"""sympy <-> gamakAST `Math` s-expression bridge.

The ONLY place sympy meets gamakAST surface syntax. Converts a sympy
expression into the egglog `Math` s-expression that `proves_equal`,
`denoise`, etc. accept. Real-domain ops only; returns None for anything the
`Math` sort doesn't model (the caller treats None as "unconvertible").

This is the single source of truth for the conversion — `parity/gen_corpus.py`
and any consumer (e.g. the HFF SR recovery checker) import it from here rather
than re-implementing the table.
"""
from __future__ import annotations


_SYMPY_TO_MATH = None


def _table():
    global _SYMPY_TO_MATH
    if _SYMPY_TO_MATH is None:
        import sympy as sp
        # sympy func -> (Math constructor, arity). Real-domain only.
        _SYMPY_TO_MATH = {
            sp.Add: ("Add", 2), sp.Mul: ("Mul", 2),
            sp.sin: ("Sin", 1), sp.cos: ("Cos", 1), sp.tan: ("Tan", 1),
            sp.exp: ("Exp", 1), sp.log: ("Log", 1), sp.tanh: ("Tanh", 1),
            sp.Abs: ("Abs", 1),
        }
    return _SYMPY_TO_MATH


def to_math(expr):
    """sympy expression -> gamakAST `Math` s-expression string, or None if it
    contains an op the `Math` sort does not model."""
    import sympy as sp
    # sympy singletons FIRST: pi/E are NumberSymbol, which is NOT a Number
    # subclass — `is_Number` is False and their `func` is not in the table, so
    # without this they silently convert to None. The Math sort models them as
    # named constant Vars; snap/eval bind the values via the constant lattice.
    if expr is sp.pi:
        return '(Var "pi")'
    if expr is sp.E:
        return '(Var "e")'
    if expr.is_Number:
        return f"(Num {float(expr)})"
    if expr.is_Symbol:
        return f'(Var "{expr.name}")'
    if expr.is_Pow:
        base, exp = expr.args
        b = to_math(base)
        if b is None:
            return None
        if exp == 2:
            return f"(Pow2 {b})"
        if exp == 3:
            return f"(Pow3 {b})"
        if exp == -1:
            return f"(Inv {b})"
        if exp == sp.Rational(1, 2):
            return f"(Sqrt {b})"
        e = to_math(exp)
        return None if e is None else f"(Pow {b} {e})"
    spec = _table().get(expr.func)
    if spec is None:
        return None
    ctor, arity = spec
    args = list(expr.args)
    # binarise n-ary Add/Mul left-leaning
    if ctor in ("Add", "Mul") and len(args) > 2:
        acc = to_math(args[0])
        for a in args[1:]:
            am = to_math(a)
            if acc is None or am is None:
                return None
            acc = f"({ctor} {acc} {am})"
        return acc
    if len(args) != arity:
        return None
    parts = [to_math(a) for a in args]
    if any(p is None for p in parts):
        return None
    return f"({ctor} {' '.join(parts)})"


def _tokenize_math(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in "()":
            out.append(c)
            i += 1
        elif c == '"':
            j = s.index('"', i + 1)
            out.append(s[i : j + 1])
            i = j + 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and s[j] not in '()"' and not s[j].isspace():
                j += 1
            out.append(s[i:j])
            i = j
    return out


def from_math(s):
    """gamakAST `Math` s-expression string -> sympy expression, or None if the
    string is malformed. Inverse of `to_math`; the ONLY sympy-side decoder of
    the `Math` grammar — consumers must not re-implement this table.

    Named constants round-trip: `(Var "pi")`/`(Var "e")` -> sympy.pi/sympy.E;
    every other Var becomes `Symbol(name)`.

    Protected ops are rendered at their generic (non-singular) point:
    ProtectedSqrt/ProtectedLog go through Abs (their actual definition),
    ProtectedExp -> exp, ProtectedInv -> 1/x, ProtectedDiv -> a/b. The
    singular-point special cases (protected_inv(0)=1, protected_div(x,0)=0)
    have no sympy analogue — do not use this rendering to reason about
    behaviour AT the singularity.
    """
    import sympy as sp

    toks = _tokenize_math(s)
    pos = [0]

    def parse():
        if pos[0] >= len(toks) or toks[pos[0]] != "(":
            return None
        pos[0] += 1
        if pos[0] >= len(toks):
            return None
        head = toks[pos[0]]
        pos[0] += 1
        if head == "Num":
            if pos[0] >= len(toks):
                return None
            try:
                node = sp.Float(toks[pos[0]])
            except (ValueError, TypeError):
                return None
            pos[0] += 1
        elif head == "Var":
            if pos[0] >= len(toks):
                return None
            name = toks[pos[0]].strip('"')
            pos[0] += 1
            node = {"pi": sp.pi, "e": sp.E}.get(name) or sp.Symbol(name)
        else:
            kids = []
            while pos[0] < len(toks) and toks[pos[0]] != ")":
                k = parse()
                if k is None:
                    return None
                kids.append(k)
            build = {
                ("Add", 2): lambda a, b: a + b,
                ("Sub", 2): lambda a, b: a - b,
                ("Mul", 2): lambda a, b: a * b,
                ("Div", 2): lambda a, b: a / b,
                ("Pow", 2): lambda a, b: a**b,
                ("ProtectedDiv", 2): lambda a, b: a / b,
                ("Neg", 1): lambda a: -a,
                ("Sin", 1): sp.sin,
                ("Cos", 1): sp.cos,
                ("Tan", 1): sp.tan,
                ("Tanh", 1): sp.tanh,
                ("Log", 1): sp.log,
                ("Exp", 1): sp.exp,
                ("Sqrt", 1): sp.sqrt,
                ("Abs", 1): sp.Abs,
                ("Pow2", 1): lambda a: a**2,
                ("Pow3", 1): lambda a: a**3,
                ("Inv", 1): lambda a: 1 / a,
                ("ProtectedSqrt", 1): lambda a: sp.sqrt(sp.Abs(a)),
                ("ProtectedLog", 1): lambda a: sp.log(sp.Abs(a)),
                ("ProtectedExp", 1): sp.exp,
                ("ProtectedInv", 1): lambda a: 1 / a,
            }.get((head, len(kids)))
            if build is None:
                return None
            node = build(*kids)
        if pos[0] >= len(toks) or toks[pos[0]] != ")":
            return None
        pos[0] += 1
        return node

    node = parse()
    return node if node is not None and pos[0] == len(toks) else None


def equals(a, b) -> bool:
    """SRBench-style symbolic equivalence: is `a` the same law as `b`, up to an
    additive or multiplicative constant? Sound, bounded, never hangs — the
    drop-in for `sympy.simplify(a - b) == 0`-style recovery checks (simplify
    spins forever on junk transcendental towers; equality saturation cannot).

    `a`, `b` may be sympy expressions or strings (sympified here). Returns True
    iff one of the three recovery tests proves equal under the bounded `wide`
    ruleset:
      1. a == b                       (exact / reordered)
      2. a - b == 0                   (additive constant offset)
      3. a / b == 1, vars nonzero     (scale constant factor)

    A True is a sound proof; a False is "not proven equal" (never a false
    positive — a genuinely-different pair is never reported recovered).
    """
    import sympy as sp
    from . import proves_equal  # PyO3 oracle

    a = sp.sympify(a) if not isinstance(a, sp.Basic) else a
    b = sp.sympify(b) if not isinstance(b, sp.Basic) else b
    # Align symbol identity by name (real vs unconstrained Symbol mismatch).
    a = a.subs({s: sp.Symbol(s.name) for s in a.free_symbols if s.is_Symbol})
    b = b.subs({s: sp.Symbol(s.name) for s in b.free_symbols if s.is_Symbol})

    am, bm = to_math(a), to_math(b)
    if am is None or bm is None:
        return False
    if proves_equal(am, bm, "wide"):
        return True
    dm = to_math(sp.expand(a - b))
    if dm is not None and proves_equal(dm, "(Num 0.0)", "wide"):
        return True
    rm = to_math(a / b)
    nz = [s.name for s in (a / b).free_symbols if s.is_Symbol]
    if rm is not None and proves_equal(rm, "(Num 1.0)", "wide", nz):
        return True
    return False
