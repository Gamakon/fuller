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
