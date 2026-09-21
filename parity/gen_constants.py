#!/usr/bin/env python3
"""Generate the constant lattice for snap_karva (offline, build-time artifact).

Idea (user, 2026-05-26): don't hand-list ~25 constants. Take every known
constant x integers x a set of symbolic combinators, compute each combination's
numeric value, and freeze the whole table. Snap then = numeric lookup -> the
SIMPLEST symbolic form for that value.

This runs OFFLINE. It may use sympy/mpmath for high-precision values — it is NOT
in the fuller runtime path; it emits a static JSON table the Rust crate loads.

Output: parity/constants_lattice.json — a list of
  {"value": <f64>, "math": "<Math s-expr over constant Vars + Num ints>",
   "label": "<human form>", "ops": <node count>}
deduplicated by a significant-figure key, keeping the SIMPLEST (fewest-ops) form
per key (that simplest-form choice is what makes snap output physics-shaped).

Constants are represented as `(Var "<name>")` in the Math s-expr so they compose
with the algebra rules in the e-graph and evaluate via the evaluator's env
(which binds the constant names to these values).

PI-MONOMIALS (2026-09-21). The rule, problem-agnostic, over sets this file
already uses — no value is listed by hand:

    q * pi^k        q = a/b,  a in SMALL = {1,2,3,4},  b in INTS = {1..12},
                    gcd(a, b) = 1,  k in PI_POWERS = {-3, -2, -1, 1, 2, 3}
    sqrt(q * pi^s)  the same q,  s in {-1, 1}          (k = -1/2, 1/2)

spelt with Div/Mul only, pi^2 as pi*pi and pi^3 as (Pow pi 3) like the rest of
the file, `ops` = the template's true node count. A (q, k) the older families
already emit (n*pi, pi/n, n/pi, 1/(n*pi), n*pi^2, pi^2/n, sqrt(n*pi), ...) comes
out with the identical label and math and is emitted once. k = 0 is NOT here:
plain rationals are `snap_karva::small_integer_entries`, kept small on purpose.

PURE FORMS ARE NOT HIDDEN BY DIMENSIONAL ONES. h = 2*pi*hbar, so hbar/h IS
1/(2*pi), and at 3 ops it used to take the key from the 5-op `1/(2*pi)`: the
only spelling of 1/(2*pi) in the table planted `hbar` and `h` in a gene. The
dedup is now two steps. (1) The older families (`lattice()`) keep the simplest
form per sig-fig key among themselves, exactly as before: every entry the table
ever had is still there under its label, never displaced by a later family.
(2) Per key, the simplest PURE form (PURE names only) of all families, the
pi-monomials included, is ADDED when the key is free or its holder names a
dimensional constant. A key held by a pure form keeps that form alone.
"""
from __future__ import annotations

import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "constants_lattice.json")

# --- base constants: (var_name, value, is_dimensionless) ----------------------
# Dimensionless math constants and fundamental physical constants (SI).
BASE = [
    ("pi",    math.pi),
    ("e",     math.e),
    ("sqrt2", math.sqrt(2.0)),
    ("sqrt3", math.sqrt(3.0)),
    ("phi",   (1.0 + math.sqrt(5.0)) / 2.0),     # golden ratio
    ("gamma", 0.5772156649015329),               # Euler–Mascheroni
    # physical (SI)
    ("G",     6.674_30e-11),
    ("c",     2.997_924_58e8),
    ("hbar",  1.054_571_817e-34),
    ("h",     6.626_070_15e-34),
    ("kB",    1.380_649e-23),
    ("qe",    1.602_176_634e-19),                # elementary charge
    ("eps0",  8.854_187_8128e-12),               # vacuum permittivity
    ("mu0",   1.256_637_062_12e-6),              # vacuum permeability
    ("NA",    6.022_140_76e23),                  # Avogadro
    ("me",    9.109_383_7015e-31),               # electron mass
    ("g_earth", 9.806_65),                       # standard gravity (m/s^2)
]

INTS = list(range(1, 13))   # 1..12
POWS = [2, 3, 4, 5, 6, 7, 8, 9]
SMALL = [1, 2, 3, 4]        # the integer grid of the rational-multiple families
# The names that carry no dimension; every other base constant is dimensional.
PURE = ("pi", "e", "sqrt2", "sqrt3", "phi", "gamma")
PI_POWERS = [-3, -2, -1, 1, 2, 3]
_VAR = re.compile(r'\(Var "([^"]+)"\)')


def is_dimensional(math_expr):
    """True iff the form names a constant outside PURE."""
    return any(name not in PURE for name in _VAR.findall(math_expr))


def node_count(math_expr):
    """Nodes of a Math s-expr: one per operator application, Var and Num."""
    return math_expr.count("(")


def pi_monomials():
    """q * pi^k and sqrt(q * pi^s) — the rule in the module docstring.
    Yields (value, math, label): ascending k, then a, then b; then the roots."""
    pv = '(Var "pi")'
    power = {1: (pv, "pi"), 2: (f"(Mul {pv} {pv})", "pi^2"), 3: (f"(Pow {pv} (Num 3.0))", "pi^3")}

    def monomial(a, b, k):
        expr, name = power[abs(k)]
        value = math.pi ** abs(k)
        if k > 0:
            top, top_label = (expr, name) if a == 1 else (f"(Mul (Num {float(a)}) {expr})", f"{a}*{name}")
            if b == 1:
                return a * value, top, top_label
            return a * value / b, f"(Div {top} (Num {float(b)}))", f"{top_label}/{b}"
        if b == 1:
            return a / value, f"(Div (Num {float(a)}) {expr})", f"{a}/{name}"
        return a / (b * value), f"(Div (Num {float(a)}) (Mul (Num {float(b)}) {expr}))", f"{a}/({b}*{name})"

    for k in PI_POWERS:
        for a in SMALL:
            for b in INTS:
                if math.gcd(a, b) != 1 or (a == 1 and b == 1 and k > 0):
                    continue  # bare pi^k is the base / POWS family's
                yield monomial(a, b, k)
    for s in (-1, 1):
        for a in SMALL:
            for b in INTS:
                if math.gcd(a, b) != 1 or (a == 1 and b == 1 and s > 0):
                    continue  # sqrt(pi) is the sqrt family's
                value, expr, label = monomial(a, b, s)
                yield math.sqrt(value), f"(Sqrt {expr})", f"sqrt({label})"


def emit(value, math_expr, label, ops):
    return {"value": value, "math": math_expr, "label": label, "ops": ops}


def lattice():
    """Enumerate combinations. Each yields (value, math_sexpr, label, op_count)."""
    out = []
    for name, val in BASE:
        cv = f'(Var "{name}")'
        out.append(emit(val, cv, name, 1))                          # c
        out.append(emit(-val, f"(Neg {cv})", f"-{name}", 2))        # -c
        for n in INTS:
            ni = float(n)
            out.append(emit(val / ni, f'(Div {cv} (Num {ni}))', f"{name}/{n}", 3))
            out.append(emit(val * ni, f'(Mul (Num {ni}) {cv})', f"{n}*{name}", 3))
            out.append(emit(ni / val, f'(Div (Num {ni}) {cv})', f"{n}/{name}", 3))
        # Reciprocals use Div(1, .) NOT Inv: `div` is in almost every pset
        # (truediv/protected_div), whereas `inv` is rare — emitting Inv made the
        # whole 1/(...) family inexpressible for real chromosomes (the snap
        # silently failed to decode back to karva). Div forms decode everywhere.
        out.append(emit(1.0 / val, f"(Div (Num 1.0) {cv})", f"1/{name}", 3))
        # 1/(n*c) and n/(m*c) — the (4pi) family the Feynman near-misses need
        # (e.g. 1/(4*pi) = 0.0796). Generic, not hardcoded.
        for n in INTS:
            ni = float(n)
            out.append(emit(1.0 / (ni * val),
                            f'(Div (Num 1.0) (Mul (Num {ni}) {cv}))', f"1/({n}*{name})", 5))
        for p in POWS:
            out.append(emit(val ** p, f'(Pow {cv} (Num {float(p)}))', f"{name}^{p}", 3))
        # --- sqrt family (Gaussian normaliser 1/sqrt(2*pi) and friends) ---
        if val > 0:
            out.append(emit(math.sqrt(val), f"(Sqrt {cv})", f"sqrt({name})", 2))
            # 1/sqrt(c)
            out.append(emit(1.0 / math.sqrt(val),
                            f"(Div (Num 1.0) (Sqrt {cv}))", f"1/sqrt({name})", 4))
            for n in INTS:
                ni = float(n)
                # sqrt(n*c) and the reciprocal 1/sqrt(n*c) — covers 1/sqrt(2*pi)
                out.append(emit(math.sqrt(ni * val),
                                f"(Sqrt (Mul (Num {ni}) {cv}))", f"sqrt({n}*{name})", 4))
                out.append(emit(1.0 / math.sqrt(ni * val),
                                f"(Div (Num 1.0) (Sqrt (Mul (Num {ni}) {cv})))",
                                f"1/sqrt({n}*{name})", 6))
                # CR_lattice_extensions: the remaining sqrt-composed forms —
                # sqrt(c/n), sqrt(n/c), 1/(n*sqrt(c)), n/sqrt(c). Cover
                # 2/sqrt(pi) (erf/diffusion), sqrt(2/pi) (scattering), oscillator
                # sqrt(g/L)-style, statistical weight factors. Div(1,.) form so
                # they decode into any pset with div+sqrt.
                out.append(emit(math.sqrt(val / ni),
                                f"(Sqrt (Div {cv} (Num {ni})))", f"sqrt({name}/{n})", 4))
                out.append(emit(math.sqrt(ni / val),
                                f"(Sqrt (Div (Num {ni}) {cv}))", f"sqrt({n}/{name})", 4))
                out.append(emit(1.0 / (ni * math.sqrt(val)),
                                f"(Div (Num 1.0) (Mul (Num {ni}) (Sqrt {cv})))",
                                f"1/({n}*sqrt({name}))", 5))
                out.append(emit(ni / math.sqrt(val),
                                f"(Div (Num {ni}) (Sqrt {cv}))", f"{n}/sqrt({name})", 4))
                # (k*pi)/sqrt(c) — pendulum period prefactor T=2π√(L/g) family.
                # 2π/√g ≈ 2.0064 is the leading constant in Feynman I_34_8-style
                # oscillator periods. Generic over integer k and base atom c.
                out.append(emit((ni * math.pi) / math.sqrt(val),
                                f'(Div (Mul (Num {ni}) (Var "pi")) (Sqrt {cv}))',
                                f"{n}*pi/sqrt({name})", 5))
    # pairwise products / ratios of two distinct constants (c1*c2, c1/c2)
    for i, (n1, v1) in enumerate(BASE):
        for n2, v2 in BASE[i + 1:]:
            cv1, cv2 = f'(Var "{n1}")', f'(Var "{n2}")'
            out.append(emit(v1 * v2, f"(Mul {cv1} {cv2})", f"{n1}*{n2}", 3))
            if v2 != 0:
                out.append(emit(v1 / v2, f"(Div {cv1} {cv2})", f"{n1}/{n2}", 3))
            if v1 != 0:
                out.append(emit(v2 / v1, f"(Div {cv2} {cv1})", f"{n2}/{n1}", 3))
    # a few three-way physics staples that recur in Feynman (c/(4 pi c2)-style)
    for n1, v1 in BASE:
        for n2, v2 in BASE:
            if n1 == n2 or v2 == 0:
                continue
            out.append(emit(v1 / (4.0 * math.pi * v2),
                            f'(Div (Var "{n1}") (Mul (Num 4.0) (Mul (Var "pi") (Var "{n2}"))))',
                            f"{n1}/(4*pi*{n2})", 5))

    # === Richer proliferation (pi^2 composites, rational-multiple ratios,
    # three-way products/ratios). These reach forms the pairwise families miss —
    # notably Kepler's 4*pi^2/(G*M)-style prefactor and n*pi^2/c oscillators.
    PI = math.pi
    PI2 = PI * PI
    pv = '(Var "pi")'
    pi2_expr = f'(Mul {pv} {pv})'                      # pi^2 as pi*pi (decodes anywhere)

    # n * pi^2  and  pi^2 / n  (small integer scalings of pi^2)
    for n in INTS:
        ni = float(n)
        out.append(emit(ni * PI2, f'(Mul (Num {ni}) {pi2_expr})', f"{n}*pi^2", 4))
        out.append(emit(PI2 / ni, f'(Div {pi2_expr} (Num {ni}))', f"pi^2/{n}", 4))

    # (n*pi^2)/c  and  pi^2/(n*c)  — the Kepler / oscillator prefactor family:
    # T = sqrt(4*pi^2/(G*M) * a^3)  ->  4*pi^2/G is exactly (n*pi^2)/c with n=4.
    for name, val in BASE:
        if val == 0 or name == "pi":
            continue
        cv = f'(Var "{name}")'
        for n in INTS:
            ni = float(n)
            out.append(emit((ni * PI2) / val,
                            f'(Div (Mul (Num {ni}) {pi2_expr}) {cv})',
                            f"{n}*pi^2/{name}", 6))
            out.append(emit(PI2 / (ni * val),
                            f'(Div {pi2_expr} (Mul (Num {ni}) {cv}))',
                            f"pi^2/({n}*{name})", 6))

    # rational-multiple ratios (n*c1)/(m*c2): the earlier family only had bare
    # c1/c2. Keep the integer grid small (SMALL, 1..4) to bound the blow-up.
    for i, (n1, v1) in enumerate(BASE):
        for n2, v2 in BASE:
            if n1 == n2 or v2 == 0:
                continue
            cv1, cv2 = f'(Var "{n1}")', f'(Var "{n2}")'
            for a in SMALL:
                for b in SMALL:
                    if a == 1 and b == 1:
                        continue  # bare c1/c2 already emitted above
                    out.append(emit((a * v1) / (b * v2),
                                    f'(Div (Mul (Num {float(a)}) {cv1}) '
                                    f'(Mul (Num {float(b)}) {cv2}))',
                                    f"({a}*{n1})/({b}*{n2})", 5))

    # three-way products c1*c2*c3 and ratios c1/(c2*c3) over distinct constants
    for i, (n1, v1) in enumerate(BASE):
        for j in range(i + 1, len(BASE)):
            n2, v2 = BASE[j]
            for k in range(j + 1, len(BASE)):
                n3, v3 = BASE[k]
                cv1, cv2, cv3 = f'(Var "{n1}")', f'(Var "{n2}")', f'(Var "{n3}")'
                out.append(emit(v1 * v2 * v3,
                                f'(Mul {cv1} (Mul {cv2} {cv3}))',
                                f"{n1}*{n2}*{n3}", 5))
                if v2 != 0 and v3 != 0:
                    out.append(emit(v1 / (v2 * v3),
                                    f'(Div {cv1} (Mul {cv2} {cv3}))',
                                    f"{n1}/({n2}*{n3})", 5))

    return out


def later_families(older):
    """The families added after the table was first frozen: the pi-monomials
    q*pi^k and sqrt(q*pi^s). A form an older family already emitted has the
    same label AND the same math: it is not emitted twice."""
    emitted = {r["label"]: r["math"] for r in older}
    out = []
    for value, expr, label in pi_monomials():
        if label in emitted:
            assert emitted[label] == expr, f"{label}: {emitted[label]} respelt as {expr}"
            continue
        emitted[label] = expr
        out.append(emit(value, expr, label, node_count(expr)))
    return out


def sig_key(v, sig=4):
    """Significant-figure key: magnitude-relative so pi (~3) and G (~7e-11) both
    work. Two values collide iff they agree to `sig` significant figures."""
    if v == 0 or not math.isfinite(v):
        return ("z", 0)
    from math import log10, floor
    exp = floor(log10(abs(v)))
    mant = round(v / (10.0 ** exp), sig - 1)
    return (mant, exp, 1 if v > 0 else -1)


def main():
    def simplest(rows):
        """Per sig key, the simplest row (fewest ops, then shortest label)."""
        best = {}
        for r in rows:
            if not math.isfinite(r["value"]):
                continue
            k = sig_key(r["value"])
            cur = best.get(k)
            if cur is None or (r["ops"], len(r["label"])) < (cur["ops"], len(cur["label"])):
                best[k] = r
        return best

    older = lattice()
    rows = older + later_families(older)
    # The older families dedup among themselves exactly as they always have, so
    # no entry the table ever had is displaced or renamed by a later family.
    best = simplest(older)
    # Then, per key, the simplest PURE form of all families is added when the
    # key is free, or when its holder names a dimensional constant.
    best_pure = simplest(r for r in rows if not is_dimensional(r["math"]))
    added = [r for k, r in best_pure.items() if k not in best or is_dimensional(best[k]["math"])]
    table = sorted(list(best.values()) + added, key=lambda r: (r["value"], r["label"], r["math"]))
    assert len({r["label"] for r in table}) == len(table), "a label twice"
    with open(OUT, "w") as f:
        json.dump(table, f)
    print(f"generated {len(rows)} combinations -> {len(best)} distinct (4 sig-fig) of the older families "
          f"+ {len(added)} pure forms (key free, or held by a dimensional form) = {len(table)} -> {OUT}")
    # show a few physics-relevant ones
    for label in ("1/pi", "pi/2", "1/G", "G/(4*pi*eps0)"):
        hit = [r for r in table if r["label"] == label]
        if hit:
            print(f"  {label:18} = {hit[0]['value']:.6g}")


if __name__ == "__main__":
    main()
