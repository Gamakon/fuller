#!/usr/bin/env python3
"""Generate the constant lattice for snap_karva (offline, build-time artifact).

Idea (user, 2026-05-26): don't hand-list ~25 constants. Take every known
constant x integers x a set of symbolic combinators, compute each combination's
numeric value, and freeze the whole table. Snap then = numeric lookup -> the
SIMPLEST symbolic form for that value.

This runs OFFLINE. It may use sympy/mpmath for high-precision values — it is NOT
in the gamakAST runtime path; it emits a static JSON table the Rust crate loads.

Output: parity/constants_lattice.json — a list of
  {"value": <f64>, "math": "<Math s-expr over constant Vars + Num ints>",
   "label": "<human form>", "ops": <node count>}
deduplicated by a significant-figure key, keeping the SIMPLEST (fewest-ops) form
per key (that simplest-form choice is what makes snap output physics-shaped).

Constants are represented as `(Var "<name>")` in the Math s-expr so they compose
with the algebra rules in the e-graph and evaluate via the evaluator's env
(which binds the constant names to these values).
"""
from __future__ import annotations

import json
import math
import os

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
]

INTS = list(range(1, 13))   # 1..12
POWS = [2, 3, 4, 5, 6, 7, 8, 9]


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
        out.append(emit(math.sqrt(val), f"(Sqrt {cv})", f"sqrt({name})", 2))
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
    rows = lattice()
    # dedup by sig key, keep the simplest (fewest ops, then shortest label)
    best = {}
    for r in rows:
        if not math.isfinite(r["value"]):
            continue
        k = sig_key(r["value"])
        cur = best.get(k)
        if cur is None or (r["ops"], len(r["label"])) < (cur["ops"], len(cur["label"])):
            best[k] = r
    table = sorted(best.values(), key=lambda r: r["value"])
    with open(OUT, "w") as f:
        json.dump(table, f)
    print(f"generated {len(rows)} combinations -> {len(table)} distinct (4 sig-fig) -> {OUT}")
    # show a few physics-relevant ones
    for label in ("1/pi", "pi/2", "1/G", "G/(4*pi*eps0)"):
        hit = [r for r in table if r["label"] == label]
        if hit:
            print(f"  {label:18} = {hit[0]['value']:.6g}")


if __name__ == "__main__":
    main()
