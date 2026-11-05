#!/usr/bin/env python3
"""The symbolic regression kingdoms as rows of the gene frame's fixed schema (docs/DESIGN_geneframe.md 3.3, 8.1
and 8.3; docs/PLAN_futhark_kingdom.md 7.7 task K1 and section 8).

ONE data type F carrying depth, order, fam, arm and ties; the 18 operators of `SymbolTable::wide` plus the input
terminals x0..x9 and the constant ?; the properties derived by up Flow rows; each limit a `must` Guard row whose repair
renders the node as its first input ({in0}). A variant differs from another ONLY by its Guard rows. The gene is one
type, 104 words at head 34, no Dv lane.

    python3 -I gen_tables.py                     # all five variants into kingdoms/master/ (write_rows.py): tsr.<variant>, depth as sr
    python3 -I gen_tables.py --variant depth
    python3 -I gen_tables.py --ceiling 3 --depth on --windows hi --named   # a custom variant's named text form
    python3 -I gen_tables.py --check             # the committed master rows equal a fresh generation (a cargo test)

Variants (the EVOLVE_SR_PROPS values in phylu):
    single               no guard (classical GEP, arm A)
    depth                the depth guard `in0.depth <= 1` on the ten raising unaries: THE SYMBOLIC REGRESSION KINGDOM
                         (K1), the kingdom `sr` of the master (not `tsr.depth`), which replaced the old depth ladder
                         of types F, T1, T2
    order                the order ceiling `out.order <= 3` on the four binaries (arm B)
    order_depth          both (arm C)
    order_depth_windows  both and the argument windows `in0.order <= 2` on seven unaries (arm F)

The rows live in the one set of tables every kingdom lives in (`kingdoms/master/`, plan 7.7 task K0), as the kingdom
`tsr.<variant>` under the parent `tsr` (the depth variant as the kingdom `sr`, no parent), and are read back by `src/tsr_kingdom/rows.rs` through `geneframe::master`.
The old row forms (one type per dual `shape.level`, types.tsv and functions.tsv with ROOT twins, and the per-variant
tables.tsv files) were deleted with the move (plan 7.7): nothing here writes them.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(os.path.dirname(HERE), "master")
sys.path.insert(0, MASTER)
import write_rows  # noqa: E402  (kingdoms/master/write_rows.py: the one writer of every kingdom's rows)

PARENT = "tsr"
DESCRIPTION = "Typed Symbolic Regression: one data type F carrying depth and Horton-Strahler stream order; each variant's guards are its limits"

# ------------------------------------------------------------ alphabet ----
# (instance, Math constructor, family, raises depth, window-limited under `hi`)
# Order is SymbolTable::wide() (phylu src/evolve/engine.rs ~160).
BINARY = [
    ("add", "Add", "A"),
    ("sub", "Sub", "A"),
    ("mul", "Mul", "M"),
    ("pdiv", "ProtectedDiv", "M"),
]
UNARY = [
    ("sin", "Sin", True, True),
    ("cos", "Cos", True, True),
    ("tanh", "Tanh", True, True),
    ("psqrt", "ProtectedSqrt", True, False),
    ("plog", "ProtectedLog", True, False),
    ("pexp", "ProtectedExp", True, False),
    ("neg", "Neg", False, False),
    ("abs", "Abs", True, False),
    ("pow2", "Pow2", False, True),
    ("pow3", "Pow3", False, True),
    ("pinv", "ProtectedInv", False, True),
    ("tan", "Tan", True, True),
    ("pasin", "ProtectedAsin", True, False),
    ("pacos", "ProtectedAcos", True, False),
]
VARIANTS = {
    # name: (ceiling R or None, depth, windows hi)
    "single": (None, False, False),
    "order": (3, False, False),
    "depth": (None, True, False),
    "order_depth": (3, True, False),
    "order_depth_windows": (3, True, True),
}

# ------------------------------------------------- the four kinds of row ----
# Read by `KingdomTables::parse_named` (fuller `src/tsr_kingdom/rows.rs`), which keeps the first n of the MAX_INPUTS
# input terminals for a law of n columns.
MAX_INPUTS = 10
PROPERTIES = ["depth", "order", "fam", "arm", "ties"]
# A leaf (an input column or the constant ?): no transcendental, Strahler order 1, no family, one arm of order 1.
LEAF_FLOWS = [("depth", "0"), ("order", "1"), ("fam", "0"), ("arm", "1"), ("ties", "1")]
FAMILY_CODE = {"A": 1, "M": 2}
SPELL_BINARY = {"add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "pdiv": "pdiv({0}, {1})"}
SUBSTITUTE = "{in0}"


def binary_flows(fam):
    """The up flows of a sum (fam 1) or a product (fam 2): Strahler order over the flattened chain (8.1)."""
    f = FAMILY_CODE[fam]
    b = [f"if(in{k}.fam = {f}, in{k}.arm, in{k}.order)" for k in (0, 1)]
    t = [f"if(in{k}.fam = {f}, in{k}.ties, 1)" for k in (0, 1)]
    return [
        ("depth", "max(in0.depth, in1.depth)"),
        ("fam", str(f)),
        ("arm", f"max({b[0]}, {b[1]})"),
        ("ties", f"if({b[0]} = out.arm, {t[0]}, 0) + if({b[1]} = out.arm, {t[1]}, 0)"),
        ("order", "if(2 <= out.ties, out.arm + 1, out.arm)"),
    ]


def unary_flows(raises):
    return [
        ("depth", "in0.depth + 1" if raises else "in0.depth"),
        ("order", "in0.order"),
        ("fam", "0"),
        ("arm", "in0.order"),
        ("ties", "1"),
    ]


def four_rows(ceiling, depth, hi):
    """The variant as the named text form of the four kinds of row. `ceiling` is the order limit (None: no order
    guard), `depth` the transcendental depth limit of 2, `hi` the argument windows (order at most 2)."""
    if hi and ceiling is None:
        raise SystemExit("windows hi needs an order ceiling: the windows bound the argument's order")
    out = ["data_type\tF\tscalar"]
    out += [f"carries\tF\t{p}" for p in PROPERTIES]
    out.append("constant\tF\t?\t" + ";".join(f"{p}={v}" for p, v in LEAF_FLOWS))
    symbols = []  # (name, alias, role, spelling, inputs, flows, guards)
    for inst, ctor, fam in BINARY:
        guards = [f"out.order <= {ceiling}"] if ceiling else []
        symbols.append((inst, ctor, "function", SPELL_BINARY[inst], 2, binary_flows(fam), guards))
    for inst, ctor, raises, windowed in UNARY:
        guards = []
        if depth and raises:
            guards.append("in0.depth <= 1")
        if hi and windowed:
            guards.append("in0.order <= 2")
        symbols.append((inst, ctor, "function", inst + "({0})", 1, unary_flows(raises), guards))
    for k in range(MAX_INPUTS):
        symbols.append((f"x{k}", "Var", "function", f"x{k}", 0, LEAF_FLOWS, []))
    symbols.append(("?", "Num", "constant", "", 0, LEAF_FLOWS, []))
    for name, alias, role, spelling, _n, _f, _g in symbols:
        out.append(f"symbol\t{name}\t{alias}\t{role}\t-\t-")
        if spelling:
            out.append(f"spelling\t{name}\t\t{spelling}")
    for name, _a, _r, _s, n, _f, _g in symbols:
        out += [f"argument\t{name}\t{i}\tin\tF\t-" for i in range(n)]
        out.append(f"argument\t{name}\t0\tout\tF\t-")
    for name, _a, _r, _s, _n, flows, _g in symbols:
        out += [f"flow\t{name}\tout\tup\t{p}\t{e}" for p, e in flows]
    for name, _a, _r, _s, _n, _f, guards in symbols:
        for order, condition in enumerate(guards):
            out.append(f"guard\t{name}\t{order}\tmust\t{condition}")
            out.append(f"repair\t{name}\t{order}\tnode\ttrue\ttrue\t{SUBSTITUTE}")
    out.append("entry\ttsr\tF")
    return "\n".join(out) + "\n"


# THE SYMBOLIC REGRESSION KINGDOM (K1): the depth variant is a kingdom of its own, `sr`, with no variants (the lead's
# ruling of 2026-11-05: regex, sr and wgsl hold their rows directly); TSR's arm E loads the same rows.
SR = "sr"
SR_VARIANT = "depth"
SR_DESCRIPTION = ("symbolic regression: one data type F whose transcendental depth is a property; the ten raising "
                  "unaries guarded by in0.depth <= 1 (must, repaired by {in0}); TSR's depth variant (arm E)")


def kingdom_of(nm):
    return SR if nm == SR_VARIANT else f"{PARENT}.{nm}"


def kingdom_row(nm):
    if nm == SR_VARIANT:
        return (SR, "", SR_DESCRIPTION, "F", "")
    return (kingdom_of(nm), PARENT, f"TSR variant {nm}", "F", "")


def check():
    """0 when the master rows of the parent and every variant are what the generator writes, else 1."""
    bad = []
    parent = [k for k in write_rows.read_master()[write_rows.KINGDOMS_FILE] if k[0] == PARENT]
    if parent != [[PARENT, "", DESCRIPTION, "", ""]]:
        bad.append(f"kingdoms/master/{write_rows.KINGDOMS_FILE}: the parent row of {PARENT} is {parent}")
    for nm, (ceiling, depth, hi) in VARIANTS.items():
        bad += write_rows.check_kingdom(kingdom_of(nm), write_rows.rows_of_named(four_rows(ceiling, depth, hi)), kingdom_row(nm))
    if f"{PARENT}.{SR_VARIANT}" in write_rows.kingdom_names(write_rows.read_master()):
        bad.append(f"kingdoms/master/{write_rows.KINGDOMS_FILE} lists {PARENT}.{SR_VARIANT}; the depth variant is the kingdom {SR}")
    for b in bad:
        print(b)
    print(f"check: {len(VARIANTS)} variants of {PARENT} (depth as {SR}) in kingdoms/master/, {len(bad)} differences")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="all", help="a named variant or `all`")
    ap.add_argument("--ceiling", help="order ceiling R, or `none` (overrides the variant; with --named)")
    ap.add_argument("--depth", choices=["on", "off"], help="the transcendental depth guard (with --named)")
    ap.add_argument("--windows", choices=["none", "hi"], help="argument windows (with --named)")
    ap.add_argument("--named", action="store_true", help="print the named text form instead of writing")
    ap.add_argument("--check", action="store_true", help="compare the committed master rows with a fresh generation")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    custom = a.ceiling is not None or a.depth is not None or a.windows is not None
    if custom:
        if not a.named:
            sys.exit("a custom variant is printed with --named; the master holds the five named variants only")
        base = VARIANTS[a.variant] if a.variant in VARIANTS else (3, False, False)
        ceiling = base[0] if a.ceiling is None else (None if a.ceiling == "none" else int(a.ceiling))
        depth = base[1] if a.depth is None else a.depth == "on"
        hi = base[2] if a.windows is None else a.windows == "hi"
        sys.stdout.write(four_rows(ceiling, depth, hi))
        return
    names = list(VARIANTS) if a.variant == "all" else [a.variant]
    for nm in names:
        if nm not in VARIANTS:
            sys.exit(f"unknown variant {nm!r}; known: {', '.join(VARIANTS)}")
    if a.named:
        for nm in names:
            sys.stdout.write(four_rows(*VARIANTS[nm]))
        return
    if not [k for k in write_rows.read_master()[write_rows.KINGDOMS_FILE] if k[0] == PARENT]:
        write_rows.write_parent(PARENT, DESCRIPTION)
    for nm in names:
        text = four_rows(*VARIANTS[nm])
        write_rows.write_kingdom(kingdom_of(nm), write_rows.rows_of_named(text), kingdom_row(nm))
        guards = sum(1 for l in text.splitlines() if l.startswith("guard\t"))
        print(f"{kingdom_of(nm)}: {text.count(chr(10))} rows, {guards} guards -> kingdoms/master/")
    retire(f"{PARENT}.{SR_VARIANT}")


def retire(kingdom):
    """Drop a kingdom the generator no longer writes (the old name of the depth variant) from every master table."""
    master = write_rows.read_master()
    if kingdom in write_rows.kingdom_names(master):
        write_rows.write_master({f: [r for r in rows if r[0] != kingdom] for f, rows in master.items()})
        print(f"{kingdom}: retired from kingdoms/master/")


if __name__ == "__main__":
    main()
