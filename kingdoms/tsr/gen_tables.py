#!/usr/bin/env python3
"""The TSR kingdom's tables (docs/PLAN_tsr_typed_gep.md section 2, Appendix A).

Horton-Strahler stream order and transcendental chain depth are carried in the
types of the SR alphabet (`SymbolTable::wide`, 18 symbols), so a well-typed gene
cannot exceed the order ceiling or stack more transcendentals than the depth
limit. A type is a dual `shape.level`; every function is a set of rows, one per
legal combination of argument duals; every row has a stable ref like the WGSL
tables. The loader reads the TSVs; the Markdown is rendered from the same rows.

    python3 -I gen_tables.py                     # all five variants into ./<variant>/
    python3 -I gen_tables.py --variant order_depth
    python3 -I gen_tables.py --ceiling 3 --depth on --windows hi --out DIR
    python3 -I gen_tables.py --check             # the committed files equal a fresh generation

Each variant folder holds the typed tables (types.tsv, functions.tsv and their Markdown) and the same variant as rows
of the gene frame's fixed schema (tables.tsv: one data type F, its properties by Flow rows, each limit a Guard row).

Variants (names are the EVOLVE_SR_KINGDOM values of plan section 5):
    single               one dual F, no limits (classical GEP, arm A)
    order                ceiling 3, depth off (arm B)
    depth                no order limit, depth on (arm E)
    order_depth          ceiling 3, depth on (arm C)
    order_depth_windows  ceiling 3, depth on, argument windows hi (arm F)
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

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
WIDE = [b[0] for b in BINARY] + [u[0] for u in UNARY]

# Every other Math constructor `math_to_nodes` reads (gpu_eval.rs `Op::from_math`),
# as an explicit absent row (plan 2.8): (constructor, instance, reason).
ABSENT = [
    ("Div", "div", "raw division: its domain differs from pdiv; a protected form is never mapped to a raw one"),
    ("Sqrt", "sqrt", "raw square root: not in the alphabet, psqrt is"),
    ("Log", "log", "raw logarithm: not in the alphabet, plog is"),
    ("Exp", "exp", "raw exponential: not in the alphabet, pexp is"),
    ("Inv", "inv", "raw inverse: not in the alphabet, pinv is"),
    ("Asin", "asin", "raw arcsine: not in the alphabet, pasin is"),
    ("Acos", "acos", "raw arccosine: not in the alphabet, pacos is"),
    ("Pow", "pow", "binary power: wide() has no Pow; a literal exponent 2 or 3 canonicalises to pow2/pow3, any other exponent is not offered (never typed by the binary merge rule: it would count the exponent as an arm)"),
    ("Select3", "select3", "ternary select: a WGSL-kingdom row, not in the SR alphabet"),
]
# Canonicalisations applied before typing (plan 2.8).
CANON = [
    ("Pow", "pow_2", "pow2", "Pow(x, 2) with the literal 2 becomes pow2(x), order preserved"),
    ("Pow", "pow_3", "pow3", "Pow(x, 3) with the literal 3 becomes pow3(x), order preserved"),
]

VARIANTS = {
    # name: (ceiling R or None, depth, windows hi)
    "single": (None, False, False),
    "order": (3, False, False),
    "depth": (None, True, False),
    "order_depth": (3, True, False),
    "order_depth_windows": (3, True, True),
}
VARIANT_ARM = {"single": "A", "order": "B", "depth": "E", "order_depth": "C", "order_depth_windows": "F"}
LEVELS = ["F", "T1", "T2"]

# ------------------------------------------------------------- profiles ----
# A profile p = (family, b, c): family "A" (sum), "M" (product) or None, the
# greatest arm order b, and whether it is shared (c = 2) or single (c = 1).
LEAF = (None, 1, 1)


def order(p):
    return p[1] + (1 if p[2] >= 2 else 0)


def unary(p):
    return (None, order(p), 1)


def binary(fam, x, y):
    b = c = 0
    for ch in (x, y):
        ob, oc = (ch[1], ch[2]) if ch[0] == fam else (order(ch), 1)
        if ob > b:
            b, c = ob, oc
        elif ob == b:
            c += oc
    return (fam, b, min(c, 2))


def shapes(ceiling):
    """Closure of LEAF under the operators, root order at most `ceiling`."""
    seen, new = {LEAF}, [LEAF]
    while new:
        new = []
        for p in list(seen):
            q = unary(p)
            if order(q) <= ceiling and q not in seen:
                seen.add(q)
                new.append(q)
        for fam in "AM":
            for x in list(seen):
                for y in list(seen):
                    q = binary(fam, x, y)
                    if order(q) <= ceiling and q not in seen:
                        seen.add(q)
                        new.append(q)
    return sorted(seen, key=lambda p: (order(p), p[0] or "", p[1], p[2]))


def shape_name(p):
    return f"P{p[1]}" if p[0] is None else f"{p[0]}{p[1]}{p[2]}"


class Dual:
    __slots__ = ("shape", "level", "name", "root")

    def __init__(self, shape, level, name, root=False):
        self.shape, self.level, self.name, self.root = shape, level, name, root


def build(ceiling, depth, hi):
    """(duals, rows). rows are (instance, constructor, class, ins, out) with
    ins/out dual names; the ROOT twin rows follow the function rows."""
    if hi and ceiling is None:
        raise SystemExit("windows hi needs an order ceiling: the windows bound the argument's order")
    levels = [0, 1, 2] if depth else [None]
    shp = shapes(ceiling) if ceiling else [LEAF]
    duals = []
    for p in shp:
        for lv in levels:
            parts = []
            if ceiling:
                parts.append(shape_name(p))
            if depth:
                parts.append(LEVELS[lv])
            duals.append(Dual(p, lv, ".".join(parts) if parts else "F"))
    has_root = bool(ceiling or depth)
    if has_root:
        duals.append(Dual(None, None, "ROOT", root=True))
    else:
        duals[0].root = True
    body = [d for d in duals if not d.root] if has_root else duals

    def un_shape(p):
        return unary(p) if ceiling else LEAF

    def bi_shape(fam, x, y):
        return binary(fam, x, y) if ceiling else LEAF

    def legal(q):
        return order(q) <= ceiling if ceiling else True

    by_key = {(d.shape, d.level): d for d in body}
    rows = []
    for inst, ctor, fam in BINARY:
        for d1 in body:
            for d2 in body:
                q = bi_shape(fam, d1.shape, d2.shape)
                if not legal(q):
                    continue
                out_level = max(d1.level, d2.level) if depth else None
                rows.append((inst, ctor, "function", (d1.name, d2.name), by_key[(q, out_level)].name))
    for inst, ctor, raises, windowed in UNARY:
        for d in body:
            if hi and windowed and order(d.shape) > 2:
                continue
            if depth and raises and d.level >= 2:
                continue
            out_level = (d.level + (1 if raises else 0)) if depth else None
            out = by_key[(un_shape(d.shape), out_level)]
            rows.append((inst, ctor, "function", (d.name,), out.name))
    if has_root:
        rows += [(i, c, "root", ins, "ROOT") for (i, c, _, ins, _o) in list(rows)]
    return duals, rows


def bounds(rows, duals, n):
    """Rev-4 sizing (MultiTypedGEP.md section 2): per dual h, P, t, d, the leaf
    bound L and the width in words. `rows` includes the root twins."""
    root = next(d.name for d in duals if d.root)
    a, nin = {}, {}
    for _i, _c, _k, ins, out in rows:
        for t in set(ins):
            a[(t, out)] = max(a.get((t, out), 0), ins.count(t))
        nin[out] = max(nin.get(out, 1), len(ins))
    reach, i = [root], 0
    while i < len(reach):
        for (t, u) in a:
            if u == reach[i] and t not in reach:
                reach.append(t)
        i += 1
    a_t = {t: max([a.get((t, u), 0) for u in reach]) for t in reach}
    h = {t: (t == root) + (n - 1) * a_t[t] for t in reach}
    leaves = 1 + sum(h[u] * (nin.get(u, 1) - 1) for u in reach)
    p = {t: (t == root) + sum(h[u] * a.get((t, u), 0) for u in reach) for t in reach}
    tl = {t: max(0, min(p[t], leaves + h[t]) - h[t]) for t in reach}
    dc = {t: min(p[t], leaves) for t in reach}
    width = sum(h[t] + tl[t] + dc[t] for t in reach)
    return {"h": h, "P": p, "t": tl, "d": dc, "L": leaves, "width": width, "root": root}


# ------------------------------------------------- the four kinds of row ----
# The same variant as rows of the gene frame's fixed schema (docs/DESIGN_futhark_types_and_properties.md 3.3 and
# 8.3; docs/PLAN_futhark_kingdom.md section 8): ONE data type F carrying depth, order, fam, arm and ties; the
# properties derived by up Flow rows; each limit a `must` Guard row whose repair renders the node as its first input
# ({in0}). The gene is the single variant's word for word (one type, no Dv lane: the inputs are terminals, not
# variables), and a variant differs from another only by its Guard rows. Read by `KingdomTables::parse_named`
# (fuller `src/tsr_kingdom/rows.rs`), which keeps the first n of the MAX_INPUTS input terminals for a law of n columns.
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


# --------------------------------------------------------------- writers ----
def meaning(d, ceiling, depth):
    if d.root:
        return "the root slot: any legal expression"
    bits = []
    if ceiling:
        p = d.shape
        fam = {"A": "a sum", "M": "a product", None: "neither a sum nor a product"}[p[0]]
        bits.append(f"upper bound: {fam}, greatest arm order {p[1]}, "
                    f"{'shared' if p[2] == 2 else 'single'}, stream order {order(p)}")
    if depth:
        bits.append(f"at most {d.level} transcendental" + ("" if d.level == 1 else "s") + " on a path")
    return "; ".join(bits) if bits else "a plain number (one type, no limits)"


def write_variant(name, ceiling, depth, hi, outdir, program_size, n_inputs):
    duals, rows = build(ceiling, depth, hi)
    os.makedirs(outdir, exist_ok=True)
    has_root = any(d.root and d.name == "ROOT" for d in duals)

    # types.tsv
    ref_of_type = {}
    lines = ["ref\tdual\tshape\tlevel\tfamily\tb\tc\torder\troot\tmeaning"]
    for i, d in enumerate(duals, 1):
        ref = f"T{i:03d}"
        ref_of_type[d.name] = ref
        if d.root and has_root:
            cols = ["-", "-", "-", "-", "-", "-"]
        else:
            p = d.shape
            cols = [
                shape_name(p) if ceiling else "-",
                LEVELS[d.level] if depth else "-",
                (p[0] or "-") if ceiling else "-",
                str(p[1]) if ceiling else "-",
                str(p[2]) if ceiling else "-",
                str(order(p)) if ceiling else "-",
            ]
        lines.append("\t".join([ref, d.name] + cols + ["yes" if d.root else "no", meaning(d, ceiling, depth)]))
    with open(os.path.join(outdir, "types.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # functions.tsv: function rows, root twins, terminals, absent, canon.
    out_lines = ["ref\tclass\tinstance\tconstructor\tin\tout\tarity\tnote"]
    n = 0

    def emit(cls, inst, ctor, ins, out, note=""):
        nonlocal n
        n += 1
        out_lines.append("\t".join([
            f"F{n:05d}", cls, inst, ctor, ",".join(ins) if ins else "-", out, str(len(ins)), note,
        ]))

    for inst, ctor, cls, ins, out in rows:
        emit(cls, inst, ctor, ins, out, "twin of the function row, output ROOT" if cls == "root" else "")
    for d in duals:
        emit("terminal", "var", "Var", (), d.name,
             "one terminal per input column (the loader expands `var` to every column)")
        emit("terminal", "const", "Num", (), d.name, "the constant placeholder ?")
    for ctor, inst, why in ABSENT:
        emit("absent", inst, ctor, (), "-", why)
    for ctor, inst, to, why in CANON:
        emit("canon", inst, ctor, (), "-", why + f" (canonicalises to {to})")
    with open(os.path.join(outdir, "functions.tsv"), "w") as f:
        f.write("\n".join(out_lines) + "\n")

    write_types_md(name, ceiling, depth, hi, outdir, duals, rows, program_size, n_inputs)
    write_symbols_md(name, ceiling, depth, hi, outdir, duals, rows, n_inputs)
    with open(os.path.join(outdir, "tables.tsv"), "w") as f:
        f.write(four_rows(ceiling, depth, hi))
    n_fn = sum(1 for r in rows if r[2] == "function")
    return {"name": name, "duals": len(duals), "rows": len(rows), "function_rows": n_fn,
            "width": bounds(rows, duals, program_size)["width"]}


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def params_text(name, ceiling, depth, hi):
    return (f"variant `{name}` (arm {VARIANT_ARM.get(name, '-')}): order ceiling "
            f"{ceiling if ceiling else 'none'}, depth {'on' if depth else 'off'}, "
            f"windows {'hi' if hi else 'none'}")


def write_types_md(name, ceiling, depth, hi, outdir, duals, rows, program_size, n_inputs):
    root = next(d for d in duals if d.root)
    body_duals = [d for d in duals if not d.root]
    md = [f"# TSR kingdom, {name}: the type table", "",
          "Generated by `gen_tables.py` from the rules of docs/PLAN_tsr_typed_gep.md section 2; "
          "edit the script, not this file.", "",
          params_text(name, ceiling, depth, hi) + ".", ""]
    md += ["## Duals", "",
           "A dual is `shape.level`. It is an UPPER BOUND on the subtree under a slot: every leaf fits every dual, "
           "and a row's output is computed from the argument duals it asks for, so the true profile of what fills "
           "a slot is never above the declared one. Soundness: a gene that expresses has true order at most the "
           "ceiling and true depth at most 2. Completeness: a tree within both limits has an exact dual at every "
           "node, so every node is an existing row.", ""]
    md += [md_table(["ref", "dual", "order", "meaning"],
                    [(f"T{i:03d}", d.name, order(d.shape) if (ceiling and not d.root) else "-", meaning(d, ceiling, depth))
                     for i, d in enumerate(duals, 1)]), ""]
    if ceiling:
        shp = shapes(ceiling)
        names = [shape_name(p) for p in shp]
        md += ["## Shapes", "",
               "Profile = (family, greatest arm order b, shared c). Order = b + 1 if c = 2 else b. "
               "Order never decreases going up, so the ceiling on the root is a ceiling on every node.", "",
               md_table(["shape", "family", "b", "shared", "order"],
                        [(shape_name(p), p[0] or "none", p[1], p[2] == 2, order(p)) for p in shp]), ""]
        md += ["## Signature of add and sub", "",
               "First argument down, second across; the result shape, `-` when the result would have order "
               f"{ceiling + 1}. Mul and pdiv are the same table with A and M exchanged. Depth levels combine by max "
               "and do not change the shape.", ""]
        cells = []
        for x in shp:
            cells.append([shape_name(x)] + [
                (shape_name(binary("A", x, y)) if order(binary("A", x, y)) <= ceiling else "-") for y in shp])
        md += [md_table(["first \\ second"] + names, cells), ""]
        md += ["## Signature of every unary operator", "",
               "The result shape is P of the argument's order; one row per argument shape.", "",
               md_table(["argument shape", "result shape"], [(shape_name(p), shape_name(unary(p))) for p in shp]), ""]
    if depth:
        md += ["## Depth levels", "",
               "F, T1, T2: the most transcendental (raising) operators on a path from the node to a leaf. "
               "Binary operators take the larger level of their arguments. Raising unaries "
               "(sin cos tan tanh psqrt plog pexp abs pasin pacos) add 1 and have no row for a T2 argument. "
               "Level-keeping unaries (neg pow2 pow3 pinv) keep the level.", ""]
    if hi:
        md += ["## Argument windows (hi)", "",
               "sin cos tan tanh pow2 pow3 pinv have no row for an argument of order above 2. "
               "The exp minimum (argument order at least 2) is a lower bound and cannot be carried in types "
               "(plan 2.4); the pair table is not carried at all.", ""]
    md += ["## Rows", ""]
    by_op = {}
    for inst, _c, cls, ins, _o in rows:
        by_op.setdefault((inst, cls), 0)
        by_op[(inst, cls)] += 1
    md += [md_table(["operator", "rows", "root twin rows"],
                    [(i, by_op.get((i, "function"), 0), by_op.get((i, "root"), 0)) for i in WIDE]), ""]
    md += [f"Function rows {sum(1 for r in rows if r[2] == 'function')}, root twins "
           f"{sum(1 for r in rows if r[2] == 'root')}, terminals {2 * len(duals)} per input column "
           f"(`var` expanded per column plus `const`; {len(duals) * (n_inputs + 1)} at {n_inputs} inputs).", ""]
    md += ["## Terminals", "",
           "Every dual, ROOT included, has a `var` terminal (one per input column) and a `const` placeholder `?`. "
           "That is the polymorphic spelling of MultiTypedGEP.md section 1, and it satisfies the contract check "
           "\"a terminal for every demanded type\" (a bare leaf has order 1 and a type such as A21 cannot otherwise "
           "have a terminal).", ""]
    if has_root_dual(duals):
        md += ["## ROOT", "",
               "A dual of its own with a twin of every function row (same operator, same argument duals, output ROOT). "
               "Nothing takes ROOT as an argument, so h = 1, P = 1, there is no tail and d = 1.", ""]
    md += ["## Sizing (rev 4)", "",
           "Words per gene, Dc included, from the generated rows. One-type SR at head 34 is N = 17 "
           "(the largest N with 1 + (N - 1) * 2 at most 34); the narrow variants use N = 8.", ""]
    srows = []
    for nn in sorted({8, 17, 32, program_size}):
        b = bounds(rows, duals, nn)
        srows.append((nn, sum(b["h"].values()), b["L"], sum(b["t"].values()), sum(b["d"].values()), b["width"]))
    md += [md_table(["N", "sum of heads", "L", "sum of tails", "sum of Dc", "width"], srows), ""]
    b = bounds(rows, duals, program_size)
    md += [f"Per-dual budgets at N = {program_size}:", "",
           md_table(["dual", "h", "P", "t", "d"],
                    [(t, b["h"][t], b["P"][t], b["t"][t], b["d"][t]) for t in b["h"]]), ""]
    with open(os.path.join(outdir, "TYPES.md"), "w") as f:
        f.write("\n".join(md))


def has_root_dual(duals):
    return any(d.name == "ROOT" for d in duals)


def write_symbols_md(name, ceiling, depth, hi, outdir, duals, rows, n_inputs):
    md = [f"# TSR kingdom, {name}: the function table", "",
          "Generated by `gen_tables.py` from `functions.tsv`; edit the script, not this file.", "",
          params_text(name, ceiling, depth, hi) + ".", "",
          "Every row of `functions.tsv` has a stable ref `F<n>`; this page summarises them. A function is an "
          "operator with a signature over duals; the evaluator operator of a row is its `constructor` "
          "(a root twin decodes to the same evaluator operator as the row it copies).", "",
          "## The 18-symbol alphabet (`SymbolTable::wide`)", ""]
    ctor = {i: c for i, c, *_ in BINARY} | {i: c for i, c, *_ in UNARY}
    cls_of = {}
    for i, _c, f in BINARY:
        cls_of[i] = "sum" if f == "A" else "product"
    for i, _c, raises, windowed in UNARY:
        cls_of[i] = ("raising unary" if raises else "level-keeping unary") + (", windowed" if (hi and windowed) else "")
    cnt = {}
    for inst, _c, cls, _i, _o in rows:
        cnt.setdefault(inst, [0, 0])[0 if cls == "function" else 1] += 1
    md += [md_table(["symbol", "constructor", "arity", "class", "rows", "root twins"],
                    [(i, ctor[i], 2 if i in {b[0] for b in BINARY} else 1, cls_of[i], cnt.get(i, [0, 0])[0],
                      cnt.get(i, [0, 0])[1]) for i in WIDE]), ""]
    md += ["## Absent rows", "",
           "Every other constructor `math_to_nodes` reads has an explicit absent entry, so a spelling that is "
           "neither a row nor an absent entry is a failing test, not prose.", "",
           md_table(["constructor", "instance", "reason"], [(c, i, w) for c, i, w in ABSENT]), "",
           "## Canonicalisation", "",
           md_table(["constructor", "instance", "becomes", "rule"], [(c, i, t, w) for c, i, t, w in CANON]), "",
           "## Terminals", "",
           f"{len(duals)} duals, each with `var` (one per input column) and `const` (`?`): "
           f"{len(duals) * (n_inputs + 1)} terminal symbols at {n_inputs} inputs.", ""]
    with open(os.path.join(outdir, "symbols.md"), "w") as f:
        f.write("\n".join(md))


FILES = ("types.tsv", "functions.tsv", "TYPES.md", "symbols.md", "tables.tsv")


def check(program_size, n_inputs):
    """0 when every committed file of every variant is byte for byte what the generator writes, else 1."""
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for nm, (ceiling, depth, hi) in VARIANTS.items():
            out = os.path.join(tmp, nm)
            write_variant(nm, ceiling, depth, hi, out, program_size, n_inputs)
            for f in FILES:
                with open(os.path.join(out, f)) as fresh:
                    want = fresh.read()
                path = os.path.join(HERE, nm, f)
                got = open(path).read() if os.path.exists(path) else None
                if got != want:
                    bad.append(os.path.relpath(path, HERE))
    for b in bad:
        print(f"differs from a fresh generation: {b}")
    print(f"check: {len(VARIANTS)} variants, {len(VARIANTS) * len(FILES)} files, {len(bad)} differ")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="all", help="a named variant or `all`")
    ap.add_argument("--ceiling", help="order ceiling R, or `none` (overrides the variant)")
    ap.add_argument("--depth", choices=["on", "off"], help="transcendental depth in the types")
    ap.add_argument("--windows", choices=["none", "hi"], help="argument windows")
    ap.add_argument("--program-size", type=int, default=17, help="N for the headline sizing (default 17)")
    ap.add_argument("--inputs", type=int, default=4, help="input columns for the terminal counts (default 4)")
    ap.add_argument("--out", help="output directory (default ./<variant>/ beside this script)")
    ap.add_argument("--check", action="store_true", help="compare the committed files with a fresh generation")
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.program_size, a.inputs))
    custom = a.ceiling is not None or a.depth is not None or a.windows is not None
    todo = []
    if custom:
        if not a.out:
            sys.exit("custom parameters need --out")
        base = VARIANTS[a.variant] if a.variant in VARIANTS else (3, False, False)
        ceiling = base[0] if a.ceiling is None else (None if a.ceiling == "none" else int(a.ceiling))
        depth = base[1] if a.depth is None else a.depth == "on"
        hi = base[2] if a.windows is None else a.windows == "hi"
        todo.append((a.variant if a.variant != "all" else "custom", ceiling, depth, hi, a.out))
    else:
        names = list(VARIANTS) if a.variant == "all" else [a.variant]
        for nm in names:
            if nm not in VARIANTS:
                sys.exit(f"unknown variant {nm!r}; known: {', '.join(VARIANTS)}")
            todo.append((nm, *VARIANTS[nm], a.out or os.path.join(HERE, nm)))
    for nm, ceiling, depth, hi, outdir in todo:
        s = write_variant(nm, ceiling, depth, hi, outdir, a.program_size, a.inputs)
        print(f"{s['name']}: {s['duals']} duals, {s['function_rows']} function rows, {s['rows']} with root twins, "
              f"width {s['width']} at N = {a.program_size} -> {outdir}")


if __name__ == "__main__":
    main()
