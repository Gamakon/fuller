#!/usr/bin/env python3
"""The shape-classed Futhark kingdom: the design written as code.

The 1-D kingdom (`kingdoms/futhark_true`) decides that *shape is a value, not a
type*: `arr.f32` is one type however long the array is. That keeps the type
count flat, and it also means the kingdom cannot state that two arrays'
dimensions relate -- `matmul`'s inner dimensions must match, and nothing in
the type system says so.

This kingdom takes the other road, and it takes it the only way the multi-typed
GEP allows:

    **A shape class is a TYPE.**

A class is not a size. `mat.n.m` means "the array the interface calls `[n][m]`",
where `n` and `m` are *dimension roles* the interface declares. The runtime
binds the actual sizes; the Futhark source is shape-polymorphic. So the class
set is finite (the interface's roles, crossed), while the sizes are not.

Where the unification happens is the whole point. Not during expression --
a cell's alphabet is enumerated before the draw, from the layout alone, and
must stay that way. Not in the GEP -- the engine never learns about shapes.
**Here, once, offline:** this generator takes the declared interface, works out
which class each operator instantiates to, and writes the two tables the
kingdom loader reads. The engine then sees an ordinary flat typed kingdom, and
legality by construction is unchanged.

The witness tests at the bottom are the reason to trust the tables: they
re-parse the emitted TSV and re-check every shape relation from the *text*,
against a relation model written separately from the enumeration code. A
mis-declared row cannot pass silently.

Output: `types.tsv` and `functions.tsv`, the format `src/futhark_kingdom.rs`
loads.

    python3 kingdoms/futhark_shapes/gen_tables.py
"""

import itertools
import os
import re
import sys

SORT_RE = re.compile(r"^\(sort\s+(\w+)\)$")
CTOR_RE = re.compile(r"^\(constructor\s+(\w+)\s+\(([^)]*)\)\s+(\w+)\)$")
PLACEHOLDER_RE = re.compile(r"\{(\d+)\}")
RULE_CLASS_RE = re.compile(r"^;.*\((\w+(?:\.\w+)+)\)\s*$")
RULE_PRODUCT_RE = re.compile(
    r"^; the product of \((\w+\.\w+\.\w+)\) and \((\w+\.\w+\.\w+)\), transposed, is the product itself$"
)

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# THE INTERFACE. What the program's signature declares.
# --------------------------------------------------------------------------

# The dimension roles. Each is a dimension variable in the generated Futhark
# source, so `mat.n.m` and `mat.m.p` share the `m`.
ROLES = ["n", "m", "p"]

# The shape classes: every ordered pair of roles. The set is the CLOSURE of
# the declared inputs under the operators' effects -- `transpose` swaps a pair
# and `matmul` composes two of them -- and the closure over three roles is all
# nine pairs. A pruned set is legitimate only if the pruning is recorded.
MATRIX_CLASSES = [(a, b) for a in ROLES for b in ROLES]

# The function-value classes. Futhark's `map` over a matrix needs an
# ARRAY-valued function (its scalar `map` cannot map over rows), so the
# kingdom's function types are row functions. `map2` needs a TWO-argument one:
# Futhark's `map2 :: (a -> b -> c) -> [a] -> [b] -> [c]`, so a one-argument
# row function does not type-check there. Two classes, because the arity is
# part of the type.
FN = "fn.arr_f32_arr_f32"
FN2 = "fn.arr2_f32_arr_f32_arr_f32"

# The roots: the classes a gene may be rooted at.
ROOTS = [("n", "p")]

# The concrete sizes the harness uses, so the generated Futhark source can be
# checked against a real compiler. The ROLES are what the source is
# polymorphic in; these are only the sizes one harness run picks.
SIZES = {"n": 2, "m": 3, "p": 4}


# --------------------------------------------------------------------------
# The class names. The dual name is what the tables carry; the tuple is what
# the generator reasons about. The witness tests re-derive the tuple from the
# NAME, so the two representations cannot drift apart silently.
# --------------------------------------------------------------------------

def mat_class(d1, d2):
    """The dual name of the matrix class with roles `d1`, `d2`."""
    return f"mat.{d1}.{d2}"


def parse_class(name):
    """The dual name back to its roles: `mat.n.m` -> `('n', 'm')`.

    Deliberately independent of how the generator builds the names: the
    witness tests use this to re-check the emitted tables.
    """
    if not name.startswith("mat."):
        return None
    parts = name.split(".")
    if len(parts) != 3:
        return None
    return (parts[1], parts[2])


def swap(pair):
    return (pair[1], pair[0])


def input_name(pair):
    """The Futhark variable a class's input terminal binds."""
    return f"m_{pair[0]}_{pair[1]}"


# --------------------------------------------------------------------------
# THE OPERATOR TEMPLATES, and the unification.
# --------------------------------------------------------------------------

# The operator templates. Each row says which FUNCTION CLASS it needs (None
# for the operators that take no function), and the spelling template. The
# enumeration below instantiates them over the class set; the witness tests
# check the emitted rows against this table -- so a row that asks for the
# wrong function class cannot pass silently. (It would not compile: Futhark's
# `map2` wants a two-argument function.) The same table pins each spelling's
# TEMPLATE, so a row whose placeholders are in the wrong order is a finding
# and not a program Futhark silently applies the wrong way round.
#
# The table and the rows are two derivations of the same fact, on purpose:
# the witness tests are the arbiter, so neither can be wrong alone.
OPERATORS = [
    ("map_rows", FN, "map {0} {1}"),
    ("map2_rows", FN2, "map2 {0} {1} {2}"),
    ("transpose", None, "transpose {0}"),
    ("matmul", None, "matmul {0} {1}"),
]


def operator_rows():
    """Every row of the kingdom: the operator templates, instantiated over the
    class set.

    A template's type is PARAMETRIC; a row's type is a concrete class. The
    instantiation is the unification, and it happens here.
    """
    rows = []
    for d1, d2 in MATRIX_CLASSES:
        c = (d1, d2)
        # `map` over the rows: shape-preserving.
        rows.append({
            "name": f"map_rows@{mat_class(d1, d2)}",
            "out": c,
            "args": [FN, c],
            "spelling": "map {0} {1}",
            "note": "map a row function over the rows",
        })
        # `map2` over the rows: shape-preserving, and the two matrices must
        # have the SAME class -- the relation the class set makes explicit.
        # The function is a TWO-argument row function, because Futhark's
        # `map2` is curried.
        rows.append({
            "name": f"map2_rows@{mat_class(d1, d2)}",
            "out": c,
            "args": [FN2, c, c],
            "spelling": "map2 {0} {1} {2}",
            "note": "map a binary row function over two matrices of the same class",
        })
        # `transpose`: swaps the roles. The row exists only if the swapped
        # class is in the set.
        if swap(c) in MATRIX_CLASSES:
            rows.append({
                "name": f"transpose@{mat_class(d1, d2)}",
                "out": swap(c),
                "args": [c],
                "spelling": "transpose {0}",
                "note": "transpose a matrix",
            })
    # `matmul`: the shape-sensitive one. Its two arguments must agree on the
    # inner role, and the result is the outer roles. Enumerating the triples
    # over the class set IS the unification.
    for d1, d2, d3 in itertools.product(ROLES, ROLES, ROLES):
        a, b, out = (d1, d2), (d2, d3), (d1, d3)
        if a in MATRIX_CLASSES and b in MATRIX_CLASSES and out in MATRIX_CLASSES:
            rows.append({
                "name": f"matmul@{mat_class(*a)}x{mat_class(*b)}",
                "out": out,
                "args": [a, b],
                "spelling": "matmul {0} {1}",
                "note": "matrix multiplication; the inner roles must agree",
            })
    return rows


def function_rows():
    """The elementary row functions: the rows of the function classes."""
    return [
        {
            "name": f"{FN}.id",
            "out": FN,
            "args": [],
            "spelling": "\\xs -> xs",
            "note": "the identity row function",
        },
        {
            "name": f"{FN}.add1",
            "out": FN,
            "args": [],
            "spelling": "\\xs -> map (+1.0) xs",
            "note": "add one to every element",
        },
        {
            "name": f"{FN}.scale2",
            "out": FN,
            "args": [],
            "spelling": "\\xs -> map (*2.0) xs",
            "note": "double every element",
        },
        {
            "name": f"{FN2}.id",
            "out": FN2,
            "args": [],
            "spelling": "\\xs _ys -> xs",
            "note": "the identity binary row function",
        },
        {
            "name": f"{FN2}.add1",
            "out": FN2,
            "args": [],
            "spelling": "\\xs _ys -> map (+1.0) xs",
            "note": "add one to every element, ignoring the second row",
        },
        {
            "name": f"{FN2}.add",
            "out": FN2,
            "args": [],
            "spelling": "\\xs ys -> map2 (+) xs ys",
            "note": "add the two rows elementwise -- a genuinely binary row function",
        },
    ]


def terminal_rows():
    """One input terminal per class.

    The kingdom contract requires a terminal for every demanded class, and a
    class is demanded by any row that mentions it. A terminal need not be an
    external input -- a derived terminal (like the 1-D kingdom's
    `scn.f32.identity`) is legal -- but for the PoC every class gets its own
    input, which is the simplest interface that satisfies the contract.
    """
    rows = []
    for c in MATRIX_CLASSES:
        rows.append({
            "name": f"{mat_class(*c)}.input",
            "out": c,
            "args": [],
            "spelling": input_name(c),
            "note": f"the interface's {c[0]}x{c[1]} matrix",
        })
    rows.append({
        "name": f"{FN}.const",
        "out": FN,
        "args": [],
        "spelling": "\\xs -> map (\\_ -> 0.0) xs",
        "note": "the constant zero row function",
    })
    rows.append({
        "name": f"{FN2}.const",
        "out": FN2,
        "args": [],
        "spelling": "\\xs _ys -> map (\\_ -> 0.0) xs",
        "note": "the constant zero binary row function",
    })
    return rows


def type_rows():
    """One dual per class, plus the function class."""
    rows = []
    for d1, d2 in MATRIX_CLASSES:
        rows.append({
            "dual": mat_class(d1, d2),
            "slot": "mat",
            "form": f"{d1}.{d2}",
            "literals": [],
            "root": (d1, d2) in ROOTS,
            "meaning": f"a 2-D array whose dims are the interface's {d1} and {d2}",
        })
    rows.append({
        "dual": FN,
        "slot": "fn",
        "form": "arr_f32_arr_f32",
        "literals": [],
        "root": False,
        "meaning": "a row function, []f32 -> []f32",
    })
    rows.append({
        "dual": FN2,
        "slot": "fn",
        "form": "arr2_f32_arr_f32_arr_f32",
        "literals": [],
        "root": False,
        "meaning": "a binary row function, []f32 -> []f32 -> []f32",
    })
    return rows


# --------------------------------------------------------------------------
# THE STATIC COST. What a row costs, in elements, as a polynomial in the
# dimension ROLES.
# --------------------------------------------------------------------------
#
# The cost is a MODEL, not a measurement, and it is deliberately a
# pre-optimisation cost: it prices the program as the search writes it,
# before Futhark's own passes. What it must not do is punish the plain
# nested map/reduce, because on the GPU that shape is exactly what the
# compiler's own tiler recognises:
#
#   * `gpuPipeline` runs `tileLoops` (`Futhark/Optimise/TileLoops.hs`);
#   * the block+register tiler (`Futhark/Optimise/BlkRegTiling.hs`) names
#     SGEMM as one of its target patterns -- a reduction quasi-perfectly
#     nested in a kernel with two parallel dimensions, which is what the
#     matmul helper is.
#
# So the `matmul` row is charged the TRAFFIC THE TILED VERSION ACHIEVES --
# each input element read once -- and not the traffic the naive nested
# map/reduce actually costs. The tiling is the compiler's business, and a
# cost model that charged for it would rank a program against a machine it
# will never run on.
#
# The three components:
#
#   * `read`   -- elements read from the inputs;
#   * `write`  -- elements written to the output;
#   * `alloc`  -- elements of fresh memory the output needs;
#   * `depth`  -- the serial depth, which is the reduction length (0 for
#                 the elementwise and the permutation rows);
#   * `absorb` -- 1 when the row is ELEMENTWISE, so it can absorb its
#                 input's output instead of materialising it.
#
# A role polynomial is a sum of products of the roles: `n*m`, `2*n*p`,
# `n*m+p`. `0` is the zero polynomial.

def _vol(c):
    """The element count of a class, as a role polynomial.

    A function class has no elements: it is a function, not an array.
    """
    if isinstance(c, str):
        return []
    d1, d2 = c
    return [(1, (d1, d2))]


def _poly(terms):
    """Render a list of (coefficient, roles) as a canonical role polynomial.

    The canonical form sorts the roles of a monomial by the interface's role
    order, sums the coefficients of a repeated monomial, drops a zero
    coefficient, and omits a coefficient of one. So `n*m` and `m*n` are the
    same polynomial, and two copies of it are `2*n*m`.
    """
    counts = {}
    for coef, roles in terms:
        key = tuple(sorted(roles, key=ROLES.index))
        counts[key] = counts.get(key, 0) + coef
    parts = []
    for key in sorted(counts, key=lambda k: tuple(ROLES.index(r) for r in k)):
        coef = counts[key]
        if coef == 0:
            continue
        mon = "*".join(key) if key else "1"
        parts.append(mon if coef == 1 else f"{coef}*{mon}")
    return "+".join(parts) if parts else "0"


def cost_rows():
    """The static cost of every row of the kingdom.

    The rules, one per operator:

    * an ELEMENTWISE row (`map_rows`, `map2_rows`) reads each of its
      matrices, writes its output, and allocates it -- and it ABSORBS its
      input, so a chain of maps costs one traversal rather than one per
      link;
    * `transpose` reads its input, writes and allocates its output;
    * `matmul` reads each operand once (the tiled traffic) and writes the
      result, and its serial depth is the inner role -- the length of the
      reduction;
    * a terminal IS the interface's own input: its output has a size, but
      nothing writes it and it allocates nothing;
    * a row function has no elements at all.
    """
    rows = []
    for r in operator_rows() + function_rows() + terminal_rows():
        name = r["name"]
        out = r["out"]
        args = r["args"]
        # The matrix arguments: the function-class arguments have no volume.
        mats = [a for a in args if not isinstance(a, str)]
        if name.startswith("map_rows@") or name.startswith("map2_rows@"):
            read = [t for m in mats for t in _vol(m)]
            absorb = "1"
            depth = "0"
        elif name.startswith("transpose@"):
            read = _vol(mats[0])
            absorb = "0"
            depth = "0"
        elif name.startswith("matmul@"):
            # The operands are (d1,d2) and (d2,d3): the inner role d2 is the
            # reduction's length, and the depth is that role.
            read = _vol(mats[0]) + _vol(mats[1])
            absorb = "0"
            depth = mats[0][1]
        else:
            # A terminal is the interface's own input: it reads nothing.
            read = []
            absorb = "0"
            depth = "0"
        vol = _vol(out)
        rows.append({
            "name": name,
            "vol": _poly(vol),
            "read": _poly(read),
            "write": _poly([] if _is_terminal(r) else vol),
            "alloc": _poly([] if _is_terminal(r) else vol),
            "depth": depth,
            "absorb": absorb,
        })
    return rows


def _is_terminal(row):
    """Whether a row is a terminal: the interface's own input."""
    return row["name"].endswith(".input")


def costs_text():
    """The text of `costs.tsv`."""
    out = ["name\tvol\tread\twrite\talloc\tdepth\tabsorb"]
    for c in cost_rows():
        out.append(f"{c['name']}\t{c['vol']}\t{c['read']}\t{c['write']}\t{c['alloc']}\t{c['depth']}\t{c['absorb']}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# The tables.
# --------------------------------------------------------------------------

def types_text():
    """The text of `types.tsv`."""
    out = ["ref\tdual\tslot\tform\tliterals\troot\tmeaning"]
    for i, t in enumerate(type_rows(), start=1):
        out.append(f"T{i:04d}\t{t['dual']}\t{t['slot']}\t{t['form']}\t"
                   f"{','.join(t['literals'])}\t{'yes' if t['root'] else 'no'}\t"
                   f"{t['meaning']}")
    return "\n".join(out) + "\n"


def functions_text():
    """The text of `functions.tsv`."""
    out = ["ref\tname\tkind\tout\targs\tspelling\tnote"]
    i = 0
    for r in operator_rows() + function_rows():
        i += 1
        out.append(f"F{i:05d}\t{r['name']}\tfunction\t{_class(r['out'])}\t"
                   f"{';'.join(_class(a) for a in r['args'])}\t{r['spelling']}\t{r['note']}")
    for r in terminal_rows():
        i += 1
        out.append(f"F{i:05d}\t{r['name']}\tterminal\t{_class(r['out'])}\t\t{r['spelling']}\t{r['note']}")
    return "\n".join(out) + "\n"


def _class(c):
    """A class, whether it is written as a name or as a pair of roles."""
    return c if isinstance(c, str) else mat_class(*c)


# --------------------------------------------------------------------------
# THE E-GRAPH. Its datatype, its ruleset and its constructor names are all
# DERIVED from the same rows, because the e-graph must not drift from the
# kingdom: a row with no constructor is a row the e-graph cannot hold, and a
# constructor with no row is a candidate the kingdom cannot express.
# --------------------------------------------------------------------------

def sort_name(dual):
    """The egglog sort of a class. One sort per class, so the SORT IS THE
    CLASS and egglog itself refuses an ill-shaped term."""
    if dual.startswith("mat."):
        return "FMat" + "".join(p.capitalize() for p in dual.split(".")[1:])
    if dual == FN:
        return "FRowFn"
    if dual == FN2:
        return "FRow2Fn"
    raise ValueError(f"no sort for {dual}")


def camel(op):
    return "".join(p.capitalize() for p in op.split("_"))


def camel_class(dual):
    """`mat.n.n` -> `MatNN`."""
    return "".join(p.capitalize() for p in dual.split("."))


def cls_of(row):
    """The class name of a row, whether it is written as a name or a pair."""
    out = row["out"]
    return out if isinstance(out, str) else mat_class(*out)


def constructor_name(row):
    """The egglog constructor of a kingdom row, derived from its name.

    * `map_rows@mat.n.m` -> `AMapRowsMatNM`
    * `matmul@mat.n.mxmat.m.p` -> `AMatmulMatNMxMatMP`
    * `fn.arr_f32_arr_f32.id` -> `FRowFnId`
    * `mat.n.m.input` -> `AInputMatNM`
    """
    if "@" in row["name"]:
        op, classes = row["name"].split("@", 1)
        parts = [camel_class(part) for part in classes.split("x")]
        return "A" + camel(op) + "".join(parts)
    # A row of a function class, or a terminal.
    if cls_of(row) in (FN, FN2):
        prefix = "FRowFn" if cls_of(row) == FN else "FRow2Fn"
        suffix = row["name"].rsplit(".", 1)[-1]
        return prefix + suffix.capitalize()
    # The only matrix-class rows without an operator are the terminals.
    return "AInput" + camel_class(cls_of(row))


def egraph_rows():
    """Every row that becomes an e-graph constructor: the operators, the
    function classes' own rows, and the terminals."""
    return operator_rows() + function_rows() + terminal_rows()


def egraph_datatype():
    """The datatype: one sort per class, one constructor per row."""
    out = ["; The shape kingdom's e-graph datatype, generated by gen_tables.py.",
           "; One sort per class: the sort IS the class.", ""]
    for d in type_rows():
        out.append(f"(sort {sort_name(d['dual'])})")
    out.append("")
    for d in type_rows():
        out.append(f"; ---- {d['dual']} " + "-" * max(1, 60 - len(d['dual'])))
        for row in egraph_rows():
            if cls_of(row) != d["dual"]:
                continue
            args = [a if isinstance(a, str) else mat_class(*a) for a in row["args"]]
            arg_sorts = " ".join(sort_name(a) for a in args)
            out.append(f"(constructor {constructor_name(row)} ({arg_sorts}) {sort_name(d['dual'])})")
        out.append("")
    return "\n".join(out)


def transposed_product_rule(a, b, c):
    """The rule text for `(Y' . X')' = X . Y`, where `X` has class
    `(a,b)` and `Y` has class `(b,c)`, and `'` is the transpose.

    The left side computes the product the transposed way and transposes the
    result; the right side computes it directly. Both are the same matrix:
    the same products, summed over the same inner role in the same order, so
    the rewrite is bit-identical.
    """
    t_ca = constructor_name({"name": f"transpose@{mat_class(c, a)}"})
    mm_cb_ba = constructor_name({"name": f"matmul@{mat_class(c, b)}x{mat_class(b, a)}"})
    t_bc = constructor_name({"name": f"transpose@{mat_class(b, c)}"})
    t_ab = constructor_name({"name": f"transpose@{mat_class(a, b)}"})
    mm_ab_bc = constructor_name({"name": f"matmul@{mat_class(a, b)}x{mat_class(b, c)}"})
    return f"({t_ca} ({mm_cb_ba} ({t_bc} u) ({t_ab} v))) ({mm_ab_bc} v u)"


def egraph_rules():
    """The ruleset, derived from the operator templates.

    Three identities, one per shape-preserving operator, instantiated per
    class:

    * `map` with the identity row function is the matrix;
    * `map2` with the identity binary row function is the FIRST matrix (the
      kingdom's binary functions ignore their second operand);
    * `transpose` twice is the matrix.

    All three are bit-identical rewrites: they permute or copy values and
    never do arithmetic, so there is no floating-point question.

    And one more, over the products:

    * `(Y' . X')' = X . Y` -- the product computed the transposed way and
      then transposed back is the product itself.

    That one is also bit-identical (the same products, summed over the same
    inner role in the same order), and it is the one rule that can WIN: the
    right side has no transposes, so when a program computes the same
    product twice -- once directly and once the transposed way -- the
    rewrite makes the two the same expression, and the compiler's own CSE
    then does the rest. The other three rules are ones Futhark's simplifier
    already applies, so they cost nothing at the compiler.
    """
    out = ["; The shape kingdom's e-graph ruleset, generated by gen_tables.py.",
           "(ruleset futhark_shape)", ""]
    for d1, d2 in MATRIX_CLASSES:
        c = (d1, d2)
        cls = mat_class(d1, d2)
        m = constructor_name({"name": f"map_rows@{cls}"})
        m2 = constructor_name({"name": f"map2_rows@{cls}"})
        t = constructor_name({"name": f"transpose@{cls}"})
        out.append(f"; `map` with the identity row function is the matrix ({cls})")
        out.append(f"(rewrite ({m} (FRowFnId) x) x :ruleset futhark_shape)")
        out.append(f"; `map2` with the identity binary row function is the first matrix ({cls})")
        out.append(f"(rewrite ({m2} (FRow2FnId) x y) x :ruleset futhark_shape)")
        if swap(c) in MATRIX_CLASSES:
            swapped = mat_class(*swap(c))
            t_swapped = constructor_name({"name": f"transpose@{swapped}"})
            out.append(f"; `transpose` is an involution ({cls})")
            out.append(f"(rewrite ({t_swapped} ({t} x)) x :ruleset futhark_shape)")
        out.append("")
    for a, b, c in itertools.product(ROLES, ROLES, ROLES):
        out.append(f"; the product of ({mat_class(a, b)}) and ({mat_class(b, c)}), "
                   f"transposed, is the product itself")
        out.append(f"(rewrite {transposed_product_rule(a, b, c)} :ruleset futhark_shape)")
        out.append("")
    return "\n".join(out)


def egraph_texts():
    """The text of every generated e-graph artefact, by file name."""
    rows = ["# row name\tconstructor\tsort"]
    for row in egraph_rows():
        rows.append(f"{row['name']}\t{constructor_name(row)}\t{sort_name(cls_of(row))}")
    sorts = ["# class\tsort"]
    for d in type_rows():
        sorts.append(f"{d['dual']}\t{sort_name(d['dual'])}")
    return {
        "egraph.datatype": egraph_datatype(),
        "egraph.rules": egraph_rules(),
        "egraph.rows": "\n".join(rows) + "\n",
        "egraph.sorts": "\n".join(sorts) + "\n",
    }


# --------------------------------------------------------------------------
# THE WITNESS TESTS. They re-read the emitted tables and re-check every shape
# relation from the TEXT, against a relation model written independently of
# the enumeration above. This is the trust boundary the design moves to: the
# engine no longer checks the shape logic, so something must.
# --------------------------------------------------------------------------

def read_tables(types_path, functions_path):
    with open(types_path) as f:
        types = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]
    with open(functions_path) as f:
        funcs = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]
    return types, funcs


def witness_camel_class(dual):
    """The class name as the constructors spell it -- a SECOND derivation.

    The generator's `camel_class` builds the name with `str.capitalize`; this
    one spells the prefixes and the role names out by hand, so the two cannot
    be wrong in the same way. A disagreement is itself a finding.
    """
    prefixes = {"mat": "Mat"}
    roles = {"n": "N", "m": "M", "p": "P"}
    parts = dual.split(".")
    return prefixes[parts[0]] + "".join(roles[r] for r in parts[1:])


def witness(types_path, functions_path):
    """Check the emitted tables. Returns the list of failures (empty = pass)."""
    bad = []
    types, funcs = read_tables(types_path, functions_path)

    declared = {t[1] for t in types}
    expected_classes = {mat_class(*c) for c in MATRIX_CLASSES} | {FN, FN2}
    if declared != expected_classes:
        bad.append(f"W1 types.tsv declares {sorted(declared)}, expected {sorted(expected_classes)}")

    # Every class a row names is a declared class.
    demanded = set()
    for row in funcs:
        out, args = row[3], [a for a in row[4].split(";") if a]
        for name in [out] + args:
            if name not in declared:
                bad.append(f"W2 row {row[1]} names the undeclared class {name}")
        demanded.update(args)
        demanded.add(out)

    # The shape relations, re-derived from the table text.
    for row in funcs:
        name, out = row[1], row[3]
        args = [a for a in row[4].split(";") if a]
        if name.startswith("matmul@"):
            a0, a1 = parse_class(args[0]), parse_class(args[1])
            o = parse_class(out)
            if a0 is None or a1 is None or o is None:
                bad.append(f"W3 matmul row {name} does not name matrix classes")
            elif not (a0[1] == a1[0]):
                bad.append(f"W3 matmul row {name}: the inner roles disagree ({a0[1]} vs {a1[0]})")
            elif o != (a0[0], a1[1]):
                bad.append(f"W3 matmul row {name}: the result class {o} is not ({a0[0]}, {a1[1]})")
        elif name.startswith("transpose@"):
            a0, o = parse_class(args[0]), parse_class(out)
            if a0 is None or o is None:
                bad.append(f"W4 transpose row {name} does not name matrix classes")
            elif o != swap(a0):
                bad.append(f"W4 transpose row {name}: the result class {o} is not the swap of {a0}")
        elif name.startswith("map_rows@") or name.startswith("map2_rows@"):
            c = parse_class(out)
            array_args = [parse_class(a) for a in args if parse_class(a)]
            if c is None or not array_args:
                bad.append(f"W5 row {name} does not name a matrix class")
            elif any(a != c for a in array_args):
                bad.append(f"W5 row {name}: the array arguments do not share the result class {c}")

    # Every demanded class has a terminal.
    terminals = {row[3] for row in funcs if row[2] == "terminal"}
    for c in sorted(demanded):
        if c not in terminals:
            bad.append(f"W6 the demanded class {c} has no terminal")

    # The class set is closed under the operators' effects.
    pairs = {parse_class(t[1]) for t in types if parse_class(t[1])}
    for c in pairs:
        if swap(c) not in pairs:
            bad.append(f"W7 the class set is not closed under transpose at {c}")
    for a, b in itertools.product(sorted(pairs), sorted(pairs)):
        if a[1] == b[0] and (a[0], b[1]) not in pairs:
            bad.append(f"W7 the class set is not closed under matmul at {a} x {b}")

    # Every spelling mentions each argument exactly once, and no more.
    for row in funcs:
        if row[2] != "function":
            continue
        args = [a for a in row[4].split(";") if a]
        for i in range(len(args)):
            if row[5].count("{%d}" % i) != 1:
                bad.append(f"W8 row {row[1]}: the spelling does not use {{{i}}} exactly once")
        if "{" in row[5].replace("{0}", "").replace("{1}", "").replace("{2}", ""):
            bad.append(f"W8 row {row[1]}: the spelling holds an unexpected placeholder")

    # The placeholders must appear in the order of the ARGUMENTS. Futhark's
    # application is positional, so a spelling that puts `{1}` before `{0}`
    # applies the operator to its arguments the wrong way round -- and a
    # counting check cannot see that, because the counts are unchanged.
    for row in funcs:
        if row[2] != "function" or "@" not in row[1]:
            continue
        args = [a for a in row[4].split(";") if a]
        seen = [int(m.group(1)) for m in PLACEHOLDER_RE.finditer(row[5])]
        if seen != sorted(seen):
            bad.append(
                f"W8 row {row[1]}: the spelling uses the arguments out of order "
                f"({seen}), but Futhark's application is positional"
            )
        if seen != list(range(len(args))):
            bad.append(f"W8 row {row[1]}: the spelling uses the arguments {seen}, not 0..{len(args) - 1}")

    # The spelling must be the OPERATOR's template. The templates live in one
    # table (OPERATORS); the rows are built elsewhere, and the two could
    # drift. This is the check that says they have not.
    templates = {op: spelling for op, _fn_class, spelling in OPERATORS}
    for row in funcs:
        if row[2] != "function" or "@" not in row[1]:
            continue
        op = row[1].split("@")[0]
        if op not in templates:
            bad.append(f"W8 row {row[1]}: there is no template for the operator {op}")
        elif row[5] != templates[op]:
            bad.append(
                f"W8 row {row[1]}: the spelling is {row[5]!r}, but the operator "
                f"{op}'s template is {templates[op]!r}"
            )

    # Each operator takes the FUNCTION CLASS it needs. This is the check that
    # caught the first draft of this kingdom: Futhark's `map2` is curried, so
    # `map2_rows` must take a TWO-argument row function, and a one-argument
    # one does not type-check there.
    for op, fn_class, _spelling in OPERATORS:
        for row in funcs:
            if not row[1].startswith(op + "@"):
                continue
            args = [a for a in row[4].split(";") if a]
            if fn_class is None:
                continue
            if not args or args[0] != fn_class:
                bad.append(
                    f"W9 row {row[1]}: the operator {op} takes the function class "
                    f"{args[0] if args else '(none)'}, but it needs {fn_class}"
                )

    return bad


# --------------------------------------------------------------------------

def witness_egraph(dir_path):
    """Check the generated e-graph: the datatype and the ruleset.

    The e-graph must not drift from the kingdom, so this checks the two
    against each other: every kingdom row has exactly one constructor, every
    constructor names declared sorts, and every rule mentions only
    constructors that exist.
    """
    bad = []
    with open(os.path.join(dir_path, "egraph.datatype")) as f:
        datatype = f.read()
    with open(os.path.join(dir_path, "egraph.rules")) as f:
        rules = f.read()

    sorts = {m.group(1) for m in (SORT_RE.match(l) for l in datatype.splitlines()) if m}
    ctors = {}
    for line in datatype.splitlines():
        m = CTOR_RE.match(line)
        if not m:
            continue
        name, arg_sorts, result = m.group(1), m.group(2).split(), m.group(3)
        if name in ctors:
            bad.append(f"W10 the constructor {name} is declared twice")
        ctors[name] = (arg_sorts, result)
        for s in arg_sorts + [result]:
            if s not in sorts:
                bad.append(f"W10 the constructor {name} names the undeclared sort {s}")

    # Every kingdom row has a constructor, and every constructor is a row.
    rows = {constructor_name(r) for r in egraph_rows()}
    if rows != set(ctors):
        missing = sorted(rows - set(ctors))
        extra = sorted(set(ctors) - rows)
        bad.append(f"W10 the datatype and the rows disagree: missing {missing}, extra {extra}")

    # The ruleset's name, and the rules paired with the comment that says
    # which class each is for.
    comments = []
    rewrite_lines = []
    pending = ""
    for line in rules.splitlines():
        s = line.strip()
        if s.startswith("(rewrite "):
            rewrite_lines.append(s)
            comments.append(pending)
            pending = ""
        elif s.startswith(";"):
            pending = s
    if "(ruleset futhark_shape)" not in rules:
        bad.append("W11 the ruleset is not named futhark_shape")
    for rule in rewrite_lines:
        if not rule.endswith(":ruleset futhark_shape)"):
            bad.append(f"W11 the rule is not in the futhark_shape ruleset: {rule}")

    # Every rule must match one of the THREE SCHEMAS below. The schemas are
    # written out by hand, independently of the code that emits the rules:
    # a rule that does not match its schema is a rule the e-graph would use
    # to justify a candidate that computes something else.
    #
    #   {CLASS} is the class in the constructors' spelling,
    #   {SWAP}  is that class with its two roles exchanged.
    schemas = {
        "map_rows": "(rewrite (AMapRows{CLASS} (FRowFnId) x) x :ruleset futhark_shape)",
        "map2_rows": "(rewrite (AMap2Rows{CLASS} (FRow2FnId) x y) x :ruleset futhark_shape)",
        "transpose": "(rewrite (ATranspose{SWAP} (ATranspose{CLASS} x)) x :ruleset futhark_shape)",
    }
    # The TRANSPOSED-PRODUCT rules: `(Y' . X')' = X . Y`. The comment names
    # the two classes whose product it is, and the schema is written out by
    # hand -- the classes of the transposes and of the two products are
    # worked out here, independently of the code that emits the rules.
    seen_products = {}
    for comment, rule in zip(comments, rewrite_lines):
        pm = RULE_PRODUCT_RE.match(comment)
        if pm:
            x_cls, y_cls = pm.group(1), pm.group(2)
            x = parse_class(x_cls)
            y = parse_class(y_cls)
            if x is None or y is None:
                bad.append(f"W11 the transposed-product rule names a class that is not one: {rule}")
                continue
            (a, b), (inner, c) = x, y
            if b != inner:
                bad.append(f"W11 the transposed-product rule's classes do not share the inner role: {rule}")
                continue
            expected = (
                f"(rewrite (ATranspose{witness_camel_class(mat_class(c, a))} "
                f"(AMatmul{witness_camel_class(mat_class(c, b))}{witness_camel_class(mat_class(b, a))} "
                f"(ATranspose{witness_camel_class(mat_class(b, c))} u) "
                f"(ATranspose{witness_camel_class(mat_class(a, b))} v))) "
                f"(AMatmul{witness_camel_class(mat_class(a, b))}{witness_camel_class(mat_class(b, c))} v u) "
                f":ruleset futhark_shape)"
            )
            if rule != expected:
                bad.append(f"W11 the transposed-product rule does not match its schema: {rule}")
                continue
            if (x_cls, y_cls) in seen_products:
                bad.append(f"W11 the transposed-product rule for ({x_cls}, {y_cls}) is stated twice")
            seen_products[(x_cls, y_cls)] = rule
            continue

    seen_rules = {}
    for comment, rule in zip(comments, rewrite_lines):
        if RULE_PRODUCT_RE.match(comment):
            continue
        m = RULE_CLASS_RE.match(comment)
        if not m:
            bad.append(f"W11 the rule {rule!r} has no class in its comment")
            continue
        cls = m.group(1)
        spelled = witness_camel_class(cls)
        swapped = witness_camel_class(mat_class(*swap(parse_class(cls))))
        expected = {
            "map_rows": schemas["map_rows"].format(CLASS=spelled),
            "map2_rows": schemas["map2_rows"].format(CLASS=spelled),
            "transpose": schemas["transpose"].format(CLASS=spelled, SWAP=swapped),
        }
        matched = [op for op, text in expected.items() if text == rule]
        if not matched:
            bad.append(f"W11 the rule for {cls} does not match any schema: {rule}")
            continue
        op = matched[0]
        if (op, cls) in seen_rules:
            bad.append(f"W11 the rule {op} for {cls} is stated twice")
        seen_rules[(op, cls)] = rule

    # Every class must have all three rules, and no rule may mention a class
    # the kingdom does not have.
    for d1, d2 in MATRIX_CLASSES:
        cls = mat_class(d1, d2)
        for op in ("map_rows", "map2_rows", "transpose"):
            if (op, cls) not in seen_rules:
                bad.append(f"W11 the class {cls} has no {op} rule")
    for (op, cls) in seen_rules:
        if parse_class(cls) not in MATRIX_CLASSES:
            bad.append(f"W11 the rule {op} mentions the class {cls}, which the kingdom does not have")

    # Every product of two classes that share an inner role must have its
    # transposed-product rule.
    for a, b, c in itertools.product(ROLES, ROLES, ROLES):
        key = (mat_class(a, b), mat_class(b, c))
        if key not in seen_products:
            bad.append(f"W11 the transposed-product rule for {key} is missing")

    # The bridge's mapping covers every row, and the sorts cover every class.
    with open(os.path.join(dir_path, "egraph.rows")) as f:
        mapped = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]
    row_names = {r["name"] for r in egraph_rows()}
    if {m[0] for m in mapped} != row_names:
        bad.append("W12 the bridge's mapping does not cover exactly the kingdom's rows")
    # The mapping must be BIJECTIVE. A row mapped to another row's
    # constructor -- `fn.arr_f32_arr_f32.add1` to `FRowFnId`, say -- would
    # pass a check that only looks at the constructor's sort, and would then
    # make the Rust bridge's reverse map silently lose a row.
    seen_ctors = {}
    seen_rows = set()
    for name, ctor, sort in mapped:
        if name in seen_rows:
            bad.append(f"W12 the row {name} is mapped twice")
        seen_rows.add(name)
        if ctor in seen_ctors:
            bad.append(
                f"W12 the constructor {ctor} is mapped to two rows: "
                f"{seen_ctors[ctor]} and {name}"
            )
        else:
            seen_ctors[ctor] = name
        if ctor not in ctors:
            bad.append(f"W12 the row {name} maps to the unknown constructor {ctor}")
            continue
        if ctors.get(ctor, ([], ""))[1] != sort:
            bad.append(f"W12 the row {name} maps to a constructor of another sort")
        # The constructor's ARGUMENT SORTS must be the row's argument
        # classes' sorts. This is what ties the datatype to the table: a
        # constructor that took its arguments in another order would have the
        # right sorts and the wrong meaning.
        row = next((r for r in egraph_rows() if r["name"] == name), None)
        if row is not None:
            want_sorts = [sort_name(a if isinstance(a, str) else mat_class(*a)) for a in row["args"]]
            if list(ctors[ctor][0]) != want_sorts:
                bad.append(
                    f"W12 the row {name} maps to a constructor taking "
                    f"{list(ctors[ctor][0])}, but the row's arguments are {want_sorts}"
                )
    with open(os.path.join(dir_path, "egraph.sorts")) as f:
        sort_map = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]
    if {s[0] for s in sort_map} != {d["dual"] for d in type_rows()}:
        bad.append("W12 the sort map does not cover exactly the kingdom's classes")
    # The sort map must be one-to-one too: two classes sharing a sort would
    # make the sort stop being the class.
    seen_sorts = {}
    for cls, sort in sort_map:
        if sort in seen_sorts:
            bad.append(f"W12 the sort {sort} is used by two classes: {seen_sorts[sort]} and {cls}")
        else:
            seen_sorts[sort] = cls
    return bad


def first_difference(got, want):
    """Where two byte strings first differ, for the report."""
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            line = got.count(b"\n", 0, i) + 1
            return f"line {line} differs"
    if len(got) != len(want):
        return f"the lengths differ ({len(got)} and {len(want)})"
    return "they are equal"


def _witness_poly(terms):
    """The canonical text of a role polynomial -- a SECOND implementation.

    The generator's `_poly` folds the terms into a dictionary and joins them
    in the order the dictionary yields; this one sorts the monomials by the
    role order first. The two are close relatives, and the difference is
    deliberate but limited: **what this witness catches is a transcription
    slip** -- a rule's cost written down wrongly, a coefficient lost -- and
    **not a modelling error**, because the rules themselves are restated
    from the same understanding in both places. A rule that is wrong in the
    generator's *model* is wrong in the witness's model too, and neither
    notices.
    """
    counts = {}
    for coef, roles in terms:
        key = tuple(sorted(roles, key=ROLES.index))
        counts[key] = counts.get(key, 0) + coef
    keys = sorted(counts, key=lambda k: (tuple(ROLES.index(r) for r in k), k))
    parts = []
    for key in keys:
        coef = counts[key]
        if coef == 0:
            continue
        mon = "*".join(key) if key else "1"
        parts.append(mon if coef == 1 else f"{coef}*{mon}")
    return "+".join(parts) if parts else "0"


def witness_costs(functions_path, costs_path):
    """Check the emitted cost table against an INDEPENDENT derivation.

    The cost of a row is re-derived here from the row's own name and
    classes -- written out by hand here, not by calling the generator's
    `cost_rows` -- so the two cannot be wrong in the same way. A
    disagreement is itself a finding.

    The rules, re-stated from scratch:

    * `map_rows@mat.X.Y`   reads X*Y, writes X*Y, allocates X*Y, depth 0,
      and it absorbs its input;
    * `map2_rows@mat.X.Y`  reads 2*X*Y, writes X*Y, allocates X*Y, depth 0,
      and it absorbs its inputs;
    * `transpose@mat.X.Y`  reads X*Y, writes X*Y, allocates X*Y, depth 0;
    * `matmul@mat.X.mxmat.m.Y` reads X*m + m*Y (the TILED traffic: each
      operand element once), writes X*Y, allocates X*Y, and its depth is
      the inner role m;
    * a terminal has a volume (it holds an array) but writes nothing and
      allocates nothing -- it IS the interface's input;
    * a row function has no elements at all.
    """
    bad = []
    with open(functions_path) as f:
        funcs = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]
    with open(costs_path) as f:
        lines = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]

    costs = {}
    for row in lines:
        if len(row) != 7:
            bad.append(f"W13 the cost row {row} has {len(row)} columns, wants 7")
            continue
        if row[0] in costs:
            bad.append(f"W13 the row {row[0]} has two cost rows")
            continue
        costs[row[0]] = row[1:]

    for row in funcs:
        name = row[1]
        out = row[3]
        args = [a for a in row[4].split(";") if a]
        if name not in costs:
            bad.append(f"W13 the row {name} has no cost")
            continue
        vol, read, write, alloc, depth, absorb = costs[name]
        classes = [parse_class(a) for a in args]
        mats = [c for c in classes if c is not None]
        o = parse_class(out)
        if name.startswith("map_rows@"):
            want = (_witness_poly([(1, mats[0])]),
                    _witness_poly([(1, mats[0])]),
                    _witness_poly([(1, o)]), _witness_poly([(1, o)]),
                    "0", "1")
        elif name.startswith("map2_rows@"):
            want = (_witness_poly([(1, o)]),
                    _witness_poly([(1, mats[0]), (1, mats[0])]),
                    _witness_poly([(1, o)]), _witness_poly([(1, o)]),
                    "0", "1")
        elif name.startswith("transpose@"):
            want = (_witness_poly([(1, mats[0])]),
                    _witness_poly([(1, mats[0])]),
                    _witness_poly([(1, o)]), _witness_poly([(1, o)]),
                    "0", "0")
        elif name.startswith("matmul@"):
            a0, a1 = mats[0], mats[1]
            want = (_witness_poly([(1, o)]),
                    _witness_poly([(1, a0), (1, a1)]),
                    _witness_poly([(1, o)]), _witness_poly([(1, o)]),
                    a0[1], "0")
        elif name.endswith(".input"):
            # A terminal HOLDS an array: it has a volume, but nothing writes
            # it and it allocates nothing.
            want = (_witness_poly([(1, o)]), "0", "0", "0", "0", "0")
        else:
            # A row function: no elements.
            want = ("0", "0", "0", "0", "0", "0")
        got = (vol, read, write, alloc, depth, absorb)
        if got != want:
            bad.append(f"W13 the cost of {name} is {got}, expected {want}")

    for name in costs:
        if not any(row[1] == name for row in funcs):
            bad.append(f"W13 the cost row {name} names no row of the kingdom")
    return bad


# --------------------------------------------------------------------------
# THE PIXEL FAMILY: a picture is a map over the pixels of a scalar program.
# --------------------------------------------------------------------------
#
# A cuneus image shader is one compute invocation per pixel: it reads the
# pixel's coordinates and a block of uniforms, runs bounded loops over scalar
# and vector arithmetic, and stores one colour. In Futhark that is
#
#     map (\yi -> map (\xi -> <the pixel's program>) (iota w)) (iota h)
#
# and the pixel's program is a chain of `let`s and bounded `loop`s over f32
# values, ending in a colour `[r, g, b]`. Vector maths is spelled per lane
# (a `vec3` is three scalars), which is what Futhark's own compiler does to it.
#
# The family is GENERIC: nothing below names a shader. An INSTANCE declares an
# interface -- the WGSL source whose uniform blocks become the program's
# inputs and whose float literals join the constant pool -- and the family's
# parameters (`PIXEL_SCOPES`, `PIXEL_ARITIES`, `PIXEL_DEPTH`, `PIXEL_COUNTS`).
#
# EVERY RULE IS A TYPE, so every gene expresses a legal Futhark program:
#
# * **Scope is a type.** `f32.s{k}` is "an f32 in a scope where `v0 .. v{k-1}`
#   are bound". Its terminals are the variables `v0 .. v{k-1}` and the
#   interface's inputs, and its functions take arguments of the SAME scope.
#   A gene cannot name an unbound variable, because no row of `f32.s{k}`
#   spells `v{k}`.
# * **A binding moves the scope.** `let v{k} = e in rest` takes an `f32.s{k}`
#   and a block of scope `k+1`. A loop of state arity S binds `v{k} ..
#   v{k+S-1}` (its state) and, inside the body only, `v{k+S}` (the trip
#   counter as an f32): its body is a block of scope `k+S+1` returning an
#   S-tuple, and the rest a block of scope `k+S`.
# * **A loop is bounded by its type.** The trip count is a `count`, whose
#   terminals are a fixed list of literals, and a loop body has a nesting
#   depth `d` in its type: a body at depth `PIXEL_DEPTH` has no loop row. So
#   the work of a pixel is at most `max(count) ** PIXEL_DEPTH` trips of a body
#   whose size the gene bounds, and no gene can spell a runaway kernel.
# * **The result arity is a type.** `body{S}.d{d}.s{k}` ends in an S-tuple,
#   the loop that owns it has state arity S; `blk.s{k}` ends in `[r, g, b]`.
#
# Every demanded type has a terminal: the f32 scopes their inputs, a block
# the black pixel, a loop body the zero tuple, the root the black image.

PIXEL_SCOPES = 40          # the most bound variables: scopes s0 .. s40
PIXEL_ARITIES = (1, 3, 5)  # the loop state arities: a scalar, a vec3, five lanes
PIXEL_DEPTH = 2            # the deepest loop nest
PIXEL_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64)

# The generic part of the constant pool, before the instance's own literals.
PIXEL_GENERIC_LITERALS = ("0.0", "1.0", "-1.0", "0.5", "2.0", "3.0", "0.25", "10.0", "0.1")

# The instances: the WGSL source of each, relative to the repository root.
PIXEL_INSTANCES = {
    "cuneus": "tests/corpus/cuneus_visual/cuneus.wgsl",
    "volumepassage": "tests/corpus/cuneus_visual/volumepassage.wgsl",
}

REPO = os.path.dirname(os.path.dirname(HERE))

# The scalar operators of every scope: (name, arity, spelling).
PIXEL_OPERATORS = [
    ("add", 2, "{0} + {1}"),
    ("sub", 2, "{0} - {1}"),
    ("mul", 2, "{0} * {1}"),
    ("div", 2, "{0} / {1}"),
    ("max", 2, "f32.max {0} {1}"),
    ("min", 2, "f32.min {0} {1}"),
    ("pow", 2, "{0} ** {1}"),
    ("mod", 2, "{0} % {1}"),
    ("neg", 1, "-{0}"),
    ("abs", 1, "f32.abs {0}"),
    ("sqrt", 1, "f32.sqrt {0}"),
    ("sin", 1, "f32.sin {0}"),
    ("cos", 1, "f32.cos {0}"),
    ("tan", 1, "f32.tan {0}"),
    ("tanh", 1, "f32.tanh {0}"),
    ("exp", 1, "f32.exp {0}"),
    ("log", 1, "f32.log {0}"),
    ("floor", 1, "f32.floor {0}"),
    ("trunc", 1, "f32.trunc {0}"),
    ("sel", 3, "if {0} >= 0f32 then {1} else {2}"),
]

STRUCT_RE = re.compile(r"struct\s+(\w+)\s*\{([^}]*)\}", re.S)
FIELD_RE = re.compile(r"(\w+)\s*:\s*([\w<>]+)")
UNIFORM_RE = re.compile(r"@group\((\d+)\)\s*@binding\((\d+)\)\s*var<uniform>\s+(\w+)\s*:\s*(\w+)\s*;")
FLOAT_RE = re.compile(r"(?<![\w.])-?\d+\.\d+(?:[eE][-+]?\d+)?")


def strip_comments(src):
    return re.sub(r"//[^\n]*", "", src)


def pixel_interface(shader_path):
    """The interface of an instance, read from its WGSL source: every f32
    field of every uniform block, in declaration order, with the block's
    group and binding and the field's word offset under WGSL's uniform layout
    (a scalar is 4 bytes aligned to 4, a vec2 8 aligned to 8, a vec3 12 and a
    vec4 16 aligned to 16). A `vecN<f32>` field is N inputs, `name_x`,
    `name_y`, `name_z`, `name_w`. A field whose name starts with `_` is padding
    and is not an input. Any other field type is a finding: the offsets would
    be wrong."""
    src = strip_comments(open(shader_path).read())
    structs = {m.group(1): FIELD_RE.findall(m.group(2)) for m in STRUCT_RE.finditer(src)}
    layout = {"f32": (4, 4, 1), "u32": (4, 4, 0), "i32": (4, 4, 0),
              "vec2<f32>": (8, 8, 2), "vec3<f32>": (12, 16, 3), "vec4<f32>": (16, 16, 4)}
    out = []
    for g, b, _var, ty in UNIFORM_RE.findall(src):
        if ty not in structs:
            raise ValueError(f"the uniform block type {ty} has no struct in {shader_path}")
        offset = 0
        for name, fty in structs[ty]:
            if fty not in layout:
                raise ValueError(f"the uniform field {ty}.{name} is a {fty}, which the interface does not lay out")
            size, align, lanes = layout[fty]
            offset = (offset + align - 1) // align * align
            if not name.startswith("_"):
                if fty == "f32":
                    out.append({"name": name, "group": int(g), "binding": int(b), "word": offset // 4, "spelling": f"u_{name}"})
                for i in range(lanes if fty.startswith("vec") else 0):
                    lane = "xyzw"[i]
                    out.append({"name": f"{name}.{lane}", "group": int(g), "binding": int(b), "word": offset // 4 + i,
                                "spelling": f"u_{name}_{lane}"})
            offset += size
    names = [f["spelling"] for f in out]
    if len(set(names)) != len(names):
        raise ValueError(f"two uniform fields share a name in {shader_path}: {names}")
    return out


def f32_bits(text):
    import struct
    return struct.unpack("<I", struct.pack("<f", float(text)))[0]


def pixel_literals(shader_path):
    """The constant pool: the generic literals, then every float literal of
    the shader's source in order of appearance, each value once (by its f32
    bits), each spelled as an f32 literal."""
    src = strip_comments(open(shader_path).read())
    seen, out = set(), []
    for text in list(PIXEL_GENERIC_LITERALS) + FLOAT_RE.findall(src):
        bits = f32_bits(text)
        if bits in seen:
            continue
        seen.add(bits)
        out.append(f"{text}f32")
    return out


def s_dual(k):
    return f"f32.s{k}"


def blk_dual(k):
    return f"blk.s{k}"


def body_dual(s, d, k):
    return f"body{s}.d{d}.s{k}"


def pixel_rows(interface):
    """Every row of the family, with its output and argument duals. The
    duals are named, not enumerated: the types are the duals the rows
    mention, and the witness tests re-derive every relation from the names."""
    rows = []
    inputs = ["px", "py", "u_width", "u_height"] + [f["spelling"] for f in interface]
    for k in range(PIXEL_SCOPES + 1):
        e = s_dual(k)
        for name, arity, spelling in PIXEL_OPERATORS:
            rows.append({"name": f"{name}@{e}", "kind": "function", "out": e, "args": [e] * arity,
                         "spelling": spelling, "note": f"{name}, in scope {k}"})
    # The block and loop-body rows. A body type is emitted only when a loop
    # row demands it, so the closure below starts from the root.
    def binders(out_of, k, d, rest_of):
        """The let and loop rows of a block (`d` = 0) or a body at depth d."""
        out = []
        if k + 1 <= PIXEL_SCOPES:
            out.append({"name": f"let@{out_of(k)}", "kind": "function", "out": out_of(k),
                        "args": [s_dual(k), rest_of(k + 1)],
                        "spelling": f"let v{k} = {{0}} in {{1}}", "note": f"bind v{k}"})
        # A block: a let-chain of its own that returns S values and releases
        # its scope. It adds no loop depth and no trips, so the work bound is
        # unchanged; its body is a body type at the SAME depth.
        for s in PIXEL_ARITIES:
            if k + s > PIXEL_SCOPES:
                continue
            state = ", ".join(f"v{k + i}" for i in range(s))
            out.append({"name": f"block{s}@{out_of(k)}", "kind": "function", "out": out_of(k),
                        "args": [body_dual(s, d, k), rest_of(k + s)],
                        "spelling": f"let ({state}) = {{0}} in {{1}}",
                        "note": f"a block of {s} values binding v{k}..v{k + s - 1}; its own bindings end with it"})
        if d < PIXEL_DEPTH:
            for s in PIXEL_ARITIES:
                if k + s + 1 > PIXEL_SCOPES:
                    continue
                state = ", ".join(f"v{k + i}" for i in range(s))
                inits = ", ".join(f"{{{i}}}" for i in range(s))
                spelling = (f"let ({state}) = loop ({state}) = ({inits}) for i{k} < {{{s}}} do "
                            f"let v{k + s} = f32.i64 i{k} in {{{s + 1}}} in {{{s + 2}}}")
                out.append({"name": f"loop{s}@{out_of(k)}", "kind": "function", "out": out_of(k),
                            "args": [s_dual(k)] * s + ["count", body_dual(s, d + 1, k + s + 1), rest_of(k + s)],
                            "spelling": spelling,
                            "note": f"a loop of state arity {s} binding v{k}..v{k + s - 1}, its counter v{k + s} in the body"})
        return out

    for k in range(PIXEL_SCOPES + 1):
        rows += binders(blk_dual, k, 0, blk_dual)
        rows.append({"name": f"rgb@{blk_dual(k)}", "kind": "function", "out": blk_dual(k),
                     "args": [s_dual(k)] * 3, "spelling": "[{0}, {1}, {2}]", "note": "the pixel's colour"})
    demanded = sorted({a for r in rows for a in r["args"] if a.startswith("body")})
    done = set()
    while demanded:
        dual = demanded.pop()
        if dual in done:
            continue
        done.add(dual)
        s, d, k = parse_body(dual)
        here = binders(lambda kk, s=s, d=d: body_dual(s, d, kk), k, d, lambda kk, s=s, d=d: body_dual(s, d, kk))
        here.append({"name": f"tuple@{dual}", "kind": "function", "out": dual, "args": [s_dual(k)] * s,
                     "spelling": "(" + ", ".join(f"{{{i}}}" for i in range(s)) + ")",
                     "note": f"the {s} values a loop body or a block returns"})
        rows += here
        for r in here:
            for a in r["args"]:
                if a.startswith("body") and a not in done:
                    demanded.append(a)
    rows.append({"name": "image", "kind": "function", "out": "image", "args": [blk_dual(0)],
                 "spelling": "map (\\yi -> map (\\xi -> let px = col0 + f32.i64 xi in let py = row0 + f32.i64 yi in {0}) (iota cols)) (iota rows)",
                 "note": "the picture: the pixel program over every pixel of the tile at (col0, row0), cols x rows (the offsets are f32: the branch compiler's WGSL has no f32-to-i64 conversion in a kernel)"})
    # The terminals.
    for k in range(PIXEL_SCOPES + 1):
        e = s_dual(k)
        for v in inputs:
            rows.append({"name": f"{v}@{e}", "kind": "terminal", "out": e, "args": [], "spelling": v,
                         "note": "an input of the interface"})
        for i in range(k):
            rows.append({"name": f"v{i}@{e}", "kind": "terminal", "out": e, "args": [], "spelling": f"v{i}",
                         "note": f"the bound variable v{i}"})
        rows.append({"name": f"?_{e}", "kind": "const", "out": e, "args": [], "spelling": "",
                     "note": "a literal of the constant pool"})
        rows.append({"name": f"black@{blk_dual(k)}", "kind": "terminal", "out": blk_dual(k), "args": [],
                     "spelling": "[0f32, 0f32, 0f32]", "note": "the black pixel"})
    for dual in sorted(done, key=parse_body):
        s, _, _ = parse_body(dual)
        rows.append({"name": f"zero@{dual}", "kind": "terminal", "out": dual, "args": [],
                     "spelling": "(" + ", ".join(["0f32"] * s) + ")", "note": "the zero state"})
    for c in PIXEL_COUNTS:
        rows.append({"name": f"n{c}@count", "kind": "terminal", "out": "count", "args": [],
                     "spelling": str(c), "note": f"{c} trips"})
    rows.append({"name": "image.black", "kind": "terminal", "out": "image", "args": [],
                 "spelling": "map (\\_ -> map (\\_ -> [0f32, 0f32, 0f32]) (iota cols)) (iota rows)",
                 "note": "the black picture"})
    return rows


def parse_body(dual):
    """`body5.d1.s13` -> (5, 1, 13); None for any other dual."""
    m = re.fullmatch(r"body(\d+)\.d(\d+)\.s(\d+)", dual)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def pixel_types(rows, literals):
    """The duals the rows mention: the root, the scopes, the blocks, the
    bodies and the count."""
    named = []
    for r in rows:
        for d in [r["out"]] + r["args"]:
            if d not in named:
                named.append(d)

    def key(d):
        if d == "image":
            return (0, 0, 0, 0)
        if d.startswith("f32.s"):
            return (1, int(d[5:]), 0, 0)
        if d.startswith("blk.s"):
            return (2, int(d[5:]), 0, 0)
        if d == "count":
            return (4, 0, 0, 0)
        s, dd, k = parse_body(d)
        return (3, s, dd, k)

    out = []
    for d in sorted(named, key=key):
        if d == "image":
            out.append({"dual": d, "slot": "img", "form": "rgb", "literals": [], "root": True,
                        "meaning": "the picture: [h][w][3]f32"})
        elif d.startswith("f32.s"):
            out.append({"dual": d, "slot": "f32", "form": d[4:], "literals": literals, "root": False,
                        "meaning": f"an f32 in scope {d[5:]} (v0 .. v{int(d[5:]) - 1} bound)"})
        elif d.startswith("blk.s"):
            out.append({"dual": d, "slot": "blk", "form": d[4:], "literals": [], "root": False,
                        "meaning": f"a pixel block in scope {d[5:]}, ending in the colour"})
        elif d == "count":
            out.append({"dual": d, "slot": "count", "form": "i64", "literals": [], "root": False,
                        "meaning": "a loop's trip count"})
        else:
            s, dd, k = parse_body(d)
            out.append({"dual": d, "slot": "body", "form": f"{s}.d{dd}.s{k}", "literals": [], "root": False,
                        "meaning": f"a loop body at depth {dd} in scope {k}, ending in a {s}-tuple"})
    return out


def pixel_tables(instance):
    """The text of every artefact of one instance."""
    shader = os.path.join(REPO, PIXEL_INSTANCES[instance])
    interface = pixel_interface(shader)
    literals = pixel_literals(shader)
    rows = pixel_rows(interface)
    types = pixel_types(rows, literals)
    ttext = ["ref\tdual\tslot\tform\tliterals\troot\tmeaning"]
    for i, t in enumerate(types, start=1):
        ttext.append(f"T{i:04d}\t{t['dual']}\t{t['slot']}\t{t['form']}\t{','.join(t['literals'])}\t"
                     f"{'yes' if t['root'] else 'no'}\t{t['meaning']}")
    ftext = ["ref\tname\tkind\tout\targs\tspelling\tnote"]
    ordered = [r for r in rows if r["kind"] == "function"] + [r for r in rows if r["kind"] != "function"]
    for i, r in enumerate(ordered, start=1):
        ftext.append(f"F{i:05d}\t{r['name']}\t{r['kind']}\t{r['out']}\t{';'.join(r['args'])}\t{r['spelling']}\t{r['note']}")
    itext = ["# input\tgroup\tbinding\tword\tspelling"]
    for f in interface:
        itext.append(f"{f['name']}\t{f['group']}\t{f['binding']}\t{f['word']}\t{f['spelling']}")
    out = {
        "types.tsv": "\n".join(ttext) + "\n",
        "functions.tsv": "\n".join(ftext) + "\n",
        "interface.tsv": "\n".join(itext) + "\n",
    }
    out.update(pixel_egraph_texts(types, ordered, literals))
    return out


# ---- the pixel family's e-graph -------------------------------------------
#
# One sort per dual, one constructor per row; a constant placeholder's
# constructor takes the literal's index in the pool (an i64). The rules are
# the IEEE identities that hold BIT FOR BIT on every finite and infinite
# input, instantiated per scope:
#
#   a + b = b + a,  a * b = b * a,  max a b = max b a,  min a b = min b a
#   a - b = a + (-b),  -(-a) = a,  |-a| = |a|,  a * 2 = a + a,  a * 1 = a
#   a / 2 = a * 0.5,  a / 4 = a * 0.25
#
# (`a + 0 = a` is NOT one: -0 + 0 is +0.) A NaN's payload may differ; a NaN
# is a NaN either way, and the picture compares NaN by its bits as a mismatch.

def pixel_sort(dual):
    return "P" + "".join(p[:1].upper() + p[1:] for p in re.split(r"[._]", dual))


def pixel_ctor(row):
    op, _, dual = row["name"].partition("@")
    if not dual:
        if row["name"].startswith("?_"):
            return "C" + pixel_sort(row["name"][2:])
        return "X" + "".join(p.capitalize() for p in row["name"].split("."))
    return "X" + "".join(p[:1].upper() + p[1:] for p in re.split(r"[._?]", op) if p) + pixel_sort(dual)


def pixel_egraph_texts(types, rows, literals):
    data = ["; The pixel family's e-graph datatype, generated by gen_tables.py.",
            "; One sort per dual: the sort IS the type (scope, block, body).", ""]
    for t in types:
        data.append(f"(sort {pixel_sort(t['dual'])})")
    data.append("")
    for r in rows:
        if r["kind"] == "const":
            data.append(f"(constructor {pixel_ctor(r)} (i64) {pixel_sort(r['out'])})")
        else:
            args = " ".join(pixel_sort(a) for a in r["args"])
            data.append(f"(constructor {pixel_ctor(r)} ({args}) {pixel_sort(r['out'])})")
    index = {f32_bits(l[:-3]): i for i, l in enumerate(literals)}
    rules = ["; The pixel family's e-graph ruleset, generated by gen_tables.py.", "(ruleset futhark_pixel)", ""]
    for k in range(PIXEL_SCOPES + 1):
        e = s_dual(k)
        c = {op: pixel_ctor({"name": f"{op}@{e}"}) for op, _, _ in PIXEL_OPERATORS}
        const = "C" + pixel_sort(e)
        rules.append(f"; scope {k}")
        for op in ("add", "mul", "max", "min"):
            rules.append(f"(rewrite ({c[op]} a b) ({c[op]} b a) :ruleset futhark_pixel)")
        rules.append(f"(rewrite ({c['sub']} a b) ({c['add']} a ({c['neg']} b)) :ruleset futhark_pixel)")
        rules.append(f"(rewrite ({c['neg']} ({c['neg']} a)) a :ruleset futhark_pixel)")
        rules.append(f"(rewrite ({c['abs']} ({c['neg']} a)) ({c['abs']} a) :ruleset futhark_pixel)")
        for value, rule in (("2.0", "({mul} a ({const} {i})) ({add} a a)"),
                            ("1.0", "({mul} a ({const} {i})) a")):
            i = index.get(f32_bits(value))
            if i is not None:
                rules.append("(rewrite " + rule.format(mul=c["mul"], add=c["add"], const=const, i=i)
                             + " :ruleset futhark_pixel)")
        for d, m in (("2.0", "0.5"), ("4.0", "0.25")):
            i, j = index.get(f32_bits(d)), index.get(f32_bits(m))
            if i is not None and j is not None:
                rules.append(f"(rewrite ({c['div']} a ({const} {i})) ({c['mul']} a ({const} {j})) :ruleset futhark_pixel)")
        rules.append("")
    rtext = ["# row name\tconstructor\tsort"] + [f"{r['name']}\t{pixel_ctor(r)}\t{pixel_sort(r['out'])}" for r in rows]
    stext = ["# dual\tsort"] + [f"{t['dual']}\t{pixel_sort(t['dual'])}" for t in types]
    return {
        "egraph.datatype": "\n".join(data) + "\n",
        "egraph.rules": "\n".join(rules) + "\n",
        "egraph.rows": "\n".join(rtext) + "\n",
        "egraph.sorts": "\n".join(stext) + "\n",
    }


# ---- the pixel family's witness tests ---------------------------------------
#
# Re-read from the emitted TEXT and re-checked against a model written apart
# from the enumeration: the scope of every variable, the binding of every let
# and loop, the loop bound, the terminals, the literal pool and the e-graph's
# mapping.

VAR_RE = re.compile(r"\bv(\d+)\b")
SCOPE_RE = re.compile(r"^(?:f32|blk)\.s(\d+)$")


def scope_of(dual):
    m = SCOPE_RE.match(dual)
    if m:
        return int(m.group(1))
    b = parse_body(dual)
    return b[2] if b else None


def witness_pixel(dir_path, instance):
    bad = []
    types, funcs = read_tables(os.path.join(dir_path, "types.tsv"), os.path.join(dir_path, "functions.tsv"))
    declared = {t[1]: t for t in types}
    roots = [t[1] for t in types if t[5] == "yes"]
    if roots != ["image"]:
        bad.append(f"P1 the roots are {roots}, expected [image]")
    terminals = set()
    for row in funcs:
        name, kind, out, args, spelling = row[1], row[2], row[3], [a for a in row[4].split(";") if a], row[5]
        for d in [out] + args:
            if d not in declared:
                bad.append(f"P2 row {name} names the undeclared dual {d}")
        if kind != "function":
            terminals.add(out)
            if args:
                bad.append(f"P2 the {kind} {name} has arguments")
        # P3: the placeholders are exactly {0} .. {n-1}.
        held = sorted({int(x) for x in PLACEHOLDER_RE.findall(spelling)})
        if kind == "function" and held != list(range(len(args))):
            bad.append(f"P3 row {name}'s spelling holds {held}, it has {len(args)} arguments")
        k = scope_of(out)
        # P4: a variable named in a spelling is bound: by the scope (a
        # terminal) or by the row itself (a let or loop).
        if kind == "terminal" and out.startswith("f32.s"):
            for v in VAR_RE.findall(spelling):
                if int(v) >= k:
                    bad.append(f"P4 the terminal {name} names v{v} in scope {k}")
        if kind == "function" and out.startswith("f32.s"):
            if any(a != out for a in args):
                bad.append(f"P4 the scalar row {name} takes an argument outside its scope: {args}")
            if VAR_RE.findall(spelling):
                bad.append(f"P4 the scalar row {name} names a variable")
        if name.startswith("let@"):
            if args != [f"f32.s{k}", out.rsplit(".s", 1)[0] + f".s{k + 1}"] or spelling != f"let v{k} = {{0}} in {{1}}":
                bad.append(f"P5 the let {name} does not bind v{k} into scope {k + 1}: {args} {spelling!r}")
        if name.startswith("loop"):
            s = int(name[4:name.index("@")])
            depth = 0 if out.startswith("blk") else parse_body(out)[1]
            body = parse_body(args[s + 1]) if len(args) == s + 3 else None
            rest = scope_of(args[s + 2]) if len(args) == s + 3 else None
            state = ", ".join(f"v{k + i}" for i in range(s))
            if (args[:s] != [f"f32.s{k}"] * s or args[s] != "count" or body != (s, depth + 1, k + s + 1)
                    or rest != k + s or args[s + 2].split(".s")[0] != out.split(".s")[0]):
                bad.append(f"P5 the loop {name} has the wrong argument duals {args}")
            if not spelling.startswith(f"let ({state}) = loop ({state}) = (") or f"let v{k + s} = f32.i64 i{k} in" not in spelling:
                bad.append(f"P5 the loop {name} does not bind v{k}..v{k + s - 1} and its counter v{k + s}: {spelling!r}")
            if depth + 1 > PIXEL_DEPTH:
                bad.append(f"P6 the loop {name} nests deeper than {PIXEL_DEPTH}")
        if name.startswith("block"):
            s = int(name[5:name.index("@")])
            depth = 0 if out.startswith("blk") else parse_body(out)[1]
            state = ", ".join(f"v{k + i}" for i in range(s))
            if (len(args) != 2 or parse_body(args[0]) != (s, depth, k) or scope_of(args[1]) != k + s
                    or args[1].split(".s")[0] != out.split(".s")[0] or spelling != f"let ({state}) = {{0}} in {{1}}"):
                bad.append(f"P5 the block {name} does not bind v{k}..v{k + s - 1} from a body of its own depth: {args} {spelling!r}")
        if name.startswith("tuple@"):
            s = parse_body(out)[0]
            if len(args) != s:
                bad.append(f"P5 the tuple {name} has {len(args)} values, its loop carries {s}")
    # P6: the work bound -- a pixel runs at most max(count) ** depth trips.
    counts = [int(r[5]) for r in funcs if r[3] == "count"]
    if not counts or max(counts) ** PIXEL_DEPTH > 4096:
        bad.append(f"P6 the work bound max(count) ** depth = {max(counts or [0]) ** PIXEL_DEPTH} exceeds 4096 trips")
    for t in types:
        if t[1] not in terminals:
            bad.append(f"P7 the dual {t[1]} has no terminal")
    # P8: the constant pool parses as f32, holds each value once, and every
    # literal past the generic ones appears in the shader's source.
    src = strip_comments(open(os.path.join(REPO, PIXEL_INSTANCES[instance])).read())
    for t in types:
        if t[2] != "f32":
            continue
        lits = t[4].split(",")
        bits = [f32_bits(l[:-3]) for l in lits if l.endswith("f32")]
        if len(bits) != len(lits) or len(set(bits)) != len(bits):
            bad.append(f"P8 the pool of {t[1]} is not a set of f32 literals")
        for l in lits[len(PIXEL_GENERIC_LITERALS):]:
            if not re.search(r"(?<![\w.])" + re.escape(l[:-3]) + r"(?![\w.])", src):
                bad.append(f"P8 the literal {l} of {t[1]} is not in the shader's source")
    # P9: the e-graph maps every row to one constructor of its dual's sort.
    with open(os.path.join(dir_path, "egraph.rows")) as f:
        mapping = [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")]
    with open(os.path.join(dir_path, "egraph.datatype")) as f:
        ctors = {}
        for l in f:
            m = re.match(r"^\(constructor\s+(\w+)\s+\(([^)]*)\)\s+(\w+)\)$", l.strip())
            if m:
                ctors[m.group(1)] = (m.group(2).split(), m.group(3))
    if sorted(m[0] for m in mapping) != sorted(r[1] for r in funcs):
        bad.append("P9 the e-graph mapping and the table name different rows")
    if len({m[1] for m in mapping}) != len(mapping):
        bad.append("P9 two rows share a constructor")
    by_name = {r[1]: r for r in funcs}
    for name, ctor, sort in mapping:
        r = by_name.get(name)
        if r is None or ctor not in ctors:
            bad.append(f"P9 the row {name} or its constructor {ctor} is missing")
            continue
        want_args = ["i64"] if r[2] == "const" else [pixel_sort(a) for a in r[4].split(";") if a]
        if ctors[ctor] != (want_args, pixel_sort(r[3])) or sort != pixel_sort(r[3]):
            bad.append(f"P9 the constructor {ctor} of {name} has the sorts {ctors[ctor]}")
    return bad


def pixel_dir(instance):
    return os.path.join(HERE, "pixel", instance)


def artefacts():
    """Every generated artefact, by file name: the text it must have."""
    out = {
        "types.tsv": types_text(),
        "functions.tsv": functions_text(),
        "costs.tsv": costs_text(),
    }
    out.update(egraph_texts())
    return out


def main(argv):
    """Check the committed artefacts, or write them.

    The DEFAULT is to CHECK. The trust boundary this kingdom rests on is the
    committed table, so the generator's job is to say whether the committed
    table is still what the generator would produce. Writing first and
    checking afterwards would launder any hand edit: the witness tests would
    only ever see the generator's own output, and the corruption would be gone
    by the time they ran.
    """
    mode = argv[1] if len(argv) > 1 else "--check"
    if mode not in ("--check", "--write"):
        print(f"usage: {os.path.basename(__file__)} [--check | --write]")
        print("  --check  verify the committed artefacts (the default)")
        print("  --write  regenerate them, then verify")
        return 2

    types_path = os.path.join(HERE, "types.tsv")
    functions_path = os.path.join(HERE, "functions.tsv")
    wanted = artefacts()

    bad = []
    if mode == "--write":
        # The new artefacts are written to TEMPORARY files and the witness
        # tests run against those. Only if they pass are the real files
        # replaced: a generator whose own model is broken must not leave a
        # broken table in the tree, where the next `cargo build` would
        # compile it.
        staged = {}
        for name, text in wanted.items():
            tmp = os.path.join(HERE, name + ".new")
            with open(tmp, "w") as f:
                f.write(text)
            staged[name] = tmp
        staged_dir = os.path.join(HERE, ".staged")
        os.makedirs(staged_dir, exist_ok=True)
        for name in wanted:
            os.replace(staged[name], os.path.join(staged_dir, name))
        staged_bad = (witness(os.path.join(staged_dir, "types.tsv"),
                             os.path.join(staged_dir, "functions.tsv"))
                      + witness_costs(os.path.join(staged_dir, "functions.tsv"),
                                      os.path.join(staged_dir, "costs.tsv"))
                      + witness_egraph(staged_dir))
        if staged_bad:
            print("THE WITNESS TESTS FAILED (the committed files are untouched):")
            for b in staged_bad:
                print("  " + b)
            for name in wanted:
                os.remove(os.path.join(staged_dir, name))
            os.rmdir(staged_dir)
            return 1
        for name in wanted:
            os.replace(os.path.join(staged_dir, name), os.path.join(HERE, name))
        os.rmdir(staged_dir)
        print(f"wrote and checked {len(wanted)} artefacts")
    else:
        # The committed artefact must be byte for byte what the generator
        # would write. A hand edit -- even a good one -- is a finding, because
        # the next regeneration would silently discard it.
        #
        # BINARY mode: text mode would translate CRLF to LF on the way in, so
        # a committed file with Windows line endings would compare equal to
        # the generator's LF output and the check would be a lie.
        for name, text in wanted.items():
            path = os.path.join(HERE, name)
            try:
                with open(path, "rb") as f:
                    got = f.read()
            except OSError as e:
                bad.append(f"the artefact {name} cannot be read: {e}")
                continue
            want = text.encode("utf-8")
            if got != want:
                bad.append(
                    f"the committed {name} is not what the generator produces "
                    f"({first_difference(got, want)}); run with --write to regenerate"
                )

    if bad:
        # The witness tests read the committed files, so they cannot run
        # against an artefact that is not there. Say so, and stop.
        print("THE COMMITTED ARTEFACTS ARE NOT USABLE:")
        for b in bad:
            print("  " + b)
        return 1

    # The witness tests read the COMMITTED files, so they check what the
    # engine will actually load. (In `--write` mode they have already run
    # against the staged copies, which are the same bytes.)
    if mode == "--check":
        bad = (witness(types_path, functions_path)
               + witness_costs(functions_path, os.path.join(HERE, "costs.tsv"))
               + witness_egraph(HERE))
        if bad:
            print("THE WITNESS TESTS FAILED:")
            for b in bad:
                print("  " + b)
            return 1

    types, funcs = read_tables(types_path, functions_path)
    classes = sorted(parse_class(t[1]) for t in types if parse_class(t[1]))
    print(f"types.tsv: {len(types)} types ({len(classes)} matrix classes + 2 function classes)")
    print(f"functions.tsv: {len(funcs)} rows")
    print(f"roles: {', '.join(ROLES)}   roots: {', '.join(mat_class(*r) for r in ROOTS)}")
    print(f"class set: {', '.join(mat_class(*c) for c in classes)}")
    by_op = {}
    for row in funcs:
        op = row[1].split("@")[0]
        by_op[op] = by_op.get(op, 0) + 1
    for op, n in sorted(by_op.items()):
        print(f"  {op:14s} {n}")
    if mode == "--check":
        print(f"committed artefacts: MATCH the generator ({len(wanted)} files)")
    print("witness tests: PASS")
    return pixel_main(mode)


def pixel_main(mode):
    """The pixel family's instances: checked (the default) or written, the
    same way as the matrix kingdom's artefacts -- written to a staging
    directory, witnessed there, and only then moved into place."""
    for instance in sorted(PIXEL_INSTANCES):
        out_dir = pixel_dir(instance)
        wanted = pixel_tables(instance)
        if mode == "--write":
            staged = os.path.join(out_dir, ".staged")
            os.makedirs(staged, exist_ok=True)
            for name, text in wanted.items():
                with open(os.path.join(staged, name), "w") as f:
                    f.write(text)
            bad = witness_pixel(staged, instance)
            if bad:
                print(f"THE PIXEL WITNESS TESTS FAILED for {instance} (the committed files are untouched):")
                for b in bad:
                    print("  " + b)
                for name in wanted:
                    os.remove(os.path.join(staged, name))
                os.rmdir(staged)
                return 1
            for name in wanted:
                os.replace(os.path.join(staged, name), os.path.join(out_dir, name))
            os.rmdir(staged)
        else:
            bad = []
            for name, text in wanted.items():
                path = os.path.join(out_dir, name)
                try:
                    with open(path, "rb") as f:
                        got = f.read()
                except OSError as e:
                    bad.append(f"the artefact pixel/{instance}/{name} cannot be read: {e}")
                    continue
                if got != text.encode("utf-8"):
                    bad.append(f"the committed pixel/{instance}/{name} is not what the generator produces "
                               f"({first_difference(got, text.encode('utf-8'))}); run with --write to regenerate")
            if not bad:
                bad = witness_pixel(out_dir, instance)
            if bad:
                print(f"THE PIXEL INSTANCE {instance} FAILED:")
                for b in bad:
                    print("  " + b)
                return 1
        types, funcs = read_tables(os.path.join(out_dir, "types.tsv"), os.path.join(out_dir, "functions.tsv"))
        kinds = {}
        for r in funcs:
            kinds[r[2]] = kinds.get(r[2], 0) + 1
        print(f"pixel/{instance}: {len(types)} types, {len(funcs)} rows "
              f"({', '.join(f'{n} {k}' for k, n in sorted(kinds.items()))}); "
              f"{'committed artefacts MATCH the generator, ' if mode == '--check' else 'written, '}witness tests PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
