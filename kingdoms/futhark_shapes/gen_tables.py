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
import sys

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
# `map2` wants a two-argument function.)
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
        "spelling": "\\xs -> map (+0.0) xs",
        "note": "the constant row function",
    })
    rows.append({
        "name": f"{FN2}.const",
        "out": FN2,
        "args": [],
        "spelling": "\\xs _ys -> map (+0.0) xs",
        "note": "the constant binary row function",
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
# The tables.
# --------------------------------------------------------------------------

def write_types(path):
    with open(path, "w") as f:
        f.write("ref\tdual\tslot\tform\tliterals\troot\tmeaning\n")
        for i, t in enumerate(type_rows(), start=1):
            f.write(f"T{i:04d}\t{t['dual']}\t{t['slot']}\t{t['form']}\t"
                    f"{','.join(t['literals'])}\t{'yes' if t['root'] else 'no'}\t"
                    f"{t['meaning']}\n")


def write_functions(path):
    with open(path, "w") as f:
        f.write("ref\tname\tkind\tout\targs\tspelling\tnote\n")
        i = 0
        for r in operator_rows() + function_rows():
            i += 1
            out = r["out"] if isinstance(r["out"], str) else mat_class(*r["out"])
            args = ";".join(a if isinstance(a, str) else mat_class(*a) for a in r["args"])
            f.write(f"F{i:05d}\t{r['name']}\tfunction\t{out}\t{args}\t{r['spelling']}\t{r['note']}\n")
        for r in terminal_rows():
            i += 1
            out = r["out"] if isinstance(r["out"], str) else mat_class(*r["out"])
            f.write(f"F{i:05d}\t{r['name']}\tterminal\t{out}\t\t{r['spelling']}\t{r['note']}\n")


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

def main():
    types_path = os.path.join(HERE, "types.tsv")
    functions_path = os.path.join(HERE, "functions.tsv")
    write_types(types_path)
    write_functions(functions_path)

    bad = witness(types_path, functions_path)
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
    print("witness tests: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
