#!/usr/bin/env python3
"""The Futhark-style parallel kingdom: the design written as code.

The kingdom evolves a *typed, high-level parallel program*: a tree of
combinators over arrays. It is the answer to the search-space problem of
`kingdoms/wgsl_true`, where 305 types and 8,841 rows bought mostly ways to
spell a statement nest (112 of the types were `stmt.*`/`block.*` scaffolding
and 5,295 of the rows produced one).

The rules that keep this kingdom small:

  * **Shape is a value, not a type.** An array's length is not in the type
    system, so `arr.f32` is ONE type however long the array is. This is the
    single decision that stops the type count from following the program size.
  * **Nesting is free.** `map`, `map2` and `scan` are *value-level*
    combinators, so a program's depth costs no types at all -- the thing
    `wgsl_true` had to encode as 112 scaffolding types.
  * **A function value is a type, not a lambda.** `fn.f32_f32` is a type whose
    own reading frame holds the elementary functions. `map` takes one. So
    `map` composes without the GEP ever needing a bound variable: the
    elementary function's *parameter* is bound by the spelling, and its
    *argument* (a captured literal) is an ordinary child slot.
  * **A reduction is a fused row.** `reduce` needs a two-argument combiner,
    and the GEP's argument slots are values, not binders, so the combiner
    family is spelled out: `reduce_add`, `reduce_mul`. Fusing costs two rows
    and buys a first-order type system.
  * **Every neutral is the operator's identity.** `reduce (+) 0.0`,
    `reduce (*) 1.0`, `scan (+) 0.0`, `scan (*) 1.0`. This is Futhark's own
    precondition, not a taste: a neutral that is not a true neutral element
    makes the result undefined, and the compiler cannot detect it. It also
    costs the kingdom nothing -- the neutral of a prefix sum is 0 and of a
    prefix product 1 -- and it buys exactness: with the identities, every
    array size 1..9 is exact; with a free neutral, `reduce (+) 0.0 (scan (+)
    0.5 xs)` returns 27.0 where 22.0 is right.
  * **One extra type buys the whole discipline.** `scn.f32` is "a scan
    result". A `scan` takes an `arr.f32` and gives an `scn.f32`, so
    scan-over-scan is *unrepresentable* rather than rejected -- and that
    matters, because Futhark 0.28.1 has an internal compiler error on
    `scan (+) 0.0 (scan (+) 1.0 xs)`. `map`, `map2` and `rev` are typed by
    what Futhark does to the accumulator: `map` and `map2` keep it, `reverse`
    drops it. Two array types and a factor of two on the rows is the whole
    price of a kingdom in which every gene is legal.

Output: `types.tsv` and `functions.tsv`, the format `src/futhark_kingdom.rs`
loads.

    python3 kingdoms/futhark_true/gen_tables.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# The types. Four of them, and the whole kingdom fits on one screen.
# --------------------------------------------------------------------------

# (dual, slot, form, literals, is_root, meaning)
#
# The literals are the values a `?_T` placeholder can take. The gene says
# "a real here"; the gene's constant array says WHICH real, by index into
# this list. The list is the whole constant space of the kingdom.
TYPES = [
    ("f32.real", "f32", "real",
     ["0.0", "1.0", "-1.0", "0.5", "2.0", "3.0", "0.25", "10.0", "-0.5", "1.5",
      "100.0", "0.1", "0.75", "7.0", "-2.5", "0.125"],
     True, "a real number"),
    ("arr.f32", "arr", "f32", [], True,
     "a one-dimensional array of reals that is NOT a scan result"),
    ("scn.f32", "scn", "f32", [], True,
     "a one-dimensional array of reals that IS a scan result"),
    ("fn.f32_f32", "fn", "f32_f32", [], False,
     "a function from a real to a real"),
]

# --------------------------------------------------------------------------
# The symbols. `out` is the output dual; `args` the ordered argument duals;
# `spelling` the Futhark text, with `{0}`, `{1}`, ... the children. The
# renderer brackets every child, so a spelling has none of its own.
# --------------------------------------------------------------------------

# (name, out, args, spelling, note)
FUNCTIONS = [
    # ---- over an ordinary array ------------------------------------------
    ("map_a", "arr.f32", ["fn.f32_f32", "arr.f32"], "map {0} {1}",
     "apply a function to every element"),
    ("map2_add_a", "arr.f32", ["arr.f32", "arr.f32"], "map2 (+) {0} {1}",
     "elementwise sum of two arrays"),
    ("map2_mul_a", "arr.f32", ["arr.f32", "arr.f32"], "map2 (*) {0} {1}",
     "elementwise product of two arrays"),
    ("map2_max_a", "arr.f32", ["arr.f32", "arr.f32"], "map2 f32.max {0} {1}",
     "elementwise maximum of two arrays"),
    ("map2_min_a", "arr.f32", ["arr.f32", "arr.f32"], "map2 f32.min {0} {1}",
     "elementwise minimum of two arrays"),
    ("rev_a", "arr.f32", ["arr.f32"], "reverse {0}", "reverse the array"),

    # ---- over a scan result ----------------------------------------------
    ("map_s", "scn.f32", ["fn.f32_f32", "scn.f32"], "map {0} {1}",
     "apply a function to every element of a scan result"),
    ("map2_add_s", "scn.f32", ["scn.f32", "scn.f32"], "map2 (+) {0} {1}",
     "elementwise sum of two scan results"),
    ("map2_mul_s", "scn.f32", ["scn.f32", "scn.f32"], "map2 (*) {0} {1}",
     "elementwise product of two scan results"),
    ("map2_max_s", "scn.f32", ["scn.f32", "scn.f32"], "map2 f32.max {0} {1}",
     "elementwise maximum of two scan results"),
    ("map2_min_s", "scn.f32", ["scn.f32", "scn.f32"], "map2 f32.min {0} {1}",
     "elementwise minimum of two scan results"),
    # `reverse` is the one combinator that DROPS the accumulator: Futhark's
    # reverse of an `acc` array is a plain array again (measured: reducing it
    # with a non-identity neutral shows no excess, so it is not an `acc`).
    ("rev_s", "arr.f32", ["scn.f32"], "reverse {0}",
     "reverse a scan result; the result is an ordinary array again"),

    # ---- the scan: its input is an ordinary array, so scan-over-scan is
    #      unrepresentable, and its output is a scan result -----------------
    #
    # The neutral is FIXED at the operator's identity, and that is not a
    # simplification: Futhark requires a reduce's or a scan's neutral element
    # to be a true neutral element for the operator, and says in so many
    # words that otherwise "we will get wrong results at run-time" and "the
    # compiler will not be able to detect that we messed up". Measured on
    # Futhark 0.28.1: `reduce (+) 0.0 (scan (+) 0.5 xs)` returns 27.0 where
    # the answer is 22.0 -- the neutral is counted once per element instead
    # of once -- and it poisons the sibling `last (scan (+) 0.5 xs)` too
    # (12.5 where 10.5 is right). With both neutrals at their identities
    # every size 1..9 is exact.
    ("scan_add_s", "scn.f32", ["arr.f32"], "scan (+) 0.0 {0}",
     "inclusive prefix sum"),
    ("scan_mul_s", "scn.f32", ["arr.f32"], "scan (*) 1.0 {0}",
     "inclusive prefix product"),

    # ---- the reductions, output a real -----------------------------------
    #
    # The same rule: the neutral is the operator's identity. There is no
    # `reduce f32.max` here, because max's identity is -infinity and Futhark
    # has no infinity literal (`f32.infinity` is an unknown name), so a max
    # reduction has no neutral element that is both exact and defined. The
    # elementwise max and min survive as `map2`, which takes no neutral.
    ("reduce_add_a", "f32.real", ["arr.f32"], "reduce (+) 0.0 {0}",
     "sum of the array"),
    ("reduce_mul_a", "f32.real", ["arr.f32"], "reduce (*) 1.0 {0}",
     "product of the array"),
    ("reduce_add_s", "f32.real", ["scn.f32"], "reduce (+) 0.0 {0}",
     "sum of a scan result"),
    ("reduce_mul_s", "f32.real", ["scn.f32"], "reduce (*) 1.0 {0}",
     "product of a scan result"),

    # ---- the elementary functions, output a function value ----------------
    ("fn.f32_f32.neg", "fn.f32_f32", [], "\\x -> -x", "negation"),
    ("fn.f32_f32.abs", "fn.f32_f32", [], "\\x -> f32.abs x", "absolute value"),
    ("fn.f32_f32.sqrt", "fn.f32_f32", [], "\\x -> f32.sqrt x", "square root"),
    ("fn.f32_f32.exp", "fn.f32_f32", [], "\\x -> f32.exp x", "exponential"),
    ("fn.f32_f32.log", "fn.f32_f32", [], "\\x -> f32.log x", "natural logarithm"),
    ("fn.f32_f32.sin", "fn.f32_f32", [], "\\x -> f32.sin x", "sine"),
    ("fn.f32_f32.cos", "fn.f32_f32", [], "\\x -> f32.cos x", "cosine"),
    ("fn.f32_f32.tan", "fn.f32_f32", [], "\\x -> f32.tan x", "tangent"),
    ("fn.f32_f32.sqr", "fn.f32_f32", [], "\\x -> x * x", "square"),
    ("fn.f32_f32.cube", "fn.f32_f32", [], "\\x -> x * x * x", "cube"),
    ("fn.f32_f32.recip", "fn.f32_f32", [], "\\x -> 1.0 / x", "reciprocal"),
    ("fn.f32_f32.floor", "fn.f32_f32", [], "\\x -> f32.floor x", "largest integer below"),
    ("fn.f32_f32.round", "fn.f32_f32", [], "\\x -> f32.round x", "nearest integer"),
    ("fn.f32_f32.add_c", "fn.f32_f32", ["f32.real"], "\\x -> x + {0}",
     "add a captured literal"),
    ("fn.f32_f32.mul_c", "fn.f32_f32", ["f32.real"], "\\x -> x * {0}",
     "multiply by a captured literal"),
    ("fn.f32_f32.sub_c", "fn.f32_f32", ["f32.real"], "\\x -> x - {0}",
     "subtract a captured literal"),
]

# The terminals. The `?_T` row is the constant placeholder: its value comes
# from the gene's constant array, so the gene says "a real here" and the
# constant array says which real. The others are leaves with a fixed meaning.
#
# The spelling of a terminal is its Futhark text. The `?_T` row has none: a
# constant placeholder's text is the literal its constant-array index names.
TERMINALS = [
    ("?_f32", "f32.real", "const", "", "a real literal, from the gene's constant array"),
    ("arr.f32.input", "arr.f32", "terminal", "xs", "the problem's input array"),
    ("scn.f32.identity", "scn.f32", "terminal", "map (\\x -> x) (scan (+) 0.0 xs)",
     "a scan result: the identity scan of the input"),
    ("fn.f32_f32.const", "fn.f32_f32", "terminal", "\\x -> 0.0", "the constant function"),
]


def write_types(path):
    with open(path, "w") as f:
        f.write("ref\tdual\tslot\tform\tliterals\troot\tmeaning\n")
        for i, (dual, slot, form, literals, root, meaning) in enumerate(TYPES, start=1):
            f.write(f"T{i:04d}\t{dual}\t{slot}\t{form}\t{','.join(literals)}\t"
                    f"{'yes' if root else 'no'}\t{meaning}\n")


def write_functions(path):
    with open(path, "w") as f:
        f.write("ref\tname\tkind\tout\targs\tspelling\tnote\n")
        i = 0
        for name, out, args, spelling, note in FUNCTIONS:
            i += 1
            f.write(f"F{i:05d}\t{name}\tfunction\t{out}\t{';'.join(args)}\t{spelling}\t{note}\n")
        for name, out, kind, spelling, note in TERMINALS:
            i += 1
            f.write(f"F{i:05d}\t{name}\t{kind}\t{out}\t\t{spelling}\t{note}\n")


def main():
    write_types(os.path.join(HERE, "types.tsv"))
    write_functions(os.path.join(HERE, "functions.tsv"))
    roots = [t[0] for t in TYPES if t[4]]
    print(f"types.tsv: {len(TYPES)} types")
    print(f"functions.tsv: {len(FUNCTIONS)} functions + {len(TERMINALS)} terminals = "
          f"{len(FUNCTIONS) + len(TERMINALS)} rows")
    print(f"roots: {', '.join(roots)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
