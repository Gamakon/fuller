# `futhark_shapes` — the shape-classed Futhark kingdom

**What it evolves:** the same kind of typed parallel program as `futhark_true`,
but over **multi-dimensional** arrays whose dimensions are *related* by the
program's own type.

**Why it exists.** `futhark_true` decides that **shape is a value, not a
type**: `arr.f32` is one type however long the array is. That keeps the type
count flat, and it also means the kingdom cannot say that two arrays' shapes
relate — `matmul`'s inner dimensions must match, and nothing in the type
system says so. So a `matmul` in the 1-D kingdom is spelled by the *caller's*
luck, and Futhark is left to reject the program if it is wrong.

This kingdom takes the other road, and the only road the multi-typed GEP
allows: **a shape class is a TYPE**.

## A class is a role, not a size

```
mat.n.m     the array the interface calls [n][m]
mat.m.p     the array the interface calls [m][p]
mat.n.p     …and the result of multiplying them
```

`n`, `m` and `p` are **dimension roles** the interface declares. The runtime
binds the actual sizes; the generated Futhark source is shape-polymorphic, so
one gene describes a program for *any* sizes the interface allows:

```futhark
def kernel [m][n][p] (m_n_n: [n][n]f32) (m_n_m: [n][m]f32) … : [n][p]f32 = …
entry main (m_n_n: [2][2]f32) (m_n_m: [2][3]f32) … : [2][4]f32 = kernel …
```

So the class set is finite (the interface's roles, crossed) while the sizes are
not — the expressive power of dependent shapes without the combinatorial
expansion of a concrete type per shape.

## Where the unification happens

**Here, once, offline.** Not during expression: a cell's alphabet is
enumerated before the draw, from the layout alone, and must stay that way. Not
in the GEP: the engine never learns that shapes exist. `gen_tables.py` takes
the declared interface, works out which class each operator instantiates to,
and writes the two tables the kingdom loader reads. The engine then sees an
ordinary flat typed kingdom, and **legality by construction is unchanged**.

| | `futhark_true` | `futhark_shapes` |
|---|---:|---:|
| types | 4 | **11** |
| symbol rows | 38 | **70** |
| dimensions | none (1-D) | 3 roles, 9 matrix classes |
| gene width (N = 3) | 40–42 words | **322 words** |

## The witness tests

The engine no longer checks the shape logic, so the **table** becomes the
trust boundary — and something must check the table. `gen_tables.py` re-parses
the emitted TSV and re-checks every shape relation from the *text*, against a
relation model written separately from the enumeration code: the `matmul`
rows' inner roles agree, the `transpose` rows' result is the swap, the
`map2_rows` rows' two arrays share a class, every demanded class has a
terminal, the class set is closed under `transpose` and `matmul`, every
operator takes the function class it declares, and every spelling mentions
each argument exactly once.

Nine checks. They run on every generation of the tables, and they fail loudly:

```
$ python3 kingdoms/futhark_shapes/gen_tables.py
types.tsv: 11 types (9 matrix classes + 2 function classes)
functions.tsv: 70 rows
witness tests: PASS
```

The ninth check exists because the Futhark compiler caught the class table
being wrong about a function's arity: `map2` is curried, so `map2_rows` takes
a *two*-argument row function. That is the shape of the whole design — the
table is the trust boundary, and both the table's own tests and the compiler
check it.

## The two function classes

`map` over a matrix needs an **array-valued** function (Futhark's scalar `map`
cannot map over rows), so the kingdom's function types are row functions. And
`map2` is curried, so it needs a *two*-argument one. Hence two classes:

```
fn.arr_f32_arr_f32            []f32 -> []f32          for map_rows
fn.arr2_f32_arr_f32_arr_f32   []f32 -> []f32 -> []f32 for map2_rows
```

## The `matmul` helper

Futhark 0.28.1 has no `matmul` in its prelude, so the kingdom brings its own —
and it is **shape-polymorphic**, which is the point: Futhark's own type checker
then enforces the inner-dimension relation the class table states, so a table
and a compiler that disagreed could not both be right.

```futhark
def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =
  map (\a -> map (\b -> reduce (+) 0.0 (map2 (*) a b)) (transpose B)) A
```

## The two Futhark kingdoms

They coexist, and they are separate kingdoms in separate `Ty` families
(`Ty::Futhark` and `Ty::FutharkShape`), so their duals can never be confused.
`futhark_true` keeps the programs whose shapes are only equal **at run time**
(`map2` over two arrays of equal length), which static classes cannot express.

| | `futhark_true` | `futhark_shapes` |
|---|---|---|
| shapes | 1-D, equal at run time | multi-dimensional, related by the type |
| equal-length inputs | runtime check | **static** (same class) |
| `matmul`'s inner dimensions | not expressible | **static** (the class relation) |

## Regenerating

```
python3 kingdoms/futhark_shapes/gen_tables.py
```

The tables are committed, like the 1-D kingdom's, so the loader's
`include_str!` sees them.
