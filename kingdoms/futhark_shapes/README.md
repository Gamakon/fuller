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
| symbol rows | 38 | **71** |
| e-graph constructors | 40 | **71** |
| e-graph rules | 9 | **27** |
| dimensions | none (1-D) | 3 roles, 9 matrix classes |
| gene width (N = 3) | 40–42 words | **322 words** |

## The e-graph, generated

The 1-D kingdom's e-graph has its datatype and its ruleset as hand-written
constants, and its bridge as a hand-written `match` over the constructor
names. That was affordable for 38 rows and 4 sorts; it is not affordable for a
kingdom whose rows are *generated*. So the generator emits the e-graph too:

| artefact | what it holds |
|---|---|
| `egraph.datatype` | one sort per class (11), one constructor per row (71) |
| `egraph.rules` | the ruleset: three identities per class, 27 rules |
| `egraph.rows` | the bridge: row name → constructor → sort |
| `egraph.sorts` | the class → sort mapping |

The Rust side never re-derives a name — it *reads* the mapping — so the
witness tests (W10–W12) are what stand between the kingdom and its e-graph.

**One sort per class: the sort IS the class.** Egglog then refuses an
ill-shaped term by itself, and the class relation cannot be broken by a
rewrite: a rewrite cannot change a node's class.

Three rules, one per shape-preserving operator, instantiated per class:

* `map` with the identity row function is the matrix;
* `map2` with the identity binary row function is the **first** matrix (the
  identity ignores its second operand; the kingdom's other binary row
  functions do not — `add` uses both);
* `transpose` twice is the matrix.

All three are **bit-identical** — they copy or permute values and never do
arithmetic — and there is **no fusion rule**, so the divergence the 1-D
kingdom measured cannot arise here.

## The checks

Twelve, and the last three are the ones that matter:

| check | what it proves |
|---|---|
| the kingdom passes the kingdom's own contract check | the tables are well formed |
| the engine derives the counts the design predicted | 11 frames, gene width 322 at N = 3, leaf bound 75 |
| every random gene expresses | legality by construction — 500 genes, zero defects |
| Futhark accepts the rendered programs | the class table and the compiler agree |
| **the compiled program and the interpreter agree, element by element** | the programs compute the *right* thing |
| the e-graph loads; every row has a constructor | the generated e-graph and the kingdom agree |
| the e-graph finds the involution and the identity map | the rules fire, and the search is deterministic |
| **a candidate writes back to a gene and comes back** | the round-trip invariant: a candidate is heritable |
| **an e-graph candidate compiles, runs, and agrees** | the whole loop, closed |
| **every row's spelling means what the interpreter says** | no row's spelling has drifted from its meaning |
| **every e-graph rule is an equivalence** | no rule can justify a candidate that computes something else |
| the committed artefacts match the generator | the table has not been hand-edited behind the engine's back |

The interpreter (`src/futhark_shapes_interp.rs`) is the shape kingdom's own:
`map` over the rows, `map2` over the pairs of rows, `transpose`, and the
`matmul` helper's own formula — a left fold over the products of one row of
`A` and one column of `B`. A program that compiles but computes the wrong
thing would pass every other check.

Two details the interpreter gets right that a `Vec<Vec<f32>>` cannot:

* **the dimensions are explicit**, so a `[0][m]` matrix is not a `[0][0]` one.
  Futhark's `transpose` of `[0][m]` is `[m][0]` — `m` empty rows — and a
  `matmul` over an empty inner dimension is a zero matrix, not an empty one;
* **the agreement is exact**, and there are two comparisons because the paths do
  not deserve the same claim. `identical` is bit for bit (a NaN excepted) and is
  used for the paths whose arithmetic is exactly specified — a row function, a
  `map`, a `map2`, a `transpose`; it tells the **signed zeros apart**. `equal`
  is exact values (a NaN excepted) with **no tolerance**, and is used for the
  reduction path, where the compiled program may legitimately differ from the
  source semantics.

  There is no tolerance because the tolerance's stated reason was false:
  Futhark 0.28.1's `c` backend reduces with a **sequential left fold**,
  measured bit-identical to the interpreter's over 600 of 600 elements, and the
  order-sensitive probe `[16777216, 1, -16777216, 1]` tells a left fold from a
  tree sum in both. A margin that is never needed is a blind spot.

  The one place the compiled program and the source semantics differ is
  Futhark's **identity fold**: the simplifier folds `x + 0` away, so the
  compiled `x + 0.0` is the identity and keeps `-0.0`'s sign, where the source
  semantics turns it into `+0.0`. (`futhark run` is Futhark's *interpreter*, so
  it shows the source semantics; the engine compiles with `futhark c`.) What is
  narrow is the fold, not the sign of a zero — the compiled program still tells
  the zeros apart where the fold does not apply, `1.0/x` on `-0.0` giving
  `-f32.inf`.

  **No row depends on the fold.** The `const` rows used to be spelled
  `map (+0.0)`, which compiles to the identity, so they differed from the
  identity only in the sign of a zero — a difference no comparison could see.
  They are now spelled `map (\_ -> 0.0)`, a genuine constant, and differ from
  the identity by **value**. A reduction over a *singleton* inner dimension has
  the same caveat, which is why `matmul` is compared on values and not on bits.

Running a 2-D program needs no C driver: Futhark's textual format carries a
matrix as `[[1.0,2.0],[3.0,4.0]]`, so the inputs go in on stdin and the result
comes out on stdout. The format carries a matrix with no elements as
`empty([rows][cols]f32)`, so even a degenerate dimension survives the round
trip — the shape is never lost in the text.

## The witness tests

The engine no longer checks the shape logic, so the **table** becomes the
trust boundary — and something must check the table. `gen_tables.py` re-parses
the emitted TSV and re-checks every shape relation from the *text*, against a
relation model written separately from the enumeration code: the `matmul`
rows' inner roles agree, the `transpose` rows' result is the swap, the
`map2_rows` rows' two arrays share a class, every demanded class has a
terminal, the class set is closed under `transpose` and `matmul`, every
operator takes the function class it declares, every spelling mentions each
argument exactly once **and uses the arguments in the order Futhark's
positional application needs**, every spelling is the operator's own template,
and every e-graph rule matches one of three hand-written schemas.

Twelve checks, and the suite runs them: a Rust test shells out to
`gen_tables.py --check`, so a hand edit to a committed table fails
`cargo test` rather than waiting to be noticed. (A missing `python3` is a
failure, not a skip.)

The checks are the generator's **self-test**: they re-derive the shape
relations from the emitted text and compare them with a relation model written
separately from the enumeration code. What guards the *committed* artefacts is
the byte-for-byte comparison — a hand edit is a finding even when the edited
table is self-consistent, which is why the comparison is the outer check and
the witnesses are the inner one.

They fail loudly:

```
$ python3 kingdoms/futhark_shapes/gen_tables.py --check
types.tsv: 11 types (9 matrix classes + 2 function classes)
functions.tsv: 71 rows
…
committed artefacts: MATCH the generator (6 files)
witness tests: PASS
```

**The default is `--check`, not `--write`.** The trust boundary is the
*committed* table, so the generator's job is to say whether the committed
table is still what the generator would produce. Writing first and checking
afterwards would launder any hand edit: the witness tests would only ever see
the generator's own output, and the corruption would be gone by the time they
ran. So `--check` compares every committed artefact byte for byte with what
the generator would write, and the witness tests then read the **committed**
files — the ones the engine's `include_str!` will load.

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
python3 kingdoms/futhark_shapes/gen_tables.py            # check the committed artefacts
python3 kingdoms/futhark_shapes/gen_tables.py --write    # regenerate them, then check
```

The tables are committed, like the 1-D kingdom's, so the loader's
`include_str!` sees them. The default is to **check** them, because the
committed table is the trust boundary; `--write` regenerates and then runs the
same checks.

## The pixel family (`pixel/<shader>/`)

A second family from the same generator: a picture as a map over the pixels of
a scalar program, for the cuneus image shaders. The image row is
`map (\yi -> map (\xi -> let px = .. in let py = .. in <block>) (iota cols)) (iota rows)`
over a tile, and a pixel block is a chain of `let`s and bounded `loop`s over
`f32` ending in `[r, g, b]`. Every rule is a type:

| rule | the type that carries it |
|---|---|
| a variable is bound before it is read | `f32.s{k}`: an f32 in scope k; its terminals are `v0 .. v{k-1}` and the interface's inputs |
| `let` binds the next variable | `let@blk.s{k}` takes `f32.s{k}` and a block of scope k+1 |
| a loop binds its state and its counter | `loopS` binds `v{k} .. v{k+S-1}`, its body is `bodyS.d{d+1}.s{k+S+1}` (counter `v{k+S}`), the rest scope k+S |
| a loop is bounded | trip counts are `count` literals (at most 64), nesting depth at most 2 (`PIXEL_DEPTH`) |
| a loop body returns its state | `bodyS.*` ends in an S-tuple |

An instance is declared by its WGSL source: the uniform blocks' f32 fields
become the inputs (`interface.tsv`), and the source's float literals join the
generic constant pool. `volumepassage`: 300 types, 3,651 rows, scopes s0..s40,
loop arities 1, 3 and 5. The e-graph (one sort a dual, constants by their pool
index) holds IEEE identities that are bit-identical: commutativity of `+`, `*`,
`max`, `min`; `a - b = a + (-b)`; `-(-a) = a`; `|-a| = |a|`; `a * 2 = a + a`;
`a * 1 = a`; `a / 2 = a * 0.5`; `a / 4 = a * 0.25`.

The witness tests P1-P9 re-read the emitted text: the root, the declared duals,
the placeholders, every variable's scope, every let and loop binding, the loop
bound, a terminal for every dual, the literal pool against the shader's source,
and the e-graph mapping. The Rust side is `src/futhark_pixel.rs`
(loader, program, work bound and tiles, the hand-written reference, the
e-graph), and the examples `futhark_pixel_reference` and `futhark_pixel_evolve`.
