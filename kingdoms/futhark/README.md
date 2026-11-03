# `futhark` — the Futhark kingdom

**What it evolves:** the same kind of typed parallel program as `futhark_1d`,
but over **multi-dimensional** arrays whose dimensions are *related* by the
program's own type.

**Why it exists.** `futhark_1d` decides that **shape is a value, not a
type**: `arr.f32` is one type however long the array is. That keeps the type
count flat, and it also means the kingdom cannot say that two arrays' shapes
relate — `matmul`'s inner dimensions must match, and nothing in the type
system says so. So a `matmul` in the 1-D Futhark kingdom is spelled by the *caller's*
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

| | `futhark_1d` | `futhark` |
|---|---:|---:|
| types | 4 | **11** |
| symbol rows | 38 | **71** |
| e-graph constructors | 40 | **71** |
| e-graph rules | 9 | **54** |
| dimensions | none (1-D) | 3 roles, 9 matrix classes |
| gene width (N = 3) | 40–42 words | **322 words** |

## The e-graph, generated

The 1-D Futhark kingdom's e-graph has its datatype and its ruleset as hand-written
constants, and its bridge as a hand-written `match` over the constructor
names. That was affordable for 38 rows and 4 sorts; it is not affordable for a
kingdom whose rows are *generated*. So the generator emits the e-graph too:

| artefact | what it holds |
|---|---|
| `egraph.datatype` | one sort per class (11), one constructor per row (71) |
| `egraph.rules` | the ruleset, 54 rules: three identities per class (27) and the transposed product for each of the 27 products |
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

The interpreter (`src/futhark_interpreter.rs`) is the Futhark kingdom's own:
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
$ python3 kingdoms/futhark/gen_tables.py --check
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
(`Ty::Futhark1d` and `Ty::Futhark`), so their duals can never be confused.
`futhark_1d` keeps the programs whose shapes are only equal **at run time**
(`map2` over two arrays of equal length), which static classes cannot express.

| | `futhark_1d` | `futhark` |
|---|---|---|
| shapes | 1-D, equal at run time | multi-dimensional, related by the type |
| equal-length inputs | runtime check | **static** (same class) |
| `matmul`'s inner dimensions | not expressible | **static** (the class relation) |

## Regenerating

```
cargo run --release --example futhark_views -- DIR              # the Rust view writes every view's table
python3 kingdoms/futhark/gen_tables.py --views DIR            # check the committed artefacts
python3 kingdoms/futhark/gen_tables.py --write --views DIR    # regenerate them, then check
```

The tables are committed, like the 1-D Futhark kingdom's, so the loader's
`include_str!` sees them. The default is to **check** them, because the
committed table is the trust boundary; `--write` regenerates and then runs the
same checks.

## The problems are views over the master (`master/`, `sig/<problem>/`)

Every row lives once, as a template, in `master/rows.tsv`. A problem's
specification `sig/<problem>/view.tsv` is written by `sniff.py` from the
problem's Futhark source (inputs, output, dimensions, element, depth), and
`src/futhark_view.rs`, the one implementation of the view, computes the table
from the master (`view.py` keeps only the master and specification readers). A signature view admits **every family** of the master;
which rows it keeps is decided by **reachability both ways**: forward, the
types that can be built from the leaves by any chain of functions (a row is
usable when all its arguments can be built); backward, the types the root can
use through usable rows. The kept rows are the usable rows whose output the
root can use. What the problem's source happens to use limits nothing
(`docs/PLAN_futhark_kingdom.md`, 3e). The per-problem tables are not
committed (3g): `examples/futhark_views` writes them, `gen_tables.py --views
DIR` runs the witnesses on them, and each view's written text is pinned in
Rust by its fingerprint.

### The elements: `i32` and `f32` (master version 3, step 4)

A view holds the elements its interface has (one, or several since master
version 4: "Two elements in one view" below). The master's rows say which elements each is defined
for (`elems`): the arithmetic, the comparisons, `min`, `max`, `abs`, `neg`, the
reductions, scans, histograms, `map2`, the binders, the index rows and the
inputs are `i32,f32`; `div` (`/`) and `sqrt` are **f32 only**, because they are
total there (NaN, the infinities) and trap on an integer 0; `mod` is `i32` only.
Widening a row's `elems` keeps its `since`: a view of another element sees the
same rows as before. Where an `f32` row's spelling must differ from the `i32`
one it is a row of its own: `clamp` and the histograms convert a value to an
index with `i64.f32`, which Futhark leaves undefined for a NaN, an infinity or a
value out of range, so the `f32` spellings test `f32.isnan` (a NaN is 0 for
`clamp`, ignored by a histogram) and clamp the value as an `f32` before the
conversion. (`x == x` is not a NaN test in Futhark: the simplifier folds it to
true; measured 2026-11-03 through `futhark c`.)

**The f32 semantics the interpreter mirrors**, measured through `futhark c`
2026-11-03 and checked bit for bit by `futhark_sig::tests_kmeans`: `f32.min`
and `f32.max` are the C library's `fmin`/`fmax` (a NaN loses to any number;
`-0.0` is below `+0.0`); `i64.f32` truncates toward zero and is 0 for a NaN, an
infinity or a value outside `i64`; `f32.sum`, `f32.minimum` and `f32.maximum`
are left folds from `0`, `f32.inf` and `-f32.inf`; division and `sqrt` are
IEEE.

**The comparison policy** (`futhark_sig::Array`, every case's exactness): bit
for bit, with one exception: any NaN agrees with any NaN (a NaN's payload is not
a value: Futhark prints every NaN as `f32.nan`, and which payload an operation
keeps is the hardware's choice). `-0.0` and `+0.0` are different values. The
stored cases are `futhark c -b`'s bytes, and the k-means exact gate of
`futhark_rust` compares bytes, so this is what they compare as.

**The f32 e-graph rules** (`master/rules.tsv`) are only the bit-exact ones:
commutativity of `+` and `*` (and of `map2 (+)`, `map2 (*)`), `-(-a) = a`,
`map2 min a a = a`, `map2 max a a = a`. No associativity, no distributivity, no
`x * 0 = 0` (a NaN), no reordering of a reduction, no commutativity of
`min`/`max` (`fmin(0, -0)` is the C library's choice). `a - b = a + (-b)` is
exact but was dropped for f32: measured on the k-means planted slowdown, the 256
ranked members of its e-class were all `add (neg ..)` forms and the reference
was not among them. The rule witness (M9) evaluates both sides of every rule on
its element's edges (the two zeros, NaN, the infinities, a denormal) under the
comparison policy.

### The class (master version 4, step 4c (b))

A class is the type of a value apart from its context, and its text reads
one way (the review's finding 1: a dotted string such as `t.a.n.a.m` reads two
ways, so the class became an AST, `src/futhark_view.rs::Class`, with this
grammar; `kingdoms/futhark/view.py::parse_class` reads it for the checks):

```
class  ::= elem                          a scalar: i32, f32
         | 'ix.' dim                     an index in range for a dimension: ix.n
         | 'a.' [elem '.'] dim {'.' dim}  an array: a.n.d, or a.f32.n.d in a view of several elements
         | '(' class ',' class ')'       a pair: (f32,i32), Futhark's own notation without the space
```

A dimension is never named as an element type, so the piece after `a.` is an
element exactly when it is one; a pair's components sit inside its own
parentheses, so `(a.f32.n.d,f32)` and `(a.f32.n,a.f32.d)` are two texts. A
view of **one** element leaves an array's element implied (`a.n`, the names of
before, so the pinned views are unchanged); a view of **several** writes it
(`a.f32.n`, `a.i32.n`). A type is a class in a context (`(f32,i32):c/{f32,i32,ix.n}@1`);
a context binds scalars, rank-1 arrays and indices, never a pair.

### Two elements in one view

The specification's `element` line lists the elements (`f32,i32`, in order of
first appearance in the entry's signature; the sniffer's `--elements` adds one
the signature does not name, for values a program keeps inside, as k-means's
assignments), and each has its own `literals` line. Every element's rows exist
(`add@f32:c` and `add@i32:c`; `mod` and the i32 `clamp` for i32, `div`, `sqrt`
and the f32 `clamp` for f32), each dimension has a rank-1 class of each element
(`a.f32.n`, `a.i32.n`, each with its zero array `zeros_f32_n`, `zeros_i32_n`),
and a loop may carry a scalar or a rank-1 array of either. A row's element `{E}`
is bound by the first quantifier that picks a class of one element, or iterated
over the elements the row is defined for; a row whose output class has no
element of its own is named by it where two could clash (`clamp.f32@ix.n:c`
beside `clamp.i32@ix.n:c`; in a view of one element `clamp@ix.n:c`, as before).
The two conversions: `cvt.i32@f32:c` is Futhark's `f32.i32` (total: exact
below 2^24, rounded to nearest above); `cvt.f32@i32:c` is `i32.f32` guarded as
the f32 `clamp` is (a NaN is 0, the value clamped into i32's range first, then
truncated toward zero), because Futhark leaves the conversion undefined there.

### Pairs

A pair of scalars is a class, one for every ordered pair of the view's elements
(`(f32,f32)`, `(f32,i32)`, `(i32,f32)`, `(i32,i32)`), in every context. Its
rows: `pair` builds one (`({0}, {1})`); `fst.A.B` and `snd.A.B` take a
component (`{0}.0`, `{0}.1`; the components' elements are in the name, since
two pairs may share a component type); `zero` is the pair of zeros, the
terminal every type needs; `psel.E` chooses between two pairs on a scalar's
sign (`if {0} < 0E then {1} else {2}`, the kingdom's one comparison, as `sel`);
and `loop.D` / `loop1.D` carry a pair state. **A loop over a pair destructures
it:** Futhark's own `loop (s, c) = (0f32, 0f32) for i < n do ...`, spelled
`loop ((s_f32, s2_f32): (f32, f32)) = {0} for i_n < n do {1}`, whose frame
(`L2`, `F2` in `master/frames.tsv`, versioned 4) binds the two components as
variables of their own classes, so the body reads `s_f32` and `s2_f32` directly
and the context is `c/{f32*2,ix.n}@1`, a count of classes as every context is.
The pair is never a bound variable, which is what keeps the context count where
it was: a view of two elements over one dimension at depth 2 has 328 types
(`sig/kahan`, `sig/argmin`).

The e-graph's pair rules are the two exact ones, `fst (pair a b) = a` and
`snd (pair a b) = b` (M9 evaluates them); no real-number identity exists, so
the Kahan step `(t - s) - y` survives saturation (tested).

**The test views** (`sig/kahan/kahan.fut`, two entries, two views): UltraDim's
k-means constructs, construct for construct. `kahan` (`xs: [n]f32`, `ms: [n]i32`
to `f32`) holds the compensated mean of the members of cluster 0: the loop over
the points whose state is the pair (sum, compensation), each step `y = x - c;
t = s + y; (t, (t - s) - y)` guarded by `|m| - 1 < 0` (that is `m == 0`), over
the count from a second loop, converted (UltraDim's third state, the count, is
a second loop here: the pair holds two). `argmin` (`ds: [n]f32` to `i32`) is
`assign_points`: the pair (best distance, best index) from the first distance,
over 1 .. n-1 with `loop1`, a later distance winning only when strictly smaller
(`d - best < 0`). Both compute, compiled with `futhark c`, the bits of the
yardstick entries (spelled with Futhark's own tuple loop, a triple state and a
range loop) and of the host interpreter, on 30 runs spanning magnitudes, n = 1,
all-member, no-member (0 / 0, a NaN), tied and NaN distances
(`futhark_sig::tests_view`). Random genes of both views are accepted by
`futhark check` (50 each) and the first 8 compile and equal the interpreter.

### Triples and the kernel boundary (master version 5, step 4c (c))

UltraDim's `update_centroids` carries THREE states in one loop, (sum,
compensation, count), and its kernels write buffers the next kernel reads. The
kingdom expresses both since version 5:

* **a triple of scalars** is a class, one for every ordered triple of the
  elements (`(f32,f32,i32)`, eight in a view of two elements), with `triple`,
  `fst.A.B.C` / `snd.A.B.C` / `thd.A.B.C` (`{0}.0`, `{0}.1`, `{0}.2`), the zero
  triple, `psel.E`, and `loop.D` / `loop1.D` destructuring it (the frames `L3`,
  `F3`: `loop ((s_f32, s2_f32, s_i32): (f32, f32, i32)) = ...`). The e-graph's
  rules are the three exact projections (`on tuple3`); the Kahan step survives
  saturation in the triple (tested, 64 variants the same bits);
* **the kernel boundary** is a `let` at the top of the program naming an array
  class once (`let s_a_i32_n = {0} in {1}`, the row `let@C` for every class `C`
  of the view, the condition `top`). Its frame `B` binds the class WITHOUT a
  trip, so the context inside it is `c/{a.i32.n}@0`, a context at depth 0: a
  `let` does not count toward the depth or the work bound (its bound expression
  and its body each run once). Which classes a view may bind is the
  specification's `lets` line (`sniff.py --lets a.i32.n`); each is bound at
  most once, so one let class doubles the contexts below it (the measured cost
  is in `docs/kmeans_seed.md`).

**Their test views** (`kahan.fut`'s entries `cell` and `kernels`): `cell` is
`update_centroids` for one (cluster, coordinate) cell, construct for construct,
the triple's loop with the membership test inside and the sum over the count
when the count is positive, else the previous centre (1,041 types); `kernels`
flags each point in a first kernel, names the `[n]i32` at the top and sums the
flagged points in a second (1,892 types). Both compute the yardstick's bits
through `futhark c` and the interpreter on 30 runs; 50 random genes a view pass
`futhark check` and 8 compile and equal the interpreter. **The k-means view**
(`sig/kmeans/`, `docs/kmeans_seed.md`) is the first with a kernel boundary:
f32, depth 3, `lets a.n` (the assignments, integers held exactly in f32), master
5: 14,359 types, 110,871 rows, a gene of 27,563,026 words at its program size
N = 192 (the seed is 131 functions). The faithful two-element view (`i32`
assignments) is 53,773 types and 62.7 million words at N = 96 with the pairs
alone (a layout of 488 s), 131,825 types with the triples and 237,325 with the
kernel boundary, over the 65,536 type codes: measured, and not the problem's
view for that reason.

**The pins.** The four i32 benchmarks' specifications are pinned to master
version 2 (`gen_tables.py`'s `SIG_PROBLEMS`, the sniffer's `--master`), so the
rows of version 3 changed none of their tables (the fingerprints in
`futhark_view::tests` are unchanged); k-means (`sig/kmeans/`, f32, depth 3) is
the first version-3 view: 3,804 types, 30,041 rows, a gene of 3,044,186 words
at its program size N = 96, pinned to version 3 until 2026-11-03, when its
seed (UltraDim's kernels) moved it to version 5 with the kernel boundary (the
sizes above); the pair rows and the frames of version 4, and the triple rows and
the `let` of version 5, changed none of the other pinned views (the fingerprints
are unchanged). Moving a pin is a decision with its regression re-measured.

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
