# Technical design: data types, properties and scope in the Futhark kingdom

**Status:** proposed 2026-11-05, for Andrew's agreement before any code. Andrew's design; this document writes it
down. It replaces the scope-and-size-in-the-type construction of the Futhark kingdom's views (14,359 types for
k-means), and supersedes the type count estimates of `docs/PLAN_futhark_kingdom.md` section 7 where they differ.

## 1. Context

**Objective.** Pass the size of an array along with the array, so that a gene's types are only the data types of
function arguments, and every gene still expresses a legal Futhark program.

**The problem.** The kingdom's views put two things in the type that are not data types:

1. the **sizes** of arrays (`a.n` and `a.k` are different types, as are `ix.n` and `ix.k`);
2. the **scope** (which variables are bound around a node, before or after the kernel boundary).

Every combination of sizes and scopes became a type: histogram 162 types, k-means 14,359 (85 % of them bodies at
depth 3), and the k-means view with integer assignments 237,325, over the 65,536 type codes, so it cannot load. The
k-means gene is 27.6 million words; its layout takes 34 s, and the start population's e-graph and fold 149 s and
193 s at 19.5 GB.

**The design in one paragraph.** A type is a data type: the data type of a function's argument or result. Sizes are
**properties** of values: every value is reached through a function that returns the value and its properties, and
the decoder carries the properties through the tree. Where a function needs two properties to agree (`map2` needs
equal lengths), the decoder compares them: if they agree, the plain Futhark call is written; if not, a bespoke
adapting function of ours is written, which makes them agree. Scope is context: a variable terminal names a data type
and an index, and the decoder resolves it from the variables in scope at that node.

## 2. Definitions

| term | meaning |
|---|---|
| **data type** (the gene's **type**) | the data type of a function argument or result, as Futhark spells it without sizes: `i32`, `i64`, `f32`, `bool`, `[]t`, `[][]t`, a tuple `(a, b, c)`, a function `a -> b` |
| **property** | a fact about a value that is not its data type. For an array: the length of each dimension. For an index: the array length it is in range of. Properties are what Futhark checks where values meet |
| **size expression** | how a property is known at decode time: a size name of the entry point (`n`, `k`, `d`), a literal, or `?` (a length known only when the program runs, e.g. after a filter) |
| **environment** | the variables visible at a node: the entry point's parameters, then each binder's variables from the root down to the node, each with its data type and properties |
| **adapter** | one of our own Futhark functions, rendered in a prelude with every program, that makes properties agree at run time |

The rank of an array is part of its data type (`[]f32` and `[][]f32` differ, as they do in Futhark); its lengths
are properties.

## 3. The data types of each problem

The kingdom's types are exactly the data types that appear as arguments or results of the problem's functions. From
the master table's families (`kingdoms/futhark/master/rows.tsv`):

| problem | data types | count (today) |
|---|---|---:|
| reduce, scan | `i32`, `[]i32`, `i64` (an index) | 3 (28) |
| histogram | `i32`, `[]i32`, `i64` | 3 (162) |
| pathfinder | `i32`, `[]i32`, `[][]i32`, `i64` | 4 (184) |
| a7 | `f32`, `[]f32`, `[][]f32`, `i64`, the two row-function types | 6 (11) |
| k-means, f32 | `f32`, `[]f32`, `[][]f32`, `i64`, `(f32, f32)`, `(f32, f32, f32)`, the row-function types | about 8 (14,359) |
| k-means, i32 + f32 | the above plus `i32`, `[]i32` and the mixed tuples | about 14 (237,325) |

These counts are to be confirmed by the first build.

## 4. Properties

### 4.1 What each data type carries

| data type | properties |
|---|---|
| scalar (`i32`, `f32`, `bool`) | none |
| index (`i64` used as an index) | `bound`: the size expression it is in range of, `[0, bound)` |
| `[]t` | `len`: a size expression |
| `[][]t` | `rows`, `cols`: size expressions |
| tuple | the properties of each component |
| function `a -> b` | the properties of its result given its argument's (a rule, below) |

### 4.2 Where properties come from: `Terminal_With_Properties`

Every terminal of a data type is reached through a wrapper that returns the value and its properties. We write these
ourselves; they exist at decode time (the decoder's attribute), not as Futhark code:

| terminal | value | properties |
|---|---|---|
| an entry parameter, `points : [n][d]f32` | `points` | rows = `n`, cols = `d` |
| an entry size, `k` | `k` | none (a scalar); also names the size `k` |
| a variable bound by a binder (a `map` element, a `loop` state, a `tabulate` index, a `let`) | its name | from the binder's rule (4.3) |
| a constant array (`zeros`) | `replicate L 0` | len = L, where L is the property the slot requires (4.4); `zeros` has no length of its own |
| a literal | the literal | none |

### 4.3 How properties flow up: each function's property rule

Each row of the master table gains a **property rule**: its result's properties from its arguments'. Examples:

| function | result properties |
|---|---|
| `map f xs`, `map2 f xs ys` (lengths agree) | len = len(xs) |
| `tabulate L body` | len = L; binds an index with bound = L for the body |
| `replicate L x` | len = L |
| `hist op ne L is vs` | len = L |
| `reduce`, `sum`, `minimum` | none (a scalar) |
| `transpose m` | rows = cols(m), cols = rows(m) |
| `xs[i]` | the element's properties: for a row of `[][]t`, len = cols |
| `loop (s = init) for i < L do body` | the properties of `init`; binds `s` with them and `i` with bound = L |
| `filter p xs` | len = `?` |

### 4.4 What a slot requires: properties flowing down

Some argument slots require properties, which flow top-down: the entry point's result (`[k][d]f32` requires rows =
`k`, cols = `d`); the destination length of `hist`; a `loop` body's result (it must equal the state's properties);
`replicate`'s and `tabulate`'s length argument when it is a size name. A slot with no requirement accepts any
properties.

## 5. Agreement: plain call or adapter

Where a function needs properties to agree, or a slot requires a property, the decoder compares the size
expressions symbolically:

- **They are the same expression** (both `n`): the plain Futhark call is written. The reference programs fall here
  everywhere, so they render to their own text, at full speed.
- **They differ, or one is `?`**: an adapter is written around the argument.

The adapters, rendered once in each program's prelude:

```futhark
-- an array of exactly length l: the first l elements of xs, padded with z when xs is shorter
def fit [m] 't (l: i64) (z: t) (xs: [m]t) : [l]t =
  tabulate l (\i -> if i < m then xs[i] else z)

-- the same for both dimensions of a matrix
def fit2 [r][c] 't (lr: i64) (lc: i64) (z: t) (xs: [r][c]t) : [lr][lc]t =
  tabulate lr (\i -> tabulate lc (\j -> if i < r && j < c then xs[i, j] else z))

-- an index into an array of length m, from any index: clamped into [0, m)
def clampix (m: i64) (i: i64) : i64 = i64.max 0 (i64.min (m - 1) i)
```

| where | rule |
|---|---|
| `map2 f xs ys`, lengths `a` and `b` differ | `map2 f (fit l z xs) (fit l z ys)` with `l = i64.min (length xs) (length ys)` |
| a slot requiring length L gets length `a` ≠ L | `fit L z x` |
| `xs[i]`, i's bound is not len(xs) | `xs[clampix (length xs) i]` |
| a loop body's result differs from the state | the body wrapped in `fit` to the state's length |
| the entry's result | `fit2 k d 0 r` when r's properties are not (`k`, `d`) |

`z` is the element type's zero (`0i32`, `0f32`). Every array dimension in these programs is at least 1, as the
entry points' contracts already state (`last` in `rows.tsv` assumes it), so `m - 1` in `clampix` is in range.

**What this changes:** the index types (`ix.n`, `ix.k`) disappear into one data type `i64` with a `bound` property;
the clamped index rows (`prev`, `next`, `clamp`) remain as functions on it.

## 6. Scope: the environment

As in Andrew's notes. A variable terminal is `var.T`: "a variable of data type T". Its index i is a word in a
Dc-style array beside the type's tail. At decode the node's environment is known (its ancestors are decoded first,
breadth-first); `var.T` becomes the (i mod c)-th variable of data type T in the environment, where c is how many
there are, and carries that variable's properties. If there are none, it becomes T's constant. Binders (`map`'s
element, `loop`'s state and counter, `tabulate`'s index, `let`) add their variables, with properties, to the
environment of the slot they bind. The kernel boundary is a `let`.

## 7. Decoding, in passes

1. **Express** (unchanged): Karva breadth-first, the head and tail of each data type, giving the tree of symbols.
2. **Environment, top-down:** each node's environment from its parent's plus the binder's variables; resolve each
   `var.T` (index word mod count, or the constant).
3. **Properties, bottom-up:** each node's properties from its children by the row's property rule.
4. **Requirements, top-down:** each slot's required properties from its parent (the entry's result at the root).
5. **Render:** each node's Futhark text, the plain call where properties agree, the adapter where they do not.

All five are deterministic in the gene alone. They run on the host, where programs are rendered. The GPU never needs
them: it draws, varies and selects symbols by data type, and every gene is legal whatever its index words and
properties, so the device's work is unchanged except that the types are fewer.

## 8. A worked example: histogram

Entry: `main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : [k]i32`. Data types `i32`, `[]i32`, `i64`.

The reference program `hist (+) 0 k (tabulate n (\i_n -> is[i_n] % k)) vs`:

| node | environment's `[]i32` | properties | rendered |
|---|---|---|---|
| `hist.add` (requires len `k`) | is, vs | len = `k` | plain: `hist (+) 0 k ...` |
| `tab` | is, vs | len = `n` | `tabulate n (\i_n -> ...)`, binds `i_n`, bound `n` |
| `var.[]i32`, word 1 | is, vs | `vs`, len = `n` | `vs`: indices and values both length `n`, plain |
| `get` | is, vs | element | `is[i_n]`: `i_n`'s bound `n` = len(is), plain |

The same text as today. Now a mutation turns `tab`'s length into `k` (a `tabulate k` row): indices have len `k`,
values len `n`, `hist` needs them equal, so the decoder writes `fit (i64.min k n) 0 ...` around both, and `is[i_k]`
becomes `is[clampix (length is) i_k]`. Legal, a different program, scored by its fitness.

## 9. What it costs and what it gives

| | today | this design |
|---|---|---|
| types, k-means | 14,359 | about 8 |
| gene, k-means (N = 192) | 27.6 million words | about 30,000 (to confirm) |
| integer assignments in k-means | cannot load | about 14 types |
| legality | by the type alone | by the data type plus adapters; every gene legal, no repair |
| the reference programs | render to their text | the same: their properties agree everywhere |
| a mismatched program | cannot be expressed | expressed with adapters, which cost run time and may block fusion |
| static binding of variables | yes | no: a mutation of a binder above can retarget `var.T` (a legal, neutral move) |

## 10. Where the code changes (by repository, after the merge plan of section 6.2 these are two crates)

- **fuller (symbolic):** the master table (`rows.tsv` gains the property rule and the requirement column; the size
  and scope columns go); the view (data types only); `typed_gep` decoding (passes 2-4); the renderer (adapters and the
  prelude); the e-graph (one sort per data type; rules unchanged, they rewrite rendered terms); the reader that turns
  a Futhark program into a gene (it emits `var.T` with the index that picks each variable).
- **phylu (evolution):** the typed layout (fewer types; the index words of `var.T` as a Dc-style array beside each
  tail); the point mutation edits an index word as it edits a constant; the start population (the reference program's
  gene from the reader).

## 11. Decisions for Andrew

1. The adapters' semantics: pad with the element's zero and truncate (`fit`), clamp indices (`clampix`), take the
   shorter length for elementwise pairs. Alternatives: pad by repeating the last element; map2 to the longer length.
2. Whether `?` lengths (filters) are in the first version, or only the size names and literals.
3. Whether properties beyond sizes are wanted now (value ranges, sortedness), or later.
