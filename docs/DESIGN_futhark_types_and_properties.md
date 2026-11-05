# Technical design: data types, properties and scope in the gene frame

**Status:** proposed 2026-11-05, for Andrew's agreement before any code. Andrew's design; this document writes it
down. It starts from the gene frame's data model, changes it, and then applies it to the Futhark kingdom first
(WGSL later). It supersedes the type estimates of `docs/PLAN_futhark_kingdom.md` section 7 where they differ.

## 1. Context

**Objective.** A gene's types are the data types of function arguments, and nothing else. Facts about a value that
are not its data type (an array's length, an index's range) travel with the value as **properties**. Which variables
a node can use is **scope**, decided while decoding. Every gene still expresses a legal program.

**The problem.** The gene frame's data model has one entity for types, and arity is a count per type. It has no place
for a property, so the Futhark kingdom folded sizes and scope into the type: histogram 162 types, k-means 14,359 (85 %
of them scope bodies at depth 3), the k-means view with integer assignments 237,325, over the 65,536 type codes, so it
cannot load. The k-means gene is 27.6 million words; its layout takes 34 s, and the start population's e-graph and
fold 149 s and 193 s at 19.5 GB.

## 2. The data model today

`src/geneframe.rs`:

```
Symbol (kingdom, symbol id, symbol_name, alias, semantic_id)
  └── Arity { inputs: Ty → count, outputs: Ty → count }

Ty = S | I | F | B | A | L | T1 | T2 | (NLP labels) | ... | Futhark(u16) | ...
```

- A **kingdom** is a filter over the master symbol table.
- **Arity** is a map from type to count: `add: in {F: 2}, out {F: 1}`. It has no argument order.
- `Ty::Futhark(u16)` is an index into a Futhark view's own type table. That table is where the 14,359 rows live:
  each row is a data type crossed with sizes (`a.n`, `a.k`, `ix.n`) and with a scope (the variables bound around the
  node).
- Karva's head and tail are laid out per type; the GPU draws, varies and selects symbols by type.

## 3. The data model proposed

```
                      ┌──────────────┐
                      │   DataType   │──────────────┐
                      └──────────────┘              │ 1
                        ▲ 1      ▲ 1                ▼ n
┌────────┐ 1   n ┌──────┴───┐   ┌┴─────────┐   ┌──────────────┐
│ Symbol │──────<│ Argument │   │ Binding  │   │ PropertyType │
└────────┘       └──────────┘   └──────────┘   └──────────────┘
   │ 1             (position,     (argument →     (name, kind)
   │               in | out)       a variable
   │ n                             of a DataType)
   ├──────< PropertyRule   (an output property = a function of input properties)
   └──────< Requirement    (an input property must equal another property, or a size)
```

### 3.1 The tables

| table | one row is | columns | example |
|---|---|---|---|
| **Symbol** | a function or terminal, as today | id, kingdom, name, alias, semantic_id, spelling | `map2.add`, spelled `map2 (+) {0} {1}` |
| **DataType** | a data type, as the target language spells it without sizes | id, name, kind (scalar, array, tuple, function), element / components | `f32`; `[]f32` (array of `f32`); `[][]f32`; `(f32, f32, f32)`; `[]f32 -> []f32` |
| **PropertyType** | a property that values of a data type carry | id, data type, name, kind (size, bound, range, flag) | `[]f32.len`; `[][]f32.rows`, `[][]f32.cols`; an index `i64.bound` |
| **Argument** | one ordered input or output of a symbol | symbol, position, direction (in, out), data type | `map2.add`: in 0 `[]f32`, in 1 `[]f32`, out `[]f32` |
| **PropertyRule** | how an output property follows from input properties | symbol, output property, expression over inputs | `map2.add`: out.len = in0.len; `transpose`: out.rows = in0.cols; `tabulate`: out.len = in0 (a size) |
| **Requirement** | a constraint an input property must meet | symbol, input property, must equal (another input property, a size, a constant) | `map2.add`: in0.len = in1.len; `hist`: in1.len = in2.len |
| **Binding** | a variable an argument makes visible to its own subtree | symbol, argument position, the variable's data type, its properties (by rule) | `tabulate`: argument 1 binds an index `i64` with bound = in0; `map`: the body binds one element of in0 |

### 3.2 Gene expression programming: arity stays explicit and fixed

GEP sizes every open reading frame from the symbols' arities, before any gene is expressed. This model keeps that:

1. **Every symbol's arity is fixed by its Argument rows.** The typed arity Karva needs (the count of inputs of each
   data type) is the count of those rows, derived once when the table loads. It is the same many-hot arity as today.
2. **The frames are sized by the same formulas:** each data type's head and tail from the most inputs of that type any
   symbol takes, `t = h·(n_max − 1) + 1` per type. Fewer types give fewer and smaller frames.
3. **Properties, rules, requirements and bindings never change arity.** They are read after Karva has expressed the
   tree.
4. **Adapters are not in the gene.** They are written into the rendered program text and take no cell.
5. **`var.T` is a terminal of arity 0.** Its index word is in a Dc-style array of fixed length beside its data type's
   region, one word a cell, as GEP-RNC keeps its constants.

The model forbids a symbol whose number of arguments depends on a property: arity must be known before expression.

### 3.3 What derives from what

- **Arity** (the count per data type that Karva's head and tail need) is derived from the Argument rows. The gene
  layout and the GPU's typed variation are unchanged: they read Argument and DataType only.
- **Ty codes** stay data types. `Ty::Futhark(u16)` indexes the DataType table: about 8 rows for k-means.
- **Properties, rules, requirements and bindings** are read only by the decoder, on the host.
- A kingdom with no PropertyType rows works exactly as today.

## 4. The decoder

The gene is unchanged: Karva, the head and tail of each data type, and a Dc-style array of index words beside each
type's region. Decoding runs in five passes, each deterministic in the gene alone:

1. **Express** (as today): breadth-first, giving the tree of symbols.
2. **Environment, top-down:** the root's environment is the entry point's parameters, each with its data type and
   properties from the signature (`points : [n][d]f32` → rows = `n`, cols = `d`). A child's environment is its
   parent's plus the Binding rows of the argument it fills.
3. **Variables:** each variable terminal `var.T` takes the index word w at its cell and becomes the (w mod c)-th
   variable of data type T in its environment (c of them). With none (c = 0) it becomes T's constant.
4. **Properties, bottom-up, and requirements:** each node's properties from its children by its PropertyRule rows,
   as symbolic expressions (`n`, `k`, `d`, a literal, or `?` when only known at run time); each Requirement is
   checked by comparing the expressions.
5. **Render:** where every requirement holds, the plain spelling; where one fails, the spelling wrapped in an
   **adapter**, one of our own functions that makes the property hold at run time.

This is `Terminal_With_Properties`: a terminal is reached as its value plus its properties (pass 2 and 3), and every
function carries its properties up (pass 4).

The GPU never runs passes 2 to 5. Every gene is legal whatever its index words and properties, so the device draws,
varies and selects symbols by data type, as now.

## 5. Futhark: the adapters

Rendered once in each program's prelude:

```futhark
-- exactly length l: the first l elements of xs, padded with z when xs is shorter
def fit [m] 't (l: i64) (z: t) (xs: [m]t) : [l]t =
  tabulate l (\i -> if i < m then xs[i] else z)

-- an index into an array of length m, from any index: clamped into [0, m)
def clampix (m: i64) (i: i64) : i64 = i64.max 0 (i64.min (m - 1) i)
```

| requirement fails at | rendered as |
|---|---|
| `map2 f xs ys`, lengths differ | `let l = i64.min (length xs) (length ys) in map2 f (fit l z xs) (fit l z ys)` |
| a slot requires length L, gets another | `fit L z x` |
| `xs[i]`, i's bound is not len(xs) | `xs[clampix (length xs) i]` |
| a loop body's result differs from the state | the body wrapped in `fit` to the state's length |
| the entry's result | `fit` on each dimension to the signature's sizes |

`z` is the element type's zero. Every array dimension is at least 1, as the entry points' contracts state, so
`clampix` is in range.

## 6. A worked example: histogram

Entry: `main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : [k]i32`. Data types: `i32`, `[]i32`, `i64`. Today: 162 types.

The reference program `hist (+) 0 k (tabulate n (\i -> is[i] % k)) vs`:

| node | environment's `[]i32` | properties | requirement | rendered |
|---|---|---|---|---|
| `hist.add` | is, vs | len = `k` | indices.len = values.len: `n` = `n` holds | `hist (+) 0 k ...` |
| `tabulate`, size `n` | is, vs | len = `n` | | `tabulate n (\i -> ...)`, binds `i`, bound `n` |
| `var.[]i32`, w = 1 | is, vs | `vs`, len = `n` | | `vs` |
| `get` | is, vs | element | i.bound = is.len: `n` = `n` holds | `is[i]` |

The same text as today. A mutation that changes the `tabulate` size to `k` breaks two requirements, so it renders as
`hist (+) 0 k (fit (i64.min k n) 0 (tabulate k (\i -> is[clampix n i] % k))) (fit (i64.min k n) 0 vs)`. That is
legal, a different program, scored by its fitness.

## 7. Sizes

| problem | data types | today |
|---|---|---:|
| reduce, scan | `i32`, `[]i32`, `i64` | 28 |
| histogram | `i32`, `[]i32`, `i64` | 162 |
| pathfinder | `i32`, `[]i32`, `[][]i32`, `i64` | 184 |
| a7 | `f32`, `[]f32`, `[][]f32`, `i64`, two function types | 11 |
| k-means, f32 | `f32`, `[]f32`, `[][]f32`, `i64`, `(f32, f32)`, `(f32, f32, f32)`, function types: about 8 | 14,359 |
| k-means, i32 + f32 | the above plus `i32`, `[]i32`, mixed tuples: about 14 | 237,325 (cannot load) |

### 7.1 The size of an individual

An individual is one Karva gene, sized by Ferreira's equations for a program of at most N functions: master head
h = 1 + (N − 1)·n_max, master tail t = h·(n_max − 1) + 1. Each data type is a column of that gene: a position holds one
symbol per type, and expression reads the column of the type demanded there. A column's length follows from how many
inputs of its type the symbols take (`kingdom_contract.rs`, `heads_from_program_size` and `bounds`), so it varies by
type; for k-means today the measured average is about 1,920 positions a column (27,563,026 words over 14,359 columns,
`docs/kmeans_seed.md`).

| k-means, N = 192 | type columns | positions a column (average) | symbols | index words (`var.T`) | words | bytes |
|---|---:|---:|---:|---:|---:|---:|
| today | 14,359 | ≈ 1,920 | 27,563,026 | 0 | 27.6 million | 110 MB |
| this design | ≈ 8 | ≈ 1,920 | ≈ 15,400 | ≈ 15,400 | ≈ 31,000 | ≈ 124 KB |

The positions a column stay about the same, because the program size and the arities do not change; the number of
columns falls from the cross-product of data types, sizes and scopes to the data types. Exact figures come from the
first build of the new table.

## 8. Is it good for many kingdoms?

A test of the model against the kingdoms we have and those we want. The question for each: does a type stay a data
type, and does everything else fit as a property, a requirement or a binding?

| kingdom | data types | properties | requirements | bindings | fits? |
|---|---|---|---|---|---|
| **symbolic regression** (have) | `F` | **transcendental depth** (today the types `T1`, `T2`) | `sin`'s input depth < 2 | none | yes, and better: the depth ladder becomes one data type with a depth property; the rule "no `T3`" becomes a requirement. Its adapter is not a wrap but a substitution (a failing node renders as its argument), to be decided |
| **regex** (have) | `Pattern`, `Char`, `I` | none needed (a repetition count could carry a bound) | none | none | yes, unchanged: no property rows, the model degenerates to today's |
| **TSR** (have) | `F` and its order and window types | the order and the window as properties | an order limit | none | yes, the same move as SR's depth |
| **WGSL** (have, unfinished) | `f32`, `i32`, `u32`, `bool`, `vecN<T>`, `matCxR<T>`, arrays, pointers | a vector's width, an array's length, a buffer's length, the address space (function, workgroup, storage), the workgroup size | `dot` needs equal widths; a store needs a writable address space; an index in range | `let`, `var`, loop variables, the invocation ids | yes, and it is where sizes and scope matter most after Futhark; the address space is a property, not a type |
| **Futhark** (have) | section 7 | lengths, bounds | section 5 | `map`, `loop`, `tabulate`, `let` | yes, the reason for this design |
| **NLP / English** (have, early) | the entity labels (`Person`, `Org`, `Money`, ...) | a phrase's number, tense, person | a verb agrees with its subject in number | none | yes: agreement is the textbook case of properties with a requirement |
| **SQL** (want) | relation, column, scalar types | a relation's schema (its columns and their types), a column's nullability | a join's keys have equal types; `UNION` needs equal schemas | a query's `FROM` binds its tables' columns for `WHERE` and `SELECT` | yes: the schema is a property of the relation; this design is how a typed SQL kingdom would avoid a type per schema |
| **Spark / Dagnetic** (want) | DataFrame, column, scalar | the schema, the partitioning, sortedness | a join needs compatible keys; a window needs a sort | a `withColumn` binds the new column downstream | yes, the same as SQL |
| **Lua / LuaJIT** (want) | the 8 Lua types | a table's shape (its fields), a string's length | a field access needs the field; arithmetic needs numbers | `local`, function parameters, loop variables, upvalues | yes for the types and scope; a dynamic language's properties are partly unknowable (`?`), so adapters do more of the work (a missing field reads `nil`, Lua's own rule) |
| **Python** (want) | `int`, `float`, `str`, `list`, `dict`, `tuple`, `ndarray` | a list's length, an array's shape and dtype | numpy broadcasting rules | the same scope rules as Lua | yes; numpy shapes are exactly Futhark's lengths |
| **Brainfuck** (want) | one: the cell | none | none | none | yes, trivially |

**Where it is weaker:**
1. **Adapters are kingdom-specific.** Futhark's are `fit` and `clampix`; SR's depth needs a substitution; SQL's schema
   mismatch needs a projection. The model holds the requirement; each kingdom supplies its adapters. That is the one
   part a new kingdom must write.
2. **Properties known only at run time** (Lua tables, Futhark `filter`, SQL `WHERE`) carry `?`, and every requirement
   on a `?` renders an adapter. That is legal but costs run time.
3. **Properties that need arithmetic** (a concatenation's length is `a + b`) need the rule language to hold
   expressions, not just equalities between names. The first version can keep sums symbolic and treat them as
   different from any name, rendering an adapter where one is required.

## 9. What changes, by repository

- **fuller (symbolic; a crate after the merge of plan 6.2):**
  - `src/geneframe.rs`: the tables of 3.1, with Arity derived from Argument.
  - The Futhark master table: `rows.tsv` gains the argument, rule, requirement and binding columns, and loses the
    size and scope columns.
  - The views: data types only.
  - `typed_gep` decoding: passes 2 to 4.
  - The renderer: the adapters and the prelude.
  - The e-graph: one sort per data type; its rules rewrite rendered programs and need no change.
  - The reader that turns a Futhark program into a gene: it emits `var.T` with the index that picks each variable.
- **phylu (evolution):**
  - The typed layout: fewer types, and the index words as a Dc-style array beside each type's region.
  - The point mutation edits an index word as it edits a constant.
  - The start population takes the reference program's gene from the reader.

## 10. Decisions for Andrew

1. The adapters' meaning for Futhark: pad with the element's zero and truncate (`fit`), clamp indices (`clampix`),
   take the shorter length for elementwise pairs. Alternatives: repeat the last element; map to the longer length.
2. Whether `?` properties (run-time lengths, after a filter) are in the first version.
3. Whether SR's transcendental depth moves to a property now, or stays as `T1`/`T2` until the Futhark kingdom is done.
