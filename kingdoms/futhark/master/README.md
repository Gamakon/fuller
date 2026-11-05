# Master 6: the Futhark master table's format

**Objective.** Master 6 restates the master table in the gene frame's data model
(`docs/DESIGN_futhark_types_and_properties.md` section 3; `src/geneframe.rs`, commit `e526a62f`): a type is a
**data type** (`f32`, `[]f32`, `[][]f32`, `i64`, `size`, tuples, function types) and nothing else. An array's length
and an index's bound are **properties**, carried by **property rules** and checked by **requirements**. The variables
a node may use are its **environment**, built by the decoder from **bindings**. Where a requirement fails, the
rendered program carries an **adapter**.

**Approach.** Master 6 keeps master 5's shape: one wide row a template in `rows.tsv`, instantiated by the view. The
size and scope columns are dropped, and four columns are added: `rules`, `requires`, `binds`, `trip`.

**Status (2026-11-05).** Plan section 7, task F0: this document is the format. No master 6 file exists yet. F1a writes
them, under `kingdoms/futhark/master/6/`, beside master 5, which stays as it is and loadable until task D1. At D1,
master 5's files are deleted and `6/`'s files move up into `master/`.

**Why.** Folding sizes and scope into the type made the k-means view 14,359 types and a gene of 27.6 million words,
and kept the `i32` + `f32` view from loading at all (plan section 7, "Why").

## 1. The files

| file | replaces | one row is |
|---|---|---|
| `VERSION` | `VERSION` | the master's version: `6` |
| `datatypes.tsv` | `types.tsv` (the matrix classes, the two function types) and every class `view.py` derived | a data type template |
| `properties.tsv` | (new) | a property a data type's values carry: the PropertyType table |
| `rows.tsv` | `rows.tsv` | a symbol template, with its Argument, PropertyRule, Requirement and Binding rows in its columns |
| `adapters.tsv` | (new; the adapters were in no table) | an adapter: what it repairs, where it runs, its rendering |
| `pools.tsv` | `pools.tsv`, unchanged | an element's literal pool. **There are no size literals**: every size is a size name |
| `rules.tsv` | `rules.tsv`, same columns | an e-graph rule. Its `{name}` placeholders name master 6 rows (task F5) |
| (none) | `frames.tsv` | deleted: a frame becomes the `binds` and `trip` columns of its binder row |

Every file is tab-separated with one header line. A field holds no tab. An empty field means "none". Within a field,
`;` separates items; no property expression contains `;`.

## 2. `datatypes.tsv`

Columns: `name`, `kind`, `members`, `over`, `tag`, `constant`, `constant_props`, `note`.

| name | kind | members | over | tag | constant | constant_props |
|---|---|---|---|---|---|---|
| `{E}` | scalar | | E | `{E}` | `0{E}` | |
| `i64` | scalar | | | `i64` | `0i64` | `bound=1` |
| `size` | scalar | | | `size` | (none) | |
| `[]{E}` | array | `{E}` | E | `a.{E}` | `[0{E}]` | `len=1` |
| `[][]{E}` | array | `[]{E}` | E | `m.{E}` | `[[0{E}]]` | `rows=1 & cols=1` |
| `({A}, {B})` | tuple | `{A};{B}` | A,B:elements | `{A}.{B}` | `(0{A}, 0{B})` | |
| `({A}, {B}, {C})` | tuple | `{A};{B};{C}` | A,B,C:elements | `{A}.{B}.{C}` | `(0{A}, 0{B}, 0{C})` | |
| `[]f32 -> []f32` | function | `[]f32 -> []f32` | | `fn.arr_f32_arr_f32` | `\xs -> map (\_ -> 0.0) xs` | |
| `[]f32 -> []f32 -> []f32` | function | `[]f32;[]f32 -> []f32` | | `fn.arr2_f32_arr_f32_arr_f32` | `\xs _ys -> map (\_ -> 0.0) xs` | |

- `name` is the data type as Futhark spells it without sizes. It is the DataType row's name. `{E}` ranges over the
  view's elements (`i32`, `f32`).
- `i64` is **the index** (master 5's `ix.D`). `size` is a length or trip count, rendered as Futhark's `i64`. Its only
  terminal is `var@size`, and no row returns it.
- `tag` names the data type inside symbol names and sexps, which hold no brackets: `map2.add@a.f32`, `var@size`.
- `constant` is what a `var` cell gives when its data type has no variable in scope (c = 0). It is also what the
  work-bound substitution gives a node that has no input of its own data type. `constant_props` are the constant's
  properties: `[0f32]` has length 1, and `fit` applies to it at any slot that requires another length.
- Every data type but `size` has a constant (witness, F1a). `size` needs none: every entry has a size in scope, and
  the view reports it as a bug if one does not.
- The A7 family's `mat.{P}` classes are all `[][]f32`.

## 3. `properties.tsv`

Columns: `data_type`, `name`, `kind`, `meaning`. These are the PropertyType rows; `kind` is G1's `PropertyKind`.

| data_type | name | kind | meaning |
|---|---|---|---|
| `[]{E}` | `len` | size | the length |
| `[][]{E}` | `rows` | size | the outer length |
| `[][]{E}` | `cols` | size | the inner length (every row's) |
| `i64` | `bound` | bound | the exclusive upper bound: the index is valid for every array of length at least `bound` |
| `size` | `value` | size | the size itself, a size name of the environment |

**The environment's property `trips`** is not a property of any data type. It is the number of trip binders above
the node: `tabulate`, `tabrows`, `loop`, `loop1`; a `let` adds none. It is 0 at the root, and is read as `env.trips`
(section 5.3).

Scalars, tuples and function types carry no properties. Every property value is a G1 `PropertyExpr`: a size name, a
literal, `?`, `in<k>.<property>`, a sum, or a `min`. In this master each is one of three things:

- a size name (`n`), because sizes are chosen by `var@size` and never computed;
- the literal 1 (a constant, `zero`);
- a `min` of those (after `map2`).

No row here produces `?` or a sum. F1a's witness reports any that appears.

## 4. `rows.tsv`

Columns: `ref`, `since`, `family`, `kind`, `elems`, `over`, `name`, `out`, `args`, `rules`, `requires`, `binds`,
`trip`, `spelling`, `note`.

| column | meaning |
|---|---|
| `ref` | Stable id. A converted row keeps its master 5 ref. A row that merges several master 5 rows (`var`) takes a new ref from M0087 on, and its `note` names the refs it replaces. |
| `since` | The version that introduced the row. A converted row keeps master 5's value (the least of those it merges), so a view pinned to version v sees the same rows, converted. A row new in master 6 has 6. |
| `family`, `elems` | As master 5. |
| `kind` | `function`, `terminal`, `const` (a Dc literal of `pools.tsv`), and the new `var`: a terminal of arity 0 whose cell has an index word in the Dv array, flagged `is_variable` (task G2). |
| `over` | **Data-type quantifiers only**, instantiated by the view. `S:states` is the scalar of each element and `[]{E}` of each element. `S:tuples` and `S:tuples3` are the pairs and triples of elements. `S:lets` is the view's `lets` data types. `Y:all` is every data type of the view but `size`. `T:vars` is every data type that has a variable (a root input, a zero array, a size, or a binding's data type). `{E}` is bound by `elems`, as in master 5. A quantified data type's components are `{S.a}`, `{S.b}`, `{S.c}`, and its tag is `{S.tag}`. |
| `name` | `<op>@<out tag>`, with no size in it: `get@{E}`, `map2.add@a.{E}`. |
| `out`, `args` | Data type templates; `args` lists the inputs in order, `;`-separated. These are the Argument rows: input k is at position k, and the output is at position 0. **A `size` argument comes first**, in the order the spelling reads the sizes. |
| `rules` | `;`-separated `<property>=<expr>`, one PropertyRule each, for the output's properties. **A rule holds whether or not the row's adapters fire** (map2's `len=min(in0.len, in1.len)`), so no adapter changes a rule. A `var` row has no rules: its properties are the chosen variable's (decoder pass 3), or the constant's when c = 0. |
| `requires` | `;`-separated `<subject> <relation> <expr> -> <adapter>`, one Requirement each (section 5). |
| `binds` | `;`-separated `<argument>:<data type> <property>=<expr> & ...`, one Binding row each. The variables an argument makes visible to its own subtree, in the order their names are given. |
| `trip` | The argument whose subtree runs once a trip of this row (its environment's `trips` is the node's plus one); empty for none. |
| `spelling` | Futhark text with placeholders (below). |
| `note` | As master 5. |

**An instantiated rule, requirement or binding property that names a property its data type does not carry is
dropped from that instance.** So one `loop` row serves a scalar state and an array state, and one `let` row serves
every data type. F1a's witness checks that every dropped item was dropped for exactly that reason.

**Spelling placeholders.**

| placeholder | renders |
|---|---|
| `{k}` | operand k's text |
| `{k.<property>}` | operand k's property as Futhark text: a size name as itself, a literal as itself, `min(a, b)` as `i64.min a b`. A `?` property binds its operand with `let` under an adapter name first and reads `length` of the name, never of the subtree. |
| `{k.type}` | operand k's data type with its properties as sizes: `[n]f32`, `[k][d]f32`, `(f32, f32, i32)`, `f32` |
| `{k.vj}` | the name of the j-th variable that argument k binds (from 0, in `binds` order). F2 reproduces master 5's `var_name` from the data type, the properties and the ordinal: `i_n`, `s2_f32`, `s_a_n` |
| `{v}` | a `var` row's chosen variable, or the data type's constant when c = 0 |
| `{E}`, `{E.tag}`, `{S.a}`, `{S.tag}`, ... | the quantifiers, expanded by the view as in master 5 |

Master 5's `{D}`, `{D.ix}`, `{D2}`, `{Y.d0}`, `{Y.ix}`, `{S.var}` and `{S.type}` are gone. A size became an operand
(`{0}`), a bound variable became `{k.vj}`, and a type became `{k.type}`.

## 5. Requirements, adapters and the work bound (the decisions F0 owns)

### 5.1 Relations

A requirement's relation is `=`, `<=` or `<`. G1's `Requirement` has no relation (its field is `must_equal`); G3 adds
the relation column. `=` is the master's default for lengths. `<=` is for indices (adversarial H2). `<` is for the work
bound. The decoder (G3) compares normal forms: sums flattened and sorted; `min` arguments sorted, duplicates removed,
and a one-argument `min` replaced by its argument. Then:

- `a = b` holds when the normal forms are identical.
- `a <= b` holds when they are identical, or a literal compares with a literal, or `a` is the literal 1 and `b` is
  any size expression (every dimension is at least 1, as the entries' contracts state), or `b` is a `min` and
  `a <= ` every argument, or `a` is a `min` and one argument `<= b`.
- `a < b` holds when both are literals and a < b. `env.trips` is always a literal at decode time.
- `?` on either side fails every relation.

A failure that a smarter comparison would have avoided only costs a needless adapter, which the cost charges (risk 4).

### 5.2 Which kinds compare (G1 review finding 1)

G3's "the two sides are the same kind" check uses this table, not equality of kinds:

| subject's kind | may compare with | relations |
|---|---|---|
| `size` (`len`, `rows`, `cols`, `value`) | `size`, `bound`, a literal | `=`, `<=` |
| `bound` | `size`, `bound`, a literal | `<=` only (`=` with a bound is a table bug) |
| `env.trips` (a count) | a literal | `<` |
| `range` | `range` | `=` |
| `flag` | `flag` | `=` |

`size` and `bound` are both counts of elements, so `get`'s `in1.bound <= in0.len` is legal. A literal has no kind. A
`min` or a sum takes its terms' kind, and its terms must all be comparable.

### 5.3 The work bound as a property (plan 7.1 choice 1; Andrew's confirmation is plan 7.5a item 1)

Every heavy row carries the requirement `env.trips < depth -> subst`. The heavy rows are S6's list (`gen_tables.py`
`HEAVY_OPS`): `tab`, `tabrows`, `replicate`, `sum`, `minimum`, `maximum`, `map2.*`, `scan.*`, `hist.*`,
`histrows.*`, `rowdiv`, `loop`, `loop1`.

- `depth` is the view specification's `depth`. The view replaces it with its literal, so the k-means view's rows read
  `env.trips < 3`.
- A view with no `depth` line (A7) drops the requirement. The matrix family, as in master 5, has no binder, so its
  trips is always 0 and it carries none.
- **`env.trips` is a requirement subject that is not an input's property.** G3 extends `Requirement`'s subject to be
  either `in<k>.<property>` or `env.trips`. G3 also reads the `trip` column as an attribute of the binding argument.
- The substitution adapter `subst` renders the failing node as its first input of its own data type when it has one
  (`map2.*`, `scan.*`, `rowdiv`, `loop`'s state, which is in1). Otherwise it renders the data type's constant (`sum`,
  `tab`, `hist.*`). G3 decides the substitution once, as an annotation. The renderer, the interpreter, `work_parts`,
  the reader and the e-graph read that annotation and never decide it again.
- **The work-bound requirement is checked first.** If it fails, the node is substituted and none of its other adapters
  render.
- **An adapter that is heavy (every `fit`) runs at the trips of the place it renders:**
  - a slot `fit`, `fit.pair` and `fit.inner` at the node's own;
  - `fit.state` and `fit.body`, which wrap a body, at the body's (the node's plus one).

  If a heavy adapter would run at `trips >= depth`, the node that owns its requirement takes the work-bound
  substitution instead. So no rendered program does more work than `max(size) ** depth`, adapters included.
- `top` is dropped (7.1 choice 2). `let` is legal anywhere and binds by its `binds` column. M5's `top` clause is
  rewritten by F1a, not carried over.

### 5.4 A requirement on a binder's body (G1 review finding 2)

A lambda body is an input argument of its row, so its length rule is an ordinary input requirement, which G1's
Requirement already expresses. `tabrows` (M0063) takes `size;size;[]{E}` (rows, cols, the body). Its requirement is
`in2.len = in1.value -> fit.body`, where in1 is a size chosen by `var@size` outside the lambda. So the requirement
holds only when the body's length is a size from outside the lambda. When it fails, the body renders as
`fit {1} 0{E} ({2})`, and no size escapes the lambda. A loop's array state is the same: `in2.len = in1.len -> fit.state`.

### 5.5 Adapters: `adapters.tsv`, and how each requirement names its adapter

Columns: `name`, `repairs`, `runs_at`, `heavy`, `rendering`. **Each requirement names its adapter after `->`**, one
adapter for each requirement. A row with two failing requirements renders both, in `requires` order.

| name | repairs | runs at | heavy | rendering |
|---|---|---|---|---|
| `clampix` | `in<i>.bound <= in<a>.len` (or `.rows`) | node | no | operand i becomes `clampix {a.len} {i}` |
| `fit` | `in<k>.<p> = <expr>` at a slot | node | yes | operand k becomes `fit <expr> 0{E} {k}` |
| `fit.pair` | `in<j>.len = in<k>.len` (`map2.*`, `hist.*`) | node | yes | `let ad_a = {k} let ad_b = {j} let ad_l = i64.min (length ad_a) (length ad_b) in <spelling with fit ad_l 0{E} ad_a and fit ad_l 0{E} ad_b as the two operands>` |
| `fit.state` | a loop body's `len` against its state's | body | yes | the body becomes `fit {1.len} 0{E} ({2})`; a `?`-length state is first `let ad_s = {1} let ad_m = length ad_s in`, and the state is declared `[ad_m]` |
| `fit.body` | a lambda body's length against a size argument | body | yes | the body becomes `fit {k} 0{E} ({body})` |
| `fit.inner` | an inner dimension (`matmul`, `map2_rows`, `histrows` columns) | node | yes | `map (fit l 0{E}) {k}` |
| `fit.root` | the entry's result against the signature's sizes | root | yes | `fit` on each dimension, inner ones by `map (fit ..)` |
| `subst` | `env.trips < depth` | node | no | section 5.3 |

- **The prelude** (design section 5) holds `fit` and `clampix`, and only those the program uses. A program with no
  failing requirement renders exactly as master 5's.
- **Reserved names.**
  - `fit` and `clampix` are the adapters' functions.
  - `ad_a`, `ad_b`, `ad_l`, `ad_m` and `ad_s` are the adapters' `let`s.
  - No operand's text ever names an `ad_` variable, so nested adapters reuse the same names by shadowing. The text
    grows linearly with nesting, and a subtree renders the same wherever it sits.
  - `sniff.py` reports as a bug an entry parameter named `fit` or `clampix`, or one that starts with `ad_`.
- **The root's requirement** comes from the view specification's `output` line. For k-means, `out.rows = k` and
  `out.cols = d`, with adapter `fit.root`.

## 6. How each master 5 column maps

| master 5 | master 6 |
|---|---|
| `rows.tsv` `ref`, `since`, `family`, `kind`, `elems`, `note` | kept (section 4; `kind` gains `var`) |
| `scope`, **dropped** | Its data-type quantifiers move to `over`: `S:states`, `S:tuples`, `S:tuples3`, `S:lets`, `Y:classes` becomes `Y:all`, `D:rank1` becomes `[]{E}`, `Y:rows(..)` becomes `[][]{E}`. Its size quantifiers (`D:dims`, `D2`, `P:pairs`, `Q:swap`, `R:triples`) become `size` arguments or properties. Its scope quantifiers (`C`, `V:vars`, `I:inputs`, `X:free`) become the one `var@{T}` row and the environment. |
| `when`, **dropped** | `shallow` becomes `env.trips < depth -> subst` in `requires`; `top` is dropped |
| `name` | Sizes removed: `get.{D}@{OUT}` becomes `get@{E}`, `hist.add.{D2}@{OUT}` becomes `hist.add@a.{E}` |
| `out`, `args` | Classes become data types. The frame suffix `+T-{D}` is removed, and becomes `binds` and `trip`. A `{D}` the spelling read becomes a `size` argument, first. |
| `spelling` | Placeholders as section 4 |
| (new) | `rules`, `requires`, `binds`, `trip` |
| `types.tsv` (the classes) | `datatypes.tsv` and `properties.tsv` |
| `frames.tsv` (`T`, `L`, `F`, `L2`, `F2`, `L3`, `F3`, `B`) | the binder rows' `binds`, in the frame's order (states, then the counter), and `trip` (every frame but `B`) |
| `pools.tsv`, `rules.tsv` | unchanged in format |

**Where master 5's rows go, by group** (F1a converts them):

- **Arithmetic, `sel`, `div`, `sqrt` and the conversions** (M0001-8, M0060, M0061, M0075, M0076): `C` is dropped,
  and nothing else changes.
- **Terminals of variables and inputs** (M0017, M0023, M0037, M0038, M0042, M0053, M0067, M0086) become `var@{T}`.
- **The zero arrays** (M0039) become root variables of `[]{E}`, one for each size and element.
- **The size as a value** (M0014) becomes `sval@{E}` (`size` to `{E}`, `{E}.i64 {0}`). This is the one conversion a
  size has.
- **`mod`, the clamps, `last`** (M0015, M0020, M0056, M0022) take a `size` argument.
- **`prev`, `next`** (M0018, M0019) carry `bound=in0.bound`. `next` is spelled `i64.min ({0.bound} - 1) ({0} + 1)`.
- **The matrix family** takes `[][]f32` with `rows`/`cols` rules. `matmul` requires `in1.rows = in0.cols ->
  fit.inner`.

## 7. Example rows, converted by hand

Fields not shown are empty. Each example lists the master 6 row first. "Instance" is the row in a named view as G1's
tables: Argument (A), PropertyRule (R), Requirement (Q, with G3's relation), Binding (B).

**map2.add** (M0026)

```
ref M0026  since 1  family core  kind function  elems i32,f32
name     map2.add@a.{E}
out      []{E}
args     []{E};[]{E}
rules    len=min(in0.len, in1.len)
requires in1.len = in0.len -> fit.pair; env.trips < depth -> subst
spelling map2 (+) {0} {1}
```

Instance, k-means (f32, depth 3):

- A: in 0 `[]f32`, in 1 `[]f32`, out `[]f32`.
- R: `out.len = min(in0.len, in1.len)`.
- Q: `in1.len = in0.len` (fit.pair), and `env.trips < 3` (subst renders in0).

With both lengths `n`, the length is `min(n, n)`, which normalises to `n`. No adapter fires, and the text is master
5's.

**get** (M0009)

```
ref M0009  since 1  family core  kind function  elems i32,f32
name     get@{E}
out      {E}
args     []{E};i64
requires in1.bound <= in0.len -> clampix
spelling {0}[{1}]
```

- A: in 0 `[]f32`, in 1 `i64`, out `f32`.
- Q: `in1.bound <= in0.len` (`<=`; bound against size is comparable, section 5.2).
- `get` is not heavy.
- `xs[i_n]` with `xs` of length `n`: `n <= n` holds. With `i_k` instead: `k <= n` fails, and the text is
  `xs[clampix n i_k]`.

**zero**, a literal index (M0021)

```
ref M0021  since 1  family core  kind terminal  elems i32,f32
name zero@i64   out i64   rules bound=1   spelling 0i64
```

Inside `row` (M0036: `args [][]{E};i64`, `rules len=in0.cols`, `requires in1.bound <= in0.rows -> clampix`), k-means's
`(centres)[(0i64)]` checks `1 <= k`, which holds. It renders unchanged, with no `clampix` (H2).

**tab**, a binder with a Binding row (M0024; frame `T`)

```
ref M0024  since 2  family core  kind function  elems i32,f32
name     tab@a.{E}
out      []{E}
args     size;{E}
rules    len=in0.value
requires env.trips < depth -> subst
binds    1:i64 bound=in0.value
trip     1
spelling tabulate {0} (\{1.v0} -> {1})
```

- A: in 0 `size`, in 1 `f32`, out `[]f32`.
- R: `out.len = in0.value`.
- Q: `env.trips < 3`; there is no own-type input, so `subst` renders `[0f32]`, of length 1.
- B: argument 1 binds an `i64` with `bound = in0.value`, and adds one trip.
- With in0 = `var@size` choosing `n`: `tabulate n (\i_n -> ...)`.

**loop**, a scalar or array state (M0040; frame `L`)

```
ref M0040  since 2  family core  kind function  elems i32,f32  over S:states
name     loop@{S.tag}
out      {S}
args     size;{S};{S}
rules    len=in1.len
requires in2.len = in1.len -> fit.state; env.trips < depth -> subst
binds    2:{S} len=in1.len; 2:i64 bound=in0.value
trip     2
spelling loop ({2.v0}: {1.type}) = {1} for {2.v1} < {0} do {2}
```

- At S = `f32`, the `len` rule, the `len` requirement and the `len` binding property are dropped (`f32` carries no
  `len`).
- At S = `[]f32`: R `out.len = in1.len`; Q `in2.len = in1.len` (fit.state, which runs at the body's trips) and
  `env.trips < 3` (subst renders the state in1); B argument 2 binds `s_a_n: []f32` with `len = in1.len`, and
  `i_d: i64` with `bound = in0.value`.
- k-means's `loop (s_f32: f32) = 0f32 for i_d < d do ...` is the scalar instance.

**the Kahan triple** (M0083, frame `L3`, with M0077 `triple`)

```
ref M0083  since 5  family core  kind function  elems i32,f32  over S:tuples3
name     loop@{S.tag}
out      {S}
args     size;{S};{S}
requires env.trips < depth -> subst
binds    2:{S.a}; 2:{S.b}; 2:{S.c}; 2:i64 bound=in0.value
trip     2
spelling loop (({2.v0}, {2.v1}, {2.v2}): {1.type}) = {1} for {2.v3} < {0} do {2}

ref M0077  since 5  family core  kind function  elems i32,f32  over S:tuples3
name triple@{S.tag}   out {S}   args {S.a};{S.b};{S.c}   spelling ({0}, {1}, {2})
```

Instance at S = `(f32, f32, i32)`, with UltraDim's (sum, compensation, count):

- A: in 0 `size`, in 1 and in 2 `(f32, f32, i32)`, out the same.
- There are no rules: a tuple of scalars carries no properties.
- Q: `env.trips < depth`.
- B: four Binding rows on argument 2, `s_f32: f32`, `s2_f32: f32`, `s_i32: i32` and `i_n: i64` (bound `n`).
- Text: `loop ((s_f32, s2_f32, s_i32): (f32, f32, i32)) = (0f32, 0f32, 0i32) for i_n < n do ...`, master 5's. The
  pair loop (M0073, frame `L2`) is the same with two components.

**hist** (M0033, i32; M0057 is its f32 spelling)

```
ref M0033  since 1  family hist  kind function  elems i32
name     hist.add@a.{E}
out      []{E}
args     size;[]{E};[]{E}
rules    len=in0.value
requires in2.len = in1.len -> fit.pair; env.trips < depth -> subst
spelling hist (+) 0{E} {0} (map i64.{E} {1}) {2}
```

Design section 6's mutation (the `tabulate` choosing `k` instead of `n`) fails `in2.len = in1.len`, since `k = n`
does not hold. `fit.pair` fits both operands to `i64.min (length ad_a) (length ad_b)`. Because the rule reads
`in0.value`, the output is still of length `k`.

**sum** (M0010)

```
ref M0010  since 1  family core  kind function  elems i32,f32
name sum@{E}   out {E}   args []{E}   requires env.trips < depth -> subst   spelling {E}.sum {0}
```

At trips 3 in a depth-3 view, the requirement fails. With no input of its own data type, `sum` renders `0f32`.

**let** (M0085; frame `B`; the `top` condition dropped)

```
ref M0085  since 5  family core  kind function  elems *  over S:lets;Y:all
name     let@{Y.tag}
out      {Y}
args     {S};{Y}
rules    len=in1.len; rows=in1.rows; cols=in1.cols; bound=in1.bound
binds    1:{S} len=in0.len & rows=in0.rows & cols=in0.cols
spelling let {1.v0} = {0} in {1}
```

- There is no `trip` and no work bound: the bound expression and the body each run once.
- At S = `[]f32` and Y = `[][]f32`: R `rows = in1.rows`, `cols = in1.cols`; B argument 1 binds `s_a_n: []f32` with
  `len = in0.len`.

**var, `var@{T}` and `var@size`** (new, M0087; replaces M0017, M0023, M0037, M0038, M0042, M0053, M0067, M0086)

```
ref M0087  since 1  family core  kind var  elems *  over T:vars
name var@{T.tag}   out {T}   spelling {v}
```

- The cell's Dv index word w chooses the (w mod c)-th variable of T in the environment.
- Its properties are that variable's: `points` has `rows=n & cols=d`, `i_n` has `bound=n`, and the size `k` has
  `value=k`.
- At c = 0 it renders T's constant, with the constant's properties.
- `var@size` is T = `size`. Its variables are the root's size names (the specification's `size` lines, in order:
  `n`, `d`, `k`) and any size a binder introduces. It is the only way a gene names a length.

**const** (M0016)

```
ref M0016  since 1  family core  kind const  elems i32,f32
name const@{E}   out {E}
```

A literal of `{E}`'s pool (`pools.tsv`), drawn from Dc as in master 5. It has no properties.

## 8. The view specification (written by `sniff.py`, task F1a)

- `input` and `output` lines are unchanged. Their element and dims give the data type and its properties.
- `size` lines replace `dims`. In order, they are the root's `size` variables.
- `lets` names data types (`[]f32`), not classes.
- `depth` is the literal of `env.trips < depth`.
- **The root environment**, in order:
  - the inputs in signature order;
  - then the zero arrays (`zeros_n`, or `zeros_f32_n` in a view of several elements), one for each rank-1 array of
    master 5's view;
  - then the sizes.

  Its order is G3's one public function; this list is its input.
