# PLAN: the Futhark kingdom: one master table, views over it, and the standard Futhark GP

## Context

**Objective.** Make the Futhark kingdom the standard genetic-programming system for Futhark: one master symbol table, versioned in git, and every problem's kingdom a view over it, chosen by a sniffer that reads the problem.

**Why.** Today each problem's symbol table is generated afresh from a hand-written declaration (`kingdoms/futhark/sig/<problem>/interface.tsv`, by `gen_tables.py`'s signature family). That limits a problem to the types the generator derives from the declaration:
- only the declared dimensions, so no program can make a new size (`filter`, concatenation, data-dependent lengths);
- only the inputs' element types (today all i32), so no f32, bool arrays or tuples;
- and an improvement to the generator reaches only problems whose tables are generated again.

This is a departure from Andrew's geneframe design, in which a kingdom is a QUERY over a master table.

**Approach (Andrew, 2026-11-01).**
- A master Futhark table holds every type and function.
- A sniffer reads the problem we are solving and writes a view specification: the WHERE clause.
- The view is a query over the master. Every improvement to the master improves every view.
- A view can be written out as an exact table (for the user, a cache or the witness tests), but that file is an output, never the source.
- Reachability from the root then shapes the gene layout: per-type heads only for types an expressed tree can reach. That is correct GEP and limits nothing, since an unreachable type can never be expressed.

**Status (2026-11-01).** Planned. Steps 1 and 2 of the overall plan (the WGSL bug fixes, and the merge into main) are done. The fitness-schemes work is finishing.

## Step 3: the rename, and the master table with views (code and docs)

### 3a. The rename, in the code and the docs
- The shape kingdom, with its signature and pixel families, becomes **the Futhark kingdom**: its files, modules, tables and docs.
- The 1-D kingdom becomes **the 1-D Futhark kingdom**: files renamed `futhark_1d_*`, marked deprecated in each file's header, its tests kept passing.
- The shared types (`FutharkKingdom`, `GeneRng`, the table loader) move into a module of their own.
- All imports are updated, phylu's included, along with CLAUDE.md, the READMEs, the papers' file references and the docs index.
- The rule against renaming files is set aside here by Andrew's instruction: "if we are going to fix it, we fix the code and the docs".
- **Done (2026-11-01).** The old-to-new mapping:

| old | new |
|---|---|
| `kingdoms/futhark_shapes/` (with `sig/`, `pixel/`) | `kingdoms/futhark/` |
| `kingdoms/futhark_true/` | `kingdoms/futhark_1d/` (deprecated) |
| `src/futhark_kingdom.rs`, shared part (`FutharkKingdom`, `FutharkFamily`, the rows, `from_tables`, `render`, the ADF rendering, `GeneRng`, `symbol_id`) | `src/futhark_table.rs` |
| `src/futhark_kingdom.rs`, 1-D part (`FutharkKingdom::load`, `futhark_relations`, `FUTHARK`, its tests) | `src/futhark_1d_kingdom.rs` (`load_1d`, `futhark_1d_relations`, `FUTHARK_1D` = "Futhark 1-D") |
| `FutharkKingdom::load_shapes`, `FUTHARK_SHAPES` = "Futhark Shapes" | `FutharkKingdom::load`, `FUTHARK` = "Futhark", in `src/futhark.rs` |
| `src/futhark_shapes.rs` | `src/futhark.rs` |
| `src/futhark_shapes_cost.rs` | `src/futhark_cost.rs` |
| `src/futhark_shapes_egraph.rs` | `src/futhark_saturation.rs` |
| `src/futhark_shapes_evolve.rs` | `src/futhark_search.rs` |
| `src/futhark_shapes_gpu.rs` | `src/futhark_gpu.rs` |
| `src/futhark_shapes_interp.rs` | `src/futhark_interpreter.rs` |
| `src/futhark_interp.rs` | `src/futhark_1d_interp.rs` (deprecated) |
| `src/futhark_egraph.rs` | `src/futhark_1d_egraph.rs` (deprecated) |
| `examples/futhark_shapes_evolve.rs` | `examples/futhark_a7_evolve.rs` |
| `examples/futhark_true_emit.rs` | `examples/futhark_1d_emit.rs` (deprecated) |
| `examples/futhark_rescue.rs` | `examples/futhark_1d_rescue.rs` (deprecated) |
| `docs/PLAN_futhark_shapes_product.md` | `docs/PLAN_futhark_product.md` |
| `Ty::FutharkShape`, `FUTHARK_SHAPE_CODE_BASE`, `MAX_FUTHARK_SHAPE_DUALS` | `Ty::Futhark`, `FUTHARK_CODE_BASE`, `MAX_FUTHARK_DUALS` |
| `Ty::Futhark`, `FUTHARK_CODE_BASE`, `MAX_FUTHARK_DUALS` | `Ty::Futhark1d`, `FUTHARK_1D_CODE_BASE`, `MAX_FUTHARK_1D_DUALS` |

  The device codes did not change: each family keeps its numeric base, and only the names moved. The Futhark kingdom's two modules that would have reused a 1-D module's name are `futhark_saturation` (the e-graph) and `futhark_interpreter`, so no new module takes an old module's name and `git log --follow` works on every file. Dated reports, the recorded phylu fixture `tests/fixtures/futhark_shapes_5gen/` and the papers' prose keep the old names.

### 3b. The master Futhark table
- **Rows are written over SIZE VARIABLES** (`get : [D]a × ix.D → a`), never over a problem's own sizes, which are not known until the problem is.
- **Element types:** i32, i64, f32, f64 and bool, with the operations each supports. Then tuples, and later records.
- **Classes:** arrays over size variables, rows of arrays, index types `ix.D`, scalars, function classes (as today), and contexts (the `tabulate`, loop and fold scopes) up to a depth bound.
- **The full function library:**
  - arithmetic and comparison per element type;
  - min, max and the modulus;
  - indexing, with prev, next and clamp;
  - map, map2, reduce and its sums, scan, hist, tabulate and replicate;
  - loops, transpose, rows;
  - later, filter, partition and concatenation once data-dependent sizes are designed (`docs/ASSESSMENT_dynamic_shapes.md`).
- **Versioning:** the table lives in git, and each row records the version it was added in, so a view can be pinned to a table version.
- **Witness tests,** carried over from `gen_tables.py`: every row's shape relation, legality and spelling is checked from the emitted text.

### 3c. The sniffer
- It reads the problem: the Futhark entry point's source (its parameters and result), and later the data.
- It writes a view specification:
  - the root type and the leaves (the inputs);
  - the size variables bound to the problem's sizes;
  - the element types;
  - the context depth;
  - (until 3e, the function families; a signature view now admits every family, see 3e).
- It replaces the hand-written `interface.tsv`.

### 3d. The view
- A query over the master: filter the rows the specification admits, bind the size variables, then compute the reachable types for the layout.
- It is optionally written out as the exact table, for the user, a cache or the witness tests.
- **Regression:** the four benchmarks (reduce, scan, histogram, pathfinder) become four view specifications. Each view's table must equal today's table, and the same genes must express the same programs.
- **Done (2026-11-01)**, commits e181563b, a2be66bf, dbbf58a6, following the design below:
  - `kingdoms/futhark/master/` (55 row templates, the three frames, the rules with their element domain, the pools, the matrix family's types; version 1), with master witnesses M1-M9 in `gen_tables.py` (6 of 6 planted faults caught);
  - `kingdoms/futhark/sniff.py`, which writes `sig/<problem>/view.tsv` from each benchmark's `yardstick.fut`; `interface.tsv` is gone, and `gen_tables.py --check` re-sniffs every benchmark;
  - `kingdoms/futhark/view.py` (the generator's view) and `src/futhark_view.rs` (the engine's, at runtime, cached once a process); A7's specification is `kingdoms/futhark/view.tsv`;
  - the regression: the four views and A7's equal the tables of d26a018a byte for byte from both implementations, and `futhark_sig::tests_view` shows the same genes express the same programs.
  - Not built here, for step 4: elements other than i32 in the master's rows (the format admits them; the value model does not), and the matrix family's cost table and e-graph stay generator-written from A7's view.

### 3e. Bidirectional reachability replaces the family filter

**The rule (Andrew, 2026-11-01): do not limit families by what the source happens to use.** Build the graph of types and functions from the rows of ALL families the view's element and shape can bind; compute reachability FORWARD from the inputs (which types can be built) and BACKWARD from the output toward the inputs (which types the root can use); where the two meet, include every type and function reachable both ways.

**As built** (`view.py::reachable`, `src/futhark_view.rs::reachable`, byte-identical):
- admission: version and element only for a signature view (the matrix view keeps its `families matrix`); a quantifier whose set the view does not have (`P:pairs` in a signature view) binds nothing, so foreign-shape rows expand to no rows;
- F: the least fixpoint, from the output of every terminal and constant, then the output of every function whose arguments are all in F. A function is USABLE when all its arguments are in F; there is no further cut. A type with no leaf of its own, which only a chain of functions builds, is kept: the GEP tail cannot hold one, so that is the gene layout's to design, not the view's to filter (Andrew's correction). Witness S8 ("every dual has a terminal") stays and will fail loudly when a master row first brings in such a class. Measured under this rule: **0** kept types without a leaf in all four views (every class in a context has `zeros`, `zero`/`last`, or the size terminals and the pool);
- B: the closure from the root over the usable rows; kept: the usable rows whose output is in B. The root must be in F, else an error naming it;
- the context-free classes (classes no kept function row produces) are recomputed expand → keep → recompute until they settle, bounded by the class count, an error if they do not. Measured: histogram, reduce and scan settle at once with none free; pathfinder in two rounds with `a.rows.cols` free, the same result as the old double expansion. It changed nothing in these four views;
- the sniffer no longer writes a `families` line (`FAMILY_CONSTRUCTS` and the source walk are gone); both readers reject one in a signature specification; S10 keeps its template checks and drops the family-membership half.

**Measured** (types; function rows; leaf rows = terminals and constants; head length and gene width at the runner's default program size N = the reference's function count; A7 at N = 3, unchanged):

| view | N | types | function rows | leaf rows | head length | gene width |
|---|---:|---|---|---|---|---|
| pathfinder | 13 | 832 → 832 | 3395 → 3975 | 2062 → 2062 | 14065 → 14065 | 82573 → 84015 |
| reduce | 1 | 73 → 73 | 351 → 408 | 221 → 221 | 1 → 1 | 13 → 13 |
| scan | 1 | 73 → 73 | 369 → 408 | 221 → 221 | 1 → 1 | 9 → 9 |
| histogram | 4 | 607 → 607 | 2610 → 2700 | 1779 → 1779 | 2506 → 2506 | 13945 → 13945 |
| A7 | 3 | 11 → 11 | 54 → 54 | 17 → 17 | 41 → 41 | 322 → 322 |

The types did not change: every type the new families' rows name was already in each view. At N = 1 a layout is the root's frame alone, so reduce's and scan's widths do not show their new rows. The regression: each problem's reference and planted slowdown, and A7's reference, write the programs recorded at 97dbbb80 (`tests/fixtures/futhark_view_programs/`, `futhark_sig::tests_view`); the 200 random genes and the stored cases as in 3d; the sizes above are pinned in `futhark_view::tests` and `futhark_sig::tests_view`.

### 3f. The row-table operators

**The rule (Andrew, 2026-11-01): treat the head as a table with a column per type, apply the inversion, transposition or crossover, then put the content back to its form.** A per-type operator acts inside one type's span, so with hundreds of types (pathfinder 832) one application touches about 1/types of what single-typed GEP's operator touches. A domain of the gene is a TABLE, row = position, column = type: column `T` of domain `D` is `D_T`, row `r` the cell `off_T + r` while `r < len_T`. The columns are ragged; a column takes part in a row only where it has that row.

**As built** (`src/typed_gep.rs`, `GeneLayout`): `columns`, `domain_rows` (the longest column); `invert_rows(domain, lo, hi)` reverses rows `[lo, hi)` in every column over the part it has; `transpose_rows(domain, src, len, dst)` is GEP's IS row-wise (the run copied to row `dst`, the rest shifted down, the overflow off each column's end; a column without row `src` or `dst` unchanged), RIS being `src = ris_row(..)` (the first row where the root's head column holds a function) and `dst = 0`; `crossover_rows(a, b, cut1, cut2)` one-point (`cut2 = None`) or two-point. Each moves a symbol only within its own column, so every child is legal by construction. The span-level IS is now `GeneLayout::transpose`, which the runner's per-type `transpose` uses unchanged.

**The crossover cut.** A cut is a row of the WHOLE gene's table, a column being `head_T`, then `tail_T`, then `Dc_T`: a cut spans head, tail and Dc together, as the classic recombination does over the linear gene. With one type that column IS the linear gene (head, tail, Dc), so this is exactly classic one- and two-point recombination. Cutting each domain's table at the same relative row was rejected: with one type it makes two cuts (one in the head, one in the tail), not the classic operator.

**Tests** (`typed_gep_tests::rows`): every child of 20,000 random genes per kingdom (SR, regex, the two-type fixture) and of pathfinder and histogram genes validates and expresses well typed, each cell where the operator says; with one type each row operator equals `invert`, `transpose` (and an independent classic IS), `swap_range` and the classic one- and two-point; the ragged two-type layout by hand; one seed, one result.

**Runner** (`examples/futhark_evolve.rs`): `--ops` names `invert-rows`, `transpose-rows` (IS or RIS, one half each, the head table), `cross-rows` (one- or two-point, one half each, both children kept) beside the old ones; rows drawn uniformly over the table's rows; each child a `note` mutation event with its operator. `--measure` takes `invert`, `transpose`, `crossover` and the three row operators, on the same parents and seeds (a crossover's partner is the next parent).

**Measured, the tables** (N = the reference's function count): pathfinder's head table has 36 rows (mean column 16.9), its gene table 2041 (mean column 101); histogram's 9 (mean 4.1) and 108 (mean 23). So a uniform cut of the gene table passes the end of most columns: on average a column takes part in 5 % of pathfinder's cuts and 21 % of histogram's.

**The relative rows, so every column takes part** (2026-11-01): `invert_rel`, `transpose_rel`, `crossover_rel` take the cuts and runs as fractions in [0, 1) (fixed point, `FRAC_ONE = 2^32` parts), and each column applies them at its own row `floor(f × len)` (`frac_row`, exact in integers). The absolute-row operators stay. With one type, `frac_of_row(r, len)` gives the fraction of any row exactly, so each relative operator is the absolute one and the classic one: tested beside the legality (every kingdom above, pathfinder and histogram) and the ragged layout by hand, where the relative inversion changes the short column the absolute one leaves alone. Runner: `--ops` and `--measure` take `invert-rel`, `transpose-rel` (IS or RIS; the IS run is 1 to 3 rows of the longest column as a fraction, at least one row in every column; RIS starts at the fraction of the root column's function row and inserts at 0) and `cross-rel` (one- or two-point); each child a `note` event.

## Step 3 design (3b, 3c, 3d), 2026-11-01

Read first, in full: `kingdoms/futhark/gen_tables.py` (the matrix kingdom, the signature family `sig_rows`, the pixel family, their witnesses), `src/futhark_sig.rs`, `src/futhark_table.rs`, `src/futhark.rs`'s loader, `src/futhark_reader.rs`'s parser. The design keeps every engine invariant: the engine still sees one flat typed table (`FutharkKingdom::from_tables`); only where the table comes from changes.

### D1. The master table: `kingdoms/futhark/master/`

The master is **the source**, hand-maintained and versioned in git. Nothing generates it; `gen_tables.py` reads it. Four TSV files and a version:

- `VERSION`: one integer, the master's version (1 at creation). Raised by the commit that adds or changes a row.
- `rows.tsv`: one **row template** a line. Columns:
  `ref  since  family  kind  elems  scope  when  name  out  args  spelling  note`
  - `ref`: a stable id (`M0001`), never reused or renumbered; a removed row's ref is retired.
  - `since`: the master version the row was added (or last changed) in. A view pinned to version `v` takes the rows with `since <= v`.
  - `family`: `core` (every signature view), an optional signature family (`scan`, `hist`, `mod`; later `filter`, `concat`), or `matrix` (the A7 kingdom's rows).
  - `kind`: `function`, `terminal`, `const`.
  - `elems`: the element types the row is defined for (`i32,i64,f32,f64,bool`, or `*` for every numeric one). An operation a type lacks is a missing element, never a guarded row: `/` is `f32,f64` only (it traps on an integer 0).
  - `scope`: the row's **quantifiers**, outermost first, `;`-separated: `C` (every context), `D:dims`, `D:rank1`, `D2:rank1`, `S:states`, `V:vars(<class>)` (the context's bound variables of a class), `I:inputs(<class>)`, `Y:rows(<class>)` (the classes whose rows are `<class>`), `X:free` (the context-free classes), and for the matrix family `P:pairs`, `R:triples`.
  - `when`: a condition on the binding: `shallow` (the context is shallower than the view's depth: binders, reductions, `map2`, `scan`, `hist`, `replicate`, the work bound), or empty.
  - `name`, `out`, `args`, `spelling`, `note`: templates. A **variable** is `{UPPER}` (`{E}` the element, `{D}`, `{S}`, `{S.type}` the Futhark type of a state, `{Y.dims}`, `{Y.brackets}`, `{K}` the frame number = depth + 1, `{OUT}` the row's output dual, `{V}`, `{I}`); a child placeholder is `{0}`, `{1}`, ... as today, so the two cannot be confused. A type in `out`/`args` is a class template in the row's context; `T+<frame>` is `T` in the context plus that frame (`{E}+T-{D}` is a `tabulate` body).
- `frames.tsv`: the **binders that make contexts**: `kind  code  scope  binds`, e.g. `T  T-{D}  D:dims  j{K}:ix.{D}` and `L  L-{D}-{S}  D:dims;S:states  s{K}:{S},i{K}:ix.{D}`. A context is a stack of frames up to the view's depth; its name is `c/<code>/<code>`, its variables the frames' `binds`. This is today's representation (`i32:c/F-rows-a.cols/T-cols`), written down once instead of in two codes.
- `types.tsv`: the matrix family's declared classes (slot, form, meaning). A signature view's types are the duals its rows name, their slot and meaning read from the class (`ix.`, `a.`, the element).
- `rules.tsv`: the e-graph rule templates: `on  elems  needs  rule` (`elem` or `arr` types; `elems` the element types the identity is exact for, `i32` only today: wrapping integer arithmetic, so no float rule is admitted by accident; `needs` a row that must exist, as today's `map2.min`), with `{add}` the constructor of row `add` at that type, rewritten left to right. A **rule witness** evaluates both sides of every rule on random values of every admitted element with the rows' own semantics (finding 12).
- `pools.tsv`: the default literal pool of each element.

**Order is meaning.** Today's tables are ordered by `sig_rows`'s nested loops. The view expands the rows **as nested blocks**: consecutive rows that share their first quantifier form one block, the block iterates that quantifier, and inside it the rows recurse on their remaining quantifiers. The master is written in the order that reproduces today's loops exactly (one context loop around everything, then the free inputs).

**Context-free classes are derived per view**, not declared: a class that no admitted function row of the view produces is context-free (today's `free`: `walls` is `a.rows.cols`, with no context). Since a dual depends on it, the view expands twice: once to find what its function rows produce, then for real. A future rank-2 producer would contextualise such a class, so it changes only views that pin the new master version (review finding 2).

### D2. The view specification: `kingdoms/futhark/sig/<problem>/view.tsv`

Written by the sniffer; it replaces `interface.tsv` and keeps its four-column lines. Rust parses it with its own `ViewSpec` parser and derives the `Signature` from it (review finding 4):

```
# kind	name	element	dims
# source	tests/corpus/futhark_problems/histogram/yardstick.fut, entry main
master	1		            (the master version pinned)
element	i32		            (the element types; one today)
size	k	i32	            (a dimension given as an i32 value parameter)
input	is	i32	n
input	vs	i32	n
output	main	i32	k
depth	2
literals	0,1,2,-1,10
dims	k,n		            (derived: order of first appearance, size inputs included)
```

Since 3e a signature specification has no `families` line (the readers reject one): the view admits every family. A7's spec (`kingdoms/futhark/view.tsv`) is `shape matrix`, `roles n,m,p`, `root n,p`, `families matrix`: hand-written, since A7 has no entry source until step 4 makes it a view problem of `futhark_evolve`.

### D3. The view algorithm (one in Python, one in Rust; a test requires them equal)

1. **Filter**: `since <= master` and the spec's element in `elems`; a signature view admits every family (3e), the matrix view the families its spec names.
2. **Bind**: derive the sets from the spec, exactly as `sig_signature` does today: `dims` (first appearance), `classes` (each input's class and its row classes, the output's, each dim's rank-1 class), `rank1`, `states` = the element then `rank1`, `free`, the inputs; the contexts from `frames.tsv` (`[()]`, then each context of depth `d` extended by every frame, `d < depth`). Expand the rows' blocks, substitute the variables, and form each type's dual (`class:context`, or the bare class when it is context-free). Name a row `name` as templated (`add@{OUT}` gives today's `add@i32:c`).
3. **Reachability**, both ways since 3e: forward from the leaves (what can be built), backward from the root (`output` in context `c`) over the rows forward reachability makes usable. It is what makes the gene layout per-type heads only for reachable types.
4. **Order and write**: functions, then terminals and constants, numbered `F00001...`; types ordered as today (the root, then depth, then name; the matrix family in template order), numbered `T0001...`; the e-graph datatype, rules, rows and sorts from the rows and `rules.tsv`, with today's sort and constructor naming.

The **Rust** side (`src/futhark_view.rs`): `View::load(master, spec) -> View { types_tsv, functions_tsv, datatype, rules, rows, sorts }`, the same text the generator writes; `futhark_sig::Instance` stays a `Copy` static handle `{ name, spec: include_str!(view.tsv) }`, and its view is computed at runtime from `include_str!(master/*)` + the spec once a process, in one cache entry a view name holding the view text and its parsed `ViewSpec`, from which the kingdom, the signature and the bridge are all read, so they cannot disagree (finding 9), and `View::write(dir)` writes the exact table (the option). The committed written-out tables (`sig/<p>/types.tsv`, ...) stay as the regression record and for the witnesses; nothing loads them at runtime. For A7 the Rust view computes `types.tsv`/`functions.tsv`; its `costs.tsv` and e-graph files stay generator-written from the same view (A7's runner is deprecated by step 4, so its family-specific cost and rule code is not ported).

### D4. Witnesses: master witnesses, then per-view checks

- **Master witnesses** (`M1...`, on `rows.tsv` itself, independent of the expansion code): refs unique and well-formed, `since <= VERSION`; every `{UPPER}` variable is bound by the row's scope or is global (`E`, `K`, `OUT`); a function's spelling holds `{0}..{n-1}` once each, in order (Futhark's application is positional); every binder's last argument is its own frame (`+T-{D}` for `tab`, `+L-..`/`+F-..` for the loops) and its spelling binds exactly that frame's variables (the tabulate and loop spellings restated by hand in the witness); every heavy row carries `when shallow`; every index row is one of the clamped templates (restated); no integer element admits `/`; every class a template names has a type template.
- **Per-view checks**: today's `S1..S10` on each written signature view (root, context parse, terminals bound, binder bodies, the work bound, the index templates, a terminal for every dual, the pool, the e-graph's bijection), and today's `W1..W13` on A7's. Their models now come from the spec (dims, families) and not from the deleted `SIG_INSTANCES`; where `S10` compared a spelling with `SIG_SCANS`, it now compares with the master row: a consistency check, no longer an independent one (recorded).

### D5. The sniffer: `kingdoms/futhark/sniff.py`

- **Input**: the problem's Futhark source and its entry (`tests/corpus/futhark_problems/<p>/yardstick.fut`, `main`). The compiler is pinned: the sniffer requires `futhark --version` 0.28.1, the version whose print it parses (finding 6).
- **The CLI types it**: `futhark dev -t` prints the type-checked program with every size inferred (measured: histogram's `: []i32` prints as `: [i64.i32 k]i32`). The sniffer parses that entry line: size parameters, `((name: T): T)` parameters, the result type. A scalar `i32` parameter that appears as `i64.i32 p` in a dimension is a **size input**; another scalar is an input value.
- **Writes** `view.tsv`: the element (the one element type of the parameters and result; two or more is an error until step 4), the inputs, the output, `dims` in first-appearance order, the master's default depth (2) and pool, `master` = the current version. (Until 3e it also wrote the **families**, admitting an optional family when its construct appeared in the problem's source (finding 5); 3e removed that: every view admits every family.) The accepted signature subset is explicit: size parameters, parameters of an element type over size parameters, a scalar `i32` parameter that appears exactly as `i64.i32 p` in a dimension (a size input, still usable as a value), a result over those; anything else stops the sniffer with a diagnostic naming it (finding 7). Whether every view should carry every family was decided by Andrew in 3e: it should.
- `gen_tables.py --check` re-runs the sniffer on every benchmark and requires its `view.tsv` byte for byte (a hand-edited spec fails `cargo test`, as a hand-edited table does).

### D6. The regression plan

1. Before any old code is deleted: the new generator's four signature views and A7's view equal today's committed `types.tsv`, `functions.tsv` and e-graph files **byte for byte** (and A7's `costs.tsv`).
2. A Rust test: for each problem, the runtime view (master + spec) equals the written-out files byte for byte, and its `Signature` equals the one today's `interface.tsv` gave.
3. A Rust test: the same genes express the same programs. Per problem, the kingdom from the runtime view and from the written-out tables have equal duals and rows, equal contract relations, constant lengths and layout alphabets at every cell (so every gene has the same alphabet and the same expression); every row is rendered with every constant index; 200 random genes (fixed seed) and the reference express to the same trees and Futhark source; the CPU interpreter gives the same outputs on the stored cases (finding 8).
4. The full fuller and phylu suites, clippy (fuller default and `gpu,wgsl`; phylu default and `gpu,wgsl`), and the GPU tests.

**Known difference, a fix**: pathfinder's yardstick entry names its parameter `wall`, the declaration `walls`. The sniffer reads the source, so the entry (our own wrapper, not the published kernel) becomes `entry main [rows][cols] (walls: [rows][cols]i32) : [cols]i32 = let wall = walls in ...`, keeping the published kernel's text and the kingdom's name (pools and genes name `walls`).

### D7. f32 and tuples (built in step 4; the format admits them now)

- **Review findings 10, 11 and 1 bind step 4**: values become typed (raw f32 bits in the binary format, a NaN and -0 policy for comparison), float rules are admitted only where bit-exact (no float associativity), the class becomes an AST with an injective text encoding before tuples (`t.a.n.a.m` is ambiguous), and two views that share one population need their own `Ty` namespace (today each kingdom object carries its duals and no two views meet).
- **f32** enters as an element: the master rows gain `f32` in `elems` where the operation exists (`+ - * /`, `min`, `max`, `abs`, `sqrt`, `exp`, `log`, the reductions `f32.sum` and friends, scans and histograms with `f32.inf`/`-f32.inf` neutrals), and `mod`, `clamp` from a value, and the integer-only rows do not. A single-element f32 view needs no new class names. A **multi-element** view puts the element in the class (`a.n` becomes `a.f32.n`, the scalar `f32`); a single-element view keeps today's names (a recorded naming rule), and conversion rows (`f32.i32 {0}`) carry an element pair. The binary format and the interpreter's `Array` become element-typed.
- **Tuples**: a tuple class `t.<c1>.<c2>` is a type; its rows are the constructor `({0}, {1})` and the projections `{0}.0`, `{0}.1`; a loop may carry a tuple state (a `states` member); an entry returning a tuple is a tuple root. Futhark's binary format writes a tuple as its components in order, so the harness reads several values. Records follow as named tuples.

## Step 4: the standard Futhark GP

- **The checklist, written down:** every Futhark problem gets
  - a named fitness scheme (`fuller::fitness`), slope-checked, with the correctness gate;
  - interleaved one-upload timing, at least 1 s a side, with the floor from the champion's own path;
  - the timing-case output check;
  - the ring canvas;
  - the operator mix with per-operator logging;
  - the runner standard: stream, killme.trigger, checkpoints, hall of fame and aa_gate.tsv.
- **`--eval rust` is the default.** The shim paths stay as test references, in the cross-path test.
- **One runner:** A7 becomes a view problem under `futhark_evolve`, and `futhark_a7_evolve` (with its loop, `fuller::futhark_search`) is deprecated. phylu's shape_typed engine keeps working through the shared types.
- **Types:** f32 first (k-means and image problems), then tuples, then records, all in the master table, so every view gains them.
- **GPU residency** (the core paper's table): today only program execution runs on the device for the signature runner. The target is a device population, with expression, variation and selection on the device and fitness that stays there, as in SR and regex.

## Steps 5 to 7 (unchanged)

5. Prove the Futhark kingdom works: the improvement run from Fable's histogram program on the fixed build, with speed reported every generation; then timing screening (a few pairs first, the full 11 for contenders).
6. Prove the k-means kingdom works: a full run with all operators and working model edits, 3 seeds, and a random-search control.
7. Done: the scratch clean-up.
