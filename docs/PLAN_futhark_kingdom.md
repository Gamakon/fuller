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

### 3g. The review's four fixes (2026-11-02)

**Why.** Fable's review (`docs/reviews/2026-11-01_futhark_kingdom_design_fable.md`) found that writing each lambda scope's frame STACK into the type multiplied the types (pathfinder 832, histogram 607), padded the gene and diluted the operators. Andrew's ruling: gene width is never a memory constraint; narrowing is only for a smaller search space, and may remove only what cannot contribute. Four fixes, each its own commit with its tests.

1. **Program size is an experiment parameter** (9e0f3061). `futhark_sig::DEFAULT_PROGRAM_SIZE = 32`, above every reference (pathfinder 13 functions, histogram 4, reduce and scan 1), the runner's default for `--program-size` (before: the reference's own count, so reduce and scan had N = 1, a gene of one root cell), recorded in the run card (`program_size`, `program_size_given`, `program_size_default`). The runner stops with a message if the reference or the planted slowdown does not fit (`typed_gep::fits`); tested for all four problems.
2. **The domain operators** (1fe8a742, `src/futhark_operators.rs`, moved from the runner unchanged). Measured on the CPU, 100 applications each, the reference and planted slowdown on random padding at N = 32 (`futhark_operators::tests`):
   - The review's 99.97 % is the share of CELLS that are non-coding (pathfinder 24 coding of 217,015 at N = 32; histogram 7 of 143,977), and its 98 % the share of (type, domain) spans (8 coding of 1,819; 5 of 1,313). It applies to an operator that picks a (type, domain) or a cell uniformly. **No operator of the runner does**: `invert`, `transpose` and `crossover` already draw a coding cell of the parent (crossover: of the first parent) and act on its span; they changed the program in 84, 57, 86 of 100 (pathfinder) and 94, 37, 27 (histogram).
   - It does apply, in effect, to the ROW operators, which act on every column but draw their rows uniformly over the whole table, mostly past the expressed cells: `cross-rows` changed the program in 0 of 100 on both problems, `cross-rel` 6 and 0, `invert-rel` 11 and 0, `invert-rows` 6 and 1, `transpose-rows` 44 and 42, `transpose-rel` 67 and 54.
   - New names, the existing ones unchanged: `invert-rows-coding`, `transpose-rows-coding`, `cross-rows-coding` draw a coding cell (of either parent for the crossover) and choose the rows around its row; 100, 59, 97 of 100 (pathfinder) and 98, 36, 93 (histogram). In `--ops` and `--measure`.
3. **One view implementation, in Rust** (0d819cf4). `futhark_view::write_committed` (command `examples/futhark_views DIR`) writes every view's table; `gen_tables.py --views DIR` runs the witnesses on them (S1 to S10 per view, the master witnesses, A7's costs and e-graph, which stay generator-written until step 4). `view.py` keeps only the master and specification readers. The committed per-problem tables (`sig/<p>/types.tsv`, `functions.tsv`, `egraph.*`, 4.5 MB) are gone; each view's written text is pinned by `View::fingerprint` (FNV-1a), equal at the change to that of the removed files.
4. **Contexts are sets, not stacks** (this section's last commit). A context was the frame stack (`c/F-rows-a.cols/T-cols`); it is now the COUNT of bound variables of each class plus the depth (`c/{a.cols,ix.cols,ix.rows}@2`, `ix.cols*2` when two binders over `cols` are in scope), and a variable is named by its class and its ordinal in that class (`futhark_view::var_name`: `i_cols`, `i2_cols`, `s_a_cols`, `s_i32`). Frames bind classes (`master/frames.tsv`); the three binder rows' spellings name their variable (`{D.ix}`, `{S.var}`, the next ordinal in the row's context); master VERSION 2, the specifications re-sniffed. The interpreter keeps a slot a (class, ordinal); the reader refuses a pattern rename that would capture an outer variable; `futhark_sig::build_stack_named` reads a program in the old names (the pools, fixtures and reports of before) and the committed pools were rewritten with it (`examples/futhark_pool_check --rename`).

**A pure SET of classes was built and measured first, as the review proposed, and lost programs**: 4 of pathfinder's 45 model programs (`sig/pathfinder/pool/llm.tsv` seq 22, 29, 40, 41, e.g. `(tab (loop.cols 0 (add s2 (sel (sub (sub (ival.cols i2) (ival.cols j1)) 1) (ival.cols i2) 0))))`) read the tabulate's `cols` index from inside a loop over `cols`; the review had counted 0 of 112. So the review's control was applied: a count per class, which loses none.

Measured (types; function rows; leaf rows = terminals and constants; gene width at the reference's function count N = 13, 4, 1, 1, and at the runner's default N = 32):

| view | contexts | types | function rows | leaf rows | width at N = ref | width at N = 32 |
|---|---|---:|---:|---:|---:|---:|
| pathfinder | stack (before) | 832 | 3975 | 2062 | 84,015 | 217,015 |
| | set (measured, not kept) | 151 | 1062 | 340 | | 50,419 |
| | count (built) | 184 | 1188 | 441 | 22,261 | 57,487 |
| histogram | stack | 607 | 2700 | 1779 | 13,945 | 143,977 |
| | set | 141 | 972 | 374 | | 44,651 |
| | count | 162 | 1032 | 455 | 4,655 | 47,999 |
| reduce | stack | 73 | 408 | 221 | 13 | 22,799 |
| | set | 24 | 188 | 61 | | 9,188 |
| | count | 28 | 202 | 79 | 11 | 10,025 |
| scan | stack | 73 | 408 | 221 | 9 | 22,794 |
| | set | 24 | 188 | 61 | | 9,184 |
| | count | 28 | 202 | 79 | 8 | 10,021 |

The review expected at most 150 types for pathfinder and 120 for histogram (sets); measured 151 and 141 with sets, 184 and 162 with counts.

**The programs** (`futhark_sig::tests_pool`, `tests_view`): all 842 pool programs (`bench`, `llm`, `alternatives` of histogram and pathfinder; 112 in `bench` + `llm`) read from their old text, compute the same values on every stored case as under the stack contexts (host interpreter, fingerprints recorded before the change in `tests/fixtures/futhark_pool_values/`), are written in the pools in the new names, and the 824 of the root type express through a gene at N = 32. The four references and planted slowdowns write programs that differ from the recorded ones (`tests/fixtures/futhark_view_programs/`) in binder names only (checked token by token) and are exact on every stored case; A7's program is unchanged byte for byte. The generator's witnesses (S1 to S10, S8 every type has a leaf) pass on every view; the random-gene, e-graph and compile tests of `futhark_sig` pass unchanged. On the new layout the operator measurement of fix 2 gives within 4 of the same counts (`futhark_operators::tests`).

**To run on the GPU later** (not run: the GPU was allocated elsewhere): the review's measurements 2 to 5 with the before binary (`/private/tmp/claude-501/keep/futhark_evolve_before_setctx`, commit b195fc6f) against this one.

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

Since 3e a signature specification has no `families` line (the readers reject one): the view admits every family. A7's spec (`kingdoms/futhark/view.tsv`) is `shape matrix`, `roles n,m,p`, `root n,p`, `families matrix`: hand-written, since A7 has no entry source (it is a problem of `futhark_evolve` since step 4b, through the same specification).

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
  **Written (2026-11-03): `docs/CHECKLIST_futhark_problem.md`**, one row an item, one column a problem (reduce, scan,
  histogram, pathfinder, kmeans, a7), each cell the code or test that proves it or `not yet` with its reason (14 cells:
  the slope check not run for reduce, scan, kmeans, a7; no test split for reduce, scan, kmeans; the GPU timing not run
  for kmeans, a7; no ring-canvas pool for reduce, scan, kmeans, a7; the measured GPU cost not wired for a7); the cheap
  claims are checked by `futhark_sig::tests_checklist` (every problem loads, has a default scheme, its reference and
  planted slowdown fit the layout and are exact on every stored case by the host interpreter, its timing input is the
  declared shape; the table has every problem and every item).
- **`--eval rust` is the default.** The shim paths stay as test references, in the cross-path test.
- **One runner:** A7 becomes a view problem under `futhark_evolve`, and `futhark_a7_evolve` (with its loop, `fuller::futhark_search`) is deprecated. phylu's shape_typed engine keeps working through the shared types. **Done (2026-11-03), 4b below.**

### 4b. One runner: A7 as a problem of `futhark_evolve` (done 2026-11-03)

- `futhark_sig::A7_PROBLEM` (`--problem a7`): the instance is the matrix view (`kingdoms/futhark/view.tsv`, `shape
  matrix`, the one `FutharkKingdom::load` loads); `signature` gives the matrix view its signature from the kingdom
  (`matrix_signature`: the nine input terminals' spellings `m_n_n .. m_p_p` with their classes' roles as dimensions,
  the root `[n][p]f32`, depth 0); the program renderer emits the kingdom's **prelude** (`futhark_sig::prelude`, the
  `matmul` helper) before the entry, and no `zeros_D` let a row does not spell; the batch host emits the prelude once
  and puts the yardstick in a module when there is one (its own `def matmul` cannot clash); the host interpreter
  covers the matrix rows (`transpose`, `matmul` as the helper's left fold, `map` and `map2` over rows, the row
  functions) and `Interp::work` costs them (a traversal its cells, the product n x m x p); the builder reads a row's
  full name when `op@dual` is not that row (`matmul@mat.n.mxmat.m.p`, `transpose@mat.n.m`, `mat.n.m.input`) and
  `sexp` writes it so; the bridge carries its ruleset's name, and the matrix view's is the generator's
  (`kingdoms/futhark/egraph.*`, `futhark_shape`), so `variants` and the saturated fold work unchanged.
- The reference is the product computed twice (`A7_REFERENCE` = `futhark_search::product_twice`), the winner the
  product once (`A7_WINNER`), the planted slowdown the reference under `map id` (`A7_PLANTED`). **Measured
  (`futhark_sig::tests_a7`):** the three programs are exact on 6 of 6 stored cases each by the host interpreter and
  on 18 of 18 case runs compiled with `futhark c` (the plain program and the quality program of every case); the
  e-graph's variants of the reference hold the winner, and the planted slowdown's hold both; 40 random genes at
  N = 32 are accepted by `futhark check` and the first 10 compile and equal the interpreter bit for bit; the batch
  source (two members with quality and timed entries, the text copy and the yardstick) holds the helper once at the
  top level and passes `futhark check`; the timing input (n = m = p = 256, the first of every matrix -0.0) has the
  declared shape, the reference's work on it is 3.4e7 node executions (under the bound), and the interpreter equals
  `futhark c` on it.
- The stored cases: `tests/corpus/futhark_problems/a7/` (`make_cases.sh`: `futhark dataset`, f32 in [-8, 8]; 3 train,
  2 held, 1 test at 64 x 64 x 64; outputs by the yardstick through `futhark c`).
- The static cost (`futhark_sig::static_cost` -> `futhark_cost`) is reported by the runner (`reference_static_cost`,
  `best_static_cost` at the timing input) and stays a speed input where a scheme asks; the `futhark_sig` schemes time
  the GPU. The measured GPU cost (`futhark_gpu::measure`) is the deprecated runner's and phylu `shape_fit`'s, not
  wired into this runner (the checklist's one a7 not-yet besides the GPU runs).
- `examples/futhark_a7_evolve.rs` is DEPRECATED: it stays, prints a deprecation line pointing to
  `futhark_evolve --problem a7`, and its tests (`tests/runner_standard.rs`, `futhark_search::tests`) pass unchanged;
  phylu's `evolve::shape_typed` compiles on the same types.
- **Not run (the GPU is the lead's):** `futhark_evolve --problem a7 --check --eval rust` (the slope check and the
  gate) and `--regime P --eval rust` (the first A7 run on the one runner: the timed pairs, `aa_gate.tsv`); the
  commands are in the checklist. The C path ran: see the checklist's smoke run.
- **Types:** f32 first (k-means and image problems), then tuples, then records, all in the master table, so every view gains them. f32 and k-means: done, 4a below.
- **GPU residency** (the core paper's table): today only program execution runs on the device for the signature runner. The target is a device population, with expression, variation and selection on the device and fitness that stays there, as in SR and regex.

### 4a. f32 first: typed values, the f32 rows, k-means as a view (done 2026-11-03)

**Typed values.** `futhark_sig::Array` carries its element (`Elem::I32 | Elem::F32`) and each value's raw 32 bits;
Futhark's binary format is read and written for both (` i32`, ` f32`); the interpreter's scalar is `Num::I | Num::F`
with Futhark's arithmetic (i32 wraps; f32 is IEEE with `fmin`/`fmax`, `i64.f32` and the folds as measured through
`futhark c`, `kingdoms/futhark/README.md`); the quality inputs are computed over the values as `f64` (exact for both)
and give the i32 formula the same bits as before (`quality_fingerprint` pinned at a535d464: `80b71efa5c5adfb0`,
the study's `7405ef68d2104e9c`, every part `756d9a9553fe716b`). **The comparison policy:** bit for bit; any NaN
agrees with any NaN (the payload is not a value: Futhark prints every NaN as `f32.nan`, hardware chooses the
payload); `-0.0` and `+0.0` differ. It is what the stored cases' bytes (`futhark c -b`) and `futhark_rust`'s k-means
exact gate compare as, and `futhark_interpreter::identical`. A cell differs when the policy says so (a sign of a zero
counts), its error is the values' distance (0 for the two zeros; infinite for a NaN against a number, squashed to 1).

**Rows added or widened (master version 3).** Widening `elems` keeps a row's `since` (a view of another element sees
what it saw); the four i32 specifications are pinned to version 2 (`SIG_PROBLEMS`, `sniff.py --master`), so their
tables and fingerprints are unchanged. Separate f32 rows where the spelling differs (the `i64.f32` conversion is
undefined for a NaN, an infinity or a value out of range, so the f32 spellings test `f32.isnan` and clamp as an f32
before converting; `x == x` is folded to true by Futhark's simplifier, measured).

| family | rows | elements | note |
|---|---|---|---|
| core (widened) | add, sub, mul, min, max, neg, abs, sel, get, sum, minimum, maximum, ival, the size, the literal, the variables, prev, next, zero, last, tab, replicate, map2.{add,sub,min,max}, row, zeros, loop, loop1, the inputs | i32 → i32,f32 | M0001-M0014, M0016-M0019, M0021-M0029, M0036-M0042 |
| scan (widened) | scan.{add,min,max} | i32 → i32,f32 | M0030-M0032 |
| core (new) | clamp (f32 spelling) | f32 | M0056 |
| hist (new) | hist.{add,min,max} (f32 spelling) | f32 | M0057-M0059 |
| core (new) | div `{0} / {1}`, sqrt | f32 | M0060, M0061: total on f32 only |
| core (new) | map2.mul | i32,f32 | M0062 |
| core (new) | tabrows: `tabulate K (\i_K -> <row>)`, a rank-2 class from its rows | i32,f32 | M0063; the binder's frame is T-K |
| hist (new) | histrows.add: `hist (map2 (+)) (replicate D 0) K (map .. is) rows` | i32 (M0064), f32 (M0065) | the published k-means's cluster sums; a new quantifier `V2:class(a.{D2}.{D})` |
| core (new) | rowdiv: `map2 (\r s -> map (/ s) r)` | f32 | M0066 |
| core (new) | a rank-2 input as a terminal in every context | i32,f32 | M0067: once a row produces a rank-2 class it is no longer context-free |
| i32 only | mod, clamp (i32), hist.* (i32) | i32 | unchanged |

Pools: `f32 0,1,2,-1,0.5,10`. Rules (f32, bit-exact only): commutativity of add and mul, `neg neg a = a`, `map2.add`
and `map2.mul` commutative, `map2.min a a = a`, `map2.max a a = a`; `sub a b = add a (neg b)` dropped (see the
README: the extraction never returned the `sub` form). The witnesses: M1-M9 over both elements (M9 evaluates each
rule on its element's edges under the policy), S1-S10 extended to f32 (the clamp template, the hist spellings, the
tabulate of rows' frame, histrows and rowdiv, the f32 pool).

**k-means as a view.** `sniff.py --depth 3` writes `sig/kmeans/view.tsv` from `kmeans/yardstick.fut` (points
`[n][d]f32`, centres `[k][d]f32`, output `[k][d]f32`, master 3). Depth 3: a point's nearest centre is a loop over
`k` inside the tabulate over `n`, and its distance is a reduction over `d` inside that (a heavy row holds only above
the depth). Measured: 3,804 types, 30,041 rows (21,620 functions, 8,421 leaves); the gene at N = 96 is 3,044,186
words (N = 32: 993,370). The reference (`KMEANS_REFERENCE`, 72 functions: the membership written twice, the kingdom
has no `let`) is the published step: `rowdiv (histrows.add.n M points) (map2.max (replicate 1) (hist.add.n M
(replicate 1)))` with `M = tab (loop.k 0 (sel (dist best - dist j) best j))`. **Exact (the policy) on 5 of 5
stored cases through the host interpreter, through `futhark c`, and batched; the planted slowdown (the ones under a
`map2 min a a`, 74 functions) exact on 5 of 5, and the e-graph's variants (256 members) exact on 5 of 5 with the
smallest at the reference's work.** The f32 rows equal `futhark c` bit for bit on 37 programs over arrays holding
NaN, the two zeros, the two infinities, a denormal and 2^24 + 1 (`tests_kmeans::the_f32_rows_equal_futhark_c_on_the_edges`).
50 random genes at N = 96 are accepted by `futhark check` and the first 12 compile and equal the interpreter on three
small cases (n = d = k = 1 included). **The champion** (`docs/kmeans_champion_s1/`) is a WGSL kernel of the scatter
form: a stable counting sort by cluster, then each cell's sum over its cluster's points in point order, which on the
sequential backend is the same left fold as `hist`; its program in the Futhark kingdom is the published (hist) form
above. Its distinct algorithm (the counting sort) has no Futhark rows: a data-dependent reorder is the dynamic-shapes
design (`docs/ASSESSMENT_dynamic_shapes.md`), not step 4.

**The runner.** `futhark_evolve --problem kmeans` loads the view, builds the layout at the problem's program size
(`Problem::program_size`, k-means 96, the i32 benchmarks `DEFAULT_PROGRAM_SIZE` = 32; recorded in the run card as
before), passes both controls (the reference exact on all 5 cases by the host interpreter and compiled), computes the
yardstick's output on the timing input (the champion's shape, n = 8,192, d = 256, k = 1,024, `timing_input`) with
`futhark c`, and starts the slope check (1,756 one-edit neighbours of the reference's gene); run on the CPU to that
point 2026-11-03 (`--backend c --eval single`, 24 programs scored in 18 s, then stopped: the check is the GPU's).
The batch host admits f32 on the rust path (`futhark-rust-rt` reads every element) and still refuses it on the C
driver (i32 only). **Not run (the GPU is the lead's):** `--eval rust` end to end. **Measured, to decide:** the
reference's work at the timing input is 2.1e11 node executions (the host interpreter's model, the membership twice)
against `TIMING_WORK_CAP` = 5.9e10, so at this timing input the runner would not time the reference; the host
interpreter runs the reference at about 4e7 node executions a second (the stored cases in 0.5 s), so the timing input
is checked through `futhark c` only (`tests_kmeans::the_kmeans_timing_input_...`).

**Not done (step 4's next pieces):** two elements in one view (k-means's membership is an `f32` holding an index
because the view has one element; D7's `a.f32.n` naming), tuples and records, GPU residency. A7 as a problem of the
one runner: done, 4b.

### 4c. Corrections after the review of 2026-11-03 (planned, for review)

**What went wrong.** Two decisions in 4a broke the project rules (`Rules.md`: no shortcuts, no failovers, no graceful
diversions, no skirting issues), and Andrew found them:

1. **The k-means seed was swapped.** The job was to translate the existing working software, UltraDim's k-means
   (`lib/projection-voting-kmeans/src/gpu_kmeans.rs`: the distance kernel, `assign_points`, and `update_centroids` with
   its Kahan-compensated sum), into a gene of this kingdom. Instead the seed became a textbook Futhark form
   (`KMEANS_REFERENCE`: `hist` cluster sums, no compensation), because the kingdom could express it. That moved the
   start of the search away from the prototype.
2. **The data was bent to a rule.** The GPU adds f32 in another order than the sequential reference, so the correct
   program was called wrong under "bit for bit". The fix chosen (06126f86) quantised the inputs to multiples of 2^-6
   and assumed every sum stays under 2^24 granules. Customer data has no such bound (magnitudes of 10^3 give sums in
   the 10^5s), and compensated sums are MORE accurate than the reference, so a bit-for-bit rule against a sequential
   fold would reject the better program. The seed must carry the prototype's compensation, the kingdom must be able
   to express it, the test data must span magnitudes, and the correctness criterion must not punish accuracy. What
   evolution then keeps or drops is evolution's result, measured by that criterion.

**What stands.** The i32 problems (reduce, scan, histogram, pathfinder) are seeded from the published programs of
`diku-dk/futhark-benchmarks`, which are the prototypes of the Futhark product; A7 is UltraDim's own matrix product.
The master table's f32 rows, the typed values and the Futhark binary format for f32 (4a) are needed and stay. The
view machinery (step 3) is untouched. The WGSL k-means champion (`docs/kmeans_champion_s1`) was seeded from UltraDim's
own kernel text, so its seed is right. It dropped A7's Kahan compensation for 1.6 % of speed, and that result was
measured on [0, 1) data on Metal only (where the compiler had already removed the compensation): the change request
(`docs/papers/2026-10-31_kmeans_centroid_update.md`) states those conditions, and the magnitude cases of item 3 below
are the measurement that says whether it holds on harder data. Evolution's result stands as measured; the correction
here is to the measurement's data and criterion, not to what evolution may find.

**What is invalidated and reverted.**
- 06126f86 (dyadic cases and timing input, `quantise.fut`, `kmeans_dyadic`, the plan paragraph "The data is dyadic").
- `KMEANS_REFERENCE` as the seed; the program may stay in the pool as one alternative, never as the reference.
- The bit-for-bit comparison as the correctness criterion for f32 problems (k-means AND A7: A7's GPU check has not
  run, and f32 matrix products meet the same ordering). The i32 problems keep bit for bit, which is exact there.
- The checklist rows "reference exact" for kmeans and a7 are re-pointed at the new criterion once built.

**The corrected design.**
1. **The seed is the prototype, construct for construct.** UltraDim's three k-means kernels become one Futhark
   program: the n x k distance matrix, the per-point argmin over k in index order (`assign_points`: the first centre
   wins a tie), and the per-(cluster, dimension) compensated sum over the points in point order with the count and the
   division (`update_centroids`). The translation is written out side by side with the WGSL (`docs/kmeans_seed.md`)
   and checked line by line BEFORE any run. What the kingdom lacks is added to the master table as rows with tests,
   never worked around: a sequential loop whose state is a PAIR (sum, compensation) — tuples, the next type of step
   4, are therefore required now, not later; an element type for the assignments (u32/i32 indices) beside f32 in one
   view — "two elements in one view" is therefore required now; the compensated-sum step as rows (`y = x - c; t = s +
   y; c = (t - s) - y; s = t`), which the e-graph must NOT simplify (`(t - s) - y` is 0 in real arithmetic and not in
   f32: no real-number rule may touch it; the rule table gets a test that the Kahan form survives saturation).
2. **SUPERSEDED (Andrew, 2026-11-03, twice: see `docs/KMEANS_FITNESS.md`, the specification as code).** k-means is not
   an exact computation: the centroid update is an approximate, iterative process with no unique right answer, so
   agreement with a reference -- bits, memberships, a distance to f64 centres under a rounding bound -- is the wrong
   measure. The fitness is the PROBLEM'S OWN OBJECTIVE: the program is run to UltraDim's stopping rule (assignment,
   update, inertia change < 1e-4 or 100 iterations), candidate and seed alike, on each fitness case, and five inputs a
   case go to the HFF, each relative to the seed's run: UltraDim's cosine inertia and mean log10 angular percentile
   of the final centroids, the wall clock to the stopping point (interleaved with the seed), the iterations, the
   inertia gained an iteration. No gate, no stage 2. The loop, the stopping test and the measures are the harness's
   (Rust, `futhark_sig::kmeans_converged`, reimplementing `cosine_inertia` and `exact_mean_log10_p`; read-only on
   UltraDim). The f64 reference program (`reference64.fut`) is kept only as a check of the translation, never to
   score. The paragraph below is the superseded design, kept for the record:
   *Correctness for f32 is a bound, not bits.* The reference is the same algorithm computed in f64 (the seed's
   semantics, Kahan included). A program is correct on a case when every output cell lies within the rounding bound
   derived from that case's inputs (Higham's gamma bound for a sum of m terms in any order, plus the division's ulps:
   the machinery of `futhark_rust::kmeans_bound`, without the dyadic shortcut) AND its total error against the f64
   reference is no larger than the seed's. The second clause makes the criterion measure accuracy rather than
   agreement with one order of additions; a program that keeps the seed's accuracy by any means passes it. The membership (assignments) is compared exactly, as integers, with a tie rule
   stated. The policy is a named fitness scheme (`fuller::fitness`), so the bit-for-bit schemes stay registered and
   marked, per the fitness rule.
3. **Test data spans magnitudes.** Cases at scales 10^-3, 1, 10^3 and 10^5, mixed scales within a case, clusters of
   very unequal counts, and a case built to break an uncompensated sum (a few large values among many small ones).
   Random draws (`futhark dataset`), never quantised. The timing input keeps UltraDim's shape (n 8,192, d 256,
   k 1,024) at scale 1 and at 10^3.
4. **The oracle for the translation.** Before evolution, the Futhark seed is run against UltraDim's own WGSL kernels
   (through fuller's wgpu harness, as the WGSL kingdom already does) on the same cases: the assignments equal, the
   centres within the bound and with the same error class (compensated). That proves the translation, not a test of
   a textbook form.
5. **Then step 6 as planned**, from this seed, with speed measured after correctness, and the change request to
   UltraDim re-issued from what that run finds.

**Status (2026-11-03).**
- (a) DONE: fc013686 reverts 06126f86; bd929652 makes the reference slot an explicit state (`futhark_sig::Seed`:
  `Program` or `NotYetTranslated`, which k-means is, with the prototype, the plan and the pool named; the runner and
  the checklist test say so and stop), moves the textbook step to `kingdoms/futhark/sig/kmeans/pool/` as
  `KMEANS_TEXTBOOK`, and makes the stored answer optional (`Case::output`, `Case::answer`): k-means stores inputs only.
- (e) DONE as `docs/KMEANS_FITNESS.md` specifies it (not as item 2 above): bd929652 adds the scheme
  `futhark-kmeans-converged` (`fuller::fitness`, k-means's default by `Problem::fitness`; `scheme_for_problem` refuses
  a scheme of the other kind), the loop, the measures and the five inputs (`futhark_sig::kmeans_converged`), the
  runner's converged judge (`--eval single`, the C backend the whole measure with the processes' wall clock, the GPU
  backend the shim's executable; not run: the seed is not translated), and the magnitude cases
  (`tests/corpus/futhark_problems/kmeans/`: scales 10^-3, 1, 10^3, 10^5; mixed scales across coordinates; cluster
  sizes 352:50:97:1 with two empty; the breaking case, 4 points at 10^4 first among 2,000 around 1, on which the plain
  f32 sum is 3.4e-5 from the f64 reference and the Kahan sum 1.1e-6; 5 train, 2 held, 1 test; the timing input at
  UltraDim's shape at scale 1 and 10^3, `kmeans_timing_input`). Tests: the inputs' arithmetic on hand-made runs; the
  measures against closed forms and UltraDim's anchors; the loop on UltraDim's step in Rust (`ultradim_step_f32`, the
  oracle of (d)), on a NaN, a wrong shape, a cycling step and a failing step; the textbook program through the
  interpreter and `futhark c` alike; the eight cases' shapes and scales; `reference64.fut` against the Rust f64 step
  bit for bit on every case. A7 is NOT included: its f32 criterion is still the bit-for-bit scheme, open.
- (c) DONE (9cbeb292, master version 5): the triple state (`S:tuples3`, frames `L3`/`F3`: UltraDim's (sum,
  compensation, count) in ONE loop, as the shader has it; the coordinator's ruling of 2026-11-03: triple rows, not a
  nested pair, since a pair loop destructures into scalar variables) and the kernel boundary (`let@C`, the frame `B`
  binding a class at depth 0 without a trip; the specification's `lets` line, `sniff.py --lets`). The e-graph's rules
  for the triple are its three projections; the Kahan form survives saturation in the triple (tested). Test views `cell`
  (`update_centroids` for one cell, 1,041 types) and `kernels` (a `let` of an `[n]i32`, 1,892 types) equal `futhark c`
  on 30 runs; 50 random genes a view pass `futhark check`, 8 compile and equal the interpreter.
- (d) DONE on the CPU: `docs/kmeans_seed.md` (the three shaders and the seed side by side, every line to its row; the
  points that are not one-to-one and why; the two views measured), `futhark_sig::kmeans_seed` in the problem's slot
  (`Seed::Translated`), `pool/seed.fut` and `seed.tsv` the rendered program and the gene, the textbook step moved to
  `pool/textbook.*`. Proven (`tests_kmeans_seed`): on 8 of 8 stored cases the assignments equal the Rust port of the
  three shaders (and the f64 reference's) as integers, the centres are bit-identical to the f32 port through the host
  interpreter and `futhark c` (plain and quality program), and no farther from the f64 reference than the port; the
  planted slowdown (the assignments twice under `map2 f32.min`) the same bits. The runner's two controls pass on the CPU
  (`--backend c --eval single`). THE DECISION FOR ANDREW: the plan's view (i32 assignments beside f32) is 53,773 types
  and 62.7 million words a gene at N = 96 with the pairs alone (the layout 488 s), which the device's 128 MiB binding
  cannot hold, and at master 5 131,825 types without the kernel boundary and 237,325 with it, over the 65,536 type codes
  (the kingdom does not load); the problem's view holds the assignments and the count as integers in f32 (exact below 2^24), 14,359 types,
  27.6 million words at N = 192 (the seed is 131 functions). Not run: the GPU (the WGSL oracle, item 4, and the run,
  item 5).
- (g) DONE on the CPU (2026-11-03, after the advisor's review of the k-means plan,
  `docs/reviews/2026-11-03_kmeans_plan_advisor_round1.md`, every finding fixed; `docs/KMEANS_FITNESS.md` revised):
  the scheme is now `futhark-kmeans-quality-first` (the retired `futhark-kmeans-converged` registered, NOT TO BE RUN
  AS IS): per case a quality group (cosine inertia ratio; the ratio of the seed's mean log10 angular percentile to
  the candidate's over the points off their centroid, UltraDim's number over every point reported beside it) and a
  speed group (the program's OWN time to the stopping point, the shim's GPU-timed passes or `futhark c`'s `-t`, the
  harness excluded exactly; the iterations), every input total on [0, 1], against the seed's run of the same
  interleaving; the two-level HFF gated by quality, so the identity step, a worsening step, a NaN step and a step
  rounded to 1e-3 rank below the seed and the run goes on (measured through the scheme on 500 dbpedia rows,
  `kmeans_converged::tests`). The cap is UltraDim's `converged: false`, a slow finished run. The work guard is 10 x
  the seed's work a case. **The data** is the dbpedia sample cut into six disjoint cases (train 1,000/16, 1,000/50,
  500/8; held 1,000/32, 500/16; test 1,000/32) plus the small `train_n200_k4` for the interpreter's tests, UltraDim's
  k-means++ starts stored per case (the same crates at the same versions), the timing input the whole sample with k
  256 (`make_cases.sh`, `examples/kmeans_cases.rs`); the uniform magnitude cases of item 3 are gone (the breaking
  case's lesson, Kahan against the plain sum, is in `docs/kmeans_seed.md`). The reimplemented measures equal
  UltraDim's vendored functions to the bit (`ln B` from statrs as UltraDim has it). The loop is Euclidean Lloyd scored
  on UltraDim's spherical measures, stated in the specification with what normalisation does. Not run: the GPU
  (the oracle, item 4; the in-run A/A and the run, item 5; the script is `/private/tmp/claude-501/fbench/kmeans_gpu_controls.sh`).
- (h) DONE on the CPU (2026-11-03, after the advisor's round-2 review and Andrew's ruling of the day: the HFF angle
  is the final fitness, nothing manipulates it; quality weighs more by repetition in one flat vector): the scheme is
  `futhark-kmeans-qqs` (`fitness::qqs`: every case's quality inputs m times beside its speed inputs once, TrueNorth
  over the whole; `--quality-weight`, default 2, in the run card and `run_start`), and every gated scheme is retired
  with the offset named as its fault, the defaults of the i32 problems, the staged runners and the shapes kingdom
  moving to `futhark-slope4-qqs` (m = 8, measured on histogram and pathfinder), `staged-qqs` and `shapes-hff-qqs`.
  The ordering table at m = 2 and 3 (the identity, a worsening, a NaN, two rounded steps, the normalised step, two
  same-partition reorderings of the seed's sums, a faster equal step, an early stop) is in `docs/KMEANS_FITNESS.md`
  and asserted by the kingdom's test. The round-2 faults: the test case is run every generation the champion changes
  and at the end (`OUT/test_case.tsv`, the report); the in-run A/A runs on both backends and its speed inputs' spread
  is the noise floor a speed claim must clear; the default pairs is 2; the P start's cost measured (149 s the
  e-graph, 193 s the fold, 19.5 GB); the index regenerated. Throughput: the c backend's program is a Futhark server
  (19.3 s to 13.2 s an evaluation of the seed); the GPU's **resident-sample path** (`--eval rust`: the sample on
  the device once a session, a generation's programs one module, each candidate's cases interleaved with the seed's
  entry, the runtime's `k` command) is built and tested on the c path, not run on the GPU
  (`/private/tmp/claude-501/fbench/kmeans_gpu_controls.sh`, then `kmeans_run_s1.sh`).

- (f): after the GPU runs.

**Status of (b), 2026-11-03: done** (`kingdoms/futhark/README.md`, "The class", "Two elements in one view", "Pairs").
The class is an AST with an injective text (`futhark_view::Class`: `a.f32.n` in a view of several elements, `a.n` in a
view of one, `(f32,i32)` a pair); a view lists its elements and has every element's rows, a rank-1 class of each
element per dimension, the two conversions (`f32.i32` total, `i32.f32` guarded); master version 4 adds pairs of scalars
as a class with `pair`, `fst`/`snd`, the zero pair, `psel` and the pair loops, which destructure the pair into two
bound variables (the frames `L2`/`F2`, versioned), so a pair is never a bound class and the context count stays where
it was. Test views `sig/kahan` (xs f32, ms i32 to f32) and `sig/argmin` (ds f32 to i32), 328 types each: UltraDim's
Kahan step and argmin as genes compute the yardstick's bits through `futhark c` on 30 runs, 50 random genes a view
pass `futhark check`, 8 compile and equal the interpreter; the five committed views' fingerprints are unchanged.
What (d) will want beyond pairs: UltraDim's `update_centroids` carries a TRIPLE (sum, compensation, count); with
pairs the count is a second loop over the points (the same values, twice the passes). A view of f32 and i32 at
k-means's depth 3 is the next measurement: the second element doubles the scalar and rank-1 classes and the contexts
grow with them.

**The paper's protocol (Andrew, 2026-11-03), which this step serves.** The point of the paper (EvoStar): a type system
for GEP, then a facility for shape scopes, carefully constructed; a gene of about 3 million words that is rich enough
to evolve efficient ML algorithms, k-means being the example. The protocol: because the search space is vast, LLMs
hallucinate 25 initial implementations at high perplexity; each is converted to a gene; population, generations, the
fitness functions and the mutation functions are set; fuller simplifies, snaps, folds and shares equivalent parts; then
evolution runs to see whether it improves on Fable's implementation. So item (d) also produces the k-means pool: 25
LLM implementations in Futhark (`futhark_pool`'s prompts extended to f32 and to UltraDim's spherical k-means), each
read into the gene by the same translation path as the prototype, Fable's among them as the one to beat; the
prototype's own translation (UltraDim's kernels, Kahan included) is in the pool as well. Histogram and pathfinder are
not ML: they stay the exact benchmarks on which the method is shown to learn.

**Order of work (each a commit with tests; no GPU step until the translation is reviewed).**
(a) revert 06126f86 and remove the textbook seed from the reference slot; (b) tuples as loop state and a second element
in a view (master rows and view code; the i32 views' fingerprints unchanged); (c) the Kahan rows and the saturation
test; (d) the translation document and the seed program, exact on the host interpreter against the f64 reference,
and the WGSL oracle run; (e) the f32 correctness scheme and the magnitude cases, A7 included; (f) the GPU check, then
the step 6 run. Vocabulary: the kingdom documents say "test", not "witness", and "type", not "dual", from here on.

## Steps 5 to 7 (unchanged)

5. Prove the Futhark kingdom works: the improvement run from Fable's histogram program on the fixed build, with speed reported every generation; then timing screening (a few pairs first, the full 11 for contenders).
6. Prove the k-means kingdom works: a full run with all operators and working model edits, 3 seeds, and a random-search control.
7. Done: the scratch clean-up.

## Step 5a: phylu runs the Futhark problems (plan, 2026-11-03, revised after the advisor's review; no code yet)

### Context

**Objective.** Every Futhark problem (`futhark_sig::PROBLEM_NAMES`: reduce, scan, histogram, pathfinder, kmeans, a7)
is evolved by phylu's engine (`phylu/src/evolve/engine.rs`: the islands, the pump, virtual ALPS, the generation loop),
and fuller is the library that scores a generation. Andrew's ruling of 2026-11-03, at the top of `CLAUDE.md`: fuller
does not evolve code.

**Approach.** fuller exposes ONE seam, `FutharkEvaluator`: it scores the rows the engine hands it in one compiled
module a call and returns each individual's HFF input vector and the angle the run's fitness scheme makes of it.
phylu's engine takes the Futhark kingdom as a typed kingdom with an external fitness, as it already takes the regex
and TSR kingdoms on the typed core, and does selection, ALPS, the pump and the islands. Everything
`examples/futhark_evolve.rs` does that is not the generation loop moves into fuller library modules; the loop's work
moves into the engine's own mechanisms, never into a second loop beside them.

**What is new.** The pump's refill (13 of an intake island's 16 rows at every pump) receives the e-graph's variants of
the best individual, the saturated fold's forms and, in regime P, mutants of the champion island's best, with a fresh
cohort label: Andrew's design in `docs/SPLIT_fuller_phylu.md` §3a ("throw it in to the empty slots that we fill when
we do the promotions"), which the SR engine never wired. The deprecated runner had no islands, no cohorts and no
refill: its "pump" put at most 4 variants into the next generation's children.

**Status.** Planned 2026-11-03 (`decdca57`); revised the same day after the advisor's review
(`docs/reviews/2026-11-03_step5a_phylu_futhark_advisor.md`, `12c9f611`): every finding is applied (5a.7 lists where)
and the four open questions are decided as the review recommends (5a.6). Read for this plan, in full:
`examples/futhark_evolve.rs` (3,448 lines), `src/futhark_search.rs`, `src/futhark_batch.rs`,
`src/futhark_sig/kmeans_converged.rs`, `src/fitness.rs`, `src/runner.rs`'s public items, `src/futhark_operators.rs`'s;
phylu `src/evolve/shape_typed.rs`, `examples/shape_fit.rs`, `engine.rs`'s `External`, `evaluate_external`, `fresh`,
`fresh_plan`, `pump`, `remember` and `Engine::new_with_table`, `checkpoint.rs`, `typed_device.rs`'s paging,
`marriage.rs`'s header, `true_wgsl_fit.rs`'s `expressed_hits`.

**Why now.** Two loops evolve Futhark programs today: `futhark_evolve` (one population of 24, a tournament of 3, two
elites) and phylu's `shape_typed` (one island, A7 only, scored by the deprecated loop's interpreter). Neither has the
engine's islands, cohorts or pump, and every improvement to the engine misses them.

**The critical path is phylu's external path, not fuller's move.** The engine's `External` hook has never run:
nothing in phylu calls `set_external` (`grep -rn set_external phylu/src phylu/examples` finds only its definition,
engine.rs 4698; its caller `wgsl_fit --model-rewrite` is gone). P2a's mock test is the first time the path runs at
all, and P1 is the first time it runs on a typed layout. Five things the plan needs are missing from it (5a.1), which
is why P2 is split in two and sized in days.

### 5a.1 What the engine has, and what it lacks for the Futhark problems

| capability | the engine today (phylu `d2264705`) | what the Futhark problems need | task |
|---|---|---|---|
| a typed population on the device | the typed core for regex and TSR (`Engine::new_with_table`: `regex_typed`, `tsr_typed`, `TypedDevice::with_parts`); `shape_typed` builds a Futhark device (the walk-only stage, the paged genome, `check_layout`) but in a one-island loop of its own | the six problems' layouts (gene words at the default program size N = 32: reduce 10,025, scan 10,021, histogram 47,999, pathfinder 57,487; kmeans 27.6 million at N = 192) on the engine's device, paged at the walk stage's `binding_limit()` (`shape_typed.rs` 316, `with_parts_paged`); the k-means layout at 14,359 types takes minutes on the host | P1, G1 |
| an external fitness, by kind | `External`, never run; keyed on a NAME: `EXTERNAL_KINGDOM = "safeWGSL"`, and `combinations_for` tests `config.kingdom == EXTERNAL_KINGDOM` (engine.rs 3861-3870) | the hook on a typed layout, keyed on "the config has an external fitness" | P1 |
| the refill's genomes, every pair | `Engine::fresh` writes the hook's k-th genome at row `pop_intake - 1 - k`, the FIRST pair's intake only (engine.rs 5359-5374); `fresh_plan` builds one `Population` for every pair's refill (5329-5338): with two pairs, pair 1 refills with random genes | the hook's individuals in EVERY pair's intake, each pair's own request | P2a |
| the refill's individuals, with their history | `ExternalFresh = Box<dyn FnMut(u32, usize) -> Vec<Vec<u32>>>` (3876): genomes only, no origin, no parents, no error | `FreshIndividual` (genome, origin, parent rows, mutation event) and a `Result`: the type is defined in fuller's `src/futhark_evaluator.rs` (F2) and phylu's widened hook imports it | F2, P2a |
| the budget | `evaluate_external` charges `eval_counts.add(source, 1)` for every row the hook scored, cache hit or not (4798-4801); `max_evaluations` reads `eval_counts.total()` (8896) | the Futhark budget: programs not scored before, by text (`Judge::evaluate`, futhark_evolve.rs 1218-1227); an elite, a repeat or a pump child equal to a scored program is free: the hook returns what it charged and the engine adds that | P2a |
| scoring on the engine's beats | the engine calls the fitness after variation (`evaluate_fresh`, 8957) and AGAIN after every beat that lands rows: the beam (9000), the snap (9151), the fold (9169), the pump (9266); a fresh row must leave the call scored (4795: NaN is an error) | ONE module an `evaluate_generation` call: two on a pump or fold generation (the children's, then the refill's or the forms'), and for k-means the champion's test case a third when its text changed (`converged_on_test` builds its own module, futhark_evolve.rs 1073-1085). The engine's order stays; the refill is never held unscored to the next generation. `GenerationCost` carries the compile seconds and the module count of the call, and G2 reports modules a generation | F3a, F3b, G2 |
| two fitness stages (`futhark-slope4-qqs`) | the re-score needs no engine change: the hook receives `&mut Generation` and the engine reads EVERY row's fitness back afterwards, elites included (4793-4804). The hall of fame does: `remember` keeps the best of the fit by angle (8968), and stage-1 angles are not comparable with stage-2 ones | quality alone until the best is exact on every case, then the flat vector with the timing slots; at the change the evaluator re-scores every row in the same call (F3a); the engine restarts the hall of fame and writes a `stage_change` event | F3a, P2b |
| the k-means fitness (`futhark-kmeans-better`) | none, and none needed: the engine sees one vector a row | a whole run per candidate to UltraDim's stopping rule, interleaved with the reference program `--pairs` times, the further iterations from the final centroids; all inside the evaluator (`futhark_batch`'s k-means command on the resident sample) | F3b |
| data resident on the GPU across generations | the engine's own data buffer (SR) | the dbpedia sample (and the timing input) held by the `futhark-rust-rt` session inside the batch host for the whole run: the evaluator lives as long as the run. The engine's wgpu device and the runtime's (loaded by `dlopen`, its own wgpu) are then two devices in one process: not yet measured | F1, G2 |
| the read-back | `evaluate_external` reads the whole population back on every call (`read_generation`, 4785): 2.65 GB for 24 k-means rows, twice on a pump generation | acceptable beside a k-means evaluation, which is minutes; G1 times it. The evaluator expresses each row by walking from the head cells it needs, never copying a row's text eagerly | G1, F3a |
| the run file | the engine checks `killme.trigger` once a generation (8880-8885); its comment promises an external fitness checks it "before every individual", which no code does (the deprecated runner checks once a generation, 2770) | a k-means generation is minutes: `evaluate_generation` checks the run file between batches and returns the rows scored so far with `stopped = true`; the engine ends the run as `killed` before its NaN check | F3a, P2a |
| the batch compile's time limit and split | none needed | already in `futhark_batch::Host` (`COMPILE_LIMIT_FLOOR` 30 s, `COMPILE_LIMIT_FACTOR` 30 x the median, the halving, the source under `bugs/`, the fusion bound, the compiler at `-N1`): the evaluator inherits it unchanged | none |
| the measures of a generation | the snapshot's `extras` are empty for an external kingdom; the snapshot is written on `progress_every` (the Futhark runner wrote one a generation) | the best's HFF as an angle AND as the percentile F(θ) with its d (today only `futhark-kmeans-better` reports the percentile); its quality and speed inputs against the reference program, per case; its exact counts; the test set's measures, reported, never a fitness input. `futhark_fit` sets `progress_every = 1` and its card says so | F4, P2b, P4 |
| checkpoints and the hall-of-fame file | `Checkpoint` (checkpoint.rs 60-116, `FORMAT_VERSION` 3) has no slot for a hook's state; the hall-of-fame file's columns are SR's (`log10_p mse_train r2_train`, engine.rs 8871-8875) | the evaluator's state (the quality cache by text, the timing slots, the stage, the counts, the first exact program) in a `serde_json::Value` slot of the checkpoint, format version 4; the hall of fame's columns from the evaluator (`input_names`, the program text) | F7, P2b |
| variation | on the device: point mutation by a per-cell rate, inversion, IS, RIS, the Dc operators, one- and two-point crossover per type; `External.rates` per island | a mutation rate in EXPRESSED cells, per regime as the deprecated runner mutated: 1-2 expressed cells in P, 1-4 in R (`m_of`, futhark_evolve.rs 3156); histogram's gene is 47,999 words and its reference program 7 coding cells, so the device's inversion and crossover over the whole row are nearly always neutral, and marriage (the typed subtree crossover, phylu `marriage.rs`) is the one structural operator, which is why P3 comes before R1; the row operators of 3f and the ring walk (fuller host operators) after R1 | P3, P3b |
| the start | random rows, or `External.seed`, which fills rows 0.. of the WHOLE population (8661-8668) | regime P (the reference program, its e-graph variants, the saturated fold's forms), the ring canvas, the task alternatives, the mutants of one program. In regime P the start population is the reference program, its e-graph variants and the fold's forms (pathfinder: 24), filled with mutants of the reference program by P's `m_of` when fewer (histogram: the reference program and 23 mutants): a departure from the deprecated runner. With intake 16 and champion 8 the first 16 of the start population land in the intake and the next 8 in the champion island at generation 0: new behaviour (the deprecated runner had no islands), stated in `futhark_fit`'s card and in the regression table (the champion island is not empty at the start) | F5, P4 |
| the pump's refill | `pump_keepers_at_most(16)` = 3 keepers, 13 refill rows a pump (3829-3831); the hook "caps what it returns and the rest stay the random draw" (5362-5366) | at 2,000 evaluations (about 80 generations of 24) and the pump every 5 generations, about 16 x 13 = 208 refill rows, a tenth of the budget; a random 47,999-word histogram gene in regime P is a wasted evaluation (section H: random programs crowd at the all-zero output), so F5 fills the refill (5a.2, `fresh`) | F5, P2a |
| the runner standard | `standard_outputs` (stream, `killme.trigger`, checkpoints), the genealogy | the ledger with the evaluator's columns, the hall of fame with program text, `per_generation.tsv`, `series.tsv`, `report.tsv`, the chart, `aa_gate.tsv` | P4 |

### 5a.2 The interface fuller exposes

One owned object, in a new module `src/futhark_evaluator.rs`. The engine keeps it inside a `Box<dyn FnMut>` for the
whole run, which is `'static`, so every borrow goes: the batch host (`Host<'a>`), the judge's interpreter
(`Interp<'a>`, futhark_evolve.rs 462) and the genetics (`Genetics<'a>`, 1283-1287) borrow the kingdom, the signature,
the contract, the layout and the constants' lengths today; they hold them by `Arc` (F1), never by `Box::leak`.

```rust
pub struct EvaluatorConfig {
    pub problem: String,               // futhark_sig::PROBLEM_NAMES
    pub fitness: Option<String>,       // a fuller::fitness scheme; None: the problem's own default
    pub quality_weight: Option<usize>, // the scheme's default when None
    pub quality_cliff: Option<f64>,    // futhark-kmeans-better only
    pub backend: Backend,              // Gpu | C
    pub eval: EvalMode,                // Rust (the default) | Batch | Single
    pub pairs: usize,
    pub runs: usize,
    pub program_size: Option<usize>,   // the problem's own default when None (k-means 192)
    pub out: PathBuf,                  // bugs.log, bugs/, the build directories; the run file is OUT/killme.trigger
}

impl FutharkEvaluator {
    pub fn new(cfg: &EvaluatorConfig) -> Result<FutharkEvaluator, String>;
    /// The controls at the start: the reference program exact (interpreted and compiled), its output on the timing
    /// input, the A/A noise floor and the in-run A/A; for k-means its run converges on every case, the work caps.
    pub fn controls(&mut self, regime: Regime) -> Result<Controls, String>;
    pub fn contract(&self) -> &KingdomContract;
    pub fn layout(&self) -> &GeneLayout;
    pub fn const_len(&self) -> &BTreeMap<Ty, u32>;
    /// The names of the HFF input vector's columns, in order (the stream's `run_start`, the hall of fame's columns).
    pub fn input_names(&self) -> Vec<String>;
    /// THE SEAM: every row the engine hands over, in row order, scored in ONE compiled module (the engine calls it
    /// once after variation and again after a pump or fold beat). A program scored before is a cache hit and costs
    /// nothing. The run file is checked between batches.
    pub fn evaluate_generation(&mut self, genes: &[&[u32]], generation: u32) -> Result<GenerationScores, String>;
    /// The measures of the generation's best (and the k-means test case when the champion's text changed: its own
    /// module, counted in the call's cost).
    pub fn report(&mut self, generation: u32, best: usize, scores: &GenerationScores) -> Result<GenerationReport, String>;
    /// New individuals for the engine to place: the start population, one pair's pump refill, the fold beat's forms.
    pub fn fresh(&mut self, request: FreshRequest) -> Result<Vec<FreshIndividual>, String>;
    pub fn state(&self) -> serde_json::Value;
    pub fn restore(&mut self, state: &serde_json::Value) -> Result<(), String>;
    /// The end: the champion's 11-pair re-time, the A/A medians, the speed verdict, the report's lines.
    pub fn final_report(&mut self, champion: &[u32]) -> Result<FinalReport, String>;
}

pub struct RowScore {
    pub expressed: bool,
    /// The program text, rendered once a distinct gene (from the head cells the walk reaches).
    pub text: String,
    /// THE HFF INPUT VECTOR as the scheme builds it: in stage 1 the quality group; in stage 2, and for k-means, the
    /// quality group `quality_weight` times and the speed group once (`fitness::flat_vector`).
    pub inputs: Vec<f64>,
    /// The scheme's angle of `inputs` (TrueNorth, `fuller::fitness`): THE number the engine's selection reads. The
    /// device reads it as `f32`; at these angles (0.72 rad, d = 66) that loses nothing selection can see.
    pub fitness: f64,
    pub theta_quality: f64,
    pub theta_speed: f64,
    pub percentile: f64, // F(theta), fitness::hff_percentile
    pub d: usize,
    pub quality: Vec<f64>, // per case, for the report
    pub speed: Vec<f64>,
    pub exact: [usize; 3], // train, held, test
    pub behaviour: String,
    pub new_evaluation: bool,
    pub ledger: Vec<String>, // the deprecated runner's LEDGER_STAGE_COLUMNS
}

pub struct GenerationScores {
    /// One a row handed over; when `stopped`, only the rows scored before the run file went.
    pub rows: Vec<RowScore>,
    pub stage: FitnessStage,
    /// The best became exact on every case: every row was scored again in stage 2 within this call.
    pub stage_changed: bool,
    /// The run file was removed between batches: the engine ends the run as `killed`.
    pub stopped: bool,
    /// This call's new evaluations (what the engine charges to the budget) and the run's total after it, so the
    /// engine does no subtraction.
    pub evaluations_new: u64,
    pub evaluations_total: u64,
    /// Compile, upload, run and timing seconds; modules compiled; halvings; unfinished compiles; resident hits.
    pub cost: GenerationCost,
}

pub struct FreshRequest {
    pub purpose: FreshPurpose, // Start | PumpRefill | FoldBeat
    pub pair: usize,           // the pair whose intake is refilled
    pub rows: usize,           // the rows the engine will place (13 a pump at intake 16)
    pub best: Option<Vec<u32>>, // the champion island's best of that pair
    pub generation: u32,
}

/// Defined here (F2); phylu's widened `ExternalFresh` imports it (P2a).
pub struct FreshIndividual {
    pub genome: Vec<u32>,
    pub origin: String,        // "reference", "egraph_variant", "saturated_fold", "pump_mutant", "random"
    pub parents: Vec<usize>,   // rows of the population the request saw
    pub event: Mutation,       // the stream's mutation event (`pump`, `saturated_fold`, `egraph_exact`, ...)
    pub new_text: bool,        // a text not scored before: a budgeted evaluation when it is scored
}
```

**What `fresh` returns for a pump refill (F5).** In order, never two with one text or one tree, never a text scored
before (a repeat would take a row for nothing):
1. the e-graph's variants of the pair's champion-island best, at most 4 (`variants(.., 2, 16, 4)`), and only when the
   best changed since that pair's last pump (futhark_evolve.rs 3055-3081);
2. on a fold beat, the saturated fold's forms of the best;
3. regime P: mutants of the champion island's best by the run's mutation operator (the expressed-cell point mutation,
   or the ring walk when the run has a ring canvas), built by `futhark_rings::mutants` as `fable_mutants_population`
   builds them (futhark_evolve.rs 1588-1631), up to the request's rows; regime R: the remainder stays the device's
   random draw.

Every refill row that is a new text is a budgeted evaluation. **The refill's share of the budget:** at most 13 a pump
(the refill's rows), about 16 pumps in a 2,000-evaluation run, so at most about 208 evaluations, a tenth of the
budget. The card prints the cap a pump and the report the refill's evaluations spent.

The angle is computed in fuller, by the scheme, so one implementation defines every fitness (`docs/FITNESS_SCHEMES.md`:
a new fitness is a new scheme); phylu writes it as the `f32` its tournament reads and never recomputes it. A gene that
does not express is a bug of our software (the layout defines what a cell may hold) and stops the run, as in
`futhark_evolve`.

**What moves where.**

| from `examples/futhark_evolve.rs` | to | owner |
|---|---|---|
| `Judge` and its types (futhark_evolve.rs 457-1280, about 820 lines with the k-means paths: `Quality`, `Eval`, `Stages`, `TimingInput`, `AaSets`, `ConvergedJudge` with its program runner, `TestCaseRun`, `pair_steps`, `pair_ratios`, `LEDGER_STAGE_COLUMNS`, `hff_dimensions`), `Genetics` | `src/futhark_evaluator.rs` | fuller |
| the per-generation prefetch, scoring and stage logic (2781-2857) | `FutharkEvaluator::evaluate_generation` | fuller |
| the controls at the start (reference exact interpreted and compiled, the timing input's output on the GPU, the A/A sets, the k-means reference program's runs, work caps and row ranges) | `FutharkEvaluator::controls` | fuller |
| the per-generation measures (`per_generation.tsv` columns, the k-means lines, `test_case.tsv`) | `FutharkEvaluator::report` | fuller |
| the P start, `ring_canvas`, `alternatives_population`, `fable_mutants_population`, the e-graph variants of the best, the saturated fold's forms, the pump refill's mutants | `src/futhark_start.rs`, behind `FutharkEvaluator::fresh` | fuller |
| the champion's re-time, `aa_medians`, `speed_verdict`, `report.tsv`, the chart | `FutharkEvaluator::final_report` | fuller |
| the measurements that are not evolution: the slope check (`--check`, `--check-replay`), `--measure`, `--rescore`, `--aa-only` (the A/A sets) | `examples/futhark_measure.rs`, over the library | fuller |
| regime C (random search: a budgeted loop over generations of fresh individuals with a ledger, a hall of fame and a stream, futhark_evolve.rs 3029-3039, 3411-3416: a runner, which fuller keeps none of) | `futhark_fit --regime C`, a control arm over the same `evaluate_generation`, no engine | phylu |
| the generation loop: elites, tournaments, marriage pairing, the pump and fold beats, the stage change, the stop rules, the population's checkpoint, the stream's snapshots, the ledger and the hall of fame | the engine, `src/evolve/futhark_typed.rs`, `examples/futhark_fit.rs` | phylu |
| `futhark_operators`, `futhark_rings`, `saturated_fold`, `futhark_batch`, `kmeans_converged`, `fitness` | unchanged | fuller |

### 5a.3 The tasks

Every brief given to an agent starts "Read docs/Rules.md, then:", then this section's context. Every task is a commit
(or a few) with its tests, zero warnings, clippy clean (fuller default and `gpu,wgsl`; phylu default and `gpu`), on
fuller's `feat/futhark-kingdom` and a phylu branch. **Agents never use the GPU**: a task's GPU tests and runs are
written as a script under `/private/tmp/claude-501/fbench/plan5a/`, `nohup` with a kill guard and a log, and the lead
runs them, one GPU process at a time, announced. A test that needs a compiler but no GPU uses the C backend. Sizes are
working days of one agent, the tests included; the GPU scripts are the lead's hours.

```
F1 ─> F2 ─┬─> F3a ─> F3b ─┬─> F4 ─┐
          │               ├─> F6 ─┼─> F8 ──────────────────────────────────────────────────┐
          │               └─> F7 ─┤                                                        │
          ├─> F5 ─────────────────┤                                                        │
          └─> P1 ─┬─> P2a ─> P2b ─┼─> P4 ─> G2 ─> G3 ─> R0 ─> R1 ─> R2 ─> R2b ─> R3 ─> R4 ─> RC ─> X1
                  ├─> G1 ─────────┤        └─> P5 (built)          └─> AB2 (two pairs), P3b (A/B)
                  └─> P3 ─────────┘
G1 runs before P2a's k-means GPU test (the paged store's first gigabytes).
```

**The lead's order (the critical path):** F1, F2, F3a, P1, P2a, G1, P2b, F3b, P3, F4 to F7, P4, G2, G3, R0, R1. F5 and
F8, P5 fill the gaps beside it.

**Status (2026-11-04).** F1 DONE (`82f6dd3e`): `futhark_sig::load` and `signature` return `Arc` handles; `Interp`,
`futhark_batch::Host` and the runner's `Judge`, `Genetics` and `Run` hold `Arc`s, no lifetime, no `Box::leak`. New
test `a_host_and_an_interpreter_built_from_dropped_handles_run_a_c_batch`. Test edits are type annotations only.
`futhark_evolve --problem histogram --regime R --gate DIR --backend c --eval single --evaluations 30 --seed 1`
before and after: `ledger.tsv` identical on 33 rows and 12 columns; the 11 `*_ms` columns (wall-clock stage times)
are excluded. Regime R needs the gate, so the histogram slope check ran first on the C path (pass, 1,254 s). Regime P
cannot run on the C path, because its stage 2 times on the GPU. fuller: 778 tests pass, clippy clean with default
features and with `gpu,wgsl`. phylu (`d2264705`, no change): release build, 250 tests pass, clippy clean with default
features and with `gpu`.

F2 DONE (`1e5a0f4c`, `e32b39f1`): `src/futhark_evaluator.rs` (feature `wgsl`, for `wgsl::work_guard`'s per-submit
cap) holds the `Judge` with its types and k-means paths, `Genetics`, `ring_canvas`, `gate_passed` and
`hff_dimensions`, moved unchanged except `Genetics::marry` (takes (tree, gene) pairs) and `walk_event` (the child's id
and tree): the library holds no population type. `FutharkEvaluator::new(&EvaluatorConfig)` loads and checks
everything (the runner's flag checks moved into it); `controls(&ControlRequest) -> Controls` runs the controls in the
runner's order; `input_names()`; accessors for the kingdom, signature, contract, layout, `const_len`, reference
program, alternatives; `judge` and `gen` are public for the deprecated runner only. `FreshRequest`, `FreshPurpose`,
`FreshIndividual` defined (`fresh` is F5). `futhark_evolve` runs through it. Before/after (946af1b1's binary against
this one, `--backend c --eval single`, random seed 1): histogram R with `--gate`, 30 evaluations: `ledger.tsv`
identical on 33 rows and 12 columns (the 11 `*_ms` excluded), the stream's objectives and card identical; k-means
`--regime P --evaluations 3 --pop 4`: the same 3 programs, quality inputs and behaviours bit for bit; the seconds
speed inputs are wall-clock ratios and differ between any two runs, the before binary against itself included
(fitness 0.722756 and 0.722735), so the F2 row's "speed inputs bit for bit" cannot hold for k-means. fuller: 778
tests pass (default), the 3 new evaluator tests pass with `wgsl`, clippy clean with default, `wgsl` and `gpu,wgsl`.
phylu (`d2264705`, no change): 250 tests pass, clippy clean with default features and with `gpu`.

P1 DONE (phylu `c3862062`): `evolve::futhark_typed` (shape, sampling, `check_layout`, carrier layout; the walk-only
paged device parts behind `gpu`), moved out of `shape_typed`; `Engine::new_with_table` takes `kingdom = "futhark"`
with `Given::Futhark` and `Data::placeholder`; `EXTERNAL_KINGDOM` is gone, `combinations_for` keys on `KingdomKind`.
Host tests: 24 rows drawn by `init_row` express, walk to the expression's node count and render, for histogram
(mean 2,421 nodes; every program passes `futhark check`) and k-means (evaluator 83 s, parts 123 s, rows 232 s; mean
1.5 million nodes, programs up to 65 MB, so no `futhark check`: 101 s for a 5.8 MB one). phylu: 252 tests pass,
clippy clean default and `gpu`; `the_engine_builds_on_the_futhark_kingdom` (GPU) is built, not run: for G1.

The F2 code review's corrections (`7db07042`): `4e730a21` the per-submit work cap moves to `futhark_sig` (re-exported
by `wgsl::work_guard`) and `futhark_evaluator` needs no feature; `2948db62` the library prints nothing, its messages go
to the caller's sink or are held for it (`Messages`, `set_messages`, `take_messages`), `futhark_evolve` prints them as
before (histogram R `--gate`, C, single, 30 evaluations: `ledger.tsv` identical to F2's on 33 x 12, the console
identical but for wall-clock times); `c4d4d90d` `controls` runs once a run, a second call is an error.

F3a DONE: `FutharkEvaluator::evaluate_generation(&mut self, genes: &[&[u32]], generation: u32) ->
Result<GenerationScores, String>` (and `evaluate_generation_with`, a callback after every batch). Each distinct gene
expressed and rendered once; one compiled module a call on batch and rust; the run file checked before every batch
(`stopped`); the stage change inside the call, every row scored again in stage 2; per row `RowScore` (the HFF input
vector, the scheme's angle, its percentile and d, the ledger's columns); `evaluations_new`, `evaluations_total`,
`GenerationCost`. Departures from 5a.2: no `expressed` field (a gene that does not express is an error of the call);
`percentile` is an `Option` (None for a retired two-level scheme, whose angle is not over one vector); `source` beside
`text`; regime P's stage 2 from the start is set by `controls`. k-means returns an error naming F3b. Tests on the C
backend against F2's runner ledger (histogram R, random seed 1, committed as
`src/futhark_evaluator/histogram_r_seed1_ledger.tsv`): the runner's generation 0 (the same 24 genes, from
`GeneRng::new(1)`) and its generation 1 (8 programs) give its angles, quality inputs, behaviours and ledger columns
bit for bit, and its counts (24, then 30); repeats free; batch: one module for 24 programs, a repeated call 0 modules
and 0 evaluations; the stage change (reference plus 5 random genes, the timing slots filled before the call since the
c backend has no GPU timing) scores every row once in each stage, 6 programs charged; the run file removed after the
first batch returns that batch's rows and `stopped`. `Judge::evaluate` now refuses the c backend in stage 2 only
when a timing slot is empty. The GPU script `f3a_rust_generation.sh` is not written (it needs a driver over the seam:
phylu's P4 or `futhark_measure`). fuller: 783 tests pass (default, the 8 evaluator tests included) and 1,243 with
`wgsl`; clippy clean with default, `wgsl` and `gpu,wgsl`. Commits `2948db62` and `c4d4d90d` carried a clippy
`type_complexity` warning (the sink's type), fixed in this commit by the `Sink` alias. F3a's code is `1a7ba021`.

P2a DONE (phylu `db3fc88d`, `54e7d4f9`, `1ce1b8d0`; GPU fixes `84a80116`, `b6045261`; review fixes `8973a6b7`): the
fitness hook (`ExternalFitness`, `ExternalScored`) and the fresh hook (`ExternalFresh`, asked once a pair); the budget
charges `evaluations_new`; the external fold rides the engine's `fold_every` beat and lands through `fold_landing`; a
stop, also in generation 0, ends the run as killed; the hook's individuals enter the genealogy with their parents and
origin. Review: `docs/reviews/2026-11-04_step5a_P2a_code_review.md`.

P2b DONE (phylu `4f648bce`; review fixes `dce9a96b`): the stage change restarts the hall of fame and writes a
`stage_change` event; the `columns`, `extras` and `state` hooks; checkpoint format 4 with the hook's state (format 3 is
a named error); a killed run writes no last checkpoint, so a resume never meets unscored rows (P4 must checkpoint every
generation); every hall-of-fame line has its header's columns. The fold beat asks for the forms of each island's best
`fold_top_k`, one request a winner, with `fold_top_k` bounded to 1..=4 by `futhark_config_errors` (one k-means fold
took 193 s); one request a pair carrying all its winners needs a `FreshRequest` change in fuller (F5). Review:
`docs/reviews/2026-11-04_step5a_P2b_code_review.md`. GPU: `p2a_mock.sh`, 6 of 6 tests pass (2026-11-04 03:18Z), phylu
`dce9a96b`, fuller `d2fafb00`'s tree.

F3a's size guard (the P1 code review, `f18d50c6`): in `evaluate_generation` an expressed tree of more than
`SIZE_BOUND_FACTOR` = 64 times the reference program's node count is scored worst on every input (as a program that
does not run), never rendered or compiled, charged once, and counted in `GenerationScores::oversized`. Basis, measured
on the CPU: the reference programs have 7 (histogram), 24 (pathfinder) and 256 (k-means) nodes; their mutants of 1-4
expressed cells reach at most 307, 141 and 392, fuller's random genes 204, 152 and 125; the largest ratio is 44
(histogram), so 64 keeps them all. It excludes phylu's rows drawn cell by cell (P1: histogram a mean of 2,421 nodes, k-means
1.5 million): **in regime R on phylu most histogram start rows will be scored worst** unless the engine's start draw
is made like `Genetics::random_gene` (a matter for P2a/P4). Test: a cell-by-cell histogram gene over 448 nodes,
twice beside one of the runner's genes: 2 oversized rows at pi/2, one module, 2 evaluations. fuller: 784 tests pass (default), the 9 evaluator tests with `wgsl`; clippy clean default, `wgsl`, `gpu,wgsl`.

F3a's code review corrections (`docs/reviews/2026-11-04_step5a_F3a_code_review.md`, `ac153b30`; findings 2 to 6;
finding 1, phylu's start draw, is phylu's, fixed in phylu `e6a7ad2b`). **The size bound is one generous safety
bound** (finding 2): `SIZE_BOUND_NODES` = 4,096 for every problem, or 4 x the reference program when larger
(`size_bound_for`; every reference program and its mutants are admitted). The real protection is the batch compile's
time limit (30 s, then the batch is split); the huge programs came from phylu's head draw. Beside it, a module's node
budget: `futhark_batch::MODULE_NODE_BUDGET` = 2,048 nodes, `Host::run` cuts a batch into modules of at most that many
expressed nodes before it compiles (a program's quality, timed and k-means entries each count it once; a program over
the budget is a module alone), raised by the evaluator to 32 x the reference program (`module_budget_for`: 8,192 for
k-means, so 24 of its mutants stay one module). Measured on the CPU, the rust path's Futhark stage of one module grows
faster than its nodes (24 histogram rows of 250 nodes: 37 s; 4 a7 rows of 1,000: 40 s; 2,000 nodes: 4 to 8.5 s); the
deprecated runner's generations are far under the budget (24 rows of 278 nodes in all). **The guard counts before it
builds** (finding 3): `typed_gep::walk_expressed` walks the expression without building the tree and stops at the
bound + 1; an oversized row is keyed by a hash of its expressed nodes' words (`Genetics::expressed_key`), and its
tree, s-expression and Futhark text are never built. **The run file inside a halving** (finding 4): `Host::run_until`
checks it before every compile (each chunk, each half); when it is gone no compile starts, the steps not run are
`Outcome::NotRun`, `cost.stopped` is set; `Judge::prefetch_until` caches none of them, and the evaluator returns the
rows before the first one not run, with `stopped` (tests: a mock c compiler that removes the run file and fails, in
`futhark_batch` and through the seam). **The GPU script** (finding 5):
`/private/tmp/claude-501/fbench/plan5a/f3a_rust_generation.sh`, a replay through the seam, not evolution: the runner on
`--backend gpu --eval rust`, histogram R, random seed 1, 72 evaluations, then the `gpu` test
`on_the_gpu_the_rust_path_replays_the_runner_s_generations_through_the_seam`, which scores the runner's first three
generations through `evaluate_generation` and asserts one module a call and the quality inputs, behaviours and angles
bit for bit; written, not run (the lead runs it). **Tests for the stage change and batch mode** (findings 6 and 8): the
run file gone before the stage-2 batch returns the stage-1 rows, `stopped`, the stage still 1; the batch test now runs
the runner's generation 1 too (one module, 6 new, 30 in all, the runner's angles bit for bit). `pub judge`/`pub gen`
stay until X1. Found on the way: Futhark 0.28.1 internal errors (a type error after the pass Fuse SOACs) on random
histogram and pathfinder rows of 225 to 1,100 nodes, 1 to 4 rows in 25 to 28; one kept at
`/private/tmp/claude-501/fbench/plan5a/size_bound/ice/histogram_225_275_k04.fut`.

F3b DONE (`3bd313cf`): `evaluate_generation` scores k-means. On `--eval rust` a call's new programs are one module on the
resident sample (`prefetch_kmeans`, now with the run file, `Host::run_until`), each scored by `evaluate_converged`;
`--eval single` compiles each. A row's vector is the scheme's (`futhark-kmeans-better`: quality 3 times, speed once,
d = 66). The evaluator keeps the champion (least angle, strictly better replaces it); when its text changes,
`champion_test_case` runs `converged_on_test`, its own module, counted, in `GenerationScores::test_case`. The runner
now calls `champion_test_case` too: one implementation. Every k-means compile and resident batch is counted
(`kmeans_batch`: modules, halvings, uploads, resident hits; `GenerationCost::uploads` new). Runner side effect: its
`upload_s` extras now include the controls' upload and the test case's runs (telemetry only). Ledger committed:
`src/futhark_evaluator/kmeans_p_seed1_ledger.tsv` and `..._test_case.tsv` (F2's k-means plumbing run, C backend, at
1192b022). CPU test (C, single, 484 s debug): the runner's 3 programs give its quality inputs, behaviours and count
(3) bit for bit, its champion's test-case inputs too; the test case runs exactly when the champion changes; a cached
call is free. The runner before and after (k-means P, C, single, 3 evaluations): ledger identical except the `*_ms`, fitness and speed columns (wall-clock). Tests: the 13 evaluator tests pass, default and `wgsl`; clippy clean default, `wgsl`, `gpu,wgsl`. GPU script `f3b_kmeans_generation.sh` (runner k-means R on rust, then the replay test
`on_the_gpu_kmeans_on_the_rust_path_replays_the_runner_s_generations_through_the_seam`): written, not run.

### 5b. The Futhark evaluator moves to phylu (Andrew, 2026-11-04; the advisor's review of df346ccc applied)

**The ruling.** fuller holds symbolic programs: the kingdom definition (symbols, arities, the type system, the
kingdom views, the Futhark kingdom among them), the tree, encoding a tree as a Karva gene (each type's head and tail,
the layout), decoding a gene to its typed tree, writing a tree as Futhark source, and symbolic rewriting (the e-graph,
share-equivalence, snapping, folding). phylu holds evolution: the geneframe, the experiment's configuration, the
runners with their stream and checkpoints, the variation and selection operators, virtual ALPS, the population, and
**every fitness evaluation**: the named HFF schemes, the Futhark compile and run, the wgpu kernels, the batch
evaluation on data. phylu sends genes to fuller to be simplified, folded and snapped. Andrew: "we went from GENES to
GPU programs directly, on device"; regression and regex already score on the device in phylu (`score.wgsl`,
`hff.wgsl`, `regex_score.wgsl`). 5a put the Futhark evaluator in fuller; this section moves it to phylu. Nothing
evolved changes: the remaining work is the same, only its home moves.

**What moves to phylu, file by file** (into `phylu/src/futhark_eval/`, each file's tests and committed ledgers with it):

| fuller today | why it is evaluation |
|---|---|
| `src/futhark_evaluator.rs`, `src/futhark_evaluator/` (tests, the three ledgers) | scores a generation |
| `src/futhark_batch.rs`, `src/futhark_batch/` | compiles and runs a generation: the module, halving, compile limit, node budget |
| `src/futhark_rust.rs` (the runtime's crates `futhark-rust-rt/`, `futhark-rust-backend/` and `futhark-wgpu-shim/` stay where they are until the merge; the moved code finds them through one path constant, set to fuller's checkout, in place of `CARGO_MANIFEST_DIR`) | the in-process wgpu runtime and the compiler that feeds it |
| `src/futhark_wgpu_host.rs` | the other GPU host |
| `src/futhark_gpu.rs` | the measured GPU cost of a compiled program |
| `src/futhark_sig/kmeans_converged.rs` and `kmeans_converged/` (`ultradim.rs`, tests) | a whole k-means run to UltraDim's stopping rule, measured |
| the measuring half of `src/futhark_sig.rs`: the timing inputs and work caps, `compile_c_server`, the problem's data and default scheme (`compile_c`, `run_exe` and the binary reader and writer stay in fuller beside `futhark_exec`: the kingdom's tests check with them that a rendering is legal Futhark computing what the interpreter says) | data and measurement, not the kingdom |
| `src/futhark_rings.rs`, `src/futhark_operators.rs`, `examples/futhark_pool.rs` | the start population and variation of genes |
| the deprecated runners (`src/futhark_search.rs`, `examples/futhark_evolve.rs`, `examples/futhark_a7_evolve.rs`, `examples/futhark_pixel_evolve.rs`, phylu `shape_typed`/`shape_fit`) | NOT moved: built once at the tag `pre-5b` into `~/.local/share/fuller/pre-5b/` (a README there names the commit), then deleted in M1b; R0 to R4 and the GPU scripts' runner step run the frozen binaries; the slope check runs from the frozen `futhark_evolve` until F8; a pixel runner on phylu is later work |
| `src/futhark_sig/tests*.rs` (nine files) | sorted test by test, not file by file: a test of the kingdom (a rendering compiles, the interpreter agrees with the compiler) stays and calls `futhark_exec`; a test of a measurement (`futhark_batch`, `kmeans_converged`, `futhark_rings`, the timing inputs) moves with the code it tests |
| `src/futhark_adf_tests.rs`, `examples/futhark_adf_saturated.rs` | stay: `futhark_search::suite` and `input_set` (the deterministic input suite, test inputs of the kingdom) move into the kingdom half of `futhark_sig` |
| `examples/futhark_pool_check.rs`, `examples/futhark_problem_cases.rs`, `examples/futhark_pixel_reference.rs`, `examples/kmeans_cases.rs` | move with the code they call |
| `tests/runner_standard.rs` (fuller's) | its Futhark entries go with the frozen runners (phylu's `tests/runner_standard.rs` registers `futhark_fit` at P4) |
| the pixel family's calls into the GPU host (`src/futhark_pixel.rs`) | its run half; the family's tables stay |

**What stays in fuller:** `futhark_table.rs`, `futhark.rs`, `futhark_view.rs`, `kingdoms/futhark/` (tables, views,
the generator and the sniffer), the kingdom half of `futhark_sig.rs` (instances, signatures, the reference programs
as text, `program*` rendering, `build`, `sexp`, `random_gene`, the e-graph bridge `variants`, `FoldEgraph`),
`typed_gep.rs`, `kingdom_contract.rs`, `futhark_reader.rs`, `futhark_saturation.rs`, `saturated_fold.rs`, `snap*.rs`,
`futhark_cost.rs` (the static cost: a function of the program text, generated with the tables), `futhark_exec.rs`
(one program compiled and run: fuller's own tests check its rendering and every e-graph rule against the compiler
with it), and the two interpreters (`futhark_interpreter.rs`, `futhark_sig::Interp`): they define what a program of
the kingdom computes and check the e-graph's rules; phylu's evaluator calls them, they compute no fitness.

**What stays until the merge (the merge makes it a move inside one workspace):** `fitness.rs`, `runner.rs`,
`telemetry.rs`, `run_file.rs` and `futhark_sig`'s quality parts (`QualityPart`, `quality_inputs*`). By the diagram
they are phylu's, but fuller's other runners (`wgsl_gram_evolve`, `futhark_sig_quality_study`) still use them, and
`fitness.rs` reads the quality parts and the runner's two-level HFF. While they stay, phylu imports them as it does
today. The dependency stays one way: phylu depends on fuller, fuller never on phylu; every moved file's imports of
fuller become `fuller::` paths, and nothing left in fuller imports a moved file (checked by fuller's build after the
move, and its full suite).

**The boundary between the engine and the evaluator (P2a's hook).** Once both live in phylu there is no library
boundary to cross, but the hook stays: it is how the engine takes ANY kingdom scored on the host (the mock fitness of
`p2a_mock.sh`, 6 GPU tests, and P3's 9 drive the engine through it), and removing it would remove tested behaviour.
What goes: `FreshRequest`, `FreshPurpose` and `FreshIndividual` are defined in phylu's `external.rs` (the
`pub use fuller::futhark_evaluator::...` line is deleted); `FutharkEvaluator`'s `pub judge` and `pub gen` go at X1 as
planned; 5a.2's "one seam fuller exposes" is read as "the evaluator's interface inside phylu". Not in this plan, noted
for later: with the evaluator beside the device, it could read the walk stage's expressed nodes instead of the whole
read-back (2.65 GB a k-means call, 5a.1); its own A/B, after R1.

**The general external evaluator (Andrew, 2026-11-04: "a general way to call external compilers, simulators and
programs").** phylu `src/evolve/program_eval.rs`: the trait `ProgramTool` (render a genome, score a batch of texts
under a time limit), `ProgramEvaluator` (the cache by text and the budget count, the time limit, the halving, the
failing source under OUT/bugs/, the run file between batches, the scheme's angle, `into_external`) and
`CommandTool` (any command line). Extracted from `futhark_batch::Host` and `Judge`, not written new. The Futhark
evaluator is one `ProgramTool`; the P2a mock is another; a `CommandTool` test with a shell script is the third. The
Futhark-only parts (the two fitness stages and their re-score, the timing pairs against the reference program, the
resident k-means sample, the champion's test case) stay inside the Futhark tool; a general piece that does not fit
Futhark without changing its numbers stays in the Futhark tool, noted.

**The remaining tasks, re-cut** (sizes in an agent's working days; the GPU scripts are the lead's hours):

```
F3b check fix, pre-5b tag ─> M1a ─> M1b ─> M1g ─> M1c ─┬─> F4 ─┐
                                                         ├─> F5 ─┤
                                                         ├─> F6 ─┼─> F8
                                                         └─> F7 ─┤
G1 (independent of the move) ────────────────────────────────────┼─> P4 ─> G2 ─> G3 ─> R0 ─> R1 ─> R2 ─> R2b ─> R3 ─> R4 ─> RC ─> X1
                                                                 └─> P5 (after P4)        └─> AB2, P3b (after R1)
```

| task | repo | size | what | proof |
|---|---|---|---|---|
| M1a | fuller | 1 | split `futhark_sig.rs` into its kingdom half and its measuring half (`futhark_sig/measure.rs`), behaviour unchanged | fuller's full suite; the evaluator's 13 tests unchanged; and `grep` finds no moved module named in any fuller file that stays |
| M1b | both | 3 | the move of the table above: `git mv` into phylu, imports rewritten, the hook's types defined in phylu, fuller's `Cargo.toml` examples and `lib.rs` entries removed, one path constant for the runtime's crates; the tests move with the code, none is edited but for paths; every fuller file that imports a moved module is listed in the move table and placed (the table of the advisor's review, recommendation 2); `compile_c`, `run_exe` and the binary reader and writer stay in fuller beside `futhark_exec`; the deprecated runners deleted, not moved | both repos' full suites, default and GPU features; the CPU plumbing tests (histogram R and k-means P ledgers) pass in phylu bit for bit; fuller builds with no moved module named |
| M1g | lead | hours | `f3a_rust_generation.sh` and `f3b_kmeans_generation.sh` rerun from phylu (their runner step from the frozen `pre-5b` binary), then `p2a_mock.sh` | the same tests pass with the same counts as in fuller: the proof the move changed nothing; F3b compares the output fingerprint only for rows whose quality inputs are not all at the worst value (the program's output is then garbage, and garbage need not be the same garbage in two processes; the fitness reads only the quality inputs) |
| M1c | phylu | 2 | the general external evaluator above, extracted; the Futhark evaluator and the P2a mock use it | the `CommandTool` test; the CPU plumbing tests bit for bit | M1g's three GPU scripts again, the same counts |
| F4 to F8 | phylu | 1, 2, 2, 1, 1 | as in 5a.3's table, in `phylu/src/futhark_eval/`; F5's `futhark_start.rs` and F8's `futhark_measure.rs` are phylu files | as in 5a.3 |
| P4, P5 | phylu | 3, 1 | as in 5a.3; `futhark_fit` calls the evaluator directly through the hook | as in 5a.3 |
| G1-G3, R0-R4, AB2, P3b, RC | lead / phylu | as 5a | unchanged; the deprecated runner's arms run from the frozen `pre-5b` binaries | as in 5a.4 |
| X1 | both | 1 | the documentation and inventory edits (5a.5): the runners' sources are already gone (M1b) | the gate stays: Andrew reads R0 to R2b, R3 and R4 have run |

**Sentences of fuller's `CLAUDE.md` that contradict the ruling (needs Andrew's OK; not edited; to be rewritten once, after the merge):**
1. Top section: "fuller is the library: kingdoms ..., the typed GEP core, evaluators (compilers, GPU runtimes), fitness
   schemes, the e-graph": evaluators and fitness schemes are phylu's.
2. Top section: "expose it here as library code (as `futhark_search::Scorer` is for phylu's `evolve::shape_typed`)":
   the Scorer moves to phylu; the example of a fuller boundary is wrong.
3. Top section: "`examples/futhark_evolve.rs` and the loop in `src/futhark_search.rs` ... DEPRECATED": they will be in
   phylu until X1.
4. Architecture: `src/futhark_gpu.rs` ("the measured GPU cost ... the cost that decides") and `src/futhark_search.rs`
   ("The SEAM for a GPU-resident population is here too: `Scorer` ... phylu calls all of it"): both move.
5. Architecture: `src/geneframe.rs`, "the nucleotable data model, owned here", and `src/population_dag.rs`: the
   diagram puts the geneframe and population management in phylu; the master symbol table inside `geneframe.rs` is
   kingdom definition. Not moved by this plan: a split for Andrew to decide.
6. Architecture: `src/wgsl/` lists "interpreter, oracle, mutation": mutation is a variation operator (phylu's).
7. "Every evolution runner meets one standard": "Every evolution loop, in fuller or phylu, writes ... through
   `fuller::telemetry`": no loop is in fuller; telemetry is phylu's by the diagram (M2).
8. "Fitness is a named scheme": "`fuller::fitness::SCHEMES` (`src/fitness.rs`) holds every fitness function": true
   until M2, then `phylu`.

**Status 2026-11-04 (M1 and M1c done; M1g's GPU scripts written, not run).** fuller: the plan with the review
applied `2e1a3b72` (tagged `pre-5b`); the F3b check fix `7ccb16b0` (the output fingerprint compared only where the
k-means output is not garbage: inertia and percentile at 1 on every case); the move `6ecbda2d`. phylu: the move
`11a643a7` (`src/futhark_eval/`, `FreshRequest`/`FreshPurpose`/`FreshIndividual` in `evolve::external`, `shape_typed`
and `shape_fit` deleted); M1c `c307fd92` (`evolve::program_eval`: `ProgramTool`, `ProgramEvaluator`, `CommandTool`;
`futhark_eval::tool::FutharkTool`; the compile limit and the failing-source writer extracted from the batch host).
Frozen: `futhark_evolve`, `futhark_a7_evolve`, `futhark_pixel_evolve` (fuller `2e1a3b72`) and `shape_fit` (phylu
`e6a7ad2b`) in `~/.local/share/fuller/pre-5b/` with a README, and histogram's gate under its `inputs/`. Tests (CPU,
`--release`, `-D warnings`): fuller's remaining futhark modules 135 of 135, `gpu_test_lock` 3, `docs_index` 3,
`runner_standard` 1; phylu `futhark_eval` 100 of 100 at M1, then `evolve::program_eval` 4, `futhark_eval::batch` 21,
`futhark_eval::evaluator` 15 (the Futhark tool gives the runner's generation-0 angles bit for bit), `evolve::external`
9, `runner_standard` 1; clippy `--all-targets` clean (fuller default, `wgsl`, `gpu,wgsl`; phylu default, `gpu`,
`gpu,wgsl`); phylu's GPU test binary builds. Not done: the full suites (run once before the merge). The GPU scripts
(`/private/tmp/claude-501/fbench/plan5a/`) build phylu: `f3a_rust_generation.sh` and `f3b_kmeans_generation.sh` run
the frozen `futhark_evolve` then phylu's `futhark_eval::evaluator::tests::on_the_gpu_*` replay; `p2a_mock.sh` runs
phylu's `futhark_typed::tests::the_external_*`; each writes a new OUT (`*_5b`). Deviations, noted: the timing
inputs' shapes (`PATHFINDER_TIMING_ROWS`, `A7_TIMING_N`, `KMEANS_TIMING_N/D/K`) stay in fuller (the problems' data
read them); `examples/futhark_problem_cases.rs` stays (it calls only `compile_c`, `run_exe` and the writer);
`experiments/virtual_blocks/bench.rs` moved too (it calls the rust runtime); a tool returns a `ToolScore` (its vector
and its own angle) where the review sketched a bare vector; the GPU tests' P2a mock keeps its closure (it drives the
stage change, the fresh hook and the state the general layer leaves to a kingdom), and the hash mock as a
`ProgramTool` is a host test.

**F5 DONE (phylu `24af1068`, 2026-11-04).** `FutharkEvaluator::fresh` in `phylu/src/futhark_eval/evaluator/start.rs`
(`set_fresh` takes the regime, the start and the random seed; `fresh_cost` the seconds and the start's evaluations).
The start reproduces the deprecated runner's generation 0 (regime P's reference program or planted slowdown with its
e-graph variants and fold forms; the task alternatives; the pool's program and its mutants; canvas or random genes).
The refill gives the best's unscored e-graph variants (at most 4, only when the best changed), then in P its mutants
(1 to 2 cells, or the ring walk); R leaves the rest to the engine's draw; C has none. The fold forms come on the fold
beat's own request, not in the refill. **One fold request a pair** carries all its winners (`FreshRequest::winners`,
the best's row in `FreshRow`); each form names its winner (`forms_by_winner`). The `fold_top_k` bound of 4 stays,
because a fold's cost grows with its programs. Found: histogram's reference program has no e-graph variant and no fold
form; pathfinder's P start is 24 programs. After the code review (phylu `20b352ad`): In regime P the start population is the reference program, its e-graph variants and the fold's forms (pathfinder: 24), filled with mutants of the reference program by P's `m_of` when fewer (histogram: the reference program and 23 mutants): a departure from the deprecated runner. R and C on histogram and pathfinder start from `Mutants` or `Drawn`; their alternatives all give the
reference program's output; no problem has an alternatives start today. Tests (CPU, `--release`,
`-D warnings`): `futhark_eval::evaluator::start` 7, `evolve::external` 10, `evolve::futhark_typed` 12, the bound test 1.
Clippy clean (default, `gpu`, `gpu,wgsl`), and the GPU test binary builds. `p2a_mock.sh`'s
`the_external_hook_runs_ten_generations_on_two_pairs` now expects 2 fold requests a beat (one a pair), each with 4
forms. Not run: the GPU tests. Left for P4: the start individuals enter through `External.seed` (no origin), and the
alternatives' evaluations are charged in `evaluations_total` only. The P start's memory is measured by the run's script.

**F4 DONE (phylu `ffc4bd98`, 2026-11-04).** `FutharkEvaluator::report(generation, row, genome)` in
`phylu/src/futhark_eval/evaluator/report.rs` reads the evaluator's caches only. It scores nothing and counts nothing,
and a best that was never scored is a bug of our software. `FutharkEvaluator::extras` has the signature of P2b's
`ExternalExtras`, so P4 boxes it. The report holds the scheme's angle and the angle over the whole vector, with that
vector's d and its percentile (`fitness::hff_percentile`), for every scheme. Where the scheme's angle is TrueNorth over
the vector the two angles are the same number; for the retired two-level schemes the second is TrueNorth over the
vector. Each group's angle and percentile sit beside them. It also holds the best's inputs case by case beside the
reference program's (when the run scored the reference program), the speed pairs, the exact counts of each split and
the test cases' measures. The answer-scored problems now measure their test cases inside the quality program
(`Quality::test_raw`, `test_inputs`), never in `inputs`. For k-means the measures are the champion's last runs on the
test cases (`ChampionMeasure`). Departures from 5a.2: `report` takes the best's genome, not `GenerationScores`,
because the engine's hook gives a genome; `Quality` keeps its all-zero flag (`zero`), so a restored program needs no
outputs. Reduce has no test set, so the check holds there trivially; the test asserts that at least two problems have one. Tests (CPU, c backend,
`--release`, `-D warnings`), 3 new: the percentile for every answer-scored scheme in both stages; the test set never
in `inputs` on every problem (k-means by its names; its scored report in the k-means seam test); the extras hook.
`futhark_eval::evaluator` passes 29 of 29. GPU: the report is checked on every call of the two replay tests in
`f3a_rust_generation.sh` and `f3b_kmeans_generation.sh` (built, not run). On the GPU no new expectation: the same
counts, and every assertion holds.

**F7 DONE (phylu `4ecc9d89`, 2026-11-04).** `FutharkEvaluator::state` and `restore`, in
`phylu/src/futhark_eval/evaluator/state.rs`, fill P2b's `ExternalState` slot. The state holds every program scored
(its quality by text, k-means's speed group, the oversized keys), the timing slots, the stage, the counts, the k-means
champion with its last test-case measures, and the pump refill's memory (`refilled`/`set_refilled`, added in phylu
`e1289ae2`: the F5 code review's finding 7). The refill's mutant draw is keyed by seed, generation and pair, and its
"scored before" filter reads the restored cache, so nothing else needs saving. Every `f64` is stored as the hex of its
bits, because an untimed slot is infinite and JSON has no infinity. A scored program's outputs are not saved: k-means's
centroids are about a megabyte a program. **The resume is `new`, then `controls`, then `restore`.** A restore before
the controls, or a state of another format or configuration, is an error that changes nothing. Tests: histogram's
state survives its JSON text; after `restore`, a recorded stage-2 generation (with one infinite slot) scores with no
new evaluation and no module, and with the same angles, inputs, ledger columns and report bit for bit. The same holds
for k-means at the end of its seam test: no test case runs again. GPU: none.

**F6 DONE (phylu `24a08cd3`, 2026-11-04).** `FutharkEvaluator::final_report(champion, run)` in
`phylu/src/futhark_eval/evaluator/final_report.rs`. The controls keep what they measured and write `aa_gate.tsv`
when they timed the A/A sets. The in-run A/A was already in the controls (F2): for the answer-scored problems the
published program goes through the timing slots, and for k-means the reference program is judged against itself
through `prefetch` + `evaluate`. The report takes the champion's score from the caches. On the GPU an exact champion
is re-timed by 11 pairs and the published program by `aa_retimes` re-times, which give the verdict. The report writes
`report.tsv` with the deprecated runner's keys in its order, `best.fut`, and the chart when the run gives its
`series.tsv`. What only the engine knows comes in a `RunSummary`: the regime, the start, the random seed, why the run
stopped, its seconds, the budget, the first exact program and the hall of fame's rank 1. `Judge::staged_eval` now
uses `report::speed_inputs`. Tests (CPU, `--release`, `-D warnings`), 2 new: the controls pass on the c backend for
all six problems, and each report, with the reference program as the champion, has the runner's keys; the GPU
block's keys and verdict from given numbers, and the chart. `futhark_eval::evaluator` passes 33 of 33. Clippy clean
(default, `gpu`, `gpu,wgsl`); the GPU test binary builds. GPU, not run: `f6_controls.sh` runs `controls` and the final
report on `--eval rust` for histogram, pathfinder, kmeans and a7, with 3 re-times to bound the GPU time.

**P4 DONE (phylu `ec141644`, 2026-11-04; GPU not run).** `examples/futhark_fit.rs` over `phylu/src/evolve/futhark_run.rs`.
The flags are a `futhark_evolve` run's (problem, regime P or R, start, pool and ring sources, gate, fitness, quality
weight and cliff, timing pairs, runs, k-means step, eval, backend, evaluations, minutes, random seed, program size,
pump and fold beats, fold winners) plus the island shape (`--island-pairs`, `--intake`, `--champion`: one pair of 16 + 8),
marriage 0.3 (subtree, offspring unmutated, as the old runner), tournament 3, elites 2; regime C is P5. The engine
checkpoints every generation and reports every generation (`progress_every` 1). `OUT/run_card.json` (the stream's
first note) records every flag, the island shape, the refill cap of 13 a pump, `expressed_edits` (2 in P, 4 in R), the
operator rates (`futhark_typed::operator_rates`), the marriage, and where generation 0 lands: pathfinder P's 24
start programs fill the intake (0..16) and the champion island (16..24). The start goes in through `External.seed`.
The runner keeps each start individual's origin for its ledger, but the genealogy names it `init`. The engine now takes
an `ExternalStream` from the hook: the stream's kingdom (`futhark_sig`), its fitness scheme (`start_with` the quality
weight and cliff), its notes, and the progress table's columns from the evaluator's report. The progress table no longer
prints SR's header (the P2b review's item 4). The `run_start`'s objectives are now the evaluator's column names.
`ExternalState::save` returns a `Result`. A null state is no longer saved silently. The runner writes `ledger.tsv`
(one line for each scored individual, with the evaluator's columns), `hall_of_fame/` (program text), `series.tsv`,
`per_generation.tsv`, `test_case.tsv` (k-means), and at the end `report.tsv`, `best.fut` (`final_report`, with
`series` None) and the chart (the runner's, the F6 review's item 6). At the stage change the runner's hall of fame
starts again and stage 1's goes to `hall_of_fame_stage1/` (the F6 review's item 2). The resume runs `new`,
`controls`, `set_fresh`, then the hook's restore. That restore puts back the evaluator's state and the runner's own
state, and with it the ORIGINAL controls (`FutharkEvaluator::replace_controls`). The resumed process's controls are
labelled a re-check in a stream note and in `aa_gate_recheck.tsv`, and `aa_gate.tsv` is the original again (the
F4/F7 review's item 5). The ledger and the series are cut back to the checkpoint's generation. Found and fixed:
`evaluate_generation`'s stage change returned the stage-2 rows with the stage-1 pass's `new_evaluation` flags lost, so
the engine's charge (`new_rows` as long as `evaluations_new`) would have failed at the first stage change. The flags
are now carried over. Registered in phylu `tests/runner_standard.rs`. Tests (CPU, `--release`, `-D warnings`):
`evolve::futhark_run` 7 of 7. These cover every flag in the card, the errors, the default fitness (k-means
`futhark-kmeans-better`), the card's statements, the cut-back, and the progress table. They also include the hooks
on histogram R on the c backend: the charge, the ledger, the stage change with stage-2 entries only in the hall of
fame, `series.tsv` and `per_generation.tsv` once a generation, and the hall of fame's values. That test ends with a
resume into a second evaluator that scores the same generation with nothing new and the same fitness bit for bit.
Also: the `gpu` host test of the engine configuration (1), `runner_standard` 2 of 2, the evaluator's stage-change
tests 2 of 2, and `evolve::external` and `evolve::program_eval` 14 of 14. Clippy is clean on default, `gpu` and
`gpu,wgsl`. The GPU test binary and the example build. Written, not run:
`/private/tmp/claude-501/fbench/plan5a/g2_futhark_fit.sh`. It runs the three engine-driven tests (histogram and
pathfinder R on the c backend end to end, and a resume against the uninterrupted run), then pathfinder P and
histogram R (the frozen gate) on `--eval rust --backend gpu`, 3 generations each.

**P5 DONE (phylu `34ff7d9e`, 2026-11-04; GPU not run).** `futhark_fit --regime C`, in
`phylu/src/evolve/futhark_run/random_search.rs`, runs without the engine, since random search has no selection.
Each generation is `pop` fresh individuals from `FutharkEvaluator::fresh`. The population is the evolution runs'
(island pairs x (intake + champion), 24 by default). Generation 0 is the start, as in regime R (`drawn` by default;
`--start random`, the engine's draw, is an error in C). Every later generation uses the new `FreshPurpose::Random`:
the deprecated runner's draw, `(seed ^ REGIME_C_TAG) * golden ^ generation`. Each generation is scored by the P/R
runner's own hooks (`FutharkRun::score` and `extras`, so `evaluate_generation` too), with the same outputs: the
stream through `RunTelemetry` (kingdom `futhark_sig`, a snapshot and the hall of fame every generation), `ledger.tsv`,
`hall_of_fame/`, `series.tsv`, `per_generation.tsv`, `stages.tsv`, the run card, and at the end `report.tsv`, `best.fut`
and the chart. The final report is of the hall of fame's rank 1 (the best ever), whose gene is kept beside the hall of
fame. The run honours `killme.trigger`, the minutes, evaluations and generations guards, and `--stop-at-exact`. A
checkpoint is written after every generation (`OUT/checkpoint_c.json`), and `--resume` continues it; a generation the
run file stopped part way writes none. A C run needs `--gate`, as R does. Tests (CPU, `--release`, `-D warnings`):
`evolve::futhark_run` 12 of 12, 3 of them new: C's arguments; the hall of fame holds the best ever, not the last
generation's best; and a short c-backend histogram run end to end (every output, a resume equal to the uninterrupted
run, report keys equal to regime P's). Also the draw test (`a_random_search_generation_is_the_runner_s_draw`) and
`runner_standard` 2 of 2 (C's markers added). Clippy is clean on default, `gpu` and `gpu,wgsl`; the GPU test binary
and the example build. RC needs a GPU script after R4. It is not written: it is the R4 script's runner step with
`--regime C`.

### 5a.3 (continued): the task table and the GPU scripts

| task | repo | size | what | files | tests (no GPU) | GPU script (the lead) | depends |
|---|---|---|---|---|---|---|---|
| F1 | fuller | 1 day | Everything the evaluator holds OWNS what it reads, by `Arc`: `Host<'a>` (`futhark_batch`), `Interp<'a>` (futhark_evolve.rs 462, held by the judge) and `Genetics<'a>` (1283-1287) borrow the kingdom, the signature, the contract, the layout and the constants' lengths today; `futhark_sig::load` and `signature` return shared handles. Never `Box::leak` | `src/futhark_batch.rs`, `src/futhark_sig.rs`, `src/futhark_interpreter.rs`, callers | every `futhark_batch`, `futhark_sig` and interpreter test unchanged; a host and an interpreter built from handles whose originals were dropped run a C batch | none | none |
| F2 | fuller | 3 days | `FutharkEvaluator::new`, the `Judge` (about 820 lines with the k-means paths), `Genetics`, the controls and the start moved behind the owned object, unchanged; `FreshIndividual` and `FreshRequest` defined here; `futhark_evolve.rs` calls the library, its behaviour unchanged | `src/futhark_evaluator.rs`, `src/lib.rs`, `examples/futhark_evolve.rs` | **the plumbing test**: histogram R, 2 generations, random seed 1, C backend, before and after the move: the ledger byte-identical except the milliseconds columns; the k-means C-backend run (`--evaluations 2`): the same quality and speed inputs bit for bit | none | F1 |
| F3a | fuller | 2 days | `evaluate_generation` for the answer-scored problems (reduce, scan, histogram, pathfinder, a7): express from the head cells, render once a distinct gene, `prefetch` + `evaluate` in one module a call, the stage change inside the call (every row scored again in stage 2), the HFF input vector and the scheme's angle a row, `evaluations_new` and `evaluations_total`, the module count and compile seconds in `GenerationCost`, the run file checked between batches (`stopped`) | `src/futhark_evaluator.rs` | the programs of F2's histogram plumbing run through the seam give the inputs the runner recorded, bit for bit; a cached row is free (`evaluations_new` 0); the stage change re-scores every row once (a mock host, as `futhark_batch::tests` mocks a compile); a removed run file between two batches returns the first batch's rows and `stopped` | `f3a_rust_generation.sh`: histogram, `--eval rust`, 3 generations of 24 through the seam: one module a call; the inputs equal the runner's on the same programs | F2 |
| F3b | fuller | 2 days | `evaluate_generation` for k-means: `prefetch_kmeans` + `evaluate_converged` on the resident sample, and the champion's test case (`converged_on_test`, its own module, counted) | `src/futhark_evaluator.rs` | the programs of F2's k-means plumbing run give the inputs the runner recorded, bit for bit (C backend) | `f3b_kmeans_generation.sh`: k-means, `--eval rust`, 3 generations of 24: one module a call, a third when the champion's text changes; one upload, the rest resident hits | F3a |
| F4 | fuller | 1 day | `report`: the best's angle and percentile with d for EVERY scheme (a report, not a fitness change), its inputs a case against the reference program, its exact counts, the test cases' measures (the answer-scored problems' test cases are in the quality program already; k-means's `converged_on_test` when the champion's text changes) | `src/futhark_evaluator.rs` | the percentile equals `fitness::hff_percentile` at the vector's d; the test set never enters `inputs` (asserted on every problem) | in `f3a_` and `f3b_` | F3b |
| F5 | fuller | 2 days | `fresh`: the start populations (regime P, the ring canvas, the alternatives, the mutants of one program) and the pump refill and fold forms, each a `FreshIndividual` with its origin, parent rows and mutation event; the refill's content per regime (5a.2: variants, fold forms, then in P mutants of the champion island's best, in R the random draw), no two with one text or one tree, no text scored before; the cap a pump (the request's rows); the P start's cost (k-means: 149 s the e-graph, 193 s the fold, 19.5 GB) reported | `src/futhark_start.rs`, `src/futhark_evaluator.rs` | pathfinder P at random seed 1 gives `futhark_evolve`'s generation 0 (same programs, same order: R0's programs; histogram's P start is filled with mutants since phylu `20b352ad`, a departure); the ring canvas and the alternatives as the runner builds them; a refill of 13 in P holds no random row, no repeat and no scored text; a variant equal to a scored program is not offered | none | F2 |
| F6 | fuller | 2 days | `controls` and `final_report`: the A/A sets, `aa_gate.tsv`, the in-run A/A (the reference program as a candidate through `prefetch` + `evaluate`, futhark_evolve.rs 2583-2595), the k-means A/A noise floor, the champion's re-time (it reads the timing slots F3a owns) and verdict, `report.tsv`, the chart | `src/futhark_evaluator.rs` | the report's keys equal the runner's (a test lists them); the C-backend controls pass for all six problems | `f6_controls.sh`: `controls` on the GPU for histogram, pathfinder, kmeans, a7 | F3b |
| F7 | fuller | 1 day | `state` and `restore` | `src/futhark_evaluator.rs` | a round trip; after `restore`, a recorded generation scores with no new evaluation and the same inputs bit for bit (C backend) | none | F3b |
| F8 | fuller | 1 day | `examples/futhark_measure.rs`: the slope check, the operator measurement, the rescore and the A/A set alone, over the library; the same flags and output files as today (regime C goes to P5) | `examples/futhark_measure.rs` | A7's slope check on the C path with 30 random genes gives `docs/measurements/2026-11-03_futhark_a7_slope_check_cpu.tsv`'s counts (63 neighbours, 7 exact) | `f8_check.sh`: the slope checks still not run (reduce, scan, a7 at 1,400 random genes) | F2, F6 |
| P1 | phylu | 2 days | The Futhark kingdom on the engine: `evolve::futhark_typed` builds the shape, sampling, walk-only stage and paged device parts from the evaluator's contract and layout (`shape_typed`'s `ShapeRun::new` and `check_layout`, moved); `Engine::new_with_table` takes `kingdom = "futhark"` beside regex and TSR, with a placeholder `Data`; `combinations_for` and every other test of `EXTERNAL_KINGDOM` key on "the config has an external fitness", not on the name `safeWGSL`. The first time `External` meets a typed layout | `src/evolve/futhark_typed.rs`, `src/evolve/engine.rs`, `src/evolve/mod.rs` | `check_layout` for all six problems at their program sizes (host only; k-means is long, minutes at 14,359 types: not ignored, timed, its seconds printed) | G1 | F2 |
| P2a | phylu | 3 days | The hook's contract, and its first run: the budget charges the hook's `evaluations_new`, not one a NaN row; the widened `ExternalFresh` returns `Result<Vec<FreshIndividual>>` (fuller's type) and is asked once a pair, its individuals placed in EVERY pair's intake (`fresh` and `fresh_plan` per pair), with the refill's fresh cohort label and each one's mutation event; a fold beat asks it for forms; `stopped` ends the run as `killed` before the NaN check | `src/evolve/engine.rs` | the pure parts on the host: the budget arithmetic (a cache hit charges nothing), the refill's row placement for one and two pairs and the cohort labels (`fresh_plan` with a mock hook) | `p2a_mock.sh` (`cargo test --features gpu futhark_typed`): a mock external fitness (a hash of the gene, no compiler) through 10 generations, histogram's layout, two pairs: both pairs' refills hold the hook's genes with fresh cohorts, the budget counts only new hashes; the k-means layout after G1 | P1 |
| P2b | phylu | 3 days | The rest of the external kingdom's run: a stage change restarts the hall of fame and writes a `stage_change` event (the re-score itself is the evaluator's, F3a); an `extras` hook fills the snapshot with the evaluator's report; a `serde_json::Value` slot for the hook's state in `Checkpoint`, format version 4 (a version-3 checkpoint is a stated error, never read as 4); the hall-of-fame file's columns from the evaluator (`input_names`, the program text), not SR's | `src/evolve/engine.rs`, `src/evolve/checkpoint.rs` | a checkpoint round trip with a hook state; a version-3 file named and stopped; the hall-of-fame header from a mock's names | in `p2a_mock.sh`: the stage change (forced by the mock at generation 5) empties the hall of fame and writes the event; a resume at generation 7 continues with the same state | P2a |
| P3 | phylu | 2 days | Variation for wide typed genes: the point mutation of the deprecated runner (`point_mutate` with `m_of`'s count) on the device. `Config::expressed_edits`, 2 in regime P and 4 in R, gives each child `1 + below(k)` edits after the device's crossover, one after another, each setting a cell the child's program reads now to another member of its alphabet (one LEAF pass per edit over a point table, `LeafTable::build_point`); the cells the program does not read are never touched. Marriage (phylu `marriage.rs`) is an engine beat for a typed external kingdom: parents are never duplicates, the subtree pair is sampled (linear in tree size), and mutated offspring get the point edits. The operator set is the old runner's (homologous crossover, the edits, marriage); any other operator that is on is named. The engine's draw keeps the GEP head rule. The per-cell rate first planned (`expressed_hits` over the median expressed node count, `true_wgsl_fit`'s rule) failed: it rewrites the unread cells too, and the P3 GPU run measured 278 changed cells per child at the median in regime P. DONE: phylu `95ed5b22`, `8249a5ff`, `e6a7ad2b`; 9 of 9 GPU tests 2026-11-04 04:41Z; review `docs/reviews/2026-11-04_step5a_P3_code_review.md`. | `src/evolve/engine.rs`, `src/evolve/marriage.rs` | The edits match `point_mutate` in distribution (host twin, histogram); every child changes at most k of its parent's expressed cells (histogram, pathfinder, P and R); the engine's draw is of the old runner's sizes; marriage's sampled pair follows the domain's weights and its offspring are legal programs. | in `p2a_mock.sh`: the device's children equal the host twin's; marriage offspring fill each island's last rows in adjacent pairs with swapped parents. | P1 |
| P3b | phylu | 2 days | The row operators of 3f (`invert-rows-coding`, `transpose-rows-coding`, `cross-rows-coding`, the relative rows) on the device, host twin first | `src/evolve/typed_vary.rs`, `typed_vary.wgsl` | the host twin equals `fuller::futhark_operators` on 20,000 genes a problem | an A/B against R1 | R1 |
| P4 | phylu | 3 days | `examples/futhark_fit.rs` and `evolve::futhark_run`: the flags of a `futhark_evolve` run (problem, regime R or P, start, fitness, quality weight, cliff, pairs, eval, backend, evaluations, minutes, random seed, program size, pump and fold beats) plus the island shape (pairs, intake, champion); `progress_every = 1`; the card states the island shape, the refill's cap a pump, the expressed-cell rate of the regime and where generation 0's rows land (the champion island not empty); the stream (kingdom `futhark_sig`, the evaluator's column names), `killme.trigger`, checkpoints, the ledger with the evaluator's columns, the hall of fame with the program text, `per_generation.tsv`, `series.tsv`, `report.tsv`, the chart; registered in phylu `tests/runner_standard.rs` | `examples/futhark_fit.rs`, `src/evolve/futhark_run.rs`, `tests/runner_standard.rs`, `experiments/README.md` | the runner-standard test (source markers); the card holds every flag | G3 | F4-F7, P2b, P3 |
| P5 | phylu | 1 day | `futhark_fit --regime C`: the random-search control arm, `pop` fresh individuals a generation from `FutharkEvaluator::fresh` (random), scored through the same `evaluate_generation`, the same outputs (stream, ledger, hall of fame, `report.tsv` with the best-ever), no engine selection | `examples/futhark_fit.rs`, `src/evolve/futhark_run.rs` | regime C's report keys equal regime P's; the hall of fame holds the best-ever, not the last batch's best | RC, after R4 | P4 |

**The GPU scripts, with what is expected before they run.**

| script | measures | expected |
|---|---|---|
| G1 `g1_futhark_device.sh` (after P1, before P2a's k-means test) | the engine's device for each problem at its program size, 24 rows: init, variation and read-back (`read_generation`) seconds, device memory, the walk stage's `binding_limit()`, page count and per-page output size, Metal's `recommendedMaxWorkingSetSize` (recorded nowhere yet), the walk's node count of every row against the host's expression | every row's node count equal; reduce to pathfinder well under 1 GB; kmeans 27.6 million words a row, 110.4 MB, 2.65 GB for 24 rows a side, 5.3 GB both sides, within the adapter's 4 GiB a binding and a buffer (one page a side, or a few if `binding_limit()` is smaller) and 4 % of the 128 GB of unified memory; the read-back of 2.65 GB in seconds |
| G2 `g2_two_devices.sh` (after P4) | the evaluator's A/A noise floor on histogram (11 pairs) alone and with the engine's device made and idle; the resident sample over 3 k-means generations; modules a generation and compile seconds a call from `GenerationCost` | the two floors within the spread of the A/A medians measured 2026-11-01; one upload in 3 generations, the rest resident hits; one module a generation, two on a pump or fold generation, a third when the k-means champion's text changed |
| G3 `g3_smoke.sh` (after G2) | histogram P, 3 generations, `--eval rust`; then `--resume` to 5; then a long run stopped by deleting `killme.trigger`; then the same stop during a k-means generation | the stream, ledger, hall of fame, `per_generation.tsv` and `report.tsv` written, a snapshot every generation; the resume continues at generation 4; both stops are `killed`, the k-means one between two batches of its generation |

### 5a.4 The regression: phylu reproduces the deprecated runner

Not bit-identical runs: ALPS, the islands and the refill change the search. What is shown, and how:

1. **The plumbing, bit for bit (CPU, F2, F3a, F3b):** the same programs give the same HFF input vectors through the
   runner and through the seam.
2. **R0, the anchor on the GPU path:** regime P generation 0 on `--eval rust`, pathfinder, whose P start is 24
   programs found, both runners scoring the same 24 programs (F5's test gives the same programs in the same order;
   histogram's P start is filled with mutants, a departure, so it is not the anchor), writing the same quality and speed inputs
   bit for bit in their ledgers. The plumbing test and R0 are the proof that phylu REPRODUCES the deprecated runner's
   fitness; with them, R1's comparison is a comparison of search dynamics and nothing else.
3. **R1, histogram, regime P** (`futhark-slope4-qqs`, m = 8, 11 pairs, 2,000 evaluations, a 60-minute guard), random
   seeds 1, 2, 3 on each runner: `futhark_evolve` (population 24) and `futhark_fit` (one pair, intake 16, champion 8,
   the pump every 5 generations, the default cohorts: 5a.6). **R2, pathfinder**, the same. Then **R2b, histogram
   regime R** from random (the slope check passed 2026-10-29), measuring the evaluations to the first exact program.
4. **R3, k-means, the first runs with `futhark-kmeans-better`** (N = 192, 24 individuals, m = 3, ε = 0.005, 2 pairs,
   regime P from UltraDim's kernels translated, 60 minutes): one run of the deprecated runner (no run of this scheme
   exists to compare with) and one of phylu, the in-run A/A first (its quality inputs exactly 1/2, its speed floor
   written). Expected: the reference program's angle is acos(3/4) = 0.7227 at every m (all inputs 1/2),
   d = 6 x (3 x 3 + 2) = 66; an individual below 0.7227 is better than the reference program on the flat vector,
   and its per-case inputs say on which measure.
5. **R4, a7, regime P** on phylu (replacing `shape_fit`), with the scheme `futhark-slope4-qqs`.
6. **After R1, the A/Bs:** AB2, two pairs of (16 + 8) against R1's one pair, histogram P, the same three random
   seeds; P3b's row operators against R1. **RC**, regime C through P5, after R4 (the ruling's order: the control
   runs after the run it controls), the same budget and random seeds.

**The table each comparison fills**, one row a measure, each with its units and conditions, a column a run (three
random seeds a runner, six numbers a measure) and the difference in percent: the best's HFF angle (rad) and
percentile F(θ) with d; exact cases of train, held and test; the champion's 11-pair re-time median against the
published program and its speed verdict against the A/A medians; the champion exact on the timing input;
evaluations to the first exact program (R2b); evaluations a second; modules a generation; the learning curve
(`per_generation.tsv`). For k-means: the per-case quality inputs (inertia, percentile, convergence) and speed inputs
of the best, and the test case's. A row states that phylu's champion island is not empty at generation 0 (8 of the
start population's programs start there) and that the refill spent the evaluations the report gives.

**What three against three can and cannot show.** It can show that every measure is produced by both runners, the
six numbers themselves, and whether each of phylu's runs is worse than the deprecated runner's worst (a gross
regression). It cannot show that one runner is better than the other, or that they are equal: three draws a runner
define a wide range, and at 3 against 3 the smallest p a rank test can give is 0.05 one-sided (1 of the 20 ways to
split six runs in two threes), 0.10 two-sided, so no difference at all can be called significant. The table is the
result; its only verdict words are "produced by both" and "worse / not worse than the deprecated runner's worst". A
worse result is reported as measured and blocks X1 until its cause is found and fixed in our software; the engine is
never tuned toward the old runner's numbers.

### 5a.5 When `futhark_evolve` and the `futhark_search` loop are removed

X1, one commit in each repo, when: P4 and P5 are registered in phylu's runner-standard test; Andrew has read the R0,
R1, R2 and R2b tables; R3 and R4 have run on phylu; `futhark_measure` covers every non-evolution command (F8) and
`futhark_fit --regime C` the random-search control (P5). Then: `examples/futhark_evolve.rs`,
`examples/futhark_a7_evolve.rs` and `futhark_search::evolve` with its `Config` go; phylu's `shape_typed` and
`shape_fit` go with them, and with them `futhark_search::Scorer`, `fitness_of` and `product_twice` (A7 runs through the
evaluator); the `shapes` fitness schemes stay registered (the rule of `docs/FITNESS_SCHEMES.md`), their runner column
naming the commit that last ran them. `docs/CHECKLIST_futhark_problem.md`'s cells move to `futhark_fit`,
`docs/STANDARD_runners.md`'s inventory, `docs/FITNESS_SCHEMES.md`'s runners table, both `tests/runner_standard.rs` and
`CLAUDE.md` (by Andrew or on his word) follow. A past experiment is reproduced at its own commit, as listed in
`docs/FITNESS_SCHEMES.md`.

### 5a.6 The four questions, decided as the review recommends (2026-11-03)

1. **k-means on the device: N = 192 and 24 individuals.** It fits: one gene is 27.6 million words = 110.4 MB; 24 rows
   are 2.65 GB a side, 5.3 GB both sides, 4 % of the 128 GB of unified memory (`hw.memsize` 137,438,953,472, Mac15,8).
   The typed device is requested with the adapter's limits (`shape_fit.rs` 100-104, `typed_device.rs` 950-951), and
   this machine's adapter reports `max_storage_buffer_binding_size` 4,294,967,292 bytes and `max_buffer_size`
   4,294,967,295 bytes (`docs/reports/2026-10-22_ST9_volumepassage_champion/timing_151/log.txt` line 8, phylu); the
   store is paged at the binding limit. Two pairs (48 rows, 10.6 GB) fit as well. G1 prints Metal's
   `recommendedMaxWorkingSetSize`, recorded nowhere in either repository, beside its seconds. The bound on the
   population is the evaluation cost (a whole k-means run a new text, 13.2 s on the c backend), not memory. N stays
   192 (the reference program is 131 functions).
2. **The island shape: one pair, intake 16 + champion 8, the pump every 5 generations, the default cohorts**
   (`cohort_merge` = 5 x 5 = 25 generations), like for like against a population of 24 for R1 to R4. Written down:
   `promote_fraction` 0.03 of 8 rounds to 0 and the floor makes it 1 promotion a beat; the refill is 13 rows a beat.
   The A/B after R1 (AB2) is two pairs of (16 + 8), which the device holds; P2a places the refill in both pairs.
3. **Operators: the device's operators plus marriage first, the row operators after R1.** The regression runs the
   old runner's point mutation in expressed cells (`expressed_edits` 2 in P: 1 to 2 cells a child; 4 in R: 1 to 4;
   `m_of`), its two-point crossover (no inversion: `futhark_typed::operator_rates`), and marriage (P3); the row operators of 3f (P3b) are an A/B after R1, host twin first.
4. **Regime C: in phylu, `futhark_fit --regime C` (P5), a control arm over the same seam, run after the evolution
   runs (RC after R4).** Random search has no selection, so it is not evolution; but it is a budgeted loop over
   generations with a ledger, a hall of fame and a stream, which is a runner, and fuller keeps none. It lives beside
   the run it controls, on the same binary and evaluator. `futhark_measure` keeps what scores a fixed list of
   programs: the slope check, the operator measurement, the rescore, the A/A sets.

### 5a.7 The advisor's review, applied

| finding | where it is applied |
|---|---|
| 1. the external path untried: no caller; keyed on `safeWGSL`; the refill in the first pair's intake only; genomes only in `fresh`; cache hits charged to the budget; the stage change through the hook, the hall of fame in the engine; no checkpoint slot, SR's hall-of-fame columns; `External.seed` across both islands | Context ("the critical path"), 5a.1 rows, 5a.2 (`FreshIndividual` in F2, `evaluations_new`), P1, P2a, P2b, P4's card, 5a.4's table |
| 2. two modules on a beat generation, k-means's test case a third | 5a.1 "scoring on the engine's beats", 5a.2's seam comment and `GenerationCost`, F3a, F3b, G2 |
| 3. the pump's refill in regime P | 5a.2 "what `fresh` returns" and its budget share, F5, P4's card |
| 4. F2 and F3 sized in days, F3 split, `Interp<'a>` in F1, F6 after F3, P1 timed on k-means, G1 before P2's k-means test | 5a.3 (sizes, F1, F3a/F3b, F6, P1, P2a, G1, the DAG and the lead's order) |
| 5. the regression at three against three; R0 | 5a.4 items 2 and "what three against three can and cannot show" |
| 6. `KMEANS_FITNESS.md` 454 against the measured limits | corrected in `docs/KMEANS_FITNESS.md` ("Known limits"); 5a.6 question 1 |
| 7. "seed" and "step" | 5a's prose says "reference program", "random seed" and "iteration"; code identifiers (`pair_steps`) stay |
| 8. the run file between batches; the read-back timed; `progress_every = 1`; the point mutation in expressed cells per regime (`expressed_edits` 2 in P, 4 in R, the old runner's `m_of`) and the operator rates (`futhark_typed::operator_rates`), stated on the card; `f32` fitness; the run total and the call's new evaluations | 5a.1 rows, 5a.2 (`stopped`, `fitness`'s comment, `evaluations_new`/`evaluations_total`), F3a, P2a, P3, P4, G1, G3 |

## 6. Next steps after this plan (Andrew, 2026-11-04)

When step 5 is complete (the regression runs R0 to R4, RC, AB2 and P3b run and read, X1 done, both repositories' full
suites passing, merged into main), the work continues in this order.

### 6.1 Next step 1: fix the faults not yet fixed

Each fault gets a fix and a test, as every fault does; a compiler fault gets its note for the compiler team in
`docs/issues` first.

| fault | where it is recorded | evidence |
|---|---|---|
| the rust-wgpu fork's internal error "Buffer lifted_const_reg_rep_tr_subhistos_mem used at multiple types" on k-means modules (an i32 and an f32 histogram share one lifted constant buffer) | `docs/issues/2026-11-04_futhark_rust_wgpu_buffer_multiple_types.md` with its `.fut` | the F3b GPU check, 2026-11-04 10:21Z: 77 halvings in the deprecated runner's k-means generation 0, the same programs through phylu |
| k-means programs built from histograms of zeros give a different (garbage) output on every GPU run; their quality inputs match | the F3b check's "not compared" lines; phylu `7ccb16b0` compares their fingerprint only when the output is not garbage | the F3b GPU checks of 2026-11-04 13:29Z and 16:23Z (`histrows.add.k zeros centres` and others); likely the same fork fault as above |
| Futhark 0.28.1 `futhark c --library`: "Type error after pass 'Fuse SOACs' ... Name inpacc_... bound twice" on random histogram rows | `/private/tmp/claude-501/fbench/plan5a/size_bound/ice/histogram_225_275_k04.fut` and its log (to be moved into `docs/issues`) | 11 of 15 failing histogram and pathfinder rows of 225 to 1,100 nodes; 1 to 4 per 25 to 28 rows |
| the rust-wgpu fork: "Migration graph sink could be reached" on one 24 x 1,000-node histogram module whose rows each compile alone on the C backend | the F3a review-fix agent's report, 2026-11-04 | `/private/tmp/claude-501/fbench/plan5a/size_bound/` |
| `wgsl::safe_reader::tests::a_tile_loop_unrolled_in_full_renders_straight_line_and_reads_back` overflows its stack in debug builds (passes with `RUST_MIN_STACK=16777216`) | the F3a review-fix agent's report | `/private/tmp/claude-501/fbench/plan5a/size_bound/test_wgsl.log` line 1122; present at HEAD before 5a |
| a `futhark rust-wgpu --fuse=100` compile of `tests/corpus/futhark_compile_failures/setmem_loop__45e18ea3f5e5.fut` left running for over 15 hours with no parent (the compile limit did not reach it) | this plan | process 76740, started about 2026-11-03 17:20Z |

### 6.2 Next step 2: plan the merge into one phylu repository with two crates

Write the plan (then its reviews, then the DAG of agents, as every plan) for moving fuller and phylu into ONE
repository named phylu, a Cargo workspace of TWO crates, following Andrew's architecture diagram of 2026-11-04:

- **the symbolic crate** (today's fuller): kingdoms and their views, the AST, gene encoding and decoding, program
  rendering, and symbolic rewriting (the e-graph, snap, fold);
- **the evolution crate** (today's phylu): the gene frame, the engine (islands, virtual ALPS, the pump), variation and
  selection, the population, fitness measurement (HFF schemes, GPU scoring, `program_eval` for external compilers and
  programs), the runners, telemetry, checkpoints and `evowatch`.

The plan must cover: what moves where file by file (including what the diagram puts in phylu that is still in fuller:
`fitness`, `telemetry`, `run_file`, `geneframe`, `population_dag`, the WGSL mutation); both histories kept (`git
subtree`); the dependency direction (the evolution crate depends on the symbolic crate, never the reverse); the Python
extension (`fuller` is imported by hff: keep the module name or change hff's import); the helper crates
(`futhark-rust-rt`, the shim); the docs index, both CLAUDE.md files rewritten once, the memory paths, the papers'
repository references; and the order: open branches merged to main first.

## 7. Data types, properties and scope: the implementation plan (Andrew, 2026-11-05)

The specification is `docs/DESIGN_futhark_types_and_properties.md` (agreed, commit `f66891be`), cited below as "the
design". This section plans its implementation and does not change it; where an earlier draft of this section
disagreed with the design (types as shape classes, a binder stack in the type), the design wins.

**Objective.** A gene's types are the data types of function arguments and nothing else (`f32`, `[]f32`, `[][]f32`,
`i64`, `size`, tuples, function types). An array's length and an index's bound are **properties** that travel with
the value; which variables a node may use is **scope**, the environment the decoder builds; where a requirement fails,
the rendered program carries an **adapter** (`fit`, `clampix`). Every gene still expresses a legal program.

**Approach, in Andrew's order.** (1) Change the gene frame's data structures: the six tables of the design's section 3
(DataType, PropertyType, Argument, PropertyRule, Requirement, Binding), with arity derived from the Argument rows, and
the decoder's five passes (design section 4). (2) Test it on Futhark: the master table and views converted, the
reference programs converted, the renderer's adapters and prelude, the interpreter, the reader, the e-graph, and
phylu's layout, variation, start population and replay. (3) Only if Futhark works, migrate SR, TSR, regex and WGSL
(7.7).

**What is unique.** Multi-typed GEP keeps its guarantee (every gene expresses a legal program) while the number of
types falls from the cross product of data types, sizes and scopes to the data types: k-means from 14,359 to about 9,
and a gene from 27.6 million words to about 31,000 (design 7.1). Arity, the frames and Ferreira's sizing are
unchanged; properties, requirements and bindings are read only after Karva has expressed the tree.

**Status.** Advisor review applied (aae172d8); the Fable review (24453707) and the adversarial review (88629b19)
applied here, with the lead's five decisions (7.8). Design agreed; this plan; G1 built.

**G1 status (2026-11-05): done.** `src/geneframe.rs`: `KingdomTables` (symbols with their spelling, `DataType`,
`PropertyType`, `Argument`, `PropertyRule`, `Requirement`, `Binding`); `PropertyExpr` (a size name, a literal, `?`,
`in<k>.<property>`, a symbolic sum, and a symbolic `min` as 7.1 now asks); `Arity::from_arguments`;
`KingdomTables::check` (every defect "our software has a bug: ..."), `signatures`, `symbol_table`; a text form that
reads back to the same tables (an exchange form for tests; master 6's file format is F0's). Tested: the derived arity
is today's row for row for SR, Law, typed depth, regex, the four TSR variants and both WGSL kingdoms; the contract
rebuilt through the tables is today's for SR, regex, TSR, the three WGSL kingdoms, the Futhark matrix and 1-D
kingdoms and the five signature views (k-means: 110,871 symbols); mixed data types; round trips; the check's defects.
No kingdom converted.

**F0 status (2026-11-05): done.** The master 6 format is in `kingdoms/futhark/master/README.md`. It has the files and
their columns, the master 5 to master 6 mapping, eleven rows converted by hand, and the decisions. Requirement
relations are `=`, `<=` and `<`; `<` is used only for the work bound `env.trips < depth`, a subject G3 adds. A table of
which property kinds compare lets `bound <= len` through. The lambda-body rule is an ordinary input requirement. Each
requirement names its adapter, and adapter `let`s use the reserved names `ad_*`. A heavy adapter that would break the
work bound substitutes its node. Andrew's confirmation of the work bound (7.5a item 1) is still open.

**F1a status (2026-11-05): done.** Master 6 is in `kingdoms/futhark/master/6/` beside master 5: 78 row templates
(master 5's 86, with its eight variable terminals merged into `var@{T}`, M0087, and its zero arrays now root
variables), 9 data type templates, 5 properties, 8 adapters. `sniff.py --format 6` writes each problem's specification
with its root environment (inputs, zero arrays, sizes); `view.py` (`View6`) writes its data types, symbols and
environment under `kingdoms/futhark/sig/<problem>/m6/`, committed and checked byte for byte by `gen_tables.py --check`
with the master 6 witnesses N0-N19 and the per-view V1-V7. Data types (symbols): reduce, scan and histogram 4 (41),
pathfinder 5 (43), a7 5 (13: the matrix family alone, so no `i64` and no `size`), k-means `f32` 9 (83). Where F0's
format was silent, master 6's README section 9 records what F1a decided.

**G2 status (2026-11-05): done.** `KingdomRelations::variables` (`with_variables`), `KingdomContract::is_variable`,
`variable_capable`, `variable_of`, `takes_word`; `TypeBound::variable` and `dv()` = `h_T + t_T`; `Domain::Dv`,
`Frame::dv_off`/`dv_len`/`dv_pos`, the Dv lanes after every Dc, `GeneLayout::lane`/`dv_of_cell`/`in_alphabet`,
`DV_WORDS` = 2^16. `express`, `walk_expressed` and `write_back` carry a `var.T`'s word in `const_index`. Every operator
that moves a head or tail cell moves its Dv word with it, and one applied to a Dv span alone is a defect. All 17
existing kingdoms lay out and express word for word as before (fingerprints taken at e526a62f). phylu has the Dv arms;
a layout with a Dv lane does not reach the device before P1. The commit lists the 34 `is_constant(` calls.

**G3 status (2026-11-05): done.** `src/typed_environment.rs`: `Decoder::new(tables, contract)` (contract symbol `s` is
the tables' symbol `s`), `decode(entry, tree)` gives every node its `Environment`, `Choice` (Dc index, variable by
`Binder` {depth, ordinal}, or the type's constant), properties, requirement `Outcome`s and `Substituted` (the work
bound, decided once); `in_scope` is the one order (the entry's parameters, then each binding outermost first);
`Decoding::resolved` the resolved tree, `failures` the rendered failures. `geneframe.rs`: `Relation` (`=`, `<=`, `<`,
`holds` on one spelling), `Subject` (`in<k>.<p>` or `env.trips`), the requirement's adapter, the `Adapter` and
`TripArgument` tables, a data type's constant, sorted sums and mins, `decode_order`, the comparable-kinds check,
`to_text` checking first, every check branch tested, a typed-depth golden. By the rule, a substituted `hist.*` renders
its first input of its own data type (in1 in the `i32` view), not the constant README 5.3's list names: the lead decides.

**F1b status (2026-11-05): done.** `src/futhark_view/master6.rs` (`Master6`, `ViewSpec6`, `View6`) computes every
master 6 view's three tables byte for byte as `view.py` does (7 of 7 committed views); `futhark_table.rs`'s
`Kingdom6::from_tables` reads them (with the master's adapters and the elements' pools) into the gene frame's tables,
checked (requirements with relation and adapter, bindings, trip arguments, constants), and the contract with a `var.T`
per `var` row. Every view lays out. At N = 192: k-means `f32`, 9 data types, head 3,057, tail 12,995, Dc 4,202, Dv
12,230, 32,484 words (design 7.1: about 31,000); the `i32` + `f32` view (`sig/kmeans_i32/m6/`, sniffed with `--elements
i32 --lets []i32`), 22 data types, 227 symbols, 122,447 words. A demanded data type with no leaf (`[][]i32`, built only
by `tabrows`) takes its constant, `zero@m.i32` (M0088), and no `var` row: no variable can be of it, so it has no Dv lane
(the F1b review's finding 3 took its dead 573-word lane away; it was 123,020 words; `view.py`, witness V6, README 9).
Master 5 unchanged.

**F2a status (2026-11-05): done.** `src/futhark_sig.rs`: all 15 reference programs as master 6 s-expressions
(`REFERENCES6`: reduce, scan, histogram and pathfinder reference and planted slowdown; k-means reference, planted
slowdown, textbook form and its planted slowdown; A7 reference, winner and planted slowdown), converted by hand. Binders
name their variables (`(tab@a.i32[i] n ...)`). `build6` sets each `var.T` word through G3's `index_of` in the leaf's
environment and checks it by decoding again. The master 5 renderings are stored in
`tests/fixtures/futhark_m6_references.txt` (`expected_text6`). Every tree writes back at the problem's program size,
expresses to itself and decodes with no failing requirement and no substitution. None needed an adapter.
`random_gene` no longer reads Dv words as symbols.

**F2 status (2026-11-05): done (`4ddf1fba`).** `src/futhark_table.rs`: `Kingdom6::program6` and `Renderer6` (pass 5),
the ADF rendering (`program6_with_adfs`, subtrees by path); `src/futhark.rs`: `FIT_HELPER`, `CLAMPIX_HELPER`. All 15
references render their master 5 texts byte for byte. Proof 1: 1,000 random genes a view pass `futhark check`, 1000 of
1000 for each of the seven views; 100 a view compile with `c` and run legally on two stored cases, N of N. Adapter-free
genes (the early guard, risk 6): reduce 97.6 %, scan 94.5 %, histogram 16.0 %, pathfinder 31.1 %, a7 3.3 %, k-means
7.8 %, k-means `i32` + `f32` 7.6 %; nodes with an adapter 0.4 % to 24.6 %. Two README adapter texts were rejected by
Futhark 0.28.1 and changed: `fit.pair`'s `ad_l` for two known lengths (an existential size, so two such `map2`s did not
match) now fits to the spelled `min`; a `?` state is bound as a size, `let [ad_m] (ad_s: [ad_m]f32) = ..`.

**F5 status (2026-11-05): done (`208e23e7`).** Master 6's e-graph for A7 and every view: `gen_tables.py` (`egraph6_texts`, witness
E1-E7, E7 "no rule names a binder") writes `sig/<problem>/m6/egraph.{datatype,rules,rows,sorts}`, one sort a data
type, one constructor a symbol, a `var` row's constructor on the resolved variable (depth, ordinal) and a second for
its constant reading; `src/futhark_view/egraph6.rs` computes the same texts byte for byte. A7's four matrix rules are
`mat` rows of `master/6/rules.tsv` (witness M9 checks them on matrices). `src/futhark_saturation/master6.rs`:
`Egraph6` (import, `place`: an extracted variable written as its index word in the destination's environment and
checked by decoding again, `variants`, `saturable`) and `FoldEgraph6`. `saturated_fold.rs` gained `saturable`,
`place`, `variables` (defaults keep every other kingdom as it was): a tree with a failing requirement is kept as
submitted, the fold's key is the resolved term, and a shared part's variables take one word naming their variable at
every use (`common_word`), else the part is inlined. The variants (3 iterations, 32 members, 48) of the eleven
references of reduce, scan, histogram, pathfinder and A7, and of the k-means reference, are master 5's texts in order
(k-means: 31 of 31; fixture `tests/fixtures/futhark_m6_kmeans_variants.txt`); all
15 references' variants write back and render their own text, none dropped; A7's reference finds the winner. `typed_chromosome::genic_cells` read a variable's index word as a symbol (a gap of the gene layout task, G2): fixed.
Code review (`docs/reviews/2026-11-05_section7_F5_code_review.md`) fixed: a refit step is taken only where the gene it
writes names the variables its extracted form names (a shared part is never put where its word names another
variable), a folded gene that does not express its form is a bug on every path, an individual the fold leaves as it
was keeps its words, and a part with no e-class is a bug. The timing of master 5's k-means variants was never recorded
with its conditions, so no figure is given.

**F3 status (2026-11-05): done.** `src/futhark_sig/interp6.rs`: master 6's interpreter, read from G3's decoding
(variables by binder, sizes and bounds from the properties, every adapter the renderer renders applied, a substitution
its substitute), with Futhark's checks (indices in range, `map2` lengths, every value's shape against its properties);
`Interp::new6` beside `Interp::new`, the same `eval`, `work`, `work_parts`, `static_cost` (phylu's callers build
unchanged). `work_parts` reads sizes from properties and charges every adapter in `Work::nodes` (a `fit` its target
cells, `clampix` 2), which phylu's `submit_work_of` reads. `kingdoms/futhark/master/6/costs.tsv` (generated, witness
C1-C6, README 10): per-row formulas over argument properties, the adapters charged; `futhark_cost::program_cost6`.
Measured: the 15 references' values and work equal master 5's on every stored case (k-means on its two smallest), A7's
static cost equal; 1,000 random genes a view run on the interpreter with no error (k-means genes over the per-submit
cap on the case, 41 and 46 of 1,000, on it cropped); 100 a view on `c` equal the interpreter bit for bit, 1,499 of
1,500 runs, the other (`kmeans_i32`) equal but for Futhark's `0 / x` fold (its simplifier folds a known zero divided by
anything to 0, where IEEE gives NaN, depending on fusion; counted apart); an adapter costs more than its clean form.

**Why.** Folding sizes and scope into the type made k-means 14,359 types (85 % depth-3 scope bodies), a gene of 27.6
million words at N = 192, a 34 s layout, a 149 s e-graph and a 193 s fold at 19.5 GB, and the `i32` + `f32` k-means
view (237,325 types) cannot load at all (65,536 type codes). Measured, `docs/kmeans_seed.md`, master 5.

### 7.1 What is built, by layer

| layer | repo, files | what |
|---|---|---|
| data model | fuller `src/geneframe.rs` | the six tables; a property expression (a size name, a literal, `?`, a sum kept symbolic, a property of an input); `Arity` derived from Argument rows; a kingdom with no property rows is today's. G3 adds the Requirement's relation (`=` or `≤`) and the check that its two sides are properties of the same kind |
| contract | fuller `src/kingdom_contract.rs` | relations built from Argument rows (in and out); a `var.T` terminal of arity 0 per data type that has variables, flagged by a new `is_variable` (never `is_constant`: `express` sends every constant to Dc, `typed_gep.rs:789`); the bounds give `T` an index-word array `Dv_T` of length `h_T + t_T`, one word a cell, a lane parallel to the head and tail |
| gene | fuller `src/typed_gep.rs` | `Domain::Dv`, `Frame::dv_off`/`dv_len`; `express` records the index word of each expressed `var.T` in `Tree::const_index` (read as a Dv word when the symbol `is_variable`, so the 629 `Tree` literals do not change); `alphabet`, `mutate`, `write_back`, `spans`, `columns` and the row operators cover Dv, every row operator moving a cell and its Dv word together |
| decoder | fuller, new `src/typed_environment.rs` (not `typed_decode`: phylu has `src/evolve/typed_decode.rs`, and the merge of 6.2 would put two in sibling crates) | passes 2 to 4 of the design, generic: the environment top-down (the root's from the entry's parameters, a child's from its parent's plus the Binding rows of its argument); `var.T` as the (w mod c)-th variable of `T`, `T`'s constant when c = 0; properties bottom-up; every Requirement checked, its failures listed per node; the work-bound substitution as an annotation; and the **resolved tree** (every variable by its binder's identity: the depth at which it was bound and its ordinal), the only form any consumer compares or hashes |
| Futhark tables | fuller `kingdoms/futhark/master/*`, `sniff.py`, `view.py`, `gen_tables.py`, `src/futhark_view.rs`, `src/futhark_table.rs` | master 6: data types and property types replace `types.tsv`'s classes; each row's arguments, rules, requirements and bindings replace its `scope` quantifiers; `frames.tsv` becomes Binding rows; a view is the data types and rows the signature reaches; `sniff.py` lists the size names in scope at the root |
| reference genes | fuller `src/futhark_sig.rs` (`reference()`, `kmeans_seed`, `KMEANS_TEXTBOOK`, the planted slowdowns), the four benchmarks' and A7's references | master 6 trees beside the master 5 sexps, each with the master 5 rendering as its expected text |
| render | fuller `src/futhark_table.rs`, `src/futhark.rs` | pass 5: the plain spelling, or the adapter of design section 5 with its operands bound by `let` under reserved names; the prelude holds `fit` and `clampix` |
| interpreter, cost, work | fuller `src/futhark_sig.rs`, `src/futhark_cost.rs`; phylu `src/futhark_eval/evaluator.rs` (`submit_work_of`) | variables by the environment, not by the duals' contexts (`binder_slot`, `Ctx::of_dual`, line 1312 on); the adapters interpreted; `work_parts` and the cost read sizes from properties and charge every adapter; `eval(tree, inputs)` keeps its signature and decodes inside |
| reader | fuller `src/futhark_reader.rs` | a name becomes `var.T` with the index of that variable in its environment; float-literal patterns; `let` read as the `let` row |
| e-graph | fuller `gen_tables.py`'s e-graph part, `kingdoms/futhark/egraph.*`, `src/futhark_saturation.rs`, `src/futhark_view.rs` (the views' e-graph), `src/futhark_sig.rs` (`Bridge`, `import`, `to_tree`, `FoldEgraph`), `src/saturated_fold.rs` | one sort a data type; a variable constructor on the **resolved** variable; the bridge writes an extracted variable back as the destination environment's index word; saturation only on trees with no failing requirement |
| evolution | phylu `src/evolve/typed_layout.rs`, `typed_init.rs`/`.wgsl`, `walk.wgsl`, `typed_vary.rs`/`.wgsl`, `marriage.rs`, `futhark_typed.rs`, `src/futhark_eval/evaluator/start.rs`, the replay tests | the frame table gains the Dv offset and length (`FRAME_WORDS` 10 to 12), a `CELL_DV` cell kind; the init draws index words; the device walk reports the index words it reads; the point mutation edits an index word as it edits a constant; marriage and the gene keys on the resolved tree; the start population enters through the reader and `write_back`; the replay from the ledgers' texts |

The GPU runs pass 1 only (design section 4): it draws, varies and selects symbols by data type, as now. Passes 2 to 5
run on the host. An index word is any `u32` below 2^16: the decoder reads it modulo the variables in scope, so a draw
or an edit needs no knowledge of the environment.

**Sizes are chosen, not computed (lead decision 2026-11-05, design section 5's closing paragraph).** A function that
takes a size (`tabulate`, `replicate`, `hist`'s length, a loop's trip count, `tabrows`' row length) takes it from the
data type `size`, rendered as Futhark's `i64`. Its only terminals are `var.size`, whose index word chooses among the
size names in scope (the entry's `n`, `k`, `d`, as `sniff.py` lists them and master 5 spells them, and the sizes a
binder introduces), and the kingdom's size literals if it has any. No function returns a `size`; a `size` may be used
as an `i64` value (a conversion row), never the reverse. So every length and bound is a size name, a literal or a
symbolic expression of them (`i64.min n k`), `work_parts` and phylu's per-submit cap read a known bound from the case's
sizes, and the `i64.max 1` adapter of the earlier draft is not needed (7.5a item 2). This answers the adversarial C1
with its option (a), as a data type rather than the function cell's own word, so the gene keeps the design's
Argument rows. `?` remains only where a row's result length is known at run time alone; the `min` of two sizes is
symbolic, not `?`.

**Two choices the design leaves to the plan, taken here (the lead writes them into master 6's README in F0):**
1. **The work bound.** Master 5's `shallow` condition bounds the trips a program runs (witnesses M5, S6), not the
   number of types. Futhark itself has no depth limit, and the head length bounds the nodes but not their nesting. So
   the bound stays, as a property and a requirement: the environment (pass 2) carries `trips`, the binders above the
   node (a `let` adds none), and every heavy row of S6's list, and every adapter that is one (`fit` is a `tabulate`),
   requires `trips < depth`. A failing heavy node renders as its first input of its own data type when it has one
   (`map2.*`, `scan.*`, `loop`'s state), and as its data type's constant when it has none (`sum`, `tabulate`): the
   substitution adapter of design section 8 (Fable's amendment). The substitution is an annotation of G3, read by the
   renderer, the interpreter, `work_parts`, the reader and the e-graph; none decides it again. Witness M5 is carried
   into master 6 against the new columns; its `top` clause (`gen_tables.py:2245`, "each let class once") is rewritten,
   not carried. This applies the design's property and requirement model, as the design itself treats SR's depth
   (design section 8); the design document is not changed. Andrew agrees to this in F0.
2. **The `let` at the top.** A `let` anywhere is legal Futhark, so the `top` condition is dropped; `let` is a row
   whose argument 1 binds its value (a Binding row).

Every data type has a constant (`0`, `0f32`, the zero tuple, the constant row function, and for an array a one-element
array of zero whose length property is 1, to which `fit` applies; `size` has no constant, since every entry has a
size in scope). `{0.len}`-style placeholders let a spelling read a property (`next`: `i64.min ({0.bound} - 1) ({0} +
1)`); a placeholder whose property is `?` binds its operand with `let` first and reads `length` of the name, never of
the subtree. The design's first options stand: `fit` pads with the element's zero and truncates, `clampix` clamps,
`map2` takes the shorter length; SR keeps `T1`/`T2` until 7.7.

**Literal indices (adversarial H2).** An index's bound may be "valid for every length of at least 1" (the literal
`0i64`, and `{size} - 1`), and an index Requirement is `bound ≤ len`, not `bound = len`. So `(centres)[(0i64)]` in the
k-means reference and `zero`/`last`/`next` in pathfinder's render unchanged, with no `clampix`. F0 writes it; G3
implements the relation.

**Adapters Futhark accepts, and text that does not grow (adversarial H1, Fable (c)).** Every adapter binds its
operands with `let` under names reserved for adapters (never a gene binder's name), so `map2`'s adapter is `let a = xs
let b = ys let l = i64.min (length a) (length b) in map2 f (fit l z a) (fit l z b)` and nested adapters grow the text
linearly, not by doubling. A loop state or a `map2` operand whose length is `?` has that length bound to a name before
Futhark sees it (`let s0 = init let m = length s0 in loop (s: [m]t) = fit m z s0 ...`). An array-valued lambda body
(`tabrows`, M0063; the value rows of `histrows`) carries a Requirement that its length is a size from outside the
lambda; when it fails, the body is wrapped in `fit` to the row's own `size` argument, so no size escapes the lambda.
An inner dimension (A7's `matmul` inner sizes, `map2_rows`, `histrows`' value columns) is fitted per dimension,
`map (fit l z)`: design section 5's "`fit` on each dimension", applied below the root.

### 7.2 Tasks

Each task is one agent, commits only its own paths, and ends with `RUSTFLAGS="-D warnings" cargo test` and clippy
clean in its own repository **and** `cargo test --no-run` in phylu (phylu builds against fuller by path); a fuller
task that changes a type phylu matches on carries the phylu arms in the same task. Each task converts the committed
fixtures of its own modules from master 5 names (adversarial M4: 48 files); D1 only deletes. Master 6 rows and views
land beside master 5's; the master 5 path stays loadable until task D1. Sizes are agent days, judged from the code
each touches (line counts at `f66891be` and phylu `95bf4317`), and taken from the adversarial review where it judged
them.

| task | repo | size | what | files | CPU tests | after |
|---|---|---|---|---|---|---|
| **G1** | fuller | 1 | the six tables and property expressions; Arity derived from Argument; the table check (positions dense, one output, rules and requirements name properties that exist, bindings name arguments that exist), each defect "our software has a bug: ..." | `src/geneframe.rs` (906 lines) | the SR, typed-depth and regex tables derive exactly today's Arity, row for row; a table with no property rows checks as today | none |
| **G2** | fuller + phylu arms | 3.5 | the contract and the gene: `var.T` with `is_variable`, `Dv_T`, the bounds, `express`, `walk_expressed`, `alphabet`, `mutate`, `write_back`, the row operators over Dv; every one of the 34 `is_constant(` calls in 18 files of both repositories audited and listed in the commit; phylu's exhaustive `Domain` matches (`rings/tests.rs:140`) given the Dv arm | `src/kingdom_contract.rs` (822), `src/typed_gep.rs` (1,331); the phylu arms | every existing `typed_gep` and contract test unchanged, and a kingdom with no `var.T` has today's width word for word; a tree with variables writes back and expresses to itself; a `var.T` takes a Dv word and no Dc word, and the Dc pool's length is unchanged; an edit of a Dv cell changes no symbol; after inversion, IS/RIS transposition and one- and two-point crossover every `var.T` cell keeps its word | G1 |
| **G3** | fuller | 2.5 | the decoder's passes 2 to 4, kingdom-free; the environment's order is one public function, the only one F2, F3, F4 and F5 use; the resolved tree (variables by binder identity) as a public function; the work-bound substitution as an annotation; the Requirement's relation `=`/`≤` and the same-kind check, added to `geneframe.rs` after G1's commit | new `src/typed_environment.rs`; `src/geneframe.rs` (the relation column) | the design's section 6 histogram as a test: the reference holds every requirement, the `tabulate` size's index word changed to choose `k` fails exactly two; `0i64` and `{size} - 1` hold `bound ≤ len` for every length; the same gene decodes twice to the same annotations, and the environment is the same across two processes (no HashMap order; `BTreeMap`); a kingdom with no property rows has no failures; two subtrees with equal index words under different environments resolve to different trees, and two with different words naming the same variable resolve equal; the WGSL contract derives today's Arity | G2 |
| **F0** | lead | 0.5 | the master 6 format: columns, the `size` data type and its terminals, the property placeholders in spellings, the `≤` bound, the reserved adapter names, the two choices above; five rows converted by hand as the example | `kingdoms/futhark/master/README` | none | G1 |
| **F1a** | fuller | 2 | master 6 rows and Binding rows; `sniff.py` writes the root environment with its properties and its size names; `view.py`; `gen_tables.py`'s witnesses (every data type but `size` has a constant; every requirement has an adapter; every rule names existing properties; no row returns `size`; M5 against the new columns, its `top` clause rewritten) | `kingdoms/futhark/master/*`, `sniff.py`, `view.py`, `gen_tables.py` (table part) | `gen_tables.py --check` | F0 |
| **F1b** | fuller | 1 | `futhark_view.rs` computes data types; `from_tables` reads the new columns | `src/futhark_view.rs` (table part), `src/futhark_table.rs` (loader) | `view.py` and `futhark_view.rs` byte for byte; the `i32` + `f32` k-means view loads; the pixel family's tests unchanged | F1a, G2 |
| **F2a** | fuller | 1.5 | the reference programs as master 6 trees: the four benchmarks', A7's reference, winner and planted slowdown, k-means's reference, planted slowdown and textbook form, converted from their master 5 sexps by hand (the reader reads 0 k-means programs today, so it cannot do this) | `src/futhark_sig.rs` (`reference()`, `kmeans_seed`, `KMEANS_TEXTBOOK`, the slowdowns), a fixture of the master 5 renderings | each tree expresses, writes back to itself, and decodes (G3) with no failing requirement and no work-bound substitution; its expected text is its master 5 rendering, stored for F2 | F1b, G3 |
| **F2** | fuller | 3.5 | pass 5: plain spelling or adapter, operands bound by `let` under reserved names; the prelude; the lambda-body requirement and its `fit`; `?`-length loop states and `map2` operands bound to a name; `map (fit l z)` for inner dimensions; binder names reproduce today's `var_name` (`i_n`, `s3_f32`, `s_a_n`) from data type, properties and ordinal, so reference texts are unchanged; a prelude holds only the adapters its program uses, so a program with no failing requirement renders exactly as today; the ADF rendering | `src/futhark_table.rs` (515), `src/futhark.rs` | every F2a tree renders its master 5 text byte for byte, `0i64` indices included; each row of the design's section 5 table as a fixture `futhark check` accepts, inside a `tabulate` lambda, inside a loop body and nested three deep; a loop of `?` state length and a `map2` over two `?`-length operands accepted; ten nested `map2` adapters give text linear in the depth; 1,000 random genes a problem pass `futhark check`, N of N, and the share of genes with no adapter and of nodes with one is printed per problem (the early guard, 7.5 risk 6); nested binders of the same data type and properties render distinct names, and the interpreter and the `c` backend agree on them; the rendered text is the same across two processes | F2a |
| **F5** | fuller | 3 | the generated e-graph a sort a data type, for A7 **and** the views (histogram, pathfinder, k-means); the variable constructor on the resolved variable; the bridge, writing an extracted variable back as the destination environment's index word; saturation only on trees with no failing requirement; the fold's key the resolved tree | `gen_tables.py`'s e-graph part, `kingdoms/futhark/egraph.*`, `src/futhark_saturation.rs` (656), `src/futhark_view.rs` (e-graph part), `src/futhark_sig.rs` (`Bridge`, `import`, `to_tree`, `FoldEgraph`), `src/saturated_fold.rs` | every extracted candidate writes back and renders its own text; the k-means reference's variants are today's texts; the extracted variable is in scope where written and is the same resolved variable (a failure is a bug); a shared part used under two different environments evaluates bit for bit as the unshared program; a tree with a failing requirement is not saturated | F2 |
| **F3** | fuller + phylu callers | 4 | the interpreter's `Op`s and variable slots from the environment (`class_dim(out)`, `env.sizes[d]` replaced by the properties); `fit` and `clampix` interpreted; `work_parts` reads sizes from properties and charges every adapter, and phylu's `submit_work_of` reads it; `costs.tsv` becomes per-row formulas over argument properties with the adapters' costs (`fit` the traffic of its target length, `clampix` two compares), regenerated through `gen_tables.py` with its witness; `eval(tree, inputs)` keeps its signature, so phylu's 14 `Interp::new` callers build unchanged | `src/futhark_sig.rs` (2,729, the interpreter part), `src/futhark_cost.rs` (520), `costs.tsv`; phylu `src/futhark_eval/evaluator.rs` (`submit_work_of`) | all 1,000 genes a problem also run on the interpreter on one stored case with no run-time error (the interpreter checks bounds and sizes as Futhark does), and 100 of them compile on the `c` backend and run, N of N, equal to the interpreter bit for bit; every reference program's value and static cost as today; a gene with an adapter costs more than the same gene without; the per-submit cap of every random gene is computed from the case's sizes, adapters counted | F5 |
| **F4** | fuller | 3 | first, the lead moves the untracked `src/futhark_reader/kinds.rs` out of `src/` (to F4's scratch, not `git stash`) if Andrew agrees F4 owns it (7.5a item 3); F4 reads it on day one and wires it or writes its own; then the reader emits `var.T` and its index, reads float-literal patterns (`sel@f32:c`) and reads `let` as the `let` row | `src/futhark_reader.rs` (1,565), `kinds.rs` if given | `read(render(t)) = t` for every F2a tree, on canonical trees (work-bound substitutions removed, index words reduced mod c); the pool's 842 programs read back with the same values on every stored case; the k-means pool reads (`pool.fut_read_by_reader` from 0 to all) | F2 |
| **P1** | phylu | 1.5 | the layout's Dv words and `CELL_DV`; the init draw of index words, tested on G2's test kingdom (P1 after G2; the `futhark_typed.rs` handover is in P3); the frame-table readers at the new stride (`walk.wgsl`, `typed_init.wgsl`, `typed_vary.wgsl`, `typed_decode.wgsl`; host twins `typed_walk.rs`, `typed_decode.rs`, `typed_write_back.rs`) | `typed_layout.rs` (745), `typed_init.rs`/`.wgsl`, the readers named | the host twin's draw expresses through G2's `express` on 1,000 genes of G2's test kingdom; the layout of a kingdom without `var.T` is today's word for word | G2 |
| **P2** | phylu | 3 | the walk reports the index words it reads; the point mutation of index words; `LeafTable::build_point` and the same-signature classes take `var.T`; crossover, inversion and transposition move a cell and its Dv word together; marriage's `subtree_hashes` and its identical-pair check, and the evaluator's gene keys from `walk_expressed`, on G3's resolved tree | `walk.wgsl` (234), `typed_vary.rs` (2,237), `typed_vary.wgsl` (1,155), `src/evolve/marriage.rs`, `walk_expressed`'s callers, `src/futhark_eval/rings.rs` (a Dv ring), `src/futhark_eval/operators.rs`, their tests (`rings/tests.rs`, `measure/tests_kmeans_seed.rs`) | the host twins: an edited index word changes only that word; after every row operator every `var.T` cell keeps its word; the ring walk edits index words; two genes differing only in an index word that names another variable get different hashes and keys, and two that name the same variable through different words get the same | P1, G3 |
| **P3** | phylu | 1 | the start population: the reference program through the reader and `write_back`; its variants through F5; `futhark_typed.rs` hands over the new contract | `src/futhark_eval/evaluator/start.rs` (672), `fresh`, `futhark_typed.rs` (1,499) | the start populations' reference and variant texts are today's (P regime, every problem), k-means included | F4, F5, P2 |
| **P4** | phylu | 1 | proof 2's replay: each frozen ledger's sexp (master 5 row names) rendered to Futhark by the master 5 renderer (loadable until D1), read by the master 6 reader, rendered again and scored; generation 0 from the ledger's texts, not from `GeneRng::new(1)`; the replay scripts committed (today in `/private/tmp/claude-501/fbench/plan5a/`) | `src/futhark_eval/evaluator/tests.rs` (`generation_zero`, `gene_of`), the scripts under `scripts/` | the master 6 rendering of every ledger row equals its master 5 rendering, text for text, and the quality and speed inputs equal the ledger's (`histogram_r_seed1_ledger.tsv`, `kmeans_p_seed1_ledger.tsv`) | P3, F3 |
| **L1** | lead | 1 | both repositories' full suites, clippy clean, the code review of each commit; the measured gene size (7.3 item 4); then the GPU checks (end of 7.2) | none | as named; the gated GPU tests of symbolic regression, regex and WGSL, since the frame table's stride changed | P4 |
| **D1** | fuller + phylu | 2 | the deletions of 7.4, master 5 included; `Ty::MAX_FUTHARK_DUALS` and the test that the `i32` + `f32` view does not load turned into the test that it loads; whatever master 5 fixture a task missed converted (each one a bug in that task, named in the commit) | the files of 7.4 | the pixel family's and the deprecated 1-D kingdom's tests unchanged; both full suites | L1 |

About 35 agent days, the adversarial review's estimate: its eight re-judged sizes are taken, G3 gains 0.5 (the
resolved tree and the `≤` relation) and P4 is the replay task it left unsized. The critical path G1, G2, G3, F2a, F2,
F5, F3, P4, L1, D1 is about 23 days, not the adversarial's 17: F5 needs F2's renderer for its own test, and F5 and F3
both edit `futhark_sig.rs` and `gen_tables.py`, so they run in sequence (F3 after F5). F1a and F1b finish by day 5.5,
P1 by day 6 and P2 by day 10 (P2 waits for G3 at day 7), F4 by day 15, P3 by day 16, all off the path. Same-file overlaps
kept apart: F1b, F5 on `futhark_view.rs` in sequence; F2a, F5, F3 on `futhark_sig.rs` in sequence; F1b and F2 on
`futhark_table.rs` in sequence; G1 then G3 on `geneframe.rs`.

```
G1 ─> G2 ─> G3 ─┬─> F2a ─> F2 ─┬─> F5 ─> F3 ──────────┐
G1 ─> F0 ─> F1a ─> F1b ─┘      └─> F4 ─┐               │
G2 ─> P1 ─> P2 (and G3) ───────────────┴─> P3 (and F5) ┴─> P4 ─> L1 ─> D1 ─> GPU checks ─> R0 ─> 7.7 (only if Futhark works)
```

**The lead's GPU checks**, one at a time: P1 and P2 on the device (the init draw and the walk on 20,000 random genes a
problem against the host twins); the replay checks; R0. Each GPU check runs a script the task's agent wrote; the lead
runs it and reads it.

### 7.3 The proof that it works

1. **Every random gene is legal.** For each Futhark problem (reduce, scan, histogram, pathfinder, a7, k-means `f32`,
   k-means `i32` + `f32`): 1,000 random genes express without a defect and every rendered program is accepted by
   `futhark check`, N of N (F2). All 1,000 genes a problem also run on F3's interpreter on one stored case with no
   run-time error (the interpreter checks bounds and sizes as Futhark does), and 100 of them compile on the `c` backend
   and run, N of N, equal to the interpreter bit for bit (F3). The share of genes with no adapter and the share of
   nodes with one are printed per problem at F2, against master 5's 100 % adapter-free. An adapter's failure is our
   bug.
2. **Reference programs are unchanged.** Every reference program (the four benchmarks', A7's reference, winner and
   planted slowdown, k-means's reference, planted slowdown and textbook form) is a master 6 tree (F2a), renders to its
   published text byte for byte (F2), and scores bit for bit as before. The replay (P4) compares **rendered Futhark
   text**: each frozen ledger row's master 5 sexp is rendered by the master 5 renderer, read by the master 6 reader,
   rendered again, and must give the same text and the same quality and speed inputs as the ledger
   (`histogram_r_seed1_ledger.tsv`, `kmeans_p_seed1_ledger.tsv`). Chosen over replaying through the frozen binary
   because master 5 stays loadable until D1 and the comparison then runs in `cargo test`, not in a second build;
   generation 0 comes from the ledger's texts, since a redraw under master 6's layout gives other genes.
3. **Host and device decoders agree.** On 20,000 random genes a problem, the device walk and the host's `express`
   give the same expressed cells and the same index words, word for word; the same random seed draws the same
   population on both.
4. **The gene size, measured** against the design's 7.1 table, reported as measured, expected, % difference, with the
   positions a column (fewer types put more symbols in a column; `triple`, `psel`, `sel` take 3):

   | view | types, master 5 (measured) | data types (design, plus `size`) | words at N = 192, master 5 | words expected (design 7.1) |
   |---|---:|---:|---:|---:|
   | reduce, scan | 28 | 4 | L1 measures | first build |
   | histogram | 162 | 4 | L1 measures | first build |
   | pathfinder | 184 | 5 | L1 measures | first build |
   | a7 | 11 | 7 | L1 measures | first build |
   | k-means, `f32` | 14,359 | about 9 | 27,563,026 | about 31,000 |
   | k-means, `i32` + `f32` | 237,325 (does not load) | about 15 | none | first build |

   The layout's, the e-graph's and the fold's seconds on k-means are measured beside them (today 34 s, 149 s, 193 s).
5. **R0 rerun.** Pathfinder regime P generation 0 on the new kingdom: the reference program and its variants give the
   same texts and the same inputs bit for bit as R0's ledger; the mutants differ by construction and are reported.
   Then the regressions of 5a.4 in their order, k-means first; the k-means regression reports, per generation, the
   share of the population with an adapter beside its best and median, so the plot shows whether selection removes
   adapters. Proof 5 is the gate for 7.7.

### 7.4 What is deleted

The scope and size in the type: the `C` quantifier and the `vars(..)`, `D:dims`, `rank1`, `rows(..)`, `pairs`,
`swap`, `triples` and `class(..)` expansions; `Ctx` as a type component, `Binder::contexts`, `Ctx::push`, the
`:c/{..}@d` names and sorts, `Ctx::of_dual` and the duals' context suffix; the type names with sizes (`a.n`, `ix.n`,
`mat.n.m`, `a.i32.n`); `frames.tsv` as a type generator; the `shallow` and `top` conditions as view filters;
`Ty::MAX_FUTHARK_DUALS` as a binding limit; the test that the `i32` + `f32` view does not load (it becomes a test that
it does); master 5's rows and its renderer, after P4's replay has run. The pixel family and the deprecated 1-D kingdom
keep their own types until their own step.

### 7.5 Risks

1. **An index word can name another variable after an edit above it** (a binder added or removed shifts the
   environment, and with `w mod c` every variable below can retarget at once). The program stays legal. The modulo
   reading stays (Andrew's design); Fable's clamp reading is not taken and no retarget rate is measured now (7.8).
2. **Binder names.** If today's names cannot be reproduced from data type and properties, the reference texts change
   and proof 2 fails; F2 finds this on its first day and the lead decides before F5. Fable's advice is recorded: the
   texts are a regression anchor, not a contract; if they cannot be kept, the reference texts are re-published and the
   quality caches re-keyed rather than the design bent to the old names.
3. **Index words are relative to their scope.** The same word names different variables in different scopes, and
   different words can name the same variable. Every consumer that compares or hashes subtrees (the e-graph, the
   saturated fold, marriage's subtree hashes, the walk's gene keys, caching) works on G3's resolved tree, never on raw
   index words (F5, P2), and the bridge writes a variable back from the destination's environment. A rule never moves
   a subtree under another binder (the `gen_tables.py` binder witness), and saturation runs only on trees with no
   failing requirement, since a rule is an identity under `fit` only where no adapter sits between its sides.
4. **Adapters cost run time** where requirements fail; a gene whose properties all hold costs nothing extra. The cost
   charges every adapter (F3), so an elite with a needless `fit` loses to its own clean form.
5. **Old genes.** Checkpoints and ledgers of master 5 genes do not load; a checkpoint is converted through its texts
   and the reader (as P4 does for the ledgers), an old result is reproduced at its own commit.
6. **The search gets worse while every proof passes** (Fable's biggest risk): the smaller gene buys a search space
   most of which may render adapters. The guard is early: the adapter-free share of random genes at F2 (day 12, not
   at R0), the cost charge on adapters, and the adapter share per generation in the k-means regression.
7. **A7's experiment changes** (adversarial M3): under data types every matrix is `[][]f32`, so a random A7 gene can
   be a program padded with `fit`, a different search space from the one the shape-typed paper and
   `docs/measurements/2026-10-27_futhark_shapes_A7_learning.md` report. Those results are reproduced at their own
   commit, not by master 6.

### 7.5a Decisions for Andrew

1. **The work bound** (7.1 choice 1): `trips < depth` on the heavy rows and the adapters that are heavy, with the
   substitution by the node's own-type input where it has one (Fable's amendment), else its constant. A plan-level
   rule the design text does not state; the design is unchanged, and Andrew confirms it in F0. Fable's review says yes.
2. **The computed-size adapter: superseded.** The earlier draft rendered a gene-computed `?` size as `i64.max 1
   (...)`. With sizes chosen from the `size` data type (7.1, lead decision 2026-11-05), no gene computes a size, so
   there is no `?` size to clamp, and the adversarial review showed the adapter would also have defeated the GPU work
   cap (a size taken from a data element can be 2^31). Nothing to confirm.
3. **`src/futhark_reader/kinds.rs`**, untracked, owner unknown: keep, commit or delete is still Andrew's. Fable's facts
   for the decision: it is not declared as a module (`futhark_reader.rs` has `mod tests;` only), so it compiles into
   nothing and the tree is green with or without it; its header describes a kind inference over the normalised Futhark
   expression (i32, i64, f32, bool, arrays, tuples, functions, the prelude's signatures, Futhark's integer-literal
   default), which is what a reader emitting `var.T` with the right data type for every name and literal needs; its
   i32-to-f32 rewrites are made unnecessary by the `i32` + `f32` view. Fable recommends giving it to F4; if Andrew
   agrees, F4 does as its row says, otherwise F4 does not touch it.

### 7.6 Kept from the earlier draft

The view is still a query over the master, written by `sniff.py` from the Futhark source and never by hand; the table
is still the trust boundary, checked offline by `gen_tables.py`'s witnesses with `--check` as the default; the
reference programs keep their published texts, so the quality caches and the scored-text sets stay valid.

### 7.7 Later phase: the other kingdoms (only if Futhark works)

After 7.3 passes and R0 has been read: migrate each kingdom into the same structures, one at a time, each with its own
legality check N of N and its own regression against today: symbolic regression (the depth ladder `T1`/`T2` becomes
one data type with a depth property and a requirement, its adapter a substitution); TSR (the order and the window as
properties); regex (no property rows, unchanged in behaviour); WGSL (vector widths, array lengths, the address space
and the workgroup size as properties, `let`/`var`/loop variables and the invocation ids as bindings). Not detailed
here: each gets its own plan when its turn comes. Also later (Andrew's note of 2026-11-05, recorded): tables over a
general set of Futhark types and functions, the declaration naming only the root type and the leaf inputs.

### 7.8 The reviews' findings and where each went

The lead's decisions (2026-11-05): (1) sizes are chosen from a `size` data type, not computed; (2) every consumer that
compares or hashes subtrees works on the resolved tree; (3) `var.T` words in their own array Dv, flagged
`is_variable`; (4) the modulo reading stays, Fable's clamp not taken, no retarget measurement now; (5) `kinds.rs`
stays Andrew's decision, with Fable's facts.

| finding | review | disposition |
|---|---|---|
| C1 computed size defeats the work cap | adversarial | decision 1 (7.1 "Sizes are chosen"); `work_parts`, `submit_work_of` in F3; adapters charged in the work bound |
| C2 index words compared across scopes | adversarial | decision 2: resolved tree in G3; consumers in F5, P2; tests there |
| C3 `var.T` sent to Dc | adversarial | decision 3: `is_variable`, Dv; 34-call audit in G2 |
| H1 adapters Futhark rejects, text doubling | adversarial; Fable (c) | 7.1 "Adapters Futhark accepts"; F2 fixtures |
| H2 literal indices change references | adversarial | 7.1 "Literal indices"; F0, G3, F2 |
| H3 proof 2 cannot match | adversarial | P4, rendered-text replay, chosen in 7.3 item 2 |
| H4 no task converts references; reader reads 0 k-means | adversarial | F2a (1.5); F4 to 3 days, float literals and `let` |
| H5 F5 misses the views' e-graph | adversarial | F5 files; F5 after F2 and before F3 |
| H6 interpreter and cost read sizes from the type; phylu callers | adversarial | F3 to 4 days; `eval` signature kept; per-row cost formulas |
| M1 G2 breaks phylu's build | adversarial | G2 carries the phylu arms; word in `const_index`; 3.5 days |
| M2 Dv separates a word from its cell | adversarial | Dv a parallel lane; G2 and P2 tests |
| M3 A7's experiment changes | adversarial | risk 7; `map (fit l z)` in F2 |
| M4 fixtures belong to no task | adversarial | each task converts its own; D1 2 days |
| M5 the substitution decided in four places | adversarial | a G3 annotation; F4's canonical round trip |
| L1 replay scripts in `/private/tmp` | adversarial | committed by P4 |
| L2 `mod` wraps a clamped `?` size | adversarial | does not arise: sizes are chosen, never clamped |
| L3 M5's `top` clause | adversarial | rewritten in F1a |
| (a) the cost charges adapters | Fable | F3, risk 4 |
| (b) saturation only on adapter-free trees | Fable | F5, risk 3 |
| (d) clamp over modulo | Fable | not taken (decision 4) |
| work bound: substitute the own-type input | Fable | 7.1 choice 1, 7.5a item 1 |
| `i64.max 1` into the design | Fable | superseded by decision 1 (7.5a item 2) |
| `kinds.rs` to F4 | Fable | recorded for Andrew (7.5a item 3) |
| biggest risk: search worse while proofs pass | Fable | risk 6; adapter share at F2 and per generation (proofs 1, 5) |
| binder names: re-publish rather than bend | Fable | risk 2 |
| index words below 16 | Fable | not taken: cosmetic, the modulo reading is the same |
| Requirement sides of the same kind; WGSL in the arity test | Fable | G3 (after G1, which is unchanged) |
| `n_max` per column | Fable | proof 4 reports the positions a column |
| sizes about 35 days, path 17 | adversarial | 35 taken; path re-judged 23 (7.2) |

## 8. Typed Symbolic Regression (TSR) on the new data model (Andrew, 2026-11-05)

TSR is Typed Symbolic Regression, Andrew's name (`kingdoms/tsr/README.md`). This section is the TSR plan that 7.7
promised; it uses only G1 to G3, which are built, and touches no Futhark file.

**Objective.** TSR's two measured limits on the 133 SRBench laws, stream order at most 3 and at most 2 transcendentals
on a path, become **properties** of one data type `F`, with **requirements** on them, instead of 34 types.
**Approach.** The gene is the one-type SR gene; the decoder derives each node's depth and order bottom-up and, where a
requirement fails, renders the node as its own first input. **What is unique.** The order limit, which lost as a
decode-time rule (84.69 % against 85.61 %, `docs/PLAN_tsr_typed_gep.md` 1.2) and cost a gene 208 times wider as
types, is a property rule of four lines and costs no gene width. **Status.** Planned; nothing built. **Why.** Andrew:
"It finds laws, or it doesn't." The deciding run (8.7) counts laws found per problem, old TSR against new.

### 8.1 Stream order, as the code defines it

From `kingdoms/tsr/gen_tables.py` (`order`, `unary`, `binary`), the device's derivation (phylu
`src/evolve/typed_decode.wgsl` step 3, `N_BEST`/`N_TIES`/`N_ORDER`) and `stream_order` (phylu `src/evolve/order.rs:193`),
which agree on 4,335,881 exhaustive and 300,000 random trees (`PLAN_tsr_typed_gep.md` 2.6): it is Horton-Strahler
order over **flattened chains**. A leaf has order 1. A unary node has its argument's order. A sum (`add`, `sub`) or a
product (`mul`, `pdiv`) collects its **arms**: the subtrees under the chain of nodes of its own family whose root is
not of that family. Its order is the greatest arm order `b`, plus 1 when two or more arms share `b`. The **profile**
`(family, b, ties)` is what a parent needs, so order is local. Example, `m·v/√(1 − v²/c²)`: `v²/c²` is a product of
two arms of order 1, order 2; `1 − v²/c²` a sum with arms 1 and 2, order 2; the root adds `√(..)` (order 2) as a third
arm of the product `m·v·..`: arms 1, 1, 2, so order 2. `(a·b + c·d)·(a·c + b·d)` has two arms of order 3: order 4,
which no law has. **Windows** (variant `order_depth_windows`): `sin cos tan tanh pow2 pow3 pinv` take an argument of
order at most 2.

### 8.2 The design in the new model

| table | TSR rows |
|---|---|
| DataType | `F` only; its constant (a fitted constant `?`) has depth 0, order 1, family none, arm 1, ties 1 |
| PropertyType on `F` | `depth`, `order`, `fam` (0 none, 1 sum, 2 product), `arm`, `ties`; kind `range` |
| Argument | the 18 operators of `SymbolTable::wide()`, `F` in and out: today's single arity, derived (G1) |
| PropertyRule | arithmetic: `depth = max(in0.depth, in1.depth)`; raising unary (sin cos tan tanh psqrt plog pexp abs pasin pacos): `depth = in0.depth + 1`; level-keeping unary (neg pow2 pow3 pinv): `depth = in0.depth`; every unary: `order = in0.order`, `fam = 0`, `arm = in0.order`, `ties = 1`; add, sub: `(fam, arm, ties, order) = strahler.sum(in0, in1)`; mul, pdiv: `strahler.product(in0, in1)` |
| Requirement | raising unary: `in0.depth <= 1 -> subst` (no `T3`); binary: `out.order <= 3 -> subst`; windows variant: the seven windowed unaries `in0.order <= 2 -> subst` |
| Binding | none. Inputs stay one terminal per data column, as in `single`, so the gene is `single`'s word for word |

Three additions to `geneframe.rs`, each with its table check: `PropertyExpr::Max` (the mirror of `Min`); the
`strahler.sum` / `strahler.product` merge of 8.1, defined on literal profiles only (anything else is `?`); and
`Subject::Output` (a requirement on the node's own derived property), with `subst` allowed on an input or output
subject in a kingdom with no heavy adapter (today `subst` is allowed only on `env.trips`, `geneframe.rs:1657`).

**When a requirement fails: the node renders as its first input (`subst`), never as a constant.** Reasons: (1) the
input has already passed every requirement below it, so the result is legal by induction, and a substituted node takes
its input's properties (`typed_environment.rs`, `Substituted`), so its parent sees a legal profile; (2) it keeps
expressed material the search built, where a constant erases the subtree (TSR_HARD1500 already ends 16 of 26 laws in
collapsed forms such as `(0*0)`, phylu `logs/TSR_HARD1500/sweep.log`; a constant substitute would push further that
way); (3) for a transcendental it is the small-argument identity (`sin x ≈ x`, `tanh x ≈ x`). Every TSR function has
an `F` input, so `subst.const` never arises. Depth is checked bottom-up, not by the top-down `env.trips` counter
(which would give the same predicate), because the device's derivation is already bottom-up (8.3).

### 8.3 Where requirements are checked: on the device, in the decoder

Today the type does the work at variation time (arm C: 34 duals, `typed_init`/`typed_vary` draw only legal rows), or
the decoder rejects (arm A: `EVOLVE_TYPED_DEPTH=2`, `gene_ok = 0`, the gene "rides along unscored").

| option | cost | decision |
|---|---|---|
| **on the device, in `typed_decode.wgsl` step 3** | the derivation of depth and Strahler order is already there, bottom-up; it adds a `fam` word a node (a substituted node's family is its input's, not its op's), one compare a requirement, and, only for a gene with a substitution, a second run of the level numbering of step 2 to drop the substituted node and its other arms; no host round trip, no new buffer, width unchanged | **taken** |
| on the host, before scoring | read back about 2,000 × 3 genes a generation, decode on the CPU, upload trees; breaks the device-resident chain; a CPU path beside the GPU one, which the house rules forbid | rejected |
| old type columns as a derived cache | arm C's layout: 20,980 words a gene, 62,940 an individual, 503.5 MB a population copy, 633 ms a generation (`PLAN_tsr_typed_gep.md` 3 and finding (e)) | rejected: it is the cost this model removes |

The device reads per-op rule codes (`max`, `max+1`, `copy`, `strahler.sum`, `strahler.product`) and the requirement
limits compiled from the TSR `KingdomTables` at load; a rule outside those codes is "our software has a bug". The host
twin (`typed_decode.rs`) and fuller's `typed_environment.rs` decoder must give the same node array.

### 8.4 Gene size, before and after (N = 17, `PLAN_tsr_typed_gep.md` section 3)

| kingdom | types | function rows | words a gene | an individual (3 genes) | a population copy (2,000 rows) |
|---|---:|---:|---:|---:|---:|
| order and depth (arm C), and with windows (F) | 34 | 8,264 / 8,094 | 20,980 | 62,940 | 503.5 MB |
| order only (B) | 12 | 1,148 | 3,656 | 10,968 | 87.7 MB |
| depth only (E) | 4 | 136 | 558 | 1,674 | 13.4 MB |
| **new TSR (properties)** | **1** | **18** | **101** | **303** | **2.4 MB** |

### 8.5 Changes, file by file

- **fuller** `src/geneframe.rs` (8.2's three additions); `src/typed_environment.rs` (the output subject; `subst` on
  property requirements); `kingdoms/tsr/gen_tables.py` (variant `props`, writing `kingdoms/tsr/props/` in master 6's
  column style: rows, rules, requirements) and `check_types.py` (its witness); `src/tsr_kingdom.rs` (`TsrVariant::Props`
  loads `KingdomTables`); `src/kingdom_contract.rs` (`tsr_relations` gives width 101).
- **phylu** `src/evolve/typed_decode.wgsl` and `typed_decode.rs` (substitution instead of `gene_failed`, rule codes
  from the tables, `fam` word); `src/evolve/tsr_typed.rs` (the `props` variant on the single layout);
  `typed_write_back.rs` (a host rewrite is offered only if it decodes with no substitution, replacing `relevel`);
  `engine.rs`, `card.rs` (`EVOLVE_SR_KINGDOM=props`, the run card names the requirements); `experiments/tsr_arms.json`
  (arm `P`); `experiments/tsr_decide20.txt` and `scripts/tsr_decide.sh` (8.7).

### 8.6 Tests

1. **The same programs as today's typed variants.** Over check_types.py's exhaustive trees (up to 9 nodes) and its
   300,000 random trees: a tree decodes with **no** substitution exactly when it has a typing in `order_depth`, and in
   `order_depth_windows` with the windows on. Python witness and Rust decoder both.
2. **Nothing illegal.** 100,000 random genes a variant: every decoded tree has `stream_order` ≤ 3 and `t_depth` ≤ 2
   by the engine's own oracles (`order.rs:193`, `engine.rs:2354`), and the windows hold.
3. **The GPU walk equals the host decoder.** 20,000 random genes a variant: node array, length, depth and hash word for
   word against `typed_decode.rs` and fuller's decoder (a gated GPU test, run by the lead).
4. **Nothing changes where nothing fails.** A gene that passes arm A's decode check gives today's node array.
5. **Every fuller feature the TSR runs use**, each with a test that its output decodes with no substitution or is not
   offered: fold (device fold and `saturated_fold`), constant snap and typed snap, denoise and simplify (`denoise_k`),
   the e-graph forms and physics candidates (`forms`, `physics` in the `chain` column), and share-equivalence (a
   shared definition carries its properties, so a reference cannot hide depth or order). Task P2 owns all five.
6. Determinism: the same gene decodes the same in two processes.

### 8.7 The benchmark, and the run that decides

**The result to beat.** The last measured TSR law rate on development seeds is phylu `logs/SR_OLD_NEW` (2026-10-20,
phylu `093aae96`, fuller `1e0c4667`; report `docs/reports/2026-10-20_sr_typed_gep.md`): TSR as run today, one type with
the depth-2 decode check, seeds 7001 to 7010, 1,000,000 counted evaluations a seed, early stop at log10 p ≤ −19 and
1 − R² ≤ 1e-6 (the value `scripts/law_runner.py` pins), the arm named "new": **380 of 400 seed-runs on the first 40
laws** (run1) and **101 of 600 on the next 60** (run2); 33 laws not run (run3 stopped at law 28 of 33, no table).

**Problem set (20), from those tables.** The 10 hardest TSR has solved (fewest seeds of 10 passing, run2):
feynman_I_10_7, feynman_I_34_1, feynman_II_13_23, feynman_III_15_12, strogatz_barmag2 (1 of 10 each);
feynman_I_26_2, feynman_II_10_9, feynman_II_6_15b, feynman_III_17_37, strogatz_glider1 (2 of 10 each): 15 of 100.
Ten it has not solved (0 of 10 in both arms, run2): feynman_I_15_10, feynman_II_13_34, feynman_I_12_11,
feynman_II_11_27, feynman_II_11_28, feynman_I_8_14, feynman_test_9, feynman_I_30_3, feynman_III_4_32,
feynman_II_35_21. Four of these are the Lorentz and Gaussian laws the order rule lost as a decode check.

**Andrew's requirements for the run.** (1) Initiated in phylu: `scripts/tsr_decide.sh` calls `sr_old_new.py` over
`law_runner.py` and phylu's engine, never a fuller loop. (2) Watched with `../phylu/target/release/evowatch OUT`
(each fit writes `OUT/stream.jsonl` through `fuller::telemetry`), with `tail -f OUT/sweep.log`. (3) Every fuller
feature of 8.6 item 5 on the new model, by task P2, before launch. (4) The 20 problems above. (5) Budget: 1,000,000
evaluations a seed, DEV_SEEDS 7001 to 7010, two arms: old TSR (`EVOLVE_TYPED_DEPTH=2`, single) and new TSR
(`EVOLVE_SR_KINGDOM=props`, order and depth), the new arm first on each law. Recorded wall time is 20 to 25 s a law
and arm for 10 seeds, so 20 × 2 × 22 s ≈ **15 minutes**; with the new decode up to twice as slow, at most 25 minutes;
the script's kill guard is 50 minutes. Both arms run at the law bar of 1e-10 (Andrew, 2026-10-08), and the counts at
1e-6 are reported beside them, since PROD1 saw an exact fit sit at 1.7e-10, above the bar's f32 floor.

| law | set | old TSR found / 10 | new TSR found / 10 | difference | median evaluations, old / new | genes with a substitution, new |
|---|---|---|---|---|---|---|

One row a problem, totals for each set. The run decides on the totals: new TSR is kept if it finds at least as many
laws (seed-runs at the bar) over the 20 as old TSR, paired by law and seed; any law of the unsolved ten is reported
first. Arm C (34 types) as a third column is outside the hour and is not run.

### 8.8 Tasks

| task | repo | size (days) | what | after |
|---|---|---:|---|---|
| T1 | fuller | 1 | `Max`, the Strahler merge, `Subject::Output`, `subst` on property requirements, their table checks | G3 |
| T2 | fuller | 1.5 | `gen_tables.py` variant `props`, `check_types.py` witness (8.6 item 1), `typed_environment.rs` output subject | T1 |
| T3 | fuller | 0.5 | `tsr_kingdom.rs` loader, contract width 101, phylu arms build | T2 |
| P1 | phylu | 2 | device and host decoders: rule codes, substitution, `fam`, renumbering; tests 2, 3, 4, 6 | T3 |
| P2 | phylu | 1.5 | host rewrites through the decoder: fold, snap, typed snap, denoise, e-graph forms, share; test 5 | T3 |
| P3 | phylu | 0.5 | card, arm `P`, problem list, `tsr_decide.sh`, the 1e-10 bar, stream for evowatch | P1, P2 |
| L1 | lead | 0.5 | both full suites, clippy, code review of each commit, the GPU test 3 | P3 |
| R1 | lead | 0.1 | the deciding run of 8.7, about 15 minutes, and its table | L1 |

About 7.6 agent days; the critical path T1, T2, T3, P1, P3, L1, R1 is 6.1. P1 and P2 run in parallel (different
files).

```
G3 ─> T1 ─> T2 ─> T3 ─┬─> P1 ─┬─> P3 ─> L1 ─> R1
                      └─> P2 ─┘
```
