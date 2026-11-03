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
