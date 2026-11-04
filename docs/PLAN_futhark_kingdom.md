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
