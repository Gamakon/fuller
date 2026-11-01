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
  - the function families.
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
families	hist,mod
literals	0,1,2,-1,10
dims	k,n		            (derived: order of first appearance, size inputs included)
```

A7's spec (`kingdoms/futhark/view.tsv`) is `shape matrix`, `roles n,m,p`, `root n,p`, `families matrix`: hand-written, since A7 has no entry source until step 4 makes it a view problem of `futhark_evolve`.

### D3. The view algorithm (one in Python, one in Rust; a test requires them equal)

1. **Filter**: the rows whose family the spec admits (`core` always for a signature view), `since <= master`, and the spec's element in `elems`.
2. **Bind**: derive the sets from the spec, exactly as `sig_signature` does today: `dims` (first appearance), `classes` (each input's class and its row classes, the output's, each dim's rank-1 class), `rank1`, `states` = the element then `rank1`, `free`, the inputs; the contexts from `frames.tsv` (`[()]`, then each context of depth `d` extended by every frame, `d < depth`). Expand the rows' blocks, substitute the variables, and form each type's dual (`class:context`, or the bare class when it is context-free). Name a row `name` as templated (`add@{OUT}` gives today's `add@i32:c`).
3. **Reachability**: the closure from the root (`output` in context `c`): a type is kept when a kept row's output is kept and it is that row's argument; rows whose output is not kept go. This is today's `sig_rows` closure, and it is what makes the gene layout per-type heads only for reachable types.
4. **Order and write**: functions, then terminals and constants, numbered `F00001...`; types ordered as today (the root, then depth, then name; the matrix family in template order), numbered `T0001...`; the e-graph datatype, rules, rows and sorts from the rows and `rules.tsv`, with today's sort and constructor naming.

The **Rust** side (`src/futhark_view.rs`): `View::load(master, spec) -> View { types_tsv, functions_tsv, datatype, rules, rows, sorts }`, the same text the generator writes; `futhark_sig::Instance` stays a `Copy` static handle `{ name, spec: include_str!(view.tsv) }`, and its view is computed at runtime from `include_str!(master/*)` + the spec once a process, in one cache entry a view name holding the view text and its parsed `ViewSpec`, from which the kingdom, the signature and the bridge are all read, so they cannot disagree (finding 9), and `View::write(dir)` writes the exact table (the option). The committed written-out tables (`sig/<p>/types.tsv`, ...) stay as the regression record and for the witnesses; nothing loads them at runtime. For A7 the Rust view computes `types.tsv`/`functions.tsv`; its `costs.tsv` and e-graph files stay generator-written from the same view (A7's runner is deprecated by step 4, so its family-specific cost and rule code is not ported).

### D4. Witnesses: master witnesses, then per-view checks

- **Master witnesses** (`M1...`, on `rows.tsv` itself, independent of the expansion code): refs unique and well-formed, `since <= VERSION`; every `{UPPER}` variable is bound by the row's scope or is global (`E`, `K`, `OUT`); a function's spelling holds `{0}..{n-1}` once each, in order (Futhark's application is positional); every binder's last argument is its own frame (`+T-{D}` for `tab`, `+L-..`/`+F-..` for the loops) and its spelling binds exactly that frame's variables (the tabulate and loop spellings restated by hand in the witness); every heavy row carries `when shallow`; every index row is one of the clamped templates (restated); no integer element admits `/`; every class a template names has a type template.
- **Per-view checks**: today's `S1..S10` on each written signature view (root, context parse, terminals bound, binder bodies, the work bound, the index templates, a terminal for every dual, the pool, the e-graph's bijection), and today's `W1..W13` on A7's. Their models now come from the spec (dims, families) and not from the deleted `SIG_INSTANCES`; where `S10` compared a spelling with `SIG_SCANS`, it now compares with the master row: a consistency check, no longer an independent one (recorded).

### D5. The sniffer: `kingdoms/futhark/sniff.py`

- **Input**: the problem's Futhark source and its entry (`tests/corpus/futhark_problems/<p>/yardstick.fut`, `main`). The compiler is pinned: the sniffer requires `futhark --version` 0.28.1, the version whose print it parses (finding 6).
- **The CLI types it**: `futhark dev -t` prints the type-checked program with every size inferred (measured: histogram's `: []i32` prints as `: [i64.i32 k]i32`). The sniffer parses that entry line: size parameters, `((name: T): T)` parameters, the result type. A scalar `i32` parameter that appears as `i64.i32 p` in a dimension is a **size input**; another scalar is an input value.
- **Writes** `view.tsv`: the element (the one element type of the parameters and result; two or more is an error until step 4), the inputs, the output, `dims` in first-appearance order, the master's default depth (2) and pool, `master` = the current version, and the **families**: an optional family is admitted when its construct appears in the source of the entry or of a definition the entry reaches (by name, transitively; comments stripped; the prelude is not the problem's text): `scan`; `hist` or `reduce_by_index`; `%` (finding 5). The accepted signature subset is explicit: size parameters, parameters of an element type over size parameters, a scalar `i32` parameter that appears exactly as `i64.i32 p` in a dimension (a size input, still usable as a value), a result over those; anything else stops the sniffer with a diagnostic naming it (finding 7). That rule reproduces today's declarations; whether every view should carry every family is a step 4 experiment, not decided here.
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
