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
