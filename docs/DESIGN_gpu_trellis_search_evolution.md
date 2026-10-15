# Design: evolving the GPU trellis search with fuller and phylu (2026-10-15)

## Context

**Objective.** Run UltraDim's UDV23 vector search — today CPU code in `ultradimdb` and `lib/trellis` — entirely on the GPU for a batch of queries, with only the top-k results per query coming back. The WGSL that does this does not exist. **The task is not to write it: it is to evolve it with our tools** — fuller (the kingdom, the harness, the safety layer, `wgsl_tune`) and phylu (the evolution engine, `wgsl_fit`).

**Approach.** Capture the CPU search's inputs and the output of each of its stages on real query batches; turn each stage into a fuller problem whose oracle is the captured output; let the model write candidate kernels from each problem's contract, the harness judge them against the captured data, and phylu evolve the correct ones for speed. One problem per kernel, composed afterwards.

**What makes this feasible (measured).**
- Model-written seeds, judged by data and evolved: on nluca tonight every gain began with a model kernel or a model rewrite and was improved by rules and section crossover (0.83-0.89x the shipped kernel's time at four of five widths, paired). Sonnet-written Gram kernels were correct 38 times in 40.
- Evolution from random genes with input/output data alone does not work: six random-start Gram runs (up to 180 generations) and the sketch start found no correct kernel, because the fitness had no slope (a near-miss scored like nonsense).
- **Search has a slope where Gram had none**: recall@k against the CPU's answers is a number between 0 and 1, and every stage has a checkable intermediate output.

**Status.** Design. Nothing built. The CPU code is the reference and the oracle; it is not a competitor, and its timing is the last step, not the first (Andrew, 2026-10-15).

---

## 1. The search, as it runs today (read in the code)

| stage | code | work per (query, seed) |
|---|---|---|
| projection | `ultradimdb/src/template_ops.rs` `build_formula_projectors`, `formula_project_sparse_one` / `_dense_one` | the raw query to one d_proj unit key per seed |
| entry | `lib/trellis/src/template.rs:1031` `walk` | score all 140 L1 nodes |
| walk | best-first over the 24-exit wiring | pop the best node; push its 9 proximity exits (stored) and 12 random exits (hashed from node, seed, epoch); the first visit to a cell pushes its claimed slots (positioned at their record's key) and scores its leaf rows (up to `cell_cap`); stop on the ef rule or `max_expansions`; keep the best `row_budget` rows |
| beam | `template.rs:1311` `beam_over_edges` | best-first over the record graph (`EdgeTier`, about 13 neighbours a row), ef-gated, `max_expansions`; optional forced seed expansion (cousin cells) |
| take | `template_ops.rs` (in `template_knn_batch_queries`) | drop tombstoned rows and rows failing the payload filter; keep `take_per_seed` |
| pool | same | per query, the union over seeds by best key cosine |
| rerank | `ultradimdb/src/ops.rs` `dense_union_and_exact_rerank`, `sparse_union_and_exact_rerank` (`lib/ultradim-sparse` `sparse_cosine_kahan`) | exact raw cosine of every pooled row; sort; keep k |

Entry point: `UltradimV23Search` / `UltradimV23SearchBatch` (`ultradimdb/src/v23_search_ops.rs`) → `template_knn_batch_queries`. Every walk and beam operation is a dot product of the query key with a template node position or a record key; the rerank is a dot product in the raw space, dense or sparse.

## 2. The target architecture (what the evolved code must implement)

**One workgroup runs one whole (query, seed) search in one dispatch**: frontier and result heaps in workgroup memory, the visited set in the job's slice of global memory, the workgroup's threads splitting the dot products and scoring a popped node's 21 exits or a cell's leaves together. A batch of Q queries over S seeds is Q x S workgroups. (Trellis's existing `template_gpu.rs` scores candidates on the GPU but keeps the heaps and the loop on the CPU, a submission and read-back every beam round; its own header names the 1-3 ms floor a round.)

Resident per family: template field centres (shared), proximity exits (shared), random exits by formula in the kernel, seating CSR and record keys and edge CSR (per seed), liveness bitmap, a per-request filter bitmap, the rerank substrate (dense n x D or sparse CSR).

| kernel | workgroup per | does |
|---|---|---|
| K1 projection | (query, seed) | sparse: each nonzero adds its hashed formula columns; dense: a matrix product; normalise |
| K2a beam | (query, seed) | best-first over the edge CSR from given seed rows, liveness and filter tested in the kernel, top `take` out |
| K2b walk | (query, seed) | the template walk, leaf gather, `row_budget` best rows out (feeds K2a in the same workgroup once both exist) |
| K3 pool | query | merge S lists, max key cosine per row |
| K4 rerank + top-k | query | exact cosine of the pooled rows (dense gather, or sparse sorted-merge one thread a candidate); top-k in the kernel, ties by id ascending |

## 3. The data: capture is the first build

The CPU search becomes the oracle by recording it. In `template_knn_batch_queries`, behind a capture switch, for real query batches on small families (so a case fits a GPU binding and runs in seconds):

- **inputs**: the queries (sparse pairs or dense rows), the knobs (`walk_ef`, `walk_expansions`, `walk_row_budget`, `beam_ef`, `beam_expansions`, `take_per_seed`, `cousin_cells`, `active_seeds`), and the resident structures flattened to the GPU layout of section 2 (template field, proximity exits, seating CSR, keys, edge CSR, liveness, substrate);
- **every stage's output**: the projected keys per (query, seed); the walk's rows; the beam's scored rows; the taken rows; the pooled set per query; the final top-k with exact cosines.

Each captured batch is a case directory, raw little-endian files named by type in binding order, as nluca's cases are (`noctiluca-dev/results/nluca/cases`, its README the model). Capture families at several sizes and both substrates; timing cases at production batch shapes (queries per batch, seeds, d_proj, rows).

## 4. The problems: one per kernel, each against its captured stage

Each kernel is evolved against the CAPTURED output of the stage before it, so the five evolve independently and in parallel, and compose afterwards.

| problem | inputs (captured) | oracle (captured) | correctness test | fitness slope |
|---|---|---|---|---|
| `trellis_project` (K1) | queries, formula seeds | projected keys | max abs error over RMS (`max_abs_over_rms`, exists) | numeric: error falls smoothly |
| `trellis_beam` (K2a) | keys, edge CSR, seed rows, liveness, filter, knobs | beam rows (after take) | recall of the captured rows | recall 0..1 |
| `trellis_walk` (K2b) | keys, template field, prox exits, seating CSR, knobs | walk rows | recall of the captured rows | recall 0..1 |
| `trellis_pool` (K3) | per-seed taken lists | pooled set | exact set and scores | small kernel |
| `trellis_rerank` (K4) | pooled rows, substrate, queries | top-k ids and cosines | top-k recall, cosines within float tolerance | recall, then error |

**Divergence policy.** A GPU dot product sums in a different order, so a walk can take a different path and still be as good. The pass rule for K2 is recall at least the agreed bar against the captured rows (not bit identity); the end-to-end gate is recall@k of the composed pipeline against the oracles the engine already builds (`MeasureRecall`, `CreateSparseOracles`).

## 5. What fuller and phylu need, beyond what exists

| change | where | why |
|---|---|---|
| a **recall norm** (set overlap of an id output against the captured ids, graded) | fuller `src/wgsl/harness.rs` `Norm` (today `mean_abs_over_max`, `cauchy_schwarz`, `max_abs_over_rms`, all numeric) | the slope that makes K2 and K4 evolvable; also feeds the graded closeness of infeasible kernels |
| **reference-free problems** | harness: expected outputs from captured files (as nluca's `expected_*` files), the speed column against the best correct kernel so far until one exists | no shipped GPU kernel exists for K2, K3 (K4's dense arm has one: udv23-umap `calculate_cosine_similarities`) |
| **kingdom: loops that run until a condition** | fuller safe grammar (`safe_reader.rs`, `kingdoms/safewgsl/functions.tsv`) | the walk and beam stop on the ef rule, not a count |
| **kingdom: bounded heaps in workgroup memory, a global-memory hash set** | same, with the safety layer's proofs for cooperative workgroup access (layer 7) | the frontier, results and visited sets |
| **kingdom: wider strided loops** | `tile_strided_*` rows take `u32.lit64` bound and step | tonight's kmeans seeds stepped by 128-256: about 25 of 49 refusals |
| **kingdom: workgroup arrays of indices** | "a workgroup array of u32.index has no safe spelling" | 11 kmeans refusals; candidate lists in workgroup memory |
| **a 64-bit hash in u32 pairs, proved bit-identical** | a WGSL helper the kingdom can name | `rand_exits` mixes in 64 bits; WGSL has no u64 |
| **multi-kernel composition** | harness: a plan that dispatches several problems' winners in one submission | the end-to-end pipeline once each kernel is evolved |

Already in place and used as-is: paired fitness, timing bands for any number of timing cases, `budget.gpu_wait_ms`, the GPU test token, `wgsl_tune` (convert, step-2 gate, evolve, paired choice, report), section crossover (2.3x more correct children than typed), model seeding under a problem with the problem's own kernels kept and its own cost bound.

## 6. The evolution, per problem

1. **Contract** (`problems/<p>/contract.md`): the bindings, the semantics in plain terms, the section-2 architecture for that kernel, the knobs, and the captured case layout. Sonnet writes seeds from it (`--seed-from model`).
2. **Gate**: every seed through the safety layer (no races, no wrapping indices) and the recall norm.
3. **Evolve** (`wgsl_tune`): section crossover, rules, rewrites, model rewrites and "beat both" calls; paired timing against the best correct kernel; the problem's timing shapes from real batches, split by regime where the work changes character (tonight's lesson from kmeans' 20 against 9,000 centroids).
4. **Hand back**: a kernel only when it passes every captured case at the recall bar and is shown faster at every timing shape; otherwise the best correct one so far.

## 7. Order and expectations (stated before any run)

| step | expectation | risk |
|---|---|---|
| 1. capture (CPU, no GPU) | cases for 3 family sizes x both substrates | the resident structures' GPU layout must match section 2 exactly |
| 2. recall norm + reference-free problems in fuller | unit-tested on the harness's CPU executor | — |
| 3. K1 projection and K4 rerank evolved | model seeds correct often (nluca-like data-parallel shapes); K4 seeded also from `calculate_cosine_similarities` | sparse K4's sorted merge may need a grammar addition |
| 4. kingdom extensions for K2 | random genes and model seeds both read and pass the safety layer | proving cooperative heap access race-free is the hardest safety work |
| 5. K2a beam, then K2b walk | the beam first: smaller, with a CPU lockstep already proved bit-identical (`trellis/tests/g9_gpu_correction_parity.rs`); recall climbs over generations | **the open risk: nobody has shown model + evolution produce a correct stateful search kernel this size** |
| 6. compose, end-to-end recall@k | at the agreed bar on `MeasureRecall` | divergence compounding across stages |
| 7. last: the CPU comparison | — | — |

## 8. Decisions needed from Andrew

- The recall bar for the GPU path, or bit identity with the CPU.
- Families and sizes to capture first (d_proj, rows, sparse or dense), and the production batch shapes for timing.
- Where capture lives in `ultradimdb` (a feature flag or an RPC), since it touches the server's search path.
