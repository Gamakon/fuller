# Design: evolving the GPU trellis — search and streamed ingest — with fuller and phylu (2026-10-15)

## Context

**Objective.** Two things, on the GPU, from one set of kernels: (1) UltraDim's UDV23 vector search for a batch of queries, only the top-k per query coming back; (2) **the trellis ingest optimisation ("settle")** on streamed microbatches — Andrew's biggest problem: data loaded through the trellis is queryable at once, but settling it (the HNSW-like edge building that follows) takes ages on the CPU. Both are today CPU code in `ultradimdb` and `lib/trellis`. The WGSL that does this does not exist. **The task is not to write it: it is to evolve it with our tools** — fuller (the kingdom, the harness, the safety layer, `wgsl_tune`) and phylu (the evolution engine, `wgsl_fit`).

**Approach.** Capture the CPU search's inputs and the output of each of its stages on real query batches; turn each stage into a fuller problem whose oracle is the captured output; let the model write candidate kernels from each problem's contract, the harness judge them against the captured data, and phylu evolve the correct ones for speed. One problem per kernel, composed afterwards.

**What makes this feasible (measured).**
- Model-written seeds, judged by data and evolved: on nluca tonight every gain began with a model kernel or a model rewrite and was improved by rules and section crossover (0.83-0.89x the shipped kernel's time at four of five widths, paired). Sonnet-written Gram kernels were correct 38 times in 40.
- Evolution from random genes with input/output data alone does not work: six random-start Gram runs (up to 180 generations) and the sketch start found no correct kernel, because the fitness had no slope (a near-miss scored like nonsense).
- **Search has a slope where Gram had none**: recall@k against the CPU's answers is a number between 0 and 1, and every stage has a checkable intermediate output.

**Status.** Design. Nothing built. The CPU code is the reference and the oracle; it is not a competitor, and its timing is the last step, not the first (Andrew, 2026-10-15).

---

**The finding that makes ingest a GPU problem, not a rewrite (read in the code).** Settle is already window-parallel by design: every row of a window proposes its edges against the state frozen at the window's start, in parallel; the streaming posture already commits with the order-independent deferred re-prune (`streaming_posture`, `deferred_reprune: true`); and a slot's contest is a maximum with ties by row id, so it is order-independent too. And a row's proposal IS a search (walk + beam, with its own key as the query). So the search kernels serve ingest, and the window is the microbatch.

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

## 2b. Streamed ingest ("settle"), as it runs today (read in the code)

Entry: the upsert path calls `trellis_template_stream_ingest` (`ultradimdb/src/template_ops.rs:255`) after rows and per-seed keys commit; under the engine's write lock, every full window that has accumulated is streamed through each seed's `SettleStream::ingest_window` (`lib/trellis/src/template.rs:1907`). Posture: `streaming_posture` (route ef 8, 48 expansions, row budget 64, frozen slots NOT skipped; beam ef 64, 128 expansions; m 13, m_max 16, the noise floor; 2 extra leaf homes; deferred re-prune; rolling correction at a lag when set).

| phase | code | work | parallel today |
|---|---|---|---|
| propose | `ingest_window`, per row of the window | walk (`walk`, insert posture) with the row's key as the query; beam over the record edges from the walk's rows; drop self and sub-floor; `spread_prune` to m (greedy: keep a candidate only if it is closer to the row than to every kept one: a pairwise dot product per candidate pair) | yes (rayon), against the frozen state |
| seat | `seat_record`, `plant_leaf` | the routed cell's floating slot picked by a hash of (row, seed); append the row to that slot's leaves and contenders; resolve the slot (best contest score, ties by row id); plant the row as a leaf in 2 runner-up cells | no: serial over the window |
| commit edges | `commit_edges_batch` (deferred) | each row's forward list set; each chosen neighbour gets the reverse edge appended; every list grown past m_max re-pruned once by `spread_prune` | appends serial; re-prunes parallel |
| advance | `advance_cell_window`, every one of the 55,900 cells | window counter + re-resolve every claimed slot's contest | no |
| rolling correction | `correction_pass_inner`, the rows of window w - lag | re-propose and re-commit against the state as it now stands ("roughly doubles per-window ingest cost") | proposals yes |
| telemetry (sampled, off in streaming) | `ingest_window` | brute top-10 over the arrived set for 1 row in 500 | — |

The code's own measurements name the walls: correction 51 % of the settle wall, `commit_edges` 33 % before the deferred re-prune, an unclaimed formula slot 33.6 µs to materialise (the insert walk does not skip them).

## 2c. The target architecture for ingest

**The GPU holds the master copy of each seed's index while a family is resident; a microbatch (window) is one submission; the CPU reads back a serving snapshot only to persist** (`save_serving_state`'s content). Layouts chosen so every update is a parallel write, not a re-pack:

- **edges**: fixed stride, n x (m_max + spill) slots per row with a per-row count (integer atomics exist in WGSL), so a reverse edge is an atomic append and no CSR is ever rebuilt;
- **seating**: per-cell leaf arrays and per-slot contender arrays at fixed capacity with counts, a holder per slot, the claimed-slot list per cell;
- **keys**: appended per window (the arrivals' per-seed keys are already computed on upsert);
- **the template**: as in section 2, plus the rotated-disk master pattern (`master_disk`, `ARENA_SLOTS` x d), because the insert walk materialises unclaimed slots (a dot-and-add on the GPU).

| kernel | workgroup per | does | reuses |
|---|---|---|---|
| I1 propose | (window row, seed) | walk (insert posture) + beam with the row's key, drop self and sub-floor | K2b + K2a of the search |
| I2 spread prune | (window row, seed) | sort the scored candidates, the pairwise dots, the greedy keep to m, the fill rule | — (also I4) |
| I3 seat | (window row, seed) | hashed slot pick (the 64-bit mix in u32 pairs), atomic appends to leaves and contenders, the 2 extra homes | the hash helper |
| I4 commit | (dirty neighbour row, seed) | forward lists written; reverse edges atomically appended; every list past m_max re-pruned once | I2 |
| I5 advance + resolve | cell, then claimed slot | window counter; each slot's contest as a max-reduction with ties by row id | — |
| I6 rolling correction | (row of window w - lag, seed) | I1 + I2 + I4 against the current state | I1, I2, I4 |

All seeds of a family run in the same submission (their streams are independent). Order independence is the contract: proposals read only the frozen state; appends land in any order and every consumer sorts or reduces by (score, row id) before use, exactly the CPU's tie rules.

## 2d. Facts from qdrant (ultradim-QandA, 2026-10-15, from the code and docs)

- **Window**: default 10,000 rows (`ultradimdb/src/template_ops.rs:345`; window 1 is the record-by-record interactive mode). Per-family window sizes and stream rates are not documented. Measured: dbpedia 990k x 1536-D at D_proj 128 upserted at 1,971 rows/s and settled (4 seeds, rolling lag 2) in about 95 min (HANDOVER.md); chembl_500k_30m settled over the wire in 81 min.
- **Families to capture first**: `chembl_500k_30m` (sparse, 30M-D raw, 500k rows, D_proj 2,048, seeds 42/137/911/4242: the R1 bar family) and `dbpedia_990k_wt_d128` (dense 1,536-D as full-index sparse, 990k rows, D_proj 128, the same 4 seeds: the cell-density stress case, about 17.7 rows a cell a seed). Settled states in `storage/udv23_trellis/`; field path and placement generation in each family's `template_manifest.txt`.
- **Quality bars**: serving recall@10 >= 0.9964 at p50 <= 29.1 ms (`docs/plans/HNSW_ULTRADIM_REQUIREMENTS.md:23`; protocol at :206: 500 held-out queries, exact brute-force truth, chembl 500k at 30M-D). Build-time gate R7: a probe must refuse a bad settled graph (precedent: 25 hash-sampled arrivals, threshold 0.8, `lib/trellis/tests/g8_movielens.rs`).
- **The GPU re-implementation precedent** (`lib/trellis/tests/g9_gpu_correction_parity.rs`): a LOGIC gate (the restructured algorithm under CPU arithmetic reproduces the sequential CPU pass bit for bit, every row) and a QUALITY gate under GPU arithmetic (serving recall no more than 0.002 below the CPU's; measured max |dot delta| 4.77e-6, which flips near-ties and reroutes about 24 % of beam searches at equal or better recall). No rule demands bit identity with the CPU index. R10's bit identity is about arrival order within a window (a shuffle gives the same seating), not implementation equivalence.
- **Capture hook**: the house precedent is an opt-in knob on the RPC, data on the response and artifacts under the family's state directory (as `telemetry_stride` on the settle request, `template_ops.rs:349`, returning per-window stats at :375 / :443; `settle_progress.txt` at :88; per-stage TRLSRV01 snapshots in `g3h_floor_correction.rs`). Not a cargo feature (those are backends), not an env var (machine config only).
- **`calculate_cosine_similarities`**: compiled into udv23-umap (which ultradimdb links, wgpu 27), but its only caller is the inventory smoke test; "thousands of times a minute" was the retired qdrant-era engine (`attic/segment_ultradim/gpu_fingerprint_search.rs`). Today's serving hot path is the CPU trellis walk + beam. (Unreached, it is absent from the 0.5.0 wheel's binary.)
- **A rule in the record, not in docs**: Andrew's 10-minute test rule: every settle-cost idea proves itself on a harness under 10 minutes before volume.

## 3. The data: capture is the first build

The CPU search becomes the oracle by recording it, through an opt-in capture field on the settle and search requests (the house precedent, section 2d), artifacts under `storage/udv23_trellis/{family}/`. In `template_knn_batch_queries` and `ingest_window`, for real query batches on small families (so a case fits a GPU binding and runs in seconds):

- **inputs**: the queries (sparse pairs or dense rows), the knobs (`walk_ef`, `walk_expansions`, `walk_row_budget`, `beam_ef`, `beam_expansions`, `take_per_seed`, `cousin_cells`, `active_seeds`), and the resident structures flattened to the GPU layout of section 2 (template field, proximity exits, seating CSR, keys, edge CSR, liveness, substrate);
- **every stage's output**: the projected keys per (query, seed); the walk's rows; the beam's scored rows; the taken rows; the pooled set per query; the final top-k with exact cosines.
- **ingest, in `ingest_window`, per window and seed**: the frozen state at the window's start (seating, edges, keys, counters), the window's rows, and after each phase: each row's walk rows, scored candidates and chosen (pruned) list; the seating after seat; the edges after commit; the slot holders after advance; and for the rolling correction, the corrected rows' new lists. A stream of consecutive windows from an empty index to a mature one is the timing and quality case: microbatch sizes as production streams them.

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
| `trellis_propose` (I1) | frozen state, window rows | each row's scored candidates | recall of the captured candidates | recall 0..1 |
| `trellis_prune` (I2) | scored candidates, keys | chosen lists | exact (deterministic given the set) | list overlap |
| `trellis_seat` (I3) | seating, routes | seating after seat | exact (hashes and appends, sorted) | per-cell overlap |
| `trellis_commit` (I4) | edges, chosen lists, keys | edges after commit | exact given the set (the deferred form) | per-row list overlap |
| `trellis_advance` (I5) | seating, counters | holders after advance | exact (max with ties by id) | per-slot agreement |

**The end-to-end ingest fitness is the index's quality, not its bytes**: after streaming a captured sequence of windows through the composed kernels, serving recall@10 against the brute-force oracle (the engine's `MeasureRecall`), compared with the CPU-settled index on the same stream. The code itself already rules this way for its own changes ("an A/B quality gate decides, never an equivalence claim", `SettleParams::deferred_reprune`).

**Divergence policy (the g9 precedent).** Two gates per kernel: a LOGIC gate (the GPU-shaped algorithm, run with CPU arithmetic, reproduces the CPU bit for bit) and a QUALITY gate under GPU arithmetic (serving recall no more than 0.002 below the CPU's; near-tie flips are expected and allowed). End to end: recall@10 >= 0.9964 on chembl_500k_30m (500 held-out queries), and the R7 probe gate on every settled graph. Every evolution case sized to Andrew's 10-minute rule.

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
| **stateful streams as cases** | harness: a case that is a sequence of windows, the kernels' output state carried into the next window (nluca's ping-pong across 64 steps is the existing pattern), judged at checkpoints | ingest evolves over a stream, not a single call |
| **kingdom: integer atomics** (`atomicAdd` on u32 counters) and the safety proof that appends to a shared array are disjoint by their returned index | safe grammar + safety layer | reverse-edge and leaf appends |

Already in place and used as-is: paired fitness, timing bands for any number of timing cases, `budget.gpu_wait_ms`, the GPU test token, `wgsl_tune` (convert, step-2 gate, evolve, paired choice, report), section crossover (2.3x more correct children than typed), model seeding under a problem with the problem's own kernels kept and its own cost bound.

## 6. The evolution, per problem

1. **Contract** (`problems/<p>/contract.md`): the bindings, the semantics in plain terms, the section-2 architecture for that kernel, the knobs, and the captured case layout. Sonnet writes seeds from it (`--seed-from model`).
2. **Gate**: every seed through the safety layer (no races, no wrapping indices) and the recall norm.
3. **Evolve** (`wgsl_tune`): section crossover, rules, rewrites, model rewrites and "beat both" calls; paired timing against the best correct kernel; the problem's timing shapes from real batches, split by regime where the work changes character (tonight's lesson from kmeans' 20 against 9,000 centroids).
4. **Hand back**: a kernel only when it passes every captured case at the recall bar and is shown faster at every timing shape; otherwise the best correct one so far.

## 7. Order and expectations (stated before any run) — ingest first

**Ingest is the primary problem: it is the first impression the system gives (Andrew, 2026-10-15).** Search follows, reusing what ingest builds (I1 is the search's walk and beam).

| step | expectation | risk |
|---|---|---|
| 1. capture ingest streams (CPU, no GPU): consecutive windows of `ingest_window` per seed, from an empty index to a mature one, at the production window (default 10,000 rows, `template_ops.rs:344`), with every phase's output | streams for 3 family sizes x both substrates | the GPU layout of section 2c must match exactly |
| 2. harness: stream cases (state carried across windows), list- and set-overlap norms, reference-free problems | unit-tested on the harness's CPU executor | — |
| 3. the deterministic ingest kernels, exact oracles: I2 spread prune, I5 advance + resolve, I3 seat, I4 commit | data-parallel and exact: model seeds often correct, as on nluca | I3 and I4 need integer atomics and their safety proof |
| 4. kingdom extensions: loops until a condition, workgroup heaps, a global hash set, atomics, wider strided loops, workgroup index arrays, the 64-bit hash in u32 pairs | random genes and model seeds read and pass the safety layer | the cooperative-heap race proof is the hardest safety work |
| 5. I1 propose = K2a beam then K2b walk (insert posture) | recall of the captured candidates climbs over generations | **the open risk: a correct stateful search kernel this size from model + evolution is unproven** |
| 6. I6 rolling correction, then ingest end to end on a captured stream | the GPU-settled index's serving recall@10 at the CPU-settled index's on the same stream | quality drift over many windows |
| 7. search: K1 projection, K4 rerank + top-k, K3 pool, composed with K2 | end-to-end recall@k at the agreed bar on `MeasureRecall` | divergence compounding across stages |
| 8. last: the CPU comparison | — | — |

## 8. Decisions needed from Andrew

- The recall bar for the GPU path, or bit identity with the CPU.
- Families and sizes to capture first (d_proj, rows, sparse or dense), and the production batch shapes for timing.
- Where capture lives in `ultradimdb` (a feature flag or an RPC), since it touches the server's search and ingest paths.
- Production microbatch (window) sizes and stream rates, for the ingest timing shapes.
