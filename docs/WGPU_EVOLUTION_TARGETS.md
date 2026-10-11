# WGPU kernels worth evolving: a target list for WGSL evolution

Written 2026-10-11. Read-only research: no code was changed and nothing was
run on the GPU (a fit was running). Every claim below cites the file it was
read from.

## Context

**Objective.** Show that evolving WGSL kernels works on a range of problems,
not one. Today there is one problem, the weighted Gram matrix H = ΦᵀΦ of the
Blue World recommender (`H_GRAM_ACCUM_WGSL`), where evolution reached about
10–14 times the shipped kernel's speed. This document lists every WGSL compute
kernel in phylu, fuller and qdrant, and ranks the ten best next problems.

**Approach.** Every `@compute` entry point in the three repositories was
listed with grep and read at its source. For each kernel: what it computes,
how often it runs, what its threads do, how much speed is plausibly left, and
what a fitness evaluation for it would need. A fitness evaluation here means
what `fuller/src/wgsl/gram.rs` (`FitnessContext`) does for Gram: a reference
answer, generated inputs, a correctness test, and timing against the shipped
kernel at a few sizes.

**Status.** Research only. Two findings change the plan before anything is
built:

1. **Decode is not worth evolving.** `docs/PLAN_wgsl_superopt.md` §6a measured
   phylu's decode pass at 0.30 ms of a 32.5 ms generation, under 1 %, and none
   of 64 variants beat the noise. The time of a generation is in the
   expression evaluator, the scoring kernel and HFF, but no per-pass timing of
   those exists yet: the same plan named it as the next step and it was not
   run. Every "how hot" judgement below for phylu is therefore read from the
   code, not measured, and the first task is to put a GPU timestamp on each
   pass of `phylu/src/evolve/resident.rs::run`.
2. **phylu's kernels need a different correctness test from Gram.** Gram
   passes a candidate whose H is within a floating-point tolerance of an f64
   reference. phylu's discrete kernels (decode, variation, typing, regex
   compile) are checked bit for bit against CPU twins, and its scoring and
   HFF kernels are certified by identical selection on a real population, not
   by a per-value tolerance (memory: kernel mutation gate is selection). Both
   are stricter and costlier than the Gram test, and the fitness evaluation
   for each phylu target must be built around them.

**Why.** Andrew needs a range of large speed-ups to claim WGSL evolution works,
and our own pipeline's kernels are the most persuasive members of that range:
if evolution speeds up the evaluator, scoring, selection and variation
kernels, symbolic-regression fits get faster. One limit on that argument, from
`phylu/examples/wgsl_fit.rs`: a WGSL individual's fitness costs about ten
seconds of compile, correctness cases and timing, so faster phylu kernels
speed up symbolic regression and the regex kingdom, not the WGSL-kernel
evolution itself.

## The ten best next problems

Ranked by Andrew's priority (2026-10-11): UltraDim's search first,
then our own evolution pipeline's kernels, then the rest by variety of
computation, expected headroom and how easily a Gram-style fitness
evaluation can be built. The UltraDim rows A–C follow ultradim-hotpath's
report (2026-10-11): every v23 search runs three stages, all on the CPU with
rayon today, and they are ranked by the share of search time they hold. "Kingdom gaps" are what the safeWGSL kingdom (`kingdoms/safewgsl`,
specialised to Gram's three bindings `phi`, `h`, `cfg`) lacks for that kernel;
the kingdom's general limits are listed after the table.

| # | kernel (file, entry point) | computation | why it matters | expected headroom | correctness test | kingdom gaps |
|---|---|---|---|---|---|---|
| A | UltraDim search, stage 1: the Trellis walk and `beam_over_edges` (qdrant `lib/trellis/src/template.rs:1031`, `:1311`, `:1340`), reached from `ultradimdb/src/template_ops.rs:859` `template_knn_batch_queries`, as ultradim-hotpath's report (2026-10-11) traced it | walk the Trellis index and widen a beam over its edges, scoring each candidate key (2,048 dimensions) against the query | every v23 search; about 28 ms of DBpedia's 29.5 ms median search | CPU with rayon today; no kernel serves it. `template_gpu.rs:23` `single_main` and `:35` `pairs_main` exist but only tests call them; a trial that ran the beam in lockstep on the device lost to the CPU (663 s against 630 s at 500,000), bound by read-backs | the target is the whole beam for a batch of queries on the device, with the frontier logic inside the kernel so nothing is read back per step; this is a new kernel, not a speed-up of one | the Rust functions as the reference; identical top-k set and order, ties broken by id | a gathered key load through an index, a frontier and visited set in workgroup or private memory, a loop with a data-dependent exit, `u32` outputs; nothing in the kingdom today covers this |
| B | UltraDim search, stage 0: query projection (`ultradimdb` `formula.rs:317`, `:409`), from ultradim-hotpath's report (2026-10-11) | project the query to 2,048 dimensions with a Rademacher matrix (entries ±1) generated by hash per seed, under a bitwise accumulation contract | every search | CPU with rayon today; no kernel | high in principle: the matrix is generated, not read, so the work is hashing and signed adds; the bitwise contract fixes the order of additions | bit-identical output to the Rust function | the hash, sign selection from bits, a fixed accumulation order |
| C | UltraDim search, stage 2: the exact re-score (`ops.rs:144` `sparse_union_and_exact_rerank` calling `ultradim-sparse` `lib.rs:32` `sparse_cosine_kahan`; dense `ops.rs:198`, `:250` `dense_cosine_one`, no Kahan), from ultradim-hotpath's report (2026-10-11) | exact cosine of the query against each surviving candidate, Kahan-compensated for sparse vectors, then the final order | every search, under 1 ms of it | CPU today; `gpu_fingerprint_search.rs:252` is a dense Kahan kernel that only a smoke test calls, and no sparse kernel exists | easy but small: under 1 ms of a 29.5 ms search | identical top-k set and order, ties broken by id, against the Rust functions | for sparse: index and value buffers and a merge of two sparse lists; for dense: a dot product like Gram's |
| 1 | fuller `src/gpu_eval.rs` `EVAL_WGSL` → `eval_main` | expression interpreter: one thread per (expression, row) walks a node array backwards over a 128-float scratch | the bulk of every symbolic-regression generation (population × genes × rows threads) | high: the 128-float private scratch is too big for registers, so the compiler spills it (moves it to slow memory); each thread re-reads the same node array as every other thread of its expression; one row per thread | bit-identical predictions to the shipped kernel on recorded populations and data (the scorer and selection depend on every bit) | a `switch` on an opcode, a struct-typed node load (`Node`), a 128-element private array, a two-dimensional dispatch with a stride, NaN and infinity bit tests, eight bindings |
| 2 | phylu `src/evolve/score.wgsl` → `score_main` | per candidate least-squares fit (the scale and offset that best match the target) and error sums: one thread per (chromosome, gene-linker combination), seven serial loops over the rows | every generation, every candidate; 15 combinations × 3 wrappers per individual | high: each of the seven row loops re-reads every gene's prediction (no reuse between loops); each thread walks its own run of consecutive rows, so one step of a workgroup touches many memory lines and uses one value from each (the loads do not coalesce, that is, merge into wide reads); no workgroup cooperation over rows | identical selection on a real population (memory: kernel mutation gate is selection); a summation-order change is a different number, so a pure tolerance test is not enough | struct `Meta` with nine fields, `arrayLength`, Neumaier compensated sums (a second float carries the rounding error of each addition, `vec2`), seven bindings, writes to a `scores` buffer at a computed slot rather than a thread-owned cell |
| 3 | phylu `src/evolve/vary.wgsl` → `first_main`, `select_main` | tournament selection: per row, `tournsize` random draws (7 % of the island) and a fitness comparison each | every generation | medium at the shipped 600 + 200 population, high at large populations: tournament size grows with the island, so the work is quadratic in population (a 200,000 island draws 14,000 per row); each draw is a full 64-bit hash emulated with 16-bit limbs (`mix64.wgsl` `mul32x32_64`) | bit for bit against the CPU twin `phylu/src/evolve/vary.rs` (device.rs tests already assert this over 20 generations) | the `hash64` dual exists, but not the `mix64` call, the per-island struct (`Island`, 24 fields), cohort comparisons, a random gather over the fitness buffer, a `u32` store to `stage1`/`parent` |
| 4 | phylu `src/evolve/vary.wgsl` → `mutate_main`, `crossover_main` | variation: copy the parent row's genome, then point, inversion, transposition and constant mutations; crossover per pair | every generation | medium: a serial copy of the whole row per thread (no vector loads), and every random draw pays the 64-bit hash | bit for bit against `vary.rs` (already tested) | as in #3, plus `u32` genome stores at computed positions, a 512-entry private array (`mut_budget`) |
| 5 | phylu `src/evolve/hff.wgsl` → `colmax_main` and `hff_main` | `colmax_main`: maximum of each of 9 objective columns over all candidates; `hff_main`: per row, the TrueNorth angle of each of 45 candidates and the argmin (the index of the smallest) | `colmax_main` once per fit (generation 0); `hff_main` every generation | `colmax_main` very high: it launches 9 threads in total, each scanning population × 45 candidates serially; a parallel maximum is a textbook reduction, and a maximum is exact in any order, so the correctness test is simple; `hff_main` low at 800 rows, medium at large populations | `colmax_main`: bit-identical maxima; `hff_main`: identical winners and fitness (selection test) | `colmax_main` needs a reduction over a non-Gram buffer with a `u32`-indexed write of 9 outputs; `hff_main` needs `acos`, NaN bit tests, 14 bindings |
| 6 | phylu `src/evolve/regex_match.wgsl` → `match_main` | regular-expression matching by a state machine (a Thompson NFA, which tracks every state the pattern could be in at once): one thread per (gene, text row), a 9-word bit set of live states stepped byte by byte | every generation of the regex kingdom | medium to high: the states reachable without reading a byte (the epsilon closure) are recomputed per byte by repeating until nothing changes; neighbouring threads run different programs, so they diverge; text bytes are re-read per gene with no workgroup sharing | bit for bit against `regex_match::is_full_match` (CPU twin exists) | bitwise operations on a 9-word private array, byte extraction, a different binding set (instructions, classes, text, offsets) |
| 7 | qdrant `lib/projection-voting-kmeans/src/lib.rs:4748` cosine assign (`build_cosine_assign_pipeline`) | k-means assignment: per point, the most similar of C centroid keys (2,048 dimensions); a matrix product and an argmax in one kernel | every cycle of the live k-means (64–100 cycles per clustering; N = 50,000 to 1,000,000 points, C up to 64,000; callers in `ultradimdb/src/cluster_ops.rs`); the hottest live qdrant kernel the helper found, up to about 1.3·10¹⁴ multiply-adds | high: each thread walks C × 2,048 serially with no tiling and no `vec4` loads; the Gram shape with an argmax on top | exact labels against an f64 reference, with an allowance where the top two are within 1e-6 | closest to Gram: a second input buffer, a `cfg` with n, C and dimensions, a `u32` output and a running maximum |
| 8 | qdrant `lib/projection-voting-kmeans/src/lib.rs:5022, :5060, :5111` block histogram, cluster offsets, scatter ids (one problem: a stable counting sort of points by cluster) | histogram, scan (running totals) and scatter (writing each id to its computed place) | every k-means cycle | high and algorithmic: the histogram does O(blocks × C) work for an O(blocks) job (55 million threads at C = 64,000); mode 1 of the offsets runs a scan of up to 64,000 on one thread; the scatter does O(256²) per thread | exact: equal to a stable CPU counting sort | `u32` histogram and offset buffers, a scan, workgroup `u32` arrays |

**Runners-up.** qdrant `generator.rs:4984` `knn_top_k_main` (top-k selection; search does not call it, per ultradim-hotpath). LoRA `GRAD_A` and `FORWARD_G` (`udv23-umap/src/ultradim/sparse_mf/lora_gpu.rs:787, :688`): matrix products on every training step of the live trainer, `GRAD_A` the same form as Gram. `centroid_project` (`lib.rs:4863`): a plain matrix product every k-means cycle. `umap_epoch_gather` (`umap/kernels_optimize.rs:400`): irregular gathers with a hash on every UMAP epoch. `nearest_neighbor_histogram_main` (`generator.rs:4757`): all-pairs with atomics. NSGA-II dominance (`hff-zenkai/src/shaders/nsga2_dominance.wgsl:28`): bit-exact, but not compiled in production.

**Not in the ten, and why.** phylu `decode_main`: measured at under 1 % of a
generation. `type_main` (`cand.wgsl`): tiny per-thread work, bounded by
launch cost. fuller's snap and lint kernels: they run every few generations
on winners, not every generation (phylu `engine.rs`: `snap_every` defaults to
`pump_every`). The Gram kernel itself is the existing problem and stays the
reference member of the suite.

**The variety this gives.** Interpreter (1), multi-pass reduction (2), random
selection with a hash (3), random variation (4), column maximum and per-row
argmin (5), finite-automaton simulation over bit sets (6), matrix product with
argmax (7), histogram, scan and scatter (8); rows A–C add a graph walk with a
beam, hash-generated projection and sparse dot products. Gram adds the dense matrix product.

### What the safeWGSL kingdom lacks for all of them

Read from `kingdoms/safewgsl/TYPES.md` and `symbols.md`:

- **One output, one input.** A store goes only to `h`, at the thread's own
  cell (`row * cfg.d + col`), as a read-add-write. A load reads `phi` (any
  index, past-the-end reads zero), `h` at the thread's cell, or a uniform
  field. Every target above writes `u32` or `f32` to buffers other than `h`,
  often at computed positions, and reads several buffers.
- **Loop bounds are Gram's.** A long loop is bounded by `cfg.m`, `cfg.d`,
  `cfg.r_rows` or an index expression; tile loops by literals up to 64 or
  4,096. Data-dependent exits (the decode closure, the kNN insertion, the
  NFA closure) have no row.
- **Workgroup memory is four fixed arrays** (`a`, `b` of 2,048 `f32`; `va`,
  `vb` of 512 `vec4<f32>`); no `u32` workgroup arrays, so no workgroup
  histogram or scan.
- **Barriers only outside conditionals; one atomic row and one subgroup row**
  in the whole table; no `atomicAdd` on a `u32` buffer as targets 7 and 10 use.
- **No struct-typed values, no `arrayLength`, no `switch`.** The reader counted
  185 nodes of phylu's kernels with no row, nearly all struct field reads and
  `arrayLength` (`kingdoms/wgsl/README.md`). Target 1's opcode `switch` is the
  centre of the kernel.
- **The binding layout is fixed** at `phi` 0, `h` 1, `cfg` 2. Each target
  needs its own layout and its own driver, as Gram has its banded driver.

The general fix is a per-problem binding table (buffers, their element types
and which are writable), store rows over any writable buffer at an index the
types can prove the thread owns, `u32` workgroup arrays with integer atomics,
and loop rows bounded by uniform fields named per problem. That is the work
that turns the Gram kingdom into a kernel kingdom.

## Full inventory

Counts were taken by grepping for `@compute` in live source (2026-10-11).

| repository | live entry points | files | notes |
|---|---|---|---|
| phylu `src/evolve/*.wgsl` | 13 | 9 | all production; `mix64.wgsl` is a hash library concatenated in front of `evolve.wgsl` and `vary.wgsl` |
| fuller production (`src/gpu_eval.rs`, `src/pattern_verify.wgsl`, `src/lint/*.wgsl`) | 10 | 6 | `splice.wgsl` is a shared prefix, no entry point |
| fuller `src/wgsl/*.rs` | 0 production kernels | 10 | 53 `@compute` strings, all inside `#[cfg(test)]` blocks except three files that emit kernels rather than being kernels: `render.rs` (renders a candidate), `generator.rs` (random test programs), `llm.rs` (a prompt template) |
| fuller `kingdoms/wgsl/` | 3 Gram kernels + test material | many | `gram/{shipped,tiled,blocked}.wgsl`; `oracle/` and `samples/` are oracle fixtures and rebuilt copies of phylu kernels |
| qdrant `lib/` | 109 | 25 | the inventory doc says 76 across 15; the difference is explained below the qdrant table heading |

### phylu and fuller

Production sizes: the SRBench configuration is a population of 600 + 200,
3 genes, head 34 (`phylu/src/evolve/engine.rs` `Config::srbench`); larger fits
run 2,000 + 2,000 to 5,000 + 5,000, and one run reached 200,000 (`hff.wgsl`
header). A candidate is one of 15 gene-linker combinations × 3 wrappers, 45 per
individual, 10 floats each (`score.rs` `WIDTH`). A generation is 22–35 ms on
the M3 Max (`docs/PLAN_population_dag.md` §2a; `phylu/docs/PLAN_arbitrary_arity.md`).

| file : line, entry | computes | threads, sizes | how often | workgroup memory, atomics | headroom | evolution target? |
|---|---|---|---|---|---|---|
| phylu `evolve.wgsl:98` `init_main` | random initial population, every draw a 64-bit hash of (seed, stream, generation, row, slot) | one per row; genome row = genes × (head + 2·tail) | once per fit, and refills | none | low (runs once) | no: not hot |
| phylu `vary.wgsl:284` `first_main` | first tournament per slot | one per row; tournsize = 7 % of the island | every generation | none | quadratic in population | yes, #3 |
| phylu `vary.wgsl:303` `select_main` | second tournament, or elite slot by repeated selection | one per row | every generation | none | as above; elite slots are a selection sort | yes, #3 |
| phylu `vary.wgsl:526` `mutate_main` | clone parent row, apply 14 mutation operators | one per row | every generation | none | serial row copy, emulated 64-bit hash | yes, #4 |
| phylu `vary.wgsl:754` `crossover_main` | one-point, two-point and gene crossover per pair | one per pair | every generation | none | small per-thread work | yes, with #4 |
| phylu `decode.wgsl:205` `decode_main` | Karva walk to node array plus typed derivation | one per gene; 600–2,400 genes | every generation | none | measured: 0.30 ms of 32.5 ms | no: under 1 % |
| fuller `gpu_eval.rs:448` `eval_main` | expression interpreter | one per (expression, row), 2-D dispatch | every generation, per block of rows | none; 128-float private scratch | high | yes, #1 |
| fuller `gpu_eval.rs:1559` `stats_main` | per subtree minimum, maximum, sum and count over rows | one per (expression, node) | typed snap and subtree studies, not every generation | none | serial row loop per thread | possible; a reduction like #2 |
| phylu `score.wgsl:202` `score_main` | least-squares scale and train / validation / extrapolation errors | one per (chromosome, combination); 7 row loops | every generation | none | high | yes, #2 |
| phylu `cand.wgsl:39` `type_main` | candidate tower depth and wrapper refusal | one per candidate | every generation | none | low: a few loads per thread | no |
| phylu `hff.wgsl:128` `hff_main` | TrueNorth angle per candidate, argmin per row | one per row, 45 candidates each | every generation | none | low at 800 rows | yes, with #5 |
| phylu `hff.wgsl:295` `colmax_main` | maximum of 9 objective columns | 9 threads, each over population × 45 | generation 0 only | none | very high | yes, #5: an easy first member |
| phylu `regex_compile.wgsl:180` `compile_main` | regex AST to Thompson program | one per gene | every regex generation | none | low (short programs) | no |
| phylu `regex_match.wgsl:86` `match_main` | NFA full-string match | one per (gene, text row) | every regex generation | none | medium to high | yes, #6 |
| phylu `regex_score.wgsl:72` `score_main` | false negatives and positives per chromosome, banded over rows | one per chromosome | every regex generation | none (u32 counters, one writer each) | serial row count per thread | possible; a counting reduction |
| fuller `pattern_verify.wgsl:35` `verify_main` | coefficient of variation of subtree / pattern over rows | one per (subtree, pattern) pair | typed snap rounds | none | serial row loop per pair | possible |
| fuller `lint/kernel.wgsl:235` `lint_main` | rule-based rewriting of an expression, greedy rounds | one per expression | finisher and snap | none | divergent by nature | no: discrete, divergent, rarely hot |
| fuller `lint/snap.wgsl:86` `snap_main` | binary search of a sorted constant table per literal | one per literal | every snap round | none | small | possible: a search problem |
| fuller `lint/snap_graft.wgsl:88` `graft_main` | graft snapped constants into expression variants | one per (expression, variant) | every snap round | none | small | no |
| fuller `lint/snap_guard.wgsl:71,111,136` `guard_lengths`, `guard_residuals`, `guard_main` | R² of each variant over guarded rows, two passes | per variant / per value | every snap round | none | serial row sum per variant | possible; same shape as #2 |

### qdrant

Production sizes and liveness below come from a second, line-by-line pass by a helper agent, which read each kernel's callers. Live k-means: N = 50,000 to 1,000,000 points, C up to 64,000 clusters, 2,048 projected dimensions, 64–100 cycles. Live UMAP: N = 70,000, 15 neighbours, 500 epochs. LoRA: 4,096 × 8,192 × rank 64 or 128, every training step. Bugs the helper found, none fixed: hff-zenkai's GPU NSGA-II gives every individual rank 0 and its crowding distances come from zero-filled buffers; Blue World's two repulsion kernels (`generator.rs:4072, :4400`) write points in place while other threads read them; `gpu_fingerprint_search.rs:328` `select_top_k` races, though nothing dispatches it; `scripts/bluenoise_alt` does not compile. Dead or test-only per the helper: `gpu_pruning.rs`, the UMAP fuzzy-graph kernels (production runs them on the CPU), `ultradim/gpu_shaders.rs`, the udv23 copy of the navigation kernels, and most query-distance kernels in `generator.rs` (5402–6190).

`docs/GPU_FUNCTIONS_INVENTORY.md` (qdrant) counts 76 entry points across 15
files. A grep of live `lib/` today finds 109 across 25 files, plus one in
`scripts/bluenoise_alt/main.rs`. The gap is explained by two things the doc
does not list: the vendored `lib/hff-zenkai` shaders (30 entry points in 10
`.wgsl` files, an optional `wgpu` feature that its one user,
`lib/fingerprint-tuner`, builds with `default-features = false`, so they are
not compiled in production), and `lib/udv23-umap/src/synthetic/gene/gpu_navigation_shaders.rs`,
a byte-identical copy of Blue World's four navigation kernels. Without those 34 the count is 75 against the doc's 76, a difference of one that was not traced. `attic/` and
`.claude/worktrees/` hold older copies and were not counted.

| file : line, entry | computes | class | how often | workgroup memory, atomics | headroom | target? |
|---|---|---|---|---|---|---|
| blueworld `sphere/generator.rs:2882` `H_GRAM_ACCUM_WGSL` `main` | weighted Gram H = ΦᵀΦ, banded | matrix product | once per fit; 419,734 items × d = 8,192 took 130 s, then cached; its only remaining caller is in `attic/` (`itemrank_ops.rs:1369`), so no live code in `lib/` calls it | none | evolved 10–14× | the existing problem |
| `generator.rs:4005` `sphere_generation_main` | Gaussian points normalised to the sphere | random generation | per build | none | low | no |
| `generator.rs:4071` `blue_noise_repulsion_with_mask` | Thomson repulsion between points with a mask | all-pairs forces (n-body) | per relaxation step | none | high: all-pairs without tiling | yes, an n-body problem |
| `generator.rs:4399` `blue_noise_repulsion_with_aggregates` | repulsion with far-field aggregates and exact near field | n-body, Barnes–Hut style | per relaxation step | none | medium | possible |
| `generator.rs:4756`, `:4834` `nearest_neighbor_histogram_main` (two variants) | nearest-neighbour distance histogram | histogram, atomics | diagnostics per build | global `atomicAdd` on `u32` bins | high | runner-up |
| `generator.rs:4911` `nearest_centroid_main` | nearest centroid by cosine | distance + argmin | clustering | none | high | as #7 |
| `generator.rs:4984`, `:5091` `knn_top_k_main` (two variants) | per-query top-k by similarity | top-k selection | kNN graph builds | none; 100-entry private arrays | high | runner-up |
| `generator.rs:5402` `batched_queries_distances_main` | distances for a batch of queries | matrix product (tall, thin) | query serving | none | medium to high | yes |
| `generator.rs:5544`, `:5656`, `:5918` `*_partial_distances_main` | partial distances over a dimension band (dense, batched, sparse) | banded dot products | query serving | none | medium | possible |
| `generator.rs:5758`, `:6023` `*_reduce_partials_main` | sum the bands' partials | reduction | query serving | none | low | no |
| `generator.rs:6134` `single_query_distances_main` | one query against all points | dot products | query serving | none | memory-bound | no |
| `generator.rs:6190` `single_query_topk_main` | per-workgroup top-k then merge | top-k selection | query serving | workgroup scratch | medium | possible, with #8 |
| `generator.rs:13282`, `:13332` `compute_pairwise_distances` | all pairwise cosine distances (triangular index) | all-pairs | analysis | none | medium | possible |
| `generator.rs:18687` `main` | batch Haversine distances | elementwise | analysis | none | low | no |
| blueworld `gene/gpu_navigation_shaders.rs:1277,1498,1575,1634` `slerp_main`, `navigation_main`, `angular_distance_main`, `rank_children_main` | spherical interpolation, navigation steps, angular distances, ranking children | elementwise, gather, ranking | gene navigation | `rank_children_main` workgroup 256 | low to medium | possible (`rank_children_main` is a small sort) |
| hffvenn-gpu `gpu_kernel.rs:975,1070` `pair_cosine_main`, `pair_cosine_masked_main` | cosine of feature pairs, optionally masked | dot products | fingerprinting | none | medium | possible |
| hffvenn-gpu `esn_gpu_kernel.rs:529` `esn_main` | echo state network reservoir update | sequential recurrence, matrix-vector | fingerprinting | none | medium | possible: a recurrence problem |
| projection-voting-kmeans `lib.rs:4686` euclidean distance | point-to-centroid distances | matrix product | baseline configuration only | none | high when fused with the next | as #7 |
| `lib.rs:4714` argmin assign, `:4747` cosine assign | nearest centroid per point; the cosine form fuses the 2,048-dimension dot products | argmin, matrix product | euclidean form: baseline only; cosine form: every cycle, the hottest live kernel | none | high | cosine assign is #7 |
| `lib.rs:4808` mean update | centroid means, banded | segmented reduction | every iteration | none | medium | possible |
| `lib.rs:4862` centroid project, `:4895` normalise keys | project centroids, normalise | small matrix product, elementwise | every iteration | none | low | no |
| `lib.rs:4945` inertia contribution, `:4980` block reduce | per-point inertia and its fixed-order block sum | reduction | every iteration | none (fixed order, no atomics) | medium | possible |
| `lib.rs:5021` block histogram, `:5059` cluster offsets, `:5110` scatter ids | counting sort of points by cluster | histogram, scan, scatter | every cycle | none; mode 1 of cluster offsets is a single-thread prefix sum | high | yes, #8 |
| projection-voting-kmeans `gpu_kmeans.rs:562,614,664,733,802` `compute_distances`, `assign_points`, `update_centroids`, `compute_indexed_distances`, `select_top_k` | dense k-means and top-k | matrix product, argmin, reduction, top-k | per iteration | none | high | as #7; `update_centroids` does O(n·k·d) work for an O(n·d) job |
| `gpu_pruning.rs:1874,1982,2098` `compute_tile_assignment`, `select_per_tile_topk`, `validate_boundary` | tile pruning for k-means | gather, top-k | per iteration | none | medium | possible |
| `gpu_chunk_shaders.rs:658,725` `energy_chunk_main`, `angle_chunk_main` | per-chunk energy and angle statistics | reduction | per chunk | none | medium | possible |
| trellis `template_gpu.rs:23,35` `single_main`, `pairs_main` | template scores, single and paired | elementwise, gather | per query | none | low | no |
| udv23-umap `ultradim/gpu_shaders.rs:296,376` `energy_ensemble_main`, `cosine_ensemble_main` | ensemble energy and cosine | reduction | per build | none | medium | possible |
| udv23-umap `gpu_fingerprint_search.rs:252,328` `calculate_cosine_similarities`, `select_top_k` | cosine search and top-k | dot products, top-k | per search | none | medium | as #8 |
| udv23-umap `sparse_mf/lora_gpu.rs:687–858` seven kernels (`FORWARD_G`, `PAIR_COEFF`, `WEK`, `GRAD_A`, `GRAD_B`, `SGD_A`, `SGD_B`) | a sparse matrix-factorisation trainer (BPR, a ranking loss over pairs of items): forward, coefficients, gradients, updates | gather, scatter, small matrix products | every training step | none | medium to high on the gradients | possible: needs a training-loss tolerance, not exact output |
| udv23-umap `umap/kernels_fuzzy.rs:72,169,466,500,539,587,634,673` (`SMOOTH_KNN_SIGMA`, `EDGE_WEIGHTS`, block histogram, cluster offsets, scatter ids, block reduce max, symmetrise combine, epsilon map) | UMAP fuzzy graph: per-point bandwidth by bisection, edge weights, counting sort, symmetrisation | search, histogram, scan, scatter | per UMAP build | none | medium | possible |
| udv23-umap `umap/kernels_optimize.rs:400,547,608,868,1005,1065` (`umap_epoch_gather` ×2, `umap_ce_contrib` ×2, `block_sum`, `mix64_kat`) | UMAP layout epochs as gathers, cross-entropy terms, block sums, a hash self-test | gather, reduction | every epoch | none | medium | possible: tolerance on the layout's loss |
| hff-zenkai `shaders/*.wgsl` (30 entry points: NSGA-II dominance, fronts, crowding, sorting; fitness angles; normalisation) | multi-objective ranking for HFF | sort, scan, histogram, pairwise dominance | not compiled in production; most entry points are stubs or never dispatched | workgroup arrays, `u32` atomics | high; the radix kernels are not dispatched and `radix_scatter` is unstable | runner-up: `nsga2_dominance` is bit-exact; `normalize_to_hypersphere` rescans all N points per thread |
| udv23-umap `synthetic/gene/gpu_navigation_shaders.rs` (4) | copy of Blue World's navigation kernels | | | | | no: duplicate |
| qdrant `scripts/bluenoise_alt/main.rs` (1) | an experiment script | | | | | no |

## Fitness evaluation notes

What each top-ten problem needs, following the Gram design: a deterministic
input generator (seeded, as `gram.rs::phi`), a reference, a correctness test,
three timing sizes (one small, one production, one larger, the shipped kernel
timed at each), and paired timing rounds (at least five; one round scattered
equal kernels by 15 %, `gram.rs` `FitnessContext::timing_rounds`).

1. **Expression evaluator.** Inputs: populations exported from real fits
   (`EVOLVE_EXPORT_POP_DIR`, already 601 MB of them in
   `phylu/logs/REUSE_predprey1/pop/`) and the matching SRBench data. Reference:
   the shipped `eval_main` output, bit for bit, which also fixes the f32
   semantics of the protected operators. Timing: 800 × 3 genes over the fit's training rows;
   5,000 + 5,000 × 3 genes; a long-expression population near 128 nodes.
2. **Scoring.** Inputs: recorded `preds`, `y` and `combinations` buffers from a
   fit. Correctness: first, scores bit-identical (cheap); then, for a
   candidate that changes summation order, identical selection over a
   recorded fit replay (phylu `scripts/wgsl_override_sweep.sh` already
   overrides a kernel in a real fit). Timing: the three population sizes
   of #1.
3. **Tournament selection** and 4. **variation.** Reference: `vary.rs`, bit for
   bit, as `device.rs` already tests over 20 generations. Inputs: a recorded
   population, fitness and cohort labels; the island table. Timing: 800,
   10,000 and 200,000 rows. These four kernels share a binding layout and
   should be one problem with four entry points, or four problems with one
   driver.
5. **Column maximum.** Reference: a host maximum over recorded scores, exact.
   Timing: 800, 10,000 and 200,000 rows × 45 candidates. The simplest
   harness in the list, and a good first one after Gram.
6. **Regex match.** Reference: `regex_match::is_full_match`. Inputs: compiled
   programs from a regex fit and its text rows. Timing: the regex fit's
   population, and twice and ten times its text.
7. **Cosine assign.** Reference: f64 dot products and argmax, ties to the
   lower index, an allowance where the top two are within 1e-6. Inputs:
   random unit keys of 2,048 dimensions. Timing: N = 100,000 with C = 1,000,
   8,000 and 64,000.
8. **Counting sort.** Reference: a stable CPU counting sort, exact. Inputs:
   random labels, labels concentrated in a few clusters. Timing: N = 1,000,000
   with C = 1,000 and 64,000.

A–C. **UltraDim search.** References: the Rust functions themselves.
   Acceptance: the identical top-k set and order, ties broken by id (row B:
   bit-identical projection). Shapes, from ultradim-hotpath's report: 150, 400
   and 2,000 candidates × 2,048-dimension keys; 1,536-dimension dense vectors;
   the ChEMBL and MovieLens sparse non-zero counts; query batches of 1 and 64.

## Can fuller's own work run on the GPU?

Partly, and part of it already does. Bounded, rule-template rewriting is on
the device today: `src/lint/kernel.wgsl` (`lint_main`) applies the linter's
rules to one expression per thread in greedy rounds over a fixed scratch, a
transcription of `src/lint/flat.rs`, and the snap kernels search, graft and
check constants without leaving the device. The decoder and the expression
evaluator are on the device in phylu. What is not plausible as a kernel is
open-ended e-graph saturation: its memory grows without a bound known in
advance, the rule families are non-confluent and explode when combined
(CLAUDE.md), and the population DAG measurements show the shapes involved
(288,918 subtree occurrences hash-consed to 37,345 on the host; a kernel of
6,836 shared definitions takes 8.9 s to compile cold; a compute pass costs
about 22 µs, `docs/PLAN_population_dag.md` §2–2b). Hash-consing a population
and evaluating its shared definitions by level is a plausible GPU job (it is
variant B of that plan); saturation stays on the host. The safe reader runs
naga, a host library, so it stays on the host.
