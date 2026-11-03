# Review: the plan to evolve a better k-means (advisor, round 1)

*fuller, 2026-11-03 13:05Z, branch `feat/futhark-kingdom` at 5c34ca49. Read in full: `docs/PLAN_futhark_kingdom.md` 4a-4c,
`docs/KMEANS_FITNESS.md`, `src/futhark_sig/kmeans_converged.rs`, `src/fitness.rs`, `docs/kmeans_seed.md`,
`src/futhark_sig/tests_kmeans_seed.rs`, `tests/corpus/futhark_problems/kmeans/` (with the new `dbpedia/`),
UltraDim's `gpu_kmeans.rs` (the loop 375-425, the three shaders 555-694, `calculate_inertia` 1794), `quality.rs` and
`lib.rs` `cosine_inertia`. Read-only; the measurements below are CPU scripts and the existing release test binary
(the GPU was not touched; another session's `futhark rust-wgpu` compile was running throughout).*

## Verdict

The seed is a faithful translation of UltraDim's three kernels and the data is now the right data, but the fitness as
built cannot be run: it has an attractor at a program that does nothing, and it ends the run with an error on the
first mutant that makes the inertia worse. Measured on the committed dbpedia sample, the identity step (return the
centres, one node) scores [0.63, 0.00, 0.02, 0.09, NaN] against the seed's [0.5, 0.5, 0.5, 0.5, 0.5]; the NaN, or any
negative "gain per iteration", fails `fitness::converged`'s range check and the `?` in the generation loop stops the
run. Three of the five inputs reward fewer iterations and say the same thing, the angular-percentile input is a
hundred times steeper than the cosine input and is dominated by clusters of one point, and the clock measures the
harness (the host's own assignment pass, 0.35 to 5.6 s an iteration at 5,000 x 1,536) rather than the candidate. The
work guard refuses every dbpedia-sized case before any of this is reached. None of these is deep: the fixes are a
handful of lines each, listed below, and the measurements to re-run afterwards are stated. Until they are made, no run.

## Findings, most severe first

### F1. The fitness rewards doing nothing, and crashes before it can reward it

`kmeans_converged.rs` 63-79 (`kmeans_inputs`). Measured with UltraDim's loop on 2,000 rows of the dbpedia sample, k 32,
starting centres the first 32 rows (`measure.py`, section 2):

| candidate step | iterations | cosine inertia | mean log10 p | inputs [cos, log10 p, seconds, iterations, rate] |
|---|---:|---:|---:|---|
| the seed (A/A) | 20 | 0.14209 | -447.7 | [0.500, 0.500, 0.496, 0.500, 0.500] |
| identity: return the centres | 2 | 0.24554 | -4053.3 | [0.633, **0.000**, **0.024**, **0.091**, **NaN**] |
| centres rounded to 1e-3 | 17 | 0.14219 | -447.5 | [0.500, 0.594, 0.457, 0.459, 0.460] |
| centres rounded to 1e-2 | 23 | 0.14900 | -433.0 | [0.512, 1.000, 0.536, 0.535, 0.548] |
| mean then unit length (spherical) | 24 | 0.14178 | -448.4 | [0.499, 0.159, 0.545, 0.545, 0.568] |
| half step towards the mean | 43 | 0.14187 | -448.3 | [0.500, 0.209, 0.678, 0.683, 0.682] |
| over-relaxed, 1.5 x the step | 30 | 0.14165 | -448.8 | [0.499, 0.070, 0.596, 0.600, 0.599] |

- The identity stops at iteration 2 (the inertia does not change), so its seconds and iterations inputs are near 0,
  and its gain per iteration is 0: `ratio(seed.gain, 0)` is `seed.gain / f64::MIN_POSITIVE` = infinity (line 67
  clamps the denominator, not the ratio), and `squash(inf)` = inf / (1 + inf) = **NaN** (line 68). A candidate whose
  final inertia is above its starting inertia (random mutants do this in generation 1) has a negative gain, so the
  ratio and the input are negative. `fitness::converged` (`src/fitness.rs` 358) returns `Err` for an input outside
  [0, 1], and `examples/futhark_evolve.rs` 1695 and 2379 propagate it with `?`: **the run stops**. The existing test
  (`kmeans_converged/tests.rs` 16) covers above, below and unfinished; not zero or negative gain.
- Had it not crashed, four of the identity's five inputs beat the seed's. It is one node; every population has it.

### F2. The angular-percentile input is a count of one-point clusters

The identity's mean log10 p is -4053 against the seed's -448 because its 32 centres are data rows: a point exactly at
its centroid has cos = 1, `quality.rs` 170 clamps sin^2 to the smallest positive f64, and `ln_betainc(767.5, 0.5,
2.2e-308)` is about -236,000 for that one point. UltraDim's comment says the clamp exists so a singleton stays finite,
not so it dominates a mean; in UltraDim the number is reported as a lift against a fitted null, never used as an
optimisation target. Here it is an input, squashed as 10^(difference) (`kmeans_inputs` line 72): a mean log10 p worse by
1 gives 0.909, by 0.1 gives 0.557, while a cosine inertia worse by 1 % gives 0.5025. **Measured sensitivity: a hundred
times the cosine input's**, and the easiest way to move it is a singleton cluster (k = 1,024 on 5,000 rows: -4075).
Any candidate that leaves some clusters with one point wins this input outright.

### F3. Three inputs reward fewer iterations, and two of them are the same number

`seconds` is iterations times the step time; `inertia_per_iteration` is (start - final) / iterations, so when two
candidates reach the same inertia it is exactly 1 / iterations. The rounded-to-1e-3 step above converges in 17
iterations instead of 20 at the same cosine inertia and takes 0.46 on three inputs; with no gate and TrueNorth over
the five, it ranks above the seed. "Rate of convergence" as defined is not independent of "iterations"; it is the
iterations input a second time.

### F4. The clock measures the harness, not the candidate

`converge` (`kmeans_converged.rs` 352, 391) times everything from `t0`: the host's `assign` (sequential f32, the
shader's rule), `euclidean_inertia`, one process launch an iteration and a 30 MB write of the points down a pipe each
time (`step_of_exe` 437-446). Measured on this machine (`assign_time.rs`, `rustc -O`):

| n x d, k | host `assign` an iteration | `euclidean_inertia` | the compiled seed's step, `futhark c` (sequential) |
|---|---:|---:|---:|
| 5,000 x 1,536, 64 | 0.350 s | 0.006 s | 0.60 s |
| 5,000 x 1,536, 256 | 1.40 s | 0.005 s | 2.10 s |
| 5,000 x 1,536, 1,024 | 5.62 s | 0.006 s | |
| 8,192 x 1,536, 1,024 | 9.20 s | 0.009 s | |
| 8,192 x 256, 1,024 (the old timing shape) | 0.99 s | 0.001 s | |

On the GPU the step is milliseconds, so the measured seconds are the harness's on both sides and the ratio sits at 1
whatever the candidate does: the speed input has no gradient. On the C path the harness is a third to a half of the
clock. The budget suffers the same way: one candidate at 5 cases x 20 iterations x 2 sides x 3 pairs x (0.35 s + step)
is over 200 s at k 64 before compiling.

### F5. The interleaved seed run is thrown away

`examples/futhark_evolve.rs` 805-806: `let (cand, _seed) = converge_interleaved(...)` then `kmeans_inputs(&cand,
&cj_seed(...))`, the seed's run from start-up (2042-2052). The seed is re-run to convergence beside every candidate,
`pairs` times, and its seconds are not used; the ratio compares a median of today's candidate runs with one seed run
taken at the start. The interleaving costs half of every evaluation and protects against nothing.

### F6. The work guard refuses every dbpedia-sized case

`evaluate_converged` (`futhark_evolve.rs` 784-787) stops with "THE WORK GUARD" when the interpreter's work model of
the candidate on the largest fitness case exceeds `CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT` = 1.7e8
(`src/wgsl/work_guard.rs` 24, a bound for a WGSL kernel on the display GPU). The seed's work is about 100 node
executions per (point, centre, coordinate) cell (measured: 1,486,931 on 300 x 6 x 8; 2.15e11 on 8,192 x 256 x 1,024),
so the guard admits n x k x d under 1.7e6: at d = 1,536, n x k under 1,100. A case of 500 rows and k 2 is the
largest that passes. The converged path never runs the interpreter on the case (the program is compiled), so this
guard is the wrong bound here; it needs its own, derived from the seed's measured step time.

### F7. The cases have not been cut; the pool prompt describes the old data

Commit 5c34ca49 adds `dbpedia/sample.py`, `row_ids.txt` and `sample_5000x1536.in` and nothing else: `cases.tsv` and
`make_cases.sh` still produce the eight uniform cases at scales 10^-3 to 10^5, which the runner and the tests load.
`examples/futhark_pool.rs` 67 still prompts the 25 implementations with "the data span magnitudes from 1e-3 to 1e5, so
accuracy matters ... the inputs here are not unit"; programs written to that prompt optimise for data that no longer
exists. `docs/KMEANS_FITNESS.md` 41-42 ("at one magnitude scale") and the plan's 4c item 3 say the same.

### F8. The seed optimises Euclidean k-means; the fitness measures spherical k-means

The loop is `gpu_kmeans.rs`'s (Euclidean inertia, no normalisation); the quality is `lib.rs`'s cosine inertia with
centroids made unit for the measure only. On unit data the two objectives differ: the step that renormalises its
centroids reaches a lower cosine inertia (0.14178 against 0.14209) and a better percentile (-448.4 against -447.7) in
more iterations (24 against 20). UltraDim's own cosine path (`lib.rs` Pass C) normalises each cycle and stops on the
cosine-inertia delta at 1e-5 (`RotatingKeyConfig::tolerance`). This is not a bug, it is a choice the plan does not
state: the first thing evolution can find is a normalisation UltraDim already has elsewhere. The paper must say which
k-means the seed is and which objective it is scored on, or the loop should stop on the measure it scores.

### F9. The stopping rule on this data is "the assignments stopped changing"

Measured on the sample, 16 runs over (n, k) from (500, 8) to (5,000, 1,024): every run stopped with a last change of
exactly 0 after 4 to 33 iterations; the f32 spacing of the inertia is 1.2e-4 at n = 5,000 (inertia 1,100-1,270) and
6.1e-5 at n = 2,000, so the tolerance 1e-4 can only fire on a bit-identical inertia. This is faithful to the prototype
(UltraDim's own data gives the same spacing) and it is why the seed finishes on every shape. Two consequences to
write down: `iterations` is Lloyd's fixed-point count, which depends strongly on the starting centres (16 against 33
at n = 2,000, k = 32, first rows against random rows), so the starting centres are part of the case; and a candidate
whose output jitters by one ulp between iterations (a non-deterministic GPU reduction) never stops, hits the cap and
is scored 1 on all five, which is "worst", not "slow".

### F10. The seed: faithful, with the one representation decision correctly measured

Checked line by line against the shaders (`docs/kmeans_seed.md` 54-92): the in-order f32 sum of squared differences
with `sqrt`, best initialised to centre 0 and the loop from 1, strict `<` so the first centre wins a tie and a NaN
never replaces, Kahan in the shader's order (`y = x - c; t = s + y; c = (t - s) - y`), the count beside the sum in one
loop, `count > 0` else the previous centre. The spellings through `e < 0` are exact on f32 with gradual underflow; a
GPU that flushes denormals to zero could make `a - b = 0` for distinct a, b below 1.2e-38, which unit data never
produces. The proof (`tests_kmeans_seed`, 8 of 8 cases bit-identical to the Rust port, through the interpreter and
`futhark c`) stands. Holding the assignment and the count in f32: acceptable for this run, and **raising the type-code
limit is not the alternative** — at 53,773 types (master 4, pairs only) the gene at N = 96 was already 62.7 M words and
488 s to lay out; the 237,325-type view at N = 192 would be of the order of a gigabyte a gene, over the 128 MiB
binding whatever the code width. Say so in the paper; the integer view waits for a smaller context set.

Re-measured here: the kingdom loads and lays out at N = 192 in 24.3 s (14,359 types, 110,871 rows, 27,563,026 words
a gene = 110 MB); the e-graph test on the view (3 iterations, 256 members, the 74-function textbook program) and the
layout test together: 184 s of CPU and 5.1 GB peak. The e-graph as a pump on a 131-function seed every generation is
minutes a generation; on the elites only, or every fifth generation.

### F11. The measures: faithful to the formula, not tested against UltraDim's numbers

`exact_mean_log10_p` and `ln_betainc` are UltraDim's line for line (the only difference: `ln_beta` by Lanczos
instead of `statrs`, about 1e-15 relative). `cosine_inertia` sums in order where UltraDim sums in rayon chunks (f32,
different bits; immaterial for a ratio). The test (`kmeans_converged/tests.rs` 56) checks anchors (equator 1/2,
antipode 1, centroid finite) and closed forms, not a value UltraDim's own tests print. One fixture pair (a cosine
vector, d, UltraDim's number) would close that.

### F12. Smaller faults

- `in_fitness` (`futhark_sig.rs` 2117) puts the held cases in the fitness; only `Test` is out. Fine, but the checklist
  (`docs/CHECKLIST_futhark_problem.md` 26) says k-means has no test split while `cases.tsv` has `test_s1_n300_d5_k4`:
  stale either way, and both will be rewritten by the cut.
- `converge` line 353 computes the start inertia with its own `assign` before the clock starts; the step's assignments
  (inside the program) are never read by the harness, so a candidate that assigns by cosine is measured on Euclidean
  assignments to its centroids. Acceptable (the measure is of the centroids) but it should be written down.
- The 100-iteration cap scores a better-but-slower program 1 on all five (`kmeans_inputs` 64-66); on this data the
  seed needs at most 33, so the cap is not binding, but a candidate at 101 is "worst", not "slow".
- `docs/KMEANS_FITNESS.md` 71 says "a difference squashed the same way" for the percentile: the code squashes
  10^(difference), which is a ratio of p. The text should say so.
- `KMEANS_PROGRAM_SIZE` 192: a host population of 64 genes is 7 GB of genes; fine on this machine, impossible on the
  device, so "GPU residency" for k-means is not this view.

## What the data section should decide (the 5,000 dbpedia rows)

Measured on `sample_5000x1536.in`: every norm 1.00000; 3 exact duplicate rows; the "5 consecutive rows" have no
structure (cosine within a run of 5: mean 0.690, between runs 0.690; a row's nearest neighbour is in its own run 0.2 %
of the time, chance 0.08 %). Treat it as 5,000 random rows; the sampling rule costs nothing and buys nothing.

- **Starting centres as UltraDim initialises them:** `GpuKMeansConfig::default()` is k-means++ (`use_kmeans_plus_plus:
  true`), ChaCha8 when `random_seed` is set. Draw them once per case with k-means++ at a fixed seed and store them as
  the case's second input; never the first k rows. The iterations depend on them (F9), and every candidate and the
  seed must start from the same ones.
- **Cases:** disjoint row sets so the held cases hold out rows, with k at most n / 20 so singleton clusters are rare
  (F2): train 1,000 rows k 16; 1,000 rows k 64; 500 rows k 8; held 1,000 rows k 32; 500 rows k 16; test 1,000 rows
  k 32 (never a fitness input). Six cases, 5,000 rows used once. Iterations at these shapes: 7 to 33.
- **The timing case** cannot be 8,192 x 1,536 from 5,000 rows. Use all 5,000 rows with k 256 (n / k about 20, 14-16
  iterations; k 1,024 gives 5 points a cluster, 4-8 iterations and the percentile at -4075, a singleton count).
  UltraDim's k 1,024 belongs with its n 8,192; at n 5,000 keep the ratio, not the k.
- The interpreter is not the path for these cases (about 4e7 node executions a second: a 1,000 x 1,536 x 16 step is
  seconds, the 5,000 x 1,536 x 256 timing case minutes); every control and test over them is the compiled program.
  `random_kmeans_genes_...` and the checklist test need one small case kept (200 x 1,536 x 4 is fine).

## Recommended changes, concrete

1. **`kmeans_inputs`:** replace the five with inputs that are total on [0, 1]: cosine inertia ratio (keep); seconds
   ratio (keep, with change 3); iterations ratio (keep); drop the gain-per-iteration input (F3) or, if a rate is wanted,
   define it so that zero and negative gain are 1, e.g. `squash(seed.gain / max(cand.gain, 0))` with the zero case
   pinned to 1 and a test for it; the percentile input either reported only, or computed over points that are not
   their cluster's only member, stated as such (F2). Add the test: identity step, a worsening step, a NaN-free result
   in [0, 1] for both. `fitness::converged` must never see an out-of-range input from a finished run.
2. **A floor the identity cannot pass:** with no gate (Andrew's ruling), the quality inputs must carry enough weight
   that a candidate at the seed's starting inertia loses. Measure after change 1: the identity, the rounded steps and
   the spherical step through `fitness::converged` against the seed; the identity must rank below the seed. If it does
   not, the quality group needs the two-level HFF (quality first), which is a named new scheme, not an edit.
3. **The clock:** time the step calls only (accumulate around `step(&centres)` in `converge`), and keep the process
   alive: `futhark c --server` / the GPU shim with `--server` (the executable then takes inputs and returns outputs
   without re-launching, so the device context and the 30 MB upload are paid once). Parallelise the harness's `assign`
   over points (the rule unchanged, the per-point sum still in order) so the budget is the candidate's.
4. **The interleaving:** use the interleaved seed's seconds (its quality from start-up is deterministic), or drop
   `pairs` to 1 and time the seed once per generation; never run it and discard it.
5. **The work guard:** on the converged path, replace `CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT` with a bound derived from
   the seed's measured step time on the largest fitness case (e.g. a candidate's work model at most 10 x the seed's),
   so a program of the seed's size on dbpedia-sized cases is admitted and a 100 x slower one is not.
6. **The cases:** `make_cases.sh` cuts the six cases above from `sample_5000x1536.in` with k-means++ starts (a Python
   or Rust step, seeded, recorded in `cases.tsv` with the row ranges); remove the uniform cases from `cases.tsv`; the
   timing input from the same rows; `tests_kmeans::the_magnitude_cases_...` becomes the dbpedia cases' shape test.
7. **The pool prompt** (`futhark_pool.rs` 67) rewritten for unit vectors of 1,536 dimensions and the actual fitness;
   `docs/KMEANS_FITNESS.md` and plan 4c item 3 updated.
8. **State the objective** (F8): either the paper says "Euclidean Lloyd on unit vectors, scored by UltraDim's spherical
   measures" and expects normalisation to be found, or the loop stops on the cosine inertia as `lib.rs` does. One
   sentence in the plan, decided before the run.
9. **The pump:** the e-graph on the elites only, bounded (the measured 2 minutes for a 74-function program at 3
   iterations is a generation's budget by itself); record its time per generation in the stream.

## The run design (item 4)

With changes 3-5 the cost of one evaluation is the compile plus 6 cases x about 20 iterations x the GPU step (ms)
plus the host's assignment (with rayon, well under 0.1 s): of the order of 10-30 s, so 2,000 evaluations is 6-16
hours a seed, not the 60 minutes in `docs/kmeans_seed.md` 187. Population 32-64 as the other problems; 3 seeds for a
range, 5 for a claim; the random-search control after the interesting runs, and dropped if they show nothing; the
WGSL oracle (plan item 4) first, since it is the only check that the GPU's compiled seed is the shader (FMA
contraction on Metal is the open question, `docs/kmeans_seed.md` 125-129). Nothing on the GPU besides.

## For the paper's claim (item 5)

- The Reader reads none of the 25 implementations today: `src/futhark_reader.rs` records a tuple and a float literal
  as gaps; `kinds.rs` (447 lines, untracked) is the start of the f32 reading and is not committed. Until the Reader
  reads f32, tuples and `let`, the pool is the seed and the textbook step, and the protocol's "25 high-perplexity
  implementations" is not run.
- Fable's implementation must be a named arm: the same prompt as the other 24, read by the same path, (a) a member of
  the pool and (b) the seed of its own run, so "beats Fable's" is measured twice: the champion against Fable's program
  on the test case under the same inputs relative to the prototype, and the run from Fable's seed against the run from
  the prototype's.
- Statistics: 5 run seeds, the champion of each scored on the test case, reported as the five values and the count of
  seeds whose champion beats the arm (as `kmeans_champion_s1/README.md` reports "11 of 11 pairs"); the A/A spread of
  the scheme (the runner's in-run A/A) as the noise floor beside them. Three seeds give a range, not a claim.

## What to measure, in order

1. After change 1: the table of section F1 through `fitness::converged` (the HFF angle, not the inputs): the seed, the
   identity, both rounded steps, the spherical step, a worsening step. The identity must be below the seed.
2. After changes 3-5: one evaluation's wall clock on the six cases, split into compile, step, harness; the in-run A/A
   (the seed against itself) on the GPU, the spread of the seconds input; the seed's iterations a case.
3. The WGSL oracle: the seed's assignments and centres on the six cases against UltraDim's shaders on the GPU; the
   centres' distance from the f64 reference on both.
4. One generation of the pump on the elites: seconds and memory at the seed's size.
5. Then the run: 1 seed first, read the stream at 200 evaluations for the three operator rates and whether any
   candidate beats the seed on quality (not only on iterations); then the other seeds; then the control.

Scripts and logs: `/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/1a542778-6355-45e6-9c87-a491851d5eb3/scratchpad/km/`
(`measure.py`, `assign_time.rs`, `egraph_test.log`, `measure.log`).
