# Review: the k-means fitness after round 1 (advisor, round 2)

*fuller, 2026-11-03 14:30Z, branch `feat/futhark-kingdom` at 92922521 (the committed state; the working tree held
four small uncommitted edits by the agent, named below). Read in full: round 1 and the commits 03a03df4, b7182a1f,
82f17171, 92922521; `docs/KMEANS_FITNESS.md`; `src/futhark_sig/kmeans_converged.rs` and its tests; `src/fitness.rs`;
the converged judge in `examples/futhark_evolve.rs`; `tests/corpus/futhark_problems/kmeans/`; plan 4c; hff_core's
TrueNorth. Read-only; the measurements are the kingdom's own tests (release, CPU, 56 s) and Python scripts over the
stored cases, with the Rust port's arithmetic reproduced to the printed digit (the seed on `train_n500_k8`: 9
iterations, cosine inertia 0.15227, mean log10 p -425.78, the identity's inputs 0.6382/0.6266, in both). The GPU
was not touched. The runner was not run on the C path: the CPU was saturated by the agent's own full `cargo test`
and another session's rust-wgpu builds (load 27-35) and a third build would have broken the one-build rule.*

## Verdict

Eleven of the twelve findings are fixed, and the fixes measure as claimed: every input is total on [0, 1], the clock
is the program's own (0.026 s of 0.66 s wall on the small case), the percentile no longer counts singletons, the
cases are the dbpedia rows with UltraDim's k-means++ starts and the seed converges on all seven in 9-16 iterations
with a last change of exactly 0. Two things stop a run. First, `cargo test` is red at the head: the F1 test fails
at its own line 67, because `squashed_ratio(NaN, 1.0)` is 0, the best score, not 1 (`f64::max` drops the NaN), and
its line 83 asserts a NaN the retired inputs do not produce at the numbers it chose. Second, the fitness has to be
written again under Andrew's ruling of today: the HFF angle is the final fitness, nothing manipulates it, so the
`π/2 + θ_quality` offset of `futhark-kmeans-quality-first` is a fault, and the scheme becomes one flat vector with
each case's quality inputs repeated. The measurement below says what repetition handles the identity, the worsening
and the NaN steps (two; three for margin) and what no repetition handles (an early stop that is 1-2 % worse). The
committed gate, for the record, had a cliff at exactly 1/2 that rejected every reordering of the seed's sums.

## The twelve findings

| finding | disposition | measured on the committed code |
|---|---|---|
| F1 the fitness rewards doing nothing and crashes | fixed: `squashed_ratio` is total, a run that does not finish is 1 on all four, the run never stops on a candidate's value. **But the head's own test is red** (`the_inputs_are_half_...`): line 67 expects `squashed_ratio(NaN, 1.0) == 1.0` and gets 0.0 (the agent's uncommitted edit adds the NaN guard); line 83 expects `kmeans_inputs_five`'s rate to be NaN on no gain, but with the seed's gain 1.0 the ratio is 1/MIN_POSITIVE = 4.5e307, finite, and squashes to 1.0 (the agent's uncommitted edit changes the seed's gain to 10 so it overflows). Both must be committed before anything else. | identity 2.498 against the seed 0.663 under the committed scheme; a worsening 2.886; a NaN 3.142; no error |
| F2 the percentile counts singletons | fixed differently, acceptable: the input is the mean over the points not at their centroid (`sin^2 < 1e-4`), UltraDim's number over every point reported beside it | identity: -254 off centroid against the seed's -426 (input 0.627); -3,106 over every point. `train_n1000_k50` has one singleton cluster (sizes 1..48): -467 off centroid, -703 over all |
| F3 three inputs reward fewer iterations | fixed: four inputs in two groups, the gain-per-iteration reported only | |
| F4 the clock measures the harness | fixed: `futhark c`'s `-t`, the shim's GPU timestamps; the harness's assignment parallel over the points | the compiled seed on 200 x 1,536 x 4: 13 iterations, the program's own 0.0262 s of 0.6625 s wall; a step 2 ms of 295 ms |
| F5 the interleaved seed run is thrown away | fixed: `converge_interleaved` returns the seed's run of that interleaving and `evaluate_converged` scores against it. Cost unchanged: the seed is re-run beside every candidate on every case (half of each evaluation); with `--pairs 1` the candidate always runs first, the seed always second | |
| F6 the work guard refuses every case | fixed differently, sound: the cap is 10 x the seed's own work on each fitness case, a relative bound, so every stored case is admitted by construction; a candidate over it is not run and scored worst, counted, never a stop | |
| F7 the cases were not cut | fixed: six disjoint cuts plus the small one, k-means++ starts stored, the pool prompt rewritten for unit embeddings and the fitness | the stored bytes equal the recomputed cuts (the kingdom's test); below, the seed on every case |
| F8 Euclidean loop, spherical score | fixed: stated in the specification and the module header; the normalised step measured | normalised centroids on `train_n500_k8`: 0.15231 against 0.15227 (input 0.5001: not a win here); on `held_n1000_k32` 0.14077 against 0.14083 (0.4999, a hair better, 12 iterations against 10) |
| F9 the stopping rule is "assignments stopped changing" | fixed: stated; the cap is slow, not worst | re-measured on all seven stored cases: every run stops with a last change of exactly 0 (table below) |
| F10 the seed and the pump's cost | not re-measured. The P regime's start runs the e-graph (3 iterations, 32 members) and the saturated fold on the 131-function seed before the first evaluation; round 1 measured 184 s and 5.1 GB for the 74-function textbook program. Measure once on the C path (`--backend c --eval single --evaluations 1`) before the GPU run | |
| F11 the measures not tested against UltraDim's numbers | fixed: vendored copies, bit-identical (804 percentiles; the cosine inertia on one thread) | |
| F12 smaller faults | fixed: the cap is a slow finished run; the harness's Euclidean assignment stated; the checklist's k-means column rewritten; the ratio wording; the program size noted | |

## The seed on every stored case (item 3)

The Rust port's loop in Python (same bits as the kingdom's test on `train_n500_k8`), from the stored starts:

| case | n | k | iterations | last change | f32 spacing of the inertia | cluster sizes | empty | at centroid | cosine inertia | mean log10 p off / all |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| train_n1000_k16 | 1,000 | 16 | 11 | 0 | 3.1e-5 | 32..119 | 0 | 0 | 0.14803 | -434.95 / -434.95 |
| train_n1000_k50 | 1,000 | 50 | 16 | 0 | 1.5e-5 | 1..48 | 0 | 1 | 0.13364 | -467.19 / -702.85 |
| train_n500_k8 | 500 | 8 | 9 | 0 | 1.5e-5 | 32..123 | 0 | 0 | 0.15227 | -425.78 / -425.78 |
| train_n200_k4 | 200 | 4 | 10 | 0 | 3.8e-6 | 9..85 | 0 | 0 | 0.15710 | -416.28 / -416.28 |
| held_n1000_k32 | 1,000 | 32 | 10 | 0 | 3.1e-5 | 5..68 | 0 | 0 | 0.14083 | -450.89 / -450.89 |
| held_n500_k16 | 500 | 16 | 11 | 0 | 1.5e-5 | 5..60 | 0 | 0 | 0.14571 | -439.76 / -439.76 |
| test_n1000_k32 | 1,000 | 32 | 11 | 0 | 3.1e-5 | 5..104 | 0 | 0 | 0.14079 | -450.96 / -450.96 |

The sizes, splits and starts are right; the stopping rule behaves (Lloyd's fixed point in 9-16 iterations, no empty
cluster). One singleton cluster appears at k 50 despite k at most n / 20; the off-centroid percentile absorbs it
(the whole-point number is -703, the fitness's -467). The timing input (5,000 rows, k 256) is checked by the
kingdom's test; it is not a fitness case and was not re-run here. The budget per evaluation on the compiled path:
67 iterations over the six fitness cases, times two sides, times one process of 0.3-0.6 s, plus the compile: of the
order of one to two minutes a candidate, as the specification says.

## The gate (item 2): Andrew's ruling, and what the measurement says about it

**The ruling (relayed by the coordinator during this review):** the HFF angle is the final fitness and nothing
manipulates it afterwards; the `π/2 + θ_quality` offset in every "gated" scheme is a fault; no gate, no stage. To
weigh quality above speed the quality coordinates are repeated in one flat vector `(quality, quality, speed)`: one
level, each case's quality inputs entered m times and its speed inputs once, the HFF over the whole vector.

**What the HFF does with that.** TrueNorth is `θ = acos(1 − mean(x²))` (hff_core, read today), so the angle is a
monotone function of the mean square of the vector and nothing else. The seed is 1/2 on every input, mean square
1/4, θ = 0.7227 at every m. A candidate beats the seed exactly when, summed over the fitness cases,
`m · (Σq² − ½·cases) < ½·cases − Σs²`: m is the exchange rate between a quality loss and a speed gain. The committed
inputs (`kmeans_inputs`, unchanged) go into it as they are.

**Measured** (`flat.py`, the committed inputs of each step on `train_n500_k8` and `held_n1000_k32`, m = 1..8; the
seed 0.7227; `*` ranks above the seed):

| step | quality inputs (k8 / k32) | speed inputs | m=1 | m=2 | m=3 | m=4 | least m that ranks it below the seed |
|---|---|---|---:|---:|---:|---:|---|
| identity | 0.638, 0.627 / 0.632, 0.617 | 0.000, 0.18 / 0.000, 0.17 | 0.657* / 0.647* | 0.756 / 0.745 | 0.801 / 0.790 | 0.827 / 0.816 | **m > 1.56 / 1.68** |
| worsening | 0.814, 0.912 / 0.823, 0.911 | 0.000, 0.25 | 0.914 / 0.915 | 1.057 / 1.060 | 1.125 / 1.129 | 1.164 / 1.168 | m > 0.44 |
| NaN at iteration 3 | 1, 1 | 1, 1 | 1.571 | 1.571 | 1.571 | 1.571 | every m |
| rounded 1e-2 | 0.511, 0.508 / 0.512, 0.508 | 0.56, 0.57 / 0.70, 0.66 | 0.782 / 0.875 | 0.768 / 0.831 | 0.760 / 0.809 | 0.756 / 0.795 | every m (worse and slower) |
| rounded 1e-3 | 0.5002, 0.5001 / 0.5001, 0.5001 | 0.50, 0.50 / 0.51, 0.47 | 0.723 / 0.718* | 0.723 / 0.720* | 0.723 / 0.721* | 0.723 / 0.721* | k8: every m; **k32: m > 65** (9 iterations against 10 at 0.02 % worse) |
| normalised centroids | 0.5001, 0.5000 / 0.4999, 0.4999 | 0.50, 0.50 / 0.60, 0.55 | 0.723 / 0.781 | 0.723 / 0.762 | 0.723 / 0.752 | 0.723 / 0.747 | below at every m (not faster) |
| plain f32 sum (same partition) | 0.5000, 0.5000 | 0.62, 0.50 / 0.66, 0.50 | 0.774 / 0.789 | 0.757 / 0.767 | 0.749 / 0.756 | 0.744 / 0.750 | below (slower in Python) |
| pairwise f32 (same partition) | 0.5000, 0.5000 | 0.22, 0.50 / 0.30, 0.50 | 0.643* / 0.660* | 0.670* / 0.681* | 0.684* / 0.692* | 0.692* / 0.698* | above at every m (faster, equal) |
| faster, same bits (seconds / 2) | 0.5, 0.5 | 0.333, 0.5 | 0.669* | 0.687* | 0.696* | 0.702* | above at every m |
| damped 0.5 / 0.2 / 0.1 | 0.4999-0.5001 | 0.71-0.92 | 0.92-1.10 | 0.85-0.99 | 0.82-0.93 | 0.80-0.89 | below at every m (slower, no better) |
| damped 0.05 / 0.02 / 0.01 / 0.001 (hit the cap) | 0.501-0.623 | 0.87-0.92 | 1.09-1.16 | 0.98-1.08 | 0.92-1.03 | 0.88-1.00 | below at every m |
| one seed step, then identity | 0.506, 0.504 / 0.508, 0.506 | 0.09, 0.25 / 0.07, 0.23 | 0.546* / 0.541* | 0.613* / 0.611* | 0.644* / 0.643* | 0.662* / 0.662* | **m > 44 / 32** |
| two seed steps, then identity | 0.502, 0.502 / 0.504, 0.502 | 0.18, 0.31 / 0.12, 0.29 | 0.569* / 0.555* | 0.625* / 0.617* | 0.651* / 0.646* | 0.667* / 0.663* | m > 95 / 69 |
| four seed steps, then identity | 0.5008, 0.5005 / 0.5009, 0.5006 | 0.31, 0.40 / 0.21, 0.38 | 0.627* / 0.594* | 0.660* / 0.640* | 0.677* / 0.662* | 0.686* / 0.675* | m > 193 / 206 |

Pooled over the two cases (the run's vector is every fitness case's inputs), the steps that beat the seed are: at
m = 1 the identity, rounded 1e-3, the two equal-partition reorderings, the three early stops and the faster seed;
at m = 2 and every m up to 8 the same list without the identity.

**What the measurement supports.**

- **m = 2 is the least repetition that handles the identity, the worsening and the NaN step; m = 3 is the
  recommendation.** At m = 2 the identity sits 3-4.5 % above the seed (0.756 and 0.745 against 0.7227), inside what
  a case with a better k-means++ start could erode; at m = 3 it sits 10 % above, at m = 4 14 %. The worsening and the
  NaN rank below at every m; the rounded-to-1e-2 step at every m. Nothing is bought back by the GPU's zero seconds
  that is not already in the table (the identity's seconds input is 0.0002 here).
- **Repetition is a price, not a rule, and the measurement says what the price buys.** A step that is 1-2 % worse
  in the clustering and stops in a third of the iterations (the "one seed step, then identity" row) ranks above the
  seed up to m = 32-44; a step 0.02 % worse that saves one iteration ("rounded 1e-3" on k32) up to m = 65. No
  repetition below that makes quality dominate. A gene cannot code "then identity" (a step has no iteration
  counter), but it can code a thresholded move (a `sel` on the size of the change per coordinate), which is the same
  shape: an early stop at a slightly worse partition. Under the ruling that is the problem's own exchange rate and
  the run will show whether evolution takes it; the report must print the champion's quality inputs beside its
  speed inputs so a 2 %-worse-but-faster champion is read as what it is, not as a better k-means.
- The gradient exists from the first generation: every step ranks by its own mean square, nothing is clamped or
  offset, and the ordering of better-faster > seed > slower > worse-and-slower > unfinished holds at every m.
- **The committed gate must go anyway, and not only for the ruling.** Measured on the six fitness cases
  (`edge.py`): three programs that reach the seed's exact partition on every case by another order of additions (the
  plain f32 sum, the f64 mean, numpy's pairwise f32) land within ±4.9e-8 of 1/2 on the cosine input and ±1e-8 on
  the percentile input, on either side at random; 7 of the 18 (case, variant) runs land above 1/2, and each of the
  three fails on at least one fitness case. Under `converged_gated` ("every quality input at most 1/2 on every
  case") every one of them scores π/2 + θ_quality = 2.29, below a half step (0.87) and below a program that is
  genuinely worse by 20 %. The reorderings are the first speed class evolution can find; the gate as coded rejected
  the whole class. The specification's sentence "whichever lands a hair better passes, the other does not" described
  this and called it an edge; it is a cliff across the main road.
- The scheme needs a NEW name (the fitness rule): `futhark-kmeans-quality-first` is retired with its fault stated
  (the offset manipulates the HFF; the cliff at 1/2), and the flat scheme is registered beside it with its
  repetition count in its name or its description, e.g. `futhark-kmeans-flat3`. The runner's `converged_fitness`,
  `converged_widths`, the objective names, the in-run A/A line and the report's `best_quality_inputs` /
  `best_speed_inputs` all carry the two-group shape and must be changed with it; there is no `θ_quality` to print.
- One level with no repetition (m = 1) is the retired five-input scheme's shape with better inputs; it is not enough:
  the identity beats the seed on both cases (0.657 and 0.647 against 0.723). The two-level ungated angle is the same
  story (identity 0.611 and 0.601). Neither should be registered.

## New findings

- **N1. The head does not pass its own tests** (F1 above). `RUSTFLAGS="-D warnings" cargo test` is the rule before
  any commit; 92922521 breaks it on `the_inputs_are_half_at_the_seed_...`. The working tree's four uncommitted edits
  (`squashed_ratio` returns 1 on a NaN on either side; the test's seed gain 10; `fitness/tests.rs` lists the new
  definition and the retired name; the specification's copy of the function) fix it; they were not committed when
  this review was written.
- **N2. The test case is never run.** For the k-means scheme the judge runs the fitness cases only; the report's
  `best_exact_test` is an exactness count, 0 of 1 by construction, and nothing scores the champion or the seed on
  `test_n1000_k32`. The paper's claim (round 1, item 5) is the champion against the seed on the test case under the
  same inputs; the runner must run both on it at the end (and the champion each generation, for the learning curve)
  and print the four inputs and the raw measures.
- **N3. The seed's own speed noise is the floor of the flat scheme.** The seed run again with the same bits scored
  0.709 and 0.736 against 0.7227 on the two cases from a ±8 % jitter of the Python step's seconds; the rounded-1e-3
  and the normalised steps differ from the seed by less than that. On the GPU the in-run A/A (the seed as a
  candidate) gives the floor; a champion's claim needs its margin above that floor, as the timed-pairs schemes
  require (`speed_claim_beats_noise_floor`). The k-means report has no such line yet.
- **N4. `--pairs 1` is not an interleaving.** With one pair the candidate always runs first and the seed second on
  every case; the specification's "neither side always runs warm after the other" holds only from two pairs. On the
  C path the `-t` time excludes the process start, so the order costs little; on the GPU the first process of a
  freshly compiled candidate pays its pipeline creation inside or outside the timestamps depending on the shim: the
  in-run A/A must show it (a seed-as-candidate seconds input away from 1/2 is this).
- **N5. The P start's cost is unmeasured on the seed** (F10): the e-graph variants and the saturated fold of a
  131-function program run before the first evaluation, every `--pump-every 5` generations after. Run
  `--backend c --eval single --evaluations 1` once and read the start's seconds and memory before the GPU run.
- **N6. The docs index.** A new review file makes `docs/index.html` stale and `cargo test` red until
  `python3 docs/build_index.py` is run with its write option; this review commits only its own file, so the agent
  regenerates the index with the next commit (round 1 did the same, f377bfee).

## The run design for the first run (item 4)

After N1, the flat scheme, N2 and N5 are in and `cargo test` is green:

1. The GPU controls (`/private/tmp/claude-501/fbench/kmeans_gpu_controls.sh`): the compiled seed converges on the
   six fitness cases on the GPU with the program's own milliseconds, and the in-run A/A gives the seed's quality
   inputs at exactly 1/2 (a deterministic device) and the spread of its seconds inputs, the floor of N3. A quality
   input of the seed against itself off 1/2 is a non-deterministic reduction and stops here: it is written down
   before any run.
2. The run: regime P from the seed, population 24, `--pairs 1`, elites 2, the pump every 5 generations on the
   elites, 400 evaluations (about 17 generations; at one to two minutes a candidate, 7-13 hours), `--minutes 900`
   as the guard, seed 1; then seeds 2 and 3 for a range. The random-search control after, and only if the runs show
   something. The command, once the scheme is registered under its name:
   `futhark_evolve --problem kmeans --regime P --start published --fitness futhark-kmeans-flat3 --eval single --backend gpu --pairs 1 --pop 24 --elites 2 --pump-every 5 --evaluations 400 --minutes 900 --seed 1 --out /private/tmp/claude-501/fbench/kmeans_P_s1`,
   with `evowatch` on the output directory.
3. What the report shows per generation (`series.tsv`, the stream): the best HFF (learning is this going down from
   the seed's 0.7227), the median HFF, the champion's four inputs on every fitness case and its raw measures beside
   the seed's (cosine inertia, mean log10 p off centroid and over every point, the program's seconds, iterations),
   the counts of candidates not run (over the work cap) and not finished, and each operator's rate of children that
   beat their parent (the mutation events). At the end: the champion and the seed on `test_n1000_k32`, the same
   five numbers, and the champion's seconds margin against the A/A floor. The chart: HFF against generation.

## Scripts and logs

`/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/1a542778-6355-45e6-9c87-a491851d5eb3/scratchpad/km2/`:
`measure2.py` (the loop, the measures, the inputs and the three rules; `measure2.log`), `edge.py` (the gate's edge;
`edge.log`), `flat.py` (the flat vector at m = 1..8; `flat.log`), `kmeans_converged_tests.log` (the kingdom's nine
k-means fitness tests, release, 8 passed and 1 failed).
