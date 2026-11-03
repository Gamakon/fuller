# The k-means fitness: the measures that go into the HFF

*fuller, 2026-11-03. **The scheme to run is `futhark-kmeans-better`** (Andrew's specification of 2026-11-03,
below, first): the first k-means run (`/private/tmp/claude-501/fbench/kmeans_qqs_s1`, scheme `futhark-kmeans-qqs`)
did not produce a better k-means, because its fitness rewarded stopping early. `futhark-kmeans-qqs`, specified after
it, is kept registered and is **NOT TO BE RUN AS IS**, with the fault "rewards early stopping: r/(1+r) compresses a
quality loss". The schemes before it (`futhark-kmeans-quality-first`, `futhark-kmeans-converged`) are NOT TO BE RUN
AS IS either (their faults are below). Andrew's ruling of 2026-11-03 holds for every scheme here: **the HFF angle is
the final fitness, nothing may manipulate it afterwards; quality weighs more than speed by repetition in one flat
vector.***

## The scheme `futhark-kmeans-better` (the default for k-means)

**Why it replaces `futhark-kmeans-qqs`.** The best individual of the first run stops after 3 iterations with a 16 %
worse cosine inertia and 31 of 32 clusters holding one point. Under `r/(1+r)` a 16 % loss moved the inertia input
only from 0.500 to 0.537, while stopping after 3 iterations put its iterations inputs at 0.16-0.25 and its seconds
inputs at 0.003-0.013 (the run's `report.tsv`): the fitness traded a large quality loss for speed.

**The run.** On each fitness data set, the candidate program and the reference program (UltraDim's centroid update,
translated) each run from the same starting centroids under UltraDim's stopping rule (the Euclidean inertia changes by
less than 1e-4, or 100 iterations), applied by the harness, interleaved `--pairs` times; the reference's run of the
same interleaving is the yardstick. Everything about the loop, the measures and the cases is as described in the
sections after this one.

**The quality inputs of a data set**, each from `x` = the candidate relative to the reference (`x = 1` equal, `x > 1`
worse):

1. the final cosine inertia: `x = I_cand / I_ref`;
2. the final mean log10 angular percentile over the points off their centroid (UltraDim's `exact_mean_log10_p`, the
   variant described below): `x = L_ref / L_cand`;
3. the convergence: `k` = the further iterations UltraDim's rule needs when the REFERENCE program's update is run on
   from the candidate's final centroids, `k_ref` likewise from the reference's own final centroids; `x = 1 + (k −
   k_ref)`, so `x ≤ 1` when the candidate's centroids are at least as converged. At a fixed point `k = 2` (the
   first iteration has no previous inertia to compare with).

Each `x` becomes an input on [0, 1] by the cliff (`fitness::quality_cliff`):

```rust
pub fn quality_cliff(x: f64, cliff: f64) -> f64 {
    if x.is_nan() {
        return 1.0;
    }
    if x <= 1.0 {
        (x / 2.0).max(0.0)
    } else {
        0.5 + 0.5 * ((x - 1.0) / cliff).min(1.0)
    }
}
```

with `ε` = `--quality-cliff`, default 0.005, recorded in the run card (`quality_cliff`) and the stream's `run_start`
(`fitness.quality_cliff`). A loss of `ε` (0.5 %) or more is the worst input there is. A run that does not finish (a
NaN or infinite centroid, a wrong shape, a failed step) or a non-finite measure is 1 on every input.

**The speed inputs of a data set:** the program's own compute time to the stopping rule (the GPU-timed passes, or
`futhark c`'s runtime, as before) and the iterations to the stopping rule, each `r/(1+r)` of candidate / reference.

**The fitness:** `truenorth_angle` (the hff library, normalisation off) over ONE flat vector: for every data set its
three quality inputs, the whole quality group repeated `m` times (`--quality-weight`, default 3 for this scheme), and
its two speed inputs once. Nothing else touches the angle: no offset, gate or stage. Beside the angle the runner
reports the **HFF percentile** `F(θ) = I_{sin²θ}((d−1)/2, ½)` (`fitness::hff_percentile`, the hff library's
`higd::cdf_beta_correction`) with `d` the length of the flat vector, in the per-generation log lines, in
`per_generation.tsv` (`best_so_far_hff_percentile`, `best_this_generation_hff_percentile`, `hff_dimensions_d`) and in
the report (`hff_dimensions_d`, `best_hff_percentile`, `best_ever_hff_percentile`). With the six fitness cases of the
run, `d = 6 × (3 × 3 + 2) = 66`.

**Where the further iterations are measured.** On the c backend (`--eval single`) the reference's server is stepped on
from each side's final centroids (`kmeans_converged::further_iterations`). On the GPU (`--eval rust`) the runtime's `k`
command takes the reference's entry as an eighth word (`k ENTRY CASE START N CAP TOL FURTHER`,
`kmeans_converged::k_command_further`): after the loop it runs that entry on from the final centroids on the device,
under the same rule, and the record's header carries the count and whether it converged (`run_of_k_payload` reads it
into `ConvergedRun::further_iterations`). The continuation's time is in no input. The GPU test
`futhark_batch::tests::on_the_gpu_the_k_command_counts_the_seeds_further_iterations_from_the_final_centroids` checks
the device's count against `futhark c`'s from the same centroids.

### The ordering table (measured 2026-11-03)

Two dbpedia cases: the 500 rows of `train_n500_k8` (k = 8) and the 1,000 rows of `held_n1000_k32` (k = 32), k-means++
starts, m = 3, ε = 0.005, the angle over the one flat vector of both cases, **d = 22**. Printed and asserted by
`kmeans_converged::tests_better::the_better_scheme_ranks_every_quality_loss_below_the_reference_and_a_faster_identical_program_above`.
Every individual except the last is a Rust step through the loop with a stated cost (the reference's measured seconds
an iteration for a step that does the reference's work, 0 for one that returns a constant, half for the faster one:
a Rust step's own clock is noise on this machine). The last is the first run's best individual (its hall of fame
rank 1, `tests/corpus/futhark_problems/kmeans/first_run/kmeans_qqs_s1_rank1.txt`, read into a gene by `build`),
compiled with `futhark c` and run as a Futhark server interleaved with the compiled reference (2 pairs), both sides'
own times; the compiled reference equals the Rust step bit for bit (asserted). Each cell gives the two cases in order
(500/k8, 1000/k32). x columns are the quality ratios before the cliff (1 equal, above 1 worse); the speed inputs are
[seconds, iterations], 1/2 equal. The reference's own `k_ref` is 2 on both cases.

| individual | iterations | further iterations k | x inertia | x percentile | x convergence | speed inputs [s, it] | HFF angle θ (rad) | HFF percentile F(θ) | d |
|---|---|---|---|---|---|---|---:|---:|---:|
| the reference program (UltraDim's step) | 9, 10 | 2, 2 | 1.00000, 1.00000 | 1.00000, 1.00000 | 1.00000, 1.00000 | [0.500, 0.500], [0.500, 0.500] | 0.7227 | 3.7772e-5 | 22 |
| returns the centroids unchanged | 2, 2 | 9, 10 | 1.76379, 1.71652 | 1.67775, 1.61390 | 8.00000, 9.00000 | [0.000, 0.182], [0.000, 0.167] | 1.3908 | 4.1367e-1 | 22 |
| makes them worse (starts pushed off the data) | 3, 3 | 19, 24 | 4.38014, 4.65385 | 10.41018, 10.24655 | 18.00000, 23.00000 | [0.000, 0.250], [0.000, 0.231] | 1.3933 | 4.2031e-1 | 22 |
| NaN at iteration 3 | 3 (unfinished), 3 (unfinished) | -, - | -, - | -, - | -, - | [1.000, 1.000], [1.000, 1.000] | 1.5708 | 1.0000e0 | 22 |
| rounded to 1e-3 | 9, 9 | 2, 2 | 1.00079, 1.00042 | 1.00054, 1.00030 | 1.00000, 1.00000 | [0.500, 0.500], [0.474, 0.474] | 0.7630 | 9.7897e-5 | 22 |
| rounded to 1e-2 | 12, 19 | 4, 10 | 1.04637, 1.04854 | 1.03384, 1.03403 | 3.00000, 9.00000 | [0.571, 0.571], [0.655, 0.655] | 1.4574 | 6.0735e-1 | 22 |
| additions reordered: plain f32 sum | 9, 10 | 2, 2 | 1.00000, 1.00000 | 1.00000, 1.00000 | 1.00000, 1.00000 | [0.500, 0.500], [0.500, 0.500] | 0.7227 | 3.7772e-5 | 22 |
| additions reordered: reversed Kahan sum | 9, 10 | 2, 2 | 1.00000, 1.00000 | 1.00000, 1.00000 | 1.00000, 1.00000 | [0.500, 0.500], [0.500, 0.500] | 0.7227 | 3.7772e-5 | 22 |
| faster, identical output (seconds / 2) | 9, 10 | 2, 2 | 1.00000, 1.00000 | 1.00000, 1.00000 | 1.00000, 1.00000 | [0.333, 0.500], [0.333, 0.500] | 0.7034 | 2.3299e-5 | 22 |
| one iteration, then unchanged | 3, 3 | 8, 9 | 1.02282, 1.03269 | 1.01634, 1.02258 | 7.00000, 8.00000 | [0.100, 0.250], [0.091, 0.231] | 1.3942 | 4.2254e-1 | 22 |
| the first run's best individual (kmeans_qqs_s1 rank 1) | 3, 3 | 2, 19 | 1.09372, 1.16403 | 1.08055, 1.14535 | 1.00000, 18.00000 | [0.074, 0.250], [0.021, 0.231] | 1.2885 | 1.9807e-1 | 22 |

What the table shows:

- Every individual worse than the reference by more than ε on a final quality measure on some case (returns the
  centroids unchanged; makes them worse; NaN; rounded to 1e-2; one iteration then unchanged; the first run's best)
  has a higher HFF than the reference (asserted): 1.2885 to 1.5708 rad against 0.7227, percentile 0.198 to 1 against
  3.8e-5.
- **The first run's best individual ranks below the reference** (asserted): 1.2885 rad, F = 0.198, against 0.7227,
  F = 3.8e-5. On the k = 32 case it is 16 % worse on the inertia and the reference needs 19 more iterations from its
  centroids (x = 18); on the k = 8 case it is 9 % worse though its centroids happen to be a fixed point (k = 2).
  Under `futhark-kmeans-qqs` it scored 0.6335 against the seed's 0.7227 and won the run.
- The faster individual with identical output has a lower HFF (asserted): 0.7034 rad, F = 2.3e-5.
- **The reorder noise** (asserted): the two reorderings of the reference's additions (a plain f32 sum; the Kahan sum
  in reverse point order) take the same iterations and further iterations as the reference and differ from it on the
  quality ratios by at most |x − 1| = 1.96e-7, which is 1/25,000 of ε. The angle that noise can move the reference
  to is 0.722758 rad; the reorderings score 0.722734, the reference's angle to six places. They rank above every
  individual that lost more than ε.
- Rounded to 1e-3 (a 0.08 % and 0.04 % worse inertia, below ε, and one iteration fewer on the k = 32 case) scores
  0.7630 rad, F = 9.8e-5: below the reference. That is the cliff's slope at work (x = 1.0008 puts the inertia input
  at 0.58); it is reported, not asserted.

# The scheme before it: `futhark-kmeans-qqs` (NOT TO BE RUN AS IS)

**Fault: rewards early stopping: r/(1+r) compresses a quality loss** (the first k-means run, above). Kept registered
so that the first run stays nameable and reproducible (`docs/FITNESS_SCHEMES.md`). What follows is its specification
as it was built; the loop, the measures, the cases, the clock and the work guard are shared with `futhark-kmeans-better`.

*Revised twice after the advisor's reviews of the same day
(`docs/reviews/2026-11-03_kmeans_plan_advisor_round1.md`, `..._round2.md`) and after Andrew's ruling of 2026-11-03:
**the HFF angle is the final fitness, nothing may manipulate it afterwards; quality weighs more than speed by
repetition in one flat vector.** The specification of the named scheme `futhark-kmeans-qqs`. The code below is what
the scheme computes; `src/fitness.rs` (`qqs`) and `src/futhark_sig/kmeans_converged.rs` implement it as written. The
schemes it replaces stay registered and are NOT TO BE RUN AS IS: `futhark-kmeans-quality-first` (fuller 92922521: the
two-level HFF gated by quality, whose `π/2 + θ_quality` offset manipulated the HFF, and whose edge at exactly 1/2
rejected every same-partition reordering of the seed's f32 sums) and `futhark-kmeans-converged` (fuller bd929652: a
candidate that gained nothing gave it a NaN input and the run stopped; four of its five inputs rewarded the step that
returns the centres unchanged).*

## What UltraDim does (the prototype, read 2026-11-03, commit 4295c335)

`lib/projection-voting-kmeans/src/gpu_kmeans.rs` lines 375-436: the algorithm iterates assignment then update, and
stops when the inertia changes by less than `tolerance` (1e-4) or after `max_iterations` (100), which it reports as
`converged: false` with the centroids it has:

```rust
for iteration in 0..self.config.max_iterations {
    iterations = iteration + 1;
    self.assign_points_gpu(...).await?;          // Step 1
    self.update_centroids_gpu(...).await?;       // Step 2 (the Kahan-compensated sums)
    let current_inertia = self.calculate_inertia(&flat_vectors, &centroids, &assignments, dimensions);
    if (prev_inertia - current_inertia).abs() < self.config.tolerance { break; }   // the stopping point
    prev_inertia = current_inertia;
}
```

**The stopping rule is Euclidean.** `calculate_inertia` (line 1794) is the sum over points of the squared Euclidean
distance to the assigned centroid, in f32; nothing in the loop is normalised. **The quality is spherical.** `quality.rs`
and `lib.rs`:

```rust
/// Spherical-cosine inertia (1/n) Σ (1 − xᵢᵀ μ_{aᵢ}) with unit centroids.      (lib.rs 5709)
pub fn cosine_inertia(flat: &[f32], centroids: &[Vec<f32>], assignments: &[usize], d_raw: usize) -> f32

/// REAL metric: mean over points of log10 F(θ_i), cosines[i] the dot of point i with its assigned unit centroid.
/// Smaller (more negative) = tighter than chance.                               (quality.rs 188)
pub fn exact_mean_log10_p(cosines: &[f64], d: usize) -> f64
```

The two objectives differ on unit data: a step that returns unit-length centroids reaches a lower cosine inertia than
the prototype's step (measured in the review: 0.14178 against 0.14209, and -448.4 against -447.7 on the percentile,
in 24 iterations against 20). The seed is UltraDim's Euclidean Lloyd, scored on UltraDim's spherical measures, exactly
as the prototype reports itself; the first thing evolution can find is a normalisation UltraDim already has in its
cosine path (`lib.rs`, Pass C). When it does, that is a finding to record, not to suppress. The table below shows what
the normalised step scores today.

**What the stopping rule does on this data.** On the dbpedia sample the f32 spacing of the inertia (1,100-1,270 at
n = 5,000) is 1.2e-4, so the tolerance 1e-4 fires only on a bit-identical inertia: "the assignments stopped changing",
Lloyd's fixed point. Measured: every seed run stopped with a last change of exactly 0 after 4 to 33 iterations. So the
iterations count depends on the starting centres (16 against 33 at n = 2,000, k = 32, first rows against random
rows), and the starting centres are part of every case (below). A candidate whose output jitters by one ulp between
iterations never stops: it reaches the cap, is measured on the centroids it has, and its cost is 100 iterations, so
it is scored slow, not worst.

**The harness's assignment.** The measures read each point's nearest centre by Euclidean distance (UltraDim's
`assign_points` rule, the first on a tie) against the centres the last step was given, as UltraDim's loop makes its
assignments before the update; the assignments a step makes inside itself are never read. A candidate that assigns
by cosine inside its step is measured on Euclidean assignments to its centroids (on unit points the two rules agree
only on unit centroids).

## The data: the dbpedia sample, cut into cases

`tests/corpus/futhark_problems/kmeans/dbpedia/sample_5000x1536.in`: 1,000 random start rows (seed 7013) of
UltraDim's dbpedia-openai-1M-angular embeddings, 5 consecutive rows each, 5,000 unit vectors of 1,536 f32 (Andrew's
sampling rule, `sample.py`; the review measured the runs of 5 to carry no structure, so the sample is 5,000 random
rows). `make_cases.sh` cuts the cases from it and nothing else (`examples/kmeans_cases.rs`, the table
`kmeans_converged::KMEANS_CASES`, recorded in `cases.tsv`):

| case | split | rows of the sample | n | k | n / k |
|---|---|---|---:|---:|---:|
| `train_n1000_k16` | train | 0..1000 | 1,000 | 16 | 62 |
| `train_n1000_k50` | train | 1000..2000 | 1,000 | 50 | 20 |
| `train_n500_k8` | train | 2000..2500 | 500 | 8 | 62 |
| `train_n200_k4` | train | 0..200 (inside `train_n1000_k16`) | 200 | 4 | 50 |
| `held_n1000_k32` | held | 2500..3500 | 1,000 | 32 | 31 |
| `held_n500_k16` | held | 3500..4000 | 500 | 16 | 31 |
| `test_n1000_k32` | test (never a fitness input) | 4000..5000 | 1,000 | 32 | 31 |
| the timing input | - | 0..5000 | 5,000 | 256 | 20 |

Six disjoint row sets, the 5,000 rows used once, k at most n / 20 so singleton clusters are rare (the review's list
had 1,000 rows with k 64, over its own rule; k 50 here); the held cases hold out rows. The small training case is the one the host interpreter can run (about 4e7 node executions a second: a
step of 1,000 x 1,536 x 16 is a minute), for the kingdom's tests; every case in the run is the compiled program's.
**The starting centres** of every case are UltraDim's k-means++ draws over the case's rows (`init_centroids_plus_plus`
with `random_seed` set: ChaCha8, the first centre uniform, each next one with probability proportional to its squared
distance to the nearest centre so far; `kmeans_plus_plus_rows`, the same crates at the same versions, so the draws are
UltraDim's), at the seed `cases.tsv` records, stored after the points in the case's `.in`: every candidate and the
seed start from the same centres. The test `tests_kmeans::the_dbpedia_cases_are_the_declared_cuts_...` recomputes
every case from the sample and requires the stored bytes.

## The run of one case

Both the candidate and the seed are run on each fitness case with UltraDim's loop and stopping rule (the two kernels
being evolved are the body of that loop; the loop, the stopping test and the quality measures are the harness's, in
Rust, and are not evolved), INTERLEAVED (`--pairs` times each, default 2, the least that interleaves: with one pair the candidate always ran first and the seed second, the review's N4; the candidate first in the even pairs), and
**the seed's run of that interleaving is the yardstick of the candidate's inputs**, never a seed run from another time.

```rust
/// One run of a k-means program to UltraDim's stopping point, and what the fitness reads from it.
pub struct ConvergedRun {
    /// The final centroids and assignments, as the program left them.
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    /// 1. QUALITY of the final centroids: UltraDim's `cosine_inertia(points, centroids, assignments, d)`, and its
    ///    `exact_mean_log10_p` over the cosines of the points that are not at their centroid (`AT_CENTROID_SIN2`).
    pub inertia: f64,
    pub mean_log10_p: f64,
    /// UltraDim's `exact_mean_log10_p` over EVERY point, the prototype's own number: reported, never an input.
    pub mean_log10_p_all: f64,
    /// The points at their centroid, left out of `mean_log10_p`.
    pub coincident: usize,
    /// 2. THE PROGRAM'S OWN TIME to the stopping point, in seconds: the sum over the iterations of what the step
    ///    itself took (the GPU-timed passes of the shim's executable; `futhark c`'s own `-t` runtime), the
    ///    harness's assignment, the process and the pipe excluded exactly.
    pub seconds: f64,
    /// The harness's wall clock from the first assignment to the stop (the budget's number; never an input).
    pub wall_seconds: f64,
    /// 3. The iterations to the stopping point (UltraDim's `iterations`), and the Euclidean inertia gained an
    ///    iteration (the retired scheme's input, reported).
    pub iterations: u32,
    pub inertia_per_iteration: f64,
    /// UltraDim's `converged`: the tolerance fired before the cap. False at the cap: the run is measured on the
    /// centroids it has (`finished` true) and its cost is `MAX_ITERATIONS`.
    pub converged: bool,
    /// Whether the run could be measured: every step returned a `[k][d]f32` of finite values. A run that did not
    /// finish (an infinite or NaN centroid, a wrong shape, a step that failed) takes the worst value on every input.
    pub finished: bool,
    pub failure: String,
}
```

**The clock.** The timed quantity is the candidate program's own time: on the GPU the device's timestamps around the
program's dispatches of each iteration's process (`FUTHARK_WGPU_SHIM_TIMESTAMPS`, the sum of the timed passes, as the
other schemes' interleaved pairs time them); with `futhark c`, the executable's `-t` runtime (the entry point's
microseconds, the input's reading and the output's writing excluded). The harness's own assignment (parallel over the
points now, each point's sums in order), the process launch and the 6-30 MB written down the pipe each iteration are
in `wall_seconds` and in nothing the fitness reads (`tests::the_programs_seconds_are_the_steps_own_time_and_the_harness_is_excluded`).

**The percentile's coincident points.** UltraDim clamps a point's `sin^2` at the smallest positive f64 so a singleton
cluster stays finite: at d = 1,536 that one point is about -236,000 where every other point is near -450, and a point
one f32 rounding off its centroid is about -5,400. The identity step (the k-means++ start rows returned as the
centres) measured -4,053 against the seed's -448 on UltraDim's mean, owing it entirely to its k rows at their own
centroids. The fitness's percentile is UltraDim's `exact_mean_log10_p` over the points that are not at their centroid
(`sin^2` of the angle below 1e-4, 0.57 degrees: in 1,536 dimensions two distinct embeddings are never that close;
nearest neighbours are 15 degrees apart and more), their count reported, and UltraDim's number over every point
reported beside it.

## The inputs of one case: two groups

```rust
/// `r / (1 + r)` of the ratio `c / s`, TOTAL on [0, 1]: a seed measure of 0 divides the smallest positive f64 (so a
/// positive candidate measure is 1, the worst), a negative candidate measure is 0 (better than perfect), an infinite
/// or NaN ratio is 1. The seed's own measure against itself is exactly 1/2.
pub fn squashed_ratio(c: f64, s: f64) -> f64 {
    if c.is_nan() || s.is_nan() {
        return 1.0;
    }
    let r = c.max(0.0) / s.max(f64::MIN_POSITIVE);
    if r.is_finite() {
        (r / (1.0 + r)).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// THE INPUTS OF ONE CASE, each defined for every outcome: the quality group is the cosine inertia ratio and the
/// ratio of the seed's mean log10 p to the candidate's (both are at most 0, so the sign is dropped: a tighter
/// candidate is more negative and scores below 1/2; a candidate with no tightness at all, 0, scores 1); the speed
/// group is the ratio of the program's own seconds and the ratio of the iterations. A run that did not finish is 1
/// on every input. Nothing here is a function of another input.
pub fn kmeans_inputs(candidate: &ConvergedRun, seed: &ConvergedRun) -> Inputs {
    if !candidate.finished {
        return Inputs::worst();
    }
    Inputs {
        quality: vec![
            squashed_ratio(candidate.inertia, seed.inertia),             // cosine inertia, candidate / seed
            squashed_ratio(-seed.mean_log10_p, -candidate.mean_log10_p), // the angular percentile: the seed's tightness over the candidate's
        ],
        speed: vec![
            squashed_ratio(candidate.seconds, seed.seconds),                                 // the program's own time to the stopping point
            squashed_ratio(f64::from(candidate.iterations), f64::from(seed.iterations)),     // iterations to the stopping point
        ],
    }
}
```

The final list, four inputs a case: **cosine inertia / seed**; **seed's mean log10 p / candidate's** (points off
their centroid); **the program's own seconds / seed's**; **iterations / seed's**. The retired scheme's fifth input, the
Euclidean inertia gained an iteration, was the iterations input a second time (two candidates at the same inertia
differ on it exactly by 1 / iterations) and is reported only.

## The fitness: one flat vector, the quality inputs repeated

```rust
/// THE ONE FLAT VECTOR (Andrew, 2026-11-03: `(quality, quality, speed)`): every quality input entered
/// `quality_weight` times (the whole quality group repeated, in order), then every speed input once.
pub fn flat_vector(quality: &[f64], speed: &[f64], quality_weight: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(quality.len() * quality_weight + speed.len());
    for _ in 0..quality_weight {
        v.extend_from_slice(quality);
    }
    v.extend_from_slice(speed);
    v
}

/// THE FLAT FITNESS: the hff library's TrueNorth over the flat vector, and THAT ANGLE IS THE FITNESS. Nothing
/// manipulates it: no gate, no offset, no nesting. The two group angles are reported beside it, never read by
/// selection. Every input must be on [0, 1]; no quality input, no speed input, or a weight of 0 is a bug.
pub fn qqs(quality: &[f64], speed: &[f64], quality_weight: usize) -> Result<TwoLevel, String> {
    if quality.is_empty() || speed.is_empty() {
        return Err("our software has a bug: the flat fitness was given an empty group (no fitness case, or no speed input)".into());
    }
    if quality_weight == 0 {
        return Err("our software has a bug: the flat fitness was given a quality weight of 0 (the quality inputs would not be entered)".into());
    }
    if let Some(x) = quality.iter().chain(speed).find(|x| !(0.0..=1.0).contains(*x)) {
        return Err(format!("our software has a bug: a fitness input {x} is outside [0, 1]"));
    }
    let fitness = crate::score::truenorth_angle(&flat_vector(quality, speed, quality_weight));
    Ok(TwoLevel { fitness, quality: crate::score::truenorth_angle(quality), speed: crate::score::truenorth_angle(speed) })
}
```

The vector of one evaluation is every fitness case's two quality inputs, the whole group entered **m** times, then
every case's two speed inputs once; the HFF (TrueNorth, the library, normalisation off) over it is the fitness.
Nothing is averaged across cases, nothing is gated, nothing is added after the angle. **m is the run's
`--quality-weight`** (default 2, Andrew's written form `(quality, quality, speed)`), recorded in the header, the run
card and the stream's `run_start` (`fitness.quality_weight`).

**What m buys, exactly.** TrueNorth is `θ = acos(1 − mean(x²))`, so with Q quality inputs and S speed inputs the seed
(1/2 everywhere) scores `acos(3/4) = 0.7227` at every m, and a candidate beats it exactly when
`m · (Σq² − Q/4) < S/4 − Σs²`: m is the exchange rate between a quality loss and a speed gain. The round-2 review
measured the identity step needing m > 1.68 to rank below the seed and recommended 3; the table below has both.

## The ordering at m = 2 and m = 3 (measured 2026-11-03)

500 dbpedia rows (`train_n500_k8`'s rows), k 8, k-means++ starts, every step a Rust step through the loop against the
seed (UltraDim's step), printed and asserted by
`kmeans_converged::tests::the_flat_scheme_at_m_2_and_3_ranks_the_identity_worsening_nan_and_rounded_steps_below_the_seed_and_a_faster_equal_step_above`.
A Rust step's own clock is noise on this machine (the review's N3), so every step reports a stated cost: a step that
does the seed's work costs the seed's measured seconds an iteration, a step that returns a constant costs nothing, the
faster step costs half; the seconds input is then a function of the iterations alone. `*` ranks above the seed
(0.7227).

| step | iterations | converged | finished | cosine inertia | mean log10 p (off centroid) | all points | at centroid | quality [cos, pct] | speed [s, it] | m = 1 | m = 2 | m = 3 |
|---|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| the seed (UltraDim's step) | 9 | yes | yes | 0.15227 | -425.78 | -425.78 | 0 | 0.5000, 0.5000 | 0.5000, 0.5000 | 0.7227 | 0.7227 | 0.7227 |
| identity: return the centres | 2 | yes | yes | 0.26857 | -253.78 | -3105.85 | 8 | 0.6382, 0.6266 | 0.0000, 0.1818 | 0.6571* | 0.7556 | 0.8011 |
| worsening: the starts pushed off the data | 3 | yes | yes | 0.66696 | -40.90 | -40.90 | 0 | 0.8141, 0.9124 | 0.0000, 0.2500 | 0.9140 | 1.0574 | 1.1247 |
| a NaN at iteration 3 | 3 | no | no | - | - | - | - | 1, 1 | 1, 1 | 1.5708 | 1.5708 | 1.5708 |
| centres rounded to 1e-3 | 9 | yes | yes | 0.15239 | -425.55 | -425.55 | 0 | 0.5002, 0.5001 | 0.5000, 0.5000 | 0.7229 | 0.7229 | 0.7229 |
| centres rounded to 1e-2 | 12 | yes | yes | 0.15933 | -411.84 | -411.84 | 0 | 0.5113, 0.5083 | 0.5714, 0.5714 | 0.7859 | 0.7701 | 0.7620 |
| mean then unit length (spherical) | 9 | yes | yes | 0.15231 | -425.71 | -425.71 | 0 | 0.5001, 0.5000 | 0.5000, 0.5000 | 0.7228 | 0.7228 | 0.7228 |
| plain f32 sum (the seed's partition) | 9 | yes | yes | 0.15227 | -425.78 | -425.78 | 0 | 0.5000, 0.5000 | 0.5000, 0.5000 | 0.7227 | 0.7227 | 0.7227 |
| reversed Kahan sum (the seed's partition) | 9 | yes | yes | 0.15227 | -425.78 | -425.78 | 0 | 0.5000, 0.5000 | 0.5000, 0.5000 | 0.7227 | 0.7227 | 0.7227 |
| faster, the same bits (seconds / 2) | 9 | yes | yes | 0.15227 | -425.78 | -425.78 | 0 | 0.5000, 0.5000 | 0.3333, 0.5000 | 0.6685* | 0.6870* | 0.6961* |
| one seed step, then identity | 3 | yes | yes | 0.15574 | -418.94 | -418.94 | 0 | 0.5056, 0.5041 | 0.1000, 0.2500 | 0.5463* | 0.6129* | 0.6439* |

The test asserts, at m = 2 and m = 3: the identity, the worsening, the NaN and both rounded steps rank below the
seed and the faster-but-equal step above it; the two reorderings of the seed's sums are within 1e-4 of 1/2 on both
quality inputs and within 1e-3 of the seed's fitness (the retired gate's cliff at 1/2 is gone: they score the seed's
0.7227 to four places); at m = 1 the identity ranks above the seed, which is why m is at least 2. Two findings,
printed, not asserted: on this case the normalised step is a hair worse than the seed (0.15231 against 0.15227; on
the review's 2,000-row case it was a hair better), and **the early stop** — one seed step, then the identity, 1-2 %
worse in the clustering in a third of the iterations — ranks above the seed at every m up to 32-44 (the review's
measurement). That is the problem's own exchange rate under the ruling; the run prints the champion's quality inputs
beside its speed inputs every generation so a worse-but-faster champion is read as what it is.

## The work guard

The runner's work guard for this scheme is the problem's own (the review's F6): a candidate's work model
(`Interp::work`) on a fitness case may be at most `WORK_CAP_TIMES_SEED` = 10 times the seed's work on that case
(`converged_work_cap`); over it the candidate is not run and takes the worst score, counted in the report
(`over_work_cap_not_run`), never a stop of the run. The WGSL per-submit cap (1.7e8 node executions) admitted n x k x d
under 1.7e6 and refused every dbpedia case; the seed's own work on the timing input is 2.3e11.

## What is NOT in it

- No comparison of bits, cells or memberships against any reference program: k-means has no exact answer.
- No rounding bound. The f64 reference program exists only to check the translation of UltraDim's kernels into the
  gene (`docs/kmeans_seed.md`), never to score.
- No slope check and no gate before a discovery run: the slope check scores neighbours against stored answers, and
  there are none; the runner says so and starts.
- The seed's own run of the same interleaving is the yardstick on every case, so the inputs read "better or worse
  than what we started from, measured beside it".

## The run (the round-2 review's N2-N5)

- **The test case** (`test_n1000_k32`, never a fitness input) is run every generation the champion's text changes,
  champion and seed interleaved as a fitness case is, and at the end: `OUT/test_case.tsv` (the learning curve: the
  four inputs and both sides' raw measures a generation) and the report's `test_case` line.
- **The noise floor** (N3): the in-run A/A, the seed judged as a candidate against itself through the same
  interleaving, on every backend; its quality inputs must be exactly 1/2 (one program; a value off 1/2 is a
  non-deterministic reduction and is written down), and the widest distance of its speed inputs from 1/2 is the
  floor (`seed_aa_speed_floor`). A champion's speed claim counts when every one of its seconds inputs is below
  1/2 − floor (`speed_claim_beats_noise_floor`). Measured on the c backend, 2 pairs: the floor 0.0029 (the seconds
  inputs 0.4988-0.5029, the iterations inputs 0.5000, 13.2 s an evaluation); 3 pairs: the floor 0.0009 (0.4991-0.5004, 19.2 s an evaluation; a first 3-pair run that overlapped a test build read 0.053: the floor is the machine's, measured quiet).
- **The pairs** (N4): the default is 2.
- **The P start's cost** (N5), measured on the c backend 2026-11-03 (`--evaluations 2`, the 131-function seed): the
  e-graph's 31 variants of the seed 149 s, the saturated fold 193 s, 19.5 GB resident at the peak; then generation 0.
  The pump every 5 generations runs the variants on the best (2 iterations, 16 members) at a fraction of that.
- **Throughput.** Before 2026-11-03 each iteration was its own process with the points down the pipe. On the c
  backend one evaluation of the seed (6 cases, 2 pairs, both sides) took 19.3 s that way and takes 13.2 s with the
  program kept alive as a Futhark server (`futhark c --server`, `kmeans_converged::Server`: the points loaded once a
  case, the centres restored an iteration): 3.1 against 4.5 evaluations a minute, the process overhead 23 ms an
  iteration on the host. On the GPU a process cost 0.3-0.6 s, which is what the **resident-sample path** removes
  (`--eval rust`, Andrew's design of 2026-11-03): every case is a row range of the one dbpedia sample, so the sample
  is uploaded ONCE for the run and kept on the device (the runtime's session holds it: `futhark-rust-rt`, `host.rs`,
  `Resident`), a generation's programs are ONE module with the seed's range entry (`futhark_batch::Step::Kmeans`),
  loaded into the in-process host, and each candidate's seven cases run against the resident sample interleaved with
  the seed's entry, the centres staying on the device between iterations (the previous call's output is the next
  call's argument), the harness's assignment and the stopping rule on the host from the k × d centres read back
  (16-300 KB). The program's entry takes the whole sample and the range and slices it itself
  (`kmeans_converged::range_entry`: `sample[start:start + n] :> [n][d]f32`, an outer-dimension slice of a row-major
  array, an offset in the compiled program, no copy; equal to the plain program on the cut bit for bit through
  `futhark c`, tested). The runtime's harness is fuller's to the bit (`kmeans_harness.rs`, included by path in
  fuller's test). The report splits each generation's cost into compile, upload (once a session; `upload_s` and
  `resident_hits` in the stream's extras) and run. The GPU measurement before (`--eval single`, a process an
  iteration) and after (`--eval rust`) is `/private/tmp/claude-501/fbench/kmeans_gpu_controls.sh`, not yet run.

## Known limits, stated

- `KMEANS_PROGRAM_SIZE` = 192: a host population of 64 genes is 7 GB of genes, fine on this machine and impossible on
  the device; "GPU residency" for k-means is not this view.
- The dbpedia-sized cases are never run through the host interpreter; its agreement with the compiled program is
  tested on the small case, and the runner's controls are the compiled program's.
- The shim's executable (`--eval single --backend gpu`) has no server mode: that path is still a process an iteration.
