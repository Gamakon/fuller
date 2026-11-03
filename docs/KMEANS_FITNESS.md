# The k-means fitness: the measures that go into the HFF

*fuller, 2026-11-03, revised after the advisor's review of the same day
(`docs/reviews/2026-11-03_kmeans_plan_advisor_round1.md`, every finding fixed). The specification of the named scheme
`futhark-kmeans-quality-first`. The code below is what the scheme computes; `src/fitness.rs` and
`src/futhark_sig/kmeans_converged.rs` implement it as written. The scheme it replaces, `futhark-kmeans-converged`
(fuller bd929652), stays registered and is NOT TO BE RUN AS IS: a candidate that gained nothing gave it a NaN input
and the run stopped; four of its five inputs rewarded the step that returns the centres unchanged.*

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
Rust, and are not evolved), INTERLEAVED (`--pairs` times each, default 1; the candidate first in the even pairs), and
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

## The fitness: quality first

```rust
/// THE K-MEANS SCHEME'S FITNESS, QUALITY FIRST: the two-level HFF of the quality group (every fitness case's cosine
/// inertia ratio and percentile ratio) and the speed group (every case's seconds ratio and iterations ratio), GATED
/// BY QUALITY: a candidate no worse than the seed on every quality input (each at most 1/2) scores the two-level
/// angle, in [0, π/2); one worse than the seed on any quality input scores π/2 + θ_quality.
pub fn converged_gated(quality: &[f64], speed: &[f64]) -> Result<TwoLevel, String> {
    if quality.is_empty() || speed.is_empty() {
        return Err("our software has a bug: the k-means scheme was given no inputs (no fitness case)".into());
    }
    if let Some(x) = quality.iter().chain(speed).find(|x| !(0.0..=1.0).contains(*x)) {
        return Err(format!("our software has a bug: a k-means input {x} is outside [0, 1]"));
    }
    let t = two_level_hff(quality, speed);
    if quality.iter().all(|x| *x <= 0.5) {
        Ok(t)
    } else {
        Ok(TwoLevel { fitness: std::f64::consts::FRAC_PI_2 + t.quality, ..t })
    }
}
```

The HFF (TrueNorth, the library, normalisation off) takes every case's inputs as its own inputs inside each group;
nothing is averaged across cases; the two group angles are combined as the other runners combine them
(`runner::two_level_hff`, each over π/2). No candidate that loses quality on any case ranks above the seed however
fast; a candidate that keeps the seed's quality ranks on its speed; one that gains quality ranks above the seed by
its quality angle. There is no stage: speed is measured in the same run as quality.

**Why a gate.** With four independent ratios and no gate, a step that rounds its centroids to 1e-3 converges in 17
iterations instead of 20 at a cosine inertia 0.07 % worse and would rank above the seed on the sum of its inputs
(the review's F3): rounding games the stopping rule, it is not a faster k-means. Quality first is Andrew's standing
rule for every runner ("quality first, then performance"); the gate is the problem's own objective, not a
comparison of bits. Its edge is where two programs reach the same partition and differ only in the order of their
f32 additions (a 1e-7 difference in the inertia): whichever lands a hair better passes, the other does not, and no
program that is worse passes. Measured on the seed against its plain-sum variant (no Kahan) the quality inputs are
within 1e-4 of 1/2 both ways.

## What the inputs are on the cases (measured 2026-11-03)

500 dbpedia rows (`train_n500_k8`'s rows), k 8, k-means++ starts, every step a Rust step through the loop, printed by
`kmeans_converged::tests::the_identity_a_worsening_a_nan_and_a_rounded_step_rank_below_the_seed_through_the_scheme`,
which asserts the ranks below:

| step | iterations | converged | cosine inertia | mean log10 p (off centroid) | over every point | at centroid | quality [cos, pct] | speed [s, it] | fitness |
|---|---:|---|---:|---:|---:|---:|---|---|---:|
| the seed (UltraDim's step) | 9 | yes | 0.15227 | -425.78 | -425.78 | 0 | 0.5000, 0.5000 | 0.5000, 0.5000 | **0.663** |
| identity: return the centres | 2 | yes | 0.26857 | -253.78 | -3105.85 | 8 | 0.6382, 0.6266 | 0.0002, 0.1818 | 2.498 |
| worsening: the starts pushed off the data | 3 | yes | 0.66696 | -40.90 | -40.90 | 0 | 0.8141, 0.9124 | 0.0004, 0.2500 | 2.886 |
| a NaN at iteration 3 | 3 | no (unfinished) | - | - | - | - | 1, 1 | 1, 1 | 3.142 |
| centres rounded to 1e-3 | 9 | yes | 0.15239 | -425.55 | -425.55 | 0 | 0.5002, 0.5001 | 0.5006, 0.5000 | 2.294 |
| centres rounded to 1e-2 | 12 | yes | 0.15933 | -411.84 | -411.84 | 0 | 0.5113, 0.5083 | 0.5648, 0.5714 | 2.308 |
| mean then unit length (spherical) | 9 | yes | 0.15231 | -425.71 | -425.71 | 0 | 0.5001, 0.5000 | 0.5093, 0.5000 | 2.294 |

The identity's percentile over every point is -3,106 (its 8 start rows at their own centroids); over the points off
their centroid it is -254, worse than the seed's -426, and the input reads 0.63. The normalised-centroid step on THIS
case is a hair worse than the seed on both quality inputs (0.15231 against 0.15227) and ranks below it; on the
review's measurement (2,000 rows, k 32) it was a hair better (0.14178 against 0.14209) and would rank above. Both are
findings; the scheme reports what the measure says. On the compiled path the seed on 200 x 1,536 x 4 through
`futhark c` takes 13 iterations, the program's own 0.026 s of 0.657 s wall: the harness is 96 % of the clock there
and none of the fitness.

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

## Known limits, stated

- Each iteration is one process of the compiled program with the points written down the pipe (6-30 MB): the budget,
  not the measure, pays for it (about 0.3-0.5 s a process on the GPU path: a candidate on the six cases at one pair
  is of the order of two minutes). Futhark's `--server` mode (the process kept alive, the points loaded once) is the
  next cut of that cost and is not built; the shim's executable has no server mode.
- `KMEANS_PROGRAM_SIZE` = 192: a host population of 64 genes is 7 GB of genes, fine on this machine and impossible on
  the device; "GPU residency" for k-means is not this view.
- The dbpedia-sized cases are never run through the host interpreter; its agreement with the compiled program is
  tested on the small case, and the runner's controls are the compiled program's.
