# The k-means fitness: the measures that go into the HFF

*fuller, 2026-11-03. The specification of the named scheme `futhark-kmeans-converged`. The code below is what the
scheme computes; `src/fitness.rs` and `src/futhark_sig.rs` implement it as written.*

## What UltraDim does (the prototype, read 2026-11-03)

`lib/projection-voting-kmeans/src/gpu_kmeans.rs` lines 375-425: the algorithm iterates assignment then update, and
stops when the inertia changes by less than `tolerance` (1e-4) or after `max_iterations` (100):

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

Its quality measures, `quality.rs` and `lib.rs`:

```rust
/// Spherical-cosine inertia (1/n) Σ (1 − xᵢᵀ μ_{aᵢ}) with unit centroids.      (lib.rs 5705)
pub fn cosine_inertia(flat: &[f32], centroids: &[Vec<f32>], assignments: &[usize], d_raw: usize) -> f32

/// REAL metric: mean over points of log10 F(θ_i), cosines[i] the dot of point i with its assigned unit centroid.
/// Smaller (more negative) = tighter than chance.                               (quality.rs 188)
pub fn exact_mean_log10_p(cosines: &[f64], d: usize) -> f64

pub struct ClusterQuality {
    pub raw_inertia: f64,            // engine/convergence number only, NOT a quality score
    pub mean_log10_p_real: f64,      // the real clustering (smaller = tighter)
    pub mean_log10_p_fitted_null: f64,
    pub log10_lift: f64,             // null − real: headline quality, bigger = better
    pub lift: f64,
}
```

## The three measures, per fitness case

A fitness case is a data set (points, starting centres, k) at one magnitude scale. Both the candidate and the seed are
run on it with UltraDim's loop and stopping rule (the two kernels being evolved are the body of that loop; the loop,
the stopping test and the quality measures are the harness's, in Rust, and are not evolved).

```rust
/// One run of a k-means program to UltraDim's stopping point, and what the fitness reads from it.
pub struct ConvergedRun {
    /// The final centroids and assignments, as the program left them.
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    /// 1. QUALITY of the final centroids, both of UltraDim's numbers:
    ///    cosine_inertia(points, centroids, assignments, d) and
    ///    exact_mean_log10_p(cosines of each point to its assigned unit centroid, d).
    pub inertia: f64,
    pub mean_log10_p: f64,
    /// 2. ELAPSED CONVERGENCE SPEED: the wall clock from the first assignment to the stopping point, in seconds,
    ///    measured interleaved with the seed's run on the same device (the A/A-gated timing of the runner).
    pub seconds: f64,
    /// 3. RATE OF CONVERGENCE: the iterations to the stopping point (UltraDim's `iterations`), and the quality
    ///    gained per iteration (inertia at the start − inertia at the stop) / iterations.
    pub iterations: u32,
    pub inertia_per_iteration: f64,
    /// A run that did not finish (an infinite or NaN centroid, a wrong shape, the iteration cap with no convergence)
    /// takes the worst value on every input below.
    pub finished: bool,
}

/// The HFF inputs of one case: every measure its own input, each relative to the seed's run on the same case, each
/// in [0, 1] with 0 best, by `r / (1 + r)` of a ratio (1/2 is "equal to the seed") or a difference squashed the same way.
pub fn kmeans_inputs(candidate: &ConvergedRun, seed: &ConvergedRun) -> Vec<f64> {
    if !candidate.finished {
        return vec![1.0; 5];
    }
    let ratio = |c: f64, s: f64| c / s.max(f64::MIN_POSITIVE);
    let squash = |r: f64| r / (1.0 + r);
    vec![
        // 1. quality of the final centroids
        squash(ratio(candidate.inertia, seed.inertia)),                               // cosine inertia, candidate / seed
        squash(10f64.powf(candidate.mean_log10_p - seed.mean_log10_p)),               // the angular-percentile metric, as a ratio of p
        // 2. elapsed convergence speed
        squash(ratio(candidate.seconds, seed.seconds)),                               // wall clock to the stopping point
        // 3. rate of convergence
        squash(ratio(f64::from(candidate.iterations), f64::from(seed.iterations))),   // iterations to the stopping point
        squash(ratio(seed.inertia_per_iteration, candidate.inertia_per_iteration)),   // quality gained an iteration (more is better, so inverted)
    ]
}
```

The HFF (TrueNorth, the library, normalisation off) takes every case's five inputs as its own inputs; nothing is
averaged across cases. There is no gate and no stage: speed and rate are measured in the same run as quality, so a
program that converges in fewer, slower iterations and one that converges in more, faster iterations are both
measured on all five, and the HFF orders them.

## What is NOT in it

- No comparison of bits, cells or memberships against any reference program: k-means has no exact answer.
- No rounding bound. The f64 reference program exists only to check the translation of UltraDim's kernels into the
  gene (`docs/kmeans_seed.md`), never to score.
- The seed's own run is the yardstick on every case, so the inputs read "better or worse than what we started from".
