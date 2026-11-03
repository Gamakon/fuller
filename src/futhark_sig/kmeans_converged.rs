//! THE K-MEANS FITNESS (`docs/KMEANS_FITNESS.md`, the specification as code; Andrew, 2026-11-03): k-means has no exact
//! answer, so a program is not compared with a reference program's bits, cells or memberships. It is RUN TO ULTRADIM'S
//! STOPPING RULE on each fitness case, candidate and seed alike, and what the fitness reads is the clustering it
//! reaches and how it got there: the quality of the final centroids (UltraDim's own two numbers) and the cost of
//! reaching them (the program's own GPU time and its iterations), each relative to the seed's run in the same
//! interleaving ([`kmeans_inputs`]). THE SCHEME TO RUN is `futhark-kmeans-better` (Andrew's specification of
//! 2026-11-03): three quality inputs a case, each the cliff of a ratio to the seed, the convergence among them
//! ([`kmeans_inputs_better`], [`further_iterations`]), in one flat vector at m = 3. The scheme before it,
//! `futhark-kmeans-qqs` (`crate::fitness::qqs`; Andrew's ruling of 2026-11-03), reads the two groups in ONE FLAT
//! VECTOR, the quality inputs entered m times (`--quality-weight`, default 2), and the HFF angle over it is the
//! fitness, nothing after it; it is NOT TO BE RUN AS IS (it rewards early stopping: r/(1+r) compresses a quality
//! loss). The retired `futhark-kmeans-quality-first`
//! (the two-level HFF gated by an offset, which manipulated the HFF; its edge at 1/2 rejected every reordering of
//! the seed's sums) and `futhark-kmeans-converged` (five inputs, [`kmeans_inputs_five`]) are NOT TO BE RUN AS IS
//! (the advisor's reviews of 2026-11-03, rounds 1 and 2).
//!
//! The prototype is `lib/projection-voting-kmeans` of UltraDim (read 2026-11-03, read-only, commit 4295c335):
//! `gpu_kmeans.rs` 375-436 is the loop (assign, update, `calculate_inertia`; stop when the inertia changes by less
//! than `tolerance` = 1e-4 or after `max_iterations` = 100, which leaves `converged` false), `lib.rs` 5709
//! `cosine_inertia` and `quality.rs` 188 `exact_mean_log10_p` are the quality measures. The loop, the stopping test
//! and the measures are the harness's, here in Rust; the program being evolved is the BODY of the loop, one step
//! `(points, centres) -> centres` (the two kernels, assignment and update).
//!
//! THE STOPPING RULE IS EUCLIDEAN, THE SCORE IS SPHERICAL (the review's F8). The loop stops on `calculate_inertia`,
//! the sum of squared Euclidean distances to the centroids (`gpu_kmeans.rs` 1794-1822, no normalisation anywhere in
//! the loop), as the prototype does; the quality is read with UltraDim's spherical measures, which normalise the
//! centroids for the measure only. The two objectives differ on unit data: a step that returns unit-length centroids
//! reaches a lower cosine inertia than the prototype's (measured in the review: 0.14178 against 0.14209), so the
//! first thing evolution can find is a normalisation UltraDim already has in its cosine path (`lib.rs`, Pass C). That
//! is a finding to record when it comes, never to suppress: the measures are UltraDim's exactly.
//!
//! WHAT THE STOPPING RULE DOES ON THIS DATA (the review's F9): on the dbpedia sample the f32 spacing of the inertia
//! (1,100-1,270 at n = 5,000) is 1.2e-4, so the tolerance 1e-4 fires only on a bit-identical inertia, which is "the
//! assignments stopped changing": Lloyd's fixed point. Measured: every seed run stopped with a last change of exactly
//! 0 after 4 to 33 iterations. So `iterations` is the fixed-point count, which depends on the starting centres (16
//! against 33 at n = 2,000, k = 32, first rows against random rows): the starting centres are part of the case
//! (`tests/corpus/futhark_problems/kmeans/`, k-means++ starts stored per case). A candidate whose output jitters by
//! one ulp between iterations never stops: it reaches the cap, which UltraDim reports as `converged: false` with the
//! centroids it has, and here too (`ConvergedRun::converged` false, `finished` true): its quality is measured on
//! those centroids and its cost is 100 iterations, so it is scored SLOW, not worst.
//!
//! THE HARNESS'S ASSIGNMENT (the review's F12): the measures read each point's nearest centre by Euclidean distance
//! (UltraDim's `assign_points` rule, the first on a tie) against the centres the last step was given, as UltraDim's
//! loop makes its assignments before the update; the assignments a step makes inside itself are never read. A
//! candidate that assigns by cosine inside its step is measured on Euclidean assignments to its centroids. On unit
//! points `|x - c|^2 = 1 + |c|^2 - 2 x . c`, so the two rules agree only on unit centroids.
//!
//! UltraDim's quality measures are spherical: its points are unit vectors by construction and its centroids are
//! normalised, so `cosine_inertia` reads `1 - x . mu` as a cosine. The harness normalises each point and each
//! centroid before the two spherical measures; on the dbpedia rows (every norm 1.00000) this is the identity and the
//! measures are UltraDim's exactly. A zero vector has cosine 0 to everything.

use std::path::Path;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::{read_one, write_binary, Array, Elem, Signature};
use crate::typed_gep::Tree;

/// UltraDim's `tolerance`: the loop stops when the inertia changes by less than this.
pub const TOLERANCE: f32 = 1e-4;

/// UltraDim's `max_iterations`: a run that reaches it stops there with `converged` false.
pub const MAX_ITERATIONS: u32 = 100;

/// A POINT AT ITS CENTROID: `sin^2` of its angle to the centroid below this, 0.01 radians (0.57 degrees). UltraDim
/// clamps such a point's `sin^2` at the smallest positive f64 so that a singleton cluster stays finite
/// (`quality.rs` 157-170): `ln_betainc(767.5, 0.5, 2.2e-308)` is about -236,000 for that one point at d = 1,536,
/// and a point one f32 rounding off its centroid is about -5,400, where every other point is near -450. The
/// measured identity step (the k-means++ start rows returned as the centres) owed its -4,053 against the seed's
/// -448 to its k rows at their own centroids (the review's F2). The fitness's percentile is the mean over the points
/// that are NOT at their centroid, and their count is reported; in 1,536 dimensions two distinct embeddings are
/// never within 0.57 degrees (nearest neighbours are 15 degrees apart and more), so the rule excludes duplicates and
/// singletons and nothing else. UltraDim's own number over every point is kept beside it, reported, never an input.
pub const AT_CENTROID_SIN2: f64 = 1e-4;

/// THE WORK CAP of the converged path (the review's F6): a candidate's work model on a fitness case may be at most
/// this many times the seed's on the same case; over it the candidate is not run and takes the worst score. The
/// WGSL per-submit cap (1.7e8 node executions) admitted n x k x d under 1.7e6 and refused every dbpedia case; the
/// seed's own work on the 5,000 x 1,536 x 256 timing case is 2.3e11.
pub const WORK_CAP_TIMES_SEED: u128 = 10;

/// The bound a candidate's work on a case must not exceed, from the seed's work on that case.
pub fn converged_work_cap(seed_work: u128) -> u128 {
    seed_work.saturating_mul(WORK_CAP_TIMES_SEED)
}

/// The names of the quality inputs a case, in [`kmeans_inputs`]' order.
pub const QUALITY_NAMES: [&str; 2] = ["cosine inertia / seed", "seed mean log10 p / candidate (points off their centroid)"];

/// The names of the speed inputs a case, in [`kmeans_inputs`]' order.
pub const SPEED_NAMES: [&str; 2] = ["program seconds to stop / seed", "iterations / seed"];

/// The names of `futhark-kmeans-better`'s quality inputs a case, in [`kmeans_inputs_better`]' order; its speed inputs
/// are [`SPEED_NAMES`].
pub const QUALITY_NAMES_BETTER: [&str; 3] = ["final cosine inertia: cliff of I_cand / I_ref", "final mean log10 p (points off their centroid): cliff of L_ref / L_cand", "convergence: cliff of 1 + (k - k_ref), k the reference's further iterations from the final centroids"];

/// The names of the retired five-input scheme's inputs ([`kmeans_inputs_five`]).
pub const INPUT_NAMES_FIVE: [&str; 5] = ["cosine inertia / seed", "angular percentile p / seed", "wall seconds to stop / seed", "iterations / seed", "seed inertia gained an iteration / candidate"];

/// One run of a k-means program to UltraDim's stopping point, and what the fitness reads from it.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvergedRun {
    /// The final centroids and assignments, as the program left them (the assignments are the harness's, made
    /// against the centres the last step was given, as UltraDim's loop makes them).
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    /// 1. QUALITY of the final centroids: UltraDim's `cosine_inertia(points, centroids, assignments, d)`, and its
    ///    `exact_mean_log10_p` over the cosines of the points that are not at their centroid ([`AT_CENTROID_SIN2`]).
    pub inertia: f64,
    pub mean_log10_p: f64,
    /// UltraDim's `exact_mean_log10_p` over EVERY point, the prototype's own number: reported, never an input.
    pub mean_log10_p_all: f64,
    /// The points at their centroid, left out of `mean_log10_p`.
    pub coincident: usize,
    /// 2. THE PROGRAM'S OWN TIME to the stopping point, in seconds: the sum over the iterations of what the step
    ///    itself took (the GPU-timed passes of the shim's executable; `futhark c`'s own `-t` runtime), the
    ///    harness's assignment, the process and the pipe excluded exactly (the review's F4).
    pub seconds: f64,
    /// The harness's wall clock from the first assignment to the stop (the budget's number; never an input).
    pub wall_seconds: f64,
    /// 3. The iterations to the stopping point (UltraDim's `iterations`), and the Euclidean inertia gained an
    ///    iteration, (the inertia of the starting centres - the inertia at the stop) / iterations (the retired
    ///    scheme's input, reported).
    pub iterations: u32,
    pub inertia_per_iteration: f64,
    /// UltraDim's `converged`: the tolerance fired before the cap. False at the cap: the run is measured on the
    /// centroids it has (`finished` true) and its cost is [`MAX_ITERATIONS`].
    pub converged: bool,
    /// Whether the run could be measured: every step returned a `[k][d]f32` of finite values. A run that did not
    /// finish (an infinite or NaN centroid, a wrong shape, a step that failed) takes the worst value on every input.
    pub finished: bool,
    /// Why a run did not finish (empty for a finished run).
    pub failure: String,
    /// THE CONVERGENCE MEASURE of `futhark-kmeans-better` ([`kmeans_inputs_better`]): the iterations UltraDim's rule
    /// needs when the REFERENCE program's update is run on from this run's final centroids ([`further_iterations`],
    /// or the runtime's `k` command with a further entry). `None` when it was not measured (the other schemes, a run
    /// that did not finish).
    pub further_iterations: Option<u32>,
}

/// The inputs of one case: the quality group and the speed group, each in [0, 1] with 0 best and 1/2 "equal to the
/// seed", by `r / (1 + r)` of a ratio to the seed's run in the same interleaving.
#[derive(Debug, Clone, PartialEq)]
pub struct Inputs {
    pub quality: Vec<f64>,
    pub speed: Vec<f64>,
}

impl Inputs {
    /// The worst inputs: 1 on every input of both groups.
    pub fn worst() -> Inputs {
        Inputs { quality: vec![1.0; QUALITY_NAMES.len()], speed: vec![1.0; SPEED_NAMES.len()] }
    }
}

/// `r / (1 + r)` of the ratio `c / s`, TOTAL on [0, 1]: a seed measure of 0 divides the smallest positive f64 (so a
/// positive candidate measure is 1, the worst), a negative candidate measure is 0 (better than perfect), an infinite
/// or NaN ratio is 1, and so is a NaN on either side (`f64::max` would drop it). The seed's own measure against itself is
/// exactly 1/2.
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

/// THE INPUTS OF ONE CASE (`docs/KMEANS_FITNESS.md`), each defined for every outcome (the review's F1): the quality
/// group is the cosine inertia ratio and the ratio of the seed's mean log10 p to the candidate's (both are at most 0,
/// so the sign is dropped: a tighter candidate is more negative and scores below 1/2; a candidate with no tightness
/// at all, 0, scores 1); the speed group is the ratio of the program's own seconds and the ratio of the iterations.
/// A run that did not finish is 1 on every input. Nothing here is a function of another input: the Euclidean gain an
/// iteration of the retired scheme was the iterations input a second time (the review's F3).
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

/// THE INPUTS OF ONE CASE UNDER `futhark-kmeans-better` (Andrew's specification of 2026-11-03,
/// `docs/KMEANS_FITNESS.md`), each from `x` = the candidate relative to the reference program's run in the same
/// interleaving (`x = 1` equal, `x > 1` worse), each QUALITY input the cliff `crate::fitness::quality_cliff(x, cliff)`:
///
/// 1. the final cosine inertia, `x = I_cand / I_ref`;
/// 2. the final mean log10 angular percentile over the points off their centroid, `x = L_ref / L_cand` (both at most
///    0: a looser candidate is less negative and `x > 1`; a candidate with no tightness, 0, is `x = inf`);
/// 3. the convergence, `x = 1 + (k - k_ref)`: `k` the further iterations UltraDim's rule needs when the REFERENCE
///    program's update is run on from the candidate's final centroids, `k_ref` likewise from the reference's own
///    ([`ConvergedRun::further_iterations`]), so `x <= 1` when the candidate's centroids are at least as converged.
///
/// The two SPEED inputs are `futhark-kmeans-qqs`'s, each `r / (1 + r)`: the program's own seconds and the iterations
/// to the stopping rule. A run that did not finish, or a non-finite measure, is 1 on every input. A ratio's divisor of
/// 0 is the smallest positive f64 (as [`squashed_ratio`]). A finished run of either side without its further
/// iterations is a bug of our software, named.
pub fn kmeans_inputs_better(candidate: &ConvergedRun, reference: &ConvergedRun, cliff: f64) -> Result<Inputs, String> {
    if !(cliff > 0.0 && cliff.is_finite()) {
        return Err(format!("our software has a bug: a quality cliff of {cliff} (it must be positive and finite)"));
    }
    let worst = Inputs { quality: vec![1.0; QUALITY_NAMES_BETTER.len()], speed: vec![1.0; SPEED_NAMES.len()] };
    if !candidate.finished {
        return Ok(worst);
    }
    let k_ref = reference.further_iterations.ok_or("our software has a bug: the reference program's run has no further iterations measured (futhark-kmeans-better needs k_ref)")?;
    let k = candidate.further_iterations.ok_or("our software has a bug: a finished candidate run has no further iterations measured (futhark-kmeans-better needs k)")?;
    if ![candidate.inertia, candidate.mean_log10_p, candidate.seconds].iter().all(|v| v.is_finite()) {
        return Ok(worst);
    }
    let ratio = |c: f64, s: f64| c.max(0.0) / s.max(f64::MIN_POSITIVE);
    let x_inertia = ratio(candidate.inertia, reference.inertia);
    let x_percentile = ratio(-reference.mean_log10_p, -candidate.mean_log10_p);
    let x_convergence = 1.0 + (f64::from(k) - f64::from(k_ref));
    Ok(Inputs {
        quality: vec![crate::fitness::quality_cliff(x_inertia, cliff), crate::fitness::quality_cliff(x_percentile, cliff), crate::fitness::quality_cliff(x_convergence, cliff)],
        speed: vec![squashed_ratio(candidate.seconds, reference.seconds), squashed_ratio(f64::from(candidate.iterations), f64::from(reference.iterations))],
    })
}

/// The `x` of each of [`kmeans_inputs_better`]' quality inputs (before the cliff), for the reports: the inertia ratio,
/// the percentile ratio and the convergence `1 + (k - k_ref)` (NaN where a side has no further iterations).
pub fn better_ratios(candidate: &ConvergedRun, reference: &ConvergedRun) -> [f64; 3] {
    let ratio = |c: f64, s: f64| c.max(0.0) / s.max(f64::MIN_POSITIVE);
    let conv = match (candidate.further_iterations, reference.further_iterations) {
        (Some(k), Some(r)) => 1.0 + (f64::from(k) - f64::from(r)),
        _ => f64::NAN,
    };
    [ratio(candidate.inertia, reference.inertia), ratio(-reference.mean_log10_p, -candidate.mean_log10_p), conv]
}

/// THE FURTHER ITERATIONS from a run's final centroids (`futhark-kmeans-better`'s convergence measure): the REFERENCE
/// program's step `reference` run on from `centroids` under UltraDim's loop and stopping rule ([`converge`]), and the
/// iterations it takes. At a fixed point that is 2 (the first iteration has no previous inertia to compare). The
/// reference's continuation not finishing is a bug of our software (the reference program on finite centroids).
pub fn further_iterations<F>(points: &Array, centroids: &[Vec<f32>], reference: F) -> Result<u32, String>
where
    F: FnMut(&Array) -> Result<Stepped, String>,
{
    let d = points.dims.get(1).copied().ok_or("the points are not a [n][d] array")?;
    if centroids.iter().any(|c| c.len() != d) {
        return Err(format!("our software has a bug: final centroids that are not of the points' width {d}"));
    }
    let start = Array::f32s(vec![centroids.len(), d], centroids.concat());
    let r = converge(points, &start, reference)?;
    if !r.finished {
        return Err(format!("our software has a bug: the reference program's continuation from a run's final centroids did not finish: {}", r.failure));
    }
    Ok(r.iterations)
}

/// THE RETIRED SCHEME'S FIVE INPUTS (`futhark-kmeans-converged`, fuller bd929652), kept so that the name computes
/// what it did: cosine inertia ratio; 10^(difference of the mean log10 p over every point); the harness's WALL
/// CLOCK ratio; the iterations ratio; the seed's Euclidean gain an iteration over the candidate's. NOT TO BE RUN AS IS:
/// a candidate that gains nothing has a NaN here (`squash(inf)`), one that worsens a negative input, and the clock is
/// the harness's (the review's F1, F3, F4).
pub fn kmeans_inputs_five(candidate: &ConvergedRun, seed: &ConvergedRun) -> Vec<f64> {
    if !candidate.finished || !candidate.converged {
        return vec![1.0; 5];
    }
    let ratio = |c: f64, s: f64| c / s.max(f64::MIN_POSITIVE);
    let squash = |r: f64| r / (1.0 + r);
    vec![
        squash(ratio(candidate.inertia, seed.inertia)),
        squash(10f64.powf(candidate.mean_log10_p_all - seed.mean_log10_p_all)),
        squash(ratio(candidate.wall_seconds, seed.wall_seconds)),
        squash(ratio(f64::from(candidate.iterations), f64::from(seed.iterations))),
        squash(ratio(seed.inertia_per_iteration, candidate.inertia_per_iteration)),
    ]
}

/// The points of a case as flat `f32` with their dimension, from the `[n][d]f32` array.
fn flat_points(points: &Array) -> Result<(Vec<f32>, usize), String> {
    if points.elem != Elem::F32 || points.dims.len() != 2 {
        return Err(format!("the points are not a [n][d]f32: {:?} {:?}", points.elem, points.dims));
    }
    Ok((points.f32_values()?, points.dims[1]))
}

/// The centroids as rows, from a `[k][d]f32` array of the expected shape; another shape or element is the plain
/// reason the run did not finish.
fn centroid_rows(centres: &Array, k: usize, d: usize) -> Result<Vec<Vec<f32>>, String> {
    if centres.elem != Elem::F32 || centres.dims != [k, d] {
        return Err(format!("the centres are not a [{k}][{d}]f32: {:?} {:?}", centres.elem, centres.dims));
    }
    Ok(centres.f32_values()?.chunks_exact(d).map(<[f32]>::to_vec).collect())
}

/// UltraDim's `compute_distances` + `assign_points`: each point's nearest centroid by Euclidean distance (the sum of
/// squared differences in order, in `f32`, then `sqrt`), the first centroid on a tie (`distance < best_distance`
/// replaces only a strictly smaller one; a NaN distance never replaces). The points are assigned in parallel, each
/// point's sums in order, so the result is the sequential one bit for bit (the review's change 3: the harness's
/// assignment was 0.35-5.6 s an iteration sequential at 5,000 x 1,536).
pub fn assign(points: &[f32], d: usize, centroids: &[Vec<f32>]) -> Vec<usize> {
    let n = points.len() / d.max(1);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let p = &points[i * d..(i + 1) * d];
            let dist = |c: &[f32]| {
                let mut s = 0f32;
                for t in 0..d {
                    s += (p[t] - c[t]) * (p[t] - c[t]);
                }
                s.sqrt()
            };
            let mut best = 0usize;
            let mut best_distance = centroids.first().map_or(f32::INFINITY, |c| dist(c));
            for (j, c) in centroids.iter().enumerate().skip(1) {
                let x = dist(c);
                if x < best_distance {
                    best_distance = x;
                    best = j;
                }
            }
            best
        })
        .collect()
}

/// UltraDim's `calculate_inertia` (the stopping rule's number, `gpu_kmeans.rs` 1794): the sum over points of the
/// squared EUCLIDEAN distance to the assigned centroid, in `f32`, an assignment out of range contributing nothing.
/// Nothing is normalised: the loop is Euclidean Lloyd (the module header).
pub fn euclidean_inertia(points: &[f32], centroids: &[Vec<f32>], assignments: &[usize], d: usize) -> f32 {
    let n = points.len() / d.max(1);
    let mut inertia = 0f32;
    for i in 0..n {
        let a = assignments[i];
        if a < centroids.len() {
            let v = &points[i * d..(i + 1) * d];
            let dist_sq = v.iter().zip(centroids[a].iter()).map(|(v, c)| (v - c).powi(2)).sum::<f32>();
            inertia += dist_sq;
        }
    }
    inertia
}

/// A vector scaled to unit length; a zero vector stays zero (its cosine to everything is 0).
fn unit(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| (f64::from(*x) / norm) as f32).collect()
}

/// UltraDim's `cosine_inertia` (`lib.rs` 5709): `(1/n) sum (1 - x_i . mu_{a_i})` with unit centroids, each term in
/// `f32`, an assignment out of range contributing nothing. UltraDim sums the terms in rayon chunks whose count is its
/// thread count, so its bits depend on the machine; here the sum is in order, which is UltraDim's own result on one
/// thread (tested against a vendored copy, `tests::the_measures_equal_ultradims_vendored_functions`). `points` and
/// `centroids` are given as unit vectors by [`spherical`].
pub fn cosine_inertia(points: &[f32], centroids: &[Vec<f32>], assignments: &[usize], d: usize) -> f32 {
    let n = points.len() / d.max(1);
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0f32;
    for i in 0..n {
        let a = assignments[i];
        if a < centroids.len() {
            let v = &points[i * d..(i + 1) * d];
            let dot: f32 = v.iter().zip(&centroids[a]).map(|(x, y)| x * y).sum();
            acc += 1.0 - dot;
        }
    }
    acc / n as f32
}

/// UltraDim's two spherical measures of a clustering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spherical {
    /// `cosine_inertia`.
    pub inertia: f64,
    /// `exact_mean_log10_p` over the points that are not at their centroid (the fitness's).
    pub mean_log10_p: f64,
    /// `exact_mean_log10_p` over every point (UltraDim's number as it is).
    pub mean_log10_p_all: f64,
    /// The points at their centroid ([`AT_CENTROID_SIN2`]).
    pub coincident: usize,
}

/// UltraDim's two spherical measures of a clustering: the points and the centroids made unit (see the module
/// header), then `cosine_inertia` and `exact_mean_log10_p` over each point's cosine to its assigned unit centroid, over
/// every point and over the points not at their centroid (an assignment out of range has no cosine, as it has no term
/// in UltraDim's inertia). The cosines are the `f64` dots of the unit `f32` vectors, clamped to [-1, 1] as
/// `log10_angular_percentile` clamps them.
pub fn spherical(points: &[f32], centroids: &[Vec<f32>], assignments: &[usize], d: usize) -> Spherical {
    let n = points.len() / d.max(1);
    let unit_points: Vec<f32> = (0..n).flat_map(|i| unit(&points[i * d..(i + 1) * d])).collect();
    let unit_centroids: Vec<Vec<f32>> = centroids.iter().map(|c| unit(c)).collect();
    let inertia = f64::from(cosine_inertia(&unit_points, &unit_centroids, assignments, d));
    let cosines: Vec<f64> = (0..n)
        .filter(|i| assignments[*i] < centroids.len())
        .map(|i| {
            let v = &unit_points[i * d..(i + 1) * d];
            v.iter().zip(&unit_centroids[assignments[i]]).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>().clamp(-1.0, 1.0)
        })
        .collect();
    let off: Vec<f64> = cosines.iter().copied().filter(|c| !at_centroid(*c)).collect();
    Spherical { inertia, mean_log10_p: exact_mean_log10_p(&off, d), mean_log10_p_all: exact_mean_log10_p(&cosines, d), coincident: cosines.len() - off.len() }
}

/// Whether a point with this cosine to its centroid is AT the centroid ([`AT_CENTROID_SIN2`]): on the centroid's side
/// (a positive cosine; the antipode has the same `sin^2` and is the worst point, not a coincident one).
pub fn at_centroid(cos_t: f64) -> bool {
    cos_t > 0.0 && 1.0 - cos_t * cos_t < AT_CENTROID_SIN2
}

// ---- quality.rs: the dimension-corrected spherical percentile ----------------------------------------------------

/// `ln B(a, b)`, UltraDim's own library call (`quality.rs` line 50: `use statrs::function::beta::ln_beta`).
pub fn ln_beta(a: f64, b: f64) -> f64 {
    statrs::function::beta::ln_beta(a, b)
}

/// Natural log of the regularised incomplete beta `I_x(a, b)` (UltraDim `quality.rs` `ln_betainc`, line for line):
/// the prefactor in log space, the Lentz continued fraction, the reflection `I_x(a, b) = 1 - I_{1-x}(b, a)` past
/// `(a + 1)/(a + b + 2)`.
pub fn ln_betainc(a: f64, b: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && b > 0.0, "ln_betainc: a,b must be > 0");
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return 0.0;
    }
    let ln_front = a * x.ln() + b * (1.0 - x).ln() - a.ln() - ln_beta(a, b);
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front + betacf(a, b, x).ln()
    } else {
        let ln_front_r = b * (1.0 - x).ln() + a * x.ln() - b.ln() - ln_beta(b, a);
        let ln_tail = ln_front_r + betacf(b, a, 1.0 - x).ln();
        ln1p_neg_exp(ln_tail)
    }
}

/// `ln(1 - exp(v))` for `v <= 0` (UltraDim's two-branch split at `ln 2`).
fn ln1p_neg_exp(v: f64) -> f64 {
    debug_assert!(v <= 1e-12, "ln1p_neg_exp expects v <= 0, got {v}");
    if v > -std::f64::consts::LN_2 {
        (-(v.exp_m1())).ln()
    } else {
        (-v.exp()).ln_1p()
    }
}

/// Lentz's continued fraction for the incomplete beta (Numerical Recipes `betacf`, as UltraDim has it).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// `log10` of the fixed-direction angular percentile `F(theta)` of a point whose cosine to its unit centroid is `cos_t`,
/// in dimension `d` (UltraDim `quality.rs` `log10_angular_percentile`, line for line): no folding, the sign of the
/// cosine selects the branch, `sin^2` clamped to the smallest positive normal so a coincident point stays finite and
/// tightest.
pub fn log10_angular_percentile(cos_t: f64, d: usize) -> f64 {
    debug_assert!(d >= 2, "spherical CDF needs d >= 2");
    let cos_t = cos_t.clamp(-1.0, 1.0);
    let alpha = (d as f64 - 1.0) / 2.0;
    let x = (1.0 - cos_t * cos_t).max(f64::MIN_POSITIVE);
    const LOG10_HALF: f64 = -std::f64::consts::LOG10_2;
    let log10_half = LOG10_HALF + ln_betainc(alpha, 0.5, x) * std::f64::consts::LOG10_E;
    if cos_t >= 0.0 {
        log10_half
    } else {
        let half = 10f64.powf(log10_half);
        (1.0 - half).log10()
    }
}

/// UltraDim's REAL metric (`quality.rs` `exact_mean_log10_p`, line for line): the mean over points of
/// `log10 F(theta_i)`; smaller (more negative) is tighter than chance. The empty set is 0.
pub fn exact_mean_log10_p(cosines: &[f64], d: usize) -> f64 {
    if cosines.is_empty() {
        return 0.0;
    }
    let sum: f64 = cosines.iter().map(|&c| log10_angular_percentile(c, d)).sum();
    sum / cosines.len() as f64
}

// ---- the loop ------------------------------------------------------------------------------------------------------

/// What one step of a program returns: the next centres and THE PROGRAM'S OWN TIME for them, in seconds (the GPU-timed
/// passes of the shim's executable, `futhark c`'s own runtime, a Rust step's clock around itself), never the
/// harness's.
#[derive(Debug, Clone, PartialEq)]
pub struct Stepped {
    pub centres: Array,
    pub seconds: f64,
}

/// A Rust step as the loop takes it: the step's own clock around the call.
pub fn timed<F>(mut f: F) -> impl FnMut(&Array) -> Result<Stepped, String>
where
    F: FnMut(&Array) -> Result<Array, String>,
{
    move |c: &Array| {
        let t = Instant::now();
        let centres = f(c)?;
        Ok(Stepped { centres, seconds: t.elapsed().as_secs_f64() })
    }
}

/// THE LOOP (UltraDim `gpu_kmeans.rs` 375-436) around a program `step: centres -> centres` (the points fixed): for up to
/// [`MAX_ITERATIONS`], the harness assigns each point to its nearest of the current centres, the program makes the
/// next centres, and the Euclidean inertia of the new centres under those assignments is compared with the last
/// iteration's: a change under [`TOLERANCE`] stops with `converged` true; the cap stops with `converged` false and the
/// centroids it has, as UltraDim's loop does. `seconds` is the sum of the steps' own times ([`Stepped::seconds`]) and
/// `wall_seconds` this function's clock from the first assignment to the stop. A step whose output is not a
/// `[k][d]f32` of finite values, or that fails, is a run that did not finish (`finished` false, `failure` says why);
/// every measure is still filled from what there is.
pub fn converge<F>(points: &Array, start: &Array, mut step: F) -> Result<ConvergedRun, String>
where
    F: FnMut(&Array) -> Result<Stepped, String>,
{
    let (flat, d) = flat_points(points)?;
    if d < 2 {
        return Err(format!("the spherical percentile needs d >= 2 (UltraDim's quality.rs); this case has d = {d}"));
    }
    let k = start.dims.first().copied().ok_or("the starting centres have no rows")?;
    let start_rows = centroid_rows(start, k, d)?;
    let t0 = Instant::now();
    let start_inertia = f64::from(euclidean_inertia(&flat, &start_rows, &assign(&flat, d, &start_rows), d));
    let mut centres = start.clone();
    let mut centroids = start_rows;
    let mut assignments = Vec::new();
    let mut prev_inertia = f32::INFINITY;
    let mut current_inertia = f32::NAN;
    let mut iterations = 0u32;
    let mut seconds = 0.0;
    let mut converged = false;
    let mut failure = String::new();
    for iteration in 0..MAX_ITERATIONS {
        iterations = iteration + 1;
        assignments = assign(&flat, d, &centroids);
        let next = match step(&centres) {
            Ok(s) => s,
            Err(e) => {
                failure = format!("iteration {iterations}: the step failed: {e}");
                break;
            }
        };
        seconds += next.seconds;
        match centroid_rows(&next.centres, k, d) {
            Ok(rows) if rows.iter().flatten().all(|x| x.is_finite()) => centroids = rows,
            Ok(_) => {
                failure = format!("iteration {iterations}: a centroid is infinite or NaN");
                break;
            }
            Err(e) => {
                failure = format!("iteration {iterations}: {e}");
                break;
            }
        }
        centres = next.centres;
        current_inertia = euclidean_inertia(&flat, &centroids, &assignments, d);
        if (prev_inertia - current_inertia).abs() < TOLERANCE {
            converged = true;
            break;
        }
        prev_inertia = current_inertia;
    }
    let wall_seconds = t0.elapsed().as_secs_f64();
    let finished = failure.is_empty();
    let s = spherical(&flat, &centroids, &assignments, d);
    let final_inertia = if current_inertia.is_finite() { f64::from(current_inertia) } else { start_inertia };
    Ok(ConvergedRun {
        centroids,
        assignments,
        inertia: s.inertia,
        mean_log10_p: s.mean_log10_p,
        mean_log10_p_all: s.mean_log10_p_all,
        coincident: s.coincident,
        seconds,
        wall_seconds,
        iterations,
        inertia_per_iteration: (start_inertia - final_inertia) / f64::from(iterations.max(1)),
        converged,
        finished,
        failure,
        further_iterations: None,
    })
}

/// THE MEDIAN OF A SIDE'S RUNS (the advisor's round-3 review, N2): the values sorted; an odd count takes the middle
/// value, an even count the MEAN of the two middle values, so two pairs give the mean of the two runs. Before
/// 2026-11-03 this was `v[v.len() / 2]`, which of two sorted values is the larger: at `--pairs 2` each side's
/// seconds was its slower run. `crate::runner::median` is that rule (the A/A verdicts take it).
pub fn median_of_runs(seconds: &[f64]) -> f64 {
    crate::runner::median(seconds)
}

/// The candidate and the seed run INTERLEAVED on one case, `pairs` times each (the candidate first in the even pairs,
/// the seed first in the odd, so neither side always runs warm after the other): each side's run is its first, with
/// its `seconds` and `wall_seconds` the median of its runs ([`median_of_runs`]: two pairs, their mean). One pair is
/// the plain [`converge`] of each. THE SEED'S RUN RETURNED HERE IS THE YARDSTICK of that evaluation (the review's
/// F5): the candidate's inputs are taken against it, never against a seed run from another time.
pub fn converge_interleaved<C, S>(points: &Array, start: &Array, pairs: usize, mut candidate: C, mut seed: S) -> Result<(ConvergedRun, ConvergedRun), String>
where
    C: FnMut(&Array) -> Result<Stepped, String>,
    S: FnMut(&Array) -> Result<Stepped, String>,
{
    if pairs == 0 {
        return Err("our software has a bug: an interleaved run of 0 pairs".into());
    }
    let (mut c_runs, mut s_runs) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
    for i in 0..pairs {
        if i % 2 == 0 {
            c_runs.push(converge(points, start, &mut candidate)?);
            s_runs.push(converge(points, start, &mut seed)?);
        } else {
            s_runs.push(converge(points, start, &mut seed)?);
            c_runs.push(converge(points, start, &mut candidate)?);
        }
    }
    sides_of_runs(c_runs, s_runs)
}

/// THE TWO SIDES OF AN INTERLEAVING FROM THEIR RUNS (the batch path, where the runs come back as records, and
/// [`converge_interleaved`]'s own runs): each side's run is its first, with its `seconds` and `wall_seconds` the
/// median of its runs ([`median_of_runs`]); no run on a side is a bug.
pub fn sides_of_runs(candidate: Vec<ConvergedRun>, seed: Vec<ConvergedRun>) -> Result<(ConvergedRun, ConvergedRun), String> {
    if candidate.is_empty() || seed.is_empty() || candidate.len() != seed.len() {
        return Err(format!("our software has a bug: an interleaving of {} candidate runs and {} seed runs", candidate.len(), seed.len()));
    }
    let side = |runs: Vec<ConvergedRun>| {
        let mut r = runs[0].clone();
        r.seconds = median_of_runs(&runs.iter().map(|r| r.seconds).collect::<Vec<f64>>());
        r.wall_seconds = median_of_runs(&runs.iter().map(|r| r.wall_seconds).collect::<Vec<f64>>());
        r
    };
    Ok((side(candidate), side(seed)))
}

/// HOW A COMPILED STEP'S OWN TIME IS READ ([`step_of_exe`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepTimer {
    /// A `futhark c` executable: its `-t FILE` runtime, the entry point's microseconds with the input's reading and
    /// the output's writing excluded (Futhark's own measure).
    FutharkC,
    /// The GPU shim's executable: the sum of the GPU-timed compute passes of the one run
    /// (`futhark_wgpu_host::run_program_bytes` with timestamps, `ShimReport::fastest_run_ns(1)`): the device's clock
    /// around the program's dispatches only, as the other schemes' interleaved pairs time them.
    Gpu,
}

/// The step of a compiled program with the declared signature `(points: [n][d]f32) (centres: [k][d]f32) -> [k][d]f32`
/// (`futhark c`, or the GPU shim's executable: both read Futhark's binary format on stdin with `-b`), as [`converge`]
/// takes it: one process per iteration, the points written each time, and the step's own time read by `timer`.
pub fn step_of_exe<'a>(exe: &'a Path, points: &'a Array, timer: StepTimer) -> impl FnMut(&Array) -> Result<Stepped, String> + 'a {
    move |centres: &Array| {
        let stdin = write_binary(&[points, centres]);
        match timer {
            StepTimer::FutharkC => {
                let dir = exe.parent().ok_or("an executable with no directory")?;
                let runtime = dir.join(format!("runtime_{}.txt", std::process::id()));
                let out = crate::futhark_exec::output_with_timeout_bytes(std::process::Command::new(exe).arg("-b").arg("-t").arg(&runtime), std::time::Duration::from_secs(120), Some(&stdin))?;
                if !out.status.success() {
                    return Err(format!("{} failed:\n{}", exe.display(), String::from_utf8_lossy(&out.stderr)));
                }
                let text = std::fs::read_to_string(&runtime).map_err(|e| format!("{} wrote no runtime ({}): {e}", exe.display(), runtime.display()))?;
                let us: u64 = text.trim().lines().last().unwrap_or("").trim().parse().map_err(|_| format!("{} wrote a runtime that is not an integer of microseconds: {text:?}", exe.display()))?;
                Ok(Stepped { centres: read_one(&out.stdout)?, seconds: us as f64 / 1e6 })
            }
            StepTimer::Gpu => {
                let (out, report) = crate::futhark_wgpu_host::run_program_bytes(exe, &stdin, true, 1)?;
                if report.backend != "Metal" {
                    return Err(format!("the program ran on {}, not Metal", report.backend));
                }
                let ns = report.fastest_run_ns(1)?;
                Ok(Stepped { centres: read_one(&out)?, seconds: ns / 1e9 })
            }
        }
    }
}

// ---- the server: one process a program, kept alive across its iterations and cases ---------------------------------

/// A COMPILED PROGRAM KEPT ALIVE AS A FUTHARK SERVER (`futhark c --server`, `futhark_sig::compile_c_server`; the
/// Futhark server protocol: commands on stdin, each answered by lines and a `%%% OK`, a failure by `%%% FAILURE` and
/// its message before the `%%% OK`). Before this (2026-11-03) every iteration of UltraDim's loop was its own process
/// with the points written down the pipe (6-30 MB), 0.3-0.6 s a process: a candidate on the six fitness cases at
/// one pair cost about two minutes. Here a case's points are loaded ONCE (`restore` from a file written once), each
/// iteration restores the centres (16-300 KB), calls the entry point, stores the output and frees it, and the
/// program's own time is the `runtime:` line of the call, the same entry-point microseconds as `-t` writes. One
/// server serves a program's every case and every iteration; the seed's server lives for the whole run.
pub struct Server {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
    exe: std::path::PathBuf,
    /// This server's own scratch files (the centres in, the output out).
    dir: std::path::PathBuf,
    /// The points files, `points_<case>.bin`, shared by every server of a run (written once, restored by each).
    points_dir: std::path::PathBuf,
    /// The cases whose points are loaded, by name.
    loaded: std::collections::BTreeSet<String>,
}

impl Server {
    /// Start the server executable `exe`, its own scratch files (the centres, the output) in `dir`, the points files
    /// it restores in `points_dir` (shared: a case's file is written once), and read its greeting.
    pub fn start(exe: &Path, dir: &Path, points_dir: &Path) -> Result<Server, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        std::fs::create_dir_all(points_dir).map_err(|e| format!("mkdir {}: {e}", points_dir.display()))?;
        let mut child = std::process::Command::new(exe)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("{}: {e}", exe.display()))?;
        let stdin = child.stdin.take().ok_or("the server's stdin")?;
        let stdout = std::io::BufReader::new(child.stdout.take().ok_or("the server's stdout")?);
        let mut s = Server { child, stdin, stdout, exe: exe.to_path_buf(), dir: dir.to_path_buf(), points_dir: points_dir.to_path_buf(), loaded: std::collections::BTreeSet::new() };
        s.response("the greeting")?;
        Ok(s)
    }

    /// Read one response: the lines up to `%%% OK`; a `%%% FAILURE` block is the error, with its message.
    fn response(&mut self, what: &str) -> Result<Vec<String>, String> {
        use std::io::BufRead;
        let mut lines = Vec::new();
        let mut failure: Option<Vec<String>> = None;
        loop {
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).map_err(|e| format!("{}: reading {what}: {e}", self.exe.display()))?;
            if n == 0 {
                let err = self.child.stderr.take().map(|mut e| { let mut s = String::new(); let _ = std::io::Read::read_to_string(&mut e, &mut s); s }).unwrap_or_default();
                return Err(format!("{}: the server ended during {what}:\n{err}", self.exe.display()));
            }
            let line = line.trim_end_matches(['\n', '\r']).to_string();
            if line == "%%% OK" {
                return match failure {
                    Some(f) => Err(format!("{}: {what} failed:\n{}", self.exe.display(), f.join("\n"))),
                    None => Ok(lines),
                };
            }
            if line == "%%% FAILURE" {
                failure = Some(Vec::new());
            } else if let Some(f) = &mut failure {
                f.push(line);
            } else {
                lines.push(line);
            }
        }
    }

    /// Send one command and read its response.
    fn command(&mut self, line: &str) -> Result<Vec<String>, String> {
        use std::io::Write;
        writeln!(self.stdin, "{line}").and_then(|()| self.stdin.flush()).map_err(|e| format!("{}: writing {line:?}: {e}", self.exe.display()))?;
        self.response(line)
    }

    /// Load a case's points under its name, once: written to the shared file in Futhark's binary format when no
    /// server has written it yet (the bytes are the case's, so an existing file of that name is the same bytes),
    /// and restored.
    pub fn load_points(&mut self, case: &str, points: &Array) -> Result<(), String> {
        if self.loaded.contains(case) {
            return Ok(());
        }
        let file = self.points_dir.join(format!("points_{case}.bin"));
        let bytes = write_binary(&[points]);
        if std::fs::metadata(&file).map_or(true, |m| m.len() != bytes.len() as u64) {
            std::fs::write(&file, &bytes).map_err(|e| format!("write {}: {e}", file.display()))?;
        }
        self.command(&format!("restore {} p_{case} [][]f32", file.display()))?;
        self.loaded.insert(case.to_string());
        Ok(())
    }

    /// ONE STEP of the program on a loaded case: the centres restored, `main` called on the points and the centres,
    /// the output stored and read, the values freed; `seconds` is the call's own `runtime:` (microseconds).
    pub fn step(&mut self, case: &str, centres: &Array) -> Result<Stepped, String> {
        if !self.loaded.contains(case) {
            return Err(format!("our software has a bug: a step on {case}, whose points are not loaded in the server"));
        }
        let c_file = self.dir.join("centres.bin");
        let o_file = self.dir.join("out.bin");
        std::fs::write(&c_file, write_binary(&[centres])).map_err(|e| format!("write {}: {e}", c_file.display()))?;
        self.command(&format!("restore {} c [][]f32", c_file.display()))?;
        let called = self.command(&format!("call main o p_{case} c"));
        // The centres are freed whatever the call did, so a failed call leaves no value behind.
        let freed = self.command("free c");
        let lines = called?;
        freed?;
        let us: u64 = lines
            .iter()
            .find_map(|l| l.strip_prefix("runtime: "))
            .ok_or_else(|| format!("{}: the call wrote no runtime line: {lines:?}", self.exe.display()))?
            .trim()
            .parse()
            .map_err(|_| format!("{}: a runtime that is not an integer of microseconds: {lines:?}", self.exe.display()))?;
        self.command(&format!("store {} o", o_file.display()))?;
        self.command("free o")?;
        let bytes = std::fs::read(&o_file).map_err(|e| format!("read {}: {e}", o_file.display()))?;
        Ok(Stepped { centres: read_one(&bytes)?, seconds: us as f64 / 1e6 })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The step of a server on a loaded case, as [`converge`] takes it.
pub fn step_of_server<'a>(server: &'a mut Server, case: &'a str) -> impl FnMut(&Array) -> Result<Stepped, String> + 'a {
    move |centres: &Array| server.step(case, centres)
}

// ---- the resident-sample path: the sample on the device once, the cases as row ranges (`--eval rust`) -------------

/// THE RANGE ENTRY of a k-means program on the resident sample (Andrew, 2026-11-03): every case is a row range of
/// the one dbpedia sample, so the sample is uploaded to the device ONCE for a program and a case is `(start, n)`
/// with its own starting centres. The entry takes the whole sample, the range and the centres, and applies the
/// program's function to the range: `sample[start:start + n]` is a Futhark slice along the outer dimension of a
/// row-major array, a contiguous range, which the compiled program reads through an offset (an index function) and
/// never copies; `:> [n][d]f32` names its size. The runtime's `k` command (`futhark-rust-rt/src/host.rs`,
/// `kmeans_loop`) calls it with the centres kept on the device between iterations.
pub fn range_entry(sig: &Signature, entry: &str, callee: &str) -> Result<String, String> {
    let (points, rest) = sig.inputs.split_first().ok_or("a signature with no inputs")?;
    let [n, d] = points.dims.as_slice() else { return Err(format!("the first input {} is not a [n][d] array", points.name)) };
    let dims: Vec<&String> = sig.shape_dims().filter(|x| *x != n).collect();
    let mut s = format!("entry {entry} [m_sample]{}", dims.iter().map(|x| format!("[{x}]")).collect::<String>());
    s.push_str(&format!(" (sample: [m_sample][{d}]{}) (start: i64) ({n}: i64)", points.elem));
    for p in rest {
        s.push_str(&format!(" ({}: {})", p.name, sig.futhark_type(p, "")));
    }
    let args: Vec<&str> = rest.iter().map(|p| p.name.as_str()).collect();
    Ok(format!("{s} : {}{} =\n  {callee} (sample[start:start + {n}] :> [{n}][{d}]{}) {}\n", "[]".repeat(sig.output.dims.len()), sig.output.elem, points.elem, args.join(" ")))
}

/// A program's text with its `entry main` made a definition named `name` (the rest of the text unchanged).
fn main_as_def(text: &str, name: &str) -> Result<String, String> {
    if text.matches("entry main ").count() != 1 {
        return Err(format!("the program has {} `entry main ` (wants exactly 1)", text.matches("entry main ").count()));
    }
    Ok(text.replacen("entry main ", &format!("def {name} "), 1))
}

/// The entry names of the resident-sample module: the candidate's range entry and the seed's.
pub const RANGE_ENTRY_CANDIDATE: &str = "kc";
pub const RANGE_ENTRY_SEED: &str = "ky";

/// THE MODULE of the resident-sample path: the seed's program (its text, `entry main`) in a module of its own as
/// the definition `ky_main` with the range entry [`RANGE_ENTRY_SEED`] over it, and, when given, the candidate's
/// program as the definition `kc_main` with the range entry [`RANGE_ENTRY_CANDIDATE`]; the kingdom's prelude once.
/// Candidate and seed in ONE module, so one `Session::run` interleaves them case by case with the sample resident.
pub fn resident_module(k: &crate::futhark_table::FutharkKingdom, sig: &Signature, candidate: Option<&Tree>, seed_text: &str) -> Result<String, String> {
    let mut s = format!("-- the resident-sample k-means module (fuller::futhark_sig::kmeans_converged)\n{}", super::prelude(k));
    let seed_def = main_as_def(seed_text, "main")?;
    s.push_str(&format!("module m_seed = {{\n{seed_def}\n}}\n{}\n", range_entry(sig, RANGE_ENTRY_SEED, "m_seed.main")?));
    if let Some(t) = candidate {
        let def = super::program_named_bare(k, sig, t, "kc_main")?.replacen("entry kc_main ", "def kc_main ", 1);
        s.push_str(&format!("{def}\n{}\n", range_entry(sig, RANGE_ENTRY_CANDIDATE, "kc_main")?));
    }
    Ok(s)
}

/// One `k` command of the runtime's host: `k ENTRY CASE START N CAP TOL` (the cap and the tolerance UltraDim's).
pub fn k_command(entry: &str, case: usize, start: usize, n: usize) -> String {
    format!("k {entry} {case} {start} {n} {MAX_ITERATIONS} {TOLERANCE:e}")
}

/// One `k` command WITH THE FURTHER ITERATIONS (`futhark-kmeans-better`): `k ENTRY CASE START N CAP TOL FURTHER`, after
/// the loop the runtime runs the entry `further` (the reference program's) on from the final centroids, on the device,
/// under the same rule, and the record's header carries its iterations ([`run_of_k_payload`]).
pub fn k_command_further(entry: &str, case: usize, start: usize, n: usize, further: &str) -> String {
    format!("{} {further}", k_command(entry, case, start, n))
}

/// THE RUN A `k` RECORD DESCRIBES, as [`converge`] would have built it: the payload's line (iterations, converged,
/// finished, the program's own nanoseconds, the start and final Euclidean inertia) and, for a finished run, its final
/// centres `[k][d]f32` and THE ASSIGNMENTS THE LOOP STOPPED ON, `[n]i32` (the harness's, made against the centres
/// the last step was given: UltraDim's loop assigns before the update, the module header's F12), else its failure.
/// The spherical measures are taken on those centres under those assignments, which is the ONE definition of the
/// final clustering, `converge`'s. Before 2026-11-03 this function assigned the points to the FINAL centres (the
/// advisor's round-3 review, N1): at a fixed point the two agree, so the seed and every converged candidate read
/// the same, but a candidate stopped by the cap or by a sub-tolerance move was scored on another partition than on
/// the c backend. A finished record without its assignments is an error naming the runtime. `wall_ns` is the
/// record's own span. A record of a `k` command with a further entry ([`k_command_further`]) has two more header
/// fields, the further iterations and whether that continuation converged: a finished run's are its
/// [`ConvergedRun::further_iterations`] (a continuation that did not finish is the runtime's failure, a bug).
pub fn run_of_k_payload(points: &Array, start: &Array, payload: &[u8], wall_ns: u64) -> Result<ConvergedRun, String> {
    let nl = payload.iter().position(|b| *b == b'\n').ok_or("a k-means record with no header line")?;
    let head = std::str::from_utf8(&payload[..nl]).map_err(|_| "a k-means record whose header is not text")?;
    let f: Vec<&str> = head.split(' ').collect();
    let bad = |what: &str| format!("a k-means record with a bad {what}: {head:?}");
    let (iterations, converged, finished, ns, start_inertia, final_inertia, further) = match f.as_slice() {
        [a, b, c, d, e, g] => (a, b, c, d, e, g, None),
        [a, b, c, d, e, g, k, kc] => (a, b, c, d, e, g, Some((k, kc))),
        _ => return Err(format!("a k-means record header {head:?}")),
    };
    let iterations: u32 = iterations.parse().map_err(|_| bad("iteration count"))?;
    let converged = *converged == "1";
    let finished = *finished == "1";
    let ns: f64 = ns.parse().map_err(|_| bad("nanoseconds"))?;
    let start_inertia: f32 = start_inertia.parse().map_err(|_| bad("start inertia"))?;
    let final_inertia: f32 = final_inertia.parse().map_err(|_| bad("final inertia"))?;
    let further_iterations = match further {
        None => None,
        Some((k, kc)) => {
            let k: u32 = k.parse().map_err(|_| bad("further iteration count"))?;
            if finished && *kc != "1" && k < MAX_ITERATIONS {
                return Err(format!("our software has a bug: the reference program's continuation stopped after {k} iterations without converging or reaching the cap: {head:?}"));
            }
            finished.then_some(k)
        }
    };
    let (flat, d) = flat_points(points)?;
    let n = flat.len() / d.max(1);
    let k = start.dims.first().copied().ok_or("the starting centres have no rows")?;
    let (centroids, assignments, failure) = if finished {
        let values = super::read_binary(&payload[nl + 1..])?;
        let [centres, assigned] = values.as_slice() else {
            return Err(format!("a finished k-means record with {} values after its header, wants 2 (the final centres and the assignments the loop stopped on): the runtime that wrote it is older than this reader", values.len()));
        };
        let centroids = centroid_rows(centres, k, d)?;
        if assigned.elem != Elem::I32 || assigned.dims != [n] {
            return Err(format!("a k-means record whose assignments are not a [{n}]i32: {:?} {:?}", assigned.elem, assigned.dims));
        }
        let assignments: Vec<usize> = assigned.i32_values()?.into_iter().map(|a| usize::try_from(a).ok().filter(|a| *a < k).ok_or_else(|| format!("our software has a bug: a k-means record assigns a point to centre {a} of {k}"))).collect::<Result<_, String>>()?;
        (centroids, assignments, String::new())
    } else {
        // A run that did not finish takes the worst inputs whatever its clustering: the start, as `converge` would
        // hold before its first step.
        let centroids = centroid_rows(start, k, d)?;
        let assignments = assign(&flat, d, &centroids);
        (centroids, assignments, String::from_utf8_lossy(&payload[nl + 1..]).into_owned())
    };
    let s = spherical(&flat, &centroids, &assignments, d);
    Ok(ConvergedRun {
        centroids,
        assignments,
        inertia: s.inertia,
        mean_log10_p: s.mean_log10_p,
        mean_log10_p_all: s.mean_log10_p_all,
        coincident: s.coincident,
        seconds: ns / 1e9,
        wall_seconds: wall_ns as f64 / 1e9,
        iterations,
        inertia_per_iteration: (f64::from(start_inertia) - f64::from(final_inertia)) / f64::from(iterations.max(1)),
        converged,
        finished,
        failure,
        further_iterations,
    })
}

/// The one line of a `k` record's payload, as the runtime writes it (a test's stand-in for the device; the runtime's
/// own text is `kmeans_payload` in `host.rs`), then `tail`: [`k_tail`] of a finished run, the failure's text of one
/// that did not finish.
pub fn k_payload(iterations: u32, converged: bool, finished: bool, ns: f64, start_inertia: f32, final_inertia: f32, tail: &[u8]) -> Vec<u8> {
    let mut p = format!("{iterations} {} {} {ns:.1} {start_inertia} {final_inertia}\n", u8::from(converged), u8::from(finished)).into_bytes();
    p.extend_from_slice(tail);
    p
}

/// A `k` record's payload with THE FURTHER ITERATIONS added ([`k_command_further`]; a test's stand-in for the runtime,
/// whose own text is `kmeans_payload` in `host.rs`): the header line of `payload` ([`k_payload`]) with two more fields,
/// the further iterations and whether that continuation converged; the tail unchanged. A payload with no header line
/// is an error.
pub fn k_payload_further(payload: &[u8], further: (u32, bool)) -> Result<Vec<u8>, String> {
    let nl = payload.iter().position(|b| *b == b'\n').ok_or("a k-means payload with no header line")?;
    let mut p = payload[..nl].to_vec();
    p.extend_from_slice(format!(" {} {}\n", further.0, u8::from(further.1)).as_bytes());
    p.extend_from_slice(&payload[nl + 1..]);
    Ok(p)
}

/// The tail of a finished `k` record, as the runtime writes it: the final centres, then the assignments the loop
/// stopped on as a `[n]i32`, both in Futhark's binary format.
pub fn k_tail(centres: &Array, assignments: &[usize]) -> Vec<u8> {
    let assigned = Array::i32s(vec![assignments.len()], assignments.iter().map(|a| *a as i32).collect());
    write_binary(&[centres, &assigned])
}

// ---- the cases ---------------------------------------------------------------------------------------------------

/// One stored case of the k-means problem, cut from the dbpedia sample: its name, split, the sample rows
/// `start..end` it holds, its `k`, and the seed of its k-means++ starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaseCut {
    pub name: &'static str,
    pub split: &'static str,
    pub start: usize,
    pub end: usize,
    pub k: usize,
    pub seed: u64,
}

/// THE CASES (the advisor's review of 2026-11-03, "What the data section should decide"): six disjoint row sets of
/// the 5,000-row sample, 5,000 rows used once, k at most n / 20 so singleton clusters are rare (the review's list had
/// 1,000 rows with k 64, over its own rule; k 50 here); the held cases hold
/// out rows; the test case is never a fitness input. The seventh, `train_n200_k4`, is the first 200 rows of
/// `train_n1000_k16` (its rows are training rows twice): the one case small enough for the host interpreter, which
/// the kingdom's tests run through (random genes, the textbook program, the seed against its Rust port); the runner
/// compiles every case. The starting centres are UltraDim's k-means++ draws over the case's rows at the case's seed
/// ([`kmeans_plus_plus_rows`]), stored in the case's `.in` after its points, so every candidate and the seed start
/// from the same centres (the iterations depend on them, the module header). `make_cases.sh` writes the files from
/// the sample alone (`examples/kmeans_cases.rs`), and `cases.tsv` records this table.
pub const KMEANS_CASES: [CaseCut; 7] = [
    CaseCut { name: "train_n1000_k16", split: "train", start: 0, end: 1000, k: 16, seed: 1 },
    CaseCut { name: "train_n1000_k50", split: "train", start: 1000, end: 2000, k: 50, seed: 2 },
    CaseCut { name: "train_n500_k8", split: "train", start: 2000, end: 2500, k: 8, seed: 3 },
    CaseCut { name: "train_n200_k4", split: "train", start: 0, end: 200, k: 4, seed: 4 },
    CaseCut { name: "held_n1000_k32", split: "held", start: 2500, end: 3500, k: 32, seed: 5 },
    CaseCut { name: "held_n500_k16", split: "held", start: 3500, end: 4000, k: 16, seed: 6 },
    CaseCut { name: "test_n1000_k32", split: "test", start: 4000, end: 5000, k: 32, seed: 7 },
];

/// The timing input's cut: every row of the sample, k = 256, its starts at this seed.
pub const KMEANS_TIMING_CUT: CaseCut = CaseCut { name: "timing_n5000_k256", split: "timing", start: 0, end: 5000, k: 256, seed: 8 };

/// A cut's points (the sample's rows `start..end`) and its k-means++ starts, as the case's two inputs.
pub fn cut_case(sample: &Array, cut: &CaseCut) -> Result<(Array, Array), String> {
    let (flat, d) = flat_points(sample)?;
    let n = flat.len() / d.max(1);
    if cut.start >= cut.end || cut.end > n {
        return Err(format!("{}: the rows {}..{} are not within the sample's {n}", cut.name, cut.start, cut.end));
    }
    let pts: Vec<f32> = flat[cut.start * d..cut.end * d].to_vec();
    let chosen = kmeans_plus_plus_rows(&pts, d, cut.k, cut.seed)?;
    let starts: Vec<f32> = chosen.iter().flat_map(|i| pts[i * d..(i + 1) * d].to_vec()).collect();
    Ok((Array::f32s(vec![cut.end - cut.start, d], pts), Array::f32s(vec![cut.k, d], starts)))
}

// ---- UltraDim's starting centres -------------------------------------------------------------------------------

/// UltraDim's `squared_distance` (`gpu_kmeans.rs` 1894): the in-order `f32` sum of squared differences.
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// ULTRADIM'S K-MEANS++ STARTING CENTRES (`gpu_kmeans.rs` `init_centroids_plus_plus`, 480-534, with `random_seed:
/// Some(seed)`, `init_rng` 470: `ChaCha8Rng::seed_from_u64`), line for line: the first centre a uniform row
/// (`random_range(0..n)`), each next one a row drawn with probability proportional to its squared distance to the
/// nearest centre so far (`random::<f32>() * total`, the cumulative sum in row order, the first row at or past the
/// target), all rows identical duplicating the first. The chosen row indices, in the order drawn; the draws are
/// UltraDim's (the same crates at the same versions, `Cargo.toml`). The distances are computed in parallel over the
/// rows (each row's fold over the centres in order), the total summed in row order, so the result is the sequential one.
pub fn kmeans_plus_plus_rows(points: &[f32], d: usize, k: usize, seed: u64) -> Result<Vec<usize>, String> {
    let n = points.len() / d.max(1);
    if n == 0 || k == 0 || k > n {
        return Err(format!("k-means++ needs 0 < k <= n: n = {n}, k = {k}"));
    }
    let row = |i: usize| &points[i * d..(i + 1) * d];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    chosen.push(rng.random_range(0..n));
    for _ in 1..k {
        let distances: Vec<f32> = (0..n).into_par_iter().map(|i| chosen.iter().map(|c| squared_distance(row(i), row(*c))).fold(f32::INFINITY, f32::min)).collect();
        let mut total_distance = 0.0f32;
        for x in &distances {
            total_distance += x;
        }
        if total_distance == 0.0 {
            chosen.push(chosen[0]);
            continue;
        }
        let target = rng.random::<f32>() * total_distance;
        let mut cumulative = 0.0f32;
        let mut next = None;
        for (idx, dist) in distances.iter().enumerate() {
            cumulative += dist;
            if cumulative >= target {
                next = Some(idx);
                break;
            }
        }
        // UltraDim's loop pushes nothing when the cumulative sum stays under the target by rounding; the centre
        // count would then be short, which its callers do not check. Here that is an error, named.
        chosen.push(next.ok_or_else(|| format!("k-means++: the cumulative distance {cumulative} never reached the target {target} of {total_distance}"))?);
    }
    Ok(chosen)
}

// ---- UltraDim's step in Rust, the oracle of the translation -------------------------------------------------------

/// ULTRADIM'S K-MEANS STEP IN RUST, construct for construct from its three kernels (`gpu_kmeans.rs`: `compute_distances`,
/// `assign_points`, `update_centroids`), in `f32`: each point's nearest centre by `sqrt` of the in-order `f32` sum of
/// squared differences, the first centre on a tie; each (cluster, coordinate) the Kahan-compensated sum over the
/// points in point order (`y = x - c; t = s + y; c = (t - s) - y; s = t`) divided by the count; a cluster with no
/// point KEEPS ITS CENTRE (`if (count > 0u)`). The oracle the translated seed is checked against (plan 4c (d)), and
/// the stand-in seed of this module's tests until it lands. `kahan` false drops the compensation (a plain sum), the
/// uncompensated form the breaking case is built to separate.
pub fn ultradim_step_f32(points: &Array, centres: &Array, kahan: bool) -> Result<Array, String> {
    let (flat, d) = flat_points(points)?;
    let k = centres.dims.first().copied().ok_or("the centres have no rows")?;
    let rows = centroid_rows(centres, k, d)?;
    let assignments = assign(&flat, d, &rows);
    let n = flat.len() / d.max(1);
    let mut out = rows.clone();
    out.par_iter_mut().enumerate().for_each(|(j, row)| {
        for (t, cell) in row.iter_mut().enumerate() {
            let (mut sum, mut sum_c, mut count) = (0f32, 0f32, 0u32);
            for i in 0..n {
                if assignments[i] == j {
                    let term = flat[i * d + t];
                    if kahan {
                        let y = term - sum_c;
                        let tt = sum + y;
                        sum_c = (tt - sum) - y;
                        sum = tt;
                    } else {
                        sum += term;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                *cell = sum / count as f32;
            }
        }
    });
    Ok(Array::f32s(vec![k, d], out.concat()))
}

/// THE F64 REFERENCE of UltraDim's step (the same algorithm in `f64`, Kahan included): what
/// `tests/corpus/futhark_problems/kmeans/reference64.fut` computes, kept to check the translation of the kernels,
/// never to score (`docs/KMEANS_FITNESS.md`). Returns the assignments and the centres as `f64`.
pub fn ultradim_step_f64(points: &Array, centres: &Array) -> Result<(Vec<usize>, Vec<f64>), String> {
    let (flat, d) = flat_points(points)?;
    let k = centres.dims.first().copied().ok_or("the centres have no rows")?;
    let rows = centroid_rows(centres, k, d)?;
    let n = flat.len() / d.max(1);
    let p64: Vec<f64> = flat.iter().map(|x| f64::from(*x)).collect();
    let c64: Vec<f64> = rows.concat().iter().map(|x| f64::from(*x)).collect();
    let assignments: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            let dist = |j: usize| {
                let mut s = 0f64;
                for t in 0..d {
                    s += (p64[i * d + t] - c64[j * d + t]) * (p64[i * d + t] - c64[j * d + t]);
                }
                s.sqrt()
            };
            let mut best = 0usize;
            let mut best_distance = if k > 0 { dist(0) } else { f64::INFINITY };
            for j in 1..k {
                let x = dist(j);
                if x < best_distance {
                    best_distance = x;
                    best = j;
                }
            }
            best
        })
        .collect();
    let mut out = c64.clone();
    out.par_chunks_mut(d.max(1)).enumerate().for_each(|(j, row)| {
        for (t, cell) in row.iter_mut().enumerate() {
            let (mut sum, mut sum_c, mut count) = (0f64, 0f64, 0u64);
            for i in 0..n {
                if assignments[i] == j {
                    let term = p64[i * d + t];
                    let y = term - sum_c;
                    let tt = sum + y;
                    sum_c = (tt - sum) - y;
                    sum = tt;
                    count += 1;
                }
            }
            if count > 0 {
                *cell = sum / count as f64;
            }
        }
    });
    Ok((assignments, out))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_better;

#[cfg(test)]
mod ultradim;
