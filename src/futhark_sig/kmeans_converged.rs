//! THE K-MEANS FITNESS, `futhark-kmeans-converged` (`docs/KMEANS_FITNESS.md`, the specification as code; Andrew,
//! 2026-11-03): k-means has no exact answer, so a program is not compared with a reference program's bits, cells or
//! memberships. It is RUN TO ULTRADIM'S STOPPING RULE on each fitness case, candidate and seed alike, and what the
//! fitness reads is the clustering it reaches and how it got there: the quality of the final centroids (UltraDim's own
//! two numbers), the wall clock to the stopping point, the iterations, and the inertia gained an iteration, each
//! relative to the seed's run on the same case ([`kmeans_inputs`]). One stage, no gate.
//!
//! The prototype is `lib/projection-voting-kmeans` of UltraDim (read 2026-11-03, read-only): `gpu_kmeans.rs` 375-425
//! is the loop (assign, update, `calculate_inertia`; stop when the inertia changes by less than `tolerance` = 1e-4 or
//! after `max_iterations` = 100), `lib.rs` 5705 `cosine_inertia` and `quality.rs` 188 `exact_mean_log10_p` are the
//! quality measures. The loop, the stopping test and the measures are the harness's, here in Rust; the program being
//! evolved is the BODY of the loop, one step `(points, centres) -> centres` (the two kernels, assignment and update).
//! The harness's assignment of each point for the measures is UltraDim's `assign_points` rule (the nearest centre by
//! Euclidean distance, the first on a tie), made against the centres the step was given, as UltraDim's loop makes it.
//!
//! UltraDim's quality measures are spherical: its points are unit vectors by construction and its centroids are
//! normalised, so `cosine_inertia` reads `1 - x . mu` as a cosine. The cases here are not unit (magnitudes from 10^-3 to
//! 10^5), so the harness normalises each point and each centroid before the two spherical measures; on unit data this
//! is the identity and the measures are UltraDim's exactly. A zero vector has cosine 0 to everything.

use std::path::Path;
use std::time::Instant;

use super::{read_one, write_binary, Array, Elem};

/// UltraDim's `tolerance`: the loop stops when the inertia changes by less than this.
pub const TOLERANCE: f32 = 1e-4;

/// UltraDim's `max_iterations`: a run that reaches it without converging did not finish.
pub const MAX_ITERATIONS: u32 = 100;

/// The names of the five inputs a case, in [`kmeans_inputs`]' order.
pub const INPUT_NAMES: [&str; 5] = ["cosine inertia / seed", "angular percentile p / seed", "seconds to stop / seed", "iterations / seed", "seed inertia gained an iteration / candidate"];

/// One run of a k-means program to UltraDim's stopping point, and what the fitness reads from it.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvergedRun {
    /// The final centroids and assignments, as the program left them (the assignments are the harness's, made
    /// against the centres the last step was given, as UltraDim's loop makes them).
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    /// 1. QUALITY of the final centroids, both of UltraDim's numbers: `cosine_inertia(points, centroids, assignments,
    ///    d)` and `exact_mean_log10_p(cosines of each point to its assigned unit centroid, d)`.
    pub inertia: f64,
    pub mean_log10_p: f64,
    /// 2. ELAPSED CONVERGENCE SPEED: the wall clock from the first assignment to the stopping point, in seconds
    ///    (measured interleaved with the seed's run on the same device: [`converge_interleaved`]).
    pub seconds: f64,
    /// 3. RATE OF CONVERGENCE: the iterations to the stopping point (UltraDim's `iterations`), and the quality gained
    ///    per iteration, (the Euclidean inertia of the starting centres - the inertia at the stop) / iterations.
    pub iterations: u32,
    pub inertia_per_iteration: f64,
    /// A run that did not finish (an infinite or NaN centroid, a wrong shape, the iteration cap with no convergence)
    /// takes the worst value on every input.
    pub finished: bool,
    /// Why a run did not finish (empty for a finished run).
    pub failure: String,
}

/// The HFF inputs of one case: every measure its own input, each relative to the seed's run on the same case, each
/// in [0, 1] with 0 best, by `r / (1 + r)` of a ratio (1/2 is "equal to the seed") or a difference squashed the same
/// way. Exactly `docs/KMEANS_FITNESS.md`.
pub fn kmeans_inputs(candidate: &ConvergedRun, seed: &ConvergedRun) -> Vec<f64> {
    if !candidate.finished {
        return vec![1.0; 5];
    }
    let ratio = |c: f64, s: f64| c / s.max(f64::MIN_POSITIVE);
    let squash = |r: f64| r / (1.0 + r);
    vec![
        // 1. quality of the final centroids
        squash(ratio(candidate.inertia, seed.inertia)),                             // cosine inertia, candidate / seed
        squash(10f64.powf(candidate.mean_log10_p - seed.mean_log10_p)),             // the angular-percentile metric, as a ratio of p
        // 2. elapsed convergence speed
        squash(ratio(candidate.seconds, seed.seconds)),                             // wall clock to the stopping point
        // 3. rate of convergence
        squash(ratio(f64::from(candidate.iterations), f64::from(seed.iterations))), // iterations to the stopping point
        squash(ratio(seed.inertia_per_iteration, candidate.inertia_per_iteration)), // quality gained an iteration (more is better, so inverted)
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
/// replaces only a strictly smaller one; a NaN distance never replaces).
pub fn assign(points: &[f32], d: usize, centroids: &[Vec<f32>]) -> Vec<usize> {
    let n = points.len() / d.max(1);
    (0..n)
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

/// UltraDim's `calculate_inertia` (the stopping rule's number): the sum over points of the squared Euclidean distance
/// to the assigned centroid, in `f32`, an assignment out of range contributing nothing.
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

/// UltraDim's `cosine_inertia`: `(1/n) sum (1 - x_i . mu_{a_i})` with unit centroids, each term in `f32`, an
/// assignment out of range contributing nothing (UltraDim sums the terms in parallel chunks whose count is the
/// thread count; here in order). `points` and `centroids` are given as unit vectors by [`spherical`].
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

/// UltraDim's two spherical measures of a clustering: the points and the centroids made unit (see the module header),
/// then `cosine_inertia` and `exact_mean_log10_p` over each point's cosine to its assigned unit centroid (an
/// assignment out of range has no cosine, as it has no term in UltraDim's inertia).
pub fn spherical(points: &[f32], centroids: &[Vec<f32>], assignments: &[usize], d: usize) -> (f64, f64) {
    let n = points.len() / d.max(1);
    let unit_points: Vec<f32> = (0..n).flat_map(|i| unit(&points[i * d..(i + 1) * d])).collect();
    let unit_centroids: Vec<Vec<f32>> = centroids.iter().map(|c| unit(c)).collect();
    let inertia = f64::from(cosine_inertia(&unit_points, &unit_centroids, assignments, d));
    let cosines: Vec<f64> = (0..n)
        .filter(|i| assignments[*i] < centroids.len())
        .map(|i| {
            let v = &unit_points[i * d..(i + 1) * d];
            v.iter().zip(&unit_centroids[assignments[i]]).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>()
        })
        .collect();
    (inertia, exact_mean_log10_p(&cosines, d))
}

// ---- quality.rs: the dimension-corrected spherical percentile ----------------------------------------------------

/// `ln Gamma(x)` for `x > 0`: the Lanczos approximation (g = 7, nine coefficients), good to about 1e-15 relative over
/// the arguments the percentile uses (`(d - 1) / 2`, `1/2` and their sum). `statrs::function::beta::ln_beta`, which
/// UltraDim calls, is `ln Gamma(a) + ln Gamma(b) - ln Gamma(a + b)` too.
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // The reflection formula keeps the series on x >= 0.5.
        let s = (std::f64::consts::PI * x).sin();
        return (std::f64::consts::PI / s.abs()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// `ln B(a, b)`.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Natural log of the regularised incomplete beta `I_x(a, b)` (UltraDim `quality.rs` `ln_betainc`): the prefactor in
/// log space, the Lentz continued fraction, the reflection `I_x(a, b) = 1 - I_{1-x}(b, a)` past `(a + 1)/(a + b + 2)`.
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
/// in dimension `d` (UltraDim `quality.rs` `log10_angular_percentile`): no folding, the sign of the cosine selects
/// the branch, `sin^2` clamped to the smallest positive normal so a coincident point stays finite and tightest.
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

/// UltraDim's REAL metric (`quality.rs` `exact_mean_log10_p`): the mean over points of `log10 F(theta_i)`; smaller
/// (more negative) is tighter than chance. The empty set is 0.
pub fn exact_mean_log10_p(cosines: &[f64], d: usize) -> f64 {
    if cosines.is_empty() {
        return 0.0;
    }
    cosines.iter().map(|&c| log10_angular_percentile(c, d)).sum::<f64>() / cosines.len() as f64
}

// ---- the loop ------------------------------------------------------------------------------------------------------

/// THE LOOP (UltraDim `gpu_kmeans.rs` 375-425) around a program `step: centres -> centres` (the points fixed): for up to
/// [`MAX_ITERATIONS`], the harness assigns each point to its nearest of the current centres, the program makes the
/// next centres, and the Euclidean inertia of the new centres under those assignments is compared with the last
/// iteration's: a change under [`TOLERANCE`] stops. The wall clock is this function's, from the first assignment to
/// the stop. A step whose output is not a `[k][d]f32` of finite values, or that fails, or the cap reached without
/// convergence, is a run that did not finish (`finished` false, `failure` says why); every measure is still filled
/// from what there is.
pub fn converge<F>(points: &Array, start: &Array, mut step: F) -> Result<ConvergedRun, String>
where
    F: FnMut(&Array) -> Result<Array, String>,
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
    let mut finished = false;
    let mut failure = String::new();
    for iteration in 0..MAX_ITERATIONS {
        iterations = iteration + 1;
        assignments = assign(&flat, d, &centroids);
        let next = match step(&centres) {
            Ok(a) => a,
            Err(e) => {
                failure = format!("iteration {iterations}: the step failed: {e}");
                break;
            }
        };
        match centroid_rows(&next, k, d) {
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
        centres = next;
        current_inertia = euclidean_inertia(&flat, &centroids, &assignments, d);
        if (prev_inertia - current_inertia).abs() < TOLERANCE {
            finished = true;
            break;
        }
        prev_inertia = current_inertia;
    }
    let seconds = t0.elapsed().as_secs_f64();
    if !finished && failure.is_empty() {
        failure = format!("the iteration cap {MAX_ITERATIONS} reached without convergence (the last change {})", (prev_inertia - current_inertia).abs());
    }
    let (inertia, mean_log10_p) = spherical(&flat, &centroids, &assignments, d);
    let final_inertia = if current_inertia.is_finite() { f64::from(current_inertia) } else { start_inertia };
    Ok(ConvergedRun { centroids, assignments, inertia, mean_log10_p, seconds, iterations, inertia_per_iteration: (start_inertia - final_inertia) / f64::from(iterations.max(1)), finished, failure })
}

/// The candidate and the seed run INTERLEAVED on one case, `pairs` times each (the candidate first in the even pairs,
/// the seed first in the odd, so neither side always runs warm after the other): each side's run is its first, with
/// its `seconds` the median of its runs. One pair is the plain [`converge`] of each.
pub fn converge_interleaved<C, S>(points: &Array, start: &Array, pairs: usize, mut candidate: C, mut seed: S) -> Result<(ConvergedRun, ConvergedRun), String>
where
    C: FnMut(&Array) -> Result<Array, String>,
    S: FnMut(&Array) -> Result<Array, String>,
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
    let median = |runs: &[ConvergedRun]| {
        let mut s: Vec<f64> = runs.iter().map(|r| r.seconds).collect();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    };
    let mut c = c_runs[0].clone();
    c.seconds = median(&c_runs);
    let mut s = s_runs[0].clone();
    s.seconds = median(&s_runs);
    Ok((c, s))
}

/// The step of a compiled program with the declared signature `(points: [n][d]f32) (centres: [k][d]f32) -> [k][d]f32`
/// (`futhark c`, or the GPU shim's executable: both read Futhark's binary format on stdin with `-b`), as [`converge`]
/// takes it: one process per iteration, the points written each time. This is the CPU path's whole measure
/// (`--backend c`): the wall clock is the processes'.
pub fn step_of_exe<'a>(exe: &'a Path, points: &'a Array) -> impl FnMut(&Array) -> Result<Array, String> + 'a {
    move |centres: &Array| {
        let stdin = write_binary(&[points, centres]);
        let out = crate::futhark_exec::output_with_timeout_bytes(std::process::Command::new(exe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin))?;
        if !out.status.success() {
            return Err(format!("{} failed:\n{}", exe.display(), String::from_utf8_lossy(&out.stderr)));
        }
        read_one(&out.stdout)
    }
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
    for (j, row) in out.iter_mut().enumerate() {
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
    }
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
    for j in 0..k {
        for t in 0..d {
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
                out[j * d + t] = sum / count as f64;
            }
        }
    }
    Ok((assignments, out))
}

#[cfg(test)]
mod tests;
