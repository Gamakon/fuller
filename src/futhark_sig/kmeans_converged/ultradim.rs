//! ULTRADIM'S OWN MEASURES, VENDORED FOR THE TEST that the reimplementation in `kmeans_converged.rs` computes their
//! numbers (the advisor's review of 2026-11-03, F11). Copied VERBATIM, comments included, from
//! `/Users/andrewmorgan/dev/minkymorgan/qdrant/lib/projection-voting-kmeans` at commit 4295c335 (2026-10-20):
//! `src/quality.rs` lines 50-197 (`ln_betainc`, `ln1p_neg_exp`, `betacf`, `log10_angular_percentile`,
//! `exact_mean_log10_p`) and `src/lib.rs` lines 5705-5741 (`cosine_inertia`). Only the `use` lines are this file's.
//! Never used by the fitness: the fitness's functions are the ones in the parent module, and
//! `tests::the_measures_equal_ultradims_vendored_functions` requires them to agree with these to the bit.
//!
//! Linking the crate itself as a dev-dependency was not done: `projection-voting-kmeans` pulls `udv23-umap`,
//! `ultradim-sparse` and wgpu 27 (fuller's GPU feature is on wgpu 30), its manifest inherits lints and versions from
//! the qdrant workspace, and fuller's `cargo test` would then build another repository's GPU engine and stop
//! wherever that checkout is not present or not building. The two functions and their helpers are 150 lines.

use statrs::function::beta::ln_beta;

/// Natural-log of the regularised incomplete beta `I_x(a, b)`.
///
/// Computed as `ln I_x = a·ln x + b·ln(1-x) - ln(a) - lnB(a,b) + ln(cf)`, where
/// the underflow-prone prefactor `x^a (1-x)^b` is kept in log space and `cf` is
/// the Lentz continued fraction (Numerical Recipes `betacf`). Returns
/// `f64::NEG_INFINITY` for `x == 0` (i.e. `I_0(a,b) = 0`) and `0.0` for
/// `x == 1` (`I_1(a,b) = 1`). The symmetry `I_x(a,b) = 1 - I_{1-x}(b,a)` is used
/// for `x > (a+1)/(a+b+2)` so the continued fraction always converges fast.
pub fn ln_betainc(a: f64, b: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && b > 0.0, "ln_betainc: a,b must be > 0");
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0 {
        return f64::NEG_INFINITY; // I_0(a,b) = 0
    }
    if x >= 1.0 {
        return 0.0; // I_1(a,b) = 1
    }
    // log of the front factor x^a (1-x)^b / (a · B(a,b)).
    let ln_front = a * x.ln() + b * (1.0 - x).ln() - a.ln() - ln_beta(a, b);
    if x < (a + 1.0) / (a + b + 2.0) {
        // Direct: I_x = front · cf(a,b,x).
        ln_front + betacf(a, b, x).ln()
    } else {
        // Reflected: I_x(a,b) = 1 - I_{1-x}(b,a). Compute the reflected tail in
        // log space, then 1 - tail via ln(1 - e^{ln_tail}) = ln1p(-e^{ln_tail}).
        let ln_front_r = b * (1.0 - x).ln() + a * x.ln() - b.ln() - ln_beta(b, a);
        let ln_tail = ln_front_r + betacf(b, a, 1.0 - x).ln();
        ln1p_neg_exp(ln_tail)
    }
}

/// `ln(1 - exp(v))` for `v <= 0`, numerically stable across the whole range
/// (the standard two-branch `log1p`/`expm1` split at `ln 2`).
fn ln1p_neg_exp(v: f64) -> f64 {
    debug_assert!(v <= 1e-12, "ln1p_neg_exp expects v <= 0, got {v}");
    if v > -std::f64::consts::LN_2 {
        (-(v.exp_m1())).ln() // -expm1(v) = 1 - e^v, accurate for v near 0
    } else {
        (-v.exp()).ln_1p() // ln1p(-e^v), accurate for very negative v
    }
}

/// Lentz continued fraction for the incomplete beta (Numerical Recipes
/// `betacf`). Converges in a handful of iterations for the reflected argument
/// range this module always uses.
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
        // even step
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
        // odd step
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

/// `log10` of the fixed-direction angular percentile `F(θ)` for a point whose
/// cosine to its (unit) centroid is `cos_t`, in dimension `d`. NO folding; the
/// sign of `cos_t` selects the branch. Survives underflow at large `d` / tight
/// points (the whole point of working in log space).
///
/// * `cos_t >= 0` (θ ≤ π/2): `p = 0.5 · I_{sin²θ}(α, ½)`.
/// * `cos_t < 0`  (θ > π/2): `p = 1 - 0.5 · I_{sin²θ}(α, ½)`.
pub fn log10_angular_percentile(cos_t: f64, d: usize) -> f64 {
    debug_assert!(d >= 2, "spherical CDF needs d >= 2");
    let cos_t = cos_t.clamp(-1.0, 1.0);
    let alpha = (d as f64 - 1.0) / 2.0;
    // sin²θ. A point EXACTLY at its centroid (cos == 1, e.g. a singleton
    // cluster) gives x = 0, whose true percentile is 0 (log10 = -inf), which
    // would poison a mean over points. Clamp x to the smallest positive normal
    // f64 instead of flooring the OUTPUT: ln_betainc(α, ½, MIN_POSITIVE) is
    // finite and scales with d (≈ -α·708 nats), so a coincident point still
    // scores strictly tighter than every non-coincident point at the same d.
    // The previous fix floored the output at -323.3 (log10 of the smallest
    // subnormal), on the false premise that values below it are
    // unrepresentable — only the LINEAR percentile underflows there; the log10
    // value itself is a perfectly representable f64. At d ~ 1.8M EVERY point's
    // log10 p sits far below -323.3, so the floor clamped real and null alike
    // and every lift collapsed to exactly 0. No output floor: the log-space
    // value is the result.
    let x = (1.0 - cos_t * cos_t).max(f64::MIN_POSITIVE);
    // log10(0.5 · I_x(α, ½)) = log10(0.5) + log10(I_x).
    // log10(0.5) = -log10(2); log10(I_x) = ln(I_x) · log10(e).
    const LOG10_HALF: f64 = -std::f64::consts::LOG10_2;
    let log10_half = LOG10_HALF + ln_betainc(alpha, 0.5, x) * std::f64::consts::LOG10_E;
    if cos_t >= 0.0 {
        log10_half // θ ≤ π/2
    } else {
        // θ > π/2: p = 1 - 0.5·I. Near π/2 (cos→0⁻) half→0.5 ⇒ log10(0.5);
        // near π (cos→-1) half→0 ⇒ log10(1)=0. No catastrophic region here.
        let half = 10f64.powf(log10_half);
        (1.0 - half).log10()
    }
}

/// REAL metric: mean over points of `log10 F(θ_i)`, where `cosines[i]` is the
/// (clamped) dot of point `i` with its assigned unit centroid. Final or sampled.
/// Smaller (more negative) = tighter than chance.
pub fn exact_mean_log10_p(cosines: &[f64], d: usize) -> f64 {
    if cosines.is_empty() {
        return 0.0;
    }
    let sum: f64 = cosines
        .iter()
        .map(|&c| log10_angular_percentile(c, d))
        .sum();
    sum / cosines.len() as f64
}

/// Spherical-cosine inertia `(1/n)Σ(1 − xᵢᵀμ_{aᵢ})` with unit centroids.
///
/// O(N·D_raw) per call; parallelised over points (deterministic chunked
/// reduction so the parity gates stay stable).
pub fn cosine_inertia(
    flat: &[f32],
    centroids: &[Vec<f32>],
    assignments: &[usize],
    d_raw: usize,
) -> f32 {
    use rayon::prelude::*;
    let n = flat.len() / d_raw;
    if n == 0 {
        return 0.0;
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(threads).max(1);
    let acc: f32 = (0..n)
        .step_by(chunk)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let end = (start + chunk).min(n);
            let mut local = 0.0f32;
            for i in start..end {
                let a = assignments[i];
                if a < centroids.len() {
                    let v = &flat[i * d_raw..(i + 1) * d_raw];
                    let dot: f32 = v.iter().zip(&centroids[a]).map(|(x, y)| x * y).sum();
                    local += 1.0 - dot;
                }
            }
            local
        })
        .sum();
    acc / n as f32
}
