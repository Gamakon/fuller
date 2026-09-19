//! Chromosome scoring over per-gene predictions — the host half of the
//! population x genes x e-class variants x wrappers join.
//!
//! The device evaluates every UNIQUE gene once over every row (one dispatch,
//! `gpu_eval`). A chromosome is a list of gene indices; a candidate is a
//! (chromosome, wrapper) pair. This module turns gene predictions into the
//! per-candidate numbers the engine's HFF objective vector is built from:
//!
//!   y_pred = a * WRAPPER( LINKER(genes) ) + b
//!
//! Sharing genes is what makes the cross-product cheap: substituting one
//! e-class variant into a 3-gene chromosome adds ONE gene to the dispatch, not
//! a whole tree, and the three wrappers re-use the same linked vector.
//!
//! The semantics mirror the engine's `compute_raw_metrics` exactly, because a
//! candidate scored here replaces one scored there:
//!   * a non-finite linked value on ANY row rejects the chromosome;
//!   * a non-finite wrapped value rejects that wrapper only;
//!   * linear scaling is the least-squares (a, b) on the TRAIN rows, rejected
//!     when the wrapped train vector is constant (numpy `allclose` to its
//!     mean, atol 1e-8) or the fit is non-finite;
//!   * any non-finite metric rejects the candidate.
//!
//! A rejected candidate is all-NaN. Nothing is substituted for it.

use rayon::prelude::*;

/// How a chromosome's genes combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linker {
    /// `sum(genes) / n` — geppy `avgval`.
    Avg,
    /// `sum(genes)`.
    Add,
    /// `product(genes)` — `mulval`.
    Mul,
}

impl Linker {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "avgval" | "_round_avg_eng" => Ok(Self::Avg),
            "addval" | "_round_add_eng" => Ok(Self::Add),
            "mulval" | "_round_mul_eng" => Ok(Self::Mul),
            other => Err(format!(
                "unknown linker {other:?} (want avgval/addval/mulval)"
            )),
        }
    }
}

/// Chromosome-level regression wrapper, applied to the linked output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrapper {
    Identity,
    /// `ln(|x| + 1e-12)` — the engine's `_w_log_abs`.
    LogAbs,
    /// `sqrt(|x|)` — `_w_sqrt_abs`.
    SqrtAbs,
    /// `exp(clamp(x, -50, 50))` — `_w_exp`.
    Exp,
    /// `x * x` — `_w_square`.
    Square,
}

impl Wrapper {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "identity" => Ok(Self::Identity),
            "log_abs" => Ok(Self::LogAbs),
            "sqrt_abs" => Ok(Self::SqrtAbs),
            "exp" => Ok(Self::Exp),
            "square" => Ok(Self::Square),
            other => Err(format!("unknown wrapper {other:?}")),
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::LogAbs => (x.abs() + 1e-12).ln(),
            Self::SqrtAbs => x.abs().sqrt(),
            Self::Exp => x.clamp(-50.0, 50.0).exp(),
            Self::Square => x * x,
        }
    }
}

/// Row layout of the resident dataset: train, then validation, then
/// extrapolation, concatenated so all three are answered by one dispatch.
#[derive(Debug, Clone, Copy)]
pub struct Splits {
    pub n_train: usize,
    pub n_val: usize,
    pub n_extrap: usize,
}

impl Splits {
    pub fn total(&self) -> usize {
        self.n_train + self.n_val + self.n_extrap
    }
}

/// How candidates are built from gene predictions.
#[derive(Debug, Clone)]
pub struct ScoreSpec {
    pub linker: Linker,
    pub wrappers: Vec<Wrapper>,
    pub splits: Splits,
    /// Fit `a*f(x)+b` on the train rows; otherwise a=1, b=0.
    pub linear_scaling: bool,
}

/// Relative spread below which a train vector is constant. Mirrored by the
/// engine's `apply_linear_scaling`; change both or neither.
pub const CONSTANT_REL_TOL: f64 = 2e-6;

/// Values per candidate: `[a, b, mse_train, mse_val, max_err_val, mse_extrap]`.
pub const SCORE_WIDTH: usize = 6;

/// Score every (chromosome, wrapper) candidate.
///
/// * `preds` — `n_genes * n_rows` device predictions, gene-major.
/// * `gene_ok` — false for a gene the device could not evaluate; any
///   chromosome using it is rejected (the caller sees NaN and must evaluate
///   that individual on its reference path).
/// * `chromosomes` — gene indices per chromosome.
/// * `y` — targets, `splits.total()` long, in the same row order as the data.
///
/// Returns `chromosomes.len() * spec.wrappers.len() * SCORE_WIDTH` values,
/// chromosome-major then wrapper.
pub fn score_chromosomes(
    preds: &[f32],
    gene_ok: &[bool],
    chromosomes: &[Vec<usize>],
    y: &[f64],
    spec: &ScoreSpec,
) -> Result<Vec<f64>, String> {
    let ScoreSpec {
        linker,
        wrappers,
        splits,
        linear_scaling,
    } = spec;
    let (linker, splits, linear_scaling) = (*linker, *splits, *linear_scaling);
    let n_rows = splits.total();
    if splits.n_train == 0 || splits.n_val == 0 || splits.n_extrap == 0 {
        return Err(format!("every split needs rows, got {splits:?}"));
    }
    if y.len() != n_rows {
        return Err(format!("y has {} values, splits total {n_rows}", y.len()));
    }
    let n_genes = gene_ok.len();
    if preds.len() != n_genes * n_rows {
        return Err(format!(
            "preds has {} values, want {n_genes} genes x {n_rows} rows",
            preds.len()
        ));
    }
    for (c, genes) in chromosomes.iter().enumerate() {
        if genes.is_empty() {
            return Err(format!("chromosome {c} has no genes"));
        }
        if let Some(&g) = genes.iter().find(|&&g| g >= n_genes) {
            return Err(format!(
                "chromosome {c} names gene {g}, only {n_genes} given"
            ));
        }
    }

    let per = wrappers.len() * SCORE_WIDTH;
    let mut out = vec![f64::NAN; chromosomes.len() * per];
    out.par_chunks_mut(per.max(1))
        .zip(chromosomes.par_iter())
        .for_each(|(slot, genes)| {
            if genes.iter().any(|&g| !gene_ok[g]) {
                return;
            }
            let Some(linked) = link(preds, genes, linker, n_rows) else {
                return;
            };
            for (w, wrapper) in wrappers.iter().enumerate() {
                if let Some(s) = score_one(&linked, *wrapper, splits, y, linear_scaling) {
                    slot[w * SCORE_WIDTH..(w + 1) * SCORE_WIDTH].copy_from_slice(&s);
                }
            }
        });
    Ok(out)
}

/// Linked output per row, or None if any row is non-finite.
fn link(preds: &[f32], genes: &[usize], linker: Linker, n_rows: usize) -> Option<Vec<f64>> {
    let mut acc: Vec<f64> = match linker {
        Linker::Mul => vec![1.0; n_rows],
        Linker::Avg | Linker::Add => vec![0.0; n_rows],
    };
    for &g in genes {
        let p = &preds[g * n_rows..(g + 1) * n_rows];
        match linker {
            Linker::Mul => acc.iter_mut().zip(p).for_each(|(a, v)| *a *= f64::from(*v)),
            Linker::Avg | Linker::Add => {
                acc.iter_mut().zip(p).for_each(|(a, v)| *a += f64::from(*v));
            }
        }
    }
    if linker == Linker::Avg {
        let n = genes.len() as f64;
        acc.iter_mut().for_each(|a| *a /= n);
    }
    acc.iter().all(|v| v.is_finite()).then_some(acc)
}

fn score_one(
    linked: &[f64],
    wrapper: Wrapper,
    splits: Splits,
    y: &[f64],
    linear_scaling: bool,
) -> Option<[f64; SCORE_WIDTH]> {
    let wrapped: Vec<f64> = linked.iter().map(|&x| wrapper.apply(x)).collect();
    if !wrapped.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (t0, t1) = (0, splits.n_train);
    let (v0, v1) = (t1, t1 + splits.n_val);
    let (e0, e1) = (v1, v1 + splits.n_extrap);

    let (a, b) = if linear_scaling {
        least_squares(&wrapped[t0..t1], &y[t0..t1])?
    } else {
        (1.0, 0.0)
    };

    let mse = |lo: usize, hi: usize| -> f64 {
        let n = (hi - lo) as f64;
        wrapped[lo..hi]
            .iter()
            .zip(&y[lo..hi])
            .map(|(x, t)| {
                let d = t - (a * x + b);
                d * d
            })
            .sum::<f64>()
            / n
    };
    let max_err = wrapped[v0..v1]
        .iter()
        .zip(&y[v0..v1])
        .map(|(x, t)| (t - (a * x + b)).abs())
        .fold(0.0_f64, f64::max);

    let s = [a, b, mse(t0, t1), mse(v0, v1), max_err, mse(e0, e1)];
    s.iter().all(|v| v.is_finite()).then_some(s)
}

/// Least-squares `(a, b)` for `a*x + b ~ y`. None when `x` is constant — the
/// engine's `np.allclose(x - mean, 0)` test, absolute tolerance 1e-8 — or the
/// fit is non-finite.
fn least_squares(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    // Constant, to the resolution the predictions HAVE. They come from an f32
    // device: x/x is exactly 1 in IEEE but Metal returns 1 +- a few ulp, so
    // an absolute 1e-8 test called protected_div(omega, omega) "not constant"
    // and the fit then amplified rounding noise into a score (the engine, in
    // f64, rejects it). 2e-6 relative is ~30 ulp of f32. It also rejects a
    // signal riding on an offset 1e6 times larger, which no f32 fit can
    // resolve; the engine's f64 path applies the same rule so both agree.
    let tol = 1e-8 + CONSTANT_REL_TOL * mx.abs();
    if x.iter().all(|v| (v - mx).abs() <= tol) {
        return None;
    }
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0_f64, 0.0_f64);
    for (xv, yv) in x.iter().zip(y) {
        sxy += (xv - mx) * (yv - my);
        sxx += (xv - mx) * (xv - mx);
    }
    let a = sxy / sxx;
    let b = my - a * mx;
    (a.is_finite() && b.is_finite()).then_some((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPLITS: Splits = Splits {
        n_train: 4,
        n_val: 3,
        n_extrap: 2,
    };

    fn xs() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    }

    fn run(
        preds: &[f32],
        ok: &[bool],
        chroms: &[Vec<usize>],
        linker: Linker,
        ws: &[Wrapper],
        y: &[f64],
        scale: bool,
    ) -> Result<Vec<f64>, String> {
        let spec = ScoreSpec {
            linker,
            wrappers: ws.to_vec(),
            splits: SPLITS,
            linear_scaling: scale,
        };
        score_chromosomes(preds, ok, chroms, y, &spec)
    }

    fn cand(out: &[f64], c: usize, w: usize, n_w: usize) -> &[f64] {
        let i = (c * n_w + w) * SCORE_WIDTH;
        &out[i..i + SCORE_WIDTH]
    }

    /// y = 3x + 2 exactly: identity wrapper with scaling must recover a=3,
    /// b=2 and zero error on every split.
    #[test]
    fn linear_scaling_recovers_exact_affine_target() {
        let x = xs();
        let y: Vec<f64> = x.iter().map(|v| 3.0 * f64::from(*v) + 2.0).collect();
        let out = run(
            &x,
            &[true],
            &[vec![0]],
            Linker::Avg,
            &[Wrapper::Identity],
            &y,
            true,
        )
        .unwrap();
        let s = cand(&out, 0, 0, 1);
        assert!(
            (s[0] - 3.0).abs() < 1e-12 && (s[1] - 2.0).abs() < 1e-12,
            "{s:?}"
        );
        assert!(s[2..].iter().all(|v| v.abs() < 1e-20), "{s:?}");
    }

    /// The wrapper dimension: y = sqrt(x) is exact under SqrtAbs and not
    /// under Identity, from the SAME gene predictions.
    #[test]
    fn wrappers_share_one_linked_vector() {
        let x = xs();
        let y: Vec<f64> = x.iter().map(|v| f64::from(*v).sqrt()).collect();
        let ws = [Wrapper::Identity, Wrapper::LogAbs, Wrapper::SqrtAbs];
        let out = run(&x, &[true], &[vec![0]], Linker::Avg, &ws, &y, true).unwrap();
        assert!(cand(&out, 0, 2, 3)[2] < 1e-20, "sqrt wrapper must be exact");
        assert!(cand(&out, 0, 0, 3)[2] > 1e-6, "identity must not be");
    }

    /// Linkers: genes [x, 2x] -> avg 1.5x, add 3x, mul 2x^2 (checked unscaled).
    #[test]
    fn linkers_match_their_definitions() {
        let x = xs();
        let mut preds = x.clone();
        preds.extend(x.iter().map(|v| 2.0 * v));
        for (linker, f) in [
            (Linker::Avg, (|v: f64| 1.5 * v) as fn(f64) -> f64),
            (Linker::Add, |v| 3.0 * v),
            (Linker::Mul, |v| 2.0 * v * v),
        ] {
            let y: Vec<f64> = x.iter().map(|v| f(f64::from(*v))).collect();
            let out = run(
                &preds,
                &[true, true],
                &[vec![0, 1]],
                linker,
                &[Wrapper::Identity],
                &y,
                false,
            )
            .unwrap();
            assert_eq!(cand(&out, 0, 0, 1)[2], 0.0, "{linker:?}");
        }
    }

    /// Variant substitution shares genes: chromosomes [0,1] and [2,1] differ
    /// in one gene only, and each is scored against its own link.
    #[test]
    fn chromosomes_index_shared_genes() {
        let x = xs();
        let mut preds = x.clone(); // gene 0 = x
        preds.extend(x.iter().map(|v| 2.0 * v)); // gene 1 = 2x
        preds.extend(x.iter().map(|v| 4.0 * v)); // gene 2 = 4x
        let y: Vec<f64> = x.iter().map(|v| 6.0 * f64::from(*v)).collect(); // 4x + 2x
        let out = run(
            &preds,
            &[true; 3],
            &[vec![0, 1], vec![2, 1]],
            Linker::Add,
            &[Wrapper::Identity],
            &y,
            false,
        )
        .unwrap();
        assert!(cand(&out, 0, 0, 1)[2] > 1.0);
        assert_eq!(cand(&out, 1, 0, 1)[2], 0.0);
    }

    /// Rejections are NaN, never a substituted number: an undecoded gene, a
    /// non-finite row, and a constant train vector under scaling.
    #[test]
    fn rejected_candidates_are_nan() {
        let x = xs();
        let y: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
        let ws = [Wrapper::Identity];

        let out = run(&x, &[false], &[vec![0]], Linker::Avg, &ws, &y, true).unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "undecoded gene");

        let mut bad = x.clone();
        bad[8] = f32::INFINITY; // an extrapolation row
        let out = run(&bad, &[true], &[vec![0]], Linker::Avg, &ws, &y, true).unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "non-finite row");

        let flat = vec![5.0_f32; 9];
        let out = run(&flat, &[true], &[vec![0]], Linker::Avg, &ws, &y, true).unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "constant train vector");
    }

    /// Every wrapper is TOTAL on finite f32 input (log is guarded, exp is
    /// clamped, and an f32 squared cannot overflow f64), so a bad wrapper is
    /// scored as bad rather than rejected. Pin that: the largest f32 under
    /// Square is finite and terrible, while Identity on the same gene is exact.
    #[test]
    fn wrappers_are_total_on_finite_input() {
        let mut x = xs();
        x[0] = 3.0e38;
        let y: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
        let ws = [Wrapper::Identity, Wrapper::Square];
        let out = run(&x, &[true], &[vec![0]], Linker::Avg, &ws, &y, false).unwrap();
        assert_eq!(cand(&out, 0, 0, 2)[2], 0.0, "identity is exact");
        assert!(
            cand(&out, 0, 1, 2)[2] > 1e100,
            "square is scored, and scored badly"
        );
    }

    /// x/x on the device is 1 +- a few f32 ulp, not exactly 1. That is a
    /// constant and must be rejected like one, not fitted.
    #[test]
    fn f32_rounding_noise_is_a_constant() {
        let noisy: Vec<f32> = (0..9).map(|i| 1.0 + (i % 3) as f32 * f32::EPSILON).collect();
        let y: Vec<f64> = xs().iter().map(|v| f64::from(*v)).collect();
        let out = run(&noisy, &[true], &[vec![0]], Linker::Avg, &[Wrapper::Identity], &y, true)
            .unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "rounding noise was fitted: {out:?}");
    }

    #[test]
    fn malformed_input_is_an_error_not_a_guess() {
        let x = xs();
        let y = vec![0.0; 9];
        let ws = [Wrapper::Identity];
        let bad_idx = run(&x, &[true], &[vec![1]], Linker::Avg, &ws, &y, true);
        assert!(bad_idx.is_err());
        let bad_y = run(&x, &[true], &[vec![0]], Linker::Avg, &ws, &y[..8], true);
        assert!(bad_y.is_err());
        assert!(Linker::parse("nope").is_err());
        assert!(Wrapper::parse("nope").is_err());
    }
}
