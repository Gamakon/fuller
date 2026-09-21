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
pub enum Combine {
    /// `sum(genes) / n` — geppy `avgval`.
    Avg,
    /// `sum(genes)`.
    Add,
    /// `product(genes)` — `mulval`.
    Mul,
}

/// A linker: a combination, optionally ROUNDED to the nearest integer — the
/// engine's `round_avg` / `round_mul` / `round_add`, used for ordinal targets.
/// Rounding is numpy's: half to EVEN, not half away from zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Linker {
    pub combine: Combine,
    pub round: bool,
}

impl Linker {
    pub const AVG: Linker = Linker { combine: Combine::Avg, round: false };
    pub const ADD: Linker = Linker { combine: Combine::Add, round: false };
    pub const MUL: Linker = Linker { combine: Combine::Mul, round: false };

    pub fn parse(name: &str) -> Result<Self, String> {
        let (combine, round) = match name {
            "avgval" => (Combine::Avg, false),
            "addval" => (Combine::Add, false),
            "mulval" => (Combine::Mul, false),
            "round_avg" | "_round_avg_eng" => (Combine::Avg, true),
            "round_add" | "_round_add_eng" => (Combine::Add, true),
            "round_mul" | "_round_mul_eng" => (Combine::Mul, true),
            other => return Err(format!("unknown linker {other:?}")),
        };
        Ok(Self { combine, round })
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
    /// Every linker to try; the engine searches linker x wrapper per
    /// individual, and all of them share the ONE set of gene predictions.
    pub linkers: Vec<Linker>,
    pub wrappers: Vec<Wrapper>,
    pub splits: Splits,
    /// Fit `a*f(x)+b` on the train rows; otherwise a=1, b=0.
    pub linear_scaling: bool,
}

/// Relative spread below which a train vector is constant. Mirrored by the
/// engine's `apply_linear_scaling`; change both or neither.
pub const CONSTANT_REL_TOL: f64 = 2e-6;

/// Rows of the behavioural signature (see [`SCORE_WIDTH`]).
pub const SIGNATURE_ROWS: usize = 16;

/// Metrics per candidate, before the signature.
pub const METRIC_WIDTH: usize = 9;

/// Values per candidate: `[a, b, mse_train, mse_val, max_err_val, mse_extrap,
/// mae_train, mae_val, mae_extrap]` followed by a SIGNATURE of
/// [`SIGNATURE_ROWS`] values — the model's scaled prediction `a*f(x)+b` at
/// fixed, evenly spaced TRAIN rows, in target units.
///
/// The signature is what makes "the same model" decidable. Comparing
/// chromosomes as text calls three orderings of four genes under a
/// commutative linker three individuals (a 30-entry hall of fame on the UCI
/// power plant data was ONE function), and comparing genomes calls two
/// identical expressed trees different when only their dormant regions
/// differ. Two candidates whose signatures agree to tolerance compute the same
/// function on this data, however either is written; two that fit equally
/// well but are different functions do not agree. A hash was not used: f32
/// noise straddling a rounding boundary would split identical models, where a
/// tolerance comparison cannot.
pub const SCORE_WIDTH: usize = METRIC_WIDTH + SIGNATURE_ROWS;

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
    check_input(preds, gene_ok, chromosomes, y, spec)?;
    let ScoreSpec {
        linkers,
        wrappers,
        splits,
        linear_scaling,
    } = spec;
    let (splits, linear_scaling) = (*splits, *linear_scaling);
    let n_rows = splits.total();

    let per = linkers.len() * wrappers.len() * SCORE_WIDTH;
    let mut out = vec![f64::NAN; chromosomes.len() * per];
    out.par_chunks_mut(per.max(1))
        .zip(chromosomes.par_iter())
        .for_each(|(slot, genes)| {
            if genes.iter().any(|&g| !gene_ok[g]) {
                return;
            }
            for (l, linker) in linkers.iter().enumerate() {
                let Some(linked) = link(preds, genes, *linker, n_rows) else {
                    continue;
                };
                for (w, wrapper) in wrappers.iter().enumerate() {
                    if let Some(s) = score_one(&linked, *wrapper, splits, y, linear_scaling) {
                        let at = (l * wrappers.len() + w) * SCORE_WIDTH;
                        slot[at..at + SCORE_WIDTH].copy_from_slice(&s);
                    }
                }
            }
        });
    Ok(out)
}

/// What every entry point asks of its input.
fn check_input(
    preds: &[f32],
    gene_ok: &[bool],
    chromosomes: &[Vec<usize>],
    y: &[f64],
    spec: &ScoreSpec,
) -> Result<(), String> {
    let splits = spec.splits;
    let n_rows = splits.total();
    // n_extrap may be 0: wild data has no truth-driven out-of-domain slice.
    if splits.n_train == 0 || splits.n_val == 0 {
        return Err(format!("train and validation need rows, got {splits:?}"));
    }
    if spec.linkers.is_empty() || spec.wrappers.is_empty() {
        return Err("need at least one linker and one wrapper".to_string());
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
    Ok(())
}

/// A GENE-LINKER COMBINATION: which of a chromosome's genes are USED — bit `g`
/// of `genes` is the chromosome's g-th gene — and the linker over them (an index
/// into `ScoreSpec::linkers`). The unused genes are not part of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneLinker {
    pub genes: u32,
    pub linker: usize,
}

impl GeneLinker {
    /// The used genes' positions in the chromosome, ascending.
    pub fn positions(self) -> impl Iterator<Item = usize> {
        (0..u32::BITS as usize).filter(move |g| self.genes >> g & 1 == 1)
    }
}

/// The most genes a chromosome may have when every subset is scored: 3 genes are
/// 15 combinations, 4 would be 37 and 5, 83 — a cost nobody has measured.
pub const MAX_SUBSET_GENES: usize = 3;

/// THE DYNAMIC GENE-SUBSET CHOICE's candidates. With `gene_subsets` off: all the
/// genes under each linker, in linker order — the engine as it always was. On:
/// every NON-EMPTY SUBSET of the genes, so a chromosome decides for itself
/// whether it is a 1-, 2- or 3-gene model — fewest genes first, then by the
/// subset's bits ({0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}), a subset of two
/// or more under each linker. A SINGLE gene has no linker: it is the gene itself
/// (every combine of one value is that value), scored once and recorded under
/// linker 0. 1 gene -> 1 combination, 2 -> 5, 3 -> 15; more than
/// [`MAX_SUBSET_GENES`] is an error. Choosing the first of equally good
/// candidates then prefers the model with fewer genes.
pub fn gene_linker_combinations(n_genes: usize, n_linkers: usize, gene_subsets: bool) -> Result<Vec<GeneLinker>, String> {
    if n_genes == 0 || n_genes > u32::BITS as usize - 1 || n_linkers == 0 {
        return Err(format!("combinations of {n_genes} genes under {n_linkers} linkers"));
    }
    let all = (1u32 << n_genes) - 1;
    if !gene_subsets {
        return Ok((0..n_linkers).map(|linker| GeneLinker { genes: all, linker }).collect());
    }
    if n_genes > MAX_SUBSET_GENES {
        return Err(format!("gene subsets are scored for at most {MAX_SUBSET_GENES} genes, this chromosome has {n_genes}"));
    }
    let mut subsets: Vec<u32> = (1..=all).collect();
    subsets.sort_by_key(|m| (m.count_ones(), *m));
    Ok(subsets
        .into_iter()
        .flat_map(|genes| (0..if genes.count_ones() == 1 { 1 } else { n_linkers }).map(move |linker| GeneLinker { genes, linker }))
        .collect())
}

/// [`score_chromosomes`] over GENE-LINKER COMBINATIONS: every chromosome is scored
/// under each of `combinations` (see [`gene_linker_combinations`]) and each of
/// `spec.wrappers`. A combination's model is `score_chromosomes`' model of the
/// chromosome made of its USED genes alone — so the avg linker divides by the
/// number of used genes, and a gene that failed to evaluate rejects only the
/// combinations that use it.
///
/// Returns `chromosomes.len() * combinations.len() * spec.wrappers.len() *
/// SCORE_WIDTH` values: chromosome-major, then combination, then wrapper.
pub fn score_gene_subsets(
    preds: &[f32],
    gene_ok: &[bool],
    chromosomes: &[Vec<usize>],
    y: &[f64],
    spec: &ScoreSpec,
    combinations: &[GeneLinker],
) -> Result<Vec<f64>, String> {
    check_input(preds, gene_ok, chromosomes, y, spec)?;
    let fewest = chromosomes.iter().map(Vec::len).min().unwrap_or(0);
    for c in combinations {
        if c.genes == 0 || c.linker >= spec.linkers.len() || c.positions().any(|g| g >= fewest) {
            return Err(format!("{c:?} does not fit chromosomes of {fewest} genes and {} linkers", spec.linkers.len()));
        }
    }
    let n_rows = spec.splits.total();
    let n_w = spec.wrappers.len();
    let per = combinations.len() * n_w * SCORE_WIDTH;
    let mut out = vec![f64::NAN; chromosomes.len() * per];
    out.par_chunks_mut(per.max(1))
        .zip(chromosomes.par_iter())
        .for_each(|(slot, genes)| {
            for (k, combination) in combinations.iter().enumerate() {
                let used: Vec<usize> = combination.positions().map(|g| genes[g]).collect();
                if used.iter().any(|&g| !gene_ok[g]) {
                    continue;
                }
                let Some(linked) = link(preds, &used, spec.linkers[combination.linker], n_rows) else {
                    continue;
                };
                for (w, wrapper) in spec.wrappers.iter().enumerate() {
                    if let Some(s) = score_one(&linked, *wrapper, spec.splits, y, spec.linear_scaling) {
                        let at = (k * n_w + w) * SCORE_WIDTH;
                        slot[at..at + SCORE_WIDTH].copy_from_slice(&s);
                    }
                }
            }
        });
    Ok(out)
}

/// Linked output per row, or None if any row is non-finite.
fn link(preds: &[f32], genes: &[usize], linker: Linker, n_rows: usize) -> Option<Vec<f64>> {
    let mut acc: Vec<f64> = match linker.combine {
        Combine::Mul => vec![1.0; n_rows],
        Combine::Avg | Combine::Add => vec![0.0; n_rows],
    };
    for &g in genes {
        let p = &preds[g * n_rows..(g + 1) * n_rows];
        match linker.combine {
            Combine::Mul => acc.iter_mut().zip(p).for_each(|(a, v)| *a *= f64::from(*v)),
            Combine::Avg | Combine::Add => {
                acc.iter_mut().zip(p).for_each(|(a, v)| *a += f64::from(*v));
            }
        }
    }
    if linker.combine == Combine::Avg {
        let n = genes.len() as f64;
        acc.iter_mut().for_each(|a| *a /= n);
    }
    if linker.round {
        acc.iter_mut().for_each(|a| *a = a.round_ties_even());
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

    // (mean squared, mean absolute) residual over a row range; (0, 0) for an
    // empty range — the absent extrapolation split of wild data.
    let errs = |lo: usize, hi: usize| -> (f64, f64) {
        if hi == lo {
            return (0.0, 0.0);
        }
        let n = (hi - lo) as f64;
        let (mut sq, mut ab) = (0.0_f64, 0.0_f64);
        for (x, t) in wrapped[lo..hi].iter().zip(&y[lo..hi]) {
            let d = t - (a * x + b);
            sq += d * d;
            ab += d.abs();
        }
        (sq / n, ab / n)
    };
    let max_err = wrapped[v0..v1]
        .iter()
        .zip(&y[v0..v1])
        .map(|(x, t)| (t - (a * x + b)).abs())
        .fold(0.0_f64, f64::max);

    let ((mse_t, mae_t), (mse_v, mae_v), (mse_e, mae_e)) =
        (errs(t0, t1), errs(v0, v1), errs(e0, e1));
    let mut s = [0.0_f64; SCORE_WIDTH];
    s[..METRIC_WIDTH].copy_from_slice(&[a, b, mse_t, mse_v, max_err, mse_e, mae_t, mae_v, mae_e]);
    // Evenly spaced train rows, first to last; with fewer rows than slots the
    // last row repeats, which is harmless for a comparison.
    let last = splits.n_train - 1;
    for (k, slot) in s[METRIC_WIDTH..].iter_mut().enumerate() {
        let row = if SIGNATURE_ROWS > 1 { k * last / (SIGNATURE_ROWS - 1) } else { 0 };
        *slot = a * wrapped[row] + b;
    }
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
            linkers: vec![linker],
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
            Linker::AVG,
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
        assert!(s[2..METRIC_WIDTH].iter().all(|v| v.abs() < 1e-20), "{s:?}");
        // and the signature is the model itself, 3x + 2, at the probe rows
        assert_eq!(&s[METRIC_WIDTH..METRIC_WIDTH + 2], &[5.0, 5.0]);
        assert_eq!(s[SCORE_WIDTH - 1], 14.0);
    }

    /// The wrapper dimension: y = sqrt(x) is exact under SqrtAbs and not
    /// under Identity, from the SAME gene predictions.
    #[test]
    fn wrappers_share_one_linked_vector() {
        let x = xs();
        let y: Vec<f64> = x.iter().map(|v| f64::from(*v).sqrt()).collect();
        let ws = [Wrapper::Identity, Wrapper::LogAbs, Wrapper::SqrtAbs];
        let out = run(&x, &[true], &[vec![0]], Linker::AVG, &ws, &y, true).unwrap();
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
            (Linker::AVG, (|v: f64| 1.5 * v) as fn(f64) -> f64),
            (Linker::ADD, |v| 3.0 * v),
            (Linker::MUL, |v| 2.0 * v * v),
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
            Linker::ADD,
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

        let out = run(&x, &[false], &[vec![0]], Linker::AVG, &ws, &y, true).unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "undecoded gene");

        let mut bad = x.clone();
        bad[8] = f32::INFINITY; // an extrapolation row
        let out = run(&bad, &[true], &[vec![0]], Linker::AVG, &ws, &y, true).unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "non-finite row");

        let flat = vec![5.0_f32; 9];
        let out = run(&flat, &[true], &[vec![0]], Linker::AVG, &ws, &y, true).unwrap();
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
        let out = run(&x, &[true], &[vec![0]], Linker::AVG, &ws, &y, false).unwrap();
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
        let out = run(&noisy, &[true], &[vec![0]], Linker::AVG, &[Wrapper::Identity], &y, true)
            .unwrap();
        assert!(out.iter().all(|v| v.is_nan()), "rounding noise was fitted: {out:?}");
    }

    /// The engine searches linker x wrapper per individual. All linkers come
    /// from ONE set of gene predictions, laid out chromosome, linker, wrapper.
    #[test]
    fn several_linkers_share_one_set_of_predictions() {
        let x = xs();
        let mut preds = x.clone();
        preds.extend(x.iter().map(|v| 2.0 * v)); // genes: x, 2x
        let y: Vec<f64> = x.iter().map(|v| 2.0 * f64::from(*v) * f64::from(*v)).collect();
        let spec = ScoreSpec {
            linkers: vec![Linker::ADD, Linker::MUL],
            wrappers: vec![Wrapper::Identity, Wrapper::SqrtAbs],
            splits: SPLITS,
            linear_scaling: false,
        };
        let out = score_chromosomes(&preds, &[true, true], &[vec![0, 1]], &y, &spec).unwrap();
        assert_eq!(out.len(), 2 * 2 * SCORE_WIDTH);
        let at = |l: usize, w: usize| &out[(l * 2 + w) * SCORE_WIDTH..][..SCORE_WIDTH];
        assert_eq!(at(1, 0)[2], 0.0, "mul x identity is 2x^2 exactly");
        assert!(at(0, 0)[2] > 1.0, "add x identity is 3x, not 2x^2");
        assert_eq!(at(1, 0)[6], 0.0, "and its MAE is 0 too");
    }

    /// Wild data has no extrapolation split; its metrics are 0.0, not NaN,
    /// and the candidate is still scored.
    #[test]
    fn extrapolation_split_is_optional() {
        let x: Vec<f32> = xs()[..7].to_vec();
        let y: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
        let spec = ScoreSpec {
            linkers: vec![Linker::AVG],
            wrappers: vec![Wrapper::Identity],
            splits: Splits { n_train: 4, n_val: 3, n_extrap: 0 },
            linear_scaling: true,
        };
        let out = score_chromosomes(&x, &[true], &[vec![0]], &y, &spec).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        assert_eq!((out[5], out[8]), (0.0, 0.0));
        assert_eq!(out.len(), SCORE_WIDTH);
    }

    /// Rounded linkers use numpy's rule: half to EVEN. 0.5 -> 0, 1.5 -> 2,
    /// 2.5 -> 2. Half-away-from-zero would give 1, 2, 3.
    #[test]
    fn rounded_linker_rounds_half_to_even() {
        let preds: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5, 0.5];
        let y: Vec<f64> = vec![0.0, 2.0, 2.0, 4.0, 0.0, 2.0, 2.0, 4.0, 0.0];
        let lk = Linker::parse("round_avg").unwrap();
        let out = run(&preds, &[true], &[vec![0]], lk, &[Wrapper::Identity], &y, false).unwrap();
        assert_eq!(cand(&out, 0, 0, 1)[2], 0.0, "{out:?}");
    }

    /// TRAIN creates the regression; VALIDATION is only scored with it. The
    /// rows share one dispatch because evaluating f(x) is independent per row
    /// — but (a, b) must come from the train slice alone. Proof: corrupt every
    /// validation (and extrapolation) prediction AND target, and (a, b) and
    /// the train error must not move by a single bit, while the validation
    /// error must.
    #[test]
    fn validation_rows_never_influence_the_regression() {
        let x = xs(); // 4 train, 3 val, 2 extrap
        let y: Vec<f64> = x.iter().map(|v| 3.0 * f64::from(*v) + 2.0).collect();
        let ws = [Wrapper::Identity];
        let clean = run(&x, &[true], &[vec![0]], Linker::AVG, &ws, &y, true).unwrap();

        let mut x_bad = x.clone();
        let mut y_bad = y.clone();
        for i in 4..9 {
            x_bad[i] = x[i] * -7.5 + 100.0;
            y_bad[i] = y[i] * 11.0 - 40.0;
        }
        let dirty = run(&x_bad, &[true], &[vec![0]], Linker::AVG, &ws, &y_bad, true).unwrap();

        let (c, d) = (cand(&clean, 0, 0, 1), cand(&dirty, 0, 0, 1));
        assert_eq!(c[0].to_bits(), d[0].to_bits(), "a moved: val leaked into the fit");
        assert_eq!(c[1].to_bits(), d[1].to_bits(), "b moved: val leaked into the fit");
        assert_eq!(c[2].to_bits(), d[2].to_bits(), "train MSE moved");
        assert_eq!(c[6].to_bits(), d[6].to_bits(), "train MAE moved");
        assert!(d[3] > 1.0 && c[3] < 1e-20, "val MSE must reflect the val rows: {c:?} {d:?}");
    }

    /// "The same model" is decided by behaviour, not by how it is written.
    /// Genes [x, 2x] under ADD, the same genes in the OTHER ORDER, and a
    /// different decomposition [1.5x, 1.5x] are all the function 3x and must
    /// carry the same signature; [x, x] (2x) fits a y = 3x target exactly as
    /// well after linear scaling — identical MSE — and the signature must STILL
    /// agree, because a*f+b is the same function; while a genuinely different
    /// function (x^2) must not.
    #[test]
    fn signature_identifies_the_function_not_the_writing() {
        let x: Vec<f32> = (1..=24).map(|v| v as f32 * 0.5).collect();
        let sp = Splits { n_train: 16, n_val: 5, n_extrap: 3 };
        let y: Vec<f64> = x.iter().map(|v| 3.0 * f64::from(*v) + 1.0).collect();
        let mut preds = x.clone(); // 0: x
        preds.extend(x.iter().map(|v| 2.0 * v)); // 1: 2x
        preds.extend(x.iter().map(|v| 1.5 * v)); // 2: 1.5x
        preds.extend(x.iter().map(|v| v * v)); // 3: x^2
        let spec = ScoreSpec {
            linkers: vec![Linker::ADD],
            wrappers: vec![Wrapper::Identity],
            splits: sp,
            linear_scaling: true,
        };
        let chroms = vec![vec![0, 1], vec![1, 0], vec![2, 2], vec![0, 0], vec![3, 0]];
        let out = score_chromosomes(&preds, &[true; 4], &chroms, &y, &spec).unwrap();
        let sig = |c: usize| &out[c * SCORE_WIDTH + METRIC_WIDTH..(c + 1) * SCORE_WIDTH];
        let same = |a: usize, b: usize| {
            sig(a).iter().zip(sig(b)).all(|(p, q)| (p - q).abs() <= 1e-9 * p.abs().max(1.0))
        };
        assert!(same(0, 1), "gene order changed the signature");
        assert!(same(0, 2), "a different decomposition of 3x changed the signature");
        assert!(same(0, 3), "2x rescaled by LSM is the same model as 3x");
        assert!(!same(0, 4), "x^2 + x is a different function and must differ");
        assert_eq!(sig(0).len(), SIGNATURE_ROWS);
    }

    /// 1 gene -> 1 combination, 2 -> 5, 3 -> 15, fewest genes first; a single gene
    /// once, not once per linker; 4 genes is an error, not a fallback. Off: all
    /// the genes under each linker, whatever their number.
    #[test]
    fn the_gene_linker_combinations_are_every_non_empty_subset() {
        let on = |n: usize| gene_linker_combinations(n, 3, true).unwrap();
        let pairs = |v: &[GeneLinker]| v.iter().map(|c| (c.genes, c.linker)).collect::<Vec<_>>();
        assert_eq!(pairs(&on(1)), vec![(0b1, 0)]);
        assert_eq!(pairs(&on(2)), vec![(0b01, 0), (0b10, 0), (0b11, 0), (0b11, 1), (0b11, 2)]);
        assert_eq!(
            pairs(&on(3)),
            vec![
                (0b001, 0), (0b010, 0), (0b100, 0),
                (0b011, 0), (0b011, 1), (0b011, 2),
                (0b101, 0), (0b101, 1), (0b101, 2),
                (0b110, 0), (0b110, 1), (0b110, 2),
                (0b111, 0), (0b111, 1), (0b111, 2),
            ]
        );
        assert!(gene_linker_combinations(4, 3, true).is_err());
        assert!(gene_linker_combinations(0, 3, true).is_err());
        for n in 1..=5 {
            let off = gene_linker_combinations(n, 3, false).unwrap();
            assert_eq!(pairs(&off), (0..3).map(|l| ((1u32 << n) - 1, l)).collect::<Vec<_>>());
        }
        assert_eq!(on(3)[4].positions().collect::<Vec<_>>(), vec![0, 1]);
    }

    /// A combination is `score_chromosomes` of its used genes alone, bit for bit:
    /// the avg linker divides by the number of USED genes, a single gene is the
    /// gene itself, and a gene that failed poisons only the subsets holding it.
    /// With the switch off the two entry points return the same values.
    #[test]
    fn a_gene_subset_scores_as_the_chromosome_of_its_used_genes() {
        let x: Vec<f32> = (1..=24).map(|v| v as f32 * 0.5).collect();
        let splits = Splits { n_train: 16, n_val: 5, n_extrap: 3 };
        let y: Vec<f64> = x.iter().map(|v| 2.0 * (f64::from(*v) * f64::from(*v) + f64::from(*v)) + 1.0).collect();
        let mut preds = x.clone(); // 0: x
        preds.extend(x.iter().map(|v| v * v)); // 1: x^2
        preds.extend(x.iter().map(|v| (v * 3.0).sin())); // 2: junk
        preds.extend(x.iter().map(|v| 1.0 / v)); // 3: not decoded
        let ok = [true, true, true, false];
        let spec = ScoreSpec {
            linkers: vec![Linker::AVG, Linker::MUL, Linker::ADD],
            wrappers: vec![Wrapper::Identity, Wrapper::SqrtAbs],
            splits,
            linear_scaling: true,
        };
        let same = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());
        let chromosomes = vec![vec![0, 1, 2], vec![3, 1, 0]];
        let combinations = gene_linker_combinations(3, 3, true).unwrap();
        let out = score_gene_subsets(&preds, &ok, &chromosomes, &y, &spec, &combinations).unwrap();
        let per = 2 * SCORE_WIDTH;
        assert_eq!(out.len(), 2 * 15 * per);
        for (c, genes) in chromosomes.iter().enumerate() {
            for (k, combination) in combinations.iter().enumerate() {
                let used: Vec<usize> = combination.positions().map(|g| genes[g]).collect();
                let alone = score_chromosomes(&preds, &ok, &[used], &y, &spec).unwrap();
                let want = &alone[combination.linker * per..(combination.linker + 1) * per];
                assert!(same(&out[(c * 15 + k) * per..(c * 15 + k + 1) * per], want), "chromosome {c}, {combination:?}");
            }
        }
        // 2 (x^2 + x) + 1 is genes {0, 1} under add (and under avg, which the scale makes
        // the same model): exact there, and nowhere it must use the junk gene alone.
        let mse = |c: usize, k: usize| out[(c * 15 + k) * per + 2];
        assert!(mse(0, 5) < 1e-20 && mse(0, 3) < 1e-20, "{} {}", mse(0, 5), mse(0, 3));
        assert!(mse(0, 2) > 1.0);
        // Chromosome 1's gene 0 failed: every subset holding it is rejected, the
        // others are scored.
        for (k, combination) in combinations.iter().enumerate() {
            assert_eq!(mse(1, k).is_nan(), combination.genes & 1 == 1, "{combination:?}");
        }
        let off = gene_linker_combinations(3, 3, false).unwrap();
        let whole = score_gene_subsets(&preds, &ok, &chromosomes, &y, &spec, &off).unwrap();
        let reference = score_chromosomes(&preds, &ok, &chromosomes, &y, &spec).unwrap();
        assert_eq!(whole.len(), reference.len());
        assert!(same(&whole, &reference));
        // A combination that names a gene the chromosome does not have is an error.
        assert!(score_gene_subsets(&preds, &ok, &[vec![0, 1]], &y, &spec, &combinations).is_err());
    }

    #[test]
    fn malformed_input_is_an_error_not_a_guess() {
        let x = xs();
        let y = vec![0.0; 9];
        let ws = [Wrapper::Identity];
        let bad_idx = run(&x, &[true], &[vec![1]], Linker::AVG, &ws, &y, true);
        assert!(bad_idx.is_err());
        let bad_y = run(&x, &[true], &[vec![0]], Linker::AVG, &ws, &y[..8], true);
        assert!(bad_y.is_err());
        assert!(Linker::parse("nope").is_err());
        assert!(Wrapper::parse("nope").is_err());
    }
}
