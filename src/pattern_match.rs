//! The reduction half of GPU-speed pattern matching: turn per-column value
//! vectors (a gene's subtree partials, and a library of pattern columns) into a
//! per-column-pair affine 1 - R², cheaply, from SUMS.
//!
//! Given a subtree's value vector `S` and a pattern's value vector `P` over the
//! same rows, "is S an affine image of P" (i.e. does `S ≈ a·P + b` for some
//! constants) is answered by the residual variance of the least-squares fit, and
//! that residual is a function of five per-column sums only:
//! `Σs, Σp, Σs², Σp², Σsp`. So the expensive part — the per-row values — is
//! produced once by [`crate::gpu_eval::GpuEvaluator::eval_with_partials`], reduced
//! to a handful of sums per column, and every (subtree × pattern) score is O(1)
//! arithmetic on those sums. No per-row host traffic per pair.
//!
//! This module is the HOST reduction (the correctness reference and the first
//! working version). A WGSL sum-reduction kernel can replace [`column_sums`]
//! later for the order-of-magnitude speedup; [`affine_residual`] stays the same
//! math on either path.

/// The five sums of one value column needed for an affine fit against a target,
/// plus the target's own sums, over the JOINTLY-FINITE rows. `n` is that finite
/// count. Rows where either side is non-finite are dropped (a NaN/inf subtree
/// value or target row carries no information and must not poison the sum).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColumnSums {
    pub n: f64,
    pub sx: f64,
    pub sy: f64,
    pub sxx: f64,
    pub syy: f64,
    pub sxy: f64,
}

impl ColumnSums {
    /// Accumulate the sums of `x` (a candidate column) against `y` (the target),
    /// over rows where BOTH are finite.
    pub fn of(x: &[f32], y: &[f64]) -> ColumnSums {
        debug_assert_eq!(x.len(), y.len(), "column and target must be the same rows");
        let mut s = ColumnSums::default();
        for (&xi, &yi) in x.iter().zip(y) {
            let xi = f64::from(xi);
            if !xi.is_finite() || !yi.is_finite() {
                continue;
            }
            s.n += 1.0;
            s.sx += xi;
            s.sy += yi;
            s.sxx += xi * xi;
            s.syy += yi * yi;
            s.sxy += xi * yi;
        }
        s
    }
}

/// The `1 - R²` of the least-squares affine fit `y ≈ a·x + b`, from the sums.
///
/// `None` when there are too few finite rows, or `x` is (near-)constant so no
/// slope is determined, or the target has no variance. A returned `0.0` means an
/// exact affine match — `x` IS the target up to scale and offset, which is the
/// "this subtree is the pattern" signal. Small values are near-matches.
///
/// This is scale- and offset-free by construction (the outer model wrap supplies
/// `a` and `b`), which is exactly why a pattern need carry no free constants.
pub fn affine_residual(s: ColumnSums, min_rows: f64) -> Option<f64> {
    if s.n < min_rows {
        return None;
    }
    let n = s.n;
    let var_x = s.sxx - s.sx * s.sx / n;
    let var_y = s.syy - s.sy * s.sy / n;
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    let cov = s.sxy - s.sx * s.sy / n;
    // R² of a simple linear regression is cov² / (var_x · var_y); 1 - R² is the
    // fraction of the target's variance the fit leaves unexplained. Clamped to
    // [0, 1] against f64 rounding at a near-perfect fit.
    let r2 = (cov * cov / (var_x * var_y)).clamp(0.0, 1.0);
    Some(1.0 - r2)
}

/// The absolute Pearson correlation |r| = sqrt(R²), from the sums — the same
/// evidence as [`affine_residual`] in the units the incomplete-beta p-value
/// consumes (`theta = acos(|r|)`). `None` under the same degenerate conditions.
pub fn abs_corr(s: ColumnSums, min_rows: f64) -> Option<f64> {
    affine_residual(s, min_rows).map(|one_minus_r2| (1.0 - one_minus_r2).sqrt())
}

/// Score every subtree column of ONE expression's partials against a target.
///
/// `partials` is the [`crate::gpu_eval::GpuEvaluator::eval_with_partials`] output;
/// `expr` is the expression's index in the batch; `n_nodes` is its node count
/// (only `k < n_nodes` are real subtrees — higher slots are 0 padding);
/// `max_nodes` is [`crate::gpu_eval::MAX_NODES`]; `n_rows` the row count.
///
/// Returns, per subtree node `k`, its affine `1 - R²` against the target (or
/// `None` for a degenerate column). The caller picks the smallest — the subtree
/// whose values are most nearly an affine image of the target.
pub fn score_subtrees(
    partials: &[f32],
    expr: usize,
    n_nodes: usize,
    max_nodes: usize,
    n_rows: usize,
    target: &[f64],
    min_rows: f64,
) -> Vec<Option<f64>> {
    (0..n_nodes)
        .map(|k| {
            let base = (expr * max_nodes + k) * n_rows;
            let col = &partials[base..base + n_rows];
            affine_residual(ColumnSums::of(col, target), min_rows)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exact_affine_image_scores_zero() {
        // y = 3x + 5 exactly → 1 - R² = 0.
        let x: Vec<f32> = (0..200).map(|i| i as f32 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&v| 3.0 * f64::from(v) + 5.0).collect();
        let r = affine_residual(ColumnSums::of(&x, &y), 10.0).expect("a fit");
        assert!(r < 1e-9, "1 - R² = {r}");
        let c = abs_corr(ColumnSums::of(&x, &y), 10.0).expect("corr");
        assert!(c > 1.0 - 1e-6, "|r| = {c}");
    }

    #[test]
    fn an_unrelated_column_scores_near_one() {
        // y independent of x → R² ≈ 0 → 1 - R² ≈ 1.
        let x: Vec<f32> = (0..500).map(|i| ((i * 7 % 13) as f32) * 0.3).collect();
        let y: Vec<f64> = (0..500).map(|i| ((i * 5 % 11) as f64) * 0.2 + 1.0).collect();
        let r = affine_residual(ColumnSums::of(&x, &y), 10.0).expect("a fit");
        assert!(r > 0.5, "unrelated should be poorly explained: 1 - R² = {r}");
    }

    #[test]
    fn a_constant_column_is_none() {
        let x = vec![2.0f32; 100];
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(affine_residual(ColumnSums::of(&x, &y), 10.0).is_none(), "no slope from a constant column");
    }

    #[test]
    fn non_finite_rows_are_dropped_not_poisoned() {
        // y = 2x + 1, but a few rows have a NaN subtree value (a pole). The fit
        // over the finite rows is still exact.
        let mut x: Vec<f32> = (0..300).map(|i| i as f32 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * f64::from(v) + 1.0).collect();
        x[10] = f32::NAN;
        x[200] = f32::INFINITY;
        let s = ColumnSums::of(&x, &y);
        assert_eq!(s.n, 298.0, "two non-finite rows dropped");
        let r = affine_residual(s, 10.0).expect("a fit");
        assert!(r < 1e-9, "finite rows fit exactly: 1 - R² = {r}");
    }

    #[test]
    fn score_subtrees_indexes_the_partials_layout() {
        // Two subtrees in one expr's partials, max_nodes 4, 3 rows.
        // node0 = target itself (score 0), node1 = unrelated.
        let max_nodes = 4;
        let n_rows = 3;
        let target = vec![1.0, 2.0, 3.0];
        let mut partials = vec![0.0f32; max_nodes * n_rows];
        // expr 0, node 0: exactly the target → affine image, 1-R²=0.
        partials[0..n_rows].copy_from_slice(&[1.0, 2.0, 3.0]);
        // expr 0, node 1: constant → None.
        partials[n_rows..2 * n_rows].copy_from_slice(&[5.0, 5.0, 5.0]);
        let scores = score_subtrees(&partials, 0, 2, max_nodes, n_rows, &target, 2.0);
        assert_eq!(scores.len(), 2);
        assert!(scores[0].unwrap() < 1e-9, "node 0 is the target");
        assert!(scores[1].is_none(), "node 1 is constant");
    }
}
