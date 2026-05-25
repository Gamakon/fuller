//! Phase 1.4: data-aware denoise via extract-many-and-rank.
//!
//! This is BRIEF.md Rule 6 and the core of the denoise operator. It is also
//! the concrete demonstration that egglog replaces sympy: given a noisy
//! expression and training data, return the smallest equivalent form that
//! still fits — deterministically, in the real domain, with no sympy.
//!
//! Pipeline:
//!   1. saturate the `algebra` ruleset on the input term;
//!   2. `extract_variants(K)` — the K lowest-structural-cost equivalent forms;
//!   3. evaluate each candidate on the training rows;
//!   4. return the smallest-cost candidate whose R^2 vs the original input is
//!      within `tolerance`; if none qualify, return the input unchanged.
//!
//! egglog's structural `CostModel` decides "smallest"; the data decides
//! "still correct". The two concerns stay cleanly separated — exactly the seam
//! that keeps the e-graph deterministic while data-awareness lives outside it.

use egglog::extract::{Extractor, TreeAdditiveCostModel};
use egglog::prelude::exprs;
use egglog::{EGraph, TermDag};

use crate::eval::{eval_term, EvalError};
use crate::expr::MATH_DATATYPE;
use crate::ruleset::identities::ALGEBRA_RULESET;

/// Outcome of a denoise call.
#[derive(Debug, Clone)]
pub struct Denoised {
    /// The chosen expression as an s-expression string.
    pub expr: String,
    /// egglog structural cost of the chosen expression.
    pub cost: u64,
    /// True if a strictly smaller equivalent form was accepted; false if the
    /// input was returned unchanged (no opportunity, or none within tolerance).
    pub changed: bool,
}

/// Denoise `input` (egglog `Math` surface syntax) against training data.
///
/// `rows` is a list of `(var_bindings, _target)` — but the target is not used
/// directly; parity is measured against the *original input expression's*
/// predictions on the same rows (we are simplifying form, not refitting). This
/// matches BRIEF.md: a candidate is accepted if it reproduces the input's
/// behaviour on the data to within `tolerance` relative R^2 loss.
///
/// `k_variants` is how many of the smallest equivalent forms to consider.
pub fn denoise(
    input: &str,
    rows: &[Vec<(String, f64)>],
    tolerance: f64,
    k_variants: usize,
) -> Result<Denoised, String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, MATH_DATATYPE)
        .map_err(|e| format!("datatype: {e}"))?;
    egraph
        .parse_and_run_program(None, crate::expr::GUARD_RELATIONS)
        .map_err(|e| format!("guards: {e}"))?;
    egraph
        .parse_and_run_program(None, ALGEBRA_RULESET)
        .map_err(|e| format!("ruleset: {e}"))?;
    egraph
        .parse_and_run_program(
            None,
            &format!("(let __root {input})\n(run-schedule (saturate (run algebra)))"),
        )
        .map_err(|e| format!("insert/saturate {input:?}: {e}"))?;

    let (sort, value) = egraph
        .eval_expr(&exprs::var("__root"))
        .map_err(|e| format!("eval root: {e}"))?;

    // Reference predictions: the input expression's value on each row, computed
    // from the *best* (lowest-cost) extraction of the root e-class — which is
    // semantically the input itself, just canonicalised.
    let extractor = Extractor::compute_costs_from_rootsorts(
        Some(vec![sort.clone()]),
        &egraph,
        TreeAdditiveCostModel::default(),
    );
    let mut termdag = TermDag::default();

    let variants = extractor.extract_variants(&egraph, &mut termdag, value, k_variants.max(1));
    if variants.is_empty() {
        return Err(format!("no variants extracted for {input:?}"));
    }

    // All variants share the root e-class, so they are semantically equal by
    // construction — any one defines the reference behaviour. We pick the
    // highest-cost variant as the reference (it is closest to the input as
    // written and least likely to be a degenerate fold), and require accepted
    // candidates to reproduce it on the data. The R^2 check is therefore a
    // guard against candidates that fold to something out-of-domain (NaN) on
    // these rows, not a re-derivation of equivalence.
    let mut ordered = variants.clone();
    ordered.sort_by_key(|(c, _)| *c); // lowest cost first
    let (ref_cost, ref_term) = *ordered.last().expect("non-empty checked above");

    let reference: Vec<f64> = rows
        .iter()
        .map(|row| eval_row(&termdag, ref_term, row))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("evaluating reference: {e}"))?;

    // Walk candidates cheapest-first; accept the first within tolerance.
    for (cost, term) in ordered.iter() {
        let preds: Result<Vec<f64>, EvalError> =
            rows.iter().map(|row| eval_row(&termdag, *term, row)).collect();
        let preds = match preds {
            Ok(p) => p,
            Err(_) => continue, // unevaluable candidate — skip
        };
        if r2_loss(&reference, &preds) <= tolerance {
            let expr = termdag.to_string(*term);
            return Ok(Denoised {
                expr,
                cost: *cost,
                changed: *cost < ref_cost,
            });
        }
    }

    // Nothing within tolerance: return the reference (input) unchanged.
    Ok(Denoised {
        expr: termdag.to_string(ref_term),
        cost: ref_cost,
        changed: false,
    })
}

/// Evaluate a term on one row of `(name, value)` bindings.
fn eval_row(
    termdag: &TermDag,
    term: egglog::TermId,
    row: &[(String, f64)],
) -> Result<f64, EvalError> {
    eval_term(termdag, term, &|name: &str| {
        row.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    })
}

/// Relative R^2 loss of `preds` against `reference`: `1 - R^2`, clamped to
/// `[0, inf)`. A perfect reproduction gives 0. Rows where either side is NaN
/// are treated as a full miss (loss contribution via large residual), so an
/// out-of-domain candidate cannot masquerade as a fit.
fn r2_loss(reference: &[f64], preds: &[f64]) -> f64 {
    debug_assert_eq!(reference.len(), preds.len());
    if reference.is_empty() {
        return 0.0;
    }
    let n = reference.len() as f64;
    let mean = reference.iter().copied().filter(|v| v.is_finite()).sum::<f64>() / n;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (r, p) in reference.iter().zip(preds.iter()) {
        if !r.is_finite() || !p.is_finite() {
            // Disqualifying: a non-finite mismatch makes this candidate unfit.
            return f64::INFINITY;
        }
        ss_res += (r - p).powi(2);
        ss_tot += (r - mean).powi(2);
    }
    if ss_tot == 0.0 {
        // Reference is constant: loss is 0 iff residuals are 0.
        return if ss_res == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ss_res / ss_tot // = 1 - R^2
}

#[cfg(test)]
mod tests {
    use super::denoise;

    fn rows(var: &str, vals: &[f64]) -> Vec<Vec<(String, f64)>> {
        vals.iter().map(|v| vec![(var.to_string(), *v)]).collect()
    }

    fn rows2(a: &str, b: &str, pairs: &[(f64, f64)]) -> Vec<Vec<(String, f64)>> {
        pairs
            .iter()
            .map(|(x, y)| vec![(a.to_string(), *x), (b.to_string(), *y)])
            .collect()
    }

    #[test]
    fn structural_noise_shrinks_and_preserves() {
        // add(mul(x,1), mul(0,y)) == x ; must shrink to (Var "x") and fit.
        let data = rows2("x", "y", &[(1.0, 5.0), (2.0, -3.0), (3.0, 0.5), (-4.0, 2.0)]);
        let out = denoise(
            r#"(Add (Mul (Var "x") (Num 1.0)) (Mul (Num 0.0) (Var "y")))"#,
            &data,
            1e-3,
            64,
        )
        .expect("denoise");
        assert_eq!(out.expr, r#"(Var "x")"#, "should collapse to x");
        assert!(out.changed, "should report a shrink");
    }

    #[test]
    fn sqrt_pow2_collapses_to_abs() {
        // sqrt(x^2) == |x| on real data.
        let data = rows("x", &[1.0, -2.0, 3.0, -4.5, 0.0]);
        let out = denoise(r#"(Sqrt (Pow2 (Var "x")))"#, &data, 1e-3, 64).expect("denoise");
        assert_eq!(out.expr, r#"(Abs (Var "x"))"#);
        assert!(out.changed);
    }

    #[test]
    fn clean_expr_is_returned_unchanged() {
        // No denoising opportunity: x + y stays x + y.
        let data = rows2("x", "y", &[(1.0, 2.0), (3.0, 4.0)]);
        let out = denoise(r#"(Add (Var "x") (Var "y"))"#, &data, 1e-3, 64).expect("denoise");
        assert!(!out.changed, "no opportunity -> unchanged");
        assert_eq!(out.expr, r#"(Add (Var "x") (Var "y"))"#);
    }
}
