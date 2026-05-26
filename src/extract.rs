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
    // Denoise loads algebra identities + powers. It does NOT load `distribute`:
    // distribute's coefficient-hoisting/distributivity generates a huge (often
    // unbounded) equivalent-form e-class, which is fine for the parity scorer
    // (it inserts two fixed terms and checks e-class equality) but FATAL for
    // denoise, which calls `extract_variants` over that class and explodes /
    // hangs. Denoise's job is shrinking via bounded identities, not normal-form
    // canonicalisation — so it stays on the confluent, bounded subset.
    egraph
        .parse_and_run_program(None, ALGEBRA_RULESET)
        .map_err(|e| format!("algebra ruleset: {e}"))?;
    egraph
        .parse_and_run_program(None, crate::ruleset::powers::POWERS_RULESET)
        .map_err(|e| format!("powers ruleset: {e}"))?;
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "(let __root {input})\n\
                 (unstable-combined-ruleset denoise_all algebra powers)\n\
                 (run-schedule (saturate (run denoise_all)))"
            ),
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
    let mut chosen_expr = termdag.to_string(ref_term);
    let mut chosen_cost = ref_cost;
    let mut changed = false;
    for (cost, term) in ordered.iter() {
        let preds: Result<Vec<f64>, EvalError> =
            rows.iter().map(|row| eval_row(&termdag, *term, row)).collect();
        let preds = match preds {
            Ok(p) => p,
            Err(_) => continue, // unevaluable candidate — skip
        };
        if r2_loss(&reference, &preds) <= tolerance {
            chosen_expr = termdag.to_string(*term);
            chosen_cost = *cost;
            changed = *cost < ref_cost;
            break;
        }
    }

    // Sound data-aware subtree pruning (the safe replacement for "substitute G
    // with its constant"): drop additive terms / collapse multiplicative
    // factors that don't change predictions on the REAL data beyond tolerance.
    // This removes wallpaper like cos(G)*... or +sin(exp(..)) WITHOUT assuming
    // anything about variable identities — equivalence is checked on the rows.
    if let Some(pruned) = prune_on_data(&chosen_expr, rows, &reference, tolerance) {
        if pruned != chosen_expr {
            chosen_expr = pruned;
            // recompute structural cost cheaply as node count (proxy);
            // smaller string-tree => fewer nodes. Mark changed.
            chosen_cost = cost_of(&chosen_expr);
            changed = true;
        }
    }

    Ok(Denoised { expr: chosen_expr, cost: chosen_cost, changed })
}

// ---------------------------------------------------------------------------
// Data-aware subtree pruning (sound #8 replacement)
// ---------------------------------------------------------------------------

/// Minimal Math tree for pruning. Mirrors the `Math` constructors we may prune
/// through (Add/Sub/Mul/Div + leaves); other ops are opaque subtrees we keep
/// whole.
#[derive(Debug, Clone)]
enum PNode {
    Num(f64),
    Var(String),
    App(String, Vec<PNode>),
}

impl PNode {
    fn to_math(&self) -> String {
        match self {
            PNode::Num(v) => format!("(Num {})", fmt_f64(*v)),
            PNode::Var(n) => format!("(Var \"{n}\")"),
            PNode::App(op, ch) => {
                let parts: Vec<String> = ch.iter().map(PNode::to_math).collect();
                format!("({op} {})", parts.join(" "))
            }
        }
    }

    fn node_count(&self) -> usize {
        match self {
            PNode::Num(_) | PNode::Var(_) => 1,
            PNode::App(_, ch) => 1 + ch.iter().map(PNode::node_count).sum::<usize>(),
        }
    }
}

fn fmt_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() { format!("{v:.1}") } else { format!("{v}") }
}

fn cost_of(expr: &str) -> u64 {
    parse_pnode(expr).map(|n| n.node_count() as u64).unwrap_or(0)
}

/// Try to shrink `expr` by removing additive terms / multiplicative factors
/// whose removal keeps predictions within `tolerance` of `reference` on `rows`.
/// Greedy + repeated to a fixpoint. Returns the smallest fitting form, or None
/// if it can't be parsed.
fn prune_on_data(
    expr: &str,
    rows: &[Vec<(String, f64)>],
    reference: &[f64],
    tolerance: f64,
) -> Option<String> {
    let mut tree = parse_pnode(expr)?;
    loop {
        let candidates = prune_candidates(&tree);
        let mut improved = false;
        for cand in candidates {
            if fits(&cand, rows, reference, tolerance) && cand.node_count() < tree.node_count() {
                tree = cand;
                improved = true;
                break;
            }
        }
        if !improved {
            break;
        }
    }
    Some(tree.to_math())
}

/// Generate one-step prunings: for each Add/Sub drop a side; for each Mul drop
/// a factor (replace the product with the surviving factor); for Div drop the
/// divisor (replace with numerator). Recurses so inner subtrees are tried too.
fn prune_candidates(node: &PNode) -> Vec<PNode> {
    let mut out = Vec::new();
    if let PNode::App(op, ch) = node {
        match (op.as_str(), ch.len()) {
            ("Add", 2) | ("Sub", 2) => {
                out.push(ch[0].clone()); // drop the second term
                if op == "Add" {
                    out.push(ch[1].clone()); // Add is symmetric for dropping
                }
            }
            ("Mul", 2) => {
                out.push(ch[0].clone());
                out.push(ch[1].clone());
            }
            ("Div", 2) => {
                out.push(ch[0].clone()); // drop divisor
            }
            _ => {}
        }
        // Recurse: rebuild this node with one child replaced by each of its
        // prunings.
        for (i, c) in ch.iter().enumerate() {
            for pc in prune_candidates(c) {
                let mut new_ch = ch.clone();
                new_ch[i] = pc;
                out.push(PNode::App(op.clone(), new_ch));
            }
        }
    }
    out
}

/// True if `node` reproduces `reference` within tolerance on the data.
fn fits(node: &PNode, rows: &[Vec<(String, f64)>], reference: &[f64], tolerance: f64) -> bool {
    // Evaluate the pruned tree through the egglog evaluator by rendering to a
    // Math string and parsing into a TermDag.
    let math = node.to_math();
    let mut egraph = match build_eval_egraph(&math) {
        Some(e) => e,
        None => return false,
    };
    let (sort, value) = match egraph.eval_expr(&exprs::var("__p")) {
        Ok(sv) => sv,
        Err(_) => return false,
    };
    let (termdag, term, _) = match egraph.extract_value(&sort, value) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let preds: Result<Vec<f64>, EvalError> =
        rows.iter().map(|row| eval_row(&termdag, term, row)).collect();
    match preds {
        Ok(p) => r2_loss(reference, &p) <= tolerance,
        Err(_) => false,
    }
}

/// A bare e-graph (datatype only) holding `__p = math`, for evaluation.
fn build_eval_egraph(math: &str) -> Option<EGraph> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, MATH_DATATYPE).ok()?;
    egraph
        .parse_and_run_program(None, &format!("(let __p {math})"))
        .ok()?;
    Some(egraph)
}

/// Parse a Math s-expression into a `PNode`.
fn parse_pnode(s: &str) -> Option<PNode> {
    let toks = pnode_tokenize(s);
    let mut pos = 0;
    let n = pnode_parse(&toks, &mut pos)?;
    if pos == toks.len() {
        Some(n)
    } else {
        None
    }
}

fn pnode_tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' => {
                out.push(c.to_string());
                chars.next();
            }
            '"' => {
                let mut t = String::from("\"");
                chars.next();
                for c2 in chars.by_ref() {
                    if c2 == '"' {
                        break;
                    }
                    t.push(c2);
                }
                t.push('"');
                out.push(t);
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut t = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '(' || c2 == ')' || c2.is_whitespace() {
                        break;
                    }
                    t.push(c2);
                    chars.next();
                }
                out.push(t);
            }
        }
    }
    out
}

fn pnode_parse(toks: &[String], pos: &mut usize) -> Option<PNode> {
    if toks.get(*pos)? != "(" {
        return None;
    }
    *pos += 1;
    let head = toks.get(*pos)?.clone();
    *pos += 1;
    let node = match head.as_str() {
        "Num" => {
            let v: f64 = toks.get(*pos)?.parse().ok()?;
            *pos += 1;
            PNode::Num(v)
        }
        "Var" => {
            let name = toks.get(*pos)?.trim_matches('"').to_string();
            *pos += 1;
            PNode::Var(name)
        }
        ctor => {
            let mut ch = Vec::new();
            while *pos < toks.len() && toks[*pos] != ")" {
                ch.push(pnode_parse(toks, pos)?);
            }
            PNode::App(ctor.to_string(), ch)
        }
    };
    if toks.get(*pos)? != ")" {
        return None;
    }
    *pos += 1;
    Some(node)
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

    /// Sound #8 replacement: a multiplicative wallpaper factor that is ~1 on
    /// the actual data gets dropped — WITHOUT assuming what the variable is.
    /// Here `g` ~ 1.0 on every row, so `x * g` should prune to `x`.
    #[test]
    fn data_aware_prune_drops_near_unit_factor() {
        // g is ~1.0 across rows; the true signal is x. mul(x, g) -> x.
        let data: Vec<Vec<(String, f64)>> = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (-4.0, 1.0)]
            .iter()
            .map(|(x, g)| vec![("x".to_string(), *x), ("g".to_string(), *g)])
            .collect();
        let out = denoise(r#"(Mul (Var "x") (Var "g"))"#, &data, 1e-3, 64).expect("denoise");
        assert_eq!(out.expr, r#"(Var "x")"#, "near-unit factor g should be pruned");
        assert!(out.changed);
    }

    /// And it must NOT prune a factor that actually matters on the data.
    #[test]
    fn data_aware_prune_keeps_real_factor() {
        // g varies and matters; mul(x, g) must NOT drop to x.
        let data: Vec<Vec<(String, f64)>> = [(1.0, 2.0), (2.0, 5.0), (3.0, 0.3), (-4.0, 9.0)]
            .iter()
            .map(|(x, g)| vec![("x".to_string(), *x), ("g".to_string(), *g)])
            .collect();
        let out = denoise(r#"(Mul (Var "x") (Var "g"))"#, &data, 1e-3, 64).expect("denoise");
        assert_eq!(out.expr, r#"(Mul (Var "x") (Var "g"))"#, "real factor must be kept");
        assert!(!out.changed);
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
