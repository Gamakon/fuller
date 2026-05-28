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
use rayon::prelude::*;

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

/// One candidate from `denoise_candidates`: a (potentially) equivalent form
/// of the input expression, with its structural cost and whether it's the
/// reference (input) itself. The caller decides which to keep — typically by
/// scoring each through HFF and picking the lowest-angle one.
#[derive(Debug, Clone)]
pub struct DenoiseCandidate {
    /// Candidate expression as Math s-expression string.
    pub expr: String,
    /// egglog structural cost (lower = simpler).
    pub cost: u64,
    /// True if this is the original input expression itself (cost-tied at
    /// the highest-cost variant in the e-class extraction).
    pub is_original: bool,
}

/// Which rule families to saturate when enumerating an equivalence class for
/// the tournament figure. Mirrors `parity::Family` but lives here so `extract`
/// has no parity dependency. distribute+trig co-saturated explode, so the caller
/// picks ONE family; iterations are bounded (never `saturate`) so a divergent
/// rule can't peg the machine — the same discipline as the parity scorer.
#[derive(Debug, Clone, Copy)]
pub enum EclassFamily {
    /// algebra + powers + distribute.
    Algebra,
    /// algebra + powers + trig.
    Trig,
    /// algebra + powers + distribute + wide (comm/assoc). FORM-GENERATING: grows
    /// the e-class so the extraction tournament has many equal members to rank.
    /// Combinatorial — callers MUST use a small bounded `iters` + kill-guard.
    Wide,
    /// algebra + powers + trig + trig_fu + wide. The STRUCTURAL family: trig_fu's
    /// bidirectional product<->sum / angle-addition / tan<->sin·cos rewrites put
    /// forms of DIFFERENT op count and transcendental profile in one e-class — so
    /// the HFF angular pick can genuinely differ from the scalar (size) pick.
    /// Strongly combinatorial (trig_fu is non-terminating unbounded) — small
    /// bounded `iters` + kill-guard, NEVER co-saturated with distribute.
    Structural,
}

/// Enumerate up to `k` lowest-cost members of `input`'s equivalence class under
/// a full rule family (NOT the bounded denoise subset). This is the wide
/// saturation the tournament figure needs: many goal-equivalent forms to score.
///
/// `iters` bounds the `run` schedule (distribute/trig do not reach a fixpoint
/// together with expand rules; a bound truncates growth — the honest, terminating
/// outcome). Returns `(cost, Math s-expression)` per distinct variant.
pub fn eclass_variants(
    input: &str,
    family: EclassFamily,
    k: usize,
    iters: u32,
) -> Result<Vec<(u64, String)>, String> {
    let (egraph, sort, value) = saturate_family(input, family, iters)?;
    let extractor = Extractor::compute_costs_from_rootsorts(
        Some(vec![sort]),
        &egraph,
        TreeAdditiveCostModel::default(),
    );
    let mut termdag = TermDag::default();
    let variants = extractor.extract_variants(&egraph, &mut termdag, value, k.max(1));

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (cost, term) in variants {
        let expr = termdag.to_string(term);
        if seen.insert(expr.clone()) {
            out.push((cost, expr));
        }
    }
    Ok(out)
}

/// Like [`eclass_variants`], but ranks the equivalence class by the CDF-corrected
/// hyperspherical-fitness angle (see [`crate::score`]) instead of egglog's scalar
/// tree cost. The extractor's per-e-class winner is chosen by the angular
/// [`MeasureVector`] ordering, evaluated as the walk proceeds — so a cleaner form
/// (fewer nodes, no transcendental self-nesting, …) wins each class.
///
/// Returns `(angle_percentile, expr)` per distinct extracted form. The percentile
/// is the term's own CDF-corrected angle (0 = best), recomputed from the rendered
/// term's structure so it is independent of the internal accumulation order.
pub fn eclass_extract_hff(
    input: &str,
    family: EclassFamily,
    k: usize,
    iters: u32,
    exclude_measures: &[String],
) -> Result<Vec<(f64, String)>, String> {
    use egglog::extract::hff_extract;

    let (egraph, sort, value) = saturate_family(input, family, iters)?;
    let mut termdag = TermDag::default();

    // The HFF tournament scorer: render the candidate whole term and score it by
    // the hyperspherical-fitness angle over its /pattern/{measure} vector (the
    // rule library in `crate::score`), with any measures in `exclude_measures`
    // turned off. NON-monotone by design — a bigger term may score better — which
    // the vendored `hff_extract` (replacing scalar Bellman-Ford) permits and the
    // stock extractor could not.
    let excl: Vec<&str> = exclude_measures.iter().map(String::as_str).collect();
    let score = |td: &TermDag, t: egglog::TermId| -> f64 {
        crate::score::score_expr_excluding(&td.to_string(t), &excl)
    };

    let ranked = hff_extract(&egraph, &mut termdag, value, sort, &score, k.max(1));
    let mut out: Vec<(f64, String)> = ranked
        .into_iter()
        .map(|(s, t)| (s, termdag.to_string(t)))
        .collect();

    // SEED the input form itself. The enumerator walks the e-graph from the root
    // e-class and, for a richly-expanded class, can fill its bound with expanded
    // members before reaching the compact e-node — so the original (often the
    // cleanest) form can be missing from `ranked`. The input is a member of its
    // own e-class by construction, so scoring it and merging guarantees the
    // tournament always considers it. (Found on lean_I_8_14: the gene IS
    // `Pow2(Sub ..)` but the enumerator only surfaced its multiplied-out forms.)
    let in_score = crate::score::score_expr_excluding(input, &excl);
    if !out.iter().any(|(_, e)| e == input) {
        out.push((in_score, input.to_string()));
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

/// Saturate `input` under a rule `family` (bounded by `iters`) and return the
/// e-graph with the root's sort and value — the shared scaffold for the two
/// e-class extractors (scalar and HFF-angular).
fn saturate_family(
    input: &str,
    family: EclassFamily,
    iters: u32,
) -> Result<(EGraph, egglog::ArcSort, egglog::Value), String> {
    use crate::ruleset::distribute::DISTRIBUTE_RULESET;
    use crate::ruleset::powers::POWERS_RULESET;
    use crate::ruleset::trig::TRIG_RULESET;
    use crate::ruleset::trig_fu::TRIG_FU_RULESET;
    use crate::ruleset::wide::WIDE_RULESET;

    // `rules` = the ruleset definitions to load. `schedule` = the run-schedule
    // body. For families with expanders, we run a TWO-PHASE schedule: first
    // saturate the CONTRACTING rules (algebra/powers/trig shrink toward compact
    // forms) so the compact member is fully reached and is the cheap default,
    // THEN a bounded `repeat` of the EXPANDERS (trig_fu/wide/distribute) for
    // extra structural variety. Doing it the other way round (one combined
    // bounded run) lets expansions dominate the e-graph before contraction
    // completes, so the compact form never surfaces in extraction.
    let (rules, schedule) = match family {
        EclassFamily::Algebra => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{DISTRIBUTE_RULESET}"),
            format!(
                "(unstable-combined-ruleset contract algebra powers)\n\
                 (unstable-combined-ruleset expand distribute)\n\
                 (run-schedule (saturate (run contract)) (repeat {iters} (run expand)))"
            ),
        ),
        EclassFamily::Trig => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{TRIG_RULESET}"),
            format!(
                "(unstable-combined-ruleset all algebra powers trig)\n\
                 (run-schedule (repeat {iters} (run all)))"
            ),
        ),
        EclassFamily::Wide => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{DISTRIBUTE_RULESET}\n{WIDE_RULESET}"),
            format!(
                "(unstable-combined-ruleset contract algebra powers)\n\
                 (unstable-combined-ruleset expand distribute wide)\n\
                 (run-schedule (saturate (run contract)) (repeat {iters} (run expand)))"
            ),
        ),
        EclassFamily::Structural => (
            format!(
                "{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{TRIG_RULESET}\n{TRIG_FU_RULESET}\n{WIDE_RULESET}"
            ),
            format!(
                "(unstable-combined-ruleset contract algebra powers trig)\n\
                 (unstable-combined-ruleset expand trig_fu wide)\n\
                 (run-schedule (saturate (run contract)) (repeat {iters} (run expand)))"
            ),
        ),
    };

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, MATH_DATATYPE)
        .map_err(|e| format!("datatype: {e}"))?;
    egraph
        .parse_and_run_program(None, crate::expr::GUARD_RELATIONS)
        .map_err(|e| format!("guards: {e}"))?;
    egraph
        .parse_and_run_program(None, &rules)
        .map_err(|e| format!("ruleset: {e}"))?;
    egraph
        .parse_and_run_program(
            None,
            &format!("(let __root {input})\n{schedule}"),
        )
        .map_err(|e| format!("insert/saturate {input:?}: {e}"))?;

    let (sort, value) = egraph
        .eval_expr(&exprs::var("__root"))
        .map_err(|e| format!("eval root: {e}"))?;
    Ok((egraph, sort, value))
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

    // Parallel: independent per-row evals. TermDag and the constants cache
    // (snap_karva::constant_values, OnceLock'd) are both Sync/&'static, so
    // rayon workers share them without contention.
    let reference: Vec<f64> = rows
        .par_iter()
        .map(|row| eval_row(&termdag, ref_term, row))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("evaluating reference: {e}"))?;

    // Walk candidates cheapest-first; accept the first within tolerance.
    let mut chosen_expr = termdag.to_string(ref_term);
    let mut chosen_cost = ref_cost;
    let mut changed = false;
    for (cost, term) in ordered.iter() {
        let preds: Result<Vec<f64>, EvalError> =
            rows.par_iter().map(|row| eval_row(&termdag, *term, row)).collect();
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

/// Like `denoise`, but returns ALL candidates instead of picking one by a
/// hardcoded tolerance. The caller (engine) is expected to score each via
/// HFF and pick the lowest-angle one — gamakAST's job is to PROPOSE
/// candidates, the engine's HFF cone DISPOSES.
///
/// Candidates include:
///   1. Every e-class variant (equivalent forms under the algebra+powers
///      rulesets).
///   2. Greedy `prune_on_data` shrinkings at multiple R²-loss budgets
///      (1e-10, 1e-6, 1e-3, 1e-2, 1e-1) — each produces a (potentially)
///      different pruned form. NOTE: r²-loss here measures drift from the
///      input's own predictions (not truth). A pruned form that drops
///      data-negligible atoms (e.g. `eps0+x → x`) has loss ~0 and is
///      strictly cleaner — the engine will see lower n_nodes and an
///      improved (or unchanged) HFF vec.
///
/// Returns candidates in no particular order; engine should score every one.
pub fn denoise_candidates(
    input: &str,
    rows: &[Vec<(String, f64)>],
    k_variants: usize,
) -> Result<Vec<DenoiseCandidate>, String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, MATH_DATATYPE)
        .map_err(|e| format!("datatype: {e}"))?;
    egraph
        .parse_and_run_program(None, crate::expr::GUARD_RELATIONS)
        .map_err(|e| format!("guards: {e}"))?;
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

    let mut ordered = variants.clone();
    ordered.sort_by_key(|(c, _)| *c);
    let (ref_cost, ref_term) = *ordered.last().expect("non-empty checked above");
    let ref_expr = termdag.to_string(ref_term);

    let reference: Vec<f64> = rows
        .par_iter()
        .map(|row| eval_row(&termdag, ref_term, row))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("evaluating reference: {e}"))?;

    // 1. Every variant — verify it can eval, include if so.
    let mut out: Vec<DenoiseCandidate> = Vec::new();
    let mut seen_exprs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (cost, term) in ordered.iter() {
        let preds: Result<Vec<f64>, EvalError> =
            rows.par_iter().map(|row| eval_row(&termdag, *term, row)).collect();
        if preds.is_err() {
            continue; // unevaluable on data — skip
        }
        let expr = termdag.to_string(*term);
        if !seen_exprs.insert(expr.clone()) {
            continue;
        }
        let is_original = *cost == ref_cost && expr == ref_expr;
        out.push(DenoiseCandidate { expr, cost: *cost, is_original });
    }

    // 2. Pruned forms at multiple tolerances. The reference for pruning is
    // the input's own predictions; a prune that drops a data-negligible
    // atom has near-zero drift and is strictly cleaner.
    for &tol in &[1e-10_f64, 1e-6, 1e-3, 1e-2, 1e-1] {
        if let Some(pruned) = prune_on_data(&ref_expr, rows, &reference, tol) {
            if seen_exprs.insert(pruned.clone()) {
                out.push(DenoiseCandidate {
                    expr: pruned.clone(),
                    cost: cost_of(&pruned),
                    is_original: false,
                });
            }
        }
    }

    Ok(out)
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
        rows.par_iter().map(|row| eval_row(&termdag, term, row)).collect();
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
///
/// Falls back to `snap_karva::constant_values()` for any name not present in
/// the row — so universal atoms (pi, G, hbar, eps0, kB, ...) that the GA
/// registers as SymbolTerminals resolve to their numeric values during
/// data-driven pruning. Without this, every candidate referencing an atom
/// errors as UnboundVar and prune_on_data can't see that the term is
/// numerically negligible against the data variables.
fn eval_row(
    termdag: &TermDag,
    term: egglog::TermId,
    row: &[(String, f64)],
) -> Result<f64, EvalError> {
    let consts = crate::snap_karva::constant_values();
    eval_term(termdag, term, &|name: &str| {
        // Row values win (a data column named e.g. "pi" would override).
        row.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
            .or_else(|| consts.get(name).copied())
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
    use super::{denoise, eclass_extract_hff, EclassFamily};

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
    fn hff_extract_ranks_class_by_angle() {
        // (x*1 + 0) has an equal, cleaner form (x). The HFF extractor returns
        // forms ranked ascending by the log-percentile score (more negative =
        // rarer/better), cleanest first.
        let input = r#"(Add (Mul (Var "x") (Num 1.0)) (Num 0.0))"#;
        let out = eclass_extract_hff(input, EclassFamily::Algebra, 32, 12, &[]).expect("hff extract");
        assert!(!out.is_empty());
        // Sorted ascending (best first).
        for w in out.windows(2) {
            assert!(w[0].0 <= w[1].0, "not sorted by score: {out:?}");
        }
        // Scores are raw TrueNorth angles: finite and >= 0.
        for (p, _) in &out {
            assert!(p.is_finite() && *p >= 0.0, "angle {p} not >= 0");
        }
        // The cleanest form reachable here is the bare variable.
        assert_eq!(out[0].1, r#"(Var "x")"#, "cleanest form should win the angle");
    }

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
