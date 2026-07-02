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

/// Iteration bound for the CONTRACTING phases that previously ran unbounded
/// `(saturate ..)`. The contracting sets (algebra/powers, plus trig for the
/// Structural family) reach fixpoint well within this on real genes; the bound
/// exists so a future divergent rule truncates instead of hanging the
/// never-raise API. Matches the parity scorer's per-family bound.
const CONTRACT_ITERS: u32 = 40;

/// Iteration bound for the denoise saturation (algebra + powers only). Same
/// rationale: bounded `repeat`, never unbounded `saturate`, in a hot GA loop.
const DENOISE_ITERS: u32 = 40;

/// Outcome of a denoise call.
#[derive(Debug, Clone)]
pub struct Denoised {
    /// The chosen expression as an s-expression string.
    pub expr: String,
    /// Structural cost (node count) of the chosen expression.
    pub cost: u64,
    /// True if a strictly smaller equivalent form was accepted; false if the
    /// input was returned unchanged (no opportunity, or none within tolerance).
    /// When false, `expr` is the input string verbatim.
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
    /// Structural cost as NODE COUNT (lower = simpler) — one unit for every
    /// candidate (extracted variants, the seeded input, pruned forms alike),
    /// so callers may sort on it directly.
    pub cost: u64,
    /// True if this is the original input expression itself, as written.
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
/// Returns `(angle, expr)` per distinct extracted form. The angle is the term's
/// own raw TrueNorth angle (0 = best; fixed measure dimension, see
/// [`crate::score`]), recomputed from the rendered term's structure so it is
/// independent of the internal accumulation order.
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

/// INSTRUMENTED e-class tournament: rank the equivalence class by TrueNorth
/// over `[form measures | measured behaviour on train rows | on val rows]`.
///
/// Every member of a sound class computes the same ideal function, but the
/// f64 behaviour of algebraically-equal forms differs MEASURABLY: rounding
/// divergence (catastrophic cancellation), introduced/removed domain failures
/// (NaN/inf) on the actual data distribution. Each candidate is therefore run
/// on the training rows AND on held-out validation rows and compared against
/// the input's own predictions — so the selected rewrite is the form whose
/// measured behaviour is cleanest and STAYS clean off the profiling set (the
/// rewrite choice cannot overfit the rows it was profiled on).
///
/// Two components are appended per row set, each in [0,1] with 0 best:
/// * `domain_mismatch` — fraction of rows where candidate and input disagree
///   about being finite (introduced OR removed NaN/inf: a behaviour change).
/// * `disagreement` — log-relative-error of candidate vs input over rows
///   where both are finite (1e-16, pure f64 rounding, maps to ~0; an O(1)
///   divergence maps to 1). This is the measured numerical-stability column.
///
/// Candidates that fail to evaluate rank last (INFINITY). Returns
/// `(angle, expr)` ascending — best first — with the input seeded like
/// [`eclass_extract_hff`]. NOTE: TrueNorth trades components against each
/// other; if "any behaviour change loses" must be a hard rule, filter on the
/// mismatch columns before adopting rather than relying on the angle.
pub fn eclass_extract_hff_instrumented(
    input: &str,
    family: EclassFamily,
    k: usize,
    iters: u32,
    exclude_measures: &[String],
    rows_train: &[Vec<(String, f64)>],
    rows_val: &[Vec<(String, f64)>],
) -> Result<Vec<(f64, String)>, String> {
    use egglog::extract::hff_extract;

    // The input's own behaviour on both row sets — the measurement reference.
    let ref_tr = eval_expr_rows(input, rows_train)?;
    let ref_va = eval_expr_rows(input, rows_val)?;

    let (egraph, sort, value) = saturate_family(input, family, iters)?;
    let mut termdag = TermDag::default();
    let excl: Vec<&str> = exclude_measures.iter().map(String::as_str).collect();

    let score = |td: &TermDag, t: egglog::TermId| -> f64 {
        let expr = td.to_string(t);
        let Some(node) = crate::score::parse(&expr) else {
            return f64::INFINITY; // unrenderable/unparseable — rank last
        };
        let mut x = crate::score::measure_vector_excluding(&node, &excl);
        // Instrumented behaviour columns, measured per row set.
        let preds_tr: Result<Vec<f64>, EvalError> =
            rows_train.iter().map(|row| eval_row(td, t, row)).collect();
        let preds_va: Result<Vec<f64>, EvalError> =
            rows_val.iter().map(|row| eval_row(td, t, row)).collect();
        let (Ok(preds_tr), Ok(preds_va)) = (preds_tr, preds_va) else {
            return f64::INFINITY; // unevaluable on the data — rank last
        };
        let (dm_tr, dis_tr) = instrumented_components(&preds_tr, &ref_tr);
        let (dm_va, dis_va) = instrumented_components(&preds_va, &ref_va);
        x.extend([dm_tr, dis_tr, dm_va, dis_va]);
        crate::score::truenorth_angle(&x)
    };

    let ranked = hff_extract(&egraph, &mut termdag, value, sort, &score, k.max(1));
    let mut out: Vec<(f64, String)> = ranked
        .into_iter()
        .map(|(s, t)| (s, termdag.to_string(t)))
        .collect();

    // SEED the input as written (see eclass_extract_hff). Its instrumented
    // components are zero by definition — it agrees with itself — so its
    // angle is the form vector's alone, extended with zeros.
    if !out.iter().any(|(_, e)| e == input) {
        let in_score = match crate::score::parse(input) {
            Some(node) => {
                let mut x = crate::score::measure_vector_excluding(&node, &excl);
                x.extend([0.0, 0.0, 0.0, 0.0]);
                crate::score::truenorth_angle(&x)
            }
            None => f64::INFINITY,
        };
        out.push((in_score, input.to_string()));
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

/// The two measured behaviour components of a candidate against the input's
/// reference predictions — `(domain_mismatch, disagreement)`, each [0,1],
/// 0 best. Both-non-finite rows count as agreement (same out-of-domain
/// behaviour); the disagreement statistic is a log-relative-error over the
/// rows where both sides are finite, mapped so that pure f64 rounding noise
/// (~1e-16 relative) scores ~0 and an O(1) divergence scores 1.
fn instrumented_components(cand: &[f64], reference: &[f64]) -> (f64, f64) {
    debug_assert_eq!(cand.len(), reference.len());
    if reference.is_empty() {
        return (0.0, 0.0);
    }
    let n = reference.len() as f64;
    let mut mismatch = 0usize;
    let mut rel_sum = 0.0;
    let mut common = 0usize;
    for (c, r) in cand.iter().zip(reference.iter()) {
        match (c.is_finite(), r.is_finite()) {
            (true, true) => {
                common += 1;
                rel_sum += (c - r).abs() / r.abs().max(1e-300);
            }
            (false, false) => {} // agree out-of-domain
            _ => mismatch += 1,
        }
    }
    let dm = mismatch as f64 / n;
    let dis = if common == 0 {
        0.0
    } else {
        let mean_rel = rel_sum / common as f64;
        if mean_rel <= 0.0 {
            0.0
        } else {
            // log-relative-error map (the FP-accuracy convention: bits of
            // error): 1e-16 -> 0, 1e-8 -> 0.5, >= 1 -> 1.
            ((mean_rel.log10() + 16.0) / 16.0).clamp(0.0, 1.0)
        }
    };
    (dm, dis)
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
    // Every combined ruleset includes `guards` (fact propagation lives in a
    // named ruleset; a schedule that omits it derives no facts and the guarded
    // rules silently never fire).
    //
    // The contract phase is `repeat CONTRACT_ITERS`, NOT unbounded `saturate`:
    // for Structural the contract set includes trig, which carries expansion
    // rules (its own header mandates bounded runs), and even for algebra/powers
    // an unbounded saturate means one future divergent rule hangs the
    // never-raise API with no cap. 40 matches the parity scorer's bound; the
    // contracting sets reach fixpoint well within it, after which iterations
    // are cheap no-ops (seminaive).
    let (rules, schedule) = match family {
        EclassFamily::Algebra => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{DISTRIBUTE_RULESET}"),
            format!(
                "(unstable-combined-ruleset contract guards algebra powers)\n\
                 (unstable-combined-ruleset expand guards distribute)\n\
                 (run-schedule (repeat {CONTRACT_ITERS} (run contract)) (repeat {iters} (run expand)))"
            ),
        ),
        EclassFamily::Trig => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{TRIG_RULESET}"),
            format!(
                "(unstable-combined-ruleset all guards algebra powers trig)\n\
                 (run-schedule (repeat {iters} (run all)))"
            ),
        ),
        EclassFamily::Wide => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{DISTRIBUTE_RULESET}\n{WIDE_RULESET}"),
            format!(
                "(unstable-combined-ruleset contract guards algebra powers)\n\
                 (unstable-combined-ruleset expand guards distribute wide)\n\
                 (run-schedule (repeat {CONTRACT_ITERS} (run contract)) (repeat {iters} (run expand)))"
            ),
        ),
        EclassFamily::Structural => (
            format!(
                "{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{TRIG_RULESET}\n{TRIG_FU_RULESET}\n{WIDE_RULESET}"
            ),
            format!(
                "(unstable-combined-ruleset contract guards algebra powers trig)\n\
                 (unstable-combined-ruleset expand guards trig_fu wide)\n\
                 (run-schedule (repeat {CONTRACT_ITERS} (run contract)) (repeat {iters} (run expand)))"
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
    denoise_assuming(input, rows, tolerance, k_variants, &[], &[])
}

/// Render caller-supplied domain facts as egglog asserts. A variable the
/// engine KNOWS is positive (from its var_ranges) unlocks the guarded
/// rewrites: Abs-shed, div-cancellation, Pow2(Sqrt), ... Never assumed —
/// only asserted by the caller, which is what keeps the guards sound.
fn guard_asserts(positive_vars: &[String], nonzero_vars: &[String]) -> String {
    let mut s = String::new();
    for v in positive_vars {
        s.push_str(&format!("(is-positive (Var \"{v}\"))\n"));
    }
    for v in nonzero_vars {
        s.push_str(&format!("(is-nonzero (Var \"{v}\"))\n"));
    }
    s
}

/// Like [`denoise`], but first asserts `is-positive` / `is-nonzero` facts for
/// the named variables (mirrors `parity::proves_equal_assuming`). This is how
/// the SR engine's domain knowledge (var_ranges) reaches the guarded rules —
/// e.g. `positive_vars=["a"]` lets `Abs(a^(3/2))` shed its Abs, the wrapper
/// that most often fails SRBench's exact symbolic-solution check on an
/// otherwise-recovered law.
pub fn denoise_assuming(
    input: &str,
    rows: &[Vec<(String, f64)>],
    tolerance: f64,
    k_variants: usize,
    positive_vars: &[String],
    nonzero_vars: &[String],
) -> Result<Denoised, String> {
    // No data, no acceptance evidence. The e-class variants are equal under
    // the rewrite theory, but the theory is sound over the reals, not
    // pointwise over f64/NaN — the R^2 gate on rows is what protects against
    // a variant that folds to something out-of-domain. Without rows that gate
    // is vacuous (and the greedy pruner, whose edits are NOT equivalences,
    // would happily shred the expression to a single leaf). Return unchanged.
    if rows.is_empty() {
        return Ok(Denoised { expr: input.to_string(), cost: cost_of(input), changed: false });
    }

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
    let asserts = guard_asserts(positive_vars, nonzero_vars);
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "(let __root {input})\n{asserts}\
                 (unstable-combined-ruleset denoise_all guards algebra powers)\n\
                 (run-schedule (repeat {DENOISE_ITERS} (run denoise_all)))"
            ),
        )
        .map_err(|e| format!("insert/saturate {input:?}: {e}"))?;

    let (sort, value) = egraph
        .eval_expr(&exprs::var("__root"))
        .map_err(|e| format!("eval root: {e}"))?;

    // Enumerate the k lowest-cost members of the root e-class as candidates.
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

    // The reference behaviour is the INPUT's own predictions, evaluated from
    // the input as written. The enumerator is NOT guaranteed to surface the
    // input among the k extracted variants (for a richly-expanded class it can
    // fill its bound with expanded members first — the same failure
    // `eclass_extract_hff` seeds against), so deriving the reference from any
    // extracted variant would measure candidates against a never-validated
    // cousin, and "unchanged" could silently return a different form.
    let reference = eval_expr_rows(input, rows)?;
    let input_nodes = cost_of(input);

    // Walk candidates cheapest-first; accept the first STRICT shrink within
    // tolerance. Non-shrinking forms are never accepted, so `changed == false`
    // always returns the input verbatim (never a same-size reformat).
    let mut ordered = variants;
    ordered.sort_by_key(|(c, _)| *c); // lowest cost first
    let mut chosen_expr = input.to_string();
    let mut chosen_cost = input_nodes;
    let mut changed = false;
    for (_, term) in ordered.iter() {
        let expr = termdag.to_string(*term);
        let nodes = cost_of(&expr);
        if nodes == 0 || nodes >= input_nodes {
            continue; // unparseable render, or not a strict shrink
        }
        // Parallel: independent per-row evals. TermDag and the constants cache
        // (snap_karva::constant_values, OnceLock'd) are both Sync/&'static, so
        // rayon workers share them without contention.
        let preds: Result<Vec<f64>, EvalError> =
            rows.par_iter().map(|row| eval_row(&termdag, *term, row)).collect();
        let preds = match preds {
            Ok(p) => p,
            Err(_) => continue, // unevaluable candidate — skip
        };
        if r2_loss(&reference, &preds) <= tolerance {
            chosen_expr = expr;
            chosen_cost = nodes;
            changed = true;
            break;
        }
    }

    // CONSTANT-SUBTREE FOLD + ADDITIVE STRIP, data-gated. The e-graph cannot
    // fold transcendental constants (no ln/sqrt f64 primitives), so remainders
    // like `pi*(r^2 + log|sqrt2|) - 1.0888` survive every saturation; the
    // evaluator folds them to literals, and the strip candidate drops ALL
    // additive constants at once — the paired-cancellation move the greedy
    // pruner below cannot reach (either term alone breaks the fit). Both
    // candidates go through the same strict-shrink + R^2 gate as everything
    // else; the fold is an f64-exact identity, the strip is justified only by
    // the data.
    let fold_base = fold_constant_subtrees(&chosen_expr).unwrap_or_else(|| chosen_expr.clone());
    let mut fold_cands: Vec<String> = Vec::new();
    if let Some(stripped) = strip_additive_constants(&fold_base) {
        fold_cands.push(stripped); // smallest first
    }
    fold_cands.push(fold_base);
    for cand in fold_cands {
        let nodes = cost_of(&cand);
        if nodes == 0 || nodes >= chosen_cost || cand == chosen_expr {
            continue;
        }
        if let Ok(preds) = eval_expr_rows(&cand, rows) {
            if r2_loss(&reference, &preds) <= tolerance {
                chosen_expr = cand;
                chosen_cost = nodes;
                changed = true;
                break;
            }
        }
    }

    // Sound data-aware subtree pruning (the safe replacement for "substitute G
    // with its constant"): drop additive terms / collapse multiplicative
    // factors that don't change predictions on the REAL data beyond tolerance.
    // This removes wallpaper like cos(G)*... or +sin(exp(..)) WITHOUT assuming
    // anything about variable identities — equivalence is checked on the rows.
    if let Some(pruned) = prune_on_data(&chosen_expr, rows, &reference, tolerance) {
        // Accept only a strict node-count shrink: string inequality alone can
        // be a pure reformat (e.g. float rendering), which must not flip
        // `changed`.
        let pruned_nodes = cost_of(&pruned);
        if pruned_nodes > 0 && pruned_nodes < cost_of(&chosen_expr) {
            chosen_expr = pruned;
            chosen_cost = pruned_nodes;
            changed = true;
        }
    }

    Ok(Denoised { expr: chosen_expr, cost: chosen_cost, changed })
}

/// Evaluate a Math s-expression string on every row: parse it into a bare
/// e-graph (datatype only, no rules), extract the term, and eval per row.
/// This is how the denoise entry points obtain the input's OWN reference
/// predictions, independent of what the variant enumerator surfaces.
fn eval_expr_rows(math: &str, rows: &[Vec<(String, f64)>]) -> Result<Vec<f64>, String> {
    let mut egraph =
        build_eval_egraph(math).ok_or_else(|| format!("could not parse {math:?}"))?;
    let (sort, value) = egraph
        .eval_expr(&exprs::var("__p"))
        .map_err(|e| format!("eval root: {e}"))?;
    let (termdag, term, _cost) = egraph
        .extract_value(&sort, value)
        .map_err(|e| format!("extract input: {e}"))?;
    rows.par_iter()
        .map(|row| eval_row(&termdag, term, row))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("evaluating reference: {e}"))
}

/// Like `denoise`, but returns ALL candidates instead of picking one by a
/// hardcoded tolerance. The caller (engine) is expected to score each via
/// HFF and pick the lowest-angle one — fuller's job is to PROPOSE
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
    denoise_candidates_assuming(input, rows, k_variants, &[], &[])
}

/// Like [`denoise_candidates`], with caller-asserted domain facts — see
/// [`denoise_assuming`].
pub fn denoise_candidates_assuming(
    input: &str,
    rows: &[Vec<(String, f64)>],
    k_variants: usize,
    positive_vars: &[String],
    nonzero_vars: &[String],
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
    let asserts = guard_asserts(positive_vars, nonzero_vars);
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "(let __root {input})\n{asserts}\
                 (unstable-combined-ruleset denoise_all guards algebra powers)\n\
                 (run-schedule (repeat {DENOISE_ITERS} (run denoise_all)))"
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

    // Reference behaviour = the INPUT's own predictions (see `denoise` for why
    // an extracted variant must not play this role).
    let reference = eval_expr_rows(input, rows)?;

    let mut ordered = variants;
    ordered.sort_by_key(|(c, _)| *c);

    // 1. Every variant — verify it can eval, include if so. `cost` is
    // recomputed as a node count so every candidate (variant, seeded input,
    // pruned form) carries the SAME cost unit — egglog's DefaultCost counts
    // literal children too, which would make the units incomparable.
    let mut out: Vec<DenoiseCandidate> = Vec::new();
    let mut seen_exprs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_, term) in ordered.iter() {
        let preds: Result<Vec<f64>, EvalError> =
            rows.par_iter().map(|row| eval_row(&termdag, *term, row)).collect();
        if preds.is_err() {
            continue; // unevaluable on data — skip
        }
        let expr = termdag.to_string(*term);
        if !seen_exprs.insert(expr.clone()) {
            continue;
        }
        let is_original = expr == input;
        let cost = cost_of(&expr);
        out.push(DenoiseCandidate { expr, cost, is_original });
    }

    // SEED the input as written: the enumerator may not surface it (see
    // `eclass_extract_hff`), and the caller's tournament must always be able
    // to keep the original.
    if seen_exprs.insert(input.to_string()) {
        out.push(DenoiseCandidate {
            expr: input.to_string(),
            cost: cost_of(input),
            is_original: true,
        });
    }

    // 2. Pruned forms at multiple tolerances. The reference for pruning is
    // the input's own predictions; a prune that drops a data-negligible
    // atom has near-zero drift and is strictly cleaner. Skipped when there
    // are no rows: prune edits are NOT equivalences, and without data there
    // is no evidence to justify them.
    if !rows.is_empty() {
        for &tol in &[1e-10_f64, 1e-6, 1e-3, 1e-2, 1e-1] {
            if let Some(pruned) = prune_on_data(input, rows, &reference, tol) {
                if seen_exprs.insert(pruned.clone()) {
                    out.push(DenoiseCandidate {
                        expr: pruned.clone(),
                        cost: cost_of(&pruned),
                        is_original: false,
                    });
                }
            }
        }
    }

    // 3. Constant-subtree fold + additive strip (see `denoise`): the e-graph
    // can't fold transcendental constants; the evaluator can. The strip is
    // NOT an identity — the caller's scoring (HFF + R^2 on data) disposes.
    if let Some(folded) = fold_constant_subtrees(input) {
        let stripped = strip_additive_constants(&folded);
        for cand in [stripped, Some(folded)].into_iter().flatten() {
            if cost_of(&cand) == 0 {
                continue; // unparseable render
            }
            if !rows.is_empty() && eval_expr_rows(&cand, rows).is_err() {
                continue; // unevaluable on data — same filter as the variants
            }
            if seen_exprs.insert(cand.clone()) {
                out.push(DenoiseCandidate {
                    expr: cand.clone(),
                    cost: cost_of(&cand),
                    is_original: false,
                });
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Constant-subtree folding (evaluator-backed; the e-graph can't do this)
// ---------------------------------------------------------------------------

/// Evaluate a variable-free Math subtree to its literal value: one empty row,
/// so every `Var` resolves through the registered-constant fallback in
/// `eval_row` (pi, sqrt2, G, ...). Exactly the evaluator's semantics — the
/// single source of truth for what an op computes.
fn eval_const_subtree(math: &str) -> Option<f64> {
    eval_expr_rows(math, &[Vec::new()]).ok().and_then(|v| v.first().copied())
}

/// Replace every MAXIMAL variable-free subtree with the literal the evaluator
/// computes for it: `Log(Abs(Var "sqrt2"))` -> `(Num 0.3465735902799726)`.
///
/// Why this exists: egglog has f64 primitives for `+ - * neg` only — no
/// `ln`/`sqrt`/trig — so a transcendental constant subtree can NEVER fold in
/// the e-graph, in any rule family. The evaluator has the real functions.
/// This is the piece that lets `pi*(r^2 + log|sqrt2|) - 1.0888` reduce: fold
/// the transcendental remainders to literals, then let the data-gated strip /
/// prune drop what cancels.
///
/// Soundness: replacing a variable-free subtree by its own evaluated value is
/// an f64-exact identity under the crate's evaluator. Subtrees that evaluate
/// non-finite (out-of-domain constants) are left symbolic. Bare leaves are
/// never touched — a lone `(Var "pi")` stays a symbol (readability + snap).
/// Caveat: a DATA COLUMN named like a registered constant would shadow it at
/// eval time; callers gate the folded form on the data (R^2), which rejects
/// the fold in that pathological case.
fn fold_constant_subtrees(expr: &str) -> Option<String> {
    fn is_const(n: &PNode) -> bool {
        match n {
            PNode::Num(_) => true,
            PNode::Var(name) => crate::snap_karva::constant_values().contains_key(name),
            PNode::App(_, ch) => ch.iter().all(is_const),
        }
    }
    fn go(n: &PNode) -> PNode {
        if let PNode::App(op, ch) = n {
            // An App is >= 2 nodes, so folding it always shrinks.
            if is_const(n) {
                if let Some(v) = eval_const_subtree(&n.to_math()) {
                    if v.is_finite() {
                        return PNode::Num(v);
                    }
                }
                // Out-of-domain / unevaluable constant: keep it symbolic.
            }
            return PNode::App(op.clone(), ch.iter().map(go).collect());
        }
        n.clone()
    }
    let tree = parse_pnode(expr)?;
    Some(go(&tree).to_math())
}

/// One aggressive candidate: the tree with EVERY additive numeric constant
/// removed — each `Add` operand that is a bare `Num`, and each `Sub`
/// subtrahend that is a bare `Num`. NOT an identity: it is only ever offered
/// through the data gate (or to a caller who scores on data).
///
/// Why all-at-once: fitted linker offsets cancel in PAIRS across the tree
/// (`pi*(r^2 + c) - pi*c`), and the greedy one-step pruner cannot cross that
/// valley — dropping either term alone breaks the fit, dropping both is
/// exact. Run after [`fold_constant_subtrees`], which turns the constant
/// remainders into the bare `Num`s this looks for.
fn strip_additive_constants(expr: &str) -> Option<String> {
    fn go(n: &PNode) -> PNode {
        match n {
            PNode::App(op, ch) if op == "Add" && ch.len() == 2 => match (&ch[0], &ch[1]) {
                (PNode::Num(_), other) | (other, PNode::Num(_)) => go(other),
                _ => PNode::App(op.clone(), vec![go(&ch[0]), go(&ch[1])]),
            },
            PNode::App(op, ch)
                if op == "Sub" && ch.len() == 2 && matches!(ch[1], PNode::Num(_)) =>
            {
                go(&ch[0])
            }
            PNode::App(op, ch) => PNode::App(op.clone(), ch.iter().map(go).collect()),
            leaf => leaf.clone(),
        }
    }
    let tree = parse_pnode(expr)?;
    Some(go(&tree).to_math())
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
/// if it can't be parsed (or there is no data to justify any prune).
fn prune_on_data(
    expr: &str,
    rows: &[Vec<(String, f64)>],
    reference: &[f64],
    tolerance: f64,
) -> Option<String> {
    // Prune edits are NOT equivalences — they are justified ONLY by the data.
    // With no rows every candidate "fits" vacuously and the greedy loop would
    // shred the expression to a single leaf. Refuse instead.
    if rows.is_empty() || reference.is_empty() {
        return None;
    }
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
    let n = pnode_parse(&toks, &mut pos, 0)?;
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

fn pnode_parse(toks: &[String], pos: &mut usize, depth: usize) -> Option<PNode> {
    // Depth cap: fail the parse instead of overflowing the stack.
    if depth > crate::MAX_EXPR_DEPTH {
        return None;
    }
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
                ch.push(pnode_parse(toks, pos, depth + 1)?);
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
/// `[0, inf)`. A perfect reproduction gives 0.
///
/// Non-finite policy, per row:
/// * both sides agree out-of-domain (NaN == NaN, +inf == +inf, -inf == -inf):
///   the candidate reproduces the reference's behaviour there — the row is
///   excluded from the finite loss but does NOT disqualify. This is what lets
///   denoise still shrink `x*1 + 0` when some OTHER subterm is NaN on a row:
///   the old blanket rule (any non-finite reference row => INF for every
///   candidate including the reference itself) silently disabled denoise on
///   any gene touching log/sqrt of a sign-varying column.
/// * mismatched (one finite, one not; or differently-signed infinities / NaN
///   vs inf): disqualifying — the candidate changes observable behaviour.
///
/// The loss statistics (mean, ss_tot) are computed over the finite-agreeing
/// rows only.
fn r2_loss(reference: &[f64], preds: &[f64]) -> f64 {
    debug_assert_eq!(reference.len(), preds.len());
    let mut finite: Vec<(f64, f64)> = Vec::with_capacity(reference.len());
    for (r, p) in reference.iter().zip(preds.iter()) {
        if r.is_finite() && p.is_finite() {
            finite.push((*r, *p));
        } else {
            // Both out-of-domain in the same way is agreement; anything else
            // is a behavioural change and disqualifies the candidate.
            let agree = (r.is_nan() && p.is_nan()) || r == p;
            if !agree {
                return f64::INFINITY;
            }
        }
    }
    if finite.is_empty() {
        // Every row agrees (possibly all out-of-domain the same way).
        return 0.0;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().map(|(r, _)| *r).sum::<f64>() / n;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (r, p) in &finite {
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
        // forms ranked ascending by the raw TrueNorth angle (lower = cleaner),
        // cleanest first.
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

    /// The evaluator folds a transcendental constant subtree the e-graph
    /// never can (egglog has no ln primitive): log(|sqrt2|) becomes a literal.
    #[test]
    fn folds_transcendental_constant_subtree() {
        let data = rows("x", &[1.0, 2.0, 3.0, -4.0]);
        let input = r#"(Add (Var "x") (Log (Abs (Var "sqrt2"))))"#;
        let out = denoise(input, &data, 1e-3, 16).expect("denoise");
        assert!(out.changed, "constant Log subtree should fold: {}", out.expr);
        assert!(!out.expr.contains("Log"), "no symbolic Log should survive: {}", out.expr);
    }

    /// The smoke-test remainder, end-to-end: pi*(r^2 + log|sqrt2|) - pi*ln(sqrt2)
    /// == pi*r^2 exactly. Dropping either constant alone breaks the fit (the
    /// greedy pruner's valley); fold + strip crosses it in one gated move.
    #[test]
    fn folds_and_strips_cancelling_linker_offset() {
        let off = std::f64::consts::PI * std::f64::consts::SQRT_2.ln();
        let input = format!(
            r#"(Sub (Mul (Var "pi") (Add (Pow2 (Var "r")) (Log (Abs (Var "sqrt2"))))) (Num {off}))"#
        );
        let data = rows("r", &[0.5, 1.0, 2.0, 3.0, -1.5]);
        let out = denoise(&input, &data, 1e-6, 32).expect("denoise");
        assert_eq!(out.expr, r#"(Mul (Var "pi") (Pow2 (Var "r")))"#, "expected clean pi*r^2");
        assert!(out.changed);
    }

    /// The instrumented components: measured behaviour deltas in [0,1].
    #[test]
    fn instrumented_components_measure_behaviour() {
        use super::instrumented_components;
        // Identical predictions (incl. matching NaN): perfect agreement.
        let r = [1.0, 2.0, f64::NAN];
        assert_eq!(instrumented_components(&[1.0, 2.0, f64::NAN], &r), (0.0, 0.0));
        // Candidate finite where reference is NaN: domain mismatch on 1/3 rows.
        let (dm, _) = instrumented_components(&[1.0, 2.0, 3.0], &r);
        assert!((dm - 1.0 / 3.0).abs() < 1e-12, "dm={dm}");
        // Pure rounding-level divergence maps to ~0; O(1) divergence to 1.
        let (_, dis_small) =
            instrumented_components(&[1.0 + 1e-16, 2.0, f64::NAN], &r);
        assert!(dis_small < 0.05, "rounding noise should score ~0, got {dis_small}");
        let (_, dis_big) = instrumented_components(&[2.0, 4.0, f64::NAN], &r);
        assert!(dis_big > 0.9, "O(1) divergence should score ~1, got {dis_big}");
    }

    /// Instrumented tournament end-to-end: for a class whose members all agree
    /// on the data, the behaviour columns are uniform and the FORM measures
    /// decide — same winner as the data-free tournament — and the ranking is
    /// well-formed on distinct train/val row sets.
    #[test]
    fn instrumented_tournament_ranks_and_picks_clean_form() {
        use super::eclass_extract_hff_instrumented;
        let input = r#"(Add (Mul (Var "x") (Num 1.0)) (Num 0.0))"#;
        let tr = rows("x", &[1.0, 2.0, 3.0, -4.0]);
        let va = rows("x", &[10.0, -20.0, 0.5]);
        let out = eclass_extract_hff_instrumented(
            input,
            EclassFamily::Algebra,
            32,
            12,
            &[],
            &tr,
            &va,
        )
        .expect("instrumented tournament");
        assert!(!out.is_empty());
        for w in out.windows(2) {
            assert!(w[0].0 <= w[1].0, "not sorted ascending: {out:?}");
        }
        assert_eq!(out[0].1, r#"(Var "x")"#, "cleanest agreeing form should win");
    }

    /// Caller-asserted positivity sheds the Abs wrapper — the keplers3 shape
    /// Abs(a^(3/2)) with a > 0 from var_ranges. Without the assertion the
    /// wrapper must survive (soundness: rewrites never guess domains).
    #[test]
    fn positive_vars_sheds_abs_of_pow() {
        use super::denoise_assuming;
        let input = r#"(Mul (Num 2.0) (Abs (Pow (Var "a") (Num 1.5))))"#;
        let data = rows("a", &[0.5, 1.0, 2.0, 4.0]);
        // With the fact: is-positive(a) propagates to Pow(a, 1.5), Abs sheds.
        let shed = denoise_assuming(input, &data, 1e-6, 32, &["a".to_string()], &[])
            .expect("denoise_assuming");
        assert_eq!(shed.expr, r#"(Mul (Num 2.0) (Pow (Var "a") (Num 1.5)))"#);
        assert!(shed.changed);
        // Without the fact: unchanged, even though the data happens to be
        // positive — domain knowledge must be asserted, never inferred.
        let kept = denoise(input, &data, 1e-6, 32).expect("denoise");
        assert_eq!(kept.expr, input);
        assert!(!kept.changed);
    }

    /// The strip is data-gated: an additive constant that MATTERS must stay.
    #[test]
    fn does_not_strip_a_constant_that_matters() {
        let data = rows("x", &[1.0, 2.0, 3.0, -4.0]);
        let out =
            denoise(r#"(Add (Var "x") (Num 10.0))"#, &data, 1e-6, 16).expect("denoise");
        assert_eq!(out.expr, r#"(Add (Var "x") (Num 10.0))"#);
        assert!(!out.changed);
    }

    /// Bare constant symbols are never folded to literals: a lone (Var "pi")
    /// stays symbolic (readability + snap round-trip).
    #[test]
    fn bare_constant_symbol_stays_symbolic() {
        let data = rows("r", &[1.0, 2.0, 3.0]);
        let input = r#"(Mul (Var "pi") (Pow2 (Var "r")))"#;
        let out = denoise(input, &data, 1e-6, 16).expect("denoise");
        assert_eq!(out.expr, input, "pi must stay a symbol, not become 3.14159...");
        assert!(!out.changed);
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
