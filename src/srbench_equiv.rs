//! SRBench-style symbolic-equivalence oracle — fuller's sympy-free replacement
//! for SRBench's `assess_symbolic_model`.
//!
//! SRBench (cavalab/srbench) counts a discovered model as the law when, AFTER
//! rounding every float constant, EITHER the difference `truth - model` is a
//! constant OR the ratio `model / truth` is a constant. That is deliberately
//! looser than exact equivalence: a model that is `k * truth` (wrong overall
//! scale) or `truth + c` (wrong offset) still counts, because SR cannot pin an
//! overall units constant. See `symbolic_utils.round_floats` and
//! `assess_symbolic_model.py` in the SRBench repo.
//!
//! This module reproduces that verdict with fuller's e-graph, no sympy:
//!
//!   1. **round_floats** (`snap_constants`): every `Num` literal with `|a| <
//!      1e-4` becomes `0`, otherwise `round(a, 3)` — SRBench's exact thresholds.
//!      Applied to BOTH sides. This is what removes the fitted junk (`-4.04e-8`)
//!      and normalises `99.00000014` to `99.0` so cancellation can fire.
//!   2. Build `(Sub model truth)` and `(ProtectedDiv model truth)`.
//!   3. Saturate a BOUNDED oracle ruleset: algebra + powers + rational (folds
//!      and cancellation) + commutativity/associativity (reordering, so a buried
//!      `k * (a / k)` can bring `k` and `/k` adjacent) + literal division folds.
//!      Every variable is assumed nonzero (a scale/offset constant is only
//!      defined where denominators are nonzero — sound for this question, and
//!      the same assumption SRBench's algebraic simplification makes implicitly).
//!   4. The verdict is TRUE iff the `Sub` OR the `Div` e-class contains a `Num`
//!      leaf — i.e. the difference or the ratio folded to a constant.
//!
//! Boundedness: comm/assoc reorder a fixed operand multiset (bounded but
//! combinatorial on long chains), so the run is iteration-capped, never
//! `saturate`. The distributivity rules that explode alongside rational are NOT
//! included here — only pure comm/assoc, which is safe with rational.

use egglog::EGraph;

use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
use crate::ruleset::identities::ALGEBRA_RULESET;
use crate::ruleset::powers::POWERS_RULESET;
use crate::ruleset::rational::RATIONAL_RULESET;

/// SRBench's `round_floats`, on a fuller Math s-expression string.
///
/// Walks the s-expr and rewrites each `(Num <f>)` literal: `|f| < 1e-4 -> 0`,
/// else `round(f, 3)`. Structural tokens are untouched. Returns the rewritten
/// s-expr. A literal fuller cannot parse back is left verbatim (the oracle then
/// simply may not fold it — a sound miss, never a crash).
pub fn snap_constants(sexpr: &str) -> String {
    let bytes = sexpr.as_bytes();
    let mut out = String::with_capacity(sexpr.len());
    let mut i = 0;
    while i < bytes.len() {
        // Look for the token "(Num " and rewrite the number that follows.
        if sexpr[i..].starts_with("(Num ") {
            out.push_str("(Num ");
            let mut j = i + 5;
            // The numeric literal runs until the closing paren of this Num.
            let start = j;
            while j < bytes.len() && bytes[j] != b')' {
                j += 1;
            }
            let lit = sexpr[start..j].trim();
            match lit.parse::<f64>() {
                Ok(f) => {
                    let snapped = if f.abs() < 1e-4 { 0.0 } else { (f * 1000.0).round() / 1000.0 };
                    // Emit with a decimal point so egglog reads it as f64.
                    if snapped == snapped.trunc() {
                        out.push_str(&format!("{snapped:.1}"));
                    } else {
                        out.push_str(&format!("{snapped}"));
                    }
                }
                Err(_) => out.push_str(lit),
            }
            i = j; // continue at the ')'
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// The oracle ruleset: algebra + powers + rational, plus pure comm/assoc for
/// Add/Mul (reordering only — NO distributivity, which would explode with
/// rational), plus literal division folds that no corpus ruleset carries
/// (`Div`/`ProtectedDiv` of two `Num`s -> the folded `Num`, and `x/x -> 1` under
/// the nonzero guard). `is-num` marks any e-class holding a numeric literal, so
/// the verdict is a single `check`.
fn oracle_program() -> String {
    // Pure reordering: comm + assoc for Add and Mul. Bounded (fixed multiset).
    const COMM_ASSOC: &str = r#"
(ruleset commassoc)
(rewrite (Add a b) (Add b a) :ruleset commassoc)
(rewrite (Mul a b) (Mul b a) :ruleset commassoc)
(rewrite (Add (Add a b) c) (Add a (Add b c)) :ruleset commassoc)
(rewrite (Add a (Add b c)) (Add (Add a b) c) :ruleset commassoc)
(rewrite (Mul (Mul a b) c) (Mul a (Mul b c)) :ruleset commassoc)
(rewrite (Mul a (Mul b c)) (Mul (Mul a b) c) :ruleset commassoc)
"#;
    // Literal folds + guarded self-cancellation that the corpus families omit.
    const ORACLE_FOLDS: &str = r#"
(ruleset oraclefolds)
; division of two literals folds to a literal (b != 0).
(rewrite (Div (Num a) (Num b)) (Num (/ a b)) :when ((!= b 0.0)) :ruleset oraclefolds)
(rewrite (ProtectedDiv (Num a) (Num b)) (Num (/ a b)) :when ((!= b 0.0)) :ruleset oraclefolds)
; x / x = 1 and x /prot x = 1 where x is known nonzero (assumed for all vars).
(rewrite (Div x x) (Num 1.0) :when ((is-nonzero x)) :ruleset oraclefolds)
(rewrite (ProtectedDiv x x) (Num 1.0) :when ((is-nonzero x)) :ruleset oraclefolds)
; a / (a/b) style and (Mul k (Div a k)) reduce via assoc + these; also fold
; Mul/Div inverse: (Mul a (Div b a)) -> b and (Div (Mul a b) a) -> b, guarded.
(rewrite (Mul k (Div a k)) a :when ((is-nonzero k)) :ruleset oraclefolds)
(rewrite (Mul k (ProtectedDiv a k)) a :when ((is-nonzero k)) :ruleset oraclefolds)
(rewrite (Mul (Div a k) k) a :when ((is-nonzero k)) :ruleset oraclefolds)
(rewrite (Mul (ProtectedDiv a k) k) a :when ((is-nonzero k)) :ruleset oraclefolds)
; (k*g)/g = k and g/(k*g)-inverse: the SCALE-constant fold. With comm/assoc the
; numerator reorders so the shared factor g sits where these match.
(rewrite (Div (Mul k g) g) k :when ((is-nonzero g)) :ruleset oraclefolds)
(rewrite (ProtectedDiv (Mul k g) g) k :when ((is-nonzero g)) :ruleset oraclefolds)
; g/g inside a product cancels: (Mul (Div g g) k) handled by x/x=1 above, but a
; bare shared factor in num & denom: (Div (Mul a b) (Mul a c)) = b/c, guarded.
(rewrite (Div (Mul a b) (Mul a c)) (Div b c) :when ((is-nonzero a)) :ruleset oraclefolds)
(rewrite (ProtectedDiv (Mul a b) (Mul a c)) (ProtectedDiv b c) :when ((is-nonzero a)) :ruleset oraclefolds)
; pull a literal coefficient OUT of a quotient numerator: (c*x)/y = c*(x/y), so
; the sqrt-quotient rule below can see the two bare roots. Size-neutral.
(rewrite (ProtectedDiv (Mul (Num c) x) y) (Mul (Num c) (ProtectedDiv x y)) :ruleset oraclefolds)
(rewrite (Div (Mul (Num c) x) y) (Mul (Num c) (Div x y)) :ruleset oraclefolds)
; ---- ProtectedSqrt QUOTIENT (radsimp), CONTRACTIVE: sqrt|a|/sqrt|b| =
; sqrt|a/b| (two roots -> one; fires only on a Div of two roots, cannot loop).
(rewrite (ProtectedDiv (ProtectedSqrt a) (ProtectedSqrt b)) (ProtectedSqrt (ProtectedDiv a b)) :ruleset oraclefolds)
; (x / k) / x = 1/k, CONTRACTIVE, leaves the coefficient OUTSIDE the sqrt so no
; expanding c*sqrt|x| -> sqrt(c^2 x) is needed (that one blew up: 99 -> 9801 ...).
; Euclidean: 1.414*sqrt((S/2)/S) -> 1.414*sqrt(1/2), a constant.
(rewrite (ProtectedDiv (ProtectedDiv x (Num k)) x) (Num (/ 1.0 k)) :when ((is-nonzero x) (!= k 0.0)) :ruleset oraclefolds)
; sum of squares with a nonzero term is nonzero (SOUND: both terms >= 0). Unlocks
; the cancel over a compound denominator S = (x2-x3)^2 + (x1-x0)^2 (only Vars are
; asserted nonzero by the caller). NO general `Add a b` rule — x + (-x) = 0.
(rule ((is-nonzero a) (= e (Add (Pow2 a) (Pow2 b)))) ((is-nonzero e)) :ruleset oraclefolds)
(rule ((is-nonzero b) (= e (Add (Pow2 a) (Pow2 b)))) ((is-nonzero e)) :ruleset oraclefolds)
; nonzero flows THROUGH a root/inv/abs to the argument (sqrt|x| != 0 => x != 0),
; so asserting the truth (denominator) nonzero reaches a compound inside it.
(rule ((is-nonzero e) (= e (ProtectedSqrt x))) ((is-nonzero x)) :ruleset oraclefolds)
(rule ((is-nonzero e) (= e (Sqrt x))) ((is-nonzero x)) :ruleset oraclefolds)
(rule ((is-nonzero e) (= e (Abs x))) ((is-nonzero x)) :ruleset oraclefolds)
(rule ((is-nonzero e) (= e (ProtectedInv x))) ((is-nonzero x)) :ruleset oraclefolds)
; ---- is-const: the VERDICT. A Num is const, and constness PROPAGATES through
; the ops egglog cannot fold numerically (it has f64 * + - but NOT sqrt), so
; sqrt(0.5) is const without computing its value — SRBench's frac.is_constant().
(relation is-const (Math))
(rule ((= e (Num a))) ((is-const e)) :ruleset oraclefolds)
(rule ((= e (ProtectedSqrt x)) (is-const x)) ((is-const e)) :ruleset oraclefolds)
(rule ((= e (Neg x)) (is-const x)) ((is-const e)) :ruleset oraclefolds)
(rule ((= e (Abs x)) (is-const x)) ((is-const e)) :ruleset oraclefolds)
(rule ((= e (Mul a b)) (is-const a) (is-const b)) ((is-const e)) :ruleset oraclefolds)
"#;
    format!(
        "{MATH_DATATYPE}\n{GUARD_RELATIONS}\n\
         {ALGEBRA_RULESET}\n{POWERS_RULESET}\n{RATIONAL_RULESET}\n\
         {COMM_ASSOC}\n{ORACLE_FOLDS}\n\
         (unstable-combined-ruleset oracle \
            guards algebra powers rational commassoc oraclefolds)"
    )
}

/// SRBench's symbolic-solution verdict for `model` against `truth`, both fuller
/// Math s-expressions over the same variables. `vars` are the variable names to
/// assume nonzero (a scale/offset constant is only defined where denominators
/// are nonzero). Returns `Ok(true)` iff, after `round_floats`, the difference OR
/// the ratio folds to a numeric constant — SRBench's exact criterion.
pub fn srbench_equivalent(model: &str, truth: &str, vars: &[String]) -> Result<bool, String> {
    // SIZE GUARD, FIRST. The comm/assoc + rational rules reorder a fixed operand
    // multiset — bounded, but COMBINATORIALLY LARGE on a deep tower with repeated
    // subterms. A 30+ node fitter tower blew the e-graph to 29 GB and took the
    // machine down. So: refuse anything past a hard token budget and return
    // Ok(false) — "not proven equal", the sound outcome — rather than build the
    // graph. Measured: real laws are < 60 tokens; the towers are the pathological
    // input, and they are exactly the ones this oracle should NOT chew on.
    const MAX_TOKENS: usize = 400;
    let toks = model.len().max(truth.len());
    if toks > MAX_TOKENS * 8 || model.matches('(').count() + truth.matches('(').count() > MAX_TOKENS {
        return Ok(false);
    }
    // round_floats both sides (SRBench rounds truth too — pi -> 3.142, etc.).
    let m = snap_constants(model);
    let t = snap_constants(truth);
    let asserts: String = vars.iter().map(|v| format!("(is-nonzero (Var \"{v}\"))\n")).collect();

    // A modest bounded iteration count: high enough to reorder+fold the products
    // these laws carry, low enough that comm/assoc stays small. 10 is the empirical
    // ceiling before the reordering set grows without adding reach on these forms.
    const ITERS: u32 = 10;

    let program = oracle_program();
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| format!("load oracle rulesets: {e}"))?;

    // Build the difference and the ratio, run the oracle, then ask whether
    // EITHER e-class is const. Two checks, OR'd. Assert the TRUTH (the ratio's
    // denominator) nonzero: model/truth == const is only a meaningful question
    // where truth != 0, and SRBench's symbolic frac.is_constant() makes the same
    // generic-nonzero assumption. This unlocks cancellation over a COMPOUND
    // denominator (e.g. S = sum of squares) that the per-Var asserts cannot reach.
    let prog = format!(
        "(let __diff (Sub {m} {t}))\n\
         (let __frac (ProtectedDiv {m} {t}))\n\
         {asserts}(is-nonzero {t})\n\
         (run-schedule (repeat {ITERS} (run oracle)))\n"
    );
    egraph
        .parse_and_run_program(None, &prog)
        .map_err(|e| format!("run oracle on {model:?}: {e}"))?;

    let diff_num = check_holds(&mut egraph, "(check (is-const __diff))");
    let frac_num = check_holds(&mut egraph, "(check (is-const __frac))");
    Ok(diff_num || frac_num)
}

/// Run a single `(check ...)` and report whether it passed (true) or was a
/// clean CheckError (false). Any other egglog error is treated as "not proven"
/// — the oracle is a lower bound, never a crash source.
fn check_holds(egraph: &mut EGraph, check: &str) -> bool {
    egraph.parse_and_run_program(None, check).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("x_{i}")).collect()
    }

    #[test]
    fn snap_zeroes_tiny_and_rounds_the_rest() {
        let s = r#"(Add (Mul (Num 99.00000014258512) (Var "x_0")) (Num -4.0370005116585617e-8))"#;
        let out = snap_constants(s);
        assert!(out.contains("(Num 99.0)"), "rounded 99.0000..: {out}");
        assert!(out.contains("(Num 0.0)"), "tiny -> 0: {out}");
        assert!(!out.contains("4.037"), "junk gone: {out}");
    }

    #[test]
    fn snap_rounds_to_three_places() {
        assert!(snap_constants(r#"(Num 0.079577472)"#).contains("0.08"));
        assert!(snap_constants(r#"(Num 3.1415926535)"#).contains("3.142"));
    }

    #[test]
    fn i_18_14_raw_winner_is_the_law() {
        // The actual RAW_MATH winner, UNROUNDED, vs truth m*r*v*sin(theta) =
        // x0*x1*x2*sin(x3). round_floats + cancellation must prove it.
        let model = r#"(Add (Mul (Num 99.00000014258512) (Mul (Mul (Var "x_0") (Mul (Var "x_2") (Sin (Var "x_3")))) (ProtectedDiv (Var "x_1") (Num 99.0)))) (Num -4.0370005116585617e-8))"#;
        let truth = r#"(Mul (Mul (Mul (Var "x_0") (Var "x_1")) (Var "x_2")) (Sin (Var "x_3")))"#;
        assert!(srbench_equivalent(model, truth, &vars(4)).unwrap(), "I_18_14 winner IS the law");
    }

    #[test]
    fn scale_constant_counts_as_solution() {
        // A model that is 2 * truth: ratio is the constant 2 -> SRBench PASS.
        let model = r#"(Mul (Num 2.0) (Mul (Var "x_0") (Var "x_1")))"#;
        let truth = r#"(Mul (Var "x_0") (Var "x_1"))"#;
        assert!(srbench_equivalent(model, truth, &vars(2)).unwrap(), "k*truth counts");
    }

    #[test]
    fn offset_constant_counts_as_solution() {
        // truth + 5: difference is the constant -5 -> SRBench PASS.
        let model = r#"(Add (Mul (Var "x_0") (Var "x_1")) (Num 5.0))"#;
        let truth = r#"(Mul (Var "x_0") (Var "x_1"))"#;
        assert!(srbench_equivalent(model, truth, &vars(2)).unwrap(), "truth+c counts");
    }

    #[test]
    fn euclidean_distance_sqrt_folds() {
        // I_8_14 tidied: 1.4142*sqrt(S/2) == sqrt(S). Coefficient pulls out of the
        // quotient, roots merge (sqrt-quotient), (S/2)/S -> 1/2, ratio = 1.414*sqrt(1/2),
        // a constant. radsimp — sympy PASSes it, the Algebra check could not.
        let s = r#"(Add (Pow2 (Sub (Var "x_2") (Var "x_3"))) (Pow2 (Sub (Var "x_1") (Var "x_0"))))"#;
        let model = format!(r#"(Mul (Num 1.4142135638900017) (ProtectedSqrt (ProtectedDiv {s} (Num 2.0))))"#);
        let truth = format!(r#"(ProtectedSqrt {s})"#);
        assert!(srbench_equivalent(&model, &truth, &vars(4)).unwrap(), "sqrt(2)*sqrt(S/2) == sqrt(S)");
    }

    #[test]
    fn sqrt_of_sum_is_not_sum_of_sqrts() {
        let model = r#"(ProtectedSqrt (Add (Var "x_0") (Var "x_1")))"#;
        let truth = r#"(Add (ProtectedSqrt (Var "x_0")) (ProtectedSqrt (Var "x_1")))"#;
        assert!(!srbench_equivalent(model, truth, &vars(2)).unwrap(), "sqrt(a+b) != sqrt a + sqrt b");
    }

    #[test]
    fn a_genuine_non_law_is_rejected() {
        // x_0 * x_0 is not x_0 * x_1: neither diff nor ratio is constant.
        let model = r#"(Mul (Var "x_0") (Var "x_0"))"#;
        let truth = r#"(Mul (Var "x_0") (Var "x_1"))"#;
        assert!(!srbench_equivalent(model, truth, &vars(2)).unwrap(), "x0^2 != x0*x1");
    }

    // ---- SOUNDNESS GUARDS: the oracle must NOT over-fire. Nonzero is assumed
    // for every var, but nonzero is NOT positive, so Abs must not be shed; and a
    // non-constant offset / a swapped ratio must stay rejected.

    #[test]
    fn abs_is_not_shed_under_nonzero_only() {
        // |x_0| == x_0 is FALSE without a positivity assumption (nonzero alone
        // does not imply positive). The oracle assumes only nonzero, so it must
        // NOT prove this — that would be the exact leniency to fear.
        let model = r#"(Abs (Var "x_0"))"#;
        let truth = r#"(Var "x_0")"#;
        assert!(!srbench_equivalent(model, truth, &vars(1)).unwrap(), "Abs(x) != x under nonzero-only");
    }

    #[test]
    fn a_variable_offset_is_rejected() {
        // x_0*x_1 + x_2 differs from x_0*x_1 by x_2, NOT a constant.
        let model = r#"(Add (Mul (Var "x_0") (Var "x_1")) (Var "x_2"))"#;
        let truth = r#"(Mul (Var "x_0") (Var "x_1"))"#;
        assert!(!srbench_equivalent(model, truth, &vars(3)).unwrap(), "+x_2 is not a constant offset");
    }

    #[test]
    fn a_swapped_ratio_is_rejected() {
        // x_0/x_1 vs x_1/x_0: ratio is (x_0/x_1)^2, not a constant.
        let model = r#"(ProtectedDiv (Var "x_0") (Var "x_1"))"#;
        let truth = r#"(ProtectedDiv (Var "x_1") (Var "x_0"))"#;
        assert!(!srbench_equivalent(model, truth, &vars(2)).unwrap(), "x0/x1 != x1/x0");
    }
}
