//! Power / exponential / logarithm identities, transcribed from SymPy
//! (`powsimp.py`, `core/power.py`, `functions/elementary/exponential.py`).
//!
//! These are the rules that were previously inexpressible: the general
//! `Pow(base, exp)` constructor and the `is-positive` domain guard (both added
//! to `expr.rs`) unlock the powsimp and log/exp families. Real domain only.
//!
//! Requires `GUARD_RELATIONS` to be loaded (for `is-positive` / `is-nonzero`).

/// The `powers` ruleset.
pub const POWERS_RULESET: &str = r#"
(ruleset powers)

; ---- Pow basics (SymPy core/power.py Pow.eval) ----
; x^0 = 1
(rewrite (Pow x (Num 0.0)) (Num 1.0) :ruleset powers)
; x^1 = x
(rewrite (Pow x (Num 1.0)) x :ruleset powers)
; x^2 and x^3 normalise to the dedicated constructors (cheaper, evaluable)
(rewrite (Pow x (Num 2.0)) (Pow2 x) :ruleset powers)
(rewrite (Pow x (Num 3.0)) (Pow3 x) :ruleset powers)
; 1^x = 1
(rewrite (Pow (Num 1.0) x) (Num 1.0) :ruleset powers)

; ---- exponent arithmetic (SymPy powsimp.py: x^a * x^b = x^(a+b)) ----
; CAUTION: these are UNSOUND to run unbounded in a pure-rewrite e-graph.
; egglog does not constant-fold the exponent, so `(Add a b)` on the RHS builds
; ever-taller unevaluated Num towers — combined with `x^2 -> Pow2 x` etc. this
; never reaches fixpoint and the e-graph grows without bound (it pegged the CPU
; in testing). They are therefore DISABLED until exponent constant-folding
; lands (a Phase-1.4-style evaluator pass that folds `(Add 2.0 1.0) -> 3.0`
; before re-applying), at which point they terminate. Tracked, not shipped.
;
; (rewrite (Mul (Pow x a) (Pow x b)) (Pow x (Add a b)) :ruleset powers)
; (rewrite (Mul (Pow x a) x)         (Pow x (Add a (Num 1.0))) :ruleset powers)
; (rewrite (Mul x (Pow x a))         (Pow x (Add a (Num 1.0))) :ruleset powers)
; (rewrite (Pow (Pow x a) b)         (Pow x (Mul a b)) :ruleset powers)
; (rewrite (Div (Pow x a) (Pow x b)) (Pow x (Sub a b)) :ruleset powers)

; ---- log / exp (SymPy exponential.py); GUARDED on positivity ----
; log(exp x) = x  — sound for all real x (exp x > 0, log defined)
(rewrite (Log (Exp x)) x :ruleset powers)
; exp(log x) = x  — only where x > 0
(rewrite (Exp (Log x)) x :when ((is-positive x)) :ruleset powers)
; log(a*b) = log a + log b  — only where a > 0 and b > 0
(rewrite (Log (Mul a b)) (Add (Log a) (Log b))
    :when ((is-positive a) (is-positive b)) :ruleset powers)
; log(x^n) = n * log x  — only where x > 0
(rewrite (Log (Pow x n)) (Mul n (Log x)) :when ((is-positive x)) :ruleset powers)
; exp(a) * exp(b) = exp(a+b)  — sound for all reals
(rewrite (Mul (Exp a) (Exp b)) (Exp (Add a b)) :ruleset powers)
"#;

#[cfg(test)]
mod tests {
    use super::POWERS_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::prelude::exprs;
    use egglog::EGraph;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        e.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        e.parse_and_run_program(None, POWERS_RULESET).unwrap();
        e
    }

    // Hard safety cap: bound saturation to a fixed iteration count rather than
    // unbounded `(saturate ...)`. A divergent rule can then never grow the
    // e-graph without limit / peg the machine — it just stops at the cap. All
    // shipped power rules reach fixpoint well within this bound.
    const SAT_ITERS: u32 = 30;

    fn simplify(input: &str, extra_facts: &str) -> String {
        let mut e = egraph();
        e.parse_and_run_program(
            None,
            &format!(
                "(let __r {input})\n{extra_facts}\n(run-schedule (repeat {SAT_ITERS} (run powers)))"
            ),
        )
        .unwrap();
        let (sort, value) = e.eval_expr(&exprs::var("__r")).unwrap();
        e.extract_value_to_string(&sort, value).unwrap().0
    }

    /// Soundness: rewritten form numerically matches input on random real
    /// points where both are finite.
    fn assert_sound(input: &str, extra_facts: &str, points: &[(&str, f64)]) {
        let mut e = egraph();
        e.parse_and_run_program(None, &format!("(let __a {input})\n{extra_facts}")).unwrap();
        let (s0, v0) = e.eval_expr(&exprs::var("__a")).unwrap();
        let (td0, t0, _) = e.extract_value(&s0, v0).unwrap();
        let before = eval_term(&td0, t0, &|n: &str| points.iter().find(|(k, _)| *k == n).map(|(_, v)| *v));

        let mut e2 = egraph();
        e2.parse_and_run_program(
            None,
            &format!("(let __b {input})\n{extra_facts}\n(run-schedule (repeat {SAT_ITERS} (run powers)))"),
        )
        .unwrap();
        let (s1, v1) = e2.eval_expr(&exprs::var("__b")).unwrap();
        let (td1, t1, _) = e2.extract_value(&s1, v1).unwrap();
        let after = eval_term(&td1, t1, &|n: &str| points.iter().find(|(k, _)| *k == n).map(|(_, v)| *v));

        if let (Ok(a), Ok(b)) = (before, after) {
            if a.is_finite() && b.is_finite() {
                assert!((a - b).abs() <= 1e-9 * (a.abs() + 1.0), "unsound: {input} -> {a} vs {b}");
            }
        }
    }

    #[test]
    fn pow_basics() {
        assert_eq!(simplify(r#"(Pow (Var "x") (Num 0.0))"#, ""), "(Num 1.0)");
        assert_eq!(simplify(r#"(Pow (Var "x") (Num 1.0))"#, ""), r#"(Var "x")"#);
        assert_eq!(simplify(r#"(Pow (Var "x") (Num 2.0))"#, ""), r#"(Pow2 (Var "x"))"#);
    }

    #[test]
    fn log_exp_unguarded_sound() {
        // log(exp x) = x  for all reals
        assert_eq!(simplify(r#"(Log (Exp (Var "x")))"#, ""), r#"(Var "x")"#);
        assert_sound(r#"(Log (Exp (Var "x")))"#, "", &[("x", -3.2)]);
        // exp(a)*exp(b) = exp(a+b)
        assert_sound(r#"(Mul (Exp (Var "a")) (Exp (Var "b")))"#, "", &[("a", 0.5), ("b", -1.1)]);
    }

    #[test]
    fn log_exp_guarded_fires_with_positivity() {
        // exp(log x) = x only fires when (is-positive x) is asserted.
        let with_guard = simplify(r#"(Exp (Log (Var "x")))"#, r#"(is-positive (Var "x"))"#);
        assert_eq!(with_guard, r#"(Var "x")"#, "guarded rule should fire");
        // sound on a positive sample
        assert_sound(r#"(Exp (Log (Var "x")))"#, r#"(is-positive (Var "x"))"#, &[("x", 2.5)]);
    }

    #[test]
    fn log_guarded_does_not_fire_without_positivity() {
        // Without the positivity fact, exp(log x) must NOT collapse (would be
        // unsound for x <= 0). It should remain as-is.
        let no_guard = simplify(r#"(Exp (Log (Var "x")))"#, "");
        assert_eq!(no_guard, r#"(Exp (Log (Var "x")))"#, "must not fire unguarded");
    }
}
