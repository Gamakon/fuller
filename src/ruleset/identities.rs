//! Phase 1.2: pure-algebra identity rules over the `Math` datatype.
//!
//! These are the BRIEF.md rules 1-5: directional rewrites that always shrink
//! or preserve, safe to run inside saturation. NO bare commutativity or
//! associativity rewrites — egglog handles operand symmetry via e-class
//! merging, and encoding it as a rewrite blows up saturation (the trap the
//! Phase 1.0 calibration surfaced). Because we don't assume commutativity, the
//! identity rules are written for BOTH operand orders explicitly where the
//! literal can sit on either side.
//!
//! Each rule is stated as its math identity, then the egglog `rewrite`.

/// The `algebra` ruleset: rules 1-5 from BRIEF.md.
pub const ALGEBRA_RULESET: &str = r#"
(ruleset algebra)

; ---- Rule 2: multiplicative identity  (x * 1) = x , (1 * x) = x ----
(rewrite (Mul x (Num 1.0)) x :ruleset algebra)
(rewrite (Mul (Num 1.0) x) x :ruleset algebra)

; ---- Rule 3: additive identity  (x + 0) = x , (0 + x) = x ----
(rewrite (Add x (Num 0.0)) x :ruleset algebra)
(rewrite (Add (Num 0.0) x) x :ruleset algebra)
; and subtractive identity  (x - 0) = x
(rewrite (Sub x (Num 0.0)) x :ruleset algebra)

; ---- Rule 4: multiplicative zero  (x * 0) = 0 , (0 * x) = 0 ----
(rewrite (Mul x (Num 0.0)) (Num 0.0) :ruleset algebra)
(rewrite (Mul (Num 0.0) x) (Num 0.0) :ruleset algebra)

; ---- Rule 5: same-op nest collapse ----
; double negation  -(-x) = x
(rewrite (Neg (Neg x)) x :ruleset algebra)
; idempotent abs  |(|x|)| = |x|
(rewrite (Abs (Abs x)) (Abs x) :ruleset algebra)
; sqrt of square  sqrt(x^2) = |x|   (real domain)
(rewrite (Sqrt (Pow2 x)) (Abs x) :ruleset algebra)

; ---- Additive / subtractive cancellation (Category 1, sound, no guard) ----
; x - x = 0
(rewrite (Sub x x) (Num 0.0) :ruleset algebra)
; (a + b) - b = a ; (b + a) - b = a
(rewrite (Sub (Add a b) b) a :ruleset algebra)
(rewrite (Sub (Add b a) b) a :ruleset algebra)
; (a - b) + b = a ; b + (a - b) = a
(rewrite (Add (Sub a b) b) a :ruleset algebra)
(rewrite (Add b (Sub a b)) a :ruleset algebra)

; ---- Argument-at-zero / unity folds (Category 1, sound) ----
(rewrite (Sin (Num 0.0)) (Num 0.0) :ruleset algebra)
(rewrite (Cos (Num 0.0)) (Num 1.0) :ruleset algebra)
(rewrite (Tan (Num 0.0)) (Num 0.0) :ruleset algebra)
(rewrite (Tanh (Num 0.0)) (Num 0.0) :ruleset algebra)
(rewrite (Exp (Num 0.0)) (Num 1.0) :ruleset algebra)
(rewrite (Log (Num 1.0)) (Num 0.0) :ruleset algebra)

; ---- Abs under a proven-positive argument (Category 2, guarded) ----
; |x| = x for x > 0. The guard is caller-supplied domain knowledge (e.g. the
; SR engine's var_ranges) or derived (Exp positivity, positive^p, ...). This
; is the rule that turns Abs(a^(3/2)) into a^(3/2) once `a` is known positive
; — Abs wrappers from protected-sqrt chains are the most common reason an
; R^2=1.0 discovery fails SRBench's exact symbolic-solution check.
(rewrite (Abs x) x :when ((is-positive x)) :ruleset algebra)

; ---- Cancellation on RAW div only (Category 2, guarded; NEVER protected) ----
; x / x = 1  (raw Div, x != 0). protected_div(x,x) is 0 at x=0 -> excluded.
(rewrite (Div x x) (Num 1.0) :when ((is-nonzero x)) :ruleset algebra)
; (a * b) / b = a  (raw Div, b != 0). Both operand orders of the product.
(rewrite (Div (Mul a b) b) a :when ((is-nonzero b)) :ruleset algebra)
(rewrite (Div (Mul b a) b) a :when ((is-nonzero b)) :ruleset algebra)

; ---- Protected-op rules (the protected variants are otherwise INERT) ----
; protected_sqrt(x^2) = sqrt(|x^2|) = |x|  — sound: |x^2| = x^2, sqrt(x^2)=|x|.
(rewrite (ProtectedSqrt (Pow2 x)) (Abs x) :ruleset algebra)
; NOTE: no other protected rule fires. In particular NONE of the div/inv
; cancellation rules lift onto ProtectedDiv/ProtectedInv — e.g.
; protected_div(x,x) is 0 at x=0 (not 1), and protected_inv(0)=1 (not undefined),
; so the raw guarded rules would be unsound on the protected forms.
"#;

// Note: Rule 1 (constant folding, e.g. cos(0) -> 1) is data/value driven and
// lands with the evaluator in Phase 1.3-1.4, not as a structural rewrite here.

#[cfg(test)]
mod tests {
    use super::ALGEBRA_RULESET;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::prelude::exprs;
    use egglog::EGraph;

    /// Load Math + guards + algebra, insert `input`, saturate, extract the
    /// lowest-cost form as a string.
    fn simplify(input: &str) -> Result<String, String> {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(None, MATH_DATATYPE)
            .map_err(|e| format!("datatype: {e}"))?;
        egraph
            .parse_and_run_program(None, GUARD_RELATIONS)
            .map_err(|e| format!("guards: {e}"))?;
        egraph
            .parse_and_run_program(None, ALGEBRA_RULESET)
            .map_err(|e| format!("ruleset: {e}"))?;
        egraph
            .parse_and_run_program(
                None,
                &format!("(let __r {input})\n(run-schedule (saturate (run guards) (run algebra)))"),
            )
            .map_err(|e| format!("insert/saturate {input:?}: {e}"))?;
        let (sort, value) = egraph
            .eval_expr(&exprs::var("__r"))
            .map_err(|e| format!("eval: {e}"))?;
        let (best, _cost) = egraph
            .extract_value_to_string(&sort, value)
            .map_err(|e| format!("extract: {e}"))?;
        Ok(best)
    }

    /// Like `simplify` but asserts extra guard facts (e.g. `(is-nonzero ...)`)
    /// before saturating — used to test that guarded rules fire when their
    /// precondition is known.
    fn simplify_with_facts(input: &str, facts: &str) -> Result<String, String> {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, MATH_DATATYPE).map_err(|e| format!("datatype: {e}"))?;
        egraph.parse_and_run_program(None, GUARD_RELATIONS).map_err(|e| format!("guards: {e}"))?;
        egraph.parse_and_run_program(None, ALGEBRA_RULESET).map_err(|e| format!("ruleset: {e}"))?;
        egraph
            .parse_and_run_program(
                None,
                &format!(
                    "(let __r {input})\n{facts}\n(run-schedule (saturate (run guards) (run algebra)))"
                ),
            )
            .map_err(|e| format!("insert/saturate {input:?}: {e}"))?;
        let (sort, value) = egraph.eval_expr(&exprs::var("__r")).map_err(|e| format!("eval: {e}"))?;
        let (best, _cost) = egraph.extract_value_to_string(&sort, value).map_err(|e| format!("extract: {e}"))?;
        Ok(best)
    }

    /// Each rule fires on its target pattern.
    #[test]
    fn each_rule_fires() {
        let cases: &[(&str, &str)] = &[
            // mul identity, both orders
            (r#"(Mul (Var "x") (Num 1.0))"#, r#"(Var "x")"#),
            (r#"(Mul (Num 1.0) (Var "x"))"#, r#"(Var "x")"#),
            // add / sub identity
            (r#"(Add (Var "x") (Num 0.0))"#, r#"(Var "x")"#),
            (r#"(Add (Num 0.0) (Var "x"))"#, r#"(Var "x")"#),
            (r#"(Sub (Var "x") (Num 0.0))"#, r#"(Var "x")"#),
            // mul zero, both orders
            (r#"(Mul (Var "x") (Num 0.0))"#, "(Num 0.0)"),
            (r#"(Mul (Num 0.0) (Var "x"))"#, "(Num 0.0)"),
            // same-op collapse
            (r#"(Neg (Neg (Var "x")))"#, r#"(Var "x")"#),
            (r#"(Abs (Abs (Var "x")))"#, r#"(Abs (Var "x"))"#),
            (r#"(Sqrt (Pow2 (Var "x")))"#, r#"(Abs (Var "x"))"#),
        ];
        let mut failures = Vec::new();
        for (input, expected) in cases {
            match simplify(input) {
                Ok(got) if got == *expected => {}
                Ok(got) => failures.push(format!("{input}\n  got:      {got}\n  expected: {expected}")),
                Err(e) => failures.push(format!("{input}\n  error: {e}")),
            }
        }
        assert!(failures.is_empty(), "{} failed:\n{}", failures.len(), failures.join("\n"));
    }

    /// Combined: a synthetic noisy chromosome collapses to `x`.
    /// add(mul(x, 1), mul(0, y)) -> x
    #[test]
    fn combined_noise_collapses() {
        let got = simplify(r#"(Add (Mul (Var "x") (Num 1.0)) (Mul (Num 0.0) (Var "y")))"#)
            .expect("simplify");
        assert_eq!(got, r#"(Var "x")"#);
    }

    /// Category 1: additive cancellation + zero/unity folds (sound, no guard).
    #[test]
    fn cancellation_and_folds() {
        let cases: &[(&str, &str)] = &[
            (r#"(Sub (Var "x") (Var "x"))"#, "(Num 0.0)"),
            (r#"(Sub (Add (Var "a") (Var "b")) (Var "b"))"#, r#"(Var "a")"#),
            (r#"(Add (Sub (Var "a") (Var "b")) (Var "b"))"#, r#"(Var "a")"#),
            (r#"(Sin (Num 0.0))"#, "(Num 0.0)"),
            (r#"(Cos (Num 0.0))"#, "(Num 1.0)"),
            (r#"(Exp (Num 0.0))"#, "(Num 1.0)"),
            (r#"(Log (Num 1.0))"#, "(Num 0.0)"),
            (r#"(Tanh (Num 0.0))"#, "(Num 0.0)"),
        ];
        for (input, expected) in cases {
            assert_eq!(simplify(input).unwrap(), *expected, "{input}");
        }
    }

    /// Category 2 soundness: raw Div cancellation fires ONLY when the
    /// denominator is provably nonzero; a bare variable (no nonzero proof) must
    /// NOT cancel.
    #[test]
    fn raw_div_cancellation_is_guarded() {
        // Bare x/x: x is not provably nonzero -> must NOT become 1.
        let bare = simplify(r#"(Div (Var "x") (Var "x"))"#).unwrap();
        assert_eq!(bare, r#"(Div (Var "x") (Var "x"))"#, "unguarded x/x must not fire");
    }

    /// When the denominator is asserted nonzero, raw Div cancellation fires.
    #[test]
    fn raw_div_cancels_when_nonzero_known() {
        let got = simplify_with_facts(
            r#"(Div (Var "x") (Var "x"))"#,
            r#"(is-nonzero (Var "x"))"#,
        )
        .unwrap();
        assert_eq!(got, "(Num 1.0)", "x/x should cancel once x is known nonzero");
    }

    /// Guarded Abs-shed: |x| = x fires ONLY with a positivity proof.
    #[test]
    fn abs_sheds_only_under_positivity() {
        // Bare |a|: no proof -> unchanged.
        let bare = simplify(r#"(Abs (Var "a"))"#).unwrap();
        assert_eq!(bare, r#"(Abs (Var "a"))"#, "unguarded Abs must not shed");
        // Asserted positive -> sheds.
        let shed = simplify_with_facts(r#"(Abs (Var "a"))"#, r#"(is-positive (Var "a"))"#).unwrap();
        assert_eq!(shed, r#"(Var "a")"#);
        // Derived positivity (Exp > 0 always) -> sheds with no assertion.
        let exp = simplify(r#"(Abs (Exp (Var "x")))"#).unwrap();
        assert_eq!(exp, r#"(Exp (Var "x"))"#);
    }

    /// Positivity propagates through Pow: is-positive(a) => is-positive(a^p),
    /// so Abs(Pow a p) sheds — the keplers3 Abs(a^(3/2)) shape.
    #[test]
    fn abs_of_pow_sheds_when_base_positive() {
        let got = simplify_with_facts(
            r#"(Abs (Pow (Var "a") (Num 1.5)))"#,
            r#"(is-positive (Var "a"))"#,
        )
        .unwrap();
        assert_eq!(got, r#"(Pow (Var "a") (Num 1.5))"#);
        // And WITHOUT the fact it must stay wrapped.
        let bare = simplify(r#"(Abs (Pow (Var "a") (Num 1.5)))"#).unwrap();
        assert_eq!(bare, r#"(Abs (Pow (Var "a") (Num 1.5)))"#);
    }

    /// Guard PROPAGATION must derive facts on its own — not only accept
    /// caller-asserted ones. `Exp x` is positive hence nonzero, so
    /// Exp(x)/Exp(x) cancels with NO assertion. This is the regression test
    /// for the wiring: the guards ruleset must actually run in the schedule
    /// (untagged rules land in egglog's default ruleset, which named runs
    /// never execute — the propagation was silently dead).
    #[test]
    fn guard_propagation_derives_exp_nonzero() {
        let got = simplify(r#"(Div (Exp (Var "x")) (Exp (Var "x")))"#).unwrap();
        assert_eq!(got, "(Num 1.0)", "exp(x)/exp(x) must cancel via derived positivity");
    }

    /// Literal-sign propagation: a nonzero numeric literal is provably
    /// nonzero, so constant/constant cancels with no assertion.
    #[test]
    fn guard_propagation_derives_literal_nonzero() {
        let got = simplify(r#"(Div (Num 2.0) (Num 2.0))"#).unwrap();
        assert_eq!(got, "(Num 1.0)");
        let neg = simplify(r#"(Div (Num -3.0) (Num -3.0))"#).unwrap();
        assert_eq!(neg, "(Num 1.0)");
    }

    /// The one sound protected rule fires: protected_sqrt(x^2) -> |x|.
    #[test]
    fn protected_sqrt_of_square_collapses() {
        let got = simplify(r#"(ProtectedSqrt (Pow2 (Var "x")))"#).expect("simplify");
        assert_eq!(got, r#"(Abs (Var "x"))"#);
    }

    /// Soundness floor: protected ops are INERT — no rule rewrites them into
    /// their raw counterparts or applies an unsound cancellation. These must
    /// come back unchanged (the raw versions WOULD rewrite, which is the bug
    /// we're preventing).
    #[test]
    fn protected_ops_are_inert() {
        let cases = [
            // protected_div(x,x) must NOT become 1 (it's 0 at x=0)
            r#"(ProtectedDiv (Var "x") (Var "x"))"#,
            // protected_inv(protected_inv x) must NOT collapse to x (inv(0)=1)
            r#"(ProtectedInv (ProtectedInv (Var "x")))"#,
            // protected_log(protected_exp x) must NOT collapse to x
            r#"(ProtectedLog (ProtectedExp (Var "x")))"#,
            // protected_sqrt(x) alone must NOT become Sqrt or Abs
            r#"(ProtectedSqrt (Var "x"))"#,
        ];
        for input in cases {
            let got = simplify(input).expect("simplify");
            assert_eq!(got, input, "protected op was rewritten (unsound!): {input}");
        }
    }
}
