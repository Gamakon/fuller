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
"#;

// Note: Rule 1 (constant folding, e.g. cos(0) -> 1) is data/value driven and
// lands with the evaluator in Phase 1.3-1.4, not as a structural rewrite here.

#[cfg(test)]
mod tests {
    use super::ALGEBRA_RULESET;
    use crate::expr::MATH_DATATYPE;
    use egglog::prelude::exprs;
    use egglog::EGraph;

    /// Load Math + algebra, insert `input`, saturate, extract the lowest-cost
    /// form as a string.
    fn simplify(input: &str) -> Result<String, String> {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(None, MATH_DATATYPE)
            .map_err(|e| format!("datatype: {e}"))?;
        egraph
            .parse_and_run_program(None, ALGEBRA_RULESET)
            .map_err(|e| format!("ruleset: {e}"))?;
        egraph
            .parse_and_run_program(
                None,
                &format!("(let __r {input})\n(run-schedule (saturate (run algebra)))"),
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
}
