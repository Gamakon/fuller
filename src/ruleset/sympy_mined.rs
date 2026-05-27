//! Identities mined line-by-line from SymPy's simplification source, covering
//! everything OUTSIDE the trig/fu family (which a sibling ruleset owns).
//!
//! Sources swept (under `sympy/`):
//!   * `functions/elementary/complexes.py` — `Abs.eval` / `sign.eval`
//!   * `functions/elementary/exponential.py` — `log.eval`, `log._eval_expand_log`,
//!     `exp.eval`
//!   * `functions/elementary/miscellaneous.py` — `sqrt` (= `Pow(_, 1/2)`)
//!   * `core/power.py` — `Pow.__new__` / `Pow._eval_power`
//!   * `simplify/powsimp.py` — base/exponent combination
//!   * `simplify/radsimp.py`, `simplify/ratsimp.py`, `simplify/simplify.py`,
//!     `simplify/sqrtdenest.py`
//!
//! Every rule below is a real-domain-sound identity, expressible in the `Math`
//! constructors, and NOT already present in identities/powers/distribute/
//! rational/trig/wide. Each carries its SymPy citation and a bounded unit test.
//!
//! ## Soundness discipline (the SKIPPED-for-soundness list, recorded here)
//! The following SymPy transforms were examined and DELIBERATELY NOT added
//! because they are unsound over the reals or need machinery we lack:
//!
//!   * `sqrt(x)*sqrt(y) -> sqrt(x*y)` UNGUARDED — `powsimp.py` itself warns this
//!     "is not true" without `force`/positivity (it fails on x,y<0). Added ONLY
//!     guarded on `(is-positive x)(is-positive y)`.
//!   * `(x^a)^b -> x^(a*b)` UNGUARDED — `power.py _eval_power`; false on the
//!     reals, e.g. `((-1)^2)^(1/2)=1` but `(-1)^1=-1`. Added ONLY guarded on
//!     `(is-positive x)`.
//!   * `log(a*b)=log a+log b`, `log(a/b)=log a-log b`, `log(x^n)=n log x`,
//!     `exp(log x)=x` — `exponential.py _eval_expand_log` gates every one on
//!     positivity (`force or x.is_positive`). The `log`-of-product/power forms
//!     already live (guarded) in `powers.rs`; we add ONLY the guarded
//!     `log(a/b)` and `exp(c·log x)` here.
//!   * `sqrt(x^2)=x` — FALSE; the real principal root is `|x|` (already covered
//!     as `(Abs x)` in `identities.rs`). Not re-added with the wrong RHS.
//!   * `sqrtdenest`: `sqrt(5+2 sqrt 6)=sqrt 2+sqrt 3` — needs perfect-square
//!     matching, not a fixed structural rewrite. Skipped.
//!   * `radsimp` denominator rationalisation (multiply by conjugate) — not a
//!     local rewrite; needs the conjugate constructed from the denominator.
//!     Skipped.
//!   * `Abs(x^n)=Abs(x)^n` for general n — only sound for the even/`Pow2` case
//!     over the reals; the even case is added, the general one skipped.
//!   * `x^a · y^a=(x·y)^a` (powsimp combine='base') — `force`-only in SymPy,
//!     unsound on negatives; the guarded positive form is the `sqrt`/`Pow`
//!     positive case, covered by the guarded `Sqrt` rule + `Pow`-positive rule.
//!
//! Requires `MATH_DATATYPE` and `GUARD_RELATIONS` to be loaded.

/// The `sympy_mined` ruleset.
pub const SYMPY_MINED_RULESET: &str = r#"
(ruleset sympy_mined)

; =====================================================================
; Abs identities  (complexes.py: Abs.eval)
; All real-domain-sound, no guard needed.
; =====================================================================
; Abs(-x) = Abs(x)             complexes.py Abs.eval: `Abs(-x) -> Abs(x)`
(rewrite (Abs (Neg x)) (Abs x) :ruleset sympy_mined)
; Abs(x^2) = x^2               complexes.py Abs.eval (even integer power, real
;                              x): x^2 >= 0, so |x^2| = x^2. Directional.
(rewrite (Abs (Pow2 x)) (Pow2 x) :ruleset sympy_mined)
; Abs(sqrt x) = sqrt x         miscellaneous.py sqrt = Pow(_,1/2); the real
;                              principal sqrt is >= 0 wherever defined.
(rewrite (Abs (Sqrt x)) (Sqrt x) :ruleset sympy_mined)
; Abs(exp x) = exp x           exponential.py / complexes.py: exp(real) > 0.
(rewrite (Abs (Exp x)) (Exp x) :ruleset sympy_mined)
; Abs(a*b) = Abs(a)*Abs(b)     complexes.py Abs.eval: Abs splits over a Mul.
(rewrite (Abs (Mul a b)) (Mul (Abs a) (Abs b)) :ruleset sympy_mined)
; Abs(a/b) = Abs(a)/Abs(b)     complexes.py Abs.eval: `cls(n)/cls(d)`. Both
;                              sides NaN where b = 0, so equal on the domain.
(rewrite (Abs (Div a b)) (Div (Abs a) (Abs b)) :ruleset sympy_mined)
; Abs(1/x) = 1/Abs(x)          the Inv special case of the Div rule above.
(rewrite (Abs (Inv x)) (Inv (Abs x)) :ruleset sympy_mined)
; Abs(Num k) = |k|             numeric fold (egglog f64 `abs` primitive).
(rewrite (Abs (Num k)) (Num (abs k)) :ruleset sympy_mined)

; =====================================================================
; log / exp identities  (exponential.py)
; Guarded on positivity — SymPy gates each on `force or x.is_positive`.
; (log(product), log(power), exp(log) already shipped guarded in powers.rs;
;  these two are the remaining guarded log/exp expansions.)
; =====================================================================
; log(a/b) = log a - log b     exponential.py _eval_expand_log (Rational/Mul
;                              branch, `log(p)-log(q)`). Guard: a>0, b>0.
(rewrite (Log (Div a b)) (Sub (Log a) (Log b))
    :when ((is-positive a) (is-positive b)) :ruleset sympy_mined)
; exp(c * log x) = x^c         exponential.py exp.eval (log_term**coeff). Sound
;                              wherever log x is defined, i.e. x > 0. Both
;                              operand orders of the product.
(rewrite (Exp (Mul c (Log x))) (Pow x c) :when ((is-positive x)) :ruleset sympy_mined)
(rewrite (Exp (Mul (Log x) c)) (Pow x c) :when ((is-positive x)) :ruleset sympy_mined)

; =====================================================================
; sqrt / Pow combination  (powsimp.py, power.py)
; UNSOUND unguarded over the reals; added ONLY on positivity.
; =====================================================================
; sqrt(a*b) = sqrt(a)*sqrt(b)  powsimp.py combine='base'. SymPy flags this as
;                              `force`-only (false for a,b<0). Guard: a>0, b>0.
;                              Directional (split a product root). Terminating.
(rewrite (Sqrt (Mul a b)) (Mul (Sqrt a) (Sqrt b))
    :when ((is-positive a) (is-positive b)) :ruleset sympy_mined)
; (x^a)^b = x^(a*b)            power.py _eval_power (`s*Pow(b, e*other)`). Sound
;                              only for x > 0 over the reals. Guard: x>0. The
;                              `(Mul a b)` exponent is folded by `distribute`
;                              when a,b are numeric; directional so terminating.
(rewrite (Pow (Pow x a) b) (Pow x (Mul a b)) :when ((is-positive x)) :ruleset sympy_mined)
"#;

#[cfg(test)]
mod tests {
    use super::SYMPY_MINED_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::prelude::exprs;
    use egglog::EGraph;

    // Hard safety cap (mirrors powers.rs / rational.rs): a divergent rule stops
    // at the cap instead of pegging the machine. Every shipped rule reaches
    // fixpoint well within this bound.
    const SAT_ITERS: u32 = 8;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        e.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        e.parse_and_run_program(None, SYMPY_MINED_RULESET).unwrap();
        e
    }

    /// Saturate (bounded) with optional extra guard facts; ask whether `a`/`b`
    /// land in the same e-class.
    fn proves_equal(a: &str, b: &str, facts: &str) -> bool {
        let mut e = egraph();
        let prog = format!(
            "(let __a {a})\n(let __b {b})\n{facts}\n\
             (run-schedule (repeat {SAT_ITERS} (run sympy_mined)))\n(check (= __a __b))"
        );
        e.parse_and_run_program(None, &prog).is_ok()
    }

    /// Numeric soundness on a random real point: the rewritten lowest-cost form
    /// must agree with the input where both are finite.
    fn assert_sound(input: &str, facts: &str, points: &[(&str, f64)]) {
        let lookup = |n: &str| points.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
        let mut e = egraph();
        e.parse_and_run_program(None, &format!("(let __a {input})\n{facts}")).unwrap();
        let (s0, v0) = e.eval_expr(&exprs::var("__a")).unwrap();
        let (td0, t0, _) = e.extract_value(&s0, v0).unwrap();
        let a = eval_term(&td0, t0, &lookup);

        let mut e2 = egraph();
        e2.parse_and_run_program(
            None,
            &format!("(let __b {input})\n{facts}\n(run-schedule (repeat {SAT_ITERS} (run sympy_mined)))"),
        )
        .unwrap();
        let (s1, v1) = e2.eval_expr(&exprs::var("__b")).unwrap();
        let (td1, t1, _) = e2.extract_value(&s1, v1).unwrap();
        let b = eval_term(&td1, t1, &lookup);

        if let (Ok(a), Ok(b)) = (a, b) {
            if a.is_finite() && b.is_finite() {
                assert!((a - b).abs() <= 1e-9 * (a.abs() + 1.0), "unsound: {input} -> {a} vs {b}");
            }
        }
    }

    #[test]
    fn abs_identities() {
        // Abs(-x) = Abs(x)
        assert!(proves_equal(r#"(Abs (Neg (Var "x")))"#, r#"(Abs (Var "x"))"#, ""));
        assert_sound(r#"(Abs (Neg (Var "x")))"#, "", &[("x", -2.3)]);
        // Abs(x^2) = x^2
        assert!(proves_equal(r#"(Abs (Pow2 (Var "x")))"#, r#"(Pow2 (Var "x"))"#, ""));
        assert_sound(r#"(Abs (Pow2 (Var "x")))"#, "", &[("x", -1.7)]);
        // Abs(sqrt x) = sqrt x
        assert!(proves_equal(r#"(Abs (Sqrt (Var "x")))"#, r#"(Sqrt (Var "x"))"#, ""));
        // Abs(exp x) = exp x
        assert!(proves_equal(r#"(Abs (Exp (Var "x")))"#, r#"(Exp (Var "x"))"#, ""));
        assert_sound(r#"(Abs (Exp (Var "x")))"#, "", &[("x", -3.0)]);
        // Abs(a*b) = Abs a * Abs b
        assert!(proves_equal(
            r#"(Abs (Mul (Var "a") (Var "b")))"#,
            r#"(Mul (Abs (Var "a")) (Abs (Var "b")))"#,
            "",
        ));
        assert_sound(r#"(Abs (Mul (Var "a") (Var "b")))"#, "", &[("a", -2.1), ("b", 3.4)]);
        // Abs(a/b) = Abs a / Abs b
        assert!(proves_equal(
            r#"(Abs (Div (Var "a") (Var "b")))"#,
            r#"(Div (Abs (Var "a")) (Abs (Var "b")))"#,
            "",
        ));
        // Abs(1/x) = 1/Abs x
        assert!(proves_equal(r#"(Abs (Inv (Var "x")))"#, r#"(Inv (Abs (Var "x")))"#, ""));
        // Abs(Num -7) = 7
        assert!(proves_equal(r#"(Abs (Num -7.0))"#, r#"(Num 7.0)"#, ""));
    }

    #[test]
    fn log_div_guarded() {
        // log(a/b) = log a - log b fires only when both are positive.
        assert!(proves_equal(
            r#"(Log (Div (Var "a") (Var "b")))"#,
            r#"(Sub (Log (Var "a")) (Log (Var "b")))"#,
            r#"(is-positive (Var "a")) (is-positive (Var "b"))"#,
        ));
        assert_sound(
            r#"(Log (Div (Var "a") (Var "b")))"#,
            r#"(is-positive (Var "a")) (is-positive (Var "b"))"#,
            &[("a", 2.3), ("b", 5.1)],
        );
    }

    #[test]
    fn log_div_does_not_fire_unguarded() {
        // Without positivity it must NOT fire (unsound on negatives).
        assert!(!proves_equal(
            r#"(Log (Div (Var "a") (Var "b")))"#,
            r#"(Sub (Log (Var "a")) (Log (Var "b")))"#,
            "",
        ));
    }

    #[test]
    fn exp_of_coeff_log_guarded() {
        // exp(c * log x) = x^c when x > 0, both operand orders.
        assert!(proves_equal(
            r#"(Exp (Mul (Num 2.0) (Log (Var "x"))))"#,
            r#"(Pow (Var "x") (Num 2.0))"#,
            r#"(is-positive (Var "x"))"#,
        ));
        assert!(proves_equal(
            r#"(Exp (Mul (Log (Var "x")) (Num 2.0)))"#,
            r#"(Pow (Var "x") (Num 2.0))"#,
            r#"(is-positive (Var "x"))"#,
        ));
        assert_sound(
            r#"(Exp (Mul (Num -1.7) (Log (Var "x"))))"#,
            r#"(is-positive (Var "x"))"#,
            &[("x", 3.2)],
        );
    }

    #[test]
    fn sqrt_of_product_guarded() {
        // sqrt(a*b) = sqrt a * sqrt b only when both positive.
        assert!(proves_equal(
            r#"(Sqrt (Mul (Var "a") (Var "b")))"#,
            r#"(Mul (Sqrt (Var "a")) (Sqrt (Var "b")))"#,
            r#"(is-positive (Var "a")) (is-positive (Var "b"))"#,
        ));
        assert_sound(
            r#"(Sqrt (Mul (Var "a") (Var "b")))"#,
            r#"(is-positive (Var "a")) (is-positive (Var "b"))"#,
            &[("a", 2.3), ("b", 5.1)],
        );
        // unguarded must NOT fire (false on negatives)
        assert!(!proves_equal(
            r#"(Sqrt (Mul (Var "a") (Var "b")))"#,
            r#"(Mul (Sqrt (Var "a")) (Sqrt (Var "b")))"#,
            "",
        ));
    }

    #[test]
    fn pow_of_pow_guarded() {
        // (x^a)^b = x^(a*b) when x > 0.
        assert!(proves_equal(
            r#"(Pow (Pow (Var "x") (Num 2.0)) (Num 3.0))"#,
            r#"(Pow (Var "x") (Mul (Num 2.0) (Num 3.0)))"#,
            r#"(is-positive (Var "x"))"#,
        ));
        assert_sound(
            r#"(Pow (Pow (Var "x") (Num 2.0)) (Num 3.0))"#,
            r#"(is-positive (Var "x"))"#,
            &[("x", 1.4)],
        );
        // unguarded must NOT fire (false on negatives: ((-1)^2)^0.5 != -1)
        assert!(!proves_equal(
            r#"(Pow (Pow (Var "x") (Num 2.0)) (Num 0.5))"#,
            r#"(Pow (Var "x") (Mul (Num 2.0) (Num 0.5)))"#,
            "",
        ));
    }
}
