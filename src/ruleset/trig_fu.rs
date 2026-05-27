//! Structurally-rich trig identities mined from SymPy's `simplify/fu.py`
//! (the TR* transforms). This is a STANDALONE ruleset, deliberately kept
//! separate from `trig.rs`: `trig.rs` ships strictly canonical (one-way)
//! directions toward sympy's trigsimp normal form, whereas the rules here are
//! the SHAPE-CHANGING equivalences in BOTH directions. Their purpose is not
//! normalisation but to POPULATE the e-class with forms that differ
//! structurally (op count, transcendental nesting, product-vs-sum profile) so
//! the HFF angular extractor has genuinely different members to choose among.
//!
//! WHY BOTH DIRECTIONS: egglog's e-graph holds every form an equation rewrites
//! to in a single equivalence class. By asserting `cos a cos b` <-> the
//! sum form, the class contains both a product (2 transcendental leaves, 1
//! Mul) and a sum (2 transcendental leaves with composite args, 1 Add). The
//! extractor — not the rewriter — decides which is "best". So directionality
//! that `trig.rs` needs for a single normal form is exactly what we DON'T want
//! here.
//!
//! NON-CONFLUENCE / DIVERGENCE: product<->sum and angle-addition expand<->
//! contract are the textbook non-terminating pairs (each application can
//! manufacture a strictly larger angle argument, e.g. `cos(a+b)*cos c ->
//! cos(a+b+c)` ...). This ruleset is therefore **NOT meant to be saturated to
//! fixpoint**. It is run under a small bounded `(repeat N (run trig_fu))`
//! schedule (the HFF extractor path and the tests below both use a small N).
//! Every individual rule is a real-domain identity true for all reals; the
//! pairing is what is unbounded, and the bound is the caller's contract. Do
//! NOT co-saturate this with `distribute` (CLAUDE.md: distribute + trig
//! explode together) and do NOT remove the bound.
//!
//! NO sec/csc/cot: the `Math` datatype has no secant/cosecant/cotangent
//! constructor, so any TR whose RHS needs them is skipped (TR's reciprocal
//! forms). `tan` is expressible as `Sin/Cos` (TR2) and is included.
//!
//! Identity sources (sympy fu.py docstrings, real-domain, verified):
//!   TR2  : tan x = sin x / cos x
//!   TR8  : cos a cos b = (cos(a-b) + cos(a+b))/2
//!          sin a sin b = (cos(a-b) - cos(a+b))/2
//!          sin a cos b = (sin(a+b) + sin(a-b))/2
//!   TR9  : the reverse of the three TR8 identities (sum -> product)
//!   TR10 : cos(a+b) = cos a cos b - sin a sin b
//!          sin(a+b) = sin a cos b + cos a sin b
//!   TR10i: the reverse of TR10 (sum-of-products -> function of a sum)
//!   TR11 : sin(2x) = 2 sin x cos x
//!          cos(2x) = cos^2 x - sin^2 x   (and 1-2sin^2 / 2cos^2-1 variants)

/// The `trig_fu` ruleset — structural trig equivalences, run BOUNDED only.
pub const TRIG_FU_RULESET: &str = r#"
(ruleset trig_fu)

; =====================================================================
; TR2 : tan x = sin x / cos x.  Real-domain identity wherever cos x != 0,
; which is exactly tan's domain. Changes the transcendental profile
; (one Tan leaf <-> a Sin and a Cos under a Div) — valuable to the
; angular measure. Both directions so the class holds tan AND sin/cos.
; =====================================================================
(birewrite (Tan x) (Div (Sin x) (Cos x)) :ruleset trig_fu)

; =====================================================================
; TR8 / TR9 : product <-> sum.  cos*cos, sin*sin, sin*cos.
; Each is a real identity for all a, b (product-to-sum formulae). Shipped
; as `birewrite` so the e-class holds BOTH the product and the sum form.
; NON-TERMINATING as a pair (angles grow under repeated expand) — bounded
; schedule only. Operand orders: only one order is written per identity;
; the symmetric partner identity (sin*cos vs cos*sin) covers the swap.
; =====================================================================
; cos a * cos b = (cos(a-b) + cos(a+b)) / 2
(birewrite (Mul (Cos a) (Cos b))
    (Mul (Num 0.5) (Add (Cos (Sub a b)) (Cos (Add a b)))) :ruleset trig_fu)
; sin a * sin b = (cos(a-b) - cos(a+b)) / 2
(birewrite (Mul (Sin a) (Sin b))
    (Mul (Num 0.5) (Sub (Cos (Sub a b)) (Cos (Add a b)))) :ruleset trig_fu)
; sin a * cos b = (sin(a+b) + sin(a-b)) / 2
(birewrite (Mul (Sin a) (Cos b))
    (Mul (Num 0.5) (Add (Sin (Add a b)) (Sin (Sub a b)))) :ruleset trig_fu)
; cos a * sin b = (sin(a+b) - sin(a-b)) / 2  (the operand-swapped partner;
; sin(a+b) - sin(a-b) = 2 cos a sin b, a real identity for all a, b)
(birewrite (Mul (Cos a) (Sin b))
    (Mul (Num 0.5) (Sub (Sin (Add a b)) (Sin (Sub a b)))) :ruleset trig_fu)

; =====================================================================
; TR10 / TR10i : angle-addition expand <-> contract.
;   cos(a+b) = cos a cos b - sin a sin b
;   sin(a+b) = sin a cos b + cos a sin b
; Real identities for all a, b. `birewrite` so the class holds both the
; single-angle-of-a-sum form and the expanded product form. The expand
; direction is the angle-growing one — bounded schedule only.
; =====================================================================
(birewrite (Cos (Add a b))
    (Sub (Mul (Cos a) (Cos b)) (Mul (Sin a) (Sin b))) :ruleset trig_fu)
(birewrite (Sin (Add a b))
    (Add (Mul (Sin a) (Cos b)) (Mul (Cos a) (Sin b))) :ruleset trig_fu)

; =====================================================================
; TR11 : double angle <-> product. The doubled-argument form sympy uses
; is `(Mul (Num 2.0) x)`.
;   sin(2x) = 2 sin x cos x
;   cos(2x) = cos^2 x - sin^2 x
; Plus the two power-reduction VARIANTS of cos(2x) that fu.py / trigsimp
; also recognise (all three RHS equal cos(2x) by sin^2+cos^2=1):
;   cos(2x) = 1 - 2 sin^2 x
;   cos(2x) = 2 cos^2 x - 1
; All real identities. `birewrite` so the class holds both the compact
; double-angle node and the expanded squared form (different transcendental
; nesting — what the measure vector wants).
; =====================================================================
(birewrite (Sin (Mul (Num 2.0) x))
    (Mul (Num 2.0) (Mul (Sin x) (Cos x))) :ruleset trig_fu)
(birewrite (Cos (Mul (Num 2.0) x))
    (Sub (Pow2 (Cos x)) (Pow2 (Sin x))) :ruleset trig_fu)
(birewrite (Cos (Mul (Num 2.0) x))
    (Sub (Num 1.0) (Mul (Num 2.0) (Pow2 (Sin x)))) :ruleset trig_fu)
(birewrite (Cos (Mul (Num 2.0) x))
    (Sub (Mul (Num 2.0) (Pow2 (Cos x))) (Num 1.0)) :ruleset trig_fu)
"#;

#[cfg(test)]
mod tests {
    use super::TRIG_FU_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::prelude::exprs;
    use egglog::EGraph;

    // SMALL bound. This ruleset is non-terminating as a whole (product<->sum
    // and expand<->contract grow angles); a small repeat count is the contract.
    // Every proof below needs only a handful of rewrites, so N is tiny.
    const SAT_ITERS: u32 = 6;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        e.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        e.parse_and_run_program(None, TRIG_FU_RULESET).unwrap();
        e
    }

    /// Bounded proof that `input` and `target` land in the same e-class.
    fn proves_equal(input: &str, target: &str) -> bool {
        let mut e = egraph();
        let prog = format!(
            "(let __in {input})\n(let __tgt {target})\n\
             (run-schedule (repeat {SAT_ITERS} (run trig_fu)))\n(check (= __in __tgt))"
        );
        e.parse_and_run_program(None, &prog).is_ok()
    }

    /// Numeric soundness: evaluate input and target at concrete reals; where
    /// both are finite they must agree to tolerance.
    fn assert_sound(input: &str, target: &str, points: &[(&str, f64)]) {
        let env = |n: &str| points.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);

        let mut e = EGraph::default();
        e.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e.parse_and_run_program(None, &format!("(let __a {input})")).unwrap();
        let (s0, v0) = e.eval_expr(&exprs::var("__a")).unwrap();
        let (td0, t0, _) = e.extract_value(&s0, v0).unwrap();
        let a = eval_term(&td0, t0, &env);

        let mut e2 = EGraph::default();
        e2.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e2.parse_and_run_program(None, &format!("(let __b {target})")).unwrap();
        let (s1, v1) = e2.eval_expr(&exprs::var("__b")).unwrap();
        let (td1, t1, _) = e2.extract_value(&s1, v1).unwrap();
        let b = eval_term(&td1, t1, &env);

        if let (Ok(a), Ok(b)) = (a, b) {
            if a.is_finite() && b.is_finite() {
                assert!(
                    (a - b).abs() <= 1e-9 * (a.abs() + 1.0),
                    "unsound: {input} = {a} vs {target} = {b}"
                );
            }
        }
    }

    #[test]
    fn tr2_tan_is_sin_over_cos() {
        assert!(proves_equal(
            r#"(Tan (Var "x"))"#,
            r#"(Div (Sin (Var "x")) (Cos (Var "x")))"#
        ));
        assert_sound(
            r#"(Tan (Var "x"))"#,
            r#"(Div (Sin (Var "x")) (Cos (Var "x")))"#,
            &[("x", 0.7)],
        );
    }

    #[test]
    fn tr8_product_to_sum() {
        // cos a cos b = (cos(a-b) + cos(a+b))/2
        assert!(proves_equal(
            r#"(Mul (Cos (Var "a")) (Cos (Var "b")))"#,
            r#"(Mul (Num 0.5) (Add (Cos (Sub (Var "a") (Var "b"))) (Cos (Add (Var "a") (Var "b")))))"#
        ));
        assert_sound(
            r#"(Mul (Cos (Var "a")) (Cos (Var "b")))"#,
            r#"(Mul (Num 0.5) (Add (Cos (Sub (Var "a") (Var "b"))) (Cos (Add (Var "a") (Var "b")))))"#,
            &[("a", 1.1), ("b", 0.4)],
        );
        // sin a sin b = (cos(a-b) - cos(a+b))/2
        assert_sound(
            r#"(Mul (Sin (Var "a")) (Sin (Var "b")))"#,
            r#"(Mul (Num 0.5) (Sub (Cos (Sub (Var "a") (Var "b"))) (Cos (Add (Var "a") (Var "b")))))"#,
            &[("a", 1.1), ("b", 0.4)],
        );
        // sin a cos b = (sin(a+b) + sin(a-b))/2
        assert_sound(
            r#"(Mul (Sin (Var "a")) (Cos (Var "b")))"#,
            r#"(Mul (Num 0.5) (Add (Sin (Add (Var "a") (Var "b"))) (Sin (Sub (Var "a") (Var "b")))))"#,
            &[("a", 1.1), ("b", 0.4)],
        );
        // cos a sin b = (sin(a+b) - sin(a-b))/2
        assert_sound(
            r#"(Mul (Cos (Var "a")) (Sin (Var "b")))"#,
            r#"(Mul (Num 0.5) (Sub (Sin (Add (Var "a") (Var "b"))) (Sin (Sub (Var "a") (Var "b")))))"#,
            &[("a", 1.1), ("b", 0.4)],
        );
    }

    #[test]
    fn tr9_sum_to_product_reverse_direction() {
        // The reverse direction must also be reachable (birewrite):
        // (cos(a-b) + cos(a+b))/2 -> cos a cos b
        assert!(proves_equal(
            r#"(Mul (Num 0.5) (Add (Cos (Sub (Var "a") (Var "b"))) (Cos (Add (Var "a") (Var "b")))))"#,
            r#"(Mul (Cos (Var "a")) (Cos (Var "b")))"#
        ));
    }

    #[test]
    fn tr10_angle_addition() {
        // cos(a+b) = cos a cos b - sin a sin b
        assert!(proves_equal(
            r#"(Cos (Add (Var "a") (Var "b")))"#,
            r#"(Sub (Mul (Cos (Var "a")) (Cos (Var "b"))) (Mul (Sin (Var "a")) (Sin (Var "b"))))"#
        ));
        assert_sound(
            r#"(Cos (Add (Var "a") (Var "b")))"#,
            r#"(Sub (Mul (Cos (Var "a")) (Cos (Var "b"))) (Mul (Sin (Var "a")) (Sin (Var "b"))))"#,
            &[("a", 0.9), ("b", -0.5)],
        );
        // sin(a+b) = sin a cos b + cos a sin b
        assert!(proves_equal(
            r#"(Sin (Add (Var "a") (Var "b")))"#,
            r#"(Add (Mul (Sin (Var "a")) (Cos (Var "b"))) (Mul (Cos (Var "a")) (Sin (Var "b"))))"#
        ));
        assert_sound(
            r#"(Sin (Add (Var "a") (Var "b")))"#,
            r#"(Add (Mul (Sin (Var "a")) (Cos (Var "b"))) (Mul (Cos (Var "a")) (Sin (Var "b"))))"#,
            &[("a", 0.9), ("b", -0.5)],
        );
    }

    #[test]
    fn tr10i_contract_reverse_direction() {
        // sum-of-products -> function of a sum (reverse of TR10)
        assert!(proves_equal(
            r#"(Sub (Mul (Cos (Var "a")) (Cos (Var "b"))) (Mul (Sin (Var "a")) (Sin (Var "b"))))"#,
            r#"(Cos (Add (Var "a") (Var "b")))"#
        ));
    }

    #[test]
    fn tr11_double_angle() {
        // sin(2x) = 2 sin x cos x
        assert!(proves_equal(
            r#"(Sin (Mul (Num 2.0) (Var "x")))"#,
            r#"(Mul (Num 2.0) (Mul (Sin (Var "x")) (Cos (Var "x"))))"#
        ));
        assert_sound(
            r#"(Sin (Mul (Num 2.0) (Var "x")))"#,
            r#"(Mul (Num 2.0) (Mul (Sin (Var "x")) (Cos (Var "x"))))"#,
            &[("x", 1.3)],
        );
        // cos(2x) = cos^2 x - sin^2 x
        assert!(proves_equal(
            r#"(Cos (Mul (Num 2.0) (Var "x")))"#,
            r#"(Sub (Pow2 (Cos (Var "x"))) (Pow2 (Sin (Var "x"))))"#
        ));
        assert_sound(
            r#"(Cos (Mul (Num 2.0) (Var "x")))"#,
            r#"(Sub (Pow2 (Cos (Var "x"))) (Pow2 (Sin (Var "x"))))"#,
            &[("x", 1.3)],
        );
        // cos(2x) = 1 - 2 sin^2 x   and   = 2 cos^2 x - 1
        assert!(proves_equal(
            r#"(Cos (Mul (Num 2.0) (Var "x")))"#,
            r#"(Sub (Num 1.0) (Mul (Num 2.0) (Pow2 (Sin (Var "x")))))"#
        ));
        assert_sound(
            r#"(Cos (Mul (Num 2.0) (Var "x")))"#,
            r#"(Sub (Mul (Num 2.0) (Pow2 (Cos (Var "x")))) (Num 1.0))"#,
            &[("x", 1.3)],
        );
    }

    /// Soundness floor: a non-identity must NOT be provable under the bound.
    #[test]
    fn does_not_prove_falsehood() {
        assert!(!proves_equal(
            r#"(Sin (Var "x"))"#,
            r#"(Cos (Var "x"))"#
        ));
    }
}
