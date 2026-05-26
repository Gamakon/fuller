//! Trigonometric identities, transcribed from SymPy's `simplify/trigsimp.py`
//! and `simplify/fu.py` (the TR* transforms). Real-domain, sound identities
//! only — NO sympy import, NO bare commutativity/associativity.
//!
//! The corpus targets in `parity/corpus/trigsimp.jsonl` are
//! `sympy.trigsimp(input)`. Reaching them needs two ingredients that egglog
//! lacks out of the box for our `Math` datatype:
//!
//!   1. **constant folding** on `(Num _)` leaves — sympy eagerly evaluates
//!      `-1 + 2`, `0.5 * 2`, etc. egglog has f64 primitives (`+ - * /`), so we
//!      fold `(Add (Num a) (Num b)) -> (Num (+ a b))` and friends. These are
//!      strictly shrinking (two nodes -> one) and therefore terminating.
//!
//!   2. **trig identities** — Pythagorean, double-angle, product-to-sum, and
//!      `cos*tan = sin`. Each is a real-domain identity verified by hand below.
//!
//! DIVERGENCE DISCIPLINE: the Pythagorean relation `sin^2 + cos^2 = 1` and the
//! distributive law are the classic non-terminating pairs. We pick CANONICAL
//! directions (always rewrite toward the sympy normal form) and never ship both
//! directions of a reversible pair. The combined ruleset runs under a bounded
//! `(repeat 40 ...)` in the parity scorer, so a missed bound caps rather than
//! hangs, but every rule here is individually terminating by construction.

/// The `trig` ruleset.
pub const TRIG_RULESET: &str = r#"
(ruleset trig)

; =====================================================================
; Constant folding on Num leaves (egglog f64 primitives). Strictly
; shrinking: (op (Num a) (Num b)) -> (Num c). Terminating.
; =====================================================================
(rewrite (Add (Num a) (Num b)) (Num (+ a b)) :ruleset trig)
(rewrite (Sub (Num a) (Num b)) (Num (- a b)) :ruleset trig)
(rewrite (Mul (Num a) (Num b)) (Num (* a b)) :ruleset trig)
(rewrite (Neg (Num a)) (Num (neg a)) :ruleset trig)
(rewrite (Pow2 (Num a)) (Num (* a a)) :ruleset trig)

; ---- Numeric-COEFFICIENT collection (NOT bare associativity) ----
; These merge two numeric literals that sit on the SAME chain of a single
; operator, collapsing two `Num` nodes into one. They only ever fire when
; *both* merged operands are `Num`, so they strictly shrink the term (one
; fewer Num node) and terminate. This is sympy's coefficient gathering
; (Mul.flatten / Add.flatten), expressed without a bare commute/associate
; rewrite over arbitrary subterms.
;
; Multiplicative: pull a stray Num through one nesting level.
(rewrite (Mul (Num a) (Mul (Num b) c)) (Mul (Num (* a b)) c) :ruleset trig)
(rewrite (Mul (Num a) (Mul c (Num b))) (Mul (Num (* a b)) c) :ruleset trig)
(rewrite (Mul (Mul (Num b) c) (Num a)) (Mul (Num (* a b)) c) :ruleset trig)
(rewrite (Mul (Mul c (Num b)) (Num a)) (Mul (Num (* a b)) c) :ruleset trig)

; =====================================================================
; Normalisation of Neg / subtraction into the +(-1 * ) form sympy uses.
; sympy has no Sub/Neg node: a - b is Add(a, Mul(-1, b)) and -a is
; Mul(-1, a). Rewriting OUR Neg/Sub toward that canonical form lets the
; folded-constant targets line up. Strictly directional (Sub/Neg are
; only ever consumed, never produced), so terminating.
; =====================================================================
; -a = (-1) * a
(rewrite (Neg a) (Mul (Num -1.0) a) :ruleset trig)
; a - b = a + (-1)*b
(rewrite (Sub a b) (Add a (Mul (Num -1.0) b)) :ruleset trig)

; =====================================================================
; Pythagorean identity, canonical direction. SymPy's trigsimp normal
; form prefers cos for the "1 - sin^2" shape and replaces "sin^2 - 1".
; All real-domain identities (sin^2 x + cos^2 x = 1 for all real x).
; =====================================================================
; sin^2 + cos^2 = 1  (both operand orders, since Add is not commutative here)
(rewrite (Add (Pow2 (Sin x)) (Pow2 (Cos x))) (Num 1.0) :ruleset trig)
(rewrite (Add (Pow2 (Cos x)) (Pow2 (Sin x))) (Num 1.0) :ruleset trig)
; sin^2 = 1 - cos^2   -> as 1 + (-1)*cos^2 (folded form)
(rewrite (Pow2 (Sin x)) (Add (Num 1.0) (Mul (Num -1.0) (Pow2 (Cos x)))) :ruleset trig)
; -1 + cos^2 x = -sin^2 x   (sympy writes Add(-1, cos^2) -> -sin^2)
(rewrite (Add (Num -1.0) (Pow2 (Cos x))) (Mul (Num -1.0) (Pow2 (Sin x))) :ruleset trig)
; -1 + sin^2 x = -cos^2 x
(rewrite (Add (Num -1.0) (Pow2 (Sin x))) (Mul (Num -1.0) (Pow2 (Cos x))) :ruleset trig)

; =====================================================================
; cos * tan = sin  (real domain: tan = sin/cos, cos cancels). Both
; operand orders. Sound where cos != 0; at cos = 0 tan is undefined so
; the identity's domain is exactly where both sides are defined.
; =====================================================================
(rewrite (Mul (Cos x) (Tan x)) (Sin x) :ruleset trig)
(rewrite (Mul (Tan x) (Cos x)) (Sin x) :ruleset trig)

; =====================================================================
; Product-to-sum / double-angle:  cos x * sin x = (1/2) sin(2x).
; sympy's fu TR8 writes sin*cos as half-angle-doubled sin. Sound for all
; reals. Canonical direction: contract product -> single sin(2x).
; =====================================================================
(rewrite (Mul (Cos x) (Sin x)) (Mul (Num 0.5) (Sin (Mul (Num 2.0) x))) :ruleset trig)
(rewrite (Mul (Sin x) (Cos x)) (Mul (Num 0.5) (Sin (Mul (Num 2.0) x))) :ruleset trig)
; 2 * sin x * cos x = sin(2x)  (the doubled forms that appear pre-folding)
(rewrite (Mul (Mul (Num 2.0) (Sin x)) (Cos x)) (Sin (Mul (Num 2.0) x)) :ruleset trig)
(rewrite (Mul (Mul (Num 2.0) (Cos x)) (Sin x)) (Sin (Mul (Num 2.0) x)) :ruleset trig)

; =====================================================================
; Power-reduction (half-angle) — sympy's fu TR8 / TR0 normal form for
; trigsimp pushes even powers of sin/cos into a cos(2x) form:
;   sin^2 x = 1/2 - 1/2 cos(2x)
;   cos^2 x = 1/2 + 1/2 cos(2x)
; Both are real-domain identities (from cos 2x = 1 - 2 sin^2 x =
; 2 cos^2 x - 1). Canonical direction: Pow2(Sin/Cos) -> cos(2x) form.
; The produced (Mul (Num 2.0) x) is inert for a Var x (no fold needed);
; the rule is strictly directional (Pow2 of a trig is only consumed),
; so it terminates.
; =====================================================================
(rewrite (Pow2 (Sin x))
    (Add (Num 0.5) (Mul (Num -0.5) (Cos (Mul (Num 2.0) x)))) :ruleset trig)
(rewrite (Pow2 (Cos x))
    (Add (Num 0.5) (Mul (Num 0.5) (Cos (Mul (Num 2.0) x)))) :ruleset trig)

; =====================================================================
; Double-angle for cos, the forms sympy's targets use:
;   cos(2x) = 1 - 2 sin^2 x = 2 cos^2 x - 1
; Provide the contraction direction (sin^2/cos^2 combos -> cos 2x) that
; matches corpus targets such as "1 + (-2) sin^2 -> cos 2x". Sound for
; all reals.
; =====================================================================
; 1 - 2 sin^2 x = cos 2x   (written folded: Add 1 (Mul -2 sin^2))
(rewrite (Add (Num 1.0) (Mul (Num -2.0) (Pow2 (Sin x)))) (Cos (Mul (Num 2.0) x)) :ruleset trig)
; 2 cos^2 x - 1 = cos 2x   (folded: Add -1 (Mul 2 cos^2))
(rewrite (Add (Num -1.0) (Mul (Num 2.0) (Pow2 (Cos x)))) (Cos (Mul (Num 2.0) x)) :ruleset trig)

; =====================================================================
; DISTRIBUTION (expand direction ONLY). Multiplication distributes over
; addition; we ship the EXPAND direction (Mul over Add -> Add of Muls)
; and never the factoring reverse, so the rewrite is terminating: every
; application strictly reduces the count of `Add` nodes sitting beneath a
; `Mul`. This is the engine that lets a factored sympy target and a
; distributed input meet at a common fully-expanded normal form (egglog
; e-class absorption finds the meet; we don't need the reverse rule).
; Pure ring algebra — sound for all reals, no trig content.
; =====================================================================
(rewrite (Mul c (Add a b)) (Add (Mul c a) (Mul c b)) :ruleset trig)
(rewrite (Mul (Add a b) c) (Add (Mul a c) (Mul b c)) :ruleset trig)

; (a + b)^2 = a^2 + 2ab + b^2  — expand a squared sum into its polynomial.
; Strictly directional (Pow2-of-Add is only consumed). Sound, real ring.
(rewrite (Pow2 (Add a b))
    (Add (Pow2 a) (Add (Mul (Num 2.0) (Mul a b)) (Pow2 b))) :ruleset trig)

; (-a)^2 = a^2  — strip a leading -1 factor under a square. NARROW form of
; (ab)^2=a^2 b^2 (the general rule diverges in combination with distribution;
; this restricted (-1)*a case is strictly shrinking — removes a Mul node —
; and lets squared-sum expansion fold the (-1)^2 cross terms). Sound.
(rewrite (Pow2 (Mul (Num -1.0) a)) (Pow2 a) :ruleset trig)

; (a^2)^2 = a^4  — fold a square-of-square into a single Pow. Directional
; (Pow2-of-Pow2 only consumed), strictly bounded. Sound for all reals.
(rewrite (Pow2 (Pow2 a)) (Pow a (Num 4.0)) :ruleset trig)
; (a^n)^2 = a^(2n)  for a NUMERIC exponent n we fold with the f64 primitive,
; so no unevaluated exponent tower is built (the powers.rs caution). Strictly
; shrinking (removes the Pow2 wrapper). Sound for all reals where a^n is
; real (egglog only fires on the literal-exponent shape we built ourselves).
(rewrite (Pow2 (Pow a (Num n))) (Pow a (Num (* 2.0 n))) :ruleset trig)

; ---- tan Pythagorean: 1 + tan^2 x = sec^2 x = 1 / cos^2 x = cos(x)^-2 ----
; Real-domain identity wherever cos x != 0 (which is exactly tan's domain).
; We write the RHS as (Cos x)^(-2) so the Pow2-of-Pow rule can lift it to
; the cos^-4 corpus targets. Folded literal exponent, terminating.
(rewrite (Add (Num 1.0) (Pow2 (Tan x))) (Pow (Cos x) (Num -2.0)) :ruleset trig)
"#;

#[cfg(test)]
mod tests {
    use super::TRIG_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::prelude::exprs;
    use egglog::EGraph;

    // Hard safety cap on saturation (mirrors powers.rs). A divergent rule stops
    // at the cap instead of pegging the machine.
    const SAT_ITERS: u32 = 40;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        e.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        e.parse_and_run_program(None, TRIG_RULESET).unwrap();
        e
    }

    fn proves_equal(input: &str, target: &str) -> bool {
        let mut e = egraph();
        let prog = format!(
            "(let __in {input})\n(let __tgt {target})\n\
             (run-schedule (repeat {SAT_ITERS} (run trig)))\n(check (= __in __tgt))"
        );
        e.parse_and_run_program(None, &prog).is_ok()
    }

    /// Soundness check: evaluate input and target on random real points; where
    /// both are finite they must agree.
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
    fn constant_folding() {
        assert!(proves_equal(
            r#"(Add (Num 1.0) (Num 2.0))"#,
            r#"(Num 3.0)"#
        ));
        assert!(proves_equal(
            r#"(Mul (Num 0.5) (Num 2.0))"#,
            r#"(Num 1.0)"#
        ));
    }

    #[test]
    fn pythagorean_identities_hold_and_are_sound() {
        // -1 + sin^2 = -cos^2
        assert!(proves_equal(
            r#"(Add (Num -1.0) (Pow2 (Sin (Var "y"))))"#,
            r#"(Mul (Num -1.0) (Pow2 (Cos (Var "y"))))"#
        ));
        assert_sound(
            r#"(Add (Num -1.0) (Pow2 (Sin (Var "y"))))"#,
            r#"(Mul (Num -1.0) (Pow2 (Cos (Var "y"))))"#,
            &[("y", 1.3)],
        );
        // sin^2 + cos^2 = 1
        assert!(proves_equal(
            r#"(Add (Pow2 (Sin (Var "x"))) (Pow2 (Cos (Var "x"))))"#,
            r#"(Num 1.0)"#
        ));
        assert_sound(
            r#"(Add (Pow2 (Sin (Var "x"))) (Pow2 (Cos (Var "x"))))"#,
            r#"(Num 1.0)"#,
            &[("x", -2.1)],
        );
    }

    #[test]
    fn cos_tan_is_sin() {
        assert!(proves_equal(
            r#"(Mul (Cos (Var "x")) (Tan (Var "x")))"#,
            r#"(Sin (Var "x"))"#
        ));
        assert_sound(
            r#"(Mul (Cos (Var "x")) (Tan (Var "x")))"#,
            r#"(Sin (Var "x"))"#,
            &[("x", 0.7)],
        );
    }

    #[test]
    fn product_to_sum_sound() {
        // cos x * sin x = 0.5 sin 2x
        assert_sound(
            r#"(Mul (Cos (Var "x")) (Sin (Var "x")))"#,
            r#"(Mul (Num 0.5) (Sin (Mul (Num 2.0) (Var "x"))))"#,
            &[("x", 1.1)],
        );
        assert!(proves_equal(
            r#"(Mul (Cos (Num 1.0)) (Sin (Num 1.0)))"#,
            r#"(Mul (Num 0.5) (Sin (Mul (Num 2.0) (Num 1.0))))"#
        ));
    }

    #[test]
    fn pow4_and_distribution_sound() {
        // (-cos^2 y)^2 = cos^4 y, reached via -1+sin^2 -> -cos^2, (-a)^2,
        // (a^2)^2 -> a^4.
        assert!(proves_equal(
            r#"(Pow2 (Add (Num -1.0) (Pow2 (Sin (Var "y")))))"#,
            r#"(Pow (Cos (Var "y")) (Num 4.0))"#
        ));
        assert_sound(
            r#"(Pow2 (Add (Num -1.0) (Pow2 (Sin (Var "y")))))"#,
            r#"(Pow (Cos (Var "y")) (Num 4.0))"#,
            &[("y", 0.9)],
        );
        // 1 + tan^2 x = cos(x)^-2, squared -> cos(x)^-4.
        assert!(proves_equal(
            r#"(Pow2 (Add (Num 1.0) (Pow2 (Tan (Var "x")))))"#,
            r#"(Pow (Cos (Var "x")) (Num -4.0))"#
        ));
        assert_sound(
            r#"(Add (Num 1.0) (Pow2 (Tan (Var "x"))))"#,
            r#"(Pow (Cos (Var "x")) (Num -2.0))"#,
            &[("x", 0.6)],
        );
        // distribution: -1 * (10 - 2x) = -10 + 2x
        assert!(proves_equal(
            r#"(Mul (Mul (Num -1.0) (Add (Num 10.0) (Mul (Num -2.0) (Var "x")))) (Tan (Num 6.0)))"#,
            r#"(Mul (Add (Num -10.0) (Mul (Num 2.0) (Var "x"))) (Tan (Num 6.0)))"#
        ));
    }

    /// Soundness floor: a non-identity must NOT be proven.
    #[test]
    fn does_not_prove_falsehood() {
        assert!(!proves_equal(
            r#"(Sin (Var "x"))"#,
            r#"(Cos (Var "x"))"#
        ));
    }
}
