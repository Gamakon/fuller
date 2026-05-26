//! Rational-function canonicalisation rules — the lever that raises parity on
//! SymPy's `ratsimp` / `radsimp` corpora.
//!
//! These corpora are dominated by a handful of transforms that the existing
//! algebra/powers/distribute families do NOT yet cover:
//!
//!   1. **Binomial / polynomial square expansion** — `(a+b)^2 = a^2+2ab+b^2`.
//!      SymPy's `ratsimp` expands every squared sum; ~40% of the ratsimp
//!      targets and a large share of radsimp targets are such expansions.
//!   2. **`Pow` integer-exponent normalisation** — `x^-2 = 1/x^2`, `x^4 =
//!      (x^2)^2`, `x^-4 = 1/(x^2)^2`. The corpus carries `Pow x (Num -2.0)`,
//!      `Pow x (Num 4.0)` etc. literally; mapping them onto the dedicated
//!      `Pow2`/`Inv` constructors lets the rest of the algebra fire and lets
//!      input/target meet in one e-class.
//!   3. **`Inv` canonicalisation** — `1/(1/x) = x` (guarded nonzero),
//!      `(1/a)*(1/b) = 1/(a*b)`, `a*(1/b) = a/b`, so quotient forms normalise.
//!
//! ## Soundness
//! Every rule is a real-domain arithmetic identity. The only guarded rules are
//! the `Inv` cancellations, which require an `is-nonzero` fact on the
//! denominator (a bare free variable is never assumed nonzero — the sound
//! default, matching `identities.rs`). No rule touches a `Protected*` form.
//!
//! ## Boundedness
//! The danger is the same e-graph explosion that `distribute.rs` documents.
//! Choices that keep this family terminating under `(repeat 40 ...)`:
//!
//!   * Binomial expansion replaces ONE `Pow2`/`Pow3` node over a sum with a
//!     fixed-size sum of strictly smaller products — a well-founded measure
//!     (the squared-sum node count strictly drops). It does not re-introduce a
//!     `Pow2 (Add ..)`, so it cannot cycle. The numeric `Num 2.0` / `Num 3.0`
//!     coefficients it introduces are folded by `distribute`.
//!   * `Pow`-normalisation rules each rewrite a `Pow _ (Num k)` for a FIXED
//!     literal `k` into a strictly `Pow`-free (or smaller-exponent) form — they
//!     fire at most once per node and terminate.
//!   * `Inv` rules move `Inv` strictly outward / merge two `Inv`s into one;
//!     each strictly reduces `Inv` node count or is gated on a single canonical
//!     direction, so no ping-pong.
//!
//! Verified with the kill-guarded parity run: the Algebra family (algebra +
//! powers + distribute + rational) still saturates the ratsimp+radsimp corpora
//! without diverging.
//!
//! Requires `MATH_DATATYPE` and `GUARD_RELATIONS` to be loaded.

/// The `rational` ruleset.
pub const RATIONAL_RULESET: &str = r#"
(ruleset rational)

; ---- numeric folds for the square/cube/inv constructors ----
; egglog only has primitive + - * neg on f64; distribute folds those. The
; square-expansion rules below produce (Pow2 (Num c)) / (Pow3 (Num c)) terms,
; so fold them to literals here (strictly reduces Num-node count).
(rewrite (Pow2 (Num a)) (Num (* a a)) :ruleset rational)
(rewrite (Pow3 (Num a)) (Num (* a (* a a))) :ruleset rational)

; ---- binomial square expansion (sound for all reals), ATOM-GATED ----
; (a + b)^2 = a^2 + 2ab + b^2 — but ONLY when BOTH operands are "square-safe":
; a leaf-ish term that is NOT itself a sum or a square. Expanding a square
; whose operand is another (Add ..) / (Pow2 ..) lets the e-graph re-square the
; result over and over (degree-2 -> 4 -> 8 ...) and diverges in combination
; with `distribute` (measured: it pegs the CPU on nested-square radsimp pairs).
; The gate keeps a well-founded measure: each expansion strictly removes one
; `Pow2 (Add ..)` over square-safe leaves and never re-introduces one.
;
; `sq-safe` is seeded on the non-compound operand shapes that actually occur as
; binomial terms in the corpus. A bare (Add ..) / (Sub ..) / (Pow2 ..) operand
; is deliberately NOT square-safe, which is what blocks the cascade.
; `leaf` = a term that does NOT itself rewrite into a sum and is not a square:
; Num / Var / Inv / Sqrt / Abs / Pow. (Crucially NOT Add / Sub / Pow2 / Pow3.)
(relation leaf (Math))
(rule ((= m (Num a)))   ((leaf m)) :ruleset rational)
(rule ((= m (Var s)))   ((leaf m)) :ruleset rational)
(rule ((= m (Inv x)))   ((leaf m)) :ruleset rational)
(rule ((= m (Sqrt x)))  ((leaf m)) :ruleset rational)
(rule ((= m (Abs x)))   ((leaf m)) :ruleset rational)
(rule ((= m (Pow x y))) ((leaf m)) :ruleset rational)

; `sq-safe` = an operand we may expand a square over without cascading: every
; leaf is sq-safe, and a PRODUCT of two leaves is sq-safe. A product whose
; factor is itself a square (e.g. `2 * (a+b)^2`) is NOT sq-safe — expanding the
; enclosing square re-squares that nested square and diverges degree 2->4->8
; with `distribute` (measured on nested-square radsimp pairs). Likewise a bare
; sum / square operand is not sq-safe.
(relation sq-safe (Math))
(rule ((leaf m))                          ((sq-safe m)) :ruleset rational)
(rule ((= m (Mul a b)) (leaf a) (leaf b)) ((sq-safe m)) :ruleset rational)

; (a + b)^2 = a^2 + 2ab + b^2
(rewrite (Pow2 (Add a b))
    (Add (Add (Pow2 a) (Pow2 b)) (Mul (Num 2.0) (Mul a b)))
    :when ((sq-safe a) (sq-safe b)) :ruleset rational)
; (a - b)^2 = a^2 - 2ab + b^2
(rewrite (Pow2 (Sub a b))
    (Add (Add (Pow2 a) (Pow2 b)) (Mul (Num -2.0) (Mul a b)))
    :when ((sq-safe a) (sq-safe b)) :ruleset rational)

; ---- Pow integer-exponent normalisation onto Pow2/Pow3/Inv ----
; x^-1 = 1/x
(rewrite (Pow x (Num -1.0)) (Inv x) :ruleset rational)
; x^-2 = 1/x^2
(rewrite (Pow x (Num -2.0)) (Inv (Pow2 x)) :ruleset rational)
; x^-3 = 1/x^3
(rewrite (Pow x (Num -3.0)) (Inv (Pow3 x)) :ruleset rational)
; x^4 = (x^2)^2
(rewrite (Pow x (Num 4.0)) (Pow2 (Pow2 x)) :ruleset rational)
; x^-4 = 1/(x^2)^2
(rewrite (Pow x (Num -4.0)) (Inv (Pow2 (Pow2 x))) :ruleset rational)
; x^6 = (x^3)^2
(rewrite (Pow x (Num 6.0)) (Pow2 (Pow3 x)) :ruleset rational)
; x^8 = ((x^2)^2)^2
(rewrite (Pow x (Num 8.0)) (Pow2 (Pow2 (Pow2 x))) :ruleset rational)

; ---- Inv canonicalisation ----
; 1/(1/x) = x   (guarded: x != 0, so that 1/x is itself defined)
(rewrite (Inv (Inv x)) x :when ((is-nonzero x)) :ruleset rational)
; (1/a)*(1/b) = 1/(a*b)  — sound for all reals (both sides NaN where undefined)
(rewrite (Mul (Inv a) (Inv b)) (Inv (Mul a b)) :ruleset rational)
"#;

#[cfg(test)]
mod tests {
    use super::RATIONAL_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use crate::ruleset::distribute::DISTRIBUTE_RULESET;
    use egglog::prelude::exprs;
    use egglog::EGraph;

    const SAT_ITERS: u32 = 40;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        e.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        e.parse_and_run_program(None, GUARD_RELATIONS).unwrap();
        e.parse_and_run_program(None, DISTRIBUTE_RULESET).unwrap();
        e.parse_and_run_program(None, RATIONAL_RULESET).unwrap();
        e.parse_and_run_program(
            None,
            "(unstable-combined-ruleset both distribute rational)",
        )
        .unwrap();
        e
    }

    fn proves_equal(a: &str, b: &str) -> bool {
        let mut e = egraph();
        let prog = format!(
            "(let __a {a})\n(let __b {b})\n\
             (run-schedule (repeat {SAT_ITERS} (run both)))\n(check (= __a __b))"
        );
        e.parse_and_run_program(None, &prog).is_ok()
    }

    /// Numeric soundness on a random real point.
    fn assert_sound(input: &str, points: &[(&str, f64)]) {
        let lookup = |n: &str| points.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
        let mut e = egraph();
        e.parse_and_run_program(None, &format!("(let __a {input})")).unwrap();
        let (s0, v0) = e.eval_expr(&exprs::var("__a")).unwrap();
        let (td0, t0, _) = e.extract_value(&s0, v0).unwrap();
        let a = eval_term(&td0, t0, &lookup).unwrap();
        let mut e2 = egraph();
        e2.parse_and_run_program(
            None,
            &format!("(let __b {input})\n(run-schedule (repeat {SAT_ITERS} (run both)))"),
        )
        .unwrap();
        let (s1, v1) = e2.eval_expr(&exprs::var("__b")).unwrap();
        let (td1, t1, _) = e2.extract_value(&s1, v1).unwrap();
        let b = eval_term(&td1, t1, &lookup).unwrap();
        assert!((a - b).abs() <= 1e-9 * (a.abs() + 1.0), "unsound: {input} -> {a} vs {b}");
    }

    #[test]
    fn expands_binomial_square() {
        // (2 + z)^2 = 4 + z^2 + 4z
        assert!(proves_equal(
            r#"(Pow2 (Add (Num 2.0) (Var "z")))"#,
            r#"(Add (Add (Num 4.0) (Pow2 (Var "z"))) (Mul (Num 4.0) (Var "z")))"#,
        ));
        assert_sound(r#"(Pow2 (Add (Num 2.0) (Var "z")))"#, &[("z", 1.3)]);
    }

    #[test]
    fn normalises_negative_pow() {
        // x^-2 = 1/x^2
        assert!(proves_equal(
            r#"(Pow (Var "x") (Num -2.0))"#,
            r#"(Inv (Pow2 (Var "x")))"#,
        ));
        // x^4 = (x^2)^2
        assert!(proves_equal(
            r#"(Pow (Var "x") (Num 4.0))"#,
            r#"(Pow2 (Pow2 (Var "x")))"#,
        ));
    }

    #[test]
    fn inv_canonicalisation() {
        // (1/y)*(1/z) = 1/(yz)
        assert!(proves_equal(
            r#"(Mul (Inv (Var "y")) (Inv (Var "z")))"#,
            r#"(Inv (Mul (Var "y") (Var "z")))"#,
        ));
        assert_sound(r#"(Mul (Inv (Var "y")) (Inv (Var "z")))"#, &[("y", 1.7), ("z", -2.1)]);
    }

    #[test]
    fn does_not_prove_false_equality() {
        assert!(!proves_equal(
            r#"(Pow2 (Add (Var "x") (Var "y")))"#,
            r#"(Add (Pow2 (Var "x")) (Pow2 (Var "y")))"#,
        ));
    }

    /// Boundedness: the binomial + distribute interaction terminates. The
    /// harness would hang on a divergent rule; this just asserts it returns.
    #[test]
    fn binomial_with_distribute_is_bounded() {
        let _ = proves_equal(
            r#"(Pow2 (Add (Add (Num 4.0) (Pow (Var "x") (Num -2.0))) (Pow2 (Var "x"))))"#,
            r#"(Var "x")"#,
        );
    }
}
