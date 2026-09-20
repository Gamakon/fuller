//! Sign normalisation: Neg / Sub / Abs and negative literals.
//!
//! Mined from the hall-of-fame dataset (847 real expressions): the shapes that
//! `smallest_form` left standing were overwhelmingly a stray negation or a
//! redundant Abs. Evidence, soundness arguments and the rejected rules are in
//! `docs/rule_proposals/sign.md`.
//!
//! Two groups. ABSORBERS each strictly remove a node. FLOAT-OUTS are
//! size-neutral and only ever move a `Neg` one level toward the root, so an
//! absorber above can delete it; nothing in algebra/powers/rational pushes a
//! `Neg` back down, so the pair terminates.
//!
//! Every rule is exact under `src/eval.rs` including NaN and +/-inf; the only
//! observable-free difference is the sign of a zero, the same standard as
//! `(Mul (Num -1.0) x) -> (Neg x)` in `algebra`. None needs a guard.
//!
//! Do NOT co-load with `trig` (`Neg a -> Mul -1 a`) or `wide`
//! (`Sub -> Add Neg`): each undoes a float-out and the pair cycles.
//!
//! `ProtectedInv` is deliberately absent: protected_inv(-0) = 1 but
//! -(protected_inv 0) = -1, so it is not odd.

/// The `sign` ruleset.
pub const SIGN_RULESET: &str = r#"
(ruleset sign)

; ===== A. ABSORBERS — each strictly removes >= 1 node =====
; A1  a - (-b) = a + b
(rewrite (Sub a (Neg b)) (Add a b) :ruleset sign)
; A2  a + (-b) = a - b ; (-a) + b = b - a
(rewrite (Add a (Neg b)) (Sub a b) :ruleset sign)
(rewrite (Add (Neg a) b) (Sub b a) :ruleset sign)
; A3  -(a - b) = b - a
(rewrite (Neg (Sub a b)) (Sub b a) :ruleset sign)
; A4  0 - x = -x
(rewrite (Sub (Num 0.0) x) (Neg x) :ruleset sign)
; A5  -(literal) folds ; -(c + x) = (-c) - x, both Add orders
(rewrite (Neg (Num a)) (Num (neg a)) :ruleset sign)
(rewrite (Neg (Add (Num c) x)) (Sub (Num (neg c)) x) :ruleset sign)
(rewrite (Neg (Add x (Num c))) (Sub (Num (neg c)) x) :ruleset sign)
; A6  sign pairs cancel through a product / quotient (incl. protected divide)
(rewrite (Mul (Neg a) (Neg b)) (Mul a b) :ruleset sign)
(rewrite (Div (Neg a) (Neg b)) (Div a b) :ruleset sign)
(rewrite (ProtectedDiv (Neg a) (Neg b)) (ProtectedDiv a b) :ruleset sign)
(rule ((= e (ProtectedDiv (Num c) (Neg b))) (< c 0.0))
      ((union e (ProtectedDiv (Num (neg c)) b))) :ruleset sign)
(rule ((= e (Div (Num c) (Neg b))) (< c 0.0))
      ((union e (Div (Num (neg c)) b))) :ruleset sign)
; A7  a negation meeting a difference inside a product flips the difference
(rewrite (Neg (Mul (Sub a b) c)) (Mul (Sub b a) c) :ruleset sign)
(rewrite (Neg (Mul c (Sub a b))) (Mul c (Sub b a)) :ruleset sign)
(rewrite (Mul (Sub a b) (Neg c)) (Mul (Sub b a) c) :ruleset sign)
(rewrite (Mul (Neg c) (Sub a b)) (Mul c (Sub b a)) :ruleset sign)
; A8  EVEN functions swallow a Neg / an Abs
(rewrite (Pow2 (Neg x)) (Pow2 x) :ruleset sign)
(rewrite (Pow2 (Abs x)) (Pow2 x) :ruleset sign)
(rewrite (Abs (Neg x)) (Abs x) :ruleset sign)
(rewrite (Cos (Neg x)) (Cos x) :ruleset sign)
(rewrite (Cos (Abs x)) (Cos x) :ruleset sign)
(rewrite (ProtectedSqrt (Neg x)) (ProtectedSqrt x) :ruleset sign)
(rewrite (ProtectedSqrt (Abs x)) (ProtectedSqrt x) :ruleset sign)
(rewrite (ProtectedLog (Neg x)) (ProtectedLog x) :ruleset sign)
(rewrite (ProtectedLog (Abs x)) (ProtectedLog x) :ruleset sign)
; A9  Abs of a function whose range is already >= 0 (or NaN)
(rewrite (Abs (Pow2 x)) (Pow2 x) :ruleset sign)
(rewrite (Abs (Sqrt x)) (Sqrt x) :ruleset sign)
(rewrite (Abs (ProtectedSqrt x)) (ProtectedSqrt x) :ruleset sign)
(rewrite (Abs (ProtectedExp x)) (ProtectedExp x) :ruleset sign)

; ===== B. FLOAT-OUTS — size-neutral; move a Neg one level ROOTWARD so an
; absorber above can delete it. Never push a Neg down. =====
; B1  (-a) - b = -(a + b)
(rewrite (Sub (Neg a) b) (Neg (Add a b)) :ruleset sign)
; B2  Neg out of a product / quotient (raw and protected divide)
(rewrite (Mul (Neg a) b) (Neg (Mul a b)) :ruleset sign)
(rewrite (Mul a (Neg b)) (Neg (Mul a b)) :ruleset sign)
(rewrite (Div (Neg a) b) (Neg (Div a b)) :ruleset sign)
(rewrite (Div a (Neg b)) (Neg (Div a b)) :ruleset sign)
(rewrite (ProtectedDiv (Neg a) b) (Neg (ProtectedDiv a b)) :ruleset sign)
(rewrite (ProtectedDiv a (Neg b)) (Neg (ProtectedDiv a b)) :ruleset sign)
; B3  ODD functions pass a Neg through (raw Inv only — NOT ProtectedInv)
(rewrite (Sin (Neg x)) (Neg (Sin x)) :ruleset sign)
(rewrite (Tan (Neg x)) (Neg (Tan x)) :ruleset sign)
(rewrite (Tanh (Neg x)) (Neg (Tanh x)) :ruleset sign)
(rewrite (Pow3 (Neg x)) (Neg (Pow3 x)) :ruleset sign)
(rewrite (Inv (Neg x)) (Neg (Inv x)) :ruleset sign)
; ===== D. ABS OVER A SIGN-DEFINITE TERM — each removes the Abs =====
; |(-a)| = |a|
(rewrite (Abs (Neg a)) (Abs a) :ruleset sign)
; a > 0, c < 0:  |a/c| = a/(-c),  |a*c| = a*(-c),  |c/a| = (-c)/a.
; Feynman III.13.18 came back as 238.76*|(1/x3)/(-19)|*..: every column there
; is positive, the only sign in the term is the literal's, and SRBench's
; checker cannot see through the Abs.
(rule ((= e (Abs (ProtectedDiv a (Num c)))) (is-positive a) (< c 0.0))
      ((union e (ProtectedDiv a (Num (neg c))))) :ruleset sign)
(rule ((= e (Abs (Div a (Num c)))) (is-positive a) (< c 0.0))
      ((union e (Div a (Num (neg c))))) :ruleset sign)
(rule ((= e (Abs (ProtectedDiv (Num c) a))) (is-positive a) (< c 0.0))
      ((union e (ProtectedDiv (Num (neg c)) a))) :ruleset sign)
(rule ((= e (Abs (Div (Num c) a))) (is-positive a) (< c 0.0))
      ((union e (Div (Num (neg c)) a))) :ruleset sign)
(rule ((= e (Abs (Mul a (Num c)))) (is-positive a) (< c 0.0))
      ((union e (Mul a (Num (neg c))))) :ruleset sign)
(rule ((= e (Abs (Mul (Num c) a))) (is-positive a) (< c 0.0))
      ((union e (Mul (Num (neg c)) a))) :ruleset sign)

"#;

#[cfg(test)]
mod tests {
    use super::SIGN_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use crate::ruleset::identities::ALGEBRA_RULESET;
    use egglog::prelude::exprs;
    use egglog::EGraph;

    // Hard safety cap (mirrors sympy_mined.rs): a divergent rule stops here
    // instead of pegging the machine.
    const SAT_ITERS: u32 = 8;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        for prog in [MATH_DATATYPE, GUARD_RELATIONS, ALGEBRA_RULESET, SIGN_RULESET] {
            e.parse_and_run_program(None, prog).expect("program loads");
        }
        e.parse_and_run_program(None, "(unstable-combined-ruleset sign_all guards algebra sign)")
            .expect("combined ruleset");
        e
    }

    /// Bounded run, then the lowest-cost form as a string.
    fn simplify(input: &str) -> String {
        let mut e = egraph();
        e.parse_and_run_program(
            None,
            &format!("(let __r {input})\n(run-schedule (repeat {SAT_ITERS} (run sign_all)))"),
        )
        .expect("saturate");
        let (sort, value) = e.eval_expr(&exprs::var("__r")).expect("eval");
        e.extract_value_to_string(&sort, value).expect("extract").0
    }

    #[test]
    fn absorbers_fire() {
        let cases: &[(&str, &str)] = &[
            (r#"(Sub (Var "a") (Neg (Var "b")))"#, r#"(Add (Var "a") (Var "b"))"#),
            (r#"(Add (Var "a") (Neg (Var "b")))"#, r#"(Sub (Var "a") (Var "b"))"#),
            (r#"(Add (Neg (Var "a")) (Var "b"))"#, r#"(Sub (Var "b") (Var "a"))"#),
            (r#"(Neg (Sub (Var "a") (Var "b")))"#, r#"(Sub (Var "b") (Var "a"))"#),
            (r#"(Sub (Num 0.0) (Var "x"))"#, r#"(Neg (Var "x"))"#),
            (r#"(Neg (Num 4.0))"#, "(Num -4.0)"),
            (r#"(Neg (Add (Num -1.0) (Var "x")))"#, r#"(Sub (Num 1.0) (Var "x"))"#),
            (r#"(Mul (Neg (Var "a")) (Neg (Var "b")))"#, r#"(Mul (Var "a") (Var "b"))"#),
            (r#"(ProtectedDiv (Num -1.0) (Neg (Var "x")))"#, r#"(ProtectedDiv (Num 1.0) (Var "x"))"#),
            (r#"(Pow2 (Neg (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
            (r#"(Pow2 (Abs (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
            (r#"(Cos (Neg (Var "x")))"#, r#"(Cos (Var "x"))"#),
            (r#"(Cos (Abs (Var "x")))"#, r#"(Cos (Var "x"))"#),
            (r#"(ProtectedSqrt (Abs (Var "x")))"#, r#"(ProtectedSqrt (Var "x"))"#),
            (r#"(ProtectedLog (Abs (Var "x")))"#, r#"(ProtectedLog (Var "x"))"#),
            (r#"(ProtectedLog (Neg (Var "x")))"#, r#"(ProtectedLog (Var "x"))"#),
            (r#"(Abs (Pow2 (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
            (r#"(Abs (Sqrt (Var "x")))"#, r#"(Sqrt (Var "x"))"#),
            (r#"(Abs (ProtectedSqrt (Var "x")))"#, r#"(ProtectedSqrt (Var "x"))"#),
            (r#"(Abs (ProtectedExp (Var "x")))"#, r#"(ProtectedExp (Var "x"))"#),
            // gap row 21: Abs(Neg(Abs t)) -> Abs t
            (r#"(Abs (Neg (Abs (Tanh (Var "x")))))"#, r#"(Abs (Tanh (Var "x")))"#),
        ];
        let mut failures = Vec::new();
        for (input, expected) in cases {
            let got = simplify(input);
            if got != *expected {
                failures.push(format!("{input}\n  got:      {got}\n  expected: {expected}"));
            }
        }
        assert!(failures.is_empty(), "{} failed:\n{}", failures.len(), failures.join("\n"));
    }

    /// Float-outs only pay when an absorber sits above; these are the real
    /// hall-of-fame shapes (gaps.jsonl rows 2, 3, 5, 17).
    #[test]
    fn float_outs_reach_an_absorber() {
        let cases: &[(&str, &str)] = &[
            // row 2: (-a - b)^2 = (a + b)^2
            (
                r#"(Pow2 (Sub (Neg (Var "a")) (Var "b")))"#,
                r#"(Pow2 (Add (Var "a") (Var "b")))"#,
            ),
            // row 3: m - tan(1/(-u - v)) = m + tan(1/(u + v))
            (
                r#"(Sub (Var "m") (Tan (Inv (Sub (Neg (Var "u")) (Var "v")))))"#,
                r#"(Add (Var "m") (Tan (Inv (Add (Var "u") (Var "v")))))"#,
            ),
            // rows 5/26: (1 - s) * -(pinv o) = (s - 1) * pinv o
            (
                r#"(Mul (Sub (Num 1.0) (Var "s")) (Neg (ProtectedInv (Var "o"))))"#,
                r#"(Mul (Sub (Var "s") (Num 1.0)) (ProtectedInv (Var "o")))"#,
            ),
            // row 17: 0 - a*(b - c) = a*(c - b)
            (
                r#"(Sub (Num 0.0) (Mul (Var "a") (Sub (Var "b") (Var "c"))))"#,
                r#"(Mul (Var "a") (Sub (Var "c") (Var "b")))"#,
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(simplify(input), *expected, "{input}");
        }
    }

    /// ProtectedInv is NOT odd: protected_inv(-0) = 1 but -(protected_inv 0) = -1.
    /// No sign rule may touch it.
    #[test]
    fn protected_inv_of_neg_is_inert() {
        let input = r#"(ProtectedInv (Neg (Var "x")))"#;
        assert_eq!(simplify(input), input, "ProtectedInv(Neg x) was rewritten (unsound at 0)");
    }

    /// Numeric soundness: input and simplified form agree bit-for-bit (or are
    /// both NaN) at negative, zero, sub-threshold and overflow points. Covers
    /// the libm-dependent parity rules (Cos/Sin/Tan/Tanh) and every protected
    /// branch the sign rules cross.
    #[test]
    fn sign_rules_are_sound_on_the_evaluator() {
        let inputs = [
            r#"(Cos (Neg (Var "x")))"#,
            r#"(Cos (Abs (Var "x")))"#,
            r#"(Sub (Var "y") (Sin (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Tan (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Tanh (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Pow3 (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Inv (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (ProtectedDiv (Neg (Var "y")) (Var "x")))"#,
            r#"(Sub (Var "y") (ProtectedDiv (Var "y") (Neg (Var "x"))))"#,
            r#"(ProtectedDiv (Neg (Var "y")) (Neg (Var "x")))"#,
            r#"(ProtectedDiv (Num -1.0) (Neg (Var "x")))"#,
            r#"(ProtectedSqrt (Abs (Var "x")))"#,
            r#"(ProtectedSqrt (Neg (Exp (Var "x"))))"#,
            r#"(ProtectedLog (Abs (Var "x")))"#,
            r#"(ProtectedLog (Neg (Var "x")))"#,
            r#"(Abs (ProtectedExp (Neg (Exp (Var "x")))))"#,
            r#"(Abs (ProtectedSqrt (Var "x")))"#,
            r#"(Abs (Sqrt (Var "x")))"#,
            r#"(Pow2 (Sub (Neg (Var "x")) (Var "y")))"#,
            r#"(Mul (Sub (Num 1.0) (Var "y")) (Neg (ProtectedInv (Var "x"))))"#,
            r#"(Neg (Add (Num -1.0) (Var "x")))"#,
        ];
        // x: negative, zero, below the protected_div threshold, ordinary, and
        // large enough that Exp(x) overflows to +inf.
        let xs = [-2.3, 0.0, -1e-7, 0.7, 1000.0];
        for input in inputs {
            let simplified = simplify(input);
            for x in xs {
                let lookup = |n: &str| match n {
                    "x" => Some(x),
                    "y" => Some(-1.25),
                    _ => None,
                };
                let value = |math: &str| {
                    let mut e = EGraph::default();
                    e.parse_and_run_program(None, MATH_DATATYPE).expect("datatype");
                    e.parse_and_run_program(None, &format!("(let __v {math})")).expect("term");
                    let (s, v) = e.eval_expr(&exprs::var("__v")).expect("eval");
                    let (td, t, _) = e.extract_value(&s, v).expect("extract");
                    eval_term(&td, t, &lookup).expect("evaluates")
                };
                let (a, b) = (value(input), value(&simplified));
                assert!(
                    (a.is_nan() && b.is_nan()) || a == b,
                    "unsound at x={x}: {input} = {a} but {simplified} = {b}"
                );
            }
        }
    }
}
