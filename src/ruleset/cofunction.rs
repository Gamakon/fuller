//! The `cofunction` ruleset — the common pi/2 phase-shift / parity identities,
//! kept SEPARATE from `trig` and loaded into the linter's `standard` tables so
//! the form beam can apply them.
//!
//! Why separate: `trig` carries CANONICALISERS (`Sub -> Add + Neg`, `Neg ->
//! Mul -1`) that fight `sign` and, co-saturated with `rational`'s square
//! expansion, explode the e-graph (the non-confluence CLAUDE.md warns about).
//! These rules do NOT: every one strictly CONSUMES a shift or a sign — a shifted
//! trig becomes an unshifted cofunction, the shift constant disappears, node
//! count does not grow. So they are safe in the standard beam.
//!
//! Why they matter: the SR engine frequently fits `cos` as a shifted `sin` (and
//! vice versa). A fitted `sin(pi/2 + x)/tan(y)` IS the law `cos(x)*cot(y)`, but
//! only these identities collapse it — without them the form is a near-miss the
//! e-graph cannot simplify and SRBench (whose own simplify also does not do
//! `sin(pi/2+x) -> cos(x)`) rejects. pi/2 = 1.5707963267948966 (the value the
//! engine and snap use). All are real-domain identities for every real x.

/// The `cofunction` ruleset: pi/2 phase shifts and trig parity, all contractive.
pub const COFUNCTION_RULESET: &str = r#"
(ruleset cofunction)

; ---- sin(pi/2 +/- x) = cos(x) ----
(rewrite (Sin (Add (Num 1.5707963267948966) x)) (Cos x) :ruleset cofunction)
(rewrite (Sin (Add x (Num 1.5707963267948966))) (Cos x) :ruleset cofunction)
(rewrite (Sin (Sub (Num 1.5707963267948966) x)) (Cos x) :ruleset cofunction)
; pi/2 - x written as pi/2 + (-1)*x (after a sign canonicalisation)
(rewrite (Sin (Add (Num 1.5707963267948966) (Mul (Num -1.0) x))) (Cos x) :ruleset cofunction)

; ---- cos(pi/2 - x) = sin(x),  cos(pi/2 + x) = -sin(x) ----
(rewrite (Cos (Sub (Num 1.5707963267948966) x)) (Sin x) :ruleset cofunction)
(rewrite (Cos (Add (Num 1.5707963267948966) (Mul (Num -1.0) x))) (Sin x) :ruleset cofunction)
(rewrite (Cos (Add (Num 1.5707963267948966) x)) (Mul (Num -1.0) (Sin x)) :ruleset cofunction)
(rewrite (Cos (Add x (Num 1.5707963267948966))) (Mul (Num -1.0) (Sin x)) :ruleset cofunction)

; ---- parity: sin(-x) = -sin(x), cos(-x) = cos(x)  (-x written as (-1)*x) ----
(rewrite (Sin (Mul (Num -1.0) x)) (Mul (Num -1.0) (Sin x)) :ruleset cofunction)
(rewrite (Cos (Mul (Num -1.0) x)) (Cos x) :ruleset cofunction)
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::EGraph;

    fn proves(input: &str, target: &str) -> bool {
        let mut e = EGraph::default();
        e.parse_and_run_program(None, &format!("{MATH_DATATYPE}\n{GUARD_RELATIONS}\n{COFUNCTION_RULESET}")).expect("load");
        let prog = format!("(let __i {input})\n(let __t {target})\n(run-schedule (repeat 6 (run cofunction)))\n(check (= __i __t))");
        e.parse_and_run_program(None, &prog).is_ok()
    }

    #[test]
    fn sin_shift_is_cos() {
        assert!(proves(r#"(Sin (Add (Num 1.5707963267948966) (Var "x")))"#, r#"(Cos (Var "x"))"#));
        assert!(proves(r#"(Sin (Sub (Num 1.5707963267948966) (Var "x")))"#, r#"(Cos (Var "x"))"#));
    }

    #[test]
    fn cos_shift_is_sin() {
        assert!(proves(r#"(Cos (Sub (Num 1.5707963267948966) (Var "x")))"#, r#"(Sin (Var "x"))"#));
    }

    #[test]
    fn parity() {
        assert!(proves(r#"(Cos (Mul (Num -1.0) (Var "x")))"#, r#"(Cos (Var "x"))"#));
        assert!(proves(r#"(Sin (Mul (Num -1.0) (Var "x")))"#, r#"(Mul (Num -1.0) (Sin (Var "x")))"#));
    }

    #[test]
    fn does_not_prove_falsehood() {
        // sin(x) is not cos(x) without a shift.
        assert!(!proves(r#"(Sin (Var "x"))"#, r#"(Cos (Var "x"))"#));
    }

    #[test]
    fn the_shearflow_shape() {
        // sin(pi/2 + x)/tan(y) == cos(x)/tan(y): the strogatz_shearflow1 recovery.
        assert!(proves(
            r#"(ProtectedDiv (Sin (Add (Num 1.5707963267948966) (Var "x"))) (Tan (Var "y")))"#,
            r#"(ProtectedDiv (Cos (Var "x")) (Tan (Var "y")))"#
        ));
    }
}
