//! Phase 1.3: real-domain evaluator for `Math` terms.
//!
//! Replaces sympy's `lambdify`. Walks an egglog-extracted `Term` over the
//! `Math` datatype and evaluates it row-by-row against numeric data, in the
//! real domain only: `sqrt(negative)`, `log(<= 0)` and division by zero all
//! return `NaN`.
//!
//! No "protection" via Abs wrapping — the evaluator reports NaN and the caller
//! decides. There is no complex domain here, which is the entire reason this
//! crate exists instead of sympy.

use egglog::{Term, TermDag, TermId};

/// Error from evaluating a `Math` term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// A `Var` in the term has no binding in the supplied environment.
    UnboundVar(String),
    /// A constructor / arity we don't recognise as a `Math` op.
    BadNode(String),
    /// The term nests deeper than [`crate::MAX_EXPR_DEPTH`] — refused before
    /// the recursive walk can overflow the stack (rayon workers have small
    /// stacks, and an overflow is an uncatchable abort).
    TooDeep,
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnboundVar(n) => write!(f, "unbound variable {n:?}"),
            EvalError::BadNode(n) => write!(f, "unevaluable node {n:?}"),
            EvalError::TooDeep => {
                write!(f, "term deeper than MAX_EXPR_DEPTH ({})", crate::MAX_EXPR_DEPTH)
            }
        }
    }
}

impl std::error::Error for EvalError {}

/// Resolves a variable name to its value for the current row. A closure
/// `|name| -> Option<f64>` is the intended implementation; over tabular data
/// the caller rebinds it per row.
pub type Env<'a> = dyn Fn(&str) -> Option<f64> + 'a;

/// Evaluate `root` in `termdag`, resolving variables through `env`. Returns the
/// real value (which may be NaN for out-of-domain operations) or an error for
/// structural problems (unbound var, unknown node).
pub fn eval_term(termdag: &TermDag, root: TermId, env: &Env) -> Result<f64, EvalError> {
    eval_inner(termdag, root, env, 0)
}

fn eval_inner(termdag: &TermDag, id: TermId, env: &Env, depth: usize) -> Result<f64, EvalError> {
    if depth > crate::MAX_EXPR_DEPTH {
        return Err(EvalError::TooDeep);
    }
    match termdag.get(id) {
        Term::Lit(lit) => match lit {
            egglog::ast::Literal::Float(of) => Ok(of.into_inner()),
            egglog::ast::Literal::Int(i) => Ok(*i as f64),
            other => Err(EvalError::BadNode(format!("{other:?}"))),
        },
        Term::Var(name) => env(name).ok_or_else(|| EvalError::UnboundVar(name.clone())),
        Term::App(op, args) => eval_app(termdag, op, args, env, depth),
    }
}

fn eval_app(
    termdag: &TermDag,
    op: &str,
    args: &[TermId],
    env: &Env,
    depth: usize,
) -> Result<f64, EvalError> {
    // Helper to evaluate the nth child.
    let child = |i: usize| -> Result<f64, EvalError> {
        eval_inner(termdag, args[i], env, depth + 1)
    };

    let val = match (op, args.len()) {
        // Leaves wrapped as constructors.
        ("Num", 1) => child(0)?,
        ("Var", 1) => {
            // (Var "name") — the name is a String literal child.
            match termdag.get(args[0]) {
                Term::Lit(egglog::ast::Literal::String(s)) => {
                    env(s).ok_or_else(|| EvalError::UnboundVar(s.clone()))?
                }
                other => return Err(EvalError::BadNode(format!("Var child {other:?}"))),
            }
        }
        // Binary ops.
        ("Add", 2) => child(0)? + child(1)?,
        ("Sub", 2) => child(0)? - child(1)?,
        ("Mul", 2) => child(0)? * child(1)?,
        ("Div", 2) => {
            let (a, b) = (child(0)?, child(1)?);
            if b == 0.0 { f64::NAN } else { a / b }
        }
        // Unary ops.
        ("Neg", 1) => -child(0)?,
        ("Sin", 1) => child(0)?.sin(),
        ("Cos", 1) => child(0)?.cos(),
        ("Tan", 1) => {
            // tan = sin/cos; NaN at the asymptote (cos == 0).
            let a = child(0)?;
            let c = a.cos();
            if c == 0.0 { f64::NAN } else { a.sin() / c }
        }
        ("Exp", 1) => child(0)?.exp(),
        ("Tanh", 1) => child(0)?.tanh(),
        ("Abs", 1) => child(0)?.abs(),
        ("Pow2", 1) => { let a = child(0)?; a * a }
        ("Pow3", 1) => { let a = child(0)?; a * a * a }
        ("Pow", 2) => {
            // a^b in the real domain. f64::powf already yields NaN for a
            // negative base with a non-integer exponent, which is exactly the
            // real-domain rule (no complex branch). But 0^negative is a
            // division by zero (powf would give +inf), and the crate's
            // div0 contract is NaN — keep Pow consistent with Div/Inv, or a
            // rewrite like Pow(x,-1) <-> Inv(x) would put a +inf-valued and a
            // NaN-valued member in the same e-class.
            let (a, b) = (child(0)?, child(1)?);
            if a == 0.0 && b < 0.0 {
                f64::NAN
            } else {
                a.powf(b)
            }
        }
        ("Log", 1) => {
            let a = child(0)?;
            if a <= 0.0 { f64::NAN } else { a.ln() }
        }
        ("Sqrt", 1) => {
            let a = child(0)?;
            if a < 0.0 { f64::NAN } else { a.sqrt() }
        }
        ("Inv", 1) => {
            let a = child(0)?;
            if a == 0.0 { f64::NAN } else { 1.0 / a }
        }
        // Inverse trig in the real domain: NaN outside [-1, 1] (NaN in -> NaN
        // out, since NaN fails the comparison).
        ("Asin", 1) => {
            let a = child(0)?;
            if a.abs() <= 1.0 { a.asin() } else { f64::NAN }
        }
        ("Acos", 1) => {
            let a = child(0)?;
            if a.abs() <= 1.0 { a.acos() } else { f64::NAN }
        }
        // Protected ops — match the SR engine's pset semantics EXACTLY. These
        // are total (never NaN on the engine's domain), distinct from the raw
        // ops above.
        // The Protected* ops ARE the engine's primitives, transcribed from
        // hff_sr_engine.py / the recovery notebook line for line — including
        // what each does with a NON-FINITE input, which is where an earlier
        // version of this file differed from the code it claimed to match:
        // it had sqrt(inf)=inf (engine: 0.0), log(0)=-inf (engine: +inf) and
        // exp(-inf)=0.0 (engine: +inf). Each of those turns a candidate the
        // engine rejects into one fuller scores, or the reverse.
        //
        //   protected_sqrt(x) = sqrt(|x|) if isfinite(x) else 0.0
        ("ProtectedSqrt", 1) => {
            let a = child(0)?;
            if a.is_finite() { a.abs().sqrt() } else { 0.0 }
        }
        //   protected_log(x)  = +inf if not isfinite(x) or x == 0 else ln(|x|)
        ("ProtectedLog", 1) => {
            let a = child(0)?;
            if !a.is_finite() || a == 0.0 { f64::INFINITY } else { a.abs().ln() }
        }
        //   protected_exp(x)  = +inf if not isfinite(x) else exp(x), and
        //   +inf on overflow (f64::exp already returns +inf there). NOT
        //   exp(min(x, 700)): a large-finite return diverges from the engine.
        ("ProtectedExp", 1) => {
            let a = child(0)?;
            if a.is_finite() { a.exp() } else { f64::INFINITY }
        }
        ("ProtectedInv", 1) => {
            let a = child(0)?;
            if a == 0.0 { 1.0 } else { 1.0 / a } // 1/x if x!=0 else 1
        }
        //   protected_div_zero(a, b) = 0 if |b| < 1e-6 else a / b
        // The threshold is part of the function: with b = -1e-7 the engine
        // returns 0 and a `b == 0` guard returns a / -1e-7.
        ("ProtectedDiv", 2) => {
            let (a, b) = (child(0)?, child(1)?);
            if b.abs() < 1e-6 { 0.0 } else { a / b }
        }
        // These two have no counterpart in hff_sr_engine.py: they are defined
        // here, on protected_sqrt's convention for a non-finite input (0.0).
        //   protected_asin(x) = asin(clamp(x, -1, 1)) if isfinite(x) else 0.0
        ("ProtectedAsin", 1) => {
            let a = child(0)?;
            if a.is_finite() { a.clamp(-1.0, 1.0).asin() } else { 0.0 }
        }
        //   protected_acos(x) = acos(clamp(x, -1, 1)) if isfinite(x) else 0.0
        ("ProtectedAcos", 1) => {
            let a = child(0)?;
            if a.is_finite() { a.clamp(-1.0, 1.0).acos() } else { 0.0 }
        }
        _ => return Err(EvalError::BadNode(format!("{op}/{}", args.len()))),
    };
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::{eval_term, EvalError};
    use crate::expr::MATH_DATATYPE;
    use egglog::prelude::exprs;
    use egglog::EGraph;

    /// Extract `input` from a fresh Math e-graph (no rules) into a TermDag,
    /// then evaluate it under `env`.
    fn eval(input: &str, env: &[(String, f64)]) -> Result<f64, EvalError> {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, MATH_DATATYPE).unwrap();
        egraph
            .parse_and_run_program(None, &format!("(let __e {input})"))
            .unwrap();
        let (sort, value) = egraph.eval_expr(&exprs::var("__e")).unwrap();
        let (termdag, term, _cost) = egraph.extract_value(&sort, value).unwrap();
        eval_term(&termdag, term, &|name: &str| {
            env.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
        })
    }

    fn env(pairs: &[(&str, f64)]) -> Vec<(String, f64)> {
        pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect()
    }

    #[test]
    fn arithmetic_matches_hand_computed() {
        let e = env(&[("x", 3.0), ("y", 4.0)]);
        // x*y + 1 = 13
        assert_eq!(
            eval(r#"(Add (Mul (Var "x") (Var "y")) (Num 1.0))"#, &e).unwrap(),
            13.0
        );
        // sqrt(x^2 + y^2) = 5
        assert_eq!(
            eval(r#"(Sqrt (Add (Pow2 (Var "x")) (Pow2 (Var "y"))))"#, &e).unwrap(),
            5.0
        );
        // x - y = -1, neg -> 1
        assert_eq!(eval(r#"(Neg (Sub (Var "x") (Var "y")))"#, &e).unwrap(), 1.0);
    }

    #[test]
    fn out_of_domain_is_nan_not_protected() {
        let e = env(&[("x", -4.0)]);
        assert!(eval(r#"(Sqrt (Var "x"))"#, &e).unwrap().is_nan(), "sqrt(-4) is NaN");
        assert!(eval(r#"(Log (Var "x"))"#, &e).unwrap().is_nan(), "log(-4) is NaN");
        let z = env(&[("x", 0.0)]);
        assert!(eval(r#"(Log (Var "x"))"#, &z).unwrap().is_nan(), "log(0) is NaN");
        assert!(eval(r#"(Inv (Var "x"))"#, &z).unwrap().is_nan(), "1/0 is NaN");
        assert!(
            eval(r#"(Div (Num 1.0) (Var "x"))"#, &z).unwrap().is_nan(),
            "1/0 via Div is NaN"
        );
    }

    #[test]
    fn transcendental_values() {
        let e = env(&[("t", 0.0)]);
        assert_eq!(eval(r#"(Cos (Var "t"))"#, &e).unwrap(), 1.0);
        assert_eq!(eval(r#"(Sin (Var "t"))"#, &e).unwrap(), 0.0);
        assert_eq!(eval(r#"(Exp (Var "t"))"#, &e).unwrap(), 1.0);
    }

    #[test]
    fn protected_ops_match_engine_semantics() {
        // protected_sqrt(|neg|), protected_log(|neg|)
        let n = env(&[("x", -4.0)]);
        assert_eq!(eval(r#"(ProtectedSqrt (Var "x"))"#, &n).unwrap(), 2.0); // sqrt|−4|=2
        assert_eq!(eval(r#"(ProtectedLog (Var "x"))"#, &n).unwrap(), 4.0_f64.ln()); // log|−4|
        // protected_inv(0)=1, protected_div(a,0)=0
        let z = env(&[("x", 0.0)]);
        assert_eq!(eval(r#"(ProtectedInv (Var "x"))"#, &z).unwrap(), 1.0);
        assert_eq!(eval(r#"(ProtectedDiv (Num 5.0) (Var "x"))"#, &z).unwrap(), 0.0);
        // protected_exp: normal below overflow, +inf above (matches engine inf,
        // NOT a capped large-finite value).
        let small = env(&[("x", 1.0)]);
        assert_eq!(eval(r#"(ProtectedExp (Var "x"))"#, &small).unwrap(), 1.0_f64.exp());
        let big = env(&[("x", 1000.0)]);
        assert!(eval(r#"(ProtectedExp (Var "x"))"#, &big).unwrap().is_infinite(),
            "protected_exp(1000) must be +inf, not a capped finite");
    }

    #[test]
    fn inverse_trig_is_real_domain_and_the_protected_forms_clamp() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_6, PI};
        let at = |expr: &str, x: f64| eval(expr, &env(&[("x", x)])).unwrap();
        let (asin, acos) = (r#"(Asin (Var "x"))"#, r#"(Acos (Var "x"))"#);
        let (pasin, pacos) = (r#"(ProtectedAsin (Var "x"))"#, r#"(ProtectedAcos (Var "x"))"#);
        for (x, s, c) in [(-1.0, -FRAC_PI_2, PI), (0.0, 0.0, FRAC_PI_2), (0.5, FRAC_PI_6, FRAC_PI_3), (1.0, FRAC_PI_2, 0.0)] {
            assert!((at(asin, x) - s).abs() < 1e-15, "asin({x}) = {}", at(asin, x));
            assert!((at(acos, x) - c).abs() < 1e-15, "acos({x}) = {}", at(acos, x));
            // Inside [-1, 1] the protected form IS the raw one.
            assert_eq!(at(pasin, x), at(asin, x));
            assert_eq!(at(pacos, x), at(acos, x));
        }
        // Outside the domain: raw is NaN, protected clamps to the nearer end.
        for x in [-3.0, -1.0000001, 1.0000001, 2.0] {
            assert!(at(asin, x).is_nan(), "asin({x}) is NaN");
            assert!(at(acos, x).is_nan(), "acos({x}) is NaN");
        }
        assert_eq!(at(pasin, 2.0), FRAC_PI_2);
        assert_eq!(at(pasin, -3.0), -FRAC_PI_2);
        assert_eq!(at(pacos, 2.0), 0.0);
        assert_eq!(at(pacos, -3.0), PI);
        // A non-finite argument is 0.0, protected_sqrt's convention — for
        // +inf, -inf and NaN alike (so ProtectedAcos(-inf) is 0, not pi).
        let e = env(&[("x", 1000.0)]); // Exp(x) = +inf
        for arg in [r#"(Exp (Var "x"))"#, r#"(Neg (Exp (Var "x")))"#, r#"(Sqrt (Neg (Var "x")))"#] {
            assert_eq!(eval(&format!("(ProtectedAsin {arg})"), &e).unwrap(), 0.0, "{arg}");
            assert_eq!(eval(&format!("(ProtectedAcos {arg})"), &e).unwrap(), 0.0, "{arg}");
            assert!(eval(&format!("(Asin {arg})"), &e).unwrap().is_nan(), "{arg}");
            assert!(eval(&format!("(Acos {arg})"), &e).unwrap().is_nan(), "{arg}");
        }
    }

    #[test]
    fn pow_zero_to_negative_is_nan_like_div0() {
        // 0^-1 is a division by zero; the crate contract is NaN, and rational
        // rules rewrite Pow(x,-1) <-> Inv(x), so the two must agree at x=0.
        let z = env(&[("x", 0.0)]);
        assert!(
            eval(r#"(Pow (Var "x") (Num -1.0))"#, &z).unwrap().is_nan(),
            "0^-1 must be NaN, matching Inv(0)"
        );
        assert!(eval(r#"(Pow (Var "x") (Num -2.0))"#, &z).unwrap().is_nan());
    }

    #[test]
    fn protected_ops_on_non_finite_input_match_the_engine() {
        // Transcribed from hff_sr_engine.py: protected_exp and protected_log
        // return +inf for ANY non-finite input, protected_sqrt returns 0.0.
        // In particular protected_exp(-inf) is +inf, not IEEE's 0.0 — an
        // earlier version of this test asserted 0.0 against a reading of the
        // engine that its code does not support.
        let e = env(&[("x", 1000.0)]); // Exp(x) = +inf, Neg(Exp(x)) = -inf
        for arg in [r#"(Exp (Var "x"))"#, r#"(Neg (Exp (Var "x")))"#] {
            assert_eq!(eval(&format!("(ProtectedExp {arg})"), &e).unwrap(), f64::INFINITY);
            assert_eq!(eval(&format!("(ProtectedLog {arg})"), &e).unwrap(), f64::INFINITY);
            assert_eq!(eval(&format!("(ProtectedSqrt {arg})"), &e).unwrap(), 0.0);
        }
        // NaN is non-finite too: Sqrt(-4) = NaN.
        let n = env(&[("x", -4.0)]);
        assert_eq!(eval(r#"(ProtectedExp (Sqrt (Var "x")))"#, &n).unwrap(), f64::INFINITY);
        assert_eq!(eval(r#"(ProtectedSqrt (Sqrt (Var "x")))"#, &n).unwrap(), 0.0);
        // protected_log(0) is +inf, not ln|0| = -inf.
        let z = env(&[("x", 0.0)]);
        assert_eq!(eval(r#"(ProtectedLog (Var "x"))"#, &z).unwrap(), f64::INFINITY);
    }

    #[test]
    fn protected_div_threshold_is_part_of_the_function() {
        // protected_div_zero: 0 if |b| < 1e-6 else a / b.
        for (b, want) in [(0.0, 0.0), (-1e-7, 0.0), (9e-7, 0.0), (2e-6, 2.5e6), (2.0, 2.5)] {
            let e = env(&[("b", b)]);
            let got = eval(r#"(ProtectedDiv (Num 5.0) (Var "b"))"#, &e).unwrap();
            assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "5/{b}: {got} vs {want}");
        }
    }

    #[test]
    fn unbound_var_errors() {
        let e = env(&[("x", 1.0)]);
        assert_eq!(
            eval(r#"(Add (Var "x") (Var "z"))"#, &e),
            Err(EvalError::UnboundVar("z".to_string()))
        );
    }
}
