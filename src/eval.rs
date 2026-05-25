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
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnboundVar(n) => write!(f, "unbound variable {n:?}"),
            EvalError::BadNode(n) => write!(f, "unevaluable node {n:?}"),
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
    eval_inner(termdag, root, env)
}

fn eval_inner(termdag: &TermDag, id: TermId, env: &Env) -> Result<f64, EvalError> {
    match termdag.get(id) {
        Term::Lit(lit) => match lit {
            egglog::ast::Literal::Float(of) => Ok(of.into_inner()),
            egglog::ast::Literal::Int(i) => Ok(*i as f64),
            other => Err(EvalError::BadNode(format!("{other:?}"))),
        },
        Term::Var(name) => env(name).ok_or_else(|| EvalError::UnboundVar(name.clone())),
        Term::App(op, args) => eval_app(termdag, op, args, env),
    }
}

fn eval_app(
    termdag: &TermDag,
    op: &str,
    args: &[TermId],
    env: &Env,
) -> Result<f64, EvalError> {
    // Helper to evaluate the nth child.
    let child = |i: usize| -> Result<f64, EvalError> { eval_inner(termdag, args[i], env) };

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
            // real-domain rule (no complex branch).
            child(0)?.powf(child(1)?)
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
    fn unbound_var_errors() {
        let e = env(&[("x", 1.0)]);
        assert_eq!(
            eval(r#"(Add (Var "x") (Var "z"))"#, &e),
            Err(EvalError::UnboundVar("z".to_string()))
        );
    }
}
