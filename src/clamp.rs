//! WHICH PROTECTED OPERATORS A MODEL OWES ITS FIT TO (docs/PLAN_outstanding_2026-10-21.md, T51).
//!
//! `ProtectedAsin` / `ProtectedAcos` clamp their argument to [-1, 1], `ProtectedLog` is `ln |x|`
//! and `ProtectedSqrt` is `sqrt |x|`. Where the data keeps the argument inside the raw domain they
//! are the raw operators; where it leaves the domain on any row, the model fits through the clamp
//! and not through the function SRBench's reader will see (`asin` of 3 is complex there). This
//! reports those nodes and, where the clamp makes a node take ONE value on every row, folds the
//! node to that constant, which is what it is on the data. It ranks and flags; it changes nothing
//! a caller does not ask it to.

use crate::gpu_eval::Op;
use crate::lint::node::{Compiled, Tree};

/// One protected node whose argument leaves the raw domain.
#[derive(Debug, Clone, PartialEq)]
pub struct Reliance {
    pub op: Op,
    /// Child indices from the root.
    pub path: Vec<u8>,
    /// Rows the argument is outside the raw domain on.
    pub rows_outside: usize,
    /// The constant the node is on every row, when it is one.
    pub constant: Option<f64>,
}

/// What [`clamp_report`] found.
#[derive(Debug, Clone)]
pub struct ClampReport {
    /// Protected Asin/Acos/Log/Sqrt nodes in the model.
    pub protected: usize,
    /// Those outside their raw domain on some row (a node inside a folded one is not counted).
    pub relies: Vec<Reliance>,
    /// The model with every such node that is one constant on the data replaced by it.
    pub folded: Tree,
}

impl ClampReport {
    /// The model owes its fit to a clamp.
    pub fn relies_on_clamping(&self) -> bool {
        !self.relies.is_empty()
    }
}

fn outside(op: Op, x: f64) -> bool {
    match op {
        Op::ProtectedAsin | Op::ProtectedAcos => x.is_finite() && x.abs() > 1.0,
        Op::ProtectedLog => x <= 0.0,
        Op::ProtectedSqrt => x < 0.0,
        _ => false,
    }
}

fn values(tree: &Tree, rows: &[Vec<(String, f64)>]) -> Result<Vec<f64>, String> {
    let c = Compiled::new(tree);
    rows.iter().map(|r| c.eval(r)).collect()
}

/// The constant `tree` is on every row, when it is finite and the same to 1e-12 of its size.
fn constant_on(tree: &Tree, rows: &[Vec<(String, f64)>]) -> Option<f64> {
    let v = values(tree, rows).ok()?;
    let (lo, hi) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
    let steady = !v.is_empty() && v.iter().all(|x| x.is_finite()) && hi - lo <= 1e-12 * lo.abs().max(hi.abs()).max(1.0);
    steady.then_some((lo + hi) / 2.0)
}

/// Every protected Asin/Acos/Log/Sqrt of `tree` whose argument is outside its raw domain on some row of `rows`,
/// and `tree` with the ones that are a constant there folded to it. The outermost node is judged first, and
/// what is inside a folded node is not looked at again.
///
/// # Errors
/// When an argument cannot be evaluated.
pub fn clamp_report(tree: &Tree, rows: &[Vec<(String, f64)>]) -> Result<ClampReport, String> {
    let (mut protected, mut relies) = (0usize, Vec::new());
    let mut folded = tree.clone();
    let mut gone: Vec<Vec<u8>> = Vec::new();
    for path in tree.positions() {
        if gone.iter().any(|g| path.starts_with(g)) {
            continue;
        }
        let Some(node @ Tree::App(op, kids)) = tree.at(&path) else { continue };
        if !matches!(op, Op::ProtectedAsin | Op::ProtectedAcos | Op::ProtectedLog | Op::ProtectedSqrt) {
            continue;
        }
        protected += 1;
        let arg = values(&kids[0], rows)?;
        let rows_outside = arg.iter().filter(|x| outside(*op, **x)).count();
        if rows_outside == 0 {
            continue;
        }
        let constant = constant_on(node, rows);
        if let Some(c) = constant {
            folded = folded.replaced(&path, Tree::Num(c));
            gone.push(path.clone());
        }
        relies.push(Reliance { op: *op, path, rows_outside, constant });
    }
    // A node that takes one value because a clamp beneath it does (x / acos(2) with the protected division giving 0) is that
    // value on the data too: climb from each folded node while its parent is one constant.
    for path in &gone {
        for depth in (0..path.len()).rev() {
            let parent = &path[..depth];
            let Some(node) = folded.at(parent) else { break };
            match constant_on(node, rows) {
                Some(c) => folded = folded.replaced(parent, Tree::Num(c)),
                None => break,
            }
        }
    }
    Ok(ClampReport { protected, relies, folded })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(xs: &[f64]) -> Vec<Vec<(String, f64)>> {
        xs.iter().map(|x| vec![("x_0".to_string(), *x)]).collect()
    }

    fn report(math: &str, xs: &[f64]) -> ClampReport {
        clamp_report(&Tree::parse(math).expect("parses"), &rows(xs)).expect("evaluates")
    }

    /// An Acos whose argument is beyond 1 on every row is the constant 0, and the model is what it is without it.
    #[test]
    fn an_acos_beyond_one_on_every_row_folds_to_its_constant() {
        let r = report(r#"(Add (Var "x_0") (ProtectedAcos (Var "x_0")))"#, &[2.0, 3.0, 5.0]);
        assert!(r.relies_on_clamping());
        assert_eq!((r.protected, r.relies.len()), (1, 1));
        assert_eq!(r.relies[0].constant, Some(0.0));
        assert_eq!(r.relies[0].rows_outside, 3);
        assert_eq!(r.folded, Tree::parse(r#"(Add (Var "x_0") (Num 0.0))"#).unwrap());
    }

    /// Outside on some rows only: reported, not folded (it is not one constant).
    #[test]
    fn a_clamp_on_some_rows_is_reported_and_left() {
        let r = report(r#"(ProtectedAsin (Var "x_0"))"#, &[0.5, 2.0]);
        assert!(r.relies_on_clamping());
        assert_eq!(r.relies[0].rows_outside, 1);
        assert_eq!(r.relies[0].constant, None);
        assert_eq!(r.folded, Tree::parse(r#"(ProtectedAsin (Var "x_0"))"#).unwrap());
    }

    /// Inside the raw domain on every row a protected operator is the raw one: nothing to flag.
    #[test]
    fn a_protected_operator_inside_its_domain_is_not_flagged() {
        let r = report(r#"(Add (ProtectedAsin (Var "x_0")) (ProtectedSqrt (Var "x_0")))"#, &[0.1, 0.5, 0.9]);
        assert_eq!(r.protected, 2);
        assert!(!r.relies_on_clamping());
    }

    /// Log at 0 and Sqrt below 0 are outside; Log of a negative number is (ln |x|) not a constant.
    #[test]
    fn log_and_sqrt_are_judged_on_their_own_domains() {
        let r = report(r#"(Add (ProtectedLog (Var "x_0")) (ProtectedSqrt (Var "x_0")))"#, &[-1.0, 4.0]);
        assert_eq!(r.relies.len(), 2);
        assert!(r.relies.iter().all(|x| x.rows_outside == 1));
        let r = report(r#"(ProtectedLog (Var "x_0"))"#, &[0.0, 4.0]);
        assert_eq!(r.relies[0].rows_outside, 1, "ln 0 is outside the raw domain");
    }

    /// A quotient by a node a clamp makes 0 is the protected division's 0: the dead term folds with it.
    #[test]
    fn a_term_a_clamp_kills_folds_whole() {
        let r = report(r#"(Add (Var "x_0") (ProtectedDiv (Var "x_0") (ProtectedAcos (Var "x_0"))))"#, &[2.0, 3.0, 5.0]);
        assert_eq!(r.relies.len(), 1);
        assert_eq!(r.folded, Tree::parse(r#"(Add (Var "x_0") (Num 0.0))"#).unwrap());
    }

    /// The inner clamp folds to 0, and the parent then reads that constant (inside its own domain).
    #[test]
    fn a_folded_node_is_the_constant_its_parent_sees() {
        let r = report(r#"(ProtectedAsin (ProtectedAcos (Add (Var "x_0") (Num 5.0))))"#, &[1.0, 2.0]);
        // Acos of x+5 beyond 1 is the constant 0; asin of that constant is inside its domain: only the inner one is outside
        assert_eq!(r.relies.len(), 1);
        assert_eq!(r.relies[0].op, Op::ProtectedAcos);
        // the parent, asin of that constant, is itself the constant 0 on the data, and folds too
        assert_eq!(r.folded, Tree::parse(r#"(Num 0.0)"#).unwrap());
    }
}
