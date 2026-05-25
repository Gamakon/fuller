//! The `Math` expression datatype and its egglog program.
//!
//! This is the shared substrate for the denoise core (Phase 1.2-1.4). Terms
//! are built in egglog surface syntax over the `Math` sort; constructors are
//! keyed on the *semantic* operation (BRIEF.md's `semantic_id`), never on a
//! pset-specific geppy name. The consumer maps geppy names -> these
//! constructors when it builds terms.
//!
//! Real-domain only: there is no complex domain here, by design. This is the
//! whole point of leaving sympy behind — the `re()/im()/sinh()` rejection
//! dance in hff's `_sympy_to_karva.py` (lines ~269-285) has no analogue.
//!
//! Constructors:
//!   (Num f64)        numeric literal
//!   (Var String)     free variable (a pset variable, by name)
//!   (Add a b) (Sub a b) (Mul a b) (Div a b)
//!   (Neg a)
//!   (Sin a) (Cos a) (Log a) (Exp a) (Sqrt a) (Abs a) (Tanh a)
//!   (Pow2 a) (Pow3 a) (Inv a)
//!
//! These cover the BRIEF.md semantic_id set
//! {add, sub, mul, div, neg, sin, cos, log, exp, sqrt, abs, tanh, pow2,
//!  pow3, inv, diff_sq}. `diff_sq(a,b) = (a-b)^2` is expressed as a rule, not
//! a constructor, so the e-graph can rewrite through it.

use egglog::EGraph;

/// The `Math` datatype declaration (egglog surface syntax).
pub const MATH_DATATYPE: &str = r#"
(datatype Math
    (Num f64)
    (Var String)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Div Math Math)
    (Neg Math)
    (Sin Math)
    (Cos Math)
    (Log Math)
    (Exp Math)
    (Sqrt Math)
    (Abs Math)
    (Tanh Math)
    (Tan Math)
    (Pow2 Math)
    (Pow3 Math)
    (Pow Math Math)
    (Inv Math))
"#;

/// Domain-guard relations shared across rule modules.
///
/// egglog 2.0 cannot guard on a free variable's sign directly, so we carry
/// domain facts as relations a rule can require via `:when`:
///   * `(is-positive m)` — `m` is known > 0  (guards log/exp identities)
///   * `(is-nonzero  m)` — `m` is known != 0  (guards div/inv identities)
///
/// Facts are introduced two ways: positive/non-zero numeric literals are
/// seeded at load, and rules may *propagate* the fact (e.g. `Exp x` is always
/// positive; a product of positives is positive). Modules add their own
/// propagation rules; this declares the relations and the literal seeds.
pub const GUARD_RELATIONS: &str = r#"
(relation is-positive (Math))
(relation is-nonzero (Math))

; Exp is positive on all reals -> seeds the log/exp domain.
(rule ((= e (Exp x))) ((is-positive e) (is-nonzero e)))
; positivity implies non-zero.
(rule ((is-positive m)) ((is-nonzero m)))
; product / quotient of positives is positive.
(rule ((is-positive a) (is-positive b) (= m (Mul a b))) ((is-positive m)))
(rule ((is-positive a) (is-positive b) (= m (Div a b))) ((is-positive m)))
"#;

/// Build a fresh e-graph with the `Math` datatype loaded (no rules yet).
pub fn math_egraph() -> Result<EGraph, String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, MATH_DATATYPE)
        .map_err(|e| format!("failed to load Math datatype: {e}"))?;
    Ok(egraph)
}

#[cfg(test)]
mod tests {
    use super::math_egraph;
    use egglog::prelude::exprs;

    #[test]
    fn datatype_loads_and_accepts_terms() {
        let mut egraph = math_egraph().expect("Math datatype loads");
        // A term using a representative spread of constructors must parse and
        // evaluate to a Math e-class.
        egraph
            .parse_and_run_program(
                None,
                r#"(let __t (Add (Mul (Var "x") (Num 1.0)) (Sqrt (Pow2 (Var "y")))))"#,
            )
            .expect("term parses");
        let (sort, _value) = egraph
            .eval_expr(&exprs::var("__t"))
            .expect("term evaluates to a value");
        assert_eq!(sort.name(), "Math");
    }
}
