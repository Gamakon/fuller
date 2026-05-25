//! Phase 1.0 calibration: drive egglog 2.0 end-to-end from Rust.
//!
//! We define a tiny boolean-algebra datatype and a 5-rule ruleset, then prove
//! we can: build a term -> saturate -> extract the lowest-cost equivalent form
//! -> read it back as a Rust string. If this round-trips on the hand-written
//! cases in the tests below, the egglog FFI substrate works on this machine
//! and the rest of the BRIEF.md plan is unblocked.
//!
//! egglog 2.0 notes (verified against the 2.0.0 crate source):
//!   * `EGraph::parse_and_run_program` is the stable textual entry point.
//!   * Extraction is `Extractor::compute_costs_from_rootsorts(..)` followed by
//!     `extract_best` / `extract_variants`, using a `CostModel`
//!     (`TreeAdditiveCostModel` is the built-in structural cost).

use egglog::EGraph;
use egglog::prelude::exprs;

/// The boolean-algebra program: one datatype `B` plus a 5-rule ruleset
/// `bool_algebra`. Pure rewrites only — no commutativity/associativity as
/// bare rewrites (egglog handles those via e-class merging; encoding them is
/// the classic blow-up mistake called out in BRIEF.md).
///
/// Rules:
///   1. identity:        (And x T) = x
///   2. double negation: (Not (Not x)) = x
///   3. De Morgan a:     (Not (And x y)) = (Or (Not x) (Not y))
///   4. De Morgan b:     (Not (Or x y))  = (And (Not x) (Not y))
///   5. absorption:      (Or x (And x y)) = x
const BOOL_ALGEBRA_PROGRAM: &str = r#"
(datatype B
    (T)
    (F)
    (Var String)
    (Not B)
    (And B B)
    (Or B B))

(ruleset bool_algebra)

; 1. identity
(rewrite (And x (T)) x :ruleset bool_algebra)
; 2. double negation
(rewrite (Not (Not x)) x :ruleset bool_algebra)
; 3. De Morgan (over And)
(rewrite (Not (And x y)) (Or (Not x) (Not y)) :ruleset bool_algebra)
; 4. De Morgan (over Or)
(rewrite (Not (Or x y)) (And (Not x) (Not y)) :ruleset bool_algebra)
; 5. absorption
(rewrite (Or x (And x y)) x :ruleset bool_algebra)
"#;

/// Build a fresh e-graph with the boolean-algebra datatype + ruleset loaded.
pub fn boolean_egraph() -> Result<EGraph, String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, BOOL_ALGEBRA_PROGRAM)
        .map_err(|e| format!("failed to load boolean-algebra program: {e}"))?;
    Ok(egraph)
}

/// Round-trip a boolean expression through egglog: insert `expr`, saturate the
/// `bool_algebra` ruleset, then extract the lowest-cost equivalent form and
/// return it as an s-expression string.
///
/// `expr` is egglog surface syntax over the `B` datatype, e.g.
/// `"(And (Var \"a\") (T))"`.
pub fn simplify(expr: &str) -> Result<String, String> {
    let mut egraph = boolean_egraph()?;

    // Bind the input term to a name so we can recover its e-class root, then
    // saturate. `run` with no iteration cap runs the ruleset to fixpoint.
    let program = format!(
        "(let __root {expr})\n(run-schedule (saturate (run bool_algebra)))"
    );
    egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| format!("failed to insert/saturate {expr:?}: {e}"))?;

    // Recover the value of `__root`, then extract its cheapest representative
    // using egglog's built-in tree-additive structural cost.
    let (sort, value) = egraph
        .eval_expr(&exprs::var("__root"))
        .map_err(|e| format!("failed to evaluate __root: {e}"))?;
    let (best, _cost) = egraph
        .extract_value_to_string(&sort, value)
        .map_err(|e| format!("failed to extract {expr:?}: {e}"))?;

    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::simplify;

    /// Each case: (input expression, expected extracted form). 20 hand-written
    /// cases per BRIEF.md Phase 1.0. Expected forms are the lowest-cost
    /// representative egglog should extract after saturating `bool_algebra`.
    const CASES: &[(&str, &str)] = &[
        // --- rule 1: identity (And x T) = x ---
        (r#"(And (Var "a") (T))"#, r#"(Var "a")"#),
        // Nested identity: inner (And a T)->a, then outer (And a T)->a.
        // NB: we deliberately do NOT assume And is commutative — egglog only
        // merges proven-equal e-classes, it does not know operator algebra
        // unless a rule says so, and BRIEF.md forbids bare commutativity
        // rewrites (they blow up). So every identity case keeps T on the right.
        (r#"(And (And (Var "a") (T)) (T))"#, r#"(Var "a")"#),
        (r#"(And (Var "x") (T))"#, r#"(Var "x")"#),
        // --- rule 2: double negation (Not (Not x)) = x ---
        (r#"(Not (Not (Var "a")))"#, r#"(Var "a")"#),
        (r#"(Not (Not (Not (Not (Var "b")))))"#, r#"(Var "b")"#),
        (r#"(Not (Not (T)))"#, "(T)"),
        (r#"(Not (Not (And (Var "a") (T))))"#, r#"(Var "a")"#),
        // --- rule 5: absorption (Or x (And x y)) = x ---
        (r#"(Or (Var "a") (And (Var "a") (Var "b")))"#, r#"(Var "a")"#),
        (r#"(Or (Var "p") (And (Var "p") (Var "q")))"#, r#"(Var "p")"#),
        (
            r#"(Or (Var "a") (And (Var "a") (T)))"#,
            r#"(Var "a")"#,
        ),
        // --- rules 3+4: De Morgan, then collapse ---
        (
            r#"(Not (And (Not (Var "a")) (Not (Var "b"))))"#,
            r#"(Or (Var "a") (Var "b"))"#,
        ),
        (
            r#"(Not (Or (Not (Var "a")) (Not (Var "b"))))"#,
            r#"(And (Var "a") (Var "b"))"#,
        ),
        (
            r#"(Not (And (Not (Var "x")) (Not (Var "y"))))"#,
            r#"(Or (Var "x") (Var "y"))"#,
        ),
        // --- terms with no applicable rule: identity round-trip ---
        (r#"(Var "a")"#, r#"(Var "a")"#),
        ("(T)", "(T)"),
        ("(F)", "(F)"),
        (r#"(And (Var "a") (Var "b"))"#, r#"(And (Var "a") (Var "b"))"#),
        (r#"(Or (Var "a") (Var "b"))"#, r#"(Or (Var "a") (Var "b"))"#),
        (r#"(Not (Var "a"))"#, r#"(Not (Var "a"))"#),
        // mixed chain: De Morgan over Or, double-neg collapse on one arm,
        // identity on the other: (Not (Or (Not a) (Not (And b T)))) -> (And a b)
        (
            r#"(Not (Or (Not (Var "a")) (Not (And (Var "b") (T)))))"#,
            r#"(And (Var "a") (Var "b"))"#,
        ),
    ];

    #[test]
    fn calibration_round_trips_all_cases() {
        let mut failures = Vec::new();
        for (input, expected) in CASES {
            match simplify(input) {
                Ok(got) if got == *expected => {}
                Ok(got) => failures.push(format!("{input}\n  got:      {got}\n  expected: {expected}")),
                Err(e) => failures.push(format!("{input}\n  error: {e}")),
            }
        }
        assert!(
            failures.is_empty(),
            "{} / {} cases failed:\n{}",
            failures.len(),
            CASES.len(),
            failures.join("\n")
        );
    }

    #[test]
    fn at_least_twenty_cases() {
        assert!(CASES.len() >= 20, "BRIEF.md requires >= 20 calibration cases");
    }
}
