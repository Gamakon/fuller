//! The `wide` ruleset — pure form-GENERATING rewrites (commutativity,
//! associativity) used ONLY to populate an equivalence class for the extraction
//! tournament, never in `denoise`.
//!
//! Why this exists: the algebra/powers rules mostly *collapse* an expression
//! toward one canonical form, so a saturated e-class has few top-level members
//! and `extract_variants` (which enumerates root e-nodes) returns ~1 form. To
//! demonstrate the HFF angular cost model picking among equivalents, the class
//! needs many *equal* forms. Commutativity and associativity generate exactly
//! that — they reorder and re-associate without changing value, so every form
//! they produce is genuinely equal.
//!
//! Boundedness: these reorder a *fixed* multiset of operands (no term ever grows
//! — `a+b` and `b+a` are the same size), so the reachable set per expression is
//! finite (the orderings/associations of its operands). That is bounded but can
//! be COMBINATORIALLY LARGE for a long Add/Mul chain — so callers MUST run this
//! with a small bounded `iters` (never `saturate`) and a kill-guard. This is the
//! classic e-graph blow-up the project warns about; it is acceptable here only
//! because it is form-generating-on-purpose and always iteration-capped.
//!
//! NOTE (measured project finding): commutativity is NOT the SymPy-parity wall,
//! and egglog's own tests ship bare comm/assoc — they are fine *bounded*, fatal
//! at unbounded fixpoint alongside expand rules. We keep them isolated in their
//! own family so they never co-saturate with distribute's expansion.

/// The `wide` ruleset: commutativity + associativity for Add and Mul.
pub const WIDE_RULESET: &str = r#"
(ruleset wide)

; ---- commutativity (reorders operands; same multiset, bounded) ----
(rewrite (Add a b) (Add b a) :ruleset wide)
(rewrite (Mul a b) (Mul b a) :ruleset wide)

; ---- associativity (re-brackets; same operands, bounded) ----
(rewrite (Add (Add a b) c) (Add a (Add b c)) :ruleset wide)
(rewrite (Add a (Add b c)) (Add (Add a b) c) :ruleset wide)
(rewrite (Mul (Mul a b) c) (Mul a (Mul b c)) :ruleset wide)
(rewrite (Mul a (Mul b c)) (Mul (Mul a b) c) :ruleset wide)

; =====================================================================
; SHAPE-CHANGING rules. Comm/assoc above only REORDER — the forms they
; produce are identical on every structural measure (node count, nesting,
; sign-ops), so an angular cost model can't tell them apart. The rules
; below CHANGE structure: they put forms of DIFFERENT node-count / sign-op /
; nesting profile into the same e-class, which is exactly what the HFF
; tournament needs to discriminate (a cleaner member to pick over a bloated
; one). All are real-domain identities. Bidirectional where a simplification
; and its inverse are both wanted in the class; the e-graph holds both and
; the cost model chooses — but bidirectional expand/contract pairs are the
; combinatorial-blowup risk, so run bounded + kill-guarded ONLY.
; =====================================================================

; distributivity, BOTH directions: the expanded form a*b + a*c and the
; factored form a*(b+c) coexist in the class with different node counts, so
; the cost model picks the smaller.
(rewrite (Mul a (Add b c)) (Add (Mul a b) (Mul a c)) :ruleset wide)
(rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)) :ruleset wide)
(rewrite (Mul a (Sub b c)) (Sub (Mul a b) (Mul a c)) :ruleset wide)
(rewrite (Sub (Mul a b) (Mul a c)) (Mul a (Sub b c)) :ruleset wide)

; sign structure: Sub <-> Add of a Neg, and double negation. These change the
; sign-op tally (a measure) so the class holds a heavier and a lighter form.
(rewrite (Sub a b) (Add a (Neg b)) :ruleset wide)
(rewrite (Add a (Neg b)) (Sub a b) :ruleset wide)
(rewrite (Neg (Neg a)) a :ruleset wide)
(rewrite (Neg (Sub a b)) (Sub b a) :ruleset wide)

; identity contraction: x+0 = x, x*1 = x, x-0 = x. These SHRINK node count,
; so a form carrying an injected identity and its clean version are both in
; the class — the cost model prefers the clean one. (Contraction only — we do
; NOT inject identities, which would be unbounded.)
(rewrite (Add a (Num 0.0)) a :ruleset wide)
(rewrite (Mul a (Num 1.0)) a :ruleset wide)
(rewrite (Sub a (Num 0.0)) a :ruleset wide)
(rewrite (Mul a (Num 0.0)) (Num 0.0) :ruleset wide)
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use egglog::EGraph;

    /// Comm + assoc must prove `a + b == b + a` (the basic generated equality)
    /// under a bounded run, without diverging.
    #[test]
    fn wide_proves_commutativity_bounded() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(None, &format!("{MATH_DATATYPE}\n{GUARD_RELATIONS}\n{WIDE_RULESET}"))
            .expect("load");
        let prog = r#"
            (let __a (Add (Var "x") (Var "y")))
            (let __b (Add (Var "y") (Var "x")))
            (run-schedule (repeat 4 (run wide)))
            (check (= __a __b))
        "#;
        egraph.parse_and_run_program(None, prog).expect("comm holds bounded");
    }

    /// Associativity across three operands, bounded.
    #[test]
    fn wide_proves_associativity_bounded() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(None, &format!("{MATH_DATATYPE}\n{GUARD_RELATIONS}\n{WIDE_RULESET}"))
            .expect("load");
        let prog = r#"
            (let __l (Mul (Mul (Var "x") (Var "y")) (Var "z")))
            (let __r (Mul (Var "x") (Mul (Var "y") (Var "z"))))
            (run-schedule (repeat 4 (run wide)))
            (check (= __l __r))
        "#;
        egraph.parse_and_run_program(None, prog).expect("assoc holds bounded");
    }
}
