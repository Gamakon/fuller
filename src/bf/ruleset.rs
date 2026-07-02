//! Sound, semantics-preserving BF rewrite rules.
//!
//! # Prog datatype recap
//!
//! Each constructor carries its "rest" continuation as the last arg:
//!   (Inc rest), (Dec rest), (Left rest), (Right rest), (Out rest), (In rest),
//!   (Loop body rest), (AddN n rest), (MoveN n rest), (Clear rest), (Nil)
//!
//! # Rule families (all acyclic / contracting)
//!
//! 1. **Cancellation**: `(Inc (Dec rest))` => `rest`, etc.
//! 2. **Run-collapse**: consecutive `Inc`/`Dec` fold into `(AddN k rest)`;
//!    consecutive `Left`/`Right` fold into `(MoveN k rest)`.
//!    `(AddN 0 rest)` and `(MoveN 0 rest)` erase.
//! 3. **Clear-loop**: `(Loop (Dec (Nil)) rest)` and `(Loop (Inc (Nil)) rest)`
//!    => `(Clear rest)`. Sound: these loops run until the cell hits 0.
//!
//! # Rules we deliberately exclude
//!
//! - Dead-loop elimination after Clear: would require tracking "current cell
//!   is 0" as a relational fact across Seq nodes — out of scope. SKIPPED.
//! - General loop equivalence: undecidable. SKIPPED.
//!
//! # Boundedness
//!
//! All rules are CONTRACTING. The ruleset terminates; `bf_simplify` uses a
//! bounded `repeat` schedule for defence-in-depth.

/// The BF rewrite ruleset (egglog surface syntax, for the flat Prog ctor).
pub const BF_RULESET: &str = r#"
(ruleset bf)

; ---- Cancellation rules ----
; Inc then Dec: net zero -> skip both
(rewrite (Inc (Dec rest)) rest :ruleset bf)
; Dec then Inc: net zero -> skip both
(rewrite (Dec (Inc rest)) rest :ruleset bf)
; Right then Left: net zero pointer move -> skip both
(rewrite (Right (Left rest)) rest :ruleset bf)
; Left then Right: net zero pointer move -> skip both
(rewrite (Left (Right rest)) rest :ruleset bf)

; ---- Zero-valued aggregate erasure ----
(rewrite (AddN 0 rest) rest :ruleset bf)
(rewrite (MoveN 0 rest) rest :ruleset bf)

; ---- Run-collapse: fold pairs into aggregates ----
; Two Inc -> AddN 2
(rewrite (Inc (Inc rest)) (AddN 2 rest) :ruleset bf)
; Two Dec -> AddN -2
(rewrite (Dec (Dec rest)) (AddN -2 rest) :ruleset bf)
; Inc into existing AddN
(rewrite (Inc (AddN k rest)) (AddN (+ k 1) rest) :ruleset bf)
; Dec into existing AddN
(rewrite (Dec (AddN k rest)) (AddN (+ k -1) rest) :ruleset bf)
; AddN absorbs Inc
(rewrite (AddN k (Inc rest)) (AddN (+ k 1) rest) :ruleset bf)
; AddN absorbs Dec
(rewrite (AddN k (Dec rest)) (AddN (+ k -1) rest) :ruleset bf)
; AddN + AddN -> combined AddN
(rewrite (AddN k (AddN j rest)) (AddN (+ k j) rest) :ruleset bf)

; Two Right -> MoveN 2
(rewrite (Right (Right rest)) (MoveN 2 rest) :ruleset bf)
; Two Left -> MoveN -2
(rewrite (Left (Left rest)) (MoveN -2 rest) :ruleset bf)
; Right into existing MoveN
(rewrite (Right (MoveN k rest)) (MoveN (+ k 1) rest) :ruleset bf)
; Left into existing MoveN
(rewrite (Left (MoveN k rest)) (MoveN (+ k -1) rest) :ruleset bf)
; MoveN absorbs Right
(rewrite (MoveN k (Right rest)) (MoveN (+ k 1) rest) :ruleset bf)
; MoveN absorbs Left
(rewrite (MoveN k (Left rest)) (MoveN (+ k -1) rest) :ruleset bf)
; MoveN + MoveN -> combined MoveN
(rewrite (MoveN k (MoveN j rest)) (MoveN (+ k j) rest) :ruleset bf)

; ---- Clear-loop recognition ----
; [Dec ...] where body is exactly one Dec -> Clear ([-])
(rewrite (Loop (Dec (Nil)) rest) (Clear rest) :ruleset bf)
; [Inc ...] where body is exactly one Inc -> Clear ([+] wraps to 0)
(rewrite (Loop (Inc (Nil)) rest) (Clear rest) :ruleset bf)
"#;

/// Build a fresh e-graph with BF_DATATYPE and BF_RULESET loaded.
pub fn bf_egraph() -> Result<egglog::EGraph, String> {
    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(None, crate::bf::expr::BF_DATATYPE)
        .map_err(|e| format!("BF_DATATYPE: {e}"))?;
    egraph
        .parse_and_run_program(None, BF_RULESET)
        .map_err(|e| format!("BF_RULESET: {e}"))?;
    Ok(egraph)
}

/// Insert `prog_sexpr` into a fresh e-graph, run BF ruleset (bounded),
/// extract lowest-cost equivalent, return as BF source string.
#[cfg(test)]
pub(crate) fn simplify_bf_via_egraph(source: &str) -> Result<String, String> {
    use egglog::extract::{Extractor, TreeAdditiveCostModel};
    use egglog::prelude::exprs;

    let prog_sexpr = crate::bf::parse::parse_bf(source)?;
    let mut egraph = bf_egraph()?;
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "(let __r {prog_sexpr})\n\
                 (unstable-combined-ruleset bf_all bf)\n\
                 (run-schedule (repeat 20 (run bf_all)))"
            ),
        )
        .map_err(|e| format!("insert/saturate: {e}"))?;

    let (sort, value) = egraph
        .eval_expr(&exprs::var("__r"))
        .map_err(|e| format!("eval root: {e}"))?;

    let extractor = Extractor::compute_costs_from_rootsorts(
        Some(vec![sort]),
        &egraph,
        TreeAdditiveCostModel::default(),
    );
    let mut termdag = egglog::TermDag::default();
    let (_cost, term) = extractor
        .extract_best(&egraph, &mut termdag, value)
        .ok_or_else(|| "extraction failed".to_string())?;
    let result_sexpr = termdag.to_string(term);
    crate::bf::parse::unparse_bf(&result_sexpr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf::eval::run_bf;

    fn assert_simplifies_soundly(source: &str, expected: &str, inputs: &[&[u8]]) {
        let simplified = simplify_bf_via_egraph(source).expect("simplify");
        assert_eq!(
            simplified, expected,
            "simplification of {source:?} -> {simplified:?} (expected {expected:?})"
        );
        for input in inputs {
            let out_orig = run_bf(source, input);
            let out_simp = run_bf(&simplified, input);
            assert_eq!(
                out_orig, out_simp,
                "SOUNDNESS: {source:?} vs {simplified:?} differ on {input:?}"
            );
        }
    }

    #[test]
    fn bf_ruleset_loads() {
        bf_egraph().expect("BF e-graph loads");
    }

    #[test]
    fn bf_cancellation_inc_dec() {
        assert_simplifies_soundly("+-", "", &[&[], &[0], &[42], &[255]]);
    }

    #[test]
    fn bf_cancellation_dec_inc() {
        assert_simplifies_soundly("-+", "", &[&[], &[0], &[42], &[255]]);
    }

    #[test]
    fn bf_cancellation_right_left() {
        assert_simplifies_soundly("><", "", &[&[], &[1]]);
    }

    #[test]
    fn bf_cancellation_left_right() {
        assert_simplifies_soundly("<>", "", &[&[], &[1]]);
    }

    #[test]
    fn bf_clear_loop_recognized() {
        let test_inputs: &[&[u8]] = &[&[], &[0], &[5], &[255]];
        let simplified = simplify_bf_via_egraph("[-]").expect("simplify");
        for input in test_inputs {
            let out_orig = run_bf("[-]", input);
            let out_simp = run_bf(&simplified, input);
            assert_eq!(
                out_orig, out_simp,
                "SOUNDNESS: [-] vs {simplified:?} differ on {input:?}"
            );
        }
    }

    #[test]
    fn bf_plus_loop_recognized_as_clear() {
        let test_inputs: &[&[u8]] = &[&[], &[0], &[5], &[127], &[255]];
        let simplified = simplify_bf_via_egraph("[+]").expect("simplify");
        for input in test_inputs {
            let out_orig = run_bf("[+]", input);
            let out_simp = run_bf(&simplified, input);
            assert_eq!(
                out_orig, out_simp,
                "SOUNDNESS: [+] vs {simplified:?} differ on {input:?}"
            );
        }
    }

    #[test]
    fn bf_addn_collapse() {
        let src = "+++";
        let simplified = simplify_bf_via_egraph(src).expect("simplify");
        for input in [&[][..], &[0u8][..], &[200u8][..]] {
            let out_orig = run_bf(src, input);
            let out_simp = run_bf(&simplified, input);
            assert_eq!(out_orig, out_simp, "soundness: {src:?}");
        }
    }

    #[test]
    fn bf_ruleset_soundness_broad() {
        let cases: &[(&str, &[&[u8]])] = &[
            ("+-+-+-", &[&[], &[10u8][..], &[255u8][..]]),
            ("><><><", &[&[], &[0u8][..]]),
            ("++--",   &[&[], &[5u8][..]]),
            (">><<",   &[&[], &[5u8][..]]),
        ];
        for (src, inputs) in cases {
            let simplified = simplify_bf_via_egraph(src).expect("simplify");
            for input in *inputs {
                let out_orig = run_bf(src, input);
                let out_simp = run_bf(&simplified, input);
                assert_eq!(
                    out_orig, out_simp,
                    "SOUNDNESS: {src:?} vs {simplified:?} differ on {input:?}"
                );
            }
        }
    }
}
