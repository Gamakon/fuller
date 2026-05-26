//! Parity scorer — measures how much of SymPy's simplification gamakAST
//! reproduces, using ONLY gamakAST's own tools (no sympy in the loop).
//!
//! A corpus pair `(input, target)` was produced offline by SymPy
//! (parity/gen_corpus.py): `target = sympy.<module>(input)`. We load both into
//! one e-graph, saturate our full ruleset, and ask egglog whether `input` and
//! `target` ended up in the SAME e-class. If so, our rules can prove the same
//! equality SymPy applied — that pair is "parity". The fraction of pairs that
//! reach parity is the parity score.
//!
//! This is honest: SymPy set the homework offline; egglog grades it. No
//! `sympy.simplify(out - tgt)` ever runs.

use egglog::EGraph;

use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
use crate::ruleset::identities::ALGEBRA_RULESET;
use crate::ruleset::powers::POWERS_RULESET;

/// One corpus pair.
#[derive(Debug, Clone)]
pub struct Pair {
    pub input: String,
    pub target: String,
}

/// Result of scoring a corpus.
#[derive(Debug, Clone)]
pub struct ParityReport {
    pub total: usize,
    pub matched: usize,
    /// Inputs egglog proved equal to their target (same e-class).
    pub matched_inputs: Vec<String>,
    /// Inputs that did NOT reach parity (rules can't derive the simplification).
    pub unmatched_inputs: Vec<String>,
}

impl ParityReport {
    pub fn pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { 100.0 * self.matched as f64 / self.total as f64 }
    }
}

/// All rulesets combined, run together to a bounded fixpoint. As more modules
/// land (trig, etc.) they are added here.
fn all_rulesets_program() -> String {
    format!(
        "{MATH_DATATYPE}\n{GUARD_RELATIONS}\n{ALGEBRA_RULESET}\n{POWERS_RULESET}\n\
         (unstable-combined-ruleset all algebra powers)"
    )
}

/// Does saturating our rules put `input` and `target` in the same e-class?
pub fn proves_equal(input: &str, target: &str) -> Result<bool, String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, &all_rulesets_program())
        .map_err(|e| format!("load rulesets: {e}"))?;
    // Insert both terms, saturate (bounded), then check e-class equality.
    let prog = format!(
        "(let __in {input})\n(let __tgt {target})\n\
         (run-schedule (repeat 40 (run all)))\n\
         (check (= __in __tgt))"
    );
    match egraph.parse_and_run_program(None, &prog) {
        Ok(_) => Ok(true),                 // (check ...) passed => same e-class
        Err(e) => {
            let msg = e.to_string();
            // A failed `check` is the normal "not proven equal" outcome.
            if msg.contains("Check failed") || msg.contains("check") {
                Ok(false)
            } else {
                Err(format!("scoring {input:?}: {msg}"))
            }
        }
    }
}

/// Score a whole corpus of pairs.
pub fn score(pairs: &[Pair]) -> ParityReport {
    let mut matched_inputs = Vec::new();
    let mut unmatched_inputs = Vec::new();
    for p in pairs {
        match proves_equal(&p.input, &p.target) {
            Ok(true) => matched_inputs.push(p.input.clone()),
            _ => unmatched_inputs.push(p.input.clone()),
        }
    }
    ParityReport {
        total: pairs.len(),
        matched: matched_inputs.len(),
        matched_inputs,
        unmatched_inputs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proves_a_known_algebra_equality() {
        // (x * 1) and x are provably equal by the algebra ruleset.
        assert_eq!(
            proves_equal(r#"(Mul (Var "x") (Num 1.0))"#, r#"(Var "x")"#).unwrap(),
            true
        );
    }

    #[test]
    fn does_not_prove_a_false_equality() {
        // x and y are not equal; rules must not "prove" it.
        assert_eq!(
            proves_equal(r#"(Var "x")"#, r#"(Var "y")"#).unwrap(),
            false
        );
    }

    #[test]
    fn scores_a_small_corpus() {
        let pairs = vec![
            Pair { input: r#"(Add (Var "x") (Num 0.0))"#.into(), target: r#"(Var "x")"#.into() },
            Pair { input: r#"(Var "x")"#.into(), target: r#"(Var "y")"#.into() }, // unmatched
        ];
        let r = score(&pairs);
        assert_eq!(r.total, 2);
        assert_eq!(r.matched, 1);
        assert_eq!(r.pct(), 50.0);
    }
}
