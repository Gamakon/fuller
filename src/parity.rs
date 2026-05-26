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
use crate::ruleset::distribute::DISTRIBUTE_RULESET;
use crate::ruleset::identities::ALGEBRA_RULESET;
use crate::ruleset::powers::POWERS_RULESET;
use crate::ruleset::rational::RATIONAL_RULESET;
use crate::ruleset::trig::TRIG_RULESET;

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

/// Which ruleset family to score a corpus with. distribute (algebra/powers/
/// rational) and trig each terminate ALONE but compose into e-graph explosion
/// when run together (distribute's distribution + trig's expand rules grow the
/// graph without repeating). They are different problem domains, so we select
/// the family per corpus rather than forcing them into one saturation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// algebra + powers + distribute (powsimp / simplify).
    Algebra,
    /// algebra + powers + rational (ratsimp / radsimp). distribute and rational
    /// are both algebra-domain but their distributivity + square-expansion
    /// ping-pong and explode the e-graph together, so they are separate
    /// families — scored on the corpora each one serves.
    Rational,
    /// algebra + powers + trig (trigsimp).
    Trig,
}

fn program_for(family: Family) -> String {
    let (rules, names) = match family {
        Family::Algebra => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{DISTRIBUTE_RULESET}"),
            "algebra powers distribute",
        ),
        Family::Rational => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{DISTRIBUTE_RULESET}\n{RATIONAL_RULESET}"),
            "algebra powers distribute rational",
        ),
        Family::Trig => (
            format!("{ALGEBRA_RULESET}\n{POWERS_RULESET}\n{TRIG_RULESET}"),
            "algebra powers trig",
        ),
    };
    format!(
        "{MATH_DATATYPE}\n{GUARD_RELATIONS}\n{rules}\n\
         (unstable-combined-ruleset all {names})"
    )
}

/// Bounded iteration count per family. Algebra/Trig are confluent-enough to
/// reach fixpoint within 40. The Rational family deliberately combines
/// distribute + rational, which DO NOT reach a fixpoint together (distributivity
/// x square-expansion keep generating terms); a low bound truncates that growth
/// — pairs whose equality needs more iterations report "not proven", the honest
/// outcome — while still deriving the many that settle in a few rounds. 6 is the
/// empirical sweet spot: high enough to expand+fold the corpus's squared sums,
/// low enough that the e-graph stays small and fast.
fn sat_iters(family: Family) -> u32 {
    match family {
        Family::Rational => 6,
        _ => 40,
    }
}

/// Does running the `family` rules put `input` and `target` in the same e-class?
pub fn proves_equal_with(input: &str, target: &str, family: Family) -> Result<bool, String> {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, &program_for(family))
        .map_err(|e| format!("load rulesets: {e}"))?;
    let iters = sat_iters(family);
    let prog = format!(
        "(let __in {input})\n(let __tgt {target})\n\
         (run-schedule (repeat {iters} (run all)))\n\
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

/// Convenience: score with the Algebra family (the default for everything
/// except the trig corpus).
pub fn proves_equal(input: &str, target: &str) -> Result<bool, String> {
    proves_equal_with(input, target, Family::Algebra)
}

/// Score a whole corpus of pairs with a chosen ruleset family.
pub fn score_with(pairs: &[Pair], family: Family) -> ParityReport {
    let mut matched_inputs = Vec::new();
    let mut unmatched_inputs = Vec::new();
    for p in pairs {
        match proves_equal_with(&p.input, &p.target, family) {
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

/// Score with the Algebra family (back-compat for existing callers/tests).
pub fn score(pairs: &[Pair]) -> ParityReport {
    score_with(pairs, Family::Algebra)
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
