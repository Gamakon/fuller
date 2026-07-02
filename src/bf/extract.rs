//! BF simplifier: insert, bounded-saturate, extract.
//!
//! `bf_simplify(source)` mirrors `extract.rs::denoise`:
//!   1. Parse BF source to Prog s-expression.
//!   2. Insert into a fresh e-graph.
//!   3. Run BF_RULESET for a bounded number of iterations.
//!   4. Extract the lowest-node-count equivalent.
//!   5. Unparse to BF source.
//!
//! Never raises on normal input; returns the input unchanged if no rule fires.

use egglog::extract::{Extractor, TreeAdditiveCostModel};
use egglog::prelude::exprs;
use egglog::EGraph;

use crate::bf::expr::BF_DATATYPE;
use crate::bf::parse::{parse_bf, unparse_bf};
use crate::bf::ruleset::BF_RULESET;

/// Maximum saturation iterations. BF rules are all contracting so a fixpoint
/// is typically reached in < 20 iterations; 50 is a generous bound.
const SIMPLIFY_ITERS: u32 = 50;

/// Outcome of a `bf_simplify` call.
#[derive(Debug, Clone)]
pub struct Simplified {
    /// The simplified BF source string.
    pub source: String,
    /// Number of BF ops in the simplified source.
    pub op_count: usize,
    /// True if the simplified form is strictly shorter than the input.
    pub changed: bool,
}

#[cfg(test)]
mod size_gate_tests {
    use super::bf_simplify;

    /// The never-raises contract must hold for arbitrarily long programs:
    /// beyond the size gate, bf_simplify returns unchanged instead of handing
    /// a 50k-deep cons nesting to egglog's recursive parser (stack abort).
    #[test]
    fn bf_simplify_huge_program_returns_unchanged() {
        let source: String = "+".repeat(50_000);
        let r = bf_simplify(&source).expect("must not error");
        assert_eq!(r.source, source);
        assert!(!r.changed);
        assert_eq!(r.op_count, 50_000);
    }
}

/// Simplify a BF source string using equality saturation.
///
/// Returns the simplified source + metadata. Never errors on normal input —
/// if parsing or saturation fails it returns the input unchanged (with
/// `changed = false`).
pub fn bf_simplify(source: &str) -> Result<Simplified, String> {
    let input_ops = op_count_source(source);

    // 0. Size gate. The Prog s-expression nests once per op, and egglog's own
    // program parser walks that nesting RECURSIVELY — our parse/unparse are
    // iterative, but handing a 50k-op program to egglog would still overflow
    // the stack inside parse_and_run_program. Cap sized like
    // crate::MAX_EXPR_DEPTH (well under the 2MB default test/rayon stacks);
    // larger programs return unchanged, keeping the never-raises contract
    // honest for arbitrary input.
    const MAX_SIMPLIFY_OPS: usize = 1024;
    if input_ops > MAX_SIMPLIFY_OPS {
        return Ok(Simplified { source: source.to_string(), op_count: input_ops, changed: false });
    }

    // 1. Parse to Prog s-expression.
    let prog_sexpr = match parse_bf(source) {
        Ok(s) => s,
        Err(_) => {
            return Ok(Simplified {
                source: source.to_string(),
                op_count: input_ops,
                changed: false,
            });
        }
    };

    // 2. Build e-graph and insert.
    let mut egraph = EGraph::default();
    if egraph.parse_and_run_program(None, BF_DATATYPE).is_err() {
        return Ok(Simplified { source: source.to_string(), op_count: input_ops, changed: false });
    }
    if egraph.parse_and_run_program(None, BF_RULESET).is_err() {
        return Ok(Simplified { source: source.to_string(), op_count: input_ops, changed: false });
    }

    // 3. Insert and saturate (bounded).
    let run_prog = format!(
        "(let __root {prog_sexpr})\n\
         (unstable-combined-ruleset bf_all bf)\n\
         (run-schedule (repeat {SIMPLIFY_ITERS} (run bf_all)))"
    );
    if egraph.parse_and_run_program(None, &run_prog).is_err() {
        return Ok(Simplified { source: source.to_string(), op_count: input_ops, changed: false });
    }

    // 4. Extract lowest-cost equivalent.
    let (sort, value) = match egraph.eval_expr(&exprs::var("__root")) {
        Ok(sv) => sv,
        Err(_) => {
            return Ok(Simplified { source: source.to_string(), op_count: input_ops, changed: false });
        }
    };

    let extractor = Extractor::compute_costs_from_rootsorts(
        Some(vec![sort]),
        &egraph,
        TreeAdditiveCostModel::default(),
    );
    let mut termdag = egglog::TermDag::default();
    let (_cost, term) = match extractor.extract_best(&egraph, &mut termdag, value) {
        Some(ct) => ct,
        None => {
            return Ok(Simplified { source: source.to_string(), op_count: input_ops, changed: false });
        }
    };
    let result_sexpr = termdag.to_string(term);

    // 5. Unparse to BF source.
    let simplified_source = match unparse_bf(&result_sexpr) {
        Ok(s) => s,
        Err(_) => {
            return Ok(Simplified { source: source.to_string(), op_count: input_ops, changed: false });
        }
    };

    let simplified_ops = op_count_source(&simplified_source);
    let changed = simplified_ops < input_ops;

    Ok(Simplified { source: simplified_source, op_count: simplified_ops, changed })
}

/// Count BF ops (chars in `+-<>.,[]`) in a source string.
fn op_count_source(source: &str) -> usize {
    source.chars().filter(|c| "+-<>.,[]".contains(*c)).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf::eval::run_bf;

    fn assert_bf_simplify_sound(source: &str, inputs: &[&[u8]]) -> Simplified {
        let result = bf_simplify(source).expect("bf_simplify");
        for input in inputs {
            let out_orig = run_bf(source, input);
            let out_simp = run_bf(&result.source, input);
            assert_eq!(
                out_orig, out_simp,
                "SOUNDNESS: {source:?} simplified to {:?} but outputs differ on {input:?}",
                result.source
            );
        }
        result
    }

    #[test]
    fn bf_simplify_cancels_inc_dec() {
        let r = assert_bf_simplify_sound("+-", &[&[], &[0], &[42], &[255]]);
        assert!(r.changed, "+-  should shrink");
        assert_eq!(r.op_count, 0);
    }

    #[test]
    fn bf_simplify_cancels_move() {
        let r = assert_bf_simplify_sound("><", &[&[], &[0]]);
        assert!(r.changed, "><  should shrink");
        assert_eq!(r.op_count, 0);
    }

    #[test]
    fn bf_simplify_clear_loop() {
        let inputs: &[&[u8]] = &[&[], &[0], &[5], &[100], &[255]];
        let r = assert_bf_simplify_sound("[-]", inputs);
        // The program is sound; it may or may not 'change' structurally
        // depending on how Clear unparsing works, but soundness is required.
        let _ = r;
    }

    #[test]
    fn bf_simplify_chain_cancels() {
        let inputs: &[&[u8]] = &[&[], &[0], &[42]];
        let r = assert_bf_simplify_sound("+-+-+-", inputs);
        assert!(r.changed);
        assert_eq!(r.op_count, 0);
    }

    #[test]
    fn bf_simplify_preserves_echo() {
        // ,. is a minimal program; nothing to simplify
        let inputs: &[&[u8]] = &[&[0], &[65], &[255]];
        let r = assert_bf_simplify_sound(",.", inputs);
        assert!(!r.changed, ",. has no simplification");
    }

    #[test]
    fn bf_simplify_no_crash_on_complex() {
        // Complex but valid BF: increment task
        let inputs: &[&[u8]] = &[&[0], &[10], &[100], &[254]];
        assert_bf_simplify_sound(",+.", inputs);
    }

    #[test]
    fn bf_soundness_500_random() {
        // Generate ~500 random short BF programs and verify soundness.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let ops = ['+', '-', '<', '>', '.', ','];
        let mut mismatches = 0usize;
        let mut tested = 0usize;

        for seed in 0u64..500 {
            // Deterministic pseudo-random program generation from seed
            let mut prog = String::new();
            let mut bracket_depth = 0i32;
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            let mut state = h.finish();

            let prog_len = (state % 12 + 3) as usize;
            for _ in 0..prog_len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (state >> 33) as usize % ops.len();
                let ch = ops[idx];
                // Avoid unmatched brackets: allow '[' only if depth < 2,
                // allow ']' only if depth > 0
                match ch {
                    '[' if bracket_depth < 2 => {
                        prog.push('[');
                        bracket_depth += 1;
                    }
                    ']' if bracket_depth > 0 => {
                        prog.push(']');
                        bracket_depth -= 1;
                    }
                    '[' | ']' => {
                        // replace with a safe op
                        prog.push('+');
                    }
                    _ => prog.push(ch),
                }
            }
            // Close any open brackets
            for _ in 0..bracket_depth {
                prog.push(']');
            }

            let simplified = match bf_simplify(&prog) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Test on a few random inputs derived from seed
            let test_inputs: &[&[u8]] = &[&[], &[0], &[seed as u8], &[255 - (seed as u8 % 128)]];
            for input in test_inputs {
                let out_orig = run_bf(&prog, input);
                let out_simp = run_bf(&simplified.source, input);
                // Only compare if both halt
                if let (Some(o), Some(s)) = (out_orig.output(), out_simp.output()) {
                    tested += 1;
                    if o != s {
                        mismatches += 1;
                        eprintln!(
                            "SOUNDNESS FAIL: prog={prog:?} simplified={:?} input={input:?} orig={o:?} simp={s:?}",
                            simplified.source
                        );
                    }
                }
            }
        }

        assert_eq!(
            mismatches, 0,
            "Soundness check: {mismatches} mismatches out of {tested} tested cases"
        );
        assert!(tested > 0, "At least some programs should have been tested");
    }
}
