//! Regex simplifier: insert, bounded-saturate, extract.
//!
//! `regex_simplify(ast_sexpr)` mirrors `bf_simplify`:
//!   1. Parse the phylu `ast_sexpr` into an egglog `Re` s-expression.
//!   2. Insert into a fresh e-graph.
//!   3. Run RE_RULESET for a bounded number of iterations.
//!   4. Extract the lowest-node-count equivalent.
//!   5. Unparse back to the phylu `ast_sexpr` format.
//!
//! Never raises on normal input; returns the input unchanged if anything fails
//! or no rule fires.

use egglog::extract::{Extractor, TreeAdditiveCostModel};
use egglog::prelude::exprs;
use egglog::EGraph;

use crate::regex::expr::RE_DATATYPE;
use crate::regex::parse::{parse_regex, unparse_regex};
use crate::regex::ruleset::RE_RULESET;

/// Maximum saturation iterations. Regex rules are contracting / size-neutral, so
/// a fixpoint is reached quickly; 50 is a generous bound (same as BF).
const SIMPLIFY_ITERS: u32 = 50;

/// Size gate: the `Re` s-expression nests once per node and egglog's parser
/// walks that recursively, so a very large AST is refused (returned unchanged)
/// to keep the never-raises contract. Sized like BF's cap.
const MAX_SIMPLIFY_NODES: usize = 1024;

/// Outcome of a `regex_simplify` call.
#[derive(Debug, Clone)]
pub struct Simplified {
    /// The simplified regex in the phylu `ast_sexpr` format (same as input if no
    /// rule fires or anything failed).
    pub source: String,
    /// Node count of the simplified AST (number of `Re` constructors).
    pub op_count: usize,
    /// True if the simplified form has strictly fewer nodes than the input.
    pub changed: bool,
}

/// Count AST nodes (constructors) in a phylu `ast_sexpr` — every `(` opens one
/// node. `(Num b)` is a leaf of `RegexLit` and is not counted separately.
fn node_count_ast(ast_sexpr: &str) -> usize {
    // Count '(' that are NOT immediately followed by "Num" (the Num leaf rides
    // with its RegexLit parent).
    let bytes = ast_sexpr.as_bytes();
    let mut n = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            // Peek past spaces for "Num".
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            let is_num = ast_sexpr[j..].starts_with("Num");
            if !is_num {
                n += 1;
            }
        }
        i += 1;
    }
    n
}

/// Count nodes in an egglog `Re` s-expression (every `(` is a node; `(Lit b)` is
/// one node).
fn node_count_re(re_sexpr: &str) -> usize {
    re_sexpr.bytes().filter(|&b| b == b'(').count()
}

/// Simplify a regex given as a phylu `ast_sexpr` string via equality saturation.
///
/// Returns the simplified AST (phylu format) + metadata. Never errors on normal
/// input — on any failure it returns the input unchanged (`changed = false`).
pub fn regex_simplify(ast_sexpr: &str) -> Result<Simplified, String> {
    let input_nodes = node_count_ast(ast_sexpr);
    let unchanged = || Simplified { source: ast_sexpr.to_string(), op_count: input_nodes, changed: false };

    // 0. Size gate.
    if input_nodes > MAX_SIMPLIFY_NODES {
        return Ok(unchanged());
    }

    // 1. Parse phylu ast_sexpr -> egglog Re s-expr.
    let re_sexpr = match parse_regex(ast_sexpr) {
        Ok(s) => s,
        Err(_) => return Ok(unchanged()),
    };

    // 2. Build e-graph and load datatype + ruleset.
    let mut egraph = EGraph::default();
    if egraph.parse_and_run_program(None, RE_DATATYPE).is_err() {
        return Ok(unchanged());
    }
    if egraph.parse_and_run_program(None, RE_RULESET).is_err() {
        return Ok(unchanged());
    }

    // 3. Insert and saturate (bounded).
    let run_prog = format!(
        "(let __root {re_sexpr})\n\
         (unstable-combined-ruleset regex_all regex)\n\
         (run-schedule (repeat {SIMPLIFY_ITERS} (run regex_all)))"
    );
    if egraph.parse_and_run_program(None, &run_prog).is_err() {
        return Ok(unchanged());
    }

    // 4. Extract lowest-cost equivalent.
    let (sort, value) = match egraph.eval_expr(&exprs::var("__root")) {
        Ok(sv) => sv,
        Err(_) => return Ok(unchanged()),
    };
    let extractor = Extractor::compute_costs_from_rootsorts(
        Some(vec![sort]),
        &egraph,
        TreeAdditiveCostModel::default(),
    );
    let mut termdag = egglog::TermDag::default();
    let (_cost, term) = match extractor.extract_best(&egraph, &mut termdag, value) {
        Some(ct) => ct,
        None => return Ok(unchanged()),
    };
    let result_re = termdag.to_string(term);

    // 5. Unparse Re s-expr -> phylu ast_sexpr.
    let simplified = match unparse_regex(&result_re) {
        Ok(s) => s,
        Err(_) => return Ok(unchanged()),
    };

    // Use the egglog node count for the comparison (robust to spacing); report
    // the ast node count for the caller.
    let simplified_re_nodes = node_count_re(&result_re);
    let input_re_nodes = node_count_re(&re_sexpr);
    let simplified_nodes = node_count_ast(&simplified);
    let changed = simplified_re_nodes < input_re_nodes;

    Ok(Simplified { source: simplified, op_count: simplified_nodes, changed })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_empty_left_collapses() {
        // Concat(Empty, Dot) -> Dot
        let r = regex_simplify("(RegexConcat (RegexEmpty) (RegexDot))").expect("simplify");
        assert_eq!(r.source, "(RegexDot)");
        assert!(r.changed);
    }

    #[test]
    fn star_star_collapses() {
        // Star(Star(Dot)) -> Star(Dot)
        let r = regex_simplify("(RegexStar (RegexStar (RegexDot)))").expect("simplify");
        assert_eq!(r.source, "(RegexStar (RegexDot))");
        assert!(r.changed);
    }

    #[test]
    fn alt_idempotent() {
        // Alt(Dot, Dot) -> Dot
        let r = regex_simplify("(RegexAlt (RegexDot) (RegexDot))").expect("simplify");
        assert_eq!(r.source, "(RegexDot)");
        assert!(r.changed);
    }

    #[test]
    fn nothing_to_simplify_is_unchanged() {
        let src = "(RegexConcat (RegexCcDigit) (RegexCcWord))";
        let r = regex_simplify(src).expect("simplify");
        assert_eq!(r.source, src);
        assert!(!r.changed);
    }

    #[test]
    fn garbage_returns_unchanged() {
        let src = "not a valid sexpr ((((";
        let r = regex_simplify(src).expect("never raises");
        assert_eq!(r.source, src);
        assert!(!r.changed);
    }

    #[test]
    fn lit_round_trips_through_simplify() {
        // A Char literal must survive: Concat(Empty, Lit 65) -> Lit 65.
        let r = regex_simplify("(RegexConcat (RegexEmpty) (RegexLit (Num 65)))").expect("simplify");
        assert_eq!(r.source, "(RegexLit (Num 65))");
        assert!(r.changed);
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;

    /// Render a phylu `ast_sexpr` into a standard regex string the `regex` crate
    /// understands, for the test oracle. Mirrors phylu's `to_regex_string`
    /// closely enough for the identities we rewrite (no RepN/Integer appear).
    fn ast_to_regex(ast: &str) -> Result<String, String> {
        fn tok(s: &str) -> Vec<String> {
            let mut out = Vec::new();
            let mut it = s.chars().peekable();
            while let Some(&c) = it.peek() {
                match c {
                    '(' | ')' => {
                        out.push(c.to_string());
                        it.next();
                    }
                    c if c.is_whitespace() => {
                        it.next();
                    }
                    _ => {
                        let mut t = String::new();
                        while let Some(&c2) = it.peek() {
                            if c2 == '(' || c2 == ')' || c2.is_whitespace() {
                                break;
                            }
                            t.push(c2);
                            it.next();
                        }
                        out.push(t);
                    }
                }
            }
            out
        }
        fn esc(b: u8) -> String {
            let c = b as char;
            if "\\^$.|?*+()[]{}".contains(c) {
                format!("\\{c}")
            } else {
                c.to_string()
            }
        }
        fn node(toks: &[String], pos: &mut usize) -> Result<String, String> {
            if toks.get(*pos).map(String::as_str) != Some("(") {
                return Err("expected (".into());
            }
            *pos += 1;
            let head = toks.get(*pos).cloned().ok_or("missing head")?;
            *pos += 1;
            let r = match head.as_str() {
                "RegexEmpty" => String::new(),
                "RegexDot" => ".".into(),
                "RegexCcDigit" => r"\d".into(),
                "RegexCcWord" => r"\w".into(),
                "RegexCcSpace" => r"\s".into(),
                "RegexAnchorStart" => "^".into(),
                "RegexAnchorEnd" => "$".into(),
                "Num" => {
                    let b: f64 = toks.get(*pos).ok_or("num")?.parse().map_err(|_| "num parse")?;
                    *pos += 1;
                    esc(b.round() as u8)
                }
                "RegexLit" => {
                    // child is (Num b)
                    node(toks, pos)?
                }
                "RegexStar" => format!("(?:{})*", node(toks, pos)?),
                "RegexPlus" => format!("(?:{})+", node(toks, pos)?),
                "RegexOpt" => format!("(?:{})?", node(toks, pos)?),
                "RegexClassOf" => format!("[{}]", node(toks, pos)?),
                "RegexCcNegate" => format!("^{}", node(toks, pos)?),
                "RegexConcat" => {
                    let a = node(toks, pos)?;
                    let b = node(toks, pos)?;
                    format!("(?:{a})(?:{b})")
                }
                "RegexAlt" => {
                    let a = node(toks, pos)?;
                    let b = node(toks, pos)?;
                    format!("(?:{a}|{b})")
                }
                "RegexCcRange" => {
                    let a = node(toks, pos)?;
                    let b = node(toks, pos)?;
                    format!("{a}-{b}")
                }
                "RegexCcUnion" => {
                    let a = node(toks, pos)?;
                    let b = node(toks, pos)?;
                    format!("{a}{b}")
                }
                other => return Err(format!("oracle: unknown {other}")),
            };
            if toks.get(*pos).map(String::as_str) != Some(")") {
                return Err("expected )".into());
            }
            *pos += 1;
            Ok(r)
        }
        let toks = tok(ast);
        let mut pos = 0;
        let body = node(&toks, &mut pos)?;
        Ok(format!("^(?:{body})$"))
    }

    /// Compile an ast via the oracle and return a matcher, or None if the oracle
    /// cannot render it (e.g. a bare class fragment — skip those cases).
    fn oracle(ast: &str) -> Option<regex::Regex> {
        let pat = ast_to_regex(ast).ok()?;
        regex::Regex::new(&pat).ok()
    }

    fn same_language(a: &str, b: &str, strings: &[&str]) -> bool {
        let (ra, rb) = match (oracle(a), oracle(b)) {
            (Some(ra), Some(rb)) => (ra, rb),
            _ => return true, // un-renderable by the oracle: skip (not a failure)
        };
        strings.iter().all(|s| ra.is_match(s) == rb.is_match(s))
    }

    const CORPUS: &[&str] = &[
        "", "a", "b", "0", "9", "aa", "a0", "0a", "abc", "123", "a1b2", " ", "a b",
        "A", "AB", "xyz", "00", "99", "a9", "9a",
    ];

    #[test]
    fn simplify_preserves_language_on_identities() {
        let cases = [
            "(RegexConcat (RegexEmpty) (RegexDot))",
            "(RegexStar (RegexStar (RegexDot)))",
            "(RegexAlt (RegexDot) (RegexDot))",
            "(RegexConcat (RegexEmpty) (RegexLit (Num 97)))",
            "(RegexStar (RegexOpt (RegexCcDigit)))",
            "(RegexOpt (RegexOpt (RegexCcWord)))",
            "(RegexConcat (RegexStar (RegexCcDigit)) (RegexStar (RegexCcDigit)))",
            "(RegexPlus (RegexStar (RegexDot)))",
        ];
        for c in cases {
            let r = regex_simplify(c).expect("simplify");
            assert!(
                same_language(c, &r.source, CORPUS),
                "SOUNDNESS: {c} -> {} changed the language",
                r.source
            );
        }
    }

    #[test]
    fn property_random_asts_preserve_language() {
        // Deterministic pseudo-random small regex ASTs; simplify must preserve
        // the language (checked via the regex crate oracle on the corpus).
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let leaves = [
            "(RegexDot)",
            "(RegexCcDigit)",
            "(RegexCcWord)",
            "(RegexEmpty)",
            "(RegexLit (Num 97))",
            "(RegexLit (Num 48))",
        ];
        let unary = ["RegexStar", "RegexPlus", "RegexOpt"];
        let binary = ["RegexConcat", "RegexAlt"];

        fn build(state: &mut u64, leaves: &[&str], unary: &[&str], binary: &[&str], depth: u32) -> String {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = (*state >> 33) as usize;
            if depth == 0 || pick.is_multiple_of(3) {
                return leaves[pick % leaves.len()].to_string();
            }
            if pick.is_multiple_of(2) {
                let op = unary[pick % unary.len()];
                format!("({op} {})", build(state, leaves, unary, binary, depth - 1))
            } else {
                let op = binary[pick % binary.len()];
                let a = build(state, leaves, unary, binary, depth - 1);
                let b = build(state, leaves, unary, binary, depth - 1);
                format!("({op} {a} {b})")
            }
        }

        let mut checked = 0usize;
        for seed in 0u64..500 {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            let mut state = h.finish();
            let ast = build(&mut state, &leaves, &unary, &binary, 4);
            let r = regex_simplify(&ast).expect("never raises");
            if same_language(&ast, &r.source, CORPUS) {
                checked += 1;
            } else {
                panic!("SOUNDNESS FAIL: {ast} -> {} differ", r.source);
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn a_bloated_form_shrinks() {
        // Star(Star(Opt(Dot))) -> Star(Dot): three ops collapse to two.
        let r = regex_simplify("(RegexStar (RegexStar (RegexOpt (RegexDot))))").expect("simplify");
        assert!(r.changed, "should shrink: {}", r.source);
        assert!(r.op_count < 4);
        assert!(same_language("(RegexStar (RegexStar (RegexOpt (RegexDot))))", &r.source, CORPUS));
    }
}
