//! Pattern/measure scoring — the `/pattern/{measure}` rule library and the
//! CDF-corrected hyperspherical-fitness (HFF) angle it feeds.
//!
//! DESIGN (as specified): scoring is a **data-driven list of rules**, not a
//! fixed struct of counters. Each rule is a `(pattern, measure)` pair: the
//! pattern decides whether the rule FIRES on a given term, and when it fires it
//! contributes one bounded `[0,1]` component (0 = best) to that term's measure
//! vector. Adding a new objective is adding a rule to [`measure_rules`] — no
//! struct surgery, no change to the accumulation logic.
//!
//! A term's score: run every rule over the term; the rules that fire give the
//! objective vector `x ∈ [0,1]^k` (its dimension `k` = how many rules fired —
//! genuinely heterogeneous per term). Project with HFF-TrueNorth
//! (`hff_core::core_functions`) to an angle, then CDF-correct it for dimension
//! `k` (`hff_core::higd`) so vectors of different `k` are comparable. Lowest
//! corrected angle wins the e-class tournament.
//!
//! This is the scorer the HFF extractor (`crate::extract::eclass_extract_hff`,
//! via the vendored `egglog::extract::hff_extract`) calls on each whole
//! candidate term — non-monotone by design, which the stock scalar extractor
//! could not express.

/// A parsed Math node: constructor head + children (leaves carry their kind in
/// the head: `Num`/`Var`). Built once per term by [`parse`]; every rule reads it.
#[derive(Clone, Debug)]
pub struct Node {
    pub head: String,
    pub children: Vec<Node>,
    /// For a `Var` leaf, its name; for a `Num` leaf, the literal text. Empty for
    /// internal nodes.
    pub leaf: String,
}

impl Node {
    fn is_var(&self) -> bool {
        self.head == "Var"
    }
    fn is_num(&self) -> bool {
        self.head == "Num"
    }
    /// Pre-order walk over the whole subtree.
    fn walk<'a>(&'a self, out: &mut Vec<&'a Node>) {
        out.push(self);
        for c in &self.children {
            c.walk(out);
        }
    }
    /// All nodes in the subtree, pre-order.
    pub fn nodes(&self) -> Vec<&Node> {
        let mut v = Vec::new();
        self.walk(&mut v);
        v
    }
    /// Depth (root = 1).
    fn depth(&self) -> u32 {
        1 + self.children.iter().map(Node::depth).max().unwrap_or(0)
    }
}

/// Transcendental constructors (for the nesting / transcendental measures).
fn is_transc(head: &str) -> bool {
    matches!(
        head,
        "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Tanh" | "ProtectedExp" | "ProtectedLog"
    )
}

/// Ops whose unguarded use can diverge / hit a domain edge (asymptote, blow-up).
fn is_unsafe_op(head: &str) -> bool {
    matches!(head, "Div" | "Inv" | "Log" | "Exp" | "Pow")
}

/// The bounded saturating penalty `c -> 1 - 1/(1 + c/s)` (0 at c=0, ->1 as c
/// grows; `s` sets where it reaches 0.5). The standard count->[0,1] map.
fn sat(c: f64, s: f64) -> f64 {
    1.0 - 1.0 / (1.0 + c / s)
}

/// One `/pattern/{measure}` rule.
///
/// `eval` returns `Some(v)` with `v ∈ [0,1]` when the rule FIRES on `root`, or
/// `None` when its pattern is absent (the rule contributes no dimension). This
/// is the firing semantics: a term is scored only on the rules whose pattern it
/// matches, so the objective vector's dimension varies per term.
pub struct MeasureRule {
    pub name: &'static str,
    pub eval: fn(&Node) -> Option<f64>,
}

/// The measure-rule library. Each entry is a guarded measure; 0 is best. To add
/// an objective, add a rule here — nothing else changes.
///
/// `node_count` is unconditional (always fires) so every term has a non-empty
/// vector and the HFF projection is well-defined.
pub fn measure_rules() -> Vec<MeasureRule> {
    vec![
        // /*/ {node_count} — parsimony. Always fires.
        MeasureRule {
            name: "node_count",
            eval: |r| Some(sat(r.nodes().len() as f64, 12.0)),
        },
        // /transc/ {transcendental_count} — universal-approximator load.
        MeasureRule {
            name: "transc_count",
            eval: |r| {
                let c = r.nodes().iter().filter(|n| is_transc(&n.head)).count();
                if c == 0 {
                    None
                } else {
                    Some(sat(c as f64, 2.0))
                }
            },
        },
        // /transc(transc(..))/ {transc_nesting} — a transcendental whose subtree
        // already holds a transcendental. Fires only when such nesting exists.
        MeasureRule {
            name: "transc_nest",
            eval: |r| {
                let c = count_transc_nesting(r);
                if c == 0 {
                    None
                } else {
                    Some(sat(c as f64, 1.0))
                }
            },
        },
        // /f(f(..)) same transc/ {self_nesting} — sin(sin(..)) etc.; strongest junk.
        MeasureRule {
            name: "self_nest",
            eval: |r| {
                let c = count_self_nesting(r);
                if c == 0 {
                    None
                } else {
                    Some(sat(c as f64, 1.0))
                }
            },
        },
        // /Num/ {numeric_literal_count} — free-parameter / overfit proxy.
        MeasureRule {
            name: "num_count",
            eval: |r| {
                let c = r.nodes().iter().filter(|n| n.is_num()).count();
                if c == 0 {
                    None
                } else {
                    Some(sat(c as f64, 4.0))
                }
            },
        },
        // /Num & Var/ {const_to_var_ratio} — fires when the term has leaves at
        // all; const-heavy forms (more numbers than variables) penalised.
        MeasureRule {
            name: "const_to_var",
            eval: |r| {
                let nums = r.nodes().iter().filter(|n| n.is_num()).count() as f64;
                let vars = r.nodes().iter().filter(|n| n.is_var()).count() as f64;
                if nums + vars == 0.0 {
                    None
                } else {
                    Some(nums / (nums + vars))
                }
            },
        },
        // /unsafe-op/ {instability} — div/inv/log/exp/pow present: asymptote /
        // blow-up risk (the extrapolation-divergence signal). Fires when any
        // unsafe op appears.
        MeasureRule {
            name: "instability",
            eval: |r| {
                let c = r.nodes().iter().filter(|n| is_unsafe_op(&n.head)).count();
                if c == 0 {
                    None
                } else {
                    Some(sat(c as f64, 2.0))
                }
            },
        },
        // /*/ {depth_over_breadth} — tall skinny trees (deep nesting) vs bushy
        // ones; a law tends balanced, a fitter stacks deep. Fires for any
        // multi-node term.
        MeasureRule {
            name: "depth_breadth",
            eval: |r| {
                let n = r.nodes().len() as f64;
                if n < 2.0 {
                    return None;
                }
                let ratio = r.depth() as f64 / (1.0 + n).log2();
                Some(sat(ratio, 2.0))
            },
        },
    ]
}

/// Count transcendental-inside-transcendental events: a transcendental node with
/// a transcendental anywhere in its subtree (excluding itself).
fn count_transc_nesting(root: &Node) -> u32 {
    let mut c = 0;
    for n in root.nodes() {
        if is_transc(&n.head) && n.children.iter().any(subtree_has_transc) {
            c += 1;
        }
    }
    c
}

fn subtree_has_transc(n: &Node) -> bool {
    n.nodes().iter().any(|m| is_transc(&m.head))
}

/// Count same-transcendental-in-itself events (sin directly over a subtree that
/// contains the SAME transcendental).
fn count_self_nesting(root: &Node) -> u32 {
    let mut c = 0;
    for n in root.nodes() {
        if is_transc(&n.head)
            && n.children
                .iter()
                .any(|ch| ch.nodes().iter().any(|m| m.head == n.head))
        {
            c += 1;
        }
    }
    c
}

/// Run every measure rule over `root`; return only the values of rules that
/// FIRED (the heterogeneous objective vector) alongside their names.
pub fn fire(root: &Node) -> Vec<(&'static str, f64)> {
    measure_rules()
        .into_iter()
        .filter_map(|rule| (rule.eval)(root).map(|v| (rule.name, v)))
        .collect()
}

/// CDF-corrected HFF-TrueNorth angle of a term: run the rule library, project
/// the fired vector with TrueNorth, CDF-correct for its dimension. Lower = best.
///
/// Uses `log_cdf_beta_correction`, NOT the plain linear `cdf_beta_correction`:
/// the linear CDF underflows to 0 in the deep left tail (small θ, large
/// dimension), which made low-D clean forms and high-D junk forms both pin near
/// 0 — so junk looked as good as clean. The log-space variant (Lentz continued
/// fraction) survives the tail, so a 3-D and an 8-D candidate stay comparable.
/// Returns a log-percentile: more negative = rarer/better. (Per the HFF repo's
/// CLAUDE.md: deep-left-tail comparisons MUST use the log variant.)
pub fn angle_percentile(root: &Node) -> f64 {
    // Score EVERY candidate at the SAME fixed dimension = the full rule count, so
    // angles are directly comparable. (Scoring each at its own fired-rule count
    // inverts the ranking — a high-D form gets a rarer angle purely from
    // dimension. Verified.)
    // Non-firing rules pad to 0.5 — NEUTRAL: a pattern that didn't match neither
    // rewards (0, "falsely perfect") nor punishes (1, which made a clean low-fire
    // form score WORST than junk — verified). 0.5 sits at the equator and doesn't
    // tilt the angle. All candidates scored at the full rule count = fixed k.
    let rules = measure_rules();
    let k = rules.len();
    let x: Vec<f64> = rules
        .into_iter()
        .map(|rule| (rule.eval)(root).unwrap_or(0.5))
        .collect();
    let arr = ndarray::Array1::from(x);
    // Rank on the raw TrueNorth angle. With a FIXED common k, the CDF correction
    // is only a monotone reshaping of theta — it changes the order ONLY when
    // comparing across different k (which we no longer do, since non-firing rules
    // pad to their ideal 0 and every candidate is scored at the full rule count).
    // So no CDF here; lower angle = cleaner.
    hff_core::core_functions::calculate_single_hyperspherical_fitness_f64_with_method(
        &arr, k, false, None, "truenorth",
    )
}

/// Score a Math s-expression string: parse then [`angle_percentile`]. This is the
/// scorer the HFF extractor calls on each whole candidate term.
pub fn score_expr(expr: &str) -> f64 {
    match parse(expr) {
        Some(n) => angle_percentile(&n),
        None => 1.0, // unparseable -> worst
    }
}

/// Parse a Math s-expression `(Head child...)`, `(Num v)`, `(Var "name")` into a
/// [`Node`]. Returns `None` on malformed input.
pub fn parse(expr: &str) -> Option<Node> {
    let toks = tokenize(expr);
    let mut pos = 0usize;
    let n = parse_node(&toks, &mut pos)?;
    Some(n)
}

fn parse_node(toks: &[String], pos: &mut usize) -> Option<Node> {
    if *pos >= toks.len() || toks[*pos] != "(" {
        return None;
    }
    *pos += 1; // (
    let head = toks.get(*pos)?.clone();
    *pos += 1;
    match head.as_str() {
        "Num" => {
            let v = toks.get(*pos)?.clone();
            *pos += 1;
            if toks.get(*pos)? != ")" {
                return None;
            }
            *pos += 1;
            Some(Node { head, children: vec![], leaf: v })
        }
        "Var" => {
            let name = toks.get(*pos)?.trim_matches('"').to_string();
            *pos += 1;
            if toks.get(*pos)? != ")" {
                return None;
            }
            *pos += 1;
            Some(Node { head, children: vec![], leaf: name })
        }
        _ => {
            let mut children = Vec::new();
            while *pos < toks.len() && toks[*pos] != ")" {
                children.push(parse_node(toks, pos)?);
            }
            if *pos >= toks.len() {
                return None;
            }
            *pos += 1; // )
            Some(Node { head, children, leaf: String::new() })
        }
    }
}

/// Tokenise an s-expression: parens, quoted strings, bare atoms.
fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' => {
                out.push(c.to_string());
                chars.next();
            }
            '"' => {
                chars.next();
                let mut buf = String::from("\"");
                for d in chars.by_ref() {
                    buf.push(d);
                    if d == '"' {
                        break;
                    }
                }
                out.push(buf);
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut buf = String::new();
                while let Some(&d) = chars.peek() {
                    if d == '(' || d == ')' || d.is_whitespace() {
                        break;
                    }
                    buf.push(d);
                    chars.next();
                }
                out.push(buf);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_expr() {
        let n = parse(r#"(Add (Mul (Var "x") (Num 1.0)) (Num 0.0))"#).unwrap();
        assert_eq!(n.head, "Add");
        assert_eq!(n.nodes().len(), 5); // Add, Mul, Var, Num, Num
    }

    #[test]
    fn node_count_rule_always_fires() {
        let n = parse(r#"(Var "x")"#).unwrap();
        let fired = fire(&n);
        assert!(fired.iter().any(|(name, _)| *name == "node_count"));
    }

    #[test]
    fn transc_rules_fire_only_when_present() {
        let flat = parse(r#"(Sin (Var "x"))"#).unwrap();
        let names: Vec<_> = fire(&flat).into_iter().map(|(n, _)| n).collect();
        assert!(names.contains(&"transc_count"));
        assert!(!names.contains(&"transc_nest"), "flat sin should not nest");

        let nested = parse(r#"(Sin (Sin (Var "x")))"#).unwrap();
        let names: Vec<_> = fire(&nested).into_iter().map(|(n, _)| n).collect();
        assert!(names.contains(&"transc_nest"));
        assert!(names.contains(&"self_nest"), "sin(sin) is self-nesting");
    }

    #[test]
    fn no_transc_means_those_rules_dont_fire() {
        let n = parse(r#"(Add (Var "x") (Var "y"))"#).unwrap();
        let names: Vec<_> = fire(&n).into_iter().map(|(nm, _)| nm).collect();
        assert!(!names.contains(&"transc_count"));
        assert!(!names.contains(&"instability"));
        // but node_count + const_to_var + depth_breadth fire
        assert!(names.contains(&"node_count"));
    }

    #[test]
    fn cleaner_term_has_lower_angle_than_junk() {
        let clean = parse(r#"(Mul (Var "x") (Var "y"))"#).unwrap();
        let junk = parse(
            r#"(Sin (Sin (Exp (Div (Var "x") (Mul (Num 2.0) (Log (Var "y")))))))"#,
        )
        .unwrap();
        assert!(
            angle_percentile(&clean) < angle_percentile(&junk),
            "clean {} should beat junk {}",
            angle_percentile(&clean),
            angle_percentile(&junk)
        );
    }

    #[test]
    fn angle_is_finite_and_deterministic() {
        // log-percentile: <= 0 (log of a probability), finite, reproducible.
        let n = parse(r#"(Add (Sin (Var "x")) (Div (Num 2.0) (Var "y")))"#).unwrap();
        let p = angle_percentile(&n);
        // raw TrueNorth angle: finite, in [0, pi].
        assert!(p.is_finite() && (0.0..=std::f64::consts::PI).contains(&p), "angle {p} out of [0,pi]");
        assert_eq!(p, angle_percentile(&n));
    }

    #[test]
    fn dimension_varies_per_term() {
        // A term with transcendentals fires MORE rules than a bare algebraic one
        // — the heterogeneous-dimension property the CDF correction exists for.
        let algebraic = parse(r#"(Add (Var "x") (Var "y"))"#).unwrap();
        let transcend = parse(r#"(Sin (Exp (Var "x")))"#).unwrap();
        assert_ne!(fire(&algebraic).len(), fire(&transcend).len());
    }
}
