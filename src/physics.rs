//! Physics-prior mutation GENERATOR — the sibling of denoise, NOT denoise.
//!
//! This is a PURE one-to-many generator. It receives one gene (a `Math`
//! s-expression) and emits a proliferation of physics-shaped candidate genes,
//! each a structural mutation that biases the form toward what real physical
//! laws look like (squared distances, axis-aligned coordinate pairs, inverse
//! powers, stripped wallpaper, ...). See docs/physics_prior_rules.md.
//!
//! It does NOT evaluate, score, or judge fit. No data rows, no tolerance, no
//! R^2. Generation is the only job; HFF + the extrapolation objective select
//! downstream. Mixing evaluation in here would smuggle selection into the
//! generator.
//!
//! Volume: internally we generate ALL distinct candidates (rules x match-sites
//! x composition, deduplicated to a fixpoint, with an internal safety cap).
//! Then, if there are more than the caller wants, we randomly sample `n` of
//! them (deterministic from `seed`). The full internal proliferation always
//! happens first; the cap only limits what is RETURNED.

/// A generated physics-prior candidate. No fitness — pure structure.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The mutated gene as a `Math` s-expression string.
    pub expr: String,
    /// Catalogue rule that produced this edit (e.g. "A1", "A2", "F").
    pub rule: String,
    /// True if this is a structural leap (changes what the gene computes) that
    /// the caller MUST gate on extrapolation, not holdout. False for edits that
    /// only reshape (e.g. axis re-pairing).
    pub speculative: bool,
}

/// Hard internal cap on how many distinct candidates we materialise before
/// sampling. Generation stops once this many distinct exprs exist — keeps a
/// pathological gene from blowing up memory/time. Tunable; well above any
/// caller's `n`.
const INTERNAL_CAP: usize = 2000;

/// Generate physics-prior candidates for `gene`, then randomly return up to `n`
/// of them.
///
/// * `gene` — a `Math` s-expression string.
/// * `paired_groups` — coordinate axes, e.g. `[["x1","x2"], ["y1","y2"]]`,
///   used by the distance-family rules.
/// * `n` — maximum candidates to RETURN (>= 1). If fewer are generated, all are
///   returned. If more, a uniform random `n` are sampled.
/// * `seed` — makes the random sample reproducible.
///
/// Returns the (sampled) candidates. Generation itself is exhaustive up to the
/// internal cap; the sample is the only thing `n` limits.
pub fn generate(
    gene: &str,
    paired_groups: &[Vec<String>],
    n: usize,
    seed: u64,
) -> Result<Vec<Candidate>, String> {
    let root = parse(gene).ok_or_else(|| format!("could not parse {gene:?}"))?;

    // BFS over reachable mutations. Start from the input; each pass applies
    // every rule at every matching site, feeding new distinct forms back in
    // (composition) until no new form appears or the internal cap is hit.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(root.to_math());
    let mut frontier: Vec<Node> = vec![root];
    // Candidates carry the rule/speculative tag of the edit that *created* them.
    let mut candidates: Vec<Candidate> = Vec::new();

    while let Some(node) = frontier.pop() {
        if candidates.len() >= INTERNAL_CAP {
            break;
        }
        for (mutated, rule, speculative) in one_step_edits(&node, paired_groups) {
            let s = mutated.to_math();
            if s == gene || !seen.insert(s.clone()) {
                continue; // skip the input itself and any form already seen
            }
            candidates.push(Candidate { expr: s, rule, speculative });
            frontier.push(mutated);
            if candidates.len() >= INTERNAL_CAP {
                break;
            }
        }
    }

    // Return all if within budget; otherwise uniformly sample `n` deterministically.
    let n = n.max(1);
    if candidates.len() <= n {
        return Ok(candidates);
    }
    Ok(sample(candidates, n, seed))
}

/// Uniform random sample of `n` items, deterministic from `seed`. Partial
/// Fisher–Yates over indices using an inline LCG (no rng crate dependency).
fn sample(mut items: Vec<Candidate>, n: usize, seed: u64) -> Vec<Candidate> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let len = items.len();
    for i in 0..n {
        let j = i + next() % (len - i);
        items.swap(i, j);
    }
    items.truncate(n);
    items
}

// ---------------------------------------------------------------------------
// Rules — each enumerates one-step edits at a single site. Composition is
// handled by the BFS in `generate` (feeding results back through these).
// ---------------------------------------------------------------------------

/// All one-step edits of `node`: for every rule, at every matching site,
/// produce `(mutated_tree, rule_name, speculative)`.
fn one_step_edits(node: &Node, groups: &[Vec<String>]) -> Vec<(Node, String, bool)> {
    let mut out = Vec::new();
    let sites = collect_sites(node);
    for path in &sites {
        let sub = at_path(node, path);
        // The parent constructor (if any) — lets rules avoid re-firing on their
        // own output (e.g. A2 must not square a Sub already inside a Pow2,
        // which would nest Pow2(Pow2(...)) without bound).
        let parent_op = parent_op_at(node, path);
        for (repl, rule, spec) in site_edits(sub, parent_op.as_deref(), groups) {
            out.push((replace_at_path(node, path, &repl), rule, spec));
        }
    }
    out
}

/// The constructor name of the parent of the node at `path`, or None for root.
fn parent_op_at(root: &Node, path: &[usize]) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let parent = at_path(root, &path[..path.len() - 1]);
    if let Node::App(op, _) = parent {
        Some(op.clone())
    } else {
        None
    }
}

/// The edits applicable to a single subtree `sub`, given its `parent_op`.
fn site_edits(sub: &Node, parent_op: Option<&str>, groups: &[Vec<String>]) -> Vec<(Node, String, bool)> {
    let mut out = Vec::new();
    if let Node::App(op, ch) = sub {
        match (op.as_str(), ch.len()) {
            // A1 — cross-coordinate -> axis-aligned pair promotion (reshape).
            ("Sub", 2) => {
                if let (Node::Var(a), Node::Var(b)) = (&ch[0], &ch[1]) {
                    if let (Some((ga, ia)), Some((gb, ib))) =
                        (locate(a, groups), locate(b, groups))
                    {
                        if ga != gb {
                            if let Some(p) = groups[ga].get(ib) {
                                out.push((
                                    Node::App("Sub".into(), vec![Node::Var(a.clone()), Node::Var(p.clone())]),
                                    "A1".into(),
                                    false,
                                ));
                            }
                            if let Some(p) = groups[gb].get(ia) {
                                out.push((
                                    Node::App("Sub".into(), vec![Node::Var(p.clone()), Node::Var(b.clone())]),
                                    "A1".into(),
                                    false,
                                ));
                            }
                        }
                    }
                }
                // A2 — square a difference (structural leap). Do NOT fire if
                // this Sub is already directly wrapped in Pow2, else
                // composition nests Pow2(Pow2(...)) without bound.
                if parent_op != Some("Pow2") {
                    out.push((Node::App("Pow2".into(), vec![sub.clone()]), "A2".into(), true));
                }
            }
            // E4 — power-law: wrap a multiplicative factor in a small power.
            ("Mul", 2) => {
                // F — strip a wallpaper factor (reshape).
                const WALL: &[&str] = &["Sqrt", "Sin", "Cos", "Tan", "Tanh", "Exp", "ProtectedSqrt"];
                if let Node::App(inner, _) = &ch[1] {
                    if WALL.contains(&inner.as_str()) {
                        out.push((ch[0].clone(), "F".into(), false));
                    }
                }
                if let Node::App(inner, _) = &ch[0] {
                    if WALL.contains(&inner.as_str()) {
                        out.push((ch[1].clone(), "F".into(), false));
                    }
                }
                // E1 — inverse-square a factor (leap): a*b -> a / Pow2(b).
                out.push((
                    Node::App("Div".into(), vec![ch[0].clone(), Node::App("Pow2".into(), vec![ch[1].clone()])]),
                    "E1".into(),
                    true,
                ));
            }
            // A2 also applies to Add as the (a+b) -> not squared; skip. Div
            // divisor inverse handled below.
            ("Div", 2) => {
                // E1 — promote divisor to its square (inverse-square leap):
                // a / b -> a / Pow2(b). Do NOT fire if the divisor is already
                // Pow2, else it nests a / Pow2(Pow2(...)) without bound.
                let divisor_is_pow2 = matches!(&ch[1], Node::App(op, _) if op == "Pow2");
                if !divisor_is_pow2 {
                    out.push((
                        Node::App("Div".into(), vec![ch[0].clone(), Node::App("Pow2".into(), vec![ch[1].clone()])]),
                        "E1".into(),
                        true,
                    ));
                }
            }
            _ => {}
        }
    }
    out
}

/// Which (group, index) a variable name belongs to in the paired groups.
fn locate(name: &str, groups: &[Vec<String>]) -> Option<(usize, usize)> {
    for (gi, g) in groups.iter().enumerate() {
        if let Some(idx) = g.iter().position(|v| v == name) {
            return Some((gi, idx));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Math tree + path-based zipper (parse / serialise / site collection / replace)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Num(f64),
    Var(String),
    App(String, Vec<Node>),
}

impl Node {
    fn to_math(&self) -> String {
        match self {
            Node::Num(v) => format!("(Num {})", fmt_f64(*v)),
            Node::Var(n) => format!("(Var \"{n}\")"),
            Node::App(op, ch) => {
                let parts: Vec<String> = ch.iter().map(Node::to_math).collect();
                format!("({op} {})", parts.join(" "))
            }
        }
    }
}

fn fmt_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() { format!("{v:.1}") } else { format!("{v}") }
}

/// All node paths in pre-order (root = empty path).
fn collect_sites(root: &Node) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    fn go(n: &Node, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        out.push(path.clone());
        if let Node::App(_, ch) = n {
            for (i, c) in ch.iter().enumerate() {
                path.push(i);
                go(c, path, out);
                path.pop();
            }
        }
    }
    let mut p = Vec::new();
    go(root, &mut p, &mut out);
    out
}

/// The subtree at `path`.
fn at_path<'a>(root: &'a Node, path: &[usize]) -> &'a Node {
    let mut cur = root;
    for &i in path {
        if let Node::App(_, ch) = cur {
            cur = &ch[i];
        }
    }
    cur
}

/// A copy of `root` with the subtree at `path` replaced by `repl`.
fn replace_at_path(root: &Node, path: &[usize], repl: &Node) -> Node {
    if path.is_empty() {
        return repl.clone();
    }
    match root {
        Node::App(op, ch) => {
            let mut new_ch = ch.clone();
            new_ch[path[0]] = replace_at_path(&ch[path[0]], &path[1..], repl);
            Node::App(op.clone(), new_ch)
        }
        other => other.clone(),
    }
}

fn parse(s: &str) -> Option<Node> {
    let toks = tok(s);
    let mut pos = 0;
    let n = pparse(&toks, &mut pos)?;
    if pos == toks.len() { Some(n) } else { None }
}

fn tok(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' => { out.push(c.to_string()); chars.next(); }
            '"' => {
                let mut t = String::from("\"");
                chars.next();
                for c2 in chars.by_ref() { if c2 == '"' { break; } t.push(c2); }
                t.push('"');
                out.push(t);
            }
            c if c.is_whitespace() => { chars.next(); }
            _ => {
                let mut t = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '(' || c2 == ')' || c2.is_whitespace() { break; }
                    t.push(c2);
                    chars.next();
                }
                out.push(t);
            }
        }
    }
    out
}

fn pparse(toks: &[String], pos: &mut usize) -> Option<Node> {
    if toks.get(*pos)? != "(" { return None; }
    *pos += 1;
    let head = toks.get(*pos)?.clone();
    *pos += 1;
    let node = match head.as_str() {
        "Num" => { let v: f64 = toks.get(*pos)?.parse().ok()?; *pos += 1; Node::Num(v) }
        "Var" => { let n = toks.get(*pos)?.trim_matches('"').to_string(); *pos += 1; Node::Var(n) }
        ctor => {
            let mut ch = Vec::new();
            while *pos < toks.len() && toks[*pos] != ")" { ch.push(pparse(toks, pos)?); }
            Node::App(ctor.to_string(), ch)
        }
    };
    if toks.get(*pos)? != ")" { return None; }
    *pos += 1;
    Some(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups() -> Vec<Vec<String>> {
        vec![
            vec!["x1".into(), "x2".into()],
            vec!["y1".into(), "y2".into()],
            vec!["z1".into(), "z2".into()],
        ]
    }

    #[test]
    fn generates_many_from_one_gene() {
        // A modestly complex gene yields a proliferation of candidates.
        let gene = r#"(Div (Var "m1") (Sub (Var "x2") (Var "y1")))"#;
        let all = generate(gene, &groups(), 1000, 0).unwrap();
        assert!(all.len() > 3, "expected a proliferation, got {}", all.len());
        // never returns the input itself
        assert!(all.iter().all(|c| c.expr != gene));
        // all distinct
        let mut exprs: Vec<&str> = all.iter().map(|c| c.expr.as_str()).collect();
        exprs.sort();
        let n_before = exprs.len();
        exprs.dedup();
        assert_eq!(n_before, exprs.len(), "candidates must be distinct");
    }

    #[test]
    fn a1_axis_promotion_present() {
        let gene = r#"(Sub (Var "x2") (Var "y1"))"#;
        let all = generate(gene, &groups(), 1000, 0).unwrap();
        let a1: Vec<&str> = all.iter().filter(|c| c.rule == "A1").map(|c| c.expr.as_str()).collect();
        assert!(a1.iter().any(|e| *e == r#"(Sub (Var "x2") (Var "x1"))"#), "{a1:?}");
        assert!(a1.iter().any(|e| *e == r#"(Sub (Var "y2") (Var "y1"))"#), "{a1:?}");
    }

    #[test]
    fn a2_square_is_speculative() {
        let gene = r#"(Sub (Var "a") (Var "b"))"#;
        let all = generate(gene, &[], 1000, 0).unwrap();
        let a2 = all.iter().find(|c| c.rule == "A2").expect("A2");
        assert_eq!(a2.expr, r#"(Pow2 (Sub (Var "a") (Var "b")))"#);
        assert!(a2.speculative);
    }

    #[test]
    fn cap_limits_returned_count_not_generation() {
        let gene = r#"(Div (Mul (Var "m1") (Var "m2")) (Sub (Var "x2") (Var "y1")))"#;
        // Ask for 5; must return exactly 5 (generation produced more).
        let five = generate(gene, &groups(), 5, 7).unwrap();
        assert_eq!(five.len(), 5);
        // Deterministic in seed.
        let five_again = generate(gene, &groups(), 5, 7).unwrap();
        assert_eq!(five, five_again, "same seed -> same sample");
        // Different seed generally differs (not asserted strictly), but asking
        // for 1 must give exactly 1.
        assert_eq!(generate(gene, &groups(), 1, 0).unwrap().len(), 1);
    }
}
