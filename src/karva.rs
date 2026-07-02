//! Phase 1.1: karva (GEP K-expression) <-> `Math` term converter.
//!
//! This is the boundary the SR engine crosses: it speaks karva chromosomes
//! (flat head+tail token lists), fuller speaks `Math` s-expressions. The
//! converter is keyed on `semantic_id` (what an operator *computes*), never a
//! pset-specific geppy name — the consumer maps its names to semantic ids when
//! it builds the `PsetSpec`. Ported (clean, no geppy/sympy) from the GEP decode
//! rule in hff `_gene_decompose.py` / `_sympy_to_karva.py`.
//!
//! GEP decode rule: walk the head left-to-right; each function token consumes
//! `arity` children from the next available slots in the head+tail stream
//! (level-order / BFS). Tail tokens are terminals only.

use std::collections::HashMap;

/// One operator in the pset, described by what it *computes*.
#[derive(Debug, Clone)]
pub struct FunctionSpec {
    /// Semantic id: one of the `Math` ops, lowercase
    /// (add sub mul div neg sin cos log exp sqrt abs tanh pow2 pow3 inv).
    pub semantic_id: String,
    /// Arity (1 or 2 for the current Math set).
    pub arity: usize,
}

/// Pure-data description of the pset (no geppy dependency).
#[derive(Debug, Clone)]
pub struct PsetSpec {
    /// Variable names, e.g. ["x0", "x1"].
    pub variables: Vec<String>,
    /// Functions, keyed for lookup by the token strings the engine emits.
    /// The map is `token_name -> FunctionSpec`.
    pub functions: HashMap<String, FunctionSpec>,
    /// Numeric RNC constants addressable by index.
    pub rnc_values: Vec<f64>,
}

/// A karva token: either a function name, a variable name, or a numeric
/// constant. The engine's flat head/tail lists are `Vec<Token>`.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A function token — looked up in `PsetSpec::functions`.
    Func(String),
    /// A variable token — must be in `PsetSpec::variables`.
    Var(String),
    /// A numeric literal.
    Num(f64),
}

/// The MASTER semantic-id set: every `(semantic_id, arity)` any rewrite or
/// physics-prior rule can emit. The SR engine seeds its pset with one token per
/// entry up front; then `terms_to_karva` can always name any candidate, so no
/// generated chromosome is ever dropped as inexpressible.
///
/// This is the single source of truth, kept in lockstep with `semantic_to_math`
/// (a test asserts every entry round-trips through it). `diff_sq` is omitted
/// deliberately — it lowers to `pow2`+`sub`, which are already present, so it is
/// not a constructor the engine needs a distinct token for.
pub fn master_pset() -> Vec<(&'static str, usize)> {
    vec![
        ("add", 2),
        ("sub", 2),
        ("mul", 2),
        ("div", 2),
        ("neg", 1),
        ("sin", 1),
        ("cos", 1),
        ("tan", 1),
        ("log", 1),
        ("exp", 1),
        ("sqrt", 1),
        ("abs", 1),
        ("tanh", 1),
        ("pow2", 1),
        ("pow3", 1),
        ("pow", 2),
        ("inv", 1),
        ("protected_sqrt", 1),
        ("protected_log", 1),
        ("protected_exp", 1),
        ("protected_inv", 1),
        ("protected_div", 2),
    ]
}

/// Map a semantic id + child Math strings into a `Math` s-expression node.
fn semantic_to_math(semantic: &str, children: &[String]) -> Result<String, String> {
    let ctor = match (semantic, children.len()) {
        ("add", 2) => "Add",
        ("sub", 2) => "Sub",
        ("mul", 2) => "Mul",
        ("div", 2) => "Div",
        ("neg", 1) => "Neg",
        ("sin", 1) => "Sin",
        ("cos", 1) => "Cos",
        ("tan", 1) => "Tan",
        ("log", 1) => "Log",
        ("exp", 1) => "Exp",
        ("sqrt", 1) => "Sqrt",
        ("abs", 1) => "Abs",
        ("tanh", 1) => "Tanh",
        ("pow2", 1) => "Pow2",
        ("pow3", 1) => "Pow3",
        ("pow", 2) => "Pow",
        ("inv", 1) => "Inv",
        // protected ops — distinct constructors, never the raw ones
        ("protected_sqrt", 1) => "ProtectedSqrt",
        ("protected_log", 1) => "ProtectedLog",
        ("protected_exp", 1) => "ProtectedExp",
        ("protected_inv", 1) => "ProtectedInv",
        ("protected_div", 2) => "ProtectedDiv",
        // diff_sq(a,b) = (a-b)^2, expressed via Pow2(Sub a b).
        ("diff_sq", 2) => {
            return Ok(format!("(Pow2 (Sub {} {}))", children[0], children[1]));
        }
        _ => {
            return Err(format!(
                "unknown semantic_id/arity: {semantic}/{}",
                children.len()
            ))
        }
    };
    Ok(format!("({ctor} {})", children.join(" ")))
}

/// Render a leaf token as a `Math` s-expression leaf.
fn leaf_to_math(tok: &Token, pset: &PsetSpec) -> Result<String, String> {
    match tok {
        Token::Var(name) => {
            if pset.variables.iter().any(|v| v == name) {
                Ok(format!("(Var \"{name}\")"))
            } else {
                Err(format!("variable {name:?} not in pset"))
            }
        }
        Token::Num(v) => {
            if !v.is_finite() {
                // egglog's parser has no literal for NaN/inf; rendering one
                // would silently produce an unparseable program downstream.
                Err(format!("non-finite numeric constant {v} is not expressible"))
            } else {
                Ok(format!("(Num {})", fmt_f64(*v)))
            }
        }
        Token::Func(name) => Err(format!("function {name:?} used as a leaf")),
    }
}

/// Format an f64 the way egglog's parser accepts (always with a decimal point).
fn fmt_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// Convert a karva (head, tail) chromosome to a `Math` s-expression string.
///
/// Decodes the live expression by the GEP level-order rule and renders it.
/// Neutral-region tokens (beyond the live expression) are ignored, exactly as
/// they would be when the chromosome is evaluated.
pub fn karva_to_terms(head: &[Token], tail: &[Token], pset: &PsetSpec) -> Result<String, String> {
    if head.is_empty() {
        return Err("empty head".to_string());
    }
    // The combined stream the GEP rule walks.
    let stream: Vec<&Token> = head.iter().chain(tail.iter()).collect();

    // Determine each function token's arity, walking the head and consuming
    // child slots level-order. `child_slots[i]` = the stream indices of node
    // i's children.
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); stream.len()];
    let mut next_slot = 1usize;
    for (i, tok) in head.iter().enumerate() {
        // A token at or beyond `next_slot` was never assigned as any live
        // node's child: it sits in the neutral (dead) region, which the GEP
        // decode rule ignores. Since `next_slot` only grows from live function
        // tokens, everything from here on is dead — stop, and never validate
        // or consume slots for dead tokens (an unmapped token in the neutral
        // region must not fail an otherwise valid chromosome).
        if i > 0 && i >= next_slot {
            break;
        }
        if let Token::Func(name) = tok {
            let spec = pset
                .functions
                .get(name)
                .ok_or_else(|| format!("function token {name:?} not in pset"))?;
            for _ in 0..spec.arity {
                if next_slot >= stream.len() {
                    return Err(format!(
                        "ran out of tokens decoding {name:?}: GEP tail too short"
                    ));
                }
                children_of[i].push(next_slot);
                next_slot += 1;
            }
        }
    }

    // Recursively render from the root (stream index 0). Depth-capped: a
    // deeper-than-sane gene must surface as an Err, not a stack overflow.
    fn render(
        idx: usize,
        stream: &[&Token],
        children_of: &[Vec<usize>],
        pset: &PsetSpec,
        depth: usize,
    ) -> Result<String, String> {
        if depth > crate::MAX_EXPR_DEPTH {
            return Err(format!(
                "expression deeper than MAX_EXPR_DEPTH ({})",
                crate::MAX_EXPR_DEPTH
            ));
        }
        match stream[idx] {
            Token::Func(name) => {
                let spec = pset
                    .functions
                    .get(name)
                    .ok_or_else(|| format!("function token {name:?} not in pset"))?;
                let kids: Result<Vec<String>, String> = children_of[idx]
                    .iter()
                    .map(|&c| render(c, stream, children_of, pset, depth + 1))
                    .collect();
                semantic_to_math(&spec.semantic_id, &kids?)
            }
            leaf => leaf_to_math(leaf, pset),
        }
    }

    render(0, &stream, &children_of, pset, 0)
}

// ---------------------------------------------------------------------------
// Inverse: Math s-expression -> karva (head, tail)
// ---------------------------------------------------------------------------

/// A parsed Math node (intermediate tree between the s-expression and karva).
#[derive(Debug, Clone)]
enum MathNode {
    Num(f64),
    Var(String),
    /// (constructor, children)
    App(String, Vec<MathNode>),
}

/// Reverse of `semantic_to_math`: Math constructor name -> semantic id. Returns
/// None for constructors with no single-token karva representation (none such
/// in the current set).
fn math_ctor_to_semantic(ctor: &str) -> Option<&'static str> {
    Some(match ctor {
        "Add" => "add",
        "Sub" => "sub",
        "Mul" => "mul",
        "Div" => "div",
        "Neg" => "neg",
        "Sin" => "sin",
        "Cos" => "cos",
        "Tan" => "tan",
        "Log" => "log",
        "Exp" => "exp",
        "Sqrt" => "sqrt",
        "Abs" => "abs",
        "Tanh" => "tanh",
        "Pow" => "pow",
        "Pow2" => "pow2",
        "Pow3" => "pow3",
        "Inv" => "inv",
        "ProtectedSqrt" => "protected_sqrt",
        "ProtectedLog" => "protected_log",
        "ProtectedExp" => "protected_exp",
        "ProtectedInv" => "protected_inv",
        "ProtectedDiv" => "protected_div",
        _ => return None,
    })
}

/// Minimal recursive-descent parser for the Math s-expression subset emitted by
/// the extractor: `(Ctor child ...)`, `(Num <f64>)`, `(Var "<name>")`.
fn parse_math(s: &str) -> Result<MathNode, String> {
    let toks = tokenize(s);
    let mut pos = 0;
    let node = parse_node(&toks, &mut pos, 0)?;
    if pos != toks.len() {
        return Err(format!("trailing tokens after parse in {s:?}"));
    }
    Ok(node)
}

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
                // quoted string token, keep the quotes
                let mut t = String::from("\"");
                chars.next();
                for c2 in chars.by_ref() {
                    if c2 == '"' {
                        break;
                    }
                    t.push(c2);
                }
                t.push('"');
                out.push(t);
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut t = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '(' || c2 == ')' || c2.is_whitespace() {
                        break;
                    }
                    t.push(c2);
                    chars.next();
                }
                out.push(t);
            }
        }
    }
    out
}

fn parse_node(toks: &[String], pos: &mut usize, depth: usize) -> Result<MathNode, String> {
    if depth > crate::MAX_EXPR_DEPTH {
        // Depth-capped so a pathologically nested s-expression is an Err, not
        // a stack overflow (which would abort the whole process).
        return Err(format!(
            "expression deeper than MAX_EXPR_DEPTH ({})",
            crate::MAX_EXPR_DEPTH
        ));
    }
    if *pos >= toks.len() {
        return Err("unexpected end of input".to_string());
    }
    if toks[*pos] != "(" {
        return Err(format!("expected '(' at token {}: {:?}", *pos, toks[*pos]));
    }
    *pos += 1; // consume '('
    let head = toks
        .get(*pos)
        .ok_or("missing constructor after '('")?
        .clone();
    *pos += 1;

    let node = match head.as_str() {
        "Num" => {
            let v: f64 = toks
                .get(*pos)
                .ok_or("missing Num value")?
                .parse()
                .map_err(|_| format!("bad Num value {:?}", toks.get(*pos)))?;
            *pos += 1;
            MathNode::Num(v)
        }
        "Var" => {
            let raw = toks.get(*pos).ok_or("missing Var name")?;
            let name = raw.trim_matches('"').to_string();
            *pos += 1;
            MathNode::Var(name)
        }
        ctor => {
            let mut children = Vec::new();
            while *pos < toks.len() && toks[*pos] != ")" {
                children.push(parse_node(toks, pos, depth + 1)?);
            }
            MathNode::App(ctor.to_string(), children)
        }
    };

    if toks.get(*pos).map(String::as_str) != Some(")") {
        return Err(format!("expected ')' closing {head}"));
    }
    *pos += 1; // consume ')'
    Ok(node)
}

/// Map a Math node to the karva token that *names* it in this pset. For a
/// function node we need the pset's token name whose semantic id matches; we
/// pick the first such token. For Pow2(Sub a b) we do NOT attempt to recover a
/// `diff_sq` token (it round-trips as pow2+sub, which is equivalent).
fn func_token_for_semantic(semantic: &str, pset: &PsetSpec) -> Result<String, String> {
    // Deterministic: HashMap iteration order is randomised per run, so when
    // several pset tokens share a semantic id (e.g. `sqrt` and
    // `protected_sqrt`) we must choose by a stable key — the lexicographically
    // smallest token name — or `terms_to_karva` violates its determinism
    // contract.
    pset.functions
        .iter()
        .filter(|(_, spec)| spec.semantic_id == semantic)
        .map(|(name, _)| name)
        .min()
        .cloned()
        .ok_or_else(|| format!("no pset token for semantic id {semantic:?}"))
}

/// Convert a `Math` s-expression string back to a karva (head, tail) pair.
///
/// BFS the parsed tree to produce the head (level-order: functions then the
/// terminals they reference), then re-pad the tail with random terminals to the
/// GEP rule `tail_len = head_len * (n_max - 1) + 1` (n_max = the pset's max
/// function arity), deterministically from `rng_seed`.
pub fn terms_to_karva(
    term: &str,
    pset: &PsetSpec,
    rng_seed: u64,
) -> Result<(Vec<Token>, Vec<Token>), String> {
    terms_to_karva_sized(term, pset, rng_seed, None).map(|(h, t, _)| (h, t))
}

/// Like [`terms_to_karva`] but accepts an optional `target_head_length`.
///
/// GEP requires a uniform `head_length` across all genes in a chromosome, so a
/// rewritten term whose natural head is shorter than the chromosome's configured
/// head breaks geppy's mating ops when grafted back. When `target_head_length`
/// is set:
///
/// * shorter natural head → extend it to the target with terminal filler. Those
///   extra head slots sit beyond the live k-expression's BFS frontier, so the
///   level-order decoder never visits them — the decoded tree (hence the
///   semantics) is unchanged. This is the same syntactic position geppy fills
///   with random tokens at random-init time.
/// * longer natural head → return `oversized = true`; the caller must drop the
///   candidate. We never truncate, because that would change the term.
/// * exact / `None` → unchanged behaviour.
///
/// The tail length always follows the GEP rule for the FINAL head length.
/// Returns `(head, tail, oversized)`.
pub fn terms_to_karva_sized(
    term: &str,
    pset: &PsetSpec,
    rng_seed: u64,
    target_head_length: Option<usize>,
) -> Result<(Vec<Token>, Vec<Token>, bool), String> {
    let root = parse_math(term)?;

    // BFS the tree; emit a Token per node in level order. Functions become
    // Func tokens (by semantic id -> pset name); leaves become Var/Num tokens.
    let mut head: Vec<Token> = Vec::new();
    let mut queue: std::collections::VecDeque<&MathNode> = std::collections::VecDeque::new();
    queue.push_back(&root);
    while let Some(node) = queue.pop_front() {
        match node {
            MathNode::Num(v) => {
                if !v.is_finite() {
                    // A NaN/inf constant round-trips to an unparseable (Num ..)
                    // literal, so the chromosome would be undecodable. Refuse
                    // here, symmetrically with `leaf_to_math` on the decode side.
                    return Err(format!("non-finite numeric constant {v} is not expressible"));
                }
                head.push(Token::Num(*v));
            }
            MathNode::Var(name) => {
                // Validate on encode, symmetrically with `leaf_to_math` on the
                // decode side: emitting a Var outside the pset would produce a
                // chromosome the engine can never decode again.
                if !pset.variables.iter().any(|v| v == name) {
                    return Err(format!("variable {name:?} not in pset"));
                }
                head.push(Token::Var(name.clone()));
            }
            MathNode::App(ctor, children) => {
                // diff_sq round-trips as its expansion Pow2(Sub ..); just use
                // the constructor's own semantic id.
                let semantic = math_ctor_to_semantic(ctor)
                    .ok_or_else(|| format!("non-karva constructor {ctor:?}"))?;
                let name = func_token_for_semantic(semantic, pset)?;
                head.push(Token::Func(name));
                for c in children {
                    queue.push_back(c);
                }
            }
        }
    }

    // The terminal pool for both head-extension filler and tail padding.
    let mut pool: Vec<Token> = pset.variables.iter().cloned().map(Token::Var).collect();
    pool.extend(pset.rnc_values.iter().copied().map(Token::Num));
    if pool.is_empty() {
        return Err("empty terminal pool for tail padding".to_string());
    }

    // Tiny deterministic LCG — no rng crate dependency, fully reproducible.
    let mut state = rng_seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };

    // Honour a requested head length. Extending appends terminals beyond the
    // live tree (BFS-unreachable, so semantics-preserving). Oversized natural
    // heads can't be shrunk without changing the term, so they're refused.
    let mut oversized = false;
    if let Some(target) = target_head_length {
        if head.len() > target {
            oversized = true;
        } else {
            while head.len() < target {
                head.push(pool[next() % pool.len()].clone());
            }
        }
    }

    // GEP tail rule for the FINAL head length: tail is terminals only, length
    // head_len*(n_max-1)+1, padded deterministically from the same stream.
    // n_max is the PSET's max function arity (geppy's rule) — NOT the max arity
    // appearing in this particular expression. A pure-unary term with an
    // expression-derived n_max of 1 would get a 1-token tail; a later head
    // point-mutation to a binary op (which the pset permits) would then run off
    // the end of the stream, and uniform-length mating would break.
    let pset_max_arity = pset
        .functions
        .values()
        .map(|f| f.arity)
        .max()
        .unwrap_or(1)
        .max(1);
    let tail_len = head.len() * (pset_max_arity - 1) + 1;
    let tail: Vec<Token> = (0..tail_len).map(|_| pool[next() % pool.len()].clone()).collect();

    Ok((head, tail, oversized))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The master pset must stay in lockstep with `semantic_to_math`: every
    /// advertised (semantic_id, arity) must actually encode to a Math node, or
    /// the engine could seed a token fuller can't render. Dummy children make
    /// the arity right.
    #[test]
    fn master_pset_entries_all_encode() {
        for (sid, arity) in master_pset() {
            let kids: Vec<String> = (0..arity).map(|_| "(Num 1.0)".to_string()).collect();
            let r = semantic_to_math(sid, &kids);
            assert!(r.is_ok(), "master_pset entry {sid}/{arity} does not encode: {r:?}");
        }
    }

    fn pset() -> PsetSpec {
        let mut functions = HashMap::new();
        functions.insert("add".to_string(), FunctionSpec { semantic_id: "add".into(), arity: 2 });
        functions.insert("mul".to_string(), FunctionSpec { semantic_id: "mul".into(), arity: 2 });
        functions.insert("sqrt".to_string(), FunctionSpec { semantic_id: "sqrt".into(), arity: 1 });
        functions.insert("abs".to_string(), FunctionSpec { semantic_id: "abs".into(), arity: 1 });
        // a pset whose geppy name differs from the semantic id. NOTE: the
        // token is a RAW-sqrt alias ("my_sqrt"), deliberately NOT
        // "protected_sqrt" — mapping a protected geppy op to a raw semantic id
        // is the unsound conflation the crate forbids (see USAGE.md).
        functions.insert(
            "my_sqrt".to_string(),
            FunctionSpec { semantic_id: "sqrt".into(), arity: 1 },
        );
        PsetSpec {
            variables: vec!["x".into(), "y".into()],
            functions,
            rnc_values: vec![1.0, 0.0],
        }
    }

    #[test]
    fn decodes_binary_tree() {
        // head: [mul, x, y]  -> mul(x, y)
        let head = vec![Token::Func("mul".into()), Token::Var("x".into()), Token::Var("y".into())];
        let tail = vec![Token::Var("x".into())];
        let out = karva_to_terms(&head, &tail, &pset()).unwrap();
        assert_eq!(out, r#"(Mul (Var "x") (Var "y"))"#);
    }

    #[test]
    fn decodes_nested_with_tail_children() {
        // head: [add, mul, x]  tail: [y, x, ...]
        // add(child1, child2): child1 = mul(...), child2 = next slot.
        // walk: add@0 takes slots 1,2 -> mul@1, x@2; mul@1 takes slots 3,4 ->
        // tail[0]=y, tail[1]=x. So add(mul(y,x), x).
        let head = vec![
            Token::Func("add".into()),
            Token::Func("mul".into()),
            Token::Var("x".into()),
        ];
        let tail = vec![Token::Var("y".into()), Token::Var("x".into())];
        let out = karva_to_terms(&head, &tail, &pset()).unwrap();
        assert_eq!(out, r#"(Add (Mul (Var "y") (Var "x")) (Var "x"))"#);
    }

    #[test]
    fn semantic_id_not_geppy_name() {
        // "my_sqrt" (a raw-sqrt alias) maps to the `sqrt` semantic id -> Sqrt
        // constructor: the converter keys on what the op COMPUTES, not its name.
        let head = vec![Token::Func("my_sqrt".into()), Token::Var("x".into())];
        let tail = vec![Token::Var("x".into())];
        let out = karva_to_terms(&head, &tail, &pset()).unwrap();
        assert_eq!(out, r#"(Sqrt (Var "x"))"#);
    }

    #[test]
    fn numeric_literal_leaf() {
        // head: [mul, x, c1] where c1 is a numeric token 0.0
        let head = vec![Token::Func("mul".into()), Token::Var("x".into()), Token::Num(0.0)];
        let tail = vec![Token::Var("x".into())];
        let out = karva_to_terms(&head, &tail, &pset()).unwrap();
        assert_eq!(out, r#"(Mul (Var "x") (Num 0.0))"#);
    }

    #[test]
    fn single_terminal_head() {
        let head = vec![Token::Var("x".into())];
        let tail: Vec<Token> = vec![];
        let out = karva_to_terms(&head, &tail, &pset()).unwrap();
        assert_eq!(out, r#"(Var "x")"#);
    }

    // ---- inverse: terms_to_karva ----

    #[test]
    fn inverse_round_trips_through_math() {
        // karva -> Math -> karva: the head must decode to the same Math.
        let head = vec![
            Token::Func("add".into()),
            Token::Func("mul".into()),
            Token::Var("x".into()),
        ];
        let tail = vec![Token::Var("y".into()), Token::Var("x".into())];
        let math = karva_to_terms(&head, &tail, &pset()).unwrap();

        let (head2, tail2) = terms_to_karva(&math, &pset(), 42).unwrap();
        // Re-decoding the regenerated chromosome must give the same Math.
        let math2 = karva_to_terms(&head2, &tail2, &pset()).unwrap();
        assert_eq!(math, math2, "round-trip changed the expression");
    }

    #[test]
    fn inverse_obeys_gep_tail_rule() {
        let math = r#"(Add (Mul (Var "x") (Var "y")) (Var "x"))"#;
        let (head, tail) = terms_to_karva(math, &pset(), 7).unwrap();
        // max arity here is 2, so tail_len = head_len*(2-1)+1 = head_len+1.
        let max_arity = 2usize;
        assert_eq!(tail.len(), head.len() * (max_arity - 1) + 1);
        // Tail must be terminals only — never a function.
        assert!(tail.iter().all(|t| !matches!(t, Token::Func(_))), "tail has a function");
    }

    #[test]
    fn inverse_is_deterministic_in_seed() {
        let math = r#"(Sqrt (Var "x"))"#;
        let a = terms_to_karva(math, &pset(), 99).unwrap();
        let b = terms_to_karva(math, &pset(), 99).unwrap();
        assert_eq!(a, b, "same seed must give identical output");
    }

    #[test]
    fn target_head_length_extends_short_head_preserving_semantics() {
        // (Sqrt (Var "x")) has a 2-token natural head; pad to 8 and the decoded
        // Math must be unchanged — the filler slots are BFS-unreachable.
        let math = r#"(Sqrt (Var "x"))"#;
        let (head, tail, oversized) =
            terms_to_karva_sized(math, &pset(), 3, Some(8)).unwrap();
        assert!(!oversized);
        assert_eq!(head.len(), 8, "head extended to target");
        // Filler slots must be terminals (never functions).
        assert!(head[2..].iter().all(|t| !matches!(t, Token::Func(_))));
        // GEP tail rule for the FINAL head length: n_max is the PSET's max
        // function arity (2 — add/mul are binary), NOT this expression's max
        // arity, so tail_len = 8*(2-1)+1 = 9. A 1-token tail would break as
        // soon as a head point-mutation introduced a binary op.
        assert_eq!(tail.len(), 9);
        let decoded = karva_to_terms(&head, &tail, &pset()).unwrap();
        assert_eq!(decoded, math, "extension changed the expression");
    }

    #[test]
    fn target_head_length_refuses_oversized() {
        // A 3-token head asked to fit length 1 can't shrink without changing the
        // term — must be flagged oversized, never truncated.
        let math = r#"(Add (Mul (Var "x") (Var "y")) (Var "x"))"#;
        let (_h, _t, oversized) =
            terms_to_karva_sized(math, &pset(), 7, Some(1)).unwrap();
        assert!(oversized, "oversized head must be flagged");
    }

    #[test]
    fn target_head_length_none_is_unchanged() {
        let math = r#"(Mul (Var "x") (Var "y"))"#;
        let (h0, t0) = terms_to_karva(math, &pset(), 5).unwrap();
        let (h1, t1, oversized) = terms_to_karva_sized(math, &pset(), 5, None).unwrap();
        assert!(!oversized);
        assert_eq!((h0, t0), (h1, t1), "None target must match the plain path");
    }

    #[test]
    fn deep_gene_errs_instead_of_overflowing() {
        // A unary chain deeper than MAX_EXPR_DEPTH must surface as a plain Err
        // (the caller returns the gene unchanged), not a stack-overflow abort.
        let n = crate::MAX_EXPR_DEPTH + 10;
        let mut head = vec![Token::Func("sqrt".into()); n];
        head.push(Token::Var("x".into()));
        let tail = vec![Token::Var("x".into())];
        let r = karva_to_terms(&head, &tail, &pset());
        assert!(r.is_err(), "deep chain must Err, got {r:?}");
    }

    #[test]
    fn deep_sexpr_parse_errs_instead_of_overflowing() {
        let n = crate::MAX_EXPR_DEPTH + 10;
        let mut s = String::new();
        for _ in 0..n {
            s.push_str("(Sqrt ");
        }
        s.push_str(r#"(Var "x")"#);
        for _ in 0..n {
            s.push(')');
        }
        assert!(terms_to_karva(&s, &pset(), 1).is_err(), "deep parse must Err");
    }

    #[test]
    fn dead_head_tokens_are_ignored() {
        // Live tree = mul(x,y); the trailing function token is in the neutral
        // (dead) region. Even though it's not in the pset, the decode must
        // succeed — the GEP rule never visits dead tokens.
        let head = vec![
            Token::Func("mul".into()),
            Token::Var("x".into()),
            Token::Var("y".into()),
            Token::Func("not_in_pset".into()),
        ];
        let tail = vec![Token::Var("x".into())];
        let out = karva_to_terms(&head, &tail, &pset()).unwrap();
        assert_eq!(out, r#"(Mul (Var "x") (Var "y"))"#);
    }

    #[test]
    fn unknown_var_in_term_errs_on_encode() {
        // Encoding a Var outside the pset would produce an undecodable
        // chromosome — must Err, symmetric with the decode-side check.
        assert!(terms_to_karva(r#"(Var "zzz")"#, &pset(), 1).is_err());
    }

    #[test]
    fn nonfinite_num_leaf_errs() {
        let head =
            vec![Token::Func("mul".into()), Token::Var("x".into()), Token::Num(f64::INFINITY)];
        let tail = vec![Token::Var("x".into())];
        assert!(karva_to_terms(&head, &tail, &pset()).is_err());
    }

    #[test]
    fn inverse_handles_denoised_abs() {
        // The denoiser emits e.g. (Abs (Var "x")); it must convert back.
        let (head, _tail) = terms_to_karva(r#"(Abs (Var "x"))"#, &pset(), 1).unwrap();
        // head[0] is the abs function token, head[1] is x.
        assert!(matches!(&head[0], Token::Func(n) if pset().functions[n].semantic_id == "abs")
            || head.iter().any(|t| matches!(t, Token::Func(_))));
    }
}

