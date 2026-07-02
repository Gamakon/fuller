//! BF source string <-> Prog s-expression conversions.
//!
//! The `Prog` datatype is a flat cons-list where each constructor carries its
//! "rest" continuation as the last argument:
//!
//!   `+-` parses to `(Inc (Dec (Nil)))`
//!   `[+]` parses to `(Loop (Inc (Nil)) (Nil))`
//!
//! `parse_bf(source)` converts raw BF source to an egglog-ready s-expression
//! string. `unparse_bf(sexpr)` goes the other direction.

/// Max bracket-NESTING depth accepted by the parser. This bounds the only
/// recursion left in this module: `parse_prog_str` recurses once per `[`,
/// never per op (the sequence walk is a loop, and the unparser is fully
/// iterative), so a cap on genuine loop nesting is both sufficient for stack
/// safety (256 × small frame ≪ the 2MB default test/rayon stacks, same
/// arithmetic as `crate::MAX_EXPR_DEPTH`) and semantically harmless — no real
/// program nests 256 loops. Straight-line program LENGTH is unlimited.
const MAX_LOOP_DEPTH: usize = 256;

// ---------------------------------------------------------------------------
// BF source -> Prog s-expression string (direct string building)
// ---------------------------------------------------------------------------

/// Convert a BF source string to a Prog s-expression string.
///
/// Non-BF characters are silently ignored (they're comments in standard BF).
/// Returns an error on unmatched brackets or loop nesting deeper than
/// [`MAX_LOOP_DEPTH`].
pub fn parse_bf(source: &str) -> Result<String, String> {
    let chars: Vec<char> = source.chars().filter(|c| "+-<>.,[]".contains(*c)).collect();
    let mut pos = 0;
    parse_prog_str(&chars, &mut pos, false, 0)
}

/// Inner parser: returns the s-expression string for the ops from `pos`
/// onward. Stops at `]` if `in_loop` is true (caller consumes `]`).
///
/// Recursion is per BRACKET only (depth-capped); the op sequence itself is
/// walked iteratively, so program length cannot overflow the stack.
fn parse_prog_str(
    chars: &[char],
    pos: &mut usize,
    in_loop: bool,
    depth: usize,
) -> Result<String, String> {
    if depth > MAX_LOOP_DEPTH {
        return Err(format!("loop nesting deeper than MAX_LOOP_DEPTH ({MAX_LOOP_DEPTH})"));
    }
    // Collect ops in order as string tags. Each tag is either a simple op name
    // or a "Loop:<body>" sentinel.
    let mut ops: Vec<String> = Vec::new();

    while *pos < chars.len() {
        match chars[*pos] {
            ']' => {
                if !in_loop {
                    return Err("unmatched ']'".to_string());
                }
                break; // caller increments pos past ']'
            }
            '[' => {
                *pos += 1; // consume '['
                let body = parse_prog_str(chars, pos, true, depth + 1)?;
                if *pos >= chars.len() || chars[*pos] != ']' {
                    return Err("unmatched '['".to_string());
                }
                *pos += 1; // consume ']'
                ops.push(format!("Loop:{body}"));
            }
            '+' => { ops.push("Inc".to_string());   *pos += 1; }
            '-' => { ops.push("Dec".to_string());   *pos += 1; }
            '<' => { ops.push("Left".to_string());  *pos += 1; }
            '>' => { ops.push("Right".to_string()); *pos += 1; }
            '.' => { ops.push("Out".to_string());   *pos += 1; }
            ',' => { ops.push("In".to_string());    *pos += 1; }
            _   => { *pos += 1; } // comment char
        }
    }

    // Build the right-leaning s-expression `(Inc (Dec (Nil)))` in ONE forward
    // pass: emit each constructor's opening, then `(Nil)`, then all the
    // closers. (The previous reverse fold re-copied the accumulated string
    // once per op — quadratic on long straight-line programs.)
    let mut result = String::new();
    for tag in &ops {
        if let Some(body) = tag.strip_prefix("Loop:") {
            result.push_str("(Loop ");
            result.push_str(body);
            result.push(' ');
        } else {
            result.push('(');
            result.push_str(tag);
            result.push(' ');
        }
    }
    result.push_str("(Nil)");
    result.push_str(&")".repeat(ops.len()));
    Ok(result)
}

// ---------------------------------------------------------------------------
// Prog s-expression -> BF source
// ---------------------------------------------------------------------------

/// Convert a Prog s-expression string back to BF source text.
///
/// FULLY ITERATIVE by design. A Prog is a cons-list, so its s-expression
/// nesting depth equals its op count: a recursive walk — or even building an
/// intermediate tree, whose recursive `Drop` walks the same spine — overflows
/// the native stack at a few tens of thousands of ops, an uncatchable abort
/// that breaks `bf_simplify`'s never-raises contract. This single pass over
/// the token stream drives an explicit heap work-stack instead: memory is
/// O(ops), the native stack stays flat, and there is no intermediate tree at
/// all.
pub fn unparse_bf(sexpr: &str) -> Result<String, String> {
    /// Work items, pushed in reverse execution order (LIFO).
    enum Task {
        /// Parse one Prog node at the current position.
        Node,
        /// Consume a closing `)`.
        RParen,
        /// Emit the `]` that closes a Loop body.
        CloseLoop,
    }

    let toks = tokenize(sexpr);
    let mut out = String::new();
    let mut pos = 0usize;
    let mut stack = vec![Task::Node];

    while let Some(task) = stack.pop() {
        match task {
            Task::CloseLoop => out.push(']'),
            Task::RParen => {
                if toks.get(pos).map(String::as_str) != Some(")") {
                    return Err(format!(
                        "expected ')' at token {pos}, got {:?}",
                        toks.get(pos)
                    ));
                }
                pos += 1;
            }
            Task::Node => {
                match toks.get(pos).map(String::as_str) {
                    Some("(") => {}
                    // A bare `Nil` atom is a valid (empty) Prog.
                    Some("Nil") => {
                        pos += 1;
                        continue;
                    }
                    other => {
                        return Err(format!("expected '(' at token {pos}, got {other:?}"))
                    }
                }
                pos += 1; // consume '('
                let head = toks
                    .get(pos)
                    .cloned()
                    .ok_or_else(|| "missing constructor after '('".to_string())?;
                pos += 1;
                match head.as_str() {
                    "Nil" => stack.push(Task::RParen),
                    "Inc" | "Dec" | "Left" | "Right" | "Out" | "In" | "Clear" => {
                        out.push_str(match head.as_str() {
                            "Inc" => "+",
                            "Dec" => "-",
                            "Left" => "<",
                            "Right" => ">",
                            "Out" => ".",
                            "In" => ",",
                            _ => "[-]", // Clear
                        });
                        // Execution order: rest, then this constructor's ')'.
                        stack.push(Task::RParen);
                        stack.push(Task::Node);
                    }
                    "AddN" | "MoveN" => {
                        let n: i64 = toks
                            .get(pos)
                            .ok_or_else(|| format!("{head} missing count arg"))?
                            .parse()
                            .map_err(|e| format!("{head} arg: {e}"))?;
                        pos += 1;
                        let (pos_c, neg_c) = if head == "AddN" { ('+', '-') } else { ('>', '<') };
                        let c = if n >= 0 { pos_c } else { neg_c };
                        for _ in 0..n.unsigned_abs() {
                            out.push(c);
                        }
                        stack.push(Task::RParen);
                        stack.push(Task::Node);
                    }
                    "Loop" => {
                        out.push('[');
                        // Execution order: body, ']', rest, this ')'.
                        stack.push(Task::RParen);
                        stack.push(Task::Node); // rest
                        stack.push(Task::CloseLoop);
                        stack.push(Task::Node); // body
                    }
                    other => return Err(format!("unknown Prog constructor: {other:?}")),
                }
            }
        }
    }
    if pos != toks.len() {
        return Err(format!("extra tokens after s-expression: {:?}", &toks[pos..]));
    }
    Ok(out)
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
            c if c.is_whitespace() => { chars.next(); }
            _ => {
                let mut tok = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '(' || c2 == ')' || c2.is_whitespace() { break; }
                    tok.push(c2);
                    chars.next();
                }
                out.push(tok);
            }
        }
    }
    out
}

/// Convert a Prog s-expression to BF source (public entry point).
pub fn sexpr_to_source(sexpr: &str) -> Result<String, String> {
    unparse_bf(sexpr)
}

/// Count BF ops in an s-expression by converting to source.
pub fn op_count(sexpr: &str) -> usize {
    unparse_bf(sexpr).map(|s| s.chars().count()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf_parse_single_ops() {
        assert_eq!(parse_bf("+").unwrap(),  "(Inc (Nil))");
        assert_eq!(parse_bf("-").unwrap(),  "(Dec (Nil))");
        assert_eq!(parse_bf("<").unwrap(),  "(Left (Nil))");
        assert_eq!(parse_bf(">").unwrap(),  "(Right (Nil))");
        assert_eq!(parse_bf(".").unwrap(),  "(Out (Nil))");
        assert_eq!(parse_bf(",").unwrap(),  "(In (Nil))");
    }

    #[test]
    fn bf_parse_sequence() {
        assert_eq!(parse_bf("+-").unwrap(), "(Inc (Dec (Nil)))");
    }

    #[test]
    fn bf_parse_loop() {
        let s = parse_bf("[+]").expect("parse");
        assert_eq!(s, "(Loop (Inc (Nil)) (Nil))");
    }

    #[test]
    fn bf_roundtrip() {
        let source = "+[->+<]>.";
        let sexpr = parse_bf(source).expect("parse");
        let back = unparse_bf(&sexpr).expect("unparse");
        assert_eq!(back, source);
    }

    #[test]
    fn bf_roundtrip_empty() {
        assert_eq!(parse_bf("").unwrap(), "(Nil)");
        assert_eq!(unparse_bf("(Nil)").unwrap(), "");
    }

    #[test]
    fn bf_unmatched_open_bracket_error() {
        assert!(parse_bf("[+").is_err());
    }

    #[test]
    fn bf_unmatched_close_bracket_error() {
        assert!(parse_bf("+]").is_err());
    }

    #[test]
    fn bf_addn_sexpr_roundtrips() {
        let src = sexpr_to_source("(AddN 3 (Nil))").expect("addn to src");
        assert_eq!(src, "+++");
    }

    #[test]
    fn bf_moven_neg_sexpr_roundtrips() {
        let src = sexpr_to_source("(MoveN -2 (Nil))").expect("moven to src");
        assert_eq!(src, "<<");
    }

    #[test]
    fn bf_clear_sexpr_roundtrips() {
        let src = sexpr_to_source("(Clear (Nil))").expect("clear to src");
        assert_eq!(src, "[-]");
    }

    #[test]
    fn bf_nested_loops_roundtrip() {
        let source = "[+[->]<]";
        let sexpr = parse_bf(source).expect("parse");
        let back = unparse_bf(&sexpr).expect("unparse");
        assert_eq!(back, source);
    }

    /// A cons-list's depth is its LENGTH: a 50k-op straight-line program must
    /// round-trip without touching the native stack (this test is the
    /// regression guard for the recursive walkers this module used to have —
    /// they aborted with a stack overflow here, which no never-raises
    /// contract can catch).
    #[test]
    fn bf_long_straight_line_roundtrips() {
        let source: String = "+".repeat(50_000);
        let sexpr = parse_bf(&source).expect("parse 50k ops");
        let back = unparse_bf(&sexpr).expect("unparse 50k ops");
        assert_eq!(back, source);
    }

    /// Loop NESTING is the only recursion left, and it is depth-capped: absurd
    /// nesting must surface as Err, never a stack overflow.
    #[test]
    fn bf_deep_nesting_errs_instead_of_overflowing() {
        let n = MAX_LOOP_DEPTH + 10;
        let source = format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(parse_bf(&source).is_err(), "deep nesting must Err");
        // Realistic nesting well under the cap still works.
        let ok = format!("{}+{}", "[".repeat(64), "]".repeat(64));
        let sexpr = parse_bf(&ok).expect("parse 64-deep");
        assert_eq!(unparse_bf(&sexpr).expect("unparse 64-deep"), ok);
    }

    /// Long body INSIDE a loop (cons-depth again, one bracket level).
    #[test]
    fn bf_long_loop_body_roundtrips() {
        let source = format!("[{}]", "+".repeat(20_000));
        let sexpr = parse_bf(&source).expect("parse");
        assert_eq!(unparse_bf(&sexpr).expect("unparse"), source);
    }

    #[test]
    fn bf_unparse_malformed_errs() {
        assert!(unparse_bf("(Bogus (Nil))").is_err());
        assert!(unparse_bf("(Inc (Nil)").is_err(), "missing ')'");
        assert!(unparse_bf("(Inc (Nil)) extra").is_err(), "trailing tokens");
        assert!(unparse_bf("(AddN x (Nil))").is_err(), "non-integer count");
    }
}
