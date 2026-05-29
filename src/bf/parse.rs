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

/// A minimal s-expression tree, used for unparsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

// ---------------------------------------------------------------------------
// BF source -> Prog s-expression string (direct string building)
// ---------------------------------------------------------------------------

/// Convert a BF source string to a Prog s-expression string.
///
/// Non-BF characters are silently ignored (they're comments in standard BF).
/// Returns an error on unmatched brackets.
pub fn parse_bf(source: &str) -> Result<String, String> {
    let chars: Vec<char> = source.chars().filter(|c| "+-<>.,[]".contains(*c)).collect();
    let mut pos = 0;
    parse_prog_str(&chars, &mut pos, false)
}

/// Inner recursive parser: returns the s-expression string for the ops from
/// `pos` onward. Stops at `]` if `in_loop` is true (caller consumes `]`).
fn parse_prog_str(chars: &[char], pos: &mut usize, in_loop: bool) -> Result<String, String> {
    // Collect ops in order as string tags, then fold right.
    // Each tag is either a simple op name or a "Loop:<body>" sentinel.
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
                let body = parse_prog_str(chars, pos, true)?;
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

    // Build right-leaning s-expression: (Inc (Dec (Nil))) etc.
    let mut result = "(Nil)".to_string();
    for tag in ops.into_iter().rev() {
        if let Some(body) = tag.strip_prefix("Loop:") {
            result = format!("(Loop {body} {result})");
        } else {
            result = format!("({tag} {result})");
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Prog s-expression -> BF source
// ---------------------------------------------------------------------------

/// Convert a Prog s-expression string back to BF source text.
pub fn unparse_bf(sexpr: &str) -> Result<String, String> {
    let expr = parse_sexpr(sexpr)?;
    let mut out = String::new();
    node_to_source(&expr, &mut out)?;
    Ok(out)
}

/// Parse an egglog s-expression string into a `SExpr` tree.
pub fn parse_sexpr(s: &str) -> Result<SExpr, String> {
    let toks = tokenize(s);
    let mut pos = 0;
    let expr = parse_sexpr_toks(&toks, &mut pos)?;
    if pos != toks.len() {
        return Err(format!("extra tokens after s-expression: {:?}", &toks[pos..]));
    }
    Ok(expr)
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

fn parse_sexpr_toks(toks: &[String], pos: &mut usize) -> Result<SExpr, String> {
    match toks.get(*pos).map(String::as_str) {
        None => Err("unexpected end of s-expression".to_string()),
        Some("(") => {
            *pos += 1;
            let mut parts = Vec::new();
            while toks.get(*pos).map(String::as_str) != Some(")") {
                if *pos >= toks.len() {
                    return Err("unclosed '('".to_string());
                }
                parts.push(parse_sexpr_toks(toks, pos)?);
            }
            *pos += 1; // consume ')'
            Ok(SExpr::List(parts))
        }
        Some(")") => Err("unexpected ')'".to_string()),
        Some(tok) => {
            let t = tok.to_string();
            *pos += 1;
            Ok(SExpr::Atom(t))
        }
    }
}

/// Convert a parsed Prog `SExpr` node to BF source text.
fn node_to_source(expr: &SExpr, out: &mut String) -> Result<(), String> {
    match expr {
        SExpr::Atom(a) => match a.as_str() {
            "Nil" => Ok(()),
            other => Err(format!("unexpected atom in Prog: {other:?}")),
        },
        SExpr::List(parts) => {
            if parts.is_empty() {
                return Err("empty list in Prog s-expression".to_string());
            }
            let head = match &parts[0] {
                SExpr::Atom(a) => a.as_str(),
                _ => return Err("list head must be an atom".to_string()),
            };
            match head {
                "Nil" => Ok(()),
                // Single-op constructors: (Inc rest)
                "Inc"   => { expect_args("Inc",   parts, 1)?; out.push('+'); node_to_source(&parts[1], out) }
                "Dec"   => { expect_args("Dec",   parts, 1)?; out.push('-'); node_to_source(&parts[1], out) }
                "Left"  => { expect_args("Left",  parts, 1)?; out.push('<'); node_to_source(&parts[1], out) }
                "Right" => { expect_args("Right", parts, 1)?; out.push('>'); node_to_source(&parts[1], out) }
                "Out"   => { expect_args("Out",   parts, 1)?; out.push('.'); node_to_source(&parts[1], out) }
                "In"    => { expect_args("In",    parts, 1)?; out.push(','); node_to_source(&parts[1], out) }
                "Clear" => {
                    expect_args("Clear", parts, 1)?;
                    out.push_str("[-]");
                    node_to_source(&parts[1], out)
                }
                "Loop" => {
                    // (Loop body rest)
                    expect_args("Loop", parts, 2)?;
                    out.push('[');
                    node_to_source(&parts[1], out)?;
                    out.push(']');
                    node_to_source(&parts[2], out)
                }
                "AddN" => {
                    // (AddN n rest)
                    expect_args("AddN", parts, 2)?;
                    let n = atom_i64(&parts[1], "AddN")?;
                    if n > 0       { for _ in 0..n    { out.push('+'); } }
                    else if n < 0  { for _ in 0..(-n) { out.push('-'); } }
                    node_to_source(&parts[2], out)
                }
                "MoveN" => {
                    // (MoveN n rest)
                    expect_args("MoveN", parts, 2)?;
                    let n = atom_i64(&parts[1], "MoveN")?;
                    if n > 0       { for _ in 0..n    { out.push('>'); } }
                    else if n < 0  { for _ in 0..(-n) { out.push('<'); } }
                    node_to_source(&parts[2], out)
                }
                other => Err(format!("unknown Prog constructor: {other:?}")),
            }
        }
    }
}

fn expect_args(name: &str, parts: &[SExpr], n: usize) -> Result<(), String> {
    if parts.len() != n + 1 {
        Err(format!("{name} expects {n} arg(s), got {}", parts.len() - 1))
    } else {
        Ok(())
    }
}

fn atom_i64(expr: &SExpr, ctx: &str) -> Result<i64, String> {
    match expr {
        SExpr::Atom(a) => a.parse::<i64>().map_err(|e| format!("{ctx} arg: {e}")),
        _ => Err(format!("{ctx}: first arg must be an integer atom")),
    }
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
}
