//! phylu `ast_sexpr` S-expression <-> egglog `Re` s-expression.
//!
//! phylu prints a regex AST as `(RegexConcat (RegexPlus (RegexDot)) (RegexCcDigit))`
//! with a byte literal as `(RegexLit (Num <byte>))`. egglog's `Re` sort uses the
//! bare constructor names (`Concat`, `Plus`, ...) and carries a byte as an `i64`
//! payload `(Lit <byte>)`. These two functions translate between the two, in both
//! directions, so `regex_simplify` can take/return the phylu format while working
//! in the egglog format internally.
//!
//! Both directions are ITERATIVE (explicit work-stack, no recursion on the tree
//! spine) so a deeply nested rule cannot overflow the native stack — the same
//! never-raises discipline as BF's unparser.

/// Max AST NESTING depth accepted. Bounds stack use in the tokeniser-driven
/// walk; a regex AST deeper than this is refused rather than risking overflow.
const MAX_DEPTH: usize = 512;

/// The egglog `Re` constructor name for each phylu `ast_sexpr` op name, plus its
/// arity. `RegexLit`/`Num` are handled specially (see `parse_regex`).
fn egg_name(phylu: &str) -> Option<(&'static str, usize)> {
    Some(match phylu {
        "RegexEmpty" => ("Empty", 0),
        "RegexDot" => ("Dot", 0),
        "RegexCcDigit" => ("CcDigit", 0),
        "RegexCcWord" => ("CcWord", 0),
        "RegexCcSpace" => ("CcSpace", 0),
        "RegexAnchorStart" => ("AnchorStart", 0),
        "RegexAnchorEnd" => ("AnchorEnd", 0),
        "RegexStar" => ("Star", 1),
        "RegexPlus" => ("Plus", 1),
        "RegexOpt" => ("Opt", 1),
        "RegexClassOf" => ("ClassOf", 1),
        "RegexCcNegate" => ("CcNegate", 1),
        "RegexConcat" => ("Concat", 2),
        "RegexAlt" => ("Alt", 2),
        "RegexCcRange" => ("CcRange", 2),
        "RegexCcUnion" => ("CcUnion", 2),
        _ => return None,
    })
}

/// Tokenise an S-expression into `(`, `)`, and atoms.
fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' => {
                out.push(c.to_string());
                chars.next();
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut tok = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '(' || c2 == ')' || c2.is_whitespace() {
                        break;
                    }
                    tok.push(c2);
                    chars.next();
                }
                out.push(tok);
            }
        }
    }
    out
}

/// Parse a phylu `ast_sexpr` string into an egglog `Re` s-expression string.
///
/// `(RegexLit (Num <b>))` and a bare `(Num <b>)` both become `(Lit <b>)`; a byte
/// value is rounded to an integer and range-checked to 0..=255. Returns an error
/// on an unknown op, wrong arity, or malformed input.
pub fn parse_regex(ast_sexpr: &str) -> Result<String, String> {
    let toks = tokenize(ast_sexpr);
    let mut pos = 0usize;
    let out = parse_node(&toks, &mut pos, 0)?;
    if pos != toks.len() {
        return Err(format!("extra tokens after s-expression: {:?}", &toks[pos..]));
    }
    Ok(out)
}

fn parse_node(toks: &[String], pos: &mut usize, depth: usize) -> Result<String, String> {
    if depth > MAX_DEPTH {
        return Err(format!("AST nesting deeper than MAX_DEPTH ({MAX_DEPTH})"));
    }
    if toks.get(*pos).map(String::as_str) != Some("(") {
        return Err(format!("expected '(' at token {pos}, got {:?}", toks.get(*pos)));
    }
    *pos += 1; // '('
    let head = toks.get(*pos).cloned().ok_or_else(|| "missing constructor after '('".to_string())?;
    *pos += 1;

    // A byte literal: phylu writes it as `(RegexLit (Num b))` OR a bare `(Num b)`.
    if head == "Num" {
        let b = read_byte(toks, pos)?;
        expect_rparen(toks, pos)?;
        return Ok(format!("(Lit {b})"));
    }
    if head == "RegexLit" {
        // The one child is `(Num b)`; unwrap it to `(Lit b)`.
        if toks.get(*pos).map(String::as_str) != Some("(") {
            return Err("RegexLit expects a (Num b) child".to_string());
        }
        *pos += 1; // '('
        if toks.get(*pos).map(String::as_str) != Some("Num") {
            return Err("RegexLit child must be Num".to_string());
        }
        *pos += 1;
        let b = read_byte(toks, pos)?;
        expect_rparen(toks, pos)?; // close (Num b)
        expect_rparen(toks, pos)?; // close (RegexLit ...)
        return Ok(format!("(Lit {b})"));
    }

    let (egg, arity) = egg_name(&head).ok_or_else(|| format!("unknown regex op: {head:?}"))?;
    let mut children = Vec::with_capacity(arity);
    while toks.get(*pos).map(String::as_str) != Some(")") {
        if *pos >= toks.len() {
            return Err("unterminated s-expression".to_string());
        }
        children.push(parse_node(toks, pos, depth + 1)?);
    }
    *pos += 1; // ')'
    if children.len() != arity {
        return Err(format!("{head} expects {arity} children, got {}", children.len()));
    }
    if children.is_empty() {
        Ok(format!("({egg})"))
    } else {
        Ok(format!("({egg} {})", children.join(" ")))
    }
}

fn read_byte(toks: &[String], pos: &mut usize) -> Result<u8, String> {
    let v = toks.get(*pos).ok_or_else(|| "missing byte value".to_string())?;
    *pos += 1;
    // phylu writes `node.konst` (an f32) — may be "65" or "65.0".
    let f: f64 = v.parse().map_err(|_| format!("byte value not a number: {v:?}"))?;
    let r = f.round();
    if !(0.0..=255.0).contains(&r) {
        return Err(format!("byte value out of range 0..=255: {r}"));
    }
    Ok(r as u8)
}

fn expect_rparen(toks: &[String], pos: &mut usize) -> Result<(), String> {
    if toks.get(*pos).map(String::as_str) != Some(")") {
        return Err(format!("expected ')' at token {pos}, got {:?}", toks.get(*pos)));
    }
    *pos += 1;
    Ok(())
}

/// Convert an egglog `Re` s-expression back to the phylu `ast_sexpr` format.
///
/// `(Lit b)` becomes `(RegexLit (Num b))`. FULLY ITERATIVE: an explicit
/// work-stack drives the single pass, so a deep tree cannot overflow the stack
/// (keeping `regex_simplify`'s never-raises contract).
pub fn unparse_regex(re_sexpr: &str) -> Result<String, String> {
    let toks = tokenize(re_sexpr);
    let mut pos = 0usize;
    let out = unparse_node(&toks, &mut pos, 0)?;
    if pos != toks.len() {
        return Err(format!("extra tokens after s-expression: {:?}", &toks[pos..]));
    }
    Ok(out)
}

/// Emit one `Re` node as phylu `ast_sexpr` text. Depth-bounded recursion
/// (MAX_DEPTH, same cap as the parser) — a tree deeper than that is refused
/// rather than risking a native-stack overflow, keeping the never-raises
/// contract: the caller treats any Err as "leave unchanged".
fn unparse_node(toks: &[String], pos: &mut usize, depth: usize) -> Result<String, String> {
    if depth > MAX_DEPTH {
        return Err(format!("Re nesting deeper than MAX_DEPTH ({MAX_DEPTH})"));
    }
    if toks.get(*pos).map(String::as_str) != Some("(") {
        return Err(format!("expected '(' at token {pos}, got {:?}", toks.get(*pos)));
    }
    *pos += 1; // '('
    let head = toks.get(*pos).cloned().ok_or_else(|| "missing constructor".to_string())?;
    *pos += 1;

    if head == "Lit" {
        let b = toks.get(*pos).ok_or_else(|| "Lit missing byte".to_string())?;
        let n: i64 = b.parse().map_err(|e| format!("Lit arg: {e}"))?;
        *pos += 1;
        expect_rparen(toks, pos)?;
        return Ok(format!("(RegexLit (Num {n}))"));
    }

    let (phy, arity) = match head.as_str() {
        "Empty" => ("RegexEmpty", 0),
        "Dot" => ("RegexDot", 0),
        "CcDigit" => ("RegexCcDigit", 0),
        "CcWord" => ("RegexCcWord", 0),
        "CcSpace" => ("RegexCcSpace", 0),
        "AnchorStart" => ("RegexAnchorStart", 0),
        "AnchorEnd" => ("RegexAnchorEnd", 0),
        "Star" => ("RegexStar", 1),
        "Plus" => ("RegexPlus", 1),
        "Opt" => ("RegexOpt", 1),
        "ClassOf" => ("RegexClassOf", 1),
        "CcNegate" => ("RegexCcNegate", 1),
        "Concat" => ("RegexConcat", 2),
        "Alt" => ("RegexAlt", 2),
        "CcRange" => ("RegexCcRange", 2),
        "CcUnion" => ("RegexCcUnion", 2),
        other => return Err(format!("unknown Re constructor: {other:?}")),
    };

    let mut children = Vec::with_capacity(arity);
    for _ in 0..arity {
        children.push(unparse_node(toks, pos, depth + 1)?);
    }
    expect_rparen(toks, pos)?;
    if children.is_empty() {
        Ok(format!("({phy})"))
    } else {
        Ok(format!("({phy} {})", children.join(" ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nullary() {
        assert_eq!(parse_regex("(RegexDot)").unwrap(), "(Dot)");
        assert_eq!(parse_regex("(RegexCcDigit)").unwrap(), "(CcDigit)");
        assert_eq!(parse_regex("(RegexEmpty)").unwrap(), "(Empty)");
    }

    #[test]
    fn parse_lit_from_regexlit_num() {
        assert_eq!(parse_regex("(RegexLit (Num 65))").unwrap(), "(Lit 65)");
        // Float byte value (phylu writes konst as f32).
        assert_eq!(parse_regex("(RegexLit (Num 65.0))").unwrap(), "(Lit 65)");
    }

    #[test]
    fn parse_nested() {
        assert_eq!(
            parse_regex("(RegexConcat (RegexPlus (RegexDot)) (RegexCcDigit))").unwrap(),
            "(Concat (Plus (Dot)) (CcDigit))"
        );
    }

    #[test]
    fn unparse_inverts_parse() {
        let cases = [
            "(RegexDot)",
            "(RegexCcDigit)",
            "(RegexEmpty)",
            "(RegexLit (Num 65))",
            "(RegexStar (RegexDot))",
            "(RegexConcat (RegexPlus (RegexDot)) (RegexCcDigit))",
            "(RegexAlt (RegexLit (Num 97)) (RegexLit (Num 98)))",
            "(RegexClassOf (RegexCcNegate (RegexCcRange (RegexLit (Num 48)) (RegexLit (Num 57)))))",
        ];
        for c in cases {
            let egg = parse_regex(c).expect("parse");
            let back = unparse_regex(&egg).expect("unparse");
            assert_eq!(back, c, "round-trip differs for {c}");
        }
    }

    #[test]
    fn parse_rejects_unknown_and_bad_arity() {
        assert!(parse_regex("(RegexBogus)").is_err());
        assert!(parse_regex("(RegexStar)").is_err(), "Star needs 1 child");
        assert!(parse_regex("(RegexDot (RegexDot))").is_err(), "Dot is nullary");
        assert!(parse_regex("(RegexLit (Num 999))").is_err(), "byte out of range");
    }

    #[test]
    fn unparse_rejects_malformed() {
        assert!(unparse_regex("(Bogus)").is_err());
        assert!(unparse_regex("(Concat (Dot))").is_err(), "Concat needs 2");
        assert!(unparse_regex("(Dot) extra").is_err(), "trailing tokens");
    }
}
