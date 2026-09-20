//! A minimal s-expression reader for the egglog rule text.
//!
//! The linter's rule table is DERIVED from the same text egglog runs
//! (`ALGEBRA_RULESET`, `SIGN_RULESET`, ...), so there is exactly one written
//! copy of every rule. This reads that text; `reader` gives it meaning.

/// One s-expression. Strings keep their quotes stripped and are tagged so a
/// `"x"` is never confused with the atom `x`.
#[derive(Debug, Clone, PartialEq)]
pub enum Sexp {
    Atom(String),
    Str(String),
    List(Vec<Sexp>),
}

impl Sexp {
    pub fn atom(&self) -> Option<&str> {
        match self {
            Sexp::Atom(a) => Some(a),
            _ => None,
        }
    }

    pub fn list(&self) -> Option<&[Sexp]> {
        match self {
            Sexp::List(l) => Some(l),
            _ => None,
        }
    }

    /// The head atom of a list: `rewrite` for `(rewrite ...)`.
    pub fn head(&self) -> Option<&str> {
        self.list().and_then(|l| l.first()).and_then(Sexp::atom)
    }
}

/// Parse every top-level form in `text`. `;` starts a comment to end of line.
pub fn parse_all(text: &str) -> Result<Vec<Sexp>, String> {
    let toks = tokenize(text)?;
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos < toks.len() {
        out.push(parse(&toks, &mut pos, 0)?);
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Open,
    Close,
    Atom(String),
    Str(String),
}

fn tokenize(text: &str) -> Result<Vec<Tok>, String> {
    let mut toks = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ';' => {
                for d in chars.by_ref() {
                    if d == '\n' {
                        break;
                    }
                }
            }
            '(' => {
                toks.push(Tok::Open);
                chars.next();
            }
            ')' => {
                toks.push(Tok::Close);
                chars.next();
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                let mut closed = false;
                for d in chars.by_ref() {
                    if d == '"' {
                        closed = true;
                        break;
                    }
                    s.push(d);
                }
                if !closed {
                    return Err(format!("unterminated string starting \"{s}"));
                }
                toks.push(Tok::Str(s));
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut s = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_whitespace() || d == '(' || d == ')' || d == ';' || d == '"' {
                        break;
                    }
                    s.push(d);
                    chars.next();
                }
                toks.push(Tok::Atom(s));
            }
        }
    }
    Ok(toks)
}

fn parse(toks: &[Tok], pos: &mut usize, depth: usize) -> Result<Sexp, String> {
    if depth > crate::MAX_EXPR_DEPTH {
        return Err("s-expression nested too deeply".to_string());
    }
    match toks.get(*pos) {
        None => Err("unexpected end of input".to_string()),
        Some(Tok::Close) => Err(format!("unexpected ')' at token {pos}")),
        Some(Tok::Atom(a)) => {
            *pos += 1;
            Ok(Sexp::Atom(a.clone()))
        }
        Some(Tok::Str(s)) => {
            *pos += 1;
            Ok(Sexp::Str(s.clone()))
        }
        Some(Tok::Open) => {
            *pos += 1;
            let mut items = Vec::new();
            loop {
                match toks.get(*pos) {
                    None => return Err("missing ')'".to_string()),
                    Some(Tok::Close) => {
                        *pos += 1;
                        return Ok(Sexp::List(items));
                    }
                    Some(_) => items.push(parse(toks, pos, depth + 1)?),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_all, Sexp};

    #[test]
    fn reads_forms_comments_and_strings() {
        let forms = parse_all(
            "; a comment\n(ruleset algebra)\n(rewrite (Mul x (Num 1.0)) x :ruleset algebra) ; tail\n(let t (Var \"x y\"))",
        )
        .unwrap();
        assert_eq!(forms.len(), 3);
        assert_eq!(forms[0].head(), Some("ruleset"));
        assert_eq!(forms[1].head(), Some("rewrite"));
        let var = &forms[2].list().unwrap()[2];
        assert_eq!(var.list().unwrap()[1], Sexp::Str("x y".to_string()));
    }

    #[test]
    fn unbalanced_input_is_an_error() {
        assert!(parse_all("(rewrite (Mul x").is_err());
        assert!(parse_all("x)").is_err());
        assert!(parse_all("(Var \"x)").is_err());
    }

    #[test]
    fn every_shipped_ruleset_parses() {
        for text in [
            crate::ruleset::identities::ALGEBRA_RULESET,
            crate::ruleset::powers::POWERS_RULESET,
            crate::ruleset::sign::SIGN_RULESET,
            crate::ruleset::rational::RATIONAL_RULESET,
            crate::expr::GUARD_RELATIONS,
        ] {
            assert!(!parse_all(text).unwrap().is_empty());
        }
    }
}
