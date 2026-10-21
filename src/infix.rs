//! Infix reader: SRBench / Feynman infix text -> fuller `Math` s-expression.
//!
//! A hand-written recursive-descent parser (no sympy). Grammar, loosest to
//! tightest binding, following Python:
//!
//! ```text
//! expr   = term (('+' | '-') term)*
//! term   = unary (('*' | '/') unary)*
//! unary  = ('-' | '+') unary | power
//! power  = atom ('**' unary)?          right-associative; -x**2 = -(x**2)
//! atom   = number | name | 'pi' | func '(' expr ')' | '(' expr ')'
//! ```
//!
//! Constructors: `Add Sub Mul ProtectedDiv Neg`, `Pow2`/`Pow3` for the
//! literal exponents 2 and 3, the general `Pow` for any other exponent,
//! `ProtectedSqrt ProtectedExp ProtectedLog ProtectedAsin ProtectedAcos Sin
//! Cos Tan Tanh`. `pi` becomes `(Num 3.14159...)`. A name must be in `vars`
//! and becomes `(Var "<name>")`. Anything the `Math` datatype cannot say
//! (`arctan`, an unknown function, an unknown name) is an `Err` naming it;
//! nothing is silently rewritten.

use crate::MAX_EXPR_DEPTH;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Plus,
    Minus,
    Star,
    Slash,
    Pow,
    LParen,
    RParen,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let cs: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        match c {
            c if c.is_whitespace() => i += 1,
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '*' => {
                if cs.get(i + 1) == Some(&'*') {
                    out.push(Tok::Pow);
                    i += 2;
                } else {
                    out.push(Tok::Star);
                    i += 1;
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < cs.len() && cs[i].is_ascii_digit() {
                    i += 1;
                }
                if i < cs.len() && cs[i] == '.' {
                    i += 1;
                    while i < cs.len() && cs[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < cs.len() && (cs[i] == 'e' || cs[i] == 'E') {
                    let mut j = i + 1;
                    if j < cs.len() && (cs[j] == '+' || cs[j] == '-') {
                        j += 1;
                    }
                    if j < cs.len() && cs[j].is_ascii_digit() {
                        while j < cs.len() && cs[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                }
                let text: String = cs[start..i].iter().collect();
                let v: f64 = text.parse().map_err(|_| format!("bad number {text:?}"))?;
                out.push(Tok::Num(v));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                    i += 1;
                }
                out.push(Tok::Name(cs[start..i].iter().collect()));
            }
            other => return Err(format!("unexpected character {other:?} at {i}")),
        }
    }
    Ok(out)
}

fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn num(v: f64) -> String {
    format!("(Num {})", fmt_num(v))
}

/// The value of `s` when it is exactly a `(Num v)` atom.
fn as_num(s: &str) -> Option<f64> {
    s.strip_prefix("(Num ")?.strip_suffix(')')?.parse().ok()
}

fn func_ctor(name: &str) -> Option<Result<&'static str, String>> {
    Some(Ok(match name {
        "sqrt" => "ProtectedSqrt",
        "exp" => "ProtectedExp",
        "log" | "ln" => "ProtectedLog",
        "sin" => "Sin",
        "cos" => "Cos",
        "tan" => "Tan",
        "tanh" => "Tanh",
        "arcsin" | "asin" => "ProtectedAsin",
        "arccos" | "acos" => "ProtectedAcos",
        "arctan" | "atan" => {
            return Some(Err(format!(
                "unsupported function {name}: Math has no arctan constructor"
            )))
        }
        _ => return None,
    }))
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    vars: &'a [String],
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expr(&mut self, depth: usize) -> Result<String, String> {
        if depth > MAX_EXPR_DEPTH {
            return Err("expression nested too deeply".into());
        }
        let mut lhs = self.term(depth)?;
        loop {
            let ctor = match self.peek() {
                Some(Tok::Plus) => "Add",
                Some(Tok::Minus) => "Sub",
                _ => return Ok(lhs),
            };
            self.pos += 1;
            let rhs = self.term(depth)?;
            lhs = format!("({ctor} {lhs} {rhs})");
        }
    }

    fn term(&mut self, depth: usize) -> Result<String, String> {
        let mut lhs = self.unary(depth)?;
        loop {
            let ctor = match self.peek() {
                Some(Tok::Star) => "Mul",
                Some(Tok::Slash) => "ProtectedDiv",
                _ => return Ok(lhs),
            };
            self.pos += 1;
            let rhs = self.unary(depth)?;
            lhs = format!("({ctor} {lhs} {rhs})");
        }
    }

    fn unary(&mut self, depth: usize) -> Result<String, String> {
        if depth > MAX_EXPR_DEPTH {
            return Err("expression nested too deeply".into());
        }
        if self.eat(&Tok::Minus) {
            let inner = self.unary(depth + 1)?;
            return Ok(match as_num(&inner) {
                Some(v) => num(-v),
                None => format!("(Neg {inner})"),
            });
        }
        if self.eat(&Tok::Plus) {
            return self.unary(depth + 1);
        }
        self.power(depth)
    }

    fn power(&mut self, depth: usize) -> Result<String, String> {
        let base = self.atom(depth)?;
        if !self.eat(&Tok::Pow) {
            return Ok(base);
        }
        let exp = self.unary(depth + 1)?;
        Ok(match as_num(&exp) {
            Some(2.0) => format!("(Pow2 {base})"),
            Some(3.0) => format!("(Pow3 {base})"),
            _ => format!("(Pow {base} {exp})"),
        })
    }

    fn atom(&mut self, depth: usize) -> Result<String, String> {
        if depth > MAX_EXPR_DEPTH {
            return Err("expression nested too deeply".into());
        }
        let tok = self.peek().cloned().ok_or("unexpected end of input")?;
        self.pos += 1;
        match tok {
            Tok::Num(v) => Ok(num(v)),
            Tok::LParen => {
                let e = self.expr(depth + 1)?;
                if !self.eat(&Tok::RParen) {
                    return Err("missing )".into());
                }
                Ok(e)
            }
            Tok::Name(name) => {
                if self.peek() == Some(&Tok::LParen) {
                    let ctor = func_ctor(&name)
                        .ok_or_else(|| format!("unknown function {name}"))??;
                    self.pos += 1;
                    let arg = self.expr(depth + 1)?;
                    if !self.eat(&Tok::RParen) {
                        return Err(format!("missing ) after {name}(..."));
                    }
                    return Ok(format!("({ctor} {arg})"));
                }
                if self.vars.contains(&name) {
                    Ok(format!("(Var \"{name}\")"))
                } else if name == "pi" {
                    Ok(num(std::f64::consts::PI))
                } else {
                    Err(format!("unknown name {name}"))
                }
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

/// Parse infix `src` into a `Math` s-expression string. Names must be in
/// `vars` (or be `pi`); anything unsupported is an `Err` saying what.
pub fn infix_to_math(src: &str, vars: &[String]) -> Result<String, String> {
    let toks = tokenize(src)?;
    if toks.is_empty() {
        return Err("empty expression".into());
    }
    let mut p = Parser { toks, pos: 0, vars };
    let out = p.expr(0)?;
    if p.pos != p.toks.len() {
        return Err(format!("unexpected trailing token {:?}", p.toks[p.pos]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::eval_expr_rows;

    fn vs(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn p(src: &str) -> String {
        infix_to_math(src, &vs(&["a", "b", "c", "x", "r"])).unwrap()
    }

    #[test]
    fn unary_minus_binds_looser_than_power() {
        assert_eq!(p("-x**2"), r#"(Neg (Pow2 (Var "x")))"#);
        assert_eq!(p("(-x)**2"), r#"(Pow2 (Neg (Var "x")))"#);
        assert_eq!(p("-2"), "(Num -2.0)");
        assert_eq!(p("-2**2"), "(Neg (Pow2 (Num 2.0)))");
        assert_eq!(p("2**-x"), r#"(Pow (Num 2.0) (Neg (Var "x")))"#);
    }

    #[test]
    fn left_associative_division_and_subtraction() {
        assert_eq!(
            p("a/b/c"),
            r#"(ProtectedDiv (ProtectedDiv (Var "a") (Var "b")) (Var "c"))"#
        );
        assert_eq!(p("a-b-c"), r#"(Sub (Sub (Var "a") (Var "b")) (Var "c"))"#);
    }

    #[test]
    fn power_is_right_associative() {
        assert_eq!(
            p("a**b**c"),
            r#"(Pow (Var "a") (Pow (Var "b") (Var "c")))"#
        );
        assert_eq!(p("x**3"), r#"(Pow3 (Var "x"))"#);
        assert_eq!(p("x**0.5"), r#"(Pow (Var "x") (Num 0.5))"#);
    }

    #[test]
    fn pi_numbers_and_precedence() {
        assert_eq!(
            p("2*pi*r"),
            format!(r#"(Mul (Mul (Num 2.0) (Num {})) (Var "r"))"#, std::f64::consts::PI)
        );
        assert_eq!(p("1e-3"), "(Num 0.001)");
        assert_eq!(
            p("a+b*c"),
            r#"(Add (Var "a") (Mul (Var "b") (Var "c")))"#
        );
        assert_eq!(p("sqrt(x)"), r#"(ProtectedSqrt (Var "x"))"#);
        assert_eq!(p("ln(x)"), r#"(ProtectedLog (Var "x"))"#);
        assert_eq!(p("asin(x)"), r#"(ProtectedAsin (Var "x"))"#);
    }

    #[test]
    fn errors_name_the_problem() {
        let v = vs(&["x"]);
        assert!(infix_to_math("y+1", &v).unwrap_err().contains("unknown name y"));
        assert!(infix_to_math("arctan(x)", &v).unwrap_err().contains("arctan"));
        assert!(infix_to_math("foo(x)", &v).unwrap_err().contains("unknown function foo"));
        assert!(infix_to_math("(x", &v).is_err());
        assert!(infix_to_math("x )", &v).is_err());
        assert!(infix_to_math("", &v).is_err());
        assert!(infix_to_math("x $ 1", &v).is_err());
    }

    const CSV: &str = "/Users/andrewmorgan/Dev/gamakon/old_regex_build/hff/notebooks/data/equations/feynman/FeynmanEquations.csv";
    const DATA: &str = "/Users/andrewmorgan/Dev/gamakon/hff/notebooks/_ledgers/pmlb_repo/datasets";

    /// Every Feynman formula parses; where the PMLB dataset exists, the
    /// parsed Math reproduces its `target` column on the first 100 rows.
    #[test]
    fn feynman_formulas_parse_and_match_data() {
        let text = std::fs::read_to_string(CSV).expect("FeynmanEquations.csv");
        let text = text.trim_start_matches('\u{feff}');
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<&str> = lines.next().unwrap().split(',').collect();
        let col = |n: &str| header.iter().position(|h| *h == n).unwrap();
        let (c_file, c_formula) = (col("Filename"), col("Formula"));
        let (mut total, mut parsed, mut checked, mut matched, mut no_data) = (0, 0, 0, 0, 0);
        let mut failures: Vec<String> = Vec::new();
        for line in lines {
            let f: Vec<&str> = line.split(',').collect();
            assert!(!line.contains('"'), "quoted csv field not handled: {line}");
            let (name, formula) = (f[c_file].trim(), f[c_formula].trim());
            if name.is_empty() {
                continue;
            }
            total += 1;
            // The "# variables" column is wrong for six rows (I.18.12,
            // I.18.14, I.38.12, II.37.1, III.10.19, III.19.51), so the
            // variables are every non-empty vN_name cell.
            let vars: Vec<String> = (1..=10)
                .map(|k| f[col(&format!("v{k}_name"))].trim().to_string())
                .filter(|v| !v.is_empty())
                .collect();
            let math = match infix_to_math(formula, &vars) {
                Ok(m) => m,
                Err(e) => {
                    failures.push(format!("{name}: parse: {e}  [{formula}]"));
                    continue;
                }
            };
            parsed += 1;
            let ds = format!("feynman_{}", name.replace('.', "_"));
            let path = format!("{DATA}/{ds}/{ds}.tsv.gz");
            if !std::path::Path::new(&path).exists() {
                no_data += 1;
                println!("no dataset for {name} ({ds})");
                continue;
            }
            checked += 1;
            let out = std::process::Command::new("gzip").args(["-dc", &path]).output().unwrap();
            assert!(out.status.success(), "gzip -dc {path}");
            let body = String::from_utf8(out.stdout).unwrap();
            let mut rows_it = body.lines();
            let cols: Vec<&str> = rows_it.next().unwrap().split('\t').collect();
            let ti = cols.iter().position(|c| *c == "target").expect("target column");
            let mut want: Vec<&str> = cols.iter().copied().filter(|c| *c != "target").collect();
            let mut have: Vec<&str> = vars.iter().map(|s| s.as_str()).collect();
            want.sort_unstable();
            have.sort_unstable();
            if want != have {
                failures.push(format!("{name}: dataset columns {want:?} != csv variables {have:?}"));
                continue;
            }
            let mut rows: Vec<Vec<(String, f64)>> = Vec::new();
            let mut target: Vec<f64> = Vec::new();
            for l in rows_it.take(100) {
                let vals: Vec<f64> = l.split('\t').map(|v| v.parse().unwrap()).collect();
                target.push(vals[ti]);
                rows.push(
                    cols.iter()
                        .zip(&vals)
                        .filter(|(c, _)| **c != "target")
                        .map(|(c, v)| (c.to_string(), *v))
                        .collect(),
                );
            }
            let pred = match eval_expr_rows(&math, &rows) {
                Ok(p) => p,
                Err(e) => {
                    failures.push(format!("{name}: eval: {e}  [{formula}]"));
                    continue;
                }
            };
            let worst = pred
                .iter()
                .zip(&target)
                .map(|(a, b)| if a == b { 0.0 } else { (a - b).abs() / b.abs().max(1e-300) })
                .fold(0.0_f64, |m, e| if e.is_nan() { f64::INFINITY } else { m.max(e) });
            if worst <= 1e-9 {
                matched += 1;
            } else {
                failures.push(format!("{name}: max rel err {worst:e} over {} rows  [{formula}]", rows.len()));
            }
        }
        println!(
            "feynman: parsed {parsed}/{total}, with dataset {checked} (no dataset {no_data}), matched {matched}/{checked}"
        );
        for f in &failures {
            println!("FAIL {f}");
        }
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
        assert_eq!(parsed, 100, "expected 100 formulas parsed, got {parsed}");
        assert_eq!(checked, 98, "expected 98 laws with a dataset, got {checked} (datasets missing?)");
        assert_eq!(matched, 98, "expected 98 laws matching their dataset, got {matched}");
    }
}
