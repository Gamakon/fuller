//! THE READER of the signature kingdom (`crate::futhark_sig`): Futhark source text to an expressed tree of a
//! benchmark's kingdom, the inverse of how the kingdom renders a tree ([`crate::futhark_kingdom::FutharkKingdom::render`]
//! inside [`crate::futhark_sig::program`]). Plan: `docs/PLAN_futhark_fitness_fixes.md` section H.
//!
//! Three steps.
//!
//! 1. **Parse** a subset of Futhark into a small syntax tree ([`E`]): `def`/`entry`, `let`, `if`, `loop ... for`,
//!    lambdas, operator sections, application, indexing, `|>`, `>->`. A construct outside the subset is a [`Gap`]
//!    that names it (a tuple, a float, `with`, `while`, ...), never skipped.
//! 2. **Normalise**: every `let` and every helper `def` is inlined, lambdas applied to arguments are reduced, binders
//!    are renamed apart, the entry's parameters are renamed to the signature's (by position), a size input `k: i32`
//!    is `i64.i32 k` = the size `k` and, as a value, the kingdom's `i32.i64 k`, and a few exact identities are applied
//!    (`i32.i32 x = x`, `copy x = x`, `opaque x = x`, `i32.i64 (i64.i32 x) = x`, `length xs` = its dimension,
//!    `(i64.i32 e) % k = i64.i32 (e % i32.i64 k)`).
//!    `let z = replicate D 0` is the kingdom's `zeros_D`.
//! 3. **Read**, typed: a slot of type `T` is filled by the first row of `T` whose SPELLING, parsed as a pattern with
//!    holes, matches the expression (binders matched up to renaming), each hole read at the row's argument type; a
//!    literal in a constant slot is the pool's entry. When no row matches, a few exact rewrites are tried: `map` and
//!    `map2` as `tabulate`, `reduce (+) 0` as `i32.sum` (and the `min`/`max` pairs), `reduce_by_index` of a
//!    `replicate` of the neutral element as `hist`, the indices of a `hist` brought to `map i64.i32`, `if x >= 0` as
//!    `if x < 0` with the branches swapped, an element of a computed array as the computation at that index
//!    (`(map f xs)[i] = f xs[i]`), and the arguments of `+`, `*`, `min` and `max` swapped. Nothing else: an
//!    expression no row reads is a gap of the kingdom, recorded with the construct it lacks.
//!
//! The round trip, every rendered program reads back to its own tree, is the test (`tests`).

use std::collections::{BTreeMap, HashMap};

use crate::futhark_kingdom::{FutharkKind, FutharkKingdom};
use crate::futhark_sig::Signature;
use crate::geneframe::Ty;
use crate::typed_gep::Tree;

/// A construct the reader cannot express: what it is, the type it was wanted at (empty in parsing), and the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    /// The construct the kingdom lacks, e.g. `operator /`, `name i32.highest`, `tuple`, `float literal`.
    pub construct: String,
    /// The dual it was wanted at (empty for a parse gap).
    pub dual: String,
    pub detail: String,
}

impl Gap {
    fn parse(construct: &str, detail: String) -> Gap {
        Gap { construct: construct.to_string(), dual: String::new(), detail }
    }
}

impl std::fmt::Display for Gap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dual.is_empty() {
            write!(f, "{} ({})", self.construct, self.detail)
        } else {
            write!(f, "{} at {} ({})", self.construct, self.dual, self.detail)
        }
    }
}

// ---- the syntax tree --------------------------------------------------------------------------------------------

/// The parsed and normalised expression.
#[derive(Debug, Clone, PartialEq)]
pub enum E {
    Int(i64),
    Var(String),
    /// A pattern's hole `{i}` (row spellings only).
    Hole(usize),
    App(Box<E>, Vec<E>),
    Bin(String, Box<E>, Box<E>),
    Neg(Box<E>),
    Index(Box<E>, Box<E>),
    If(Box<E>, Box<E>, Box<E>),
    Lam(Vec<String>, Box<E>),
    Let(String, Box<E>, Box<E>),
    /// `loop state = init for var in lo ..< hi do body` (`for var < hi` has `lo` 0).
    Loop { state: String, init: Box<E>, var: String, lo: Box<E>, hi: Box<E>, body: Box<E> },
}

fn b(e: E) -> Box<E> {
    Box::new(e)
}

fn var(s: &str) -> E {
    E::Var(s.to_string())
}

fn app(f: &str, args: Vec<E>) -> E {
    E::App(b(var(f)), args)
}

impl E {
    /// The construct at the head of an expression, for a gap.
    fn head(&self) -> String {
        match self {
            E::Int(v) => format!("literal {v}"),
            E::Var(x) => format!("name {x}"),
            E::Hole(i) => format!("hole {i}"),
            E::App(f, _) => match f.as_ref() {
                E::Var(x) => x.clone(),
                _ => "application of a non-name".into(),
            },
            E::Bin(op, _, _) => format!("operator {op}"),
            E::Neg(_) => "negation".into(),
            E::Index(..) => "indexing".into(),
            E::If(c, _, _) => match c.as_ref() {
                E::Bin(op, _, r) => format!("if with condition {op} {}", if matches!(r.as_ref(), E::Int(0)) { "0" } else { "an expression" }),
                _ => "if with a condition that is not a comparison".into(),
            },
            E::Lam(..) => "lambda".into(),
            E::Let(..) => "let".into(),
            E::Loop { .. } => "loop".into(),
        }
    }

    fn mentions(&self, x: &str) -> bool {
        match self {
            E::Var(y) => y == x,
            E::Int(_) | E::Hole(_) => false,
            E::App(f, a) => f.mentions(x) || a.iter().any(|e| e.mentions(x)),
            E::Bin(_, l, r) | E::Index(l, r) => l.mentions(x) || r.mentions(x),
            E::Neg(e) => e.mentions(x),
            E::If(c, t, e) => c.mentions(x) || t.mentions(x) || e.mentions(x),
            E::Lam(ps, body) => !ps.iter().any(|p| p == x) && body.mentions(x),
            E::Let(y, e, body) => e.mentions(x) || (y != x && body.mentions(x)),
            E::Loop { state, init, var, lo, hi, body } => init.mentions(x) || lo.mentions(x) || hi.mentions(x) || (state != x && var != x && body.mentions(x)),
        }
    }

    /// Rename the free `from` to `to` (the names of a normalised expression are unique, so nothing is captured).
    fn rename(&self, from: &str, to: &str) -> E {
        let r = |e: &E| e.rename(from, to);
        match self {
            E::Var(y) if y == from => var(to),
            E::Int(_) | E::Hole(_) | E::Var(_) => self.clone(),
            E::App(f, a) => E::App(b(r(f)), a.iter().map(r).collect()),
            E::Bin(op, l, rr) => E::Bin(op.clone(), b(r(l)), b(r(rr))),
            E::Index(l, rr) => E::Index(b(r(l)), b(r(rr))),
            E::Neg(e) => E::Neg(b(r(e))),
            E::If(c, t, e) => E::If(b(r(c)), b(r(t)), b(r(e))),
            E::Lam(ps, body) => {
                if ps.iter().any(|p| p == from) {
                    self.clone()
                } else {
                    E::Lam(ps.clone(), b(r(body)))
                }
            }
            E::Let(y, e, body) => E::Let(y.clone(), b(r(e)), if y == from { body.clone() } else { b(r(body)) }),
            E::Loop { state, init, var: v, lo, hi, body } => E::Loop {
                state: state.clone(),
                init: b(r(init)),
                var: v.clone(),
                lo: b(r(lo)),
                hi: b(r(hi)),
                body: if state == from || v == from { body.clone() } else { b(r(body)) },
            },
        }
    }

    /// The function a lambda is, when it only applies a name to its parameters (`\a b -> f a b` is `f`).
    fn eta(&self) -> Option<&str> {
        let E::Lam(ps, body) = self else { return None };
        let E::App(f, args) = body.as_ref() else { return None };
        let E::Var(name) = f.as_ref() else { return None };
        (args.len() == ps.len() && args.iter().zip(ps).all(|(a, p)| a == &E::Var(p.clone())) && !ps.iter().any(|p| p == name)).then_some(name.as_str())
    }
}

/// `loop s = init for t < D - 1 do let i = t + 1 in body` is the loop from 1 (`for i in 1..<D`): the spelling of the
/// kingdom's `loop1` rows and the form a person writes are one form.
fn canonical_loop(state: String, init: E, v: String, lo: E, hi: E, body: E) -> E {
    if lo == E::Int(0) {
        if let (E::Bin(minus, d, one), E::Let(i, step, rest)) = (&hi, &body) {
            if minus == "-" && **one == E::Int(1) && **step == E::Bin("+".into(), b(E::Var(v.clone())), b(E::Int(1))) && !rest.mentions(&v) {
                return E::Loop { state, init: b(init), var: i.clone(), lo: b(E::Int(1)), hi: d.clone(), body: rest.clone() };
            }
        }
    }
    E::Loop { state, init: b(init), var: v, lo: b(lo), hi: b(hi), body: b(body) }
}

// ---- the lexer --------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum K {
    Int(i64),
    Float,
    Ident,
    Sym,
    Hole(usize),
    Str,
}

#[derive(Debug, Clone)]
struct Tok {
    k: K,
    text: String,
    /// No whitespace before it (indexing `a[i]` against an array literal argument `f [1]`).
    tight: bool,
    /// The first token of its line, in column 0: a top-level item starts here.
    line_start: bool,
}

const KEYWORDS: [&str; 18] = ["let", "in", "if", "then", "else", "loop", "for", "while", "do", "def", "entry", "type", "module", "import", "open", "local", "with", "case"];

const SYMS: [&str; 41] = [
    "...", "..<", "..>", ">->", "<-<", "..", "->", "|>", "<|", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "**", "//", "%%", "+", "-", "*", "/", "%", "<", ">", "!", "=", "\\", "(", ")", "[", "]", ",", ":",
    "&", "|", "^", "{", "}",
];

fn is_operand(t: &Tok) -> bool {
    match t.k {
        K::Int(_) | K::Float | K::Hole(_) | K::Str => true,
        K::Ident => !KEYWORDS.contains(&t.text.as_str()),
        K::Sym => t.text == ")" || t.text == "]",
    }
}

fn lex(src: &str, holes: bool) -> Result<Vec<Tok>, Gap> {
    let c: Vec<char> = src.chars().collect();
    let mut out: Vec<Tok> = Vec::new();
    let mut i = 0;
    let mut tight = false;
    let mut col0 = true;
    let mut line_start = true;
    while i < c.len() {
        let ch = c[i];
        if ch == '\n' {
            i += 1;
            tight = false;
            line_start = true;
            col0 = true;
            continue;
        }
        if ch.is_whitespace() {
            i += 1;
            tight = false;
            col0 = false;
            continue;
        }
        if ch == '-' && c.get(i + 1) == Some(&'-') {
            while i < c.len() && c[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if ch == '#' && c.get(i + 1) == Some(&'[') {
            let mut depth = 0;
            while i < c.len() {
                if c[i] == '[' {
                    depth += 1;
                } else if c[i] == ']' {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                i += 1;
            }
            tight = false;
            continue;
        }
        let first = line_start && col0;
        line_start = false;
        col0 = false;
        let start = i;
        // `-1` is a negative literal where an operand cannot precede it (`x -1` is a subtraction).
        let neg = ch == '-' && c.get(i + 1).is_some_and(|d| d.is_ascii_digit()) && !out.last().is_some_and(is_operand);
        if ch.is_ascii_digit() || neg {
            if neg {
                i += 1;
            }
            let mut text = String::new();
            let mut float = false;
            if c[i] == '0' && c.get(i + 1).is_some_and(|x| *x == 'x' || *x == 'X') {
                i += 2;
                while i < c.len() && (c[i].is_ascii_hexdigit() || c[i] == '_') {
                    if c[i] != '_' {
                        text.push(c[i]);
                    }
                    i += 1;
                }
                let v = i64::from_str_radix(&text, 16).map_err(|e| Gap::parse("literal", format!("{text}: {e}")))?;
                text = v.to_string();
            } else {
                while i < c.len() && (c[i].is_ascii_digit() || c[i] == '_') {
                    if c[i] != '_' {
                        text.push(c[i]);
                    }
                    i += 1;
                }
                if i + 1 < c.len() && c[i] == '.' && c[i + 1].is_ascii_digit() {
                    float = true;
                    i += 1;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < c.len() && (c[i] == 'e' || c[i] == 'E') && c.get(i + 1).is_some_and(|d| d.is_ascii_digit() || *d == '-') {
                    float = true;
                    i += 2;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            // A suffix: i8..i64, u8..u64 (integers), f16..f64 (floats).
            if i < c.len() && matches!(c[i], 'i' | 'u' | 'f') {
                let s0 = i;
                i += 1;
                while i < c.len() && c[i].is_ascii_digit() {
                    i += 1;
                }
                let suffix: String = c[s0..i].iter().collect();
                if suffix.starts_with('f') {
                    float = true;
                } else if suffix != "i32" && suffix != "i64" {
                    out.push(Tok { k: K::Float, text: format!("integer of type {suffix}"), tight, line_start: first });
                    tight = true;
                    continue;
                }
            }
            let k = if float {
                K::Float
            } else {
                let v: i64 = text.parse().map_err(|e| Gap::parse("literal", format!("{text}: {e}")))?;
                K::Int(if neg { -v } else { v })
            };
            out.push(Tok { k, text: c[start..i].iter().collect(), tight, line_start: first });
            tight = true;
            continue;
        }
        if ch.is_alphabetic() || ch == '_' {
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_' || c[i] == '\'') {
                i += 1;
            }
            // A qualified name: `i32.min`, `f32.sqrt`.
            while i + 1 < c.len() && c[i] == '.' && (c[i + 1].is_alphabetic() || c[i + 1] == '_') {
                i += 1;
                while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_' || c[i] == '\'') {
                    i += 1;
                }
            }
            out.push(Tok { k: K::Ident, text: c[start..i].iter().collect(), tight, line_start: first });
            tight = true;
            continue;
        }
        if ch == '"' {
            i += 1;
            while i < c.len() && c[i] != '"' {
                i += 1;
            }
            i += 1;
            out.push(Tok { k: K::Str, text: c[start..i.min(c.len())].iter().collect(), tight, line_start: first });
            tight = true;
            continue;
        }
        if holes && ch == '{' {
            let mut j = i + 1;
            while j < c.len() && c[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && c.get(j) == Some(&'}') {
                let n: usize = c[i + 1..j].iter().collect::<String>().parse().map_err(|_| Gap::parse("hole", "a hole number".into()))?;
                out.push(Tok { k: K::Hole(n), text: c[i..=j].iter().collect(), tight, line_start: first });
                i = j + 1;
                tight = true;
                continue;
            }
        }
        let rest: String = c[i..c.len().min(i + 3)].iter().collect();
        // A character outside the subset is a token of its own: only an item that uses it is a gap (`token .`).
        let sym = SYMS.iter().find(|s| rest.starts_with(**s)).map_or_else(|| ch.to_string(), |s| s.to_string());
        i += sym.chars().count();
        out.push(Tok { k: K::Sym, text: sym, tight, line_start: first });
        tight = true;
    }
    Ok(out)
}

// ---- the parser -------------------------------------------------------------------------------------------------

/// A parameter's or a result's type: its dimension names (`None` for an anonymous `[]` or an expression) and element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FType {
    pub dims: Vec<Option<String>>,
    pub elem: String,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Option<FType>,
}

/// A top-level `def` or `entry`.
#[derive(Debug, Clone)]
pub struct Def {
    pub name: String,
    pub entry: bool,
    pub size_params: Vec<String>,
    pub params: Vec<Param>,
    pub ret: Option<FType>,
    pub body: Result<E, Gap>,
}

struct Parser<'t> {
    t: &'t [Tok],
    p: usize,
    fresh: usize,
}

const BINOPS: [(&str, u8); 23] = [
    ("|>", 0),
    ("<|", 0),
    (">->", 0),
    ("<-<", 0),
    ("||", 1),
    ("&&", 2),
    ("==", 3),
    ("!=", 3),
    ("<", 3),
    ("<=", 3),
    (">", 3),
    (">=", 3),
    ("&", 4),
    ("^", 4),
    ("|", 4),
    ("<<", 5),
    (">>", 5),
    ("+", 6),
    ("-", 6),
    ("*", 7),
    ("/", 7),
    ("%", 7),
    ("//", 7),
];

fn binop(s: &str) -> Option<u8> {
    BINOPS.iter().find(|(o, _)| *o == s).map(|(_, p)| *p).or(match s {
        "%%" => Some(7),
        "**" => Some(8),
        _ => None,
    })
}

impl<'t> Parser<'t> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.p)
    }
    fn peek_at(&self, d: usize) -> Option<&Tok> {
        self.t.get(self.p + d)
    }
    fn is(&self, s: &str) -> bool {
        self.peek().is_some_and(|t| t.text == s && matches!(t.k, K::Sym | K::Ident))
    }
    fn eat(&mut self, s: &str) -> bool {
        if self.is(s) {
            self.p += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, s: &str) -> Result<(), Gap> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(self.unexpected(&format!("expected {s}")))
        }
    }
    fn unexpected(&self, why: &str) -> Gap {
        match self.peek() {
            Some(t) if t.k == K::Float => Gap::parse("float literal", format!("{} ({why})", t.text)),
            Some(t) if t.text == "with" => Gap::parse("in-place update (with)", why.into()),
            Some(t) if t.text == "while" => Gap::parse("while loop", why.into()),
            Some(t) if t.text == "case" || t.text == "match" => Gap::parse("match", why.into()),
            Some(t) => Gap::parse(&format!("token {}", t.text), why.into()),
            None => Gap::parse("end of text", why.into()),
        }
    }
    fn ident(&mut self) -> Result<String, Gap> {
        match self.peek() {
            Some(t) if t.k == K::Ident && !KEYWORDS.contains(&t.text.as_str()) => {
                let s = t.text.clone();
                self.p += 1;
                Ok(s)
            }
            _ => Err(self.unexpected("expected a name")),
        }
    }
    fn fresh(&mut self) -> String {
        self.fresh += 1;
        format!("%s{}", self.fresh)
    }

    /// A type: `*`, `[d]`... then an element; a tuple, record or function type is a gap only where it matters, so it
    /// is returned with the element `tuple` or `function`.
    fn ftype(&mut self) -> Result<FType, Gap> {
        self.eat("*");
        let mut dims = Vec::new();
        while self.is("[") {
            self.p += 1;
            let mut toks = Vec::new();
            let mut depth = 0;
            while let Some(t) = self.peek() {
                if t.text == "]" && depth == 0 {
                    break;
                }
                if t.text == "[" {
                    depth += 1;
                }
                if t.text == "]" {
                    depth -= 1;
                }
                toks.push(t.clone());
                self.p += 1;
            }
            self.expect("]")?;
            dims.push(match toks.as_slice() {
                [t] if t.k == K::Ident => Some(t.text.clone()),
                // `[i64.i32 k]`: the size input's dimension, as the kingdom spells it.
                [a, t] if a.text == "i64.i32" && t.k == K::Ident => Some(t.text.clone()),
                _ => None,
            });
        }
        let elem = if self.is("(") || self.is("{") {
            let open = self.peek().map(|t| t.text.clone()).unwrap_or_default();
            let close = if open == "(" { ")" } else { "}" };
            let mut depth = 0;
            while let Some(t) = self.peek() {
                if t.text == open {
                    depth += 1;
                } else if t.text == close {
                    depth -= 1;
                    if depth == 0 {
                        self.p += 1;
                        break;
                    }
                }
                self.p += 1;
            }
            "tuple".to_string()
        } else if self.peek().is_some_and(|t| t.text == "'" || t.text.starts_with('\'')) {
            self.p += 1;
            "type parameter".into()
        } else {
            self.ident()?
        };
        if self.is("->") {
            self.p += 1;
            self.ftype()?;
            return Ok(FType { dims, elem: "function".into() });
        }
        Ok(FType { dims, elem })
    }

    /// A binder: a name, `(name)`, `(name: t)`, `name: t`; `_` is a fresh name; a tuple is a gap.
    fn binder(&mut self) -> Result<(String, Option<FType>), Gap> {
        if self.eat("(") {
            if self.is(")") {
                return Err(Gap::parse("unit pattern", "()".into()));
            }
            let n = if self.is("_") {
                self.p += 1;
                self.fresh()
            } else {
                self.ident()?
            };
            let mut ty = None;
            if self.eat(":") {
                ty = Some(self.ftype()?);
            }
            if self.is(",") {
                return Err(Gap::parse("tuple pattern", n));
            }
            self.expect(")")?;
            return Ok((n, ty));
        }
        if self.is("_") {
            self.p += 1;
            return Ok((self.fresh(), None));
        }
        Ok((self.ident()?, None))
    }

    fn expr(&mut self) -> Result<E, Gap> {
        if self.is("let") {
            return self.let_expr();
        }
        if self.eat("if") {
            let c = self.expr()?;
            self.expect("then")?;
            let t = self.expr()?;
            self.expect("else")?;
            let e = self.expr()?;
            return Ok(E::If(b(c), b(t), b(e)));
        }
        if self.is("loop") {
            return self.loop_expr();
        }
        if self.is("\\") {
            return self.lambda();
        }
        if self.is("while") {
            return Err(Gap::parse("while loop", String::new()));
        }
        self.binary(0)
    }

    fn lambda(&mut self) -> Result<E, Gap> {
        self.expect("\\")?;
        let mut ps = Vec::new();
        while !self.is("->") && !self.is(":") {
            ps.push(self.binder()?.0);
        }
        if self.eat(":") {
            self.ftype()?;
        }
        self.expect("->")?;
        let body = self.expr()?;
        Ok(E::Lam(ps, b(body)))
    }

    fn let_expr(&mut self) -> Result<E, Gap> {
        self.expect("let")?;
        if self.is("(") && self.peek_at(2).is_some_and(|t| t.text == ",") {
            return Err(Gap::parse("tuple pattern", "let (a, b) = ...".into()));
        }
        let (name, _) = self.binder()?;
        if self.is("[") && self.peek().is_some_and(|t| t.tight) {
            return Err(Gap::parse("in-place update (let a[i] = ...)", name));
        }
        // `let f x y = ...`: a local function.
        let mut params = Vec::new();
        while !self.is("=") && !self.is(":") {
            params.push(self.binder()?.0);
        }
        if self.eat(":") {
            self.ftype()?;
        }
        self.expect("=")?;
        let mut rhs = self.expr()?;
        if !params.is_empty() {
            rhs = E::Lam(params, b(rhs));
        }
        let body = if self.eat("in") {
            self.expr()?
        } else if self.is("let") {
            self.let_expr()?
        } else {
            return Err(self.unexpected("a let with no body"));
        };
        Ok(E::Let(name, b(rhs), b(body)))
    }

    fn loop_expr(&mut self) -> Result<E, Gap> {
        self.expect("loop")?;
        if self.is("(") && self.peek_at(2).is_some_and(|t| t.text == ",") {
            return Err(Gap::parse("tuple pattern", "loop (a, b) = ...".into()));
        }
        let (state, _) = self.binder()?;
        let init = if self.eat("=") { self.expr()? } else { E::Var(state.clone()) };
        if self.is("while") {
            return Err(Gap::parse("while loop", String::new()));
        }
        self.expect("for")?;
        let v = self.binder()?.0;
        let (lo, hi) = if self.eat("<") {
            (E::Int(0), self.binary(1)?)
        } else if self.eat("in") {
            let lo = self.binary(4)?;
            if self.eat("..<") {
                (lo, self.binary(4)?)
            } else if self.is("..") || self.is("...") {
                return Err(Gap::parse("inclusive range", String::new()));
            } else {
                return Err(Gap::parse("loop over an array (for x in xs)", String::new()));
            }
        } else {
            return Err(self.unexpected("a loop form"));
        };
        self.expect("do")?;
        let body = self.expr()?;
        Ok(canonical_loop(state, init, v, lo, hi, body))
    }

    fn binary(&mut self, min: u8) -> Result<E, Gap> {
        let mut lhs = self.unary()?;
        while let Some(t) = self.peek() {
            if t.k != K::Sym {
                break;
            }
            let op = t.text.clone();
            let Some(prec) = binop(&op) else { break };
            if prec < min {
                break;
            }
            // `(e op)`: a left section, closed by the caller.
            if self.peek_at(1).is_some_and(|t| t.text == ")") {
                break;
            }
            self.p += 1;
            let rhs = if self.is("\\") || self.is("if") || self.is("let") || self.is("loop") {
                self.expr()?
            } else if op == "<|" {
                self.binary(prec)?
            } else {
                self.binary(prec + 1)?
            };
            lhs = combine(&op, lhs, rhs, &mut self.fresh);
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<E, Gap> {
        if self.is("-") {
            self.p += 1;
            return Ok(E::Neg(b(self.unary()?)));
        }
        if self.is("!") {
            return Err(Gap::parse("logical not", String::new()));
        }
        self.application()
    }

    fn starts_atom(&self) -> bool {
        match self.peek() {
            Some(t) => match &t.k {
                K::Int(_) | K::Hole(_) | K::Float => true,
                K::Ident => !KEYWORDS.contains(&t.text.as_str()),
                K::Sym => t.text == "(" || (t.text == "[" && !t.tight),
                K::Str => false,
            },
            None => false,
        }
    }

    fn application(&mut self) -> Result<E, Gap> {
        let f = self.postfix()?;
        let mut args = Vec::new();
        while self.starts_atom() {
            args.push(self.postfix()?);
        }
        Ok(if args.is_empty() { f } else { E::App(b(f), args) })
    }

    fn postfix(&mut self) -> Result<E, Gap> {
        let mut e = self.atom()?;
        while self.peek().is_some_and(|t| t.text == "[" && t.tight && t.k == K::Sym) {
            self.p += 1;
            loop {
                let i = self.expr()?;
                if self.is(":") {
                    return Err(Gap::parse("slice", String::new()));
                }
                e = E::Index(b(e), b(i));
                if !self.eat(",") {
                    break;
                }
            }
            self.expect("]")?;
        }
        if self.peek().is_some_and(|t| t.text == "." && t.tight) {
            return Err(Gap::parse("field access", String::new()));
        }
        Ok(e)
    }

    fn atom(&mut self) -> Result<E, Gap> {
        let Some(t) = self.peek().cloned() else { return Err(self.unexpected("an expression")) };
        match t.k {
            K::Int(v) => {
                self.p += 1;
                Ok(E::Int(v))
            }
            K::Hole(i) => {
                self.p += 1;
                Ok(E::Hole(i))
            }
            K::Float => Err(Gap::parse("float literal", t.text)),
            K::Str => Err(Gap::parse("string", t.text)),
            K::Ident => {
                let n = self.ident()?;
                Ok(E::Var(n))
            }
            K::Sym if t.text == "[" => Err(Gap::parse("array literal", String::new())),
            K::Sym if t.text == "(" => {
                self.p += 1;
                if self.is(")") {
                    return Err(Gap::parse("unit", "()".into()));
                }
                // `(op)` and `(op e)`.
                if let Some(op) = self.peek().filter(|t| t.k == K::Sym && binop(&t.text).is_some()).map(|t| t.text.clone()) {
                    if self.peek_at(1).is_some_and(|t| t.text == ")") {
                        self.p += 2;
                        let (x, y) = (self.fresh(), self.fresh());
                        return Ok(E::Lam(vec![x.clone(), y.clone()], b(combine(&op, E::Var(x), E::Var(y), &mut self.fresh))));
                    }
                    if op != "-" {
                        self.p += 1;
                        let e = self.binary(0)?;
                        self.expect(")")?;
                        let x = self.fresh();
                        return Ok(E::Lam(vec![x.clone()], b(combine(&op, E::Var(x), e, &mut self.fresh))));
                    }
                }
                let e = self.expr()?;
                if self.is(",") {
                    return Err(Gap::parse("tuple", String::new()));
                }
                if self.eat(":") {
                    self.ftype()?;
                }
                // `(e op)`.
                if let Some(op) = self.peek().filter(|t| t.k == K::Sym && binop(&t.text).is_some()).map(|t| t.text.clone()) {
                    if self.peek_at(1).is_some_and(|t| t.text == ")") {
                        self.p += 2;
                        let x = self.fresh();
                        return Ok(E::Lam(vec![x.clone()], b(combine(&op, e, E::Var(x), &mut self.fresh))));
                    }
                }
                self.expect(")")?;
                Ok(e)
            }
            _ => Err(self.unexpected("an expression")),
        }
    }

    /// A top-level `def`/`entry`/`let`, from its keyword.
    fn def(&mut self) -> Result<Def, Gap> {
        let entry = self.is("entry");
        self.p += 1;
        if self.is("(") {
            return Err(Gap::parse("operator definition", String::new()));
        }
        let name = self.ident()?;
        let mut size_params = Vec::new();
        let mut params = Vec::new();
        loop {
            if self.is("[") {
                self.p += 1;
                size_params.push(self.ident()?);
                self.expect("]")?;
            } else if self.peek().is_some_and(|t| t.text.starts_with('\'')) {
                return Err(Gap::parse("type parameter", name));
            } else if self.is("(") {
                if self.peek_at(2).is_some_and(|t| t.text == ",") {
                    return Err(Gap::parse("tuple parameter", name));
                }
                let (n, ty) = self.binder()?;
                params.push(Param { name: n, ty });
            } else if self.peek().is_some_and(|t| t.k == K::Ident && !KEYWORDS.contains(&t.text.as_str())) {
                let n = self.ident()?;
                params.push(Param { name: n, ty: None });
            } else {
                break;
            }
        }
        let ret = if self.eat(":") { Some(self.ftype()?) } else { None };
        self.expect("=")?;
        let body = self.expr().and_then(|e| if self.p < self.t.len() { Err(self.unexpected("text after the body")) } else { Ok(e) });
        Ok(Def { name, entry, size_params, params, ret, body })
    }
}

/// A binary operator's node; `|>`, `<|`, `>->` and `<-<` are applications and compositions.
fn combine(op: &str, l: E, r: E, fresh: &mut usize) -> E {
    match op {
        "|>" => E::App(b(r), vec![l]),
        "<|" => E::App(b(l), vec![r]),
        ">->" | "<-<" => {
            *fresh += 1;
            let x = format!("%s{fresh}");
            let (f, g) = if op == ">->" { (l, r) } else { (r, l) };
            E::Lam(vec![x.clone()], b(E::App(b(g), vec![E::App(b(f), vec![E::Var(x)])])))
        }
        _ => E::Bin(op.to_string(), b(l), b(r)),
    }
}

/// Parse an expression (a row's spelling, with holes).
pub fn parse_pattern(text: &str) -> Result<E, Gap> {
    let toks = lex(text, true)?;
    let mut p = Parser { t: &toks, p: 0, fresh: 0 };
    let e = p.expr()?;
    if p.p != toks.len() {
        return Err(p.unexpected("text after the expression"));
    }
    Ok(e)
}

/// Parse a Futhark source file into its top-level `def`s and `entry`s (other items, `import`, `type`, `module`,
/// `open`, are skipped; an item that does not parse is a `Def` whose body is the gap).
pub fn parse_file(src: &str) -> Result<Vec<Def>, Gap> {
    let toks = lex(src, false)?;
    let starts: Vec<usize> = (0..toks.len()).filter(|&i| toks[i].line_start && toks[i].k == K::Ident && ["def", "entry", "let", "type", "module", "import", "open", "local", "include"].contains(&toks[i].text.as_str())).collect();
    let mut out = Vec::new();
    for (j, &s) in starts.iter().enumerate() {
        let end = starts.get(j + 1).copied().unwrap_or(toks.len());
        let mut s = s;
        if toks[s].text == "local" {
            s += 1;
        }
        if s >= end || !["def", "entry", "let"].contains(&toks[s].text.as_str()) {
            continue;
        }
        let name = toks.get(s + 1).map(|t| t.text.clone()).unwrap_or_default();
        let mut p = Parser { t: &toks[s..end], p: 0, fresh: 0 };
        match p.def() {
            Ok(d) => out.push(d),
            Err(g) => out.push(Def { name, entry: toks[s].text == "entry", size_params: vec![], params: vec![], ret: None, body: Err(g) }),
        }
    }
    Ok(out)
}

// ---- the normaliser ---------------------------------------------------------------------------------------------

struct Norm<'a> {
    dims: &'a [String],
    /// The dimensions of the inputs, by canonical name (for `length`).
    input_dims: &'a BTreeMap<String, Vec<String>>,
    fresh: usize,
    steps: usize,
}

const NORM_STEPS: usize = 2_000_000;

impl Norm<'_> {
    fn fresh(&mut self) -> String {
        self.fresh += 1;
        format!("%v{}", self.fresh)
    }

    fn is_dim(&self, e: &E) -> Option<String> {
        match e {
            E::Var(d) if self.dims.contains(d) => Some(d.clone()),
            _ => None,
        }
    }

    fn dim_of(&self, e: &E) -> Option<String> {
        match e {
            E::Var(x) => self.input_dims.get(x).and_then(|d| d.first().cloned()).or_else(|| x.strip_prefix("zeros_").filter(|d| self.dims.iter().any(|x| x == d)).map(str::to_string)),
            E::Index(a, _) => match a.as_ref() {
                E::Var(x) => self.input_dims.get(x).and_then(|d| d.get(1).cloned()),
                _ => None,
            },
            E::App(f, args) => match (f.as_ref(), args.as_slice()) {
                (E::Var(f), [d, _]) if f == "replicate" || f == "tabulate" => self.is_dim(d),
                (E::Var(f), [d]) if f == "iota" => self.is_dim(d),
                (E::Var(f), [_, xs]) if f == "map" => self.dim_of(xs),
                (E::Var(f), [_, xs, _]) if f == "map2" || f == "scan" => self.dim_of(xs),
                (E::Var(f), [_, _, d, _, _]) if f == "hist" => self.is_dim(d),
                _ => None,
            },
            _ => None,
        }
    }

    fn ev(&mut self, e: &E, env: &HashMap<String, E>) -> Result<E, Gap> {
        self.steps += 1;
        if self.steps > NORM_STEPS {
            return Err(Gap::parse("program too large to inline", format!("over {NORM_STEPS} normalisation steps")));
        }
        Ok(match e {
            E::Int(_) | E::Hole(_) => e.clone(),
            E::Var(x) => env.get(x).cloned().unwrap_or_else(|| e.clone()),
            E::Neg(x) => E::Neg(b(self.ev(x, env)?)),
            E::Bin(op, l, r) => {
                let (l, r) = (self.ev(l, env)?, self.ev(r, env)?);
                // `(i64.i32 e) % D` is `i64.i32 (e % i32.i64 D)` for a size D (sizes are positive i32).
                if op == "%" {
                    if let (E::App(f, a), Some(d)) = (&l, self.is_dim(&r)) {
                        if **f == var("i64.i32") && a.len() == 1 {
                            return Ok(app("i64.i32", vec![E::Bin("%".into(), b(a[0].clone()), b(app("i32.i64", vec![var(&d)])))]));
                        }
                    }
                }
                E::Bin(op.clone(), b(l), b(r))
            }
            E::Index(a, i) => E::Index(b(self.ev(a, env)?), b(self.ev(i, env)?)),
            E::If(c, t, f) => E::If(b(self.ev(c, env)?), b(self.ev(t, env)?), b(self.ev(f, env)?)),
            E::Lam(ps, body) => {
                let mut env2 = env.clone();
                let mut fresh = Vec::with_capacity(ps.len());
                for p in ps {
                    let f = self.fresh();
                    env2.insert(p.clone(), E::Var(f.clone()));
                    fresh.push(f);
                }
                E::Lam(fresh, b(self.ev(body, &env2)?))
            }
            E::Let(x, rhs, body) => {
                let v = self.ev(rhs, env)?;
                // `let z = replicate D 0` is the kingdom's `zeros_D`.
                let v = match &v {
                    E::App(f, a) if **f == var("replicate") && a.len() == 2 && a[1] == E::Int(0) && self.is_dim(&a[0]).is_some() => var(&format!("zeros_{}", self.is_dim(&a[0]).unwrap_or_default())),
                    _ => v,
                };
                let mut env2 = env.clone();
                env2.insert(x.clone(), v);
                self.ev(body, &env2)?
            }
            E::Loop { state, init, var: v, lo, hi, body } => {
                let init = self.ev(init, env)?;
                let (lo, hi) = (self.ev(lo, env)?, self.ev(hi, env)?);
                let (s2, v2) = (self.fresh(), self.fresh());
                let mut env2 = env.clone();
                env2.insert(state.clone(), E::Var(s2.clone()));
                env2.insert(v.clone(), E::Var(v2.clone()));
                let body = self.ev(body, &env2)?;
                E::Loop { state: s2, init: b(init), var: v2, lo: b(lo), hi: b(hi), body: b(body) }
            }
            E::App(f, args) => {
                let f = self.ev(f, env)?;
                let args = args.iter().map(|a| self.ev(a, env)).collect::<Result<Vec<_>, _>>()?;
                self.apply(f, args)?
            }
        })
    }

    /// Apply a normalised function to normalised arguments: a lambda is reduced, an application is extended, and the
    /// exact identities of the module's header are applied.
    fn apply(&mut self, f: E, mut args: Vec<E>) -> Result<E, Gap> {
        if args.is_empty() {
            return Ok(f);
        }
        match f {
            E::Lam(ps, body) => {
                let n = ps.len().min(args.len());
                let mut env = HashMap::new();
                for (p, a) in ps.iter().zip(args.drain(..n)) {
                    env.insert(p.clone(), a);
                }
                let rest: Vec<String> = ps[n..].to_vec();
                let body = self.ev(&body, &env)?;
                let r = if rest.is_empty() { body } else { self.ev(&E::Lam(rest, b(body)), &HashMap::new())? };
                self.apply(r, args)
            }
            E::App(g, mut a0) => {
                a0.extend(args);
                self.apply(*g, a0)
            }
            E::Var(name) => {
                let one = args.len() == 1;
                match name.as_str() {
                    "i32.i32" | "i64.i64" | "copy" if one => return Ok(args.remove(0)),
                    // A size input `k: i32` as a size: `i64.i32 k` is the size `k`.
                    "i64.i32" if one => {
                        if let E::Var(x) = &args[0] {
                            if let Some(d) = x.strip_suffix(SIZE_I32).filter(|d| self.dims.iter().any(|x| x == d)) {
                                return Ok(var(d));
                            }
                        }
                    }
                    "opaque" if one => return Ok(args.remove(0)),
                    "i32.i64" if one => {
                        if let E::App(g, a) = &args[0] {
                            if **g == var("i64.i32") && a.len() == 1 {
                                return Ok(a[0].clone());
                            }
                        }
                    }
                    "length" if one => {
                        if let Some(d) = self.dim_of(&args[0]) {
                            return Ok(var(&d));
                        }
                    }
                    "map" if args.len() == 2 && (args[0] == var("i32.i32") || args[0].eta() == Some("i32.i32")) => return Ok(args.remove(1)),
                    _ => {}
                }
                Ok(E::App(b(E::Var(name)), args))
            }
            other => Ok(E::App(b(other), args)),
        }
    }
}

/// A function of a file and what reading it gave: its type and tree, or the gap.
pub type FunctionRead = (String, Result<(Ty, Tree), Gap>);

/// The suffix of a size input's `i32` parameter (`k: i32` is `k%i32`; `i64.i32 k` is the size `k`).
const SIZE_I32: &str = "%i32";

/// `opaque x` is `x` (Futhark's optimisation barrier, an identity): stripped from patterns and programs alike.
fn strip_opaque(e: E) -> E {
    let s = |e: &E| strip_opaque(e.clone());
    match e {
        E::App(f, mut a) if *f == var("opaque") && a.len() == 1 => strip_opaque(a.remove(0)),
        E::App(f, a) => E::App(b(s(&f)), a.iter().map(s).collect()),
        E::Bin(op, l, r) => E::Bin(op, b(s(&l)), b(s(&r))),
        E::Index(l, r) => E::Index(b(s(&l)), b(s(&r))),
        E::Neg(x) => E::Neg(b(s(&x))),
        E::If(c, t, f) => E::If(b(s(&c)), b(s(&t)), b(s(&f))),
        E::Lam(ps, body) => E::Lam(ps, b(s(&body))),
        E::Let(x, v, body) => E::Let(x, b(s(&v)), b(s(&body))),
        E::Loop { state, init, var: v, lo, hi, body } => E::Loop { state, init: b(s(&init)), var: v, lo: b(s(&lo)), hi: b(s(&hi)), body: b(s(&body)) },
        other => other,
    }
}

// ---- the reader -------------------------------------------------------------------------------------------------

/// The reader of one instance: every row's spelling as a pattern, by output type.
pub struct Reader<'a> {
    pub k: &'a FutharkKingdom,
    pub sig: &'a Signature,
    by_ty: BTreeMap<Ty, Vec<(u32, E)>>,
    consts: BTreeMap<Ty, u32>,
    input_dims: BTreeMap<String, Vec<String>>,
    steps: std::cell::Cell<usize>,
}

/// The most pattern matches one read may try (a guard: reading a program always ends).
const READ_STEPS: usize = 5_000_000;

impl<'a> Reader<'a> {
    pub fn new(k: &'a FutharkKingdom, sig: &'a Signature) -> Result<Reader<'a>, String> {
        let mut by_ty: BTreeMap<Ty, Vec<(u32, E)>> = BTreeMap::new();
        let mut consts = BTreeMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            if r.kind == FutharkKind::Const {
                consts.insert(r.output, i as u32);
                continue;
            }
            let p = strip_opaque(parse_pattern(&r.spelling).map_err(|g| format!("our software has a bug: the spelling of {} does not parse as a pattern: {g}", r.name))?);
            by_ty.entry(r.output).or_default().push((i as u32, p));
        }
        let input_dims = sig.inputs.iter().filter(|p| !sig.size_inputs.contains(&p.name)).map(|p| (p.name.clone(), p.dims.clone())).collect();
        Ok(Reader { k, sig, by_ty, consts, input_dims, steps: std::cell::Cell::new(0) })
    }

    fn norm(&self) -> Norm<'_> {
        Norm { dims: &self.sig.dims, input_dims: &self.input_dims, fresh: 0, steps: 0 }
    }

    /// A whole program with the declared signature (its entry `main`, or its only entry), read at the root type.
    pub fn read_program(&self, src: &str) -> Result<Tree, Gap> {
        let defs = parse_file(src)?;
        let entries: Vec<&Def> = defs.iter().filter(|d| d.entry).collect();
        let entry = entries.iter().find(|d| d.name == "main").copied().or(if entries.len() == 1 { Some(entries[0]) } else { None }).ok_or_else(|| Gap::parse("no entry point", format!("{} entries, none named main", entries.len())))?;
        let mut norm = self.norm();
        let mut env = self.helpers(&defs, entry, &mut norm)?;
        if entry.params.len() != self.sig.inputs.len() {
            return Err(Gap::parse("entry signature", format!("{} parameters, the signature has {}", entry.params.len(), self.sig.inputs.len())));
        }
        let mut dim_map: HashMap<String, String> = HashMap::new();
        for (p, s) in entry.params.iter().zip(&self.sig.inputs) {
            env.insert(p.name.clone(), self.canonical_input(s));
            if let Some(t) = &p.ty {
                for (d, sd) in t.dims.iter().zip(&s.dims) {
                    if let Some(d) = d {
                        dim_map.insert(d.clone(), sd.clone());
                    }
                }
            }
        }
        for sp in &entry.size_params {
            let d = dim_map.get(sp).ok_or_else(|| Gap::parse("entry signature", format!("the size parameter {sp} is no dimension of the signature")))?;
            env.insert(sp.clone(), var(d));
        }
        let body = entry.body.clone()?;
        let e = norm.ev(&body, &env)?;
        let root = *self.k.roots().first().ok_or_else(|| Gap::parse("no root", String::new()))?;
        self.steps.set(0);
        self.read(&e, root)
    }

    /// A size input `k: i32` is the name `k%i32` (its `i64.i32` is the size `k`); an array input is its own name.
    fn canonical_input(&self, s: &crate::futhark_sig::Param) -> E {
        if self.sig.size_inputs.contains(&s.name) {
            var(&format!("{}{SIZE_I32}", s.name))
        } else {
            var(&s.name)
        }
    }

    /// The helper definitions before `upto`, normalised, as the environment of a body.
    fn helpers(&self, defs: &[Def], upto: &Def, norm: &mut Norm<'_>) -> Result<HashMap<String, E>, Gap> {
        let mut env: HashMap<String, E> = HashMap::new();
        for d in defs {
            if std::ptr::eq(d, upto) {
                break;
            }
            let Ok(body) = &d.body else { continue };
            // A size parameter of a helper is bound by its arguments' shapes: kept free (a gap if it is used).
            let ps: Vec<String> = d.params.iter().map(|p| p.name.clone()).collect();
            let lam = if ps.is_empty() { body.clone() } else { E::Lam(ps, b(body.clone())) };
            match norm.ev(&lam, &env) {
                Ok(v) => {
                    env.insert(d.name.clone(), v);
                }
                Err(_) => continue,
            }
        }
        Ok(env)
    }

    /// Every function of a source file read in this kingdom (`docs/PLAN_futhark_fitness_fixes.md` §H, the published
    /// benchmarks as a source of the rings): each `def`/`entry`, its parameters bound to the kingdom's inputs of the
    /// same type (an `i32` array of rank r to the next unused input of rank r, a scalar to the next unused size input
    /// or dimension), its body read at the type of its result in the top context. In file order.
    pub fn read_functions(&self, src: &str) -> Vec<FunctionRead> {
        let defs = match parse_file(src) {
            Ok(d) => d,
            Err(g) => return vec![("(file)".into(), Err(g))],
        };
        let mut out = Vec::new();
        for d in &defs {
            let mut norm = self.norm();
            let r = self.helpers(&defs, d, &mut norm).and_then(|env| self.read_function(d, env, &mut norm));
            out.push((d.name.clone(), r));
        }
        out
    }

    fn read_function(&self, d: &Def, mut env: HashMap<String, E>, norm: &mut Norm<'_>) -> Result<(Ty, Tree), Gap> {
        let body = d.body.clone()?;
        let arrays: Vec<&crate::futhark_sig::Param> = self.sig.inputs.iter().filter(|p| !self.sig.size_inputs.contains(&p.name)).collect();
        let mut used_arrays: Vec<&str> = Vec::new();
        let mut scalars: Vec<E> = self.sig.size_inputs.iter().map(|s| var(&format!("{s}{SIZE_I32}"))).collect();
        scalars.extend(self.sig.dims.iter().filter(|x| !self.sig.size_inputs.contains(x)).map(|x| app("i32.i64", vec![var(x)])));
        let mut sizes: Vec<String> = self.sig.dims.clone();
        let mut dim_map: HashMap<String, String> = HashMap::new();
        for p in &d.params {
            let Some(t) = &p.ty else { return Err(Gap::parse("untyped parameter", p.name.clone())) };
            match (t.elem.as_str(), t.dims.len()) {
                ("i32", 0) => {
                    if scalars.is_empty() {
                        return Err(Gap::parse("parameter i32 (no scalar terminal left)", p.name.clone()));
                    }
                    let s = scalars.remove(0);
                    env.insert(p.name.clone(), norm.ev(&s, &HashMap::new())?);
                }
                ("i64", 0) => {
                    if sizes.is_empty() {
                        return Err(Gap::parse("parameter i64 (no dimension left)", p.name.clone()));
                    }
                    env.insert(p.name.clone(), var(&sizes.remove(0)));
                }
                ("i32", r) => {
                    let Some(a) = arrays.iter().find(|a| a.dims.len() == r && !used_arrays.contains(&a.name.as_str())) else {
                        return Err(Gap::parse(&format!("parameter {}i32 (no input of rank {r} left)", "[]".repeat(r)), p.name.clone()));
                    };
                    used_arrays.push(&a.name);
                    for (dd, sd) in t.dims.iter().zip(&a.dims) {
                        if let Some(dd) = dd {
                            if dim_map.insert(dd.clone(), sd.clone()).is_some_and(|old| &old != sd) {
                                return Err(Gap::parse("parameter shapes", format!("the dimension {dd} is two dimensions of the signature")));
                            }
                        }
                    }
                    env.insert(p.name.clone(), var(&a.name));
                }
                (elem, r) => return Err(Gap::parse(&format!("parameter of type {}{elem}", "[]".repeat(r)), p.name.clone())),
            }
        }
        for sp in &d.size_params {
            match dim_map.get(sp) {
                Some(x) => {
                    env.insert(sp.clone(), var(x));
                }
                None => return Err(Gap::parse("size parameter no input binds", sp.clone())),
            }
        }
        let e = norm.ev(&body, &env)?;
        // An untyped result: every class of the kingdom's top context is tried.
        let ret = d.ret.clone().unwrap_or(FType { dims: vec![], elem: "untyped".into() });
        let classes: Vec<String> = match (ret.elem.as_str(), ret.dims.len()) {
            ("untyped", _) => std::iter::once("i32".to_string()).chain(self.sig.dims.iter().map(|x| format!("a.{x}"))).collect(),
            ("i32", 0) => vec!["i32".into()],
            ("i32", 1) => match &ret.dims[0] {
                Some(dd) if dim_map.contains_key(dd) => vec![format!("a.{}", dim_map[dd])],
                Some(dd) if self.sig.dims.contains(dd) => vec![format!("a.{dd}")],
                _ => self.sig.dims.iter().map(|x| format!("a.{x}")).collect(),
            },
            (elem, r) => return Err(Gap::parse(&format!("result of type {}{elem}", "[]".repeat(r)), d.name.clone())),
        };
        let mut first = None;
        for c in classes {
            let Some(ty) = self.k.ty(&format!("{c}:c")) else { continue };
            self.steps.set(0);
            match self.read(&e, ty) {
                Ok(t) => return Ok((ty, t)),
                Err(g) => {
                    first.get_or_insert(g);
                }
            }
        }
        Err(first.unwrap_or_else(|| Gap::parse("result class", "no class of the kingdom".into())))
    }

    /// Read a normalised expression at a type.
    pub fn read(&self, e: &E, ty: Ty) -> Result<Tree, Gap> {
        self.steps.set(self.steps.get() + 1);
        let dual = self.k.dual(ty).ok_or_else(|| Gap::parse("type", "a type outside the kingdom".into()))?;
        if self.steps.get() > READ_STEPS {
            return Err(Gap { construct: "reading budget".into(), dual: dual.dual.clone(), detail: format!("over {READ_STEPS} matches") });
        }
        // A literal in a constant slot.
        if let (E::Int(v), Some(&sym)) = (e, self.consts.get(&ty)) {
            if let Some(i) = dual.literals.iter().position(|l| l.trim_end_matches(dual.slot.as_str()).parse::<i64>() == Ok(*v)) {
                return Ok(Tree { symbol: sym, const_index: Some(i as u32), kids: vec![] });
            }
        }
        let mut first: Option<Gap> = None;
        if let Some(rows) = self.by_ty.get(&ty) {
            for (sym, pat) in rows {
                let n = self.k.rows[*sym as usize].inputs.len();
                for holes in pmatch_all(pat, e, n) {
                    let inputs = &self.k.rows[*sym as usize].inputs;
                    let mut kids = Vec::with_capacity(n);
                    let mut failed = None;
                    for (h, t) in holes.iter().zip(inputs) {
                        let Some(h) = h else {
                            failed = Some(Gap { construct: "spelling".into(), dual: dual.dual.clone(), detail: "a hole the spelling does not bind".into() });
                            break;
                        };
                        match self.read(h, *t) {
                            Ok(t) => kids.push(t),
                            Err(g) => {
                                failed = Some(g);
                                break;
                            }
                        }
                    }
                    match failed {
                        None => return Ok(Tree { symbol: *sym, const_index: None, kids }),
                        Some(g) => {
                            first.get_or_insert(g);
                        }
                    }
                }
            }
        }
        let mut norm = self.norm();
        norm.fresh = 1_000_000 + self.steps.get();
        for alt in self.rewrites(e, dual, &mut norm) {
            match self.read(&alt, ty) {
                Ok(t) => return Ok(t),
                Err(g) => {
                    first.get_or_insert(g);
                }
            }
        }
        Err(first.unwrap_or_else(|| Gap { construct: e.head(), dual: dual.dual.clone(), detail: "no row of this type reads it".into() }))
    }

    /// The exact rewrites tried when no row reads an expression.
    fn rewrites(&self, e: &E, dual: &crate::futhark_kingdom::FutharkDual, norm: &mut Norm<'_>) -> Vec<E> {
        let mut out = Vec::new();
        let class = dual.dual.split(':').next().unwrap_or("");
        let arr_dim = class.strip_prefix("a.").filter(|d| !d.contains('.')).map(str::to_string);
        let is_add = |f: &E| matches!(f, E::Lam(ps, body) if ps.len() == 2 && **body == E::Bin("+".into(), b(var(&ps[0])), b(var(&ps[1]))));
        let named = |f: &E, n: &str| f == &var(n) || f.eta() == Some(n);
        // A size input as an `i32` value is the kingdom's `i32.i64 k`.
        if let E::Var(x) = e {
            if let Some(d) = x.strip_suffix(SIZE_I32) {
                out.push(app("i32.i64", vec![var(d)]));
            }
        }
        // An element of a computed array is the computation at that index (exact: the index is in range for the
        // array, which has the length of the arrays it is computed from): `(map f xs)[i] = f xs[i]`,
        // `(map2 f xs ys)[i] = f xs[i] ys[i]`, `(tabulate d g)[i] = g i`, `(replicate d x)[i] = x`.
        if let E::Index(a, i) = e {
            if let E::App(h, ha) = a.as_ref() {
                if let E::Var(hn) = h.as_ref() {
                    let at = |x: &E| match x {
                        E::App(g, ga) if (**g == var("iota") || **g == var("indices")) && ga.len() == 1 => *i.clone(),
                        _ => E::Index(b(x.clone()), i.clone()),
                    };
                    let r = match (hn.as_str(), ha.as_slice()) {
                        ("map" | "map2" | "map3", [g, arrays @ ..]) if !arrays.is_empty() => norm.apply(g.clone(), arrays.iter().map(at).collect()).ok(),
                        ("tabulate", [_, g]) => norm.apply(g.clone(), vec![*i.clone()]).ok(),
                        ("replicate", [_, x]) => Some(x.clone()),
                        _ => None,
                    };
                    out.extend(r);
                }
            }
        }
        let E::App(f, args) = e else {
            if let E::If(c, t, f) = e {
                if let E::Bin(op, l, r) = c.as_ref() {
                    let lt0 = |x: &E| E::Bin("<".into(), b(x.clone()), b(E::Int(0)));
                    match (op.as_str(), l.as_ref(), r.as_ref()) {
                        (">=", x, E::Int(0)) => out.push(E::If(b(lt0(x)), f.clone(), t.clone())),
                        (">", E::Int(0), x) => out.push(E::If(b(lt0(x)), t.clone(), f.clone())),
                        ("<=", E::Int(0), x) => out.push(E::If(b(lt0(x)), f.clone(), t.clone())),
                        _ => {}
                    }
                }
            }
            return out;
        };
        let E::Var(fname) = f.as_ref() else { return out };
        match (fname.as_str(), args.as_slice()) {
            ("map", [g, _]) | ("map2", [g, _, _]) | ("map3", [g, _, _, _]) => {
                if let Some(d) = &arr_dim {
                    let j = norm.fresh();
                    let arrays = &args[1..];
                    let elems: Vec<E> = arrays
                        .iter()
                        .map(|a| match a {
                            E::App(h, x) if (**h == var("iota") || **h == var("indices")) && x.len() == 1 => var(&j),
                            _ => E::Index(b(a.clone()), b(var(&j))),
                        })
                        .collect();
                    if let Ok(body) = norm.apply(g.clone(), elems) {
                        out.push(app("tabulate", vec![var(d), E::Lam(vec![j], b(body))]));
                    }
                }
            }
            ("reduce_by_index", [E::App(r, ra), op, ne, is, vs]) => {
                if **r == var("replicate") && ra.len() == 2 && &ra[1] == ne {
                    out.push(app("hist", vec![op.clone(), ne.clone(), ra[0].clone(), is.clone(), vs.clone()]));
                }
            }
            ("hist", [op, ne, d, E::App(m, ma), vs]) => {
                if **m == var("map") && ma.len() == 2 && !named(&ma[0], "i64.i32") {
                    let v = norm.fresh();
                    if let Ok(E::App(h, ha)) = norm.apply(ma[0].clone(), vec![var(&v)]) {
                        if *h == var("i64.i32") && ha.len() == 1 {
                            let inner = app("map", vec![E::Lam(vec![v], b(ha[0].clone())), ma[1].clone()]);
                            out.push(app("hist", vec![op.clone(), ne.clone(), d.clone(), app("map", vec![var("i64.i32"), inner]), vs.clone()]));
                        }
                    }
                }
            }
            ("reduce", [op, ne, xs]) => {
                if is_add(op) && *ne == E::Int(0) {
                    out.push(app("i32.sum", vec![xs.clone()]));
                } else if named(op, "i32.min") && *ne == var("i32.highest") {
                    out.push(app("i32.minimum", vec![xs.clone()]));
                } else if named(op, "i32.max") && *ne == var("i32.lowest") {
                    out.push(app("i32.maximum", vec![xs.clone()]));
                }
            }
            ("i64.min", [x, E::App(g, ga)]) => {
                // `min (D-1) (max 0 x)` is `max 0 (min (D-1) x)` for D >= 1: the clamp's other order.
                if **g == var("i64.max") && ga.len() == 2 && ga[0] == E::Int(0) {
                    out.push(app("i64.max", vec![E::Int(0), app("i64.min", vec![x.clone(), ga[1].clone()])]));
                }
            }
            _ => {}
        }
        out
    }
}

/// Whether a pattern node may be matched with its two arguments swapped (an exact commutative operator) and is worth
/// it (one argument is not a bare hole, or the two holes differ in type: a pattern of two holes reads both orders
/// alike).
fn commutative(p: &E) -> Option<(E, E)> {
    match p {
        E::Bin(op, l, r) if (op == "+" || op == "*") && !(matches!(**l, E::Hole(_)) && matches!(**r, E::Hole(_))) => Some((*l.clone(), *r.clone())),
        E::App(f, a) if a.len() == 2 && matches!(f.as_ref(), E::Var(n) if ["i32.min", "i32.max", "i64.min", "i64.max"].contains(&n.as_str())) && !(matches!(a[0], E::Hole(_)) && matches!(a[1], E::Hole(_))) => Some((a[0].clone(), a[1].clone())),
        _ => None,
    }
}

/// Every way a pattern matches an expression, the direct one first: the holes' expressions.
fn pmatch_all(p: &E, e: &E, n: usize) -> Vec<Vec<Option<E>>> {
    let mut out = Vec::new();
    let mut holes = vec![None; n];
    pm(p, e, &mut holes, &mut |h| out.push(h.to_vec()));
    out
}

/// The matcher, by continuation: `k` is called with the holes of every complete match.
fn pm(p: &E, e: &E, holes: &mut Vec<Option<E>>, k: &mut dyn FnMut(&[Option<E>])) {
    pm_seq(&[(p.clone(), e.clone())], holes, k);
}

fn pm_seq(pairs: &[(E, E)], holes: &mut Vec<Option<E>>, k: &mut dyn FnMut(&[Option<E>])) {
    let Some(((p, e), rest)) = pairs.split_first() else {
        k(holes);
        return;
    };
    let then = |extra: Vec<(E, E)>, holes: &mut Vec<Option<E>>, k: &mut dyn FnMut(&[Option<E>])| {
        let mut v = extra;
        v.extend_from_slice(rest);
        pm_seq(&v, holes, k);
    };
    match (p, e) {
        (E::Hole(i), _) => {
            if *i >= holes.len() {
                return;
            }
            match &holes[*i] {
                Some(x) if x != e => {}
                Some(_) => then(vec![], holes, k),
                None => {
                    holes[*i] = Some(e.clone());
                    then(vec![], holes, k);
                    holes[*i] = None;
                }
            }
        }
        (E::Int(a), E::Int(c)) if a == c => then(vec![], holes, k),
        // The kingdom's `i32.i64 k` against a size input's own `i32` parameter.
        (E::App(f, a), E::Var(x)) if **f == var("i32.i64") && a.len() == 1 && matches!(&a[0], E::Var(d) if x.strip_suffix(SIZE_I32) == Some(d.as_str())) => then(vec![], holes, k),
        (E::Var(a), E::Var(c)) if a == c => then(vec![], holes, k),
        // A name against a lambda that only applies it, and the other way round.
        (E::Var(a), E::Lam(..)) if e.eta() == Some(a.as_str()) => then(vec![], holes, k),
        (E::Lam(..), E::Var(c)) if p.eta() == Some(c.as_str()) => then(vec![], holes, k),
        (E::Neg(a), E::Neg(c)) => then(vec![(*a.clone(), *c.clone())], holes, k),
        (E::Index(a, i), E::Index(c, j)) => then(vec![(*a.clone(), *c.clone()), (*i.clone(), *j.clone())], holes, k),
        (E::If(a, t, f), E::If(c, u, g)) => then(vec![(*a.clone(), *c.clone()), (*t.clone(), *u.clone()), (*f.clone(), *g.clone())], holes, k),
        (E::Bin(o1, a, x), E::Bin(o2, c, y)) if o1 == o2 => {
            then(vec![(*a.clone(), *c.clone()), (*x.clone(), *y.clone())], holes, k);
            if commutative(p).is_some() {
                then(vec![(*a.clone(), *y.clone()), (*x.clone(), *c.clone())], holes, k);
            }
        }
        (E::App(f, a), E::App(g, c)) if a.len() == c.len() => {
            let mut v = vec![(*f.clone(), *g.clone())];
            v.extend(a.iter().cloned().zip(c.iter().cloned()));
            then(v, holes, k);
            if let Some((x, y)) = commutative(p) {
                then(vec![(*f.clone(), *g.clone()), (x, c[1].clone()), (y, c[0].clone())], holes, k);
            }
        }
        (E::Lam(ps, body), E::Lam(qs, ebody)) if ps.len() == qs.len() => {
            let mut eb = *ebody.clone();
            for (p, q) in ps.iter().zip(qs) {
                eb = eb.rename(q, p);
            }
            then(vec![(*body.clone(), eb)], holes, k);
        }
        (E::Loop { state, init, var: v, lo, hi, body }, E::Loop { state: s2, init: i2, var: v2, lo: l2, hi: h2, body: b2 }) => {
            let eb = b2.rename(s2, state).rename(v2, v);
            then(vec![(*init.clone(), *i2.clone()), (*lo.clone(), *l2.clone()), (*hi.clone(), *h2.clone()), (*body.clone(), eb)], holes, k);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
