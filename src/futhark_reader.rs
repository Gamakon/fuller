//! THE READER of the signature kingdom (`crate::futhark_sig`): Futhark source text to an expressed tree of a
//! benchmark's kingdom, the inverse of how the kingdom renders a tree ([`crate::futhark_table::FutharkKingdom::render`]
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
//!
//! MASTER 6 ([`Reader6`], plan `docs/PLAN_futhark_kingdom.md` section 7 task F4) reads a master 6 view's program into
//! its tree with nothing normalised: a name is the view's `var@T` with the index word G3's environment gives it, a
//! float literal is read by its value, a `let` is the `let` row, an adapter is read as the plain row (`tests6`).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::futhark_table::{FutharkKind, FutharkKingdom};
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
    /// `loop state = init for var in lo ..< hi do body` (`for var < hi` has `lo` 0); a tuple state
    /// `loop ((a, b): t) = ..` names each component (master 6 only: master 5's parser names one).
    Loop { state: Vec<String>, init: Box<E>, var: String, lo: Box<E>, hi: Box<E>, body: Box<E> },
    /// A float literal (master 6 only), in one spelling: its value as Rust prints an `f64`, then its suffix
    /// (`0f32` and `0.0f32` are `0.0f32`; `1.0` is `1.0`).
    Float(String),
    /// An integer literal with its type's suffix (master 6 only: `0i32`, `-1i64`; master 5 reads the value alone,
    /// [`E::Int`], as does an unsuffixed literal).
    Lit(String),
    /// A tuple `(a, b)` (master 6 only).
    Tuple(Vec<E>),
    /// An expression in parentheses (master 6 only): pass 5 puts every operand but a size in parentheses, so a pair of
    /// literals `((0f32), (0f32))` and the pair's zero `(0f32, 0f32)` are two texts, read as two trees.
    Paren(Box<E>),
    /// A tuple's component `e.0` (master 6 only).
    Proj(Box<E>, usize),
    /// An array literal `[a, b]` (master 6 only: a data type's constant, `[0f32]`).
    Array(Vec<E>),
    /// A master 6 spelling's placeholder `{k.p}`: property `p` of operand `k` as Futhark text (patterns only).
    Prop(usize, String),
}

/// A float literal's one spelling ([`E::Float`]): its value as Rust prints an `f64`, then its suffix; `None` for a text
/// that is no float literal.
fn float_literal(text: &str) -> Option<String> {
    let t: String = text.chars().filter(|c| *c != '_').collect();
    let (num, suffix) = match t.len().checked_sub(3).map(|i| t.split_at(i)) {
        Some((n, s)) if ["f16", "f32", "f64"].contains(&s) => (n.to_string(), s.to_string()),
        _ => (t.clone(), String::new()),
    };
    let v: f64 = num.parse().ok()?;
    Some(format!("{v:?}{suffix}"))
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
            E::Float(v) => format!("float literal {v}"),
            E::Lit(v) => format!("literal {v}"),
            E::Paren(e) => e.head(),
            E::Tuple(_) => "tuple".into(),
            E::Proj(..) => "tuple component".into(),
            E::Array(_) => "array literal".into(),
            E::Prop(i, p) => format!("placeholder {{{i}.{p}}}"),
        }
    }

    fn mentions(&self, x: &str) -> bool {
        match self {
            E::Var(y) => y == x,
            E::Int(_) | E::Hole(_) | E::Float(_) | E::Lit(_) | E::Prop(..) => false,
            E::App(f, a) => f.mentions(x) || a.iter().any(|e| e.mentions(x)),
            E::Tuple(a) | E::Array(a) => a.iter().any(|e| e.mentions(x)),
            E::Proj(e, _) | E::Paren(e) => e.mentions(x),
            E::Bin(_, l, r) | E::Index(l, r) => l.mentions(x) || r.mentions(x),
            E::Neg(e) => e.mentions(x),
            E::If(c, t, e) => c.mentions(x) || t.mentions(x) || e.mentions(x),
            E::Lam(ps, body) => !ps.iter().any(|p| p == x) && body.mentions(x),
            E::Let(y, e, body) => e.mentions(x) || (y != x && body.mentions(x)),
            E::Loop { state, init, var, lo, hi, body } => init.mentions(x) || lo.mentions(x) || hi.mentions(x) || (!state.iter().any(|s| s == x) && var != x && body.mentions(x)),
        }
    }

    /// Rename the free `from` to `to` (the names of a normalised expression are unique, so nothing is captured).
    fn rename(&self, from: &str, to: &str) -> E {
        let r = |e: &E| e.rename(from, to);
        match self {
            E::Var(y) if y == from => var(to),
            E::Int(_) | E::Hole(_) | E::Var(_) | E::Float(_) | E::Lit(_) | E::Prop(..) => self.clone(),
            E::App(f, a) => E::App(b(r(f)), a.iter().map(r).collect()),
            E::Tuple(a) => E::Tuple(a.iter().map(r).collect()),
            E::Array(a) => E::Array(a.iter().map(r).collect()),
            E::Proj(e, i) => E::Proj(b(r(e)), *i),
            E::Paren(e) => E::Paren(b(r(e))),
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
                body: if state.iter().any(|s| s == from) || v == from { body.clone() } else { b(r(body)) },
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
fn canonical_loop(state: Vec<String>, init: E, v: String, lo: E, hi: E, body: E) -> E {
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
    /// A master 6 spelling's placeholder of a property or a type of an operand: `{0.bound}`, `{1.type}`.
    Place(usize, String),
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
        K::Int(_) | K::Float | K::Hole(_) | K::Place(..) | K::Str => true,
        K::Ident => !KEYWORDS.contains(&t.text.as_str()),
        K::Sym => t.text == ")" || t.text == "]",
    }
}

/// The end of a master 6 binder placeholder `{k.vj}` starting at `i` (the j-th variable argument k binds), if one does.
fn binder_place(c: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    if c.get(j) != Some(&'{') {
        return None;
    }
    j += 1;
    let d0 = j;
    while c.get(j).is_some_and(char::is_ascii_digit) {
        j += 1;
    }
    if j == d0 || c.get(j) != Some(&'.') || c.get(j + 1) != Some(&'v') {
        return None;
    }
    j += 2;
    let d1 = j;
    while c.get(j).is_some_and(char::is_ascii_digit) {
        j += 1;
    }
    (j > d1 && c.get(j) == Some(&'}')).then_some(j + 1)
}

/// Lex a text: `holes` reads `{k}` as a pattern's hole; `rich` is master 6's spelling and program syntax (a binder
/// placeholder `{k.vj}` is a name, glued to the name before it: `t{2.v1}`; `{k.p}` is a placeholder of a property or
/// type).
fn lex(src: &str, holes: bool, rich: bool) -> Result<Vec<Tok>, Gap> {
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
            if holes && rich {
                if let Some(end) = binder_place(&c, i) {
                    i = end;
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
        if holes && rich && ch == '{' {
            if let Some(end) = binder_place(&c, i) {
                out.push(Tok { k: K::Ident, text: c[i..end].iter().collect(), tight, line_start: first });
                i = end;
                tight = true;
                continue;
            }
            let mut j = i + 1;
            while j < c.len() && c[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && c.get(j) == Some(&'.') {
                let mut e = j + 1;
                while e < c.len() && (c[e].is_alphanumeric() || c[e] == '_') {
                    e += 1;
                }
                if e > j + 1 && c.get(e) == Some(&'}') {
                    let n: usize = c[i + 1..j].iter().collect::<String>().parse().map_err(|_| Gap::parse("placeholder", "a placeholder number".into()))?;
                    out.push(Tok { k: K::Place(n, c[j + 1..e].iter().collect()), text: c[i..=e].iter().collect(), tight, line_start: first });
                    i = e + 1;
                    tight = true;
                    continue;
                }
            }
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
    /// Master 6's syntax: float literals, tuples and their components, array literals, a tuple loop state, a size
    /// binding `let [m] (x: t) = ..`, and a spelling's placeholders.
    rich: bool,
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
        if self.rich {
            if let Some(Tok { k: K::Place(_, f), .. }) = self.peek() {
                if f == "type" {
                    self.p += 1;
                    return Ok(FType { dims: Vec::new(), elem: "placeholder".into() });
                }
            }
        }
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
        // `let [m] (x: [m]t) = ..`: a size binding (master 6's adapters); the size's name is only a type's.
        if self.rich && self.is("[") {
            self.p += 1;
            self.ident()?;
            self.expect("]")?;
        }
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
        let state = if self.rich && self.is("(") && self.peek_at(1).is_some_and(|t| t.text == "(") {
            self.p += 2;
            let mut names = vec![self.ident()?];
            while self.eat(",") {
                names.push(self.ident()?);
            }
            self.expect(")")?;
            if self.eat(":") {
                self.ftype()?;
            }
            self.expect(")")?;
            names
        } else {
            vec![self.binder()?.0]
        };
        let init = match (self.eat("="), state.as_slice()) {
            (true, _) => self.expr()?,
            (false, [one]) => E::Var(one.clone()),
            (false, _) => return Err(Gap::parse("tuple loop state with no initial value", String::new())),
        };
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
                K::Int(_) | K::Hole(_) | K::Float | K::Place(..) => true,
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
        while self.rich && self.peek().is_some_and(|t| t.text == "." && t.tight) {
            match self.peek_at(1).map(|t| t.k.clone()) {
                Some(K::Int(i)) if i >= 0 => {
                    self.p += 2;
                    e = E::Proj(b(e), i as usize);
                }
                _ => return Err(Gap::parse("field access", String::new())),
            }
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
                let suffix = t.text.rfind(['i', 'u']).map(|at| &t.text[at..]).filter(|x| x.len() > 1 && x[1..].bytes().all(|c| c.is_ascii_digit())).unwrap_or("");
                Ok(if self.rich && !suffix.is_empty() { E::Lit(format!("{v}{suffix}")) } else { E::Int(v) })
            }
            K::Hole(i) => {
                self.p += 1;
                Ok(E::Hole(i))
            }
            K::Float if self.rich => {
                self.p += 1;
                float_literal(&t.text).map(E::Float).ok_or_else(|| Gap::parse("literal", t.text.clone()))
            }
            K::Float => Err(Gap::parse("float literal", t.text)),
            K::Place(i, ref f) if self.rich && f != "type" => {
                self.p += 1;
                Ok(E::Prop(i, f.clone()))
            }
            K::Place(..) => Err(Gap::parse("placeholder", t.text)),
            K::Str => Err(Gap::parse("string", t.text)),
            K::Ident => {
                let n = self.ident()?;
                Ok(E::Var(n))
            }
            K::Sym if t.text == "[" && self.rich => {
                self.p += 1;
                let mut items = Vec::new();
                if !self.is("]") {
                    items.push(self.expr()?);
                    while self.eat(",") {
                        items.push(self.expr()?);
                    }
                }
                self.expect("]")?;
                Ok(E::Array(items))
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
                if self.rich && self.is(",") {
                    let mut items = vec![e];
                    while self.eat(",") {
                        items.push(self.expr()?);
                    }
                    self.expect(")")?;
                    return Ok(E::Tuple(items));
                }
                if self.is(",") {
                    return Err(Gap::parse("tuple", String::new()));
                }
                if self.eat(":") {
                    self.ftype()?;
                }
                let e = if self.rich { E::Paren(b(e)) } else { e };
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
    parse_expression(text, true, false)
}

/// The first fresh name's number of a parse: a master 6 program's names a section or `_` introduces start far past a
/// spelling's, so a pattern's `\%s1 %s2 -> ..` never meets a program's `\%s2 %s3 -> ..` (the matcher renames one
/// binder at a time, and a program binder named as another pattern binder would read as a capture). Master 5's
/// normaliser renames every binder apart, so its parses keep 0.
fn program_fresh(holes: bool, rich: bool) -> usize {
    if rich && !holes {
        1_000_000
    } else {
        0
    }
}

/// Parse an expression: `holes` reads a spelling's holes, `rich` master 6's syntax.
fn parse_expression(text: &str, holes: bool, rich: bool) -> Result<E, Gap> {
    let toks = lex(text, holes, rich)?;
    let mut p = Parser { t: &toks, p: 0, fresh: program_fresh(holes, rich), rich };
    let e = p.expr()?;
    if p.p != toks.len() {
        return Err(p.unexpected("text after the expression"));
    }
    Ok(e)
}

/// Parse a Futhark source file into its top-level `def`s and `entry`s (other items, `import`, `type`, `module`,
/// `open`, are skipped; an item that does not parse is a `Def` whose body is the gap).
pub fn parse_file(src: &str) -> Result<Vec<Def>, Gap> {
    parse_file_with(src, false)
}

/// [`parse_file`], in master 6's syntax when `rich`.
fn parse_file_with(src: &str, rich: bool) -> Result<Vec<Def>, Gap> {
    let toks = lex(src, false, rich)?;
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
        let mut p = Parser { t: &toks[s..end], p: 0, fresh: program_fresh(false, rich), rich };
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
    /// `replicate D 0` anywhere is the kingdom's `zeros_D` ([`Reader::for_pool`]).
    zeros_everywhere: bool,
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
            E::Int(_) | E::Hole(_) | E::Float(_) | E::Lit(_) | E::Prop(..) => e.clone(),
            E::Tuple(a) => E::Tuple(a.iter().map(|x| self.ev(x, env)).collect::<Result<_, _>>()?),
            E::Array(a) => E::Array(a.iter().map(|x| self.ev(x, env)).collect::<Result<_, _>>()?),
            E::Proj(x, i) => E::Proj(b(self.ev(x, env)?), *i),
            E::Paren(x) => self.ev(x, env)?,
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
                let mut env2 = env.clone();
                let mut s2 = Vec::with_capacity(state.len());
                for x in state {
                    let f = self.fresh();
                    env2.insert(x.clone(), E::Var(f.clone()));
                    s2.push(f);
                }
                let v2 = self.fresh();
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
                    "replicate" if self.zeros_everywhere && args.len() == 2 && args[1] == E::Int(0) && self.is_dim(&args[0]).is_some() => {
                        return Ok(var(&format!("zeros_{}", self.is_dim(&args[0]).unwrap_or_default())));
                    }
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
        E::Tuple(a) => E::Tuple(a.iter().map(s).collect()),
        E::Array(a) => E::Array(a.iter().map(s).collect()),
        E::Proj(x, i) => E::Proj(b(s(&x)), i),
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
    /// Read every `replicate D 0` as the kingdom's `zeros_D` (the same value): set by [`Reader::for_pool`].
    zeros_everywhere: bool,
}

/// The most pattern matches one read may try (a guard: reading a program always ends).
const READ_STEPS: usize = 5_000_000;
/// The most rows [`Reader6::read_tree`] rejects at their nodes before a text is a gap.
const READ_REJECTED: usize = 64;

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
        Ok(Reader { k, sig, by_ty, consts, input_dims, steps: std::cell::Cell::new(0), zeros_everywhere: false })
    }

    /// The reader of the pool's programs (`examples/futhark_pool.rs`): as [`Reader::new`], and every `replicate D 0`
    /// is read as the kingdom's `zeros_D`, the same array, so the rings hold the zero arrays programs build (the zero
    /// array is the only terminal of a computed array's type: measured 2026-10-28, without this no topic program put
    /// a terminal in the `a.k` rings of histogram). The round trip uses [`Reader::new`], which keeps an inline
    /// `replicate D 0` as `replicate`.
    pub fn for_pool(k: &'a FutharkKingdom, sig: &'a Signature) -> Result<Reader<'a>, String> {
        Ok(Reader { zeros_everywhere: true, ..Reader::new(k, sig)? })
    }

    fn norm(&self) -> Norm<'_> {
        Norm { dims: &self.sig.dims, input_dims: &self.input_dims, fresh: 0, steps: 0, zeros_everywhere: self.zeros_everywhere }
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
    fn rewrites(&self, e: &E, dual: &crate::futhark_table::FutharkDual, norm: &mut Norm<'_>) -> Vec<E> {
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
        // A binder's variable is named by its class (`futhark_view::var_name`), so the pattern's name may be one an
        // outer binder already gave: renaming the source's binder to it would capture an outer variable the body
        // reads (the body reads an outer variable through a same-class inner binder). No row can say that, so the
        // match fails and the reader reports the gap.
        (E::Lam(ps, _), E::Lam(qs, ebody)) if ps.len() == qs.len() && ps.iter().zip(qs).any(|(p, q)| p != q && ebody.mentions(p)) => {}
        (E::Loop { state, var: v, .. }, E::Loop { state: s2, var: v2, body: b2, .. }) if state.len() != s2.len() || state.iter().zip(s2).any(|(p, q)| p != q && b2.mentions(p)) || (v != v2 && b2.mentions(v)) => {}
        (E::Let(x, _, _), E::Let(y, _, ebody)) if x != y && ebody.mentions(x) => {}
        (E::Let(x, v, body), E::Let(y, w, ebody)) => then(vec![(*v.clone(), *w.clone()), (*body.clone(), ebody.rename(y, x))], holes, k),
        (E::Float(a), E::Float(c)) | (E::Lit(a), E::Lit(c)) if a == c => then(vec![], holes, k),
        (E::Proj(a, i), E::Proj(c, j)) if i == j => then(vec![(*a.clone(), *c.clone())], holes, k),
        (E::Paren(a), E::Paren(c)) => then(vec![(*a.clone(), *c.clone())], holes, k),
        (E::Tuple(a), E::Tuple(c)) | (E::Array(a), E::Array(c)) if a.len() == c.len() => then(a.iter().cloned().zip(c.iter().cloned()).collect(), holes, k),
        (E::Lam(ps, body), E::Lam(qs, ebody)) if ps.len() == qs.len() => {
            let mut eb = *ebody.clone();
            for (p, q) in ps.iter().zip(qs) {
                eb = eb.rename(q, p);
            }
            then(vec![(*body.clone(), eb)], holes, k);
        }
        (E::Loop { state, init, var: v, lo, hi, body }, E::Loop { state: s2, init: i2, var: v2, lo: l2, hi: h2, body: b2 }) => {
            let eb = state.iter().zip(s2).fold(*b2.clone(), |e, (p, q)| e.rename(q, p)).rename(v2, v);
            then(vec![(*init.clone(), *i2.clone()), (*lo.clone(), *l2.clone()), (*hi.clone(), *h2.clone()), (*body.clone(), eb)], holes, k);
        }
        _ => {}
    }
}

// ---- master 6: the reader (plan section 7, task F4) -----------------------------------------------------------------
//
// Futhark text to a tree of a master 6 view (`kingdoms/futhark/sig/<problem>/m6/`), the inverse of pass 5
// ([`Kingdom6::program6`]). A slot of data type `T` is read, in this order, as: a name in scope of data type `T` (the
// view's `var@T`, its index word set below); a literal of `T`'s pool (`const@T`, the literal's position in the data
// type's pool); the first row of `T`
// whose spelling, parsed as a pattern, matches (each `{k}` operand read at its argument's data type, each `{k.vj}` a
// variable that argument binds, each `{k.p}` checked against operand k's property once the tree is decoded); and, where
// no variable of `T` is in scope, `T`'s constant as `var@T` (what such a cell renders). Nothing is normalised: a `let`
// is the `let` row, a float literal is read by its value, a tuple and its components are rows.
//
// THE INDEX WORDS are never computed here: the tree is decoded (G3, `typed_environment`) and each variable leaf's word
// is the one [`crate::typed_environment::Environment::index_of`] gives for its binder in the leaf's environment (the
// least word: a gene's word is read modulo the variables in scope, so the tree read is the CANONICAL tree), and the
// tree is decoded again to check every leaf names its variable.
//
// ADAPTERS are read as the plain row, not as rows of their own: no row of the gene is an adapter, pass 5 renders one
// wherever a requirement fails, so the plain row is the only tree that can render it. `fit l z (e)` and
// `map (fit l z) (e)` are read as `e`, `clampix l (e)` as `e`, and the adapters' `let ad_* = ..` bound names as their
// values. The tree read decodes with the same failing requirement and renders the same adapter again
// (`render(read(x)) = x`, tested on random genes of every view). A text whose adapters the tree read does not render
// (another length, an adapter where the requirement holds, none where it fails) is a program no gene expresses, with
// other values: each adapter is recorded with its kind, length and node, the tree read is rendered and read again, and
// a difference is a gap, never a silent change of the program. A substituted node (the
// work bound) renders as its substitute and is read as it; a substitution by the constant where a variable of its data
// type is in scope is read as the first row the decode substitutes there.
//
// THE UNSPELLED OPERAND (master 6 README 9: `tabrows` never spells its `cols` size, so the text does not hold its
// index word): where the body is fitted (`fit d 0f32 (body)`, its `fit.body` failed), `cols` is the fit's length; where
// it is not, the requirement `in2.len = in1.value` held, so `cols` is the FIRST size in scope (the least index word)
// whose value is the body's length.
//
// Parentheses are kept ([`E::Paren`]): pass 5 puts every operand but a size in them, so they tell a pair of literals
// `((0f32), (0f32))` from the pair's zero `(0f32, 0f32)`.

use crate::futhark_table::Kingdom6;
use crate::geneframe::{FlowName, Owner, Place, PropertyExpr, Role};
use crate::kingdom_contract::SymbolId;
use crate::typed_environment::template::{parse_template, substitute_of};
use crate::typed_environment::{Binder, Choice, Decoding, Entry, Node, Substitution};

impl E {
    /// This expression with `f` applied to each of its direct sub-expressions (binders kept).
    fn map_kids(&self, f: &mut dyn FnMut(&E) -> E) -> E {
        let mut all = |xs: &[E]| -> Vec<E> {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(f(x));
            }
            out
        };
        match self {
            E::Int(_) | E::Var(_) | E::Hole(_) | E::Float(_) | E::Lit(_) | E::Prop(..) => self.clone(),
            E::App(g, a) => {
                let mut v = all(std::slice::from_ref(g.as_ref()));
                v.extend(all(a));
                let g = v.remove(0);
                E::App(b(g), v)
            }
            E::Bin(op, l, r) => {
                let mut v = all(&[*l.clone(), *r.clone()]);
                let r = v.pop().unwrap_or(E::Int(0));
                let l = v.pop().unwrap_or(E::Int(0));
                E::Bin(op.clone(), b(l), b(r))
            }
            E::Neg(x) => E::Neg(b(all(std::slice::from_ref(x.as_ref())).remove(0))),
            E::Index(a, i) => {
                let mut v = all(&[*a.clone(), *i.clone()]);
                let i = v.pop().unwrap_or(E::Int(0));
                let a = v.pop().unwrap_or(E::Int(0));
                E::Index(b(a), b(i))
            }
            E::If(c, t, e) => {
                let mut v = all(&[*c.clone(), *t.clone(), *e.clone()]);
                let e = v.pop().unwrap_or(E::Int(0));
                let t = v.pop().unwrap_or(E::Int(0));
                let c = v.pop().unwrap_or(E::Int(0));
                E::If(b(c), b(t), b(e))
            }
            E::Lam(ps, body) => E::Lam(ps.clone(), b(all(std::slice::from_ref(body.as_ref())).remove(0))),
            E::Let(x, v, body) => {
                let mut w = all(&[*v.clone(), *body.clone()]);
                let body = w.pop().unwrap_or(E::Int(0));
                let v = w.pop().unwrap_or(E::Int(0));
                E::Let(x.clone(), b(v), b(body))
            }
            E::Loop { state, init, var: v, lo, hi, body } => {
                let mut w = all(&[*init.clone(), *lo.clone(), *hi.clone(), *body.clone()]);
                let body = w.pop().unwrap_or(E::Int(0));
                let hi = w.pop().unwrap_or(E::Int(0));
                let lo = w.pop().unwrap_or(E::Int(0));
                let init = w.pop().unwrap_or(E::Int(0));
                E::Loop { state: state.clone(), init: b(init), var: v.clone(), lo: b(lo), hi: b(hi), body: b(body) }
            }
            E::Tuple(a) => E::Tuple(all(a)),
            E::Array(a) => E::Array(all(a)),
            E::Proj(x, i) => E::Proj(b(all(std::slice::from_ref(x.as_ref())).remove(0)), *i),
            E::Paren(x) => E::Paren(b(all(std::slice::from_ref(x.as_ref())).remove(0))),
        }
    }

    /// The free name `x` replaced by `v` (a binder of `x` shadows it; the names of a rendered program are distinct in
    /// every scope, so nothing in `v` is captured).
    fn subst(&self, x: &str, v: &E) -> E {
        match self {
            E::Var(y) if y == x => v.clone(),
            E::Lam(ps, _) if ps.iter().any(|p| p == x) => self.clone(),
            E::Let(y, r, body) if y == x => E::Let(y.clone(), b(r.subst(x, v)), body.clone()),
            E::Loop { state, init, var: w, lo, hi, body } if w == x || state.iter().any(|s| s == x) => {
                E::Loop { state: state.clone(), init: b(init.subst(x, v)), var: w.clone(), lo: b(lo.subst(x, v)), hi: b(hi.subst(x, v)), body: body.clone() }
            }
            _ => self.map_kids(&mut |k| k.subst(x, v)),
        }
    }

    /// Each placeholder `{k.p}` replaced by a hole past the `n` operands, its (k, p) appended to `props`.
    fn props_to_holes(&self, n: usize, props: &mut Vec<(usize, String)>) -> E {
        match self {
            E::Prop(k, p) => {
                let i = props.iter().position(|q| q.0 == *k && q.1 == *p).unwrap_or_else(|| {
                    props.push((*k, p.clone()));
                    props.len() - 1
                });
                E::Hole(n + i)
            }
            _ => self.map_kids(&mut |e| e.props_to_holes(n, props)),
        }
    }
}

/// The mark a `fit` leaves where [`strip_adapters`] takes it away: `%fit l e`, the length `l` it fitted `e` to (an
/// operand's text no row spells, `tabrows`' `cols`, is read from it).
const FIT_MARK: &str = "%fit";
/// The mark of `map (fit l z)`, a fit of the inner dimension.
const FIT_COLS_MARK: &str = "%fitcols";
/// The mark of `clampix l`, an index clamped to `l`.
const CLAMP_MARK: &str = "%clampix";

/// THE ADAPTERS READ AS THE PLAIN ROW: `fit l z (e)`, `fitrows l z (e)`, `map (fit l z) (e)` and `clampix l (e)` are
/// `e` marked with the adapter's kind and length ([`FIT_MARK`], [`FIT_COLS_MARK`], [`CLAMP_MARK`]; the reader reads the
/// operand under the marks and records them), and an adapter's `let ad_* = v` (or the size binding `let [ad_m] (ad_s:
/// ..) = v`) is its body with the name's value in place. Each adapter application is numbered in pre-order (its mark is
/// `%fit#<i>`): the applications numbered in `keep` are left as they are, to be read as the ADAPTER SYMBOLS evolution
/// placed ([`Reader6::read_body`] keeps those whose mark the tree read does not render).
fn strip_adapters(e: &E, keep: &BTreeSet<usize>) -> E {
    strip_counted(e, keep).0
}

/// [`strip_adapters`] and the number of adapter applications the expression holds.
fn strip_counted(e: &E, keep: &BTreeSet<usize>) -> (E, usize) {
    fn go(e: &E, keep: &BTreeSet<usize>, n: &mut usize) -> E {
        let is = |f: &E, name: &str| matches!(f, E::Var(x) if x == name);
        let fit_cols = |a: &[E]| a.len() == 2 && matches!(unmark(&a[0]).0, E::App(g, ga) if is(g, "fit") && ga.len() == 2);
        let adapter = match e {
            E::App(f, a) => ((is(f, "fit") || is(f, "fitrows")) && a.len() == 3) || (is(f, "clampix") && a.len() == 2) || (is(f, "map") && fit_cols(a)),
            _ => false,
        };
        if adapter {
            let i = *n;
            *n += 1;
            if keep.contains(&i) {
                return e.map_kids(&mut |k| go(k, keep, n));
            }
            let mark = |m: &str| format!("{m}#{i}");
            return match e {
                E::App(f, a) if is(f, "clampix") => app(&mark(CLAMP_MARK), vec![a[0].clone(), go(&a[1], keep, n)]),
                E::App(f, a) if is(f, "map") => match unmark(&a[0]).0 {
                    E::App(_, ga) => app(&mark(FIT_COLS_MARK), vec![ga[0].clone(), go(&a[1], keep, n)]),
                    _ => go(&a[1], keep, n),
                },
                E::App(_, a) => app(&mark(FIT_MARK), vec![a[0].clone(), go(&a[2], keep, n)]),
                _ => e.clone(),
            };
        }
        match e {
            E::Let(x, v, body) if x.starts_with("ad_") => go(&body.subst(x, v), keep, n),
            _ => e.map_kids(&mut |k| go(k, keep, n)),
        }
    }
    let mut n = 0;
    let out = go(e, keep, &mut n);
    (out, n)
}

/// A mark's kind and its application's number, from its name (`%fit#3`).
fn mark_of(name: &str) -> Option<(&'static str, usize)> {
    let (kind, i) = name.split_once('#')?;
    [FIT_MARK, FIT_COLS_MARK, CLAMP_MARK].into_iter().find(|m| *m == kind).zip(i.parse().ok())
}

/// An expression without the parentheses and adapter marks around it, and the length of the outermost fit.
fn unmark(e: &E) -> (&E, Option<&E>) {
    let (e, marks) = unmark_all(e);
    (e, marks.into_iter().find(|m| m.0 == FIT_MARK).map(|m| m.1))
}

/// An expression without the parentheses and adapter marks around it, and every mark taken away, outermost first:
/// its kind and its length.
fn unmark_all(e: &E) -> (&E, Vec<(&'static str, &E)>) {
    let (inner, marks) = unmark_numbered(e);
    (inner, marks.into_iter().map(|(k, l, _)| (k, l)).collect())
}

/// [`unmark_all`] with each mark's application number.
fn unmark_numbered(e: &E) -> (&E, Vec<(&'static str, &E, usize)>) {
    let mut e = e;
    let mut marks = Vec::new();
    loop {
        if let E::Paren(x) = e {
            e = x;
            continue;
        }
        let E::App(f, a) = e else { break };
        let E::Var(mark) = f.as_ref() else { break };
        let Some((kind, i)) = mark_of(mark).filter(|_| a.len() == 2) else { break };
        marks.push((kind, &a[0], i));
        e = &a[1];
    }
    (e, marks)
}

/// An expression without its parentheses.
fn unparen(e: &E) -> E {
    match e {
        E::Paren(x) => unparen(x),
        _ => e.map_kids(&mut unparen),
    }
}

/// A property as the expression pass 5 spells it (`futhark_table`'s size text): a size name, a literal, a `min` as
/// nested `i64.min`, a sum with `+`; `None` for `?` or an input's property.
fn property_expr(p: &PropertyExpr) -> Option<E> {
    match p {
        PropertyExpr::Size(n) => Some(var(n)),
        PropertyExpr::Literal(v) => Some(E::Int(*v)),
        PropertyExpr::Min(args) => {
            let es: Vec<E> = args.iter().map(property_expr).collect::<Option<_>>()?;
            let (last, rest) = es.split_last()?;
            Some(rest.iter().rev().fold(last.clone(), |acc, a| app("i64.min", vec![a.clone(), acc])))
        }
        PropertyExpr::Sum(terms) => {
            let es: Vec<E> = terms.iter().map(property_expr).collect::<Option<_>>()?;
            let (first, rest) = es.split_first()?;
            Some(rest.iter().fold(first.clone(), |acc, t| E::Bin("+".into(), b(acc), b(t.clone()))))
        }
        _ => None,
    }
}

/// One row's spelling as a pattern: its operands' data types, the variables each argument binds (Binding row order),
/// and the property each hole past the operands reads.
#[derive(Debug, Clone)]
struct Pattern6 {
    symbol: SymbolId,
    pattern: E,
    inputs: Vec<Ty>,
    binds: BTreeMap<u32, Vec<Ty>>,
    props: Vec<(usize, String)>,
    /// Whether the spelling holds operand k's hole (`tabrows` never spells its `cols`).
    spelled: Vec<bool>,
}

/// Whether an expression holds the hole `i`.
fn has_hole(e: &E, i: usize) -> bool {
    let mut found = matches!(e, E::Hole(h) if *h == i);
    e.map_kids(&mut |k| {
        found |= has_hole(k, i);
        k.clone()
    });
    found
}

/// Why a decoding has no canonical tree ([`Reader6::canonical`]): the first two by design (a terminal of the same
/// spelling, or the first substitute candidate, reads first at such a node), the third a bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotCanonical {
    SubstitutedByConstant,
    VariableAsConstant,
    Bug(String),
}

/// Where a name read in a program is bound: a root variable, or the `ordinal`-th variable the argument at path `at`
/// binds.
#[derive(Debug, Clone)]
enum Origin6 {
    Root(String),
    Bound { at: Vec<u32>, ordinal: usize },
}

/// A name in scope while reading: the program's name, its data type and where it is bound.
#[derive(Debug, Clone)]
struct Named6 {
    name: String,
    ty: Ty,
    origin: Origin6,
}

/// An operand no row spells (`tabrows`' `cols`), read from the requirement that ties it to a spelled operand: the
/// node's path, the operand, and the requirement's other side (operand `j`'s property `pj`) against the operand's
/// property `pi`.
#[derive(Debug, Clone)]
struct Unspelled6 {
    node: Vec<u32>,
    operand: usize,
    j: usize,
    pj: String,
    pi: String,
}

/// What a read records for the decode that follows it: each variable leaf's path and origin, each property placeholder
/// (node path, operand, property, the expression read), each leaf read as its data type's constant, each node read as
/// a work-bound substitution by its constant, and each unspelled operand.
#[derive(Debug, Default)]
struct Read6 {
    leaves: Vec<(Vec<u32>, Origin6)>,
    props: Vec<(Vec<u32>, usize, String, E)>,
    constants: Vec<Vec<u32>>,
    substituted: Vec<Vec<u32>>,
    unspelled: Vec<Unspelled6>,
    /// Each adapter the text has: the path of the operand it adapts, its mark ([`FIT_MARK`], [`FIT_COLS_MARK`],
    /// [`CLAMP_MARK`]) and its length, outermost first.
    adapters: Adapters6,
    /// The gap of the deepest expression no row read: what a failed read reports (the first gap is often a row whose
    /// pattern matches everything of its head, `map2 {0} {1} {2}`, failing at its first operand).
    deepest: Option<(usize, Gap)>,
    /// The rows not to try at a node (path, symbol): each one read there before, whose property placeholder the
    /// decode then contradicted ([`Reader6::read_tree`]).
    rejected: BTreeSet<(Vec<u32>, SymbolId)>,
}

type Mark6 = [usize; 6];
/// A row read at a node (path, symbol) whose property placeholder the decode contradicted, with its gap.
type Rejected6 = ((Vec<u32>, SymbolId), Gap);

/// The marks a tree read does not render (each its depth and its number), with the gap.
type Unrendered6 = (Vec<(usize, usize)>, Gap);

/// The adapters a text has: each the path of the operand it adapts, its mark and its length, outermost first.
type Adapters6 = Vec<(Vec<u32>, String, E, usize)>;

impl Read6 {
    fn mark(&self) -> Mark6 {
        [self.leaves.len(), self.props.len(), self.constants.len(), self.substituted.len(), self.unspelled.len(), self.adapters.len()]
    }
    fn undo(&mut self, m: Mark6) {
        self.leaves.truncate(m[0]);
        self.props.truncate(m[1]);
        self.constants.truncate(m[2]);
        self.substituted.truncate(m[3]);
        self.unspelled.truncate(m[4]);
        self.adapters.truncate(m[5]);
    }
}

/// THE READER OF A MASTER 6 VIEW: its rows' spellings as patterns, its constants and literal pools, and its kingdom
/// (for the decode that sets the index words).
pub struct Reader6<'a> {
    pub k: &'a Kingdom6,
    entry: Entry,
    rows: BTreeMap<Ty, Vec<Pattern6>>,
    consts: BTreeMap<Ty, SymbolId>,
    vars: BTreeMap<Ty, SymbolId>,
    constants: BTreeMap<Ty, E>,
    steps: std::cell::Cell<usize>,
}

fn bug6(k: &Kingdom6, what: String) -> String {
    format!("our software has a bug: {}: {what}", k.kingdom)
}

impl<'a> Reader6<'a> {
    /// The reader of kingdom `k`. A spelling or a constant that does not parse is a bug of the master.
    pub fn new(k: &'a Kingdom6) -> Result<Reader6<'a>, String> {
        let mut rows: BTreeMap<Ty, Vec<Pattern6>> = BTreeMap::new();
        let (mut consts, mut vars) = (BTreeMap::new(), BTreeMap::new());
        for row in k.symbols() {
            let symbol = row.symbol as SymbolId;
            let (inputs, output) = (k.inputs(symbol).to_vec(), k.output(symbol));
            match row.role {
                Role::Variable => {
                    vars.insert(output, symbol);
                }
                Role::Constant => {
                    consts.insert(output, symbol);
                }
                Role::Function | Role::Formula => {
                    let spelling = pattern6(k, symbol, row.spelling("").unwrap_or_default());
                    let parsed = parse_expression(&spelling, true, true).map_err(|g| bug6(k, format!("the spelling of {} does not parse as a pattern: {g}", row.symbol_name)))?;
                    let mut props = Vec::new();
                    let pattern = parsed.props_to_holes(inputs.len(), &mut props);
                    let mut binds: BTreeMap<u32, Vec<Ty>> = BTreeMap::new();
                    for f in k.tables.flows.iter().filter(|f| f.owner == Owner::Symbol(row.symbol)) {
                        if let (FlowName::Var { ty, .. }, Place::In(pos)) = (&f.name, f.at) {
                            binds.entry(pos).or_default().push(*ty);
                        }
                    }
                    let spelled = (0..inputs.len()).map(|i| has_hole(&pattern, i)).collect();
                    rows.entry(output).or_default().push(Pattern6 { symbol, pattern, inputs, binds, props, spelled });
                }
            }
        }
        // A row of no input reads first: a text its spelling is (`(0f32, 0f32, 0f32)`, the triple's zero) is that leaf,
        // not a function of leaves that spells the same.
        for ps in rows.values_mut() {
            ps.sort_by_key(|p| !p.inputs.is_empty());
        }
        let mut constants = BTreeMap::new();
        for d in &k.tables.data_types {
            if let Some(c) = &d.constant {
                let e = parse_expression(&c.spelling, false, true).map_err(|g| bug6(k, format!("the constant {:?} of {} does not parse: {g}", c.spelling, d.name)))?;
                constants.insert(d.id, e);
            }
        }
        Ok(Reader6 { k, entry: k.entry.clone(), rows, consts, vars, constants, steps: std::cell::Cell::new(0) })
    }

    fn type_name(&self, ty: Ty) -> String {
        self.k.data_type(ty).map_or_else(|| format!("{ty:?}"), |d| d.name.clone())
    }

    /// The root environment's names, in its order.
    fn root_scope(&self) -> Vec<Named6> {
        self.k.entry.parameters.iter().map(|v| Named6 { name: v.name.clone(), ty: v.data_type, origin: Origin6::Root(v.name.clone()) }).collect()
    }

    /// Whether `name` is an entry parameter the signature declares of `kind` (`input`, `zeros`).
    fn root_of_kind(&self, name: &str, kind: &str) -> bool {
        self.k.spec.env.iter().any(|l| l.kind == kind && l.name == name) && self.k.entry.parameters.iter().any(|p| p.name == name)
    }

    /// A WHOLE PROGRAM of this view (its entry `main`, or its only entry): the size inputs' `i64` rebinding and the
    /// zero arrays' bindings, which the entry declares (pass 5's header), are the root environment, and the body is
    /// read at the root's data type. The entry's array parameters are the view's inputs by position.
    pub fn read_program(&self, src: &str) -> Result<Tree, Gap> {
        let (defs, body) = self.program_body(src)?;
        // A call of one of the program's own functions: the reader reads the entry alone (no row is a helper, and a
        // `let` is the `let` row, so nothing is inlined), so the gap names the helper and why its body is no row.
        self.read_body(&body, self.entry.clone()).map_err(|g| match defs.iter().find(|d| !d.entry && d.name == g.construct) {
            Some(d) => Gap {
                construct: format!("call of the program's function {}", d.name),
                dual: g.dual,
                detail: match &d.body {
                    Ok(_) => "the reader reads the entry alone and inlines no function".into(),
                    Err(h) => format!("the reader reads the entry alone and inlines no function; its body has {h}"),
                },
            },
            None => g,
        })
    }

    /// A program's definitions and its entry's body: the entry's array parameters renamed to the view's inputs, the
    /// header's bindings (each size input's `i64` rebinding, the zero arrays) taken away.
    fn program_body(&self, src: &str) -> Result<(Vec<Def>, E), Gap> {
        let defs = parse_file_with(src, true)?;
        let entries: Vec<&Def> = defs.iter().filter(|d| d.entry).collect();
        let entry = entries.iter().find(|d| d.name == "main").copied().or(if entries.len() == 1 { Some(entries[0]) } else { None }).ok_or_else(|| Gap::parse("no entry point", format!("{} entries, none named main", entries.len())))?;
        let mut body = entry.body.clone()?;
        let size_inputs = self.k.size_inputs();
        let inputs: Vec<&str> = self.k.spec.env.iter().filter(|l| l.kind == "input").map(|l| l.name.as_str()).collect();
        let arrays: Vec<&Param> = entry.params.iter().filter(|p| !size_inputs.contains(&p.name)).collect();
        if arrays.len() != inputs.len() {
            return Err(Gap::parse("entry signature", format!("{} array parameters, the view has {} inputs", arrays.len(), inputs.len())));
        }
        for (p, name) in arrays.iter().zip(&inputs) {
            if p.name != *name {
                body = body.rename(&p.name, name);
            }
        }
        // The header: `let k = i64.i32 k` for each size input, then the zero arrays.
        loop {
            match body {
                E::Let(x, rhs, rest) if (size_inputs.contains(&x) && *rhs == app("i64.i32", vec![var(&x)])) || (self.root_of_kind(&x, "zeros") && matches!(rhs.as_ref(), E::App(f, a) if **f == var("replicate") && a.len() == 2)) => body = *rest,
                other => {
                    body = other;
                    break;
                }
            }
        }
        Ok((defs, body))
    }

    /// An EXPRESSION of data type `ty` in the root environment (the entry's names in scope): what a program's body is
    /// when it is not of the root's data type (a pool's `i32` constant).
    pub fn read_expression(&self, src: &str, ty: Ty) -> Result<Tree, Gap> {
        let e = parse_expression(src, false, true)?;
        let entry = Entry { output: ty, guards: Vec::new(), ..self.entry.clone() };
        self.read_body(&e, entry)
    }

    /// The tree a decoding renders as, in the reader's canonical form: a substituted node as its substitute input,
    /// every variable's index word the least that names it, and an operand no row spells (`tabrows`' `cols`) by the
    /// reader's own rule ([`Reader6::read_body`]). A constant substitution or a variable read as its data type's
    /// constant has no one tree the reader gives (a terminal of the same spelling reads first), so it is an error.
    pub fn canonical(&self, tree: &Tree) -> Result<Tree, NotCanonical> {
        let d = self.decode(&self.entry, tree).map_err(NotCanonical::Bug)?;
        self.canonical_node(&d.root)
    }

    fn canonical_node(&self, n: &Node) -> Result<Tree, NotCanonical> {
        if let Some(s) = n.substituted {
            return match s.by {
                Substitution::Input(p) => self.canonical_node(n.kids.get(p as usize).ok_or_else(|| NotCanonical::Bug(format!("no input {p} substitutes {}", self.k.name(n.symbol))))?),
                Substitution::Constant => Err(NotCanonical::SubstitutedByConstant),
            };
        }
        let const_index = match n.choice {
            Some(Choice::Variable(binder)) => Some(n.environment.index_of(n.data_type, binder).ok_or_else(|| NotCanonical::Bug(format!("a variable out of scope: {binder:?}")))?),
            Some(Choice::Constant) => return Err(NotCanonical::VariableAsConstant),
            Some(Choice::Dc(i)) => Some(i),
            Some(Choice::Free(_)) | None => None,
        };
        let mut kids = n.kids.iter().map(|k| self.canonical_node(k)).collect::<Result<Vec<_>, _>>()?;
        // THE UNSPELLED OPERAND'S RULE, as the reader applies it: where the tied requirement held (no fit rendered),
        // the first variable in scope whose property equals the tied operand's.
        let spelled = self.rows.values().flatten().find(|p| p.symbol == n.symbol).map(|p| p.spelled.clone()).unwrap_or_default();
        for (i, kid) in n.kids.iter().enumerate() {
            if spelled.get(i).copied().unwrap_or(true) || !matches!(kid.choice, Some(Choice::Variable(_))) {
                continue;
            }
            let Some((j, pj, pi, guard)) = self.tie(n.symbol, i) else { continue };
            if n.outcomes.iter().any(|o| !o.holds && o.requirement == guard) {
                continue;
            }
            let want = n.kids.get(j).and_then(|k| k.properties.get(&pj)).ok_or_else(|| NotCanonical::Bug(format!("operand {j} of {} has no {pj}", self.k.name(n.symbol))))?;
            let v = crate::typed_environment::in_scope(&kid.environment, kid.data_type).into_iter().rev().find(|v| v.properties.get(&pi) == Some(want)).ok_or_else(|| NotCanonical::Bug(format!("no variable in scope has {pi} = {want}, though the requirement held")))?;
            kids[i].const_index = Some(kid.environment.index_of(kid.data_type, v.binder).ok_or_else(|| NotCanonical::Bug(format!("{:?} is not in scope", v.binder)))?);
        }
        Ok(self.read_first(n, &kids).unwrap_or(Tree { symbol: n.symbol, const_index, kids }))
    }

    /// THE ROW READ FIRST where two rows spell one text: a row tried before the node's own whose spelling is the
    /// node's with a property placeholder where the node spells a size operand (`next`'s `{0.bound}` against
    /// `nextin`'s size), when the decoded property is that size. The reader reads that row ([`Reader6::read_tree`]).
    fn read_first(&self, n: &Node, kids: &[Tree]) -> Option<Tree> {
        let rows = self.rows.get(&n.data_type)?;
        let own = rows.iter().find(|p| p.symbol == n.symbol)?;
        for r in rows.iter().take_while(|p| p.symbol != n.symbol).filter(|p| !p.props.is_empty()) {
            'matches: for holes in pmatch_all(&r.pattern, &own.pattern, r.inputs.len() + r.props.len()) {
                let operand = |h: Option<&Option<E>>| match h {
                    Some(Some(E::Hole(i))) if *i < own.inputs.len() => Some(*i),
                    _ => None,
                };
                let mut at = Vec::with_capacity(r.inputs.len());
                for (i, ty) in r.inputs.iter().enumerate() {
                    match operand(holes.get(i)) {
                        Some(si) if own.inputs[si] == *ty => at.push(si),
                        _ => continue 'matches,
                    }
                }
                for (j, (k, prop)) in r.props.iter().enumerate() {
                    let Some(sj) = operand(holes.get(r.inputs.len() + j)) else { continue 'matches };
                    let have = n.kids.get(at[*k]).and_then(|kid| kid.properties.get(prop));
                    if have.is_none() || have != n.kids.get(sj).and_then(|kid| kid.properties.get("value")) {
                        continue 'matches;
                    }
                }
                return Some(Tree { symbol: r.symbol, const_index: None, kids: at.iter().map(|&si| kids[si].clone()).collect() });
            }
        }
        None
    }

    fn decode(&self, entry: &Entry, tree: &Tree) -> Result<Decoding, String> {
        self.k.decoder.decode(entry, tree)
    }

    /// Read `e` at the entry's data type, then decode: every variable leaf's index word from G3's environment, every
    /// property placeholder checked against the decoded property.
    fn read_body(&self, e: &E, entry: Entry) -> Result<Tree, Gap> {
        // An adapter the tree read does not render is read again as the adapter SYMBOL evolution placed there, the
        // innermost first (a placed adapter changes the lengths its parents' repairs see), one a round: up to as many
        // rounds as the text has adapters. A text no round reads gives the first round's gap.
        let mut keep: BTreeSet<usize> = BTreeSet::new();
        let (mut first, mut tried, mut extra): (Option<Gap>, BTreeSet<usize>, Vec<(usize, usize)>) = (None, BTreeSet::new(), Vec::new());
        for _ in 0..READ_REJECTED {
            // The next adapter to read as placed: the innermost not tried at this keep.
            let mut next = keep.clone();
            if first.is_some() {
                match extra.iter().filter(|(_, i)| !tried.contains(i) && !keep.contains(i)).max() {
                    Some((_, i)) => {
                        next.insert(*i);
                    }
                    None => break,
                }
            }
            match self.read_body_keeping(e, &entry, &next) {
                Ok(Ok(tree)) => return Ok(tree),
                Ok(Err((more, gap))) => {
                    if first.is_none() {
                        first = Some(gap);
                    }
                    keep = next;
                    tried.clear();
                    extra = more;
                }
                Err(g) if first.is_none() => return Err(g),
                Err(_) => {
                    tried.extend(next.difference(&keep).copied());
                }
            }
        }
        Err(first.unwrap_or_else(|| Gap { construct: "adapter the gene does not render".into(), dual: self.type_name(entry.output), detail: "no reading of its adapters renders the text".into() }))
    }

    /// One read of [`Reader6::read_body`] with the adapter applications `keep` read as placed adapter symbols: the
    /// tree, or the marks the tree read does not render (each its depth and its number), with the gap.
    fn read_body_keeping(&self, e: &E, entry: &Entry, keep: &BTreeSet<usize>) -> Result<Result<Tree, Unrendered6>, Gap> {
        let entry = entry.clone();
        let (tree, adapters) = self.read_tree(e, &entry, keep)?;
        // THE ADAPTERS THE TREE RENDERS: the tree read is rendered and its text read again; the adapters of the two
        // reads (node, kind, length) must be the same, or the text is a program the gene does not express.
        let bug = |what: String| Gap { construct: "decode".into(), dual: self.type_name(entry.output), detail: bug6(self.k, what) };
        let d = self.decode(&entry, &tree).map_err(&bug)?;
        let text = self.k.program_of(&d).map_err(&bug)?.text;
        let (_, body) = self.program_body(&text).map_err(|g| bug(format!("the rendering of the tree read does not parse: {g}")))?;
        // The tree renders the text itself: read, whatever its adapters.
        if unparen(&body) == unparen(e) {
            return Ok(Ok(tree));
        }
        // Else (a text pass 5 spells otherwise, kept apart) the adapters of the two reads must be the same, every one a
        // repair: with an adapter read as placed, only the text itself will do.
        let (_, rendered) = self.read_tree(&body, &entry, &BTreeSet::new()).map_err(|g| bug(format!("the rendering of the tree read does not read: {g}")))?;
        let plain = |a: &Adapters6| a.iter().map(|(p, k, l, _)| (p.clone(), k.clone(), l.clone())).collect::<Vec<_>>();
        let same_count = strip_counted(e, &BTreeSet::new()).1 == strip_counted(&body, &BTreeSet::new()).1;
        if !keep.is_empty() || !same_count || plain(&adapters) != plain(&rendered) {
            let show = |a: &Adapters6| a.iter().map(|(p, k, l, _)| format!("{} {l:?} at {p:?}", k.trim_start_matches('%'))).collect::<Vec<_>>().join(", ");
            let gap = Gap { construct: "adapter the gene does not render".into(), dual: self.type_name(entry.output), detail: format!("the text has [{}], the tree read renders [{}]", show(&adapters), show(&rendered)) };
            let theirs = plain(&rendered);
            let missing: Vec<(usize, usize)> = adapters.iter().filter(|(p, k, l, _)| !theirs.contains(&(p.clone(), k.clone(), l.clone()))).map(|a| (a.0.len(), a.3)).collect();
            let extra = if missing.is_empty() { adapters.iter().map(|a| (a.0.len(), a.3)).collect() } else { missing };
            return Ok(Err((extra, gap)));
        }
        Ok(Ok(tree))
    }

    /// Read `e` at the entry's data type and set its index words (no check of its adapters): the tree and the
    /// adapters the text has.
    ///
    /// A PROPERTY PLACEHOLDER is checked only on the decoded tree, so a row whose spelling holds one (`next`'s
    /// `{0.bound}`) can read a text another row spells too (`nextin`'s size operand) and fail there: the read is then
    /// made again with that row rejected at that node, until the placeholders hold or no row is left.
    fn read_tree(&self, e: &E, entry: &Entry, keep: &BTreeSet<usize>) -> Result<(Tree, Adapters6), Gap> {
        let e = strip_adapters(e, keep);
        let mut rejected = BTreeSet::new();
        loop {
            match self.read_tree_once(&e, entry, &rejected)? {
                Ok(read) => return Ok(read),
                Err((at, gap)) => {
                    if !rejected.insert(at) || rejected.len() > READ_REJECTED {
                        return Err(gap);
                    }
                }
            }
        }
    }

    /// One read of [`Reader6::read_tree`] with the rows `rejected` not tried at their nodes: the tree, or the node and
    /// row whose property placeholder failed, with its gap.
    fn read_tree_once(&self, e: &E, entry: &Entry, rejected: &BTreeSet<(Vec<u32>, SymbolId)>) -> Result<Result<(Tree, Adapters6), Rejected6>, Gap> {
        self.steps.set(0);
        let mut rec = Read6 { rejected: rejected.clone(), ..Read6::default() };
        let mut scope = self.root_scope();
        let mut tree = match self.read6(e, entry.output, &mut Vec::new(), &mut scope, &mut rec) {
            Ok(t) => t,
            Err(g) => return Err(rec.deepest.map_or(g, |d| d.1)),
        };
        let bug = |what: String| Gap { construct: "decode".into(), dual: self.type_name(entry.output), detail: bug6(self.k, what) };
        let d = self.decode(entry, &tree).map_err(&bug)?;
        let mut wanted = Vec::with_capacity(rec.leaves.len());
        for (path, origin) in &rec.leaves {
            let leaf = d.node(path).ok_or_else(|| bug(format!("no node at {path:?}")))?;
            let binder: Binder = match origin {
                Origin6::Root(name) => d.root.environment.variables.iter().find(|v| v.name.as_deref() == Some(name.as_str())).map(|v| v.binder).ok_or_else(|| bug(format!("{name} is no root variable")))?,
                Origin6::Bound { at, ordinal } => {
                    let env = &d.node(at).ok_or_else(|| bug(format!("no node at {at:?}")))?.environment;
                    env.variables.iter().filter(|v| v.binder.depth == env.depth).nth(*ordinal).map(|v| v.binder).ok_or_else(|| bug(format!("the argument at {at:?} binds no variable {ordinal}")))?
                }
            };
            let word = leaf.environment.index_of(leaf.data_type, binder).ok_or_else(|| bug(format!("the variable at {path:?} ({binder:?}) is not in scope")))?;
            tree_at(&mut tree, path).ok_or_else(|| bug(format!("no node at {path:?}")))?.const_index = Some(word);
            wanted.push((path.clone(), Choice::Variable(binder)));
        }
        // THE UNSPELLED OPERAND'S RULE (master 6 README 9: `tabrows` never spells its `cols` size): where the text fits
        // the tied operand (`fit d 0f32 (body)`), the operand is the fit's length, read as any operand; where it does
        // not, the requirement held, so the operand is the FIRST variable in scope (the least index word) whose
        // property equals the tied operand's. Deepest first, each on a decode with every word set so far.
        let mut unspelled = rec.unspelled.clone();
        unspelled.sort_by_key(|u| std::cmp::Reverse(u.node.len()));
        for u in &unspelled {
            let d = self.decode(entry, &tree).map_err(&bug)?;
            let node = d.node(&u.node).ok_or_else(|| bug(format!("no node at {:?}", u.node)))?;
            let want = node.kids.get(u.j).and_then(|n| n.properties.get(&u.pj)).ok_or_else(|| bug(format!("operand {} of the node at {:?} has no {}", u.j, u.node, u.pj)))?;
            let kid = node.kids.get(u.operand).ok_or_else(|| bug(format!("the node at {:?} has no operand {}", u.node, u.operand)))?;
            let v = crate::typed_environment::in_scope(&kid.environment, kid.data_type).into_iter().rev().find(|v| v.properties.get(&u.pi) == Some(want)).ok_or_else(|| Gap {
                construct: "unspelled operand".into(),
                dual: self.type_name(kid.data_type),
                detail: format!("{} at {:?}: no variable in scope has {} = {want}, the {} of its operand {}", self.k.name(node.symbol), u.node, u.pi, u.pj, u.j),
            })?;
            let mut path = u.node.clone();
            path.push(u.operand as u32);
            let word = kid.environment.index_of(kid.data_type, v.binder).ok_or_else(|| bug(format!("the variable at {path:?} is not in scope")))?;
            tree_at(&mut tree, &path).ok_or_else(|| bug(format!("no node at {path:?}")))?.const_index = Some(word);
            wanted.push((path, Choice::Variable(v.binder)));
        }
        // A CONSTANT WITH A VARIABLE OF ITS DATA TYPE IN SCOPE is a node the work bound substitutes by its constant: the
        // first candidate ([`Reader6::constant_substitutes`]) the decode substitutes there.
        for path in &rec.substituted {
            let ty = self.decode(entry, &tree).map_err(&bug)?.node(path).map(|n| n.data_type).ok_or_else(|| bug(format!("no node at {path:?}")))?;
            let mut found = false;
            for candidate in self.constant_substitutes(ty) {
                *tree_at(&mut tree, path).ok_or_else(|| bug(format!("no node at {path:?}")))? = candidate;
                let d = self.decode(entry, &tree).map_err(&bug)?;
                if d.node(path).and_then(|n| n.substituted).map(|s| s.by) == Some(Substitution::Constant) {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Gap { construct: "constant".into(), dual: self.type_name(ty), detail: format!("the text has the constant at {path:?} where a variable of its data type is in scope, and no node the work bound substitutes by its constant can stand there") });
            }
        }
        wanted.extend(rec.constants.iter().map(|p| (p.clone(), Choice::Constant)));
        let again = self.decode(entry, &tree).map_err(&bug)?;
        for (path, choice) in wanted {
            let got = again.node(&path).and_then(|n| n.choice);
            if got != Some(choice) {
                return Err(bug(format!("the leaf at {path:?} reads as {got:?}, not {choice:?}")));
            }
        }
        for (path, k, p, read) in &rec.props {
            let node = again.node(path).ok_or_else(|| bug(format!("no node at {path:?}")))?;
            let value = node.kids.get(*k).and_then(|kid| kid.properties.get(p));
            if value.and_then(property_expr).as_ref() != Some(&unparen(read)) {
                let gap = Gap { construct: format!("{{{k}.{p}}}"), dual: self.type_name(node.data_type), detail: format!("{} reads {p} = {value:?} of its operand {k}, the text has {read:?}", self.k.name(node.symbol)) };
                return Ok(Err(((path.clone(), node.symbol), gap)));
            }
        }
        Ok(Ok((tree, rec.adapters)))
    }

    /// Read an expression at data type `ty`, at `path`, with `scope` in scope.
    fn read6(&self, e: &E, ty: Ty, path: &mut Vec<u32>, scope: &mut Vec<Named6>, rec: &mut Read6) -> Result<Tree, Gap> {
        let (e, marks) = unmark_numbered(e);
        rec.adapters.extend(marks.into_iter().map(|(kind, length, i)| (path.clone(), kind.to_string(), unparen(length), i)));
        self.steps.set(self.steps.get() + 1);
        if self.steps.get() > READ_STEPS {
            return Err(Gap { construct: "reading budget".into(), dual: self.type_name(ty), detail: format!("over {READ_STEPS} matches") });
        }
        if let E::Var(x) = e {
            if let (Some(v), Some(&sym)) = (scope.iter().rev().find(|v| &v.name == x), self.vars.get(&ty)) {
                if v.ty == ty {
                    rec.leaves.push((path.clone(), v.origin.clone()));
                    return Ok(Tree { symbol: sym, const_index: Some(0), kids: Vec::new() });
                }
            }
        }
        if let (Some(&sym), Some(d)) = (self.consts.get(&ty), self.k.data_type(ty)) {
            let integer = d.name.starts_with('i') || d.name.starts_with('u');
            let at = |i: &usize| {
                let l = self.k.literal_text(sym, *i).unwrap_or_default();
                match e {
                    E::Int(v) => integer && l.parse::<i64>() == Ok(*v),
                    E::Lit(t) => integer && l.parse::<i64>().is_ok_and(|v| format!("{v}{}", d.name) == *t),
                    E::Float(f) => !integer && float_literal(&format!("{l}{}", d.name)).as_ref() == Some(f),
                    _ => false,
                }
            };
            if let Some(i) = (0..self.k.literals(sym).len()).find(at) {
                return Ok(Tree { symbol: sym, const_index: Some(i as u32), kids: Vec::new() });
            }
        }
        let mut first: Option<Gap> = None;
        for p in self.rows.get(&ty).into_iter().flatten() {
            if rec.rejected.contains(&(path.clone(), p.symbol)) {
                continue;
            }
            for holes in pmatch_all(&p.pattern, e, p.inputs.len() + p.props.len()) {
                let mark = rec.mark();
                match self.read_row(p, &holes, path, scope, rec) {
                    Ok(t) => return Ok(t),
                    Err(g) => {
                        rec.undo(mark);
                        first.get_or_insert(g);
                    }
                }
            }
        }
        if self.constants.get(&ty) == Some(e) {
            // No variable of the data type in scope: a `var@T` renders its constant.
            if let Some(&sym) = self.vars.get(&ty).filter(|_| !scope.iter().any(|v| v.ty == ty)) {
                rec.constants.push(path.clone());
                return Ok(Tree { symbol: sym, const_index: Some(0), kids: Vec::new() });
            }
            // Else only a heavy node the work bound substitutes by its constant renders it, chosen once the tree is
            // decoded ([`Reader6::read_body`]).
            if let Some(t) = self.constant_substitutes(ty).into_iter().next() {
                rec.substituted.push(path.clone());
                return Ok(t);
            }
        }
        let g = first.unwrap_or_else(|| Gap { construct: e.head(), dual: self.type_name(ty), detail: "no row of this data type reads it".into() });
        if rec.deepest.as_ref().is_none_or(|d| d.0 < path.len()) {
            rec.deepest = Some((path.len(), g.clone()));
        }
        Err(g)
    }

    /// One row's match: each operand read at its argument's data type, the variables an argument binds in scope while
    /// it is read, the property placeholders recorded.
    fn read_row(&self, p: &Pattern6, holes: &[Option<E>], path: &mut Vec<u32>, scope: &mut Vec<Named6>, rec: &mut Read6) -> Result<Tree, Gap> {
        let name = self.k.name(p.symbol);
        let unbound = || Gap { construct: "spelling".into(), dual: self.type_name(p.inputs.first().copied().unwrap_or(Ty::Futhark(0))), detail: format!("{name}: a hole the spelling does not bind") };
        let mut kids = Vec::with_capacity(p.inputs.len());
        for (i, ty) in p.inputs.iter().enumerate() {
            let mut h = match holes.get(i).cloned().flatten() {
                Some(h) => h,
                None => {
                    let (j, pj, pi, _) = self.tie(p.symbol, i).ok_or_else(unbound)?;
                    match holes.get(j).and_then(Option::as_ref).and_then(|hj| unmark(hj).1) {
                        Some(length) => length.clone(),
                        None => {
                            let sym = *self.vars.get(ty).ok_or_else(unbound)?;
                            rec.unspelled.push(Unspelled6 { node: path.clone(), operand: i, j, pj, pi });
                            kids.push(Tree { symbol: sym, const_index: Some(0), kids: Vec::new() });
                            continue;
                        }
                    }
                }
            };
            let outer = scope.len();
            let mut at = path.clone();
            at.push(i as u32);
            for (j, bty) in p.binds.get(&(i as u32)).into_iter().flatten().enumerate() {
                // Named by its binding position, so two reads of one structure name it alike (an adapter's length
                // is compared by its names across reads, [`Reader6::read_body`]) and two binders never share a name.
                let fresh = format!("%b{}/{j}", at.iter().map(u32::to_string).collect::<Vec<_>>().join("."));
                h = h.rename(&format!("{{{i}.v{j}}}"), &fresh);
                scope.push(Named6 { name: fresh, ty: *bty, origin: Origin6::Bound { at: at.clone(), ordinal: j } });
            }
            path.push(i as u32);
            let r = self.read6(&h, *ty, path, scope, rec);
            path.pop();
            scope.truncate(outer);
            kids.push(r?);
        }
        for (j, (k, prop)) in p.props.iter().enumerate() {
            let h = holes.get(p.inputs.len() + j).cloned().flatten().ok_or_else(unbound)?;
            rec.props.push((path.clone(), *k, prop.clone(), h));
        }
        Ok(Tree { symbol: p.symbol, const_index: None, kids })
    }
}

impl Reader6<'_> {
    /// The must guard of `symbol` that ties its operand `i` to another (`in<j>.<pj> = in<i>.<pi>`, either way round):
    /// (the other operand, its property, operand `i`'s property, the guard's index in the tables).
    fn tie(&self, symbol: SymbolId, i: usize) -> Option<(usize, String, String, usize)> {
        self.k.tables.guards.iter().enumerate().filter(|(_, g)| g.owner == Owner::Symbol(i64::from(symbol))).find_map(|(gi, g)| {
            let (_, a, b) = g.condition.comparison()?;
            match (a, b) {
                (PropertyExpr::Input { position: j, property: pj }, PropertyExpr::Input { position, property }) if *position as usize == i => Some((*j as usize, pj.clone(), property.clone(), gi)),
                (PropertyExpr::Input { position, property }, PropertyExpr::Input { position: j, property: pj }) if *position as usize == i => Some((*j as usize, pj.clone(), property.clone(), gi)),
                _ => None,
            }
        })
    }

    /// The first leaf of data type `ty`: its `var` (word 0), its literal 0, or a row of no input.
    fn leaf(&self, ty: Ty) -> Option<Tree> {
        if let Some(&s) = self.vars.get(&ty) {
            return Some(Tree { symbol: s, const_index: Some(0), kids: Vec::new() });
        }
        if let Some(&s) = self.consts.get(&ty) {
            return Some(Tree { symbol: s, const_index: Some(0), kids: Vec::new() });
        }
        self.k.symbols().find(|r| r.role == Role::Function && self.k.output(r.symbol as SymbolId) == ty && self.k.inputs(r.symbol as SymbolId).is_empty()).map(|r| Tree { symbol: r.symbol as SymbolId, const_index: None, kids: Vec::new() })
    }

    /// THE CANDIDATES FOR A SUBSTITUTED CONSTANT, in order: each row of data type `ty` whose work bound (its guard of
    /// order 0 on `env.trips`) substitutes it by its constant (its first repair `{const}`, or `{in.own}` with no input of
    /// its own data type), in symbol order, its inputs leaves (never rendered), each `var` leaf's word running over 0 to
    /// 3, the first input's fastest. A node is substituted where `env.trips >= depth`, or one trip short of it when a
    /// guard whose heavy repair runs in its body fails (`tabrows`' body fit), which the leaves' words decide: so the
    /// words vary.
    fn constant_substitutes(&self, ty: Ty) -> Vec<Tree> {
        const WORDS: u32 = 4;
        let mut out = Vec::new();
        for row in self.k.symbols() {
            let s = row.symbol as SymbolId;
            let (inputs, output) = (self.k.inputs(s), self.k.output(s));
            let work = self.k.tables.guards.iter().find(|g| g.owner == Owner::Symbol(row.symbol) && g.condition.references().iter().any(|r| matches!(r, crate::geneframe::Reference::Env("trips"))));
            let Some(first) = work.and_then(|g| g.repairs.first()).and_then(|r| parse_template(&r.template).ok()).and_then(|t| substitute_of(&t)) else { continue };
            let by_constant = match first {
                crate::typed_environment::template::Substitute::Const => true,
                crate::typed_environment::template::Substitute::Own => !inputs.contains(&output),
                crate::typed_environment::template::Substitute::Input(_) => false,
            };
            if output != ty || !by_constant {
                continue;
            }
            let Some(kids) = inputs.iter().map(|t| self.leaf(*t)).collect::<Option<Vec<Tree>>>() else { continue };
            let vars: Vec<usize> = (0..kids.len()).filter(|k| self.vars.values().any(|v| *v == kids[*k].symbol)).collect();
            for n in 0..WORDS.pow(vars.len() as u32) {
                let mut t = Tree { symbol: s, const_index: None, kids: kids.clone() };
                for (place, k) in vars.iter().enumerate() {
                    t.kids[*k].const_index = Some(n / WORDS.pow(place as u32) % WORDS);
                }
                out.push(t);
            }
        }
        out
    }
}

/// A symbol's spelling (the template engine's placeholders) as a pattern of the reader's grammar: each argument's type
/// spelling after a binder's `:` as `{k.type}`, and `{=in<k>.<p>}` as `{k.p}`.
fn pattern6(k: &Kingdom6, symbol: SymbolId, spelling: &str) -> String {
    let mut out = spelling.to_string();
    for (i, ty) in k.inputs(symbol).iter().enumerate() {
        let Some(d) = k.data_type(*ty) else { continue };
        let ts = match &d.kind {
            crate::geneframe::DataKind::Array { .. } => {
                let elem = k.data_type(k.innermost(*ty)).map_or(String::new(), |e| e.name.clone());
                format!("{}{elem}", d.properties.iter().map(|p| format!("[{{=in{i}.{p}}}]")).collect::<String>())
            }
            _ => d.name.clone(),
        };
        out = out.replace(&format!(": {ts})"), &format!(": {{{i}.type}})"));
    }
    let mut rest = out.as_str();
    let mut done = String::new();
    while let Some(at) = rest.find("{=in") {
        done.push_str(&rest[..at]);
        let after = &rest[at + 4..];
        match after.find('}') {
            Some(end) => {
                done.push('{');
                done.push_str(&after[..end]);
                done.push('}');
                rest = &after[end + 1..];
            }
            None => {
                done.push_str(&rest[at..]);
                rest = "";
            }
        }
    }
    done.push_str(rest);
    done
}

/// The node of a tree at `path`.
fn tree_at<'t>(t: &'t mut Tree, path: &[u32]) -> Option<&'t mut Tree> {
    match path.split_first() {
        None => Some(t),
        Some((i, rest)) => tree_at(t.kids.get_mut(*i as usize)?, rest),
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests6;
