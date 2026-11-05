//! THE TEMPLATE ENGINE of pass 5 (design section 8.14.2): a symbol's spelling, a repair's template, a binding's name
//! and a transform's text are templates over one CLOSED list of placeholders, rendered by this one kingdom-free engine.
//! A kingdom never writes renderer code for an adapter; it writes rows.
//!
//! The placeholders:
//!
//! | placeholder | renders |
//! |---|---|
//! | `{k}` | operand k's text |
//! | `{k@form}` | operand k in the form `form` (a Spelling form, passed to the operand's head argument when it lacks one) |
//! | `{k.vj}` | the name of the j-th variable argument k introduces |
//! | `{=formula}`, `{=formula@form}` | a formula's value, spelled by the kingdom's formula rows in the form (below) |
//! | `{name}` | shorthand for `{=name}` (`{each}`, `{depth}`) |
//! | `{nth}` | in a variable's spelling only: nothing for the first variable in scope of its spelling, else its ordinal among them (2, 3, ...) |
//! | `{p0}` | repair parameter 0's value |
//! | `{zero}`, `{const}` | the site's data type's constant spelling (`{zero}`: its element's, for an array) |
//! | `{in<k>}`, `{in.own}` | input k's text; the first input of the node's own data type |
//! | `{fresh name}` | `name` made unique by a counter over the program; the same within one template |
//! | `{self}`, `{self(k := template, ...)}` | the node's own spelling, with operands replaced |
//! | `{call S(template, ...)}` | the adapter symbol S (its spelling) applied to the arguments; its prelude is added once |
//! | `{if condition : template else template}` | the first template where the condition holds, else the second |
//! | `{for i < n sep text : template}` | the template for `i = 0 .. n-1`, joined by `text` (` sep text` optional) |
//!
//! `{{` and `}}` write a literal brace. A repair's template that is ONE substitute (`{in<k>}`, `{in.own}`, `{const}`)
//! replaces its node ([`substitute_of`]); any other template wraps or rewrites.
//!
//! FORMULAS ARE SPELLED BY ROWS (code review 2026-11-05, finding 3): a size name, a non-negative literal, a tag and a
//! flag are spelled as themselves; every operator (`min`, `max`, `+`, `*`, a negative literal, the grouping of a
//! compound operand) by the kingdom's FORMULA rows ([`Lexicon`]: the Symbol rows of role `formula`, `=min`, `=max`, `=+`,
//! `=*`, `=negative`, `=group`, one spelling per form), so Futhark writes `i64.min k n` and a variable's name
//! `min_k_n`. An operator with no row is "our software has a bug", and so is a `?` in a rendered value: a template that
//! may meet one is chosen by a `when` with `known(...)`. A formula given as a `{call ..}` argument is an operand: grouped
//! when compound.
//!
//! A repair's template is also EXECUTABLE ([`Scope::exec`]): the interpreter and the cost run the same template the
//! renderer writes, over values instead of text, so an adapter's meaning is its symbol's and nothing is matched on an
//! adapter's name.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::geneframe::formula::{is_identifier, Condition, PropertyExpr, Reference, SizeBounds, Truth};

/// One piece of a parsed template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Text(String),
    Operand { k: u32, form: Option<String> },
    /// `{k.vj}`: the name of the j-th variable argument k introduces.
    Bound { k: u32, j: u32 },
    /// `{=formula}` or `{=formula@form}`.
    Value { formula: PropertyExpr, form: String },
    /// `{nth}`.
    Nth,
    Param(u32),
    Zero,
    Const,
    /// `{in<k>}`: input k, as a substitute.
    Input(u32),
    /// `{in.own}`: the first input of the node's own data type, as a substitute.
    Own,
    Fresh(String),
    /// `{self}` (no replacement) or `{self(k := template, ...)}`.
    SelfWith(Vec<(u32, Template)>),
    /// `{call S(template, ...)}`.
    Call { symbol: String, args: Vec<Template> },
    If { cond: Condition, then: Template, otherwise: Template },
    For { index: String, count: PropertyExpr, sep: String, body: Template },
}

/// A parsed template: its pieces in order.
pub type Template = Vec<Piece>;

fn bad(text: &str, what: &str) -> String {
    format!("our software has a bug: the template {text:?} {what}")
}

/// The index of the `}` closing the placeholder opened just before `s` (brace depth counted), in `s`.
fn closing(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' if depth == 0 => return Some(i),
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// The first occurrence of `pat` in `s` outside any brace or parenthesis.
fn top_level(s: &str, pat: &str) -> Option<usize> {
    let (mut braces, mut parens) = (0i32, 0i32);
    for (i, c) in s.char_indices() {
        if braces == 0 && parens == 0 && s[i..].starts_with(pat) {
            return Some(i);
        }
        match c {
            '{' => braces += 1,
            '}' => braces -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            _ => {}
        }
    }
    None
}

/// `s` split at its top-level commas (outside braces and parentheses), each part trimmed; none for a blank `s`.
fn top_level_items(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = s;
    while !rest.trim().is_empty() {
        let end = top_level(rest, ",").unwrap_or(rest.len());
        out.push(rest[..end].trim());
        rest = if end < rest.len() { &rest[end + 1..] } else { "" };
    }
    out
}

/// Parse a template's text into its pieces. Every malformed placeholder is "our software has a bug: ...".
pub fn parse_template(text: &str) -> Result<Template, String> {
    let mut out: Template = Vec::new();
    let mut lit = String::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if rest.starts_with("{{") || rest.starts_with("}}") {
            lit.push(c);
            rest = &rest[2..];
            continue;
        }
        if c == '}' {
            return Err(bad(text, "has a } that closes nothing (write }} for a brace)"));
        }
        if c != '{' {
            lit.push(c);
            rest = &rest[c.len_utf8()..];
            continue;
        }
        let inner = &rest[1..];
        let end = closing(inner).ok_or_else(|| bad(text, "has a { that is not closed"))?;
        if !lit.is_empty() {
            out.push(Piece::Text(std::mem::take(&mut lit)));
        }
        out.push(placeholder(text, inner[..end].trim())?);
        rest = &inner[end + 1..];
    }
    if !lit.is_empty() {
        out.push(Piece::Text(lit));
    }
    Ok(out)
}

fn digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

fn placeholder(text: &str, p: &str) -> Result<Piece, String> {
    if p.is_empty() {
        return Err(bad(text, "has an empty placeholder {}"));
    }
    if let Some(f) = p.strip_prefix('=') {
        // `@form` after the formula (a formula holds no `@`).
        let (formula, form) = match f.rsplit_once('@') {
            Some((e, form)) if is_identifier(form) => (e, form.to_string()),
            Some(_) => return Err(bad(text, &format!("has {{={f}}}, whose form is not a name"))),
            None => (f, String::new()),
        };
        return Ok(Piece::Value { formula: PropertyExpr::parse(formula.trim())?, form });
    }
    if p == "nth" {
        return Ok(Piece::Nth);
    }
    if let Some((k, v)) = p.split_once('.') {
        if digits(k) {
            let j = v.strip_prefix('v').filter(|j| digits(j)).ok_or_else(|| bad(text, &format!("has {{{p}}}, which is not {{k.vj}}")))?;
            return Ok(Piece::Bound { k: k.parse().map_err(|_| bad(text, "has an operand number out of range"))?, j: j.parse().map_err(|_| bad(text, "has a variable number out of range"))? });
        }
    }
    let (head, form) = match p.split_once('@') {
        Some((h, f)) if digits(h) => (h, Some(f.to_string())),
        _ => (p, None),
    };
    if digits(head) {
        return head.parse().map(|k| Piece::Operand { k, form }).map_err(|_| bad(text, "has an operand number out of range"));
    }
    if let Some(d) = p.strip_prefix('p').filter(|d| digits(d)) {
        return d.parse().map(Piece::Param).map_err(|_| bad(text, "has a parameter number out of range"));
    }
    match p {
        "zero" => return Ok(Piece::Zero),
        "const" => return Ok(Piece::Const),
        "self" => return Ok(Piece::SelfWith(Vec::new())),
        "in.own" => return Ok(Piece::Own),
        _ => {}
    }
    if let Some(d) = p.strip_prefix("in").filter(|d| digits(d)) {
        return d.parse().map(Piece::Input).map_err(|_| bad(text, "has an input number out of range"));
    }
    if let Some(name) = p.strip_prefix("fresh ") {
        let name = name.trim();
        if !is_identifier(name) {
            return Err(bad(text, &format!("has {{fresh {name}}}, whose name is not an identifier")));
        }
        return Ok(Piece::Fresh(name.to_string()));
    }
    if let Some(args) = p.strip_prefix("self(").and_then(|a| a.strip_suffix(')')) {
        let mut repl = Vec::new();
        for item in top_level_items(args) {
            let (k, t) = item.split_once(":=").ok_or_else(|| bad(text, "has a self( argument that is not k := template"))?;
            let k: u32 = k.trim().parse().map_err(|_| bad(text, "has a self( argument whose operand is not a number"))?;
            repl.push((k, parse_template(t.trim())?));
        }
        return Ok(Piece::SelfWith(repl));
    }
    if let Some(call) = p.strip_prefix("call ") {
        let open = call.find('(').ok_or_else(|| bad(text, "has a call with no ("))?;
        let symbol = call[..open].trim();
        let args = call[open + 1..].strip_suffix(')').ok_or_else(|| bad(text, "has a call whose arguments are not closed by )"))?;
        if symbol.is_empty() || symbol.contains(char::is_whitespace) {
            return Err(bad(text, "has a call of no symbol"));
        }
        let args = top_level_items(args).into_iter().map(parse_template).collect::<Result<Vec<_>, _>>()?;
        return Ok(Piece::Call { symbol: symbol.to_string(), args });
    }
    if let Some(body) = p.strip_prefix("if ") {
        let colon = top_level(body, ":").ok_or_else(|| bad(text, "has an if with no :"))?;
        let cond = Condition::parse(body[..colon].trim())?;
        let after = &body[colon + 1..];
        let (then, otherwise) = match top_level(after, " else ") {
            Some(e) => (&after[..e], &after[e + " else ".len()..]),
            None => (after, ""),
        };
        return Ok(Piece::If { cond, then: parse_template(then.trim())?, otherwise: parse_template(otherwise.trim())? });
    }
    if let Some(body) = p.strip_prefix("for ") {
        let colon = top_level(body, ":").ok_or_else(|| bad(text, "has a for with no :"))?;
        let head = &body[..colon];
        let (head, sep) = match head.find(" sep ") {
            Some(s) => (&head[..s], head[s + " sep ".len()..].to_string()),
            None => (head, String::new()),
        };
        let (index, count) = head.split_once('<').ok_or_else(|| bad(text, "has a for with no <"))?;
        let index = index.trim();
        if !is_identifier(index) {
            return Err(bad(text, "has a for whose index is not a name"));
        }
        let count = PropertyExpr::parse(count.trim())?;
        let raw = &body[colon + 1..];
        let raw = raw.strip_prefix(' ').unwrap_or(raw);
        return Ok(Piece::For { index: index.to_string(), count, sep, body: parse_template(raw)? });
    }
    if is_identifier(p) {
        return Ok(Piece::Value { formula: PropertyExpr::parse(p)?, form: String::new() });
    }
    Err(bad(text, &format!("has the placeholder {{{p}}}, which is not in the closed list")))
}

/// What a repair's template substitutes for its node when it is one substitute and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Substitute {
    Input(u32),
    Own,
    Const,
}

/// The substitute a template is, when it is exactly one of `{in<k>}`, `{in.own}`, `{const}`.
pub fn substitute_of(t: &Template) -> Option<Substitute> {
    match t.as_slice() {
        [Piece::Input(k)] => Some(Substitute::Input(*k)),
        [Piece::Own] => Some(Substitute::Own),
        [Piece::Const] => Some(Substitute::Const),
        _ => None,
    }
}

/// The operand rewrites a template is, when it is exactly one `{self(k := template, ...)}` with at least one: at the
/// node they rewrite its operands in order, as an operand repair rewrites one.
pub fn rewrites_of(t: &Template) -> Option<&[(u32, Template)]> {
    match t.as_slice() {
        [Piece::SelfWith(r)] if !r.is_empty() => Some(r),
        _ => None,
    }
}

/// Every symbol a template calls, in order of first appearance (in every branch, rendered or not).
pub fn calls_of(t: &Template) -> Vec<String> {
    fn go(t: &Template, out: &mut Vec<String>) {
        for p in t {
            match p {
                Piece::Call { symbol, args } => {
                    if !out.contains(symbol) {
                        out.push(symbol.clone());
                    }
                    args.iter().for_each(|a| go(a, out));
                }
                Piece::SelfWith(r) => r.iter().for_each(|(_, a)| go(a, out)),
                Piece::If { then, otherwise, .. } => {
                    go(then, out);
                    go(otherwise, out);
                }
                Piece::For { body, .. } => go(body, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    go(t, &mut out);
    out
}

/// Every formula a template renders, in every branch (the table check reads what each needs `known`).
pub fn values_of(t: &Template) -> Vec<&PropertyExpr> {
    fn go<'a>(t: &'a Template, out: &mut Vec<&'a PropertyExpr>) {
        for p in t {
            match p {
                Piece::Value { formula, .. } => out.push(formula),
                Piece::Call { args, .. } => args.iter().for_each(|a| go(a, out)),
                Piece::SelfWith(r) => r.iter().for_each(|(_, a)| go(a, out)),
                Piece::If { then, otherwise, .. } => {
                    go(then, out);
                    go(otherwise, out);
                }
                Piece::For { body, .. } => go(body, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    go(t, &mut out);
    out
}

/// The counter of `{fresh name}` over one program: `ad_a1`, `ad_a2`, ...
#[derive(Debug, Default)]
pub struct Fresh {
    next: BTreeMap<String, u32>,
}

impl Fresh {
    fn make(&mut self, name: &str) -> String {
        let n = self.next.entry(name.to_string()).or_insert(0);
        *n += 1;
        format!("{name}{n}")
    }
}

/// THE KINGDOM'S LEXICON, read from its Symbol rows once: each formula operator's spelling by form (the rows of role
/// `formula`: `=min`, `=max`, `=+`, `=*`, `=negative`, `=group`), and each adapter symbol's spelling and prelude, which
/// a `{call ..}` renders.
#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    /// `(operator, form)` to its parsed template (`{0}`, `{1}` its operands).
    pub formulas: BTreeMap<(String, String), Template>,
    /// An adapter symbol's name to its default spelling and its prelude.
    pub adapters: BTreeMap<String, (Template, String)>,
}

/// True when a value spells as one word: a size name, a non-negative literal, a tag, a flag.
fn atomic(e: &PropertyExpr) -> bool {
    matches!(e, PropertyExpr::Size(_) | PropertyExpr::Tag(_) | PropertyExpr::Flag(_)) || matches!(e, PropertyExpr::Literal(v) if *v >= 0)
}

impl Lexicon {
    /// The operator row `op` in `form`, rendered with `operands`.
    fn apply(&self, op: &str, form: &str, operands: Vec<String>) -> Result<String, String> {
        let t = self.formulas.get(&(op.to_string(), form.to_string())).ok_or_else(|| format!("our software has a bug: a formula's {op} is spelled in the form {form:?}, and the kingdom has no formula row ={op} of that form"))?;
        let mut s = String::new();
        for p in t {
            match p {
                Piece::Text(x) => s.push_str(x),
                Piece::Operand { k, form: None } => s.push_str(operands.get(*k as usize).ok_or_else(|| format!("our software has a bug: the formula row ={op} reads operand {k}"))?),
                other => return Err(format!("our software has a bug: the formula row ={op} holds {other:?}, which a formula row may not")),
            }
        }
        Ok(s)
    }

    /// A compound text grouped by the form's `=group` row, or as it is when the kingdom has none.
    fn group(&self, text: String, form: &str) -> Result<String, String> {
        if self.formulas.contains_key(&("group".to_string(), form.to_string())) {
            self.apply("group", form, vec![text])
        } else {
            Ok(text)
        }
    }

    /// `e` as an operand: grouped by the form's `=group` row when it is compound.
    pub fn operand(&self, e: &PropertyExpr, form: &str) -> Result<String, String> {
        let text = self.spell(e, form)?;
        if atomic(e) {
            Ok(text)
        } else {
            self.group(text, form)
        }
    }

    /// THE SPELLING OF A VALUE in `form`: an atom as itself, an operator by its row (an n-ary operator folded from the
    /// right, `min(a, b, c)` as `min a (min b c)`, each operand grouped when compound). A `?`, an unresolved reference
    /// or an operator with no row is a bug.
    pub fn spell(&self, e: &PropertyExpr, form: &str) -> Result<String, String> {
        use PropertyExpr as P;
        let fold = |op: &str, xs: &[P]| -> Result<String, String> {
            let (last, rest) = xs.split_last().ok_or_else(|| format!("our software has a bug: an empty {op}"))?;
            let mut acc = self.operand(last, form)?;
            for (i, x) in rest.iter().rev().enumerate() {
                let right = if i == 0 { acc } else { self.group(acc, form)? };
                acc = self.apply(op, form, vec![self.operand(x, form)?, right])?;
            }
            Ok(acc)
        };
        match e {
            P::Size(n) => Ok(n.clone()),
            P::Literal(v) if *v >= 0 => Ok(v.to_string()),
            P::Literal(v) => match self.formulas.get(&("negative".to_string(), form.to_string())) {
                Some(_) => self.apply("negative", form, vec![v.unsigned_abs().to_string()]),
                None => Ok(v.to_string()),
            },
            P::Tag(t) => Ok(t.clone()),
            P::Flag(b) => Ok(b.to_string()),
            P::Min(xs) => fold("min", xs),
            P::Max(xs) => fold("max", xs),
            P::Sum(xs) => fold("+", xs),
            P::Product(xs) => fold("*", xs),
            P::Unknown => Err("our software has a bug: a template renders the value ?, known only at run time (a template that may meet one is chosen by a when with known(..))".to_string()),
            other => Err(format!("our software has a bug: a template renders the value {other}, which no formula row spells")),
        }
    }
}

/// What a template is rendered against: the operands' texts (and their texts in forms, rendered before), the names of
/// the variables each argument introduces, the node's own spelling, the references' values, the parameters, the site's
/// constants, the size bounds, the kingdom's lexicon, and a variable spelling's ordinal. The adapters a `{call ..}`
/// renders are recorded in `called`.
pub struct Scope<'a> {
    pub operands: Vec<String>,
    /// The first input of the node's own data type, for `{in.own}`.
    pub own: Option<u32>,
    pub forms: BTreeMap<(u32, String), String>,
    pub bound: Vec<Vec<String>>,
    pub spelling: Option<&'a Template>,
    pub values: &'a dyn Fn(&Reference) -> Option<PropertyExpr>,
    pub params: &'a [PropertyExpr],
    pub zero: Option<String>,
    pub constant: Option<String>,
    pub bounds: &'a SizeBounds,
    pub lexicon: &'a Lexicon,
    pub nth: Option<String>,
    pub called: RefCell<Vec<String>>,
}

impl<'a> Scope<'a> {
    /// A scope with no operand, spelling, parameter, constant or ordinal.
    pub fn new(values: &'a dyn Fn(&Reference) -> Option<PropertyExpr>, bounds: &'a SizeBounds, lexicon: &'a Lexicon) -> Scope<'a> {
        Scope { operands: Vec::new(), own: None, forms: BTreeMap::new(), bound: Vec::new(), spelling: None, values, params: &[], zero: None, constant: None, bounds, lexicon, nth: None, called: RefCell::new(Vec::new()) }
    }
}

/// The operands a template reads in a form, `(k, form)`, so a renderer renders them before the scope is made.
pub fn forms_of(t: &Template) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    fn go(t: &Template, out: &mut Vec<(u32, String)>) {
        for p in t {
            match p {
                Piece::Operand { k, form: Some(f) } if !out.contains(&(*k, f.clone())) => out.push((*k, f.clone())),
                Piece::Call { args, .. } => args.iter().for_each(|a| go(a, out)),
                Piece::SelfWith(r) => r.iter().for_each(|(_, a)| go(a, out)),
                Piece::If { then, otherwise, .. } => {
                    go(then, out);
                    go(otherwise, out);
                }
                Piece::For { body, .. } => go(body, out),
                _ => {}
            }
        }
    }
    go(t, &mut out);
    out
}

/// The values a template's executable pieces give (the interpreter's values, the cost's properties): what each
/// executable piece reads, and what a call of an adapter symbol computes. [`Scope::exec`] walks the template.
pub trait Exec {
    type V: Clone;
    /// Operand k's value (as earlier rewrites left it).
    fn operand(&mut self, k: u32) -> Result<Self::V, String>;
    /// A formula's value (already read in the scope, in its one spelling).
    fn value(&mut self, e: &PropertyExpr) -> Result<Self::V, String>;
    /// The site's element constant (`{zero}`).
    fn zero(&mut self) -> Result<Self::V, String>;
    /// The site's constant (`{const}`).
    fn constant(&mut self) -> Result<Self::V, String>;
    /// The node itself (`{self}`, a root repair's result).
    fn whole(&mut self) -> Result<Self::V, String>;
    /// The adapter symbol `symbol` applied to `args`.
    fn call(&mut self, symbol: &str, args: Vec<Self::V>) -> Result<Self::V, String>;
}

impl Scope<'_> {
    /// The value a reference reads here: an index from `indices` (a template's formula reads a `for` index as a name), a
    /// size name `p<k>` as parameter k, else the scope's.
    fn read(&self, r: &Reference, indices: &BTreeMap<String, i64>) -> Option<PropertyExpr> {
        match r {
            Reference::Index(i) => indices.get(*i).map(|v| PropertyExpr::Literal(*v)).or_else(|| (self.values)(r)),
            Reference::Size(n) if indices.contains_key(*n) => indices.get(*n).map(|v| PropertyExpr::Literal(*v)),
            Reference::Size(n) => n.strip_prefix('p').and_then(|d| d.parse::<usize>().ok()).and_then(|k| self.params.get(k)).cloned().or_else(|| (self.values)(r)),
            other => (self.values)(other),
        }
    }

    /// The value of `e` here.
    pub fn eval(&self, e: &PropertyExpr, indices: &BTreeMap<String, i64>) -> PropertyExpr {
        e.substitute(&|r| self.read(r, indices), self.bounds)
    }

    /// Whether `c` holds here.
    pub fn holds(&self, c: &Condition, indices: &BTreeMap<String, i64>) -> bool {
        c.substitute(&|r| self.read(r, indices), self.bounds).truth(self.bounds) == Truth::Holds
    }

    /// Render `t` here; `fresh` numbers the fresh names of the program.
    pub fn render(&self, t: &Template, fresh: &mut Fresh) -> Result<String, String> {
        self.render_in(t, fresh, &mut BTreeMap::new(), &BTreeMap::new())
    }

    /// Render `t` with the indices `indices` bound (a variable's spelling reads `each` and `depth` this way).
    pub fn render_with(&self, t: &Template, fresh: &mut Fresh, indices: &BTreeMap<String, i64>) -> Result<String, String> {
        self.render_in(t, fresh, &mut BTreeMap::new(), indices)
    }

    /// The adapters the renderings so far called, in order, the list emptied.
    pub fn take_called(&self) -> Vec<String> {
        std::mem::take(&mut *self.called.borrow_mut())
    }

    fn render_in(&self, t: &Template, fresh: &mut Fresh, mine: &mut BTreeMap<String, String>, indices: &BTreeMap<String, i64>) -> Result<String, String> {
        let mut s = String::new();
        for piece in t {
            match piece {
                Piece::Text(x) => s.push_str(x),
                Piece::Operand { k, form: None } => s.push_str(self.operands.get(*k as usize).ok_or_else(|| format!("our software has a bug: a template reads operand {k}, which the node has not"))?),
                Piece::Operand { k, form: Some(f) } => s.push_str(self.forms.get(&(*k, f.clone())).ok_or_else(|| format!("our software has a bug: a template reads operand {k} in the form {f}, which was not rendered"))?),
                Piece::Bound { k, j } => s.push_str(self.bound.get(*k as usize).and_then(|b| b.get(*j as usize)).ok_or_else(|| format!("our software has a bug: a template reads the variable {j} argument {k} introduces, which it does not"))?),
                Piece::Value { formula, form } => s.push_str(&self.lexicon.spell(&self.eval(formula, indices), form)?),
                Piece::Nth => s.push_str(self.nth.as_deref().ok_or("our software has a bug: {nth} is read outside a variable's spelling")?),
                Piece::Param(k) => {
                    let e = self.params.get(*k as usize).ok_or_else(|| format!("our software has a bug: a template reads parameter {k}, which the call has not"))?;
                    s.push_str(&self.lexicon.spell(&self.eval(e, indices), "")?);
                }
                Piece::Zero => s.push_str(self.zero.as_deref().ok_or("our software has a bug: a template reads {zero}, and the site's data type has no constant")?),
                Piece::Const => s.push_str(self.constant.as_deref().ok_or("our software has a bug: a template reads {const}, and the site's data type has no constant")?),
                Piece::Input(k) => s.push_str(self.operands.get(*k as usize).ok_or_else(|| format!("our software has a bug: a template reads in{k}, which the node has not"))?),
                Piece::Own => {
                    let k = self.own.ok_or("our software has a bug: a template reads {in.own}, and the node has no input of its own data type")?;
                    s.push_str(self.operands.get(k as usize).ok_or("our software has a bug: {in.own} names an input the node has not")?);
                }
                Piece::Fresh(name) => {
                    let n = match mine.get(name) {
                        Some(n) => n.clone(),
                        None => {
                            let n = fresh.make(name);
                            mine.insert(name.clone(), n.clone());
                            n
                        }
                    };
                    s.push_str(&n);
                }
                Piece::SelfWith(repl) => {
                    let spelling = self.spelling.ok_or("our software has a bug: a template reads {self}, and the site has no spelling")?;
                    let mut operands = self.operands.clone();
                    for (k, sub) in repl {
                        let text = self.render_in(sub, fresh, mine, indices)?;
                        *operands.get_mut(*k as usize).ok_or_else(|| format!("our software has a bug: {{self}} replaces operand {k}, which the node has not"))? = text;
                    }
                    let inner = Scope {
                        operands,
                        own: self.own,
                        forms: self.forms.clone(),
                        bound: self.bound.clone(),
                        spelling: None,
                        values: self.values,
                        params: self.params,
                        zero: self.zero.clone(),
                        constant: self.constant.clone(),
                        bounds: self.bounds,
                        lexicon: self.lexicon,
                        nth: self.nth.clone(),
                        called: RefCell::new(Vec::new()),
                    };
                    s.push_str(&inner.render_in(spelling, fresh, mine, indices)?);
                    self.called.borrow_mut().extend(inner.take_called());
                }
                Piece::Call { symbol, args } => {
                    let (spelling, _) = self.lexicon.adapters.get(symbol).ok_or_else(|| format!("our software has a bug: a template calls {symbol}, which is not an adapter symbol of the kingdom"))?;
                    let mut texts = Vec::with_capacity(args.len());
                    for a in args {
                        texts.push(match a.as_slice() {
                            [Piece::Value { formula, form }] => self.lexicon.operand(&self.eval(formula, indices), form)?,
                            _ => self.render_in(a, fresh, mine, indices)?,
                        });
                    }
                    let mut out = String::new();
                    for p in spelling {
                        match p {
                            Piece::Text(x) => out.push_str(x),
                            Piece::Operand { k, form: None } => out.push_str(texts.get(*k as usize).ok_or_else(|| format!("our software has a bug: {symbol}'s spelling reads operand {k}, and the call gives {}", texts.len()))?),
                            other => return Err(format!("our software has a bug: the adapter {symbol}'s spelling holds {other:?}; an adapter is spelled by its operands alone")),
                        }
                    }
                    self.called.borrow_mut().push(symbol.clone());
                    s.push_str(&out);
                }
                Piece::If { cond, then, otherwise } => {
                    let branch = if self.holds(cond, indices) { then } else { otherwise };
                    s.push_str(&self.render_in(branch, fresh, mine, indices)?);
                }
                Piece::For { index, count, sep, body } => {
                    let n = match self.eval(count, indices) {
                        PropertyExpr::Literal(n) => n.max(0),
                        other => return Err(format!("our software has a bug: a template's for counts {other}, which is not a literal here")),
                    };
                    let mut parts = Vec::new();
                    for i in 0..n {
                        let mut ix = indices.clone();
                        ix.insert(index.clone(), i);
                        parts.push(self.render_in(body, fresh, mine, &ix)?);
                    }
                    s.push_str(&parts.join(sep));
                }
            }
        }
        Ok(s)
    }

    /// EXECUTE a repair's template over values (the interpreter's, the cost's): it must be one executable piece, an
    /// operand, an input, a formula, `{zero}`, `{const}`, `{self}`, a `{call ..}` of executable arguments, or an `{if ..}`
    /// whose branches are (decided as the renderer decides it). Text is not executable: a bug.
    pub fn exec<X: Exec>(&self, t: &Template, x: &mut X) -> Result<X::V, String> {
        let empty = BTreeMap::new();
        let pieces: Vec<&Piece> = t.iter().filter(|p| !matches!(p, Piece::Text(s) if s.trim().is_empty())).collect();
        let [piece] = pieces.as_slice() else { return Err(format!("our software has a bug: the template {t:?} is not one executable piece")) };
        match piece {
            Piece::Operand { k, form: None } | Piece::Input(k) => x.operand(*k),
            Piece::Own => x.operand(self.own.ok_or("our software has a bug: {in.own} executed at a node with no input of its own data type")?),
            Piece::Value { formula, .. } => x.value(&self.eval(formula, &empty)),
            Piece::Zero => x.zero(),
            Piece::Const => x.constant(),
            Piece::SelfWith(r) if r.is_empty() => x.whole(),
            Piece::Call { symbol, args } => {
                let mut vs = Vec::with_capacity(args.len());
                for a in args {
                    vs.push(self.exec(a, x)?);
                }
                x.call(symbol, vs)
            }
            Piece::If { cond, then, otherwise } => self.exec(if self.holds(cond, &empty) { then } else { otherwise }, x),
            other => Err(format!("our software has a bug: {other:?} is not executable (an executable repair is made of operands, formulas, constants, self and calls)")),
        }
    }
}

#[cfg(test)]
mod tests;
