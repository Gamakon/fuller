//! THE TEMPLATE ENGINE of pass 5 (design section 8.14.2): a symbol's Spelling, an adapter's AdapterTemplate, a
//! binding's name and a transform's text are templates over one CLOSED list of placeholders, rendered by this one
//! kingdom-free engine. A kingdom never writes renderer code for an adapter; it writes a template row.
//!
//! The placeholders:
//!
//! | placeholder | renders |
//! |---|---|
//! | `{k}` | operand k's text |
//! | `{k@form}` | operand k in the form `form` (a Spelling form, passed to the operand's head argument when it lacks one) |
//! | `{=formula}` | a formula's value; in a template `p<k>` names repair parameter k, `each` a binding's ordinal, `depth` its depth |
//! | `{name}` | shorthand for `{=name}` (`{each}`, `{depth}`) |
//! | `{p0}` | repair parameter 0's value |
//! | `{zero}`, `{const}` | the site's data type's constant spelling (`{zero}`: its element's, for an array) |
//! | `{in<k>}`, `{in.own}` | input k's text; the first input of the node's own data type |
//! | `{fresh name}` | `name` made unique by a counter over the program; the same within one template |
//! | `{self}`, `{self(k := template, ...)}` | the node's own spelling, with operands replaced |
//! | `{if condition : template else template}` | the first template where the condition holds, else the second |
//! | `{for i < n sep text : template}` | the template for `i = 0 .. n-1`, joined by `text` (` sep text` optional) |
//!
//! `{{` and `}}` write a literal brace. A repair's template that is ONE substitute (`{in<k>}`, `{in.own}`, `{const}`)
//! replaces its node ([`substitute_of`]); any other template wraps or rewrites.

use std::collections::BTreeMap;

use crate::geneframe::formula::{is_identifier, Condition, PropertyExpr, Reference, SizeBounds, Truth};

/// One piece of a parsed template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Text(String),
    Operand { k: u32, form: Option<String> },
    Value(PropertyExpr),
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

fn placeholder(text: &str, p: &str) -> Result<Piece, String> {
    if p.is_empty() {
        return Err(bad(text, "has an empty placeholder {}"));
    }
    if let Some(f) = p.strip_prefix('=') {
        return Ok(Piece::Value(PropertyExpr::parse(f)?));
    }
    let (head, form) = match p.split_once('@') {
        Some((h, f)) if h.bytes().all(|b| b.is_ascii_digit()) => (h, Some(f.to_string())),
        _ => (p, None),
    };
    if !head.is_empty() && head.bytes().all(|b| b.is_ascii_digit()) {
        return head.parse().map(|k| Piece::Operand { k, form }).map_err(|_| bad(text, "has an operand number out of range"));
    }
    if let Some(d) = p.strip_prefix('p').filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())) {
        return d.parse().map(Piece::Param).map_err(|_| bad(text, "has a parameter number out of range"));
    }
    match p {
        "zero" => return Ok(Piece::Zero),
        "const" => return Ok(Piece::Const),
        "self" => return Ok(Piece::SelfWith(Vec::new())),
        "in.own" => return Ok(Piece::Own),
        _ => {}
    }
    if let Some(d) = p.strip_prefix("in").filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())) {
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
        let mut rest = args;
        while !rest.trim().is_empty() {
            let end = top_level(rest, ",").unwrap_or(rest.len());
            let (k, t) = rest[..end].split_once(":=").ok_or_else(|| bad(text, "has a self( argument that is not k := template"))?;
            let k: u32 = k.trim().parse().map_err(|_| bad(text, "has a self( argument whose operand is not a number"))?;
            repl.push((k, parse_template(t.trim())?));
            rest = if end < rest.len() { &rest[end + 1..] } else { "" };
        }
        return Ok(Piece::SelfWith(repl));
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
        return Ok(Piece::Value(PropertyExpr::parse(p)?));
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

/// What a template is rendered against: the operands' texts, a way to render an operand in a form, the node's own
/// spelling, the references' values, the parameters, the site's constants and the size bounds.
pub struct Scope<'a> {
    pub operands: Vec<String>,
    /// The first input of the node's own data type, for `{in.own}`.
    pub own: Option<u32>,
    pub in_form: &'a dyn Fn(u32, &str) -> Result<String, String>,
    pub spelling: Option<&'a Template>,
    pub values: &'a dyn Fn(&Reference) -> Option<PropertyExpr>,
    pub params: &'a [PropertyExpr],
    pub zero: Option<String>,
    pub constant: Option<String>,
    pub bounds: &'a SizeBounds,
}

/// A formula's value as text: an integer, a tag's name, a flag, a size name; anything symbolic as the formula's
/// spelling.
pub fn value_text(v: &PropertyExpr) -> String {
    match v {
        PropertyExpr::Tag(t) => t.clone(),
        other => other.to_string(),
    }
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
    fn eval(&self, e: &PropertyExpr, indices: &BTreeMap<String, i64>) -> PropertyExpr {
        e.substitute(&|r| self.read(r, indices), self.bounds)
    }

    fn holds(&self, c: &Condition, indices: &BTreeMap<String, i64>) -> bool {
        c.substitute(&|r| self.read(r, indices), self.bounds).truth(self.bounds) == Truth::Holds
    }

    /// Render `t` here; `fresh` numbers the fresh names of the program, `mine` those of this template.
    pub fn render(&self, t: &Template, fresh: &mut Fresh) -> Result<String, String> {
        self.render_in(t, fresh, &mut BTreeMap::new(), &BTreeMap::new())
    }

    /// Render `t` with the indices `indices` bound (a variable's spelling reads `each` and `depth` this way).
    pub fn render_with(&self, t: &Template, fresh: &mut Fresh, indices: &BTreeMap<String, i64>) -> Result<String, String> {
        self.render_in(t, fresh, &mut BTreeMap::new(), indices)
    }

    fn render_in(&self, t: &Template, fresh: &mut Fresh, mine: &mut BTreeMap<String, String>, indices: &BTreeMap<String, i64>) -> Result<String, String> {
        let mut s = String::new();
        for piece in t {
            match piece {
                Piece::Text(x) => s.push_str(x),
                Piece::Operand { k, form: None } => s.push_str(self.operands.get(*k as usize).ok_or_else(|| format!("our software has a bug: a template reads operand {k}, which the node has not"))?),
                Piece::Operand { k, form: Some(f) } => s.push_str(&(self.in_form)(*k, f)?),
                Piece::Value(e) => s.push_str(&value_text(&self.eval(e, indices))),
                Piece::Param(k) => s.push_str(&value_text(&self.eval(self.params.get(*k as usize).ok_or_else(|| format!("our software has a bug: a template reads parameter {k}, which the call has not"))?, indices))),
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
                    let inner = Scope { operands, own: self.own, in_form: self.in_form, spelling: None, values: self.values, params: self.params, zero: self.zero.clone(), constant: self.constant.clone(), bounds: self.bounds };
                    s.push_str(&inner.render_in(spelling, fresh, mine, indices)?);
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
}

#[cfg(test)]
mod tests;
