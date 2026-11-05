//! THE FORMULA LANGUAGE of the gene frame (design `docs/DESIGN_futhark_types_and_properties.md` section 3.4): a
//! CLOSED list of terms, operators and conditions that every kingdom's rows are written in. It is the same for every
//! kingdom; a kingdom never adds an operator (Andrew Morgan, 2026-11-05: the schema is fixed, a kingdom is rows).
//!
//! A **formula** ([`PropertyExpr`], also called [`Formula`]) gives a value; a **condition** ([`Condition`]) gives
//! true, false or undecided ([`Truth`]). Every formula has ONE spelling (its normal form), so symbolic equality is
//! equality of spellings: a sum is a polynomial (monomials sorted, like monomials' coefficients added, literals
//! added), a product is distributed over sums, `min` and `max` are flattened, sorted and without repeats with one
//! literal kept, a decided `if` is its branch, a `for` of a literal count is expanded. [`PropertyExpr::normal`] and
//! [`PropertyExpr::parse`] build only that form; [`PropertyExpr::shape_defects`] reports one built by hand in another.
//!
//! References: `in<k>.<p>` (a property of input k), `out.<p>` (another property of the node itself), `env.<a>` (an
//! attribute of the node's environment, or of the site's in a guard), `each` (a binding's ordinal) and a
//! comprehension's index. A size name (`n`) is a value of the entry's SizeName table, whose lower bound `<=` reads
//! ([`SizeBounds`]) instead of assuming every size is at least 1.
//!
//! Comparison (3.4.4, fixed here): `?` anywhere makes a comparison undecided; `=` holds on identical spellings and
//! fails on two different literals; `<=` holds by the rules of [`Relation::truth`] and fails on two literals.

use std::collections::BTreeMap;
use std::fmt;

/// The lower bound of each size name of an entry (its SizeName rows): what `<=` reads, never a built-in `n >= 1`. A
/// size name with no row has no known lower bound.
pub type SizeBounds = BTreeMap<String, i64>;

/// A comparison's relation: `=`, `<=`, `<`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Relation {
    Eq,
    Le,
    Lt,
}

/// The three values of a condition (3.4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Truth {
    Holds,
    Fails,
    Undecided,
}

impl Truth {
    /// True only when the condition is decided true: a requirement HOLDS only then.
    pub fn holds(self) -> bool {
        self == Truth::Holds
    }

    fn of(b: bool) -> Truth {
        if b {
            Truth::Holds
        } else {
            Truth::Fails
        }
    }

    fn not(self) -> Truth {
        match self {
            Truth::Holds => Truth::Fails,
            Truth::Fails => Truth::Holds,
            Truth::Undecided => Truth::Undecided,
        }
    }

    /// Kleene's `and`: fails if either fails, holds if both hold.
    fn and(self, o: Truth) -> Truth {
        match (self, o) {
            (Truth::Fails, _) | (_, Truth::Fails) => Truth::Fails,
            (Truth::Holds, Truth::Holds) => Truth::Holds,
            _ => Truth::Undecided,
        }
    }

    /// Kleene's `or`: holds if either holds, fails if both fail.
    fn or(self, o: Truth) -> Truth {
        self.not().and(o.not()).not()
    }
}

/// A reference a formula reads, as the decoder resolves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reference<'a> {
    /// `in<k>.<p>`.
    Input { position: u32, property: &'a str },
    /// `out.<p>`.
    Output(&'a str),
    /// `env.<a>`.
    Env(&'a str),
    /// `each`, or a comprehension's index.
    Index(&'a str),
}

impl fmt::Display for Reference<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reference::Input { position, property } => write!(f, "in{position}.{property}"),
            Reference::Output(p) => write!(f, "out.{p}"),
            Reference::Env(a) => write!(f, "env.{a}"),
            Reference::Index(i) => write!(f, "{i}"),
        }
    }
}

/// A formula: a property's value as the decoder knows it before run time (section 3.4.1 and 3.4.2). The first six
/// variants are the first version's, in its order, so values written in it sort and spell as before.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropertyExpr {
    /// A size name of the entry (`n`, `k`).
    Size(String),
    /// An integer literal.
    Literal(i64),
    /// `?`: known only at run time.
    Unknown,
    /// `in<k>.<p>`.
    Input { position: u32, property: String },
    /// `a + b + ...`: at least two terms, sorted, no two of one monomial, at most one literal (not 0), none a sum.
    Sum(Vec<PropertyExpr>),
    /// `min(a, b, ...)`: at least two arguments, sorted, no repeats, at most one literal, none a min.
    Min(Vec<PropertyExpr>),
    /// `max(a, b, ...)`: as a min.
    Max(Vec<PropertyExpr>),
    /// `a * b * ...`: at least two factors, a literal coefficient first if not 1 (never 0), the others sorted, none a
    /// product or a sum (a product is distributed over a sum).
    Product(Vec<PropertyExpr>),
    /// `if(c, a, b)`: `a` where `c` holds, `b` where it fails, itself while `c` is undecided.
    If { cond: Box<Condition>, then: Box<PropertyExpr>, otherwise: Box<PropertyExpr> },
    /// A tag literal: `'read_write'`.
    Tag(String),
    /// A flag literal: `true`, `false`.
    Flag(bool),
    /// `out.<p>`.
    Output(String),
    /// `env.<a>`.
    Env(String),
    /// `each` (a binding's ordinal) or a comprehension's index.
    Index(String),
    /// `list(a, b, ...)`.
    List(Vec<PropertyExpr>),
    /// `len(L)`.
    Len(Box<PropertyExpr>),
    /// `at(L, i, d)`: element `i` (from 0), `d` when `i` is out of range.
    At(Box<PropertyExpr>, Box<PropertyExpr>, Box<PropertyExpr>),
    /// `concat(L, M, ...)`: flattened, adjacent literal lists joined.
    Concat(Vec<PropertyExpr>),
    /// `find(L, x)`: the position of `x`, `-1` when absent, `?` when undecided.
    Find(Box<PropertyExpr>, Box<PropertyExpr>),
    /// `for(i < n : f)`: the list of `f` for `i = 0 .. n-1`, expanded when `n` is a literal.
    For { index: String, count: Box<PropertyExpr>, body: Box<PropertyExpr> },
    /// A condition used as a flag value (`env.uniform and in0.uniform`).
    Cond(Box<Condition>),
}

/// The formula language's value type, by its design name.
pub type Formula = PropertyExpr;

/// A condition (section 3.4.3): three-valued.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Condition {
    /// `a = b`, `a <= b`, `a < b`.
    Compare { relation: Relation, left: PropertyExpr, right: PropertyExpr },
    /// `x in L`.
    In { item: PropertyExpr, list: PropertyExpr },
    /// `known(a)`: `a` contains no `?`.
    Known(PropertyExpr),
    /// A flag-valued formula as a condition: `true`, `env.uniform`, `in0.uniform`.
    Flag(PropertyExpr),
    And(Vec<Condition>),
    Or(Vec<Condition>),
    Not(Box<Condition>),
    /// `all(i < n : c)`.
    All { index: String, count: PropertyExpr, body: Box<Condition> },
}

/// The words a size name or an index may not be: the closed list's own.
const RESERVED: [&str; 7] = ["and", "or", "not", "in", "true", "false", "each"];

pub(crate) fn is_identifier(s: &str) -> bool {
    let mut cs = s.chars();
    cs.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_name(s: &str) -> bool {
    is_identifier(s) && !RESERVED.contains(&s)
}

/// A term of a sum as `(coefficient, its other factors)`: `3` is `(3, [])`, `2 * n` is `(2, [n])`, `n` is `(1, [n])`.
fn monomial(t: PropertyExpr) -> (i64, Vec<PropertyExpr>) {
    match t {
        PropertyExpr::Literal(v) => (v, Vec::new()),
        PropertyExpr::Product(mut fs) => match fs.first() {
            Some(PropertyExpr::Literal(c)) => {
                let c = *c;
                fs.remove(0);
                (c, fs)
            }
            _ => (1, fs),
        },
        other => (1, vec![other]),
    }
}

/// The term of coefficient `c` and factors `fs` (sorted, no literal), in its one spelling.
fn term(c: i64, mut fs: Vec<PropertyExpr>) -> PropertyExpr {
    match (c, fs.len()) {
        (_, 0) => PropertyExpr::Literal(c),
        (0, _) => PropertyExpr::Literal(0),
        (1, 1) => fs.remove(0),
        (1, _) => PropertyExpr::Product(fs),
        _ => {
            fs.insert(0, PropertyExpr::Literal(c));
            PropertyExpr::Product(fs)
        }
    }
}

/// True when the formula is a literal value: an integer, a tag, a flag, or a list of literals.
fn is_literal(e: &PropertyExpr) -> bool {
    match e {
        PropertyExpr::Literal(_) | PropertyExpr::Tag(_) | PropertyExpr::Flag(_) => true,
        PropertyExpr::List(xs) => xs.iter().all(is_literal),
        _ => false,
    }
}

impl PropertyExpr {
    /// The sum of `terms` in its one spelling: a polynomial. Nested sums are spliced in; each term is split into its
    /// literal coefficient and its other factors; terms of one monomial add their coefficients and a zero term is
    /// dropped; the terms are sorted. A term that holds `?` is never merged (two unknowns are not one). The sum of no
    /// term is 0.
    pub fn sum(terms: Vec<PropertyExpr>) -> PropertyExpr {
        let mut flat = Vec::new();
        for t in terms {
            match t {
                PropertyExpr::Sum(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        let mut opaque: Vec<PropertyExpr> = Vec::new();
        let mut monomials: BTreeMap<Vec<PropertyExpr>, i64> = BTreeMap::new();
        for t in flat {
            if t.has_unknown() {
                opaque.push(t);
                continue;
            }
            let (c, fs) = monomial(t);
            let e = monomials.entry(fs).or_insert(0);
            *e = e.saturating_add(c);
        }
        let mut out: Vec<PropertyExpr> = monomials.into_iter().filter(|(_, c)| *c != 0).map(|(fs, c)| term(c, fs)).collect();
        out.extend(opaque);
        out.sort();
        match out.len() {
            0 => PropertyExpr::Literal(0),
            1 => out.remove(0),
            _ => PropertyExpr::Sum(out),
        }
    }

    /// The product of `factors` in its one spelling: nested products spliced in, distributed over a sum, literal
    /// factors multiplied into one coefficient (0 makes the product 0), the other factors sorted.
    pub fn product(factors: Vec<PropertyExpr>) -> PropertyExpr {
        let mut flat = Vec::new();
        for f in factors {
            match f {
                PropertyExpr::Product(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if let Some(i) = flat.iter().position(|f| matches!(f, PropertyExpr::Sum(_))) {
            let PropertyExpr::Sum(terms) = flat.remove(i) else { unreachable!("a sum at the position just found") };
            return PropertyExpr::sum(terms.into_iter().map(|t| PropertyExpr::product(flat.iter().cloned().chain(std::iter::once(t)).collect())).collect());
        }
        let mut c: i64 = 1;
        let mut rest = Vec::new();
        for f in flat {
            match f {
                PropertyExpr::Literal(v) => c = c.saturating_mul(v),
                other => rest.push(other),
            }
        }
        rest.sort();
        term(c, rest)
    }

    /// The least of `args` in its one spelling: nested mins spliced in, the least literal kept of several, sorted,
    /// without repeats; a single argument is itself. The min of no argument is `?` (no table writes one).
    pub fn min(args: Vec<PropertyExpr>) -> PropertyExpr {
        PropertyExpr::extremum(args, true)
    }

    /// The greatest of `args` in its one spelling, as [`PropertyExpr::min`].
    pub fn max(args: Vec<PropertyExpr>) -> PropertyExpr {
        PropertyExpr::extremum(args, false)
    }

    fn extremum(args: Vec<PropertyExpr>, least: bool) -> PropertyExpr {
        let mut flat = Vec::new();
        for a in args {
            match a {
                PropertyExpr::Min(inner) if least => flat.extend(inner),
                PropertyExpr::Max(inner) if !least => flat.extend(inner),
                other => flat.push(other),
            }
        }
        let literals = flat.iter().filter_map(|a| if let PropertyExpr::Literal(v) = a { Some(*v) } else { None });
        let keep = if least { literals.min() } else { literals.max() };
        if let Some(k) = keep {
            flat.retain(|a| !matches!(a, PropertyExpr::Literal(v) if *v != k));
        }
        flat.sort();
        flat.dedup();
        match flat.len() {
            0 => PropertyExpr::Unknown,
            1 => flat.remove(0),
            _ if least => PropertyExpr::Min(flat),
            _ => PropertyExpr::Max(flat),
        }
    }

    /// `if(cond, then, otherwise)` in its one spelling, deciding `cond` with no size bounds.
    pub fn when(cond: Condition, then: PropertyExpr, otherwise: PropertyExpr) -> PropertyExpr {
        PropertyExpr::If { cond: Box::new(cond), then: Box::new(then), otherwise: Box::new(otherwise) }.normal()
    }

    /// The formula in its one spelling, at every depth, deciding nothing that needs a size's lower bound.
    pub fn normal(&self) -> PropertyExpr {
        self.normal_in(&SizeBounds::new())
    }

    /// The formula in its one spelling, at every depth, with `bounds` the sizes' lower bounds.
    pub fn normal_in(&self, b: &SizeBounds) -> PropertyExpr {
        use PropertyExpr as P;
        let all = |xs: &[P]| xs.iter().map(|x| x.normal_in(b)).collect::<Vec<_>>();
        match self {
            P::Sum(xs) => P::sum(all(xs)),
            P::Product(xs) => P::product(all(xs)),
            P::Min(xs) => P::min(all(xs)),
            P::Max(xs) => P::max(all(xs)),
            P::If { cond, then, otherwise } => {
                let (c, t, o) = (cond.normal_in(b), then.normal_in(b), otherwise.normal_in(b));
                match c.truth(b) {
                    Truth::Holds => t,
                    Truth::Fails => o,
                    Truth::Undecided if t == o => t,
                    Truth::Undecided => P::If { cond: Box::new(c), then: Box::new(t), otherwise: Box::new(o) },
                }
            }
            P::List(xs) => P::List(all(xs)),
            P::Len(l) => match l.normal_in(b) {
                P::List(xs) => P::Literal(xs.len() as i64),
                P::Unknown => P::Unknown,
                other => P::Len(Box::new(other)),
            },
            P::At(l, i, d) => {
                let (l, i, d) = (l.normal_in(b), i.normal_in(b), d.normal_in(b));
                match (&l, &i) {
                    (P::List(xs), P::Literal(k)) => usize::try_from(*k).ok().and_then(|k| xs.get(k)).cloned().unwrap_or(d),
                    (P::Unknown, _) | (_, P::Unknown) => P::Unknown,
                    _ => P::At(Box::new(l), Box::new(i), Box::new(d)),
                }
            }
            P::Concat(xs) => {
                let mut flat: Vec<P> = Vec::new();
                for x in all(xs) {
                    match x {
                        P::Concat(inner) => flat.extend(inner),
                        other => flat.push(other),
                    }
                }
                if flat.contains(&P::Unknown) {
                    return P::Unknown;
                }
                let mut joined: Vec<P> = Vec::new();
                for x in flat {
                    match (joined.last_mut(), x) {
                        (Some(P::List(a)), P::List(c)) => a.extend(c),
                        (_, x) => joined.push(x),
                    }
                }
                match joined.len() {
                    0 => P::List(Vec::new()),
                    1 => joined.remove(0),
                    _ => P::Concat(joined),
                }
            }
            P::Find(l, x) => {
                let (l, x) = (l.normal_in(b), x.normal_in(b));
                if let P::List(xs) = &l {
                    for (i, y) in xs.iter().enumerate() {
                        match Relation::Eq.truth(y, &x, b) {
                            Truth::Holds => return P::Literal(i as i64),
                            Truth::Fails => continue,
                            Truth::Undecided => return if l.is_ground() && x.is_ground() { P::Unknown } else { P::Find(Box::new(l), Box::new(x)) },
                        }
                    }
                    return P::Literal(-1);
                }
                if l == P::Unknown || x == P::Unknown || (l.is_ground() && x.is_ground()) {
                    P::Unknown
                } else {
                    P::Find(Box::new(l), Box::new(x))
                }
            }
            P::For { index, count, body } => {
                let (count, body) = (count.normal_in(b), body.normal_in(b));
                match count {
                    P::Literal(n) => P::List((0..n.max(0)).map(|k| body.bind(index, &P::Literal(k)).normal_in(b)).collect()),
                    count => P::For { index: index.clone(), count: Box::new(count), body: Box::new(body) },
                }
            }
            P::Cond(c) => match c.normal_in(b) {
                Condition::Flag(f) => f,
                other => P::Cond(Box::new(other)),
            },
            P::Size(_) | P::Literal(_) | P::Unknown | P::Input { .. } | P::Tag(_) | P::Flag(_) | P::Output(_) | P::Env(_) | P::Index(_) => self.clone(),
        }
    }

    /// The formula with the index `name` (not shadowed by an inner `for` or `all`) replaced by `value`, not normalised.
    fn bind(&self, name: &str, value: &PropertyExpr) -> PropertyExpr {
        self.map_refs(&|r| matches!(r, Reference::Index(i) if *i == name).then(|| value.clone()), &mut Vec::new())
    }

    /// Every reference `f` resolves replaced by its value, the result in its one spelling with `bounds`.
    pub fn substitute(&self, f: &dyn Fn(&Reference) -> Option<PropertyExpr>, bounds: &SizeBounds) -> PropertyExpr {
        self.map_refs(f, &mut Vec::new()).normal_in(bounds)
    }

    /// Every `in<k>.<p>` replaced by `value(k, p)`, the result in its one spelling (no size bounds).
    pub fn substitute_inputs(&self, value: &dyn Fn(u32, &str) -> PropertyExpr) -> PropertyExpr {
        self.substitute(&|r| if let Reference::Input { position, property } = r { Some(value(*position, property)) } else { None }, &SizeBounds::new())
    }

    /// The formula with each reference `f` resolves replaced, NOT normalised; `shadow` holds the comprehension indices
    /// bound around the current point, which a reference to an index never escapes.
    fn map_refs(&self, f: &dyn Fn(&Reference) -> Option<PropertyExpr>, shadow: &mut Vec<String>) -> PropertyExpr {
        use PropertyExpr as P;
        let all = |xs: &[P], shadow: &mut Vec<String>| xs.iter().map(|x| x.map_refs(f, shadow)).collect::<Vec<_>>();
        let reference = match self {
            P::Input { position, property } => Some(Reference::Input { position: *position, property }),
            P::Output(p) => Some(Reference::Output(p)),
            P::Env(a) => Some(Reference::Env(a)),
            P::Index(i) if !shadow.contains(i) => Some(Reference::Index(i)),
            _ => None,
        };
        if let Some(r) = reference {
            return f(&r).unwrap_or_else(|| self.clone());
        }
        match self {
            P::Sum(xs) => P::Sum(all(xs, shadow)),
            P::Product(xs) => P::Product(all(xs, shadow)),
            P::Min(xs) => P::Min(all(xs, shadow)),
            P::Max(xs) => P::Max(all(xs, shadow)),
            P::List(xs) => P::List(all(xs, shadow)),
            P::Concat(xs) => P::Concat(all(xs, shadow)),
            P::If { cond, then, otherwise } => P::If { cond: Box::new(cond.map_refs(f, shadow)), then: Box::new(then.map_refs(f, shadow)), otherwise: Box::new(otherwise.map_refs(f, shadow)) },
            P::Len(l) => P::Len(Box::new(l.map_refs(f, shadow))),
            P::At(l, i, d) => P::At(Box::new(l.map_refs(f, shadow)), Box::new(i.map_refs(f, shadow)), Box::new(d.map_refs(f, shadow))),
            P::Find(l, x) => P::Find(Box::new(l.map_refs(f, shadow)), Box::new(x.map_refs(f, shadow))),
            P::For { index, count, body } => {
                let count = count.map_refs(f, shadow);
                shadow.push(index.clone());
                let body = body.map_refs(f, shadow);
                shadow.pop();
                P::For { index: index.clone(), count: Box::new(count), body: Box::new(body) }
            }
            P::Cond(c) => P::Cond(Box::new(c.map_refs(f, shadow))),
            P::Size(_) | P::Literal(_) | P::Unknown | P::Input { .. } | P::Tag(_) | P::Flag(_) | P::Output(_) | P::Env(_) | P::Index(_) => self.clone(),
        }
    }

    /// The formula's direct sub-formulas (a condition's formulas included).
    fn parts(&self) -> Vec<&PropertyExpr> {
        use PropertyExpr as P;
        match self {
            P::Sum(xs) | P::Product(xs) | P::Min(xs) | P::Max(xs) | P::List(xs) | P::Concat(xs) => xs.iter().collect(),
            P::If { cond, then, otherwise } => {
                let mut v = cond.formulas();
                v.extend([then.as_ref(), otherwise.as_ref()]);
                v
            }
            P::Len(l) => vec![l],
            P::At(l, i, d) => vec![l, i, d],
            P::Find(l, x) => vec![l, x],
            P::For { count, body, .. } => vec![count, body],
            P::Cond(c) => c.formulas(),
            P::Size(_) | P::Literal(_) | P::Unknown | P::Input { .. } | P::Tag(_) | P::Flag(_) | P::Output(_) | P::Env(_) | P::Index(_) => Vec::new(),
        }
    }

    /// True when `?` occurs anywhere in the formula.
    pub fn has_unknown(&self) -> bool {
        matches!(self, PropertyExpr::Unknown) || self.parts().into_iter().any(PropertyExpr::has_unknown)
    }

    /// True when a size name occurs anywhere in the formula.
    pub fn has_size_name(&self) -> bool {
        matches!(self, PropertyExpr::Size(_)) || self.parts().into_iter().any(PropertyExpr::has_size_name)
    }

    /// True when the formula reads no reference (no `in<k>.p`, `out.p`, `env.a` or free index): its value depends on
    /// the size names alone.
    pub fn is_ground(&self) -> bool {
        self.references().is_empty()
    }

    /// Every reference the formula reads, in reading order (a comprehension's own index is not one).
    pub fn references(&self) -> Vec<Reference<'_>> {
        let mut out = Vec::new();
        self.collect_refs(&mut Vec::new(), &mut out);
        out
    }

    fn collect_refs<'a>(&'a self, shadow: &mut Vec<&'a str>, out: &mut Vec<Reference<'a>>) {
        use PropertyExpr as P;
        match self {
            P::Input { position, property } => out.push(Reference::Input { position: *position, property }),
            P::Output(p) => out.push(Reference::Output(p)),
            P::Env(a) => out.push(Reference::Env(a)),
            P::Index(i) if !shadow.contains(&i.as_str()) => out.push(Reference::Index(i)),
            P::For { index, count, body } => {
                count.collect_refs(shadow, out);
                shadow.push(index);
                body.collect_refs(shadow, out);
                shadow.pop();
            }
            P::If { cond, then, otherwise } => {
                cond.collect_refs(shadow, out);
                then.collect_refs(shadow, out);
                otherwise.collect_refs(shadow, out);
            }
            P::Cond(cond) => cond.collect_refs(shadow, out),
            _ => self.parts().into_iter().for_each(|x| x.collect_refs(shadow, out)),
        }
    }

    /// Every `(input position, property)` the formula reads.
    pub fn inputs(&self) -> Vec<(u32, &str)> {
        self.references().into_iter().filter_map(|r| if let Reference::Input { position, property } = r { Some((position, property)) } else { None }).collect()
    }

    /// The least value the formula can take whatever the sizes are, from the sizes' lower bounds (`None`: none known).
    pub fn lower_bound(&self, b: &SizeBounds) -> Option<i64> {
        use PropertyExpr as P;
        match self {
            P::Size(n) => b.get(n).copied(),
            P::Literal(v) => Some(*v),
            P::Min(xs) => xs.iter().map(|x| x.lower_bound(b)).collect::<Option<Vec<i64>>>()?.into_iter().min(),
            P::Max(xs) => xs.iter().filter_map(|x| x.lower_bound(b)).max(),
            P::Sum(xs) => xs.iter().map(|x| x.lower_bound(b)).try_fold(0i64, |acc, v| v.map(|v| acc.saturating_add(v))),
            P::Product(xs) => {
                let lbs = xs.iter().map(|x| x.lower_bound(b)).collect::<Option<Vec<i64>>>()?;
                lbs.iter().all(|v| *v >= 0).then(|| lbs.iter().fold(1i64, |a, v| a.saturating_mul(*v)))
            }
            P::If { then, otherwise, .. } => Some(then.lower_bound(b)?.min(otherwise.lower_bound(b)?)),
            P::Len(_) => Some(0),
            _ => None,
        }
    }

    /// Parse the text [`fmt::Display`] writes, or any other spelling of it, into its one spelling. A condition read
    /// where a value is wanted is a flag value ([`PropertyExpr::Cond`]).
    pub fn parse(text: &str) -> Result<PropertyExpr, String> {
        let mut p = Parser::new(text);
        let e = p.value()?;
        p.end()?;
        Ok(e.normal())
    }

    /// What is malformed in this formula, at any depth: a value not in its one spelling, a size name or an index that
    /// is not a name, a property or attribute name that is not an identifier, or a tag that is empty.
    pub fn shape_defects(&self) -> Vec<String> {
        let mut v = Vec::new();
        let normal = self.normal();
        if *self != normal {
            v.push(format!("has {self}, which is not in its one spelling ({normal})"));
        }
        self.name_defects(&mut v);
        v
    }

    fn name_defects(&self, v: &mut Vec<String>) {
        use PropertyExpr as P;
        match self {
            P::Size(n) if !is_name(n) => v.push(format!("names {n:?}, which is not a name")),
            P::Index(n) if n != "each" && !is_name(n) => v.push(format!("names {n:?}, which is not a name")),
            P::Input { property: p, .. } | P::Output(p) | P::Env(p) if !is_identifier(p) => v.push(format!("reads the property {p:?}, which is not an identifier")),
            P::Tag(t) if t.is_empty() || t.contains(['\'', '\t', '\n']) => v.push(format!("has the tag {t:?}, which is empty or holds a quote")),
            P::For { index, .. } if !is_name(index) => v.push(format!("has the index {index:?}, which is not a name")),
            _ => {}
        }
        for x in self.parts() {
            x.name_defects(v);
        }
    }
}

/// Write a list of formulas separated by `, `.
fn join(f: &mut fmt::Formatter<'_>, xs: &[PropertyExpr]) -> fmt::Result {
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{x}")?;
    }
    Ok(())
}

/// A sum's term as it is written after a minus: `2` for the literal -2, `k` for `-1 * k`, `3 * k` for `-3 * k`; `None`
/// for a term that is not negative.
fn negated(t: &PropertyExpr) -> Option<String> {
    match t {
        PropertyExpr::Literal(v) if *v < 0 => Some(v.unsigned_abs().to_string()),
        PropertyExpr::Product(fs) => match fs.first() {
            Some(PropertyExpr::Literal(c)) if *c < 0 => {
                let rest: Vec<String> = fs[1..].iter().map(factor).collect();
                Some(if *c == -1 { rest.join(" * ") } else { format!("{} * {}", c.unsigned_abs(), rest.join(" * ")) })
            }
            _ => None,
        },
        _ => None,
    }
}

/// A factor of a product, parenthesised when it is a sum or a condition.
fn factor(x: &PropertyExpr) -> String {
    match x {
        PropertyExpr::Sum(_) | PropertyExpr::Cond(_) => format!("({x})"),
        other => other.to_string(),
    }
}

impl fmt::Display for PropertyExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PropertyExpr as P;
        match self {
            P::Size(n) | P::Index(n) => write!(f, "{n}"),
            P::Literal(v) => write!(f, "{v}"),
            P::Unknown => write!(f, "?"),
            P::Input { position, property } => write!(f, "in{position}.{property}"),
            P::Output(p) => write!(f, "out.{p}"),
            P::Env(a) => write!(f, "env.{a}"),
            P::Tag(t) => write!(f, "'{t}'"),
            P::Flag(b) => write!(f, "{b}"),
            P::Sum(terms) => {
                for (i, t) in terms.iter().enumerate() {
                    match (i, negated(t)) {
                        (0, _) => write!(f, "{}", factor(t))?,
                        (_, Some(n)) => write!(f, " - {n}")?,
                        (_, None) => write!(f, " + {}", factor(t))?,
                    }
                }
                Ok(())
            }
            P::Product(xs) => write!(f, "{}", xs.iter().map(factor).collect::<Vec<_>>().join(" * ")),
            P::Min(xs) => {
                write!(f, "min(")?;
                join(f, xs)?;
                write!(f, ")")
            }
            P::Max(xs) => {
                write!(f, "max(")?;
                join(f, xs)?;
                write!(f, ")")
            }
            P::If { cond, then, otherwise } => write!(f, "if({cond}, {then}, {otherwise})"),
            P::List(xs) => {
                write!(f, "list(")?;
                join(f, xs)?;
                write!(f, ")")
            }
            P::Len(l) => write!(f, "len({l})"),
            P::At(l, i, d) => write!(f, "at({l}, {i}, {d})"),
            P::Concat(xs) => {
                write!(f, "concat(")?;
                join(f, xs)?;
                write!(f, ")")
            }
            P::Find(l, x) => write!(f, "find({l}, {x})"),
            P::For { index, count, body } => write!(f, "for({index} < {count} : {body})"),
            P::Cond(c) => write!(f, "{c}"),
        }
    }
}

impl Relation {
    pub fn symbol(self) -> &'static str {
        match self {
            Relation::Eq => "=",
            Relation::Le => "<=",
            Relation::Lt => "<",
        }
    }

    pub fn from_symbol(s: &str) -> Option<Relation> {
        [Relation::Eq, Relation::Le, Relation::Lt].into_iter().find(|r| r.symbol() == s)
    }

    /// THE COMPARISON (3.4.4), on the one spelling of each side, with `bounds` the sizes' lower bounds. `?` anywhere
    /// makes it undecided. `a = b` holds on identical spellings and fails on two different literals (integers, tags,
    /// flags, literal lists; lists element by element). `a <= b` holds when identical, both literals and `a <= b`, `a`
    /// a literal at most `b`'s lower bound, `b` a min and `a <=` each argument, `a` a min and one argument `<= b`, `b` a
    /// max and `a <=` one argument, `a` a max and each argument `<= b`; it fails on two literals. `a < b` holds when
    /// `a + 1 <= b` holds and fails on two literals or identical sides. A comparison a cleverer prover could decide
    /// costs only a needless adapter.
    pub fn truth(self, a: &PropertyExpr, b: &PropertyExpr, bounds: &SizeBounds) -> Truth {
        let (a, b) = (a.normal_in(bounds), b.normal_in(bounds));
        if a.has_unknown() || b.has_unknown() {
            return Truth::Undecided;
        }
        match self {
            Relation::Eq => equal(&a, &b),
            Relation::Le => {
                if at_most(&a, &b, bounds) {
                    Truth::Holds
                } else if let (PropertyExpr::Literal(x), PropertyExpr::Literal(y)) = (&a, &b) {
                    Truth::of(x <= y)
                } else {
                    Truth::Undecided
                }
            }
            Relation::Lt => {
                if let (PropertyExpr::Literal(x), PropertyExpr::Literal(y)) = (&a, &b) {
                    Truth::of(x < y)
                } else if a == b {
                    Truth::Fails
                } else if at_most(&PropertyExpr::sum(vec![a, PropertyExpr::Literal(1)]), &b, bounds) {
                    Truth::Holds
                } else {
                    Truth::Undecided
                }
            }
        }
    }
}

/// `a = b` on two normal formulas with no `?`.
fn equal(a: &PropertyExpr, b: &PropertyExpr) -> Truth {
    if a == b {
        return Truth::Holds;
    }
    match (a, b) {
        (PropertyExpr::List(xs), PropertyExpr::List(ys)) => {
            if xs.len() != ys.len() {
                return Truth::Fails;
            }
            xs.iter().zip(ys).fold(Truth::Holds, |t, (x, y)| t.and(equal(x, y)))
        }
        _ if is_literal(a) && is_literal(b) => Truth::Fails,
        _ => Truth::Undecided,
    }
}

/// `a <= b` holds, for two normal formulas with no `?`.
fn at_most(a: &PropertyExpr, b: &PropertyExpr, bounds: &SizeBounds) -> bool {
    use PropertyExpr as P;
    if a == b {
        return true;
    }
    match (a, b) {
        (P::Literal(x), P::Literal(y)) => return x <= y,
        (P::Literal(x), _) if b.lower_bound(bounds).is_some_and(|lb| *x <= lb) => return true,
        _ => {}
    }
    if let P::Min(ys) = b {
        if ys.iter().all(|y| at_most(a, y, bounds)) {
            return true;
        }
    }
    if let P::Min(xs) = a {
        if xs.iter().any(|x| at_most(x, b, bounds)) {
            return true;
        }
    }
    if let P::Max(ys) = b {
        if ys.iter().any(|y| at_most(a, y, bounds)) {
            return true;
        }
    }
    if let P::Max(xs) = a {
        if xs.iter().all(|x| at_most(x, b, bounds)) {
            return true;
        }
    }
    false
}

impl Condition {
    /// The formulas the condition reads directly.
    fn formulas(&self) -> Vec<&PropertyExpr> {
        match self {
            Condition::Compare { left, right, .. } => vec![left, right],
            Condition::In { item, list } => vec![item, list],
            Condition::Known(x) | Condition::Flag(x) => vec![x],
            Condition::And(cs) | Condition::Or(cs) => cs.iter().flat_map(Condition::formulas).collect(),
            Condition::Not(c) => c.formulas(),
            Condition::All { count, body, .. } => {
                let mut v = vec![count];
                v.extend(body.formulas());
                v
            }
        }
    }

    /// Every reference the condition reads (an `all`'s own index is not one).
    pub fn references(&self) -> Vec<Reference<'_>> {
        let mut out = Vec::new();
        self.collect_refs(&mut Vec::new(), &mut out);
        out
    }

    fn collect_refs<'a>(&'a self, shadow: &mut Vec<&'a str>, out: &mut Vec<Reference<'a>>) {
        match self {
            Condition::All { index, count, body } => {
                count.collect_refs(shadow, out);
                shadow.push(index);
                body.collect_refs(shadow, out);
                shadow.pop();
            }
            Condition::And(cs) | Condition::Or(cs) => cs.iter().for_each(|c| c.collect_refs(shadow, out)),
            Condition::Not(c) => c.collect_refs(shadow, out),
            _ => self.formulas().into_iter().for_each(|f| f.collect_refs(shadow, out)),
        }
    }

    /// True when the condition reads no reference.
    pub fn is_ground(&self) -> bool {
        self.references().is_empty()
    }

    /// The condition with each reference `f` resolves replaced and every formula in its one spelling with `bounds`;
    /// its structure is kept (a requirement's outcome shows both sides of its comparison).
    pub fn substitute(&self, f: &dyn Fn(&Reference) -> Option<PropertyExpr>, bounds: &SizeBounds) -> Condition {
        self.map_refs(f, &mut Vec::new()).map_formulas(&|x| x.normal_in(bounds))
    }

    fn map_formulas(&self, g: &dyn Fn(&PropertyExpr) -> PropertyExpr) -> Condition {
        match self {
            Condition::Compare { relation, left, right } => Condition::Compare { relation: *relation, left: g(left), right: g(right) },
            Condition::In { item, list } => Condition::In { item: g(item), list: g(list) },
            Condition::Known(x) => Condition::Known(g(x)),
            Condition::Flag(x) => Condition::Flag(g(x)),
            Condition::And(cs) => Condition::And(cs.iter().map(|c| c.map_formulas(g)).collect()),
            Condition::Or(cs) => Condition::Or(cs.iter().map(|c| c.map_formulas(g)).collect()),
            Condition::Not(c) => Condition::Not(Box::new(c.map_formulas(g))),
            Condition::All { index, count, body } => Condition::All { index: index.clone(), count: g(count), body: Box::new(body.map_formulas(g)) },
        }
    }

    fn map_refs(&self, f: &dyn Fn(&Reference) -> Option<PropertyExpr>, shadow: &mut Vec<String>) -> Condition {
        match self {
            Condition::Compare { relation, left, right } => Condition::Compare { relation: *relation, left: left.map_refs(f, shadow), right: right.map_refs(f, shadow) },
            Condition::In { item, list } => Condition::In { item: item.map_refs(f, shadow), list: list.map_refs(f, shadow) },
            Condition::Known(x) => Condition::Known(x.map_refs(f, shadow)),
            Condition::Flag(x) => Condition::Flag(x.map_refs(f, shadow)),
            Condition::And(cs) => Condition::And(cs.iter().map(|c| c.map_refs(f, shadow)).collect()),
            Condition::Or(cs) => Condition::Or(cs.iter().map(|c| c.map_refs(f, shadow)).collect()),
            Condition::Not(c) => Condition::Not(Box::new(c.map_refs(f, shadow))),
            Condition::All { index, count, body } => {
                let count = count.map_refs(f, shadow);
                shadow.push(index.clone());
                let body = body.map_refs(f, shadow);
                shadow.pop();
                Condition::All { index: index.clone(), count, body: Box::new(body) }
            }
        }
    }

    fn bind(&self, name: &str, value: &PropertyExpr) -> Condition {
        self.map_refs(&|r| matches!(r, Reference::Index(i) if *i == name).then(|| value.clone()), &mut Vec::new())
    }

    /// THE THREE-VALUED VALUE of the condition (3.4.4) with `bounds` the sizes' lower bounds. A reference left in it
    /// reads as undecided.
    pub fn truth(&self, b: &SizeBounds) -> Truth {
        match self {
            Condition::Compare { relation, left, right } => {
                let (l, r) = (left.normal_in(b), right.normal_in(b));
                if (!l.is_ground() || !r.is_ground()) && l != r {
                    return Truth::Undecided;
                }
                relation.truth(&l, &r, b)
            }
            Condition::In { item, list } => {
                let (x, l) = (item.normal_in(b), list.normal_in(b));
                match &l {
                    PropertyExpr::List(ys) => {
                        let mut t = Truth::Fails;
                        for y in ys {
                            let e = if x.has_unknown() || y.has_unknown() { Truth::Undecided } else { equal(&x, y) };
                            t = t.or(e);
                        }
                        if t == Truth::Fails && !(is_literal(&x) && ys.iter().all(is_literal)) {
                            Truth::Undecided
                        } else {
                            t
                        }
                    }
                    _ => Truth::Undecided,
                }
            }
            Condition::Known(x) => {
                let x = x.normal_in(b);
                if x.has_unknown() {
                    Truth::Fails
                } else if x.is_ground() {
                    Truth::Holds
                } else {
                    Truth::Undecided
                }
            }
            Condition::Flag(x) => match x.normal_in(b) {
                PropertyExpr::Flag(v) => Truth::of(v),
                PropertyExpr::Cond(c) => c.truth(b),
                _ => Truth::Undecided,
            },
            Condition::And(cs) => cs.iter().fold(Truth::Holds, |t, c| t.and(c.truth(b))),
            Condition::Or(cs) => cs.iter().fold(Truth::Fails, |t, c| t.or(c.truth(b))),
            Condition::Not(c) => c.truth(b).not(),
            Condition::All { index, count, body } => match count.normal_in(b) {
                PropertyExpr::Literal(n) => (0..n.max(0)).fold(Truth::Holds, |t, k| t.and(body.bind(index, &PropertyExpr::Literal(k)).truth(b))),
                _ => Truth::Undecided,
            },
        }
    }

    /// The condition in its one spelling: formulas in theirs, a decided condition `true` or `false`, `and` and `or`
    /// flattened, sorted and without repeats, `=`'s two sides sorted, `not not c` as `c`.
    pub fn normal_in(&self, b: &SizeBounds) -> Condition {
        let decided = |t: Truth| match t {
            Truth::Holds => Some(Condition::Flag(PropertyExpr::Flag(true))),
            Truth::Fails => Some(Condition::Flag(PropertyExpr::Flag(false))),
            Truth::Undecided => None,
        };
        match self {
            Condition::Compare { relation, left, right } => {
                let (mut l, mut r) = (left.normal_in(b), right.normal_in(b));
                if *relation == Relation::Eq && r < l {
                    std::mem::swap(&mut l, &mut r);
                }
                let c = Condition::Compare { relation: *relation, left: l, right: r };
                decided(c.truth(b)).unwrap_or(c)
            }
            Condition::In { item, list } => {
                let c = Condition::In { item: item.normal_in(b), list: list.normal_in(b) };
                decided(c.truth(b)).unwrap_or(c)
            }
            Condition::Known(x) => {
                let c = Condition::Known(x.normal_in(b));
                decided(c.truth(b)).unwrap_or(c)
            }
            Condition::Flag(x) => match x.normal_in(b) {
                PropertyExpr::Cond(c) => c.normal_in(b),
                other => Condition::Flag(other),
            },
            Condition::Not(c) => match c.normal_in(b) {
                Condition::Not(inner) => *inner,
                Condition::Flag(PropertyExpr::Flag(v)) => Condition::Flag(PropertyExpr::Flag(!v)),
                other => Condition::Not(Box::new(other)),
            },
            Condition::And(cs) | Condition::Or(cs) => {
                let and = matches!(self, Condition::And(_));
                let mut flat: Vec<Condition> = Vec::new();
                for c in cs.iter().map(|c| c.normal_in(b)) {
                    match c {
                        Condition::And(inner) if and => flat.extend(inner),
                        Condition::Or(inner) if !and => flat.extend(inner),
                        Condition::Flag(PropertyExpr::Flag(v)) if v == and => {}
                        Condition::Flag(PropertyExpr::Flag(v)) => return Condition::Flag(PropertyExpr::Flag(v)),
                        other => flat.push(other),
                    }
                }
                flat.sort();
                flat.dedup();
                match flat.len() {
                    0 => Condition::Flag(PropertyExpr::Flag(and)),
                    1 => flat.remove(0),
                    _ if and => Condition::And(flat),
                    _ => Condition::Or(flat),
                }
            }
            Condition::All { index, count, body } => match count.normal_in(b) {
                PropertyExpr::Literal(n) => Condition::And((0..n.max(0)).map(|k| body.bind(index, &PropertyExpr::Literal(k))).collect()).normal_in(b),
                count => Condition::All { index: index.clone(), count, body: Box::new(body.normal_in(b)) },
            },
        }
    }

    /// Parse a condition in the text [`fmt::Display`] writes.
    pub fn parse(text: &str) -> Result<Condition, String> {
        let mut p = Parser::new(text);
        let c = p.condition()?;
        p.end()?;
        Ok(c.map_formulas(&PropertyExpr::normal))
    }

    /// The comparison's relation and two sides when the condition is one comparison.
    pub fn comparison(&self) -> Option<(Relation, &PropertyExpr, &PropertyExpr)> {
        match self {
            Condition::Compare { relation, left, right } => Some((*relation, left, right)),
            _ => None,
        }
    }

    /// What is malformed in the condition's formulas ([`PropertyExpr::shape_defects`]).
    pub fn shape_defects(&self) -> Vec<String> {
        let mut v: Vec<String> = self.formulas().into_iter().flat_map(PropertyExpr::shape_defects).collect();
        if let Condition::All { index, body, .. } = self {
            if !is_name(index) {
                v.push(format!("has the index {index:?}, which is not a name"));
            }
            v.extend(body.shape_defects());
        }
        v
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Condition::Compare { relation, left, right } => write!(f, "{left} {} {right}", relation.symbol()),
            Condition::In { item, list } => write!(f, "{item} in {list}"),
            Condition::Known(x) => write!(f, "known({x})"),
            Condition::Flag(x) => write!(f, "{x}"),
            Condition::Not(c) => match **c {
                Condition::And(_) | Condition::Or(_) => write!(f, "not ({c})"),
                _ => write!(f, "not {c}"),
            },
            Condition::And(cs) => {
                let parts: Vec<String> = cs.iter().map(|c| if matches!(c, Condition::Or(_)) { format!("({c})") } else { c.to_string() }).collect();
                write!(f, "{}", parts.join(" and "))
            }
            Condition::Or(cs) => {
                let parts: Vec<String> = cs.iter().map(Condition::to_string).collect();
                write!(f, "{}", parts.join(" or "))
            }
            Condition::All { index, count, body } => write!(f, "all({index} < {count} : {body})"),
        }
    }
}

/// The recursive-descent reader of the closed language. Grammar (lowest precedence first):
/// `condition := and ('or' and)*`, `and := not ('and' not)*`, `not := 'not' not | atom-condition`,
/// `atom-condition := known(f) | all(i < f : c) | '(' condition ')' | f (rel f | 'in' f)?`,
/// `formula := product (('+' | '-') product)*`, `product := unary ('*' unary)*`, `unary := '-' unary | atom`.
struct Parser<'a> {
    text: &'a str,
    rest: &'a str,
    indices: Vec<String>,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Parser<'a> {
        Parser { text, rest: text, indices: Vec::new() }
    }

    fn bug(&self, what: &str) -> String {
        format!("our software has a bug: the formula {:?} {what} at {:?}", self.text, self.rest.trim_start())
    }

    fn ws(&mut self) {
        self.rest = self.rest.trim_start();
    }

    fn peek(&mut self) -> Option<char> {
        self.ws();
        self.rest.chars().next()
    }

    /// Consume `s` (after white space) if it is next.
    fn eat(&mut self, s: &str) -> bool {
        self.ws();
        match self.rest.strip_prefix(s) {
            Some(after) => {
                self.rest = after;
                true
            }
            None => false,
        }
    }

    fn expect(&mut self, s: &str) -> Result<(), String> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(self.bug(&format!("wants {s:?}")))
        }
    }

    fn end(&mut self) -> Result<(), String> {
        self.ws();
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(self.bug("has text left over"))
        }
    }

    /// The word next (identifier characters and dots), not consumed.
    fn word(&mut self) -> &'a str {
        self.ws();
        let end = self.rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.')).unwrap_or(self.rest.len());
        &self.rest[..end]
    }

    /// The keyword `w` as a whole word next (consumed when it is).
    fn eat_word(&mut self, w: &str) -> bool {
        if self.word() == w {
            self.rest = &self.rest[w.len()..];
            true
        } else {
            false
        }
    }

    /// `name(` next (consumed when it is).
    fn eat_call(&mut self, name: &str) -> bool {
        self.ws();
        match self.rest.strip_prefix(name).and_then(|r| r.strip_prefix('(')) {
            Some(after) => {
                self.rest = after;
                true
            }
            None => false,
        }
    }

    /// A value: a condition, read as a formula when it is a bare formula.
    fn value(&mut self) -> Result<PropertyExpr, String> {
        Ok(match self.condition()? {
            Condition::Flag(f) => f,
            c => PropertyExpr::Cond(Box::new(c)),
        })
    }

    fn condition(&mut self) -> Result<Condition, String> {
        let mut cs = vec![self.cond_and()?];
        while self.eat_word("or") {
            cs.push(self.cond_and()?);
        }
        Ok(if cs.len() == 1 { cs.remove(0) } else { Condition::Or(cs) })
    }

    fn cond_and(&mut self) -> Result<Condition, String> {
        let mut cs = vec![self.cond_not()?];
        while self.eat_word("and") {
            cs.push(self.cond_not()?);
        }
        Ok(if cs.len() == 1 { cs.remove(0) } else { Condition::And(cs) })
    }

    fn cond_not(&mut self) -> Result<Condition, String> {
        if self.eat_word("not") {
            return Ok(Condition::Not(Box::new(self.cond_not()?)));
        }
        self.cond_atom()
    }

    /// `i < n :` of a comprehension, the index pushed.
    fn comprehension(&mut self) -> Result<(String, PropertyExpr), String> {
        let index = self.word().to_string();
        if !is_name(&index) {
            return Err(self.bug("wants an index name"));
        }
        self.rest = &self.rest[index.len()..];
        self.expect("<")?;
        let count = self.formula()?;
        self.expect(":")?;
        self.indices.push(index.clone());
        Ok((index, count))
    }

    fn cond_atom(&mut self) -> Result<Condition, String> {
        if self.eat_call("known") {
            let x = self.value()?;
            self.expect(")")?;
            return Ok(Condition::Known(x));
        }
        if self.eat_call("all") {
            let (index, count) = self.comprehension()?;
            let body = self.condition();
            self.indices.pop();
            let body = body?;
            self.expect(")")?;
            return Ok(Condition::All { index, count, body: Box::new(body) });
        }
        if self.peek() == Some('(') {
            // A parenthesised condition, unless the parentheses open a formula that a relation or an operator follows.
            let saved = self.rest;
            self.eat("(");
            if let Ok(c) = self.condition() {
                if self.eat(")") {
                    let next = self.peek();
                    let continues = matches!(next, Some('=' | '<' | '+' | '-' | '*')) || self.word() == "in";
                    if !continues {
                        return Ok(c);
                    }
                }
            }
            self.rest = saved;
        }
        let left = self.formula()?;
        for relation in [Relation::Le, Relation::Lt, Relation::Eq] {
            if self.eat(relation.symbol()) {
                let right = self.formula()?;
                return Ok(Condition::Compare { relation, left, right });
            }
        }
        if self.eat_word("in") {
            let list = self.formula()?;
            return Ok(Condition::In { item: left, list });
        }
        Ok(Condition::Flag(left))
    }

    fn formula(&mut self) -> Result<PropertyExpr, String> {
        let mut terms = vec![self.product()?];
        loop {
            if self.eat("+") {
                terms.push(self.product()?);
            } else if self.peek() == Some('-') {
                self.eat("-");
                terms.push(PropertyExpr::Product(vec![PropertyExpr::Literal(-1), self.product()?]));
            } else {
                break;
            }
        }
        Ok(if terms.len() == 1 { terms.remove(0) } else { PropertyExpr::Sum(terms) })
    }

    fn product(&mut self) -> Result<PropertyExpr, String> {
        let mut fs = vec![self.unary()?];
        while self.eat("*") {
            fs.push(self.unary()?);
        }
        Ok(if fs.len() == 1 { fs.remove(0) } else { PropertyExpr::Product(fs) })
    }

    fn unary(&mut self) -> Result<PropertyExpr, String> {
        if self.peek() == Some('-') {
            self.eat("-");
            if self.rest.starts_with(|c: char| c.is_ascii_digit()) {
                let digits = self.word();
                self.rest = &self.rest[digits.len()..];
                return format!("-{digits}").parse::<i64>().map(PropertyExpr::Literal).map_err(|_| self.bug("has a literal that is not an integer"));
            }
            return Ok(PropertyExpr::Product(vec![PropertyExpr::Literal(-1), self.unary()?]));
        }
        self.atom()
    }

    /// The comma-separated values up to `)`; at least `least` of them.
    fn args(&mut self, least: usize, name: &str) -> Result<Vec<PropertyExpr>, String> {
        let mut xs = Vec::new();
        if !self.eat(")") {
            loop {
                xs.push(self.value()?);
                if self.eat(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        if xs.len() < least {
            return Err(self.bug(&format!("has a {name} of {} arguments, not at least {least}", xs.len())));
        }
        Ok(xs)
    }

    fn atom(&mut self) -> Result<PropertyExpr, String> {
        use PropertyExpr as P;
        if self.eat("(") {
            let v = self.value()?;
            self.expect(")")?;
            return Ok(v);
        }
        if self.eat("?") {
            return Ok(P::Unknown);
        }
        if self.eat("'") {
            let end = self.rest.find('\'').ok_or_else(|| self.bug("has a tag that is not closed"))?;
            let tag = &self.rest[..end];
            self.rest = &self.rest[end + 1..];
            if tag.is_empty() {
                return Err(self.bug("has an empty tag"));
            }
            return Ok(P::Tag(tag.to_string()));
        }
        for (name, least) in [("min", 2), ("max", 2), ("list", 0), ("concat", 1)] {
            if self.eat_call(name) {
                let xs = self.args(least, name)?;
                return Ok(match name {
                    "min" => P::Min(xs),
                    "max" => P::Max(xs),
                    "list" => P::List(xs),
                    _ => P::Concat(xs),
                });
            }
        }
        if self.eat_call("if") {
            let cond = self.condition()?;
            self.expect(",")?;
            let then = self.value()?;
            self.expect(",")?;
            let otherwise = self.value()?;
            self.expect(")")?;
            return Ok(P::If { cond: Box::new(cond), then: Box::new(then), otherwise: Box::new(otherwise) });
        }
        if self.eat_call("len") {
            let mut xs = self.args(1, "len")?;
            return if xs.len() == 1 { Ok(P::Len(Box::new(xs.remove(0)))) } else { Err(self.bug("has a len of more than one argument")) };
        }
        if self.eat_call("at") {
            let xs = self.args(3, "at")?;
            let [l, i, d]: [PropertyExpr; 3] = xs.try_into().map_err(|_| self.bug("has an at of more than three arguments"))?;
            return Ok(P::At(Box::new(l), Box::new(i), Box::new(d)));
        }
        if self.eat_call("find") {
            let xs = self.args(2, "find")?;
            let [l, x]: [PropertyExpr; 2] = xs.try_into().map_err(|_| self.bug("has a find of more than two arguments"))?;
            return Ok(P::Find(Box::new(l), Box::new(x)));
        }
        if self.eat_call("for") {
            let (index, count) = self.comprehension()?;
            let body = self.value();
            self.indices.pop();
            let body = body?;
            self.expect(")")?;
            return Ok(P::For { index, count: Box::new(count), body: Box::new(body) });
        }
        let w = self.word();
        if w.is_empty() {
            return Err(self.bug("wants a term"));
        }
        if w.starts_with(|c: char| c.is_ascii_digit()) {
            self.rest = &self.rest[w.len()..];
            return w.parse::<i64>().map(P::Literal).map_err(|_| self.bug(&format!("has {w:?}, which is not an integer")));
        }
        self.rest = &self.rest[w.len()..];
        match w {
            "true" => return Ok(P::Flag(true)),
            "false" => return Ok(P::Flag(false)),
            "each" => return Ok(P::Index("each".to_string())),
            _ => {}
        }
        if let Some((head, property)) = w.split_once('.') {
            if !is_identifier(property) {
                return Err(self.bug(&format!("has {w:?}, whose property is not an identifier")));
            }
            return match head {
                "out" => Ok(P::Output(property.to_string())),
                "env" => Ok(P::Env(property.to_string())),
                _ => match head.strip_prefix("in").filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())).and_then(|d| d.parse::<u32>().ok()) {
                    Some(position) => Ok(P::Input { position, property: property.to_string() }),
                    None => Err(self.bug(&format!("has {w:?}, which is not in<k>.<p>, out.<p> or env.<a>"))),
                },
            };
        }
        if self.indices.iter().any(|i| i == w) {
            return Ok(P::Index(w.to_string()));
        }
        if is_name(w) {
            return Ok(P::Size(w.to_string()));
        }
        Err(self.bug(&format!("has {w:?}, which is not a term of the formula language")))
    }
}

#[cfg(test)]
mod tests;
