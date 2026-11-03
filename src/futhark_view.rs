//! THE VIEW: a Futhark kingdom as a query over the master table (`docs/PLAN_futhark_kingdom.md`, "Step 3 design").
//!
//! `kingdoms/futhark/master/` holds every row of every Futhark kingdom as a TEMPLATE over size variables (`{D}`), the
//! element (`{E}`) and contexts (`C`), each row versioned (`since`). A problem's view specification (`view.tsv`,
//! written by `kingdoms/futhark/sniff.py` from the problem's Futhark source) is the WHERE clause: the element, the
//! inputs and output over named dimensions, the depth, the master version. The view
//!
//! 1. **filters** the master rows the specification admits (version and element; a signature view admits EVERY
//!    family, the matrix view its `families`);
//! 2. **binds** them: a row's scope is a list of quantifiers, outermost first; consecutive rows that share their first
//!    quantifier are one block, the block iterates it, and its rows recurse on their remaining quantifiers, so the
//!    master's row order is the expansion order;
//! 3. keeps the types **reachable both ways** (`docs/PLAN_futhark_kingdom.md`, 3e): forward, F is the least fixpoint
//!    of what can be built (every leaf's output, then the output of every function whose arguments are all in F), and
//!    a row is usable when all its arguments are in F; backward, B is the closure from the root over the usable rows;
//!    the kept rows are the usable rows whose output is in B. Which families a view has is decided by the types, never
//!    by what the problem's source happens to use;
//! 4. **writes** the exact table: `types.tsv`, `functions.tsv` and, for a signature view, the e-graph artefacts.
//!
//! This is THE ONE implementation of the view (`docs/PLAN_futhark_kingdom.md` 3g): the engine runs it, and
//! [`write_committed`] writes its tables for the generator's witnesses (`kingdoms/futhark/gen_tables.py --views DIR`);
//! the per-problem tables are no longer committed, and each view's written text is pinned by its
//! [`View::fingerprint`]. The engine sees an ordinary flat typed table
//! ([`FutharkKingdom::from_tables`]): legality by construction is unchanged, the view only decides which rows exist.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

use crate::futhark_table::{FutharkFamily, FutharkKingdom};

const MASTER_VERSION: &str = include_str!("../kingdoms/futhark/master/VERSION");
const MASTER_ROWS: &str = include_str!("../kingdoms/futhark/master/rows.tsv");
const MASTER_FRAMES: &str = include_str!("../kingdoms/futhark/master/frames.tsv");
const MASTER_RULES: &str = include_str!("../kingdoms/futhark/master/rules.tsv");
const MASTER_POOLS: &str = include_str!("../kingdoms/futhark/master/pools.tsv");
const MASTER_TYPES: &str = include_str!("../kingdoms/futhark/master/types.tsv");

// ---- the master -------------------------------------------------------------------------------------------------

/// One row template of `master/rows.tsv`.
#[derive(Debug, Clone)]
struct MasterRow {
    reference: String,
    since: u32,
    family: String,
    kind: String,
    elems: String,
    scope: Vec<String>,
    when: String,
    name: String,
    out: String,
    args: Vec<String>,
    spelling: String,
    note: String,
}

/// A frame template of `master/frames.tsv`: a binder that extends a context, and the classes of the variables it
/// binds (a variable is named by its class, [`var_name`]).
#[derive(Debug, Clone)]
struct FrameTemplate {
    /// The master version the frame was added in: a view pinned before it has no context through it.
    since: u32,
    code: String,
    scope: Vec<String>,
    binds: Vec<String>,
}

#[derive(Debug, Clone)]
struct RuleTemplate {
    on: String,
    elems: String,
    needs: String,
    rule: String,
}

#[derive(Debug, Clone)]
struct TypeTemplate {
    family: String,
    scope: Vec<String>,
    dual: String,
    slot: String,
    form: String,
    meaning: String,
}

/// The master table: its version, rows, frames, rules, literal pools and declared types.
#[derive(Debug, Clone)]
pub struct Master {
    pub version: u32,
    rows: Vec<MasterRow>,
    frames: Vec<FrameTemplate>,
    rules: Vec<RuleTemplate>,
    pools: BTreeMap<String, Vec<String>>,
    types: Vec<TypeTemplate>,
}

fn tsv<'a>(name: &str, text: &'a str, columns: &[&str]) -> Result<Vec<Vec<&'a str>>, String> {
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    if head != columns {
        return Err(format!("master/{name}: the header is {head:?}, expected {columns:?}"));
    }
    let mut out = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != columns.len() {
            return Err(format!("master/{name}:{}: {} columns, expected {}", i + 2, f.len(), columns.len()));
        }
        out.push(f);
    }
    Ok(out)
}

fn scope_list(s: &str) -> Vec<String> {
    s.split(';').filter(|q| !q.is_empty()).map(str::to_string).collect()
}

impl Master {
    /// The committed master, as compiled in.
    pub fn committed() -> Result<Master, String> {
        Master::parse(MASTER_VERSION, MASTER_ROWS, MASTER_FRAMES, MASTER_RULES, MASTER_POOLS, MASTER_TYPES)
    }

    /// A master from the text of its files.
    pub fn parse(version: &str, rows: &str, frames: &str, rules: &str, pools: &str, types: &str) -> Result<Master, String> {
        let version = version.trim().parse::<u32>().map_err(|e| format!("master/VERSION: {e}"))?;
        let rows = tsv("rows.tsv", rows, &["ref", "since", "family", "kind", "elems", "scope", "when", "name", "out", "args", "spelling", "note"])?
            .into_iter()
            .map(|f| {
                Ok(MasterRow {
                    reference: f[0].into(),
                    since: f[1].parse().map_err(|e| format!("master/rows.tsv: the row {} has the version {:?}: {e}", f[0], f[1]))?,
                    family: f[2].into(),
                    kind: f[3].into(),
                    elems: f[4].into(),
                    scope: scope_list(f[5]),
                    when: f[6].into(),
                    name: f[7].into(),
                    out: f[8].into(),
                    args: f[9].split(';').filter(|a| !a.is_empty()).map(str::to_string).collect(),
                    spelling: f[10].into(),
                    note: f[11].into(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let frames = tsv("frames.tsv", frames, &["family", "kind", "since", "code", "scope", "binds"])?
            .into_iter()
            .map(|f| {
                let binds: Vec<String> = f[5].split(',').map(str::to_string).collect();
                if let Some(b) = binds.iter().find(|b| b.is_empty() || b.contains(':')) {
                    return Err(format!("master/frames.tsv: the binding {b:?} is not a class (a variable is named by its class)"));
                }
                let since = f[2].parse().map_err(|e| format!("master/frames.tsv: the frame {} has the version {:?}: {e}", f[1], f[2]))?;
                if since > version {
                    return Err(format!("master/frames.tsv: the frame {} is of version {since}, the master is at {version}", f[1]));
                }
                Ok(FrameTemplate { since, code: f[3].into(), scope: scope_list(f[4]), binds })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let rules = tsv("rules.tsv", rules, &["on", "elems", "needs", "rule"])?
            .into_iter()
            .map(|f| RuleTemplate { on: f[0].into(), elems: f[1].into(), needs: f[2].into(), rule: f[3].into() })
            .collect();
        let pools = tsv("pools.tsv", pools, &["element", "literals"])?.into_iter().map(|f| (f[0].to_string(), f[1].split(',').map(str::to_string).collect())).collect();
        let types = tsv("types.tsv", types, &["family", "scope", "dual", "slot", "form", "meaning"])?
            .into_iter()
            .map(|f| TypeTemplate { family: f[0].into(), scope: scope_list(f[1]), dual: f[2].into(), slot: f[3].into(), form: f[4].into(), meaning: f[5].into() })
            .collect();
        Ok(Master { version, rows, frames, rules, pools, types })
    }

    /// The default literal pool of an element.
    pub fn pool(&self, elem: &str) -> Option<&[String]> {
        self.pools.get(elem).map(Vec::as_slice)
    }
}

// ---- the specification ------------------------------------------------------------------------------------------

/// What a view is a view of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// A program's signature: inputs and an output over named dimensions, in contexts (the signature family).
    Signature,
    /// The matrix family (the A7 kingdom): dimension roles and a root class.
    Matrix,
}

/// An input of a signature view: its name, element (`size` for a size input) and dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecInput {
    pub name: String,
    pub elem: String,
    pub dims: Vec<String>,
}

/// A view specification (`view.tsv`): four-column lines `kind name element dims`; `#` lines are comments. A
/// `families` line belongs to the matrix view only: a signature view admits every family of the master. The
/// `element` line lists the view's element types (one, or several: `f32,i32`), and a `literals` line gives one
/// element's pool (the element in its third column, empty for a view of one element).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewSpec {
    pub shape: Shape,
    pub master: Option<u32>,
    /// The first of the view's elements (the matrix view's one).
    pub element: String,
    /// The view's element types, in the order the `element` line gives them. A view of several elements writes an
    /// array class with its element (`a.f32.n`); a view of one keeps the names of before (`a.n`).
    pub elements: Vec<String>,
    /// The inputs in order, a size input with the element `size`.
    pub inputs: Vec<SpecInput>,
    pub sizes: Vec<String>,
    /// The output's name, element and dimensions.
    pub output: Option<SpecInput>,
    pub depth: u32,
    /// The matrix view's families (a signature view has none: it admits every family).
    pub families: Vec<String>,
    /// The literal pool of each element.
    pub literals: BTreeMap<String, Vec<String>>,
    pub dims: Vec<String>,
    pub roles: Vec<String>,
    pub root: Vec<String>,
}

impl ViewSpec {
    pub fn parse(text: &str) -> Result<ViewSpec, String> {
        let mut s = ViewSpec {
            shape: Shape::Signature,
            master: None,
            element: String::new(),
            elements: Vec::new(),
            inputs: Vec::new(),
            sizes: Vec::new(),
            output: None,
            depth: 0,
            families: Vec::new(),
            literals: BTreeMap::new(),
            dims: Vec::new(),
            roles: Vec::new(),
            root: Vec::new(),
        };
        let list = |x: &str| x.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect::<Vec<_>>();
        let mut families_line = false;
        let mut pools: Vec<(String, Vec<String>)> = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            let [kind, name, elem, dims] = f.as_slice() else {
                return Err(format!("the specification line {line:?} has {} fields, wants 4", f.len()));
            };
            let num = |x: &str| x.parse::<u32>().map_err(|e| format!("the specification line {line:?}: {e}"));
            match *kind {
                "input" => s.inputs.push(SpecInput { name: name.to_string(), elem: elem.to_string(), dims: list(dims) }),
                "size" => {
                    if *elem != "i32" || !dims.is_empty() {
                        return Err(format!("the size input {name} is not an i32 scalar"));
                    }
                    s.inputs.push(SpecInput { name: name.to_string(), elem: "size".into(), dims: vec![] });
                    s.sizes.push(name.to_string());
                }
                "output" => s.output = Some(SpecInput { name: name.to_string(), elem: elem.to_string(), dims: list(dims) }),
                "master" => s.master = Some(num(name)?),
                "depth" => s.depth = num(name)?,
                "element" => {
                    s.elements = list(name);
                    s.element = s.elements.first().cloned().unwrap_or_default();
                }
                "shape" => {
                    s.shape = match *name {
                        "matrix" => Shape::Matrix,
                        "signature" => Shape::Signature,
                        other => return Err(format!("the view shape {other}")),
                    }
                }
                "families" => {
                    families_line = true;
                    s.families = list(name);
                }
                "literals" => pools.push((elem.to_string(), list(name))),
                "dims" => s.dims = list(name),
                "roles" => s.roles = list(name),
                "root" => s.root = list(name),
                other => return Err(format!("a specification line of kind {other}")),
            }
        }
        if s.elements.is_empty() {
            return Err("the specification names no element".into());
        }
        if let Some(e) = s.elements.iter().find(|e| !ELEMENT_NAMES.contains(&e.as_str())) {
            return Err(format!("the element {e} is not a Futhark primitive type"));
        }
        if (1..s.elements.len()).any(|i| s.elements[i..].contains(&s.elements[i - 1])) {
            return Err(format!("the element line repeats an element: {:?}", s.elements));
        }
        for (elem, pool) in pools {
            let elem = if elem.is_empty() {
                let [one] = s.elements.as_slice() else { return Err(format!("a literals line names no element and the view has {}", s.elements.len())) };
                one.clone()
            } else if s.elements.contains(&elem) {
                elem
            } else {
                return Err(format!("a literals line is of the element {elem}, which the view does not have"));
            };
            if s.literals.insert(elem.clone(), pool).is_some() {
                return Err(format!("two literals lines for the element {elem}"));
            }
        }
        if s.shape == Shape::Signature && families_line {
            return Err("a signature specification has a families line: a signature view admits every family".into());
        }
        if s.shape == Shape::Signature {
            let output = s.output.as_ref().ok_or("the signature specification has no output")?;
            for p in s.inputs.iter().filter(|p| p.elem != "size").chain(std::iter::once(output)) {
                if !s.elements.contains(&p.elem) {
                    return Err(format!("{} is of the element {}, which the element line does not list", p.name, p.elem));
                }
                if p.dims.is_empty() && p.name != output.name {
                    return Err(format!("the input {} is a scalar that is not a size: the master has no row for a scalar input", p.name));
                }
                if let Some(d) = p.dims.iter().find(|d| !is_dim(d)) {
                    return Err(format!("the dimension {d} of {} is not a dimension name (a lower-case identifier that is not an element's)", p.name));
                }
            }
            if let Some(d) = s.dims.iter().chain(&s.sizes).find(|d| !is_dim(d)) {
                return Err(format!("the dimension {d} is not a dimension name (a lower-case identifier that is not an element's)"));
            }
            if let Some(e) = s.elements.iter().find(|e| !s.literals.contains_key(*e)) {
                return Err(format!("the element {e} has no literals line"));
            }
        }
        Ok(s)
    }

    /// Whether the view has several elements, so an array class is written with its element.
    pub fn multi(&self) -> bool {
        self.elements.len() > 1
    }
}

// ---- the class --------------------------------------------------------------------------------------------------

/// The element types a class may name (Futhark's primitive types). A dimension may not be named as one: the text of
/// an array class would then read two ways.
pub const ELEMENT_NAMES: [&str; 12] = ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64", "bool"];

/// A CLASS: the type of a value apart from its context, as an AST with an INJECTIVE text (the review's finding 1:
/// a dotted string such as `t.a.n.a.m` reads two ways; this grammar reads one way). The grammar, as the tables write
/// it (`kingdoms/futhark/README.md`, "The class"):
///
/// ```text
/// class  ::= elem                         a scalar: `i32`, `f32`
///          | 'ix.' dim                    an index in range for a dimension: `ix.n`
///          | 'a.' [elem '.'] dim {'.' dim} an array: `a.n.d`; `a.f32.n.d` in a view of more than one element
///          | '(' class ',' class ')'      a tuple: `(f32,i32)` (no space; Futhark's own notation)
///          | name                         a class the matrix family declares: `mat.n.m`, `fn.arr_f32_arr_f32`
/// ```
///
/// The array's element is written only in a view of several elements: a view of one element keeps the names of
/// before (`a.n`), its one element implied (D7's naming rule, `docs/PLAN_futhark_kingdom.md`). Both texts are
/// injective: a dimension is never named as an element ([`ELEMENT_NAMES`]), so the piece after `a.` is an element
/// exactly when it is one. A tuple's components are separated by a comma inside its own parentheses, so a tuple of a
/// rank-2 array and a scalar cannot be confused with a tuple of a rank-1 array and an array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Scalar(String),
    Index(String),
    Array { elem: String, dims: Vec<String> },
    Tuple(Vec<Class>),
    /// A class the matrix family declares by name (`master/types.tsv`).
    Named(String),
}

/// The pieces of `text` between its top-level commas (a comma inside parentheses separates a tuple's components).
fn split_top(text: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0usize, 0usize);
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&text[start..]);
    out
}

impl Class {
    /// A class from its text in a view whose elements are `elements`. An array written without its element takes
    /// `default_elem` (the row's `{E}`), which a view of one element always has; an error names what does not parse.
    pub fn parse(text: &str, elements: &[String], default_elem: Option<&str>) -> Result<Class, String> {
        if let Some(inner) = text.strip_prefix('(') {
            let inner = inner.strip_suffix(')').ok_or_else(|| format!("the class {text} opens a tuple it does not close"))?;
            let parts = split_top(inner);
            if parts.len() < 2 {
                return Err(format!("the tuple class {text} has {} component", parts.len()));
            }
            return Ok(Class::Tuple(parts.iter().map(|p| Class::parse(p, elements, default_elem)).collect::<Result<_, _>>()?));
        }
        if let Some(d) = text.strip_prefix("ix.") {
            return if is_dim(d) { Ok(Class::Index(d.to_string())) } else { Err(format!("the index class {text} is not over a dimension")) };
        }
        if let Some(rest) = text.strip_prefix("a.") {
            let mut pieces: Vec<&str> = rest.split('.').collect();
            let elem = if pieces.first().is_some_and(|p| ELEMENT_NAMES.contains(p)) {
                let e = pieces.remove(0);
                if !elements.iter().any(|x| x == e) {
                    return Err(format!("the class {text} is of the element {e}, which the view does not have"));
                }
                e.to_string()
            } else {
                match (default_elem, elements) {
                    (Some(e), _) => e.to_string(),
                    (None, [e]) => e.clone(),
                    (None, _) => return Err(format!("the array class {text} names no element and the view has {}", elements.len())),
                }
            };
            if pieces.is_empty() || !pieces.iter().all(|d| is_dim(d)) {
                return Err(format!("the array class {text} is not over dimensions"));
            }
            return Ok(Class::Array { elem, dims: pieces.iter().map(|d| d.to_string()).collect() });
        }
        if elements.iter().any(|e| e == text) {
            return Ok(Class::Scalar(text.to_string()));
        }
        if ELEMENT_NAMES.contains(&text) {
            return Err(format!("the class {text} is an element the view does not have"));
        }
        if text.starts_with("mat.") || text.starts_with("fn.") {
            return Ok(Class::Named(text.to_string()));
        }
        Err(format!("the class {text:?} does not parse"))
    }

    /// The class's text in a view of one element (`multi` false: an array leaves its element implied) or of several.
    pub fn text(&self, multi: bool) -> String {
        match self {
            Class::Scalar(e) => e.clone(),
            Class::Index(d) => format!("ix.{d}"),
            Class::Array { elem, dims } => {
                if multi {
                    format!("a.{elem}.{}", dims.join("."))
                } else {
                    format!("a.{}", dims.join("."))
                }
            }
            Class::Tuple(parts) => format!("({})", parts.iter().map(|p| p.text(multi)).collect::<Vec<_>>().join(",")),
            Class::Named(n) => n.clone(),
        }
    }

    /// The element of a scalar or an array; an index, a tuple and a named class have none.
    pub fn elem(&self) -> Option<&str> {
        match self {
            Class::Scalar(e) | Class::Array { elem: e, .. } => Some(e),
            _ => None,
        }
    }

    /// An array's dimensions (empty for every other class).
    pub fn dims(&self) -> &[String] {
        match self {
            Class::Array { dims, .. } => dims,
            _ => &[],
        }
    }

    /// The class's Futhark type: `f32`, `[n][d]f32`, `(f32, i32)`; an index is an `i64`.
    pub fn futhark_type(&self) -> String {
        match self {
            Class::Scalar(e) => e.clone(),
            Class::Index(_) => "i64".into(),
            Class::Array { elem, dims } => dims.iter().map(|d| format!("[{d}]")).collect::<String>() + elem,
            Class::Tuple(parts) => format!("({})", parts.iter().map(Class::futhark_type).collect::<Vec<_>>().join(", ")),
            Class::Named(n) => n.clone(),
        }
    }

    /// The class as a piece of a Futhark identifier: the text without its `a.` or `ix.` prefix, `.` as `_` (`n`,
    /// `f32_n`; a tuple's parentheses and comma as `_`). The zero array of a rank-1 class is `zeros_<tag>`.
    pub fn tag(&self, multi: bool) -> String {
        let t = self.text(multi);
        let t = t.strip_prefix("a.").or_else(|| t.strip_prefix("ix.")).unwrap_or(&t);
        t.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
    }
}

/// Whether `d` may name a dimension: a lower-case identifier that is not an element's name.
fn is_dim(d: &str) -> bool {
    let mut c = d.chars();
    c.next().is_some_and(|x| x.is_ascii_lowercase()) && c.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '_') && !ELEMENT_NAMES.contains(&d)
}

// ---- binding ----------------------------------------------------------------------------------------------------

/// A CONTEXT: how many variables of each class are in scope, and the depth (the binders around a node, for the
/// `shallow` rule and the work bound). Not the frame stack (one type a stack, the review's F1), and not only the set
/// of classes: a set loses a program that reads an outer variable through an inner binder of its class (measured
/// 2026-11-02, 4 of pathfinder's 45 model programs), so the count is kept (the review's control). Its name is `c` at
/// depth 0, else `c/{<class>[*<count above one>], sorted, comma-separated}@<depth>`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Ctx {
    pub classes: BTreeMap<String, u32>,
    pub depth: u32,
}

impl Ctx {
    pub fn name(&self) -> String {
        if self.depth == 0 {
            "c".into()
        } else {
            let items: Vec<String> = self.classes.iter().map(|(c, n)| if *n == 1 { c.clone() } else { format!("{c}*{n}") }).collect();
            format!("c/{{{}}}@{}", items.join(","), self.depth)
        }
    }

    /// The context inside a binder that binds `classes`.
    fn push(&self, classes: &[String]) -> Ctx {
        let mut c = self.clone();
        for k in classes {
            *c.classes.entry(k.clone()).or_default() += 1;
        }
        c.depth += 1;
        c
    }

    /// The variables in scope of a class.
    pub fn count(&self, cls: &str) -> u32 {
        self.classes.get(cls).copied().unwrap_or(0)
    }

    /// The context of a dual's name (`cls:c`, `cls:c/{a,b}@2`); `None` for a context-free dual.
    pub fn of_dual(dual: &str) -> Result<Option<Ctx>, String> {
        let Some((_, ctx)) = dual.split_once(':') else { return Ok(None) };
        if ctx == "c" {
            return Ok(Some(Ctx::default()));
        }
        let body = ctx.strip_prefix("c/{").ok_or_else(|| format!("the context of {dual} does not parse"))?;
        let (set, depth) = body.split_once("}@").ok_or_else(|| format!("the context of {dual} does not parse"))?;
        let depth: u32 = depth.parse().map_err(|_| format!("the depth of {dual} does not parse"))?;
        let mut classes: BTreeMap<String, u32> = BTreeMap::new();
        for item in set.split(',') {
            let (c, n) = match item.split_once('*') {
                Some((c, n)) => (c, n.parse::<u32>().map_err(|_| format!("the count of {item} in {dual} does not parse"))?),
                None => (item, 1),
            };
            if c.is_empty() || n == 0 || classes.insert(c.to_string(), n).is_some() {
                return Err(format!("the context of {dual} names {item:?}"));
            }
        }
        if depth == 0 || classes.is_empty() {
            return Err(format!("the context of {dual} is not a binder's"));
        }
        Ok(Some(Ctx { classes, depth }))
    }
}

/// The bindings of an expansion: the template variables, and the context of a row inside the context loop.
#[derive(Debug, Clone, Default)]
struct Env {
    vars: BTreeMap<String, String>,
    ctx: Option<Ctx>,
    shallow: bool,
}

impl Env {
    fn with(&self, more: &[(String, String)]) -> Env {
        let mut e = self.clone();
        e.vars.extend(more.iter().cloned());
        e
    }
}

/// Whether `{name}` is a template variable (`{E}`, `{S.type}`) rather than a child placeholder (`{0}`).
fn is_var(name: &str) -> bool {
    let (head, tail) = name.split_once('.').unwrap_or((name, ""));
    let mut c = head.chars();
    c.next().is_some_and(|x| x.is_ascii_uppercase())
        && c.all(|x| x.is_ascii_uppercase() || x.is_ascii_digit())
        && (tail.is_empty() || tail.chars().all(|x| x.is_ascii_lowercase() || x.is_ascii_digit()))
        && !tail.contains('.')
}

/// Every `{VAR}` of the text replaced by its binding; `{0}`, `{1}`, ... (the children) are left alone.
fn subst(text: &str, env: &Env) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        match after.find('}') {
            Some(j) if is_var(&after[..j]) => {
                let v = env.vars.get(&after[..j]).ok_or_else(|| format!("the variable {{{}}} of {text:?} is not bound", &after[..j]))?;
                out.push_str(v);
                rest = &after[j + 1..];
            }
            _ => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// THE NAME OF A BOUND VARIABLE: the `k`-th variable of its class in scope (`docs/PLAN_futhark_kingdom.md` 3g), so a
/// context is the count of each class's variables, never the frame stack. An index `ix.D` is `i_D`, a loop state of
/// class `C` is `s_C` with `.` as `_` (`s_i32`, `s_a_cols`); the `k`-th, `k >= 2`, puts `k` after the letter
/// (`i2_cols`, `s2_a_cols`).
pub fn var_name(cls: &str, k: u32) -> String {
    let ord = if k == 1 { String::new() } else { k.to_string() };
    match cls.strip_prefix("ix.") {
        Some(d) => format!("i{ord}_{d}"),
        None => format!("s{ord}_{}", cls.replace('.', "_")),
    }
}

/// The name a binder of class `cls` gives its variable in context `ctx` (one past those in scope).
fn next_var(cls: &str, ctx: Option<&Ctx>) -> String {
    var_name(cls, ctx.map_or(0, |c| c.count(cls)) + 1)
}

/// The sets a view's quantifiers range over.
#[derive(Debug, Clone, Default)]
struct Sets {
    /// The view's elements, in order; `multi` when there are several (an array class is then written with its
    /// element).
    elements: Vec<String>,
    multi: bool,
    dims: Vec<String>,
    classes: Vec<Class>,
    rank1: Vec<Class>,
    /// The classes a loop may carry: the scalar of each element, then the rank-1 classes.
    states: Vec<Class>,
    /// The pairs of scalars a loop may carry (destructured: `loop (s, c) = ...`), every ordered pair of the
    /// elements.
    tuples: Vec<Class>,
    free: Vec<Class>,
    inputs: Vec<(String, Class)>,
    roles: Vec<String>,
    pairs: Vec<(String, String)>,
}

impl Sets {
    fn text(&self, c: &Class) -> String {
        c.text(self.multi)
    }
}

/// The sets of a signature view, derived from the specification: the dims in order of first appearance, the classes
/// (each input's, its rows', the output's, each dim's rank-1 class of each element), the rank-1 classes, the states
/// a loop may carry, the pairs of scalars. The context-free classes are derived by [`view_rows`].
fn signature_sets(spec: &ViewSpec) -> Result<Sets, String> {
    let elements = spec.elements.clone();
    let mut dims: Vec<String> = Vec::new();
    for p in &spec.inputs {
        if p.elem == "size" {
            dims.push(p.name.clone());
        }
        for d in &p.dims {
            if !dims.contains(d) {
                dims.push(d.clone());
            }
        }
    }
    let output = spec.output.as_ref().ok_or("the signature specification has no output")?;
    if let Some(d) = output.dims.iter().find(|d| !dims.contains(d)) {
        return Err(format!("the output's dimension {d} is not a dimension of an input {dims:?}"));
    }
    let mut classes: Vec<Class> = Vec::new();
    fn add(classes: &mut Vec<Class>, c: Class) {
        if !classes.contains(&c) {
            let rows = match &c {
                Class::Array { elem, dims } if dims.len() >= 2 => Some(Class::Array { elem: elem.clone(), dims: dims[1..].to_vec() }),
                _ => None,
            };
            classes.push(c);
            if let Some(r) = rows {
                add(classes, r);
            }
        }
    }
    let class_of = |p: &SpecInput| if p.dims.is_empty() { Class::Scalar(p.elem.clone()) } else { Class::Array { elem: p.elem.clone(), dims: p.dims.clone() } };
    for p in spec.inputs.iter().filter(|p| p.elem != "size") {
        add(&mut classes, class_of(p));
    }
    add(&mut classes, class_of(output));
    for d in &dims {
        for e in &elements {
            add(&mut classes, Class::Array { elem: e.clone(), dims: vec![d.clone()] });
        }
    }
    let rank1: Vec<Class> = classes.iter().filter(|c| c.dims().len() == 1).cloned().collect();
    let states: Vec<Class> = elements.iter().map(|e| Class::Scalar(e.clone())).chain(rank1.iter().cloned()).collect();
    let tuples: Vec<Class> = elements.iter().flat_map(|a| elements.iter().map(move |b| Class::Tuple(vec![Class::Scalar(a.clone()), Class::Scalar(b.clone())]))).collect();
    let inputs = spec.inputs.iter().filter(|p| p.elem != "size").map(|p| (p.name.clone(), class_of(p))).collect();
    Ok(Sets { multi: elements.len() > 1, elements, dims, classes, rank1, states, tuples, free: Vec::new(), inputs, ..Sets::default() })
}

/// The frames of a view (a frame template expanded over its scope): the code and the classes it binds.
type Frames = Vec<(String, Vec<String>)>;

struct Binder {
    sets: Sets,
    depth: u32,
    frames: Frames,
    contexts: Vec<Ctx>,
}

impl Binder {
    fn new(spec: &ViewSpec, master: &Master) -> Result<Binder, String> {
        let mut b = match spec.shape {
            Shape::Signature => Binder { sets: signature_sets(spec)?, depth: spec.depth, frames: Vec::new(), contexts: Vec::new() },
            Shape::Matrix => {
                let pairs = spec.roles.iter().flat_map(|a| spec.roles.iter().map(move |c| (a.clone(), c.clone()))).collect();
                Binder { sets: Sets { elements: spec.elements.clone(), roles: spec.roles.clone(), pairs, ..Sets::default() }, depth: 0, frames: Vec::new(), contexts: vec![Ctx::default()] }
            }
        };
        if spec.shape == Shape::Signature {
            let base = b.base_env();
            // The frames the pinned master version has (a frame added later, as the pair loop's, makes contexts a
            // view of before must not see).
            for f in master.frames.iter().filter(|f| f.since <= spec.master.unwrap_or(master.version)) {
                for env in b.expand_scope(&f.scope, &base)? {
                    let code = subst(&f.code, &env)?;
                    b.frames.push((code, f.binds.iter().map(|c| subst(c, &env)).collect::<Result<Vec<_>, _>>()?));
                }
            }
            // The contexts: the empty one, then every context of depth d extended by every frame (d < depth), each
            // (set, depth) once, in order of first appearance.
            let (mut ctxs, mut frontier) = (vec![Ctx::default()], vec![Ctx::default()]);
            for _ in 0..b.depth {
                let mut next: Vec<Ctx> = Vec::new();
                for c in &frontier {
                    for (_, binds) in &b.frames {
                        let n = c.push(binds);
                        if !next.contains(&n) {
                            next.push(n);
                        }
                    }
                }
                ctxs.extend(next.iter().cloned());
                frontier = next;
            }
            b.contexts = ctxs;
            // A variable's name must not be an input's, a size's or a zero array's: it would shadow the interface.
            // And two classes a frame binds must not share a variable name (the name is made from the class's text,
            // so it is a Futhark identifier, and one name a class).
            let zeros = b.sets.rank1.iter().map(|c| format!("zeros_{}", c.tag(b.sets.multi)));
            let taken: BTreeSet<String> = b.sets.inputs.iter().map(|(n, _)| n.clone()).chain(b.sets.dims.iter().cloned()).chain(zeros).collect();
            let mut named: BTreeMap<String, String> = BTreeMap::new();
            for (_, binds) in &b.frames {
                for c in binds {
                    for k in 1..=b.depth.max(1) {
                        let v = var_name(c, k);
                        if !v.chars().all(|x| x.is_ascii_alphanumeric() || x == '_') {
                            return Err(format!("the bound variable {v} of the class {c} is not a Futhark identifier"));
                        }
                        let t = c.strip_prefix("ix.").map(|_| format!("t{v}"));
                        if let Some(x) = std::iter::once(&v).chain(t.as_ref()).find(|x| taken.contains(*x)) {
                            return Err(format!("the bound variable {x} has the name of an input, a size or a zero array of the view"));
                        }
                        if let Some(other) = named.insert(v.clone(), c.clone()).filter(|o| o != c) {
                            return Err(format!("the classes {other} and {c} would both name a variable {v}"));
                        }
                    }
                }
            }
        }
        Ok(b)
    }

    /// The base bindings of an expansion: the element `{E}` when the view has one (a view of several binds it per
    /// row, from the class a quantifier picks or by iterating the elements).
    fn base_env(&self) -> Env {
        match self.sets.elements.as_slice() {
            [one] => self.elem_env(&Env::default(), one),
            _ => Env::default(),
        }
    }

    /// The bindings with the element `{E}` set to `e`, and `{E.tag}`, the element as a piece of a row's name: `.e`
    /// in a view of several elements, nothing in a view of one (`clamp@ix.n:c` keeps its name; a view of two has
    /// `clamp.f32@ix.n:c` and `clamp.i32@ix.n:c`, two rows of one output type told apart by their element).
    fn elem_env(&self, env: &Env, e: &str) -> Env {
        env.with(&[("E".into(), e.to_string()), ("E.tag".into(), if self.sets.multi { format!(".{e}") } else { String::new() })])
    }

    /// The elements a row is defined for among the view's, in the view's order.
    fn row_elements(&self, row: &MasterRow) -> Vec<String> {
        self.sets.elements.iter().filter(|e| row.elems == "*" || row.elems.split(',').any(|x| x == e.as_str())).cloned().collect()
    }

    /// A class from a template's text, bound: an array written without its element is of the row's `{E}`.
    fn class_of(&self, text: &str, env: &Env) -> Result<Class, String> {
        let t = subst(text, env)?;
        if self.sets.elements.is_empty() || t.starts_with("mat.") || t.starts_with("fn.") {
            return Ok(Class::Named(t));
        }
        Class::parse(&t, &self.sets.elements, env.vars.get("E").map(String::as_str))
    }

    /// The variables a class binds: the class, the next variable of its class (`.var`), its Futhark type, its
    /// dimensions (`.dims`, `.br` as brackets, `.d0` the first), `.ix`, the next index variable over its first
    /// dimension (a tabulate of its rows binds it), and `.tag`, the class as a piece of an identifier (`zeros_{D.tag}`).
    fn class_env(&self, var: &str, cls: &Class, ctx: Option<&Ctx>) -> Vec<(String, String)> {
        let text = self.sets.text(cls);
        let ds = cls.dims();
        vec![
            (var.to_string(), text.clone()),
            (format!("{var}.var"), next_var(&text, ctx)),
            (format!("{var}.type"), cls.futhark_type()),
            (format!("{var}.dims"), ds.join(".")),
            (format!("{var}.br"), ds.join("][")),
            (format!("{var}.d0"), ds.first().cloned().unwrap_or_default()),
            (format!("{var}.ix"), ds.first().map_or(String::new(), |d| next_var(&format!("ix.{d}"), ctx))),
            (format!("{var}.tag"), cls.tag(self.sets.multi)),
        ]
    }

    /// The bindings with the row's element `{E}` set from a class a quantifier picked: the class's own when `{E}` is
    /// unbound; unchanged when it is the bound one; a class of another element binds nothing (`None`).
    fn with_elem(&self, env: &Env, cls: &Class) -> Option<Env> {
        match (cls.elem(), env.vars.get("E")) {
            (Some(elem), Some(bound)) if bound != elem => None,
            (Some(elem), None) => Some(self.elem_env(env, elem)),
            _ => Some(env.clone()),
        }
    }

    /// A class's bindings ([`Self::class_env`]) with the row's element ([`Self::with_elem`]).
    fn class_env_elem(&self, var: &str, cls: &Class, env: &Env) -> Option<Env> {
        self.with_elem(env, cls).map(|e| e.with(&self.class_env(var, cls, env.ctx.as_ref())))
    }

    /// The variables a context binds: (name, class), by class, then outermost first.
    fn frame_vars(&self, ctx: &Ctx) -> Vec<(String, String)> {
        ctx.classes.iter().flat_map(|(c, n)| (1..=*n).map(move |k| (var_name(c, k), c.clone()))).collect()
    }

    fn expand_scope(&self, scope: &[String], env: &Env) -> Result<Vec<Env>, String> {
        let Some((q, rest)) = scope.split_first() else { return Ok(vec![env.clone()]) };
        let mut out = Vec::new();
        for b in self.quantifier(q, env)? {
            out.extend(self.expand_scope(rest, &b)?);
        }
        Ok(out)
    }

    /// The bindings of one quantifier, in order.
    fn quantifier(&self, q: &str, env: &Env) -> Result<Vec<Env>, String> {
        if q == "C" {
            return Ok(self
                .contexts
                .iter()
                .map(|ctx| {
                    let mut e = env.with(&[("K".into(), (ctx.depth + 1).to_string()), ("CTX".into(), ctx.name())]);
                    e.ctx = Some(ctx.clone());
                    e.shallow = ctx.depth < self.depth;
                    e
                })
                .collect());
        }
        let (var, kind) = q.split_once(':').ok_or_else(|| format!("the quantifier {q:?} does not parse"))?;
        let (kind, arg) = match kind.split_once('(') {
            Some((k, a)) => (k, Some(subst(a.strip_suffix(')').ok_or_else(|| format!("the quantifier {q:?} does not parse"))?, env)?)),
            None => (kind, None),
        };
        let s = &self.sets;
        let one = |pairs: Vec<(String, String)>| env.with(&pairs);
        let v = |x: &str| vec![(var.to_string(), x.to_string())];
        let arg = arg.unwrap_or_default();
        // A quantifier's class argument, bound (`vars(a.{D})`, `rows(a.{D})`, `class(a.{D2}.{D})`, `inputs({X})`).
        let arg_class =|| -> Result<Option<Class>, String> { Ok(if arg.is_empty() { None } else { Some(self.class_of(&arg, env)?) }) };
        Ok(match kind {
            "dims" => s.dims.iter().map(|d| one(vec![(var.to_string(), d.clone()), (format!("{var}.ix"), next_var(&format!("ix.{d}"), env.ctx.as_ref()))])).collect(),
            // A rank-1 class binds its dimension, and the row's element: the class's own, or none of another
            // element's when `{E}` is bound already.
            "rank1" => s
                .rank1
                .iter()
                .filter_map(|c| {
                    let d = c.dims()[0].clone();
                    let e = self.with_elem(env, c)?;
                    Some(e.with(&[(var.to_string(), d.clone()), (format!("{var}.ix"), next_var(&format!("ix.{d}"), env.ctx.as_ref())), (format!("{var}.tag"), c.tag(s.multi))]))
                })
                .collect(),
            "states" => s.states.iter().filter_map(|c| self.class_env_elem(var, c, env)).collect(),
            "free" => s.free.iter().filter_map(|c| self.class_env_elem(var, c, env)).collect(),
            // The pairs of scalars a loop may carry, destructured: `{S}` the class, `.a` and `.b` its components,
            // `.type` its Futhark type, `.var` the pattern of its two variables (the next of each component's
            // class; the second one further when the components are of one class), `.tag` the components joined by
            // `.` (a row's name), `.zero` the pair of the components' zeros.
            "tuples" => s
                .tuples
                .iter()
                .map(|t| {
                    let Class::Tuple(parts) = t else { return Err(format!("our software has a bug: the tuple {} is not one", s.text(t))) };
                    let [a, b] = parts.as_slice() else { return Err(format!("our software has a bug: the tuple {} is not a pair", s.text(t))) };
                    let (ta, tb) = (s.text(a), s.text(b));
                    let va = next_var(&ta, env.ctx.as_ref());
                    let vb = var_name(&tb, env.ctx.as_ref().map_or(0, |c| c.count(&tb)) + 1 + u32::from(a == b));
                    let zero = |c: &Class| format!("0{}", c.elem().unwrap_or(""));
                    Ok(one(vec![
                        (var.to_string(), s.text(t)),
                        (format!("{var}.a"), ta.clone()),
                        (format!("{var}.b"), tb.clone()),
                        (format!("{var}.type"), t.futhark_type()),
                        (format!("{var}.var"), format!("({va}, {vb})")),
                        (format!("{var}.tag"), format!("{ta}.{tb}")),
                        (format!("{var}.zero"), format!("({}, {})", zero(a), zero(b))),
                    ]))
                })
                .collect::<Result<Vec<_>, _>>()?,
            "vars" => {
                let ctx = env.ctx.as_ref().ok_or_else(|| format!("the quantifier {q} outside a context"))?;
                let want = arg_class()?.map(|c| s.text(&c)).unwrap_or_default();
                self.frame_vars(ctx).into_iter().filter(|(_, c)| *c == want).map(|(n, _)| one(v(&n))).collect()
            }
            "inputs" => {
                let want = arg_class()?;
                s.inputs.iter().filter(|(_, c)| Some(c) == want.as_ref()).map(|(n, _)| one(v(n))).collect()
            }
            // The named class, when the view has it (`V2:class(a.{D2}.{D})`: the rows a histogram of rows reads).
            "class" => {
                let want = arg_class()?;
                s.classes.iter().filter(|c| Some(*c) == want.as_ref()).filter_map(|c| self.class_env_elem(var, c, env)).collect()
            }
            "rows" => {
                let want = arg_class()?;
                s.classes
                    .iter()
                    .filter(|y| match y {
                        Class::Array { elem, dims } if dims.len() >= 2 => Some(Class::Array { elem: elem.clone(), dims: dims[1..].to_vec() }) == want,
                        _ => false,
                    })
                    .filter_map(|y| self.class_env_elem(var, y, env))
                    .collect()
            }
            "pairs" => s.pairs.iter().map(|(a, b)| one(vec![(var.into(), format!("{a}.{b}")), (format!("{var}.a"), a.clone()), (format!("{var}.b"), b.clone())])).collect(),
            "swap" => {
                let (a, b) = arg.split_once('.').ok_or_else(|| format!("the swap of {arg:?}"))?;
                if s.pairs.contains(&(b.to_string(), a.to_string())) {
                    vec![one(v(&format!("{b}.{a}")))]
                } else {
                    vec![]
                }
            }
            "triples" => {
                let mut out = Vec::new();
                for a in &s.roles {
                    for b in &s.roles {
                        for c in &s.roles {
                            let has = |x: &String, y: &String| s.pairs.contains(&(x.clone(), y.clone()));
                            if has(a, b) && has(b, c) && has(a, c) {
                                out.push(one(vec![
                                    (var.into(), format!("{a}.{b}.{c}")),
                                    (format!("{var}.ab"), format!("{a}.{b}")),
                                    (format!("{var}.bc"), format!("{b}.{c}")),
                                    (format!("{var}.ac"), format!("{a}.{c}")),
                                ]));
                            }
                        }
                    }
                }
                out
            }
            other => return Err(format!("the quantifier kind {other:?} is not one of the master's")),
        })
    }

    /// A type template in a row's context as the dual: `class:context`, or the bare class when the row is outside any
    /// context or the class is context-free. `T+<frame>` is `T` in the context plus that frame.
    fn dual(&self, ty: &str, env: &Env) -> Result<String, String> {
        let ty = subst(ty, env)?;
        let (cls, frame) = match ty.split_once('+') {
            Some((c, f)) => (c.to_string(), Some(f.to_string())),
            None => (ty.clone(), None),
        };
        let class = self.class_of(&cls, env)?;
        let cls = self.sets.text(&class);
        let Some(ctx) = env.ctx.as_ref().filter(|_| !self.sets.free.contains(&class)) else {
            if frame.is_some() {
                return Err(format!("the type {ty} pushes a frame outside a context"));
            }
            return Ok(cls);
        };
        let ctx = match frame {
            Some(f) => {
                let (_, binds) = self.frames.iter().find(|(c, _)| *c == f).ok_or_else(|| format!("the type {ty} pushes the frame {f}, which the view does not have"))?;
                ctx.push(binds)
            }
            None => ctx.clone(),
        };
        Ok(format!("{cls}:{}", ctx.name()))
    }
}

/// A row of a view, bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewRow {
    /// The master row it is an instance of.
    pub reference: String,
    pub name: String,
    pub kind: String,
    pub out: String,
    pub args: Vec<String>,
    pub spelling: String,
    pub note: String,
}

/// Whether the specification admits a master row: its version and its element, and for the matrix view its family. A
/// signature view admits every family; a row whose quantifiers range over sets the view does not have (`P:pairs` in a
/// signature view) expands to no rows, and [`reachable`] drops what nothing can use.
fn admitted(row: &MasterRow, spec: &ViewSpec, master: &Master) -> bool {
    (spec.shape == Shape::Signature || spec.families.contains(&row.family))
        && row.since <= spec.master.unwrap_or(master.version)
        && (row.elems == "*" || row.elems.split(',').any(|e| spec.elements.iter().any(|v| v == e)))
}

/// Whether a template's text reads the element `{E}`.
fn reads_elem(text: &str) -> bool {
    text.contains("{E}")
}

/// THE ELEMENT OF A ROW in a view of several elements. `{E}` is bound by the first quantifier that picks a class of
/// one element (`rank1`, `states`, `free`, `rows`, `class`: the class's own), and a class of another element binds
/// nothing then. A quantifier whose argument reads `{E}` before any binds it (`V:vars({E})`), and a row whose
/// templates read `{E}` with none bound (`add@{OUT}` over `{E};{E}`, `ival.{D}` over `D:dims`), iterate the
/// elements the row is defined for, in the view's order. A row whose templates read no `{E}` is one row. In a view
/// of one element `{E}` is bound from the start, so the rows are those of before.
fn expand(rows: &[(&MasterRow, &[String])], binder: &Binder, env: &Env, out: &mut Vec<ViewRow>) -> Result<(), String> {
    let mut i = 0;
    while i < rows.len() {
        let (row, scope) = rows[i];
        let Some(q) = scope.first() else {
            leaf(row, binder, env, out)?;
            i += 1;
            continue;
        };
        let mut j = i;
        while j < rows.len() && rows[j].1.first() == Some(q) {
            j += 1;
        }
        let block: Vec<(&MasterRow, &[String])> = rows[i..j].iter().map(|(r, s)| (*r, &s[1..])).collect();
        let envs: Vec<Env> = if reads_elem(q) && !env.vars.contains_key("E") {
            binder.sets.elements.iter().map(|e| binder.elem_env(env, e)).collect()
        } else {
            vec![env.clone()]
        };
        for e in &envs {
            for b in binder.quantifier(q, e)? {
                expand(&block, binder, &b, out)?;
            }
        }
        i = j;
    }
    Ok(())
}

/// A row with every quantifier bound: instantiated when its condition holds, once an element it is defined for
/// (the bound `{E}`, or each of the view's it reads `{E}` for; see [`expand`]).
fn leaf(row: &MasterRow, binder: &Binder, env: &Env, out: &mut Vec<ViewRow>) -> Result<(), String> {
    match row.when.as_str() {
        "" => {}
        "shallow" if env.shallow => {}
        "shallow" => return Ok(()),
        other => return Err(format!("the row {} has the condition {other:?}", row.reference)),
    }
    let mine = binder.row_elements(row);
    match env.vars.get("E") {
        Some(e) => {
            if mine.contains(e) {
                out.push(instantiate(row, binder, env)?);
            }
        }
        None if [&row.name, &row.out, &row.spelling, &row.note].into_iter().any(|t| reads_elem(t)) || row.args.iter().any(|a| reads_elem(a)) => {
            for e in mine {
                out.push(instantiate(row, binder, &binder.elem_env(env, &e))?);
            }
        }
        None => out.push(instantiate(row, binder, env)?),
    }
    Ok(())
}

fn instantiate(row: &MasterRow, binder: &Binder, env: &Env) -> Result<ViewRow, String> {
    let out = binder.dual(&row.out, env)?;
    let args = row.args.iter().map(|a| binder.dual(a, env)).collect::<Result<Vec<_>, _>>()?;
    let e = env.with(&[("OUT".into(), out.clone())]);
    Ok(ViewRow { reference: row.reference.clone(), name: subst(&row.name, &e)?, kind: row.kind.clone(), out, args, spelling: subst(&row.spelling, &e)?, note: subst(&row.note, &e)? })
}

/// The rows reachable both ways. Forward: F, the least fixpoint of the types that can be built (the output of every
/// terminal and constant, then of every function whose arguments are all in F); a row is usable when all its arguments
/// are in F. Backward: B, the closure from the root over the usable rows. Kept: the usable rows whose output is in B,
/// in expansion order. The root must be in F.
fn reachable(rows: Vec<ViewRow>, root: &str) -> Result<Vec<ViewRow>, String> {
    let mut built: BTreeSet<&str> = rows.iter().filter(|r| r.kind != "function").map(|r| r.out.as_str()).collect();
    loop {
        let before = built.len();
        for r in &rows {
            if !built.contains(r.out.as_str()) && r.args.iter().all(|a| built.contains(a.as_str())) {
                built.insert(&r.out);
            }
        }
        if built.len() == before {
            break;
        }
    }
    if !built.contains(root) {
        return Err(format!("the root {root} cannot be built from the view's leaves by any chain of its functions"));
    }
    let usable: Vec<bool> = rows.iter().map(|r| r.args.iter().all(|a| built.contains(a.as_str()))).collect();
    let mut keep: BTreeSet<&str> = BTreeSet::from([root]);
    loop {
        let before = keep.len();
        for (r, _) in rows.iter().zip(&usable).filter(|(_, u)| **u) {
            if keep.contains(r.out.as_str()) {
                keep.extend(r.args.iter().map(String::as_str));
            }
        }
        if keep.len() == before {
            break;
        }
    }
    let kept: Vec<bool> = rows.iter().zip(&usable).map(|(r, u)| *u && keep.contains(r.out.as_str())).collect();
    Ok(rows.into_iter().zip(kept).filter(|(_, k)| *k).map(|(r, _)| r).collect())
}

/// Steps 1 to 3: the view's rows, filtered, bound, and kept when reachable both ways; the root; the binder.
fn view_rows(spec: &ViewSpec, master: &Master) -> Result<(Vec<ViewRow>, String, Binder), String> {
    let mut binder = Binder::new(spec, master)?;
    let picked: Vec<(&MasterRow, &[String])> = master.rows.iter().filter(|r| admitted(r, spec, master)).map(|r| (r, r.scope.as_slice())).collect();
    let base = binder.base_env();
    let o = match spec.shape {
        Shape::Matrix => {
            let root = format!("mat.{}", spec.root.join("."));
            let mut rows = Vec::new();
            expand(&picked, &binder, &base, &mut rows)?;
            return Ok((reachable(rows, &root)?, root, binder));
        }
        Shape::Signature => spec.output.as_ref().ok_or("the signature specification has no output")?,
    };
    let cls = binder.sets.text(&if o.dims.is_empty() { Class::Scalar(o.elem.clone()) } else { Class::Array { elem: o.elem.clone(), dims: o.dims.clone() } });
    // A class that no KEPT function row of the view produces is CONTEXT-FREE (the interface's value, one type in every
    // context). Which rows are kept depends on which classes are free, so the view iterates expand -> keep ->
    // recompute until the free classes settle (from none free; bounded by the class count: a run that does not settle
    // is a bug in the master or here, never a guess).
    for _ in 0..binder.sets.classes.len() + 2 {
        let mut rows = Vec::new();
        expand(&picked, &binder, &base, &mut rows)?;
        let root = binder.dual(&cls, &Env { ctx: Some(Ctx::default()), ..Env::default() })?;
        let rows = reachable(rows, &root)?;
        let produced: BTreeSet<&str> = rows.iter().filter(|r| r.kind == "function").map(|r| r.out.split(':').next().unwrap_or("")).collect();
        let free: Vec<Class> = binder.sets.classes.iter().filter(|c| !produced.contains(binder.sets.text(c).as_str())).cloned().collect();
        if free == binder.sets.free {
            return Ok((rows, root, binder));
        }
        binder.sets.free = free;
    }
    Err(format!("the context-free classes of the view do not settle (last {:?})", binder.sets.free))
}

// ---- the written tables -----------------------------------------------------------------------------------------

/// A type of a view, as `types.tsv` writes it.
#[derive(Debug, Clone)]
struct ViewType {
    dual: String,
    slot: String,
    form: String,
    literals: Vec<String>,
    root: bool,
    meaning: String,
}

/// The types of a signature view: every dual its kept rows name, the root first, then by depth and name. A scalar's
/// slot is its element and it carries the element's literal pool; an index's is `ix`, an array's `arr`, a tuple's
/// `tuple`.
fn signature_types(spec: &ViewSpec, rows: &[ViewRow], root: &str) -> Result<Vec<ViewType>, String> {
    let mut named: Vec<&str> = Vec::new();
    for r in rows {
        for d in std::iter::once(&r.out).chain(&r.args) {
            if !named.contains(&d.as_str()) {
                named.push(d);
            }
        }
    }
    let depth = |d: &str| Ctx::of_dual(d).ok().flatten().map_or(0, |c| c.depth);
    named.sort_by(|a, b| (*a != root, depth(a), *a).cmp(&(*b != root, depth(b), *b)));
    named
        .into_iter()
        .map(|d| {
            let (cls, ctx) = d.split_once(':').unwrap_or((d, ""));
            let class = Class::parse(cls, &spec.elements, None)?;
            let (slot, literals, meaning) = match &class {
                Class::Index(dim) => ("ix".to_string(), vec![], format!("an index in range for [{dim}]")),
                Class::Array { elem, dims } => ("arr".to_string(), vec![], format!("a [{}]{elem}", dims.join("]["))),
                Class::Tuple(parts) => ("tuple".to_string(), vec![], format!("a pair of {}", parts.iter().map(|p| format!("a {}", p.futhark_type())).collect::<Vec<_>>().join(" and "))),
                Class::Scalar(elem) => {
                    let pool = spec.literals.get(elem).ok_or_else(|| format!("the element {elem} has no literal pool"))?;
                    (elem.clone(), pool.iter().map(|l| format!("{l}{elem}")).collect(), format!("an {elem}"))
                }
                Class::Named(n) => return Err(format!("our software has a bug: the signature view names the class {n}")),
            };
            let meaning = meaning + &if ctx.is_empty() { ", the interface's input class".to_string() } else { format!(", in the context {ctx}") };
            Ok(ViewType { dual: d.to_string(), slot, form: cls.to_string(), literals, root: d == root, meaning })
        })
        .collect()
}

fn ordered(rows: Vec<ViewRow>) -> Vec<ViewRow> {
    let (f, t): (Vec<ViewRow>, Vec<ViewRow>) = rows.into_iter().partition(|r| r.kind == "function");
    f.into_iter().chain(t).collect()
}

fn types_text(types: &[ViewType]) -> String {
    let mut out = String::from("ref\tdual\tslot\tform\tliterals\troot\tmeaning\n");
    for (i, t) in types.iter().enumerate() {
        out.push_str(&format!("T{:04}\t{}\t{}\t{}\t{}\t{}\t{}\n", i + 1, t.dual, t.slot, t.form, t.literals.join(","), if t.root { "yes" } else { "no" }, t.meaning));
    }
    out
}

fn functions_text(rows: &[ViewRow]) -> String {
    let mut out = String::from("ref\tname\tkind\tout\targs\tspelling\tnote\n");
    for (i, r) in rows.iter().enumerate() {
        out.push_str(&format!("F{:05}\t{}\t{}\t{}\t{}\t{}\t{}\n", i + 1, r.name, r.kind, r.out, r.args.join(";"), r.spelling, r.note));
    }
    out
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

fn sig_sort(dual: &str) -> String {
    format!("S{}", sanitize(dual))
}

fn sig_ctor(name: &str, kind: &str, out: &str) -> String {
    if kind == "const" {
        return format!("C{}", sanitize(out));
    }
    format!("X{}__{}", sanitize(name.split('@').next().unwrap_or("")), sanitize(out))
}

/// A rule template's `{op}` constructors at the type `dual`.
fn rule_text(rule: &str, dual: &str) -> String {
    let mut out = String::new();
    let mut rest = rule;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let j = after.find('}').unwrap_or(after.len());
        let op = &after[..j];
        if !op.is_empty() && op.chars().next().is_some_and(|c| c.is_ascii_lowercase()) && j < after.len() {
            out.push_str(&sig_ctor(&format!("{op}@{dual}"), "function", dual));
            rest = &after[j + 1..];
        } else {
            out.push('{');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

// ---- the view ---------------------------------------------------------------------------------------------------

/// A view: its specification and the exact table it writes. `datatype`, `rules`, `egraph_rows` and `sorts` are the
/// e-graph artefacts of a signature view (empty for the matrix view, whose e-graph the generator writes).
#[derive(Debug, Clone)]
pub struct View {
    pub spec: ViewSpec,
    pub types_tsv: String,
    pub functions_tsv: String,
    pub datatype: String,
    pub rules: String,
    pub egraph_rows: String,
    pub sorts: String,
}

impl View {
    /// The view of a specification over a master.
    pub fn load(master: &Master, spec_text: &str) -> Result<View, String> {
        let spec = ViewSpec::parse(spec_text)?;
        let (rows, root, binder) = view_rows(&spec, master)?;
        match spec.shape {
            Shape::Signature => {
                let types = signature_types(&spec, &rows, &root)?;
                let rows = ordered(rows);
                let mut v = View { types_tsv: types_text(&types), functions_tsv: functions_text(&rows), datatype: String::new(), rules: String::new(), egraph_rows: String::new(), sorts: String::new(), spec };
                v.signature_egraph(&types, &rows, master)?;
                Ok(v)
            }
            Shape::Matrix => {
                let named: BTreeSet<&str> = rows.iter().flat_map(|r| std::iter::once(r.out.as_str()).chain(r.args.iter().map(String::as_str))).collect();
                let mut types = Vec::new();
                let base = binder.base_env();
                for t in master.types.iter().filter(|t| spec.families.contains(&t.family)) {
                    for env in binder.expand_scope(&t.scope, &base)? {
                        let dual = subst(&t.dual, &env)?;
                        if named.contains(dual.as_str()) {
                            types.push(ViewType { root: dual == root, form: subst(&t.form, &env)?, meaning: subst(&t.meaning, &env)?, slot: t.slot.clone(), literals: vec![], dual });
                        }
                    }
                }
                let rows = ordered(rows);
                Ok(View { types_tsv: types_text(&types), functions_tsv: functions_text(&rows), datatype: String::new(), rules: String::new(), egraph_rows: String::new(), sorts: String::new(), spec })
            }
        }
    }

    /// The signature view's e-graph: one sort a dual, one constructor a row (a constant takes its pool index), and the
    /// master's rules on every type they apply to (`elem`: the scalar types; `arr`: the rank-1 array types in a
    /// context that hold the row the rule `needs`; `tuple`: the pair types, whose `{fst}` and `{snd}` are the
    /// projections at the components' types), for the element types each rule is exact for (a pair's: both
    /// components').
    fn signature_egraph(&mut self, types: &[ViewType], rows: &[ViewRow], master: &Master) -> Result<(), String> {
        let elements = self.spec.elements.clone();
        let mut data = String::from("; The signature family's e-graph datatype, generated by gen_tables.py.\n; One sort per dual: the sort IS the type (class and context).\n\n");
        for t in types {
            data.push_str(&format!("(sort {})\n", sig_sort(&t.dual)));
        }
        data.push('\n');
        for r in rows {
            let args = if r.kind == "const" { "i64".to_string() } else { r.args.iter().map(|a| sig_sort(a)).collect::<Vec<_>>().join(" ") };
            data.push_str(&format!("(constructor {} ({args}) {})\n", sig_ctor(&r.name, &r.kind, &r.out), sig_sort(&r.out)));
        }
        let names: BTreeSet<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        let mut rules = String::from("; The signature family's e-graph ruleset, generated by gen_tables.py.\n(ruleset futhark_sig)\n\n");
        for on in ["elem", "arr", "tuple"] {
            for t in types {
                let d = &t.dual;
                let (cls, ctx) = d.split_once(':').unwrap_or((d, ""));
                let class = Class::parse(cls, &elements, None)?;
                // The elements a rule must be exact for to apply at this type.
                let exact_for: Vec<&str> = match (on, &class) {
                    ("elem", Class::Scalar(e)) => vec![e],
                    ("arr", Class::Array { elem, dims }) if dims.len() == 1 && !ctx.is_empty() => vec![elem],
                    ("tuple", Class::Tuple(parts)) => parts.iter().filter_map(Class::elem).collect(),
                    _ => continue,
                };
                let mine: Vec<&RuleTemplate> = master
                    .rules
                    .iter()
                    .filter(|r| r.on == on && exact_for.iter().all(|e| r.elems == "*" || r.elems.split(',').any(|x| x == *e)) && (r.needs.is_empty() || names.contains(format!("{}@{d}", r.needs).as_str())))
                    .collect();
                if mine.is_empty() {
                    continue;
                }
                rules.push_str(&format!("; {d}\n"));
                for r in mine {
                    let text = match &class {
                        Class::Tuple(parts) => {
                            let tag = parts.iter().map(|p| p.text(self.spec.multi())).collect::<Vec<_>>().join(".");
                            let at = |p: &Class| format!("{}:{ctx}", p.text(self.spec.multi()));
                            rule_text(&r.rule, d).replace(&sig_ctor(&format!("fst@{d}"), "function", d), &sig_ctor(&format!("fst.{tag}@{}", at(&parts[0])), "function", &at(&parts[0]))).replace(&sig_ctor(&format!("snd@{d}"), "function", d), &sig_ctor(&format!("snd.{tag}@{}", at(&parts[1])), "function", &at(&parts[1])))
                        }
                        _ => rule_text(&r.rule, d),
                    };
                    rules.push_str(&text);
                    rules.push('\n');
                }
                rules.push('\n');
            }
        }
        let mut rtext = String::from("# row name\tconstructor\tsort\n");
        for r in rows {
            rtext.push_str(&format!("{}\t{}\t{}\n", r.name, sig_ctor(&r.name, &r.kind, &r.out), sig_sort(&r.out)));
        }
        let mut stext = String::from("# dual\tsort\n");
        for t in types {
            stext.push_str(&format!("{}\t{}\n", t.dual, sig_sort(&t.dual)));
        }
        self.datatype = data;
        self.rules = rules;
        self.egraph_rows = rtext;
        self.sorts = stext;
        Ok(())
    }

    /// The kingdom the view is, as the engine loads every Futhark kingdom.
    pub fn kingdom(&self, name: &str) -> Result<FutharkKingdom, String> {
        FutharkKingdom::from_tables(FutharkFamily::Shape, name, &self.types_tsv, &self.functions_tsv)
    }

    /// The view written out as an exact table: every artefact, by file name (the option the plan keeps: for the user,
    /// a cache, or the witness tests).
    pub fn artefacts(&self) -> Vec<(&'static str, &str)> {
        let mut out = vec![("types.tsv", self.types_tsv.as_str()), ("functions.tsv", self.functions_tsv.as_str())];
        if self.spec.shape == Shape::Signature {
            out.extend([("egraph.datatype", self.datatype.as_str()), ("egraph.rules", self.rules.as_str()), ("egraph.rows", self.egraph_rows.as_str()), ("egraph.sorts", self.sorts.as_str())]);
        }
        out
    }

    /// Write the exact table into `dir`.
    pub fn write(&self, dir: &std::path::Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        for (name, text) in self.artefacts() {
            std::fs::write(dir.join(name), text).map_err(|e| format!("write {}: {e}", dir.join(name).display()))?;
        }
        Ok(())
    }
}

impl View {
    /// THE FINGERPRINT of the written text: FNV-1a (64 bits) over every artefact's file name, a NUL, its text and a
    /// NUL, in [`View::artefacts`] order. It pins a view's regression without committing the tables
    /// (`docs/PLAN_futhark_kingdom.md` 3g): a change to any written byte changes it.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for (name, text) in self.artefacts() {
            for b in name.bytes().chain([0]).chain(text.bytes()).chain([0]) {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }
}

/// Every committed specification, by the directory [`write_committed`] writes its tables into: the signature
/// problems (`sig/<problem>`), the two test views of the pair constructs (`sig/argmin`, `sig/kahan`, sniffed from
/// `kingdoms/futhark/sig/kahan/kahan.fut`) and A7's matrix view (`a7`).
pub const COMMITTED: [(&str, &str); 8] = [
    ("sig/argmin", include_str!("../kingdoms/futhark/sig/argmin/view.tsv")),
    ("sig/histogram", include_str!("../kingdoms/futhark/sig/histogram/view.tsv")),
    ("sig/kahan", include_str!("../kingdoms/futhark/sig/kahan/view.tsv")),
    ("sig/kmeans", include_str!("../kingdoms/futhark/sig/kmeans/view.tsv")),
    ("sig/pathfinder", include_str!("../kingdoms/futhark/sig/pathfinder/view.tsv")),
    ("sig/reduce", include_str!("../kingdoms/futhark/sig/reduce/view.tsv")),
    ("sig/scan", include_str!("../kingdoms/futhark/sig/scan/view.tsv")),
    ("a7", include_str!("../kingdoms/futhark/view.tsv")),
];

/// THE ONE VIEW IMPLEMENTATION writes every committed view's exact table under `dir` (`dir/sig/<problem>/`,
/// `dir/a7/`): what `kingdoms/futhark/gen_tables.py --views DIR` reads to run its witnesses, so the witnesses check
/// the tables the engine computes. `examples/futhark_views.rs` is the command.
pub fn write_committed(dir: &std::path::Path) -> Result<Vec<(&'static str, &'static View)>, String> {
    let mut out = Vec::new();
    for (sub, spec) in COMMITTED {
        let v = cached(spec)?;
        v.write(&dir.join(sub))?;
        out.push((sub, v));
    }
    Ok(out)
}

/// The view of a committed specification over the committed master, computed once a process: every caller of one
/// specification shares one view, so its kingdom, signature and e-graph cannot disagree.
pub fn cached(spec_text: &'static str) -> Result<&'static View, String> {
    static VIEWS: OnceLock<Mutex<BTreeMap<&'static str, &'static View>>> = OnceLock::new();
    let mut map = VIEWS.get_or_init(|| Mutex::new(BTreeMap::new())).lock().map_err(|_| "our software has a bug: the view cache is poisoned".to_string())?;
    if let Some(v) = map.get(spec_text) {
        return Ok(v);
    }
    let v: &'static View = Box::leak(Box::new(View::load(&Master::committed()?, spec_text)?));
    map.insert(spec_text, v);
    Ok(v)
}

#[cfg(test)]
mod tests;
