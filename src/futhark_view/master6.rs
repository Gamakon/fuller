//! THE MASTER 6 VIEW (`docs/PLAN_futhark_kingdom.md` section 7, task F1b): a Futhark kingdom over master 6
//! (`kingdoms/futhark/master/6/`, the format in `kingdoms/futhark/master/README.md`), where a type is a DATA TYPE and
//! nothing else, an array's length and an index's bound are PROPERTIES, and the variables a node may use are its
//! ENVIRONMENT.
//!
//! This computes the same three tables as `kingdoms/futhark/view.py` (`View6`), byte for byte (the test below compares
//! them with the committed `sig/<problem>/m6/` files, which `gen_tables.py --check` compares with `view.py`'s):
//!
//! 1. the CANDIDATE data types: `datatypes.tsv`'s templates, in order, each over the view's elements (`E`, or the
//!    ordered pairs and triples of elements), a function type only where its members are candidates;
//! 2. the CANDIDATE rows: each admitted row (version at most the pin, its family admitted, the `var` row in every
//!    view, one of its elements in the view) once for each of its elements when its templates read `{E}`, then over
//!    its `over` quantifiers in order; an instance whose output, an argument or a binding is not a candidate data type
//!    does not exist; `depth` is replaced by the view's depth (the work-bound requirement is dropped in a view with
//!    none), and a rule, requirement or binding property that names a property its data type does not carry is dropped;
//! 3. kept when REACHABLE both ways, with `var@{T}` for every data type that has a variable (a root variable or a
//!    binding of a kept row, iterated until they settle), and for every data type the root or a kept row demands that
//!    no kept row of no arguments gives (its constant when no variable is in scope);
//! 4. written: `datatypes.tsv`, `symbols.tsv`, `environment.tsv`.
//!
//! [`View6::kingdom`] loads the written text through [`Kingdom6::from_tables`], so the engine's kingdom is the table
//! the witnesses checked. Master 5 ([`super::View`]) is untouched and stays loadable until task D1.

use std::collections::{BTreeMap, BTreeSet};

use crate::futhark_table::Kingdom6;

const VERSION6: &str = include_str!("../../kingdoms/futhark/master/6/VERSION");
const DATATYPES6: &str = include_str!("../../kingdoms/futhark/master/6/datatypes.tsv");
const PROPERTIES6: &str = include_str!("../../kingdoms/futhark/master/6/properties.tsv");
const ROWS6: &str = include_str!("../../kingdoms/futhark/master/6/rows.tsv");
/// The master's adapters, which every master 6 kingdom loads.
pub const ADAPTERS6: &str = include_str!("../../kingdoms/futhark/master/6/adapters.tsv");

const ROW6_COLUMNS: [&str; 15] = ["ref", "since", "family", "kind", "elems", "over", "name", "out", "args", "rules", "requires", "binds", "trip", "spelling", "note"];
const DATATYPE6_COLUMNS: [&str; 8] = ["name", "kind", "members", "over", "tag", "constant", "constant_props", "note"];
const PROPERTY6_COLUMNS: [&str; 4] = ["data_type", "name", "kind", "meaning"];
/// The written view's columns.
pub const VIEW6_DATATYPE_COLUMNS: [&str; 8] = ["id", "name", "kind", "members", "tag", "constant", "constant_props", "properties"];
pub const VIEW6_SYMBOL_COLUMNS: [&str; 11] = ["id", "ref", "name", "kind", "out", "args", "rules", "requires", "binds", "trip", "spelling"];
pub const VIEW6_ENV_COLUMNS: [&str; 5] = ["kind", "name", "data_type", "properties", "requires"];

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

/// A tab-separated file of master 6 with its header, as rows of fields (blank lines skipped).
fn tsv6<'a>(name: &str, text: &'a str, columns: &[&str]) -> Result<Vec<Vec<&'a str>>, String> {
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    if head != columns {
        return Err(bug(format!("master/6/{name}: the header is {head:?}, expected {columns:?}")));
    }
    let mut out = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != columns.len() {
            return Err(bug(format!("master/6/{name}:{}: {} columns, expected {}", i + 2, f.len(), columns.len())));
        }
        out.push(f);
    }
    Ok(out)
}

/// A field's `sep`-separated items, trimmed; an empty field has none (`view.py`'s `items`).
fn items(field: &str, sep: char) -> Vec<String> {
    field.split(sep).map(str::trim).filter(|x| !x.is_empty()).map(str::to_string).collect()
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// One data type template of `master/6/datatypes.tsv`.
#[derive(Debug, Clone)]
struct DataTypeTemplate {
    name: String,
    kind: String,
    members: String,
    over: String,
    tag: String,
    constant: String,
    constant_props: String,
}

/// One row template of `master/6/rows.tsv`, its fields by column name.
#[derive(Debug, Clone)]
struct RowTemplate {
    fields: BTreeMap<&'static str, String>,
    since: u32,
}

impl RowTemplate {
    fn get(&self, column: &str) -> &str {
        self.fields.get(column).map_or("", String::as_str)
    }
}

/// The template columns a row's `{E}` and quantifier variables are substituted in.
const ROW6_TEMPLATED: [&str; 8] = ["name", "out", "args", "rules", "requires", "binds", "spelling", "note"];

/// Master 6: its version, data type templates, properties (by data type template, in order) and row templates.
#[derive(Debug, Clone)]
pub struct Master6 {
    pub version: u32,
    datatypes: Vec<DataTypeTemplate>,
    properties: BTreeMap<String, Vec<(String, String)>>,
    rows: Vec<RowTemplate>,
}

impl Master6 {
    /// The committed master 6, as compiled in.
    pub fn committed() -> Result<Master6, String> {
        Master6::parse(VERSION6, DATATYPES6, PROPERTIES6, ROWS6)
    }

    /// Master 6 from the text of its files.
    pub fn parse(version: &str, datatypes: &str, properties: &str, rows: &str) -> Result<Master6, String> {
        let version = version.trim().parse::<u32>().map_err(|e| bug(format!("master/6/VERSION: {e}")))?;
        let datatypes = tsv6("datatypes.tsv", datatypes, &DATATYPE6_COLUMNS)?
            .into_iter()
            .map(|f| DataTypeTemplate { name: f[0].into(), kind: f[1].into(), members: f[2].into(), over: f[3].into(), tag: f[4].into(), constant: f[5].into(), constant_props: f[6].into() })
            .collect();
        let mut props: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        for f in tsv6("properties.tsv", properties, &PROPERTY6_COLUMNS)? {
            props.entry(f[0].to_string()).or_default().push((f[1].to_string(), f[2].to_string()));
        }
        let rows = tsv6("rows.tsv", rows, &ROW6_COLUMNS)?
            .into_iter()
            .map(|f| {
                let since = f[1].parse::<u32>().map_err(|e| bug(format!("master/6/rows.tsv: the row {} has the version {:?}: {e}", f[0], f[1])))?;
                Ok(RowTemplate { fields: ROW6_COLUMNS.iter().zip(&f).map(|(c, v)| (*c, v.to_string())).collect(), since })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Master6 { version, datatypes, properties: props, rows })
    }
}

// ---- the specification ------------------------------------------------------------------------------------------

/// A root environment line of a master 6 specification: `input`, `zeros` or `size`, its name, element and dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvLine {
    pub kind: String,
    pub name: String,
    pub elem: String,
    pub dims: Vec<String>,
}

/// A master 6 view specification (written by `sniff.py --format 6`; `view.py`'s `read_spec6`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewSpec6 {
    pub master: u32,
    pub elements: Vec<String>,
    pub families: Vec<String>,
    /// The root environment in order: inputs, zero arrays, sizes.
    pub env: Vec<EnvLine>,
    /// The output's name, element and dimensions.
    pub output: (String, String, Vec<String>),
    /// The literal of `env.trips < depth`; none in a view with no work bound (A7).
    pub depth: Option<u32>,
    /// The data types a `let` may bind.
    pub lets: Vec<String>,
    /// Each element's literal pool.
    pub literals: BTreeMap<String, Vec<String>>,
}

impl ViewSpec6 {
    pub fn parse(text: &str) -> Result<ViewSpec6, String> {
        let list = |x: &str| x.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect::<Vec<_>>();
        let (mut format, mut master, mut depth, mut output) = (None, None, None, None);
        let (mut elements, mut families, mut lets, mut env) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut literals = BTreeMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            let [kind, name, elem, dims] = f.as_slice() else {
                return Err(bug(format!("the master 6 specification line {line:?} has {} fields, wants 4", f.len())));
            };
            let num = |x: &str| x.parse::<u32>().map_err(|e| bug(format!("the master 6 specification line {line:?}: {e}")));
            match *kind {
                "format" => format = Some(num(name)?),
                "master" => master = Some(num(name)?),
                "depth" => depth = Some(num(name)?),
                "element" => elements = list(name),
                "families" => families = list(name),
                "lets" => lets = list(name),
                "input" | "zeros" | "size" => env.push(EnvLine { kind: kind.to_string(), name: name.to_string(), elem: elem.to_string(), dims: list(dims) }),
                "output" => output = Some((name.to_string(), elem.to_string(), list(dims))),
                "literals" => {
                    let key = if !elem.is_empty() {
                        elem.to_string()
                    } else if elements.len() == 1 {
                        elements[0].clone()
                    } else {
                        String::new()
                    };
                    literals.insert(key, list(name));
                }
                other => return Err(bug(format!("a master 6 specification line of kind {other:?}"))),
            }
        }
        match (format, master, output) {
            (Some(6), Some(master), Some(output)) if !elements.is_empty() => Ok(ViewSpec6 { master, elements, families, env, output, depth, lets, literals }),
            _ => Err(bug("a master 6 specification needs its format 6, master, element and output lines".to_string())),
        }
    }
}

/// The data type of an array of `rank` dimensions of `elem` (a scalar at rank 0).
fn array_type(elem: &str, rank: usize) -> String {
    format!("{}{elem}", "[]".repeat(rank))
}

/// The properties a value of rank 0, 1 or 2 carries, in order.
fn rank_properties(rank: usize) -> Option<&'static [&'static str]> {
    match rank {
        0 => Some(&[]),
        1 => Some(&["len"]),
        2 => Some(&["rows", "cols"]),
        _ => None,
    }
}

/// A root variable: (name, data type, [(property, value)]).
type RootVar = (String, String, Vec<(String, String)>);

/// A root environment line as (name, data type, [(property, value)]).
fn env_variable(e: &EnvLine) -> Result<RootVar, String> {
    if e.kind == "size" {
        return Ok((e.name.clone(), "size".to_string(), vec![("value".to_string(), e.name.clone())]));
    }
    let props = rank_properties(e.dims.len()).ok_or_else(|| bug(format!("the root variable {} has {} dimensions; master 6 has arrays of rank 1 and 2", e.name, e.dims.len())))?;
    Ok((e.name.clone(), array_type(&e.elem, e.dims.len()), props.iter().map(|p| p.to_string()).zip(e.dims.iter().cloned()).collect()))
}

// ---- templates ------------------------------------------------------------------------------------------------

type Env = BTreeMap<String, String>;

/// Replace every `{V}` / `{V.x}` quantifier variable (`[A-Z][A-Z0-9]*`, optionally `.` and `[a-z0-9]+`) by its value in
/// `env`; one `env` does not bind is a bug. Every other `{...}` (`{0}`, `{1.v0}`, `{v}`) is left as it is.
fn subst(text: &str, env: &Env) -> Result<String, String> {
    let b = text.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    let mut last = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(end) = quantifier_var_end(b, i + 1) {
                let var = &text[i + 1..end];
                let value = env.get(var).ok_or_else(|| bug(format!("the template {text:?} reads {{{var}}}, which nothing binds")))?;
                out.push_str(&text[last..i]);
                out.push_str(value);
                i = end + 1;
                last = i;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&text[last..]);
    Ok(out)
}

/// The index of the `}` closing a quantifier variable that starts at `start`, if one does.
fn quantifier_var_end(b: &[u8], start: usize) -> Option<usize> {
    let mut j = start;
    if !b.get(j)?.is_ascii_uppercase() {
        return None;
    }
    j += 1;
    while b.get(j).is_some_and(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        j += 1;
    }
    if b.get(j) == Some(&b'}') {
        return Some(j);
    }
    if b.get(j) != Some(&b'.') {
        return None;
    }
    let k = j + 1;
    let mut e = k;
    while b.get(e).is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        e += 1;
    }
    (e > k && b.get(e) == Some(&b'}')).then_some(e)
}

/// Every `in<k>.<property>` term of an expression, at word boundaries (`view.py`'s `PROPERTY_TERM_RE`).
fn property_terms(expr: &str) -> Vec<(usize, String)> {
    let cs: Vec<char> = expr.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < cs.len() {
        if cs[i] == 'i' && cs[i + 1] == 'n' && (i == 0 || !is_word(cs[i - 1])) {
            let mut j = i + 2;
            while j < cs.len() && cs[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 2 && j < cs.len() && cs[j] == '.' && cs.get(j + 1).is_some_and(|c| c.is_ascii_lowercase() || *c == '_') {
                let mut e = j + 2;
                while e < cs.len() && (cs[e].is_ascii_lowercase() || cs[e].is_ascii_digit() || cs[e] == '_') {
                    e += 1;
                }
                if e == cs.len() || !is_word(cs[e]) {
                    let k: String = cs[i + 2..j].iter().collect();
                    out.push((k.parse().unwrap_or(usize::MAX), cs[j + 1..e].iter().collect()));
                    i = e;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// `expr` with every whole word `depth` replaced by `value`.
fn replace_depth(expr: &str, value: u32) -> String {
    let cs: Vec<char> = expr.chars().collect();
    let word: Vec<char> = "depth".chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        let at = cs[i..].starts_with(&word) && (i == 0 || !is_word(cs[i - 1])) && cs.get(i + word.len()).is_none_or(|c| !is_word(*c));
        if at {
            out.push_str(&value.to_string());
            i += word.len();
        } else {
            out.push(cs[i]);
            i += 1;
        }
    }
    out
}

/// A requirement's text in its four parts: (subject, relation, expression, adapter).
pub type RequirementParts = (String, String, String, String);

/// A requirement's text as (subject, relation, expression, adapter): `<subject> <relation> <expr> -> <adapter>`, the
/// subject `in<k>.<p>`, `env.trips` or `out.<p>`, the relation `=`, `<=` or `<`.
pub fn parse_requirement(q: &str) -> Result<RequirementParts, String> {
    let no = || bug(format!("the requirement {q:?} does not parse"));
    let (subject, rest) = q.split_once(' ').ok_or_else(no)?;
    let ident = |s: &str| s.chars().next().is_some_and(|c| c.is_ascii_lowercase() || c == '_') && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    let subject_ok = subject == "env.trips"
        || subject.strip_prefix("out.").is_some_and(ident)
        || subject.strip_prefix("in").and_then(|s| s.split_once('.')).is_some_and(|(k, p)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()) && ident(p));
    let (relation, rest) = rest.split_once(' ').ok_or_else(no)?;
    let (expr, adapter) = rest.rsplit_once(" -> ").ok_or_else(no)?;
    let adapter_ok = adapter.chars().next().is_some_and(|c| c.is_ascii_lowercase()) && adapter.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.');
    if !subject_ok || !["=", "<=", "<"].contains(&relation) || expr.is_empty() || !adapter_ok {
        return Err(no());
    }
    Ok((subject.to_string(), relation.to_string(), expr.to_string(), adapter.to_string()))
}

/// A binding's text as (argument, data type, the properties text): `<argument>:<data type>[ <properties>]`.
fn parse_binding(b: &str) -> Result<(usize, String, String), String> {
    let no = || bug(format!("the binding {b:?} does not parse"));
    let (arg, rest) = b.split_once(':').ok_or_else(no)?;
    if arg.is_empty() || !arg.bytes().all(|c| c.is_ascii_digit()) {
        return Err(no());
    }
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let (dt, props) = rest.split_at(end);
    if dt.is_empty() {
        return Err(no());
    }
    let props = match props.strip_prefix(' ') {
        Some(p) => p,
        None if props.is_empty() => "",
        None => return Err(no()),
    };
    Ok((arg.parse().map_err(|_| no())?, dt.to_string(), props.to_string()))
}

// ---- the view -------------------------------------------------------------------------------------------------

/// A candidate data type of a view.
#[derive(Debug, Clone)]
struct Candidate {
    name: String,
    kind: String,
    members: String,
    member_list: Vec<String>,
    tag: String,
    constant: String,
    constant_props: String,
    properties: Vec<(String, String)>,
}

/// A row of a view: a row template instantiated.
#[derive(Debug, Clone)]
struct Instance {
    reference: String,
    name: String,
    kind: String,
    out: String,
    args: Vec<String>,
    rules: Vec<String>,
    requires: Vec<String>,
    binds: Vec<(usize, String, Vec<String>)>,
    trip: String,
    spelling: String,
}

/// A master 6 view: its specification and the three tables it writes.
#[derive(Debug, Clone)]
pub struct View6 {
    pub spec: ViewSpec6,
    pub datatypes_tsv: String,
    pub symbols_tsv: String,
    pub environment_tsv: String,
}

struct Builder<'a> {
    spec: &'a ViewSpec6,
    master: &'a Master6,
    multi: bool,
    candidates: Vec<Candidate>,
    by_name: BTreeMap<String, usize>,
    root: String,
}

impl Builder<'_> {
    fn candidate(&self, dt: &str) -> Option<&Candidate> {
        self.by_name.get(dt).map(|i| &self.candidates[*i])
    }

    fn carries(&self, dt: &str, prop: &str) -> bool {
        self.candidate(dt).is_some_and(|t| t.properties.iter().any(|(p, _)| p == prop))
    }

    fn tagged(&self, env: &Env, var: &str, dt: &str) -> Result<Env, String> {
        let t = self.candidate(dt).ok_or_else(|| bug(format!("the quantifier {var} names the data type {dt}, which the view does not have")))?;
        let mut b = env.clone();
        b.insert(var.to_string(), dt.to_string());
        b.insert(format!("{var}.tag"), t.tag.clone());
        Ok(b)
    }

    fn admitted(&self, r: &RowTemplate) -> bool {
        r.since <= self.spec.master
            && (self.spec.families.is_empty() || self.spec.families.iter().any(|f| f == r.get("family")) || r.get("kind") == "var")
            && (r.get("elems") == "*" || r.get("elems").split(',').any(|e| self.spec.elements.iter().any(|v| v == e)))
    }

    fn quantifier(&self, q: &str, env: &Env, row_elems: &[String], var_types: &BTreeSet<String>) -> Result<Vec<Env>, String> {
        let (var, kind) = q.split_once(':').unwrap_or((q, ""));
        let mut out = Vec::new();
        match kind {
            "states" => {
                let elems: Vec<String> = match env.get("E") {
                    Some(e) => vec![e.clone()],
                    None => row_elems.to_vec(),
                };
                for dt in elems.iter().cloned().chain(elems.iter().map(|e| format!("[]{e}"))) {
                    if self.by_name.contains_key(&dt) {
                        out.push(self.tagged(env, var, &dt)?);
                    }
                }
            }
            "tuples" | "tuples3" => {
                let n = if kind == "tuples" { 2 } else { 3 };
                let els = &self.spec.elements;
                let mut combos: Vec<Vec<&String>> = vec![vec![]];
                for _ in 0..n {
                    combos = combos.into_iter().flat_map(|c| els.iter().map(move |e| c.iter().copied().chain(std::iter::once(e)).collect())).collect();
                }
                for parts in combos {
                    let dt = format!("({})", parts.iter().map(|p| p.as_str()).collect::<Vec<_>>().join(", "));
                    if self.by_name.contains_key(&dt) && parts.iter().all(|p| row_elems.contains(p)) {
                        let mut b = self.tagged(env, var, &dt)?;
                        for (x, p) in ["a", "b", "c"].iter().zip(&parts) {
                            b.insert(format!("{var}.{x}"), p.to_string());
                        }
                        b.insert(format!("{var}.zero"), format!("({})", parts.iter().map(|p| format!("0{p}")).collect::<Vec<_>>().join(", ")));
                        out.push(b);
                    }
                }
            }
            "lets" => {
                for dt in &self.spec.lets {
                    out.push(self.tagged(env, var, dt)?);
                }
            }
            "all" => {
                for t in self.candidates.iter().filter(|t| t.name != "size") {
                    out.push(self.tagged(env, var, &t.name)?);
                }
            }
            "vars" => {
                for t in self.candidates.iter().filter(|t| var_types.contains(&t.name)) {
                    out.push(self.tagged(env, var, &t.name)?);
                }
            }
            _ => return Err(bug(format!("the quantifier {q:?} is not one of master 6's"))),
        }
        Ok(out)
    }

    /// One instance of row `r` under `env`, or none when one of its data types is not a candidate.
    fn instance(&self, r: &RowTemplate, env: &Env) -> Result<Option<Instance>, String> {
        let mut t: BTreeMap<&str, String> = BTreeMap::new();
        for c in ROW6_TEMPLATED {
            t.insert(c, subst(r.get(c), env)?);
        }
        let args = items(&t["args"], ';');
        if !self.by_name.contains_key(&t["out"]) || args.iter().any(|a| !self.by_name.contains_key(a)) {
            return Ok(None);
        }
        let held = |expr: &str| property_terms(expr).iter().all(|(k, p)| *k < args.len() && self.carries(&args[*k], p));
        let mut rules = Vec::new();
        for rule in items(&t["rules"], ';') {
            let (prop, expr) = rule.split_once('=').unwrap_or((rule.as_str(), ""));
            if self.carries(&t["out"], prop) && held(expr) {
                rules.push(rule.clone());
            }
        }
        let mut requires = Vec::new();
        for q in items(&t["requires"], ';') {
            let (subject, rel, expr, adapter) = parse_requirement(&q).map_err(|e| format!("{e} (row {})", r.get("ref")))?;
            if subject == "env.trips" {
                if let Some(depth) = self.spec.depth {
                    requires.push(format!("{subject} {rel} {} -> {adapter}", replace_depth(&expr, depth)));
                }
                continue;
            }
            let (k, p) = subject.strip_prefix("in").and_then(|s| s.split_once('.')).ok_or_else(|| bug(format!("the row {} has a requirement on {subject}, which is not an input", r.get("ref"))))?;
            let k: usize = k.parse().map_err(|_| bug(format!("the requirement {q:?} of {} names no input", r.get("ref"))))?;
            if k < args.len() && self.carries(&args[k], p) && held(&expr) {
                requires.push(q.clone());
            }
        }
        let mut binds = Vec::new();
        for b in items(&t["binds"], ';') {
            let (arg, dt, props) = parse_binding(&b).map_err(|e| format!("{e} (row {})", r.get("ref")))?;
            if !self.by_name.contains_key(&dt) {
                return Ok(None);
            }
            let kept = items(&props, '&')
                .into_iter()
                .filter(|pe| {
                    let (p, e) = pe.split_once('=').unwrap_or((pe.as_str(), ""));
                    self.carries(&dt, p) && held(e)
                })
                .collect();
            binds.push((arg, dt, kept));
        }
        Ok(Some(Instance {
            reference: r.get("ref").to_string(),
            name: t["name"].clone(),
            kind: r.get("kind").to_string(),
            out: t["out"].clone(),
            args,
            rules,
            requires,
            binds,
            trip: r.get("trip").to_string(),
            spelling: t["spelling"].clone(),
        }))
    }

    fn rows(&self, var_types: &BTreeSet<String>) -> Result<Vec<Instance>, String> {
        let mut out = Vec::new();
        for r in self.master.rows.iter().filter(|r| self.admitted(r)) {
            let row_elems: Vec<String> = if r.get("elems") == "*" {
                self.spec.elements.clone()
            } else {
                self.spec.elements.iter().filter(|e| r.get("elems").split(',').any(|x| x == e.as_str())).cloned().collect()
            };
            let mut envs: Vec<Env> = if ROW6_TEMPLATED.iter().any(|c| r.get(c).contains("{E}") || r.get(c).contains("{E.tag}")) {
                row_elems.iter().map(|e| Env::from([("E".to_string(), e.clone()), ("E.tag".to_string(), if self.multi { format!(".{e}") } else { String::new() })])).collect()
            } else {
                vec![Env::new()]
            };
            for q in items(r.get("over"), ';') {
                let mut next = Vec::new();
                for env in &envs {
                    next.extend(self.quantifier(&q, env, &row_elems, var_types)?);
                }
                envs = next;
            }
            for env in &envs {
                if let Some(inst) = self.instance(r, env)? {
                    out.push(inst);
                }
            }
        }
        Ok(out)
    }

    /// The rows reachable both ways: forward from the leaves, backward from the root.
    fn reachable(&self, rows: Vec<Instance>) -> Result<Vec<Instance>, String> {
        let mut built: BTreeSet<String> = rows.iter().filter(|r| r.kind != "function").map(|r| r.out.clone()).collect();
        loop {
            let before = built.len();
            for r in &rows {
                if !built.contains(&r.out) && r.args.iter().all(|a| built.contains(a)) {
                    built.insert(r.out.clone());
                }
            }
            if built.len() == before {
                break;
            }
        }
        if !built.contains(&self.root) {
            return Err(bug(format!("the root {} cannot be built from the view's leaves", self.root)));
        }
        let usable: Vec<bool> = rows.iter().map(|r| r.args.iter().all(|a| built.contains(a))).collect();
        let mut keep: BTreeSet<String> = BTreeSet::from([self.root.clone()]);
        loop {
            let before = keep.len();
            for (r, u) in rows.iter().zip(&usable) {
                if *u && keep.contains(&r.out) {
                    keep.extend(r.args.iter().cloned());
                }
            }
            if keep.len() == before {
                break;
            }
        }
        Ok(rows.into_iter().zip(usable).filter(|(r, u)| *u && keep.contains(&r.out)).map(|(r, _)| r).collect())
    }

    /// The data types the root and `rows` demand that no row of no arguments in `rows` gives; each has a constant.
    fn leafless(&self, rows: &[Instance]) -> Result<BTreeSet<String>, String> {
        let leaves: BTreeSet<&str> = rows.iter().filter(|r| r.args.is_empty()).map(|r| r.out.as_str()).collect();
        let demanded: BTreeSet<&str> = std::iter::once(self.root.as_str()).chain(rows.iter().flat_map(|r| r.args.iter().map(String::as_str))).collect();
        let out: BTreeSet<String> = demanded.into_iter().filter(|t| !leaves.contains(t)).map(str::to_string).collect();
        if let Some(t) = out.iter().find(|t| self.candidate(t).is_none_or(|c| c.constant.is_empty())) {
            return Err(bug(format!("the data type {t} is demanded, has no leaf and no constant")));
        }
        Ok(out)
    }
}

impl View6 {
    /// The view of a master 6 specification over a master 6.
    pub fn load(master: &Master6, spec_text: &str) -> Result<View6, String> {
        let spec = ViewSpec6::parse(spec_text)?;
        let elements = &spec.elements;
        // 1. The candidate data types.
        let mut candidates: Vec<Candidate> = Vec::new();
        for t in &master.datatypes {
            let envs: Vec<Env> = match t.over.as_str() {
                "E" => elements.iter().map(|e| Env::from([("E".to_string(), e.clone())])).collect(),
                "A,B:elements" => elements.iter().flat_map(|a| elements.iter().map(move |b| Env::from([("A".to_string(), a.clone()), ("B".to_string(), b.clone())]))).collect(),
                "A,B,C:elements" => elements
                    .iter()
                    .flat_map(|a| elements.iter().flat_map(move |b| elements.iter().map(move |c| Env::from([("A".to_string(), a.clone()), ("B".to_string(), b.clone()), ("C".to_string(), c.clone())]))))
                    .collect(),
                "" => vec![Env::new()],
                other => return Err(bug(format!("the data type {} is over {other:?}", t.name))),
            };
            for env in envs {
                let name = subst(&t.name, &env)?;
                let members = subst(&t.members, &env)?;
                let member_list = if t.kind == "function" {
                    let (params, result) = members.split_once(" -> ").ok_or_else(|| bug(format!("the function type {name} has members {members:?}, not `params -> result`")))?;
                    items(params, ';').into_iter().chain(std::iter::once(result.to_string())).collect()
                } else {
                    items(&members, ';')
                };
                if member_list.iter().any(|m| !candidates.iter().any(|c| &c.name == m)) {
                    continue;
                }
                candidates.push(Candidate {
                    name,
                    kind: t.kind.clone(),
                    members,
                    member_list,
                    tag: subst(&t.tag, &env)?,
                    constant: subst(&t.constant, &env)?,
                    constant_props: t.constant_props.clone(),
                    properties: master.properties.get(&t.name).cloned().unwrap_or_default(),
                });
            }
        }
        let mut by_name = BTreeMap::new();
        for (i, c) in candidates.iter().enumerate() {
            by_name.insert(c.name.clone(), i);
        }
        let root_vars = spec.env.iter().map(env_variable).collect::<Result<Vec<_>, _>>()?;
        let root = array_type(&spec.output.1, spec.output.2.len());
        for dt in root_vars.iter().map(|(_, dt, _)| dt).chain(std::iter::once(&root)) {
            if !by_name.contains_key(dt) {
                return Err(bug(format!("the specification names the data type {dt}, which the master does not have for {elements:?}")));
            }
        }
        let b = Builder { spec: &spec, master, multi: elements.len() > 1, candidates, by_name, root };
        // 2 and 3: the rows, reachable both ways, the data types with a variable settled.
        let root_types: BTreeSet<String> = root_vars.iter().map(|(_, dt, _)| dt.clone()).collect();
        let binders = |rows: &[Instance]| -> BTreeSet<String> { rows.iter().flat_map(|r| r.binds.iter().map(|(_, dt, _)| dt.clone())).collect() };
        let mut var_types: BTreeSet<String> = root_types.union(&binders(&b.rows(&BTreeSet::new())?)).cloned().collect();
        let mut settled_rows = None;
        for _ in 0..b.candidates.len() + 2 {
            let rows = b.reachable(b.rows(&var_types)?)?;
            let have: BTreeSet<String> = root_types.union(&binders(&rows)).cloned().collect();
            let settled: BTreeSet<String> = var_types.intersection(&have).cloned().collect();
            if settled == var_types {
                settled_rows = Some(rows);
                break;
            }
            var_types = settled;
        }
        let mut rows = settled_rows.ok_or_else(|| bug("the data types with a variable do not settle".to_string()))?;
        let leafless = b.leafless(&rows)?;
        if !leafless.is_empty() {
            var_types.extend(leafless);
            rows = b.reachable(b.rows(&var_types)?)?;
        }
        let mut used: BTreeSet<String> = BTreeSet::from([b.root.clone()]);
        for r in &rows {
            used.insert(r.out.clone());
            used.extend(r.args.iter().cloned());
            used.extend(r.binds.iter().map(|(_, dt, _)| dt.clone()));
        }
        loop {
            let before = used.len();
            for t in &b.candidates {
                if used.contains(&t.name) {
                    used.extend(t.member_list.iter().cloned());
                }
            }
            if used.len() == before {
                break;
            }
        }
        let types: Vec<&Candidate> = b.candidates.iter().filter(|t| used.contains(&t.name)).collect();
        let var_rows: BTreeSet<&str> = rows.iter().filter(|r| r.kind == "var").map(|r| r.out.as_str()).collect();
        // 4. The written tables.
        let mut dt = VIEW6_DATATYPE_COLUMNS.join("\t") + "\n";
        for (i, t) in types.iter().enumerate() {
            let props = t.properties.iter().map(|(p, k)| format!("{p}:{k}")).collect::<Vec<_>>().join(";");
            dt.push_str(&[i.to_string(), t.name.clone(), t.kind.clone(), t.members.clone(), t.tag.clone(), t.constant.clone(), t.constant_props.clone(), props].join("\t"));
            dt.push('\n');
        }
        let mut sy = VIEW6_SYMBOL_COLUMNS.join("\t") + "\n";
        for (i, r) in rows.iter().enumerate() {
            let binds = r.binds.iter().map(|(a, d, ps)| format!("{a}:{d}{}", if ps.is_empty() { String::new() } else { format!(" {}", ps.join(" & ")) })).collect::<Vec<_>>().join("; ");
            sy.push_str(&[i.to_string(), r.reference.clone(), r.name.clone(), r.kind.clone(), r.out.clone(), r.args.join(";"), r.rules.join("; "), r.requires.join("; "), binds, r.trip.clone(), r.spelling.clone()].join("\t"));
            sy.push('\n');
        }
        let (out_name, _, out_dims) = &spec.output;
        let props = rank_properties(out_dims.len()).ok_or_else(|| bug(format!("the output has {} dimensions", out_dims.len())))?;
        let root_props = props.iter().zip(out_dims).map(|(p, d)| format!("{p}={d}")).collect::<Vec<_>>().join(" & ");
        let root_req = props.iter().zip(out_dims).map(|(p, d)| format!("out.{p} = {d} -> fit.root")).collect::<Vec<_>>().join("; ");
        let mut en = VIEW6_ENV_COLUMNS.join("\t") + "\n";
        en.push_str(&["root", out_name, &b.root, &root_props, &root_req].join("\t"));
        en.push('\n');
        for (line, (vname, vdt, ps)) in spec.env.iter().zip(&root_vars) {
            if var_rows.contains(vdt.as_str()) {
                let ps = ps.iter().map(|(p, v)| format!("{p}={v}")).collect::<Vec<_>>().join(" & ");
                en.push_str(&[line.kind.as_str(), vname, vdt, &ps, ""].join("\t"));
                en.push('\n');
            }
        }
        Ok(View6 { spec: spec.clone(), datatypes_tsv: dt, symbols_tsv: sy, environment_tsv: en })
    }

    /// The view's three tables, by file name.
    pub fn artefacts(&self) -> [(&'static str, &str); 3] {
        [("datatypes.tsv", self.datatypes_tsv.as_str()), ("symbols.tsv", self.symbols_tsv.as_str()), ("environment.tsv", self.environment_tsv.as_str())]
    }

    /// The kingdom the view is, loaded from its written text.
    pub fn kingdom(&self, name: &str) -> Result<Kingdom6, String> {
        Kingdom6::from_tables(name, &self.datatypes_tsv, &self.symbols_tsv, &self.environment_tsv, ADAPTERS6, &self.spec.literals)
    }
}

macro_rules! m6 {
    ($p:literal) => {
        (
            $p,
            include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/view.tsv")),
            include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/datatypes.tsv")),
            include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/symbols.tsv")),
            include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/environment.tsv")),
        )
    };
}

/// Every committed master 6 view: (problem, the specification, and the three tables `view.py` wrote), under
/// `kingdoms/futhark/sig/<problem>/m6/`.
pub const COMMITTED6: [(&str, &str, &str, &str, &str); 7] = [
    m6!("a7"),
    m6!("histogram"),
    m6!("kmeans"),
    m6!("kmeans_i32"),
    m6!("pathfinder"),
    m6!("reduce"),
    m6!("scan"),
];

/// The committed view of a problem, computed over the committed master 6.
pub fn committed(problem: &str) -> Result<View6, String> {
    let (_, spec, ..) = COMMITTED6.iter().find(|c| c.0 == problem).ok_or_else(|| bug(format!("{problem} is not a committed master 6 view")))?;
    View6::load(&Master6::committed()?, spec)
}

#[cfg(test)]
mod tests;
