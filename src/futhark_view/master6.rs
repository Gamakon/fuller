//! THE MASTER 6 VIEW (`docs/PLAN_futhark_kingdom.md` section 7): a Futhark kingdom over master 6
//! (`kingdoms/futhark/master/6/`, the format in `kingdoms/futhark/master/README.md`), where a type is a DATA TYPE and
//! nothing else, an array's length and an index's bound are PROPERTIES, and the variables a node may use are its
//! ENVIRONMENT, written in the gene frame's FOUR KINDS OF ROW (design `docs/DESIGN_futhark_types_and_properties.md`
//! 3.3: Argument, Flow, Guard, Entry) and nothing else: no row form of its own.
//!
//! This computes the same table as `kingdoms/futhark/view.py` (`View6`), byte for byte (the test below compares it
//! with the committed `sig/<problem>/m6/tables.tsv`, which `gen_tables.py --check` compares with `view.py`'s):
//!
//! 1. the CANDIDATE data types: `datatypes.tsv`'s templates, in order, each over the view's elements (`E`, or the
//!    ordered pairs and triples of elements), a function type only where its members are candidates;
//! 2. the CANDIDATE rows: each admitted row (version at most the pin and its family admitted; a `var` row and an
//!    ADAPTER in every view) once for each of its elements when its templates read `{E}`, then over its `over`
//!    quantifiers in order; an instance whose output, an argument or a variable's data type is not a candidate does not
//!    exist; `depth` is the view's depth (the work bound dropped in a view with none), and an up flow, a variable's
//!    property or a guard that reads a property its data type does not carry is dropped;
//! 3. kept when REACHABLE both ways, with `var@{T}` for every data type that has a variable, and `zero@{T}`, T's
//!    constant (M0088), for every data type the root or a kept row demands that no kept row of no arguments gives, to
//!    a fixed point;
//! 4. written (`tables.tsv`, read by `KingdomTables::parse_named`): the data types (properties, constants, the entry's
//!    size names at least 1 on `size`), the symbols (role, adapter, spellings, prelude; the master's formula rows last),
//!    their Argument rows, their Flow rows (up flows, variables with their properties and spellings, the trip
//!    attribute), their Guard rows with their repairs inline (the master's repair lists, `repairs.tsv`, with the
//!    guard's operands and the adapter that fits each bound, then the row's work-bound substitutes), and the entry.
//!
//! [`View6::kingdom`] loads the written text, so the engine's kingdom is the table the witnesses checked.

use std::collections::{BTreeMap, BTreeSet};

use crate::futhark_table::Kingdom6;

const VERSION6: &str = include_str!("../../kingdoms/futhark/master/6/VERSION");
const DATATYPES6: &str = include_str!("../../kingdoms/futhark/master/6/datatypes.tsv");
const PROPERTIES6: &str = include_str!("../../kingdoms/futhark/master/6/properties.tsv");
const ROWS6: &str = include_str!("../../kingdoms/futhark/master/6/rows.tsv");
const REPAIRS6: &str = include_str!("../../kingdoms/futhark/master/6/repairs.tsv");
const FORMULAS6: &str = include_str!("../../kingdoms/futhark/master/6/formulas.tsv");
/// The master's preludes, by name (`master/6/preludes/<name>.fut`).
const PRELUDES6: [(&str, &str); 4] = [
    ("clampix", include_str!("../../kingdoms/futhark/master/6/preludes/clampix.fut")),
    ("fit", include_str!("../../kingdoms/futhark/master/6/preludes/fit.fut")),
    ("fitrows", include_str!("../../kingdoms/futhark/master/6/preludes/fitrows.fut")),
    ("matmul", include_str!("../../kingdoms/futhark/master/6/preludes/matmul.fut")),
];

const ROW6_COLUMNS: [&str; 14] = ["ref", "since", "family", "kind", "elems", "over", "name", "out", "args", "flows", "guards", "spelling", "prelude", "note"];
const DATATYPE6_COLUMNS: [&str; 10] = ["name", "kind", "members", "over", "tag", "constant", "constant_props", "spell", "type_spelling", "note"];
const PROPERTY6_COLUMNS: [&str; 4] = ["data_type", "name", "kind", "meaning"];
const REPAIR6_COLUMNS: [&str; 5] = ["list", "site", "when", "guard", "template"];
const FORMULA6_COLUMNS: [&str; 3] = ["operator", "form", "spelling"];

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
    spell: String,
    type_spelling: String,
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

/// One repair of a repair list of `master/6/repairs.tsv`.
#[derive(Debug, Clone)]
struct RepairTemplate {
    list: String,
    site: String,
    when: String,
    guard: String,
    template: String,
}

/// The template columns a row's `{E}` and quantifier variables are substituted in (its spelling apart: it also reads
/// each argument's type spelling, `{TYPE<k>}`).
const ROW6_TEMPLATED: [&str; 7] = ["name", "out", "args", "flows", "guards", "spelling", "note"];

/// The adapter that fits a property of a value, by its data type's shape (the repairs' `{FITS}`, `{FITJ}`).
const FIT6: [(&str, &str, &str); 3] = [("[]", "len", "fit@a.{E}"), ("[][]", "rows", "fitrows@m.{E}"), ("[][]", "cols", "fitcols@m.{E}")];

/// Master 6: its version, data type templates, properties (by data type template, in order), row templates, repair
/// lists, formula rows and preludes.
#[derive(Debug, Clone)]
pub struct Master6 {
    pub version: u32,
    datatypes: Vec<DataTypeTemplate>,
    properties: BTreeMap<String, Vec<(String, String)>>,
    rows: Vec<RowTemplate>,
    repairs: Vec<RepairTemplate>,
    formulas: Vec<(String, String, String)>,
    preludes: BTreeMap<String, String>,
}

impl Master6 {
    /// The committed master 6, as compiled in.
    pub fn committed() -> Result<Master6, String> {
        Master6::parse(VERSION6, DATATYPES6, PROPERTIES6, ROWS6, REPAIRS6, FORMULAS6, &PRELUDES6)
    }

    /// Master 6 from the text of its files.
    pub fn parse(version: &str, datatypes: &str, properties: &str, rows: &str, repairs: &str, formulas: &str, preludes: &[(&str, &str)]) -> Result<Master6, String> {
        let version = version.trim().parse::<u32>().map_err(|e| bug(format!("master/6/VERSION: {e}")))?;
        let datatypes = tsv6("datatypes.tsv", datatypes, &DATATYPE6_COLUMNS)?
            .into_iter()
            .map(|f| DataTypeTemplate { name: f[0].into(), kind: f[1].into(), members: f[2].into(), over: f[3].into(), tag: f[4].into(), constant: f[5].into(), constant_props: f[6].into(), spell: f[7].into(), type_spelling: f[8].into() })
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
        let repairs = tsv6("repairs.tsv", repairs, &REPAIR6_COLUMNS)?.into_iter().map(|f| RepairTemplate { list: f[0].into(), site: f[1].into(), when: f[2].into(), guard: f[3].into(), template: f[4].into() }).collect();
        let formulas = tsv6("formulas.tsv", formulas, &FORMULA6_COLUMNS)?.into_iter().map(|f| (f[0].to_string(), f[1].to_string(), f[2].to_string())).collect();
        let preludes = preludes.iter().map(|(n, t)| (n.to_string(), t.to_string())).collect();
        Ok(Master6 { version, datatypes, properties: props, rows, repairs, formulas, preludes })
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

/// A guard item of a row as (strength, subject, relation, expression, repair list): `must <subject> <relation> <expr>
/// -> <list>`, the subject `in<k>.<p>` or `env.trips`, the relation `=`, `<=` or `<`.
pub type GuardParts = (String, String, String, String, String);

/// A guard item's text in its five parts ([`GuardParts`]).
pub fn parse_guard(q: &str) -> Result<GuardParts, String> {
    let no = || bug(format!("the guard {q:?} does not parse"));
    let (strength, rest) = q.split_once(' ').ok_or_else(no)?;
    let (subject, rest) = rest.split_once(' ').ok_or_else(no)?;
    let ident = |s: &str| s.chars().next().is_some_and(|c| c.is_ascii_lowercase() || c == '_') && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    let subject_ok = subject == "env.trips" || subject.strip_prefix("in").and_then(|s| s.split_once('.')).is_some_and(|(k, p)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()) && ident(p));
    let (relation, rest) = rest.split_once(' ').ok_or_else(no)?;
    let (expr, list) = rest.rsplit_once(" -> ").ok_or_else(no)?;
    let list_ok = list.chars().next().is_some_and(|c| c.is_ascii_lowercase()) && list.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.');
    if !["must", "advise"].contains(&strength) || !subject_ok || !["=", "<=", "<"].contains(&relation) || expr.is_empty() || !list_ok {
        return Err(no());
    }
    Ok((strength.to_string(), subject.to_string(), relation.to_string(), expr.to_string(), list.to_string()))
}

/// One flow item of a row.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FlowItem {
    /// `out.<p> := <formula>`.
    Up(String, String),
    /// `in<k>: env.<a> := <formula>`.
    Env(usize, String, String),
    /// `in<k>: var <T>[ with p := f & ..]`.
    Var(usize, String, Vec<(String, String)>),
}

/// A flow item's text.
fn parse_flow(item: &str) -> Result<FlowItem, String> {
    let no = || bug(format!("the flow {item:?} does not parse"));
    if let Some(rest) = item.strip_prefix("out.") {
        let (p, e) = rest.split_once(" := ").ok_or_else(no)?;
        return Ok(FlowItem::Up(p.to_string(), e.to_string()));
    }
    let (at, rest) = item.split_once(": ").ok_or_else(no)?;
    let k: usize = at.strip_prefix("in").and_then(|d| d.parse().ok()).ok_or_else(no)?;
    if let Some(rest) = rest.strip_prefix("env.") {
        let (a, e) = rest.split_once(" := ").ok_or_else(no)?;
        return Ok(FlowItem::Env(k, a.to_string(), e.to_string()));
    }
    let rest = rest.strip_prefix("var ").ok_or_else(no)?;
    let (dt, props) = rest.split_once(" with ").unwrap_or((rest, ""));
    let props = items(props, '&').into_iter().map(|pe| pe.split_once(" := ").map(|(p, e)| (p.to_string(), e.to_string())).ok_or_else(no)).collect::<Result<Vec<_>, _>>()?;
    Ok(FlowItem::Var(k, dt.to_string(), props))
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
    spell: String,
    type_spelling: String,
}

/// A variable a row's flow introduces: the argument, its data type, its properties.
type VarFlow = (usize, String, Vec<(String, String)>);

/// A row of a view: a row template instantiated.
#[derive(Debug, Clone)]
struct Instance {
    reference: String,
    name: String,
    kind: String,
    out: String,
    args: Vec<String>,
    ups: Vec<(String, String)>,
    envs: Vec<(usize, String, String)>,
    vars: Vec<VarFlow>,
    guards: Vec<GuardParts>,
    spelling: String,
    prelude: String,
}

/// A master 6 view: its specification, the table it writes, and what its e-graph is written from: each data type's
/// (name, kind, members, tag) and each symbol's (name, kind, output, arguments), in the table's order.
#[derive(Debug, Clone)]
pub struct View6 {
    pub spec: ViewSpec6,
    pub tables_tsv: String,
    pub types: Vec<[String; 4]>,
    pub symbols: Vec<(String, String, String, Vec<String>)>,
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

    /// A var row and an ADAPTER are in every view, whatever its pin and families: every view's repairs call the
    /// adapters, and evolution may place them.
    fn admitted(&self, r: &RowTemplate) -> bool {
        let everywhere = r.get("kind") == "var" || r.get("kind") == "adapter";
        (everywhere || r.since <= self.spec.master)
            && (self.spec.families.is_empty() || self.spec.families.iter().any(|f| f == r.get("family")) || everywhere || items(r.get("over"), ';').iter().any(|q| q.split_once(':').is_some_and(|(_, k)| k == "leafless")))
            && (r.get("elems") == "*" || r.get("elems").split(',').any(|e| self.spec.elements.iter().any(|v| v == e)))
    }

    fn quantifier(&self, q: &str, env: &Env, row_elems: &[String], var_types: &BTreeSet<String>, leafless: &BTreeSet<String>) -> Result<Vec<Env>, String> {
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
            "leafless" => {
                for t in self.candidates.iter().filter(|t| leafless.contains(&t.name)) {
                    let mut b = self.tagged(env, var, &t.name)?;
                    b.insert(format!("{var}.constant"), t.constant.clone());
                    out.push(b);
                }
            }
            _ => return Err(bug(format!("the quantifier {q:?} is not one of master 6's"))),
        }
        Ok(out)
    }

    /// One instance of row `r` under `env`, or none when one of its data types is not a candidate.
    fn instance(&self, r: &RowTemplate, env: &Env) -> Result<Option<Instance>, String> {
        let mut t: BTreeMap<&str, String> = BTreeMap::new();
        for c in ROW6_TEMPLATED.iter().filter(|c| **c != "spelling") {
            t.insert(c, subst(r.get(c), env)?);
        }
        let args = items(&t["args"], ';');
        if !self.by_name.contains_key(&t["out"]) || args.iter().any(|a| !self.by_name.contains_key(a)) {
            return Ok(None);
        }
        let mut with_types = env.clone();
        for (k, a) in args.iter().enumerate() {
            let ts = self.candidate(a).map(|c| c.type_spelling.replace("@.", &format!("in{k}."))).unwrap_or_default();
            with_types.insert(format!("TYPE{k}"), ts);
        }
        let spelling = subst(r.get("spelling"), &with_types)?;
        let held = |expr: &str| property_terms(expr).iter().all(|(k, p)| *k < args.len() && self.carries(&args[*k], p));
        let (mut ups, mut envs, mut vars) = (Vec::new(), Vec::new(), Vec::new());
        for item in items(&t["flows"], ';') {
            match parse_flow(&item).map_err(|e| format!("{e} (row {})", r.get("ref")))? {
                FlowItem::Up(p, e) => {
                    if self.carries(&t["out"], &p) && held(&e) {
                        ups.push((p, e));
                    }
                }
                FlowItem::Env(k, a, e) => envs.push((k, a, e)),
                FlowItem::Var(k, dt, props) => {
                    if !self.by_name.contains_key(&dt) {
                        return Ok(None);
                    }
                    let kept = props.into_iter().filter(|(p, e)| self.carries(&dt, p) && held(e)).collect();
                    vars.push((k, dt, kept));
                }
            }
        }
        let mut guards = Vec::new();
        for q in items(&t["guards"], ';') {
            let (strength, subject, rel, expr, list) = parse_guard(&q).map_err(|e| format!("{e} (row {})", r.get("ref")))?;
            if subject == "env.trips" {
                if let Some(depth) = self.spec.depth {
                    guards.push((strength, subject, rel, replace_depth(&expr, depth), list));
                }
                continue;
            }
            let (k, p) = subject.strip_prefix("in").and_then(|s| s.split_once('.')).ok_or_else(|| bug(format!("the row {} has a guard on {subject}, which is not an input", r.get("ref"))))?;
            let k: usize = k.parse().map_err(|_| bug(format!("the guard {q:?} of {} names no input", r.get("ref"))))?;
            if k < args.len() && self.carries(&args[k], p) && held(&expr) {
                guards.push((strength, subject, rel, expr, list));
            }
        }
        Ok(Some(Instance { reference: r.get("ref").to_string(), name: t["name"].clone(), kind: r.get("kind").to_string(), out: t["out"].clone(), args, ups, envs, vars, guards, spelling, prelude: r.get("prelude").to_string() }))
    }

    fn rows(&self, var_types: &BTreeSet<String>, leafless: &BTreeSet<String>) -> Result<Vec<Instance>, String> {
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
                    next.extend(self.quantifier(&q, env, &row_elems, var_types, leafless)?);
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
        let mut built: BTreeSet<String> = rows.iter().filter(|r| r.kind != "function" && r.kind != "adapter").map(|r| r.out.clone()).collect();
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

    /// The adapter symbol that fits property `prop` of a value of data type `dt`.
    fn fitter(&self, dt: &str, prop: &str) -> Result<String, String> {
        let shape = if dt.starts_with("[][]") {
            "[][]"
        } else if dt.starts_with("[]") {
            "[]"
        } else {
            ""
        };
        let name = FIT6.iter().find(|(s, p, _)| *s == shape && *p == prop).map(|(_, _, n)| *n).ok_or_else(|| bug(format!("no adapter fits the {prop} of {dt}")))?;
        Ok(name.replace("{E}", &dt[shape.len()..]))
    }

    /// The repairs of one guard ([`View6`], step 4): its list with the guard's operands, properties and adapters bound,
    /// then the row's work-bound substitutes (its own work bound's list, else `subst`), so the last always applies.
    fn repairs(&self, args: &[String], work: Option<&str>, subject: &str, expr: &str, list: &str) -> Result<Vec<[String; 4]>, String> {
        let mut env = Env::new();
        if let Some(rest) = subject.strip_prefix("in") {
            let (s, ps) = rest.split_once('.').ok_or_else(|| bug(format!("the subject {subject}")))?;
            let si: usize = s.parse().map_err(|_| bug(format!("the subject {subject}")))?;
            for (k, v) in [("S", s.to_string()), ("PS", ps.to_string()), ("SUBJECT", subject.to_string()), ("AGAINST", expr.to_string()), ("MIN", format!("min({subject}, {expr})"))] {
                env.insert(k.to_string(), v);
            }
            env.insert("FITS".to_string(), if list == "clampix" { String::new() } else { self.fitter(&args[si], ps)? });
            if let Some((j, pj)) = expr.strip_prefix("in").and_then(|r| r.split_once('.')).filter(|(j, pj)| !j.is_empty() && j.bytes().all(|b| b.is_ascii_digit()) && !pj.is_empty() && pj.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')) {
                let ji: usize = j.parse().map_err(|_| bug(format!("the guard's {expr}")))?;
                env.insert("J".to_string(), j.to_string());
                env.insert("PJ".to_string(), pj.to_string());
                env.insert("FITJ".to_string(), if list == "fit.pair" || list == "fit.against" { self.fitter(&args[ji], pj)? } else { String::new() });
            }
        } else if let Some(p) = subject.strip_prefix("out.") {
            for (k, v) in [("PS", p.to_string()), ("SUBJECT", subject.to_string()), ("AGAINST", expr.to_string()), ("FITS", self.fitter(&self.root, p)?)] {
                env.insert(k.to_string(), v);
            }
        }
        let list_of = |name: &str| -> Result<Vec<&RepairTemplate>, String> {
            let v: Vec<&RepairTemplate> = self.master.repairs.iter().filter(|r| r.list == name).collect();
            if v.is_empty() {
                Err(bug(format!("the repair list {name} is not in master/6/repairs.tsv")))
            } else {
                Ok(v)
            }
        };
        let mut out = Vec::new();
        for r in list_of(list)? {
            let guard = if r.guard.contains("depth") {
                match self.spec.depth {
                    None => "true".to_string(),
                    Some(d) => replace_depth(&r.guard, d),
                }
            } else {
                r.guard.clone()
            };
            out.push([subst(&r.site, &env)?, subst(&r.when, &env)?, guard, subst(&r.template, &env)?]);
        }
        if !["subst", "subst.const", "fit.root"].contains(&list) {
            for r in list_of(work.unwrap_or("subst"))? {
                out.push([r.site.clone(), r.when.clone(), r.guard.clone(), r.template.clone()]);
            }
        }
        Ok(out)
    }

    /// Every size name of the entry, sorted, each at least 1 (the entry points' contract).
    fn size_names(&self) -> Vec<String> {
        let mut names: BTreeSet<String> = BTreeSet::new();
        for e in &self.spec.env {
            names.extend(e.dims.iter().cloned());
            if e.kind == "size" {
                names.insert(e.name.clone());
            }
        }
        names.extend(self.spec.output.2.iter().cloned());
        names.into_iter().collect()
    }
}

/// A prelude's text on one line of the table (`\` as `\\`, a line break as `\n`).
fn escape6(text: &str) -> String {
    crate::geneframe::escape(text)
}

impl View6 {
    /// The view of a master 6 specification over a master 6.
    pub fn load(master: &Master6, spec_text: &str) -> Result<View6, String> {
        let spec = ViewSpec6::parse(spec_text)?;
        let elements = &spec.elements;
        let multi = elements.len() > 1;
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
                let mut spell_env = env.clone();
                spell_env.insert("M".to_string(), match env.get("E") {
                    Some(e) if multi => format!("_{e}"),
                    _ => String::new(),
                });
                candidates.push(Candidate {
                    name,
                    kind: t.kind.clone(),
                    members,
                    member_list,
                    tag: subst(&t.tag, &env)?,
                    constant: subst(&t.constant, &env)?,
                    constant_props: t.constant_props.clone(),
                    properties: master.properties.get(&t.name).cloned().unwrap_or_default(),
                    spell: subst(&t.spell, &spell_env)?,
                    type_spelling: subst(&t.type_spelling, &env)?,
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
        let b = Builder { spec: &spec, master, multi, candidates, by_name, root };
        // 2 and 3: the rows, reachable both ways, the data types with a variable settled.
        let root_types: BTreeSet<String> = root_vars.iter().map(|(_, dt, _)| dt.clone()).collect();
        let binders = |rows: &[Instance]| -> BTreeSet<String> { rows.iter().flat_map(|r| r.vars.iter().map(|(_, dt, _)| dt.clone())).collect() };
        let none = BTreeSet::new();
        let mut var_types: BTreeSet<String> = root_types.union(&binders(&b.rows(&none, &none)?)).cloned().collect();
        let mut settled_rows = None;
        for _ in 0..b.candidates.len() + 2 {
            let rows = b.reachable(b.rows(&var_types, &none)?)?;
            let have: BTreeSet<String> = root_types.union(&binders(&rows)).cloned().collect();
            let settled: BTreeSet<String> = var_types.intersection(&have).cloned().collect();
            if settled == var_types {
                settled_rows = Some(rows);
                break;
            }
            var_types = settled;
        }
        let mut rows = settled_rows.ok_or_else(|| bug("the data types with a variable do not settle".to_string()))?;
        let mut leafless = BTreeSet::new();
        let mut settled = false;
        for _ in 0..b.candidates.len() + 2 {
            let more: BTreeSet<String> = leafless.union(&b.leafless(&rows)?).cloned().collect();
            if more == leafless {
                settled = true;
                break;
            }
            leafless = more;
            rows = b.reachable(b.rows(&var_types, &leafless)?)?;
        }
        if !settled {
            return Err(bug("the data types with no leaf do not settle".to_string()));
        }
        let stray: Vec<String> = binders(&rows).difference(&var_types).cloned().collect();
        if !stray.is_empty() {
            return Err(bug(format!("the leaf rows of {leafless:?} admit binders of {stray:?}, which have no var row")));
        }
        let mut used: BTreeSet<String> = BTreeSet::from([b.root.clone()]);
        for r in &rows {
            used.insert(r.out.clone());
            used.extend(r.args.iter().cloned());
            used.extend(r.vars.iter().map(|(_, dt, _)| dt.clone()));
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
        // 4. The written table: the four kinds of row.
        let mut lines: Vec<String> = Vec::new();
        let holder = if types.iter().any(|t| t.name == "size") { Some("size".to_string()) } else { types.iter().find(|t| t.properties.iter().any(|(_, k)| k == "size")).map(|t| t.name.clone()) };
        for t in &types {
            match t.kind.as_str() {
                "scalar" => lines.push(format!("data_type\t{}\tscalar", t.name)),
                "array" | "tuple" => lines.push(format!("data_type\t{}\t{}\t{}", t.name, t.kind, t.members)),
                _ => {
                    let (params, result) = t.members.split_once(" -> ").ok_or_else(|| bug(format!("the function type {} has members {:?}", t.name, t.members)))?;
                    lines.push(format!("data_type\t{}\tfunction\t{params}\t{result}", t.name));
                }
            }
            for (p, _) in &t.properties {
                lines.push(format!("carries\t{}\t{p}", t.name));
            }
            if !t.constant.is_empty() {
                let props = items(&t.constant_props, '&').join(";");
                lines.push(format!("constant\t{}\t{}\t{}", t.name, t.constant, if props.is_empty() { "-" } else { &props }));
            }
            if holder.as_deref() == Some(t.name.as_str()) {
                for n in b.size_names() {
                    lines.push(format!("size_name\t{}\t{n}\t1", t.name));
                }
            }
        }
        for r in &rows {
            let role = match r.kind.as_str() {
                "const" => "constant",
                "var" => "variable",
                _ => "function",
            };
            let prelude = if r.prelude.is_empty() { "-".to_string() } else { escape6(master.preludes.get(&r.prelude).ok_or_else(|| bug(format!("the row {} names the prelude {}, which master/6/preludes/ does not hold", r.reference, r.prelude)))?) };
            lines.push(format!("symbol\t{}\t{}\t{role}\t{}\t{prelude}", r.name, r.reference, if r.kind == "adapter" { "adapter" } else { "-" }));
            if r.kind == "const" {
                let pool = spec.literals.get(&r.out).ok_or_else(|| bug(format!("the constant {} has no literal pool", r.name)))?;
                lines.push(format!("spelling\t{}\tpool\t{}", r.name, pool.iter().map(|x| format!("{x}{}", r.out)).collect::<Vec<_>>().join(",")));
            } else if !r.spelling.is_empty() {
                lines.push(format!("spelling\t{}\t\t{}", r.name, r.spelling));
            }
        }
        let mut ops: Vec<&str> = Vec::new();
        for (op, _, _) in &master.formulas {
            if !ops.contains(&op.as_str()) {
                ops.push(op);
            }
        }
        for op in ops {
            lines.push(format!("symbol\t={op}\t=\tformula\t-\t-"));
            for (o, form, sp) in master.formulas.iter().filter(|(o, _, _)| o == op) {
                lines.push(format!("spelling\t={o}\t{form}\t{sp}"));
            }
        }
        for r in &rows {
            for (k, a) in r.args.iter().enumerate() {
                lines.push(format!("argument\t{}\t{k}\tin\t{a}\t-", r.name));
            }
            lines.push(format!("argument\t{}\t0\tout\t{}\t-", r.name, r.out));
        }
        for r in &rows {
            for (p, e) in &r.ups {
                lines.push(format!("flow\t{}\tout\tup\t{p}\t{e}", r.name));
            }
            let mut groups: BTreeMap<(usize, &str), u32> = BTreeMap::new();
            for (k, dt, props) in &r.vars {
                let g = groups.entry((*k, dt.as_str())).or_insert(0);
                let name = if *g == 0 { format!("var.{dt}") } else { format!("var.{dt}/{g}") };
                *g += 1;
                lines.push(format!("flow\t{}\tin{k}\tdown\t{name}\t1", r.name));
                for (p, e) in props {
                    lines.push(format!("flow\t{}\tin{k}\tdown\t{name}.{p}\t{e}", r.name));
                }
                let spell = &b.candidate(dt).ok_or_else(|| bug(format!("no data type {dt}")))?.spell;
                if !spell.is_empty() {
                    lines.push(format!("flow\t{}\tin{k}\tdown\t{name}.spell\t{spell}", r.name));
                }
            }
            for (k, a, e) in &r.envs {
                lines.push(format!("flow\t{}\tin{k}\tdown\tenv.{a}\t{e}", r.name));
            }
        }
        for r in &rows {
            let work = r.guards.iter().find(|g| g.1 == "env.trips").map(|g| g.4.as_str());
            let mut ordered: Vec<&GuardParts> = r.guards.iter().filter(|g| g.1 == "env.trips").collect();
            ordered.extend(r.guards.iter().filter(|g| g.1 != "env.trips"));
            let mut order = 1;
            for (strength, subject, rel, expr, list) in ordered {
                let at = if subject == "env.trips" {
                    0
                } else {
                    order += 1;
                    order - 1
                };
                lines.push(format!("guard\t{}\t{at}\t{strength}\t{subject} {rel} {expr}", r.name));
                for [site, when, guard, template] in b.repairs(&r.args, work, subject, expr, list)? {
                    lines.push(format!("repair\t{}\t{at}\t{site}\t{when}\t{guard}\t{template}", r.name));
                }
            }
        }
        let (out_name, _, out_dims) = &spec.output;
        lines.push(format!("entry\t{out_name}\t{}", b.root));
        for (vname, vdt, ps) in &root_vars {
            if var_rows.contains(vdt.as_str()) {
                let ps = ps.iter().map(|(p, v)| format!("{p}={v}")).collect::<Vec<_>>().join(";");
                lines.push(format!("parameter\t{vname}\t{vdt}\t{}\t-\t-\t-\t-", if ps.is_empty() { "-" } else { &ps }));
            }
        }
        let props = rank_properties(out_dims.len()).ok_or_else(|| bug(format!("the output has {} dimensions", out_dims.len())))?;
        for (i, (p, d)) in props.iter().zip(out_dims).enumerate() {
            let subject = format!("out.{p}");
            lines.push(format!("guard\tentry\t{i}\tmust\t{subject} = {d}"));
            for [site, when, guard, template] in b.repairs(&[], None, &subject, d, "fit.root")? {
                lines.push(format!("repair\tentry\t{i}\t{site}\t{when}\t{guard}\t{template}"));
            }
        }
        let mut tables_tsv = lines.join("\n");
        tables_tsv.push('\n');
        let types = types.iter().map(|t| [t.name.clone(), t.kind.clone(), t.members.clone(), t.tag.clone()]).collect();
        let symbols = rows.iter().map(|r| (r.name.clone(), r.kind.clone(), r.out.clone(), r.args.clone())).collect();
        Ok(View6 { spec: spec.clone(), tables_tsv, types, symbols })
    }

    /// The view's table, by file name.
    pub fn artefacts(&self) -> [(&'static str, &str); 1] {
        [("tables.tsv", self.tables_tsv.as_str())]
    }

    /// The kingdom the view is, loaded from its written text.
    pub fn kingdom(&self, name: &str) -> Result<Kingdom6, String> {
        Kingdom6::load(name, &self.tables_tsv, &self.spec)
    }
}

macro_rules! m6 {
    ($p:literal) => {
        ($p, include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/view.tsv")), include_str!(concat!("../../kingdoms/futhark/sig/", $p, "/m6/tables.tsv")))
    };
}

/// Every committed master 6 view: (problem, the specification, and the table `view.py` wrote), under
/// `kingdoms/futhark/sig/<problem>/m6/`.
pub const COMMITTED6: [(&str, &str, &str); 7] = [m6!("a7"), m6!("histogram"), m6!("kmeans"), m6!("kmeans_i32"), m6!("pathfinder"), m6!("reduce"), m6!("scan")];

/// The committed view of a problem, computed over the committed master 6.
pub fn committed(problem: &str) -> Result<View6, String> {
    let (_, spec, _) = COMMITTED6.iter().find(|c| c.0 == problem).ok_or_else(|| bug(format!("{problem} is not a committed master 6 view")))?;
    View6::load(&Master6::committed()?, spec)
}

/// The committed kingdom of a problem, loaded from its committed table (`tables.tsv`).
pub fn committed_kingdom(problem: &str) -> Result<Kingdom6, String> {
    let (_, spec, tables) = COMMITTED6.iter().find(|c| c.0 == problem).ok_or_else(|| bug(format!("{problem} is not a committed master 6 view")))?;
    Kingdom6::load(problem, tables, &ViewSpec6::parse(spec)?)
}

#[cfg(test)]
mod tests;
