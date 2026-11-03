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
        let frames = tsv("frames.tsv", frames, &["family", "kind", "code", "scope", "binds"])?
            .into_iter()
            .map(|f| {
                let binds: Vec<String> = f[4].split(',').map(str::to_string).collect();
                if let Some(b) = binds.iter().find(|b| b.is_empty() || b.contains(':')) {
                    return Err(format!("master/frames.tsv: the binding {b:?} is not a class (a variable is named by its class)"));
                }
                Ok(FrameTemplate { code: f[2].into(), scope: scope_list(f[3]), binds })
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
/// `families` line belongs to the matrix view only: a signature view admits every family of the master.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewSpec {
    pub shape: Shape,
    pub master: Option<u32>,
    pub element: String,
    /// The inputs in order, a size input with the element `size`.
    pub inputs: Vec<SpecInput>,
    pub sizes: Vec<String>,
    /// The output's name, element and dimensions.
    pub output: Option<SpecInput>,
    pub depth: u32,
    /// The matrix view's families (a signature view has none: it admits every family).
    pub families: Vec<String>,
    pub literals: Vec<String>,
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
            inputs: Vec::new(),
            sizes: Vec::new(),
            output: None,
            depth: 0,
            families: Vec::new(),
            literals: Vec::new(),
            dims: Vec::new(),
            roles: Vec::new(),
            root: Vec::new(),
        };
        let list = |x: &str| x.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect::<Vec<_>>();
        let mut families_line = false;
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
                "element" => s.element = name.to_string(),
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
                "literals" => s.literals = list(name),
                "dims" => s.dims = list(name),
                "roles" => s.roles = list(name),
                "root" => s.root = list(name),
                other => return Err(format!("a specification line of kind {other}")),
            }
        }
        if s.element.is_empty() {
            return Err("the specification names no element".into());
        }
        if s.shape == Shape::Signature && families_line {
            return Err("a signature specification has a families line: a signature view admits every family".into());
        }
        if s.shape == Shape::Signature && s.output.is_none() {
            return Err("the signature specification has no output".into());
        }
        Ok(s)
    }
}

// ---- binding ----------------------------------------------------------------------------------------------------

fn class_dims(cls: &str) -> Vec<&str> {
    cls.strip_prefix("a.").map(|r| r.split('.').collect()).unwrap_or_default()
}

fn array_class(dims: &[&str]) -> String {
    format!("a.{}", dims.join("."))
}

fn futhark_type(cls: &str, elem: &str) -> String {
    class_dims(cls).iter().map(|d| format!("[{d}]")).collect::<String>() + elem
}

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

/// The variables a class binds: the class, the next variable of its class (`.var`), its Futhark type, its dimensions
/// (`.dims`, `.br` as brackets, `.d0` the first), and `.ix`, the next index variable over its first dimension (a
/// tabulate of its rows binds it).
fn class_env(var: &str, cls: &str, elem: &str, ctx: Option<&Ctx>) -> Vec<(String, String)> {
    let ds = class_dims(cls);
    vec![
        (var.to_string(), cls.to_string()),
        (format!("{var}.var"), next_var(cls, ctx)),
        (format!("{var}.type"), futhark_type(cls, elem)),
        (format!("{var}.dims"), ds.join(".")),
        (format!("{var}.br"), ds.join("][")),
        (format!("{var}.d0"), ds.first().map_or(String::new(), |d| d.to_string())),
        (format!("{var}.ix"), ds.first().map_or(String::new(), |d| next_var(&format!("ix.{d}"), ctx))),
    ]
}

/// The sets a view's quantifiers range over.
#[derive(Debug, Clone, Default)]
struct Sets {
    elem: String,
    dims: Vec<String>,
    classes: Vec<String>,
    rank1: Vec<String>,
    states: Vec<String>,
    free: Vec<String>,
    inputs: Vec<(String, String)>,
    roles: Vec<String>,
    pairs: Vec<(String, String)>,
}

/// The sets of a signature view, derived from the specification: the dims in order of first appearance, the classes
/// (each input's, its rows', the output's, each dim's rank-1 class), the rank-1 classes, the states a loop may carry.
/// The context-free classes are derived by [`view_rows`].
fn signature_sets(spec: &ViewSpec) -> Result<Sets, String> {
    let elem = spec.element.clone();
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
    let mut classes: Vec<String> = Vec::new();
    fn add(classes: &mut Vec<String>, c: String) {
        if !classes.contains(&c) {
            let ds: Vec<String> = class_dims(&c).iter().map(|d| d.to_string()).collect();
            classes.push(c);
            if ds.len() >= 2 {
                add(classes, array_class(&ds[1..].iter().map(String::as_str).collect::<Vec<_>>()));
            }
        }
    }
    let class_of = |p: &SpecInput| if p.dims.is_empty() { elem.clone() } else { array_class(&p.dims.iter().map(String::as_str).collect::<Vec<_>>()) };
    for p in spec.inputs.iter().filter(|p| p.elem != "size") {
        add(&mut classes, class_of(p));
    }
    add(&mut classes, class_of(output));
    for d in &dims {
        add(&mut classes, array_class(&[d.as_str()]));
    }
    let rank1: Vec<String> = classes.iter().filter(|c| class_dims(c).len() == 1).cloned().collect();
    let states = std::iter::once(elem.clone()).chain(rank1.iter().cloned()).collect();
    let inputs = spec.inputs.iter().filter(|p| p.elem != "size").map(|p| (p.name.clone(), class_of(p))).collect();
    Ok(Sets { elem, dims, classes, rank1, states, free: Vec::new(), inputs, ..Sets::default() })
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
                Binder { sets: Sets { elem: spec.element.clone(), roles: spec.roles.clone(), pairs, ..Sets::default() }, depth: 0, frames: Vec::new(), contexts: vec![Ctx::default()] }
            }
        };
        if spec.shape == Shape::Signature {
            let base = Env::default().with(&[("E".into(), spec.element.clone())]);
            for f in &master.frames {
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
            let taken: BTreeSet<String> = b.sets.inputs.iter().map(|(n, _)| n.clone()).chain(b.sets.dims.iter().cloned()).chain(b.sets.dims.iter().map(|d| format!("zeros_{d}"))).collect();
            for (_, binds) in &b.frames {
                for c in binds {
                    for k in 1..=b.depth.max(1) {
                        let v = var_name(c, k);
                        let t = c.strip_prefix("ix.").map(|_| format!("t{v}"));
                        if let Some(x) = std::iter::once(&v).chain(t.as_ref()).find(|x| taken.contains(*x)) {
                            return Err(format!("the bound variable {x} has the name of an input, a size or a zero array of the view"));
                        }
                    }
                }
            }
        }
        Ok(b)
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
        Ok(match kind {
            "dims" => s.dims.iter().map(|d| one(vec![(var.to_string(), d.clone()), (format!("{var}.ix"), next_var(&format!("ix.{d}"), env.ctx.as_ref()))])).collect(),
            "rank1" => s.rank1.iter().map(|c| one(vec![(var.to_string(), class_dims(c)[0].to_string()), (format!("{var}.ix"), next_var(&format!("ix.{}", class_dims(c)[0]), env.ctx.as_ref()))])).collect(),
            "states" => s.states.iter().map(|c| one(class_env(var, c, &s.elem, env.ctx.as_ref()))).collect(),
            "free" => s.free.iter().map(|c| one(class_env(var, c, &s.elem, env.ctx.as_ref()))).collect(),
            "vars" => {
                let ctx = env.ctx.as_ref().ok_or_else(|| format!("the quantifier {q} outside a context"))?;
                self.frame_vars(ctx).into_iter().filter(|(_, c)| *c == arg).map(|(n, _)| one(v(&n))).collect()
            }
            "inputs" => s.inputs.iter().filter(|(_, c)| *c == arg).map(|(n, _)| one(v(n))).collect(),
            // The named class, when the view has it (`V2:class(a.{D2}.{D})`: the rows a histogram of rows reads).
            "class" => s.classes.iter().filter(|c| **c == arg).map(|c| one(class_env(var, c, &s.elem, env.ctx.as_ref()))).collect(),
            "rows" => s
                .classes
                .iter()
                .filter(|y| {
                    let ys = class_dims(y);
                    ys.len() >= 2 && array_class(&ys[1..]) == arg
                })
                .map(|y| one(class_env(var, y, &s.elem, env.ctx.as_ref())))
                .collect(),
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
        let Some(ctx) = env.ctx.as_ref().filter(|_| !self.sets.free.contains(&cls)) else {
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
        && (row.elems == "*" || row.elems.split(',').any(|e| e == spec.element))
}

fn expand(rows: &[(&MasterRow, &[String])], binder: &Binder, env: &Env, out: &mut Vec<ViewRow>) -> Result<(), String> {
    let mut i = 0;
    while i < rows.len() {
        let (row, scope) = rows[i];
        let Some(q) = scope.first() else {
            match row.when.as_str() {
                "" => out.push(instantiate(row, binder, env)?),
                "shallow" if env.shallow => out.push(instantiate(row, binder, env)?),
                "shallow" => {}
                other => return Err(format!("the row {} has the condition {other:?}", row.reference)),
            }
            i += 1;
            continue;
        };
        let mut j = i;
        while j < rows.len() && rows[j].1.first() == Some(q) {
            j += 1;
        }
        let block: Vec<(&MasterRow, &[String])> = rows[i..j].iter().map(|(r, s)| (*r, &s[1..])).collect();
        for b in binder.quantifier(q, env)? {
            expand(&block, binder, &b, out)?;
        }
        i = j;
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
    let base = Env::default().with(&[("E".into(), spec.element.clone())]);
    let o = match spec.shape {
        Shape::Matrix => {
            let root = format!("mat.{}", spec.root.join("."));
            let mut rows = Vec::new();
            expand(&picked, &binder, &base, &mut rows)?;
            return Ok((reachable(rows, &root)?, root, binder));
        }
        Shape::Signature => spec.output.as_ref().ok_or("the signature specification has no output")?,
    };
    let cls = if o.dims.is_empty() { spec.element.clone() } else { array_class(&o.dims.iter().map(String::as_str).collect::<Vec<_>>()) };
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
        let free: Vec<String> = binder.sets.classes.iter().filter(|c| !produced.contains(c.as_str())).cloned().collect();
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

fn signature_types(spec: &ViewSpec, rows: &[ViewRow], root: &str) -> Vec<ViewType> {
    let elem = spec.element.as_str();
    let literals: Vec<String> = spec.literals.iter().map(|l| format!("{l}{elem}")).collect();
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
            let slot = if cls.starts_with("ix.") {
                "ix"
            } else if cls.starts_with("a.") {
                "arr"
            } else {
                elem
            };
            let meaning = match slot {
                "ix" => format!("an index in range for [{}]", &cls[3..]),
                "arr" => format!("a [{}]{elem}", class_dims(cls).join("][")),
                _ => format!("an {elem}"),
            } + &if ctx.is_empty() { ", the interface's input class".to_string() } else { format!(", in the context {ctx}") };
            ViewType { dual: d.to_string(), slot: slot.to_string(), form: cls.to_string(), literals: if slot == elem { literals.clone() } else { vec![] }, root: d == root, meaning }
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
                let types = signature_types(&spec, &rows, &root);
                let rows = ordered(rows);
                let mut v = View { types_tsv: types_text(&types), functions_tsv: functions_text(&rows), datatype: String::new(), rules: String::new(), egraph_rows: String::new(), sorts: String::new(), spec };
                v.signature_egraph(&types, &rows, master);
                Ok(v)
            }
            Shape::Matrix => {
                let named: BTreeSet<&str> = rows.iter().flat_map(|r| std::iter::once(r.out.as_str()).chain(r.args.iter().map(String::as_str))).collect();
                let mut types = Vec::new();
                let base = Env::default().with(&[("E".into(), spec.element.clone())]);
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
    /// master's rules on every type they apply to (`elem`: the element types; `arr`: the rank-1 array types in a
    /// context that hold the row the rule `needs`), for the element types each rule is exact for.
    fn signature_egraph(&mut self, types: &[ViewType], rows: &[ViewRow], master: &Master) {
        let elem = self.spec.element.clone();
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
        for on in ["elem", "arr"] {
            for t in types {
                let d = &t.dual;
                let applies = match on {
                    "elem" => t.slot == elem,
                    _ => t.slot == "arr" && t.form.matches('.').count() == 1 && d.contains(':'),
                };
                if !applies {
                    continue;
                }
                let mine: Vec<&RuleTemplate> = master.rules.iter().filter(|r| r.on == on && r.elems.split(',').any(|e| e == elem) && (r.needs.is_empty() || names.contains(format!("{}@{d}", r.needs).as_str()))).collect();
                if mine.is_empty() {
                    continue;
                }
                rules.push_str(&format!("; {d}\n"));
                for r in mine {
                    rules.push_str(&rule_text(&r.rule, d));
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

/// Every committed specification, by the directory [`write_committed`] writes its tables into: the four signature
/// problems (`sig/<problem>`) and A7's matrix view (`a7`).
pub const COMMITTED: [(&str, &str); 6] = [
    ("sig/histogram", include_str!("../kingdoms/futhark/sig/histogram/view.tsv")),
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
