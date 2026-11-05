//! THE WGSL KINGDOM AS ROWS of the gene frame's fixed schema (design `docs/DESIGN_geneframe.md` 3.3, 8.4 and 8.5;
//! plan `docs/PLAN_futhark_kingdom.md` 7.7, task K3). The kingdom `wgsl` (its Gram problem the variant `wgsl.gram`) in
//! the one set of tables `kingdoms/master/`, written by `kingdoms/wgsl_true/gen_rows.py` through `write_rows.py` (never
//! by hand; its `--check` is a test), holds the four kinds of row and nothing else: nine data types where the old table had 305, because every rule the old table carried in a type is a property
//! here, moved by Flow rows and enforced by `must` Guard rows with their repairs inline:
//!
//! | rule | old carrier | rows |
//! |---|---|---|
//! | no two invocations write one element | ownership types | `u32.owned` (injective in the invocation id): `up add_u: owned := (in0.owned and in1.uniform) or (in1.owned and in0.uniform)`; `guard store: in1.owned must -> {const}`; workgroup memory likewise by `local_owned` (injective in the local invocation index): `guard wg_store: in0.local_owned` |
//! | barriers only in uniform control flow | statement rungs `L`/`C`/`W`/`O` | `down if_ in1, in2: env.uniform := env.uniform and in0.uniform`; `guard barrier: env.uniform must -> {const}` |
//! | every loop ends, within the work bound | fuel strata `_f<k>` | `count` terminals with a literal `value`; `down loop in1: env.work := env.work * in0.value`; `guard loop: env.work * in0.value <= 64 must -> {const}` (the Gram interface's old K = 1 budget) |
//! | every index is in range | masked index types | `guard load, store: in1.bound <= in0.len must -> [in1] min({1}, arrayLength(&{0}) - 1u)` |
//! | the output is written | the `O` rung | the kernel row's own spelling ends in the primary output's store |
//! | vector widths agree | one type per width | `vecf.width`; `guard vadd, dot: in1.width = in0.width must -> [in1] a swizzle or an extension by zeros` |
//!
//! Scope is context (design 8.5): the entry's builtins (`lin`, `gid_x`, `cfg_d`, ...) and the buffers (`phi`, `h`) are
//! the root environment's variables, `let` bindings and loop counters are variables introduced by down flows, and
//! `var.T`'s index word chooses among those in scope. Nothing is folded into a type. The constant `{}` of `stmt` (the
//! empty block) is the repair that drops a statement; an illegal statement stays representable in the gene and is
//! INEXPRESSIBLE in the program.
//!
//! [`WgslRows::load`] reads the table, checks it, builds the contract and the decoder once and checks the entry;
//! [`WgslRows::program`] is pass 5, the decoder's rendering ([`Decoder::render`]) with the kernel's prelude (the
//! bindings) before it. [`WgslRows::device_rules`] compiles the Flow and Guard rows into the closed forms a device
//! decoder evaluates, and [`WgslRows::twin_decode`] evaluates them on the host, so a test can require the two decodes
//! equal ([`Twin`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::geneframe::{Entry, FlowName, KingdomTables, Owner, Place, PropertyExpr, Relation, Role, Site, Strength, Ty, Way};
use crate::geneframe::formula::Condition;
use crate::kingdom_contract::{relations_from_tables, KingdomContract, SymbolId};
use crate::typed_environment::{Choice, Decoder, Decoding, Failure, Node, Substituted, Substitution};
use crate::typed_gep::{express, Domain, GeneLayout, Tree};

/// The kingdom in the one set of tables (`kingdoms/master/`, [`crate::geneframe::master`]): the parent, and the Gram
/// problem's variant, which holds every row.
pub const KINGDOM: &str = "wgsl";
pub const VARIANT: &str = "wgsl.gram";

/// The per-invocation work bound the loop guard enforces (`gen_rows.py` BUDGET).
pub const BUDGET: i64 = 64;

fn bug(what: String) -> String {
    format!("our software has a bug: {KINGDOM}: {what}")
}

/// The loaded WGSL kingdom: its rows, entry, contract and decoder, built once.
#[derive(Debug, Clone)]
pub struct WgslRows {
    pub tables: Arc<KingdomTables>,
    pub entry: Entry,
    pub contract: Arc<KingdomContract>,
    pub decoder: Decoder,
}

/// A rendered kernel: its text (the prelude, then the module), the nodes it renders, and each statement or operand a
/// `must` guard repaired, by its node's path and the guard's condition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgslProgram {
    pub text: String,
    pub nodes: usize,
    /// Nodes replaced by their data type's constant or an input (`{const}`, `{in.own}`), with the condition that failed.
    pub substitutions: Vec<(Vec<u32>, String)>,
    /// Operands rewritten in place (an index clamp, a width fit), with the condition that failed.
    pub rewrites: Vec<(Vec<u32>, String)>,
}

impl WgslRows {
    /// Load the committed rows of [`VARIANT`] from the one set of tables.
    pub fn load() -> Result<WgslRows, String> {
        WgslRows::from_text(&crate::geneframe::master::select(VARIANT)?.named_text()?)
    }

    /// Load a table's text (the named text form): parse, check, build the contract and the decoder, check the entry.
    pub fn from_text(text: &str) -> Result<WgslRows, String> {
        let ty = |i: usize| -> Result<Ty, String> {
            Ty::check_dual_count(KINGDOM, i + 1, Ty::MAX_WGSL_DUALS)?;
            Ok(Ty::Wgsl(i as u16))
        };
        let (tables, entry) = KingdomTables::parse_named(VARIANT, text, &ty)?;
        tables.check().map_err(|d| d.join("\n"))?;
        let contract = relations_from_tables(&tables, vec![entry.output]).check().map_err(|d| d.join("\n"))?;
        let (tables, contract) = (Arc::new(tables), Arc::new(contract));
        let decoder = Decoder::from_arcs(Arc::clone(&tables), Arc::clone(&contract)).map_err(|d| d.join("\n"))?;
        decoder.check_entry(&entry).map_err(|d| d.join("\n"))?;
        Ok(WgslRows { tables, entry, contract, decoder })
    }

    /// The root data type (`kernel`).
    pub fn root(&self) -> Ty {
        self.entry.output
    }

    /// A data type by name.
    pub fn ty(&self, name: &str) -> Option<Ty> {
        self.tables.data_types.iter().find(|d| d.name == name).map(|d| d.id)
    }

    /// A data type's name.
    pub fn type_name(&self, ty: Ty) -> &str {
        self.tables.data_type(ty).map_or("?", |d| d.name.as_str())
    }

    /// A symbol's id by name.
    pub fn symbol_named(&self, name: &str) -> Option<SymbolId> {
        self.decoder.symbol_named(name)
    }

    /// A symbol's name.
    pub fn name(&self, s: SymbolId) -> &str {
        self.contract.name(s)
    }

    /// The literals of a constant symbol's pool (its `pool` spelling).
    pub fn literals(&self, s: SymbolId) -> Vec<String> {
        self.tables.symbol(i64::from(s)).and_then(|r| r.spelling("pool")).map(|p| p.split(',').map(str::to_string).collect()).unwrap_or_default()
    }

    /// The Dc lengths of the gene: each constant symbol's data type and its pool's length.
    pub fn const_len(&self) -> BTreeMap<Ty, u32> {
        self.tables.symbols.iter().filter(|s| s.role == Role::Constant).filter_map(|s| self.tables.output(s.symbol).map(|ty| (ty, self.literals(s.symbol as SymbolId).len() as u32))).collect()
    }

    /// The gene's layout at program size `n` (Ferreira's head from the contract's arities, one frame per type the
    /// root reaches, a Dc lane per constant-capable type, a Dv lane per type with a variable terminal).
    pub fn layout(&self, n: usize) -> Result<GeneLayout, String> {
        let heads = self.contract.heads_from_program_size(self.root(), n)?;
        GeneLayout::from_heads(&self.contract, self.root(), &heads).map_err(|e| bug(format!("the layout at program size {n}: {e:?}")))
    }

    /// Passes 2 to 4 on an expressed tree: every node's properties and environment, and where a `must` guard fails,
    /// its repair.
    pub fn decode(&self, tree: &Tree) -> Result<Decoding, String> {
        self.decoder.decode(&self.entry, tree)
    }

    /// Express `gene` over `layout` (Karva's walk) to its tree.
    pub fn express(&self, layout: &GeneLayout, gene: &[u32]) -> Result<Tree, String> {
        Ok(express(&self.contract, layout, gene, &self.const_len()).map_err(|d| bug(format!("the gene does not express: {d}")))?.tree())
    }

    /// Pass 5: the kernel a tree spells, through the decoder's renderer, the prelude first.
    pub fn program(&self, tree: &Tree) -> Result<WgslProgram, String> {
        self.program_of(&self.decode(tree)?)
    }

    /// [`WgslRows::program`] of a decoding.
    pub fn program_of(&self, d: &Decoding) -> Result<WgslProgram, String> {
        let dc = |s: SymbolId, w: u32| -> Result<String, String> {
            let pool = self.literals(s);
            pool.get(w as usize).cloned().ok_or_else(|| bug(format!("{}'s literal {w} is past its pool of {}", self.name(s), pool.len())))
        };
        let r = self.decoder.render(&self.entry, d, &dc)?;
        let mut text = String::new();
        for p in &r.preludes {
            text.push_str(p);
            if !p.ends_with('\n') {
                text.push('\n');
            }
        }
        text.push_str(&lines(&r.body));
        let (mut substitutions, mut rewrites) = (Vec::new(), Vec::new());
        for f in d.failures() {
            match f {
                Failure::Substituted { path, requirement, .. } => substitutions.push((path, self.tables.guards[requirement].condition.to_string())),
                Failure::Adapter { path, requirement, .. } => rewrites.push((path, self.tables.guards[requirement].condition.to_string())),
                _ => {}
            }
        }
        Ok(WgslProgram { text, nodes: d.rendered_nodes().len(), substitutions, rewrites })
    }

    /// A random gene over `layout` (the typed initialiser's draws: a head cell a function or a terminal of its type at
    /// one half each, a tail cell a terminal, a Dc cell a word below its pool, a Dv cell a word below `DV_WORDS`; the
    /// root's first cell a function). Nothing is masked or redrawn.
    pub fn random_gene(&self, layout: &GeneLayout, rng: &mut crate::futhark_table::GeneRng) -> Vec<u32> {
        let k = &self.contract;
        let const_len = self.const_len();
        let mut gene = vec![0u32; layout.width];
        for s in layout.spans() {
            let (functions, terminals) = (k.functions_of(s.ty), k.terminals_of(s.ty));
            for (i, cell) in gene[s.off..s.off + s.len].iter_mut().enumerate() {
                *cell = match s.domain {
                    Domain::Head => {
                        let force = s.ty == layout.root && i == 0 && !functions.is_empty();
                        if !functions.is_empty() && (terminals.is_empty() || force || rng.below(2) == 0) {
                            functions[rng.below(functions.len())]
                        } else {
                            terminals[rng.below(terminals.len())]
                        }
                    }
                    Domain::Tail => terminals[rng.below(terminals.len())],
                    Domain::Dc => rng.below(const_len.get(&s.ty).copied().unwrap_or(1).max(1) as usize) as u32,
                    Domain::Dv => rng.below(crate::typed_gep::DV_WORDS as usize) as u32,
                };
            }
        }
        gene
    }
}

/// The rendered body broken into lines after each `;`, `{` and `}` (the rows spell statements on one line because a
/// table holds no line break; WGSL reads either). Cosmetic: no WGSL token of the rows holds those characters.
fn lines(body: &str) -> String {
    let mut out = String::with_capacity(body.len() + body.len() / 8);
    for c in body.chars() {
        out.push(c);
        if matches!(c, ';' | '{' | '}') {
            out.push('\n');
        }
    }
    out
}

// ---- THE DEVICE'S RULES (the closed forms a device decoder evaluates), and the host twin --------------------------

/// How one property of a symbol's result is derived, in the closed set the device decodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpRule {
    /// A literal flag or integer.
    Literal(PropertyExpr),
    /// `in<k>.p` copied.
    Copy { k: u32, property: String },
    /// `in0.p and in1.p and ...`, over the listed inputs.
    All { property: String, ks: Vec<u32> },
    /// `(in0.p and in1.uniform) or (in1.p and in0.uniform)` for the flag `p` (`owned`, `local_owned`): the one way a
    /// sum stays injective.
    InjectiveSum { property: String },
    /// `in0.bound + in1.bound`.
    SumBound,
    /// `in0.bound * in1.value`.
    ProductBound,
    /// `min(in0.bound, in1.bound)`.
    MinBound,
}

/// What a down flow of an argument does, in the closed set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownRule {
    /// `env.uniform := env.uniform and in0.uniform`.
    UniformAnd,
    /// `env.work := env.work * in0.value`.
    WorkTimes,
    /// One variable of `ty` introduced, each property copied from `in0.p` or set to a literal.
    Var { ty: Ty, properties: Vec<(String, UpRule)> },
}

/// A guard's condition, in the closed set, with how its repair acts: a substitution (`{const}` or `{in.own}`) or a
/// rewrite of operand `k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardRule {
    /// `in<k>.access = 'read_write'`.
    Access { k: u32 },
    /// `in<k>.<property>`, a flag (`owned`, `local_owned`).
    Flag { k: u32, property: String },
    /// `in<k>.bound <= in<b>.len`, repaired by a clamp of operand k.
    Bound { k: u32, buffer: u32 },
    /// `in<k>.bound <= limit` (a workgroup array's length), repaired by a clamp of operand k.
    BoundLiteral { k: u32, limit: i64 },
    /// `in<k>.width = in<w>.width`, repaired by a fit of operand k.
    Width { k: u32, other: u32 },
    /// `env.work * in0.value <= limit`.
    Work { limit: i64 },
    /// `env.uniform`.
    Uniform,
}

/// One symbol's rules as the device reads them, compiled from its Flow and Guard rows.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceRule {
    pub ups: Vec<(String, UpRule)>,
    pub downs: BTreeMap<u32, Vec<DownRule>>,
    /// In `order`: the guard and how its repairs act.
    pub guards: Vec<(GuardRule, Repairs)>,
}

/// How a guard's repair list acts: `rewrite`: its first repair rewrites that operand in place (a clamp, a fit), and the
/// node then stands; `own`: a `{in.own}` substitute is offered before `{const}`. The decoder skips a substitute the node
/// lacks (no input of its own data type) and takes the next repair, `{const}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Repairs {
    pub rewrite: Option<u32>,
    pub own: bool,
}

/// A node of the twin's decode: what the twin and the decoder must agree on. `bound_known`: the bound is a literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Twin {
    pub symbol: SymbolId,
    /// `Some(k)`: the node was replaced by input `k`; `None` with `dropped`: replaced by its constant.
    pub substituted_by: Option<u32>,
    pub dropped: bool,
    /// The operands rewritten in place (clamped, fitted).
    pub rewritten: Vec<u32>,
    pub uniform: Option<bool>,
    pub owned: Option<bool>,
    pub local_owned: Option<bool>,
    pub bound: Option<i64>,
    pub width: Option<i64>,
    pub kids: Vec<Twin>,
}

fn formula(text: &str) -> PropertyExpr {
    PropertyExpr::parse(text).unwrap_or_else(|e| panic!("our software has a bug: the WGSL rule form {text:?} does not parse: {e}"))
}

impl WgslRows {
    /// THE DEVICE'S RULES, one per symbol id, compiled from the Flow and Guard rows: each row must be one of the closed
    /// forms of [`UpRule`], [`DownRule`] and [`GuardRule`]. A row outside them is "our software has a bug" at compile
    /// time, never a silent approximation.
    pub fn device_rules(&self) -> Result<Vec<DeviceRule>, String> {
        (0..self.contract.n_symbols() as SymbolId).map(|s| self.device_rule(s).map_err(|e| bug(format!("symbol {s} ({}): {e}", self.name(s))))).collect()
    }

    fn up_rule(&self, arity: usize, property: &str, f: &PropertyExpr) -> Result<UpRule, String> {
        let is = |text: &str| f == &formula(text);
        if matches!(f, PropertyExpr::Literal(_) | PropertyExpr::Flag(_)) {
            return Ok(UpRule::Literal(f.clone()));
        }
        if let PropertyExpr::Input { position, property: p } = f {
            return Ok(UpRule::Copy { k: *position, property: p.clone() });
        }
        let all: Vec<u32> = (0..arity as u32).collect();
        for n in 2..=arity {
            let text = all[..n].iter().map(|k| format!("in{k}.{property}")).collect::<Vec<_>>().join(" and ");
            if is(&text) {
                return Ok(UpRule::All { property: property.to_string(), ks: all[..n].to_vec() });
            }
        }
        if (property == "owned" || property == "local_owned") && is(&format!("(in0.{property} and in1.uniform) or (in1.{property} and in0.uniform)")) {
            return Ok(UpRule::InjectiveSum { property: property.to_string() });
        }
        if property == "bound" {
            if is("in0.bound + in1.bound") {
                return Ok(UpRule::SumBound);
            }
            if is("in0.bound * in1.value") {
                return Ok(UpRule::ProductBound);
            }
            if is("min(in0.bound, in1.bound)") {
                return Ok(UpRule::MinBound);
            }
        }
        Err(format!("the flow {property} := {f} is outside the device's forms"))
    }

    fn device_rule(&self, s: SymbolId) -> Result<DeviceRule, String> {
        let id = i64::from(s);
        let arity = self.contract.signature(s).len();
        let mut rule = DeviceRule::default();
        let mut vars: BTreeMap<u32, (Ty, Vec<(String, UpRule)>)> = BTreeMap::new();
        for f in self.tables.flows.iter().filter(|f| f.owner == Owner::Symbol(id)) {
            match (&f.name, f.way, f.at) {
                (FlowName::Property(p), Way::Up, Place::Out) => rule.ups.push((p.clone(), self.up_rule(arity, p, &f.formula)?)),
                (FlowName::Env(a), Way::Down, Place::In(k)) => {
                    let d = if a == "uniform" && f.formula == formula("env.uniform and in0.uniform") {
                        DownRule::UniformAnd
                    } else if a == "work" && f.formula == formula("env.work * in0.value") {
                        DownRule::WorkTimes
                    } else {
                        return Err(format!("the down flow env.{a} := {} is outside the device's forms", f.formula));
                    };
                    rule.downs.entry(k).or_default().push(d);
                }
                (FlowName::Var { ty, group: 0 }, Way::Down, Place::In(k)) => {
                    if f.formula != PropertyExpr::Literal(1) {
                        return Err("a binding of other than one variable".into());
                    }
                    vars.insert(k, (*ty, Vec::new()));
                }
                (FlowName::VarProperty { ty, group: 0, property }, Way::Down, Place::In(k)) => {
                    let (t, props) = vars.get_mut(&k).ok_or("a variable's property before its binding")?;
                    if t != ty {
                        return Err("a variable's property of another type".into());
                    }
                    props.push((property.clone(), self.up_rule(arity, property, &f.formula)?));
                }
                (FlowName::VarSpell { .. }, Way::Down, Place::In(_)) => {}
                _ => return Err(format!("a flow outside the device's forms: {:?} {:?} {:?}", f.name, f.way, f.at)),
            }
        }
        for (k, (ty, properties)) in vars {
            rule.downs.entry(k).or_default().push(DownRule::Var { ty, properties });
        }
        let mut guards: Vec<_> = self.tables.guards.iter().filter(|g| g.owner == Owner::Symbol(id)).collect();
        guards.sort_by_key(|g| g.order);
        for g in guards {
            if g.strength != Strength::Must {
                return Err(format!("guard {}: not a must guard", g.order));
            }
            let first = g.repairs.first().ok_or_else(|| format!("guard {}: no repair", g.order))?;
            let rewrite = match first.site {
                Site::Node => None,
                Site::In(k) => Some(k),
                Site::Root => return Err(format!("guard {}: a root repair on a symbol", g.order)),
            };
            for r in g.repairs.iter().skip(usize::from(rewrite.is_some())) {
                if r.site != Site::Node || !matches!(r.template.as_str(), "{const}" | "{in.own}") || !r.unconditional() {
                    return Err(format!("guard {}: the repair {} is not an unconditional {{const}} or {{in.own}} at the node", g.order, r.template));
                }
            }
            let own_input = self.contract.signature(s).contains(&self.contract.output(s));
            if g.repairs.last().is_none_or(|r| !(r.template == "{const}" || (r.template == "{in.own}" && own_input))) {
                return Err(format!("guard {}: its repairs do not end in a substitute that always applies ({{const}}, or {{in.own}} with an input of the node's own data type)", g.order));
            }
            let repairs = Repairs { rewrite, own: g.repairs.iter().any(|r| r.template == "{in.own}") };
            let rule_of = match &g.condition {
                Condition::Compare { relation: Relation::Eq, left: PropertyExpr::Input { position: k, property }, right: PropertyExpr::Tag(t) } if property == "access" && t == "read_write" => GuardRule::Access { k: *k },
                Condition::Flag(PropertyExpr::Input { position: k, property }) if property == "owned" || property == "local_owned" => GuardRule::Flag { k: *k, property: property.clone() },
                Condition::Flag(PropertyExpr::Env(a)) if a == "uniform" => GuardRule::Uniform,
                Condition::Compare { relation: Relation::Le, left: PropertyExpr::Input { position: k, property: p }, right: PropertyExpr::Input { position: b, property: q } } if p == "bound" && q == "len" => GuardRule::Bound { k: *k, buffer: *b },
                Condition::Compare { relation: Relation::Le, left: PropertyExpr::Input { position: k, property: p }, right: PropertyExpr::Literal(limit) } if p == "bound" => GuardRule::BoundLiteral { k: *k, limit: *limit },
                Condition::Compare { relation: Relation::Eq, left: PropertyExpr::Input { position: k, property: p }, right: PropertyExpr::Input { position: w, property: q } } if p == "width" && q == "width" => GuardRule::Width { k: *k, other: *w },
                Condition::Compare { relation: Relation::Le, left, right: PropertyExpr::Literal(limit) } if *left == formula("env.work * in0.value") => GuardRule::Work { limit: *limit },
                c => return Err(format!("guard {}: the condition {c} is outside the device's forms", g.order)),
            };
            rule.guards.push((rule_of, repairs));
        }
        Ok(rule)
    }

    /// THE HOST TWIN of a device decode: the rules of [`WgslRows::device_rules`] evaluated over `tree` in the entry's
    /// root environment (its parameters' literal properties), with the same reading of `var.T` (the index word modulo
    /// the variables of T in scope, outermost first) and the same repair order as the decoder. The result is what a
    /// device decode would carry back; [`WgslRows::twin_of`] reads the same facts from the decoder's decoding.
    pub fn twin_decode(&self, rules: &[DeviceRule], tree: &Tree) -> Result<Twin, String> {
        #[derive(Clone)]
        struct Var {
            ty: Ty,
            props: Props,
        }
        type Props = BTreeMap<String, PropertyExpr>;
        #[derive(Clone)]
        struct Env {
            uniform: bool,
            work: i64,
            vars: Vec<Var>,
        }
        fn flag(p: &Props, name: &str) -> Option<bool> {
            match p.get(name) {
                Some(PropertyExpr::Flag(b)) => Some(*b),
                _ => None,
            }
        }
        fn int(p: &Props, name: &str) -> Option<i64> {
            match p.get(name) {
                Some(PropertyExpr::Literal(v)) => Some(*v),
                _ => None,
            }
        }
        fn eval(rule: &UpRule, kids: &[Props]) -> PropertyExpr {
            let unknown = PropertyExpr::Unknown;
            let f = |k: u32, p: &str| flag(&kids[k as usize], p);
            let i = |k: u32, p: &str| int(&kids[k as usize], p);
            match rule {
                UpRule::Literal(l) => l.clone(),
                UpRule::Copy { k, property } => kids[*k as usize].get(property).cloned().unwrap_or(unknown),
                UpRule::All { property, ks } => {
                    let vs: Vec<Option<bool>> = ks.iter().map(|k| f(*k, property)).collect();
                    if vs.contains(&Some(false)) {
                        PropertyExpr::Flag(false)
                    } else if vs.iter().all(|v| *v == Some(true)) {
                        PropertyExpr::Flag(true)
                    } else {
                        unknown
                    }
                }
                UpRule::InjectiveSum { property } => {
                    let a = (f(0, property), f(1, "uniform"));
                    let b = (f(1, property), f(0, "uniform"));
                    let and = |(x, y): (Option<bool>, Option<bool>)| match (x, y) {
                        (Some(false), _) | (_, Some(false)) => Some(false),
                        (Some(true), Some(true)) => Some(true),
                        _ => None,
                    };
                    match (and(a), and(b)) {
                        (Some(true), _) | (_, Some(true)) => PropertyExpr::Flag(true),
                        (Some(false), Some(false)) => PropertyExpr::Flag(false),
                        _ => unknown,
                    }
                }
                UpRule::SumBound => i(0, "bound").zip(i(1, "bound")).map_or(unknown, |(a, b)| PropertyExpr::Literal(a + b)),
                UpRule::ProductBound => i(0, "bound").zip(i(1, "value")).map_or(unknown, |(a, b)| PropertyExpr::Literal(a * b)),
                UpRule::MinBound => i(0, "bound").zip(i(1, "bound")).map_or(unknown, |(a, b)| PropertyExpr::Literal(a.min(b))),
            }
        }
        fn go(k: &WgslRows, rules: &[DeviceRule], t: &Tree, env: &Env) -> Result<(Twin, Props, Ty), String> {
            let s = t.symbol;
            let rule = rules.get(s as usize).ok_or_else(|| bug(format!("symbol {s} has no rule")))?;
            let out_ty = k.contract.output(s);
            let row = k.tables.symbol(i64::from(s)).ok_or_else(|| bug(format!("symbol {s} has no row")))?;
            // a variable: its properties are the chosen variable's (a Dc constant is a function of no inputs: its flows')
            if row.role == Role::Variable {
                let word = t.const_index.ok_or_else(|| bug("a variable without a word".into()))?;
                // the environment's order (`typed_environment::in_scope`): the innermost first, the entry's parameters last
                let in_scope: Vec<&Var> = env.vars.iter().rev().filter(|v| v.ty == out_ty).collect();
                let props = if in_scope.is_empty() { constant_props(k, out_ty) } else { in_scope[word as usize % in_scope.len()].props.clone() };
                return Ok((leaf(s, &props), props, out_ty));
            }
            // pass 2: the inputs, each in its environment (its argument's down flows read the inputs before it)
            let mut kids: Vec<Option<(Twin, Props, Ty)>> = vec![None; t.kids.len()];
            let order = k.tables.decode_order(i64::from(s))?;
            for pos in order {
                let mut e = env.clone();
                let done: Vec<Props> = kids.iter().map(|c| c.as_ref().map(|(_, p, _)| p.clone()).unwrap_or_default()).collect();
                for d in rule.downs.get(&pos).map(Vec::as_slice).unwrap_or(&[]) {
                    match d {
                        DownRule::UniformAnd => e.uniform = e.uniform && flag(&done[0], "uniform") == Some(true),
                        DownRule::WorkTimes => e.work = e.work.saturating_mul(int(&done[0], "value").unwrap_or(i64::MAX / 4)),
                        DownRule::Var { ty, properties } => {
                            let props: Props = properties.iter().map(|(p, r)| (p.clone(), eval(r, &done))).collect();
                            e.vars.push(Var { ty: *ty, props });
                        }
                    }
                }
                let child = &t.kids[pos as usize];
                kids[pos as usize] = Some(go(k, rules, child, &e)?);
            }
            let kids: Vec<(Twin, Props, Ty)> = kids.into_iter().map(|c| c.ok_or_else(|| bug("an input not decoded".into()))).collect::<Result<_, _>>()?;
            let kid_props: Vec<Props> = kids.iter().map(|(_, p, _)| p.clone()).collect();
            // pass 3: the up flows
            let mut props: Props = rule.ups.iter().map(|(p, r)| (p.clone(), eval(r, &kid_props))).collect();
            for p in k.tables.properties_of(out_ty) {
                props.entry(p.name.clone()).or_insert(PropertyExpr::Unknown);
            }
            // pass 4: the guards in order; the first that fails repairs (a rewrite keeps checking; a substitution ends it)
            let mut twin = Twin { symbol: s, substituted_by: None, dropped: false, rewritten: Vec::new(), uniform: None, owned: None, local_owned: None, bound: None, width: None, kids: kids.into_iter().map(|(tw, _, _)| tw).collect() };
            for (g, repairs) in &rule.guards {
                let holds = match g {
                    GuardRule::Access { k: a } => kid_props[*a as usize].get("access") == Some(&PropertyExpr::Tag("read_write".into())),
                    GuardRule::Flag { k: a, property } => flag(&kid_props[*a as usize], property) == Some(true),
                    GuardRule::Bound { k: a, buffer } => int(&kid_props[*a as usize], "bound").zip(int(&kid_props[*buffer as usize], "len")).is_some_and(|(b, l)| b <= l),
                    GuardRule::BoundLiteral { k: a, limit } => int(&kid_props[*a as usize], "bound").is_some_and(|b| b <= *limit),
                    GuardRule::Width { k: a, other } => int(&kid_props[*a as usize], "width").zip(int(&kid_props[*other as usize], "width")).is_some_and(|(x, y)| x == y),
                    GuardRule::Work { limit } => int(&kid_props[0], "value").is_some_and(|v| env.work.saturating_mul(v) <= *limit),
                    GuardRule::Uniform => env.uniform,
                };
                if holds {
                    continue;
                }
                match repairs.rewrite {
                    Some(a) => twin.rewritten.push(a),
                    None => {
                        let own = k.contract.signature(s).iter().position(|t| *t == out_ty);
                        match own.filter(|_| repairs.own) {
                            Some(kk) => {
                                twin.substituted_by = Some(kk as u32);
                                props = kid_props[kk].clone();
                            }
                            None => {
                                twin.dropped = true;
                                props = constant_props(k, out_ty);
                            }
                        }
                        break;
                    }
                }
            }
            twin.uniform = flag(&props, "uniform");
            twin.owned = flag(&props, "owned");
            twin.local_owned = flag(&props, "local_owned");
            twin.bound = int(&props, "bound");
            twin.width = int(&props, "width");
            Ok((twin, props, out_ty))
        }
        /// A data type's constant's properties, every property it carries present (`?` where the constant gives none).
        fn constant_props(k: &WgslRows, ty: Ty) -> Props {
            let mut props: Props = k.tables.data_type(ty).and_then(|d| d.constant.as_ref()).map(|c| c.properties.iter().cloned().collect()).unwrap_or_default();
            for p in k.tables.properties_of(ty) {
                props.entry(p.name.clone()).or_insert(PropertyExpr::Unknown);
            }
            props
        }
        fn leaf(s: SymbolId, props: &Props) -> Twin {
            Twin { symbol: s, substituted_by: None, dropped: false, rewritten: Vec::new(), uniform: flag(props, "uniform"), owned: flag(props, "owned"), local_owned: flag(props, "local_owned"), bound: int(props, "bound"), width: int(props, "width"), kids: Vec::new() }
        }
        let root_env = self.decoder.root_environment(&self.entry);
        let env = Env {
            uniform: matches!(root_env.attribute("uniform"), Some(PropertyExpr::Flag(true))),
            work: match root_env.attribute("work") {
                Some(PropertyExpr::Literal(w)) => *w,
                _ => 1,
            },
            vars: root_env.variables.iter().map(|v| Var { ty: v.data_type, props: v.properties.clone() }).collect(),
        };
        Ok(go(self, rules, tree, &env)?.0)
    }

    /// The decoder's decoding read as a [`Twin`]: the same facts, so the two can be compared.
    pub fn twin_of(&self, d: &Decoding) -> Twin {
        fn go(k: &WgslRows, n: &Node) -> Twin {
            let (substituted_by, dropped) = match n.substituted {
                Some(Substituted { by: Substitution::Input(k), .. }) => (Some(k), false),
                Some(Substituted { by: Substitution::Constant, .. }) => (None, true),
                None => (None, false),
            };
            let mut rewritten: Vec<u32> = n
                .outcomes
                .iter()
                .filter(|o| !o.holds)
                .filter_map(|o| match k.tables.guards[o.requirement].repairs.get(o.repair?)?.site {
                    Site::In(a) => Some(a),
                    Site::Node | Site::Root => None,
                })
                .collect();
            rewritten.sort_unstable();
            rewritten.dedup();
            let flag = |p: &str| match n.properties.get(p) {
                Some(PropertyExpr::Flag(b)) => Some(*b),
                _ => None,
            };
            let int = |p: &str| match n.properties.get(p) {
                Some(PropertyExpr::Literal(v)) => Some(*v),
                _ => None,
            };
            let kids = if matches!(n.choice, Some(Choice::Dc(_))) { Vec::new() } else { n.kids.iter().map(|c| go(k, c)).collect() };
            Twin { symbol: n.symbol, substituted_by, dropped, rewritten, uniform: flag("uniform"), owned: flag("owned"), local_owned: flag("local_owned"), bound: int("bound"), width: int("width"), kids }
        }
        go(self, &d.root)
    }
}

pub mod share;

#[cfg(test)]
mod tests;
