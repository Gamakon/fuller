//! THE DECODER, passes 2 to 5, on the host and kingdom-free (design `docs/DESIGN_futhark_types_and_properties.md`
//! section 4), and the entry points of the second engine, the cost (section 4, "The second engine").
//!
//! Pass 1 (`typed_gep::express`) gives the tree of symbols. Then, in one walk, reading only the Flow rows and the `must`
//! Guard rows of the tables (`crate::geneframe`, section 3.3):
//!
//! 2. **The environment, top-down (down flows).** The root's is the entry's parameters (raw values and alternative
//!    forms, each with its properties and their sources) and the environment attributes at their catalogue roots,
//!    overridden by the entry's down flows. A child's is its parent's with the down flows of the argument it fills: a
//!    `var.T` flow introduces its variables (their properties by `var.T.p`, their names by `var.T.spell`) on top of T's
//!    stack, an `env.a` flow sets an attribute.
//! 3. **Variables (binder-relative, design 3.6).** A `var.T` takes its index word `w` (`Tree::const_index`, a Dv word)
//!    and becomes the `(w mod c)`-th of the `c` variables of T in scope counted from the INNERMOST (0 is the innermost
//!    binder's variable; the entry's parameters are outermost), or T's constant when `c = 0`. [`in_scope`] is THE
//!    order: the only one the renderer, the interpreter, the reader and the e-graph use.
//! 4. **Properties, bottom-up (up flows), and the `must` guards,** checked three-valued in their `order` with every
//!    measured value read as `?` (a measurement never makes a `must` guard hold). A guard that does not hold takes its
//!    first repair whose guard (read in its site's environment) and `when` hold: a wrap or an operand rewrite renders
//!    the repair's template; a substitute (`{in<k>}`, `{in.own}`, `{const}`) REPLACES the node, which takes the
//!    substitute's properties, and none of the node's other repairs render ([`Substituted`]). The renderer, the
//!    interpreter, the cost, the reader and the e-graph read this annotation and never decide it again.
//! 5. **Render** ([`Decoder::render`]): the spellings and the repairs' templates by the one template engine
//!    ([`template`]), each used symbol's prelude once, each alternative form bound once at the head. A node's repairs
//!    render in their guards' `order`: first every operand rewrite (a repair at `in<k>`, or one `{self(k := ..)}` at the
//!    node), then every wrap of the node ([`Decoder::steps`]).
//!
//! `advise` guards are never read here: they are the cost's ([`Decoder::advice`]), with every adapter in the program,
//! placed by evolution or called by a repair's template ([`Decoder::rescues`], [`Decoder::calls`]). The decoder
//! repairs, the cost charges. The interpreter and the cost EXECUTE the repairs' templates ([`Decoder::exec`]): an
//! adapter means what its symbol means, wherever it is written.
//!
//! The entry is checked once per decoder (code review finding 16), the first time it is decoded.
//!
//! The **resolved tree** ([`Decoding::resolved`]) names every variable by its binder's identity (the binding depth and
//! the ordinal), never by its index word: the only form any consumer compares or hashes. A SUBTREE is compared by its
//! key ([`Decoding::subtree_key`]), which adds what the subtree decodes by that is outside it: its environment's
//! attributes and its free variables' binders, data types and properties.
//!
//! Deterministic in the gene and the rows alone: every map is a `BTreeMap`, every list in table or position order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

pub use crate::geneframe::Entry;
use crate::geneframe::{Condition, Constant, FlowName, Guard, KingdomTables, Owner, PropertyExpr, Reference, Role, Site, SizeBounds, Strength, Truth, Ty};
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::typed_gep::Tree;
use template::{calls_of, forms_of, parse_template, rewrites_of, substitute_of, Exec, Fresh, Lexicon, Scope, Substitute, Template};

pub mod template;

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

/// A variable's identity: the binding depth at which it was bound (0 for the entry's parameters, then one more for each
/// argument on the path that introduces any variable) and its ordinal among the variables introduced there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Binder {
    pub depth: u32,
    pub ordinal: u32,
}

/// A variable in scope: its identity, its name (an entry parameter's, or its `var.T.spell` flow's; `None` where the
/// kingdom has no spelling for it), the stem its spelling's `{nth}` numbers it among (`None` for an entry's parameter),
/// its data type, its properties (every property its data type carries; `?` where unknown) and the names of those whose
/// source is measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub binder: Binder,
    pub name: Option<String>,
    pub stem: Option<String>,
    pub data_type: Ty,
    pub properties: BTreeMap<String, PropertyExpr>,
    pub measured: BTreeSet<String>,
}

/// What a node may use: the variables in binding order (outermost first), its binding depth, its environment
/// attributes (`trips`, `uniform`, ...) and the names of the attributes whose value is measured.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Environment {
    pub variables: Vec<Variable>,
    pub depth: u32,
    pub attributes: BTreeMap<String, PropertyExpr>,
    pub measured: BTreeSet<String>,
}

/// THE ENVIRONMENT'S ORDER (design 3.6): the variables of data type `ty` in scope, the INNERMOST first (the last binder's
/// variables, the highest ordinal first), the entry's parameters last. A `var.T`'s word `w` names the `(w mod c)`-th of
/// them ([`Environment::choose`]); a reader writes a variable back as its place here ([`Environment::index_of`]). Every
/// consumer reads this one function.
pub fn in_scope(env: &Environment, ty: Ty) -> Vec<&Variable> {
    env.variables.iter().rev().filter(|v| v.data_type == ty).collect()
}

impl Environment {
    /// The variable a `var.T` of index word `word` names, `None` when no variable of `ty` is in scope.
    pub fn choose(&self, ty: Ty, word: u32) -> Option<&Variable> {
        let vs = in_scope(self, ty);
        if vs.is_empty() {
            None
        } else {
            Some(vs[word as usize % vs.len()])
        }
    }

    /// The index word that names `binder` among the variables of `ty` in scope (the least one; any word congruent to it
    /// modulo their number names it too).
    pub fn index_of(&self, ty: Ty, binder: Binder) -> Option<u32> {
        in_scope(self, ty).iter().position(|v| v.binder == binder).map(|i| i as u32)
    }

    /// An attribute's value, `None` when the kingdom has no such attribute.
    pub fn attribute(&self, name: &str) -> Option<&PropertyExpr> {
        self.attributes.get(name)
    }
}

/// What a node's word reads as: a `?_T`'s Dc index, a `var.T`'s variable, or (no variable of its type in scope) its
/// type's constant. In a [`SubtreeKey`] only, a variable bound outside the subtree is `Free(j)`, the `j`-th of the key's
/// free variables, and one bound inside it has its depth counted from the subtree's root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Choice {
    Dc(u32),
    Variable(Binder),
    Constant,
    Free(u32),
}

/// One `must` guard checked at one node: its row (an index into the tables' guards, or the entry's for a root guard),
/// its condition with every reference read (measured values as `?`), whether it holds, and, where it does not, the
/// repair taken (its index in the guard's repairs) and the adapter symbols that repair's template may call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub requirement: usize,
    pub condition: Condition,
    pub holds: bool,
    pub repair: Option<usize>,
    pub calls: Vec<String>,
}

static UNKNOWN: PropertyExpr = PropertyExpr::Unknown;

impl Outcome {
    /// The left side of the guard's comparison, as read (`?` for a guard that is no single comparison).
    pub fn subject(&self) -> &PropertyExpr {
        self.condition.comparison().map_or(&UNKNOWN, |(_, a, _)| a)
    }

    /// The right side of the guard's comparison, as read.
    pub fn against(&self) -> &PropertyExpr {
        self.condition.comparison().map_or(&UNKNOWN, |(_, _, b)| b)
    }

    /// True when the guard failed and its repair may call `adapter`.
    pub fn repaired_by(&self, adapter: &str) -> bool {
        !self.holds && self.calls.iter().any(|c| c == adapter)
    }
}

/// What a replaced node renders as: its input at this position, or its data type's constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Substitution {
    Input(u32),
    Constant,
}

/// A node replaced by a substitute repair: what it renders as, and the guard whose repair replaced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Substituted {
    pub by: Substitution,
    pub requirement: usize,
}

/// One decoded node: its symbol and data type, its environment, what its word reads as, its properties (every property
/// its data type carries; a replaced node's are its substitute's) and those measured, each of its `must` guards' outcome
/// in table order, its replacement, and its inputs in position order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub symbol: SymbolId,
    pub data_type: Ty,
    pub environment: Environment,
    pub choice: Option<Choice>,
    pub properties: BTreeMap<String, PropertyExpr>,
    pub measured: BTreeSet<String>,
    pub outcomes: Vec<Outcome>,
    pub substituted: Option<Substituted>,
    pub kids: Vec<Node>,
}

/// The tree with every variable by its binder's identity: the form two WHOLE trees of one entry are compared or hashed
/// by. A subtree is compared by its [`SubtreeKey`] instead.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedTree {
    pub symbol: SymbolId,
    pub choice: Option<Choice>,
    pub kids: Vec<ResolvedTree>,
}

/// A variable a subtree uses but does not bind: its binder, data type, properties and measured properties.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FreeVariable {
    pub binder: Binder,
    pub data_type: Ty,
    pub properties: BTreeMap<String, PropertyExpr>,
    pub measured: BTreeSet<String>,
}

/// THE KEY OF A SUBTREE: its resolved tree, with each variable bound outside it as `Choice::Free(j)` and each bound
/// inside at its depth below the subtree's root; the attributes of its root's environment; and its free variables in
/// order of first use. Every repair inside a subtree is decided by these alone, so two subtrees of one key render with
/// the same repairs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubtreeKey {
    pub tree: ResolvedTree,
    pub attributes: BTreeMap<String, PropertyExpr>,
    pub free: Vec<FreeVariable>,
}

/// A `must` guard that fails where the program is rendered (a node under a replacement that drops it is not rendered):
/// the node's path of argument positions from the root, and what renders there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// Guard `requirement` fails at the node; its repair may call the adapters `calls`.
    Adapter { path: Vec<u32>, requirement: usize, calls: Vec<String> },
    /// The node is replaced by a substitute repair.
    Substituted { path: Vec<u32>, by: Substitution, requirement: usize },
    /// The entry's guard `requirement` fails on the result; its repair may call the adapters `calls`.
    Root { requirement: usize, calls: Vec<String> },
}

/// An `advise` guard that does not hold where the program is rendered: what the cost charges (`None` path: the
/// entry's, on the result).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Advice {
    pub path: Option<Vec<u32>>,
    pub guard: usize,
    pub truth: Truth,
}

/// An adapter in the rendered program: evolution placed it (an adapter symbol of the tree) or a repair wrote it. The
/// cost charges both alike.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rescue {
    pub path: Option<Vec<u32>>,
    pub adapter: String,
    pub placed: bool,
}

/// A decoded gene: the annotated tree and the outcomes of the entry's `must` guards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoding {
    pub root: Node,
    pub root_outcomes: Vec<Outcome>,
}

impl Decoding {
    /// The resolved tree ([`ResolvedTree`]).
    pub fn resolved(&self) -> ResolvedTree {
        fn go(n: &Node) -> ResolvedTree {
            ResolvedTree { symbol: n.symbol, choice: n.choice, kids: n.kids.iter().map(go).collect() }
        }
        go(&self.root)
    }

    /// The key of the subtree at `path` ([`SubtreeKey`]), `None` when there is no node there.
    pub fn subtree_key(&self, path: &[u32]) -> Option<SubtreeKey> {
        fn go(n: &Node, root_depth: u32, free: &mut Vec<FreeVariable>) -> ResolvedTree {
            let choice = match n.choice {
                Some(Choice::Variable(b)) if b.depth > root_depth => Some(Choice::Variable(Binder { depth: b.depth - root_depth, ordinal: b.ordinal })),
                Some(Choice::Variable(b)) => {
                    let j = free.iter().position(|f| f.binder == b).unwrap_or_else(|| {
                        free.push(FreeVariable { binder: b, data_type: n.data_type, properties: n.properties.clone(), measured: n.measured.clone() });
                        free.len() - 1
                    });
                    Some(Choice::Free(j as u32))
                }
                c => c,
            };
            ResolvedTree { symbol: n.symbol, choice, kids: n.kids.iter().map(|k| go(k, root_depth, free)).collect() }
        }
        let n = self.node(path)?;
        let mut free = Vec::new();
        let tree = go(n, n.environment.depth, &mut free);
        Some(SubtreeKey { tree, attributes: n.environment.attributes.clone(), free })
    }

    /// Every failing `must` guard and every replacement of the rendered program, the entry's first, then the nodes in
    /// pre-order. A replaced node lists its replacement alone, and only the input it renders as is walked below.
    pub fn failures(&self) -> Vec<Failure> {
        fn go(n: &Node, path: &mut Vec<u32>, out: &mut Vec<Failure>) {
            if let Some(s) = n.substituted {
                out.push(Failure::Substituted { path: path.clone(), by: s.by, requirement: s.requirement });
                if let Substitution::Input(k) = s.by {
                    path.push(k);
                    go(&n.kids[k as usize], path, out);
                    path.pop();
                }
                return;
            }
            for o in n.outcomes.iter().filter(|o| !o.holds) {
                out.push(Failure::Adapter { path: path.clone(), requirement: o.requirement, calls: o.calls.clone() });
            }
            for (k, kid) in n.kids.iter().enumerate() {
                path.push(k as u32);
                go(kid, path, out);
                path.pop();
            }
        }
        let mut out: Vec<Failure> = self.root_outcomes.iter().filter(|o| !o.holds).map(|o| Failure::Root { requirement: o.requirement, calls: o.calls.clone() }).collect();
        go(&self.root, &mut Vec::new(), &mut out);
        out
    }

    /// The node at `path` (argument positions from the root).
    pub fn node(&self, path: &[u32]) -> Option<&Node> {
        let mut n = &self.root;
        for k in path {
            n = n.kids.get(*k as usize)?;
        }
        Some(n)
    }

    /// The nodes that render, in pre-order with their paths (a replaced node's dropped inputs left out).
    pub fn rendered_nodes(&self) -> Vec<(Vec<u32>, &Node)> {
        fn go<'a>(n: &'a Node, path: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, &'a Node)>) {
            if let Some(Substituted { by, .. }) = n.substituted {
                if let Substitution::Input(k) = by {
                    path.push(k);
                    go(&n.kids[k as usize], path, out);
                    path.pop();
                }
                return;
            }
            out.push((path.clone(), n));
            for (k, kid) in n.kids.iter().enumerate() {
                path.push(k as u32);
                go(kid, path, out);
                path.pop();
            }
        }
        let mut out = Vec::new();
        go(&self.root, &mut Vec::new(), &mut out);
        out
    }
}

/// One down flow of an argument, compiled.
#[derive(Debug, Clone)]
enum Down {
    Env(String, PropertyExpr),
    /// A group of variables: their data type, count, properties and spelling template.
    Vars { ty: Ty, count: PropertyExpr, properties: Vec<(String, PropertyExpr)>, spell: Option<Template> },
}

/// What the decoder reads of one symbol, compiled once (design 3.5, the compiled rules).
#[derive(Debug, Clone)]
struct SymbolInfo {
    inputs: Vec<Ty>,
    output: Ty,
    role: Role,
    /// The up flows, in an order without cycles.
    ups: Vec<(String, PropertyExpr)>,
    /// The `must` guards' indices, in their `order`.
    musts: Vec<usize>,
    /// The `advise` guards' indices, in their `order` (the cost's).
    advise: Vec<usize>,
    down: BTreeMap<u32, Vec<Down>>,
    order: Vec<u32>,
    spellings: BTreeMap<String, Template>,
    head: Option<u32>,
}

/// The decoder of one kingdom: its tables and its contract, whose symbol `s` is the tables' symbol `s`. It OWNS them
/// through `Arc`, so a caller may keep it as long as it likes and share the tables with it at no copy.
#[derive(Debug, Clone)]
pub struct Decoder {
    tables: Arc<KingdomTables>,
    contract: Arc<KingdomContract>,
    symbols: Vec<SymbolInfo>,
    properties_of: BTreeMap<Ty, Vec<String>>,
    constants: BTreeMap<Ty, Option<Constant>>,
    attributes: Vec<String>,
    bounds: SizeBounds,
    /// Every guard's repairs' templates, parsed: `templates[g][r]`.
    templates: Vec<Vec<Template>>,
    lexicon: Lexicon,
    by_name: BTreeMap<String, SymbolId>,
    /// The entries already checked ([`Decoder::check_entry`]), so each is checked once (finding 16).
    checked: Arc<Mutex<Vec<Entry>>>,
}

/// The value of a reference at a node, reading measured values as `?` when `must`.
struct Values<'a> {
    kids: &'a [Option<Node>],
    out: &'a BTreeMap<String, PropertyExpr>,
    out_measured: &'a BTreeSet<String>,
    env: &'a Environment,
    must: bool,
}

impl Values<'_> {
    fn read(&self, r: &Reference) -> Option<PropertyExpr> {
        let hide = |measured: bool, v: Option<&PropertyExpr>| if measured && self.must { Some(PropertyExpr::Unknown) } else { Some(v.cloned().unwrap_or(PropertyExpr::Unknown)) };
        match r {
            Reference::Input { position, property } => match self.kids.get(*position as usize).and_then(Option::as_ref) {
                Some(n) => hide(n.measured.contains(*property), n.properties.get(*property)),
                None => Some(PropertyExpr::Unknown),
            },
            Reference::Output(p) => hide(self.out_measured.contains(*p), self.out.get(*p)),
            Reference::Env(a) => hide(self.env.measured.contains(*a), self.env.attributes.get(*a)),
            Reference::Index(_) | Reference::Size(_) => None,
        }
    }

    /// True when the formula reads any measured value.
    fn measured(&self, e: &PropertyExpr) -> bool {
        e.references().iter().any(|r| match r {
            Reference::Input { position, property } => self.kids.get(*position as usize).and_then(Option::as_ref).is_some_and(|n| n.measured.contains(*property)),
            Reference::Output(p) => self.out_measured.contains(*p),
            Reference::Env(a) => self.env.measured.contains(*a),
            Reference::Index(_) | Reference::Size(_) => false,
        })
    }
}

impl Decoder {
    /// The decoder of `tables` and `contract`. Every defect, each "our software has a bug: ...": the tables' own check,
    /// a contract symbol whose inputs, output or role the tables do not give it.
    pub fn new(tables: &KingdomTables, contract: &KingdomContract) -> Result<Decoder, Vec<String>> {
        Decoder::from_arcs(Arc::new(tables.clone()), Arc::new(contract.clone()))
    }

    /// [`Decoder::new`] over shared tables and contract, kept alive by the decoder's own handles.
    pub fn from_arcs(tables: Arc<KingdomTables>, contract: Arc<KingdomContract>) -> Result<Decoder, Vec<String>> {
        tables.check()?;
        let k = &tables.kingdom;
        let mut d: Vec<String> = Vec::new();
        let mut symbols = Vec::with_capacity(contract.n_symbols());
        for s in 0..contract.n_symbols() as SymbolId {
            let id = s as i64;
            let (inputs, output) = (tables.inputs(id), tables.output(id));
            let Some(row) = tables.symbol(id) else {
                d.push(bug(format!("kingdom {k}: contract symbol {s} ({}) is not a symbol of the tables", contract.name(s))));
                continue;
            };
            if output != Some(contract.output(s)) || inputs != contract.signature(s) {
                d.push(bug(format!("kingdom {k}: contract symbol {s} ({}) takes {:?} to {:?}, but the tables' symbol {id} takes {inputs:?} to {output:?}", contract.name(s), contract.signature(s), contract.output(s))));
                continue;
            }
            let role = if contract.is_constant(s) {
                Role::Constant
            } else if contract.is_variable(s) {
                Role::Variable
            } else {
                Role::Function
            };
            if role != row.role {
                d.push(bug(format!("kingdom {k}: contract symbol {s} ({}) is a {}, but the tables' symbol is a {}", contract.name(s), role.name(), row.role.name())));
                continue;
            }
            let ups = match tables.up_order(id) {
                Ok(order) => order
                    .into_iter()
                    .filter_map(|i| match &tables.flows[i].name {
                        FlowName::Property(p) => Some((p.clone(), tables.flows[i].formula.clone())),
                        _ => None,
                    })
                    .collect(),
                Err(e) => return Err(vec![e]),
            };
            let mut guards: Vec<(u32, usize)> = tables.guards.iter().enumerate().filter(|(_, g)| g.owner == Owner::Symbol(id)).map(|(i, g)| (g.order, i)).collect();
            guards.sort();
            let musts = guards.iter().filter(|(_, i)| tables.guards[*i].strength == Strength::Must).map(|(_, i)| *i).collect();
            let advise = guards.iter().filter(|(_, i)| tables.guards[*i].strength == Strength::Advise).map(|(_, i)| *i).collect();
            let mut down: BTreeMap<u32, Vec<Down>> = BTreeMap::new();
            for (pos, _) in inputs.iter().enumerate() {
                let pos = pos as u32;
                let flows = tables.down_flows(id, pos);
                let mut list = Vec::new();
                for f in &flows {
                    match &f.name {
                        FlowName::Env(a) => list.push(Down::Env(a.clone(), f.formula.clone())),
                        FlowName::Var { ty, group } => {
                            let properties = flows.iter().filter_map(|g| match &g.name {
                                FlowName::VarProperty { ty: t2, group: g2, property } if t2 == ty && g2 == group => Some((property.clone(), g.formula.clone())),
                                _ => None,
                            });
                            let spell = flows.iter().find(|g| g.name == FlowName::VarSpell { ty: *ty, group: *group }).map(|g| parse_template(&g.template)).transpose().map_err(|e| vec![e])?;
                            list.push(Down::Vars { ty: *ty, count: f.formula.clone(), properties: properties.collect(), spell });
                        }
                        FlowName::VarProperty { .. } | FlowName::VarSpell { .. } | FlowName::Property(_) => {}
                    }
                }
                if !list.is_empty() {
                    down.insert(pos, list);
                }
            }
            let spellings = row.spellings.iter().map(|(f, t)| parse_template(t).map(|t| (f.clone(), t))).collect::<Result<_, _>>().map_err(|e| vec![e])?;
            symbols.push(SymbolInfo { inputs, output: contract.output(s), role, ups, musts, advise, down, order: tables.decode_order(id).map_err(|e| vec![e])?, spellings, head: tables.head(id) });
        }
        if !d.is_empty() {
            return Err(d);
        }
        let mut properties_of: BTreeMap<Ty, Vec<String>> = BTreeMap::new();
        for t in &tables.data_types {
            let mut names = t.properties.clone();
            names.sort();
            properties_of.insert(t.id, names);
        }
        let constants = tables.data_types.iter().map(|t| (t.id, t.constant.clone())).collect();
        let templates = tables.guards.iter().map(|g| g.repairs.iter().map(|r| parse_template(&r.template)).collect::<Result<Vec<_>, _>>()).collect::<Result<Vec<_>, _>>().map_err(|e| vec![e])?;
        let (attributes, bounds) = (tables.attributes(), tables.bounds());
        let lexicon = tables.lexicon().map_err(|e| vec![e])?;
        let by_name = (0..contract.n_symbols() as SymbolId).map(|s| (contract.name(s).to_string(), s)).collect();
        Ok(Decoder { tables, contract, symbols, properties_of, constants, attributes, bounds, templates, lexicon, by_name, checked: Arc::new(Mutex::new(Vec::new())) })
    }

    /// The kingdom's tables.
    pub fn tables(&self) -> &KingdomTables {
        &self.tables
    }

    /// The kingdom's contract.
    pub fn contract(&self) -> &KingdomContract {
        &self.contract
    }

    /// Every property `ty` carries, from `given` where it is given and `?` where not, each in its one spelling.
    fn complete(&self, ty: Ty, given: impl IntoIterator<Item = (String, PropertyExpr)>) -> BTreeMap<String, PropertyExpr> {
        let given: BTreeMap<String, PropertyExpr> = given.into_iter().collect();
        self.properties_of.get(&ty).into_iter().flatten().map(|p| (p.clone(), given.get(p).map_or(PropertyExpr::Unknown, |e| e.normal_in(&self.bounds)))).collect()
    }

    /// The properties of `ty`'s constant.
    fn constant_of(&self, ty: Ty) -> Result<BTreeMap<String, PropertyExpr>, String> {
        match self.constants.get(&ty) {
            Some(Some(c)) => Ok(self.complete(ty, c.properties.iter().cloned())),
            _ => Err(bug(format!("kingdom {}: data type {ty:?} has no constant, and a node of it needs one", self.tables.kingdom))),
        }
    }

    /// THE ENTRY CHECK: the tables' [`KingdomTables::entry_defects`], and a data type of a variable terminal that has no
    /// constant has a parameter of the entry (a `var` of it at the root would otherwise have nothing to read).
    pub fn check_entry(&self, entry: &Entry) -> Result<(), Vec<String>> {
        let mut d = self.tables.entry_defects(entry);
        for s in (0..self.contract.n_symbols() as SymbolId).filter(|s| self.contract.is_variable(*s)) {
            let ty = self.contract.output(s);
            if matches!(self.constants.get(&ty), Some(None) | None) && !entry.parameters.iter().any(|v| v.data_type == ty) {
                d.push(bug(format!("kingdom {}: the entry has no parameter of {ty:?}, which has no constant, so {} has nothing to read at the root", self.tables.kingdom, self.contract.name(s))));
            }
        }
        if d.is_empty() {
            Ok(())
        } else {
            Err(d)
        }
    }

    /// The root environment of `entry` (pass 2's start): its parameters at depth 0, in its order, and the attributes.
    pub fn root_environment(&self, entry: &Entry) -> Environment {
        let mut env = Environment::default();
        for a in &self.attributes {
            if let Some(v) = crate::geneframe::property(a).and_then(|p| p.root.clone()) {
                env.attributes.insert(a.clone(), v);
            }
        }
        for f in &entry.flows {
            if let FlowName::Env(a) = &f.name {
                env.attributes.insert(a.clone(), f.formula.normal_in(&self.bounds));
            }
        }
        for (ordinal, p) in entry.parameters.iter().enumerate() {
            let mut given: Vec<(String, PropertyExpr)> = Vec::new();
            let mut measured: BTreeSet<String> = BTreeSet::new();
            if let Some((raw, t)) = &p.form_of {
                if let Some(r) = env.variables.iter().find(|v| v.name.as_deref() == Some(raw.as_str())) {
                    for (q, e) in &t.properties {
                        let read = |x: &Reference| match x {
                            Reference::Input { position: 0, property } => r.properties.get(*property).cloned(),
                            _ => None,
                        };
                        if e.inputs().iter().any(|(_, prop)| r.measured.contains(*prop)) {
                            measured.insert(q.clone());
                        }
                        given.push((q.clone(), e.substitute(&read, &self.bounds)));
                    }
                }
            }
            for (q, v, source) in &p.properties {
                given.retain(|(x, _)| x != q);
                given.push((q.clone(), v.clone()));
                if *source == crate::geneframe::Source::Declared {
                    measured.remove(q);
                } else {
                    measured.insert(q.clone());
                }
            }
            env.variables.push(Variable { binder: Binder { depth: 0, ordinal: ordinal as u32 }, name: Some(p.name.clone()), stem: None, data_type: p.data_type, properties: self.complete(p.data_type, given), measured });
        }
        env
    }

    /// Passes 2 to 4 on the expressed tree `tree`, rooted in `entry`. An error is a bug: an entry that fails
    /// [`Decoder::check_entry`] (checked the first time this decoder decodes it), a malformed tree, or a `must` guard
    /// none of whose repairs applies (which the table check rules out).
    pub fn decode(&self, entry: &Entry, tree: &Tree) -> Result<Decoding, String> {
        let k = &self.tables.kingdom;
        let poisoned = || bug(format!("kingdom {k}: the checked entries' lock is poisoned"));
        if !self.checked.lock().map_err(|_| poisoned())?.contains(entry) {
            self.check_entry(entry).map_err(|d| d.join("\n"))?;
            self.checked.lock().map_err(|_| poisoned())?.push(entry.clone());
        }
        let root_env = self.root_environment(entry);
        let root = self.node(tree, root_env.clone())?;
        if root.data_type != entry.output {
            return Err(bug(format!("kingdom {k}: the tree's root is of {:?}, the entry returns {:?}", root.data_type, entry.output)));
        }
        let mut musts: Vec<(u32, usize)> = entry.guards.iter().enumerate().filter(|(_, g)| g.strength == Strength::Must).map(|(i, g)| (g.order, i)).collect();
        musts.sort();
        let mut root_outcomes = Vec::new();
        for (_, i) in musts {
            let g = &entry.guards[i];
            let values = Values { kids: &[], out: &root.properties, out_measured: &root.measured, env: &root_env, must: true };
            let condition = g.condition.substitute(&|r| values.read(r), &self.bounds);
            let holds = condition.truth(&self.bounds).holds();
            let (mut repair, mut calls) = (None, Vec::new());
            if !holds {
                let plain = Values { must: false, ..values };
                let r = g.repairs.iter().position(|r| self.applies(&r.guard, &root_env) && r.when.substitute(&|x| plain.read(x), &self.bounds).truth(&self.bounds).holds());
                let r = r.ok_or_else(|| bug(format!("kingdom {k}: the entry's must guard {} fails and none of its repairs applies", g.condition)))?;
                repair = Some(r);
                calls = calls_of(&parse_template(&g.repairs[r].template)?);
            }
            root_outcomes.push(Outcome { requirement: i, condition, holds, repair, calls });
        }
        root_outcomes.sort_by_key(|o| o.requirement);
        Ok(Decoding { root, root_outcomes })
    }

    /// The resolved tree of `tree` rooted in `entry` ([`Decoding::resolved`]).
    pub fn resolve(&self, entry: &Entry, tree: &Tree) -> Result<ResolvedTree, String> {
        Ok(self.decode(entry, tree)?.resolved())
    }

    /// True when a repair's guard holds in `env` (read with measured values as they are).
    fn applies(&self, guard: &Condition, env: &Environment) -> bool {
        guard.substitute(&|r| if let Reference::Env(a) = r { Some(env.attributes.get(*a).cloned().unwrap_or(PropertyExpr::Unknown)) } else { None }, &self.bounds).truth(&self.bounds).holds()
    }

    /// The environment of argument `pos` of a node of `info` in `env`, its earlier-decoded siblings `kids`.
    fn child_environment(&self, info: &SymbolInfo, pos: u32, env: &Environment, kids: &[Option<Node>]) -> Result<Environment, String> {
        let k = &self.tables.kingdom;
        let mut child = env.clone();
        let Some(flows) = info.down.get(&pos) else { return Ok(child) };
        let empty = (BTreeMap::new(), BTreeSet::new());
        let values = Values { kids, out: &empty.0, out_measured: &empty.1, env, must: false };
        let mut bound: Vec<Variable> = Vec::new();
        let depth = env.depth + 1;
        for f in flows {
            match f {
                Down::Env(a, e) => {
                    let v = e.substitute(&|r| values.read(r), &self.bounds);
                    if values.measured(e) {
                        child.measured.insert(a.clone());
                    } else {
                        child.measured.remove(a);
                    }
                    child.attributes.insert(a.clone(), v);
                }
                Down::Vars { ty, count, properties, spell } => {
                    // A count that is not a literal here would bind nothing in silence (code review finding 7): a bug.
                    let n = match count.substitute(&|r| values.read(r), &self.bounds) {
                        PropertyExpr::Literal(n) if n >= 0 => n,
                        other => return Err(bug(format!("kingdom {k}: a binding introduces {other} variables of {ty:?}, which is not a literal count here"))),
                    };
                    for each in 0..n {
                        let read = |r: &Reference| match r {
                            Reference::Index("each") => Some(PropertyExpr::Literal(each)),
                            other => values.read(other),
                        };
                        let given: Vec<(String, PropertyExpr)> = properties.iter().map(|(p, e)| (p.clone(), e.substitute(&read, &self.bounds))).collect();
                        let measured: BTreeSet<String> = properties.iter().filter(|(_, e)| values.measured(e)).map(|(p, _)| p.clone()).collect();
                        let props = self.complete(*ty, given);
                        let (name, stem) = match spell {
                            None => (None, None),
                            Some(t) => {
                                let own = |r: &Reference| match r {
                                    Reference::Output(p) => props.get(*p).cloned(),
                                    other => values.read(other),
                                };
                                let ix = BTreeMap::from([("each".to_string(), each), ("depth".to_string(), i64::from(depth))]);
                                // NAME HYGIENE (design 8.14.2): `{nth}` is nothing for the first variable in scope of its
                                // stem, else its ordinal among them, so two variables in scope never share a name.
                                let mut scope = Scope::new(&own, &self.bounds, &self.lexicon);
                                scope.nth = Some(String::new());
                                let stem = scope.render_with(t, &mut Fresh::default(), &ix)?;
                                let before = env.variables.iter().chain(bound.iter()).filter(|v| v.stem.as_deref() == Some(stem.as_str())).count();
                                scope.nth = Some(if before == 0 { String::new() } else { (before + 1).to_string() });
                                (Some(scope.render_with(t, &mut Fresh::default(), &ix)?), Some(stem))
                            }
                        };
                        let ordinal = bound.len() as u32;
                        bound.push(Variable { binder: Binder { depth, ordinal }, name, stem, data_type: *ty, properties: props, measured });
                    }
                }
            }
        }
        if !bound.is_empty() {
            child.depth = depth;
            child.variables.extend(bound);
        }
        Ok(child)
    }

    fn node(&self, t: &Tree, env: Environment) -> Result<Node, String> {
        let k = &self.tables.kingdom;
        let info = self.symbols.get(t.symbol as usize).ok_or_else(|| bug(format!("kingdom {k}: the tree names symbol {}, which is not in the contract", t.symbol)))?;
        if t.kids.len() != info.inputs.len() {
            return Err(bug(format!("kingdom {k}: a node of {} has {} inputs, not {}", self.contract.name(t.symbol), t.kids.len(), info.inputs.len())));
        }
        let word = || t.const_index.ok_or_else(|| bug(format!("kingdom {k}: a node of {} carries no word", self.contract.name(t.symbol))));
        if info.role == Role::Variable {
            let (choice, properties, measured) = match env.choose(info.output, word()?) {
                Some(v) => (Choice::Variable(v.binder), v.properties.clone(), v.measured.clone()),
                None => (Choice::Constant, self.constant_of(info.output)?, BTreeSet::new()),
            };
            return Ok(Node { symbol: t.symbol, data_type: info.output, environment: env, choice: Some(choice), properties, measured, outcomes: Vec::new(), substituted: None, kids: Vec::new() });
        }
        let choice = if info.role == Role::Constant { Some(Choice::Dc(word()?)) } else { None };
        // Pass 2: each input after those its argument's down flows read.
        let mut kids: Vec<Option<Node>> = vec![None; info.inputs.len()];
        for &pos in &info.order {
            let child = self.child_environment(info, pos, &env, &kids)?;
            kids[pos as usize] = Some(self.node(&t.kids[pos as usize], child)?);
        }
        // Pass 4: the up flows, in their order.
        let mut out: BTreeMap<String, PropertyExpr> = BTreeMap::new();
        let mut out_measured: BTreeSet<String> = BTreeSet::new();
        for (p, e) in &info.ups {
            let values = Values { kids: &kids, out: &out, out_measured: &out_measured, env: &env, must: false };
            let v = e.substitute(&|r| values.read(r), &self.bounds);
            let m = values.measured(e);
            out.insert(p.clone(), v);
            if m {
                out_measured.insert(p.clone());
            }
        }
        let mut properties = self.complete(info.output, out);
        let mut measured = out_measured;
        // The must guards, in their order.
        let mut outcomes: Vec<Outcome> = Vec::new();
        let mut substituted: Option<Substituted> = None;
        for &gi in &info.musts {
            let g: &Guard = &self.tables.guards[gi];
            let values = Values { kids: &kids, out: &properties, out_measured: &measured, env: &env, must: true };
            let condition = g.condition.substitute(&|r| values.read(r), &self.bounds);
            let holds = condition.truth(&self.bounds).holds();
            let (mut repair, mut calls) = (None, Vec::new());
            if !holds && substituted.is_none() {
                let plain = Values { must: false, ..values };
                let mut chosen = None;
                for (ri, r) in g.repairs.iter().enumerate() {
                    let site_env = match r.site {
                        Site::In(j) => kids.get(j as usize).and_then(Option::as_ref).map_or(&env, |n| &n.environment),
                        Site::Node | Site::Root => &env,
                    };
                    if !self.applies(&r.guard, site_env) || !r.when.substitute(&|x| plain.read(x), &self.bounds).truth(&self.bounds).holds() {
                        continue;
                    }
                    match substitute_of(&self.templates[gi][ri]) {
                        Some(sub) => {
                            let own = info.inputs.iter().position(|ty| *ty == info.output).map(|p| p as u32);
                            let by = match sub {
                                Substitute::Input(i) if info.inputs.get(i as usize) == Some(&info.output) => Some(Substitution::Input(i)),
                                Substitute::Own => own.map(Substitution::Input),
                                Substitute::Const if matches!(self.constants.get(&info.output), Some(Some(_))) => Some(Substitution::Constant),
                                _ => None,
                            };
                            if let Some(by) = by {
                                substituted = Some(Substituted { by, requirement: gi });
                                chosen = Some(ri);
                                break;
                            }
                        }
                        None => {
                            chosen = Some(ri);
                            calls = calls_of(&self.templates[gi][ri]);
                            break;
                        }
                    }
                }
                repair = Some(chosen.ok_or_else(|| bug(format!("kingdom {k}: {}'s must guard {} fails and none of its repairs applies", self.contract.name(t.symbol), g.condition)))?);
            }
            outcomes.push(Outcome { requirement: gi, condition, holds, repair, calls });
        }
        outcomes.sort_by_key(|o| o.requirement);
        let kids: Vec<Node> = kids.into_iter().collect::<Option<Vec<Node>>>().ok_or_else(|| bug(format!("kingdom {k}: {}'s decode order skips an input", self.contract.name(t.symbol))))?;
        if let Some(s) = substituted {
            (properties, measured) = match s.by {
                Substitution::Input(p) => (kids[p as usize].properties.clone(), kids[p as usize].measured.clone()),
                Substitution::Constant => (self.constant_of(info.output)?, BTreeSet::new()),
            };
        }
        Ok(Node { symbol: t.symbol, data_type: info.output, environment: env, choice, properties, measured, outcomes, substituted, kids })
    }

    /// THE COST'S GUARDS (the second engine): every `advise` guard, of the entry and of each rendered node, that does
    /// not hold, read with measured values as they are. Nothing is repaired.
    pub fn advice(&self, entry: &Entry, decoding: &Decoding) -> Vec<Advice> {
        let mut out = Vec::new();
        let no_kids: [Option<Node>; 0] = [];
        let root = &decoding.root;
        for (i, g) in entry.guards.iter().enumerate().filter(|(_, g)| g.strength == Strength::Advise) {
            let values = Values { kids: &no_kids, out: &root.properties, out_measured: &root.measured, env: &root.environment, must: false };
            let truth = g.condition.substitute(&|r| values.read(r), &self.bounds).truth(&self.bounds);
            if !truth.holds() {
                out.push(Advice { path: None, guard: i, truth });
            }
        }
        for (path, n) in decoding.rendered_nodes() {
            let Some(info) = self.symbols.get(n.symbol as usize) else { continue };
            let kids: Vec<Option<Node>> = n.kids.iter().cloned().map(Some).collect();
            for &gi in &info.advise {
                let values = Values { kids: &kids, out: &n.properties, out_measured: &n.measured, env: &n.environment, must: false };
                let truth = self.tables.guards[gi].condition.substitute(&|r| values.read(r), &self.bounds).truth(&self.bounds);
                if !truth.holds() {
                    out.push(Advice { path: Some(path.clone()), guard: gi, truth });
                }
            }
        }
        out
    }

    /// THE REPAIRS A RENDERED NODE TAKES, in the order they render (design 4, pass 5): first every operand rewrite in
    /// its guard's `order` (a repair at `in<k>`, or each `k := ..` of a `{self(k := ..)}` at the node), then every wrap
    /// of the node (any other template at the node). A substituted node takes none (it is its substitute).
    pub fn steps<'a>(&'a self, n: &Node) -> Vec<Step<'a>> {
        if n.substituted.is_some() {
            return Vec::new();
        }
        let mut failing: Vec<(u32, usize, usize)> = n.outcomes.iter().filter(|o| !o.holds).filter_map(|o| o.repair.map(|r| (self.tables.guards[o.requirement].order, o.requirement, r))).collect();
        failing.sort();
        let mut out = Vec::new();
        for &(_, g, r) in &failing {
            let t = &self.templates[g][r];
            match (self.tables.guards[g].repairs[r].site, rewrites_of(t)) {
                (Site::In(k), _) => out.push(Step::Operand { k, template: t, guard: g }),
                (Site::Node, Some(rw)) => out.extend(rw.iter().map(|(k, t)| Step::Operand { k: *k, template: t, guard: g })),
                _ => {}
            }
        }
        for &(_, g, r) in &failing {
            let t = &self.templates[g][r];
            if self.tables.guards[g].repairs[r].site == Site::Node && rewrites_of(t).is_none() {
                out.push(Step::Wrap { template: t, guard: g });
            }
        }
        out
    }

    /// The repairs of the entry's failing `must` guards, in their order: each wraps the result.
    pub fn root_steps(&self, entry: &Entry, decoding: &Decoding) -> Result<Vec<(usize, Template)>, String> {
        let mut out: Vec<(u32, usize, Template)> = Vec::new();
        for o in decoding.root_outcomes.iter().filter(|o| !o.holds) {
            let g = &entry.guards[o.requirement];
            let r = o.repair.ok_or_else(|| bug("a failing root guard has no repair".to_string()))?;
            out.push((g.order, o.requirement, parse_template(&g.repairs[r].template)?));
        }
        out.sort_by_key(|(order, i, _)| (*order, *i));
        Ok(out.into_iter().map(|(_, i, t)| (i, t)).collect())
    }

    /// The data type a step's template runs at: operand k's, or the node's.
    pub fn step_type(&self, n: &Node, step: &Step) -> Ty {
        match step {
            Step::Operand { k, .. } => n.kids.get(*k as usize).map_or(n.data_type, |c| c.data_type),
            Step::Wrap { .. } => n.data_type,
        }
    }

    /// EXECUTE a template at node `n` with `x` (the interpreter's values, the cost's properties): its conditions and
    /// formulas read the node's decoded properties and environment, as the renderer reads them.
    pub fn exec<X: Exec>(&self, n: &Node, template: &Template, x: &mut X) -> Result<X::V, String> {
        let kids: Vec<Option<Node>> = n.kids.iter().cloned().map(Some).collect();
        let values = Values { kids: &kids, out: &n.properties, out_measured: &n.measured, env: &n.environment, must: false };
        let read = |r: &Reference| values.read(r);
        let mut scope = Scope::new(&read, &self.bounds, &self.lexicon);
        scope.own = self.symbols.get(n.symbol as usize).and_then(|i| i.inputs.iter().position(|ty| *ty == i.output)).map(|p| p as u32);
        scope.exec(template, x)
    }

    /// EXECUTE a root repair's template with `x`: it reads the result's properties and the root environment.
    pub fn exec_root<X: Exec>(&self, decoding: &Decoding, template: &Template, x: &mut X) -> Result<X::V, String> {
        let root = &decoding.root;
        let no_kids: [Option<Node>; 0] = [];
        let values = Values { kids: &no_kids, out: &root.properties, out_measured: &root.measured, env: &root.environment, must: false };
        let read = |r: &Reference| values.read(r);
        Scope::new(&read, &self.bounds, &self.lexicon).exec(template, x)
    }

    /// The symbol id of a name of the contract.
    pub fn symbol_named(&self, name: &str) -> Option<SymbolId> {
        self.by_name.get(name).copied()
    }

    /// The properties the adapter symbol `name` gives on arguments of properties `args` (its up flows).
    pub fn call_properties(&self, name: &str, args: &[BTreeMap<String, PropertyExpr>]) -> Result<(Ty, BTreeMap<String, PropertyExpr>), String> {
        let s = self.symbol_named(name).ok_or_else(|| bug(format!("kingdom {}: a call names {name}, which is no symbol", self.tables.kingdom)))?;
        let info = &self.symbols[s as usize];
        let mut out: BTreeMap<String, PropertyExpr> = BTreeMap::new();
        for (p, e) in &info.ups {
            let v = e.substitute(
                &|r| match r {
                    Reference::Input { position, property } => Some(args.get(*position as usize).and_then(|a| a.get(*property)).cloned().unwrap_or(PropertyExpr::Unknown)),
                    Reference::Output(q) => Some(out.get(*q).cloned().unwrap_or(PropertyExpr::Unknown)),
                    _ => None,
                },
                &self.bounds,
            );
            out.insert(p.clone(), v);
        }
        Ok((info.output, self.complete(info.output, out)))
    }

    /// The properties of `ty`'s constant (`{const}`), and of its element's for an array (`{zero}`).
    fn constants_of(&self, ty: Ty) -> (BTreeMap<String, PropertyExpr>, BTreeMap<String, PropertyExpr>) {
        let constant = self.constant_of(ty).unwrap_or_default();
        let mut z = ty;
        while let Some(crate::geneframe::DataKind::Array { element }) = self.tables.data_type(z).map(|t| &t.kind) {
            z = *element;
        }
        (constant, self.constant_of(z).unwrap_or_default())
    }

    /// The adapters the repairs of node `n` call, in the order they render, each with the operand its step rewrites
    /// (`None` for a wrap of the node): the interpreter and the cost charge a body's once a trip. Each call's path is
    /// left `None` for the caller to set.
    pub fn node_calls(&self, n: &Node) -> Result<Vec<(Option<u32>, Call)>, String> {
        self.repaired(n).map(|(calls, _)| calls)
    }

    /// The properties of node `n`'s operands as its repairs leave them (a fitted operand at its new length).
    pub fn adapted_operands(&self, n: &Node) -> Result<Vec<BTreeMap<String, PropertyExpr>>, String> {
        self.repaired(n).map(|(_, ops)| ops)
    }

    /// [`Decoder::node_calls`] and [`Decoder::adapted_operands`] in one walk of the node's repairs.
    fn repaired(&self, n: &Node) -> Result<NodeRepairs, String> {
        let mut out = Vec::new();
        let mut ops: Vec<BTreeMap<String, PropertyExpr>> = n.kids.iter().map(|c| c.properties.clone()).collect();
        for step in self.steps(n) {
            let (constant, zero) = self.constants_of(self.step_type(n, &step));
            let mut x = PropExec { d: self, ops: ops.clone(), whole: n.properties.clone(), zero, constant, calls: Vec::new() };
            let v = self.exec(n, step.template(), &mut x)?;
            let at = match step {
                Step::Operand { k, .. } => {
                    ops[k as usize] = v;
                    Some(k)
                }
                Step::Wrap { .. } => None,
            };
            out.extend(x.calls.into_iter().map(|(adapter, args, result)| (at, Call { path: None, adapter, args, out: result })));
        }
        Ok((out, ops))
    }

    /// EVERY ADAPTER A REPAIR CALLS where the program renders (the second engine's charge for a rescue written by the
    /// decoder): the entry's first, then each rendered node's in the order its repairs render, each with its arguments'
    /// properties and its result's. A call inside an `{if ..}` that renders the other branch is not one.
    pub fn calls(&self, entry: &Entry, decoding: &Decoding) -> Result<Vec<Call>, String> {
        let mut out = Vec::new();
        let mut whole = decoding.root.properties.clone();
        for (_, t) in self.root_steps(entry, decoding)? {
            let (constant, zero) = self.constants_of(entry.output);
            let mut x = PropExec { d: self, ops: Vec::new(), whole: whole.clone(), zero, constant, calls: Vec::new() };
            whole = self.exec_root(decoding, &t, &mut x)?;
            out.extend(x.calls.into_iter().map(|(adapter, args, result)| Call { path: None, adapter, args, out: result }));
        }
        for (path, n) in decoding.rendered_nodes() {
            out.extend(self.node_calls(n)?.into_iter().map(|(_, mut c)| {
                c.path = Some(path.clone());
                c
            }));
        }
        Ok(out)
    }

    /// THE COST'S ADAPTERS (the second engine): every adapter in the rendered program, the entry's repairs' calls first,
    /// then in pre-order each node that is an adapter symbol (placed by evolution) and each adapter a repair calls
    /// (written by the decoder). The cost charges both alike, so selection prefers programs that need no rescue.
    pub fn rescues(&self, entry: &Entry, decoding: &Decoding) -> Result<Vec<Rescue>, String> {
        let calls = self.calls(entry, decoding)?;
        let mut out: Vec<Rescue> = calls.iter().filter(|c| c.path.is_none()).map(|c| Rescue { path: None, adapter: c.adapter.clone(), placed: false }).collect();
        for (path, n) in decoding.rendered_nodes() {
            if let Some(row) = self.tables.symbol(i64::from(n.symbol)).filter(|s| s.adapter) {
                out.push(Rescue { path: Some(path.clone()), adapter: row.symbol_name.clone(), placed: true });
            }
            out.extend(calls.iter().filter(|c| c.path.as_ref() == Some(&path)).map(|c| Rescue { path: Some(path.clone()), adapter: c.adapter.clone(), placed: false }));
        }
        Ok(out)
    }

    /// PASS 5, kingdom-free: the program `decoding` renders as, by the symbols' spellings and the repairs' templates
    /// ([`template`]). `dc` gives a `?_T`'s text from its symbol and Dc word. The preludes of the symbols used (placed or
    /// called) come first, once each in the tables' order, then each alternative form the program uses, bound once,
    /// then the program.
    pub fn render(&self, entry: &Entry, decoding: &Decoding, dc: &dyn Fn(SymbolId, u32) -> Result<String, String>) -> Result<Rendered, String> {
        let mut r = Renderer { d: self, dc, fresh: Fresh::default(), preludes: Vec::new(), used: BTreeSet::new() };
        let mut body = r.node(&decoding.root, "")?;
        for (_, t) in self.root_steps(entry, decoding)? {
            let spelling: Template = vec![template::Piece::Text(body.clone())];
            let root = &decoding.root;
            let no_kids: [Option<Node>; 0] = [];
            let values = Values { kids: &no_kids, out: &root.properties, out_measured: &root.measured, env: &root.environment, must: false };
            let read = |x: &Reference| values.read(x);
            let mut scope = Scope::new(&read, &self.bounds, &self.lexicon);
            scope.operands = vec![body.clone()];
            scope.spelling = Some(&spelling);
            // The root's {zero} is its element's constant, as at any node (code review finding 8).
            scope.zero = r.zero(entry.output);
            scope.constant = r.constant(entry.output);
            body = scope.render(&t, &mut r.fresh)?;
            for c in scope.take_called() {
                r.prelude_of(&c);
            }
        }
        let mut forms = Vec::new();
        for (i, p) in entry.parameters.iter().enumerate() {
            let Some((raw, t)) = &p.form_of else { continue };
            if !r.used.contains(&Binder { depth: 0, ordinal: i as u32 }) {
                continue;
            }
            let template = parse_template(&t.template)?;
            let none = |_: &Reference| None;
            let mut scope = Scope::new(&none, &self.bounds, &self.lexicon);
            scope.operands = vec![raw.clone()];
            let text = scope.render_with(&template, &mut r.fresh, &BTreeMap::new())?;
            forms.push(format!("let {} = {text}", p.name));
        }
        // The preludes in the tables' order (of the first symbol that has each), so two trees of one program text give
        // it with one prelude order whether an adapter was placed or called.
        let mut preludes = r.preludes;
        preludes.sort_by_key(|t| self.tables.symbols.iter().position(|s| &s.prelude == t));
        Ok(Rendered { preludes, forms, body })
    }
}

/// One repair step of a rendered node ([`Decoder::steps`]): a rewrite of operand `k` by `template`, or a wrap of the
/// node; `guard` is the tables' guard it repairs.
#[derive(Debug, Clone, Copy)]
pub enum Step<'a> {
    Operand { k: u32, template: &'a Template, guard: usize },
    Wrap { template: &'a Template, guard: usize },
}

impl<'a> Step<'a> {
    /// The step's template.
    pub fn template(&self) -> &'a Template {
        match self {
            Step::Operand { template, .. } | Step::Wrap { template, .. } => template,
        }
    }
}

/// One adapter a repair calls where the program renders ([`Decoder::calls`]): its node's path (`None`: the entry's
/// result), the adapter symbol, its arguments' properties in order, and its result's properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub path: Option<Vec<u32>>,
    pub adapter: String,
    pub args: Vec<BTreeMap<String, PropertyExpr>>,
    pub out: BTreeMap<String, PropertyExpr>,
}

/// A node's repairs: the calls they make (each with the operand its step rewrites) and the operands they leave.
type NodeRepairs = (Vec<(Option<u32>, Call)>, Vec<BTreeMap<String, PropertyExpr>>);

/// A call an executed template made: the adapter, its arguments' properties, its result's.
type CallMade = (String, Vec<BTreeMap<String, PropertyExpr>>, BTreeMap<String, PropertyExpr>);

/// The executor of a repair's template over properties: what each call reads and gives.
struct PropExec<'a> {
    d: &'a Decoder,
    ops: Vec<BTreeMap<String, PropertyExpr>>,
    whole: BTreeMap<String, PropertyExpr>,
    zero: BTreeMap<String, PropertyExpr>,
    constant: BTreeMap<String, PropertyExpr>,
    calls: Vec<CallMade>,
}

impl Exec for PropExec<'_> {
    type V = BTreeMap<String, PropertyExpr>;

    fn operand(&mut self, k: u32) -> Result<Self::V, String> {
        self.ops.get(k as usize).cloned().ok_or_else(|| bug(format!("a repair reads operand {k}, which the node has not")))
    }

    fn value(&mut self, e: &PropertyExpr) -> Result<Self::V, String> {
        Ok(BTreeMap::from([("value".to_string(), e.clone())]))
    }

    fn zero(&mut self) -> Result<Self::V, String> {
        Ok(self.zero.clone())
    }

    fn constant(&mut self) -> Result<Self::V, String> {
        Ok(self.constant.clone())
    }

    fn whole(&mut self) -> Result<Self::V, String> {
        Ok(self.whole.clone())
    }

    fn call(&mut self, symbol: &str, args: Vec<Self::V>) -> Result<Self::V, String> {
        let (_, out) = self.d.call_properties(symbol, &args)?;
        self.calls.push((symbol.to_string(), args, out.clone()));
        Ok(out)
    }
}

/// A rendered program: the used symbols' preludes (once each, in first use), the alternative forms' bindings, the body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub preludes: Vec<String>,
    pub forms: Vec<String>,
    pub body: String,
}

struct Renderer<'a> {
    d: &'a Decoder,
    dc: &'a dyn Fn(SymbolId, u32) -> Result<String, String>,
    fresh: Fresh,
    preludes: Vec<String>,
    used: BTreeSet<Binder>,
}

impl Renderer<'_> {
    fn prelude_of(&mut self, symbol: &str) {
        if let Some(s) = self.d.tables.symbols.iter().find(|s| s.symbol_name == symbol && !s.prelude.is_empty()) {
            if !self.preludes.contains(&s.prelude) {
                self.preludes.push(s.prelude.clone());
            }
        }
    }

    fn constant(&self, ty: Ty) -> Option<String> {
        self.d.constants.get(&ty).cloned().flatten().map(|c| c.spelling)
    }

    /// The constant of `ty`'s element, for an array; else its own.
    fn zero(&self, ty: Ty) -> Option<String> {
        match self.d.tables.data_type(ty).map(|t| &t.kind) {
            Some(crate::geneframe::DataKind::Array { element }) => self.zero(*element),
            _ => self.constant(ty),
        }
    }

    /// The operands `t` reads in a form, rendered here (through this renderer, so their fresh names, preludes and
    /// variables are the program's: code review finding 9).
    fn forms(&mut self, n: &Node, t: &Template) -> Result<BTreeMap<(u32, String), String>, String> {
        let mut out = BTreeMap::new();
        for (k, f) in forms_of(t) {
            let kid = n.kids.get(k as usize).ok_or_else(|| bug(format!("a template reads operand {k} in a form, which the node has not")))?;
            out.insert((k, f.clone()), self.node(kid, &f)?);
        }
        Ok(out)
    }

    /// The names of the variables each argument of `n` introduces, by argument.
    fn bound(n: &Node) -> Vec<Vec<String>> {
        n.kids
            .iter()
            .map(|kid| {
                let env = &kid.environment;
                if env.depth > n.environment.depth {
                    env.variables.iter().filter(|v| v.binder.depth == env.depth).map(|v| v.name.clone().unwrap_or_default()).collect()
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    /// The text of node `n` in `form` (empty: the default).
    fn node(&mut self, n: &Node, form: &str) -> Result<String, String> {
        let k = &self.d.tables.kingdom;
        if let Some(s) = n.substituted {
            return match s.by {
                Substitution::Input(p) => self.node(&n.kids[p as usize], form),
                Substitution::Constant => self.constant(n.data_type).ok_or_else(|| bug(format!("kingdom {k}: a node replaced by the constant of {:?}, which has none", n.data_type))),
            };
        }
        match n.choice {
            Some(Choice::Variable(b)) => {
                self.used.insert(b);
                let v = n.environment.variables.iter().find(|v| v.binder == b).ok_or_else(|| bug(format!("kingdom {k}: the variable {b:?} is not in scope")))?;
                return v.name.clone().ok_or_else(|| bug(format!("kingdom {k}: the variable {b:?} has no name (no var.T.spell flow)")));
            }
            Some(Choice::Constant) => return self.constant(n.data_type).ok_or_else(|| bug(format!("kingdom {k}: {:?} has no constant", n.data_type))),
            Some(Choice::Dc(w)) => return (self.dc)(n.symbol, w),
            Some(Choice::Free(_)) | None => {}
        }
        let info = &self.d.symbols[n.symbol as usize];
        let name = self.d.contract.name(n.symbol).to_string();
        self.prelude_of(&name);
        let mut operands: Vec<String> = Vec::with_capacity(n.kids.len());
        for kid in &n.kids {
            operands.push(self.node(kid, "")?);
        }
        // A form the symbol lacks passes to its head argument.
        let spelling = match (info.spellings.get(form), info.head) {
            (Some(t), _) => t.clone(),
            (None, Some(h)) if !form.is_empty() => {
                operands[h as usize] = self.node(&n.kids[h as usize], form)?;
                info.spellings.get("").cloned().ok_or_else(|| bug(format!("kingdom {k}: {name} has no spelling")))?
            }
            _ => info.spellings.get("").cloned().ok_or_else(|| bug(format!("kingdom {k}: {name} has no spelling")))?,
        };
        let kids_opt: Vec<Option<Node>> = n.kids.iter().cloned().map(Some).collect();
        let values = Values { kids: &kids_opt, out: &n.properties, out_measured: &n.measured, env: &n.environment, must: false };
        let read = |x: &Reference| values.read(x);
        let own = info.inputs.iter().position(|ty| *ty == info.output).map(|p| p as u32);
        let bound = Renderer::bound(n);
        let steps = self.d.steps(n);
        let mut text: Option<String> = None;
        for step in &steps {
            let ty = self.d.step_type(n, step);
            if let Step::Wrap { .. } = step {
                if text.is_none() {
                    let forms = self.forms(n, &spelling)?;
                    let mut scope = Scope::new(&read, &self.d.bounds, &self.d.lexicon);
                    (scope.operands, scope.own, scope.forms, scope.bound) = (operands.clone(), own, forms, bound.clone());
                    (scope.zero, scope.constant) = (self.zero(n.data_type), self.constant(n.data_type));
                    text = Some(scope.render(&spelling, &mut self.fresh)?);
                    for c in scope.take_called() {
                        self.prelude_of(&c);
                    }
                }
            }
            let current: Template = match &text {
                Some(t) => vec![template::Piece::Text(t.clone())],
                None => spelling.clone(),
            };
            let forms = self.forms(n, step.template())?;
            let mut scope = Scope::new(&read, &self.d.bounds, &self.d.lexicon);
            (scope.operands, scope.own, scope.forms, scope.bound) = (operands.clone(), own, forms, bound.clone());
            (scope.spelling, scope.zero, scope.constant) = (Some(&current), self.zero(ty), self.constant(ty));
            let rendered = scope.render(step.template(), &mut self.fresh)?;
            for c in scope.take_called() {
                self.prelude_of(&c);
            }
            match step {
                Step::Operand { k: at, .. } => *operands.get_mut(*at as usize).ok_or_else(|| bug(format!("kingdom {k}: a repair rewrites operand {at} of {name}, which has {} inputs", n.kids.len())))? = rendered,
                Step::Wrap { .. } => text = Some(rendered),
            }
        }
        match text {
            Some(t) => Ok(t),
            None => {
                let forms = self.forms(n, &spelling)?;
                let mut scope = Scope::new(&read, &self.d.bounds, &self.d.lexicon);
                (scope.operands, scope.own, scope.forms, scope.bound) = (operands, own, forms, bound);
                (scope.zero, scope.constant) = (self.zero(n.data_type), self.constant(n.data_type));
                let t = scope.render(&spelling, &mut self.fresh)?;
                for c in scope.take_called() {
                    self.prelude_of(&c);
                }
                Ok(t)
            }
        }
    }
}

#[cfg(test)]
mod tests;
