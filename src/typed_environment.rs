//! The decoder's passes 2 to 4, on the host and kingdom-free (design `docs/DESIGN_futhark_types_and_properties.md`
//! section 4; plan `docs/PLAN_futhark_kingdom.md` section 7, task G3; master 6's format
//! `kingdoms/futhark/master/README.md` section 5).
//!
//! Pass 1 (`typed_gep::express`) gives the tree of symbols. Then, in one walk:
//!
//! 2. **The environment, top-down.** The root's is the entry's parameters with their properties ([`Entry`]); a
//!    child's is its parent's with the Binding rows of the argument it fills appended, in Binding row order, and one
//!    more trip when that argument is its symbol's trip argument. [`in_scope`] is THE order of the variables of a data
//!    type in an environment: the only one the renderer, the interpreter, the reader and the e-graph use.
//! 3. **Variables.** A `var.T` takes its index word `w` (`Tree::const_index`, a Dv word) and becomes the `(w mod c)`-th
//!    of the `c` variables of `T` in scope, or `T`'s constant when `c = 0`. `var.size` is the same over the size names.
//! 4. **Properties, bottom-up, and requirements.** Each node's properties by its PropertyRule rows over its inputs'
//!    properties, as symbolic [`PropertyExpr`]s in their one spelling (a var's are its variable's); each Requirement
//!    checked by its relation ([`Relation::holds`]). A requirement that fails names its adapter. The work bound
//!    `env.trips < depth` is checked first: when it fails, or when a heavy adapter of the node would run at
//!    `trips >= depth`, the node is **substituted** as its work bound's adapter says ([`Substitute`]): `subst` by its
//!    first input of its own data type, else by its type's constant; `subst.const` (`hist.*`, `histrows.*`) by its
//!    type's constant. None of its other adapters render ([`Substituted`]). The renderer, the interpreter, the cost, the
//!    reader and the e-graph read this annotation and never decide it again.
//!
//! The **resolved tree** ([`Decoding::resolved`]) names every variable by its binder's identity (the binding depth and
//! the ordinal), never by its index word: the only form any consumer compares or hashes (plan 7.5 risk 3). Rendering
//! (pass 5) is task F2's. A SUBTREE is compared by its key ([`Decoding::subtree_key`]), which adds what the subtree
//! renders by that is outside it: its trips and its free variables' binders, data types and properties.
//!
//! Deterministic in the gene alone: every map is a `BTreeMap`, every list in table or position order.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::futhark_table::Kingdom6;
use crate::geneframe::{Adapter, AdapterSite, EntryParameter, EntryRequirement, KingdomTables, PropertyExpr, Relation, Subject, Substitute, Ty};
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::typed_gep::Tree;

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

/// A variable's identity: the binding depth at which it was bound (0 for the entry's parameters, then one more for
/// each argument on the path that binds any variable) and its ordinal among the variables bound there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Binder {
    pub depth: u32,
    pub ordinal: u32,
}

/// A variable in scope: its identity, its name (the entry's parameters have one; a bound variable is named by the
/// renderer), its data type and its properties (every property its data type carries; `?` where unknown).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub binder: Binder,
    pub name: Option<String>,
    pub data_type: Ty,
    pub properties: BTreeMap<String, PropertyExpr>,
}

/// The variables a node may use, in binding order, its binding depth, and its trip count `env.trips` (the trip binders
/// above it).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Environment {
    pub variables: Vec<Variable>,
    pub depth: u32,
    pub trips: u32,
}

/// THE ENVIRONMENT'S ORDER: the variables of data type `ty` in scope, the entry's parameters first in their order, then
/// each binding's variables, outermost first, in Binding row order. A `var.T`'s word `w` names the `(w mod c)`-th of
/// them ([`Environment::choose`]); a reader writes a variable back as its place here ([`Environment::index_of`]). Every
/// consumer reads this one function.
pub fn in_scope(env: &Environment, ty: Ty) -> Vec<&Variable> {
    env.variables.iter().filter(|v| v.data_type == ty).collect()
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

    /// The index word that names `binder` among the variables of `ty` in scope (the least one; any word congruent to
    /// it modulo their number names it too).
    pub fn index_of(&self, ty: Ty, binder: Binder) -> Option<u32> {
        in_scope(self, ty).iter().position(|v| v.binder == binder).map(|i| i as u32)
    }

    /// A child's environment: this one with `bound` appended at one more binding depth (none added when `bound` is
    /// empty), and one more trip when `trip`.
    fn child(&self, bound: Vec<(Ty, BTreeMap<String, PropertyExpr>)>, trip: bool) -> Environment {
        let mut e = self.clone();
        if !bound.is_empty() {
            e.depth += 1;
            for (ordinal, (data_type, properties)) in bound.into_iter().enumerate() {
                e.variables.push(Variable { binder: Binder { depth: e.depth, ordinal: ordinal as u32 }, name: None, data_type, properties });
            }
        }
        if trip {
            e.trips += 1;
        }
        e
    }
}

/// One parameter of the entry (`points : [n][d]f32` is `points` of `[][]f32` with rows = `n`, cols = `d`; a size `k`
/// is `k` of `size` with value = `k`), in the root environment's order (master 6 README section 8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryVariable {
    pub name: String,
    pub data_type: Ty,
    pub properties: Vec<(String, PropertyExpr)>,
}

/// A requirement on the entry's result (`out.len = k -> fit.root`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootRequirement {
    pub property: String,
    pub relation: Relation,
    pub against: PropertyExpr,
    pub adapter: String,
}

/// The entry point: its result's data type, its parameters (the root environment) and the requirements on its result.
/// The decoder checks it against the tables on every decode ([`Decoder::check_entry`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub output: Ty,
    pub variables: Vec<EntryVariable>,
    pub requirements: Vec<RootRequirement>,
}

impl Entry {
    /// The entry of a loaded master 6 kingdom: its root's data type and requirements, and its root environment's
    /// variables in the view's order (`environment.tsv`).
    pub fn of_kingdom6(k: &Kingdom6) -> Entry {
        Entry {
            output: k.root.data_type,
            variables: k.variables.iter().map(|v| EntryVariable { name: v.name.clone(), data_type: v.data_type, properties: v.properties.clone() }).collect(),
            requirements: k.root.requires.iter().map(|q| RootRequirement { property: q.property.clone(), relation: q.relation, against: q.against.clone(), adapter: q.adapter.clone() }).collect(),
        }
    }
}

/// What a node's word reads as: a `?_T`'s Dc index, a `var.T`'s variable, or (no variable of its type in scope) its
/// type's constant. In a [`SubtreeKey`] only, a variable bound outside the subtree is `Free(j)`, the `j`-th of the
/// key's free variables, and one bound inside it has its depth counted from the subtree's root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Choice {
    Dc(u32),
    Variable(Binder),
    Constant,
    Free(u32),
}

/// One requirement checked at one node: its row (an index into the tables' requirements, or into the entry's for a
/// root requirement), both sides' values, whether it holds, and the adapter it names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub requirement: usize,
    pub subject: PropertyExpr,
    pub against: PropertyExpr,
    pub holds: bool,
    pub adapter: String,
}

/// What a substituted node renders as: its input at this position, or its data type's constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Substitution {
    Input(u32),
    Constant,
}

/// The work-bound substitution of a node (master 6 README 5.3): what it renders as, and the requirement that decided
/// it (the failing work bound, or the requirement whose heavy adapter would run at `trips >= depth`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Substituted {
    pub by: Substitution,
    pub requirement: usize,
}

/// One decoded node: its symbol and data type, its environment, what its word reads as, its properties (every property
/// its data type carries; a substituted node's are its substitute's), each of its requirements' outcome in table
/// order, its substitution, and its inputs in position order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub symbol: SymbolId,
    pub data_type: Ty,
    pub environment: Environment,
    pub choice: Option<Choice>,
    pub properties: BTreeMap<String, PropertyExpr>,
    pub outcomes: Vec<Outcome>,
    pub substituted: Option<Substituted>,
    pub kids: Vec<Node>,
}

/// The tree with every variable by its binder's identity: the form two WHOLE trees of one entry are compared or hashed
/// by. Two whose equal index words name different variables differ; two whose different words name the same variable
/// are equal. A subtree is compared by its [`SubtreeKey`] instead, which carries what it renders by outside itself.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedTree {
    pub symbol: SymbolId,
    pub choice: Option<Choice>,
    pub kids: Vec<ResolvedTree>,
}

/// A variable a subtree uses but does not bind: its binder, its data type and its properties.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FreeVariable {
    pub binder: Binder,
    pub data_type: Ty,
    pub properties: BTreeMap<String, PropertyExpr>,
}

/// THE KEY OF A SUBTREE (G3 review finding 4): the form F5's sharing and P2's marriage compare and hash a subtree by.
/// Its resolved tree, with each variable bound outside it as `Choice::Free(j)` and each bound inside it at its depth
/// below the subtree's root; the trips at its root; and its free variables in order of first use, each with its
/// binder, data type and properties. Every adapter and substitution inside a subtree is decided by these alone, so
/// two subtrees of one key render with the same adapters; `get(is, i)` under `tabulate n` and under `tabulate k`
/// differ (`i`'s bound). The same subtree under one more `let` has a free variable of another binder depth, so its key
/// differs too, which costs sharing only.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubtreeKey {
    pub tree: ResolvedTree,
    pub trips: u32,
    pub free: Vec<FreeVariable>,
}

/// A requirement that fails where the program is rendered (a node under a substitution that drops it is not
/// rendered): the node's path of argument positions from the root, and what renders there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// Requirement `requirement` fails at the node; adapter `adapter` makes it hold.
    Adapter { path: Vec<u32>, requirement: usize, adapter: String },
    /// The node is substituted (the work bound).
    Substituted { path: Vec<u32>, by: Substitution, requirement: usize },
    /// The entry's requirement `requirement` fails on the result; adapter `adapter` makes it hold.
    Root { requirement: usize, adapter: String },
}

/// A decoded gene: the annotated tree and the outcomes of the entry's requirements.
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
                        free.push(FreeVariable { binder: b, data_type: n.data_type, properties: n.properties.clone() });
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
        Some(SubtreeKey { tree, trips: n.environment.trips, free })
    }

    /// Every failing requirement and every substitution of the rendered program, the entry's first, then the nodes in
    /// pre-order. A substituted node lists its substitution alone, and only the input it renders as is walked below.
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
                out.push(Failure::Adapter { path: path.clone(), requirement: o.requirement, adapter: o.adapter.clone() });
            }
            for (k, kid) in n.kids.iter().enumerate() {
                path.push(k as u32);
                go(kid, path, out);
                path.pop();
            }
        }
        let mut out: Vec<Failure> = self.root_outcomes.iter().filter(|o| !o.holds).map(|o| Failure::Root { requirement: o.requirement, adapter: o.adapter.clone() }).collect();
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
}

/// The Binding rows of one argument: each variable's data type and its properties by rule, in table order.
type BindingRows = Vec<(Ty, Vec<(String, PropertyExpr)>)>;

/// What the decoder reads of one symbol, gathered once.
#[derive(Debug, Clone)]
struct SymbolInfo {
    inputs: Vec<Ty>,
    output: Ty,
    rules: BTreeMap<String, PropertyExpr>,
    /// Indices into the tables' requirements, in table order.
    requirements: Vec<usize>,
    /// Argument position -> its Binding rows' (data type, properties), in table order.
    bindings: BTreeMap<u32, BindingRows>,
    trip: Option<u32>,
    order: Vec<u32>,
}

/// The decoder of one kingdom: its tables and its contract, whose symbol `s` is the tables' symbol `s`. It OWNS
/// them through `Arc` (plan step 5a F1: Arc handles, no lifetime, no `Box::leak`), so a caller may keep it as long as
/// it likes and share the tables with it at no copy ([`Decoder::from_arcs`]).
#[derive(Debug, Clone)]
pub struct Decoder {
    tables: Arc<KingdomTables>,
    contract: Arc<KingdomContract>,
    symbols: Vec<SymbolInfo>,
    properties_of: BTreeMap<Ty, Vec<String>>,
    constants: BTreeMap<Ty, Option<Vec<(String, PropertyExpr)>>>,
    adapters: BTreeMap<String, Adapter>,
}

impl Decoder {
    /// The decoder of `tables` and `contract`. Every defect, each "our software has a bug: ...": the tables' own check
    /// (which finds a work-bound substitution by the constant of a data type that has none, before any gene runs),
    /// and a contract symbol whose inputs or output the tables do not give it, or a `var.T` with a property rule (its
    /// properties are its variable's).
    ///
    /// It copies `tables` and `contract` once; [`Decoder::from_arcs`] shares them instead.
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
            if !tables.symbols.iter().any(|r| r.symbol == id) || output != Some(contract.output(s)) || inputs != contract.signature(s) {
                d.push(bug(format!(
                    "kingdom {k}: contract symbol {s} ({}) takes {:?} to {:?}, but the tables' symbol {id} takes {inputs:?} to {output:?}",
                    contract.name(s),
                    contract.signature(s),
                    contract.output(s)
                )));
                continue;
            }
            let rules: BTreeMap<String, PropertyExpr> = tables.property_rules.iter().filter(|r| r.symbol == id).map(|r| (r.property.clone(), r.expr.clone())).collect();
            if contract.is_variable(s) && !rules.is_empty() {
                d.push(bug(format!("kingdom {k}: the variable terminal {} has property rules; its properties are its variable's", contract.name(s))));
            }
            let mut bindings: BTreeMap<u32, BindingRows> = BTreeMap::new();
            for b in tables.bindings.iter().filter(|b| b.symbol == id) {
                bindings.entry(b.position).or_default().push((b.data_type, b.properties.clone()));
            }
            symbols.push(SymbolInfo {
                inputs,
                output: contract.output(s),
                rules,
                requirements: tables.requirements.iter().enumerate().filter(|(_, q)| q.symbol == id).map(|(i, _)| i).collect(),
                bindings,
                trip: tables.trip_arguments.iter().find(|t| t.symbol == id).map(|t| t.position),
                order: tables.decode_order(id).map_err(|e| vec![e])?,
            });
        }
        if !d.is_empty() {
            return Err(d);
        }
        let mut properties_of: BTreeMap<Ty, Vec<String>> = BTreeMap::new();
        for p in &tables.property_types {
            properties_of.entry(p.data_type).or_default().push(p.name.clone());
        }
        for names in properties_of.values_mut() {
            names.sort();
        }
        let constants = tables.data_types.iter().map(|t| (t.id, t.constant.clone())).collect();
        let adapters = tables.adapters.iter().map(|a| (a.name.clone(), a.clone())).collect();
        Ok(Decoder { tables, contract, symbols, properties_of, constants, adapters })
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
        self.properties_of.get(&ty).into_iter().flatten().map(|p| (p.clone(), given.get(p).map_or(PropertyExpr::Unknown, PropertyExpr::normal))).collect()
    }

    /// The properties of `ty`'s constant.
    fn constant_of(&self, ty: Ty) -> Result<BTreeMap<String, PropertyExpr>, String> {
        match self.constants.get(&ty) {
            Some(Some(c)) => Ok(self.complete(ty, c.iter().cloned())),
            _ => Err(bug(format!("kingdom {}: data type {ty:?} has no constant, and a node of it needs one", self.tables.kingdom))),
        }
    }

    /// THE ENTRY CHECK (G3 review findings 5 and 6): the tables' [`KingdomTables::entry_defects`], and a data type of a
    /// variable terminal that has no constant (`size`) has a parameter of the entry, since a `var` of it at the root
    /// would otherwise have nothing to read (a binding binds an `i64`, never a size).
    pub fn check_entry(&self, entry: &Entry) -> Result<(), Vec<String>> {
        let vars: Vec<EntryParameter> = entry.variables.iter().map(|v| (v.name.as_str(), v.data_type, v.properties.as_slice())).collect();
        let reqs: Vec<EntryRequirement> = entry.requirements.iter().map(|q| (q.property.as_str(), q.relation, &q.against, q.adapter.as_str())).collect();
        let mut d = self.tables.entry_defects(entry.output, &vars, &reqs);
        for s in (0..self.contract.n_symbols() as SymbolId).filter(|s| self.contract.is_variable(*s)) {
            let ty = self.contract.output(s);
            if matches!(self.constants.get(&ty), Some(None) | None) && !entry.variables.iter().any(|v| v.data_type == ty) {
                d.push(bug(format!("kingdom {}: the entry has no parameter of {ty:?}, which has no constant, so {} has nothing to read at the root", self.tables.kingdom, self.contract.name(s))));
            }
        }
        if d.is_empty() {
            Ok(())
        } else {
            Err(d)
        }
    }

    /// Passes 2 to 4 on the expressed tree `tree`, rooted in `entry`. An error is a bug: an entry that fails
    /// [`Decoder::check_entry`], or a malformed tree.
    pub fn decode(&self, entry: &Entry, tree: &Tree) -> Result<Decoding, String> {
        let k = &self.tables.kingdom;
        self.check_entry(entry).map_err(|d| d.join("\n"))?;
        let mut root_env = Environment::default();
        for (ordinal, v) in entry.variables.iter().enumerate() {
            root_env.variables.push(Variable {
                binder: Binder { depth: 0, ordinal: ordinal as u32 },
                name: Some(v.name.clone()),
                data_type: v.data_type,
                properties: self.complete(v.data_type, v.properties.iter().cloned()),
            });
        }
        let root = self.node(tree, root_env)?;
        if root.data_type != entry.output {
            return Err(bug(format!("kingdom {k}: the tree's root is of {:?}, the entry returns {:?}", root.data_type, entry.output)));
        }
        let root_outcomes = entry
            .requirements
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let subject = root.properties.get(&q.property).cloned().unwrap_or(PropertyExpr::Unknown);
                let against = q.against.normal();
                Outcome { requirement: i, holds: q.relation.holds(&subject, &against), subject, against, adapter: q.adapter.clone() }
            })
            .collect();
        Ok(Decoding { root, root_outcomes })
    }

    /// The resolved tree of `tree` rooted in `entry` ([`Decoding::resolved`]).
    pub fn resolve(&self, entry: &Entry, tree: &Tree) -> Result<ResolvedTree, String> {
        Ok(self.decode(entry, tree)?.resolved())
    }

    fn node(&self, t: &Tree, env: Environment) -> Result<Node, String> {
        let k = &self.tables.kingdom;
        let info = self.symbols.get(t.symbol as usize).ok_or_else(|| bug(format!("kingdom {k}: the tree names symbol {}, which is not in the contract", t.symbol)))?;
        if t.kids.len() != info.inputs.len() {
            return Err(bug(format!("kingdom {k}: a node of {} has {} inputs, not {}", self.contract.name(t.symbol), t.kids.len(), info.inputs.len())));
        }
        let word = || t.const_index.ok_or_else(|| bug(format!("kingdom {k}: a node of {} carries no word", self.contract.name(t.symbol))));
        if self.contract.is_variable(t.symbol) {
            let (choice, properties) = match env.choose(info.output, word()?) {
                Some(v) => (Choice::Variable(v.binder), v.properties.clone()),
                None => (Choice::Constant, self.constant_of(info.output)?),
            };
            return Ok(Node { symbol: t.symbol, data_type: info.output, environment: env, choice: Some(choice), properties, outcomes: Vec::new(), substituted: None, kids: Vec::new() });
        }
        let choice = if self.contract.is_constant(t.symbol) { Some(Choice::Dc(word()?)) } else { None };
        // The inputs, each after those its own bindings read (pass 2 needs their properties).
        let mut kids: Vec<Option<Node>> = vec![None; info.inputs.len()];
        for &pos in &info.order {
            let value = |j: u32, p: &str| kids.get(j as usize).and_then(Option::as_ref).and_then(|n| n.properties.get(p)).cloned().unwrap_or(PropertyExpr::Unknown);
            let bound: Vec<(Ty, BTreeMap<String, PropertyExpr>)> = info
                .bindings
                .get(&pos)
                .into_iter()
                .flatten()
                .map(|(ty, props)| (*ty, self.complete(*ty, props.iter().map(|(p, e)| (p.clone(), e.substitute(&value))))))
                .collect();
            let child = env.child(bound, info.trip == Some(pos));
            let node = self.node(&t.kids[pos as usize], child)?;
            kids[pos as usize] = Some(node);
        }
        let kids: Vec<Node> = kids.into_iter().collect::<Option<Vec<Node>>>().ok_or_else(|| bug(format!("kingdom {k}: {}'s decode order skips an input", self.contract.name(t.symbol))))?;
        let value = |j: u32, p: &str| kids.get(j as usize).and_then(|n| n.properties.get(p)).cloned().unwrap_or(PropertyExpr::Unknown);
        let mut properties = self.complete(info.output, info.rules.iter().map(|(p, e)| (p.clone(), e.substitute(&value))));
        let outcomes: Vec<Outcome> = info
            .requirements
            .iter()
            .map(|&i| {
                let q = &self.tables.requirements[i];
                let subject = match &q.subject {
                    Subject::Input { position, property } => value(*position, property),
                    Subject::Trips => PropertyExpr::Literal(i64::from(env.trips)),
                };
                let against = q.against.substitute(&value);
                Outcome { requirement: i, holds: q.relation.holds(&subject, &against), subject, against, adapter: q.adapter.clone() }
            })
            .collect();
        let substituted = self.substitution(info, &env, &outcomes).map(|(requirement, how)| {
            let own = info.inputs.iter().position(|ty| *ty == info.output);
            let by = match (how, own) {
                (Substitute::OwnInput, Some(p)) => Substitution::Input(p as u32),
                (Substitute::OwnInput, None) | (Substitute::Constant, _) => Substitution::Constant,
            };
            Substituted { by, requirement }
        });
        if let Some(s) = substituted {
            properties = match s.by {
                Substitution::Input(p) => kids[p as usize].properties.clone(),
                Substitution::Constant => self.constant_of(info.output)?,
            };
        }
        Ok(Node { symbol: t.symbol, data_type: info.output, environment: env, choice, properties, outcomes, substituted, kids })
    }

    /// THE WORK-BOUND DECISION (master 6 README 5.3): the requirement that substitutes this node, if any, and how
    /// (its work bound's adapter, [`Substitute`]). The work bound `env.trips < depth` is checked first; when it holds,
    /// a failing requirement whose heavy adapter would run at `trips >= depth` (a node adapter at the node's trips, a
    /// body adapter at the body's, one more) substitutes the node instead. The table check gives a symbol at most one
    /// work bound, naming a substitution adapter, and one on every symbol that names a heavy adapter.
    fn substitution(&self, info: &SymbolInfo, env: &Environment, outcomes: &[Outcome]) -> Option<(usize, Substitute)> {
        let work = outcomes.iter().find(|o| self.tables.requirements[o.requirement].subject == Subject::Trips)?;
        let how = Substitute::of_adapter(&work.adapter)?;
        if !work.holds {
            return Some((work.requirement, how));
        }
        let depth = match work.against {
            PropertyExpr::Literal(d) => d,
            _ => return None,
        };
        outcomes
            .iter()
            .filter(|o| !o.holds)
            .find(|o| {
                self.adapters.get(&o.adapter).is_some_and(|a| {
                    let at = match a.runs_at {
                        AdapterSite::Node | AdapterSite::Root => env.trips,
                        AdapterSite::Body => env.trips + u32::from(info.trip.is_some()),
                    };
                    a.heavy && i64::from(at) >= depth
                })
            })
            .map(|o| (o.requirement, how))
    }
}

#[cfg(test)]
mod tests;
