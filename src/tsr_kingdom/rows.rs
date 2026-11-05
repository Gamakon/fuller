//! THE TSR KINGDOM AS ROWS of the gene frame's fixed schema (design `docs/DESIGN_futhark_types_and_properties.md`
//! 3.3 and 8.3; plan `docs/PLAN_futhark_kingdom.md` section 8). TSR is Typed Symbolic Regression.
//!
//! Each variant's rows are the kingdom `tsr.<variant>` of the master tables (`kingdoms/master/`), written by
//! `kingdoms/tsr/gen_tables.py` through `write_rows.py` (never by hand; its `--check` is a test): the four kinds of row
//! and nothing else:
//!
//! - ONE data type `F`, carrying `depth`, `order`, `fam`, `arm` and `ties`;
//! - Argument rows: the 18 operators of `SymbolTable::wide()`, `F` in and out, then the input terminals `x0..x9` and
//!   the constant `?` (role constant: a Dc word), in `wide()`'s order;
//! - up Flow rows: depth is the greatest input depth, plus 1 for a raising unary; stream order is Horton-Strahler order
//!   over flattened sum and product chains, through `fam`, `arm` and `ties` (`out.p` read acyclically);
//! - `must` Guard rows, the variant's only difference: `out.order <= 3` on the four binaries (order), `in0.depth <= 1`
//!   on the ten raising unaries (depth), `in0.order <= 2` on the seven windowed unaries (windows); each repaired by
//!   `{in0}`, the node rendered as its first input. `single` has no guard.
//!
//! The gene is the single kingdom's word for word: one type, no Dv lane (the inputs are terminals, not variables), 104
//! words at head 34. A law of `n` columns keeps the first `n` input terminals ([`TsrRows::load`]), so its symbol ids
//! are `wide(n)`'s. An illegal shape stays representable in the gene and is INEXPRESSIBLE in the program: the decoder
//! ([`crate::typed_environment::Decoder`]) substitutes, bottom-up, and [`program`] is the tree that is scored.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::TsrVariant;
use crate::geneframe::{Entry, FlowName, KingdomTables, Owner, Place, PropertyExpr, Relation, Role, Site, Strength, Ty, Way};
use crate::kingdom_contract::{relations_from_tables, KingdomContract, SymbolId};
use crate::typed_environment::{Choice, Decoder, Decoding, Node, Substituted, Substitution};
use crate::typed_gep::{express, GeneLayout, Tree};

/// The input terminals the tables hold; a law keeps the first `n`.
pub const MAX_INPUTS: usize = 10;

/// The single kingdom's head (`Config::head`; program size 17), which gives the 104-word gene.
pub const HEAD: usize = 34;

impl TsrVariant {
    /// The variant's kingdom in the master tables (`kingdoms/master/`): `tsr.<variant>`.
    pub fn master_kingdom(self) -> String {
        format!("tsr.{}", self.dir())
    }

    /// The variant's four kinds of row, selected from the master tables, as the named text form.
    pub fn rows_tsv(self) -> Result<String, String> {
        crate::geneframe::master::select(&self.master_kingdom())?.named_text()
    }
}

/// What a symbol of the alphabet is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsrSymbol<'a> {
    /// An operator, by its `Math` constructor (`Add`, `ProtectedDiv`, `Sin`).
    Operator(&'a str),
    /// Input column `k`.
    Input(usize),
    /// The constant `?`.
    Constant,
}

/// A loaded TSR variant: its rows, entry, contract and decoder, built once.
#[derive(Debug, Clone)]
pub struct TsrRows {
    pub variant: TsrVariant,
    pub n_inputs: usize,
    pub tables: Arc<KingdomTables>,
    pub entry: Entry,
    pub contract: Arc<KingdomContract>,
    pub decoder: Decoder,
}

fn bug(what: String) -> String {
    format!("our software has a bug: {what}")
}

/// The input column a symbol name `x<k>` names.
fn input_column(name: &str) -> Option<usize> {
    name.strip_prefix('x').filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())).and_then(|d| d.parse().ok())
}

impl TsrRows {
    /// Load `variant` for a law of `n_inputs` columns (1 to [`MAX_INPUTS`]): the rows naming an input terminal
    /// `x<k>`, `k >= n_inputs`, are left out (a view of the rows, as a Futhark problem's is of its master), then the
    /// tables are read and checked, the contract built and checked, the decoder built and the entry checked.
    pub fn load(variant: TsrVariant, n_inputs: usize) -> Result<TsrRows, String> {
        if !(1..=MAX_INPUTS).contains(&n_inputs) {
            return Err(bug(format!("TSR {}: a law of {n_inputs} input columns; the rows hold 1 to {MAX_INPUTS}", variant.dir())));
        }
        let text = variant.rows_tsv()?;
        let kept: Vec<&str> = text.lines().filter(|l| l.split('\t').nth(1).and_then(input_column).is_none_or(|k| k < n_inputs)).collect();
        let kingdom = format!("{}/{}", super::TSR, variant.dir());
        let ty = |i: usize| if i == 0 { Ok(Ty::F) } else { Err(bug(format!("{kingdom}: data type {i}; TSR has one, F"))) };
        let (tables, entry) = KingdomTables::parse_named(&kingdom, &kept.join("\n"), &ty)?;
        tables.check().map_err(|d| d.join("\n"))?;
        let contract = relations_from_tables(&tables, vec![entry.output]).check().map_err(|d| d.join("\n"))?;
        let (tables, contract) = (Arc::new(tables), Arc::new(contract));
        let decoder = Decoder::from_arcs(Arc::clone(&tables), Arc::clone(&contract)).map_err(|d| d.join("\n"))?;
        decoder.check_entry(&entry).map_err(|d| d.join("\n"))?;
        Ok(TsrRows { variant, n_inputs, tables, entry, contract, decoder })
    }

    /// What symbol `s` is.
    pub fn symbol(&self, s: SymbolId) -> Option<TsrSymbol<'_>> {
        let row = self.tables.symbol(i64::from(s))?;
        Some(match (row.role, input_column(&row.symbol_name)) {
            (Role::Constant, _) => TsrSymbol::Constant,
            (Role::Function, Some(k)) => TsrSymbol::Input(k),
            (Role::Function, None) => TsrSymbol::Operator(row.alias.as_str()),
            (Role::Variable | Role::Formula, _) => return None,
        })
    }

    /// The gene's layout at head `head` (the single kingdom's: one frame, `F`; [`HEAD`] gives 104 words).
    pub fn layout(&self, head: usize) -> Result<GeneLayout, String> {
        GeneLayout::from_heads(&self.contract, Ty::F, &BTreeMap::from([(Ty::F, head)])).map_err(|e| bug(format!("TSR {}: the layout at head {head}: {e:?}", self.variant.dir())))
    }

    /// Passes 2 to 4 on an expressed tree: every node's properties, and where a `must` guard fails, its substitution.
    pub fn decode(&self, tree: &Tree) -> Result<Decoding, String> {
        self.decoder.decode(&self.entry, tree)
    }
}

/// THE PROGRAM a decoded gene expresses, the tree that is scored: every substituted node replaced by the input it
/// renders as (a TSR guard's only repair is `{in0}`, so never a constant). A node of the program is a node of the
/// expressed tree, with its symbol and its word.
pub fn program(d: &Decoding) -> Tree {
    fn go(n: &Node) -> Tree {
        if let Some(Substituted { by: Substitution::Input(k), .. }) = n.substituted {
            return go(&n.kids[k as usize]);
        }
        let const_index = match n.choice {
            Some(Choice::Dc(w)) => Some(w),
            _ => None,
        };
        Tree { symbol: n.symbol, const_index, kids: n.kids.iter().map(go).collect() }
    }
    go(&d.root)
}

/// How many nodes of the expressed tree a substitution drops (the program's size subtracted from the tree's).
pub fn dropped(tree: &Tree, d: &Decoding) -> usize {
    fn size(t: &Tree) -> usize {
        1 + t.kids.iter().map(size).sum::<usize>()
    }
    size(tree) - size(&program(d))
}

// ---- THE DECODER'S SURFACE FOR PHYLU (plan section 8.8, task T3) --------------------------------------------------

/// One node of the program a TSR gene expresses, in the program's level order: what it is, where in the gene it was
/// read from, where its inputs are, and its derived properties (a node the decoder kept is never substituted, so these
/// are its own).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramNode {
    pub symbol: SymbolId,
    /// The Dc word a constant read; `None` for every other symbol.
    pub const_index: Option<u32>,
    /// The gene position the symbol was read from.
    pub pos: usize,
    /// The level-order index of its first input in the program; its inputs are contiguous from there. 0 for a leaf.
    pub first_child: usize,
    pub arity: usize,
    pub depth: u32,
    pub order: u32,
    pub fam: u32,
    pub arm: u32,
    pub ties: u32,
}

/// What a TSR gene decodes to ([`TsrRows::decode_gene`]): the expressed tree, the program (substitutions applied) in
/// level order, the gene positions of the substituted nodes and how many expressed nodes the program drops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneProgram {
    pub expressed: Tree,
    pub nodes: Vec<ProgramNode>,
    /// Gene positions of the nodes rendered as their first input, ascending: those the program passes through (a
    /// substitution inside an input a substitution drops is silent material, and is not listed).
    pub substituted: Vec<usize>,
    pub dropped: usize,
}

impl GeneProgram {
    /// The program as a tree (equal to [`program`] of the decoding).
    pub fn tree(&self) -> Tree {
        fn go(nodes: &[ProgramNode], i: usize) -> Tree {
            let n = &nodes[i];
            Tree { symbol: n.symbol, const_index: n.const_index, kids: (0..n.arity).map(|j| go(nodes, n.first_child + j)).collect() }
        }
        go(&self.nodes, 0)
    }
}

/// How a TSR symbol's `depth` is derived (its Flow row), in the closed set the device decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthRule {
    /// A leaf: `0`.
    Zero,
    /// A level-keeping unary: `in0.depth`.
    Copy,
    /// A raising unary: `in0.depth + 1`.
    PlusOne,
    /// A binary: `max(in0.depth, in1.depth)`.
    Max,
}

/// How a TSR symbol's stream order (`order`, `fam`, `arm`, `ties`) is derived, in the closed set the device decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderRule {
    /// A leaf: order 1, fam 0, arm 1, ties 1.
    Leaf,
    /// A unary: order and arm its input's order, fam 0, ties 1.
    Unary,
    /// A binary of family `family` (1 sums, 2 products): Horton-Strahler order over the flattened chain (plan 8.1).
    Strahler { family: u32 },
}

/// ONE SYMBOL'S RULES AS THE DEVICE READS THEM, compiled from its Flow and Guard rows: the depth and order rules and
/// the `must` limits, each repaired by `{in0}` (the node rendered as its first input). A row outside these forms is
/// "our software has a bug" at compile time ([`TsrRows::device_rules`]), never a silent approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRule {
    pub depth: DepthRule,
    pub order: OrderRule,
    /// `in0.depth <= L`.
    pub in0_depth_max: Option<u32>,
    /// `in0.order <= L`.
    pub in0_order_max: Option<u32>,
    /// `out.order <= L`.
    pub out_order_max: Option<u32>,
}

impl DeviceRule {
    /// Whether the node fails a limit, given its first input's depth and order and its own derived order.
    pub fn fails(&self, in0_depth: u32, in0_order: u32, out_order: u32) -> bool {
        self.in0_depth_max.is_some_and(|l| in0_depth > l) || self.in0_order_max.is_some_and(|l| in0_order > l) || self.out_order_max.is_some_and(|l| out_order > l)
    }
}

/// The five properties a TSR node carries: depth, order, fam, arm, ties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    pub depth: u32,
    pub order: u32,
    pub fam: u32,
    pub arm: u32,
    pub ties: u32,
}

impl Profile {
    /// A leaf's (the rows' constant and every input terminal).
    pub const LEAF: Profile = Profile { depth: 0, order: 1, fam: 0, arm: 1, ties: 1 };
}

impl DeviceRule {
    /// The profile of a node of this rule over its inputs' profiles (post-substitution), before its limits are read.
    pub fn derive(&self, kids: &[Profile]) -> Profile {
        let depth = kids.iter().map(|k| k.depth).max().unwrap_or(0) + u32::from(self.depth == DepthRule::PlusOne);
        match self.order {
            OrderRule::Leaf => Profile::LEAF,
            OrderRule::Unary => Profile { depth, order: kids[0].order, fam: 0, arm: kids[0].order, ties: 1 },
            OrderRule::Strahler { family } => {
                let view = |k: &Profile| if k.fam == family { (k.arm, k.ties) } else { (k.order, 1) };
                let arm = kids.iter().map(|k| view(k).0).max().unwrap_or(0);
                let ties = kids.iter().map(view).filter(|(v, _)| *v == arm).map(|(_, t)| t).sum::<u32>();
                Profile { depth, order: arm + u32::from(ties >= 2), fam: family, arm, ties }
            }
        }
    }
}

fn formula(text: &str) -> PropertyExpr {
    PropertyExpr::parse(text).unwrap_or_else(|e| panic!("our software has a bug: the TSR rule form {text:?} does not parse: {e}"))
}

impl TsrRows {
    /// THE DEVICE'S RULES, one per symbol id, compiled from the Flow and Guard rows (plan 8.3): each symbol's five up
    /// flows must be one of the closed forms of [`DepthRule`] and [`OrderRule`], and each guard a `must` comparison
    /// `in0.depth <= L`, `in0.order <= L` or `out.order <= L` with one unconditional repair `{in0}` at the node.
    ///
    /// # Errors
    /// "our software has a bug" naming the first row outside those forms.
    pub fn device_rules(&self) -> Result<Vec<DeviceRule>, String> {
        let kingdom = format!("TSR {}", self.variant.dir());
        (0..self.contract.n_symbols() as SymbolId).map(|s| self.device_rule(s).map_err(|e| bug(format!("{kingdom}, symbol {s} ({}): {e}", self.contract.name(s))))).collect()
    }

    fn device_rule(&self, s: SymbolId) -> Result<DeviceRule, String> {
        let id = i64::from(s);
        let mut flows: BTreeMap<&str, &PropertyExpr> = BTreeMap::new();
        for f in self.tables.flows.iter().filter(|f| f.owner == Owner::Symbol(id)) {
            match (&f.name, f.way, f.at) {
                (FlowName::Property(p), Way::Up, Place::Out) => {
                    if flows.insert(p.as_str(), &f.formula).is_some() {
                        return Err(format!("two up flows of {p}"));
                    }
                }
                _ => return Err(format!("a flow outside the device's forms: {:?} {:?} {:?}", f.name, f.way, f.at)),
            }
        }
        let names: Vec<&str> = flows.keys().copied().collect();
        if names != ["arm", "depth", "fam", "order", "ties"] {
            return Err(format!("its up flows are {names:?}, not depth, order, fam, arm and ties"));
        }
        let is = |p: &str, text: &str| flows[p] == &formula(text);
        let arity = self.contract.signature(s).len();
        let (depth, order) = match arity {
            0 => {
                let leaf = is("depth", "0") && is("order", "1") && is("fam", "0") && is("arm", "1") && is("ties", "1");
                if !leaf {
                    return Err("a leaf's flows are not depth 0, order 1, fam 0, arm 1, ties 1".into());
                }
                (DepthRule::Zero, OrderRule::Leaf)
            }
            1 => {
                let depth = if is("depth", "in0.depth") {
                    DepthRule::Copy
                } else if is("depth", "in0.depth + 1") {
                    DepthRule::PlusOne
                } else {
                    return Err(format!("a unary's depth flow {} is neither in0.depth nor in0.depth + 1", flows["depth"]));
                };
                if !(is("order", "in0.order") && is("fam", "0") && is("arm", "in0.order") && is("ties", "1")) {
                    return Err("a unary's order flows are not order in0.order, fam 0, arm in0.order, ties 1".into());
                }
                (depth, OrderRule::Unary)
            }
            2 => {
                if !is("depth", "max(in0.depth, in1.depth)") {
                    return Err(format!("a binary's depth flow {} is not max(in0.depth, in1.depth)", flows["depth"]));
                }
                let family = match flows["fam"] {
                    PropertyExpr::Literal(f @ (1 | 2)) => *f as u32,
                    other => return Err(format!("a binary's fam flow {other} is not 1 (sums) or 2 (products)")),
                };
                let v = |k: u32| format!("if(in{k}.fam = {family}, in{k}.arm, in{k}.order)");
                let t = |k: u32| format!("if(in{k}.fam = {family}, in{k}.ties, 1)");
                let arm = format!("max({}, {})", v(0), v(1));
                let ties = format!("if({} = out.arm, {}, 0) + if({} = out.arm, {}, 0)", v(0), t(0), v(1), t(1));
                if !(is("arm", &arm) && is("ties", &ties) && is("order", "if(2 <= out.ties, out.arm + 1, out.arm)")) {
                    return Err(format!("a binary's arm, ties and order flows are not the Strahler merge of family {family}"));
                }
                (DepthRule::Max, OrderRule::Strahler { family })
            }
            a => return Err(format!("arity {a}: the device decodes leaves, unaries and binaries")),
        };
        let mut rule = DeviceRule { depth, order, in0_depth_max: None, in0_order_max: None, out_order_max: None };
        for g in self.tables.guards.iter().filter(|g| g.owner == Owner::Symbol(id)) {
            let repaired = matches!(g.repairs.as_slice(), [r] if r.site == Site::Node && r.template == "{in0}" && r.unconditional());
            if g.strength != Strength::Must || !repaired {
                return Err(format!("guard {}: not a must guard with the one unconditional repair {{in0}} at the node", g.order));
            }
            let limit = match g.condition.comparison() {
                Some((Relation::Le, left, PropertyExpr::Literal(l))) if *l >= 0 => (left, u32::try_from(*l).map_err(|_| format!("guard {}: limit {l}", g.order))?),
                _ => return Err(format!("guard {}: {} is not `in0.depth`, `in0.order` or `out.order` <= a literal", g.order, g.condition)),
            };
            let slot = match limit.0 {
                PropertyExpr::Input { position: 0, property } if property == "depth" && arity >= 1 => &mut rule.in0_depth_max,
                PropertyExpr::Input { position: 0, property } if property == "order" && arity >= 1 => &mut rule.in0_order_max,
                PropertyExpr::Output(p) if p == "order" && arity >= 1 => &mut rule.out_order_max,
                other => return Err(format!("guard {}: the subject {other} is not in0.depth, in0.order or out.order of a function", g.order)),
            };
            if slot.replace(limit.1).is_some() {
                return Err(format!("guard {}: a second limit on the same subject", g.order));
            }
        }
        Ok(rule)
    }

    /// DECODE A GENE TO ITS PROGRAM (the surface phylu's host twin is tested against): Karva's walk over `layout`
    /// (`const_len`: each type's constant pool length), the decoder's passes, and the program with each node's gene
    /// position, the gene positions of the substituted nodes and the nodes dropped.
    ///
    /// # Errors
    /// A gene that does not fit the layout, or a decoding the rows do not give literal properties for: our software's
    /// bug.
    pub fn decode_gene(&self, layout: &GeneLayout, gene: &[u32], const_len: &BTreeMap<Ty, u32>) -> Result<GeneProgram, String> {
        let e = express(&self.contract, layout, gene, const_len).map_err(|d| bug(format!("TSR {}: the gene does not express: {d}", self.variant.dir())))?;
        let expressed = e.tree();
        let dec = self.decode(&expressed)?;
        // The expressed nodes' gene positions, by path (the walk's level order: a node's inputs follow in order).
        let mut kids: Vec<Vec<(u32, usize)>> = vec![Vec::new(); e.nodes.len()];
        for (i, n) in e.nodes.iter().enumerate() {
            if let (Some(p), Some(a)) = (n.parent_step, n.arg_index) {
                kids[p - 1].push((a, i));
            }
        }
        for k in &mut kids {
            k.sort_unstable();
        }
        // The program, top down: a substituted node is replaced by its input's program; a kept node keeps its gene
        // position. (A node's properties are its own when it is kept.)
        struct Kept<'a> {
            node: &'a Node,
            walk: usize,
            kids: Vec<Kept<'a>>,
        }
        let mut substituted = Vec::new();
        fn keep<'a>(n: &'a Node, w: usize, kids: &[Vec<(u32, usize)>], e: &crate::typed_gep::Expressed, subst: &mut Vec<usize>) -> Result<Kept<'a>, String> {
            if let Some(s) = n.substituted {
                let Substitution::Input(k) = s.by else { return Err("a TSR node substituted by a constant".into()) };
                subst.push(e.nodes[w].pos);
                return keep(&n.kids[k as usize], kids[w][k as usize].1, kids, e, subst);
            }
            let ks = n.kids.iter().zip(&kids[w]).map(|(c, (_, cw))| keep(c, *cw, kids, e, subst)).collect::<Result<Vec<_>, _>>()?;
            Ok(Kept { node: n, walk: w, kids: ks })
        }
        let root = keep(&dec.root, 0, &kids, &e, &mut substituted)?;
        substituted.sort_unstable();
        let prop = |n: &Node, p: &str| match n.properties.get(p) {
            Some(PropertyExpr::Literal(v)) if *v >= 0 => u32::try_from(*v).map_err(|_| format!("{p} = {v}")),
            other => Err(bug(format!("TSR {}: a node's {p} is {other:?}, not a literal", self.variant.dir()))),
        };
        let mut nodes = Vec::new();
        let mut queue = std::collections::VecDeque::from([&root]);
        let mut next = 1usize;
        while let Some(k) = queue.pop_front() {
            let w = &e.nodes[k.walk];
            nodes.push(ProgramNode {
                symbol: k.node.symbol,
                const_index: match k.node.choice {
                    Some(Choice::Dc(c)) => Some(c),
                    _ => None,
                },
                pos: w.pos,
                first_child: if k.kids.is_empty() { 0 } else { next },
                arity: k.kids.len(),
                depth: prop(k.node, "depth")?,
                order: prop(k.node, "order")?,
                fam: prop(k.node, "fam")?,
                arm: prop(k.node, "arm")?,
                ties: prop(k.node, "ties")?,
            });
            next += k.kids.len();
            queue.extend(k.kids.iter());
        }
        let dropped = e.nodes.len() - nodes.len();
        Ok(GeneProgram { expressed, nodes, substituted, dropped })
    }
}

#[cfg(test)]
mod tests;
