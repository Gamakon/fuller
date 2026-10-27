//! Population fold by SATURATED share-equivalence, for any kingdom that has an
//! egglog e-graph (`docs/PLAN_saturated_share_equivalence_v1.md`, applied to
//! the ADFs of `nucleotable/docs/MultiTypedGEP.md` rev 4 section 6).
//!
//! Structural hashing ([`crate::population_fold::AdfTable::fold`]) shares two
//! subtrees only when they are the same tree. Here two subtrees are shared when
//! the kingdom's e-graph proves them equal, so a part one gene spells one way and
//! another gene spells another way is found too. The pipeline is the one
//! `extract::maximal_shared_saturated` runs for SR, on the kingdom's own
//! datatype and ruleset:
//!
//! 1. **The equivalent forms.** The population's distinct expressed trees (the
//!    structural hashing is the fast first step: identical trees are asserted
//!    once, with their multiplicity) are asserted as roots of ONE e-graph and
//!    saturated one iteration at a time, the e-graph's size checked between
//!    iterations (`Budget`), stopping early at a fixpoint.
//! 2. **The joint extraction.** The e-graph is serialized and extracted jointly
//!    over every root with [`crate::extract_dag::extract_shared_dag_with_budget`]
//!    (the marginal-cost fixpoint), so a form wins when other genes already pay
//!    for its parts. A node the kingdom cannot spell as a row (an e-graph-only
//!    constructor such as the 1-D kingdom's `FId`/`FCompose`, or a literal not
//!    in the dual's `C_T`) is given an infinite cost, so the extractor never
//!    chooses it: every chosen form is a tree of kingdom rows.
//! 3. **The ADFs.** In the extracted DAG (one node per e-class), an e-class that
//!    holds a function, has at least two DAG edges into it and is reached from
//!    at least two individuals is a shared part (the population DAG's rule, as
//!    in the structural fold). Each becomes an ADF with a stable id from the
//!    caller's [`AdfTable`] (keyed by its expanded body), its root type and its
//!    level (one more than the deepest shared part below it). Every individual's
//!    gene is rebuilt from its extracted form, the shared parts as `A_d`
//!    references, by the per-type write-back of section 5.
//!
//! A rebuilt form differs from the submitted tree where the e-graph found an
//! equal form that shares better; every rewrite it used is one of the kingdom's
//! rules, each checked bit-identical against Futhark 0.28.1, so the value is the
//! same. Each folded gene is expressed again, inlined, and compared with its
//! extracted form here; the value comparison against the submitted tree is the
//! kingdoms' tests.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use egglog::prelude::exprs;
use egglog::{EGraph, SerializeConfig};
use egraph_serialize::{ClassId, EGraph as SerEGraph, NodeId};

use crate::geneframe::Ty;
use crate::kingdom_contract::{KingdomContract, SymbolId};
use crate::population_fold::{count_nodes, AdfTable, PopulationShare};
use crate::typed_chromosome::{express_chromosome, Adf, AdfDecl, AdfId, AdfSet, Chromosome, ChromosomeLayout, Epoch};
use crate::typed_gep::{write_back, GeneLayout, Layout, Tree};

fn bug(what: impl std::fmt::Display) -> String {
    format!("our software has a bug: {what}")
}

/// A kingdom's e-graph, as the fold needs it.
pub trait EgraphKingdom {
    /// The datatype and the ruleset, as egglog programs, loaded in order.
    fn programs(&self) -> Vec<&str>;
    /// The rulesets one saturation iteration runs, in order.
    fn rulesets(&self) -> Vec<&str>;
    /// An expressed tree (kingdom symbols only) as an egglog term.
    fn import(&self, tree: &Tree) -> Result<String, String>;
    /// The kingdom row a constructor stands for, with its `C_T` index when the
    /// row is a constant placeholder; `literals` are the texts of the node's
    /// primitive children, in order. `None` when the kingdom cannot spell the
    /// node as a row.
    fn row(&self, constructor: &str, literals: &[&str]) -> Option<(SymbolId, Option<u32>)>;
}

/// The saturation and extraction budget.
#[derive(Debug, Clone)]
pub struct Budget {
    /// The most saturation iterations.
    pub iters: u32,
    /// The size guard: saturation stops once the e-graph holds more than
    /// `growth` times its asserted size plus `slack` tuples. A ceiling relative
    /// to the population's own size, because one population of a thousand genes
    /// already asserts more tuples than a fixed SR-sized cap.
    pub growth: usize,
    pub slack: usize,
    /// The marginal-cost fixpoint's round cap.
    pub max_rounds: usize,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget { iters: 8, growth: 4, slack: 5_000, max_rounds: 8 }
    }
}

/// One population folded by saturated share-equivalence.
#[derive(Debug, Clone)]
pub struct SaturatedFold {
    pub epoch: Epoch,
    pub adfs: AdfSet,
    /// The ADFs this population uses: id, root type, level, body nodes (expanded).
    pub used: Vec<(AdfId, Ty, u32, usize)>,
    /// Each individual's gene after the write-back.
    pub genes: Vec<Vec<u32>>,
    /// Each individual's tree over the epoch alphabet: its extracted form with the shared parts as references.
    pub folded: Vec<Tree>,
    /// Each individual's extracted form, unfolded (kingdom symbols only).
    pub forms: Vec<Tree>,
    /// Individuals whose gene words changed.
    pub rewritten: usize,
    /// Individuals whose gene holds a form that is not the tree they were submitted with.
    pub reformed: usize,
    /// Individuals whose jointly extracted form did not fit their layout, so their form was searched among the
    /// forms of their e-class that fit.
    pub refitted: usize,
    /// The subtrees of those genes that took their class's joint form or reference.
    pub refit_adopted: usize,
    /// References written into the genes.
    pub references: usize,
    /// The structural-hashing baseline on the same population (its `adfs` is the count to compare with `used`).
    pub structural: PopulationShare,
    /// Distinct trees asserted, saturation iterations run, the e-graph's tuples before and after, whether the size
    /// guard stopped it, and the joint extraction's cost.
    pub distinct: usize,
    pub iterations: u32,
    pub tuples_asserted: usize,
    pub tuples_saturated: usize,
    pub stopped_by_size: bool,
    pub total_cost: u64,
    /// The DAG the population needs, as the submitted trees spell it (hash-consed: distinct subtrees, and those
    /// whose root is a function) and as the extraction chose it (extracted e-classes, and those holding a function).
    /// The second number of each is the cost both extractions minimise.
    pub structural_dag: (usize, u64),
    /// What became of each part structural hashing shares.
    pub fate: StructuralFate,
    pub dag: (usize, u64),
    /// Wall-clock time of the structural baseline, the saturation, the joint extraction, and the ADFs with the
    /// write-back.
    pub timings: [std::time::Duration; 4],
}

/// What became, in the saturated fold, of each part structural hashing shares (its e-class, found from its body).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuralFate {
    /// Its e-class is a shared part, and no other structural part's is.
    pub matched: usize,
    /// Its e-class is a shared part another structural part also maps to: two spellings of one part.
    pub merged: usize,
    /// Its e-class's chosen form holds no function (`transpose (transpose A)` is `A`), so it is no part at all.
    pub collapsed: usize,
    /// Its e-class holds a function but is no longer shared: the extraction reached it fewer ways.
    pub unshared: usize,
    /// No root's chosen form reaches its e-class.
    pub unreached: usize,
}

/// The `(constructor Name` declarations of the programs.
fn constructors(programs: &[&str]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for p in programs {
        for line in p.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("(constructor ") {
                if let Some(name) = rest.split_whitespace().next() {
                    out.insert(name.to_string());
                }
            }
        }
    }
    out
}

/// The extracted DAG's node of each class, as a kingdom row with its program children.
struct DagNode {
    symbol: SymbolId,
    const_index: Option<u32>,
    kids: Vec<ClassId>,
}

/// Fold a population by saturated share-equivalence. `genes[i]` expresses to `trees[i]` (kingdom symbols only) under
/// `layout`; ADF ids come from `table` (an existing body keeps its id, a new one appends).
pub fn fold_saturated(table: &mut AdfTable, kingdom: &dyn EgraphKingdom, base: &KingdomContract, layout: &GeneLayout, const_len: &BTreeMap<Ty, u32>, genes: &[Vec<u32>], trees: &[Tree], budget: &Budget) -> Result<SaturatedFold, String> {
    if genes.len() != trees.len() {
        return Err(bug("genes and trees differ in number"));
    }
    if trees.is_empty() {
        return Err(bug("an empty population has nothing to fold"));
    }
    // The baseline, on a scratch table so the caller's ids are not spent on it.
    let clock = std::time::Instant::now();
    let structural_fold = AdfTable::default().fold(base, layout, const_len, genes, trees)?;
    let structural = structural_fold.stats.clone();
    let t_structural = clock.elapsed();
    let structural_dag = {
        let mut seen: std::collections::HashSet<&Tree> = std::collections::HashSet::new();
        let mut stack: Vec<&Tree> = trees.iter().collect();
        let mut functions = 0u64;
        while let Some(t) = stack.pop() {
            if seen.insert(t) {
                functions += u64::from(!base.is_terminal(t.symbol));
                stack.extend(t.kids.iter());
            }
        }
        (seen.len(), functions)
    };

    // ---- 1. the equivalent forms --------------------------------------------------------------------------------
    let mut distinct: Vec<&Tree> = Vec::new();
    let mut multiplicity: Vec<usize> = Vec::new();
    let mut of_tree: HashMap<&Tree, usize> = HashMap::new();
    let mut root_of: Vec<usize> = Vec::with_capacity(trees.len());
    for t in trees {
        let d = *of_tree.entry(t).or_insert_with(|| {
            distinct.push(t);
            multiplicity.push(0);
            distinct.len() - 1
        });
        multiplicity[d] += 1;
        root_of.push(d);
    }
    let programs = kingdom.programs();
    let ctors = constructors(&programs);
    let mut egraph = EGraph::default();
    for p in &programs {
        egraph.parse_and_run_program(None, p).map_err(|e| bug(format!("the kingdom's e-graph does not load: {e}")))?;
    }
    let mut assert = String::new();
    for (i, t) in distinct.iter().enumerate() {
        assert.push_str(&format!("(let __g{i} {})\n", kingdom.import(t)?));
    }
    egraph.parse_and_run_program(None, &assert).map_err(|e| bug(format!("the population does not assert: {e}")))?;
    let tuples_asserted = egraph.num_tuples();
    let cap = tuples_asserted.saturating_mul(budget.growth).saturating_add(budget.slack);
    let step = format!("(run-schedule (repeat 1 (seq {})))\n", kingdom.rulesets().iter().map(|r| format!("(run {r})")).collect::<Vec<_>>().join(" "));
    let (mut iterations, mut stopped_by_size) = (0u32, false);
    let mut last = tuples_asserted;
    while iterations < budget.iters {
        egraph.parse_and_run_program(None, &step).map_err(|e| bug(format!("saturation: {e}")))?;
        iterations += 1;
        let now = egraph.num_tuples();
        if now > cap {
            stopped_by_size = true;
            break;
        }
        if now == last {
            break; // a fixpoint: nothing new was proven
        }
        last = now;
    }
    let tuples_saturated = egraph.num_tuples();
    let mut root_values = Vec::with_capacity(distinct.len());
    for i in 0..distinct.len() {
        root_values.push(egraph.eval_expr(&exprs::var(&format!("__g{i}"))).map_err(|e| bug(format!("root {i}: {e}")))?);
    }

    let t_saturate = clock.elapsed();
    // ---- 2. the joint extraction --------------------------------------------------------------------------------
    let mut ser: SerEGraph = egraph.serialize(SerializeConfig { max_functions: None, max_calls_per_function: None, include_temporary_functions: false, root_eclasses: root_values }).egraph;
    let infinite = egraph_serialize::Cost::new(f64::INFINITY).map_err(bug)?;
    let mut spell: HashMap<NodeId, (SymbolId, Option<u32>)> = HashMap::new();
    let ids: Vec<NodeId> = ser.nodes.keys().cloned().collect();
    for id in &ids {
        let node = &ser.nodes[id];
        if !ctors.contains(&node.op) {
            continue; // a primitive: a literal's own node
        }
        let literals: Vec<&str> = node.children.iter().map(|c| ser.nodes[c].op.as_str()).filter(|op| !ctors.contains(*op)).collect();
        match kingdom.row(&node.op, &literals) {
            Some(s) => {
                spell.insert(id.clone(), s);
            }
            None => {
                if let Some(n) = ser.nodes.get_mut(id) {
                    n.cost = infinite;
                }
            }
        }
    }
    // Every node by its constructor and its children (a program child by class, a literal by its value): how a
    // submitted subtree finds its e-class.
    let key_of = |ser: &SerEGraph, op: &str, children: &[NodeId]| -> (String, Vec<String>) {
        (op.to_string(), children.iter().map(|c| { let n = &ser.nodes[c]; if ctors.contains(&n.op) { format!("class {}", n.eclass) } else { literal_key(&n.op) } }).collect())
    };
    let mut class_of: HashMap<(String, Vec<String>), ClassId> = HashMap::new();
    for (_, n) in ser.nodes.iter().filter(|(_, n)| ctors.contains(&n.op)) {
        class_of.insert(key_of(&ser, &n.op, &n.children), n.eclass.clone());
    }
    let roots: Vec<ClassId> = ser.root_eclasses.clone();
    let extracted = crate::extract_dag::extract_shared_dag_with_budget(&ser, &roots, &crate::extract_dag::ExtractionBudget { max_rounds: budget.max_rounds })?;
    let dag = crate::extract_dag::materialize(&ser, &extracted)?;
    let mut nodes: BTreeMap<ClassId, DagNode> = BTreeMap::new();
    for (cid, class) in dag.classes() {
        let nid = class.nodes.first().ok_or_else(|| bug(format!("extracted class {cid} has no node")))?;
        let node = &dag.nodes[nid];
        if !ctors.contains(&node.op) {
            continue;
        }
        let (symbol, const_index) = *spell.get(nid).ok_or_else(|| bug(format!("the extractor chose {}, which the kingdom cannot spell", node.op)))?;
        let kids = node.children.iter().filter(|c| ctors.contains(&dag.nodes[*c].op)).map(|c| dag.nodes[c].eclass.clone()).collect();
        nodes.insert(cid.clone(), DagNode { symbol, const_index, kids });
    }

    let t_extract = clock.elapsed();
    // ---- 3. the ADFs ----------------------------------------------------------------------------------------------
    // Post-order over the DAG from every root: a class after everything below it.
    let mut order: Vec<ClassId> = Vec::new();
    let mut placed: BTreeSet<ClassId> = BTreeSet::new();
    fn post(c: &ClassId, nodes: &BTreeMap<ClassId, DagNode>, placed: &mut BTreeSet<ClassId>, order: &mut Vec<ClassId>) -> Result<(), String> {
        if !placed.insert(c.clone()) {
            return Ok(());
        }
        let n = nodes.get(c).ok_or_else(|| bug(format!("class {c} has no kingdom node")))?;
        for k in &n.kids {
            post(k, nodes, placed, order)?;
        }
        order.push(c.clone());
        Ok(())
    }
    for r in &roots {
        post(r, &nodes, &mut placed, &mut order)?;
    }
    let mut functions: BTreeMap<&ClassId, u64> = BTreeMap::new();
    let mut edges: BTreeMap<&ClassId, usize> = BTreeMap::new();
    for c in &order {
        let n = &nodes[c];
        let f = u64::from(!base.is_terminal(n.symbol)) + n.kids.iter().map(|k| functions[k]).sum::<u64>();
        functions.insert(c, f);
        for k in &n.kids {
            *edges.entry(k).or_insert(0) += 1;
        }
    }
    let mut individuals: BTreeMap<&ClassId, usize> = BTreeMap::new();
    for (d, r) in roots.iter().enumerate() {
        *edges.entry(r).or_insert(0) += multiplicity[d];
        let mut seen = BTreeSet::new();
        let mut stack = vec![r];
        while let Some(c) = stack.pop() {
            if seen.insert(c) {
                stack.extend(nodes[c].kids.iter());
            }
        }
        for c in seen {
            *individuals.entry(c).or_insert(0) += multiplicity[d];
        }
    }
    let dag_size = (order.len(), order.iter().filter(|c| !base.is_terminal(nodes[*c].symbol)).count() as u64);
    let shared: BTreeSet<&ClassId> = order.iter().filter(|c| functions[c] >= 1 && edges.get(c).copied().unwrap_or(0) >= 2 && individuals.get(c).copied().unwrap_or(0) >= 2).collect();
    // What became of each part structural hashing shares.
    let mut fate = StructuralFate::default();
    let mut seen_classes: BTreeSet<ClassId> = BTreeSet::new();
    for body in structural_fold.adfs.expanded(&structural_fold.epoch).values() {
        let c = class_of_term(&parse_sexpr(&kingdom.import(body)?)?, &|op, kids| class_of.get(&(op.to_string(), kids)).cloned())?.ok_or_else(|| bug("a structural part has no e-class"))?;
        if !nodes.contains_key(&c) {
            fate.unreached += 1;
        } else if functions[&c] == 0 {
            fate.collapsed += 1;
        } else if !shared.contains(&c) {
            fate.unshared += 1;
        } else if !seen_classes.insert(c) {
            fate.merged += 1;
        } else {
            fate.matched += 1;
        }
    }
    // The expanded tree of every class.
    let mut expanded: BTreeMap<&ClassId, Tree> = BTreeMap::new();
    for c in &order {
        let n = &nodes[c];
        let kids = n.kids.iter().map(|k| expanded[k].clone()).collect();
        expanded.insert(c, Tree { symbol: n.symbol, const_index: n.const_index, kids });
    }
    let mut adf_of: BTreeMap<&ClassId, AdfId> = BTreeMap::new();
    let mut level: BTreeMap<&ClassId, u32> = BTreeMap::new();
    let mut decls = Vec::new();
    let mut shared_order = Vec::new();
    for c in order.iter().filter(|c| shared.contains(c)) {
        let (id, _) = table.id_of(&expanded[c]);
        fn deepest<'a>(c: &'a ClassId, top: bool, nodes: &'a BTreeMap<ClassId, DagNode>, level: &BTreeMap<&ClassId, u32>) -> Option<u32> {
            if !top {
                if let Some(l) = level.get(c) {
                    return Some(*l);
                }
            }
            nodes[c].kids.iter().filter_map(|k| deepest(k, false, nodes, level)).max()
        }
        let l = deepest(c, true, &nodes, &level).map_or(0, |d| d + 1);
        level.insert(c, l);
        adf_of.insert(c, id);
        decls.push(AdfDecl { id, root: base.output(nodes[c].symbol), level: l });
        shared_order.push(c);
    }
    let epoch = Epoch::new(base, &[layout.root], &decls).map_err(|e| e.to_string())?;
    fn folded<'a>(c: &'a ClassId, top_too: bool, nodes: &'a BTreeMap<ClassId, DagNode>, adf_of: &BTreeMap<&ClassId, AdfId>, epoch: &Epoch) -> Result<(Tree, usize), String> {
        if top_too {
            if let Some(a) = adf_of.get(c) {
                let s = epoch.adf_terminal(*a).ok_or_else(|| bug(format!("ADF {a} has no reference terminal")))?;
                return Ok((Tree { symbol: s, const_index: None, kids: Vec::new() }, 1));
            }
        }
        let n = &nodes[c];
        let mut refs = 0;
        let mut kids = Vec::with_capacity(n.kids.len());
        for k in &n.kids {
            let (t, m) = folded(k, true, nodes, adf_of, epoch)?;
            refs += m;
            kids.push(t);
        }
        Ok((Tree { symbol: n.symbol, const_index: n.const_index, kids }, refs))
    }
    let mut bodies = Vec::new();
    let mut used = Vec::new();
    for (c, d) in shared_order.iter().zip(&decls) {
        let (body, _) = folded(c, false, &nodes, &adf_of, &epoch)?;
        bodies.push(Adf::from_tree(&epoch, d.id, d.level, &body));
        used.push((d.id, d.root, d.level, count_nodes(&expanded[c])));
    }
    let adfs = AdfSet::new(&epoch, bodies).map_err(|e| e.to_string())?;
    let cl = ChromosomeLayout { normal: Layout::new(0, vec![layout.clone()]), mains: Vec::new() };
    let (mut out_genes, mut out_folded, mut forms) = (Vec::with_capacity(genes.len()), Vec::with_capacity(genes.len()), Vec::with_capacity(genes.len()));
    let (mut references, mut rewritten, mut reformed, mut refitted) = (0usize, 0usize, 0usize, 0usize);
    let mut refit_adopted = 0usize;
    let bodies_expanded = adfs.expanded(&epoch);
    for (i, (g, t)) in genes.iter().zip(trees).enumerate() {
        let r = &roots[root_of[i]];
        let (mut f, mut n) = folded(r, true, &nodes, &adf_of, &epoch)?;
        let mut form = expanded[r].clone();
        // The write-back's domain is the forms that fit the gene's layout (section 5: encodable exactly when
        // `j_T <= h_T` for every type). An extracted form that does not fit is outside it, so the gene keeps its own
        // spelling with its shared parts as references: a removal-only rewrite, which always fits.
        // The domain of a root is the forms of its e-class that fit its layout (section 5: encodable exactly when
        // `j_T <= h_T` for every type); a form that does not fit costs infinity for this root, and the other roots
        // keep the joint choice. The joint form is the best when it fits. When it does not, the root's form is
        // searched among the forms of its class that do: starting from the form it was submitted with (one of its
        // class's forms, and one that fits), every subtree, top down, takes its class's joint form, or its reference
        // when the class is shared, whenever the whole still fits. Every step is checked before it is taken, so no
        // form outside the domain is ever written.
        let words = match write_back(epoch.kingdom(), layout, g, &f) {
            Ok(w) => w,
            Err(_) => {
                let sexpr = parse_sexpr(&kingdom.import(t)?)?;
                let start = annotate(t, &sexpr, &|op, kids| class_of.get(&(op.to_string(), kids)).cloned())?;
                let joint = |c: &ClassId| -> Result<Option<Tree>, String> {
                    if adf_of.contains_key(c) {
                        let a = adf_of[c];
                        return Ok(Some(Tree { symbol: epoch.adf_terminal(a).ok_or_else(|| bug(format!("ADF {a} has no reference terminal")))?, const_index: None, kids: Vec::new() }));
                    }
                    if nodes.contains_key(c) {
                        return Ok(Some(folded(c, false, &nodes, &adf_of, &epoch)?.0));
                    }
                    Ok(None)
                };
                let (tree, adopted) = refit(start, epoch.kingdom(), layout, &joint)?;
                refitted += 1;
                refit_adopted += adopted;
                n = count_refs(&epoch, &tree);
                f = tree;
                form = crate::typed_chromosome::unfold_with(&epoch, &bodies_expanded, &f);
                write_back(epoch.kingdom(), layout, g, &f).map_err(|e| bug(format!("individual {i}: a form checked to fit the layout did not write back: {e}")))?
            }
        };
        let c = Chromosome { chromosome_id: i as u64, genes: vec![words.clone()], mains: Vec::new() };
        let p = express_chromosome(&epoch, &adfs, &cl, &c, const_len).map_err(|e| format!("individual {i}: {e}"))?;
        let back = p.inlined(&epoch, &adfs);
        if back.len() != 1 || back[0].1 != form {
            return Err(bug(format!("individual {i}: the folded gene does not express its extracted form")));
        }
        references += n;
        rewritten += usize::from(words != *g);
        reformed += usize::from(form != *t);
        out_genes.push(words);
        out_folded.push(f);
        forms.push(form);
    }
    Ok(SaturatedFold { epoch, adfs, used, genes: out_genes, folded: out_folded, forms, rewritten, reformed, refitted, refit_adopted, references, structural, distinct: distinct.len(), iterations, tuples_asserted, tuples_saturated, stopped_by_size, total_cost: extracted.total_cost, structural_dag, fate, dag: dag_size, timings: [t_structural, t_saturate - t_structural, t_extract - t_saturate, clock.elapsed() - t_extract] })
}

/// A literal child's key: its value when it is a number, so `0.5` asserted and `0.5` serialized agree however each
/// was spelled.
fn literal_key(op: &str) -> String {
    match op.parse::<f64>() {
        Ok(v) => format!("literal {:?}", v.to_bits()),
        Err(_) => format!("literal {op}"),
    }
}

/// An egglog term: a constructor application or an atom.
#[derive(Debug, Clone)]
enum Sexpr {
    App(String, Vec<Sexpr>),
    Atom(String),
}

fn parse_sexpr(text: &str) -> Result<Sexpr, String> {
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let mut tokens = spaced.split_whitespace();
    fn one<'a>(first: &'a str, rest: &mut dyn Iterator<Item = &'a str>) -> Result<Sexpr, String> {
        if first != "(" {
            return Ok(Sexpr::Atom(first.to_string()));
        }
        let op = rest.next().ok_or_else(|| bug("a term ends after an open bracket"))?.to_string();
        let mut kids = Vec::new();
        loop {
            match rest.next() {
                Some(")") => return Ok(Sexpr::App(op, kids)),
                Some(t) => kids.push(one(t, rest)?),
                None => return Err(bug("a term is not closed")),
            }
        }
    }
    let first = tokens.next().ok_or_else(|| bug("an empty term"))?;
    one(first, &mut tokens)
}

/// The e-class of a submitted term: found by its constructor and its children's classes.
fn class_of_term(s: &Sexpr, class: &dyn Fn(&str, Vec<String>) -> Option<ClassId>) -> Result<Option<ClassId>, String> {
    let Sexpr::App(op, kids) = s else { return Ok(None) };
    let mut keys = Vec::with_capacity(kids.len());
    for k in kids {
        match k {
            Sexpr::Atom(a) => keys.push(literal_key(a)),
            Sexpr::App(..) => match class_of_term(k, class)? {
                Some(c) => keys.push(format!("class {c}")),
                None => return Err(bug(format!("a submitted subtree of {op} has no e-class"))),
            },
        }
    }
    Ok(class(op, keys))
}


/// A tree with each node's e-class (`None` for a reference).
#[derive(Clone)]
struct ATree {
    symbol: SymbolId,
    const_index: Option<u32>,
    class: Option<ClassId>,
    kids: Vec<ATree>,
}

impl ATree {
    fn tree(&self) -> Tree {
        Tree { symbol: self.symbol, const_index: self.const_index, kids: self.kids.iter().map(ATree::tree).collect() }
    }
}

/// A submitted tree with the e-class of every node.
fn annotate(t: &Tree, s: &Sexpr, class: &dyn Fn(&str, Vec<String>) -> Option<ClassId>) -> Result<ATree, String> {
    let c = class_of_term(s, class)?.ok_or_else(|| bug("a submitted subtree has no e-class"))?;
    let Sexpr::App(_, kids) = s else { return Err(bug("a submitted tree is an atom")) };
    let program: Vec<&Sexpr> = kids.iter().filter(|k| matches!(k, Sexpr::App(..))).collect();
    if program.len() != t.kids.len() {
        return Err(bug("a submitted tree and its term differ in arity"));
    }
    let kids = t.kids.iter().zip(program).map(|(kt, ks)| annotate(kt, ks, class)).collect::<Result<Vec<_>, _>>()?;
    Ok(ATree { symbol: t.symbol, const_index: t.const_index, class: Some(c), kids })
}

/// The refit search: from `start` (which fits), each subtree in level order takes `joint(its class)` (fixed from
/// then on) when the whole tree still fits the layout; otherwise its children are tried. Returns the tree and how
/// many subtrees took their joint form.
fn refit(start: ATree, k: &KingdomContract, layout: &GeneLayout, joint: &dyn Fn(&ClassId) -> Result<Option<Tree>, String>) -> Result<(Tree, usize), String> {
    crate::typed_gep::fits(k, layout, &start.tree()).map_err(|e| bug(format!("the submitted form does not fit its own layout: {e}")))?;
    /// A tree whose fixed parts are plain trees and whose open parts are still annotated.
    #[derive(Clone)]
    enum Part {
        Fixed(Tree),
        Open(SymbolId, Option<u32>, Option<ClassId>, Vec<Part>),
    }
    fn tree(p: &Part) -> Tree {
        match p {
            Part::Fixed(t) => t.clone(),
            Part::Open(s, c, _, kids) => Tree { symbol: *s, const_index: *c, kids: kids.iter().map(tree).collect() },
        }
    }
    fn open(a: &ATree) -> Part {
        Part::Open(a.symbol, a.const_index, a.class.clone(), a.kids.iter().map(open).collect())
    }
    fn at<'a>(p: &'a mut Part, path: &[usize]) -> &'a mut Part {
        match (path.split_first(), p) {
            (None, p) => p,
            (Some((i, rest)), Part::Open(_, _, _, kids)) => at(&mut kids[*i], rest),
            (Some(_), p) => p,
        }
    }
    let mut whole = open(&start);
    let mut adopted = 0;
    let mut queue: std::collections::VecDeque<Vec<usize>> = std::collections::VecDeque::from([Vec::new()]);
    while let Some(path) = queue.pop_front() {
        let here = at(&mut whole, &path).clone();
        let Part::Open(_, _, class, kids) = &here else { continue };
        let Some(c) = class else { continue };
        if let Some(j) = joint(c)? {
            if j != tree(&here) {
                let mut trial = whole.clone();
                *at(&mut trial, &path) = Part::Fixed(j);
                if crate::typed_gep::fits(k, layout, &tree(&trial)).is_ok() {
                    whole = trial;
                    adopted += 1;
                    continue;
                }
            }
        }
        for i in 0..kids.len() {
            let mut p = path.clone();
            p.push(i);
            queue.push_back(p);
        }
    }
    Ok((tree(&whole), adopted))
}

fn count_refs(epoch: &Epoch, t: &Tree) -> usize {
    usize::from(epoch.reference(t.symbol).is_some()) + t.kids.iter().map(|k| count_refs(epoch, k)).sum::<usize>()
}
