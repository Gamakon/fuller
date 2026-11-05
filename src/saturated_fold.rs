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
//!    constructor such as the 1-D Futhark kingdom's `FId`/`FCompose`, or a literal not
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
    /// Whether the e-graph may rewrite this tree. A tree it may not (master 6: one with a failing requirement, since a
    /// rule is an identity under an adapter only where no adapter sits between its sides) is not asserted: its form is
    /// itself and its gene is unchanged.
    fn saturable(&self, _tree: &Tree) -> Result<bool, String> {
        Ok(true)
    }
    /// THE PLACEMENT: an extracted form (nodes spelled by [`EgraphKingdom::row`]) as the tree a gene expresses, every
    /// word written for where it stands (master 6: a variable, extracted as its resolved identity, written as its index
    /// word in the destination's environment). `None` when the form is not a program there (a variable not in scope,
    /// a failing requirement): the individual then keeps its own tree. The form may mix extracted parts with nodes of
    /// a submitted tree, whose words already stand where they are.
    fn place(&self, form: &Tree) -> Result<Option<Tree>, String> {
        Ok(Some(form.clone()))
    }
    /// The variable nodes of a placed form: each node's path and the words that name its variable there.
    fn variables(&self, _form: &Tree) -> Result<Vec<(Vec<u32>, VariableSite)>, String> {
        Ok(Vec::new())
    }
}

/// What names the variable of one variable node of a placed form (master 6) where it stands: every word `w` with
/// `w mod count == place` (`count` 0: no variable of its data type is in scope, and every word reads the data type's
/// constant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariableSite {
    pub place: u32,
    pub count: u32,
}

/// The least word below `limit` that names, at every one of `sites`, the variable each wants (`None` when none does):
/// the word a shared part's variable takes so that it names the same resolved variable wherever the part is used.
pub fn common_word(sites: &[VariableSite], limit: u32) -> Option<u32> {
    let named: Vec<&VariableSite> = sites.iter().filter(|s| s.count > 0).collect();
    let Some(first) = named.first() else { return Some(0) };
    let mut w = first.place;
    while w < limit {
        if named.iter().all(|s| w % s.count == s.place) {
            return Some(w);
        }
        w += first.count;
    }
    None
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
    /// Individuals whose gene words changed. An individual whose program the fold left as it was keeps its gene word
    /// for word; one whose form changed has its own variables at their least index words (the same variables).
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
    /// Individuals kept as submitted: their tree may not be saturated ([`EgraphKingdom::saturable`]), or their
    /// extracted form is not a program where it stands ([`EgraphKingdom::place`]).
    pub kept: usize,
    /// Parts the e-graph shares that are not ADFs, because no one word of one of their variables names the same
    /// resolved variable wherever the part is used ([`common_word`]): each is inlined where it is used.
    pub unshared_by_scope: usize,
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
    // The distinct trees by their e-graph term: the kingdom's import writes a variable as its resolved identity, so
    // two trees whose words name the same variables are one root (the fold's key is the resolved tree).
    let mut distinct: Vec<&Tree> = Vec::new();
    let mut terms: Vec<String> = Vec::new();
    let mut multiplicity: Vec<usize> = Vec::new();
    let mut of_term: HashMap<String, usize> = HashMap::new();
    let mut root_of: Vec<Option<usize>> = Vec::with_capacity(trees.len());
    for t in trees {
        if !kingdom.saturable(t)? {
            root_of.push(None);
            continue;
        }
        let term = kingdom.import(t)?;
        let d = match of_term.get(&term) {
            Some(d) => *d,
            None => {
                of_term.insert(term.clone(), distinct.len());
                distinct.push(t);
                terms.push(term);
                multiplicity.push(0);
                distinct.len() - 1
            }
        };
        multiplicity[d] += 1;
        root_of.push(Some(d));
    }
    if distinct.is_empty() {
        // Nothing may be saturated: every individual is kept as submitted.
        let epoch = Epoch::new(base, &[layout.root], &[]).map_err(|e| e.to_string())?;
        let adfs = AdfSet::new(&epoch, Vec::new()).map_err(|e| e.to_string())?;
        let t = clock.elapsed();
        return Ok(SaturatedFold { epoch, adfs, used: Vec::new(), genes: genes.to_vec(), folded: trees.to_vec(), forms: trees.to_vec(), rewritten: 0, reformed: 0, refitted: 0, refit_adopted: 0, references: 0, structural, distinct: 0, iterations: 0, tuples_asserted: 0, tuples_saturated: 0, stopped_by_size: false, total_cost: 0, structural_dag, fate: StructuralFate::default(), dag: (0, 0), kept: trees.len(), unshared_by_scope: 0, timings: [t_structural, t - t_structural, std::time::Duration::ZERO, std::time::Duration::ZERO] });
    }
    let programs = kingdom.programs();
    let ctors = constructors(&programs);
    let mut egraph = EGraph::default();
    for p in &programs {
        egraph.parse_and_run_program(None, p).map_err(|e| bug(format!("the kingdom's e-graph does not load: {e}")))?;
    }
    let mut assert = String::new();
    for (i, term) in terms.iter().enumerate() {
        assert.push_str(&format!("(let __g{i} {term})\n"));
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
    let lookup = |op: &str, kids: Vec<String>| class_of.get(&(op.to_string(), kids)).cloned();
    // Every subtree of every asserted tree, by its first occurrence, with its e-class: how a structural part finds its
    // e-class (a subtree alone has no environment, so it is never imported by itself). Every asserted tree, not only
    // the first of its term: two trees whose words differ but name the same variables are one root, and each has its
    // own subtrees.
    let mut class_of_subtree: HashMap<&Tree, ClassId> = HashMap::new();
    for (t, d) in trees.iter().zip(&root_of) {
        let Some(d) = d else { continue };
        if class_of_subtree.contains_key(t) {
            continue; // the same tree again: its subtrees are recorded
        }
        fn record<'a>(t: &'a Tree, a: &ATree, out: &mut HashMap<&'a Tree, ClassId>) {
            if let Some(c) = &a.class {
                out.entry(t).or_insert_with(|| c.clone());
            }
            for (kt, ka) in t.kids.iter().zip(&a.kids) {
                record(kt, ka, out);
            }
        }
        record(t, &annotate(t, &parse_sexpr(&terms[*d])?, &lookup)?, &mut class_of_subtree);
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
    let mut shared: BTreeSet<&ClassId> = order.iter().filter(|c| functions[c] >= 1 && edges.get(c).copied().unwrap_or(0) >= 2 && individuals.get(c).copied().unwrap_or(0) >= 2).collect();
    // The expanded tree of every class.
    let mut expanded: BTreeMap<&ClassId, Tree> = BTreeMap::new();
    for c in &order {
        let n = &nodes[c];
        let kids = n.kids.iter().map(|k| expanded[k].clone()).collect();
        expanded.insert(c, Tree { symbol: n.symbol, const_index: n.const_index, kids });
    }
    // Each root's form placed where it stands, and its variables' words (none outside master 6's kingdoms: with none,
    // the scope step below, the refit's scope check and the comparison by resolved tree are skipped, and the fold is
    // what it was before master 6).
    let mut forms_of: Vec<Option<Tree>> = Vec::with_capacity(roots.len());
    let mut variables_of: Vec<BTreeMap<Vec<u32>, VariableSite>> = Vec::with_capacity(roots.len());
    for r in &roots {
        let form = kingdom.place(&expanded[r])?;
        variables_of.push(match &form {
            Some(f) => kingdom.variables(f)?.into_iter().collect(),
            None => BTreeMap::new(),
        });
        forms_of.push(form);
    }
    let has_variables = variables_of.iter().any(|v| !v.is_empty());
    // A shared part's variables take, each, one word that names its resolved variable wherever the part is used (in
    // every placed form, at every occurrence); a part with a variable no word does that for is not an ADF, and is
    // inlined. To a fixed point: inlining a part puts its nodes into the bodies of the parts around it.
    let mut words_of: BTreeMap<&ClassId, BTreeMap<Vec<u32>, u32>> = BTreeMap::new();
    let mut unshared_by_scope = 0usize;
    let mut settled = !has_variables;
    while !settled {
        // Every occurrence of every shared class, by the root form it is in.
        let mut sites: BTreeMap<&ClassId, Vec<(usize, Vec<u32>)>> = BTreeMap::new();
        for (d, r) in roots.iter().enumerate() {
            if forms_of[d].is_none() {
                continue;
            }
            let mut stack: Vec<(&ClassId, Vec<u32>)> = vec![(r, Vec::new())];
            while let Some((c, path)) = stack.pop() {
                if shared.contains(c) {
                    sites.entry(c).or_default().push((d, path.clone()));
                }
                for (i, k) in nodes[c].kids.iter().enumerate() {
                    let mut p = path.clone();
                    p.push(i as u32);
                    stack.push((k, p));
                }
            }
        }
        words_of.clear();
        let mut drop: Option<&ClassId> = None;
        for c in shared.iter().copied() {
            // The body's own nodes: below the part, not inside a part it shares in turn.
            let mut body: Vec<Vec<u32>> = Vec::new();
            let mut stack: Vec<(&ClassId, Vec<u32>)> = vec![(c, Vec::new())];
            while let Some((k, q)) = stack.pop() {
                if !q.is_empty() && shared.contains(k) {
                    continue;
                }
                for (i, kk) in nodes[k].kids.iter().enumerate() {
                    let mut p = q.clone();
                    p.push(i as u32);
                    stack.push((kk, p));
                }
                body.push(q);
            }
            let mut words = BTreeMap::new();
            for q in body {
                let wants: Vec<VariableSite> = sites.get(c).into_iter().flatten().filter_map(|(d, s)| variables_of[*d].get(&[s.as_slice(), q.as_slice()].concat()).copied()).collect();
                if wants.is_empty() {
                    continue;
                }
                match common_word(&wants, crate::typed_gep::DV_WORDS) {
                    Some(w) => {
                        words.insert(q, w);
                    }
                    None => {
                        drop = Some(c);
                        break;
                    }
                }
            }
            if drop.is_some() {
                break;
            }
            words_of.insert(c, words);
        }
        match drop {
            Some(c) => {
                shared.remove(c);
                unshared_by_scope += 1;
            }
            None => settled = true,
        }
    }
    // What became of each part structural hashing shares.
    let mut fate = StructuralFate::default();
    let mut seen_classes: BTreeSet<ClassId> = BTreeSet::new();
    for body in structural_fold.adfs.expanded(&structural_fold.epoch).values() {
        let Some(c) = class_of_subtree.get(body).cloned() else {
            // Every subtree of an asserted tree has its e-class, so a part with none is a part only of trees that
            // may not be saturated (kept as submitted); anything else is a bug.
            let in_kept = trees.iter().zip(&root_of).any(|(t, d)| d.is_none() && holds(t, body));
            if !in_kept {
                return Err(bug("a part structural hashing shares has no e-class, and no tree kept as submitted holds it"));
            }
            fate.unreached += 1;
            continue;
        };
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
    /// The tree of class `c` over the epoch alphabet, the shared parts below it (and `c` itself when `top_too`) as
    /// references; a node's word is `word(its path from c)` where that gives one.
    fn folded<'a>(c: &'a ClassId, top_too: bool, nodes: &'a BTreeMap<ClassId, DagNode>, adf_of: &BTreeMap<&ClassId, AdfId>, epoch: &Epoch, path: &mut Vec<u32>, word: &dyn Fn(&[u32]) -> Option<u32>) -> Result<(Tree, usize), String> {
        if top_too {
            if let Some(a) = adf_of.get(c) {
                let s = epoch.adf_terminal(*a).ok_or_else(|| bug(format!("ADF {a} has no reference terminal")))?;
                return Ok((Tree { symbol: s, const_index: None, kids: Vec::new() }, 1));
            }
        }
        let n = &nodes[c];
        let mut refs = 0;
        let mut kids = Vec::with_capacity(n.kids.len());
        for (i, k) in n.kids.iter().enumerate() {
            path.push(i as u32);
            let (t, m) = folded(k, true, nodes, adf_of, epoch, path, word)?;
            path.pop();
            refs += m;
            kids.push(t);
        }
        Ok((Tree { symbol: n.symbol, const_index: word(path).or(n.const_index), kids }, refs))
    }
    let mut bodies = Vec::new();
    let mut used = Vec::new();
    for (c, d) in shared_order.iter().zip(&decls) {
        let words = words_of.get(*c).cloned().unwrap_or_default();
        let (body, _) = folded(c, false, &nodes, &adf_of, &epoch, &mut Vec::new(), &|q| words.get(q).copied())?;
        bodies.push(Adf::from_tree(&epoch, d.id, d.level, &body));
        used.push((d.id, d.root, d.level, count_nodes(&expanded[c])));
    }
    let adfs = AdfSet::new(&epoch, bodies).map_err(|e| e.to_string())?;
    let cl = ChromosomeLayout { normal: Layout::new(0, vec![layout.clone()]), mains: Vec::new() };
    let (mut out_genes, mut out_folded, mut forms) = (Vec::with_capacity(genes.len()), Vec::with_capacity(genes.len()), Vec::with_capacity(genes.len()));
    let (mut references, mut rewritten, mut reformed, mut refitted) = (0usize, 0usize, 0usize, 0usize);
    let (mut refit_adopted, mut kept) = (0usize, 0usize);
    // Each ADF's body twice: as the gene holds it (its variables by their common words), and as the extraction chose
    // it (its variables by their resolved identities, which `place` writes as the words that name them wherever the
    // body is put).
    let bodies_expanded = adfs.expanded(&epoch);
    let bodies_resolved: BTreeMap<AdfId, Tree> = adf_of.iter().map(|(c, a)| (*a, expanded[c].clone())).collect();
    // A REFIT STEP KEEPS THE PROGRAM. A refit may put a shared part where no counted site stands (inside a subtree the
    // extracted form dropped, `fst (pair a b) -> a`), and there the part's common word may name another variable. A
    // trial tree (epoch alphabet) is taken only when it is a program where it stands with the bodies as extracted, and
    // the gene it writes (the bodies as the gene holds them) names the same variables: otherwise the part stays
    // unshared there (the refit tries its children instead).
    let scope_holds = |trial: &Tree| -> Result<bool, String> {
        if !has_variables {
            return Ok(true);
        }
        let Some(placed) = kingdom.place(&crate::typed_chromosome::unfold_with(&epoch, &bodies_resolved, trial))? else { return Ok(false) };
        let written = crate::typed_chromosome::unfold_with(&epoch, &bodies_expanded, &with_words(&epoch, trial, &placed));
        Ok(kingdom.import(&written)? == kingdom.import(&placed)?)
    };
    for (i, (g, t)) in genes.iter().zip(trees).enumerate() {
        let Some(d) = root_of[i] else {
            kept += 1;
            out_genes.push(g.clone());
            out_folded.push(t.clone());
            forms.push(t.clone());
            continue;
        };
        let Some(placed_form) = forms_of[d].clone() else {
            kept += 1;
            out_genes.push(g.clone());
            out_folded.push(t.clone());
            forms.push(t.clone());
            continue;
        };
        let r = &roots[d];
        let vars = &variables_of[d];
        // A form's own variables take the least word that names their variable where they stand (`w` and
        // `w mod count` name the same variable, so the program is the same); a shared part's take their common word.
        let (mut f, mut n) = folded(r, true, &nodes, &adf_of, &epoch, &mut Vec::new(), &|p| vars.get(p).map(|v| v.place))?;
        let mut form = placed_form;
        // The write-back's domain is the forms that fit the gene's layout (section 5: encodable exactly when
        // `j_T <= h_T` for every type). The domain of a root is the forms of its e-class that fit its layout; a form
        // that does not fit costs infinity for this root, and the other roots keep the joint choice. The joint form is
        // the best when it fits. When it does not, the root's form is searched among the forms of its class that do:
        // starting from the form it was submitted with (one of its class's forms, and one that fits), every subtree,
        // top down, takes its class's joint form, or its reference when the class is shared, whenever the whole still
        // fits and keeps its program (`scope_holds`). Every step is checked before it is taken, so no form outside the
        // domain is ever written.
        let words = match write_back(epoch.kingdom(), layout, g, &f) {
            Ok(w) => w,
            Err(_) => {
                let start = annotate(t, &parse_sexpr(&terms[d])?, &lookup)?;
                let joint = |c: &ClassId| -> Result<Option<Tree>, String> {
                    if adf_of.contains_key(c) {
                        let a = adf_of[c];
                        return Ok(Some(Tree { symbol: epoch.adf_terminal(a).ok_or_else(|| bug(format!("ADF {a} has no reference terminal")))?, const_index: None, kids: Vec::new() }));
                    }
                    if nodes.contains_key(c) {
                        return Ok(Some(folded(c, false, &nodes, &adf_of, &epoch, &mut Vec::new(), &|_| None)?.0));
                    }
                    Ok(None)
                };
                let (tree, adopted) = refit(start, epoch.kingdom(), layout, &joint, &scope_holds)?;
                // The refit form's extracted parts are placed where they now stand, and its own nodes take their words.
                let p = kingdom.place(&crate::typed_chromosome::unfold_with(&epoch, &bodies_resolved, &tree))?.ok_or_else(|| bug(format!("individual {i}: the refit form is no program where it stands, though its start was one and every step was checked")))?;
                refitted += 1;
                refit_adopted += adopted;
                f = with_words(&epoch, &tree, &p);
                n = count_refs(&epoch, &f);
                form = p;
                write_back(epoch.kingdom(), layout, g, &f).map_err(|e| bug(format!("individual {i}: a form checked to fit the layout did not write back: {e}")))?
            }
        };
        // The folded gene expresses its form: the same tree, or (master 6: a shared part's variables taking their
        // common word, an own variable its least word) the same resolved tree. On the direct path the common words
        // were chosen for every counted site, and every refit step was checked, so a difference is a bug either way:
        // never a changed program.
        let c = Chromosome { chromosome_id: i as u64, genes: vec![words.clone()], mains: Vec::new() };
        let p = express_chromosome(&epoch, &adfs, &cl, &c, const_len).map_err(|e| format!("individual {i}: {e}"))?;
        let back = p.inlined(&epoch, &adfs);
        let same = |x: &Tree, y: &Tree| -> Result<bool, String> { Ok(if has_variables { kingdom.import(x)? == kingdom.import(y)? } else { x == y }) };
        if back.len() != 1 || !same(&back[0].1, &form)? {
            return Err(bug(format!("individual {i}: the folded gene does not express its extracted form")));
        }
        // An individual whose program the fold left as it was (the same resolved tree, no reference) keeps its gene
        // word for word: its index words are not moved to their least form, which would erase the neutral variation
        // they carry (`w` and `w mod count` differ once a mutation changes `count`).
        if has_variables && n == 0 && same(&form, t)? {
            out_genes.push(g.clone());
            out_folded.push(t.clone());
            forms.push(t.clone());
            continue;
        }
        references += n;
        rewritten += usize::from(words != *g);
        reformed += usize::from(!same(&form, t)?);
        out_genes.push(words);
        out_folded.push(f);
        forms.push(form);
    }
    Ok(SaturatedFold { epoch, adfs, used, genes: out_genes, folded: out_folded, forms, rewritten, reformed, refitted, refit_adopted, references, structural, distinct: distinct.len(), iterations, tuples_asserted, tuples_saturated, stopped_by_size, total_cost: extracted.total_cost, structural_dag, fate, dag: dag_size, kept, unshared_by_scope, timings: [t_structural, t_saturate - t_structural, t_extract - t_saturate, clock.elapsed() - t_extract] })
}

/// `tree` (over the epoch alphabet) with each of its own nodes' words taken from `form`, its unfolded form: a
/// reference's subtree in `form` is the reference's body, and is skipped.
fn with_words(epoch: &Epoch, tree: &Tree, form: &Tree) -> Tree {
    if epoch.reference(tree.symbol).is_some() {
        return tree.clone();
    }
    Tree { symbol: tree.symbol, const_index: form.const_index, kids: tree.kids.iter().zip(&form.kids).map(|(t, f)| with_words(epoch, t, f)).collect() }
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
/// then on) when the whole tree still fits the layout and `keeps_program` holds for it; otherwise its children are
/// tried. Returns the tree and how many subtrees took their joint form.
fn refit(start: ATree, k: &KingdomContract, layout: &GeneLayout, joint: &dyn Fn(&ClassId) -> Result<Option<Tree>, String>, keeps_program: &dyn Fn(&Tree) -> Result<bool, String>) -> Result<(Tree, usize), String> {
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
                let whole_trial = tree(&trial);
                if crate::typed_gep::fits(k, layout, &whole_trial).is_ok() && keeps_program(&whole_trial)? {
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

/// Whether `sub` is a subtree of `t`.
fn holds(t: &Tree, sub: &Tree) -> bool {
    t == sub || t.kids.iter().any(|k| holds(k, sub))
}

fn count_refs(epoch: &Epoch, t: &Tree) -> usize {
    usize::from(epoch.reference(t.symbol).is_some()) + t.kids.iter().map(|k| count_refs(epoch, k)).sum::<usize>()
}
