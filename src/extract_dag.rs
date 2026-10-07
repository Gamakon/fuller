//! MARGINAL-COST FIXPOINT extraction over a saturated, serialized egglog
//! e-graph: pick exactly one winning e-node per e-class, jointly across
//! every gene root, so that forms that are only cheaper BECAUSE another
//! root already "pays for" part of them are found — e.g. `exp(x+y)`
//! versus `exp(x)*exp(y)`, where the multiplicative form only wins once
//! `exp(x)` and `exp(y)` are known to be covered by other genes.
//!
//! `docs/PLAN_saturated_share_equivalence_v1.md` §2 is the design this
//! implements; read it for the full worked arithmetic on the motivating
//! case. In short: plain greedy-DAG extraction (price each e-class only by
//! what is reachable *through it*, independently per root) is the WRONG
//! algorithm here — on the motivating case it picks `exp(x+y)` (cost 2)
//! over `exp(x)*exp(y)` (cost 3), exactly backwards, because it has no way
//! to know `exp(x)`/`exp(y)` are already covered by other roots. This
//! module instead runs a bounded, iterative MARGINAL-cost heuristic: price
//! a candidate relative to what the OTHER roots' current choices already
//! cover, and iterate — explicitly a heuristic, not a provably-optimal
//! algorithm (see `extract_shared_dag`'s own doc for the oscillation risk
//! and the round cap/best-seen safeguard).
//!
//! Not a port of `extraction-gym` (the reference greedy-DAG crate in the
//! egg/egglog ecosystem): that crate is bin-only upstream (no `lib.rs`,
//! cannot be a Cargo dependency), and its algorithm is the wrong one for
//! this use case regardless. Only its data-structure shape (operating
//! directly on `egraph_serialize` types) is reused as a model.

use egraph_serialize::{ClassId, EGraph as SerEGraph, Node, NodeId};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// A concrete, acyclic extraction result: exactly one winning `NodeId` per
/// `ClassId` reachable from `roots`, plus the TOTAL joint cost (every
/// class in the union of every root's reachable set, counted once) — this
/// total IS the per-individual DAG cost a caller (phylu) exposes for
/// fitness/selection use, so it is a first-class field here, not
/// recomputed by a separate pass.
#[derive(Debug, Clone)]
pub struct ExtractedDag {
    pub chosen: HashMap<ClassId, NodeId>,
    pub roots: Vec<ClassId>,
    pub total_cost: u64,
}

/// 0 for `Num`/`Var` leaves, 0 for any other zero-children (primitive)
/// node, 1 for every other op — matches `extract::internal_op_count`'s
/// existing convention exactly: that function stops recursing AT a
/// `Num`/`Var` node and never visits its own child (e.g. `Var`'s string
/// literal, serialized as a SEPARATE node/class of its own by egglog's
/// `serialize()`), so that string-literal child is never counted by it
/// either. This extractor DOES visit every class, including those string
/// -literal primitive nodes, since they are genuinely reachable classes
/// — so they must be explicitly costed at 0 (not implicitly skipped by
/// never being visited), or every `Var`/`Num` leaf would silently cost 1
/// extra for its own literal payload, inflating every total by one per
/// leaf. Skips (by returning `None`, so callers never silently use a
/// bogus cost) nodes that are `subsumed` or carry a non-finite serialized
/// cost (truncation placeholders egglog's `serialize()` can emit under
/// `max_functions`/`max_calls_per_function` — not reachable via this
/// crate's own call sites today, which always pass `None`/`None`, but
/// guarded against regardless since this function does not control its
/// caller's config).
fn op_cost(node: &Node) -> Option<u64> {
    if node.subsumed || !node.cost.into_inner().is_finite() {
        return None;
    }
    Some(if node.op == "Num" || node.op == "Var" || node.children.is_empty() { 0 } else { 1 })
}

/// One e-class's current best candidate during a round: which node wins,
/// and the FULL set of classes reachable through it (own class included)
/// — the reachable SET, not a tree-summed cost, is what makes reuse free:
/// a class counted in two different candidates' reachable sets still only
/// contributes its own op_cost once to a joint total built from the union
/// of those sets.
#[derive(Clone, Debug)]
struct Candidate {
    node: NodeId,
    reachable: BTreeSet<ClassId>,
}

/// Bottom-up worklist (Bellman-Ford style — e-graphs built under
/// commutative/associative rules can contain cycles among classes, so
/// this cannot be a single topological pass) computing, for every class
/// reachable from `roots`, the candidate node that minimizes cost under
/// FREE REUSE: a node's cost = its own op_cost + the sum of its
/// children's CURRENT best candidate cost, each child's class counted
/// once even if reachable multiple ways.
///
/// `root_reach` is EMPTY for round 0 (plain free-reuse, no marginal
/// pricing yet). For later rounds, `extract_shared_dag` passes the
/// PREVIOUS round's PER-ROOT reachable sets (one `BTreeSet<ClassId>` per
/// root, not a flattened union) — this is the fix for a real bug a flat
/// union had: pricing class `k` as "free" for target class `c` must mean
/// "SOME OTHER root already pays for `k`," never "c's own previous choice
/// happens to include k." A flat union cannot distinguish those two
/// cases — it let a root's own prior pick make its own subtree free to
/// itself, which is circular and silently prevented a strictly cheaper
/// joint form from ever winning (confirmed: it produced a tie, not a
/// strict improvement, so the cheaper form never displaced the prior
/// round's choice). The real rule, computed by `rank` from `root_reach`
/// directly: `k` is free to `c` iff some root `r` has `k` in
/// `root_reach[r]` but does NOT have `c` in `root_reach[r]`.
///
/// Every child is resolved via `ser.nodes[child].eclass` — NEVER by
/// trusting a `children[i]` `NodeId` as a stable reference across calls.
/// This matters once saturation has run: egglog's own `serialize()`
/// resolves a child edge by round-robin ROTATION over that child class's
/// node pool (`vendor-egglog/src/serialize.rs::serialize_value`), so two
/// different parent edges into the SAME class can carry two DIFFERENT
/// `NodeId`s. Re-resolving through `.eclass` every time sidesteps this
/// entirely; trusting the raw `NodeId` would not.
fn bottom_up_pass(ser: &SerEGraph, roots: &[ClassId], root_reach: &HashMap<ClassId, BTreeSet<ClassId>>) -> Result<HashMap<ClassId, Candidate>, String> {
    let mut best: HashMap<ClassId, Candidate> = HashMap::new();

    // Reverse index: ClassId -> every NodeId that has at least one child
    // resolving to that class (via `.eclass`, never the raw `children[i]`).
    let mut depends_on: HashMap<ClassId, Vec<NodeId>> = HashMap::new();
    for (nid, node) in &ser.nodes {
        if node.subsumed {
            continue;
        }
        let mut seen_child_classes: HashSet<ClassId> = HashSet::new();
        for child in &node.children {
            let cid = ser.nodes[child].eclass.clone();
            if seen_child_classes.insert(cid.clone()) {
                depends_on.entry(cid).or_default().push(nid.clone());
            }
        }
    }

    // Collect every class reachable from `roots` (BFS over classes, via
    // `.eclass` on every member node's children) -- the worklist only
    // ever needs to consider classes in this set.
    let mut reachable_classes: HashSet<ClassId> = HashSet::new();
    let mut frontier: VecDeque<ClassId> = roots.iter().cloned().collect();
    while let Some(cid) = frontier.pop_front() {
        if !reachable_classes.insert(cid.clone()) {
            continue;
        }
        let Some(class) = ser.classes().get(&cid) else {
            continue;
        };
        for nid in &class.nodes {
            let node = &ser.nodes[nid];
            if node.subsumed {
                continue;
            }
            for child in &node.children {
                frontier.push_back(ser.nodes[child].eclass.clone());
            }
        }
    }

    fn try_candidate(ser: &SerEGraph, best: &HashMap<ClassId, Candidate>, node_id: &NodeId) -> Option<Candidate> {
        let node = &ser.nodes[node_id];
        if node.subsumed {
            return None;
        }
        let own_class = node.eclass.clone();
        let mut reachable: BTreeSet<ClassId> = BTreeSet::new();
        reachable.insert(own_class.clone());
        let mut child_classes: HashSet<ClassId> = HashSet::new();
        for child in &node.children {
            let cid = ser.nodes[child].eclass.clone();
            if !child_classes.insert(cid.clone()) {
                continue;
            }
            let child_best = best.get(&cid)?; // not ready yet: this node's candidate can't be built this round
            reachable.extend(child_best.reachable.iter().cloned());
        }
        // A cycle through already-decided classes: a child's reachable
        // set already contained this node's own class before we added it
        // ourselves. Never choose it.
        if !node.children.is_empty() {
            let would_recount = node.children.iter().filter_map(|c| best.get(&ser.nodes[c].eclass)).any(|cand| cand.reachable.contains(&own_class));
            if would_recount {
                return None;
            }
        }
        Some(Candidate { node: node_id.clone(), reachable })
    }

    // Seed: every leaf node (Num/Var, zero children) is an immediate
    // candidate for its class.
    let mut worklist: VecDeque<ClassId> = VecDeque::new();
    for cid in &reachable_classes {
        let Some(class) = ser.classes().get(cid) else { continue };
        for nid in &class.nodes {
            let node = &ser.nodes[nid];
            if node.subsumed || !node.children.is_empty() {
                continue;
            }
            if let Some(cand) = try_candidate(ser, &best, nid) {
                let new_rank = rank(ser, &best, &cand, cid, root_reach);
                let better = best.get(cid).map(|b| cmp_candidates(new_rank, &cand.node, rank(ser, &best, b, cid, root_reach), &b.node)).unwrap_or(true);
                if better {
                    best.insert(cid.clone(), cand);
                }
            }
        }
        worklist.push_back(cid.clone());
    }

    while let Some(cid) = worklist.pop_front() {
        let Some(dependents) = depends_on.get(&cid).cloned() else { continue };
        for nid in dependents {
            let target_class = ser.nodes[&nid].eclass.clone();
            if !reachable_classes.contains(&target_class) {
                continue;
            }
            if let Some(cand) = try_candidate(ser, &best, &nid) {
                let new_rank = rank(ser, &best, &cand, &target_class, root_reach);
                let better = best
                    .get(&target_class)
                    .map(|b| cmp_candidates(new_rank, &cand.node, rank(ser, &best, b, &target_class, root_reach), &b.node))
                    .unwrap_or(true);
                if better {
                    best.insert(target_class.clone(), cand);
                    worklist.push_back(target_class);
                }
            }
        }
    }

    for cid in &reachable_classes {
        if !best.contains_key(cid) {
            return Err(format!("class {cid} has no leaf-bottomed path: fully cyclic, no candidate found"));
        }
    }
    Ok(best)
}

/// Is class `k` FREE to class `c` — i.e. does paying for `k` cost `c`
/// nothing, because some OTHER root already pays for it regardless of
/// what `c` picks? True iff some root `r`'s reach (from `root_reach`,
/// the PREVIOUS round's per-root reachable sets) contains `k` but does
/// NOT contain `c` — "a root whose own subtree doesn't even touch `c`
/// already needs `k` anyway." If `k`'s only coverage comes from roots
/// that also reach `c` itself, `k` is NOT free (that root's "coverage"
/// of `k` could be entirely a consequence of `c`'s OWN current choice,
/// which is circular — exactly the bug a flat global `covered` union had
/// before this function existed).
fn is_free_to(root_reach: &HashMap<ClassId, BTreeSet<ClassId>>, k: &ClassId, c: &ClassId) -> bool {
    root_reach.values().any(|reach| reach.contains(k) && !reach.contains(c))
}

/// A candidate's rank for comparison: its cost, with any class free to
/// `target_class` (per `is_free_to`, using the PREVIOUS round's per-root
/// reach) contributing 0 instead of its own op_cost. This is the
/// "marginal cost relative to the OTHER roots' current choices" pricing;
/// with `root_reach` empty (round 0) nothing is free yet, identical to
/// plain free-reuse cost.
///
/// CRITICAL: every class in `cand.reachable` other than `target_class`
/// MUST be costed via `best`'s CURRENT winning node for that class, never
/// via `ser.classes()[cid].nodes.first()` — a class can have more than
/// one member (that is the entire reason this module exists), and
/// `nodes.first()` returns an ARBITRARY one, not necessarily the one any
/// candidate actually depends on. `target_class` itself is costed via
/// `cand.node` (the node THIS candidate represents), since `best` has not
/// been updated with it yet at the point this is called. Every other
/// class in `reachable` is guaranteed to already have a `best` entry by
/// the time this runs (`try_candidate` only includes a child's reachable
/// set after confirming `best.get(&cid)` succeeded).
fn rank(ser: &SerEGraph, best: &HashMap<ClassId, Candidate>, cand: &Candidate, target_class: &ClassId, root_reach: &HashMap<ClassId, BTreeSet<ClassId>>) -> u64 {
    cand.reachable
        .iter()
        .filter(|cid| *cid == target_class || !is_free_to(root_reach, cid, target_class))
        .filter_map(|cid| {
            let nid = if cid == target_class { &cand.node } else { &best.get(cid)?.node };
            op_cost(&ser.nodes[nid])
        })
        .sum()
}

/// Deterministic candidate comparison: strictly lower rank wins; on a
/// tie, the LOWER `NodeId` wins (via its `Ord` impl) — never leave a tie
/// to depend on iteration/insertion order.
fn cmp_candidates(new_rank: u64, new_node: &NodeId, old_rank: u64, old_node: &NodeId) -> bool {
    (new_rank, new_node) < (old_rank, old_node)
}

const MAX_ROUNDS: usize = 8;

/// MARGINAL-COST FIXPOINT, jointly over ALL roots — see this module's doc
/// and `docs/PLAN_saturated_share_equivalence_v1.md` §2 for why plain
/// greedy-DAG (independent per-root pricing) is the wrong algorithm here.
///
/// Round 0: plain free-reuse bottom-up pass (`root_reach` empty) — a
/// baseline, not the answer. Each subsequent round re-runs the SAME
/// bottom-up pass with `root_reach` set to the PREVIOUS round's per-root
/// reachable sets (keyed by root `ClassId`, NOT flattened into one union
/// — see `bottom_up_pass`'s doc for why a flat union is unsound: it lets
/// a root's own prior choice make its own subtree look "free" to itself).
/// This is explicitly a HEURISTIC, not a provably-terminating fixpoint:
/// pricing each root against the others' current picks can oscillate
/// rather than settle, so this keeps the best `total_cost` seen across
/// all rounds (not just the last), caps at `MAX_ROUNDS`, and stops early
/// if a round's per-root reach set repeats one already seen (a detected
/// cycle, not just "unchanged from last round").
pub fn extract_shared_dag(ser: &SerEGraph, roots: &[ClassId]) -> Result<ExtractedDag, String> {
    let mut root_reach: HashMap<ClassId, BTreeSet<ClassId>> = HashMap::new();
    let mut seen_root_reach: HashSet<Vec<(ClassId, BTreeSet<ClassId>)>> = HashSet::new();
    let mut best_total: Option<(u64, HashMap<ClassId, NodeId>)> = None;

    for _round in 0..MAX_ROUNDS {
        let pass = bottom_up_pass(ser, roots, &root_reach)?;
        let mut new_root_reach: HashMap<ClassId, BTreeSet<ClassId>> = HashMap::new();
        let mut covered: BTreeSet<ClassId> = BTreeSet::new();
        let mut chosen: HashMap<ClassId, NodeId> = HashMap::new();
        for root in roots {
            let cand = pass.get(root).ok_or_else(|| format!("root class {root} has no candidate"))?;
            new_root_reach.insert(root.clone(), cand.reachable.clone());
            covered.extend(cand.reachable.iter().cloned());
        }
        // Record every reachable class's chosen node (not just roots'
        // own direct candidate), so `materialize` can rewrite every
        // child edge, not just root edges.
        for cid in &covered {
            if let Some(cand) = pass.get(cid) {
                chosen.insert(cid.clone(), cand.node.clone());
            }
        }
        let total_cost: u64 = covered.iter().filter_map(|cid| chosen.get(cid)).filter_map(|nid| op_cost(&ser.nodes[nid])).sum();

        let better = best_total.as_ref().map(|(best, _)| total_cost < *best).unwrap_or(true);
        if better {
            best_total = Some((total_cost, chosen));
        }

        let snapshot: Vec<(ClassId, BTreeSet<ClassId>)> = {
            let mut v: Vec<_> = new_root_reach.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        if !seen_root_reach.insert(snapshot) {
            break; // cycle detected: this per-root reach set was already tried
        }
        root_reach = new_root_reach;
    }

    let (total_cost, chosen) = best_total.ok_or_else(|| "no round produced a candidate".to_string())?;
    Ok(ExtractedDag { chosen, roots: roots.to_vec(), total_cost })
}

/// Rebuild `extracted` as a fresh, concrete `egraph_serialize::EGraph`:
/// one node per class (the winner), every `children[i]` REWRITTEN to
/// point at that child's class's winning `NodeId` (resolved via the
/// ORIGINAL `ser`'s `.eclass`, then `extracted.chosen`) — NOT copied
/// verbatim from `ser`, which would silently reintroduce the rotation
/// hazard this module's doc describes (`ser`'s own `children` fields were
/// populated by the SAME rotating logic). This is the object
/// `extract::shared_sites_in` walks; one member per class makes
/// `class.nodes.first()` unambiguous by construction, so no further
/// `ClassId`-rekeying is needed downstream.
pub fn materialize(ser: &SerEGraph, extracted: &ExtractedDag) -> SerEGraph {
    let mut out = SerEGraph::default();
    for (cid, nid) in &extracted.chosen {
        let node = &ser.nodes[nid];
        let new_children: Vec<NodeId> = node
            .children
            .iter()
            .map(|c| {
                let child_class = &ser.nodes[c].eclass;
                extracted.chosen.get(child_class).cloned().unwrap_or_else(|| c.clone())
            })
            .collect();
        out.add_node(
            nid.clone(),
            Node { op: node.op.clone(), children: new_children, eclass: cid.clone(), cost: node.cost, subsumed: false },
        );
    }
    out.root_eclasses = extracted.roots.clone();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(id: &str, class: &str, op: &str) -> (NodeId, Node) {
        (id.into(), Node { op: op.to_string(), children: vec![], eclass: class.into(), cost: egraph_serialize::Cost::new(1.0).unwrap(), subsumed: false })
    }

    fn op_node(id: &str, class: &str, op: &str, children: &[&str]) -> (NodeId, Node) {
        (
            id.into(),
            Node {
                op: op.to_string(),
                children: children.iter().map(|c| NodeId::from(*c)).collect(),
                eclass: class.into(),
                cost: egraph_serialize::Cost::new(1.0).unwrap(),
                subsumed: false,
            },
        )
    }

    /// `op_cost`'s `node.children.is_empty() -> 0` rule is sound only
    /// because `MATH_DATATYPE` (src/expr.rs) has exactly two zero-Math
    /// -children constructors, `Num` and `Var`, and both are already
    /// covered by the `op == "Num" || op == "Var"` check -- the
    /// `children.is_empty()` clause exists ONLY to zero-cost their own
    /// serialized primitive-payload nodes (a `Var`'s string literal, a
    /// `Num`'s float literal), which `op_cost` is called on directly but
    /// `extract::internal_op_count` never visits (it stops recursing at
    /// `Num`/`Var` itself). If a future constructor took a bare `Math`
    /// -sort zero-arity leaf that was NOT `Num`/`Var` (e.g. a named
    /// constant symbol), this rule would silently cost it 0 instead of
    /// its real cost -- this test is the regression guard: it pins
    /// today's actual arity-0 constructor set, so a future datatype
    /// change that violates the assumption fails loudly here instead of
    /// silently miscosting in `extract_dag`.
    #[test]
    fn every_zero_math_arity_constructor_is_num_or_var() {
        for line in crate::expr::MATH_DATATYPE.lines() {
            let trimmed = line.trim_start().trim_start_matches('(');
            let Some(name) = trimmed.split_whitespace().next() else { continue };
            if !trimmed.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                continue; // not a constructor line (datatype header, comment, etc.)
            }
            if name == "Num" || name == "Var" {
                continue;
            }
            assert!(
                trimmed.contains("Math"),
                "{name} takes no Math argument -- it would be costed as a free leaf by extract_dag::op_cost's children.is_empty() rule, same as Num/Var, which is likely wrong for a new non-Num/Var zero-Math-arity constructor; revisit that rule"
            );
        }
    }

    /// CYCLE/TERMINATION: `(Neg (Neg x))` -- class `cx` has two members,
    /// `Var("x")` (a leaf) and `neg_neg` (whose own child resolves back
    /// into `cx` itself, a direct self-cycle). `extract_shared_dag` must
    /// terminate and pick the leaf (cost 0), never the self-referential
    /// candidate -- a naive single-pass topological implementation would
    /// hang or panic here; this is the test that would catch that.
    #[test]
    fn cycle_terminates_and_picks_the_leaf() {
        let mut ser = SerEGraph::default();
        let (id, node) = leaf("x", "cx", "Var");
        ser.add_node(id, node);
        let (id, node) = op_node("neg_neg", "cx", "Neg", &["neg_inner"]);
        ser.add_node(id, node);
        let (id, node) = op_node("neg_inner", "c_neg", "Neg", &["x"]);
        ser.add_node(id, node);
        ser.root_eclasses = vec!["cx".into()];

        let extracted = extract_shared_dag(&ser, &["cx".into()]).expect("should terminate");
        let winner = &extracted.chosen[&ClassId::from("cx")];
        assert_eq!(ser.nodes[winner].op, "Var", "the leaf must win over the self-referential Neg(Neg x) candidate");
        assert_eq!(extracted.total_cost, 0);
    }

    /// FREE-REUSE, round 0 only: a root with two children both pointing
    /// at the SAME shared leaf class -- total extracted cost must count
    /// the shared leaf ONCE, not twice. Pins the "reachable set, not tree
    /// sum" cost rule directly, independent of the marginal-cost rounds
    /// (this shape has only one root, so round 0's plain free-reuse
    /// answer is already correct and rounds never need to improve on it).
    #[test]
    fn free_reuse_diamond_counts_the_shared_leaf_once() {
        let mut ser = SerEGraph::default();
        let (id, node) = leaf("leaf", "c_leaf", "Var");
        ser.add_node(id, node);
        let (id, node) = op_node("a", "c_a", "Neg", &["leaf"]);
        ser.add_node(id, node);
        let (id, node) = op_node("b", "c_b", "Neg", &["leaf"]);
        ser.add_node(id, node);
        let (id, node) = op_node("root", "c_root", "Add", &["a", "b"]);
        ser.add_node(id, node);
        ser.root_eclasses = vec!["c_root".into()];

        let extracted = extract_shared_dag(&ser, &["c_root".into()]).expect("should extract");
        // root(1) + a(1) + b(1) + leaf(0), leaf counted once not twice.
        assert_eq!(extracted.total_cost, 3);
    }

    /// THE MARGINAL-COST FIXPOINT ITSELF, the regression guard against
    /// reintroducing plain greedy-DAG: three independent roots where
    /// round 0's per-root-independent minimum (picking `add_class`, cost
    /// 2 total for the shared root) disagrees with the jointly cheaper
    /// form (picking `mul_class`, cost 1 marginal once x/y's own roots
    /// already cover their classes) -- exactly the motivating case's
    /// arithmetic (`docs/PLAN_saturated_share_equivalence_v1.md` §2's
    /// worked trace), built by hand instead of through egglog so the
    /// extractor is tested in isolation from saturation.
    ///
    /// Shape: root0's class has two members: `add_form` (cost 1, one
    /// child -> `add_class`, itself cost 1, two children -> x_class,
    /// y_class) and `mul_form` (cost 1, two children -> x_class,
    /// y_class directly). root1 = x_class's own node. root2 = y_class's
    /// own node. Greedy-DAG (round 0, independent): root0 alone prices
    /// `add_form` at 1+1=2 (own + add_class) vs `mul_form` at 1+0+0=1
    /// (own + x_class's cost which round 0 ALSO must price into root0's
    /// OWN total if evaluated jointly -- the actual round-0 trap is that
    /// bottom_up_pass with empty `covered` already gives free reuse
    /// WITHIN one pass, so to truly reproduce the greedy-DAG failure mode
    /// this test checks the JOINT total across all three roots, where
    /// round 0 (no marginal pricing across separate pass invocations per
    /// root) is not actually comparable -- the real regression check is:
    /// total_cost must equal the jointly optimal 3, not a worse value a
    /// buggy implementation might settle on by, e.g., failing to re-run
    /// the pass with `covered` updated.
    #[test]
    fn marginal_cost_fixpoint_finds_the_jointly_cheaper_shared_form() {
        let mut ser = SerEGraph::default();
        let (id, node) = leaf("x", "x_class", "Var");
        ser.add_node(id, node);
        let (id, node) = leaf("y", "y_class", "Var");
        ser.add_node(id, node);
        let (id, node) = op_node("add_class_node", "add_class", "Add", &["x", "y"]);
        ser.add_node(id, node);
        let (id, node) = op_node("add_form", "root0_class", "Exp", &["add_class_node"]);
        ser.add_node(id, node);
        let (id, node) = op_node("mul_form", "root0_class", "Mul", &["exp_x", "exp_y"]);
        ser.add_node(id, node);
        let (id, node) = op_node("exp_x", "expx_class", "Exp", &["x"]);
        ser.add_node(id, node);
        let (id, node) = op_node("exp_y", "expy_class", "Exp", &["y"]);
        ser.add_node(id, node);
        // mul_form's children must resolve, via .eclass, to classes that
        // ALSO appear as other roots -- root1/root2 below point at
        // exp_x/exp_y's OWN classes directly (mirroring gene1=Exp(x),
        // gene2=Exp(y) in the real motivating case).
        ser.root_eclasses = vec!["root0_class".into(), "expx_class".into(), "expy_class".into()];

        let extracted = extract_shared_dag(&ser, &["root0_class".into(), "expx_class".into(), "expy_class".into()]).expect("should extract");
        assert_eq!(extracted.total_cost, 3, "jointly cheapest: exp_x(1) + exp_y(1) + mul_form(1) = 3, not add_form's path (exp_add=1+add=1+... =4 total)");
        let winner = &extracted.chosen[&ClassId::from("root0_class")];
        assert_eq!(ser.nodes[winner].op, "Mul", "the marginal-cost fixpoint must choose Mul, not Exp(Add x y)");
    }

    /// BRUTE-FORCE CROSS-CHECK on the same fixture as the fixpoint test
    /// above: exhaustively enumerate every combination of one e-node per
    /// reachable class and confirm the TRUE minimum joint cost equals
    /// what the heuristic found. Required before trusting the heuristic
    /// on real chromosomes (per the plan's revision note) -- a heuristic
    /// that only ever gets spot-checked by eye is not actually checked.
    #[test]
    fn brute_force_confirms_the_fixpoint_found_the_true_optimum() {
        let mut ser = SerEGraph::default();
        let (id, node) = leaf("x", "x_class", "Var");
        ser.add_node(id, node);
        let (id, node) = leaf("y", "y_class", "Var");
        ser.add_node(id, node);
        let (id, node) = op_node("add_class_node", "add_class", "Add", &["x", "y"]);
        ser.add_node(id, node);
        let (id, node) = op_node("add_form", "root0_class", "Exp", &["add_class_node"]);
        ser.add_node(id, node);
        let (id, node) = op_node("mul_form", "root0_class", "Mul", &["exp_x", "exp_y"]);
        ser.add_node(id, node);
        let (id, node) = op_node("exp_x", "expx_class", "Exp", &["x"]);
        ser.add_node(id, node);
        let (id, node) = op_node("exp_y", "expy_class", "Exp", &["y"]);
        ser.add_node(id, node);
        ser.root_eclasses = vec!["root0_class".into(), "expx_class".into(), "expy_class".into()];

        // Brute force: only root0_class has a real choice (2 members);
        // every other reachable class has exactly 1 member. Enumerate
        // both root0_class choices, compute the true joint op count by
        // walking each concrete choice's reachable set.
        let candidates = ["add_form", "mul_form"];
        let mut true_min = u64::MAX;
        for choice in candidates {
            // Walk the chosen root0_class member, PLUS the two other
            // independent roots (expx_class, expy_class each have only
            // one member, so there's no ambiguity there) -- track which
            // concrete NODE was visited for each class, not an arbitrary
            // "nodes[0]" of the class, so root0_class's own cost is
            // costed as `choice` specifically, never the other member.
            let mut reachable: HashMap<ClassId, NodeId> = HashMap::new();
            let mut stack = vec![NodeId::from(choice), NodeId::from("exp_x"), NodeId::from("exp_y")];
            let mut visited_nodes: HashSet<NodeId> = HashSet::new();
            while let Some(nid) = stack.pop() {
                if !visited_nodes.insert(nid.clone()) {
                    continue;
                }
                let node = &ser.nodes[&nid];
                reachable.entry(node.eclass.clone()).or_insert_with(|| nid.clone());
                for c in &node.children {
                    stack.push(c.clone());
                }
            }
            let cost: u64 = reachable.values().map(|nid| if ser.nodes[nid].op == "Var" { 0 } else { 1 }).sum();
            true_min = true_min.min(cost);
        }

        let extracted = extract_shared_dag(&ser, &["root0_class".into(), "expx_class".into(), "expy_class".into()]).expect("should extract");
        assert_eq!(extracted.total_cost, true_min, "the heuristic's total_cost must equal the brute-force true minimum on this fixture");
    }

    /// NOT an oscillation witness -- named for what it actually checks,
    /// after an attempt to build a genuine round-to-round flip (see the
    /// long in-body comment) showed this specific shape cannot produce
    /// one: once both roots' round-0 independent choice avoids the
    /// shared class entirely, no later round's marginal re-pricing can
    /// discover that sharing would help, because marginal pricing only
    /// ever changes the cost of a class some round's choice already
    /// reached. This test instead confirms termination plus a
    /// self-consistent, genuinely-optimal settled answer on a
    /// multi-root shape with a real (non-zero-cost) shareable class, as
    /// a check distinct from the headline test's (which DOES require
    /// the fixpoint to move past round 0 to find the right answer).
    #[test]
    fn two_roots_with_an_unshared_optimum_settle_without_crashing() {
        // A genuine flip, worked by hand before writing this fixture:
        // `shared_class` is a REAL op (Pow2(leaf), cost 1, not a free
        // leaf) -- this is what makes sharing it actually worth
        // something, unlike the leaf-only fixture this test replaces
        // (which tied at cost 2 regardless of what the fixpoint did,
        // and so never exercised a real flip).
        //
        // ra_class: `ra_via_shared` = Add(shared_class, extra_a) cost
        // own(1)+shared(1 if not free, else 0)+extra_a(0, leaf) = 2 or 1.
        // `ra_alone` = a plain leaf, cost 0 always.
        // rb_class: symmetric (`rb_via_shared`, `rb_alone`).
        //
        // Round 0 (nothing free): `ra_alone`(0) beats `ra_via_shared`(2)
        // for ra, and symmetrically for rb -- NEITHER root touches
        // `shared_class` at all. With neither root ever reaching it,
        // `shared_class` can never become "free" to the other in any
        // later round either (is_free_to requires some root's reach to
        // actually CONTAIN it) -- so round 0's `ra_alone`/`rb_alone`
        // pick is a stable fixpoint, total_cost 0, and that IS the best
        // possible answer the algorithm could find given the LOCAL
        // comparison it makes each round. This is the real limitation
        // the "heuristic, not proof of optimality" doc comment names: a
        // form neither root ever tries in round 0 can never be
        // discovered later purely by marginal re-pricing, because
        // marginal pricing only changes costs for classes SOME round's
        // choice already reached. Confirmed by hand -- this fixture
        // settles immediately (round 0 == round 1), not because nothing
        // COULD flip, but because this specific shape never gives the
        // fixpoint a reason to explore the shared branch at all.
        //
        // This is recorded here, not swept under a passing assertion:
        // the test now checks TERMINATION and the SETTLED answer being
        // self-consistent (round 0's independent-optimal choice, which
        // is also the right answer here since sharing a cost-1 op
        // between two roots that could otherwise both go free is never
        // beneficial) -- not a flip, because this fixture provably
        // cannot produce one. A fixture that WOULD need marginal
        // re-pricing to flip requires the cheap-alone option to not
        // exist (forcing round 0 to touch the shared class on both
        // sides) -- removing `ra_alone`/`rb_alone` turns this into the
        // headline test's own shape (`marginal_cost_fixpoint_finds_the_
        // jointly_cheaper_shared_form`), which already covers that case
        // end to end. No separate witness fixture for a genuine
        // round-to-round flip exists yet; the headline test is the
        // closest thing to one in this suite.
        let mut ser = SerEGraph::default();
        let (id, node) = leaf("shared_leaf", "shared_leaf_class", "Var");
        ser.add_node(id, node);
        let (id, node) = op_node("shared_op", "shared_class", "Pow2", &["shared_leaf"]);
        ser.add_node(id, node);
        let (id, node) = op_node("ra_via_shared", "ra_class", "Add", &["shared_op", "extra_a"]);
        ser.add_node(id, node);
        let (id, node) = leaf("extra_a", "extra_a_class", "Num");
        ser.add_node(id, node);
        let (id, node) = leaf("ra_alone", "ra_class", "Num");
        ser.add_node(id, node);
        let (id, node) = op_node("rb_via_shared", "rb_class", "Sub", &["shared_op", "extra_b"]);
        ser.add_node(id, node);
        let (id, node) = leaf("extra_b", "extra_b_class", "Num");
        ser.add_node(id, node);
        let (id, node) = leaf("rb_alone", "rb_class", "Num");
        ser.add_node(id, node);
        ser.root_eclasses = vec!["ra_class".into(), "rb_class".into()];

        let extracted = extract_shared_dag(&ser, &["ra_class".into(), "rb_class".into()]).expect("should settle, not hang");
        assert_eq!(extracted.total_cost, 0, "round 0's independent-optimal (ra_alone, rb_alone, both cost 0) is also the global optimum here -- sharing shared_class is never worth it when going alone is free");
        let ra_winner = &extracted.chosen[&ClassId::from("ra_class")];
        let rb_winner = &extracted.chosen[&ClassId::from("rb_class")];
        assert_eq!(ser.nodes[ra_winner].op, "Num");
        assert_eq!(ser.nodes[rb_winner].op, "Num");
    }
}

