//! The rewriting engine: facts, match, apply, fold, and the search over forms.
//!
//! Spec §4 (K1–K4) and §6, on the CPU. This is the reference the device kernel
//! is checked against.

use std::collections::BTreeMap;

use super::node::{Measure, Tree};
use super::tables::{Exactness, Facts, GuardRule, Order, Pat, Rule, Tmpl};
use crate::gpu_eval::Op;

/// How literals are compared and computed. `F32` reproduces what the device
/// would do, so device parity can be tested exactly; forms used for reporting
/// or write-back are always derived in `F64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LitMode {
    F64,
    F32,
}

/// Facts the caller asserts about input variables.
#[derive(Debug, Clone, Default)]
pub struct CallerFacts {
    pub positive: Vec<String>,
    pub nonzero: Vec<String>,
}

/// Per-node facts, in the shape of the tree they annotate.
#[derive(Debug, Clone)]
pub struct Ann {
    pub facts: Facts,
    pub kids: Vec<Ann>,
}

/// K1: one bottom-up pass. A node's facts come from its symbol (seeds), its
/// children's facts (propagation), and implications between facts.
pub fn annotate(t: &Tree, guards: &[GuardRule], caller: &CallerFacts, strict: bool) -> Ann {
    let kids: Vec<Ann> = match t {
        Tree::App(_, ks) => ks.iter().map(|k| annotate(k, guards, caller, strict)).collect(),
        _ => Vec::new(),
    };
    let mut facts = Facts::NONE;
    if let Tree::Var(name) = t {
        if caller.positive.contains(name) {
            facts = facts.union(Facts::POSITIVE);
        }
        if caller.nonzero.contains(name) {
            facts = facts.union(Facts::NONZERO);
        }
    }
    let (op, literal) = match t {
        Tree::Num(v) => (Some(Op::Num), Some(*v)),
        Tree::Var(_) => (None, None),
        Tree::App(op, _) => (Some(*op), None),
    };
    // Implications can chain (positive => nonzero, positive => nonneg), so
    // repeat until nothing is added. Facts only grow; at most 3 bits.
    loop {
        let before = facts;
        // `strict`: only rows that hold everywhere. A `range_only` row (`Exp x`
        // is positive — except where it underflows to 0) is used only alongside
        // the finite-exact rules.
        for g in guards.iter().filter(|g| !(strict && g.range_only)) {
            let shape_ok = match g.op {
                None => true,
                Some(want) => {
                    op == Some(want)
                        && g.child_req
                            .iter()
                            .zip(&kids)
                            .all(|(need, kid)| kid.facts.contains(*need))
                        && literal.map_or(g.num_preds.is_empty(), |v| {
                            g.num_preds.iter().all(|(cmp, k)| cmp.holds(v, *k))
                        })
                }
            };
            // `self_req` is empty for a seed or propagation row, and the reader
            // guarantees it is non-empty for a pure implication.
            if shape_ok && facts.contains(g.self_req) {
                facts = facts.union(g.gives);
            }
        }
        if facts == before {
            break;
        }
    }
    Ann { facts, kids }
}

fn ann_at<'a>(ann: &'a Ann, path: &[u8]) -> Option<&'a Ann> {
    let mut a = ann;
    for &i in path {
        a = a.kids.get(i as usize)?;
    }
    Some(a)
}

/// Bindings of one match.
struct Bind<'a> {
    math: Vec<Option<(&'a Tree, &'a Ann)>>,
    num: Vec<Option<f64>>,
}

fn same_literal(a: f64, b: f64, mode: LitMode) -> bool {
    match mode {
        LitMode::F64 => a == b,
        LitMode::F32 => (a as f32) == (b as f32),
    }
}

/// K2 at one position.
fn matches<'a>(p: &Pat, t: &'a Tree, a: &'a Ann, b: &mut Bind<'a>, mode: LitMode) -> bool {
    match (p, t) {
        (Pat::Mv(i), _) => match b.math[*i as usize] {
            Some((bound, _)) => bound == t,
            None => {
                b.math[*i as usize] = Some((t, a));
                true
            }
        },
        (Pat::NumExact(want), Tree::Num(v)) => same_literal(*v, *want, mode),
        (Pat::NumMv(i), Tree::Num(v)) => match b.num[*i as usize] {
            Some(bound) => bound.to_bits() == v.to_bits(),
            None => {
                b.num[*i as usize] = Some(*v);
                true
            }
        },
        (Pat::Op(want, pats), Tree::App(op, kids)) => {
            want == op
                && pats.len() == kids.len()
                && pats
                    .iter()
                    .zip(kids.iter().zip(&a.kids))
                    .all(|(p, (k, ka))| matches(p, k, ka, b, mode))
        }
        _ => false,
    }
}

/// Try `rule` at `t`. Returns the replacement subtree on a hit.
fn apply_at(rule: &Rule, t: &Tree, a: &Ann, mode: LitMode) -> Option<Tree> {
    let mut b = Bind {
        math: vec![None; rule.n_mv as usize],
        num: vec![None; rule.n_num as usize],
    };
    if !matches(&rule.lhs, t, a, &mut b, mode) {
        return None;
    }
    let nums: Vec<f64> = b.num.iter().map(|n| n.unwrap_or(f64::NAN)).collect();
    if !rule.preds.iter().all(|p| p.cmp.holds(nums[p.num as usize], p.k)) {
        return None;
    }
    for (mv, need) in &rule.when {
        let (_, ann) = b.math[*mv as usize]?;
        if !ann.facts.contains(*need) {
            return None;
        }
    }
    instantiate(&rule.rhs, &b, &nums, mode)
}

/// K3: build the template with the bound subtrees in place.
fn instantiate(t: &Tmpl, b: &Bind, nums: &[f64], mode: LitMode) -> Option<Tree> {
    Some(match t {
        Tmpl::Mv(i) => b.math[*i as usize]?.0.clone(),
        Tmpl::Num(e) => {
            let v = match mode {
                LitMode::F64 => e.eval(nums),
                LitMode::F32 => {
                    let rounded: Vec<f64> = nums.iter().map(|n| *n as f32 as f64).collect();
                    e.eval(&rounded) as f32 as f64
                }
            };
            Tree::Num(v)
        }
        Tmpl::Op(op, kids) => Tree::App(
            *op,
            kids.iter()
                .map(|k| instantiate(k, b, nums, mode))
                .collect::<Option<Vec<_>>>()?,
        ),
    })
}

/// Rules grouped by the opcode at the root of their pattern.
pub struct RuleIndex<'a> {
    by_root: BTreeMap<u32, Vec<&'a Rule>>,
}

impl<'a> RuleIndex<'a> {
    pub fn new(rules: &[&'a Rule]) -> RuleIndex<'a> {
        let mut by_root: BTreeMap<u32, Vec<&'a Rule>> = BTreeMap::new();
        for r in rules {
            let root = match &r.lhs {
                Pat::Op(op, _) => *op as u32,
                Pat::NumExact(_) | Pat::NumMv(_) => Op::Num as u32,
                Pat::Mv(_) => continue, // a bare metavariable matches everything; never admitted
            };
            by_root.entry(root).or_default().push(r);
        }
        RuleIndex { by_root }
    }

    fn at(&self, t: &Tree) -> &[&'a Rule] {
        let root = match t {
            Tree::App(op, _) => *op as u32,
            Tree::Num(_) => Op::Num as u32,
            Tree::Var(_) => return &[],
        };
        self.by_root.get(&root).map_or(&[], Vec::as_slice)
    }
}

/// One rewrite: the rule that fired and the whole new tree.
pub struct Step {
    pub rule: usize,
    pub order: Order,
    /// The rule's own level — or `Finite` when the rule fired only thanks to a
    /// range-only guard fact (`Exp x` is positive, except where it is 0).
    pub exactness: Exactness,
    pub tree: Tree,
}

/// Every single-step rewrite of `t`, in (level-order position, rule id) order.
///
/// `strict` holds the facts that are true everywhere; `loose`, when given,
/// also those true only within the float range. A hit that needs `loose` is
/// labelled `Finite`.
pub fn steps(t: &Tree, index: &RuleIndex, strict: &Ann, loose: Option<&Ann>, mode: LitMode) -> Vec<Step> {
    let mut out = Vec::new();
    for path in t.positions() {
        let (Some(sub), Some(strict_at)) = (t.at(&path), ann_at(strict, &path)) else {
            continue;
        };
        for rule in index.at(sub) {
            let hit = match apply_at(rule, sub, strict_at, mode) {
                Some(r) => Some((r, rule.exactness)),
                None => loose
                    .and_then(|l| ann_at(l, &path))
                    .and_then(|loose_at| apply_at(rule, sub, loose_at, mode))
                    .map(|r| (r, Exactness::Finite)),
            };
            if let Some((replacement, exactness)) = hit {
                out.push(Step {
                    rule: rule.id,
                    order: rule.order,
                    exactness,
                    tree: t.replaced(&path, replacement),
                });
            }
        }
    }
    out
}

/// K4: fold closed subtrees to literals. An input variable is never folded,
/// whatever its name; only finite values fold. Delegates to the one
/// implementation in `extract`, so the linter cannot drift from
/// `smallest_form` here.
pub fn fold(t: Tree, inputs: &[String]) -> Tree {
    if !has_closed_app(&t, inputs) {
        return t;
    }
    crate::extract::fold_constant_subtrees_excluding(&t.to_math(), inputs)
        .and_then(|s| Tree::parse(&s).ok())
        .unwrap_or(t)
}

fn has_closed_app(t: &Tree, inputs: &[String]) -> bool {
    fn closed(t: &Tree, inputs: &[String]) -> bool {
        match t {
            Tree::Num(_) => true,
            Tree::Var(name) => {
                !inputs.contains(name) && crate::snap_karva::constant_values().contains_key(name.as_str())
            }
            Tree::App(_, kids) => kids.iter().all(|k| closed(k, inputs)),
        }
    }
    match t {
        Tree::App(_, kids) => closed(t, inputs) || kids.iter().any(|k| has_closed_app(k, inputs)),
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Search {
    /// Take the first hit — class A before B, then level-order position, then
    /// rule id — and repeat. What the device kernel does.
    Greedy,
    /// Keep the `beam` smallest distinct forms each round.
    Beam(usize),
}

pub struct Config<'a> {
    pub inputs: &'a [String],
    pub caller: &'a CallerFacts,
    pub mode: LitMode,
    pub search: Search,
    /// Upper bound on rewrites along any one chain. The measure guarantees
    /// termination; this bounds the work.
    pub max_steps: usize,
    /// The weakest exactness admitted. `Bit` for anything written back into a
    /// gene or used for fitness.
    pub admit: Exactness,
    /// Admit rules whose template COMPUTES a literal. The v1 device kernel
    /// cannot: a literal made on the device would need classifying in f32.
    pub computed_literals: bool,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    /// The smallest form found (the input if nothing shrank).
    pub best: Tree,
    /// Every distinct form met, smallest first, the input included.
    pub forms: Vec<Tree>,
    /// For each form, the weakest exactness on the chain that reached it:
    /// how far it can be trusted to compute what the input computes.
    pub levels: Vec<Exactness>,
    /// Rewrites along the chain to `best` (greedy) or rounds run (beam).
    pub steps: usize,
    /// How often each rule fired, by rule id.
    pub hits: BTreeMap<usize, usize>,
}

/// Run the linter on one expression.
pub fn run(input: &Tree, rules: &[&Rule], guards: &[GuardRule], cfg: &Config) -> Outcome {
    let admitted: Vec<&Rule> = rules
        .iter()
        .copied()
        .filter(|r| r.exactness <= cfg.admit)
        .filter(|r| cfg.computed_literals || !r.computes_literal())
        .collect();
    let index = RuleIndex::new(&admitted);
    let start = fold(input.clone(), cfg.inputs);
    let mut hits: BTreeMap<usize, usize> = BTreeMap::new();
    // form -> the best (lowest) level it has been reached at.
    let mut seen: BTreeMap<String, Exactness> = BTreeMap::new();
    seen.insert(start.to_math(), Exactness::Bit);
    let mut forms: Vec<Tree> = vec![start.clone()];
    let mut frontier: Vec<(Tree, Exactness)> = vec![(start, Exactness::Bit)];
    let mut rounds = 0usize;

    while rounds < cfg.max_steps && !frontier.is_empty() {
        let mut next: Vec<(Measure, String, Tree, Exactness)> = Vec::new();
        for (t, level) in &frontier {
            let strict = annotate(t, guards, cfg.caller, true);
            let loose = (cfg.admit == Exactness::Finite).then(|| annotate(t, guards, cfg.caller, false));
            let mut found = steps(t, &index, &strict, loose.as_ref(), cfg.mode);
            if cfg.search == Search::Greedy {
                // First hit in (class A before B, position, rule id) order.
                // `steps` is already in (position, rule id) order and the sort
                // is stable.
                found.sort_by_key(|s| s.order == Order::B);
                found.truncate(1);
            }
            for s in found {
                let reached = (*level).max(s.exactness);
                let tree = fold(s.tree, cfg.inputs);
                let key = tree.to_math();
                match seen.get_mut(&key) {
                    Some(known) => *known = (*known).min(reached),
                    None => {
                        seen.insert(key.clone(), reached);
                        *hits.entry(s.rule).or_insert(0) += 1;
                        next.push((tree.measure(), key, tree, reached));
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        rounds += 1;
        next.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        if let Search::Beam(width) = cfg.search {
            next.truncate(width.max(1));
        }
        frontier = next.iter().map(|(_, _, t, l)| (t.clone(), *l)).collect();
        forms.extend(next.into_iter().map(|(_, _, t, _)| t));
    }

    forms.sort_by_cached_key(|t| (t.measure(), t.to_math()));
    let levels = forms.iter().map(|t| seen.get(&t.to_math()).copied().unwrap_or(Exactness::Finite)).collect();
    let best = forms.first().cloned().unwrap_or_else(|| input.clone());
    Outcome { best, forms, levels, steps: rounds, hits }
}
