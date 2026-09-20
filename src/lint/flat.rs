//! The kernel-shaped engine: the device's algorithm, run on the CPU.
//!
//! Everything here is written the way the WGSL kernel has to work — flat
//! level-order node arrays, fixed-size scratch, explicit queues, no recursion,
//! integer comparisons only — so the kernel is a transcription of this file,
//! and this file is checked against the tree engine (`engine.rs`) for exact
//! agreement.
//!
//! One round:
//!   facts     backward scan, a fact word per node (K1)
//!   match     first hit in (class A before B, position, rule id) order (K2)
//!   apply     append the template's nodes, redirect one pointer (K3)
//!   relayout  breadth-first renumbering from the root through a queue
//!
//! The relayout does three jobs at once: it drops the nodes the rewrite
//! orphaned (a dead `Sin(inf)` must not reach the evaluator's poison flag), it
//! restores child index > parent index, and it leaves the result in CANONICAL
//! level order — a genuine K-expression, not merely something the evaluator
//! can run.

use super::engine::CallerFacts;
use super::node::Tree;
use super::pack::{arity, LiteralCodes, PackedRule, HEAP_SLOTS, KIND_MV, KIND_NUM, KIND_OP, KIND_UNUSED, MAX_METAVARS};
use super::tables::{Exactness, Facts, GuardRule};
use crate::gpu_eval::Op;

const NONE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LNode {
    pub op: u32,
    /// First child; for a `Var`, the index into `Flat::vars`.
    pub arg0: u32,
    /// Second child of a binary op. Meaningless (0) otherwise — every reader
    /// must consult the arity first.
    pub arg1: u32,
    pub lit: f64,
}

/// One expression in canonical level order: root at 0, a node's children
/// contiguous and after it.
#[derive(Debug, Clone, PartialEq)]
pub struct Flat {
    pub nodes: Vec<LNode>,
    pub vars: Vec<String>,
}

impl Flat {
    pub fn from_tree(tree: &Tree) -> Flat {
        let mut vars: Vec<String> = Vec::new();
        let mut nodes: Vec<LNode> = vec![LNode { op: 0, arg0: 0, arg1: 0, lit: 0.0 }];
        let mut queue: Vec<(&Tree, usize)> = vec![(tree, 0)];
        let mut next = 0usize;
        while next < queue.len() {
            let (t, at) = queue[next];
            next += 1;
            nodes[at] = match t {
                Tree::Num(v) => LNode { op: Op::Num as u32, arg0: 0, arg1: 0, lit: *v },
                Tree::Var(name) => {
                    let i = match vars.iter().position(|v| v == name) {
                        Some(i) => i,
                        None => {
                            vars.push(name.clone());
                            vars.len() - 1
                        }
                    };
                    LNode { op: Op::Var as u32, arg0: i as u32, arg1: 0, lit: 0.0 }
                }
                Tree::App(op, kids) => {
                    let first = nodes.len();
                    for k in kids {
                        nodes.push(LNode { op: 0, arg0: 0, arg1: 0, lit: 0.0 });
                        queue.push((k, nodes.len() - 1));
                    }
                    LNode {
                        op: *op as u32,
                        arg0: first as u32,
                        arg1: if kids.len() > 1 { first as u32 + 1 } else { 0 },
                        lit: 0.0,
                    }
                }
            };
        }
        Flat { nodes, vars }
    }

    /// Like [`Flat::from_tree`], but every `Var` indexes the GLOBAL column list
    /// `vars` — what the device needs, where a batch shares one data buffer. A
    /// name that is not a column is an error, never a default column.
    pub fn from_tree_in(tree: &Tree, vars: &[String]) -> Result<Flat, String> {
        let mut f = Flat::from_tree(tree);
        let local = std::mem::take(&mut f.vars);
        for n in f.nodes.iter_mut().filter(|n| n.op == Op::Var as u32) {
            let name = &local[n.arg0 as usize];
            let at = vars.iter().position(|v| v == name).ok_or_else(|| format!("{name} is not a column"))?;
            n.arg0 = at as u32;
        }
        f.vars = vars.to_vec();
        Ok(f)
    }

    pub fn to_tree(&self) -> Tree {
        fn go(f: &Flat, i: usize) -> Tree {
            let n = &f.nodes[i];
            match arity(n.op) {
                0 if n.op == Op::Num as u32 => Tree::Num(n.lit),
                0 => Tree::Var(f.vars[n.arg0 as usize].clone()),
                1 => Tree::App(op_of(n.op), vec![go(f, n.arg0 as usize)]),
                _ => Tree::App(op_of(n.op), vec![go(f, n.arg0 as usize), go(f, n.arg1 as usize)]),
            }
        }
        go(self, 0)
    }
}

fn op_of(code: u32) -> Op {
    [
        Op::Var, Op::Num, Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Neg, Op::Abs, Op::Sqrt, Op::Log,
        Op::Exp, Op::Sin, Op::Cos, Op::Tan, Op::Tanh, Op::Pow, Op::Pow2, Op::Pow3, Op::Inv,
        Op::ProtectedDiv, Op::ProtectedSqrt, Op::ProtectedLog, Op::ProtectedExp, Op::ProtectedInv,
    ][code as usize]
}

/// K1. One backward scan: children are at higher indices, so their facts are
/// ready when the parent is reached.
fn facts_pass(f: &Flat, guards: &[GuardRule], caller: &CallerFacts, strict: bool) -> Vec<u8> {
    let mut facts = vec![0u8; f.nodes.len()];
    for k in (0..f.nodes.len()).rev() {
        let n = &f.nodes[k];
        let mut here = Facts::NONE;
        if n.op == Op::Var as u32 {
            let name = &f.vars[n.arg0 as usize];
            if caller.positive.contains(name) {
                here = here.union(Facts::POSITIVE);
            }
            if caller.nonzero.contains(name) {
                here = here.union(Facts::NONZERO);
            }
        }
        let kids = [n.arg0 as usize, n.arg1 as usize];
        loop {
            let before = here;
            for g in guards.iter().filter(|g| !(strict && g.range_only)) {
                let shape_ok = match g.op {
                    None => true,
                    Some(want) => {
                        n.op == want as u32
                            && n.op != Op::Var as u32
                            && g.child_req
                                .iter()
                                .enumerate()
                                .all(|(i, need)| Facts(facts[kids[i]]).contains(*need))
                            && (n.op != Op::Num as u32
                                || g.num_preds.iter().all(|(cmp, v)| cmp.holds(n.lit, *v)))
                    }
                };
                if shape_ok && here.contains(g.self_req) {
                    here = here.union(g.gives);
                }
            }
            if here == before {
                break;
            }
        }
        facts[k] = here.0;
    }
    facts
}

/// Same-subtree test without recursion: walk both in lockstep through a queue
/// of index pairs. (On the device a per-node hash filters first; acceptance is
/// always this exact walk — a collision would turn `Sub x y` into 0.)
fn same_subtree(f: &Flat, a: u32, b: u32) -> bool {
    let mut queue: Vec<(u32, u32)> = vec![(a, b)];
    let mut next = 0usize;
    while next < queue.len() {
        let (x, y) = queue[next];
        next += 1;
        if x == y {
            continue;
        }
        let (nx, ny) = (&f.nodes[x as usize], &f.nodes[y as usize]);
        if nx.op != ny.op {
            return false;
        }
        match arity(nx.op) {
            0 => {
                let same = if nx.op == Op::Num as u32 {
                    nx.lit.to_bits() == ny.lit.to_bits()
                } else {
                    nx.arg0 == ny.arg0
                };
                if !same {
                    return false;
                }
            }
            1 => queue.push((nx.arg0, ny.arg0)),
            _ => {
                queue.push((nx.arg0, ny.arg0));
                queue.push((nx.arg1, ny.arg1));
            }
        }
    }
    true
}

/// K2 at one position. Returns the bindings, and whether any required fact was
/// available only in the loose (range-only) facts.
fn match_at(
    f: &Flat,
    rule: &PackedRule,
    pos: u32,
    codes: &LiteralCodes,
    strict: &[u8],
    loose: Option<&[u8]>,
) -> Option<([u32; MAX_METAVARS], bool)> {
    let mut at = [NONE; HEAP_SLOTS];
    let mut bind = [NONE; MAX_METAVARS];
    let mut needed_loose = false;
    at[0] = pos;
    for slot in 0..HEAP_SLOTS {
        let s = &rule.pat[slot];
        if s.kind == KIND_UNUSED || at[slot] == NONE {
            continue;
        }
        let idx = at[slot];
        let n = &f.nodes[idx as usize];
        match s.kind {
            KIND_OP => {
                if n.op != s.id {
                    return None;
                }
                let a = arity(n.op);
                if a >= 1 {
                    at[2 * slot + 1] = n.arg0;
                }
                if a >= 2 {
                    at[2 * slot + 2] = n.arg1;
                }
            }
            KIND_MV => {
                let mv = s.id as usize;
                if bind[mv] == NONE {
                    bind[mv] = idx;
                    let need = s.req as u8;
                    if strict[idx as usize] & need != need {
                        match loose {
                            Some(l) if l[idx as usize] & need == need => needed_loose = true,
                            _ => return None,
                        }
                    }
                } else if !same_subtree(f, bind[mv], idx) {
                    return None;
                }
            }
            KIND_NUM => {
                if n.op != Op::Num as u32 {
                    return None;
                }
                if s.lit != 0 {
                    if codes.class_of(n.lit) != s.lit {
                        return None;
                    }
                } else if codes.pred_bits(n.lit) & s.req != s.req {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some((bind, needed_loose))
}

/// K3 + relayout. Append the template's nodes after the existing ones, point
/// the hit's parent at the new subtree, then renumber breadth-first from the
/// root. Orphans are simply never reached.
fn apply(f: &Flat, rule: &PackedRule, pos: u32, bind: &[u32; MAX_METAVARS], codes: &LiteralCodes) -> Flat {
    let mut nodes = f.nodes.clone();
    // Where each template slot lives: a fresh index, or the bound subtree.
    let mut home = [NONE; HEAP_SLOTS];
    for (place, s) in home.iter_mut().zip(&rule.tmpl) {
        match s.kind {
            KIND_OP | KIND_NUM => {
                *place = nodes.len() as u32;
                nodes.push(LNode { op: 0, arg0: 0, arg1: 0, lit: 0.0 });
            }
            KIND_MV => *place = bind[s.id as usize],
            _ => {}
        }
    }
    for slot in 0..HEAP_SLOTS {
        let s = &rule.tmpl[slot];
        if s.kind == KIND_OP {
            let a = arity(s.id);
            nodes[home[slot] as usize] = LNode {
                op: s.id,
                arg0: if a >= 1 { home[2 * slot + 1] } else { 0 },
                arg1: if a >= 2 { home[2 * slot + 2] } else { 0 },
                lit: 0.0,
            };
        } else if s.kind == KIND_NUM {
            nodes[home[slot] as usize] =
                LNode { op: Op::Num as u32, arg0: 0, arg1: 0, lit: codes.pool[s.id as usize] };
        }
    }
    let mut root = 0u32;
    if pos == 0 {
        root = home[0];
    } else {
        for n in nodes.iter_mut().take(f.nodes.len()) {
            let a = arity(n.op);
            if a >= 1 && n.arg0 == pos {
                n.arg0 = home[0];
            }
            if a >= 2 && n.arg1 == pos {
                n.arg1 = home[0];
            }
        }
    }

    // Breadth-first renumbering. `queue[i]` is the old index of the node that
    // lands at new index `i`.
    let mut queue: Vec<u32> = vec![root];
    let mut out: Vec<LNode> = Vec::with_capacity(f.nodes.len());
    let mut next = 0usize;
    while next < queue.len() {
        let old = nodes[queue[next] as usize];
        next += 1;
        let a = arity(old.op);
        let first = queue.len() as u32;
        if a >= 1 {
            queue.push(old.arg0);
        }
        if a >= 2 {
            queue.push(old.arg1);
        }
        out.push(LNode {
            op: old.op,
            arg0: if a >= 1 { first } else { old.arg0 },
            arg1: if a >= 2 { first + 1 } else { 0 },
            lit: old.lit,
        });
    }
    Flat { nodes: out, vars: f.vars.clone() }
}

pub struct GreedyOutcome {
    pub form: Flat,
    pub steps: usize,
    /// The weakest exactness used on the way.
    pub level: Exactness,
}

/// Greedy rounds, as the device runs them: first hit in (class A before B,
/// position, rule id) order, until nothing fires or `max_steps` is reached.
/// No constant folding between rounds — the device does none.
pub fn run_greedy(
    start: &Flat,
    rules: &[PackedRule],
    codes: &LiteralCodes,
    guards: &[GuardRule],
    caller: &CallerFacts,
    admit: Exactness,
    max_steps: usize,
) -> GreedyOutcome {
    let admit_code = admit as u32;
    let mut form = start.clone();
    let mut level = Exactness::Bit;
    let mut steps = 0usize;
    while steps < max_steps {
        let strict = facts_pass(&form, guards, caller, true);
        let loose = (admit == Exactness::Finite).then(|| facts_pass(&form, guards, caller, false));
        let mut hit: Option<(u32, &PackedRule, [u32; MAX_METAVARS], bool)> = None;
        'search: for order in [0u32, 1] {
            for pos in 0..form.nodes.len() as u32 {
                let op = form.nodes[pos as usize].op;
                for rule in rules {
                    if rule.order != order || rule.exactness > admit_code {
                        continue;
                    }
                    if rule.root_kind == KIND_OP && rule.root_op != op {
                        continue;
                    }
                    if let Some((bind, needed_loose)) =
                        match_at(&form, rule, pos, codes, &strict, loose.as_deref())
                    {
                        hit = Some((pos, rule, bind, needed_loose));
                        break 'search;
                    }
                }
            }
        }
        let Some((pos, rule, bind, needed_loose)) = hit else {
            break;
        };
        let used = if needed_loose {
            Exactness::Finite
        } else {
            [Exactness::Bit, Exactness::Rounding, Exactness::Finite][rule.exactness as usize]
        };
        level = level.max(used);
        form = apply(&form, rule, pos, &bind, codes);
        steps += 1;
    }
    GreedyOutcome { form, steps, level }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lint::engine::{run, Config, LitMode, Search};
    use crate::lint::pack::pack;
    use crate::lint::tables::{Rule, Tables};

    fn tree(s: &str) -> Tree {
        Tree::parse(s).unwrap()
    }

    /// Level order as Karva reads it: walking the array front to back, each
    /// function's children are the next unclaimed slots.
    fn is_canonical(f: &Flat) -> bool {
        let mut next_free = 1u32;
        for n in &f.nodes {
            let a = arity(n.op) as u32;
            if a >= 1 && n.arg0 != next_free {
                return false;
            }
            if a >= 2 && n.arg1 != next_free + 1 {
                return false;
            }
            next_free += a;
        }
        next_free as usize == f.nodes.len()
    }

    #[test]
    fn a_tree_survives_the_round_trip_through_level_order() {
        let t = tree(r#"(Add (Mul (Neg (Var "a")) (Var "b")) (Sin (Num 2.5)))"#);
        let f = Flat::from_tree(&t);
        assert_eq!(f.to_tree(), t);
        // canonical level order: Add, Mul, Sin, Neg, b, 2.5, a
        let ops: Vec<u32> = f.nodes.iter().map(|n| n.op).collect();
        assert_eq!(ops[..3], [Op::Add as u32, Op::Mul as u32, Op::Sin as u32]);
        for (i, n) in f.nodes.iter().enumerate() {
            if arity(n.op) >= 1 {
                assert!(n.arg0 as usize > i);
            }
        }
    }

    /// The layout the evaluator can run is NOT enough: after a rewrite the
    /// result must still be a K-expression. `Add (Mul (Neg a) b) c` with the
    /// float-out is the case a naive splice gets wrong.
    #[test]
    fn a_rewrite_leaves_a_canonical_k_expression() {
        let tables = Tables::standard().unwrap();
        let rules: Vec<&Rule> = tables.rules.iter().collect();
        let p = pack(&rules);
        let start = Flat::from_tree(&tree(r#"(Add (Mul (Neg (Var "a")) (Var "b")) (Var "c"))"#));
        let out = run_greedy(&start, &p.rules, &p.codes, &tables.guards, &CallerFacts::default(), Exactness::Finite, 1);
        assert_eq!(out.steps, 1);
        assert!(is_canonical(&out.form), "not in canonical level order: {:?}", out.form.nodes);
        assert_eq!(out.form.to_tree().to_math(), r#"(Add (Neg (Mul (Var "a") (Var "b"))) (Var "c"))"#);
    }

    #[test]
    fn dropped_subtrees_do_not_survive() {
        let tables = Tables::standard().unwrap();
        let rules: Vec<&Rule> = tables.rules.iter().collect();
        let p = pack(&rules);
        // Mul x 0 -> 0 drops x entirely; nothing of Sin(..) may remain.
        let start = Flat::from_tree(&tree(r#"(Add (Var "y") (Mul (Sin (Var "x")) (Num 0.0)))"#));
        let out = run_greedy(&start, &p.rules, &p.codes, &tables.guards, &CallerFacts::default(), Exactness::Finite, 8);
        assert!(out.form.nodes.iter().all(|n| n.op != Op::Sin as u32));
        assert_eq!(out.form.to_tree().to_math(), r#"(Var "y")"#);
    }

    /// The kernel-shaped engine and the tree engine make the same choices and
    /// reach the same form, at every exactness level.
    #[test]
    fn agrees_with_the_tree_engine() {
        let tables = Tables::standard().unwrap();
        let rules: Vec<&Rule> = tables.rules.iter().collect();
        let p = pack(&rules);
        let caller = CallerFacts::default();
        // `c` and `m` are data columns here, not constants: never folded.
        let inputs: Vec<String> = ["a", "b", "c", "x", "m"].iter().map(|s| s.to_string()).collect();
        let cases = [
            r#"(Pow2 (Sub (Neg (Var "a")) (Var "b")))"#,
            r#"(Sub (Neg (Mul (Var "a") (Num -1.0))) (Neg (Abs (Pow2 (Var "b")))))"#,
            r#"(ProtectedDiv (Sin (Neg (ProtectedInv (Abs (Exp (Var "x")))))) (Mul (Num -1.0) (Var "m")))"#,
            r#"(Sub (Add (Var "a") (Var "b")) (Var "b"))"#,
            r#"(Mul (Mul (Var "c") (Var "c")) (Inv (Var "c")))"#,
            r#"(Abs (ProtectedInv (ProtectedInv (Pow2 (Var "x")))))"#,
        ];
        for admit in [Exactness::Bit, Exactness::Rounding, Exactness::Finite] {
            for case in cases {
                let t = tree(case);
                let cfg = Config {
                    inputs: &inputs,
                    caller: &caller,
                    mode: LitMode::F64,
                    search: Search::Greedy,
                    max_steps: 64,
                    admit,
                    computed_literals: false,
                    fold_in_rounds: false,
                };
                let want = run(&t, &rules, &tables.guards, &cfg).best;
                // The host folds once at ingestion; the device never folds.
                let start = Flat::from_tree(&crate::lint::engine::fold(t.clone(), &inputs));
                let got = run_greedy(&start, &p.rules, &p.codes, &tables.guards, &caller, admit, 64);
                assert!(is_canonical(&got.form), "{case}: not canonical");
                assert_eq!(got.form.to_tree().to_math(), want.to_math(), "{case} at {admit:?}");
            }
        }
    }
}
