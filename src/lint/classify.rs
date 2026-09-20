//! Decide, per rule, whether it may enter the table and in which class.
//!
//! Nothing here is asserted by hand. Order (A / B) is derived from the rule's
//! shape and then CHECKED by instantiating it with sample subtrees and
//! comparing the termination measure. Exactness (exact / finite-exact /
//! unsound) is decided by evaluating both sides on probe values that include
//! zero, the protected-divide band, overflow, ±inf and NaN.

use super::node::{Compiled, Tree};
use super::reader::Draft;
use super::tables::{Exactness, Facts, GuardRule, NumExpr, Order, Pat, Rule, Tmpl};
use crate::gpu_eval::Op;

/// Values a metavariable's subtree may take. Finite first, then non-finite.
const PROBES: [f64; 20] = [
    -1e200,
    -std::f64::consts::E,
    -2.3,
    -1.0,
    0.0,
    1e-7,
    0.1,
    0.123456789,
    0.7,
    // Values that do NOT survive divide-then-multiply by 3, 0.1, -9: without
    // them `(x / c) * c -> x` measured as bit-exact, which it is not.
    0.9,
    1.0,
    1.7,
    3.0,
    4.35,
    7.1,
    1234.5678,
    1e200,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

/// Values a bound literal may take (always finite: a gene's constant is).
const LITERALS: [f64; 13] =
    [-9.0, -3.0, -2.7182736588201304, -2.0, -1.0, -0.5, 0.0, 5e-7, 0.1, 0.5, 1.0, 2.0, 3.0];

const AGREE_REL_TOL: f64 = 1e-9;

pub fn classify(d: &Draft, id: usize) -> Result<Rule, String> {
    if matches!(d.lhs, Pat::Mv(_)) {
        return Err("pattern is a bare metavariable".to_string());
    }
    for mv in 0..d.n_mv {
        let (l, r) = (count_pat(&d.lhs, mv), count_tmpl(&d.rhs, mv));
        if r > l {
            return Err(format!(
                "expanding: metavariable {mv} appears {l}x in the pattern, {r}x in the template"
            ));
        }
    }
    let delta = skeleton_tmpl(&d.rhs) as i64 - skeleton_pat(&d.lhs) as i64
        + (0..d.n_mv)
            .map(|mv| count_tmpl(&d.rhs, mv) as i64 - count_pat(&d.lhs, mv) as i64)
            .sum::<i64>();
    let order = match delta {
        i64::MIN..=-1 => Order::A,
        0 => Order::B,
        _ => return Err(format!("expanding: +{delta} nodes at the smallest binding")),
    };

    let nums = literal_bindings(d)?;
    check_measure(d, order, &nums)?;
    let exactness = check_exactness(d, &nums)?;

    Ok(Rule {
        id,
        ruleset: d.ruleset.clone(),
        text: d.text.clone(),
        lhs: d.lhs.clone(),
        rhs: d.rhs.clone(),
        preds: d.preds.clone(),
        when: d.when.clone(),
        n_mv: d.n_mv,
        n_num: d.n_num,
        order,
        exactness,
    })
}

fn count_pat(p: &Pat, mv: u8) -> usize {
    match p {
        Pat::Mv(i) => usize::from(*i == mv),
        Pat::Op(_, kids) => kids.iter().map(|k| count_pat(k, mv)).sum(),
        _ => 0,
    }
}

fn count_tmpl(t: &Tmpl, mv: u8) -> usize {
    match t {
        Tmpl::Mv(i) => usize::from(*i == mv),
        Tmpl::Op(_, kids) => kids.iter().map(|k| count_tmpl(k, mv)).sum(),
        Tmpl::Num(_) => 0,
    }
}

/// Nodes the rule itself writes (metavariables count zero).
fn skeleton_pat(p: &Pat) -> usize {
    match p {
        Pat::Mv(_) => 0,
        Pat::NumExact(_) | Pat::NumMv(_) => 1,
        Pat::Op(_, kids) => 1 + kids.iter().map(skeleton_pat).sum::<usize>(),
    }
}

fn skeleton_tmpl(t: &Tmpl) -> usize {
    match t {
        Tmpl::Mv(_) => 0,
        Tmpl::Num(_) => 1,
        Tmpl::Op(_, kids) => 1 + kids.iter().map(skeleton_tmpl).sum::<usize>(),
    }
}

/// Every assignment of the rule's bound literals that satisfies its
/// predicates. A rule whose predicates no probe satisfies cannot be checked,
/// and is refused rather than admitted untested.
fn literal_bindings(d: &Draft) -> Result<Vec<Vec<f64>>, String> {
    let mut all: Vec<Vec<f64>> = vec![Vec::new()];
    for _ in 0..d.n_num {
        all = all
            .into_iter()
            .flat_map(|prefix| {
                LITERALS.iter().map(move |v| {
                    let mut p = prefix.clone();
                    p.push(*v);
                    p
                })
            })
            .collect();
    }
    all.retain(|nums| d.preds.iter().all(|p| p.cmp.holds(nums[p.num as usize], p.k)));
    if all.is_empty() {
        return Err("no probe literal satisfies the rule's predicates".to_string());
    }
    Ok(all)
}

fn build_pat(p: &Pat, math: &[Tree], nums: &[f64]) -> Tree {
    match p {
        Pat::Mv(i) => math[*i as usize].clone(),
        Pat::NumExact(v) => Tree::Num(*v),
        Pat::NumMv(i) => Tree::Num(nums[*i as usize]),
        Pat::Op(op, kids) => Tree::App(*op, kids.iter().map(|k| build_pat(k, math, nums)).collect()),
    }
}

fn build_tmpl(t: &Tmpl, math: &[Tree], nums: &[f64]) -> Tree {
    match t {
        Tmpl::Mv(i) => math[*i as usize].clone(),
        Tmpl::Num(e) => Tree::Num(NumExpr::eval(e, nums)),
        Tmpl::Op(op, kids) => Tree::App(*op, kids.iter().map(|k| build_tmpl(k, math, nums)).collect()),
    }
}

/// All `choices.len() ^ n` assignments, in a fixed order.
fn assignments<T: Clone>(choices: &[T], n: usize) -> Vec<Vec<T>> {
    let mut all: Vec<Vec<T>> = vec![Vec::new()];
    for _ in 0..n {
        all = all
            .into_iter()
            .flat_map(|prefix| {
                choices.iter().map(move |c| {
                    let mut p = prefix.clone();
                    p.push(c.clone());
                    p
                })
            })
            .collect();
    }
    all
}

/// The termination claim, checked mechanically: with a leaf, a `Neg`, and a
/// product of `Neg`s in every metavariable position, the measure must strictly
/// fall — and for class A, the node count must.
fn check_measure(d: &Draft, order: Order, nums: &[Vec<f64>]) -> Result<(), String> {
    let var = |n: &str| Tree::Var(n.to_string());
    let samples = [
        var("s"),
        Tree::App(Op::Neg, vec![var("s")]),
        Tree::App(
            Op::Mul,
            vec![
                Tree::App(Op::Neg, vec![var("u")]),
                Tree::App(Op::Neg, vec![var("v")]),
            ],
        ),
    ];
    for math in assignments(&samples, d.n_mv as usize) {
        for n in nums {
            let (before, after) = (build_pat(&d.lhs, &math, n), build_tmpl(&d.rhs, &math, n));
            let (mb, ma) = (before.measure(), after.measure());
            let ok = match order {
                Order::A => ma.nodes < mb.nodes,
                Order::B => ma.nodes == mb.nodes && ma < mb,
            };
            if !ok {
                return Err(format!(
                    "does not reduce the measure: {} {mb:?} -> {} {ma:?}",
                    before.to_math(),
                    after.to_math()
                ));
            }
        }
    }
    Ok(())
}

/// Does `v` carry the facts `need`? A fact is a claim about a real value, so
/// NaN satisfies every fact vacuously: what a rule does to a NaN is judged by
/// the exactness check, not excused by a guard.
fn satisfies(v: f64, need: Facts) -> bool {
    v.is_nan()
        || ((!need.contains(Facts::POSITIVE) || v > 0.0)
            && (!need.contains(Facts::NONZERO) || v != 0.0)
            && (!need.contains(Facts::NONNEG) || v >= 0.0))
}

/// Magnitudes far enough from 1 that f64 itself can give out on the way
/// (overflow to inf, underflow to 0: `(1e-7)^1234` IS zero). A guard row that
/// fails only with such a value in play is true of the reals and false of the
/// floats — `range_only`. One that fails on ordinary values is simply wrong.
fn is_extreme(v: f64) -> bool {
    !(v == 0.0 || (1e-3..=1e3).contains(&v.abs()))
}

/// Check one guard row by evaluation: with children carrying the facts it
/// requires, does the node carry the facts it grants?
///
/// `Ok(false)`: always. `Ok(true)`: on ordinary values, but not at the edge of
/// the float range (`Exp(-1e200)` is 0, `Inv(inf)` is 0) — a `range_only` row.
/// `Err`: it fails on ordinary values, and the row is refused.
pub fn classify_guard(g: &GuardRule) -> Result<bool, String> {
    let Some(op) = g.op else {
        // A pure implication between facts: check it on the values directly.
        let mut range_only = false;
        for v in PROBES {
            if satisfies(v, g.self_req) && !satisfies(v, g.gives) {
                if !is_extreme(v) {
                    return Err(format!("implication fails at {v:e}"));
                }
                range_only = true;
            }
        }
        return Ok(range_only);
    };
    if op == Op::Num {
        for v in LITERALS {
            if g.num_preds.iter().all(|(cmp, k)| cmp.holds(v, *k)) && !satisfies(v, g.gives) {
                return Err(format!("literal seed fails at {v:e}"));
            }
        }
        return Ok(false);
    }
    let names: Vec<String> = (0..g.child_req.len()).map(|i| format!("__c{i}")).collect();
    let node = Compiled::new(&Tree::App(op, names.iter().map(|n| Tree::Var(n.clone())).collect()));
    let mut range_only = false;
    for values in assignments(&PROBES, names.len()) {
        if !values.iter().zip(&g.child_req).all(|(v, need)| satisfies(*v, *need)) {
            continue;
        }
        let row: Vec<(String, f64)> = names.iter().cloned().zip(values.iter().copied()).collect();
        let out = node.eval(&row)?;
        if satisfies(out, g.gives) {
            continue;
        }
        if values.iter().any(|v| is_extreme(*v)) {
            range_only = true;
        } else {
            return Err(format!("fails on ordinary values: at {values:?} the node is {out:e}"));
        }
    }
    Ok(range_only)
}

/// Agreement at the precision the inputs allow. `scale` is the largest
/// magnitude in play: `(a + b) - b` with `b = 1e200` evaluates to 0 where the
/// rewrite gives `a` — the rewrite is the MORE accurate side, and the gap is
/// rounding at 1e200, not a different function.
fn agree(a: f64, b: f64, scale: f64) -> bool {
    (a.is_nan() && b.is_nan())
        || a == b
        || (a - b).abs() <= AGREE_REL_TOL * scale.max(a.abs()).max(b.abs())
}

/// Every subtree, the root included.
fn subtrees(t: &Tree, out: &mut Vec<Tree>) {
    out.push(t.clone());
    if let Tree::App(_, kids) = t {
        for k in kids {
            subtrees(k, out);
        }
    }
}

/// Evaluate both sides with every metavariable bound to a probe value, and
/// return the strongest level the rule earns: `Bit` if the results are
/// identical everywhere, `Rounding` if they agree to the precision the inputs
/// allow, `Finite` if they agree wherever every subterm of the pattern — bound
/// values, intermediates and the result — is finite. `Err`, with the
/// counterexample, if they disagree where everything is finite.
fn check_exactness(d: &Draft, nums: &[Vec<f64>]) -> Result<Exactness, String> {
    let names: Vec<String> = (0..d.n_mv).map(|i| format!("__m{i}")).collect();
    let math: Vec<Tree> = names.iter().map(|n| Tree::Var(n.clone())).collect();
    let need = |mv: usize| {
        d.when
            .iter()
            .filter(|(m, _)| *m as usize == mv)
            .fold(Facts::NONE, |acc, (_, f)| acc.union(*f))
    };
    let mut level = Exactness::Bit;
    for n in nums {
        let lhs_tree = build_pat(&d.lhs, &math, n);
        let mut parts = Vec::new();
        subtrees(&lhs_tree, &mut parts);
        let parts: Vec<Compiled> = parts.iter().map(Compiled::new).collect();
        let lhs = Compiled::new(&lhs_tree);
        let rhs = Compiled::new(&build_tmpl(&d.rhs, &math, n));
        let literal_scale = n.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for values in assignments(&PROBES, d.n_mv as usize) {
            if !values.iter().enumerate().all(|(mv, v)| satisfies(*v, need(mv))) {
                continue;
            }
            let row: Vec<(String, f64)> = names.iter().cloned().zip(values.iter().copied()).collect();
            let (a, b) = (lhs.eval(&row)?, rhs.eval(&row)?);
            let scale = values
                .iter()
                .filter(|v| v.is_finite())
                .fold(literal_scale, |m, v| m.max(v.abs()));
            if (a.is_nan() && b.is_nan()) || a == b {
                continue;
            }
            if agree(a, b, scale) {
                level = level.max(Exactness::Rounding);
                continue;
            }
            let mut all_finite = true;
            for part in &parts {
                all_finite &= part.eval(&row)?.is_finite();
            }
            if all_finite {
                return Err(format!(
                    "unsound where every subterm is finite: at {values:?} literals {n:?} the pattern is {a:e}, the template {b:e}"
                ));
            }
            level = Exactness::Finite;
        }
    }
    Ok(level)
}
