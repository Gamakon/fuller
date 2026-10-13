//! GENERIC TYPED REWRITES (2026-10-13, Andrew: a mutation is pattern ->
//! action, like awk). Each rewrite walks the tree, collects the nodes its
//! pattern matches, picks one with the run's RNG and applies its action.
//! The patterns are typed by the kingdom's dual table only: a rewrite knows
//! nothing of what the kernel computes, nor whether it is correct, so it
//! applies in every kingdom. The child goes through the layered check like
//! any child (`knob::check_text`). With the structural rules (`rules.rs`)
//! they make one pool: [`mutate`] draws among every operator whose pattern
//! matches the individual.

use crate::wgsl::knob;
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::render::Tree;
use crate::wgsl::rules::{is_plain_block, kernel_statements, rebuild, replace_statement, rung, statements, Rule};

/// A generic rewrite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rewrite {
    SwapOperands,
    SwapLeaves,
    IndexLiteral,
    ReplaceLeaf,
    SwapOp,
    DeleteStatement,
    DuplicateStatement,
    SwapAdjacent,
    WrapUnary,
    UnwrapUnary,
    LoopEnd,
    InsertAccumulate,
}

impl Rewrite {
    pub const ALL: [Rewrite; 12] = [
        Rewrite::SwapOperands,
        Rewrite::SwapLeaves,
        Rewrite::IndexLiteral,
        Rewrite::ReplaceLeaf,
        Rewrite::SwapOp,
        Rewrite::DeleteStatement,
        Rewrite::DuplicateStatement,
        Rewrite::SwapAdjacent,
        Rewrite::WrapUnary,
        Rewrite::UnwrapUnary,
        Rewrite::LoopEnd,
        Rewrite::InsertAccumulate,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Rewrite::SwapOperands => "swap-operands",
            Rewrite::SwapLeaves => "swap-leaves",
            Rewrite::IndexLiteral => "index-literal",
            Rewrite::ReplaceLeaf => "replace-leaf",
            Rewrite::SwapOp => "swap-op",
            Rewrite::DeleteStatement => "delete-statement",
            Rewrite::DuplicateStatement => "duplicate-statement",
            Rewrite::SwapAdjacent => "swap-adjacent",
            Rewrite::WrapUnary => "wrap-unary",
            Rewrite::UnwrapUnary => "unwrap-unary",
            Rewrite::LoopEnd => "loop-end",
            Rewrite::InsertAccumulate => "insert-accumulate",
        }
    }
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn pick<T: Clone>(xs: &[T], rng: &mut u64) -> Option<T> {
    (!xs.is_empty()).then(|| xs[(next(rng) % xs.len() as u64) as usize].clone())
}

/// The paths (child indices from the root) of every node matching `p`.
fn matches(t: &Tree, p: &dyn Fn(&Tree) -> bool) -> Vec<Vec<usize>> {
    fn walk(t: &Tree, p: &dyn Fn(&Tree) -> bool, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if p(t) {
            out.push(path.clone());
        }
        for (i, k) in t.kids.iter().enumerate() {
            path.push(i);
            walk(k, p, path, out);
            path.pop();
        }
    }
    let mut out = Vec::new();
    walk(t, p, &mut Vec::new(), &mut out);
    out
}

fn at<'t>(t: &'t Tree, path: &[usize]) -> &'t Tree {
    path.iter().fold(t, |n, &i| &n.kids[i])
}

fn replace_at(t: &Tree, path: &[usize], with: Tree) -> Tree {
    match path.split_first() {
        None => with,
        Some((&i, rest)) => {
            let mut kids = t.kids.clone();
            kids[i] = replace_at(&t.kids[i], rest, with);
            Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
        }
    }
}

/// The input duals of a node's row (the row of its name, output and arity).
fn row_inputs<'k>(t: &Tree, kingdom: &'k WgslKingdom) -> Option<&'k [crate::geneframe::Ty]> {
    kingdom.rows.iter().find(|r| r.name == t.name && r.output == t.ty && r.inputs.len() == t.kids.len()).map(|r| r.inputs.as_slice())
}

/// A variable leaf: a terminal row that is not a literal.
fn is_variable(t: &Tree) -> bool {
    t.kids.is_empty() && !t.name.starts_with("literal.") && t.name != "fallback"
}

/// ONE REWRITE of `root` (its references written as their values), or None
/// when the pattern matches nowhere: the description and the child.
pub fn apply(rw: Rewrite, root: &Tree, kingdom: &WgslKingdom, rng: &mut u64) -> Result<Option<(String, Tree)>, String> {
    let plain = crate::wgsl::params::to_literals(root, kingdom)?;
    let root = &plain;
    Ok(match rw {
        // A binary node whose two children have the same dual: swapped.
        Rewrite::SwapOperands => pick(&matches(root, &|n| n.kids.len() == 2 && n.kids[0].ty == n.kids[1].ty && n.kids[0] != n.kids[1] && !n.name.starts_with("block.")), rng).map(|p| {
            let n = at(root, &p);
            (n.name.clone(), replace_at(root, &p, Tree { name: n.name.clone(), ty: n.ty, konst: n.konst, kids: vec![n.kids[1].clone(), n.kids[0].clone()] }))
        }),
        // Two variable leaves of the same dual and different names: exchanged.
        Rewrite::SwapLeaves => {
            let leaves = matches(root, &is_variable);
            let Some(a) = pick(&leaves, rng) else { return Ok(None) };
            let la = at(root, &a).clone();
            let same: Vec<Vec<usize>> = leaves.into_iter().filter(|p| { let l = at(root, p); l.ty == la.ty && l.name != la.name }).collect();
            pick(&same, rng).map(|b| {
                let lb = at(root, &b).clone();
                (format!("{} <-> {}", la.name, lb.name), replace_at(&replace_at(root, &a, lb.clone()), &b, la.clone()))
            })
        }
        // An integer literal of an index dual: plus or minus one or two.
        Rewrite::IndexLiteral => {
            let index = crate::wgsl::render::dual_ty(kingdom, "u32.index")?;
            pick(&matches(root, &|n| n.kids.is_empty() && n.name.starts_with("literal.") && n.ty == index), rng).and_then(|p| {
                let n = at(root, &p);
                let d = [-2.0f32, -1.0, 1.0, 2.0][(next(rng) % 4) as usize];
                let v = n.konst + d;
                (v >= 0.0).then(|| (format!("{} {} -> {v}", n.name, n.konst), replace_at(root, &p, Tree { konst: v, ..n.clone() })))
            })
        }
        // A counted loop's trip count (its first child, a literal) plus or minus one.
        Rewrite::LoopEnd => pick(&matches(root, &|n| n.name.starts_with("loop.") && n.kids.len() >= 2 && n.kids[0].kids.is_empty() && n.kids[0].name.starts_with("literal.") && n.kids[0].konst >= 1.0), rng).map(|p| {
            let n = at(root, &p);
            let c = &n.kids[0];
            let v = if c.konst >= 2.0 && next(rng).is_multiple_of(2) { c.konst - 1.0 } else { c.konst + 1.0 };
            let mut kids = n.kids.clone();
            kids[0] = Tree { konst: v, ..c.clone() };
            (format!("{} count {} -> {v}", n.name, c.konst), replace_at(root, &p, Tree { name: n.name.clone(), ty: n.ty, konst: n.konst, kids }))
        }),
        // `r = r + a * b` after a statement: r a register leaf of the kernel
        // whose dual has a setter at the statement's rung, a getter, an add
        // and a multiply; a and b values of that dual in the statement.
        Rewrite::InsertAccumulate => {
            let stmts: Vec<_> = kernel_statements(root).into_iter().filter(|s| !s.tree.kids.iter().any(|k| k.name.starts_with("block."))).collect();
            let Some(s) = pick(&stmts, rng) else { return Ok(None) };
            let Some(r) = rung(&s.tree.name) else { return Ok(None) };
            let regs = matches(root, &|n| is_variable(n) && n.name.starts_with("reg."));
            let Some(rp) = pick(&regs, rng) else { return Ok(None) };
            let reg = at(root, &rp).clone();
            let suffix = format!("_{r}");
            let Some(set) = kingdom.rows.iter().find(|x| x.name.starts_with("reg.set_") && x.name.ends_with(&suffix) && x.inputs.len() == 2 && x.inputs[0] == reg.ty) else { return Ok(None) };
            let v = set.inputs[1];
            let row = |pred: &dyn Fn(&crate::wgsl::loader::WgslRow) -> bool| kingdom.rows.iter().find(|x| pred(x)).map(|x| x.name.clone());
            let (Some(get), Some(add), Some(mul)) = (
                row(&|x| x.name.starts_with("reg.get") && x.inputs == [reg.ty] && x.output == v),
                row(&|x| x.name.ends_with(".add") && x.inputs == [v, v] && x.output == v),
                row(&|x| x.name.ends_with(".mul") && x.inputs == [v, v] && x.output == v),
            ) else {
                return Ok(None);
            };
            let values = matches(&s.tree, &|n| n.ty == v);
            let (Some(a), Some(b)) = (pick(&values, rng), pick(&values, rng)) else { return Ok(None) };
            let (a, b) = (at(&s.tree, &a).clone(), at(&s.tree, &b).clone());
            let sum = Tree::new(&add, v, vec![Tree::new(&get, v, vec![reg.clone()]), Tree::new(&mul, v, vec![a, b])]);
            let new = Tree::new(&set.name, set.output, vec![reg.clone(), sum]);
            Some((format!("{} = {} + a * b after statement {}", reg.name, reg.name, s.index), replace_statement(root, s.index, vec![s.tree.clone(), new], kingdom)?))
        }
        // A variable leaf replaced by another terminal row of its dual.
        Rewrite::ReplaceLeaf => pick(&matches(root, &is_variable), rng).and_then(|p| {
            let n = at(root, &p);
            let others: Vec<&str> = kingdom.rows.iter().filter(|r| r.terminal && r.inputs.is_empty() && r.output == n.ty && r.name != n.name && !r.name.starts_with("literal.")).map(|r| r.name.as_str()).collect();
            pick(&others, rng).map(|o| (format!("{} -> {o}", n.name), replace_at(root, &p, Tree::new(o, n.ty, vec![]))))
        }),
        // A binary op replaced by another row of the same signature.
        Rewrite::SwapOp => pick(&matches(root, &|n| n.kids.len() == 2 && row_inputs(n, kingdom).is_some()), rng).and_then(|p| {
            let n = at(root, &p);
            let ins = row_inputs(n, kingdom).unwrap_or(&[]);
            let others: Vec<&str> = kingdom.rows.iter().filter(|r| r.name != n.name && r.output == n.ty && r.inputs.as_slice() == ins && !r.name.starts_with("block.")).map(|r| r.name.as_str()).collect();
            pick(&others, rng).map(|o| (format!("{} -> {o}", n.name), replace_at(root, &p, Tree { name: o.to_string(), ty: n.ty, konst: n.konst, kids: n.kids.clone() })))
        }),
        Rewrite::DeleteStatement | Rewrite::DuplicateStatement => {
            let stmts = kernel_statements(root);
            match pick(&stmts, rng) {
                None => None,
                Some(s) => {
                    let with = if rw == Rewrite::DeleteStatement { Vec::new() } else { vec![s.tree.clone(), s.tree.clone()] };
                    Some((format!("statement {} ({})", s.index, s.tree.name), replace_statement(root, s.index, with, kingdom)?))
                }
            }
        }
        // Two adjacent statements of a block exchanged.
        Rewrite::SwapAdjacent => match pick(&matches(root, &|n| is_plain_block(&n.name) && statements(n).len() >= 2), rng) {
            None => None,
            Some(p) => {
                let b = at(root, &p);
                let mut stmts = statements(b);
                let i = (next(rng) % (stmts.len() as u64 - 1)) as usize;
                stmts.swap(i, i + 1);
                let r = rung(&b.name).ok_or_else(|| format!("{} has no rung", b.name))?;
                Some((format!("statements {i} and {} of {}", i + 1, b.name), replace_at(root, &p, rebuild(stmts, r, kingdom)?)))
            }
        },
        // A node wrapped in a unary row from its dual to its dual.
        Rewrite::WrapUnary => pick(&matches(root, &|n| !n.kids.is_empty() || is_variable(n)), rng).and_then(|p| {
            let n = at(root, &p);
            let unary: Vec<&str> = kingdom.rows.iter().filter(|r| r.inputs.len() == 1 && r.inputs[0] == n.ty && r.output == n.ty).map(|r| r.name.as_str()).collect();
            pick(&unary, rng).map(|u| (format!("{} wrapped in {u}", n.name), replace_at(root, &p, Tree::new(u, n.ty, vec![n.clone()]))))
        }),
        // A unary row from a dual to the same dual, replaced by its child.
        Rewrite::UnwrapUnary => pick(&matches(root, &|n| n.kids.len() == 1 && n.kids[0].ty == n.ty && row_inputs(n, kingdom).is_some()), rng).map(|p| {
            let n = at(root, &p);
            (format!("{} unwrapped", n.name), replace_at(root, &p, n.kids[0].clone()))
        }),
    })
}

/// A rewrite or rule child, checked.
#[derive(Clone, Debug)]
pub struct Child {
    pub tree: Tree,
    pub text: String,
    /// The operator's name and what it changed.
    pub operator: &'static str,
    pub what: String,
}

/// ONE MUTATION from the pool: the structural rules `rules` and every
/// generic rewrite. An operator is drawn among those whose pattern matches
/// the individual, its site with the RNG; the child through the layered
/// check; another draw when it is illegal, up to `tries`. None when no
/// draw gives a legal child.
pub fn mutate(root: &Tree, rules: &[Rule], kingdom: &WgslKingdom, rng: &mut u64, tries: usize) -> Result<Option<Child>, String> {
    let lit: Vec<Rule> = rules.iter().copied().filter(|r| !r.sites(root, kingdom).is_empty()).collect();
    let n = lit.len() + Rewrite::ALL.len();
    for _ in 0..tries {
        let k = (next(rng) % n as u64) as usize;
        let made = if k < lit.len() {
            let r = lit[k];
            let sites = r.sites(root, kingdom);
            let Some(site) = pick(&sites, rng) else { continue };
            r.apply(root, &site, kingdom).ok().map(|t| (r.name(), site.what, t))
        } else {
            let rw = Rewrite::ALL[k - lit.len()];
            apply(rw, root, kingdom, rng)?.map(|(what, t)| (rw.name(), what, t))
        };
        let Some((operator, what, tree)) = made else { continue };
        if let Ok((text, _)) = knob::check_text(&tree, kingdom) {
            return Ok(Some(Child { tree, text, operator, what }));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::safe_reader::read_safe;

    /// Every rewrite matches somewhere in the corpus, and its children that
    /// pass the layered check are kernels other than the parent; the pool
    /// draws legal children. Counts are printed.
    #[test]
    fn every_rewrite_matches_and_makes_checked_children() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/safewgsl");
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        let trees: Vec<Tree> = files.iter().filter_map(|f| read_safe(&std::fs::read_to_string(f).ok()?, &k).ok().map(|r| r.tree)).take(12).collect();
        let mut rng = 3u64;
        for rw in Rewrite::ALL {
            let (mut made, mut legal) = (0usize, 0usize);
            for t in &trees {
                for _ in 0..4 {
                    let Some((what, child)) = apply(rw, t, &k, &mut rng).unwrap() else { continue };
                    made += 1;
                    if let Ok((text, _)) = knob::check_text(&child, &k) {
                        legal += 1;
                        let parent = crate::wgsl::safe_karva::render_tree(t, &k).unwrap().text;
                        // A swap of equal-rendering parts, or the epilogue's only store
                        // deleted (its fallback is that store), renders the parent.
                        assert!(text != parent || matches!(rw, Rewrite::SwapOperands | Rewrite::SwapAdjacent | Rewrite::SwapLeaves | Rewrite::DeleteStatement), "{} {what}: the child is its parent", rw.name());
                    }
                }
            }
            println!("{}: {made} children, {legal} legal", rw.name());
            assert!(made > 0, "{} matches nothing in the corpus", rw.name());
        }
        let mut pool = 0usize;
        for t in &trees {
            if let Some(c) = mutate(t, &Rule::ALL, &k, &mut rng, 16).unwrap() {
                pool += 1;
                println!("pool: {} {}", c.operator, c.what.chars().take(80).collect::<String>());
            }
        }
        assert!(pool > 0);
    }
}
