//! The first mutations (`docs/PLAN_wgsl_lineage.md` §7 step 5): the Math
//! kingdoms' algebraic rewrites applied to pure float REGIONS of a
//! kernel's roots, gated by the oracle's numerical criterion on the device.
//!
//! A region is a maximal subtree of `arith`/`trig` rows the `Math`
//! datatype has an operator for (add, sub, mul, div, neg, sqrt, abs, exp,
//! log, pow; sin, cos, tan, tanh, asin, acos). Everything else under it is
//! an OPAQUE LEAF: a `let` use, an argument, a builtin, an `href`, a
//! literal, a versioned load, or a whole subtree of another class
//! (`convert`, `select`, `min`…). An opaque leaf enters the e-graph as
//! `(Var "L<n>")` and comes back by that index to its exact text, so the
//! versions guarantee (equal text, equal lineage) rides along for free and
//! a rewrite can neither move a load across a store nor change what a
//! leaf reads. The plan's first increment admits only regions whose
//! leaves hold NO load; regions with loads are counted, not mutated.
//!
//! Variants come from `extract::eclass_variants` over the bounded Algebra
//! family (identities, powers, distribute; a small iteration bound, as the
//! crate's non-confluence rule demands). A variant that needs an operator
//! with no WGSL row (`Protected*`) is dropped and counted; `Pow2`, `Pow3`
//! and `Inv` are spelled with the rows that exist.

use std::collections::BTreeMap;

use crate::extract::{eclass_variants, EclassFamily};
use crate::karva::{parse_math, MathNode};

/// One region of a root: where it sits and what the e-graph saw.
#[derive(Debug, Clone)]
pub(crate) struct Region {
    pub(crate) path: Vec<u8>,
    /// Operators inside the region.
    pub ops: usize,
    /// Whether a leaf (or an opaque subtree) reads a load.
    pub has_load: bool,
    /// The region as a `Math` term, leaves `(Var "L<n>")`.
    pub term: String,
    leaves: Vec<MathNode>,
}

/// Rows the `Math` datatype can rewrite, both ways.
const ROWS: &[(&str, &str, usize)] = &[
    ("arith.add", "Add", 2),
    ("arith.sub", "Sub", 2),
    ("arith.mul", "Mul", 2),
    ("arith.div", "Div", 2),
    ("arith.pow", "Pow", 2),
    ("arith.neg", "Neg", 1),
    ("arith.sqrt", "Sqrt", 1),
    ("arith.abs", "Abs", 1),
    ("arith.exp", "Exp", 1),
    ("arith.log", "Log", 1),
    ("trig.sin", "Sin", 1),
    ("trig.cos", "Cos", 1),
    ("trig.tan", "Tan", 1),
    ("trig.tanh", "Tanh", 1),
    ("trig.asin", "Asin", 1),
    ("trig.acos", "Acos", 1),
];

fn math_op(row: &str, arity: usize) -> Option<&'static str> {
    ROWS.iter().find(|(r, _, a)| *r == row && *a == arity).map(|(_, m, _)| *m)
}

fn row_of(math: &str) -> Option<(&'static str, usize)> {
    ROWS.iter().find(|(_, m, _)| *m == math).map(|(r, _, a)| (*r, *a))
}

fn render(n: &MathNode) -> String {
    match n {
        MathNode::Num(v) => format!("(Num {v:?})"),
        MathNode::Var(name) => format!("(Var {name:?})"),
        MathNode::App(ctor, children) => {
            let parts: Vec<String> = children.iter().map(render).collect();
            if parts.is_empty() {
                format!("({ctor})")
            } else {
                format!("({ctor} {})", parts.join(" "))
            }
        }
    }
}

fn reads_load(n: &MathNode) -> bool {
    match n {
        MathNode::Var(name) => name.starts_with("load."),
        MathNode::App(ctor, kids) => ctor.starts_with("load.") || kids.iter().any(reads_load),
        MathNode::Num(_) => false,
    }
}

/// Whether `n` is a rewritable node (a mapped row with the right arity).
fn mapped(n: &MathNode) -> bool {
    matches!(n, MathNode::App(ctor, kids) if math_op(ctor, kids.len()).is_some())
}

/// Render a region as a Math term, collecting its opaque leaves.
fn to_term(n: &MathNode, leaves: &mut Vec<MathNode>, ops: &mut usize, has_load: &mut bool) -> String {
    match n {
        MathNode::App(ctor, kids) if ctor == "literal.real" && kids.len() == 1 => {
            if let MathNode::Num(v) = &kids[0] {
                return format!("(Num {v:?})");
            }
            opaque(n, leaves, has_load)
        }
        MathNode::App(ctor, kids) => match math_op(ctor, kids.len()) {
            Some(op) => {
                *ops += 1;
                let parts: Vec<String> = kids.iter().map(|k| to_term(k, leaves, ops, has_load)).collect();
                format!("({op} {})", parts.join(" "))
            }
            None => opaque(n, leaves, has_load),
        },
        _ => opaque(n, leaves, has_load),
    }
}

fn opaque(n: &MathNode, leaves: &mut Vec<MathNode>, has_load: &mut bool) -> String {
    if reads_load(n) {
        *has_load = true;
    }
    let key = render(n);
    let i = match leaves.iter().position(|l| render(l) == key) {
        Some(i) => i,
        None => {
            leaves.push(n.clone());
            leaves.len() - 1
        }
    };
    format!("(Var \"L{i}\")")
}

/// The maximal rewritable regions of `root`, with at least `min_ops`
/// operators, in pre-order.
pub(crate) fn regions(root: &MathNode, min_ops: usize) -> Vec<Region> {
    fn walk(n: &MathNode, path: Vec<u8>, min_ops: usize, out: &mut Vec<Region>) {
        if mapped(n) {
            let mut leaves = Vec::new();
            let (mut ops, mut has_load) = (0usize, false);
            let term = to_term(n, &mut leaves, &mut ops, &mut has_load);
            if ops >= min_ops {
                out.push(Region { path, ops, has_load, term, leaves });
            }
            return;
        }
        if let MathNode::App(_, kids) = n {
            for (i, k) in kids.iter().enumerate() {
                let mut p = path.clone();
                p.push(i as u8);
                walk(k, p, min_ops, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, Vec::new(), min_ops, &mut out);
    out
}

/// A Math variant back into the kingdom's rows over the region's leaves;
/// `None` when it needs an operator with no row.
fn from_term(n: &MathNode, leaves: &[MathNode]) -> Option<MathNode> {
    match n {
        MathNode::Num(v) => Some(MathNode::App("literal.real".into(), vec![MathNode::Num(*v)])),
        MathNode::Var(name) => name.strip_prefix('L').and_then(|i| i.parse::<usize>().ok()).and_then(|i| leaves.get(i).cloned()),
        MathNode::App(ctor, kids) => {
            let back: Vec<MathNode> = kids.iter().map(|k| from_term(k, leaves)).collect::<Option<_>>()?;
            match (ctor.as_str(), back.as_slice()) {
                ("Pow2", [x]) => Some(MathNode::App("arith.mul".into(), vec![x.clone(), x.clone()])),
                ("Pow3", [x]) => Some(MathNode::App("arith.mul".into(), vec![MathNode::App("arith.mul".into(), vec![x.clone(), x.clone()]), x.clone()])),
                ("Inv", [x]) => Some(MathNode::App("arith.div".into(), vec![MathNode::App("literal.real".into(), vec![MathNode::Num(1.0)]), x.clone()])),
                _ => {
                    let (row, arity) = row_of(ctor)?;
                    if arity != back.len() {
                        return None;
                    }
                    Some(MathNode::App(row.into(), back))
                }
            }
        }
    }
}

fn replace_at(root: &MathNode, path: &[u8], with: &MathNode) -> MathNode {
    match path.split_first() {
        None => with.clone(),
        Some((&i, rest)) => match root {
            MathNode::App(ctor, kids) => MathNode::App(ctor.clone(), kids.iter().enumerate().map(|(k, c)| if k == i as usize { replace_at(c, rest, with) } else { c.clone() }).collect()),
            other => other.clone(),
        },
    }
}

/// One mutant: the root with a region rewritten.
#[derive(Debug, Clone)]
pub struct Mutant {
    pub region_path: Vec<u8>,
    /// The region's Math term and the variant that replaced it.
    pub from: String,
    pub to: String,
    pub(crate) root: MathNode,
}

impl Mutant {
    /// The mutated root as the reader renders a root.
    pub fn root_text(&self) -> String {
        render(&self.root)
    }
}

/// What the generator of mutants found for one root.
#[derive(Debug, Clone, Default)]
pub struct MutateStats {
    pub regions: usize,
    pub regions_with_loads: usize,
    /// Operators over all regions (the size of what the e-graph saw).
    pub region_ops: usize,
    pub variants: usize,
    pub dropped_no_row: usize,
}

/// Mutants of `root` from its load-free regions: up to `k` variants per
/// region from the Algebra family at `iters` iterations.
pub(crate) fn mutants(root: &MathNode, min_ops: usize, k: usize, iters: u32, with_loads: bool, stats: &mut MutateStats) -> Result<Vec<Mutant>, String> {
    let mut out = Vec::new();
    for region in regions(root, min_ops) {
        stats.regions += 1;
        stats.region_ops += region.ops;
        if region.has_load {
            stats.regions_with_loads += 1;
            if !with_loads {
                continue;
            }
        }
        let original = crate::homeotic::canonical(&region.term)?;
        let variants = eclass_variants(&region.term, EclassFamily::Algebra, k + 1, iters)?;
        for (_, v) in variants {
            if crate::homeotic::canonical(&v)? == original {
                continue;
            }
            let term = parse_math(&v)?;
            match from_term(&term, &region.leaves) {
                Some(rewritten) => {
                    stats.variants += 1;
                    out.push(Mutant { region_path: region.path.clone(), from: region.term.clone(), to: v.clone(), root: replace_at(root, &region.path, &rewritten) });
                }
                None => stats.dropped_no_row += 1,
            }
        }
    }
    Ok(out)
}

/// Mutants for every root of a function (roots as the reader renders
/// them), keyed by root index. `with_loads` admits regions whose opaque
/// leaves read a load (the second increment: a versioned load is a fixed
/// leaf like a `let`, so the rewrite cannot change what it reads).
pub fn mutants_of_roots(roots: &[String], min_ops: usize, k: usize, iters: u32, with_loads: bool) -> Result<(BTreeMap<usize, Vec<Mutant>>, MutateStats), String> {
    let mut stats = MutateStats::default();
    let mut out = BTreeMap::new();
    for (i, r) in roots.iter().enumerate() {
        let r = parse_math(r)?;
        let m = mutants(&r, min_ops, k, iters, with_loads, &mut stats)?;
        if !m.is_empty() {
            out.insert(i, m);
        }
    }
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_region_maps_to_a_math_term_with_opaque_leaves_and_back() {
        let root = parse_math(r#"(store.buffer.out.# (literal.index (Num 0.0)) (arith.add (arith.mul (Var "let.a@3") (literal.real (Num 1.0))) (convert.index_to_f32 (Var "load.local.i@1@v2"))))"#).unwrap();
        let rs = regions(&root, 1);
        assert_eq!(rs.len(), 1);
        let r = &rs[0];
        assert_eq!(r.path, vec![1]);
        assert_eq!(r.ops, 2);
        assert!(r.has_load, "the convert of a load is an opaque leaf that reads a load");
        assert_eq!(r.term, r#"(Add (Mul (Var "L0") (Num 1.0)) (Var "L1"))"#);
        let back = from_term(&parse_math(&r.term).unwrap(), &r.leaves).unwrap();
        assert_eq!(render(&back), render(&parse_math(r#"(arith.add (arith.mul (Var "let.a@3") (literal.real (Num 1.0))) (convert.index_to_f32 (Var "load.local.i@1@v2")))"#).unwrap()));
        let pow2 = from_term(&parse_math(r#"(Add (Pow2 (Var "L0")) (Inv (Var "L1")))"#).unwrap(), &r.leaves).unwrap();
        assert!(render(&pow2).starts_with("(arith.add (arith.mul (Var \"let.a@3\") (Var \"let.a@3\")) (arith.div (literal.real (Num 1.0))"));
        assert!(from_term(&parse_math(r#"(ProtectedSqrt (Var "L0"))"#).unwrap(), &r.leaves).is_none());
    }

    #[test]
    fn a_load_free_region_yields_variants_and_a_region_with_loads_is_counted_not_mutated() {
        let root = parse_math(r#"(store.buffer.out.# (literal.index (Num 0.0)) (arith.add (arith.mul (Var "let.a@3") (literal.real (Num 1.0))) (literal.real (Num 0.0))))"#).unwrap();
        let mut stats = MutateStats::default();
        let m = mutants(&root, 1, 3, 2, false, &mut stats).unwrap();
        assert_eq!((stats.regions, stats.regions_with_loads), (1, 0));
        assert!(!m.is_empty(), "x * 1 + 0 has simpler forms");
        assert!(m.iter().any(|mu| render(&mu.root) == r#"(store.buffer.out.# (literal.index (Num 0.0)) (Var "let.a@3"))"#), "{:?}", m.iter().map(|mu| render(&mu.root)).collect::<Vec<_>>());
        let loaded = parse_math(r#"(store.buffer.out.# (literal.index (Num 0.0)) (arith.add (arith.mul (Var "load.buffer.xs.#@v0") (literal.real (Num 1.0))) (literal.real (Num 0.0))))"#).unwrap();
        let mut stats = MutateStats::default();
        let m = mutants(&loaded, 1, 3, 2, false, &mut stats).unwrap();
        assert_eq!((stats.regions, stats.regions_with_loads, m.len()), (1, 1, 0));
        let mut stats = MutateStats::default();
        let m = mutants(&loaded, 1, 3, 2, true, &mut stats).unwrap();
        assert!(!m.is_empty(), "with loads admitted, the load is an opaque leaf");
    }
}
