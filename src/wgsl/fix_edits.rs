//! CORRECTNESS EDITS (2026-10-13, Andrew: operators for a population with
//! no correct kernel): single typed edits of a wrong kernel, each a small
//! step toward a kernel that computes H = ΦᵀΦ, every child through the
//! layered check (`knob::check_text`: the legality checker `check.rs`,
//! render, naga validation, read back). The structural rules
//! (`rules.rs`, `rules_more.rs`) keep a correct kernel's values; these
//! change a wrong kernel's.
//!
//! | edit | what it changes |
//! |---|---|
//! | swap-ij | in one statement, the row and column index sources exchanged (`lid.x` with `lid.y`, `wid.x` with `wid.y`, `gid.x` with `gid.y`) |
//! | index-const | an index literal ±1 or ±2 |
//! | loop-end | a tile loop's count ±1 (every loop starts at 0) |
//! | phi-layout | a Φ read's layout: `t * d + i` to `i * m + t`, or back |
//! | index-var | an index variable replaced by another of the same dual in the kernel |
//! | bound-to-cfg | an item loop's bound set to `cfg.m`, `cfg.d` or `cfg.r_rows` |
//! | store-accumulator | the value the epilogue stores made a register the kernel accumulates (`r = r + E`), or a lane of one |
//! | insert-accumulate | `r = r + a * b` inserted after a statement in a loop, `a`, `b` values the loop reads from memory |
//! | swap-op | `+` with `*`, `min` with `max` (index or real) |
//! | delete-statement, duplicate-statement | one statement removed or written twice |
//! | graft | a subtree replaced by one of the same dual from a correct kernel |
//!
//! The edit is chosen by the parent's closeness columns where one stands
//! out (`closeness.rs`): the transpose of the output nearly right while
//! most cells are wrong is a swap of i and j; a shift by one or two nearly
//! right, an index constant or a loop end ±1; cells left unwritten or an
//! output blind to Φ, the store made the accumulator. Otherwise at random.

use crate::wgsl::closeness;
use crate::wgsl::knob;
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::render::{dual_ty, safe_base, Tree};
use crate::wgsl::rules::{add_terms, kernel_statements, plain_value, replace_statement, rung};

/// A correctness edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Edit {
    SwapIJ,
    IndexConst,
    LoopEnd,
    PhiLayout,
    IndexVar,
    BoundToCfg,
    StoreAccumulator,
    InsertAccumulate,
    SwapOp,
    DeleteStatement,
    DuplicateStatement,
    Graft,
}

impl Edit {
    pub const ALL: [Edit; 12] = [
        Edit::SwapIJ,
        Edit::IndexConst,
        Edit::LoopEnd,
        Edit::PhiLayout,
        Edit::IndexVar,
        Edit::BoundToCfg,
        Edit::StoreAccumulator,
        Edit::InsertAccumulate,
        Edit::SwapOp,
        Edit::DeleteStatement,
        Edit::DuplicateStatement,
        Edit::Graft,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Edit::SwapIJ => "swap-ij",
            Edit::IndexConst => "index-const",
            Edit::LoopEnd => "loop-end",
            Edit::PhiLayout => "phi-layout",
            Edit::IndexVar => "index-var",
            Edit::BoundToCfg => "bound-to-cfg",
            Edit::StoreAccumulator => "store-accumulator",
            Edit::InsertAccumulate => "insert-accumulate",
            Edit::SwapOp => "swap-op",
            Edit::DeleteStatement => "delete-statement",
            Edit::DuplicateStatement => "duplicate-statement",
            Edit::Graft => "graft",
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

// ---- paths into a tree ----

/// The node at `path` (child indices from the root).
fn at<'t>(t: &'t Tree, path: &[usize]) -> &'t Tree {
    path.iter().fold(t, |n, &i| &n.kids[i])
}

/// `t` with the node at `path` replaced by `with`.
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

/// The paths of every node satisfying `p`, in pre-order.
fn paths(t: &Tree, p: &dyn Fn(&Tree) -> bool) -> Vec<Vec<usize>> {
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

fn any_node(t: &Tree, p: &dyn Fn(&Tree) -> bool) -> bool {
    p(t) || t.kids.iter().any(|k| any_node(k, p))
}

/// Whether the kingdom has the row `name` producing `ty` with `arity` children.
fn has_row(kingdom: &WgslKingdom, name: &str, ty: crate::geneframe::Ty, arity: usize) -> bool {
    kingdom.rows.iter().any(|r| r.name == name && r.output == ty && r.inputs.len() == arity)
}

// ---- the edits ----

/// The row and column sources: `index.of_col(axis.x_*)` and `index.of_row(axis.y_*)`.
fn ij_partner(t: &Tree) -> Option<(&'static str, &'static str)> {
    let a = t.kids.first().filter(|_| t.kids.len() == 1)?;
    match (t.name.as_str(), a.name.as_str()) {
        ("index.of_col", "axis.x_lid") => Some(("index.of_row", "axis.y_lid")),
        ("index.of_col", "axis.x_wid") => Some(("index.of_row", "axis.y_wid")),
        ("index.of_col", "axis.x_gid") => Some(("index.of_row", "axis.y_gid")),
        ("index.of_row", "axis.y_lid") => Some(("index.of_col", "axis.x_lid")),
        ("index.of_row", "axis.y_wid") => Some(("index.of_col", "axis.x_wid")),
        ("index.of_row", "axis.y_gid") => Some(("index.of_col", "axis.x_gid")),
        _ => None,
    }
}

fn swap_ij(t: &Tree, kingdom: &WgslKingdom) -> Tree {
    if let Some((wrap, axis)) = ij_partner(t) {
        let a = &t.kids[0];
        let axis_ty = kingdom.rows.iter().find(|r| r.name == axis).map_or(a.ty, |r| r.output);
        return Tree::new(wrap, t.ty, vec![Tree { name: axis.to_string(), ty: axis_ty, konst: a.konst, kids: a.kids.clone() }]);
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| swap_ij(k, kingdom)).collect() }
}

/// Every child of `edit` at every site (a description and the tree);
/// edits with a random part (insert-accumulate, graft) give a few.
fn children(edit: Edit, root: &Tree, library: &[Tree], kingdom: &WgslKingdom, rng: &mut u64) -> Result<Vec<(String, Tree)>, String> {
    // On the kernel's values: a loop count or a stride that is a knob
    // reference is edited as the number it is.
    let plain = crate::wgsl::params::to_literals(root, kingdom)?;
    let root = &plain;
    let index = dual_ty(kingdom, "u32.index")?;
    let real = dual_ty(kingdom, "f32.real")?;
    let mut out = Vec::new();
    match edit {
        Edit::SwapIJ => {
            for s in kernel_statements(root) {
                let has = |w: &str| any_node(&s.tree, &|n| n.name == w && ij_partner(n).is_some());
                if has("index.of_col") && has("index.of_row") && !crate::wgsl::rules::is_compound(&s.tree) {
                    out.push((format!("statement {} ({})", s.index, s.tree.name), replace_statement(root, s.index, vec![swap_ij(&s.tree, kingdom)], kingdom)?));
                }
            }
        }
        Edit::IndexConst => {
            for p in paths(root, &|n| n.name == "literal.index" && plain_value(n).is_some()) {
                let v = plain_value(at(root, &p)).unwrap_or(0) as i64;
                for d in [-2i64, -1, 1, 2] {
                    if v + d >= 0 {
                        out.push((format!("index literal {v} -> {}", v + d), replace_at(root, &p, Tree::literal("literal.index", index, (v + d) as f32))));
                    }
                }
            }
        }
        Edit::LoopEnd => {
            for p in paths(root, &|n| safe_base(&n.name).starts_with("loop.tile") && n.kids.first().and_then(plain_value).is_some()) {
                let lp = at(root, &p);
                let c = &lp.kids[0];
                let v = plain_value(c).unwrap_or(0);
                for nv in [v.saturating_sub(1), v + 1] {
                    if nv >= 1 && nv != v {
                        let mut kids = lp.kids.clone();
                        kids[0] = Tree { konst: nv as f32, ..c.clone() };
                        out.push((format!("{} count {v} -> {nv}", lp.name), replace_at(root, &p, Tree { name: lp.name.clone(), ty: lp.ty, konst: lp.konst, kids })));
                    }
                }
            }
        }
        Edit::PhiLayout => {
            for p in paths(root, &|n| matches!(n.name.as_str(), "load.phi_safe" | "load.phi4_safe") && n.kids.len() == 1) {
                if let Some((x, what)) = flip_layout(&at(root, &p).kids[0], index) {
                    let load = at(root, &p);
                    out.push((what, replace_at(root, &p, Tree { name: load.name.clone(), ty: load.ty, konst: load.konst, kids: vec![x] })));
                }
            }
        }
        Edit::IndexVar => {
            let is_var = |n: &Tree| n.ty == index && (ij_partner(n).is_some() || (n.name.starts_with("reg.get_i") && n.kids.len() == 1) || n.name.starts_with("safe.tile_index") || n.name == "safe.counter_index" || n.name == "index.of_lidx");
            let ps = paths(root, &is_var);
            let mut vars: Vec<Tree> = Vec::new();
            for p in &ps {
                let v = at(root, p).clone();
                if !vars.contains(&v) {
                    vars.push(v);
                }
            }
            for p in &ps {
                let cur = at(root, p);
                for v in vars.iter().filter(|v| *v != cur) {
                    out.push((format!("{} -> {}", cur.to_sexpr(), v.to_sexpr()), replace_at(root, p, v.clone())));
                }
            }
        }
        Edit::BoundToCfg => {
            for p in paths(root, &|n| safe_base(&n.name).starts_with("loop.items") && n.kids.iter().any(|k| k.name.starts_with("bound."))) {
                let lp = at(root, &p);
                let (stem, cur) = ["boundm", "boundd", "boundrows"].iter().find_map(|b| lp.name.strip_suffix(b).map(|s| (s.to_string(), *b))).unwrap_or_default();
                for (b, leaf) in [("boundm", "bound.m"), ("boundd", "bound.d"), ("boundrows", "bound.rows")] {
                    if b == cur || stem.is_empty() {
                        continue;
                    }
                    let name = format!("{stem}{b}");
                    let Some(leaf_row) = kingdom.rows.iter().find(|r| r.name == leaf) else { continue };
                    if !has_row(kingdom, &name, lp.ty, lp.kids.len()) {
                        continue;
                    }
                    let kids = lp.kids.iter().map(|k| if k.name.starts_with("bound.") { Tree::new(leaf, leaf_row.output, vec![]) } else { k.clone() }).collect();
                    out.push((format!("{} -> {name}", lp.name), replace_at(root, &p, Tree { name, ty: lp.ty, konst: lp.konst, kids })));
                }
            }
        }
        Edit::StoreAccumulator => {
            let accs = accumulators(root, kingdom, real)?;
            for p in paths(root, &|n| safe_base(&n.name).starts_with("store.h") && n.kids.len() == 2) {
                let st = at(root, &p);
                for (what, v) in &accs {
                    if v.ty == st.kids[1].ty && *v != st.kids[1] {
                        let mut kids = st.kids.clone();
                        kids[1] = v.clone();
                        out.push((format!("the store's value -> {what}"), replace_at(root, &p, Tree { name: st.name.clone(), ty: st.ty, konst: st.konst, kids })));
                    }
                }
            }
        }
        Edit::InsertAccumulate => {
            let stmts: Vec<_> = kernel_statements(root).into_iter().filter(|s| s.in_loop && !crate::wgsl::rules::is_compound(&s.tree)).collect();
            let regs: Vec<String> = f32_registers_read(root);
            for _ in 0..8 {
                let (Some(s), Some(r)) = (pick(&stmts, rng), pick(&regs, rng)) else { break };
                let Some(r_at) = rung(&s.tree.name) else { continue };
                let loads = paths(&s.tree, &|n| n.ty == real && matches!(safe_base(&n.name), "load.phi_safe" | "wg.get_safe" | "shape.access_0" | "shape.access_1" | "shape.access_2" | "shape.access_3") && any_node(n, &|m| safe_base(&m.name).starts_with("load.") || safe_base(&m.name).starts_with("wg.get")));
                let (Some(a), Some(b)) = (pick(&loads, rng), pick(&loads, rng)) else { continue };
                let (a, b) = (at(&s.tree, &a).clone(), at(&s.tree, &b).clone());
                let get = reg_get(&r, kingdom)?;
                let value = Tree::new("arith.add", real, vec![get, Tree::new("arith.mul", real, vec![a, b])]);
                let setter = format!("reg.set_f_{r_at}");
                let Some(row) = kingdom.rows.iter().find(|x| x.name == setter) else { continue };
                let leaf_ty = kingdom.rows.iter().find(|x| x.name == r).map(|x| x.output).ok_or_else(|| format!("no register {r}"))?;
                let new = Tree::new(&setter, row.output, vec![Tree::new(&r, leaf_ty, vec![]), value]);
                out.push((format!("{r} += a * b after statement {}", s.index), replace_statement(root, s.index, vec![s.tree.clone(), new], kingdom)?));
            }
        }
        Edit::SwapOp => {
            let pairs = [("arith.add", "arith.mul"), ("arith.mul", "arith.add"), ("arith.min", "arith.max"), ("arith.max", "arith.min"), ("index.add", "index.mul"), ("index.mul", "index.add"), ("index.min", "index.max"), ("index.max", "index.min")];
            for (from, to) in pairs {
                for p in paths(root, &|n| n.name == from && n.kids.len() == 2) {
                    let n = at(root, &p);
                    if has_row(kingdom, to, n.ty, 2) {
                        out.push((format!("{from} -> {to}"), replace_at(root, &p, Tree { name: to.to_string(), ty: n.ty, konst: n.konst, kids: n.kids.clone() })));
                    }
                }
            }
        }
        Edit::DeleteStatement | Edit::DuplicateStatement => {
            for s in kernel_statements(root) {
                let with = if edit == Edit::DeleteStatement { Vec::new() } else { vec![s.tree.clone(), s.tree.clone()] };
                out.push((format!("statement {} ({})", s.index, s.tree.name), replace_statement(root, s.index, with, kingdom)?));
            }
        }
        Edit::Graft => {
            let donors: Vec<&Tree> = library.iter().collect();
            let points = paths(root, &|n| !n.kids.is_empty());
            for _ in 0..8 {
                let (Some(p), Some(d)) = (pick(&points[1.min(points.len())..], rng), pick(&donors, rng)) else { break };
                let target = at(root, &p);
                let same = paths(d, &|n| n.ty == target.ty && !n.kids.is_empty());
                let Some(q) = pick(&same, rng) else { continue };
                let sub = at(d, &q).clone();
                if sub != *target {
                    out.push((format!("{} from a correct kernel", target.name), replace_at(root, &p, sub)));
                }
            }
        }
    }
    Ok(out)
}

/// A Φ read's index flipped between the layouts `t * d + i` (item rows of
/// d) and `i * m + t` (column rows of m): the term multiplied by `cfg.d`
/// (or `cfg.m`) becomes the multiplier's partner, the rest the multiplied.
fn flip_layout(x: &Tree, index: crate::geneframe::Ty) -> Option<(Tree, String)> {
    let mut terms = Vec::new();
    add_terms(x, &mut terms);
    let is_count = |n: &Tree, f: &str| n.name == "index.of_count" && n.kids.first().is_some_and(|l| l.name == f);
    let mut found: Option<(usize, Tree, Tree, &str)> = None;
    for (i, t) in terms.iter().enumerate() {
        if t.name != "index.mul" || t.kids.len() != 2 {
            continue;
        }
        for (f, to) in [("load.cfg_d", "load.cfg_m"), ("load.cfg_m", "load.cfg_d")] {
            for k in 0..2 {
                if is_count(&t.kids[k], f) && found.is_none() {
                    let count = &t.kids[k];
                    let flipped = Tree { name: count.name.clone(), ty: count.ty, konst: count.konst, kids: vec![Tree { name: to.to_string(), ..count.kids[0].clone() }] };
                    found = Some((i, t.kids[1 - k].clone(), flipped, f));
                }
            }
        }
    }
    let (i, multiplied, count, f) = found?;
    let rest: Vec<Tree> = terms.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, t)| (*t).clone()).collect();
    let rest = rest.into_iter().reduce(|a, b| Tree::new("index.add", index, vec![a, b]))?;
    let new = Tree::new("index.add", index, vec![Tree::new("index.mul", index, vec![rest, count]), multiplied]);
    Some((new, format!("Φ layout over {f} flipped")))
}

/// A register's value read (`reg.get_f reg.f3`).
fn reg_get(leaf: &str, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let kind = leaf.strip_prefix("reg.").and_then(|r| r.chars().next()).ok_or_else(|| format!("{leaf} is not a register"))?;
    let getter = format!("reg.get_{kind}");
    let row = kingdom.rows.iter().find(|r| r.name == getter).ok_or_else(|| format!("no row {getter}"))?;
    let leaf_row = kingdom.rows.iter().find(|r| r.name == leaf).ok_or_else(|| format!("no register {leaf}"))?;
    Ok(Tree::new(&getter, row.output, vec![Tree::new(leaf, leaf_row.output, vec![])]))
}

/// The f32 registers the kernel reads somewhere.
fn f32_registers_read(root: &Tree) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in paths(root, &|n| n.name == "reg.get_f" && n.kids.len() == 1) {
        let r = at(root, &p).kids[0].name.clone();
        if !out.contains(&r) {
            out.push(r);
        }
    }
    out
}

/// The kernel's accumulators as f32 values: every register written as
/// `r = r + E`, an f32 one read whole, a vec4 one lane by lane.
fn accumulators(root: &Tree, kingdom: &WgslKingdom, real: crate::geneframe::Ty) -> Result<Vec<(String, Tree)>, String> {
    let mut regs: Vec<String> = Vec::new();
    for s in kernel_statements(root) {
        let t = &s.tree;
        if !(t.name.starts_with("reg.set_") && t.kids.len() == 2 && t.kids[1].name == "arith.add") {
            continue;
        }
        let r = &t.kids[0].name;
        if t.kids[1].kids.iter().any(|k| k.name.starts_with("reg.get") && k.kids.first().is_some_and(|l| l.name == *r)) && !regs.contains(r) {
            regs.push(r.clone());
        }
    }
    let mut out = Vec::new();
    for r in regs {
        let get = reg_get(&r, kingdom)?;
        if get.ty == real {
            out.push((r.clone(), get));
        } else if get.name == "reg.get_v" {
            for j in 0..4 {
                let lane = format!("shape.access_{j}");
                if has_row(kingdom, &lane, real, 1) {
                    out.push((format!("{r}.{}", ["x", "y", "z", "w"][j]), Tree::new(&lane, real, vec![get.clone()])));
                }
            }
        }
    }
    Ok(out)
}

/// The edit the parent's closeness columns point to, when one stands out:
/// most cells wrong (`within` at least one half) while the transpose, or a
/// shift by one or two, is nearly right (at most 0.05); or many cells
/// unwritten or the output blind to Φ (`written` or `sensitivity` at least
/// one half).
pub fn indicated(c: &[f64; closeness::N]) -> Option<Edit> {
    let col = |name: &str| closeness::COLUMNS.iter().position(|x| *x == name).map_or(1.0, |i| c[i]);
    let wrong = col("within") >= 0.5;
    if wrong && col("transpose") <= 0.05 {
        return Some(Edit::SwapIJ);
    }
    if wrong && col("shift") <= 0.05 {
        return Some(Edit::IndexConst);
    }
    if col("written") >= 0.5 || col("sensitivity") >= 0.5 {
        return Some(Edit::StoreAccumulator);
    }
    None
}

/// A correctness edit's child, checked.
#[derive(Clone, Debug)]
pub struct FixChild {
    pub tree: Tree,
    pub text: String,
    pub edit: Edit,
    pub what: String,
    /// Whether the closeness columns chose the edit.
    pub indicated: bool,
    /// Children the layered check found illegal before this one.
    pub illegal: usize,
}

/// ONE CORRECTNESS EDIT of `root`: the edit the closeness columns indicate
/// (for the first half of the tries, while it has a child; shift adds the
/// loop end ±1 to the index literals), else one drawn at random among those
/// with a child; the site at random; the child through the layered check,
/// another drawn when it is illegal. An error when `tries` draws give no
/// legal child.
pub fn fix_edit(root: &Tree, c: &[f64; closeness::N], library: &[Tree], kingdom: &WgslKingdom, rng: &mut u64, tries: usize) -> Result<FixChild, String> {
    let want = indicated(c);
    let mut illegal = 0usize;
    let mut last = String::new();
    for t in 0..tries {
        let (edit, by_columns) = match want {
            Some(e) if t < tries.div_ceil(2) => {
                if e == Edit::IndexConst && next(rng).is_multiple_of(2) {
                    (Edit::LoopEnd, true)
                } else {
                    (e, true)
                }
            }
            _ => (Edit::ALL[(next(rng) % Edit::ALL.len() as u64) as usize], false),
        };
        let made = children(edit, root, library, kingdom, rng)?;
        let Some((what, child)) = pick(&made, rng) else { continue };
        match knob::check_text(&child, kingdom) {
            Ok((text, _)) => return Ok(FixChild { tree: child, text, edit, what, indicated: by_columns, illegal }),
            Err((stage, e)) => {
                illegal += 1;
                last = format!("{} {what}: {}: {}", edit.name(), stage.name(), e.chars().take(160).collect::<String>());
            }
        }
    }
    Err(format!("no legal correctness edit in {tries} draws ({illegal} illegal; the last: {last})"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::safe_reader::read_safe;

    fn kernels(k: &WgslKingdom) -> Vec<(String, Tree)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for dir in ["tests/corpus/safewgsl", "logs/HGP/001/popR_rules/texts"] {
            if let Ok(rd) = std::fs::read_dir(root.join(dir)) {
                files.extend(rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")));
            }
        }
        files.sort();
        files.into_iter().filter_map(|f| Some((f.file_name()?.to_string_lossy().into_owned(), read_safe(&std::fs::read_to_string(&f).ok()?, k).ok()?.tree))).collect()
    }

    /// Every edit has children on the kernels, and a drawn child of every
    /// edit, when one passes the layered check, is a different kernel; the
    /// share that passes is printed by edit.
    #[test]
    fn every_edit_makes_checked_children_that_differ_from_the_parent() {
        let k = WgslKingdom::load_safe();
        let all = kernels(&k);
        let library: Vec<Tree> = all.iter().take(8).map(|(_, t)| t.clone()).collect();
        let mut rng = 11u64;
        for edit in Edit::ALL {
            let (mut made, mut legal) = (0usize, 0usize);
            for (name, t) in all.iter().take(20) {
                let kids = children(edit, t, &library, &k, &mut rng).unwrap_or_else(|e| panic!("{name} {}: {e}", edit.name()));
                for (what, child) in kids.iter().take(6) {
                    made += 1;
                    if let Ok((text, _)) = knob::check_text(child, &k) {
                        legal += 1;
                        let parent = crate::wgsl::safe_karva::render_tree(t, &k).unwrap().text;
                        assert!(text != parent || matches!(edit, Edit::IndexVar | Edit::Graft), "{name} {} {what}: the child is its parent", edit.name());
                    }
                }
            }
            println!("{}: {made} children, {legal} legal", edit.name());
            assert!(made > 0, "{} makes no child on 20 kernels", edit.name());
        }
    }

    /// The columns choose the edit: a transposed output swaps i and j, a
    /// shifted one moves an index, unwritten cells store the accumulator,
    /// and nothing standing out leaves the draw at random; the chosen edit
    /// gives a legal child of a corpus kernel.
    #[test]
    fn the_closeness_columns_choose_the_edit() {
        let col = |pairs: &[(&str, f64)]| {
            let mut c = [0.6f64; closeness::N];
            for (n, v) in pairs {
                c[closeness::COLUMNS.iter().position(|x| x == n).unwrap()] = *v;
            }
            c
        };
        assert_eq!(indicated(&col(&[("transpose", 0.01), ("within", 0.9), ("written", 0.0), ("sensitivity", 0.0)])), Some(Edit::SwapIJ));
        assert_eq!(indicated(&col(&[("shift", 0.02), ("within", 0.9), ("written", 0.0), ("sensitivity", 0.0)])), Some(Edit::IndexConst));
        assert_eq!(indicated(&col(&[("written", 0.8)])), Some(Edit::StoreAccumulator));
        assert_eq!(indicated(&col(&[("written", 0.1), ("sensitivity", 0.1)])), None);
        let k = WgslKingdom::load_safe();
        let all = kernels(&k);
        let library: Vec<Tree> = all.iter().take(4).map(|(_, t)| t.clone()).collect();
        let mut rng = 5u64;
        let (_, t) = &all[0];
        let c = fix_edit(t, &col(&[("written", 0.8)]), &library, &k, &mut rng, 64).unwrap_or_else(|e| panic!("{e}"));
        println!("{} {} ({} illegal before)", c.edit.name(), c.what, c.illegal);
    }
}
