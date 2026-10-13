//! MORE STRUCTURAL REWRITE RULES (2026-10-13, Andrew's list for the beam):
//! small changes that keep a kernel's values, each with its preconditions,
//! its inverse where one makes sense, and every child through the layered
//! checker (`knob::check_text`) before it is used. The rule catalogue and
//! the statement editor are `rules.rs`; this module holds the rules added
//! to it:
//!
//! - drop-dead-store (mined: "register initialisation removed", the largest
//!   class of `docs/mined_rewrites/rules.tsv`): every write of a register
//!   nothing reads is removed.
//! - unroll-k4: unroll-k by four (`rules::unroll_k_by`).
//! - roll-k: the inverse of unroll-k and unroll-k4 (same-register unrolls).
//! - split-accumulator: a K loop's accumulations alternate between the
//!   register and a fresh partner, added once after the loop.
//! - wg-shape: the workgroup reshaped at the same thread count (16 x 16,
//!   32 x 8, 8 x 32), every stride that depends on it a knob reference.
//! - hoist-load: a value read from memory that does not change in a loop is
//!   read once before it, into a fresh register.
//! - pad-shared: a workgroup array padded by one element a row (row width
//!   W, the K stride of its reads): index x is read at x + x / W.
//! - grow-cols / shrink-cols: the column analogue of grow-rows and
//!   shrink-rows; a thread's columns 4..7 read the column tile's vec4 at
//!   `2 x + 1` into twin storages, the epilogue reads them at element
//!   r + by of the private array.
//! - double-buffer: the item loop's loader moved one pass ahead into the
//!   other half of each tile, one barrier a pass instead of two.
//!
//! Symmetric-half is not here: the kingdom lets a thread store only its own
//! cell of h (checker layer 2, `store_not_owned_cell`) and the fitness
//! dispatches h in row bands, so a mirrored write is outside both.

use std::collections::{BTreeMap, BTreeSet};

use crate::wgsl::knob;
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::params::to_literals;
use crate::wgsl::render::{dual_ty, safe_base, Tree};
use crate::wgsl::rules::{
    any_node, k_loops, k_steps, kernel_statements, leaf_positions, plain_value, re_k, read_keys, rebuild, replace_statement, rewrite_k_loop, rung, statements, strip_k_constant, target_key, tile_counter_name, unroll_plan, Site, Stmt,
};
use crate::wgsl::target::Target;

// ---- shared pieces ----

/// A register leaf (`reg.v6`).
fn is_register(t: &Tree) -> bool {
    t.kids.is_empty() && t.name.starts_with("reg.") && t.name[4..].starts_with(|c: char| c.is_ascii_lowercase())
}

/// The kind of a register leaf: `v` (vec4), `f` (f32), `i` (index), `b` (flag).
fn register_kind(leaf: &str) -> Option<&str> {
    let rest = leaf.strip_prefix("reg.")?;
    let kind = rest.trim_end_matches(|c: char| c.is_ascii_digit() || c == 'r');
    matches!(kind, "v" | "f" | "i" | "b").then_some(kind)
}

/// A row of the kingdom by name, or a bug in our software.
fn row_exists(kingdom: &WgslKingdom, name: &str) -> Result<(), String> {
    if kingdom.rows.iter().any(|r| r.name == name) {
        Ok(())
    } else {
        Err(format!("our software has a bug: the kingdom has no row {name}"))
    }
}

/// A register of `kind` the kernel never names (the kingdom's own leaves,
/// lowest number first), marked used.
fn fresh_register(kind: &str, used: &mut BTreeSet<String>, kingdom: &WgslKingdom) -> Result<String, String> {
    let fresh = (0..64).map(|n| format!("reg.{kind}{n}")).find(|c| !used.contains(c) && kingdom.rows.iter().any(|r| r.name == *c)).ok_or_else(|| format!("no free {kind} register"))?;
    used.insert(fresh.clone());
    Ok(fresh)
}

/// The leaves a kernel names.
fn names_used(root: &Tree) -> BTreeSet<String> {
    leaf_positions(root).into_keys().collect()
}

/// A register's value read: `(reg.get_v reg.v14)`.
fn get_register(leaf: &str, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let kind = register_kind(leaf).ok_or_else(|| format!("{leaf} is not a register"))?;
    let getter = format!("reg.get_{kind}");
    let row = kingdom.rows.iter().find(|r| r.name == getter).ok_or_else(|| format!("our software has a bug: no row {getter}"))?;
    let leaf_row = kingdom.rows.iter().find(|r| r.name == leaf).ok_or_else(|| format!("our software has a bug: no register {leaf}"))?;
    Ok(Tree::new(&getter, row.output, vec![Tree::new(leaf, leaf_row.output, vec![])]))
}

/// A whole-register write at `rung`: `(reg.set_v_L1 reg.v14 value)`.
fn set_register(leaf: &str, value: Tree, rung: &str, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let kind = register_kind(leaf).ok_or_else(|| format!("{leaf} is not a register"))?;
    let setter = format!("reg.set_{kind}_{rung}");
    let row = kingdom.rows.iter().find(|r| r.name == setter).ok_or_else(|| format!("our software has a bug: no row {setter}"))?;
    let leaf_row = kingdom.rows.iter().find(|r| r.name == leaf).ok_or_else(|| format!("our software has a bug: no register {leaf}"))?;
    Ok(Tree::new(&setter, row.output, vec![Tree::new(leaf, leaf_row.output, vec![]), value]))
}

/// The zero of a register's kind: `0.0`, or the zero vec4 splat.
fn zero_of(kind: &str, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let f = dual_ty(kingdom, "f32.real")?;
    let zero = Tree::literal("literal.real", f, 0.0);
    match kind {
        "f" => Ok(zero),
        "v" => {
            let v = dual_ty(kingdom, "vec4<f32>.real")?;
            row_exists(kingdom, "shape.splat4")?;
            Ok(Tree::new("shape.splat4", v, vec![zero]))
        }
        other => Err(format!("no zero for a {other} register")),
    }
}

/// The statement whose node is `node` (a loop), with its index.
fn statement_at(root: &Tree, node: usize) -> Option<Stmt> {
    kernel_statements(root).into_iter().find(|s| s.node == node)
}

/// `t` with every leaf named in `map` renamed.
fn rename_leaves(t: &Tree, map: &BTreeMap<String, String>) -> Tree {
    if t.kids.is_empty() {
        if let Some(n) = map.get(&t.name) {
            return Tree { name: n.clone(), ..t.clone() };
        }
        return t.clone();
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| rename_leaves(k, map)).collect() }
}

/// A node with a side effect beyond its value: an atomic, a store, a
/// workgroup write or reduction, a barrier.
fn has_effect(t: &Tree) -> bool {
    any_node(t, &|n| {
        let b = safe_base(&n.name);
        b.starts_with("atomic.") || b.starts_with("store.") || b.starts_with("wg.set") || b.starts_with("wg.reduce") || b.starts_with("sync.") || b.contains("barrier")
    })
}

/// A K-loop statement's accumulation shape: `r = r + E` with E reading
/// neither r nor any register written elsewhere in the body. The register.
fn accumulation(st: &Tree, body_targets: &BTreeSet<String>) -> Result<String, String> {
    let key = target_key(st).filter(|k| k.starts_with("reg.")).ok_or_else(|| format!("{} in the K loop assigns no register", st.name))?;
    let add = st.kids.get(1).filter(|v| st.name.starts_with("reg.set_") && st.kids.len() == 2 && v.name == "arith.add" && v.kids.len() == 2).ok_or_else(|| format!("{key} is not accumulated as {key} + E"))?;
    let self_read = |k: &Tree| k.name.starts_with("reg.get") && k.kids.first().is_some_and(|l| l.name == key);
    let e = match (self_read(&add.kids[0]), self_read(&add.kids[1])) {
        (true, false) => &add.kids[1],
        (false, true) => &add.kids[0],
        _ => return Err(format!("{key} is not accumulated as {key} + E")),
    };
    let mut reads = BTreeSet::new();
    read_keys(e, &mut reads);
    if let Some(r) = reads.iter().find(|r| **r == key || body_targets.contains(*r)) {
        return Err(format!("{key}'s term reads {r}, written in the loop"));
    }
    if has_effect(e) {
        return Err(format!("{key}'s term has a side effect"));
    }
    Ok(key)
}

// ---- RULE: DROP DEAD STORES (mined) ----

/// The registers a kernel reads: a register leaf anywhere but as the
/// target of a register write (a whole write or a lane write).
fn registers_read(t: &Tree, out: &mut BTreeSet<String>) {
    let write = t.name.starts_with("reg.set_");
    for (i, k) in t.kids.iter().enumerate() {
        if is_register(k) && !(write && i == 0) {
            out.insert(k.name.clone());
        }
        registers_read(k, out);
    }
}

/// The register a statement writes (whole or a lane), if it is a register write.
fn written_register(s: &Tree) -> Option<&str> {
    (s.name.starts_with("reg.set_") && s.kids.first().is_some_and(is_register)).then(|| s.kids[0].name.as_str())
}

/// Every register written and never read, with the tree-order indices of
/// its writes; a write whose value has a side effect keeps its register out.
fn dead_registers(root: &Tree) -> BTreeMap<String, Vec<usize>> {
    let mut read = BTreeSet::new();
    registers_read(root, &mut read);
    let mut dead: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut effect: BTreeSet<String> = BTreeSet::new();
    for s in kernel_statements(root) {
        if let Some(r) = written_register(&s.tree).filter(|r| !read.contains(*r)) {
            if s.tree.kids[1..].iter().any(has_effect) {
                effect.insert(r.to_string());
            }
            dead.entry(r.to_string()).or_default().push(s.index);
        }
    }
    dead.retain(|r, _| !effect.contains(r));
    dead
}

/// RULE: DROP DEAD STORES. One site per register written and never read
/// (`i0 = 0u` in most kernels of the saved populations): all its writes go.
/// The identity: nothing observes the register.
pub fn drop_dead_store_sites(root: &Tree) -> Vec<Site> {
    dead_registers(root).into_iter().enumerate().map(|(i, (r, w))| Site { index: i, what: format!("{r}: {} writes, never read", w.len()) }).collect()
}

pub fn drop_dead_store(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let (_, writes) = dead_registers(root).into_iter().nth(site.index).ok_or("our software has a bug: the dead register is gone")?;
    let mut t = root.clone();
    for &i in writes.iter().rev() {
        t = replace_statement(&t, i, Vec::new(), kingdom)?;
    }
    Ok(t)
}

// ---- RULES: UNROLL K BY FOUR, AND ROLL AN UNROLL BACK ----

pub fn unroll_k4_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = to_literals(root, kingdom) else { return Vec::new() };
    k_loops(&plain).into_iter().filter_map(|(n, d, lp)| unroll_plan(n, d, &lp, 4).ok().map(|_| Site { index: n, what: format!("K loop of {} statements", statements(&lp.kids[1]).len()) })).collect()
}

pub fn unroll_k4(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    crate::wgsl::rules::unroll_k_by(root, site.index, 4, kingdom)
}

/// What rolling an unrolled K loop needs: the unroll factor, the K stride,
/// and the body indices of the statements of row m k (they stay).
#[derive(Clone, Debug)]
pub struct RollPlan {
    pub factor: u32,
    pub stride: u32,
    pub keep: Vec<usize>,
}

/// THE PRECONDITIONS of rolling the K loop at `node` (unrolled by m = 2 or
/// 4 into the same registers): every statement of its body is an
/// accumulation `r = r + E` reading the tiles at one K row m k + j, E
/// reading no register the body writes; each register's statements, in
/// body order, are rows 0, 1, .., m - 1 again and again, and the statement
/// of row j is the one of row 0 read j rows on. The rolled loop adds each
/// register's terms in the same order.
pub fn roll_plan(depth: usize, lp: &Tree, index: crate::geneframe::Ty) -> Result<RollPlan, String> {
    let counter = tile_counter_name(depth);
    let body = statements(&lp.kids[1]);
    let targets: BTreeSet<String> = body.iter().filter_map(target_key).collect();
    let (mut factor, mut stride) = (None, None);
    let mut rows = Vec::with_capacity(body.len());
    for st in &body {
        let (mut ks, mut bad) = (BTreeSet::new(), false);
        k_steps(st, counter, &mut ks, &mut bad);
        let steps: Vec<(u32, u32, u32)> = ks.into_iter().collect();
        let (m, s, j) = match (bad, steps.as_slice()) {
            (false, [(m, s, j)]) if *m >= 2 => (*m, *s, *j),
            _ => return Err(format!("{} in the K loop does not read one K row of an unroll", st.name)),
        };
        if factor.is_some_and(|f| f != m) || stride.is_some_and(|x| x != s) {
            return Err("the K loop's statements unroll by different factors or strides".into());
        }
        let others: BTreeSet<String> = targets.iter().filter(|t| Some((*t).clone()) != target_key(st)).cloned().collect();
        let key = accumulation(st, &others)?;
        factor = Some(m);
        stride = Some(s);
        rows.push((key, j));
    }
    let (m, s) = (factor.ok_or("an empty K loop")?, stride.ok_or("an empty K loop")?);
    // Each register's rows in body order: 0, 1, .., m - 1, repeated; the
    // statement of row j the one of row 0 read j rows on.
    let mut last: BTreeMap<&str, (u32, usize)> = BTreeMap::new();
    let mut keep = Vec::new();
    for (i, (key, j)) in rows.iter().enumerate() {
        let want = last.get(key.as_str()).map_or(0, |(pj, _)| (pj + 1) % m);
        if *j != want {
            return Err(format!("{key} adds K row {j} where row {want} comes next"));
        }
        if *j == 0 {
            keep.push(i);
        } else {
            let (_, first) = last[key.as_str()];
            let rolled = re_k(&body[first], counter, 1, 0, index);
            let expect = re_k(&strip_k_constant(&rolled, s), counter, m, *j, index);
            if expect != body[i] {
                return Err(format!("{key}'s row {j} is not its row 0 read {j} rows on"));
            }
        }
        let first = if *j == 0 { i } else { last[key.as_str()].1 };
        last.insert(key.as_str(), (*j, first));
    }
    if last.values().any(|(j, _)| *j != m - 1) {
        return Err("a register's last pass stops short of the unroll".into());
    }
    Ok(RollPlan { factor: m, stride: s, keep })
}

pub fn roll_k_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let (Ok(plain), Ok(index)) = (to_literals(root, kingdom), dual_ty(kingdom, "u32.index")) else { return Vec::new() };
    k_loops(&plain).into_iter().filter_map(|(n, d, lp)| roll_plan(d, &lp, index).ok().map(|p| Site { index: n, what: format!("K loop unrolled by {}, {} statements kept", p.factor, p.keep.len()) })).collect()
}

/// ROLL K: the loop at `node` steps one K row a pass, the statements of
/// rows m k + 1 .. m k + m - 1 gone, m times the passes.
pub fn roll_k(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = to_literals(root, kingdom)?;
    let (n, depth, lp) = k_loops(&plain).into_iter().find(|(n, _, _)| *n == site.index).ok_or("our software has a bug: the K loop is gone")?;
    let index = dual_ty(kingdom, "u32.index")?;
    let plan = roll_plan(depth, &lp, index)?;
    let counter = tile_counter_name(depth);
    let body = statements(&lp.kids[1]);
    let rolled: Vec<Tree> = plan.keep.iter().map(|&i| re_k(&body[i], counter, 1, 0, index)).collect();
    let count = plain_value(&lp.kids[0]).ok_or("the K loop's count is not a literal")? * plan.factor;
    rewrite_k_loop(&plain, n, rolled, count, kingdom)
}

fn roll_why(root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
    let plain = to_literals(root, kingdom).ok()?;
    let index = dual_ty(kingdom, "u32.index").ok()?;
    let loops = k_loops(&plain);
    if loops.is_empty() {
        return Some("no counted loop reads the tiles".into());
    }
    let why: Vec<String> = loops.iter().filter_map(|(_, d, lp)| roll_plan(*d, lp, index).err()).collect();
    (why.len() == loops.len()).then(|| why.join("; "))
}

// ---- RULE: SPLIT THE ACCUMULATOR ----

/// What splitting a K loop's accumulators needs: each register with its
/// fresh partner, and the loop's statement index and rung.
#[derive(Clone, Debug)]
pub struct SplitPlan {
    pub node: usize,
    pub depth: usize,
    pub statement: usize,
    pub rung: String,
    pub partner: BTreeMap<String, String>,
    pub stride: u32,
}

/// THE PRECONDITIONS of splitting the accumulators of the K loop at `node`:
/// the loop unrolls by two (every statement `r = r + E` at K row k, E
/// reading no register the body writes, the count even); each register is
/// a vec4 or f32 with a free partner of its kind; the loop is a statement
/// of a plain block (so a zero store can go before it and the sum after).
pub fn split_plan(plain: &Tree, node: usize, depth: usize, lp: &Tree, kingdom: &WgslKingdom) -> Result<SplitPlan, String> {
    let k = unroll_plan(node, depth, lp, 2)?;
    let body = statements(&lp.kids[1]);
    let targets: BTreeSet<String> = body.iter().filter_map(target_key).collect();
    let mut used = names_used(plain);
    let mut partner = BTreeMap::new();
    for st in &body {
        let others: BTreeSet<String> = targets.iter().filter(|t| Some((*t).clone()) != target_key(st)).cloned().collect();
        let key = accumulation(st, &others)?;
        let kind = register_kind(&key).filter(|k| matches!(*k, "v" | "f")).ok_or_else(|| format!("{key} is not a vec4 or f32 register"))?.to_string();
        if let std::collections::btree_map::Entry::Vacant(e) = partner.entry(key) {
            e.insert(fresh_register(&kind, &mut used, kingdom)?);
        }
    }
    let s = statement_at(plain, node).ok_or("the K loop is not a statement of a block")?;
    let r = rung(&lp.name).ok_or_else(|| format!("{} has no rung", lp.name))?.to_string();
    Ok(SplitPlan { node, depth, statement: s.index, rung: r, partner, stride: k.stride })
}

pub fn split_accumulator_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = to_literals(root, kingdom) else { return Vec::new() };
    k_loops(&plain).into_iter().filter_map(|(n, d, lp)| split_plan(&plain, n, d, &lp, kingdom).ok().map(|p| Site { index: n, what: format!("K loop, {} accumulators split", p.partner.len()) })).collect()
}

/// SPLIT THE ACCUMULATOR: the K loop at `node` steps two K rows a pass;
/// row 2 k adds into each register as before, row 2 k + 1 into its partner
/// (zeroed before the loop); after the loop each register adds its partner.
/// Each term is the same; the order of the additions changes (the register
/// holds the even rows' sum plus the odd rows' sum), which moves the result
/// by rounding only: the fitness's correctness criterion judges it.
pub fn split_accumulator(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = to_literals(root, kingdom)?;
    let (n, depth, lp) = k_loops(&plain).into_iter().find(|(n, _, _)| *n == site.index).ok_or("our software has a bug: the K loop is gone")?;
    let plan = split_plan(&plain, n, depth, &lp, kingdom)?;
    let counter = tile_counter_name(depth);
    let index = dual_ty(kingdom, "u32.index")?;
    let mut body = Vec::new();
    for st in statements(&lp.kids[1]) {
        let key = target_key(&st).ok_or("our software has a bug: an accumulation lost its register")?;
        let one: BTreeMap<String, String> = [(key.clone(), plan.partner[&key].clone())].into();
        body.push(re_k(&st, counter, 2, 0, index));
        body.push(rename_leaves(&re_k(&strip_k_constant(&st, plan.stride), counter, 2, 1, index), &one));
    }
    let count = plain_value(&lp.kids[0]).ok_or("the K loop's count is not a literal")? / 2;
    let inner = crate::wgsl::rules::rung(&lp.kids[1].name).ok_or("the K loop's body has no rung")?;
    let new_loop = Tree { name: lp.name.clone(), ty: lp.ty, konst: lp.konst, kids: vec![Tree::literal(&lp.kids[0].name, lp.kids[0].ty, count as f32), rebuild(body, inner, kingdom)?] };
    let mut with = Vec::new();
    for p in plan.partner.values() {
        let kind = register_kind(p).ok_or_else(|| format!("our software has a bug: {p} is not a register"))?;
        with.push(set_register(p, zero_of(kind, kingdom)?, &plan.rung, kingdom)?);
    }
    with.push(new_loop);
    for (r, p) in &plan.partner {
        let (a, b) = (get_register(r, kingdom)?, get_register(p, kingdom)?);
        let sum = Tree::new("arith.add", a.ty, vec![a, b]);
        with.push(set_register(r, sum, &plan.rung, kingdom)?);
    }
    replace_statement(&plain, plan.statement, with, kingdom)
}

fn split_why(root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
    let plain = to_literals(root, kingdom).ok()?;
    let loops = k_loops(&plain);
    if loops.is_empty() {
        return Some("no counted loop reads the tiles".into());
    }
    let why: Vec<String> = loops.iter().filter_map(|(n, d, lp)| split_plan(&plain, *n, *d, lp, kingdom).err()).collect();
    (why.len() == loops.len()).then(|| why.join("; "))
}

// ---- RULE: RESHAPE THE WORKGROUP ----

/// The workgroup shapes the rule moves between: 256 threads each.
pub const SHAPES: [(u32, u32); 3] = [(16, 16), (32, 8), (8, 32)];

/// A plain index literal that may encode the workgroup's extent (a whole
/// multiple of x or y, its mask x - 1 or y - 1, or its log2 as a shift):
/// the stride a reference would have followed, written as a number.
fn plain_workgroup_literal(t: &Tree, x: u32, y: u32) -> Option<String> {
    let lit = |n: &Tree| (n.name == "literal.index").then(|| plain_value(n)).flatten();
    if let Some(v) = lit(t) {
        if v > 0 && (v % x == 0 || v % y == 0) {
            return Some(format!("the index literal {v} (a multiple of the workgroup's {x} or {y})"));
        }
    }
    if t.kids.len() == 2 {
        if let Some(v) = lit(&t.kids[1]) {
            match t.name.as_str() {
                "index.and" if v + 1 == x || v + 1 == y => return Some(format!("the mask {v} (the workgroup's {x} or {y} less one)")),
                "index.shr" | "index.shl" if 1u32.checked_shl(v).is_some_and(|p| p == x || p == y) => return Some(format!("the shift {v} (log2 of the workgroup's {x} or {y})")),
                _ => {}
            }
        }
    }
    t.kids.iter().find_map(|k| plain_workgroup_literal(k, x, y))
}

/// THE PRECONDITIONS of reshaping the workgroup to `to`: the grid's
/// extents are literals (knob sites), the thread count stays (every loader
/// strides by the threads), every stride that depends on the extents is a
/// reference (no plain literal that may encode one), and the child passes
/// the joins (adapter, block, references exact, workgroup memory).
pub fn wg_shape_plan(root: &Tree, to: (u32, u32), kingdom: &WgslKingdom, target: &Target) -> Result<Tree, String> {
    let sites = knob::sites(root);
    let get = |s: knob::Site| sites.iter().find(|(x, _)| *x == s).map(|(_, v)| *v);
    let (x, y) = (get(knob::Site::Wx).ok_or("the workgroup's x is not a literal")?, get(knob::Site::Wy).ok_or("the workgroup's y is not a literal")?);
    if (x, y) == to {
        return Err("the workgroup already has that shape".into());
    }
    if x * y != to.0 * to.1 {
        return Err(format!("{x} x {y} and {} x {} differ in threads", to.0, to.1));
    }
    if let Some(why) = root.kids.get(1..).and_then(|ks| ks.iter().find_map(|k| plain_workgroup_literal(k, x, y))) {
        return Err(format!("{why} would not follow the workgroup"));
    }
    let child = knob::set(&knob::set(root, &knob::Site::Wx, to.0), &knob::Site::Wy, to.1);
    knob::joins(&child, kingdom, target)?;
    Ok(child)
}

pub fn wg_shape_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let target = Target::current();
    SHAPES.iter().enumerate().filter_map(|(i, &s)| wg_shape_plan(root, s, kingdom, &target).ok().map(|_| Site { index: i, what: format!("workgroup {} x {}", s.0, s.1) })).collect()
}

pub fn wg_shape(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let to = *SHAPES.get(site.index).ok_or("our software has a bug: no such shape")?;
    wg_shape_plan(root, to, kingdom, &Target::current())
}

fn wg_shape_why(root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
    let target = Target::current();
    let why: Vec<String> = SHAPES.iter().filter_map(|&s| wg_shape_plan(root, s, kingdom, &target).err()).collect();
    (why.len() == SHAPES.len()).then(|| why.join("; "))
}

// ---- RULE: HOIST A LOOP-INVARIANT LOAD ----

/// A loop statement, with its tile depth (counted loops nest by depth; the
/// item loop's counter is `safe.counter`).
fn loops_with_depth(root: &Tree) -> Vec<(usize, usize, Tree)> {
    fn walk(t: &Tree, counter: &mut usize, depth: usize, out: &mut Vec<(usize, usize, Tree)>) {
        let me = *counter;
        *counter += 1;
        let base = safe_base(&t.name);
        let is_loop = base.starts_with("loop.");
        if is_loop {
            out.push((me, depth, t.clone()));
        }
        let counted = is_loop && !base.starts_with("loop.items");
        for k in &t.kids {
            walk(k, counter, depth + usize::from(counted), out);
        }
    }
    let mut out = Vec::new();
    walk(root, &mut 0, 0, &mut out);
    out
}

/// What a loop changes: the registers and private arrays it writes, the
/// workgroup arrays it writes, whether it stores to h, and the counters
/// that step inside it (its own and its nested loops').
struct LoopWrites {
    storages: BTreeSet<String>,
    wg: BTreeSet<String>,
    barrier: bool,
    counters: BTreeSet<String>,
}

fn loop_writes(lp: &Tree, depth: usize) -> LoopWrites {
    let mut w = LoopWrites { storages: BTreeSet::new(), wg: BTreeSet::new(), barrier: false, counters: BTreeSet::new() };
    fn walk(t: &Tree, depth: usize, w: &mut LoopWrites) {
        let base = safe_base(&t.name);
        if base.starts_with("loop.items") {
            w.counters.insert("safe.counter".into());
        } else if base.starts_with("loop.") {
            w.counters.insert(tile_counter_name(depth).into());
        }
        if let Some(k) = target_key(t) {
            w.storages.insert(k.split('[').next().unwrap_or(&k).to_string());
        }
        if (t.name.starts_with("reg.set_v_lane") || base == "parr.set_v_lane") && t.kids.first().is_some_and(|l| l.kids.is_empty()) {
            w.storages.insert(t.kids[0].name.clone());
        }
        if base == "parr.set_safe" && t.kids.first().is_some_and(|l| l.kids.is_empty()) {
            w.storages.insert(t.kids[0].name.clone());
        }
        if (base.starts_with("wg.set") || base.starts_with("atomic.") || base.starts_with("wg.reduce")) && t.kids.first().is_some_and(|l| l.kids.is_empty()) {
            w.wg.insert(t.kids[0].name.clone());
        }
        if base.contains("barrier") {
            w.barrier = true;
        }
        let counted = base.starts_with("loop.") && !base.starts_with("loop.items");
        for k in &t.kids {
            walk(k, depth + usize::from(counted), w);
        }
    }
    walk(lp, depth, &mut w);
    w
}

/// A memory read of a value: storage (phi), a workgroup array, h.
fn is_load(t: &Tree) -> bool {
    let b = safe_base(&t.name);
    b.starts_with("load.") || b.starts_with("wg.get")
}

/// The value subtrees of `t` that read memory and do not change in the
/// loop: f32 or vec4 typed, a load inside, no counter that steps in the
/// loop, no storage or workgroup array the loop writes, no read of h, no
/// register or private array read; a workgroup read only in a loop with
/// no barrier (with one, the read may be ordered after other threads'
/// writes made before the loop that the barrier publishes).
fn invariant_loads(t: &Tree, w: &LoopWrites, ty_ok: &dyn Fn(&Tree) -> bool, out: &mut Vec<Tree>) {
    let invariant = |n: &Tree| {
        let varies = any_node(n, &|m| w.counters.contains(&m.name))
            || (w.barrier && any_node(n, &|m| safe_base(&m.name).starts_with("wg.get")))
            || any_node(n, &|m| m.kids.is_empty() && (w.storages.contains(&m.name) || w.wg.contains(&m.name)))
            || any_node(n, &|m| safe_base(&m.name).starts_with("load.h"))
            || has_effect(n)
            || any_node(n, &|m| m.name.starts_with("reg.get") || m.name.starts_with("parr.get"));
        !varies
    };
    if ty_ok(t) && any_node(t, &is_load) && invariant(t) {
        if !out.contains(t) {
            out.push(t.clone());
        }
        return;
    }
    t.kids.iter().for_each(|k| invariant_loads(k, w, ty_ok, out));
}

/// Every (loop node, invariant load) of the kernel, in tree order.
fn hoist_candidates(plain: &Tree, kingdom: &WgslKingdom) -> Vec<(usize, Tree)> {
    let (Ok(f), Ok(v)) = (dual_ty(kingdom, "f32.real"), dual_ty(kingdom, "vec4<f32>.real")) else { return Vec::new() };
    let ty_ok = |n: &Tree| (n.ty == f || n.ty == v) && !n.name.starts_with("literal.") && !n.kids.is_empty();
    let mut out = Vec::new();
    for (node, depth, lp) in loops_with_depth(plain) {
        if statement_at(plain, node).is_none() {
            continue;
        }
        let w = loop_writes(&lp, depth);
        let mut found = Vec::new();
        for k in &lp.kids[1..] {
            invariant_loads(k, &w, &ty_ok, &mut found);
        }
        out.extend(found.into_iter().map(|t| (node, t)));
    }
    out
}

/// `t` with every subtree equal to `from` replaced by `to`.
fn replace_subtree(t: &Tree, from: &Tree, to: &Tree) -> Tree {
    if t == from {
        return to.clone();
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| replace_subtree(k, from, to)).collect() }
}

pub fn hoist_load_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = to_literals(root, kingdom) else { return Vec::new() };
    hoist_candidates(&plain, kingdom).into_iter().enumerate().map(|(i, (node, t))| Site { index: i, what: format!("{} out of the loop at node {node}", t.name) }).collect()
}

/// HOIST A LOAD: the value read once into a fresh register before the
/// loop, every occurrence of it in the loop reading the register. Every
/// read of the kingdom is in range by its mask (a safe load never faults),
/// so reading it once before a loop that might not run is harmless.
pub fn hoist_load(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = to_literals(root, kingdom)?;
    let (node, value) = hoist_candidates(&plain, kingdom).into_iter().nth(site.index).ok_or("our software has a bug: the load is gone")?;
    let s = statement_at(&plain, node).ok_or("our software has a bug: the loop is not a statement")?;
    let r = rung(&s.tree.name).ok_or_else(|| format!("{} has no rung", s.tree.name))?.to_string();
    let v4 = dual_ty(kingdom, "vec4<f32>.real")?;
    let kind = if value.ty == v4 { "v" } else { "f" };
    let mut used = names_used(&plain);
    let fresh = fresh_register(kind, &mut used, kingdom)?;
    let read = get_register(&fresh, kingdom)?;
    let lp = &s.tree;
    let mut kids = vec![lp.kids[0].clone()];
    kids.extend(lp.kids[1..].iter().map(|k| replace_subtree(k, &value, &read)));
    let new_loop = Tree { name: lp.name.clone(), ty: lp.ty, konst: lp.konst, kids };
    replace_statement(&plain, s.index, vec![set_register(&fresh, value, &r, kingdom)?, new_loop], kingdom)
}

// ---- RULE: PAD A WORKGROUP ARRAY ----

/// The upper bound of an index expression, when this module can prove one:
/// literals, the thread's ids under the workgroup, the tile counters under
/// their loops' counts, sums, products, shifts right, masks, minimums.
fn index_hi(t: &Tree, env: &BTreeMap<String, u64>) -> Option<u64> {
    if let Some(v) = plain_value(t) {
        return Some(u64::from(v));
    }
    let k = |i: usize| t.kids.get(i).and_then(|c| index_hi(c, env));
    match t.name.as_str() {
        "index.add" => Some(k(0)? + k(1)?),
        "index.mul" => Some(k(0)? * k(1)?),
        "index.shr" => Some(k(0)? >> plain_value(&t.kids[1])?),
        "index.and" => Some(k(0).unwrap_or(u64::MAX).min(k(1)?)),
        "index.min" => Some(k(0).unwrap_or(u64::MAX).min(k(1).unwrap_or(u64::MAX))).filter(|v| *v < u64::MAX),
        _ => {
            if let Some(v) = env.get(&t.name) {
                return Some(*v);
            }
            match t.kids.as_slice() {
                [only] if t.name.starts_with("safe.tile_index") || t.name.starts_with("index.of_") || t.name.starts_with("axis.") => index_hi(only, env),
                _ => None,
            }
        }
    }
}

/// Every access of workgroup array `array` (reads and writes), with the
/// bounds of the counters of the loops around it.
fn accesses(root: &Tree, array: &str, base_env: &BTreeMap<String, u64>) -> Vec<(Tree, BTreeMap<String, u64>)> {
    fn walk(t: &Tree, array: &str, depth: usize, env: &BTreeMap<String, u64>, out: &mut Vec<(Tree, BTreeMap<String, u64>)>) {
        let base = safe_base(&t.name);
        if (base.starts_with("wg.get") || base.starts_with("wg.set")) && t.kids.first().is_some_and(|a| a.name == array) {
            out.push((t.clone(), env.clone()));
        }
        let counted = base.starts_with("loop.") && !base.starts_with("loop.items");
        let mut inner = env.clone();
        if counted {
            if let Some(c) = t.kids.first().and_then(plain_value) {
                inner.insert(tile_counter_name(depth).to_string(), u64::from(c.saturating_sub(1)));
            }
        }
        for (i, k) in t.kids.iter().enumerate() {
            let e = if counted && i > 0 { &inner } else { env };
            walk(k, array, depth + usize::from(counted && i > 0), e, out);
        }
    }
    let mut out = Vec::new();
    walk(root, array, 0, base_env, &mut out);
    out
}

/// What padding a workgroup array needs, found in it.
#[derive(Clone, Debug)]
pub struct PadPlan {
    pub array: String,
    /// The row width W (a power of two) and the declared length L.
    pub width: u32,
    pub len: u32,
}

/// The declared length of a workgroup array (after `to_literals`).
fn declared_len(plain: &Tree, array: &str) -> Option<u32> {
    fn walk(t: &Tree, array: &str) -> Option<u32> {
        if t.name.starts_with("wg.decl_") && t.kids.len() == 2 && t.kids[0].name == array {
            return plain_value(&t.kids[1]);
        }
        t.kids.iter().find_map(|k| walk(k, array))
    }
    plain.kids.first().and_then(|g| walk(g, array))
}

/// THE PRECONDITIONS of padding workgroup array `array`: every read in a
/// K loop steps K by one stride W (a power of two, the row width); the
/// declared length L is a whole number of rows; every access's index is
/// wrapped by the kingdom's mask (`safe.wg_of`/`safe.wgv_of`) around an
/// index this module bounds below L (so the mask never wrapped it, and
/// the padded index x + x / W stays below the padded L + L / W); the
/// padded arrays fit the adapter's workgroup memory.
pub fn pad_plan(plain: &Tree, array: &str) -> Result<PadPlan, String> {
    let len = declared_len(plain, array).ok_or_else(|| format!("{array} has no literal length"))?;
    let mut strides = BTreeSet::new();
    for (_, depth, lp) in k_loops(plain) {
        let counter = tile_counter_name(depth);
        for st in statements(&lp.kids[1]) {
            let mut reads = Vec::new();
            fn wg_reads<'a>(t: &'a Tree, array: &str, out: &mut Vec<&'a Tree>) {
                if safe_base(&t.name).starts_with("wg.get") && t.kids.first().is_some_and(|a| a.name == array) {
                    out.push(t);
                }
                t.kids.iter().for_each(|k| wg_reads(k, array, out));
            }
            wg_reads(&st, array, &mut reads);
            for r in reads {
                let (mut ks, mut bad) = (BTreeSet::new(), false);
                k_steps(r, counter, &mut ks, &mut bad);
                if bad {
                    return Err(format!("a read of {array} does not step K by a stride"));
                }
                strides.extend(ks.into_iter().map(|(_, s, _)| s));
            }
        }
    }
    let width = match strides.into_iter().collect::<Vec<_>>().as_slice() {
        [w] if w.is_power_of_two() && *w >= 2 => *w,
        [] => return Err(format!("no K loop reads {array}")),
        other => return Err(format!("{array} is read at K strides {other:?}, not one power of two")),
    };
    if len % width != 0 {
        return Err(format!("{array}'s length {len} is not whole rows of {width}"));
    }
    let knobs = crate::wgsl::params::Knobs::of_root(plain)?;
    let g = knobs.grid;
    let mut env = BTreeMap::new();
    env.insert("builtin.local_invocation_index".to_string(), u64::from(g.x * g.y - 1));
    env.insert("axis.x_lid".to_string(), u64::from(g.x - 1));
    env.insert("axis.y_lid".to_string(), u64::from(g.y - 1));
    for (acc, e) in accesses(plain, array, &env) {
        let idx = acc.kids.get(1).ok_or("an access with no index")?;
        let ib = safe_base(&idx.name);
        if !(matches!(ib, "safe.wg_of" | "safe.wgv_of") && idx.kids.len() == 1) {
            return Err(format!("an index of {array} is not the kingdom's mask ({})", idx.name));
        }
        let hi = index_hi(&idx.kids[0], &e).ok_or_else(|| format!("an index of {array} has no bound this rule can prove"))?;
        if hi >= u64::from(len) {
            return Err(format!("an index of {array} reaches {hi}, the mask wraps it below {len}"));
        }
    }
    Ok(PadPlan { array: array.to_string(), width, len })
}

/// The workgroup arrays a kernel declares.
fn wg_arrays(plain: &Tree) -> Vec<String> {
    fn walk(t: &Tree, out: &mut Vec<String>) {
        if t.name.starts_with("wg.decl_") && t.kids.len() == 2 {
            out.push(t.kids[0].name.clone());
        }
        t.kids.iter().for_each(|k| walk(k, out));
    }
    let mut out = Vec::new();
    if let Some(g) = plain.kids.first() {
        walk(g, &mut out);
    }
    out
}

/// `t` with every access of `array` at x read at x + (x >> log2 W), and
/// its declaration L + L / W long.
fn pad_walk(t: &Tree, plan: &PadPlan, index: crate::geneframe::Ty) -> Tree {
    let base = safe_base(&t.name);
    let kids: Vec<Tree> = t.kids.iter().map(|k| pad_walk(k, plan, index)).collect();
    if (base.starts_with("wg.get") || base.starts_with("wg.set")) && t.kids.first().is_some_and(|a| a.name == plan.array) {
        let mut kids = kids;
        let mask = &t.kids[1];
        let x = mask.kids[0].clone();
        let shift = Tree::literal("literal.index", index, plan.width.trailing_zeros() as f32);
        let row = Tree::new("index.shr", index, vec![x.clone(), shift]);
        let padded = Tree::new("index.add", index, vec![x, row]);
        kids[1] = Tree { name: mask.name.clone(), ty: mask.ty, konst: mask.konst, kids: vec![padded] };
        return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids };
    }
    if t.name.starts_with("wg.decl_") && t.kids.len() == 2 && t.kids[0].name == plan.array {
        let mut kids = kids;
        kids[1] = Tree { konst: (plan.len + plan.len / plan.width) as f32, ..t.kids[1].clone() };
        return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids };
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
}

pub fn pad_shared_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = to_literals(root, kingdom) else { return Vec::new() };
    wg_arrays(&plain).into_iter().enumerate().filter_map(|(i, a)| pad_plan(&plain, &a).ok().map(|p| Site { index: i, what: format!("{a}: rows of {}, {} -> {}", p.width, p.len, p.len + p.len / p.width) })).collect()
}

/// PAD SHARED: the array's rows a stride W + 1 apart (a bank further on
/// each row); every access moved in step, so each thread reads what it
/// wrote. The joins check the padded memory against the adapter.
pub fn pad_shared(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = to_literals(root, kingdom)?;
    let array = wg_arrays(&plain).into_iter().nth(site.index).ok_or("our software has a bug: the array is gone")?;
    let plan = pad_plan(&plain, &array)?;
    let child = pad_walk(&plain, &plan, dual_ty(kingdom, "u32.index")?);
    knob::joins(&child, kingdom, &Target::current())?;
    Ok(child)
}

fn pad_why(root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
    let plain = to_literals(root, kingdom).ok()?;
    let arrays = wg_arrays(&plain);
    if arrays.is_empty() {
        return Some("no workgroup array".into());
    }
    let why: Vec<String> = arrays.iter().filter_map(|a| pad_plan(&plain, a).err()).collect();
    (why.len() == arrays.len()).then(|| why.join("; "))
}

// ---- RULES: GROW THE COLUMN BLOCK, AND SHRINK IT BACK ----

/// `index.of_col` of a local invocation id: the thread's column in its workgroup.
fn is_col_lid(t: &Tree) -> bool {
    t.name == "index.of_col" && t.kids.first().is_some_and(|a| a.name.starts_with("axis.") && a.name.ends_with("_lid"))
}

/// The private vec4 array a key names (`parr.v[2]` -> `parr.v`).
fn array_of(key: &str) -> Option<&str> {
    key.split_once('[').map(|(a, _)| a)
}

/// The literal element of a private array key (`parr.v[2]` -> 2).
fn element_of(key: &str) -> Option<u32> {
    let (_, i) = key.split_once('[')?;
    i.trim_end_matches(']').parse::<f32>().ok().filter(|v| v.fract() == 0.0 && *v >= 0.0).map(|v| v as u32)
}

/// What growing or shrinking the column block needs, found in a kernel.
#[derive(Clone, Debug)]
pub struct ColPlan {
    /// The vec4 workgroup array read at the thread's column id (`wg.va`).
    pub tile: String,
    /// Every storage of the columns' accumulation flow, with its column
    /// group (0: the first four columns of the block, 1: the next four).
    pub family: BTreeMap<String, usize>,
    /// Group 0's storage to its group-1 twin (growing only).
    pub map: BTreeMap<String, String>,
    pub threads: u32,
    /// The row block (the private array holds a row's columns a vec4 at
    /// element r, its next four columns at r + by).
    pub by: u32,
}

/// The column tile: the one vec4 workgroup array read at the thread's
/// column id, and every use of that id inside its reads.
fn col_tile(root: &Tree) -> Result<String, String> {
    let mut tiles = BTreeSet::new();
    fn find(t: &Tree, out: &mut BTreeSet<String>) {
        if let Some(a) = crate::wgsl::rules::wg_vec4_read(t) {
            if any_node(&t.kids[1], &is_col_lid) {
                out.insert(a.to_string());
            }
        }
        t.kids.iter().for_each(|k| find(k, out));
    }
    find(root, &mut tiles);
    let tile = match tiles.into_iter().collect::<Vec<_>>().as_slice() {
        [t] => t.clone(),
        other => return Err(format!("{} vec4 arrays are read at the thread's column id, not one", other.len())),
    };
    fn outside(t: &Tree, tile: &str) -> bool {
        if crate::wgsl::rules::wg_vec4_read(t) == Some(tile) {
            return false;
        }
        is_col_lid(t) || t.kids.iter().any(|k| outside(k, tile))
    }
    if outside(root, &tile) {
        return Err("the thread's column id is used outside the column tile's reads".into());
    }
    Ok(tile)
}

/// The column group a statement's reads of the column tile hold: with
/// `spread` false every read is at `x + c` (group 0); with `spread` the
/// read is at `2 x + c` over even terms and c's parity is the group.
fn col_groups(t: &Tree, tile: &str, spread: bool, out: &mut BTreeSet<usize>, bad: &mut Option<String>) {
    if crate::wgsl::rules::wg_vec4_read(t) == Some(tile) {
        let Some(idx) = t.kids[1].kids.first() else {
            *bad = Some("a column tile read without an index".into());
            return;
        };
        let mut terms = Vec::new();
        crate::wgsl::rules::add_terms(idx, &mut terms);
        let (mut ids, mut constant, mut odd) = (0usize, 0u32, false);
        for term in terms {
            if let Some(v) = plain_value(term) {
                constant += v;
            } else if !spread && is_col_lid(term) {
                ids += 1;
            } else if term.name == "index.mul" && term.kids.len() == 2 {
                let (e, m) = match (plain_value(&term.kids[0]), plain_value(&term.kids[1])) {
                    (_, Some(m)) => (&term.kids[0], m),
                    (Some(m), None) => (&term.kids[1], m),
                    (None, None) => {
                        odd |= spread || any_node(term, &is_col_lid);
                        continue;
                    }
                };
                if spread && is_col_lid(e) && m == 2 {
                    ids += 1;
                } else if any_node(e, &is_col_lid) || (spread && m % 2 != 0) {
                    odd = true;
                }
            } else if any_node(term, &is_col_lid) || spread {
                odd = true;
            }
        }
        if ids != 1 || odd {
            *bad = Some(if spread { "a column tile read is not 2 x + c over even terms".into() } else { "a column tile read is not x + c".into() });
            return;
        }
        out.insert(if spread { (constant % 2) as usize } else { 0 });
        return;
    }
    for k in &t.kids {
        col_groups(k, tile, spread, out, bad);
        if bad.is_some() {
            return;
        }
    }
}

/// The column tile and the storages of the columns' flow with their groups.
fn col_families(root: &Tree, spread: bool) -> Result<(String, BTreeMap<String, usize>), String> {
    let tile = col_tile(root)?;
    let stmts = kernel_statements(root);
    let mut family: BTreeMap<String, usize> = BTreeMap::new();
    for s in stmts.iter().filter(|s| !crate::wgsl::rules::is_compound(&s.tree)) {
        let (mut groups, mut bad) = (BTreeSet::new(), None);
        col_groups(&s.tree, &tile, spread, &mut groups, &mut bad);
        if let Some(b) = bad {
            return Err(format!("{}: {b}", s.tree.name));
        }
        if groups.is_empty() {
            continue;
        }
        if groups.len() != 1 {
            return Err(format!("{} reads both column groups", s.tree.name));
        }
        let key = target_key(&s.tree).filter(|k| k.starts_with("reg.")).ok_or_else(|| format!("{} reads the column tile and assigns no register", s.tree.name))?;
        let g = *groups.iter().next().expect("one group");
        if family.insert(key.clone(), g).is_some_and(|x| x != g) {
            return Err(format!("{key} accumulates both column groups"));
        }
    }
    if family.is_empty() {
        return Err("no accumulation reads the column tile".into());
    }
    crate::wgsl::rules::close_flow(&stmts, &mut family)?;
    Ok((tile, family))
}

/// The epilogue's reads of the flow: every read of a family storage in the
/// epilogue is a lane of a private vec4 array of the flow (`pv[k][k2]`).
fn epilogue_reads_lanes(epilogue: &Tree, family: &BTreeMap<String, usize>) -> Result<BTreeSet<String>, String> {
    let arrays: BTreeSet<String> = family.keys().filter_map(|k| array_of(k).map(str::to_string)).collect();
    let registers: BTreeSet<&String> = family.keys().filter(|k| !k.contains('[')).collect();
    if any_node(epilogue, &|n| n.name.starts_with("reg.get") && n.kids.first().is_some_and(|l| registers.contains(&l.name))) {
        return Err("the epilogue reads a register of the columns' flow".into());
    }
    let mut lanes = 0usize;
    let mut other = false;
    fn walk(t: &Tree, arrays: &BTreeSet<String>, lanes: &mut usize, other: &mut bool) {
        if t.name.starts_with("parr.get") && t.kids.first().is_some_and(|a| arrays.contains(&a.name)) {
            if t.name == "parr.get_v_lane" && t.kids.len() == 3 {
                *lanes += 1;
            } else {
                *other = true;
            }
            return;
        }
        t.kids.iter().for_each(|k| walk(k, arrays, lanes, other));
    }
    walk(epilogue, &arrays, &mut lanes, &mut other);
    if other || lanes == 0 {
        return Err("the epilogue reads the columns other than as lanes of a private vec4 array".into());
    }
    Ok(arrays)
}

/// THE PRECONDITIONS of growing the column block 4 -> 8 at a constant
/// workgroup tile (the workgroup's column extent halved): one vec4
/// workgroup array is read at the thread's column id `x + c`, every use of
/// that id in such a read; every storage the columns' sums flow through is
/// a register (twinned by a fresh one) or element r < by of a private vec4
/// array (twinned by element r + by, free); the epilogue reads the flow
/// only as lanes of those arrays; every use of the flat thread index is in
/// a loop striding by the threads.
pub fn grow_cols_plan(root: &Tree, kingdom: &WgslKingdom) -> Result<ColPlan, String> {
    use crate::wgsl::params::{Knob, Knobs};
    let knobs = Knobs::of_root(root)?;
    if knobs.grid.bx != 4 {
        return Err(format!("the column block is {}, not 4", knobs.grid.bx));
    }
    let wcol = knobs.value(Knob::Wcol);
    if wcol < 2 || wcol % 2 != 0 {
        return Err(format!("the workgroup's column extent {wcol} does not halve"));
    }
    let by = knobs.grid.by;
    let (tile, family) = col_families(root, false)?;
    let epilogue = root.kids.get(2).ok_or("the kernel has no epilogue")?;
    let arrays = epilogue_reads_lanes(epilogue, &family)?;
    let mut used = names_used(root);
    let mut literal_parr: BTreeSet<String> = BTreeSet::new();
    for s in kernel_statements(root) {
        if let Some(k) = target_key(&s.tree).filter(|k| k.contains('[')) {
            literal_parr.insert(k);
        }
        let mut reads = BTreeSet::new();
        read_keys(&s.tree, &mut reads);
        literal_parr.extend(reads.into_iter().filter(|k| k.contains('[') && !k.ends_with("[*]")));
    }
    let mut map = BTreeMap::new();
    for key in family.keys() {
        if let Some(arr) = array_of(key) {
            let i = element_of(key).ok_or_else(|| format!("{key}: no literal element"))?;
            if i >= by {
                return Err(format!("{key} is past the row block {by}"));
            }
            let new = format!("{arr}[{}]", i + by);
            if i + by >= 16 || literal_parr.contains(&new) {
                return Err(format!("{new} is in use or past the array"));
            }
            map.insert(key.clone(), new);
        } else {
            let kind = register_kind(key).ok_or_else(|| format!("{key} is not a register"))?;
            map.insert(key.clone(), fresh_register(kind, &mut used, kingdom)?);
        }
    }
    for a in &arrays {
        if (0..by).any(|r| !family.contains_key(&format!("{a}[{r}]"))) {
            return Err(format!("{a} does not hold every row of the block in the columns' flow"));
        }
    }
    let threads = knobs.value(Knob::Threads);
    let mut outside = false;
    crate::wgsl::rules::lidx_outside_thread_loops(root, threads, 0, false, &mut outside);
    if outside {
        return Err("the flat thread index is used outside a loop striding by the threads, or such a loop's counter strides something else".into());
    }
    Ok(ColPlan { tile, family, map, threads, by })
}

/// `t` with the column tile's reads at column id `x` read at `2 x + extra`.
fn spread_cols(t: &Tree, tile: &str, extra: Option<u32>, in_tile: bool) -> Tree {
    if in_tile && is_col_lid(t) {
        let doubled = Tree::new("index.mul", t.ty, vec![t.clone(), Tree::literal("literal.index", t.ty, 2.0)]);
        return match extra {
            Some(e) => Tree::new("index.add", t.ty, vec![doubled, Tree::literal("literal.index", t.ty, e as f32)]),
            None => doubled,
        };
    }
    let inside = in_tile || crate::wgsl::rules::wg_vec4_read(t) == Some(tile);
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| spread_cols(k, tile, extra, inside)).collect() }
}

/// `t` with the column tile's reads at `2 x + c` read at `x + c - c % 2`.
fn unspread_cols(t: &Tree, tile: &str, in_tile: bool) -> Tree {
    if in_tile && t.name == "index.mul" && t.kids.len() == 2 {
        for i in 0..2 {
            if is_col_lid(&t.kids[i]) && plain_value(&t.kids[1 - i]) == Some(2) {
                return t.kids[i].clone();
            }
        }
    }
    let inside = in_tile || crate::wgsl::rules::wg_vec4_read(t) == Some(tile);
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| unspread_cols(k, tile, inside)).collect() }
}

/// The epilogue's lane read of a flow array: `a[r][c]` read as
/// `a[r + by * (c >> 2)][c & 3]` (growing), or back (shrinking).
fn reroute_lanes(t: &Tree, arrays: &BTreeSet<String>, by: u32, grow: bool) -> Result<Tree, String> {
    let kids: Vec<Tree> = t.kids.iter().map(|k| reroute_lanes(k, arrays, by, grow)).collect::<Result<_, _>>()?;
    if !(t.name == "parr.get_v_lane" && t.kids.len() == 3 && arrays.contains(&t.kids[0].name)) {
        return Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids });
    }
    let (arr, wrap, lane) = (&kids[0], &kids[1], &kids[2]);
    let row = wrap.kids.first().ok_or("a private array read with no index")?;
    let ix = row.ty;
    let lit = |v: u32| Tree::literal("literal.index", ix, v as f32);
    let (new_row, new_lane) = if grow {
        let group = Tree::new("index.mul", ix, vec![lit(by), Tree::new("index.shr", lane.ty, vec![lane.clone(), lit(2)])]);
        (Tree::new("index.add", ix, vec![row.clone(), group]), Tree::new("index.and", lane.ty, vec![lane.clone(), lit(3)]))
    } else {
        // Exactly the form growing wrote, or the kernel is not one this rule shrinks.
        let ok = row.name == "index.add" && row.kids.len() == 2 && row.kids[1].name == "index.mul" && plain_value(&row.kids[1].kids[0]) == Some(by) && row.kids[1].kids[1].name == "index.shr" && plain_value(&row.kids[1].kids[1].kids[1]) == Some(2) && lane.name == "index.and" && plain_value(&lane.kids[1]) == Some(3) && row.kids[1].kids[1].kids[0] == lane.kids[0];
        if !ok {
            return Err("the epilogue does not read a row's second column group at r + by".into());
        }
        (row.kids[0].clone(), lane.kids[0].clone())
    };
    let wrap = Tree { name: wrap.name.clone(), ty: wrap.ty, konst: wrap.konst, kids: vec![new_row] };
    Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: vec![arr.clone(), wrap, new_lane] })
}

/// Every statement mapped for growing: an assignment of the flow becomes
/// itself (its tile reads at `2 x`) followed by its twin (renamed, at
/// `2 x + 1`); counted loops striding by the threads run twice the passes.
fn grow_cols_walk(t: &Tree, plan: &ColPlan, depth: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
    if crate::wgsl::rules::is_plain_block(&t.name) {
        let mut out = Vec::new();
        for s in statements(t) {
            match target_key(&s).filter(|k| plan.family.contains_key(k)) {
                Some(_) => {
                    out.push(spread_cols(&s, &plan.tile, None, false));
                    out.push(crate::wgsl::rules::rename(&spread_cols(&s, &plan.tile, Some(1), false), &plan.map));
                }
                None => out.push(grow_cols_walk(&s, plan, depth, kingdom)?),
            }
        }
        return rebuild(out, rung(&t.name).ok_or_else(|| format!("{} has no rung", t.name))?, kingdom);
    }
    let base = safe_base(&t.name);
    let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
    let mut kids = Vec::with_capacity(t.kids.len());
    for (i, k) in t.kids.iter().enumerate() {
        kids.push(grow_cols_walk(k, plan, depth + usize::from(looped), kingdom)?);
        if i == 0 && crate::wgsl::rules::is_counted_loop(&t.name) && t.kids.len() >= 2 && crate::wgsl::rules::strides_by_threads(t.kids.last().expect("a body"), depth, plan.threads) {
            kids[0].konst *= 2.0;
        }
    }
    Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids })
}

/// GROW THE COLUMN BLOCK: 4 -> 8 columns a thread, the workgroup's column
/// extent halved (the same workgroup tile, half the threads); a thread's
/// columns 4..7 read the tile's vec4 at `2 x + 1` into twin storages, the
/// epilogue reads them at element r + by of the private array.
pub fn grow_cols(root: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    use crate::wgsl::knob::{set, Site as KnobSite};
    use crate::wgsl::params::{Knob, Knobs};
    let plan = grow_cols_plan(root, kingdom)?;
    let knobs = Knobs::of_root(root)?;
    let arrays: BTreeSet<String> = plan.family.keys().filter_map(|k| array_of(k).map(str::to_string)).collect();
    let mut kids = root.kids.clone();
    kids[1] = grow_cols_walk(&root.kids[1], &plan, 0, kingdom)?;
    kids[2] = reroute_lanes(&grow_cols_walk(&root.kids[2], &plan, 0, kingdom)?, &arrays, plan.by, true)?;
    let t = Tree { name: root.name.clone(), ty: root.ty, konst: root.konst, kids };
    let col_site = if knobs.grid.rows_x { KnobSite::Wy } else { KnobSite::Wx };
    let t = set(&t, &col_site, knobs.value(Knob::Wcol) / 2);
    Ok(set(&t, &KnobSite::Bx, 8))
}

pub fn grow_cols_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    match grow_cols_plan(root, kingdom) {
        Ok(p) => vec![Site { index: 0, what: format!("columns 4 -> 8 over {}, {} storages twinned", p.tile, p.map.len()) }],
        Err(_) => Vec::new(),
    }
}

/// THE PRECONDITIONS of shrinking the column block 8 -> 4 (the column
/// extent doubled): the column tile is read at `2 x + c`, group c % 2;
/// each group's flow its own; the epilogue reads a row's second group at
/// element r + by as growing writes it; the threads double within the
/// adapter; every loop striding the flat thread index by the threads has
/// an even number of passes.
pub fn shrink_cols_plan(root: &Tree, kingdom: &WgslKingdom) -> Result<ColPlan, String> {
    use crate::wgsl::params::{Knob, Knobs};
    let plain = to_literals(root, kingdom)?;
    let knobs = Knobs::of_root(&plain)?;
    if knobs.grid.bx != 8 {
        return Err(format!("the column block is {}, not 8", knobs.grid.bx));
    }
    let target = Target::current();
    let threads = knobs.value(Knob::Threads);
    let wcol = knobs.value(Knob::Wcol);
    if threads * 2 > target.max_invocations || wcol * 2 > target.max_size[if knobs.grid.rows_x { 1 } else { 0 }] {
        return Err(format!("{threads} threads do not double within the adapter"));
    }
    let (tile, family) = col_families(&plain, true)?;
    if !family.values().any(|&g| g == 1) {
        return Err("no storage of the second column group".into());
    }
    let epilogue = plain.kids.get(2).ok_or("the kernel has no epilogue")?;
    let arrays = epilogue_reads_lanes(epilogue, &family)?;
    reroute_lanes(epilogue, &arrays, knobs.grid.by, false)?;
    let mut outside = false;
    crate::wgsl::rules::lidx_outside_thread_loops(&plain, threads, 0, false, &mut outside);
    if outside {
        return Err("the flat thread index is used outside a loop striding by the threads, or such a loop's counter strides something else".into());
    }
    fn odd_thread_loop(t: &Tree, threads: u32, depth: usize) -> bool {
        let base = safe_base(&t.name);
        let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
        if crate::wgsl::rules::is_counted_loop(&t.name) && t.kids.len() >= 2 && crate::wgsl::rules::strides_by_threads(t.kids.last().expect("a body"), depth, threads) && plain_value(&t.kids[0]).is_none_or(|v| v % 2 != 0) {
            return true;
        }
        t.kids.iter().any(|k| odd_thread_loop(k, threads, depth + usize::from(looped)))
    }
    if odd_thread_loop(&plain, threads, 0) {
        return Err("a loop striding by the threads has an odd or unknown number of passes".into());
    }
    Ok(ColPlan { tile, family, map: BTreeMap::new(), threads, by: knobs.grid.by })
}

/// SHRINK THE COLUMN BLOCK: 8 -> 4, the column extent doubled; the second
/// group's statements and storages dropped, the first group's tile reads at
/// `x` where they were at `2 x`, the epilogue's lanes read back at row r,
/// the loops striding by the threads half the passes.
pub fn shrink_cols(root: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    use crate::wgsl::knob::{set, Site as KnobSite};
    use crate::wgsl::params::{Knob, Knobs};
    let plan = shrink_cols_plan(root, kingdom)?;
    let knobs = Knobs::of_root(root)?;
    fn walk(t: &Tree, plan: &ColPlan, depth: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
        if crate::wgsl::rules::is_plain_block(&t.name) {
            let mut out = Vec::new();
            for s in statements(t) {
                match target_key(&s).and_then(|k| plan.family.get(&k).copied()) {
                    Some(1) => {}
                    Some(_) => out.push(unspread_cols(&s, &plan.tile, false)),
                    None => out.push(walk(&s, plan, depth, kingdom)?),
                }
            }
            return rebuild(out, rung(&t.name).ok_or_else(|| format!("{} has no rung", t.name))?, kingdom);
        }
        let base = safe_base(&t.name);
        let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
        let mut kids = Vec::with_capacity(t.kids.len());
        for (i, k) in t.kids.iter().enumerate() {
            kids.push(walk(k, plan, depth + usize::from(looped), kingdom)?);
            if i == 0 && crate::wgsl::rules::is_counted_loop(&t.name) && t.kids.len() >= 2 && crate::wgsl::rules::strides_by_threads(t.kids.last().expect("a body"), depth, plan.threads) {
                kids[0].konst /= 2.0;
            }
        }
        Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids })
    }
    let arrays: BTreeSet<String> = plan.family.keys().filter_map(|k| array_of(k).map(str::to_string)).collect();
    let mut kids = root.kids.clone();
    kids[1] = walk(&root.kids[1], &plan, 0, kingdom)?;
    kids[2] = reroute_lanes(&walk(&root.kids[2], &plan, 0, kingdom)?, &arrays, plan.by, false)?;
    let t = Tree { name: root.name.clone(), ty: root.ty, konst: root.konst, kids };
    let col_site = if knobs.grid.rows_x { KnobSite::Wy } else { KnobSite::Wx };
    let t = set(&t, &col_site, knobs.value(Knob::Wcol) * 2);
    Ok(set(&t, &KnobSite::Bx, 4))
}

pub fn shrink_cols_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    match shrink_cols_plan(root, kingdom) {
        Ok(p) => vec![Site { index: 0, what: format!("columns 8 -> 4 over {}, {} storages dropped", p.tile, p.family.values().filter(|&&g| g == 1).count()) }],
        Err(_) => Vec::new(),
    }
}

// ---- RULE: DOUBLE-BUFFER THE TILES ----

/// `t` at the type `want` (a statement or block dual of another rung): its
/// row renamed to that rung, and each child given the type the new row's
/// signature demands of it, down to where the types already agree (a tile
/// loop's body keeps its rung: `tile_L0` and `tile_L1` both take an `L2`
/// block). The statements of a loop body written one loop further out.
fn rerung_to(t: &Tree, want: crate::geneframe::Ty, kingdom: &WgslKingdom) -> Result<Tree, String> {
    if t.ty == want {
        return Ok(t.clone());
    }
    let Some(r) = rung(&t.name) else { return Err(format!("{} has no rung to move to another type", t.name)) };
    let dual = kingdom.dual(want).map(|x| x.dual.clone()).ok_or("our software has a bug: a type with no dual")?;
    let to = dual.rsplit_once('.').map(|(_, x)| x.to_string()).filter(|x| rung(&format!("x_{x}")).is_some()).ok_or_else(|| format!("{dual} is not a rung's type"))?;
    let name = format!("{}{to}", &t.name[..t.name.len() - r.len()]);
    let row = kingdom.rows.iter().find(|x| x.name == name && x.output == want && x.inputs.len() == t.kids.len()).ok_or_else(|| format!("the kingdom has no row {name} -> {dual} with {} children", t.kids.len()))?;
    let kids = t.kids.iter().zip(&row.inputs).map(|(k, &d)| if k.ty == d { Ok(k.clone()) } else { rerung_to(k, d, kingdom) }).collect::<Result<Vec<_>, _>>()?;
    Ok(Tree { name, ty: want, konst: t.konst, kids })
}

/// A statement of a loop body as a statement of the block around the loop.
fn rung_down(t: &Tree, outer: &str, kingdom: &WgslKingdom) -> Result<Tree, String> {
    rerung_to(t, dual_ty(kingdom, &format!("stmt.{outer}"))?, kingdom)
}

/// The item counter's value `(safe.counter_index safe.counter)`.
fn is_item_counter(t: &Tree) -> bool {
    t.name == "safe.counter_index" && t.kids.len() == 1 && t.kids[0].name == "safe.counter"
}

/// `t` with the item counter's value replaced by `with`.
fn with_counter(t: &Tree, with: &Tree) -> Tree {
    if is_item_counter(t) {
        return with.clone();
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| with_counter(k, with)).collect() }
}

/// `t` with every access of the arrays in `tiles` (reads when `reads`,
/// writes otherwise) at index x moved to x + offset.
fn offset_accesses(t: &Tree, tiles: &BTreeSet<String>, reads: bool, offset: &Tree) -> Tree {
    let kids: Vec<Tree> = t.kids.iter().map(|k| offset_accesses(k, tiles, reads, offset)).collect();
    let base = safe_base(&t.name);
    let hit = if reads { base.starts_with("wg.get") } else { base.starts_with("wg.set") };
    if hit && t.kids.first().is_some_and(|a| tiles.contains(&a.name)) && t.kids.len() >= 2 {
        let mut kids = kids;
        let mask = &t.kids[1];
        if let Some(x) = mask.kids.first() {
            let moved = Tree::new("index.add", x.ty, vec![x.clone(), offset.clone()]);
            kids[1] = Tree { name: mask.name.clone(), ty: mask.ty, konst: mask.konst, kids: vec![moved] };
        }
        return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids };
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
}

/// What double-buffering an item loop needs, found in it.
#[derive(Clone, Debug)]
pub struct BufferPlan {
    pub statement: usize,
    /// The body's statements: the loader (before the first barrier) and the
    /// compute (between it and the last).
    pub load: Vec<Tree>,
    pub compute: Vec<Tree>,
    pub barrier: Tree,
    pub tiles: BTreeSet<String>,
    /// The item loop's stride T, and each tile's half H (a power of two at
    /// least every index the loop reaches) with its declared length.
    pub stride: u32,
    pub half: BTreeMap<String, (u32, u32)>,
    pub body_rung: String,
    pub outer_rung: String,
}

/// How often each register is written (whole or a lane) and read in trees.
fn register_use(trees: &[Tree]) -> (BTreeMap<String, usize>, BTreeMap<String, usize>) {
    let (mut written, mut read) = (BTreeMap::new(), BTreeMap::new());
    for s in trees {
        fn walk(t: &Tree, written: &mut BTreeMap<String, usize>, read: &mut BTreeMap<String, usize>) {
            let write = t.name.starts_with("reg.set_");
            for (i, k) in t.kids.iter().enumerate() {
                if is_register(k) {
                    let m = if write && i == 0 { &mut *written } else { &mut *read };
                    *m.entry(k.name.clone()).or_default() += 1;
                }
                walk(k, written, read);
            }
        }
        walk(s, &mut written, &mut read);
    }
    (written, read)
}

/// THE PRECONDITIONS of double-buffering the item loop at statement `s`:
/// its body is a loader, a barrier, a computation and a closing barrier;
/// the loader writes the tiles (workgroup arrays) and reads none of them,
/// holds no barrier, uses the item counter only as its value, and its
/// registers are its own (written and read only in it); the computation
/// writes no workgroup array; the loop steps the items from 0 by a power
/// of two T; every tile index the loop reaches has a bound this rule can
/// prove, and two halves of the next power of two fit the array (its
/// declared length grown when they do not, within the adapter's memory).
pub fn buffer_plan(plain: &Tree, s: &Stmt) -> Result<BufferPlan, String> {
    let lp = &s.tree;
    if !safe_base(&lp.name).starts_with("loop.items_strided") || lp.kids.len() != 3 {
        return Err(format!("{} is not an item loop striding by its tile", lp.name));
    }
    let stride = plain_value(&lp.kids[1]).filter(|t| t.is_power_of_two()).ok_or("the item loop's stride is not a literal power of two")?;
    let body = statements(&lp.kids[2]);
    let is_barrier = |t: &Tree| rung(&t.name).is_some() && t.name.starts_with("stmt.barrier_");
    let barriers: Vec<usize> = body.iter().enumerate().filter(|(_, t)| any_node(t, &|n| safe_base(&n.name).contains("barrier"))).map(|(i, _)| i).collect();
    let (b1, b2) = match barriers.as_slice() {
        [a, b] if is_barrier(&body[*a]) && is_barrier(&body[*b]) && *b == body.len() - 1 && *a > 0 && *b > *a + 1 => (*a, *b),
        _ => return Err("the item loop's body is not a loader, a barrier, a computation and a closing barrier".into()),
    };
    let (load, compute) = (body[..b1].to_vec(), body[b1 + 1..b2].to_vec());
    let mut tiles = BTreeSet::new();
    for st in &load {
        fn writes(t: &Tree, out: &mut BTreeSet<String>) {
            if safe_base(&t.name).starts_with("wg.set") && t.kids.first().is_some_and(|a| a.kids.is_empty()) {
                out.insert(t.kids[0].name.clone());
            }
            t.kids.iter().for_each(|k| writes(k, out));
        }
        writes(st, &mut tiles);
    }
    if tiles.is_empty() {
        return Err("the loader writes no workgroup array".into());
    }
    if load.iter().any(|t| any_node(t, &|n| safe_base(&n.name).starts_with("wg.get") && n.kids.first().is_some_and(|a| tiles.contains(&a.name)))) {
        return Err("the loader reads a tile".into());
    }
    if compute.iter().any(|t| any_node(t, &|n| safe_base(&n.name).starts_with("wg.set") || safe_base(&n.name).starts_with("atomic.") || safe_base(&n.name).starts_with("wg.reduce"))) {
        return Err("the computation writes workgroup memory".into());
    }
    fn counter_uses(t: &Tree, bad: &mut bool) {
        if is_item_counter(t) {
            return;
        }
        if t.name == "safe.counter" {
            *bad = true;
        }
        t.kids.iter().for_each(|k| counter_uses(k, bad));
    }
    let mut bad = false;
    load.iter().for_each(|t| counter_uses(t, &mut bad));
    if bad {
        return Err("the loader uses the item counter other than as its value".into());
    }
    // The loader's registers are its own.
    let (load_w, load_r) = register_use(&load);
    let (all_w, all_r) = register_use(std::slice::from_ref(plain));
    let elsewhere = |all: &BTreeMap<String, usize>, mine: &BTreeMap<String, usize>, r: &str| all.get(r).copied().unwrap_or(0) > mine.get(r).copied().unwrap_or(0);
    if let Some(r) = load_w.keys().find(|r| elsewhere(&all_r, &load_r, r)) {
        return Err(format!("{r} is written by the loader and read elsewhere"));
    }
    if let Some(r) = load_r.keys().find(|r| !load_w.contains_key(*r) || elsewhere(&all_w, &load_w, r)) {
        return Err(format!("the loader reads {r}, written elsewhere"));
    }
    // Each tile's half: a power of two at least every index the loop reaches.
    let knobs = crate::wgsl::params::Knobs::of_root(plain)?;
    let g = knobs.grid;
    let mut env = BTreeMap::new();
    env.insert("builtin.local_invocation_index".to_string(), u64::from(g.x * g.y - 1));
    env.insert("axis.x_lid".to_string(), u64::from(g.x - 1));
    env.insert("axis.y_lid".to_string(), u64::from(g.y - 1));
    let mut half = BTreeMap::new();
    for tile in &tiles {
        let mut hi = 0u64;
        for (acc, e) in accesses(lp, tile, &env) {
            let idx = acc.kids.get(1).ok_or("an access with no index")?;
            if !(matches!(safe_base(&idx.name), "safe.wg_of" | "safe.wgv_of") && idx.kids.len() == 1) {
                return Err(format!("an index of {tile} is not the kingdom's mask"));
            }
            hi = hi.max(index_hi(&idx.kids[0], &e).ok_or_else(|| format!("an index of {tile} has no bound this rule can prove"))?);
        }
        if accesses(plain, tile, &env).len() != accesses(lp, tile, &env).len() {
            return Err(format!("{tile} is used outside the item loop"));
        }
        let h = u32::try_from(hi + 1).map_err(|_| format!("{tile}'s index is too large"))?.next_power_of_two();
        let len = declared_len(plain, tile).ok_or_else(|| format!("{tile} has no literal length"))?;
        half.insert(tile.clone(), (h, len.max(2 * h)));
    }
    // The two halves within the adapter's workgroup memory.
    let elem = |a: &str| if a.starts_with("wg.v2") { 8u64 } else if a.starts_with("wg.v") { 16 } else { 4 };
    let bytes: u64 = wg_arrays(plain).iter().map(|a| elem(a) * u64::from(half.get(a).map_or_else(|| declared_len(plain, a).unwrap_or(0), |(_, l)| *l))).sum();
    let limit = u64::from(Target::current().max_workgroup_storage);
    if bytes > limit {
        return Err(format!("two halves of the tiles take {bytes} bytes of workgroup memory, the adapter has {limit}"));
    }
    let body_rung = rung(&lp.kids[2].name).ok_or("the item loop's body has no rung")?.to_string();
    let d = body_rung.as_bytes()[1] - b'0';
    if d == 0 {
        return Err("the item loop's body is at rung 0".into());
    }
    let outer_rung = format!("{}{}", &body_rung[..1], d - 1);
    Ok(BufferPlan { statement: s.index, load, compute, barrier: body[b2].clone(), tiles, stride, half, body_rung, outer_rung })
}

fn buffer_candidates(plain: &Tree) -> Vec<Result<BufferPlan, String>> {
    kernel_statements(plain).into_iter().filter(|s| safe_base(&s.tree.name).starts_with("loop.items")).map(|s| buffer_plan(plain, &s)).collect()
}

pub fn double_buffer_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = to_literals(root, kingdom) else { return Vec::new() };
    buffer_candidates(&plain).into_iter().enumerate().filter_map(|(i, p)| p.ok().map(|p| Site { index: i, what: format!("item loop by {}, tiles {:?}", p.stride, p.half) })).collect()
}

/// DOUBLE-BUFFER: the tiles of items t..t + T are loaded before the loop
/// (half 0); each pass loads the next tiles into the other half (items t +
/// T, half ((t / T) + 1) % 2) and computes from this pass's half (t / T) %
/// 2, with one barrier a pass instead of two. The pass that computes from
/// a half and the pass that next writes it are separated by the closing
/// barrier between them; the loads of items past the end are guarded off
/// by the loader's own bounds, as they were.
pub fn double_buffer(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = to_literals(root, kingdom)?;
    let plan = buffer_candidates(&plain).into_iter().nth(site.index).ok_or("our software has a bug: the item loop is gone")??;
    let index = dual_ty(kingdom, "u32.index")?;
    let lit = |v: u32| Tree::literal("literal.index", index, v as f32);
    fn find_counter(t: &Tree) -> Option<Tree> {
        if is_item_counter(t) {
            return Some(t.clone());
        }
        t.kids.iter().find_map(find_counter)
    }
    let t = plan.load.iter().find_map(find_counter).ok_or("the loader does not read the item counter")?;
    let pass = Tree::new("index.shr", index, vec![t.clone(), lit(plan.stride.trailing_zeros())]);
    let cur = Tree::new("index.and", index, vec![pass.clone(), lit(1)]);
    let next = Tree::new("index.and", index, vec![Tree::new("index.add", index, vec![pass, lit(1)]), lit(1)]);
    let mut prologue = Vec::new();
    for st in &plan.load {
        prologue.push(rung_down(&with_counter(st, &lit(0)), &plan.outer_rung, kingdom)?);
    }
    prologue.push(rung_down(&plan.barrier, &plan.outer_rung, kingdom)?);
    let ahead = Tree::new("index.add", index, vec![t.clone(), lit(plan.stride)]);
    let mut body = Vec::new();
    for st in &plan.load {
        let mut x = with_counter(st, &ahead);
        for (tile, (h, _)) in &plan.half {
            let one: BTreeSet<String> = [tile.clone()].into();
            x = offset_accesses(&x, &one, false, &Tree::new("index.mul", index, vec![lit(*h), next.clone()]));
        }
        body.push(x);
    }
    for st in &plan.compute {
        let mut x = st.clone();
        for (tile, (h, _)) in &plan.half {
            let one: BTreeSet<String> = [tile.clone()].into();
            x = offset_accesses(&x, &one, true, &Tree::new("index.mul", index, vec![lit(*h), cur.clone()]));
        }
        body.push(x);
    }
    body.push(plan.barrier.clone());
    let s = kernel_statements(&plain).into_iter().find(|x| x.index == plan.statement).ok_or("our software has a bug: the item loop is gone")?;
    let lp = &s.tree;
    let new_loop = Tree { name: lp.name.clone(), ty: lp.ty, konst: lp.konst, kids: vec![lp.kids[0].clone(), lp.kids[1].clone(), rebuild(body, &plan.body_rung, kingdom)?] };
    prologue.push(new_loop);
    let mut child = replace_statement(&plain, plan.statement, prologue, kingdom)?;
    for (tile, (_, len)) in &plan.half {
        if declared_len(&child, tile).is_some_and(|l| l < *len) {
            child = pad_len(&child, tile, *len);
        }
    }
    knob::joins(&child, kingdom, &Target::current())?;
    Ok(child)
}

/// The kernel with workgroup array `array` declared `len` long.
fn pad_len(t: &Tree, array: &str, len: u32) -> Tree {
    let kids: Vec<Tree> = t.kids.iter().map(|k| pad_len(k, array, len)).collect();
    if t.name.starts_with("wg.decl_") && t.kids.len() == 2 && t.kids[0].name == array {
        let mut kids = kids;
        kids[1] = Tree { konst: len as f32, ..t.kids[1].clone() };
        return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids };
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
}

fn buffer_why(root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
    let plain = to_literals(root, kingdom).ok()?;
    let c = buffer_candidates(&plain);
    if c.is_empty() {
        return Some("no item loop".into());
    }
    let why: Vec<String> = c.iter().filter_map(|p| p.as_ref().err().cloned()).collect();
    (why.len() == c.len()).then(|| why.join("; "))
}

/// Why each rule of this module does not fire (None when it fires).
pub fn why_not(rule: crate::wgsl::rules::Rule, root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
    use crate::wgsl::rules::Rule;
    match rule {
        Rule::DropDeadStore => drop_dead_store_sites(root).is_empty().then(|| "every register written is read".to_string()),
        Rule::UnrollK4 => {
            let plain = to_literals(root, kingdom).ok()?;
            let loops = k_loops(&plain);
            let why: Vec<String> = loops.iter().filter_map(|(n, d, lp)| unroll_plan(*n, *d, lp, 4).err()).collect();
            (loops.is_empty() || why.len() == loops.len()).then(|| if loops.is_empty() { "no counted loop reads the tiles".to_string() } else { why.join("; ") })
        }
        Rule::RollK => roll_why(root, kingdom),
        Rule::SplitAccumulator => split_why(root, kingdom),
        Rule::WgShape => wg_shape_why(root, kingdom),
        Rule::HoistLoad => hoist_load_sites(root, kingdom).is_empty().then(|| "no loop reads a value from memory that it does not change".to_string()),
        Rule::PadShared => pad_why(root, kingdom),
        Rule::GrowCols => grow_cols_plan(root, kingdom).err(),
        Rule::ShrinkCols => shrink_cols_plan(root, kingdom).err(),
        Rule::DoubleBuffer => buffer_why(root, kingdom),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::rules::Rule;
    use crate::wgsl::safe_karva::render_tree;
    use crate::wgsl::safe_reader::read_safe;

    /// The corpus kernels and the rule population popR_rules (when this
    /// checkout holds it), read.
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

    const NEW: [Rule; 10] = [Rule::DropDeadStore, Rule::UnrollK4, Rule::RollK, Rule::SplitAccumulator, Rule::WgShape, Rule::HoistLoad, Rule::PadShared, Rule::GrowCols, Rule::ShrinkCols, Rule::DoubleBuffer];

    /// EVERY CHILD OF EVERY NEW RULE PASSES THE LAYERED CHECK (legality,
    /// render, naga validation, read back) on every kernel where the rule
    /// fires; a rule that fires and fails is a bug in the rule. The counts
    /// by rule are printed; the rules this population should light are
    /// asserted to fire.
    #[test]
    fn every_child_of_the_new_rules_passes_the_layered_check() {
        let k = WgslKingdom::load_safe();
        let all = kernels(&k);
        let mut failures = Vec::new();
        let mut fired: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for (name, t) in &all {
            for rule in NEW {
                for site in rule.sites(t, &k) {
                    let e = fired.entry(rule.name()).or_default();
                    e.0 += 1;
                    match rule.apply(t, &site, &k).map_err(|e| ("apply", e)).and_then(|c| knob::check_text(&c, &k).map_err(|(s, e)| (s.name(), e))) {
                        Ok(_) => e.1 += 1,
                        Err((stage, err)) => failures.push(format!("{name} {} {}: {stage}: {}", rule.name(), site.what, err.chars().take(240).collect::<String>())),
                    }
                }
            }
        }
        for (r, (n, ok)) in &fired {
            println!("{r}: {n} children, {ok} pass the check");
        }
        assert!(failures.is_empty(), "{} children fail:\n{}", failures.len(), failures.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
        if all.len() > 40 {
            for r in [Rule::DropDeadStore, Rule::SplitAccumulator, Rule::GrowCols, Rule::DoubleBuffer, Rule::UnrollK4] {
                assert!(fired.get(r.name()).is_some_and(|(n, _)| *n > 0), "{} fires on no kernel of popR_rules", r.name());
            }
        }
    }

    /// The inverses: rolling an unroll by two or four gives back the
    /// kernel's text; shrinking a grown column block gives back the kernel's
    /// text; a grown block's child shrinks.
    #[test]
    fn the_inverses_give_back_the_kernel() {
        let k = WgslKingdom::load_safe();
        let text = |t: &Tree| render_tree(t, &k).unwrap().text;
        let (mut rolled, mut shrunk) = (0usize, 0usize);
        for (name, t) in kernels(&k) {
            let plain = to_literals(&t, &k).unwrap();
            for site in crate::wgsl::rules::unroll_k_sites(&t, &k) {
                for f in [2u32, 4] {
                    let Ok(u) = crate::wgsl::rules::unroll_k_by(&t, site.index, f, &k) else { continue };
                    let sites = roll_k_sites(&u, &k);
                    assert_eq!(sites.len(), 1, "{name}: unrolled by {f}, {} roll sites", sites.len());
                    let back = roll_k(&u, &sites[0], &k).unwrap();
                    assert_eq!(text(&back), text(&plain), "{name}: unroll by {f} then roll is another kernel");
                    rolled += 1;
                }
            }
            if let Ok(g) = grow_cols(&t, &k) {
                let back = shrink_cols(&g, &k).unwrap_or_else(|e| panic!("{name}: the grown column block does not shrink: {e}"));
                assert_eq!(text(&back), text(&t), "{name}: grow then shrink the columns is another kernel");
                shrunk += 1;
            }
        }
        println!("rolled back {rolled}, shrunk back {shrunk}");
        assert!(rolled > 0 && shrunk > 0);
    }

    /// A dead store's register is never written in the child, and the
    /// child's text is the parent's without those lines.
    #[test]
    fn a_dead_register_is_written_nowhere_in_the_child() {
        let k = WgslKingdom::load_safe();
        let mut n = 0usize;
        for (name, t) in kernels(&k) {
            for site in drop_dead_store_sites(&t) {
                let reg = site.what.split(':').next().unwrap().to_string();
                let child = drop_dead_store(&t, &site, &k).unwrap();
                assert!(!any_node(&child, &|x| x.name.starts_with("reg.set_") && x.kids.first().is_some_and(|l| l.name == reg)), "{name}: {reg} still written");
                let (a, b) = (render_tree(&t, &k).unwrap().text, render_tree(&child, &k).unwrap().text);
                let kept: Vec<&str> = a.lines().collect();
                assert!(b.lines().all(|l| kept.contains(&l)), "{name} {}: a line changed", site.what);
                n += 1;
            }
        }
        assert!(n > 0);
    }

    /// Splitting a K loop's accumulators adds one partner a register, zeroed
    /// before the loop and added after it, and halves the passes.
    #[test]
    fn a_split_accumulator_zeroes_its_partner_and_adds_it_after_the_loop() {
        let k = WgslKingdom::load_safe();
        let mut n = 0usize;
        for (name, t) in kernels(&k) {
            for site in split_accumulator_sites(&t, &k) {
                let child = split_accumulator(&t, &site, &k).unwrap();
                let text = render_tree(&child, &k).unwrap().text;
                let plain = to_literals(&t, &k).unwrap();
                let (_, depth, lp) = k_loops(&plain).into_iter().find(|(x, _, _)| *x == site.index).unwrap();
                let plan = split_plan(&plain, site.index, depth, &lp, &k).unwrap();
                for (r, p) in &plan.partner {
                    // Registers 16 and 32 are spelled with an r (f16 and f32 are WGSL types).
                    let spelled = |x: &str| {
                        let n = x.trim_start_matches("reg.");
                        if matches!(n, "f16" | "f32" | "i32") { format!("{n}r") } else { n.to_string() }
                    };
                    let (rr, pp) = (spelled(r), spelled(p));
                    assert!(text.contains(&format!("{rr} = ({rr} + {pp});")), "{name}: {r} does not add {p} after the loop");
                }
                n += 1;
            }
        }
        assert!(n > 0);
    }
}
