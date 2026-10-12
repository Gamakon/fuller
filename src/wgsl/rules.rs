//! STRUCTURAL REWRITE RULES (`docs/PLAN_constrained_mutation.md` task 2,
//! sections 3 and 5): a rule is a pattern on the typed tree with holes, the
//! constraints under which it may fire, and a replacement that changes every
//! coupled position together. A rule lists the sites where it fires in an
//! individual; the operator picks one and rewrites; the child then passes
//! the layered checker (`knob::check_text`: rendered, validated for the
//! adapter, read back by the reader's ownership proof). Rules come from the
//! model edits mined in `docs/MINED_REWRITES.md`.
//!
//! The statement editor here views every block as the flat list of its
//! statements (the grammar spells a block as `one`/`seq2..4` rows nested
//! through `stmt.block`), edits the list and rebuilds the block in the
//! grammar's spelling at the same rung.

use std::collections::{BTreeMap, BTreeSet};

use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::render::{dual_ty, safe_base, Tree};

/// The rung of a block or statement row (`block.seq4_L2` -> `L2`).
fn rung(name: &str) -> Option<&str> {
    let (_, r) = name.rsplit_once('_')?;
    (r.len() == 2 && matches!(r.as_bytes()[0], b'L' | b'C' | b'O' | b'B') && r.as_bytes()[1].is_ascii_digit()).then_some(r)
}

/// A block row of the plain spellings (`block.one`, `block.seq2..4`).
fn is_plain_block(name: &str) -> bool {
    matches!(name.rsplit_once('_').map(|(s, _)| s), Some("block.one" | "block.seq2" | "block.seq3" | "block.seq4")) && rung(name).is_some()
}

/// The statements of a plain block, its `stmt.block` chains opened.
pub fn statements(block: &Tree) -> Vec<Tree> {
    fn walk(t: &Tree, out: &mut Vec<Tree>) {
        if is_plain_block(&t.name) || is_chain_link(t) {
            t.kids.iter().for_each(|k| walk(k, out));
        } else if t.name != "fallback" {
            out.push(t.clone());
        }
    }
    let mut out = Vec::new();
    walk(block, &mut out);
    out
}

/// A plain block at `rung` holding `stmts` in the grammar's spelling (at
/// most four a row, the rest nested through `stmt.block`); an empty list is
/// the block's fallback.
pub fn rebuild(stmts: Vec<Tree>, rung: &str, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let bt = dual_ty(kingdom, &format!("block.{rung}"))?;
    let st = dual_ty(kingdom, &format!("stmt.{rung}"))?;
    if stmts.is_empty() {
        return Ok(Tree::fallback(bt));
    }
    let mut items = stmts;
    while items.len() > 4 {
        let rest = items.split_off(3);
        let inner = rebuild(rest, rung, kingdom)?;
        items.push(Tree::new(&format!("stmt.block_{rung}"), st, vec![inner]));
    }
    let name = match items.len() {
        1 => "one",
        2 => "seq2",
        3 => "seq3",
        _ => "seq4",
    };
    Ok(Tree::new(&format!("block.{name}_{rung}"), bt, items))
}

/// Every statement of the kernel in tree order (pre-order: the order a
/// thread starts them outside loops), each with its pre-order node number
/// in the whole kernel, whether a loop encloses it, and its tree.
#[derive(Clone, Debug)]
pub struct Stmt {
    pub index: usize,
    pub node: usize,
    pub in_loop: bool,
    pub tree: Tree,
}

/// A `stmt.block` row that only links a block to the rest of its statements.
fn is_chain_link(t: &Tree) -> bool {
    t.name.rsplit_once('_').map(|(s, _)| s) == Some("stmt.block") && t.kids.len() == 1 && is_plain_block(&t.kids[0].name)
}

/// The kernel's statements (see [`Stmt`]), numbered in the order
/// [`replace_statement`] meets them.
pub fn kernel_statements(root: &Tree) -> Vec<Stmt> {
    fn walk(t: &Tree, counter: &mut usize, in_loop: bool, is_stmt: bool, out: &mut Vec<Stmt>) {
        let me = *counter;
        *counter += 1;
        if is_stmt {
            out.push(Stmt { index: out.len(), node: me, in_loop, tree: t.clone() });
        }
        let looped = in_loop || safe_base(&t.name).starts_with("loop.");
        let container = is_plain_block(&t.name);
        for k in &t.kids {
            walk(k, counter, looped, container && !is_chain_link(k) && k.name != "fallback", out);
        }
    }
    let mut out = Vec::new();
    walk(root, &mut 0, false, false, &mut out);
    out
}

/// The pre-order node numbers at which each leaf name occurs in the kernel.
fn leaf_positions(root: &Tree) -> std::collections::HashMap<String, Vec<usize>> {
    fn walk(t: &Tree, counter: &mut usize, out: &mut std::collections::HashMap<String, Vec<usize>>) {
        if t.kids.is_empty() {
            out.entry(t.name.clone()).or_default().push(*counter);
        }
        *counter += 1;
        t.kids.iter().for_each(|k| walk(k, counter, out));
    }
    let mut out = std::collections::HashMap::new();
    walk(root, &mut 0, &mut out);
    out
}

/// The kernel with the statement of tree-order index `index` (as
/// [`kernel_statements`] numbers them) replaced by `with` (none removes it).
pub fn replace_statement(root: &Tree, index: usize, with: Vec<Tree>, kingdom: &WgslKingdom) -> Result<Tree, String> {
    fn walk(t: &Tree, seen: &mut usize, index: usize, with: &mut Option<Vec<Tree>>, kingdom: &WgslKingdom) -> Result<Tree, String> {
        if is_plain_block(&t.name) {
            // Walk the statements in the order kernel_statements numbers them.
            let mut out = Vec::new();
            for s in statements(t) {
                let me = *seen;
                *seen += 1;
                if me == index {
                    if let Some(w) = with.take() {
                        // The replacement's own statements are not walked.
                        *seen += count_statements(&s);
                        out.extend(w);
                        continue;
                    }
                }
                out.push(walk_kids(&s, seen, index, with, kingdom)?);
            }
            let r = rung(&t.name).ok_or_else(|| format!("{} has no rung", t.name))?;
            return rebuild(out, r, kingdom);
        }
        walk_kids(t, seen, index, with, kingdom)
    }
    fn walk_kids(t: &Tree, seen: &mut usize, index: usize, with: &mut Option<Vec<Tree>>, kingdom: &WgslKingdom) -> Result<Tree, String> {
        Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| walk(k, seen, index, with, kingdom)).collect::<Result<_, _>>()? })
    }
    /// The statements nested in a statement (its own blocks', recursively).
    fn count_statements(t: &Tree) -> usize {
        if is_plain_block(&t.name) {
            return statements(t).iter().map(|s| 1 + count_statements(s)).sum();
        }
        t.kids.iter().map(count_statements).sum()
    }
    let mut with = Some(with);
    let out = walk(root, &mut 0, index, &mut with, kingdom)?;
    if with.is_some() {
        return Err(format!("our software has a bug: no statement {index} in the kernel"));
    }
    Ok(out)
}

/// A zero value: a literal 0 or false, a fallback value, or a zero splat.
fn is_zero(t: &Tree) -> bool {
    (t.kids.is_empty() && t.name.starts_with("literal.") && t.konst == 0.0) || t.name == "fallback" || (t.name.starts_with("shape.splat") && t.kids.len() == 1 && is_zero(&t.kids[0]))
}

/// The storage a zero store writes (`parr.a`, `reg.i3`), when `s` is one:
/// a private array element or a whole register set to zero.
fn zero_store(s: &Tree) -> Option<&str> {
    let base = safe_base(&s.name);
    let whole_register = s.name.starts_with("reg.set_") && !s.name.starts_with("reg.set_v_lane") && s.kids.len() == 2;
    if (base == "parr.set_safe" && s.kids.len() == 3 && is_zero(&s.kids[2])) || (whole_register && is_zero(&s.kids[1])) {
        let leaf = &s.kids[0];
        return (leaf.kids.is_empty()).then_some(leaf.name.as_str());
    }
    None
}

/// The storages a statement only zeroes: a zero store, or a counted tile
/// loop whose body is nothing but zero stores (a zero-fill loop).
fn zeroes(s: &Tree) -> Option<Vec<String>> {
    if let Some(x) = zero_store(s) {
        return Some(vec![x.to_string()]);
    }
    let base = safe_base(&s.name);
    if matches!(base, "loop.tile_bounded" | "loop.unroll") && s.kids.len() == 2 {
        let body = statements(&s.kids[1]);
        let all: Option<Vec<String>> = body.iter().map(|b| zero_store(b).map(str::to_string)).collect();
        return all.filter(|v| !v.is_empty());
    }
    None
}

/// A site where a rule fires: the statement's tree-order index and what it is.
#[derive(Clone, Debug)]
pub struct Site {
    pub index: usize,
    pub what: String,
}

/// RULE: DROP A DEAD ZERO-FILL (mined: "drop the zero-fill tile loop when
/// its array is fully written before read", mean gain 0.162 over the pairs
/// showing it, confounded). A statement that only stores zero into private
/// arrays or registers, outside every loop, where nothing in the kernel
/// touches those storages before it, is the identity: WGSL gives every
/// function-scope variable its zero value (the renderer declares registers
/// `= 0.0` and private arrays without an initializer).
pub fn drop_zero_fill_sites(root: &Tree) -> Vec<Site> {
    let positions = leaf_positions(root);
    kernel_statements(root)
        .into_iter()
        .filter(|s| !s.in_loop)
        .filter_map(|s| {
            let stores = zeroes(&s.tree)?;
            let untouched = stores.iter().all(|x| positions.get(x).is_some_and(|p| p.iter().all(|&n| n >= s.node)));
            untouched.then(|| Site { index: s.index, what: format!("{} zeroing {}", s.tree.name, stores.join(", ")) })
        })
        .collect()
}

/// The child of dropping the dead zero-fill at `site`.
pub fn drop_zero_fill(root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
    replace_statement(root, site.index, Vec::new(), kingdom)
}

// ---- RULE: GROW THE ROW BLOCK AT A CONSTANT WORKGROUP TILE ----

/// `index.of_row` of a local invocation id: the thread's row in its workgroup.
fn is_row_lid(t: &Tree) -> bool {
    t.name == "index.of_row" && t.kids.first().is_some_and(|a| a.name.starts_with("axis.") && a.name.ends_with("_lid"))
}

/// A workgroup vec4 read (`wg.get_v_safe`) and its array leaf.
fn wg_vec4_read(t: &Tree) -> Option<&str> {
    (t.name == "wg.get_v_safe" && t.kids.len() == 2 && t.kids[0].kids.is_empty()).then(|| t.kids[0].name.as_str())
}

/// The storage key a statement assigns whole: a register (`reg.v6`) or a
/// private array element at a literal index (`parr.v[2]`).
fn target_key(s: &Tree) -> Option<String> {
    if s.name.starts_with("reg.set_") && !s.name.starts_with("reg.set_v_lane") && s.kids.len() == 2 && s.kids[0].kids.is_empty() {
        return Some(s.kids[0].name.clone());
    }
    if safe_base(&s.name) == "parr.set_safe" && s.kids.len() == 3 {
        let i = s.kids[1].kids.first().filter(|l| l.name.starts_with("literal.") && l.kids.is_empty())?;
        return Some(format!("{}[{}]", s.kids[0].name, i.konst));
    }
    None
}

/// The storage keys a value expression reads: registers, private array
/// elements at literal indices; `arr[*]` for one at any other index.
fn read_keys(t: &Tree, out: &mut BTreeSet<String>) {
    if t.name.starts_with("reg.get_") && t.kids.first().is_some_and(|l| l.kids.is_empty()) {
        out.insert(t.kids[0].name.clone());
    }
    if t.name.starts_with("parr.get_") && t.kids.len() >= 2 {
        match t.kids[1].kids.first().filter(|l| l.name.starts_with("literal.") && l.kids.is_empty()) {
            Some(i) => out.insert(format!("{}[{}]", t.kids[0].name, i.konst)),
            None => out.insert(format!("{}[*]", t.kids[0].name)),
        };
    }
    t.kids.iter().for_each(|k| read_keys(k, out));
}

/// The value read by an assignment (every child but the target's).
fn assigned_values(s: &Tree) -> &[Tree] {
    if s.name.starts_with("reg.set_") {
        &s.kids[1..]
    } else if safe_base(&s.name) == "parr.set_safe" {
        &s.kids[2..]
    } else {
        &s.kids[..]
    }
}

/// The lanes of the row tile a statement reads (`shape.access_j` of a read of `tile`).
fn row_lanes(t: &Tree, tile: &str, out: &mut BTreeSet<usize>, bare: &mut bool) {
    for k in &t.kids {
        if wg_vec4_read(k) == Some(tile) {
            match t.name.strip_prefix("shape.access_").and_then(|j| j.parse::<usize>().ok()) {
                Some(j) => {
                    out.insert(j);
                }
                None => *bare = true,
            }
        }
        row_lanes(k, tile, out, bare);
    }
}

/// What growing the row block of a kernel needs, found in it.
#[derive(Clone, Debug)]
pub struct GrowPlan {
    /// The workgroup array holding the rows, four to a vec4 (`wg.vb`).
    pub tile: String,
    /// Every storage of the rows' accumulation, with its row 0..4.
    pub family: BTreeMap<String, usize>,
    /// Old storage key to the new row's (registers renamed, `parr.v[i]` to `[i + 4]`).
    pub map: BTreeMap<String, String>,
    pub threads: u32,
}

/// THE PRECONDITIONS of growing the row block 4 -> 8 at a constant
/// workgroup tile (the workgroup's row extent halved), or why not:
/// one vec4 workgroup array holds the rows, every read of it a lane of a
/// vec4 at an index of the thread's row id; every use of that row id is in
/// such a read; every accumulation reading it is a statement of one lane
/// (one row); every storage the rows' sums flow through belongs to one
/// row; a private array element of the flow sits at its row's index; every
/// use of the flat thread index is inside a loop striding by the threads.
pub fn grow_rows_plan(root: &Tree, kingdom: &WgslKingdom) -> Result<GrowPlan, String> {
    use crate::wgsl::params::{Knob, Knobs};
    let knobs = Knobs::of_root(root)?;
    if knobs.grid.by != 4 {
        return Err(format!("the row block is {}, not 4", knobs.grid.by));
    }
    let wrow = knobs.value(Knob::Wrow);
    if wrow < 2 || wrow % 2 != 0 {
        return Err(format!("the workgroup's row extent {wrow} does not halve"));
    }
    // The row tile: the vec4 array read at the thread's row id.
    let mut tiles = BTreeSet::new();
    fn find_tiles(t: &Tree, out: &mut BTreeSet<String>) {
        if let Some(a) = wg_vec4_read(t) {
            if any_node(&t.kids[1], &is_row_lid) {
                out.insert(a.to_string());
            }
        }
        t.kids.iter().for_each(|k| find_tiles(k, out));
    }
    find_tiles(root, &mut tiles);
    let tile = match tiles.into_iter().collect::<Vec<_>>().as_slice() {
        [t] => t.clone(),
        other => return Err(format!("{} vec4 arrays are read at the thread's row id, not one", other.len())),
    };
    // Every row-id use inside a read of the tile.
    fn row_lid_outside(t: &Tree, tile: &str) -> bool {
        if wg_vec4_read(t) == Some(tile) {
            return false;
        }
        is_row_lid(t) || t.kids.iter().any(|k| row_lid_outside(k, tile))
    }
    if row_lid_outside(root, &tile) {
        return Err("the thread's row id is used outside the row tile's reads".into());
    }
    let stmts = kernel_statements(root);
    // Seeds: the statements reading the tile, one lane each.
    let mut family: BTreeMap<String, usize> = BTreeMap::new();
    for s in stmts.iter().filter(|s| !is_compound(&s.tree)) {
        let (mut lanes, mut bare) = (BTreeSet::new(), false);
        row_lanes(&s.tree, &tile, &mut lanes, &mut bare);
        if lanes.is_empty() && !bare {
            continue;
        }
        if bare || lanes.len() != 1 {
            return Err(format!("{} reads the row tile other than as one lane", s.tree.name));
        }
        let key = target_key(&s.tree).filter(|k| k.starts_with("reg.")).ok_or_else(|| format!("{} reads the row tile and assigns no register", s.tree.name))?;
        let row = *lanes.iter().next().expect("one lane");
        if family.insert(key.clone(), row).is_some_and(|r| r != row) {
            return Err(format!("{key} accumulates two rows"));
        }
    }
    if family.is_empty() {
        return Err("no accumulation reads the row tile".into());
    }
    // The flow of the rows' sums, to a fixed point.
    loop {
        let mut grew = false;
        for s in stmts.iter().filter(|s| !is_compound(&s.tree)) {
            if !(s.tree.name.starts_with("reg.set_") || safe_base(&s.tree.name) == "parr.set_safe" || s.tree.name.starts_with("store.") || s.tree.name.starts_with("wg.set")) {
                continue;
            }
            let mut reads = BTreeSet::new();
            assigned_values(&s.tree).iter().for_each(|v| read_keys(v, &mut reads));
            let rows: BTreeSet<usize> = reads.iter().filter_map(|k| family.get(k).copied()).collect();
            if rows.is_empty() {
                continue;
            }
            if rows.len() > 1 {
                return Err(format!("{} mixes rows {rows:?}", s.tree.name));
            }
            let row = *rows.iter().next().expect("one row");
            let key = target_key(&s.tree).ok_or_else(|| format!("{} reads a row's sum and assigns no whole storage", s.tree.name))?;
            match family.get(&key) {
                Some(&r) if r != row => return Err(format!("{key} holds rows {r} and {row}")),
                Some(_) => {}
                None => {
                    family.insert(key, row);
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    // A private array element of the flow at its row's index, its row + 4 free.
    let positions = leaf_positions(root);
    let mut map = BTreeMap::new();
    let mut used: BTreeSet<String> = positions.keys().cloned().collect();
    let mut literal_parr: BTreeSet<String> = BTreeSet::new();
    for s in &stmts {
        if let Some(k) = target_key(&s.tree).filter(|k| k.contains('[')) {
            literal_parr.insert(k);
        }
        let mut reads = BTreeSet::new();
        read_keys(&s.tree, &mut reads);
        literal_parr.extend(reads.into_iter().filter(|k| k.contains('[') && !k.ends_with("[*]")));
    }
    for (key, &row) in &family {
        if let Some((arr, idx)) = key.split_once('[') {
            let i: f32 = idx.trim_end_matches(']').parse().map_err(|_| format!("{key}: no index"))?;
            if i != row as f32 {
                return Err(format!("{key} holds row {row} at another index"));
            }
            let new = format!("{arr}[{}]", i + 4.0);
            if literal_parr.contains(&new) {
                return Err(format!("{new} is in use"));
            }
            map.insert(key.clone(), new);
        } else {
            // A fresh register of the same kind.
            let stem = key.trim_end_matches(|c: char| c.is_ascii_digit());
            let fresh = (0..64).map(|n| format!("{stem}{n}")).find(|c| !used.contains(c) && kingdom.rows.iter().any(|r| r.name == *c)).ok_or_else(|| format!("no free register like {key}"))?;
            used.insert(fresh.clone());
            map.insert(key.clone(), fresh);
        }
    }
    // Every flat thread index inside a loop striding by the threads.
    let threads = knobs.value(Knob::Threads);
    let mut outside = false;
    lidx_outside_thread_loops(root, threads, 0, false, &mut outside);
    if outside {
        return Err("the flat thread index is used outside a loop striding by the threads, or such a loop's counter strides something else".into());
    }
    Ok(GrowPlan { tile, family, map, threads })
}

/// A statement holding blocks (a loop, an if): its work is its nested statements.
fn is_compound(t: &Tree) -> bool {
    t.kids.iter().any(|k| any_node(k, &|n| is_plain_block(&n.name) || n.name.starts_with("block.")))
}

/// Any node of `t` satisfying `p`.
fn any_node(t: &Tree, p: &dyn Fn(&Tree) -> bool) -> bool {
    p(t) || t.kids.iter().any(|k| any_node(k, p))
}

/// The threads as a value: a reference to them or the literal.
fn is_threads(t: &Tree, threads: u32) -> bool {
    t.name == "literal.p_threads" || (t.kids.is_empty() && t.name.starts_with("literal.") && !t.name.starts_with("literal.p_") && t.konst == threads as f32)
}

/// Whether a counted loop at tile depth `d` strides its body by the threads
/// (`lidx + threads * k`, k its counter).
fn strides_by_threads(body: &Tree, d: usize, threads: u32) -> bool {
    let counter = tile_counter_name(d);
    any_node(body, &|n| n.name == "index.mul" && n.kids.len() == 2 && (0..2).any(|i| is_threads(&n.kids[i], threads) && any_node(&n.kids[1 - i], &|m| m.name == counter)))
}

/// Whether every use of the counter of the loop at tile depth `d` in its
/// body strides the threads (`threads * k`): a counter also used another
/// way (`t + (lidx >> 4) + 16u * k`, 16 rows a pass of 256 threads) covers
/// the tile only at this thread count.
fn counter_only_strides_threads(body: &Tree, d: usize, threads: u32) -> bool {
    let counter = tile_counter_name(d);
    fn walk(t: &Tree, counter: &str, threads: u32) -> bool {
        if t.name == "index.mul" && t.kids.len() == 2 && (0..2).any(|i| is_threads(&t.kids[i], threads)) {
            return true;
        }
        if t.name == counter {
            return false;
        }
        t.kids.iter().all(|k| walk(k, counter, threads))
    }
    walk(body, counter, threads)
}

fn tile_counter_name(d: usize) -> &'static str {
    ["safe.tile_counter", "safe.tile_counter2", "safe.tile_counter3", "safe.tile_counter4"][d.min(3)]
}

fn is_counted_loop(name: &str) -> bool {
    matches!(safe_base(name), "loop.tile_bounded" | "loop.unroll" | "loop.tile_strided_bounded")
}

fn lidx_outside_thread_loops(t: &Tree, threads: u32, depth: usize, inside: bool, outside: &mut bool) {
    if t.name == "index.of_lidx" && !inside {
        *outside = true;
    }
    let base = safe_base(&t.name);
    if base.starts_with("loop.") && !base.starts_with("loop.items") {
        let body = t.kids.last().expect("a loop has a body");
        let striding = is_counted_loop(&t.name) && t.kids.len() >= 2 && strides_by_threads(body, depth, threads);
        if striding && !counter_only_strides_threads(body, depth, threads) {
            *outside = true;
        }
        for k in &t.kids {
            lidx_outside_thread_loops(k, threads, depth + 1, inside || striding, outside);
        }
        return;
    }
    t.kids.iter().for_each(|k| lidx_outside_thread_loops(k, threads, depth, inside, outside));
}

/// `t` with the row tile's reads at row id `x` read at `x * 2 + extra`.
fn spread_rows(t: &Tree, tile: &str, extra: Option<u32>, in_tile: bool) -> Tree {
    if in_tile && is_row_lid(t) {
        let two = Tree::literal("literal.index", t.ty, 2.0);
        let doubled = Tree::new("index.mul", t.ty, vec![t.clone(), two]);
        return match extra {
            Some(e) => Tree::new("index.add", t.ty, vec![doubled, Tree::literal("literal.index", t.ty, e as f32)]),
            None => doubled,
        };
    }
    let inside = in_tile || wg_vec4_read(t) == Some(tile);
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| spread_rows(k, tile, extra, inside)).collect() }
}

/// `t` with the family's storages renamed to the new row's.
fn rename(t: &Tree, map: &BTreeMap<String, String>) -> Tree {
    if t.kids.is_empty() {
        if let Some(n) = map.get(&t.name) {
            return Tree { name: n.clone(), ty: t.ty, konst: t.konst, kids: Vec::new() };
        }
    }
    // A private array element at a literal index: the index moves with its row.
    let parr = (t.name.starts_with("parr.get_") || safe_base(&t.name) == "parr.set_safe") && t.kids.len() >= 2;
    let mut kids: Vec<Tree> = t.kids.iter().map(|k| rename(k, map)).collect();
    if parr {
        if let Some(i) = t.kids[1].kids.first().filter(|l| l.name.starts_with("literal.") && l.kids.is_empty()) {
            if let Some(new) = map.get(&format!("{}[{}]", t.kids[0].name, i.konst)) {
                let v: f32 = new.rsplit_once('[').and_then(|(_, x)| x.trim_end_matches(']').parse().ok()).unwrap_or(i.konst);
                kids[1].kids[0].konst = v;
            }
        }
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
}

/// Every statement mapped: an assignment of the family becomes itself
/// (its tile reads spread to `x * 2`) followed by the new row's copy
/// (renamed, its tile reads at `x * 2 + 1`); counted loops striding by the
/// threads run twice as many passes.
fn grow_walk(t: &Tree, plan: &GrowPlan, depth: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
    if is_plain_block(&t.name) {
        let mut out = Vec::new();
        for s in statements(t) {
            match target_key(&s).filter(|k| plan.family.contains_key(k)) {
                Some(_) => {
                    out.push(spread_rows(&s, &plan.tile, None, false));
                    out.push(rename(&spread_rows(&s, &plan.tile, Some(1), false), &plan.map));
                }
                None => out.push(grow_walk(&s, plan, depth, kingdom)?),
            }
        }
        return rebuild(out, rung(&t.name).ok_or_else(|| format!("{} has no rung", t.name))?, kingdom);
    }
    let base = safe_base(&t.name);
    let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
    let mut kids = Vec::with_capacity(t.kids.len());
    for (i, k) in t.kids.iter().enumerate() {
        kids.push(grow_walk(k, plan, depth + usize::from(looped), kingdom)?);
        if i == 0 && is_counted_loop(&t.name) && t.kids.len() >= 2 && strides_by_threads(t.kids.last().expect("a body"), depth, plan.threads) {
            kids[0].konst *= 2.0;
        }
    }
    Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids })
}

/// The kernel with its row block grown 4 -> 8 and its workgroup's row
/// extent halved: the same workgroup tile, half the threads, each thread
/// twice the rows (the miner's register block 4 -> 8, with the epilogue's
/// row loop 4 -> 8 that its references give).
pub fn grow_rows(root: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    use crate::wgsl::knob::{set, Site as KnobSite};
    use crate::wgsl::params::{Knob, Knobs};
    let plan = grow_rows_plan(root, kingdom)?;
    let knobs = Knobs::of_root(root)?;
    let mut kids = root.kids.clone();
    kids[1] = grow_walk(&root.kids[1], &plan, 0, kingdom)?;
    kids[2] = grow_walk(&root.kids[2], &plan, 0, kingdom)?;
    let t = Tree { name: root.name.clone(), ty: root.ty, konst: root.konst, kids };
    let row_site = if knobs.grid.rows_x { KnobSite::Wx } else { KnobSite::Wy };
    let t = set(&t, &row_site, knobs.value(Knob::Wrow) / 2);
    Ok(set(&t, &KnobSite::By, 8))
}

/// RULE: GROW THE ROW BLOCK (one site: the kernel).
pub fn grow_rows_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    match grow_rows_plan(root, kingdom) {
        Ok(p) => vec![Site { index: 0, what: format!("rows 4 -> 8 over {}, {} storages", p.tile, p.family.len()) }],
        Err(_) => Vec::new(),
    }
}

/// A rule of the catalogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rule {
    DropZeroFill,
    GrowRows,
}

impl Rule {
    pub const ALL: [Rule; 2] = [Rule::DropZeroFill, Rule::GrowRows];

    pub fn name(self) -> &'static str {
        match self {
            Rule::DropZeroFill => "drop-zero-fill",
            Rule::GrowRows => "grow-rows",
        }
    }

    pub fn sites(self, root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
        match self {
            Rule::DropZeroFill => drop_zero_fill_sites(root),
            Rule::GrowRows => grow_rows_sites(root, kingdom),
        }
    }

    /// Why the rule does not fire in this kernel (None when it fires).
    pub fn why_not(self, root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
        match self {
            Rule::DropZeroFill => drop_zero_fill_sites(root).is_empty().then(|| "no dead zero store outside a loop".to_string()),
            Rule::GrowRows => grow_rows_plan(root, kingdom).err(),
        }
    }

    pub fn apply(self, root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
        match self {
            Rule::DropZeroFill => drop_zero_fill(root, site, kingdom),
            Rule::GrowRows => grow_rows(root, kingdom),
        }
    }
}

/// A rule's child: the tree, the rule and the site it fired at.
#[derive(Clone, Debug)]
pub struct RuleChild {
    pub tree: Tree,
    pub rule: Rule,
    pub site: Site,
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One rule mutation: among `rules`, the (rule, site) pairs that light up
/// in this kernel, one chosen uniformly. An error when none fires.
pub fn mutate(root: &Tree, rules: &[Rule], kingdom: &WgslKingdom, rng: &mut u64) -> Result<RuleChild, String> {
    let options: Vec<(Rule, Site)> = rules.iter().flat_map(|&r| r.sites(root, kingdom).into_iter().map(move |s| (r, s))).collect();
    if options.is_empty() {
        return Err("no rule fires in this kernel".into());
    }
    let (rule, site) = options[(next(rng) % options.len() as u64) as usize].clone();
    let tree = rule.apply(root, &site, kingdom)?;
    Ok(RuleChild { tree, rule, site })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::safe_karva::render_tree;
    use crate::wgsl::safe_reader::read_safe;

    fn corpus(k: &WgslKingdom) -> Vec<(String, Tree)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for dir in ["tests/corpus/safewgsl", "logs/HGP/001/popB_g080/texts"] {
            if let Ok(rd) = std::fs::read_dir(root.join(dir)) {
                files.extend(rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")));
            }
        }
        files.sort();
        files.into_iter().filter_map(|f| Some((f.display().to_string(), read_safe(&std::fs::read_to_string(&f).ok()?, k).ok()?.tree))).collect()
    }

    /// The editor changes nothing it is not asked to: every block of every
    /// corpus kernel flattened and rebuilt renders the same text, and the
    /// statements are numbered the same way by both walks.
    #[test]
    fn a_block_flattened_and_rebuilt_renders_the_same_text() {
        let k = WgslKingdom::load_safe();
        for (name, t) in corpus(&k) {
            let stmts = kernel_statements(&t);
            assert!(!stmts.is_empty(), "{name}");
            let same = replace_statement(&t, 0, vec![stmts[0].tree.clone()], &k).unwrap();
            assert_eq!(render_tree(&t, &k).unwrap().text, render_tree(&same, &k).unwrap().text, "{name}");
            let last = stmts.len() - 1;
            let same = replace_statement(&t, last, vec![stmts[last].tree.clone()], &k).unwrap();
            assert_eq!(render_tree(&t, &k).unwrap().text, render_tree(&same, &k).unwrap().text, "{name}");
        }
    }

    /// A dead zero-fill is dropped and nothing else changes: the child's
    /// text is the parent's without those lines, and it passes the check.
    #[test]
    fn a_dead_zero_fill_is_dropped_and_the_child_checks() {
        let k = WgslKingdom::load_safe();
        let mut fired = 0usize;
        for (name, t) in corpus(&k) {
            for site in drop_zero_fill_sites(&t) {
                let child = drop_zero_fill(&t, &site, &k).unwrap();
                let (a, b) = (render_tree(&t, &k).unwrap().text, render_tree(&child, &k).unwrap().text);
                assert!(b.lines().count() < a.lines().count(), "{name} {}: nothing dropped", site.what);
                let kept: Vec<&str> = a.lines().collect();
                assert!(b.lines().all(|l| kept.contains(&l)), "{name} {}: a line changed", site.what);
                crate::wgsl::knob::check_text(&child, &k).map(|_| ()).unwrap_or_else(|(s, e)| panic!("{name} {}: {}: {e}", site.what, s.name()));
                fired += 1;
            }
        }
        assert!(fired > 0);
    }

    /// GROW THE ROW BLOCK on saved kernels of reserve2 (2026-10-12): a
    /// 16 x 16 workgroup of 4 x 4 blocks becomes 16 x 8 of 4 x 8 with the
    /// same tile (its 24 children were correct on all 128 device cases); a
    /// kernel already at 8 rows, and one whose loader strides rows by a
    /// literal fixed by the thread count (three children computed wrong
    /// values before that precondition), are not sites.
    #[test]
    fn the_row_block_grows_at_a_constant_tile_where_its_preconditions_hold() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/reserve2/texts");
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok().map(|src| read_safe(&src, &k).unwrap().tree);
        let Some(four) = read("0.1219_fit_A2_stopped_1729Z_g000_r11_ca96ca.wgsl") else { return };
        let plan = grow_rows_plan(&four, &k).unwrap();
        assert_eq!(plan.tile, "wg.vb");
        let child = grow_rows(&four, &k).unwrap();
        let (text, _) = crate::wgsl::knob::check_text(&child, &k).unwrap_or_else(|(s, e)| panic!("{}: {e}", s.name()));
        assert!(text.starts_with("// grid 16 8 4 8"), "{}", &text[..40]);
        assert!(text.contains("(lid.y * 2u)") && text.contains("((lid.y * 2u) + 1u)") && text.contains("pv[7u]") && text.contains("128u * k"), "{text}");
        let eight = read("0.1066_fit_A2_stopped_1702Z_g001_r12_d93a44.wgsl").unwrap();
        assert!(grow_rows_plan(&eight, &k).unwrap_err().contains("row block is 8"));
        let row_strided = read("0.1222_fit_A2_paused_1912Z_g000_r5_13236c.wgsl").unwrap();
        assert!(grow_rows_plan(&row_strided, &k).unwrap_err().contains("flat thread index"));
    }

    /// Never a zero store inside a loop, nor one whose storage was touched
    /// before (a register zeroed again after a use is not dead).
    #[test]
    fn a_zero_store_after_a_use_or_in_a_loop_is_not_a_site() {
        let k = WgslKingdom::load_safe();
        for (name, t) in corpus(&k) {
            let stmts = kernel_statements(&t);
            let positions = leaf_positions(&t);
            for site in drop_zero_fill_sites(&t) {
                let s = &stmts[site.index];
                assert!(!s.in_loop, "{name}: {}", site.what);
                for x in zeroes(&s.tree).unwrap() {
                    assert!(positions[&x].iter().all(|&n| n >= s.node), "{name}: {x} touched before");
                }
            }
        }
    }
}
