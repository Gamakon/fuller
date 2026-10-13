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
use crate::wgsl::rules_more as more;

/// The rung of a block or statement row (`block.seq4_L2` -> `L2`).
pub(crate) fn rung(name: &str) -> Option<&str> {
    let (_, r) = name.rsplit_once('_')?;
    (r.len() == 2 && matches!(r.as_bytes()[0], b'L' | b'C' | b'O' | b'B') && r.as_bytes()[1].is_ascii_digit()).then_some(r)
}

/// A block row of the plain spellings (`block.one`, `block.seq2..4`).
pub(crate) fn is_plain_block(name: &str) -> bool {
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
pub(crate) fn is_chain_link(t: &Tree) -> bool {
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
pub(crate) fn leaf_positions(root: &Tree) -> std::collections::HashMap<String, Vec<usize>> {
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
pub(crate) fn is_zero(t: &Tree) -> bool {
    (t.kids.is_empty() && t.name.starts_with("literal.") && t.konst == 0.0) || t.name == "fallback" || (t.name.starts_with("shape.splat") && t.kids.len() == 1 && is_zero(&t.kids[0]))
}

/// The storage a zero store writes (`parr.a`, `reg.i3`), when `s` is one:
/// a private array element or a whole register set to zero.
pub(crate) fn zero_store(s: &Tree) -> Option<&str> {
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
pub(crate) fn wg_vec4_read(t: &Tree) -> Option<&str> {
    (t.name == "wg.get_v_safe" && t.kids.len() == 2 && t.kids[0].kids.is_empty()).then(|| t.kids[0].name.as_str())
}

/// The storage key a statement assigns whole: a register (`reg.v6`) or a
/// private array element at a literal index (`parr.v[2]`).
pub(crate) fn target_key(s: &Tree) -> Option<String> {
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
pub(crate) fn read_keys(t: &Tree, out: &mut BTreeSet<String>) {
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
pub(crate) fn assigned_values(s: &Tree) -> &[Tree] {
    if s.name.starts_with("reg.set_") {
        &s.kids[1..]
    } else if safe_base(&s.name) == "parr.set_safe" {
        &s.kids[2..]
    } else {
        &s.kids[..]
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
    let RowFamilies { tile, family } = row_families(root, false)?;
    let stmts = kernel_statements(root);
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
pub(crate) fn is_compound(t: &Tree) -> bool {
    t.kids.iter().any(|k| any_node(k, &|n| is_plain_block(&n.name) || n.name.starts_with("block.")))
}

/// The storages of the rows' accumulation flow closed to a fixed point:
/// an assignment reading one row's storages joins that row; one reading two
/// rows, or reading a row's sum and assigning no whole storage, is an error.
pub(crate) fn close_flow(stmts: &[Stmt], family: &mut BTreeMap<String, usize>) -> Result<(), String> {
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
    Ok(())
}

/// A kernel's row tile and the storages of its rows' accumulation flow.
struct RowFamilies {
    tile: String,
    family: BTreeMap<String, usize>,
}

/// The row tile (the one vec4 array read at the thread's row id), and the
/// storages of the rows' accumulation flow with their rows: a seed is an
/// assignment reading the tile at one row (the lane, plus 4 for a read at
/// `2 y + 1` when `spread`), the flow closed over assignments reading one
/// row's storages. With every use of the row id inside the tile's reads.
fn row_families(root: &Tree, spread: bool) -> Result<RowFamilies, String> {
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
    let mut family: BTreeMap<String, usize> = BTreeMap::new();
    for s in stmts.iter().filter(|s| !is_compound(&s.tree)) {
        let (mut rows, mut bad) = (BTreeSet::new(), None);
        tile_read_rows(&s.tree, &tile, spread, &mut rows, &mut bad);
        if let Some(b) = bad {
            return Err(format!("{}: {b}", s.tree.name));
        }
        if rows.is_empty() {
            continue;
        }
        if rows.len() != 1 {
            return Err(format!("{} reads the row tile at rows {rows:?}, not one", s.tree.name));
        }
        let key = target_key(&s.tree).filter(|k| k.starts_with("reg.")).ok_or_else(|| format!("{} reads the row tile and assigns no register", s.tree.name))?;
        let row = *rows.iter().next().expect("one row");
        if family.insert(key.clone(), row).is_some_and(|r| r != row) {
            return Err(format!("{key} accumulates two rows"));
        }
    }
    if family.is_empty() {
        return Err("no accumulation reads the row tile".into());
    }
    close_flow(&stmts, &mut family)?;
    Ok(RowFamilies { tile, family })
}

/// The rows a statement's reads of the row tile hold: lane j of the vec4
/// at the thread's row id is row j; with `spread`, the read is at `2 y + c`
/// over terms of even factors and c's parity adds 4 (`2 y + 1` holds rows 4..7).
fn tile_read_rows(t: &Tree, tile: &str, spread: bool, out: &mut BTreeSet<usize>, bad: &mut Option<String>) {
    for k in &t.kids {
        if wg_vec4_read(k) == Some(tile) {
            let Some(j) = t.name.strip_prefix("shape.access_").and_then(|j| j.parse::<usize>().ok()) else {
                *bad = Some("reads the row tile other than as one lane".into());
                return;
            };
            if !spread {
                out.insert(j);
                continue;
            }
            let Some(idx) = k.kids[1].kids.first() else {
                *bad = Some("a row tile read without an index".into());
                return;
            };
            let mut terms = Vec::new();
            add_terms(idx, &mut terms);
            let (mut doubled, mut constant, mut odd) = (0usize, 0u32, false);
            for term in terms {
                if let Some(v) = plain_value(term) {
                    constant += v;
                } else if term.name == "index.mul" && term.kids.len() == 2 {
                    let (e, m) = match (plain_value(&term.kids[0]), plain_value(&term.kids[1])) {
                        (_, Some(m)) => (&term.kids[0], m),
                        (Some(m), None) => (&term.kids[1], m),
                        (None, None) => {
                            odd = true;
                            continue;
                        }
                    };
                    if is_row_lid(e) && m == 2 {
                        doubled += 1;
                    } else if m % 2 != 0 || any_node(e, &is_row_lid) {
                        odd = true;
                    }
                } else {
                    odd = true;
                }
            }
            if doubled != 1 || odd {
                *bad = Some("a row tile read is not 2 y + c over even terms".into());
                return;
            }
            out.insert(j + 4 * (constant % 2) as usize);
            continue;
        }
        tile_read_rows(k, tile, spread, out, bad);
        if bad.is_some() {
            return;
        }
    }
}

/// `t` with the row tile's reads at `2 y + c` read at `y + c - c % 2`.
fn unspread_rows(t: &Tree, tile: &str, in_tile: bool) -> Tree {
    if in_tile && t.name == "index.mul" && t.kids.len() == 2 {
        for i in 0..2 {
            if is_row_lid(&t.kids[i]) && plain_value(&t.kids[1 - i]) == Some(2) {
                return t.kids[i].clone();
            }
        }
    }
    // A row-group-0 read's constant is even (the K steps a tile row apart):
    // it stays where it is.
    let inside = in_tile || wg_vec4_read(t) == Some(tile);
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| unspread_rows(k, tile, inside)).collect() }
}

/// THE PRECONDITIONS of shrinking the row block 8 -> 4 at a constant
/// workgroup tile (the row extent doubled): the tile is read at `2 y + c`
/// (lane j of row j + 4 (c % 2)); every row's flow is its own; the threads
/// double within the adapter; every loop striding the flat thread index by
/// the threads has an even number of passes.
pub fn shrink_rows_plan(root: &Tree, kingdom: &WgslKingdom) -> Result<(String, BTreeMap<String, usize>), String> {
    use crate::wgsl::params::{Knob, Knobs};
    let plain = crate::wgsl::params::to_literals(root, kingdom)?;
    let knobs = Knobs::of_root(&plain)?;
    if knobs.grid.by != 8 {
        return Err(format!("the row block is {}, not 8", knobs.grid.by));
    }
    let target = crate::wgsl::target::Target::current();
    let threads = knobs.value(Knob::Threads);
    let wrow = knobs.value(Knob::Wrow);
    if threads * 2 > target.max_invocations || wrow * 2 > target.max_size[if knobs.grid.rows_x { 0 } else { 1 }] {
        return Err(format!("{threads} threads do not double within the adapter"));
    }
    let RowFamilies { tile, family } = row_families(&plain, true)?;
    if !family.values().any(|&r| r >= 4) {
        return Err("no row of the second group".into());
    }
    let mut outside = false;
    lidx_outside_thread_loops(&plain, threads, 0, false, &mut outside);
    if outside {
        return Err("the flat thread index is used outside a loop striding by the threads, or such a loop's counter strides something else".into());
    }
    fn odd_thread_loop(t: &Tree, threads: u32, depth: usize) -> bool {
        let base = safe_base(&t.name);
        let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
        if is_counted_loop(&t.name) && t.kids.len() >= 2 && strides_by_threads(t.kids.last().expect("a body"), depth, threads) && plain_value(&t.kids[0]).is_none_or(|v| v % 2 != 0) {
            return true;
        }
        t.kids.iter().any(|k| odd_thread_loop(k, threads, depth + usize::from(looped)))
    }
    if odd_thread_loop(&plain, threads, 0) {
        return Err("a loop striding by the threads has an odd or unknown number of passes".into());
    }
    Ok((tile, family))
}

/// The kernel with its row block shrunk 8 -> 4 and its workgroup's row
/// extent doubled: rows 4..7 and every storage of their flow dropped, the
/// tile read at `y` where it was read at `2 y`, the loops striding the flat
/// thread index by the threads run half the passes.
pub fn shrink_rows(root: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    use crate::wgsl::knob::{set, Site as KnobSite};
    use crate::wgsl::params::{Knob, Knobs};
    // Planned on the plain values; rewritten on the kernel with its
    // references, so every reference to the row block, the row extent and
    // the threads follows the grid (on the plain tree the threads' stride
    // and the guards kept the old values: all 5 children computed wrong
    // values on the device).
    let (tile, family) = shrink_rows_plan(root, kingdom)?;
    let knobs = Knobs::of_root(root)?;
    let threads = knobs.value(Knob::Threads);
    fn walk(t: &Tree, tile: &str, family: &BTreeMap<String, usize>, threads: u32, depth: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
        if is_plain_block(&t.name) {
            let mut out = Vec::new();
            for s in statements(t) {
                match target_key(&s).and_then(|k| family.get(&k).copied()) {
                    Some(r) if r >= 4 => {}
                    Some(_) => out.push(unspread_rows(&s, tile, false)),
                    None => out.push(walk(&s, tile, family, threads, depth, kingdom)?),
                }
            }
            return rebuild(out, rung(&t.name).ok_or_else(|| format!("{} has no rung", t.name))?, kingdom);
        }
        let base = safe_base(&t.name);
        let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
        let mut kids = Vec::with_capacity(t.kids.len());
        for (i, k) in t.kids.iter().enumerate() {
            kids.push(walk(k, tile, family, threads, depth + usize::from(looped), kingdom)?);
            if i == 0 && is_counted_loop(&t.name) && t.kids.len() >= 2 && strides_by_threads(t.kids.last().expect("a body"), depth, threads) {
                kids[0].konst /= 2.0;
            }
        }
        Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids })
    }
    let mut kids = root.kids.clone();
    kids[1] = walk(&root.kids[1], &tile, &family, threads, 0, kingdom)?;
    kids[2] = walk(&root.kids[2], &tile, &family, threads, 0, kingdom)?;
    let t = Tree { name: root.name.clone(), ty: root.ty, konst: root.konst, kids };
    let row_site = if knobs.grid.rows_x { KnobSite::Wx } else { KnobSite::Wy };
    let t = set(&t, &row_site, knobs.value(Knob::Wrow) * 2);
    Ok(set(&t, &KnobSite::By, 4))
}

/// RULE: SHRINK THE ROW BLOCK (one site: the kernel).
pub fn shrink_rows_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    match shrink_rows_plan(root, kingdom) {
        Ok((tile, family)) => vec![Site { index: 0, what: format!("rows 8 -> 4 over {tile}, {} storages dropped", family.values().filter(|&&r| r >= 4).count()) }],
        Err(_) => Vec::new(),
    }
}

/// Any node of `t` satisfying `p`.
pub(crate) fn any_node(t: &Tree, p: &dyn Fn(&Tree) -> bool) -> bool {
    p(t) || t.kids.iter().any(|k| any_node(k, p))
}

/// The threads as a value: a reference to them or the literal.
pub(crate) fn is_threads(t: &Tree, threads: u32) -> bool {
    t.name == "literal.p_threads" || (t.kids.is_empty() && t.name.starts_with("literal.") && !t.name.starts_with("literal.p_") && t.konst == threads as f32)
}

/// Whether a counted loop at tile depth `d` strides its body by the threads
/// (`lidx + threads * k`, k its counter).
pub(crate) fn strides_by_threads(body: &Tree, d: usize, threads: u32) -> bool {
    let counter = tile_counter_name(d);
    any_node(body, &|n| n.name == "index.mul" && n.kids.len() == 2 && (0..2).any(|i| is_threads(&n.kids[i], threads) && any_node(&n.kids[1 - i], &|m| m.name == counter)))
}

/// Whether every use of the counter of the loop at tile depth `d` in its
/// body strides the threads (`threads * k`): a counter also used another
/// way (`t + (lidx >> 4) + 16u * k`, 16 rows a pass of 256 threads) covers
/// the tile only at this thread count.
pub(crate) fn counter_only_strides_threads(body: &Tree, d: usize, threads: u32) -> bool {
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

pub(crate) fn tile_counter_name(d: usize) -> &'static str {
    ["safe.tile_counter", "safe.tile_counter2", "safe.tile_counter3", "safe.tile_counter4"][d.min(3)]
}

pub(crate) fn is_counted_loop(name: &str) -> bool {
    matches!(safe_base(name), "loop.tile_bounded" | "loop.unroll" | "loop.tile_strided_bounded")
}

pub(crate) fn lidx_outside_thread_loops(t: &Tree, threads: u32, depth: usize, inside: bool, outside: &mut bool) {
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
pub(crate) fn rename(t: &Tree, map: &BTreeMap<String, String>) -> Tree {
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

// ---- the row block grown where a thread owns its rows a workgroup apart ----

/// The kernel with every `lid.y * X + lid.x`, X the workgroup's x extent,
/// written as the flat thread index `lidx` (WGSL's definition of the local
/// invocation index): the loaders of the strided kernels spell it that way,
/// and growing the row block halves the workgroup's row extent, which the
/// flat index follows and the sum written with literals does not.
pub fn flat_thread_ids(root: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let knobs = crate::wgsl::params::Knobs::of_root(root)?;
    let x = knobs.grid.x;
    let index = dual_ty(kingdom, "u32.index")?;
    let lidx = Tree::new("index.of_lidx", index, vec![Tree::new("builtin.local_invocation_index", dual_ty(kingdom, "u32.lidx")?, vec![])]);
    let id_of = |t: &Tree, axis: &str| t.kids.len() == 1 && t.kids[0].name == axis && t.name.starts_with("index.of_");
    let value = |t: &Tree| -> Option<u32> {
        match crate::wgsl::params::parse_ref(&t.name) {
            Some((k, form)) => crate::wgsl::params::exact_value(knobs.value(k), t.konst, form),
            None => plain_value(t),
        }
    };
    fn walk(t: &Tree, f: &dyn Fn(&Tree) -> Option<Tree>) -> Tree {
        if let Some(n) = f(t) {
            return n;
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| walk(k, f)).collect() }
    }
    let flat = |t: &Tree| -> Option<Tree> {
        if t.name != "index.add" || t.kids.len() != 2 {
            return None;
        }
        for (m, c) in [(0usize, 1usize), (1, 0)] {
            let (mul, col) = (&t.kids[m], &t.kids[c]);
            if id_of(col, "axis.x_lid") && mul.name == "index.mul" && mul.kids.len() == 2 && (0..2).any(|i| id_of(&mul.kids[i], "axis.y_lid") && value(&mul.kids[1 - i]) == Some(x)) {
                return Some(lidx.clone());
            }
        }
        None
    };
    Ok(walk(root, &flat))
}

/// Whether a kernel's threads own their rows a workgroup apart (the
/// epilogue's store is at `wid * span + lid + k * w`, `safe.row_plus_tile_s`).
fn owns_rows_strided(root: &Tree) -> bool {
    root.kids.get(2).is_some_and(|e| any_node(e, &|n| n.name.starts_with("safe.row_plus") && n.name.ends_with("_s")) && !any_node(e, &|n| n.name == "safe.thread_row"))
}

/// A workgroup array read and its array leaf: f32 (`wg.get_safe`) or vec4 (`wg.get_v_safe`).
pub(crate) fn wg_read(t: &Tree) -> Option<(&str, bool)> {
    match t.name.as_str() {
        "wg.get_safe" | "wg.get_v_safe" if t.kids.len() == 2 && t.kids[0].kids.is_empty() => Some((t.kids[0].name.as_str(), t.name == "wg.get_v_safe")),
        _ => None,
    }
}

/// What growing a strided row block needs, found in it.
#[derive(Clone, Debug)]
pub struct StridedPlan {
    pub tile: String,
    pub vec: bool,
    pub family: BTreeMap<String, usize>,
    /// Elements of a row in each private array of the flow.
    pub per_row: BTreeMap<String, u32>,
    pub fresh: BTreeMap<String, String>,
    pub threads: u32,
    /// The new row extent: a new row reads its old neighbour's cell this far on.
    pub step: u32,
}

/// The rows (0..4) a statement's reads of the row tile hold, a thread
/// owning rows `lid + w j`: a cell offset δ within the tile's row (f32: the
/// index's constant modulo the row span R; vec4 interleaved by vec4-tile:
/// the vec4's offset plus the lane times R / 4) is row δ / w.
fn strided_rows(t: &Tree, tile: &str, r: u32, w: u32, out: &mut BTreeSet<usize>, bad: &mut Option<String>) {
    for k in &t.kids {
        if let Some((a, vec)) = wg_read(k).filter(|(a, _)| *a == tile) {
            let Some(x) = k.kids[1].kids.first() else {
                *bad = Some(format!("a read of {a} without an index"));
                return;
            };
            let mut terms = Vec::new();
            add_terms(x, &mut terms);
            let (mut constant, mut lids) = (0u32, 0usize);
            for term in &terms {
                if let Some(v) = plain_value(term) {
                    constant += v;
                } else if is_row_lid(term) {
                    lids += 1;
                } else if any_node(term, &is_row_lid) {
                    lids += 2;
                }
            }
            if lids != 1 {
                *bad = Some(format!("a read of {a} is not at the thread's row id plus a constant"));
                return;
            }
            let delta = if vec {
                let Some(lane) = t.name.strip_prefix("shape.access_").and_then(|j| j.parse::<u32>().ok()) else {
                    *bad = Some(format!("a read of {a} other than as one lane"));
                    return;
                };
                let s = r / 4;
                let off = constant % s;
                if off + w > s {
                    *bad = Some(format!("a read of {a} at vec4 offset {off} leaves no room for the new row"));
                    return;
                }
                off + lane * s
            } else {
                constant % r
            };
            if delta % w != 0 || delta / w >= 4 {
                *bad = Some(format!("a read of {a} at cell {delta} is not one of the thread's 4 rows {w} apart"));
                return;
            }
            out.insert((delta / w) as usize);
            continue;
        }
        strided_rows(k, tile, r, w, out, bad);
        if bad.is_some() {
            return;
        }
    }
}

/// THE PRECONDITIONS of growing a strided row block 4 -> 8 at a constant
/// workgroup tile (the row extent halved, a thread then owning rows
/// `lid + (w / 2) j'`, j' < 8): the threads own their rows w apart; one
/// array (f32, or vec4 interleaved by vec4-tile) is read at the thread's
/// row id, nowhere else in the body; each accumulation reading it holds one
/// row; every storage of the flow belongs to one row, a private array's
/// elements a row at a time; the flat thread index is used only in loops
/// striding by the threads.
pub fn grow_strided_plan(root: &Tree, kingdom: &WgslKingdom) -> Result<StridedPlan, String> {
    use crate::wgsl::params::{Knob, Knobs};
    let flat = flat_thread_ids(root, kingdom)?;
    let plain = crate::wgsl::params::to_literals(&flat, kingdom)?;
    let knobs = Knobs::of_root(&plain)?;
    if knobs.grid.by != 4 {
        return Err(format!("the row block is {}, not 4", knobs.grid.by));
    }
    if !owns_rows_strided(&plain) {
        return Err("the threads do not own their rows a workgroup apart".into());
    }
    let (w, r, threads) = (knobs.value(Knob::Wrow), knobs.value(Knob::RowSpan), knobs.value(Knob::Threads));
    if w < 2 || w % 2 != 0 || r != 4 * w {
        return Err(format!("the row extent {w} (span {r}) does not halve"));
    }
    // The row tile: the one array read at the thread's row id in the body.
    let mut tiles = BTreeSet::new();
    fn find(t: &Tree, out: &mut BTreeSet<(String, bool)>) {
        if let Some((a, vec)) = wg_read(t) {
            if any_node(&t.kids[1], &is_row_lid) {
                out.insert((a.to_string(), vec));
            }
        }
        t.kids.iter().for_each(|k| find(k, out));
    }
    find(&plain.kids[1], &mut tiles);
    let (tile, vec) = match tiles.into_iter().collect::<Vec<_>>().as_slice() {
        [t] => t.clone(),
        other => return Err(format!("{} arrays are read at the thread's row id, not one", other.len())),
    };
    fn row_lid_outside(t: &Tree, tile: &str) -> bool {
        if wg_read(t).is_some_and(|(a, _)| a == tile) {
            return false;
        }
        is_row_lid(t) || t.kids.iter().any(|k| row_lid_outside(k, tile))
    }
    if row_lid_outside(&plain.kids[1], &tile) {
        return Err("the thread's row id is used in the body outside the row tile's reads".into());
    }
    let stmts = kernel_statements(&plain);
    let mut family: BTreeMap<String, usize> = BTreeMap::new();
    for s in stmts.iter().filter(|s| !is_compound(&s.tree)) {
        let (mut rows, mut bad) = (BTreeSet::new(), None);
        strided_rows(&s.tree, &tile, r, w, &mut rows, &mut bad);
        if let Some(b) = bad {
            return Err(format!("{}: {b}", s.tree.name));
        }
        match rows.into_iter().collect::<Vec<_>>().as_slice() {
            [] => {}
            [row] => {
                let key = target_key(&s.tree).filter(|k| k.starts_with("reg.")).ok_or_else(|| format!("{} reads the row tile and assigns no register", s.tree.name))?;
                if family.insert(key.clone(), *row).is_some_and(|x| x != *row) {
                    return Err(format!("{key} accumulates two rows"));
                }
            }
            more => return Err(format!("{} reads rows {more:?}", s.tree.name)),
        }
    }
    if family.is_empty() {
        return Err("no accumulation reads the row tile".into());
    }
    close_flow(&stmts, &mut family)?;
    // A private array of the flow holds its rows a row at a time; every
    // literal store into it is the flow's; a store at a computed index is a
    // zero-fill outside every loop (the array is zero then anyway).
    let mut per_row: BTreeMap<String, u32> = BTreeMap::new();
    let arrays: BTreeSet<String> = family.keys().filter_map(|k| k.split_once('[').map(|(a, _)| a.to_string())).collect();
    for a in &arrays {
        let elems: Vec<(u32, usize)> = family.iter().filter_map(|(k, &row)| k.strip_prefix(&format!("{a}[")).and_then(|i| i.trim_end_matches(']').parse::<f32>().ok()).map(|i| (i as u32, row))).collect();
        let e = elems.iter().filter(|(_, row)| *row == 0).count() as u32;
        if e == 0 || elems.iter().any(|&(i, row)| i / e != row as u32) {
            return Err(format!("{a} does not hold its rows {e} elements at a time"));
        }
        per_row.insert(a.clone(), e);
    }
    // The zero-fill loops outside every loop: their statements' node ranges.
    let fills: Vec<(usize, usize)> = stmts.iter().filter(|s| !s.in_loop && is_counted_loop(&s.tree.name) && zeroes(&s.tree).is_some()).map(|s| (s.node, s.node + s.tree.size())).collect();
    let in_fill = |n: usize| fills.iter().any(|&(a, b)| n > a && n < b);
    for s in &stmts {
        let Some(arr) = s.tree.kids.first().filter(|_| safe_base(&s.tree.name) == "parr.set_safe").map(|l| l.name.clone()) else { continue };
        if !arrays.contains(&arr) {
            continue;
        }
        match target_key(&s.tree) {
            Some(k) => {
                let e = per_row[&arr];
                let i: f32 = k.split_once('[').and_then(|(_, x)| x.trim_end_matches(']').parse().ok()).unwrap_or(f32::MAX);
                let row = (i as u32 / e) as usize;
                if row >= 4 || family.get(&k).is_some_and(|&x| x != row) {
                    return Err(format!("{k} is outside the rows' flow"));
                }
                family.insert(k, row);
            }
            None => {
                if zero_store(&s.tree).is_none() || (s.in_loop && !in_fill(s.node)) {
                    return Err(format!("{arr} is stored at a computed index other than a zero-fill outside every loop"));
                }
            }
        }
    }
    // Zero-fill loops of the arrays: only outside every loop, only zeroes.
    for s in &stmts {
        if is_counted_loop(&s.tree.name) && any_node(&s.tree, &|n| safe_base(&n.name) == "parr.set_safe" && n.kids.first().is_some_and(|l| arrays.contains(&l.name))) && (s.in_loop || zeroes(&s.tree).is_none()) {
            return Err("a loop stores a flow array other than as a zero-fill outside every loop".into());
        }
    }
    let mut outside = false;
    lidx_outside_thread_loops(&plain, threads, 0, false, &mut outside);
    if outside {
        return Err("the flat thread index is used outside a loop striding by the threads, or such a loop's counter strides something else".into());
    }
    let mut used: BTreeSet<String> = leaf_positions(&plain).keys().cloned().collect();
    let mut fresh = BTreeMap::new();
    for key in family.keys().filter(|k| !k.contains('[')) {
        let stem = key.trim_end_matches(|c: char| c.is_ascii_digit());
        let f = (0..64).map(|n| format!("{stem}{n}")).find(|c| !used.contains(c) && kingdom.rows.iter().any(|row| row.name == *c)).ok_or_else(|| format!("no free register like {key}"))?;
        used.insert(f.clone());
        fresh.insert(key.clone(), f);
    }
    Ok(StridedPlan { tile, vec, family, per_row, fresh, threads, step: w / 2 })
}

/// `t` with the flow's private array elements of row j at row `to(j)`
/// (`a[j E + e]` to `a[to(j) E + e]`), its registers renamed by `regs`, and
/// (with `shift`) the row tile's reads `shift` further on.
fn restride(t: &Tree, plan: &StridedPlan, to: &dyn Fn(u32) -> u32, regs: &BTreeMap<String, String>, shift: Option<u32>) -> Tree {
    if t.kids.is_empty() {
        if let Some(n) = regs.get(&t.name) {
            return Tree { name: n.clone(), ty: t.ty, konst: t.konst, kids: Vec::new() };
        }
    }
    let mut kids: Vec<Tree> = t.kids.iter().map(|k| restride(k, plan, to, regs, shift)).collect();
    let parr = (t.name.starts_with("parr.get_") || safe_base(&t.name) == "parr.set_safe") && t.kids.len() >= 2;
    if parr {
        if let Some(&e) = plan.per_row.get(&t.kids[0].name) {
            if let Some(i) = kids[1].kids.first_mut().filter(|l| l.name.starts_with("literal.") && l.kids.is_empty()) {
                let old = i.konst as u32;
                i.konst = (to(old / e) * e + old % e) as f32;
            }
        }
    }
    if let (Some(d), Some((a, _))) = (shift, wg_read(t)) {
        if a == plan.tile {
            let idx = &mut kids[1];
            let x = idx.kids[0].clone();
            idx.kids[0] = Tree::new("index.add", x.ty, vec![x.clone(), Tree::literal("literal.index", x.ty, d as f32)]);
        }
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
}

/// The kernel with its strided row block grown 4 -> 8 and its row extent
/// halved: old row j is new row 2 j (the same cell); new row 2 j + 1 is its
/// copy over fresh registers reading the row tile half the old extent on;
/// the flow's private array elements move with their rows; the loops
/// striding the flat thread index by the threads run twice the passes.
pub fn grow_strided(root: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    use crate::wgsl::knob::{set, Site as KnobSite};
    use crate::wgsl::params::{Knob, Knobs};
    let plan = grow_strided_plan(root, kingdom)?;
    let root = flat_thread_ids(root, kingdom)?;
    let knobs = Knobs::of_root(&root)?;
    let none = BTreeMap::new();
    fn walk(t: &Tree, plan: &StridedPlan, none: &BTreeMap<String, String>, depth: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
        if is_plain_block(&t.name) {
            let mut out = Vec::new();
            for s in statements(t) {
                match target_key(&s).filter(|k| plan.family.contains_key(k)) {
                    Some(_) => {
                        out.push(restride(&s, plan, &|j| 2 * j, none, None));
                        out.push(restride(&s, plan, &|j| 2 * j + 1, &plan.fresh, Some(plan.step)));
                    }
                    None => out.push(walk(&s, plan, none, depth, kingdom)?),
                }
            }
            return rebuild(out, rung(&t.name).ok_or_else(|| format!("{} has no rung", t.name))?, kingdom);
        }
        let base = safe_base(&t.name);
        let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
        let mut kids = Vec::with_capacity(t.kids.len());
        for (i, k) in t.kids.iter().enumerate() {
            kids.push(walk(k, plan, none, depth + usize::from(looped), kingdom)?);
            if i == 0 && is_counted_loop(&t.name) && t.kids.len() >= 2 && strides_by_threads(t.kids.last().expect("a body"), depth, plan.threads) {
                kids[0].konst *= 2.0;
            }
        }
        Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids })
    }
    let mut kids = root.kids.clone();
    kids[1] = walk(&root.kids[1], &plan, &none, 0, kingdom)?;
    kids[2] = walk(&root.kids[2], &plan, &none, 0, kingdom)?;
    let t = Tree { name: root.name.clone(), ty: root.ty, konst: root.konst, kids };
    let row_site = if knobs.grid.rows_x { KnobSite::Wx } else { KnobSite::Wy };
    let t = set(&t, &row_site, knobs.value(Knob::Wrow) / 2);
    Ok(set(&t, &KnobSite::By, 8))
}

/// RULE: GROW THE ROW BLOCK (one site: the kernel). Site 0 is the
/// contiguous form (a thread's 4 rows in one vec4), site 1 the strided
/// form (a thread's rows a workgroup apart).
pub fn grow_rows_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    if let Ok(p) = grow_rows_plan(root, kingdom) {
        return vec![Site { index: 0, what: format!("rows 4 -> 8 over {}, {} storages", p.tile, p.family.len()) }];
    }
    match grow_strided_plan(root, kingdom) {
        Ok(p) => vec![Site { index: 1, what: format!("rows 4 -> 8 strided over {}, {} storages", p.tile, p.family.len()) }],
        Err(_) => Vec::new(),
    }
}

// ---- RULE: A STRIDED f32 TILE AS A vec4 TILE, INTERLEAVED ----

/// The terms of a sum, however it is nested.
pub(crate) fn add_terms<'t>(t: &'t Tree, out: &mut Vec<&'t Tree>) {
    if t.name == "index.add" && t.kids.len() == 2 {
        add_terms(&t.kids[0], out);
        add_terms(&t.kids[1], out);
    } else {
        out.push(t);
    }
}

pub(crate) fn plain_value(t: &Tree) -> Option<u32> {
    (t.kids.is_empty() && t.name.starts_with("literal.") && !t.name.starts_with("literal.p_") && t.konst >= 0.0 && t.konst.fract() == 0.0).then_some(t.konst as u32)
}

/// A read index of a strided tile decomposed: `row * R + lid + c * S`:
/// the row terms (each `e * m`, m a multiple of R), the thread's id on the
/// tile's axis, the literal rows and the lane c.
struct StridedIndex<'t> {
    rows: Vec<(&'t Tree, u32)>,
    lid: &'t Tree,
    literal_rows: u32,
    lane: u32,
    /// The vec4 offset within the quarter row that holds the cell's lane.
    off: u32,
    col: bool,
}

/// `x` as a read index of a tile row of R cells laid out as vec4s of the
/// row's quarters (cell j at vec4 `j % (R / 4)`, lane `j / (R / 4)`), when
/// it is one: `row * R + lid + c`, the thread's id on one axis (below its
/// workgroup extent `w`) plus a constant whose quarter offset leaves the
/// id inside the quarter.
fn strided_index(x: &Tree, r: [u32; 2], w: [u32; 2]) -> Option<StridedIndex<'_>> {
    let mut terms = Vec::new();
    add_terms(x, &mut terms);
    let (mut rows, mut lid, mut constant) = (Vec::new(), None, 0u32);
    for t in terms {
        if let Some(v) = plain_value(t) {
            constant += v;
        } else if t.name == "index.mul" && t.kids.len() == 2 {
            let (e, m) = match (plain_value(&t.kids[0]), plain_value(&t.kids[1])) {
                (_, Some(m)) => (&t.kids[0], m),
                (Some(m), None) => (&t.kids[1], m),
                (None, None) => return None,
            };
            rows.push((e, m));
        } else if is_row_lid(t) || (t.name == "index.of_col" && t.kids.first().is_some_and(|a| a.name.starts_with("axis.") && a.name.ends_with("_lid"))) {
            if lid.replace(t).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }
    let lid = lid?;
    let col = lid.name == "index.of_col";
    let (r, w) = if col { (r[0], w[0]) } else { (r[1], w[1]) };
    let q = r / 4;
    if r % 4 != 0 || rows.iter().any(|&(_, m)| m % r != 0) {
        return None;
    }
    let (literal_rows, rem) = (constant / r, constant % r);
    let (off, lane) = (rem % q, rem / q);
    (off + w <= q && lane < 4).then_some(StridedIndex { rows, lid, literal_rows, lane, off, col })
}

/// What vectorising one f32 tile needs.
#[derive(Clone, Debug)]
pub struct VecTilePlan {
    pub array: String,
    pub len: u32,
    /// The row width R on the tile's axis and S = R / 4, the vec4s of a row.
    pub r: u32,
    pub s: u32,
    pub col: bool,
}

/// THE READ PRECONDITIONS of laying the f32 tile `array` out as vec4s of
/// its rows' quarters (cell j of a row of R at vec4 `j % (R / 4)`, lane
/// `j / (R / 4)`): every read of the array is `row * R + lid + c` over the
/// thread's id on one axis, c keeping the id inside its quarter (a thread
/// owning 4 cells a workgroup apart reads one vec4; 8 cells, two); every
/// other use is a scalar store; R and the length are powers of two.
pub fn vec_tile_plan(plain: &Tree, array: &str) -> Result<VecTilePlan, String> {
    use crate::wgsl::params::{Knob, Knobs};
    let knobs = Knobs::of_root(plain)?;
    let (rc, sc, rr, sr) = (knobs.value(Knob::ColSpan), knobs.value(Knob::Wcol), knobs.value(Knob::RowSpan), knobs.value(Knob::Wrow));
    let (mut reads, mut col_axis, mut bad) = (0usize, BTreeSet::new(), None::<String>);
    fn walk(t: &Tree, array: &str, r: [u32; 2], s: [u32; 2], reads: &mut usize, axes: &mut BTreeSet<bool>, bad: &mut Option<String>) {
        let is_array = |k: &Tree| k.kids.is_empty() && k.name == array;
        if t.name == "wg.get_safe" && t.kids.len() == 2 && is_array(&t.kids[0]) {
            let idx = t.kids[1].kids.first().filter(|_| t.kids[1].name == "safe.wg_of");
            match idx.and_then(|x| strided_index(x, r, s)) {
                Some(si) => {
                    *reads += 1;
                    axes.insert(si.col);
                }
                None => *bad = Some(format!("a read of {array} is not row * R + lid + c * S: {}", t.kids[1].to_sexpr())),
            }
            return;
        }
        if safe_base(&t.name) == "wg.set_safe" && t.name.starts_with("wg.set_") && !t.name.starts_with("wg.set_v") && t.kids.first().is_some_and(is_array) {
            if t.kids.get(1).is_none_or(|i| i.name != "safe.wg_of") {
                *bad = Some(format!("a store to {array} not through its index row"));
            }
            for k in &t.kids[1..] {
                walk(k, array, r, s, reads, axes, bad);
            }
            return;
        }
        if t.name.starts_with("wg.decl_") {
            return;
        }
        if t.kids.iter().any(is_array) {
            *bad = Some(format!("{} uses {array}", t.name));
            return;
        }
        t.kids.iter().for_each(|k| walk(k, array, r, s, reads, axes, bad));
    }
    walk(plain, array, [rc, rr], [sc, sr], &mut reads, &mut col_axis, &mut bad);
    if let Some(b) = bad {
        return Err(b);
    }
    if reads == 0 {
        return Err(format!("{array} is never read"));
    }
    let col = match col_axis.into_iter().collect::<Vec<_>>().as_slice() {
        [c] => *c,
        _ => return Err(format!("{array} is read on both axes")),
    };
    let (r, w) = if col { (rc, sc) } else { (rr, sr) };
    let s = r / 4;
    if !r.is_power_of_two() || r < 4 || s < w {
        return Err(format!("the tile's row of {r} cells has no quarter of {w} or more"));
    }
    // The array's length: declared, or the default of an f32 array.
    let mut len = 2048u32;
    fn decl_len(t: &Tree, array: &str, len: &mut u32) {
        if t.name == "wg.decl_f" && t.kids.len() == 2 && t.kids[0].name == array {
            if let Some(v) = plain_value(&t.kids[1]) {
                *len = v;
            }
        }
        t.kids.iter().for_each(|k| decl_len(k, array, len));
    }
    decl_len(&plain.kids[0], array, &mut len);
    if !len.is_power_of_two() || len < 4 * r {
        return Err(format!("{array}'s length {len} is not a power of two of whole rows"));
    }
    Ok(VecTilePlan { array: array.to_string(), len, r, s, col })
}

/// A scalar store into an f32 workgroup array: the array leaf, the flat
/// index F under its index row, the value.
fn f32_store(t: &Tree) -> Option<(&str, &Tree, &Tree)> {
    let is = t.name.starts_with("wg.set_") && !t.name.starts_with("wg.set_v") && t.kids.len() == 3 && t.kids[0].kids.is_empty() && ["wg.a", "wg.b", "wg.c", "wg.d"].contains(&t.kids[0].name.as_str()) && t.kids[1].name == "safe.wg_of" && t.kids[1].kids.len() == 1;
    is.then(|| (t.kids[0].name.as_str(), &t.kids[1].kids[0], &t.kids[2]))
}

/// `t` with every subtree equal to `from` replaced by `to`.
fn substitute(t: &Tree, from: &str, to: &Tree) -> Tree {
    if t.to_sexpr() == from {
        return to.clone();
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| substitute(k, from, to)).collect() }
}

/// A loop that fills f32 tiles, found in the kernel (plain literals): its
/// pre-order node, its tile depth, its arrays, R and S of their reads.
#[derive(Clone, Debug)]
pub struct LoaderPlan {
    pub node: usize,
    pub depth: usize,
    pub arrays: Vec<(String, String)>,
    pub r: u32,
    pub s: u32,
    pub passes: u32,
}

/// Every counted loop (tile or unroll) whose body stores f32 tiles, with
/// its pre-order node and tile depth.
fn loader_loops(t: &Tree) -> Vec<(usize, usize, Tree)> {
    fn walk(t: &Tree, counter: &mut usize, depth: usize, out: &mut Vec<(usize, usize, Tree)>) {
        let me = *counter;
        *counter += 1;
        let base = safe_base(&t.name);
        let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
        if matches!(base, "loop.tile_bounded" | "loop.unroll") && t.kids.len() == 2 && statements(&t.kids[1]).iter().any(|s| f32_store(s).is_some()) {
            out.push((me, depth, t.clone()));
        }
        for k in &t.kids {
            walk(k, counter, depth + usize::from(looped), out);
        }
    }
    let mut out = Vec::new();
    walk(t, &mut 0, 0, &mut out);
    out
}

/// THE PRECONDITIONS of vectorising the f32 tiles a loader loop fills:
/// the tiles' stores end the loop's body, at one flat index F = G +
/// threads * k (k the loop's counter, used nowhere else in the body); the
/// passes divide by 4; every tile is read only as `row * R + lid + c`
/// with one R for all; a vec4 array is free for each; the registers
/// the body writes are used nowhere outside the loop.
pub fn loader_plan(plain: &Tree, node: usize, depth: usize, lp: &Tree) -> Result<LoaderPlan, String> {
    let body = statements(&lp.kids[1]);
    let first = body.iter().position(|s| f32_store(s).is_some()).ok_or("no store")?;
    if body[first..].iter().any(|s| f32_store(s).is_none()) {
        return Err("the tile stores do not end the loader's body".into());
    }
    let f = f32_store(&body[first]).map(|(_, f, _)| f.to_sexpr()).ok_or("no store")?;
    if body[first..].iter().any(|s| f32_store(s).map(|(_, x, _)| x.to_sexpr()) != Some(f.clone())) {
        return Err("the tile stores are at different flat indices".into());
    }
    let counter = tile_counter_name(depth);
    let threads = crate::wgsl::params::Knobs::of_root(plain)?.value(crate::wgsl::params::Knob::Threads);
    let fx = f32_store(&body[first]).map(|(_, x, _)| x.clone()).ok_or("no store")?;
    let mut terms = Vec::new();
    add_terms(&fx, &mut terms);
    let strides = terms.iter().filter(|t| t.name == "index.mul" && t.kids.len() == 2 && (0..2).any(|i| plain_value(&t.kids[i]) == Some(threads) && any_node(&t.kids[1 - i], &|m| m.name == counter))).count();
    if strides != 1 || terms.iter().filter(|t| any_node(t, &|m| m.name == counter)).count() != 1 {
        return Err("the flat index is not G + threads * k".into());
    }
    let marker = Tree::new("fallback", fx.ty, vec![]);
    if body.iter().any(|s| any_node(&substitute(s, &f, &marker), &|m| m.name == counter)) {
        return Err("the loader's counter is used outside its flat index".into());
    }
    let passes = plain_value(&lp.kids[0]).ok_or("the loader's count is not a literal")?;
    if passes % 4 != 0 {
        return Err(format!("{passes} passes do not divide by 4"));
    }
    let mut arrays = Vec::new();
    let (mut rs, mut used) = (BTreeSet::new(), leaf_positions(plain));
    for s in &body[first..] {
        let a = f32_store(s).map(|(a, _, _)| a.to_string()).ok_or("no store")?;
        if arrays.iter().any(|(x, _)| *x == a) {
            return Err(format!("{a} is stored twice in the loader"));
        }
        let p = vec_tile_plan(plain, &a)?;
        rs.insert((p.r, p.s));
        let vec = ["wg.va", "wg.vb", "wg.vc", "wg.vd"].into_iter().find(|v| !used.contains_key(*v)).ok_or("no free vec4 workgroup array")?;
        used.insert(vec.to_string(), Vec::new());
        arrays.push((a, vec.to_string()));
    }
    let (r, s) = match rs.into_iter().collect::<Vec<_>>().as_slice() {
        [x] => *x,
        _ => return Err("the tiles' rows differ".into()),
    };
    // The body's registers live only in the loop.
    let size = lp.size();
    let positions = leaf_positions(plain);
    for st in &body {
        if let Some(k) = target_key(st).filter(|k| k.starts_with("reg.")) {
            if positions.get(&k).is_some_and(|ps| ps.iter().any(|&p| p < node || p >= node + size)) {
                return Err(format!("{k} is used outside the loader"));
            }
        }
    }
    Ok(LoaderPlan { node, depth, arrays, r, s, passes })
}

/// The kernel with the f32 tiles of one loader loop stored as interleaved
/// vec4 tiles: cell j of a row at vec4 `j % S`, lane `j / S`, so a
/// thread's 4 cells (S apart) are one vec4. The loader counts vec4s (a
/// quarter of the passes): its body runs once per lane c with the flat
/// index replaced by lane c's cell of vec4 F and fresh registers, and one
/// thread stores each vec4 whole (lane stores by four threads raced: all 62
/// children of the lane-store version computed wrong values). Every read
/// is a lane of the vec4 at `row * S + lid` (one index for the thread's
/// four cells, which the compiler loads once).
pub fn vec_tile(root: &Tree, loader: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = crate::wgsl::params::to_literals(root, kingdom)?;
    let (node, depth, lp) = loader_loops(&plain).into_iter().find(|(n, _, _)| *n == loader).ok_or("our software has a bug: the loader is gone")?;
    let plan = loader_plan(&plain, node, depth, &lp)?;
    let index = dual_ty(kingdom, "u32.index")?;
    let real = dual_ty(kingdom, "f32.real")?;
    let vindex = dual_ty(kingdom, "u32.wgvindex")?;
    let varr = dual_ty(kingdom, "wgmem<vec4<f32>>.real")?;
    let (v2, v3, v4) = (dual_ty(kingdom, "vec2<f32>.real")?, dual_ty(kingdom, "vec3<f32>.real")?, dual_ty(kingdom, "vec4<f32>.real")?);
    let lit = |v: u32| Tree::literal("literal.index", index, v as f32);
    let op = |name: &str, a: Tree, b: Tree| Tree::new(name, index, vec![a, b]);
    let (r, s) = (plan.r, plan.s);
    let knobs = crate::wgsl::params::Knobs::of_root(&plain)?;
    let extents = [knobs.value(crate::wgsl::params::Knob::Wcol), knobs.value(crate::wgsl::params::Knob::Wrow)];
    let vec_of = |a: &str| plan.arrays.iter().find(|(x, _)| x == a).map(|(_, v)| v.clone());
    // The new loader.
    let body = statements(&lp.kids[1]);
    let first = body.iter().position(|x| f32_store(x).is_some()).ok_or("no store")?;
    let f = f32_store(&body[first]).map(|(_, x, _)| x.clone()).ok_or("no store")?;
    let fs = f.to_sexpr();
    let mut used: BTreeSet<String> = leaf_positions(&plain).keys().cloned().collect();
    let written: BTreeSet<String> = body.iter().filter_map(target_key).filter(|k| k.starts_with("reg.")).collect();
    let mut new_body = Vec::new();
    let mut lanes: BTreeMap<String, Vec<Tree>> = BTreeMap::new();
    for c in 0..4u32 {
        let mut map = BTreeMap::new();
        if c > 0 {
            for w in &written {
                let stem = w.trim_end_matches(|ch: char| ch.is_ascii_digit());
                let fresh = (0..64).map(|n| format!("{stem}{n}")).find(|x| !used.contains(x) && kingdom.rows.iter().any(|row| row.name == *x)).ok_or_else(|| format!("no free register like {w}"))?;
                used.insert(fresh.clone());
                map.insert(w.clone(), fresh);
            }
        }
        // Lane c's cell of vec4 F: (F / S) * R + F % S + c * S.
        let cell = op("index.add", op("index.add", op("index.mul", op("index.shr", f.clone(), lit(s.trailing_zeros())), lit(r)), op("index.and", f.clone(), lit(s - 1))), lit(c * s));
        for st in &body {
            let moved = rename(&substitute(st, &fs, &cell), &map);
            match f32_store(st) {
                Some((a, _, _)) => {
                    let v = rename(&substitute(&st.kids[2], &fs, &cell), &map);
                    lanes.entry(a.to_string()).or_default().push(v);
                }
                None => new_body.push(moved),
            }
        }
    }
    let body_rung = rung(&lp.kids[1].name).ok_or("the loader's body has no rung")?.to_string();
    for (a, vals) in &lanes {
        let [x, y, z, w] = vals.as_slice() else { return Err("our software has a bug: not four lanes".into()) };
        let vec = vec_of(a).ok_or("our software has a bug: no vec4 tile")?;
        let composed = Tree::new("shape.compose4", v4, vec![Tree::new("shape.compose3", v3, vec![Tree::new("shape.compose2", v2, vec![x.clone(), y.clone()]), z.clone()]), w.clone()]);
        let st_ty = dual_ty(kingdom, &format!("stmt.{body_rung}"))?;
        new_body.push(Tree::new(&format!("wg.set_v_{body_rung}"), st_ty, vec![Tree::new(&vec, varr, vec![]), Tree::new("safe.wgv_of", vindex, vec![f.clone()]), composed]));
    }
    let new_loop = Tree { name: lp.name.clone(), ty: lp.ty, konst: lp.konst, kids: vec![Tree::literal(&lp.kids[0].name, lp.kids[0].ty, (plan.passes / 4) as f32), rebuild(new_body, &body_rung, kingdom)?] };
    // The loader replaced, the reads and declarations rewritten.
    fn walk(t: &Tree, counter: &mut usize, plan: &LoaderPlan, new_loop: &Tree, f: &dyn Fn(&Tree) -> Option<Tree>) -> Tree {
        let me = *counter;
        *counter += 1;
        if me == plan.node {
            *counter += t.size() - 1;
            return new_loop.clone();
        }
        if let Some(n) = f(t) {
            *counter += t.size() - 1;
            return n;
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| walk(k, counter, plan, new_loop, f)).collect() }
    }
    let rewrite = |t: &Tree| -> Option<Tree> {
        let is_tile = |k: &Tree| k.kids.is_empty() && vec_of(&k.name).is_some();
        if t.name == "wg.get_safe" && t.kids.len() == 2 && is_tile(&t.kids[0]) {
            let si = strided_index(&t.kids[1].kids[0], [r, r], extents)?;
            let mut y = si.lid.clone();
            for (e, m) in &si.rows {
                y = op("index.add", op("index.mul", (*e).clone(), lit(m / 4)), y);
            }
            if si.literal_rows * s + si.off > 0 {
                y = op("index.add", y, lit(si.literal_rows * s + si.off));
            }
            let read = Tree::new("wg.get_v_safe", v4, vec![Tree::new(&vec_of(&t.kids[0].name)?, varr, vec![]), Tree::new("safe.wgv_of", vindex, vec![y])]);
            return Some(Tree::new(&format!("shape.access_{}", si.lane), real, vec![read]));
        }
        if t.name == "wg.decl_f" && t.kids.len() == 2 && is_tile(&t.kids[0]) {
            let len = plain_value(&t.kids[1])?;
            return Some(Tree::new("wg.decl_v4", t.ty, vec![Tree::new(&vec_of(&t.kids[0].name)?, varr, vec![]), Tree::literal("literal.wglen", t.kids[1].ty, (len / 4) as f32)]));
        }
        None
    };
    Ok(walk(&plain, &mut 0, &plan, &new_loop, &rewrite))
}

/// RULE: VECTORISE THE STRIDED f32 TILES OF A LOADER (one site per loader loop).
pub fn vec_tile_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = crate::wgsl::params::to_literals(root, kingdom) else { return Vec::new() };
    loader_loops(&plain)
        .into_iter()
        .filter_map(|(n, d, lp)| loader_plan(&plain, n, d, &lp).ok().map(|p| Site { index: n, what: format!("{} as vec4 (R {}, S {}, {} passes)", p.arrays.iter().map(|(a, v)| format!("{a}->{v}")).collect::<Vec<_>>().join(" "), p.r, p.s, p.passes) }))
        .collect()
}

/// Why no loader's tiles vectorise.
fn vec_tile_why(root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
    let plain = crate::wgsl::params::to_literals(root, kingdom).ok()?;
    let loops = loader_loops(&plain);
    if loops.is_empty() {
        return Some("no counted loop stores an f32 tile".into());
    }
    let why: Vec<String> = loops.iter().filter_map(|(n, d, lp)| loader_plan(&plain, *n, *d, lp).err()).collect();
    (why.len() == loops.len()).then(|| why.join("; "))
}

// ---- RULES: UNROLL K BY TWO, AND UNDO IT ----

/// A workgroup tile read's K step, its index over a counted loop's counter
/// `c`: `(m * k) * S + j S + rest` (m = 2 in an unrolled loop, 1 not), the
/// K row being `m k + j`. Returns (m, S, j) when every term with the counter
/// is that product and the constant's K part is j S.
pub(crate) fn k_step(index: &Tree, counter: &str) -> Option<(u32, u32, u32)> {
    let mut terms = Vec::new();
    add_terms(index, &mut terms);
    let (mut step, mut constant) = (None, 0u32);
    for t in &terms {
        if let Some(v) = plain_value(t) {
            constant += v;
            continue;
        }
        if !any_node(t, &|n| n.name == counter) {
            continue;
        }
        // (m * k) * S, or k * S.
        if t.name != "index.mul" || t.kids.len() != 2 || step.is_some() {
            return None;
        }
        let (inner, s) = match (plain_value(&t.kids[0]), plain_value(&t.kids[1])) {
            (_, Some(s)) => (&t.kids[0], s),
            (Some(s), None) => (&t.kids[1], s),
            _ => return None,
        };
        let m = if inner.name.starts_with("safe.tile_index") && any_node(inner, &|n| n.name == counter) {
            1
        } else if inner.name == "index.mul" && inner.kids.len() == 2 {
            let (a, b) = (&inner.kids[0], &inner.kids[1]);
            match (plain_value(a), plain_value(b)) {
                (Some(m), None) if b.name.starts_with("safe.tile_index") => m,
                (None, Some(m)) if a.name.starts_with("safe.tile_index") => m,
                _ => return None,
            }
        } else {
            return None;
        };
        step = Some((m, s));
    }
    let (m, s) = step?;
    // The K part of the constant: whole K rows, below the unroll.
    let j = constant / s;
    (s > 0 && j < m.max(1) + u32::from(m == 1)).then_some((m, s, j))
}

/// The K steps of the workgroup tile reads in a statement, over the counter.
pub(crate) fn k_steps(t: &Tree, counter: &str, out: &mut BTreeSet<(u32, u32, u32)>, bad: &mut bool) {
    if let Some((_, _)) = wg_read(t) {
        match t.kids[1].kids.first().and_then(|x| k_step(x, counter)) {
            Some(k) => {
                out.insert(k);
            }
            None => *bad = true,
        }
        return;
    }
    t.kids.iter().for_each(|k| k_steps(k, counter, out, bad));
}

/// The counted loops that step K through the workgroup tiles: (pre-order
/// node, tile depth, the loop), each body's tile reads all K steps of its counter.
pub(crate) fn k_loops(root: &Tree) -> Vec<(usize, usize, Tree)> {
    fn walk(t: &Tree, counter: &mut usize, depth: usize, out: &mut Vec<(usize, usize, Tree)>) {
        let me = *counter;
        *counter += 1;
        let base = safe_base(&t.name);
        let looped = base.starts_with("loop.") && !base.starts_with("loop.items");
        if matches!(base, "loop.tile_bounded") && t.kids.len() == 2 && any_node(&t.kids[1], &|n| wg_read(n).is_some()) && !any_node(&t.kids[1], &|n| is_wg_write(&n.name)) {
            out.push((me, depth, t.clone()));
        }
        for k in &t.kids {
            walk(k, counter, depth + usize::from(looped), out);
        }
    }
    let mut out = Vec::new();
    walk(root, &mut 0, 0, &mut out);
    out
}

/// A workgroup array write (any `wg.set`).
pub(crate) fn is_wg_write(name: &str) -> bool {
    name.starts_with("wg.set")
}

/// What un-unrolling or unrolling a K loop needs.
#[derive(Clone, Debug)]
pub struct KPlan {
    pub node: usize,
    pub depth: usize,
    pub stride: u32,
    /// Un-unroll: the statements of K row 2 k + 1 to drop (their indices in the body).
    pub drop: BTreeSet<usize>,
}

/// THE PRECONDITIONS of undoing a K unroll by two in the loop at `node`:
/// every statement of its body is an accumulation reading the tiles at one
/// K row, `2 k` or `2 k + 1` (in the same stride); a statement of row
/// `2 k + 1` writes a register nothing else writes but zero stores (it
/// stays zero once its statements are gone, so every sum reading it is
/// unchanged); the loop's count is a literal or a reference that doubles.
pub fn ununroll_plan(plain: &Tree, node: usize, depth: usize, lp: &Tree) -> Result<KPlan, String> {
    let counter = tile_counter_name(depth);
    let body = statements(&lp.kids[1]);
    let (mut drop, mut stride) = (BTreeSet::new(), None);
    for (i, st) in body.iter().enumerate() {
        let (mut ks, mut bad) = (BTreeSet::new(), false);
        k_steps(st, counter, &mut ks, &mut bad);
        if bad || is_compound(st) {
            return Err(format!("{} in the K loop is not an accumulation over K steps of its counter", st.name));
        }
        let steps: Vec<(u32, u32, u32)> = ks.into_iter().collect();
        match steps.as_slice() {
            [] => return Err(format!("{} in the K loop reads no tile", st.name)),
            [(2, s, j)] if *j < 2 && stride.is_none_or(|x| x == *s) => {
                stride = Some(*s);
                if *j == 1 {
                    drop.insert(i);
                }
            }
            other => return Err(format!("{} reads K steps {other:?}, not one row of an unroll by two", st.name)),
        }
        if target_key(st).filter(|k| k.starts_with("reg.")).is_none() {
            return Err(format!("{} in the K loop assigns no register", st.name));
        }
    }
    if drop.is_empty() || drop.len() * 2 != body.len() {
        return Err(format!("{} of {} statements read row 2 k + 1, not half", drop.len(), body.len()));
    }
    // The dropped registers: written elsewhere only by zero stores.
    let dropped: BTreeSet<String> = drop.iter().filter_map(|&i| target_key(&body[i])).collect();
    let loop_end = node + lp.size();
    for s in kernel_statements(plain) {
        if s.node > node && s.node < loop_end {
            continue;
        }
        if let Some(k) = target_key(&s.tree).filter(|k| dropped.contains(k)) {
            if zero_store(&s.tree).is_none() {
                return Err(format!("{k} is written outside the K loop other than with zero"));
            }
        }
    }
    Ok(KPlan { node, depth, stride: stride.unwrap_or(1), drop })
}

/// THE PRECONDITIONS of unrolling a K loop by `factor` (2 or 4): every
/// statement of its body is an accumulation `r = r + E` reading the tiles
/// at K row `k`; the loop's count is a multiple of the factor.
pub fn unroll_plan(node: usize, depth: usize, lp: &Tree, factor: u32) -> Result<KPlan, String> {
    let counter = tile_counter_name(depth);
    let count = plain_value(&lp.kids[0]).ok_or("the K loop's count is not a literal")?;
    if count % factor != 0 || count < factor {
        return Err(format!("the K loop's count {count} is not a multiple of {factor}"));
    }
    let mut stride = None;
    for st in statements(&lp.kids[1]) {
        let (mut ks, mut bad) = (BTreeSet::new(), false);
        k_steps(&st, counter, &mut ks, &mut bad);
        let steps: Vec<(u32, u32, u32)> = ks.into_iter().collect();
        match (bad, steps.as_slice()) {
            (false, [(1, s, 0)]) if stride.is_none_or(|x| x == *s) => stride = Some(*s),
            _ => return Err(format!("{} in the K loop is not an accumulation at K row k", st.name)),
        }
        let key = target_key(&st).filter(|k| k.starts_with("reg.")).ok_or_else(|| format!("{} in the K loop assigns no register", st.name))?;
        let mut reads = BTreeSet::new();
        assigned_values(&st).iter().for_each(|v| read_keys(v, &mut reads));
        if !(st.name.starts_with("reg.set_") && st.kids.len() == 2 && st.kids[1].name == "arith.add" && st.kids[1].kids.iter().any(|k| k.name.starts_with("reg.get") && k.kids.first().is_some_and(|l| l.name == key))) {
            return Err(format!("{key} is not accumulated as {key} + E"));
        }
    }
    Ok(KPlan { node, depth, stride: stride.ok_or("an empty K loop")?, drop: BTreeSet::new() })
}

/// `t` with the counter `c`'s K step rewritten: `(m * k) * S` to
/// `(to_m * k) * S` (to_m 1: `k * S`), plus `extra` K rows.
pub(crate) fn re_k(t: &Tree, counter: &str, to_m: u32, extra: u32, kingdom_index: crate::geneframe::Ty) -> Tree {
    if t.name == "index.mul" && t.kids.len() == 2 && any_node(t, &|n| n.name == counter) {
        let (inner_i, s) = match (plain_value(&t.kids[0]), plain_value(&t.kids[1])) {
            (_, Some(s)) => (0usize, s),
            (Some(s), None) => (1usize, s),
            _ => return t.clone(),
        };
        let inner = &t.kids[inner_i];
        let tile_index = if inner.name.starts_with("safe.tile_index") { inner.clone() } else { inner.kids.iter().find(|k| k.name.starts_with("safe.tile_index")).cloned().unwrap_or_else(|| inner.clone()) };
        let k = if to_m == 1 { tile_index } else { Tree::new("index.mul", kingdom_index, vec![Tree::literal("literal.index", kingdom_index, to_m as f32), tile_index]) };
        let base = Tree::new("index.mul", kingdom_index, vec![k, Tree::literal("literal.index", kingdom_index, s as f32)]);
        return if extra > 0 { Tree::new("index.add", kingdom_index, vec![base, Tree::literal("literal.index", kingdom_index, (extra * s) as f32)]) } else { base };
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| re_k(k, counter, to_m, extra, kingdom_index)).collect() }
}

/// `t` with the reads' constant K rows (`j S` in an index) removed: what
/// is left of a row-2k+1 statement's indices once re_k adds the K row back.
pub(crate) fn strip_k_constant(t: &Tree, s: u32) -> Tree {
    if wg_read(t).is_some() {
        let mut kids = t.kids.clone();
        if let Some(x) = kids[1].kids.first().cloned() {
            let mut terms = Vec::new();
            add_terms(&x, &mut terms);
            let constant: u32 = terms.iter().filter_map(|t| plain_value(t)).sum();
            let keep = constant % s;
            let rest: Vec<Tree> = terms.iter().filter(|t| plain_value(t).is_none()).map(|t| (*t).clone()).collect();
            let mut acc = rest.into_iter().reduce(|a, b| Tree::new("index.add", a.ty, vec![a, b])).expect("a term with the counter");
            if keep > 0 {
                acc = Tree::new("index.add", acc.ty, vec![acc.clone(), Tree::literal("literal.index", acc.ty, keep as f32)]);
            }
            kids[1].kids[0] = acc;
        }
        return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids };
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| strip_k_constant(k, s)).collect() }
}

/// The kernel with the K loop at `node` rewritten by `body` and its count scaled.
pub(crate) fn rewrite_k_loop(plain: &Tree, node: usize, body: Vec<Tree>, count: u32, kingdom: &WgslKingdom) -> Result<Tree, String> {
    fn walk(t: &Tree, counter: &mut usize, node: usize, f: &dyn Fn(&Tree) -> Result<Tree, String>) -> Result<Tree, String> {
        let me = *counter;
        *counter += 1;
        if me == node {
            *counter += t.size() - 1;
            return f(t);
        }
        Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| walk(k, counter, node, f)).collect::<Result<_, _>>()? })
    }
    let f = |lp: &Tree| -> Result<Tree, String> {
        let r = rung(&lp.kids[1].name).ok_or("the K loop's body has no rung")?;
        Ok(Tree { name: lp.name.clone(), ty: lp.ty, konst: lp.konst, kids: vec![Tree::literal(&lp.kids[0].name, lp.kids[0].ty, count as f32), rebuild(body.clone(), r, kingdom)?] })
    };
    walk(plain, &mut 0, node, &f)
}

/// UN-UNROLL K: the loop at `node` steps one K row a pass, its statements
/// of row 2 k + 1 gone (their registers stay zero), twice the passes.
pub fn ununroll_k(root: &Tree, node: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = crate::wgsl::params::to_literals(root, kingdom)?;
    let (n, depth, lp) = k_loops(&plain).into_iter().find(|(n, _, _)| *n == node).ok_or("our software has a bug: the K loop is gone")?;
    let plan = ununroll_plan(&plain, n, depth, &lp)?;
    let index = dual_ty(kingdom, "u32.index")?;
    let counter = tile_counter_name(depth);
    let body: Vec<Tree> = statements(&lp.kids[1]).iter().enumerate().filter(|(i, _)| !plan.drop.contains(i)).map(|(_, st)| re_k(st, counter, 1, 0, index)).collect();
    let count = plain_value(&lp.kids[0]).ok_or("the K loop's count is not a literal")? * 2;
    rewrite_k_loop(&plain, n, body, count, kingdom)
}

/// UNROLL K: the loop at `node` steps two K rows a pass: each accumulation
/// reads row 2 k, then again row 2 k + 1 into the same register; half the passes.
pub fn unroll_k(root: &Tree, node: usize, kingdom: &WgslKingdom) -> Result<Tree, String> {
    unroll_k_by(root, node, 2, kingdom)
}

/// UNROLL K BY `factor`: the loop at `node` steps `factor` K rows a pass:
/// each accumulation reads row f k, then rows f k + 1 .. f k + f - 1 into
/// the same register, in that order (each register's sum is added in the
/// order the rolled loop adds it); 1 / factor of the passes.
pub fn unroll_k_by(root: &Tree, node: usize, factor: u32, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let plain = crate::wgsl::params::to_literals(root, kingdom)?;
    let (n, depth, lp) = k_loops(&plain).into_iter().find(|(n, _, _)| *n == node).ok_or("our software has a bug: the K loop is gone")?;
    let plan = unroll_plan(n, depth, &lp, factor)?;
    let index = dual_ty(kingdom, "u32.index")?;
    let counter = tile_counter_name(depth);
    let mut body = Vec::new();
    for st in statements(&lp.kids[1]) {
        body.push(re_k(&st, counter, factor, 0, index));
        for j in 1..factor {
            body.push(re_k(&strip_k_constant(&st, plan.stride), counter, factor, j, index));
        }
    }
    let count = plain_value(&lp.kids[0]).ok_or("the K loop's count is not a literal")? / factor;
    rewrite_k_loop(&plain, n, body, count, kingdom)
}

/// RULES: UN-UNROLL K / UNROLL K (one site per K loop whose plan holds).
pub fn ununroll_k_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = crate::wgsl::params::to_literals(root, kingdom) else { return Vec::new() };
    k_loops(&plain).into_iter().filter_map(|(n, d, lp)| ununroll_plan(&plain, n, d, &lp).ok().map(|p| Site { index: n, what: format!("K loop of {} statements, {} dropped", p.drop.len() * 2, p.drop.len()) })).collect()
}

pub fn unroll_k_sites(root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
    let Ok(plain) = crate::wgsl::params::to_literals(root, kingdom) else { return Vec::new() };
    k_loops(&plain).into_iter().filter_map(|(n, d, lp)| unroll_plan(n, d, &lp, 2).ok().map(|_| Site { index: n, what: format!("K loop of {} statements", statements(&lp.kids[1]).len()) })).collect()
}

fn k_why(root: &Tree, kingdom: &WgslKingdom, unroll: bool) -> Option<String> {
    let plain = crate::wgsl::params::to_literals(root, kingdom).ok()?;
    let loops = k_loops(&plain);
    if loops.is_empty() {
        return Some("no counted loop reads the tiles".into());
    }
    let why: Vec<String> = loops.iter().filter_map(|(n, d, lp)| if unroll { unroll_plan(*n, *d, lp, 2).err() } else { ununroll_plan(&plain, *n, *d, lp).err() }).collect();
    (why.len() == loops.len()).then(|| why.join("; "))
}

/// A rule of the catalogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rule {
    DropZeroFill,
    GrowRows,
    VecTile,
    ShrinkRows,
    UnunrollK,
    UnrollK,
    DropDeadStore,
    UnrollK4,
    RollK,
    SplitAccumulator,
    WgShape,
    HoistLoad,
    PadShared,
    GrowCols,
    ShrinkCols,
    DoubleBuffer,
}

impl Rule {
    /// The pool. wg-shape is out (2026-10-13, run 3: all 10 children wrong on
    /// the device; the reader's references cannot tell equal knobs apart);
    /// its rule stays for the table and the tests.
    pub const ALL: [Rule; 15] = [
        Rule::DropZeroFill,
        Rule::GrowRows,
        Rule::VecTile,
        Rule::ShrinkRows,
        Rule::UnunrollK,
        Rule::UnrollK,
        Rule::DropDeadStore,
        Rule::UnrollK4,
        Rule::RollK,
        Rule::SplitAccumulator,
        Rule::HoistLoad,
        Rule::PadShared,
        Rule::GrowCols,
        Rule::ShrinkCols,
        Rule::DoubleBuffer,
    ];

    /// Rules kept out of the pool until they are fixed, still measurable by name.
    pub const OUT_OF_POOL: [Rule; 1] = [Rule::WgShape];

    /// A rule of the pool or out of it, by name.
    pub fn by_name(name: &str) -> Option<Rule> {
        Rule::ALL.into_iter().chain(Rule::OUT_OF_POOL).find(|r| r.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Rule::DropZeroFill => "drop-zero-fill",
            Rule::GrowRows => "grow-rows",
            Rule::VecTile => "vec4-tile",
            Rule::ShrinkRows => "shrink-rows",
            Rule::UnunrollK => "un-unroll-k",
            Rule::UnrollK => "unroll-k",
            Rule::DropDeadStore => "drop-dead-store",
            Rule::UnrollK4 => "unroll-k4",
            Rule::RollK => "roll-k",
            Rule::SplitAccumulator => "split-accumulator",
            Rule::WgShape => "wg-shape",
            Rule::HoistLoad => "hoist-load",
            Rule::PadShared => "pad-shared",
            Rule::GrowCols => "grow-cols",
            Rule::ShrinkCols => "shrink-cols",
            Rule::DoubleBuffer => "double-buffer",
        }
    }

    pub fn sites(self, root: &Tree, kingdom: &WgslKingdom) -> Vec<Site> {
        match self {
            Rule::DropZeroFill => drop_zero_fill_sites(root),
            Rule::GrowRows => grow_rows_sites(root, kingdom),
            Rule::VecTile => vec_tile_sites(root, kingdom),
            Rule::ShrinkRows => shrink_rows_sites(root, kingdom),
            Rule::UnunrollK => ununroll_k_sites(root, kingdom),
            Rule::UnrollK => unroll_k_sites(root, kingdom),
            Rule::DropDeadStore => more::drop_dead_store_sites(root),
            Rule::UnrollK4 => more::unroll_k4_sites(root, kingdom),
            Rule::RollK => more::roll_k_sites(root, kingdom),
            Rule::SplitAccumulator => more::split_accumulator_sites(root, kingdom),
            Rule::WgShape => more::wg_shape_sites(root, kingdom),
            Rule::HoistLoad => more::hoist_load_sites(root, kingdom),
            Rule::PadShared => more::pad_shared_sites(root, kingdom),
            Rule::GrowCols => more::grow_cols_sites(root, kingdom),
            Rule::ShrinkCols => more::shrink_cols_sites(root, kingdom),
            Rule::DoubleBuffer => more::double_buffer_sites(root, kingdom),
        }
    }

    /// Why the rule does not fire in this kernel (None when it fires).
    pub fn why_not(self, root: &Tree, kingdom: &WgslKingdom) -> Option<String> {
        match self {
            Rule::DropZeroFill => drop_zero_fill_sites(root).is_empty().then(|| "no dead zero store outside a loop".to_string()),
            Rule::GrowRows => match (grow_rows_plan(root, kingdom), grow_strided_plan(root, kingdom)) {
                (Err(a), Err(b)) => Some(format!("contiguous: {a}; strided: {b}")),
                _ => None,
            },
            Rule::VecTile => vec_tile_why(root, kingdom),
            Rule::ShrinkRows => shrink_rows_plan(root, kingdom).err(),
            Rule::UnunrollK => k_why(root, kingdom, false),
            Rule::UnrollK => k_why(root, kingdom, true),
            other => more::why_not(other, root, kingdom),
        }
    }

    pub fn apply(self, root: &Tree, site: &Site, kingdom: &WgslKingdom) -> Result<Tree, String> {
        match self {
            Rule::DropZeroFill => drop_zero_fill(root, site, kingdom),
            Rule::GrowRows => {
                if site.index == 1 {
                    grow_strided(root, kingdom)
                } else {
                    grow_rows(root, kingdom)
                }
            }
            Rule::VecTile => vec_tile(root, site.index, kingdom),
            Rule::ShrinkRows => shrink_rows(root, kingdom),
            Rule::UnunrollK => ununroll_k(root, site.index, kingdom),
            Rule::UnrollK => unroll_k(root, site.index, kingdom),
            Rule::DropDeadStore => more::drop_dead_store(root, site, kingdom),
            Rule::UnrollK4 => more::unroll_k4(root, site, kingdom),
            Rule::RollK => more::roll_k(root, site, kingdom),
            Rule::SplitAccumulator => more::split_accumulator(root, site, kingdom),
            Rule::WgShape => more::wg_shape(root, site, kingdom),
            Rule::HoistLoad => more::hoist_load(root, site, kingdom),
            Rule::PadShared => more::pad_shared(root, site, kingdom),
            Rule::GrowCols => more::grow_cols(root, kingdom),
            Rule::ShrinkCols => more::shrink_cols(root, kingdom),
            Rule::DoubleBuffer => more::double_buffer(root, site, kingdom),
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

    /// VECTORISE THE STRIDED f32 TILES of a saved kernel (popC_rand20's
    /// r01, 16 x 16 workgroup, 4 x 4 blocks, a thread's four columns 16
    /// apart): the loader counts vec4s, stores each whole from one thread,
    /// and the 16 products read lanes of two vec4s (its child was correct
    /// on all 128 device cases, 2026-10-12).
    #[test]
    fn the_strided_tiles_of_a_loader_become_interleaved_vec4_tiles() {
        let k = WgslKingdom::load_safe();
        let f = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/popC_rand20/texts/r01_0.1954_fit_A_g001_r12_86aa44.wgsl");
        let Ok(src) = std::fs::read_to_string(&f) else { return };
        let t = read_safe(&src, &k).unwrap().tree;
        let sites = vec_tile_sites(&t, &k);
        assert_eq!(sites.len(), 1, "{sites:?}");
        let child = vec_tile(&t, sites[0].index, &k).unwrap();
        let (text, _) = crate::wgsl::knob::check_text(&child, &k).unwrap_or_else(|(s, e)| panic!("{}: {e}", s.name()));
        assert!(!text.contains("wg_a[") && !text.contains("wg_b["), "{text}");
        assert!(text.contains("var<workgroup> wg_va: array<vec4<f32>, 512>;") && text.contains("for (k = 0u; k < 1u; k = k + 1u)"), "{text}");
        assert!(text.contains("wg_va[((((k * 16u) + lid.x)) & 511u)].w"), "{text}");
        // No lane is stored into a workgroup vec4 (four threads would race on it).
        assert!(!text.lines().any(|l| l.contains("wg_v") && l.contains("][") && l.contains("] = ")), "{text}");
    }

    /// SHRINK THE ROW BLOCK, the inverse of growing it: reserve2's 16 x 8
    /// kernel of 4 x 8 blocks (rows 4..7 read at 2 y + 1) becomes 16 x 16 of
    /// 4 x 4, its threads' stride and guards following the grid (rewritten
    /// on the plain tree they did not: 5 of 5 children were wrong on the
    /// device; now 5 of 5 correct).
    #[test]
    fn the_row_block_shrinks_back_with_every_reference_following_the_grid() {
        let k = WgslKingdom::load_safe();
        let f = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/reserve2/texts/0.1066_fit_A2_stopped_1702Z_g001_r12_d93a44.wgsl");
        let Ok(src) = std::fs::read_to_string(&f) else { return };
        let t = read_safe(&src, &k).unwrap().tree;
        let (tile, family) = shrink_rows_plan(&t, &k).unwrap();
        assert_eq!(tile, "wg.vb");
        assert!(family.values().any(|&r| r >= 4));
        let child = shrink_rows(&t, &k).unwrap();
        let (text, _) = crate::wgsl::knob::check_text(&child, &k).unwrap_or_else(|(s, e)| panic!("{}: {e}", s.name()));
        assert!(text.starts_with("// grid 16 16 4 4"), "{}", &text[..40]);
        assert!(text.contains("256u * k2") && !text.contains("128u * k2") && !text.contains("lid.y * 2u") && !text.contains("gid.y * 8u"), "{text}");
        assert!(grow_rows_plan(&t, &k).is_err() && shrink_rows_plan(&grow_rows(&read_safe(&text, &k).unwrap().tree, &k).unwrap(), &k).is_ok());
    }

    /// THE CHAIN of the strided family: popC_rand20's r01 (16 x 16, 4 x 4
    /// blocks, rows owned 16 apart, f32 tiles) grows its rows and lays its
    /// tiles out as quarter-row vec4s in either order, and both two-step
    /// children check (a 16 x 8 workgroup of 4 x 8 blocks, rows 8 apart).
    #[test]
    fn a_strided_kernel_climbs_two_steps_in_either_order() {
        let k = WgslKingdom::load_safe();
        let f = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/popC_rand20/texts/r01_0.1954_fit_A_g001_r12_86aa44.wgsl");
        let Ok(src) = std::fs::read_to_string(&f) else { return };
        let t = read_safe(&src, &k).unwrap().tree;
        assert!(grow_rows_plan(&t, &k).is_err(), "not the contiguous form");
        let step = |t: &Tree, rule: Rule| -> Tree {
            let site = rule.sites(t, &k).into_iter().next().unwrap_or_else(|| panic!("{} does not fire: {:?}", rule.name(), rule.why_not(t, &k)));
            let child = rule.apply(t, &site, &k).unwrap();
            crate::wgsl::knob::check_text(&child, &k).unwrap_or_else(|(s, e)| panic!("{}: {}: {e}", rule.name(), s.name())).1.tree
        };
        for (first, second) in [(Rule::GrowRows, Rule::VecTile), (Rule::VecTile, Rule::GrowRows)] {
            let two = step(&step(&t, first), second);
            let text = crate::wgsl::safe_karva::render_tree(&two, &k).unwrap().text;
            assert!(text.starts_with("// grid 16 8 4 8"), "{} then {}: {}", first.name(), second.name(), &text[..40]);
            assert!(text.contains("var<workgroup> wg_va: array<vec4<f32>,") && !text.contains("wg_a["), "{text}");
            assert!(text.contains("lidx + (128u * k)"), "{text}");
        }
    }

    /// UNROLL K AND UNDO IT, on two saved kernels of reserve2 (2026-10-12):
    /// one stepping K two rows a pass into two sets of accumulators (r01 of
    /// popR24) steps one row a pass, twice the passes, and then grows its
    /// rows; one stepping one row (r00 of popR24) steps two, half the
    /// passes. Every child checks; each was correct on the device.
    #[test]
    fn a_k_unroll_by_two_is_undone_and_redone() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/reserve2/texts");
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok().map(|src| read_safe(&src, &k).unwrap().tree);
        let Some(unrolled) = read("0.1209_fit_A2_stopped_1729Z_g001_r16_c65da8.wgsl") else { return };
        let site = Rule::UnunrollK.sites(&unrolled, &k).into_iter().next().unwrap_or_else(|| panic!("{:?}", Rule::UnunrollK.why_not(&unrolled, &k)));
        let child = crate::wgsl::knob::check_text(&Rule::UnunrollK.apply(&unrolled, &site, &k).unwrap(), &k).unwrap_or_else(|(s, e)| panic!("{}: {e}", s.name())).1.tree;
        let text = crate::wgsl::safe_karva::render_tree(&child, &k).unwrap().text;
        assert!(text.contains("for (k = 0u; k < 32u; k = k + 1u)") && text.contains("(k * 16u) + lid.y") && !text.contains("(2u * k)"), "{text}");
        assert!(!Rule::GrowRows.sites(&child, &k).is_empty(), "grow-rows fires after un-unroll-K");
        let one = read("0.1208_fit_A2_stopped_1702Z_g000_r19_6d34a7.wgsl").unwrap();
        let site = Rule::UnrollK.sites(&one, &k).into_iter().next().unwrap_or_else(|| panic!("{:?}", Rule::UnrollK.why_not(&one, &k)));
        let child = crate::wgsl::knob::check_text(&Rule::UnrollK.apply(&one, &site, &k).unwrap(), &k).unwrap_or_else(|(s, e)| panic!("{}: {e}", s.name())).1.tree;
        let text = crate::wgsl::safe_karva::render_tree(&child, &k).unwrap().text;
        assert!(text.contains("for (k = 0u; k < 16u; k = k + 1u)") && text.contains("((2u * k) * 16u) + 16u)"), "{text}");
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
