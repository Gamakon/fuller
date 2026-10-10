//! A drafted kernel read into the safeWGSL grammar (`kingdoms/safewgsl`): the
//! statements of its entry function walked in naga's order (the reader's
//! root numbering), its `let`s inlined, its expressions carried over as typed
//! trees, and the safe rows substituted wherever the grammar demands
//! provenance — a load of `phi` must be `item · d + column` over the loop's
//! own counter, a store to `h` must be read-add-write at this thread's cell,
//! a loop must count to a binding or a small literal in steps of one, a
//! barrier must stand outside any condition. A draft that does not fit is
//! refused at the first statement that fails, by name; it is never admitted
//! by guessing.

use std::collections::{BTreeMap, HashMap};

use naga::{Expression, Function, Handle, Statement};

use crate::geneframe::Ty;
use crate::wgsl::infer::infer_function;
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::reader::{read, Node, Root, RootKind};
use crate::wgsl::render::{dual_ty, Tree};

/// The conversion's result: the safe tree and what was noted on the way.
pub struct SafeRead {
    pub tree: Tree,
    pub notes: Vec<String>,
    pub registers: BTreeMap<String, String>,
}

struct Ctx<'a> {
    kingdom: &'a WgslKingdom,
    duals: &'a BTreeMap<String, Option<Ty>>,
    lets: &'a HashMap<String, Node>,
    roots: &'a HashMap<Vec<usize>, Vec<&'a Root>>,
    /// The item loop's counter local (`load.local.t@1`), while inside it.
    item_counter: Option<String>,
    /// The tile loops' counter locals, outer then inner, while inside them.
    tile_counter: Option<String>,
    tile_counter2: Option<String>,
    tile_counter3: Option<String>,
    /// Register allocation: local name (`local.acc@0`) → register row (`reg.f0`).
    registers: BTreeMap<String, String>,
    next_reg: BTreeMap<char, usize>,
    /// The thread guard from a leading `if (c) { return; }`: every later
    /// store to h is wrapped in `if (!c)`.
    guard: Option<Tree>,
    /// Whether the block being walked ends the function (a `return` that
    /// is its last statement is the implicit return naga writes).
    tail: bool,
    /// Workgroup arrays by name → `wg.a` / `wg.b`.
    arrays: BTreeMap<String, &'static str>,
    /// Local arrays by name → a private array row, its kind and its mask row.
    parrs: BTreeMap<String, (&'static str, char, &'static str)>,
    grid: (u32, u32, u32, u32),
    notes: Vec<String>,
}

fn ty(ctx: &Ctx, name: &str) -> Result<Ty, String> {
    dual_ty(ctx.kingdom, name)
}

/// The reader's node as a typed tree, duals by the inference's text, lets inlined.
fn tree_of(ctx: &mut Ctx, n: &Node) -> Result<Tree, String> {
    let text = n.to_sexpr();
    match n {
        Node::Leaf { name, .. } => {
            if let Some(def) = name.strip_prefix("let.").and_then(|k| ctx.lets.get(k)) {
                let def = def.clone();
                return tree_of(ctx, &def);
            }
            let t = ctx.duals.get(&text).copied().flatten().ok_or_else(|| format!("untyped leaf {name}"))?;
            Ok(Tree::new(name, t, vec![]))
        }
        Node::Literal { form, value, .. } => {
            let t = match ctx.duals.get(&text).copied().flatten() {
                Some(t) => t,
                None => {
                    // Two forms in the inference (an index here, a count there): the form's own scalar dual.
                    let dual = match form.as_str() { "real" => "f32.real", "int" => "i32.int", "flag" => "bool.flag", "count" => "u32.count", "bits" => "u32.bits", _ => "u32.index" };
                    ty(ctx, dual)?
                }
            };
            Ok(Tree::literal(&format!("literal.{form}"), t, *value as f32))
        }
        Node::App { name, kids, known, .. } => {
            if !known {
                return Err(format!("{name} has no row in the table"));
            }
            let t = ctx.duals.get(&text).copied().flatten().ok_or_else(|| format!("untyped node {}", &text[..text.len().min(80)]))?;
            let kids = kids.iter().map(|k| tree_of(ctx, k)).collect::<Result<Vec<_>, _>>()?;
            Ok(Tree::new(name, t, kids))
        }
    }
}

fn is_leaf(t: &Tree, name: &str) -> bool {
    t.kids.is_empty() && t.name == name
}

fn local_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("load.").or_else(|| name.strip_prefix("store."))?;
    let rest = rest.strip_prefix("local.")?;
    Some(format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@")))
}

/// The private array dual of a kind letter.
fn parr_dual(kind: char) -> &'static str {
    match kind {
        'v' => "parr<vec4<f32>>.real",
        'i' => "parr<u32>.index",
        'b' => "parr<bool>.flag",
        _ => "parr<f32>.real",
    }
}

fn literal_value(t: &Tree) -> Option<f32> {
    t.name.starts_with("literal.").then_some(t.konst)
}

/// `gid.y` / `gid.x` as a tree: `(shape.access_1 builtin.global_invocation_id)`.
fn gid_axis(t: &Tree, axis: usize) -> bool {
    t.name == format!("shape.access_{axis}") && t.kids.len() == 1 && is_leaf(&t.kids[0], "builtin.global_invocation_id")
}

fn id_axis(t: &Tree, axis: usize, builtin: &str) -> bool {
    t.name == format!("shape.access_{axis}") && t.kids.len() == 1 && is_leaf(&t.kids[0], builtin)
}

/// `(index.mul X K)` in either order with X the builtin's axis → K.
fn scaled_id(t: &Tree, axis: usize, builtin: &str) -> Option<u32> {
    if t.name != "index.mul" || t.kids.len() != 2 {
        return None;
    }
    for (a, b) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
        if id_axis(a, axis, builtin) {
            return literal_value(b).map(|k| k as u32);
        }
    }
    None
}

/// `wid.axis * (W * B) + lid.axis * B` (either order) with W the workgroup
/// size on the axis and B the block factor: equal to `gid.axis * B`.
fn wid_lid_axis(t: &Tree, axis: usize, workgroup: u32) -> Option<(u32, u32)> {
    if t.name != "index.add" || t.kids.len() != 2 {
        return None;
    }
    for (a, b) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
        let kl = if id_axis(b, axis, "builtin.local_invocation_id") { Some(1) } else { scaled_id(b, axis, "builtin.local_invocation_id") };
        if let (Some(kw), Some(kl)) = (scaled_id(a, axis, "builtin.work_group_id").or_else(|| scaled_id(a, axis, "builtin.workgroup_id")), kl) {
            if kw == workgroup * kl {
                return Some((kl, 0));
            }
        }
    }
    None
}

/// An offset within the thread's block: a literal, or a tile counter; for a
/// strided block (rows W apart) a literal multiple of W.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Off {
    Lit(u32),
    Tile(u8),
    Strided(u32),
    StridedTile(u8),
}

/// The tile counters in effect, outer to inner.
type Tiles<'t> = [&'t Option<String>; 3];

/// Which tile counter (1, 2 or 3) a local is, if any.
fn tile_of(local: &str, tiles: Tiles) -> Option<u8> {
    tiles.iter().position(|c| c.as_deref() == Some(local)).map(|p| p as u8 + 1)
}

/// A row or column expression of this thread: `gid.axis`, `gid.axis * K`,
/// `gid.axis * K + off` (off a literal below K, or a tile counter), with K
/// the grid's block factor → `(blk, offset)`.
fn thread_axis_off(t: &Tree, axis: usize, workgroup: u32, tiles: Tiles) -> Option<(u32, Off)> {
    if let Some((k, 0)) = thread_axis(t, axis, workgroup) {
        return Some((k, Off::Lit(0)));
    }
    // A strided block: wid * (W * B) + lid, the i-th row at + i * W.
    if let Some(b) = strided_axis(t, axis, workgroup) {
        return Some((b, Off::Strided(0)));
    }
    if t.name == "index.add" && t.kids.len() == 2 {
        for (x, y) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
            if let Some(b) = strided_axis(x, axis, workgroup) {
                if let Some(i) = literal_value(y) {
                    let i = i as u32;
                    if i.is_multiple_of(workgroup) && i / workgroup < b {
                        return Some((b, Off::Strided(i / workgroup)));
                    }
                }
                // + k * W (either order): the k-th row of the strided block.
                if y.name == "index.mul" && y.kids.len() == 2 {
                    for (p, q) in [(&y.kids[0], &y.kids[1]), (&y.kids[1], &y.kids[0])] {
                        if literal_value(p) == Some(workgroup as f32) && q.kids.is_empty() {
                            if let Some(n) = local_name(&q.name).and_then(|l| tile_of(&l, tiles)) {
                                return Some((b, Off::StridedTile(n)));
                            }
                        }
                    }
                }
            }
        }
    }
    if t.name == "index.add" && t.kids.len() == 2 {
        for (x, y) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
            if let Some((k, 0)) = thread_axis(x, axis, workgroup) {
                if let Some(i) = literal_value(y) {
                    if (i as u32) < k {
                        return Some((k, Off::Lit(i as u32)));
                    }
                }
                if y.kids.is_empty() {
                    if let Some(n) = local_name(&y.name).and_then(|l| tile_of(&l, tiles)) {
                        return Some((k, Off::Tile(n)));
                    }
                }
            }
        }
    }
    None
}

/// `wid.axis * (W * B) + lid.axis` (lid unscaled, B > 1): a strided block → B.
fn strided_axis(t: &Tree, axis: usize, workgroup: u32) -> Option<u32> {
    if t.name != "index.add" || t.kids.len() != 2 {
        return None;
    }
    for (a, b) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
        if id_axis(b, axis, "builtin.local_invocation_id") {
            if let Some(kw) = scaled_id(a, axis, "builtin.work_group_id").or_else(|| scaled_id(a, axis, "builtin.workgroup_id")) {
                if workgroup > 0 && kw.is_multiple_of(workgroup) && kw / workgroup > 1 {
                    return Some(kw / workgroup);
                }
            }
        }
    }
    None
}

fn thread_axis(t: &Tree, axis: usize, workgroup: u32) -> Option<(u32, u32)> {
    if gid_axis(t, axis) {
        return Some((1, 0));
    }
    if let Some(x) = wid_lid_axis(t, axis, workgroup) {
        return Some(x);
    }
    if t.name == "index.mul" && t.kids.len() == 2 {
        let (a, b) = (&t.kids[0], &t.kids[1]);
        if gid_axis(a, axis) {
            return literal_value(b).map(|k| (k as u32, 0));
        }
        if gid_axis(b, axis) {
            return literal_value(a).map(|k| (k as u32, 0));
        }
    }
    if t.name == "index.add" && t.kids.len() == 2 {
        for (x, y) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
            if let (Some((k, 0)), Some(i)) = (thread_axis(x, axis, workgroup), literal_value(y)) {
                if (i as u32) < k {
                    return Some((k, i as u32));
                }
            }
        }
    }
    None
}

impl<'a> Ctx<'a> {
    fn register(&mut self, local: &str, dual: &str) -> Result<String, String> {
        if let Some(r) = self.registers.get(local) {
            return Ok(r.clone());
        }
        let (kind, max) = match dual {
            "f32.real" => ('f', 16),
            "vec4<f32>.real" => ('v', 8),
            "u32.index" => ('i', 16),
            "bool.flag" => ('b', 4),
            other => return Err(format!("local {local} has dual {other}, which has no register kind"),)
        };
        let n = *self.next_reg.get(&kind).unwrap_or(&0);
        if n >= max {
            return Err(format!("more than {max} {kind} registers (local {local})"));
        }
        self.next_reg.insert(kind, n + 1);
        let row = format!("reg.{kind}{n}");
        self.registers.insert(local.to_string(), row.clone());
        Ok(row)
    }

    fn reg_dual(kind: char) -> &'static str {
        match kind {
            'f' => "reg<f32>.real",
            'v' => "reg<vec4<f32>>.real",
            'i' => "reg<u32>.index",
            _ => "reg<bool>.flag",
        }
    }

    /// A value tree in the safe grammar.
    fn value(&mut self, t: &Tree) -> Result<Tree, String> {
        let d = self.kingdom.dual(t.ty).map(|d| d.dual.clone()).unwrap_or_default();
        // Leaves.
        if t.kids.is_empty() {
            if let Some(local) = local_name(&t.name) {
                if self.item_counter.as_deref() == Some(local.as_str()) {
                    return Ok(Tree::new("safe.counter_index", ty(self, "u32.index")?, vec![Tree::new("safe.counter", ty(self, "u32.counter")?, vec![])]));
                }
                if self.tile_counter.as_deref() == Some(local.as_str()) {
                    return Ok(Tree::new("safe.tile_index", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter", ty(self, "u32.tcounter")?, vec![])]));
                }
                if self.tile_counter2.as_deref() == Some(local.as_str()) {
                    return Ok(Tree::new("safe.tile_index2", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter2", ty(self, "u32.tcounter2")?, vec![])]));
                }
                if self.tile_counter3.as_deref() == Some(local.as_str()) {
                    return Ok(Tree::new("safe.tile_index3", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter3", ty(self, "u32.tcounter3")?, vec![])]));
                }
                let row = self.register(&local, &d)?;
                let kind = row.chars().nth(4).unwrap_or('f');
                return Ok(Tree::new(&format!("reg.get_{kind}"), t.ty, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![])]));
            }
            if let Some(field) = t.name.strip_prefix("load.uniform.cfg.") {
                return match field {
                    "d" | "m" | "r_rows" => Ok(Tree::new(&format!("load.cfg_{field}"), ty(self, "u32.count")?, vec![])),
                    "r_off" => Ok(Tree::new("load.cfg_r_off", ty(self, "u32.index")?, vec![])),
                    other => Err(format!("uniform field {other}")),
                };
            }
            if t.name == "builtin.work_group_id" {
                // naga's name for the builtin; the table's row is `builtin.workgroup_id`.
                return Ok(Tree::new("builtin.workgroup_id", t.ty, vec![]));
            }
            if t.name.starts_with("builtin.") || t.name.starts_with("literal.") {
                return Ok(t.clone());
            }
            return Err(format!("leaf {} has no safe spelling", t.name));
        }
        // Loads with an index.
        if t.name.starts_with("load.buffer.phi.#") {
            // Any index: a read past the buffer is zero by WGSL's rule.
            let idx = self.value(&t.kids[0])?;
            return Ok(Tree::new("load.phi_safe", ty(self, "f32.real")?, vec![idx]));
        }
        if t.name.starts_with("load.buffer.h.#") {
            let cell = self.cell(&t.kids[0])?;
            return Ok(Tree::new("load.h_safe", ty(self, "f32.real")?, vec![cell]));
        }
        if let Some(rest) = t.name.strip_prefix("load.workgroup.") {
            let arr = rest.split('.').next().unwrap_or("");
            let which = self.array(arr, &d)?;
            let (of, idx_dual, arr_dual, get, _) = Self::wg_rows(which);
            let idx = self.value(&t.kids[0])?;
            let wg = Tree::new(of, ty(self, idx_dual)?, vec![idx]);
            return Ok(Tree::new(get, t.ty, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), wg]));
        }
        if let Some(rest) = t.name.strip_prefix("load.local.") {
            if rest.contains(".#") {
                // A private array: `load.local.acc@0.#@v3 (idx)`.
                let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
                let (arr, kind, mask) = self.private_array(&local, &d)?;
                let idx = self.value(&t.kids[0])?;
                let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![idx]);
                let arr_dual = parr_dual(kind);
                return Ok(Tree::new(&format!("parr.get_{kind}"), t.ty, vec![Tree::new(arr, ty(self, arr_dual)?, vec![]), pidx]));
            }
        }
        if t.name.starts_with("load.") {
            return Err(format!("load {} has no safe spelling", t.name));
        }
        let kids = t.kids.iter().map(|k| self.value(k)).collect::<Result<Vec<_>, _>>()?;
        // The reader suffixes a compose with its slot (`shape.compose4.vec4<f32>`);
        // the table's row is the bare name, the tree's dual carries the slot.
        let name = if t.name.starts_with("shape.compose") { t.name.split('.').take(2).collect::<Vec<_>>().join(".") } else { t.name.clone() };
        Ok(Tree { name, ty: t.ty, konst: t.konst, kids })
    }

    /// A local array as a private array row: (`parr.a|b|v|i`, kind, mask row).
    fn private_array(&mut self, local: &str, elem_dual: &str) -> Result<(&'static str, char, &'static str), String> {
        if let Some(a) = self.parrs.get(local) {
            return Ok(*a);
        }
        let (kind, mask) = match elem_dual {
            "f32.real" => ('f', "safe.pidx64"),
            "vec4<f32>.real" => ('v', "safe.pidx16"),
            "u32.index" => ('i', "safe.pidx16"),
            "bool.flag" => ('b', "safe.pidx16"),
            other => return Err(format!("a local array {local} of {other} has no private array")),
        };
        let used: Vec<&'static str> = self.parrs.values().map(|a| a.0).collect();
        let arr = match kind {
            'f' => ["parr.a", "parr.b", "parr.c", "parr.d", "parr.e", "parr.f"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a seventh f32 local array {local}"))?,
            'v' => if used.contains(&"parr.v") { return Err(format!("a second vec4 local array {local}")) } else { "parr.v" },
            'b' => ["parr.m", "parr.n"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a third bool local array {local}"))?,
            _ => ["parr.i", "parr.j", "parr.k"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a fourth u32 local array {local}"))?,
        };
        self.parrs.insert(local.to_string(), (arr, kind, mask));
        Ok((arr, kind, mask))
    }

    /// A workgroup array by name and element dual: `wg.a`/`wg.b` hold f32,
    /// `wg.va`/`wg.vb` hold vec4<f32>.
    fn array(&mut self, name: &str, elem_dual: &str) -> Result<&'static str, String> {
        if let Some(w) = self.arrays.get(name) {
            return Ok(w);
        }
        let candidates: &[&'static str] = match elem_dual {
            "f32.real" => &["wg.a", "wg.b"],
            "vec4<f32>.real" => &["wg.va", "wg.vb"],
            other => return Err(format!("a workgroup array {name} of {other} has no safe spelling")),
        };
        let used: Vec<&'static str> = self.arrays.values().copied().collect();
        let which = candidates.iter().copied().find(|c| !used.contains(c)).ok_or_else(|| format!("a third workgroup array {name} of {elem_dual}; the grammar has two"))?;
        self.arrays.insert(name.to_string(), which);
        Ok(which)
    }

    /// The index and array duals, the get row and the set stem of a workgroup array row.
    fn wg_rows(which: &str) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
        if which.starts_with("wg.v") {
            ("safe.wgv_of", "u32.wgvindex", "wgmem<vec4<f32>>.real", "wg.get_v_safe", "wg.set_v")
        } else {
            ("safe.wg_of", "u32.wgindex", "wgmem<f32>.real", "wg.get_safe", "wg.set")
        }
    }

    /// `(index.add (index.mul R D) C)` with R and C this thread's row and
    /// column → `safe.cell_of(row, col)`.
    fn cell(&mut self, idx: &Tree) -> Result<Tree, String> {
        let fail = |what: &str| Err(format!("a store or load of h is not at this thread's cell: {what}: {}", idx.to_sexpr()));
        if idx.name != "index.add" || idx.kids.len() != 2 {
            return fail("not row * d + col");
        }
        // `(row * d + col) + j` with j a literal: the literal belongs to the column.
        let mut idx = idx.clone();
        loop {
            let mut folded = None;
            for (inner, lit) in [(&idx.kids[0], &idx.kids[1]), (&idx.kids[1], &idx.kids[0])] {
                if let Some(j) = literal_value(lit) {
                    if inner.name == "index.add" && inner.kids.len() == 2 {
                        // The column is the kid that is not `row * cfg.d`.
                        let is_prod = |t: &Tree| t.name == "index.mul" && t.kids.iter().any(|k| is_leaf(k, "load.uniform.cfg.d"));
                        let (p, c) = if is_prod(&inner.kids[0]) { (0, 1) } else { (1, 0) };
                        let col = Tree::new("index.add", inner.ty, vec![inner.kids[c].clone(), Tree::literal(&lit.name, lit.ty, j)]);
                        folded = Some(Tree::new("index.add", inner.ty, vec![inner.kids[p].clone(), col]));
                        break;
                    }
                }
            }
            match folded {
                Some(t) => idx = t,
                None => break,
            }
        }
        let idx = &idx;
        for (prod, col) in [(&idx.kids[0], &idx.kids[1]), (&idx.kids[1], &idx.kids[0])] {
            if prod.name == "index.mul" && prod.kids.len() == 2 {
                for (r, dd) in [(&prod.kids[0], &prod.kids[1]), (&prod.kids[1], &prod.kids[0])] {
                    if !is_leaf(dd, "load.uniform.cfg.d") {
                        continue;
                    }
                    let tiles = (self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone());
                    let tiles: Tiles = [&tiles.0, &tiles.1, &tiles.2];
                    let (Some((by, i)), Some((bx, j))) = (thread_axis_off(r, 1, self.grid.1, tiles), thread_axis_off(col, 0, self.grid.0, tiles)) else { continue };
                    if by != self.grid.3 || bx != self.grid.2 {
                        return fail(&format!("block {bx} x {by} in the index, the grid says {} x {}", self.grid.2, self.grid.3));
                    }
                    let gid = Tree::new("builtin.global_invocation_id", ty(self, "vec3<u32>.index")?, vec![]);
                    let blk = |v: u32| Tree::literal("literal.blk", ty(self, "u32.blk").expect("dual"), v as f32);
                    let plus = |ctx: &Self, base: Tree, off: Off, axis: &str| -> Result<Tree, String> {
                        let dual = ty(ctx, if axis == "row" { "u32.threadrow" } else { "u32.threadcol" })?;
                        let counter = |ctx: &Self, n: u8| -> Result<Tree, String> {
                            let suffix = if n == 1 { String::new() } else { n.to_string() };
                            Ok(Tree::new(&format!("safe.tile_counter{suffix}"), ty(ctx, &format!("u32.tcounter{suffix}"))?, vec![]))
                        };
                        let suffix = |n: u8| if n == 1 { String::new() } else { n.to_string() };
                        Ok(match off {
                            Off::Lit(0) | Off::Strided(0) => base,
                            Off::Strided(v) => Tree::new(&format!("safe.{axis}_plus_s"), dual, vec![base, Tree::literal("literal.lit64", ty(ctx, "u32.lit64")?, v as f32)]),
                            Off::Lit(v) => Tree::new(&format!("safe.{axis}_plus"), dual, vec![base, Tree::literal("literal.lit64", ty(ctx, "u32.lit64")?, v as f32)]),
                            Off::Tile(n) => Tree::new(&format!("safe.{axis}_plus_tile{}", suffix(n)), dual, vec![base, counter(ctx, n)?]),
                            Off::StridedTile(n) => Tree::new(&format!("safe.{axis}_plus_tile{}_s", suffix(n)), dual, vec![base, counter(ctx, n)?]),
                        })
                    };
                    let wid = Tree::new("builtin.workgroup_id", ty(self, "vec3<u32>.index")?, vec![]);
                    let lid = Tree::new("builtin.local_invocation_id", ty(self, "vec3<u32>.index")?, vec![]);
                    let base_row = if matches!(i, Off::Strided(_) | Off::StridedTile(_)) { Tree::new("safe.thread_row_s", ty(self, "u32.threadrow")?, vec![wid.clone(), lid.clone(), blk(by)]) } else { Tree::new("safe.thread_row", ty(self, "u32.threadrow")?, vec![gid.clone(), blk(by)]) };
                    let base_col = if matches!(j, Off::Strided(_) | Off::StridedTile(_)) { Tree::new("safe.thread_col_s", ty(self, "u32.threadcol")?, vec![wid, lid, blk(bx)]) } else { Tree::new("safe.thread_col", ty(self, "u32.threadcol")?, vec![gid, blk(bx)]) };
                    let row = plus(self, base_row, i, "row")?;
                    let c = plus(self, base_col, j, "col")?;
                    return Ok(Tree::new("safe.cell_of", ty(self, "u32.cell")?, vec![row, c]));
                }
            }
        }
        fail(&format!("row or column is not this thread's (tile counters {:?} / {:?} / {:?})", self.tile_counter, self.tile_counter2, self.tile_counter3))
    }

    fn stmt_ty(&self, rung: &str) -> Result<Ty, String> {
        ty(self, &format!("stmt.{rung}"))
    }

    /// A list of statements as a block of the rung.
    fn block(&self, stmts: Vec<Tree>, rung: &str) -> Result<Tree, String> {
        let bt = ty(self, &format!("block.{rung}"))?;
        let st = self.stmt_ty(rung)?;
        if stmts.is_empty() {
            return Ok(Tree::fallback(bt));
        }
        let mut items = stmts;
        // More than four: the tail nested as a `stmt.block`.
        while items.len() > 4 {
            let rest = items.split_off(3);
            let inner = self.block(rest, rung)?;
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

    fn root_at(&self, path: &[usize], kind: fn(&RootKind) -> bool) -> Option<&'a Root> {
        self.roots.get(path).and_then(|rs| rs.iter().copied().find(|r| kind(&r.kind)))
    }

    /// A naga block at a rung: `L0`/`L1`/`L2` or `C0`/`C1`/`C2`.
    fn walk(&mut self, block: &naga::Block, path: &mut Vec<usize>, rung: &str, start: usize) -> Result<Vec<Tree>, String> {
        let mut out = Vec::new();
        let mut index = start;
        let cond = rung.starts_with('C');
        let level: usize = rung[1..].parse().unwrap_or(0);
        let st = self.stmt_ty(rung)?;
        let block_tail = self.tail;
        let last = block.iter().rposition(|s| !matches!(s, Statement::Emit(_)));
        for (position, stmt) in block.iter().enumerate() {
            if matches!(stmt, Statement::Emit(_)) {
                continue;
            }
            let stmt_tail = block_tail && Some(position) == last;
            path.push(index);
            index += 1;
            let here = path.clone();
            let result: Result<Option<Tree>, String> = (|| {
                match stmt {
                    Statement::Emit(_) => Ok(None),
                    Statement::Block(b) => {
                        self.tail = stmt_tail;
                        let inner = self.walk(b, path, rung, 0);
                        self.tail = block_tail;
                        let inner = inner?;
                        Ok(Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(inner, rung)?])))
                    }
                    Statement::Store { .. } => {
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Store { .. })).ok_or("no store root at this statement")?;
                        let tree = tree_of(self, &root.tree)?;
                        self.store(&tree, rung)
                    }
                    Statement::If { accept, reject, .. } => {
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at this if")?;
                        let cond_tree = tree_of(self, &root.tree)?;
                        let c = self.value(&cond_tree.kids[0])?;
                        // The thread guard: `if (c) { return; }` at L0 with nothing else.
                        let is_return_only = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Return { value: None }));
                        if rung == "L0" && is_return_only(accept) && reject.iter().all(|s| matches!(s, Statement::Emit(_))) {
                            let not = Tree::new("logic.not", ty(self, "bool.flag")?, vec![c]);
                            self.guard = Some(match self.guard.take() {
                                Some(g) => Tree::new("logic.and", ty(self, "bool.flag")?, vec![g, not]),
                                None => not,
                            });
                            self.notes.push("a leading thread guard became a guard on every store to h".into());
                            return Ok(None);
                        }
                        let crung = format!("C{level}");
                        self.tail = stmt_tail;
                        path.push(0);
                        let a = self.walk(accept, path, &crung, 0);
                        path.pop();
                        path.push(1);
                        let r = self.walk(reject, path, &crung, 0);
                        path.pop();
                        self.tail = block_tail;
                        let a = a?;
                        let r = r?;
                        if r.is_empty() {
                            Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![c, self.block(a, &crung)?])))
                        } else {
                            Ok(Some(Tree::new(&format!("stmt.if_else_{rung}"), st, vec![c, self.block(a, &crung)?, self.block(r, &crung)?])))
                        }
                    }
                    Statement::Loop { body, continuing, break_if } => {
                        self.tail = false;
                        let looped = self.looped(body, continuing, *break_if, path, rung, cond);
                        self.tail = block_tail;
                        looped
                    }
                    Statement::Barrier(_) => {
                        if cond {
                            return Err("a barrier under a condition".into());
                        }
                        Ok(Some(Tree::new(&format!("stmt.barrier_{rung}"), st, vec![])))
                    }
                    Statement::Return { value: None } => {
                        if stmt_tail {
                            // The function's implicit return (naga writes one at
                            // the end of each branch of a trailing `if`).
                            return Ok(None);
                        }
                        if cond {
                            return Err("a return under a condition (not a leading guard)".into());
                        }
                        Ok(Some(Tree::new(&format!("stmt.return_{rung}"), st, vec![])))
                    }
                    other => Err(format!("statement {} has no safe spelling", stmt_name(other))),
                }
            })();
            path.pop();
            match result {
                Ok(Some(t)) => out.push(t),
                Ok(None) => {}
                Err(e) => return Err(format!("at statement {here:?}: {e}")),
            }
        }
        Ok(out)
    }

    fn store(&mut self, tree: &Tree, rung: &str) -> Result<Option<Tree>, String> {
        let st = self.stmt_ty(rung)?;
        let name = tree.name.as_str();
        if name.starts_with("store.buffer.h.#") {
            let [idx, value] = tree.kids.as_slice() else { return Err("store to h without an index".into()) };
            // read-add-write: value = (arith.add (load.buffer.h.# idx) x) in either order
            let x = (|| {
                if value.name == "arith.add" && value.kids.len() == 2 {
                    for (l, x) in [(&value.kids[0], &value.kids[1]), (&value.kids[1], &value.kids[0])] {
                        if l.name.starts_with("load.buffer.h.#") && l.kids.len() == 1 && l.kids[0].to_sexpr() == idx.to_sexpr() {
                            return Some(x);
                        }
                    }
                }
                None
            })()
            .ok_or("a store to h is not read-add-write of the same cell")?;
            let cell = self.cell(idx)?;
            let v = self.value(x)?;
            let store = |ctx: &Self, r: &str| -> Result<Tree, String> { Ok(Tree::new(&format!("store.h_{r}"), ctx.stmt_ty(r)?, vec![cell.clone(), v.clone()])) };
            return match self.guard.clone() {
                Some(g) if !rung.starts_with('C') => {
                    let level = &rung[1..];
                    let crung = format!("C{level}");
                    let inner = self.block(vec![store(self, &crung)?], &crung)?;
                    Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![g, inner])))
                }
                _ => Ok(Some(store(self, rung)?)),
            };
        }
        if let Some(rest) = name.strip_prefix("store.workgroup.") {
            let arr = rest.split('.').next().unwrap_or("");
            let [idx, value] = tree.kids.as_slice() else { return Err("store to a workgroup array without an index".into()) };
            let d = self.kingdom.dual(value.ty).map(|d| d.dual.clone()).unwrap_or_default();
            let which = self.array(arr, &d)?;
            let (of, idx_dual, arr_dual, _, set) = Self::wg_rows(which);
            let i = self.value(idx)?;
            let v = self.value(value)?;
            let wg = Tree::new(of, ty(self, idx_dual)?, vec![i]);
            return Ok(Some(Tree::new(&format!("{set}_{rung}"), st, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), wg, v])));
        }
        if let Some(rest) = name.strip_prefix("store.local.") {
            if rest.contains(".#") {
                let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
                let [idx, value] = tree.kids.as_slice() else { return Err("a store to a local array without an index".into()) };
                let d = self.kingdom.dual(value.ty).map(|d| d.dual.clone()).unwrap_or_default();
                let (arr, kind, mask) = self.private_array(&local, &d)?;
                let i = self.value(idx)?;
                let v = self.value(value)?;
                let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![i]);
                let arr_dual = parr_dual(kind);
                return Ok(Some(Tree::new(&format!("parr.set_{kind}_{rung}"), st, vec![Tree::new(arr, ty(self, arr_dual)?, vec![]), pidx, v])));
            }
        }
        if let Some(local) = local_name(name) {
            // A store to the loop counter inside its loop is the loop's own step.
            if [&self.item_counter, &self.tile_counter, &self.tile_counter2, &self.tile_counter3].iter().any(|c| c.as_deref() == Some(local.as_str())) {
                return Ok(None);
            }
            let d = self.kingdom.dual(tree.kids[0].ty).map(|d| d.dual.clone()).unwrap_or_default();
            let row = self.register(&local, &d)?;
            let kind = row.chars().nth(4).unwrap_or('f');
            let v = self.value(&tree.kids[0])?;
            return Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), v])));
        }
        Err(format!("store {name} has no safe spelling"))
    }

    /// A naga loop as the grammar's bounded loop: the body's first statement
    /// is `if (!(counter < bound)) { break; }` (or the reverse arms), the
    /// continuing block steps the counter by one, the bound is a binding's
    /// count or a literal at most 64.
    #[allow(clippy::too_many_arguments)]
    fn looped(&mut self, body: &naga::Block, continuing: &naga::Block, break_if: Option<Handle<Expression>>, path: &mut Vec<usize>, rung: &str, cond: bool) -> Result<Option<Tree>, String> {
        if break_if.is_some() {
            return Err("a loop with a break_if (not a `for`)".into());
        }
        // The condition: the first non-Emit statement of the body is an If whose one arm is a Break.
        let first = body.iter().find(|s| !matches!(s, Statement::Emit(_))).ok_or("an empty loop")?;
        let Statement::If { accept, reject, .. } = first else { return Err("a loop whose first statement is not its test".into()) };
        let is_break = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Break));
        let is_empty = |b: &naga::Block| b.iter().all(|s| matches!(s, Statement::Emit(_)));
        let negated = if is_break(reject) && is_empty(accept) { false } else if is_break(accept) && is_empty(reject) { true } else { return Err("a loop whose test does not break".into()) };
        path.push(0);
        path.push(0);
        let root = self.root_at(path, |k| matches!(k, RootKind::Condition { .. }));
        path.pop();
        path.pop();
        let root = root.ok_or("no condition root at the loop's test")?;
        let test = tree_of(self, &root.tree)?;
        let mut c = test.kids[0].clone();
        if negated {
            // naga lowers `for (; t < m;)` into `if (t >= m) { break; }`.
            if c.name == "logic.not" && c.kids.len() == 1 {
                c = c.kids[0].clone();
            } else if c.name == "compare.ge" && c.kids.len() == 2 {
                c = Tree::new("compare.lt", c.ty, c.kids.clone());
            } else {
                return Err(format!("a loop test `if (cond) break` that is not a negated counter < bound: {}", c.to_sexpr()));
            }
        }
        if c.name != "compare.lt" || c.kids.len() != 2 {
            return Err(format!("a loop test that is not counter < bound: {}", c.to_sexpr()));
        }
        let counter = local_name(&c.kids[0].name).filter(|_| c.kids[0].kids.is_empty()).ok_or_else(|| format!("a loop test whose left side is not a local counter: {}", c.kids[0].to_sexpr()))?;
        let bound = &c.kids[1];
        let (bound_tree, kind) = if is_leaf(bound, "load.uniform.cfg.m") {
            (Tree::new("bound.m", ty(self, "u32.boundm")?, vec![]), "boundm")
        } else if is_leaf(bound, "load.uniform.cfg.d") {
            (Tree::new("bound.d", ty(self, "u32.boundd")?, vec![]), "boundd")
        } else if is_leaf(bound, "load.uniform.cfg.r_rows") {
            (Tree::new("bound.rows", ty(self, "u32.boundrows")?, vec![]), "boundrows")
        } else if let Some(v) = literal_value(bound).filter(|v| *v >= 1.0 && *v <= 64.0) {
            (Tree::literal("literal.lit64", ty(self, "u32.lit64")?, v), "lit64")
        } else {
            return Err(format!("a loop bound that is not cfg.m, cfg.d, cfg.r_rows or a literal at most 64: {}", bound.to_sexpr()));
        };
        // The step: the continuing block's store to the counter is counter + s, s a literal in 1..=64.
        let mut step: Option<u32> = None;
        let mut cpath = path.clone();
        cpath.push(1);
        let mut ci = 0usize;
        for s in continuing.iter() {
            if matches!(s, Statement::Emit(_)) {
                continue;
            }
            cpath.push(ci);
            ci += 1;
            if let Statement::Store { .. } = s {
                if let Some(r) = self.root_at(&cpath, |k| matches!(k, RootKind::Store { .. })) {
                    let t = tree_of(self, &r.tree)?;
                    if local_name(&t.name).as_deref() == Some(counter.as_str()) {
                        let v = &t.kids[0];
                        let lit = if v.name == "index.add" && v.kids.len() == 2 && v.kids.iter().any(|k| local_name(&k.name).as_deref() == Some(counter.as_str())) { v.kids.iter().find_map(literal_value) } else { None };
                        match lit {
                            Some(x) if (1.0..=64.0).contains(&x) => step = Some(x as u32),
                            _ => {
                                cpath.pop();
                                return Err(format!("a loop step that is not counter + a literal in 1..=64: {}", v.to_sexpr()));
                            }
                        }
                    }
                }
            }
            cpath.pop();
        }
        // A `loop { if (t >= m) { break; } ... t = t + s; }` steps at the body's end.
        let body_stmts_all: Vec<&Statement> = body.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        let mut last_is_step = false;
        if step.is_none() {
            if let Some(Statement::Store { .. }) = body_stmts_all.last() {
                let mut lpath = path.clone();
                lpath.push(0);
                lpath.push(body_stmts_all.len() - 1);
                if let Some(r) = self.root_at(&lpath, |k| matches!(k, RootKind::Store { .. })) {
                    let t = tree_of(self, &r.tree)?;
                    if local_name(&t.name).as_deref() == Some(counter.as_str()) {
                        let v = &t.kids[0];
                        let lit = if v.name == "index.add" && v.kids.len() == 2 && v.kids.iter().any(|k| local_name(&k.name).as_deref() == Some(counter.as_str())) { v.kids.iter().find_map(literal_value) } else { None };
                        if let Some(x) = lit.filter(|x| (1.0..=64.0).contains(x)) {
                            step = Some(x as u32);
                            last_is_step = true;
                        }
                    }
                }
            }
        }
        let step = step.ok_or_else(|| format!("a loop whose continuing block or last statement does not step its counter {counter} by a literal in 1..=64"))?;
        // The rung of the body and the loop row.
        // A tile loop's depth is the number of tile counters already in effect;
        // its body is one rung deeper (L0 and L1 both open at L2).
        let depth = [&self.tile_counter, &self.tile_counter2, &self.tile_counter3].iter().filter(|c| c.is_some()).count() as u8 + 1;
        let tile = if step == 1 { "tile" } else { "tile_strided" };
        let (body_rung, row_name, out_rung) = match (kind, rung) {
            ("lit64", "L0" | "L1") => ("L2".to_string(), format!("loop.{tile}_{rung}"), rung),
            ("lit64", "L2") => ("L3".to_string(), format!("loop.{tile}_L2"), "L2"),
            ("lit64", "L3") => ("L4".to_string(), format!("loop.{tile}_L3"), "L3"),
            ("lit64", "C0") => ("C1".to_string(), format!("loop.{tile}_C0"), "C0"),
            ("lit64", "C1") => ("C2".to_string(), format!("loop.{tile}_C1"), "C1"),
            ("lit64", "C2") => ("C3".to_string(), format!("loop.{tile}_C2"), "C2"),
            ("lit64", "C3") => ("C4".to_string(), format!("loop.{tile}_C3"), "C3"),
            ("lit64", _) => return Err(format!("a tile loop at rung {rung} (too deep)")),
            (b, "L0") if step == 1 => ("L1".to_string(), format!("loop.items_{b}"), "L0"),
            (b, "L0") => ("L1".to_string(), format!("loop.items_strided_{b}"), "L0"),
            (_, _) => return Err(format!("a long loop at rung {rung} (only at the top level)")),
        };
        let _ = cond;
        if kind == "lit64" && depth > 3 {
            return Err("a fourth nested tile loop".into());
        }
        let saved = (self.item_counter.clone(), self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone());
        if body_rung == "L1" {
            self.item_counter = Some(counter.clone());
        } else {
            match depth {
                1 => self.tile_counter = Some(counter.clone()),
                2 => self.tile_counter2 = Some(counter.clone()),
                _ => self.tile_counter3 = Some(counter.clone()),
            }
        }
        // The body after its test, at the body rung: the test is statement 0
        // of the body block (path + [0]); the rest are numbered from 1.
        let mut rest: Vec<Statement> = body.iter().filter(|s| !matches!(s, Statement::Emit(_))).skip(1).cloned().collect();
        if last_is_step {
            rest.pop();
        }
        let rest = naga::Block::from_vec(rest);
        path.push(0);
        let body_stmts = self.walk(&rest, path, &body_rung, 1);
        path.pop();
        self.item_counter = saved.0;
        self.tile_counter = saved.1;
        self.tile_counter2 = saved.2;
        self.tile_counter3 = saved.3;
        let body_stmts = body_stmts?;
        let block = self.block(body_stmts, &body_rung)?;
        let st = self.stmt_ty(out_rung)?;
        let kids = if step != 1 { vec![bound_tree, Tree::literal("literal.lit64", ty(self, "u32.lit64")?, step as f32), block] } else { vec![bound_tree, block] };
        Ok(Some(Tree::new(&row_name, st, kids)))
    }

}

fn stmt_name(s: &Statement) -> &'static str {
    match s {
        Statement::Emit(_) => "Emit",
        Statement::Block(_) => "Block",
        Statement::If { .. } => "If",
        Statement::Switch { .. } => "Switch",
        Statement::Loop { .. } => "Loop",
        Statement::Break => "Break",
        Statement::Continue => "Continue",
        Statement::Return { .. } => "Return",
        Statement::Kill => "Kill",
        Statement::Barrier(_) => "Barrier",
        Statement::Store { .. } => "Store",
        Statement::ImageStore { .. } => "ImageStore",
        Statement::Atomic { .. } => "Atomic",
        Statement::WorkGroupUniformLoad { .. } => "WorkGroupUniformLoad",
        Statement::Call { .. } => "Call",
        Statement::RayQuery { .. } => "RayQuery",
        Statement::SubgroupBallot { .. } | Statement::SubgroupGather { .. } | Statement::SubgroupCollectiveOperation { .. } => "Subgroup",
    }
}

/// The `// grid x y bx by` header.
fn grid_header(source: &str) -> Option<(u32, u32, u32, u32)> {
    source.lines().map(str::trim).find_map(|l| {
        let rest = l.strip_prefix("// grid ")?;
        let v: Vec<u32> = rest.split_whitespace().map(|t| t.parse().ok()).collect::<Option<Vec<u32>>>()?;
        (v.len() == 4).then_some((v[0], v[1], v[2], v[3]))
    })
}

/// The entry function's statements by path, one line each (for reading a refusal).
pub fn statement_outline(source: &str) -> Result<Vec<String>, String> {
    let kernel = read(source)?;
    let f: &Function = &kernel.module.entry_points.first().ok_or("no entry point")?.function;
    fn walk(b: &naga::Block, path: &mut Vec<usize>, out: &mut Vec<String>) {
        let mut i = 0;
        for s in b.iter() {
            if matches!(s, Statement::Emit(_)) {
                continue;
            }
            path.push(i);
            i += 1;
            out.push(format!("{path:?} {}", stmt_name(s)));
            match s {
                Statement::Block(x) => walk(x, path, out),
                Statement::If { accept, reject, .. } => {
                    path.push(0);
                    walk(accept, path, out);
                    path.pop();
                    path.push(1);
                    walk(reject, path, out);
                    path.pop();
                }
                Statement::Loop { body, continuing, .. } => {
                    path.push(0);
                    walk(body, path, out);
                    path.pop();
                    path.push(1);
                    walk(continuing, path, out);
                    path.pop();
                }
                _ => {}
            }
            path.pop();
        }
    }
    let mut out = Vec::new();
    walk(&f.body, &mut Vec::new(), &mut out);
    Ok(out)
}

/// Read `source` into the safe grammar, or say why not.
pub fn read_safe(source: &str, kingdom: &WgslKingdom) -> Result<SafeRead, String> {
    let grid = grid_header(source).ok_or("no `// grid x y bx by` header")?;
    let kernel = read(source)?;
    if kernel.module.functions.iter().next().is_some() {
        return Err("the kernel has helper functions (no safe spelling yet)".into());
    }
    let f = kernel.functions.iter().find(|f| f.entry_point).ok_or("no entry point")?;
    let naga_fn: &Function = &kernel.module.entry_points.first().ok_or("no entry point")?.function;
    // Roots by statement path; lets by name; duals by text (one form, else untyped).
    let mut roots: HashMap<Vec<usize>, Vec<&Root>> = HashMap::new();
    let mut lets: HashMap<String, Node> = HashMap::new();
    let mut inits: Vec<&Root> = Vec::new();
    for r in &f.roots {
        match &r.kind {
            RootKind::Let { name } => {
                if let Node::App { kids, .. } = &r.tree {
                    lets.insert(name.clone(), kids[0].clone());
                }
            }
            RootKind::Init { .. } => inits.push(r),
            _ => roots.entry(r.path.clone()).or_default().push(r),
        }
    }
    let types = infer_function(f, kingdom);
    let duals: BTreeMap<String, Option<Ty>> = types.by_text.iter().map(|(text, set)| (text.clone(), if set.len() == 1 { set.iter().next().copied().flatten() } else { None })).collect();
    let mut ctx = Ctx { kingdom, duals: &duals, lets: &lets, roots: &roots, item_counter: None, tile_counter: None, tile_counter2: None, tile_counter3: None, registers: BTreeMap::new(), next_reg: BTreeMap::new(), guard: None, tail: true, arrays: BTreeMap::new(), parrs: BTreeMap::new(), grid, notes: Vec::new() };
    // Register initialisations first, in root order.
    let mut stmts = Vec::new();
    for r in inits {
        let t = tree_of(&mut ctx, &r.tree)?;
        if let Some(st) = ctx.store(&t, "L0").map_err(|e| format!("at init {}: {e}", r.tree.to_sexpr()))? {
            stmts.push(st);
        }
    }
    let body = ctx.walk(&naga_fn.body, &mut Vec::new(), "L0", 0)?;
    stmts.extend(body);
    let block = ctx.block(stmts, "L0")?;
    let g = |n: &str, v: u32| Tree::literal(n, dual_ty(kingdom, if n == "literal.wgdim" { "u32.wgdim" } else { "u32.blk" }).expect("dual"), v as f32);
    let grid_tree = Tree::new("grid.of", dual_ty(kingdom, "grid.grid")?, vec![g("literal.wgdim", grid.0), g("literal.wgdim", grid.1), g("literal.blk", grid.2), g("literal.blk", grid.3)]);
    let tree = Tree::new("kernel.entry_safe", dual_ty(kingdom, "kernel.safe")?, vec![grid_tree, block]);
    // Row-consistent: every node under the dual its parent's row demands.
    let per_kernel = crate::wgsl::safe_karva::per_kernel_rows(kingdom, &crate::wgsl::render::gram_role());
    let tree = crate::wgsl::safe_karva::conform(&tree, tree.ty, kingdom, &per_kernel)?;
    Ok(SafeRead { tree, notes: ctx.notes, registers: ctx.registers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::render::{gram_role, validate_text, Renderer};
    use crate::wgsl::table::FunctionTable;

    const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");
    const BLOCKED: &str = include_str!("../../kingdoms/wgsl/gram/blocked.wgsl");

    /// The shipped kernel reads into the safe grammar: its guard becomes a
    /// guard on the store, its loop is the bounded item loop, its loads are
    /// phi indices and its store the owned cell; the text renders and
    /// validates.
    #[test]
    fn the_shipped_kernel_reads_into_the_safe_grammar_and_renders() {
        let k = WgslKingdom::load_safe();
        let src = format!("// grid 16 16 1 1\n{SHIPPED}");
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        let sx = r.tree.to_sexpr();
        assert!(sx.contains("loop.items_boundm"), "{sx}");
        assert!(sx.contains("load.phi_safe (index.add (index.mul (safe.counter_index safe.counter) (index.of_count load.cfg_d))"), "{sx}");
        assert!(sx.contains("store.h_C0 (safe.cell_of (safe.thread_row"), "{sx}");
        assert!(r.notes.iter().any(|n| n.contains("guard")), "{:?}", r.notes);
        let (t, role) = (FunctionTable::safe(), gram_role());
        let out = Renderer::new(&k, &t, &role).kernel(&r.tree).unwrap();
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        assert!(out.text.contains("for (t = 0u; t < cfg.m; t = t + 1u)"), "{}", out.text);
    }

    /// A kernel with a helper function is refused by name.
    #[test]
    fn a_kernel_with_a_helper_function_is_refused() {
        let k = WgslKingdom::load_safe();
        let err = read_safe(BLOCKED, &k).err().expect("refused");
        assert!(err.contains("helper function"), "{err}");
    }
}
