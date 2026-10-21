//! KERNEL PARAMETERS (`docs/PLAN_constrained_mutation.md` task 0, sections
//! 2.3 and 3.1): the tuning knobs of a safe kernel as first-class, closed
//! nodes, and the positions that depend on them as references.
//!
//! A knob is DECLARED once under the kernel root: the grid's workgroup
//! extent and block factors (`grid.of`), the tile depth T (`grid.with_ktile`)
//! and each workgroup array's length (`grid.with_wg`). A position that depends
//! on a knob (a stride `wid.x * 64u`, a loop bound, a mask, an array's
//! length) is a REFERENCE leaf `literal.p_<knob>[_log2|_m1]` whose payload is
//! an integer multiplier of the knob in units of 1/[`Q`]. [`resolve`] writes
//! every reference as the value the kernel's own declarations give, so a
//! change of one knob rewrites every reference in step, and a reference
//! grafted into another kernel reads that kernel's knob. A reference names
//! no counter, register, binding or local.
//!
//! The knobs are named by the output's axes, not by x and y (the layout knob
//! `grid.of_rows_x` swaps which thread axis carries the rows).

use std::collections::BTreeSet;

use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::render::{grid_of, safe_base, Grid, Tree};

/// The payload unit of a reference: a payload of `Q` is the knob itself.
pub const Q: u32 = 256;

/// A knob a reference can name: one declaration or a product of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Knob {
    /// The workgroup's extent along the output's columns.
    Wcol,
    /// The workgroup's extent along the output's rows.
    Wrow,
    /// The column block factor (cells a thread owns along a row).
    Bcol,
    /// The row block factor.
    Brow,
    /// Columns a workgroup covers: wcol * bcol.
    ColSpan,
    /// Rows a workgroup covers: wrow * brow.
    RowSpan,
    /// Threads in a workgroup: wcol * wrow.
    Threads,
    /// The tile depth T: items a pass of the item loop takes.
    Ktile,
}

impl Knob {
    pub const ALL: [Knob; 8] = [Knob::Wcol, Knob::Wrow, Knob::Bcol, Knob::Brow, Knob::ColSpan, Knob::RowSpan, Knob::Threads, Knob::Ktile];

    pub fn name(self) -> &'static str {
        match self {
            Knob::Wcol => "wcol",
            Knob::Wrow => "wrow",
            Knob::Bcol => "bcol",
            Knob::Brow => "brow",
            Knob::ColSpan => "colspan",
            Knob::RowSpan => "rowspan",
            Knob::Threads => "threads",
            Knob::Ktile => "ktile",
        }
    }
}

/// What a reference writes of its scaled knob v = knob * payload / Q.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Form {
    /// v itself.
    Plain,
    /// log2(v): a shift.
    Log2,
    /// v - 1: a mask.
    Minus1,
}

/// Every reference row name, by knob (in [`Knob::ALL`]'s order) and form
/// (plain, `_log2`, `_m1`).
pub const REF_NAMES: [[&str; 3]; 8] = [
    ["literal.p_wcol", "literal.p_wcol_log2", "literal.p_wcol_m1"],
    ["literal.p_wrow", "literal.p_wrow_log2", "literal.p_wrow_m1"],
    ["literal.p_bcol", "literal.p_bcol_log2", "literal.p_bcol_m1"],
    ["literal.p_brow", "literal.p_brow_log2", "literal.p_brow_m1"],
    ["literal.p_colspan", "literal.p_colspan_log2", "literal.p_colspan_m1"],
    ["literal.p_rowspan", "literal.p_rowspan_log2", "literal.p_rowspan_m1"],
    ["literal.p_threads", "literal.p_threads_log2", "literal.p_threads_m1"],
    ["literal.p_ktile", "literal.p_ktile_log2", "literal.p_ktile_m1"],
];

/// The row name of a reference.
pub fn ref_name(knob: Knob, form: Form) -> &'static str {
    let k = Knob::ALL.iter().position(|&x| x == knob).unwrap_or(0);
    REF_NAMES[k][match form {
        Form::Plain => 0,
        Form::Log2 => 1,
        Form::Minus1 => 2,
    }]
}

/// The knob and form a row name references, if it is a reference.
pub fn parse_ref(name: &str) -> Option<(Knob, Form)> {
    let rest = name.strip_prefix("literal.p_")?;
    let (stem, form) = if let Some(s) = rest.strip_suffix("_log2") {
        (s, Form::Log2)
    } else if let Some(s) = rest.strip_suffix("_m1") {
        (s, Form::Minus1)
    } else {
        (rest, Form::Plain)
    };
    Knob::ALL.iter().find(|k| k.name() == stem).map(|&k| (k, form))
}

/// The knobs a kernel declares: its grid as the renderer reads it, and its
/// tile depth when it declares one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Knobs {
    pub grid: Grid,
    pub ktile: Option<u32>,
}

impl Knobs {
    /// The knobs of a kernel root (`kernel.entry_safe`, `kernel.entry`, or an
    /// own terminal, whose grid is the renderer's default).
    pub fn of_root(root: &Tree) -> Result<Knobs, String> {
        let g = match root.kids.first() {
            Some(g) if root.name.starts_with("kernel.") => g,
            _ => return Ok(Knobs { grid: crate::wgsl::render::Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, ktile: None }),
        };
        Ok(Knobs { grid: grid_of(g)?, ktile: ktile_of(g) })
    }

    /// The value of a knob. A kernel that declares no tile depth steps its
    /// items one at a time: its tile depth is 1 (so a reference to T in any
    /// gene renders; a random gene draws one anywhere).
    pub fn value(&self, k: Knob) -> u32 {
        let g = &self.grid;
        let (wcol, wrow) = (g.col_axis().1, g.row_axis().1);
        match k {
            Knob::Wcol => wcol,
            Knob::Wrow => wrow,
            Knob::Bcol => g.bx,
            Knob::Brow => g.by,
            Knob::ColSpan => wcol * g.bx,
            Knob::RowSpan => wrow * g.by,
            Knob::Threads => g.x * g.y,
            Knob::Ktile => self.ktile.unwrap_or(1),
        }
    }
}

/// The tile depth a grid declares (`grid.with_ktile`, under any grid wrapper).
pub fn ktile_of(g: &Tree) -> Option<u32> {
    match (g.name.as_str(), g.kids.as_slice()) {
        ("grid.with_ktile", [_, t]) => Some(if t.name.starts_with("literal.") && t.konst.is_finite() { (t.konst.round() as i64).clamp(1, 4096) as u32 } else { 1 }),
        ("grid.with_wg", [inner, _]) => ktile_of(inner),
        _ => None,
    }
}

/// The value a reference writes when its knob is `knob` and its payload `q`:
/// v = knob * q / Q rounded, then its form. Used to render; a payload a
/// point mutation drew is written as the nearest value.
pub fn value_of(knob: u32, q: f64, form: Form) -> f64 {
    let q = if q.is_finite() { q.max(0.0) } else { 0.0 };
    let v = (f64::from(knob) * q / f64::from(Q)).round();
    match form {
        Form::Plain => v,
        Form::Log2 => v.max(1.0).log2().round(),
        Form::Minus1 => v.max(1.0) - 1.0,
    }
}

/// The value a reference writes, only when it is exact: knob * q a whole
/// multiple of Q, at least one, and a power of two under `_log2`. The knob
/// mutation's join: a new knob value every reference of the kernel writes
/// exactly.
pub fn exact_value(knob: u32, q: f64, form: Form) -> Option<u32> {
    if !q.is_finite() || q < 1.0 || q.fract() != 0.0 {
        return None;
    }
    let prod = u64::from(knob) * q as u64;
    if !prod.is_multiple_of(u64::from(Q)) {
        return None;
    }
    let v = u32::try_from(prod / u64::from(Q)).ok()?;
    match form {
        _ if v == 0 => None,
        Form::Plain => Some(v),
        Form::Log2 => v.is_power_of_two().then(|| v.trailing_zeros()),
        Form::Minus1 => Some(v - 1),
    }
}

/// Whether a tree holds a reference.
pub fn has_refs(t: &Tree) -> bool {
    parse_ref(&t.name).is_some() || t.kids.iter().any(has_refs)
}

/// The tree with every reference's payload replaced by the value its
/// kernel's knobs give (the name and dual unchanged: the renderer writes a
/// reference by its dual's form). An error is a grid row the renderer
/// cannot read.
pub fn resolve(root: &Tree) -> Result<Tree, String> {
    if !has_refs(root) {
        return Ok(root.clone());
    }
    // A module of entry points (`kernel.module_N`, v24's ingest and search):
    // each entry's references are its own grid's; the module has no grid.
    if root.name.starts_with("kernel.module") {
        return Ok(Tree { kids: root.kids.iter().map(resolve).collect::<Result<_, _>>()?, ..root.clone() });
    }
    let knobs = Knobs::of_root(root)?;
    fn walk(t: &Tree, knobs: &Knobs) -> Tree {
        if let Some((k, form)) = parse_ref(&t.name) {
            return Tree { name: t.name.clone(), ty: t.ty, konst: value_of(knobs.value(k), t.konst, form), kids: Vec::new() };
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| walk(k, knobs)).collect() }
    }
    Ok(walk(root, &knobs))
}

/// The tree with every reference written as the plain literal of its
/// dual's form at the value its kernel's knobs give (`literal.index 64` for
/// a column span of 64): what a structural rule rewrites, since its
/// rewrite changes what the references are relative to. The child's
/// read-back assigns them again.
pub fn to_literals(root: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    let resolved = resolve(root)?;
    fn walk(t: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
        if parse_ref(&t.name).is_some() {
            let form = kingdom.dual(t.ty).map(|d| d.form.clone()).ok_or_else(|| format!("{} has no dual", t.name))?;
            let name = format!("literal.{form}");
            if !kingdom.rows.iter().any(|r| r.name == name && r.output == t.ty) {
                return Err(format!("our software has a bug: no literal row {name} for {}", t.name));
            }
            return Ok(Tree::literal(&name, t.ty, t.konst));
        }
        Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| walk(k, kingdom)).collect::<Result<_, _>>()? })
    }
    walk(&resolved, kingdom)
}

/// A plain literal's whole value (not a reference).
fn lit(t: &Tree) -> Option<u32> {
    (t.kids.is_empty() && t.name.starts_with("literal.") && parse_ref(&t.name).is_none() && t.konst.is_finite() && t.konst >= 0.0 && t.konst.fract() == 0.0).then_some(t.konst as u32)
}

/// Lanes of a workgroup array leaf (`wg.va` 4, `wg.v2a` 2, `wg.a` 1).
fn wg_lanes(name: &str) -> u32 {
    if name.starts_with("wg.v2") {
        2
    } else if name.starts_with("wg.v") {
        4
    } else {
        1
    }
}

fn is_wg_write(name: &str) -> bool {
    name.starts_with("wg.set")
}

fn is_wg_read(name: &str) -> bool {
    name.starts_with("wg.get")
}

/// A tile loop of the safe grammar with a literal count as its first child
/// (`for (k = 0u; k < n; k = k + 1u)`, its unrolled form, and the stepped
/// form `k = k + s`); its body is its last child.
fn is_tile_loop(name: &str) -> bool {
    matches!(safe_base(name), "loop.tile_bounded" | "loop.unroll" | "loop.tile_strided_bounded")
}

/// Any loop inside the item loop (a tile loop, an expression-bounded one):
/// its body runs under its own counter.
fn is_inner_loop(name: &str) -> bool {
    let b = safe_base(name);
    b.starts_with("loop.") && !b.starts_with("loop.items")
}

fn is_strided_items(name: &str) -> bool {
    matches!(safe_base(name), "loop.items_strided_bounded" | "loop.items_cont_strided_bounded")
}

/// Whether any node of `t` satisfies `p`.
fn any(t: &Tree, p: &dyn Fn(&Tree) -> bool) -> bool {
    p(t) || t.kids.iter().any(|k| any(k, p))
}

/// The workgroup array writes in `t` outside any loop nested in it: the
/// array leaf's name and the write's index.
fn direct_writes<'t>(t: &'t Tree, out: &mut Vec<(String, &'t Tree)>) {
    if is_wg_write(&t.name) && t.kids.len() >= 2 {
        if let Some(a) = t.kids.first().filter(|a| a.kids.is_empty() && a.name.starts_with("wg.")) {
            out.push((a.name.clone(), &t.kids[1]));
        }
    }
    for k in &t.kids {
        if !(safe_base(&k.name).starts_with("loop.")) {
            direct_writes(k, out);
        }
    }
}

/// The counter leaf of the tile loop at nesting depth `d` (the renderer's k, k2, k3, k4).
fn tile_counter(d: usize) -> &'static str {
    ["safe.tile_counter", "safe.tile_counter2", "safe.tile_counter3", "safe.tile_counter4"][d.min(3)]
}

/// Which output axis and thread id a `index.of_col` / `index.of_row` node
/// names: (true for columns, the id's suffix wid, lid or gid).
fn axis_id(t: &Tree) -> Option<(bool, &'static str)> {
    let col = match t.name.as_str() {
        "index.of_col" => true,
        "index.of_row" => false,
        _ => return None,
    };
    let a = t.kids.first()?.name.as_str();
    let id = ["wid", "lid", "gid"].into_iter().find(|id| a.starts_with("axis.") && a.ends_with(id))?;
    Some((col, id))
}

struct Assigner<'a> {
    knobs: Knobs,
    kingdom: &'a WgslKingdom,
    /// The tile depth's references made (the grid then declares it).
    ktile_refs: usize,
    /// Workgroup arrays a tile-depth loop writes directly at an index of its
    /// counter: they hold the whole tile, their lengths scale with T.
    tiles: BTreeSet<String>,
    /// A tile-depth loop writes an array at an index held in a register:
    /// whether the array holds the tile or one pass of the loop is unknown,
    /// so T is not a knob of this kernel.
    t_unknown: bool,
    /// The epilogue's block loops open around the walk: counter leaf, knob.
    block_counters: Vec<(&'static str, Knob)>,
}

#[derive(Clone, Copy, Default)]
struct Env {
    /// Inside an item loop whose stride is the tile depth.
    in_t_items: bool,
    /// Inside a tile loop of that item loop (its tile-depth loops are the outermost).
    in_tile: bool,
    /// The lanes of the arrays the innermost enclosing loop writes directly.
    write_lanes: Option<u32>,
    /// Inside the index of a workgroup array read: the array's lanes.
    read_lanes: Option<u32>,
}

impl Assigner<'_> {
    /// `t` as a reference to `knob` in `form` at payload `q`, when that
    /// reference writes exactly `t`'s value and the kingdom has its row.
    fn try_ref(&mut self, t: &Tree, knob: Knob, form: Form, q: u32) -> Option<Tree> {
        let v = lit(t)?;
        let k = self.knobs.value(knob);
        if exact_value(k, q as f64, form) != Some(v) {
            return None;
        }
        let name = ref_name(knob, form);
        if !self.kingdom.rows.iter().any(|r| r.name == name && r.output == t.ty && r.inputs.is_empty()) {
            return None;
        }
        if knob == Knob::Ktile {
            self.ktile_refs += 1;
        }
        Some(Tree::literal(name, t.ty, q as f32))
    }

    /// The payload that makes `knob` scaled by it write `v` in `form`: v (or
    /// 2^v, v + 1) * Q / knob, when whole.
    fn payload(&self, knob: Knob, form: Form, v: u32) -> Option<u32> {
        let k = u64::from(self.knobs.value(knob));
        let scaled = match form {
            Form::Plain => u64::from(v),
            Form::Log2 => 1u64.checked_shl(v)?,
            Form::Minus1 => u64::from(v) + 1,
        };
        let num = scaled * u64::from(Q);
        (k > 0 && num.is_multiple_of(k)).then(|| u32::try_from(num / k).ok()).flatten()
    }

    /// `t` as a reference to the first knob of `knobs` whose scaled value
    /// (knob / divisor) writes it in `form`.
    fn ref_scaled(&mut self, t: &Tree, knobs: &[Knob], form: Form, divisor: u32) -> Option<Tree> {
        for &k in knobs {
            if let Some(q) = Q.is_multiple_of(divisor).then_some(Q / divisor) {
                if let Some(r) = self.try_ref(t, k, form, q) {
                    return Some(r);
                }
            }
        }
        None
    }

    fn walk(&mut self, t: &Tree, env: Env, epi_depth: Option<usize>) -> Tree {
        let base = safe_base(&t.name);
        // A strided item loop at the tile depth: its stride is the knob.
        if is_strided_items(&t.name) && t.kids.len() == 3 {
            let t_here = lit(&t.kids[1]).is_some_and(|s| Some(s) == self.knobs.ktile);
            let stride = if t_here { self.try_ref(&t.kids[1], Knob::Ktile, Form::Plain, Q).unwrap_or_else(|| t.kids[1].clone()) } else { t.kids[1].clone() };
            let inner = Env { in_t_items: t_here, in_tile: false, ..env };
            return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: vec![self.walk(&t.kids[0], env, epi_depth), stride, self.walk(&t.kids[2], inner, epi_depth)] };
        }
        if is_inner_loop(&t.name) && !t.kids.is_empty() {
            let last = t.kids.len() - 1;
            let counted = is_tile_loop(&t.name) && last >= 1;
            let mut writes = Vec::new();
            direct_writes(&t.kids[last], &mut writes);
            let lanes: BTreeSet<u32> = writes.iter().map(|(a, _)| wg_lanes(a)).collect();
            let write_lanes = if lanes.len() == 1 { lanes.into_iter().next() } else { env.write_lanes };
            let mut bound = t.kids[0].clone();
            // The outermost tile loops of a tile-depth item loop that touch the
            // workgroup tile, or step through the tile's items, run T-proportionally.
            if counted && env.in_t_items && !env.in_tile {
                let counter = tile_counter(0);
                let touches = any(&t.kids[last], &|n| is_wg_write(&n.name) || is_wg_read(&n.name));
                let items = any(&t.kids[last], &|n| n.name == "index.add" && n.kids.iter().any(|k| k.name == "safe.counter_index") && n.kids.iter().any(|k| any(k, &|m| m.name == counter)));
                if touches || items {
                    if let Some(v) = lit(&bound) {
                        if let Some(q) = self.payload(Knob::Ktile, Form::Plain, v) {
                            if let Some(r) = self.try_ref(&bound, Knob::Ktile, Form::Plain, q) {
                                bound = r;
                                // An array written at an index of this loop's counter holds the
                                // whole tile (its length scales with T); at an index without it,
                                // one pass of the loop (p120's chunk loop rewrites wg_vb[lidx]).
                                for (a, idx) in &writes {
                                    if any(idx, &|n| n.name == counter) {
                                        self.tiles.insert(a.clone());
                                    } else if any(idx, &|n| n.name.starts_with("reg.") || n.name.starts_with("parr.")) {
                                        self.t_unknown = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // The epilogue's block loops: a loop whose counter offsets the
            // thread's row (column) runs over the row (column) block.
            let mut pushed = false;
            if let Some(d) = epi_depth.filter(|_| counted) {
                let counter = tile_counter(d);
                let offsets = |prefix: &str| any(&t.kids[last], &|n| n.name.starts_with(prefix) && n.kids.iter().any(|k| any(k, &|m| m.name == counter)));
                let knob = if offsets("safe.row_plus") { Some(Knob::Brow) } else if offsets("safe.col_plus") { Some(Knob::Bcol) } else { None };
                if let Some((k, r)) = knob.and_then(|k| self.try_ref(&bound, k, Form::Plain, Q).map(|r| (k, r))) {
                    bound = r;
                    self.block_counters.push((counter, k));
                    pushed = true;
                }
            }
            let inner = Env { in_tile: env.in_t_items || env.in_tile, write_lanes, ..env };
            let mut kids: Vec<Tree> = Vec::with_capacity(t.kids.len());
            for (i, k) in t.kids.iter().enumerate() {
                kids.push(match i {
                    0 if counted => bound.clone(),
                    i if i == last => self.walk(k, inner, epi_depth.map(|d| d + 1)),
                    _ => self.walk(k, env, epi_depth),
                });
            }
            if pushed {
                self.block_counters.pop();
            }
            return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids };
        }
        // The index of a workgroup array read: its lanes are the read's.
        if is_wg_read(&t.name) && t.kids.len() == 2 {
            let lanes = wg_lanes(&t.kids[0].name);
            let inner = Env { read_lanes: Some(lanes), ..env };
            return Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: vec![t.kids[0].clone(), self.walk(&t.kids[1], inner, epi_depth)] };
        }
        let mut kids: Vec<Tree> = t.kids.iter().map(|k| self.walk(k, env, epi_depth)).collect();
        // A block loop's counter masked or clamped to the block (`pv[k & 3u]`,
        // `min(k, 3u)` in a 4-row epilogue), and stepping the workgroup's
        // extent on its axis (`16u * k`: the rows a thread owns 16 apart).
        if let ("index.and" | "index.min", [e, m]) = (base, kids.as_mut_slice()) {
            let knob = self.block_counters.iter().rev().find(|(c, _)| any(e, &|n| n.name == *c)).map(|&(_, k)| k);
            if let Some(r) = knob.and_then(|k| self.try_ref(m, k, Form::Minus1, Q)) {
                *m = r;
            }
        }
        if let ("index.mul", [a, b]) = (base, kids.as_mut_slice()) {
            for (x, l) in [(0usize, 1usize), (1, 0)] {
                let side = if x == 0 { &*a } else { &*b };
                let knob = self.block_counters.iter().rev().find(|(c, _)| any(side, &|n| n.name == *c)).map(|&(_, k)| k);
                let extent = match knob {
                    Some(Knob::Brow) => Some(Knob::Wrow),
                    Some(Knob::Bcol) => Some(Knob::Wcol),
                    _ => None,
                };
                let target = if l == 0 { &mut *a } else { &mut *b };
                if let Some(r) = extent.and_then(|k| self.try_ref(target, k, Form::Plain, Q)) {
                    *target = r;
                }
            }
        }
        match (base, kids.as_mut_slice()) {
            // Strides of the thread ids: wid by the span, lid and gid by the block.
            ("index.mul", [a, b]) => {
                for (x, l) in [(0usize, 1usize), (1, 0)] {
                    let other = if x == 0 { &*a } else { &*b };
                    let Some((col, id)) = axis_id(other) else { continue };
                    let knob = match (col, id) {
                        (true, "wid") => Knob::ColSpan,
                        (false, "wid") => Knob::RowSpan,
                        (true, _) => Knob::Bcol,
                        (false, _) => Knob::Brow,
                    };
                    let target = if l == 0 { &mut *a } else { &mut *b };
                    if let Some(r) = self.try_ref(target, knob, Form::Plain, Q) {
                        *target = r;
                    }
                }
                // The threads of a workgroup striding a flat index (lidx + 256u * k).
                for l in [0usize, 1] {
                    let other_has_tile = any(if l == 0 { &*b } else { &*a }, &|n| n.name.starts_with("safe.tile_index"));
                    let target = if l == 0 { &mut *a } else { &mut *b };
                    if other_has_tile && env.write_lanes.is_some() {
                        if let Some(r) = self.try_ref(target, Knob::Threads, Form::Plain, Q) {
                            *target = r;
                        }
                    }
                }
            }
            // A row of the tile in a workgroup read: tile counter * (span / lanes) + the thread's lid.
            ("index.add", [a, b]) if env.read_lanes.is_some() => {
                let lanes = env.read_lanes.unwrap_or(1);
                for (m, o) in [(0usize, 1usize), (1, 0)] {
                    let other = if o == 0 { &*a } else { &*b };
                    let axis = if any(other, &|n| axis_id(n).is_some_and(|(c, id)| c && id == "lid")) {
                        Some(Knob::ColSpan)
                    } else if any(other, &|n| axis_id(n).is_some_and(|(c, id)| !c && id == "lid")) {
                        Some(Knob::RowSpan)
                    } else {
                        None
                    };
                    let mul = if m == 0 { &mut *a } else { &mut *b };
                    if let (Some(knob), "index.mul", 2) = (axis, mul.name.as_str(), mul.kids.len()) {
                        for i in 0..2 {
                            if any(&mul.kids[1 - i], &|n| n.name.starts_with("safe.tile_index")) {
                                if let Some(r) = self.ref_scaled(&mul.kids[i], &[knob], Form::Plain, lanes) {
                                    mul.kids[i] = r;
                                }
                            }
                        }
                    }
                }
            }
            // The flat thread index mapped onto the tile's rows and columns:
            // >> log2(span / lanes), & (span / lanes - 1), in a loop that writes the tile.
            ("index.shr", [e, s]) | ("index.and", [e, s]) if env.write_lanes.is_some() && any(e, &|n| n.name == "index.of_lidx") => {
                let form = if base == "index.shr" { Form::Log2 } else { Form::Minus1 };
                if let Some(r) = self.ref_scaled(s, &[Knob::ColSpan, Knob::RowSpan], form, env.write_lanes.unwrap_or(1)) {
                    *s = r;
                }
            }
            // The tile's parity of the item counter (t >> log2(T)).
            ("index.shr", [e, s]) if env.in_t_items && e.name == "safe.counter_index" => {
                if let Some(r) = self.try_ref(s, Knob::Ktile, Form::Log2, Q) {
                    *s = r;
                }
            }
            _ => {}
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
    }

    /// The grid's array declarations: a tile array's length as a reference to T.
    fn decls(&mut self, g: &Tree) -> Tree {
        let mut kids: Vec<Tree> = g.kids.iter().map(|k| self.decls(k)).collect();
        if g.name.starts_with("wg.decl_") && kids.len() == 2 && self.tiles.contains(&kids[0].name) {
            if let Some(q) = lit(&kids[1]).and_then(|v| self.payload(Knob::Ktile, Form::Plain, v)) {
                if let Some(r) = self.try_ref(&kids[1], Knob::Ktile, Form::Plain, q) {
                    kids[1] = r;
                }
            }
        }
        Tree { name: g.name.clone(), ty: g.ty, konst: g.konst, kids }
    }
}

/// Whether every workgroup access of the tile-depth item loops' bodies sits
/// inside a tile loop whose count references T: a write outside one (a
/// single pass of the threads loading the tile, p290 of the saved
/// populations: 256 threads, 16 rows of 16) covers T rows only because the
/// thread count equals T times the row's width, and a read outside one
/// walks a tile whose depth the loop around it does not follow (p270's
/// compute loop `k < 32u; k = k + 2u` before stepped loops took their
/// reference): T is not a knob there.
fn tile_depth_coherent(t: &Tree) -> bool {
    fn writes_outside_t_loops(t: &Tree) -> bool {
        if is_tile_loop(&t.name) && t.kids.first().is_some_and(|b| parse_ref(&b.name).is_some_and(|(k, _)| k == Knob::Ktile)) {
            return false;
        }
        is_wg_write(&t.name) || is_wg_read(&t.name) || t.kids.iter().any(writes_outside_t_loops)
    }
    if is_strided_items(&t.name) && t.kids.len() == 3 && t.kids[1].name == ref_name(Knob::Ktile, Form::Plain) {
        return !writes_outside_t_loops(&t.kids[2]);
    }
    t.kids.iter().all(tile_depth_coherent)
}

/// The first strided item loop's stride: the kernel's tile depth T.
fn first_stride(t: &Tree) -> Option<u32> {
    if is_strided_items(&t.name) && t.kids.len() == 3 {
        return lit(&t.kids[1]);
    }
    t.kids.iter().find_map(first_stride)
}

/// THE READER'S PARAMETERS: a safe kernel tree with every position that
/// depends on a knob written as a reference to it, and the tile depth
/// declared on the grid when a reference names it. Only a literal whose
/// reference writes exactly its value is replaced, so the kernel renders
/// the same text; a kernel with nothing to reference is returned as it is.
pub fn assign(tree: &Tree, kingdom: &WgslKingdom) -> Result<Tree, String> {
    if tree.name != "kernel.entry_safe" || tree.kids.len() != 3 || has_refs(tree) {
        return Ok(tree.clone());
    }
    let grid = grid_of(&tree.kids[0])?;
    let mut ktile = first_stride(&tree.kids[1]);
    let mut a = Assigner { knobs: Knobs { grid, ktile }, kingdom, ktile_refs: 0, tiles: BTreeSet::new(), t_unknown: false, block_counters: Vec::new() };
    let mut body = a.walk(&tree.kids[1], Env::default(), None);
    // The tile depth is a knob only when the tile's loads scale with it;
    // otherwise the kernel is read without it.
    if ktile.is_some() && (a.t_unknown || !tile_depth_coherent(&body)) {
        ktile = None;
        a = Assigner { knobs: Knobs { grid, ktile }, kingdom, ktile_refs: 0, tiles: BTreeSet::new(), t_unknown: false, block_counters: Vec::new() };
        body = a.walk(&tree.kids[1], Env::default(), None);
    }
    let epilogue = a.walk(&tree.kids[2], Env::default(), Some(0));
    let mut g = a.decls(&tree.kids[0]);
    if a.ktile_refs > 0 {
        let t = ktile.ok_or("our software has a bug: a tile-depth reference without a tile depth")?;
        let lit_ty = crate::wgsl::render::dual_ty(kingdom, "u32.lit4096")?;
        let decl = |inner: Tree| Tree::new("grid.with_ktile", inner.ty, vec![inner, Tree::literal("literal.lit4096", lit_ty, t as f32)]);
        g = if g.name == "grid.with_wg" && g.kids.len() == 2 {
            let mut kids = g.kids.clone();
            kids[0] = decl(kids[0].clone());
            Tree { name: g.name.clone(), ty: g.ty, konst: g.konst, kids }
        } else {
            decl(g)
        };
    }
    Ok(Tree { name: tree.name.clone(), ty: tree.ty, konst: tree.konst, kids: vec![g, body, epilogue] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::render::dual_ty;

    /// A MODULE'S REFERENCES RESOLVE ENTRY BY ENTRY: every v24 run-1 seed
    /// that reads (a `kernel.module_2` of ingest and search, each entry its
    /// own grid) is written as literals (`to_literals`, what the legality
    /// checker and the rewrites take) with no reference left; before this,
    /// a module root's first child, an entry, was taken for its grid ("is
    /// not a grid row") and every check of such a kernel failed.
    #[test]
    fn a_modules_references_resolve_entry_by_entry() {
        let k = WgslKingdom::load_safe();
        let mut modules = 0;
        for (name, text) in crate::wgsl::fixture_texts::set("v24_run1_seeds") {
            let Ok(r) = crate::wgsl::safe_reader::read_safe(&text, &k) else { continue };
            if !r.tree.name.starts_with("kernel.module") || !has_refs(&r.tree) {
                continue;
            }
            let plain = to_literals(&r.tree, &k).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(!has_refs(&plain), "{name}: a reference left");
            modules += 1;
        }
        assert!(modules >= 10, "{modules} module seeds with references");
    }

    /// THE READER'S PARAMETERS CHANGE NO TEXT: every corpus kernel and
    /// every kernel of the saved population popB_g080 that reads renders
    /// the same text with its knobs as references as with its literals, and
    /// a register-blocked population kernel declares its tile depth with
    /// references to it.
    #[test]
    fn a_parameterised_kernel_renders_its_literal_text() {
        use crate::wgsl::safe_karva::render_tree_with;
        use crate::wgsl::safe_reader::{read_safe, read_safe_literal};
        let k = WgslKingdom::load_safe();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for dir in ["tests/corpus/safewgsl", "logs/HGP/001/popB_g080/texts"] {
            if let Ok(rd) = std::fs::read_dir(root.join(dir)) {
                files.extend(rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")));
            }
        }
        files.sort();
        let (mut read, mut with_t) = (0usize, 0usize);
        for f in &files {
            let src = std::fs::read_to_string(f).unwrap();
            let (Ok(lit), Ok(par)) = (read_safe_literal(&src, &k), read_safe(&src, &k)) else {
                assert_eq!(read_safe_literal(&src, &k).is_ok(), read_safe(&src, &k).is_ok(), "{}: the two reads disagree", f.display());
                continue;
            };
            read += 1;
            assert_eq!(render_tree_with(&lit.tree, &k, &crate::wgsl::render::gram_role()).unwrap().text, render_tree_with(&par.tree, &k, &crate::wgsl::render::gram_role()).unwrap().text, "{}", f.display());
            if ktile_of(&par.tree.kids[0]).is_some() {
                with_t += 1;
                assert!(has_refs(&par.tree.kids[1]), "{}", f.display());
            }
        }
        println!("{read} of {} kernels read; {with_t} declare a tile depth", files.len());
        assert!(read >= 100, "{read}");
        assert!(with_t > 0);
    }

    /// THE TILE DEPTH IS A KNOB ONLY WHERE THE TILE SCALES WITH IT: three
    /// saved kernels of reserve2 whose tile-depth children computed wrong
    /// values on the device (2026-10-12) before the reader learned why.
    #[test]
    fn the_tile_depth_is_declared_only_where_the_tile_scales_with_it() {
        use crate::wgsl::knob::{set, Site};
        use crate::wgsl::safe_karva::render_tree_with;
        use crate::wgsl::safe_reader::read_safe;
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/reserve2/texts");
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok().map(|src| read_safe(&src, &k).unwrap().tree);
        // wg_vb loaded by one unlooped pass of 256 threads, 16 rows of 16: no tile depth.
        let Some(single_pass) = read("0.1692_fit_A2_stopped_1554Z_g000_r9_5825fe.wgsl") else { return };
        assert_eq!(ktile_of(&single_pass.kids[0]), None);
        // The compute loop `k < 32u; k = k + 2u` follows T; the loop nested in it does not.
        let stepped = read("0.1607_fit_A2_stopped_1400Z_g001_r4_9a65d4.wgsl").unwrap();
        assert_eq!(ktile_of(&stepped.kids[0]), Some(32));
        let text = render_tree_with(&set(&stepped, &Site::Ktile, 8), &k, &crate::wgsl::render::gram_role()).unwrap().text;
        assert!(text.contains("for (k = 0u; k < 8u; k = k + 2u)") && text.contains("for (k2 = 0u; k2 < 4u; k2 = k2 + 1u)"), "{text}");
        // A chunk loop rewriting wg_vb[lidx] on every pass: wg_vb holds one pass, its length stays.
        let chunked = read("0.1208_fit_A2_g001_r9_2ddca4.wgsl").unwrap();
        assert_eq!(ktile_of(&chunked.kids[0]), Some(128));
        let text = render_tree_with(&set(&chunked, &Site::Ktile, 256), &k, &crate::wgsl::render::gram_role()).unwrap().text;
        assert!(text.contains("var<workgroup> wg_vb: array<vec4<f32>, 512>;") && text.contains("t = t + 256u"), "{text}");
    }

    #[test]
    fn a_kernel_written_with_plain_literals_renders_its_text() {
        use crate::wgsl::safe_karva::render_tree_with;
        use crate::wgsl::safe_reader::read_safe;
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/HGP/001/popB_g080/texts");
        let Ok(rd) = std::fs::read_dir(&dir) else { return };
        for f in rd.flatten().map(|e| e.path()) {
            let t = read_safe(&std::fs::read_to_string(&f).unwrap(), &k).unwrap().tree;
            let plain = to_literals(&t, &k).unwrap();
            assert!(!has_refs(&plain), "{}", f.display());
            assert_eq!(render_tree_with(&t, &k, &crate::wgsl::render::gram_role()).unwrap().text, render_tree_with(&plain, &k, &crate::wgsl::render::gram_role()).unwrap().text, "{}", f.display());
        }
    }

    #[test]
    fn a_reference_name_round_trips() {
        for k in Knob::ALL {
            for f in [Form::Plain, Form::Log2, Form::Minus1] {
                assert_eq!(parse_ref(ref_name(k, f)), Some((k, f)));
                let suffix = match f {
                    Form::Plain => "",
                    Form::Log2 => "_log2",
                    Form::Minus1 => "_m1",
                };
                assert_eq!(ref_name(k, f), format!("literal.p_{}{suffix}", k.name()));
            }
        }
        assert_eq!(parse_ref("literal.index"), None);
        assert_eq!(parse_ref("literal.p_nothing"), None);
    }

    #[test]
    fn every_reference_has_its_rows_in_the_kingdom() {
        let k = WgslKingdom::load_safe();
        for knob in Knob::ALL {
            for (f, duals) in [(Form::Plain, &["u32.index", "u32.lit64", "u32.lit4096", "u32.wglen"][..]), (Form::Log2, &["u32.index"][..]), (Form::Minus1, &["u32.index"][..])] {
                for d in duals {
                    let ty = dual_ty(&k, d).unwrap();
                    assert!(k.rows.iter().any(|r| r.name == ref_name(knob, f) && r.output == ty && r.inputs.is_empty()), "{} -> {d}", ref_name(knob, f));
                }
            }
        }
        assert!(k.rows.iter().any(|r| r.name == "grid.with_ktile" && r.inputs.len() == 2));
    }

    #[test]
    fn exact_values_are_whole_multiples_only() {
        assert_eq!(exact_value(32, 256.0, Form::Plain), Some(32));
        assert_eq!(exact_value(32, 16.0, Form::Plain), Some(2));
        assert_eq!(exact_value(8, 16.0, Form::Plain), None, "8 / 16 is not whole");
        assert_eq!(exact_value(64, 64.0, Form::Log2), Some(4));
        assert_eq!(exact_value(48, 64.0, Form::Log2), None, "12 is not a power of two");
        assert_eq!(exact_value(64, 64.0, Form::Minus1), Some(15));
        assert_eq!(exact_value(64, 0.0, Form::Plain), None);
        assert_eq!(value_of(64, 64.0, Form::Minus1), 15.0);
        assert_eq!(value_of(64, 64.0, Form::Log2), 4.0);
    }

    #[test]
    fn a_reference_resolves_against_its_own_kernels_knobs() {
        let k = WgslKingdom::load_safe();
        let t = |d: &str| dual_ty(&k, d).unwrap();
        let grid = |x: f32, y: f32, bx: f32, by: f32, ktile: Option<f32>| {
            let g = Tree::new("grid.of", t("grid.grid"), vec![Tree::literal("literal.wgdim", t("u32.wgdim"), x), Tree::literal("literal.wgdim", t("u32.wgdim"), y), Tree::literal("literal.blk", t("u32.blk"), bx), Tree::literal("literal.blk", t("u32.blk"), by)]);
            match ktile {
                Some(v) => Tree::new("grid.with_ktile", t("grid.grid"), vec![g, Tree::literal("literal.lit4096", t("u32.lit4096"), v)]),
                None => g,
            }
        };
        let refs = Tree::new("index.add", t("u32.index"), vec![Tree::literal("literal.p_colspan", t("u32.index"), 256.0), Tree::literal("literal.p_ktile_m1", t("u32.index"), 128.0)]);
        let root = |g: Tree| Tree::new("kernel.entry_safe", t("kernel.safe"), vec![g, refs.clone(), crate::wgsl::render::own_leaf(&k, t("block.O0")).unwrap()]);
        let r = resolve(&root(grid(16.0, 8.0, 4.0, 8.0, Some(32.0)))).unwrap();
        assert_eq!((r.kids[1].kids[0].konst, r.kids[1].kids[1].konst), (64.0, 15.0));
        let r = resolve(&root(grid(8.0, 8.0, 2.0, 8.0, Some(64.0)))).unwrap();
        assert_eq!((r.kids[1].kids[0].konst, r.kids[1].kids[1].konst), (16.0, 31.0));
        // No tile depth declared: the items are stepped one at a time, T = 1.
        let r = resolve(&root(grid(8.0, 8.0, 2.0, 8.0, None))).unwrap();
        assert_eq!((r.kids[1].kids[0].konst, r.kids[1].kids[1].konst), (16.0, 0.0));
    }
}
