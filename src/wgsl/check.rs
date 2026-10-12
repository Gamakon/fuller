//! The layered legality checker of the safeWGSL kingdom
//! (`docs/PLAN_constrained_mutation.md` §1, review finding 1): it runs after
//! every rewrite of a typed tree and names the first layer that fails.
//! Joins, patterns and table lookups choose where a rule may fire; this
//! module is the proof that the child is legal before it reaches the GPU.
//!
//! | layer | what is proved |
//! |---|---|
//! | 1 typed tree | every node is a row of the kingdom (or the role) at its signature (a loop counter read outside its loop is the constant 0, as the rows say and the renderer writes) |
//! | 2 structural grammar | the root is the safe kernel; a store to `h`, and a read of `h`, only at the thread's owned cell (a fallback row, column or cell renders the thread's own); a barrier never inside an `if`; loops only from the bounded constructors, no item loop in an item loop, at most four tile loops deep |
//! | 3 range | every workgroup and private array index in its array's range (by the mask its row writes, or by its value at the worst shape, d 8,192, m 65,536, the band 8,192 rows); no `u32` wrap in index arithmetic anywhere (Codex's review of Andrew's decision, 2026-10-12): `store_cell_wraps` in a store's cell, `index_wraps` elsewhere |
//! | 4 barrier uniformity | a barrier only where every condition it depends on (an `if`, a loop's trip count, a `break` or `continue` guard, an earlier conditional `return`) is workgroup-uniform |
//! | 5 adapter | the workgroup's shape and its workgroup memory within the target's limits; subgroup and float features the target has |
//! | 6 bounded loops | every loop's trip count bounded: a literal, a binding, or a computed bound, which renders capped at MAX_LOOP_TRIP (2026-10-12: a uniform one too) and so is known; no `u32` wrap in the arithmetic of a loop's bound (a wrapped bound runs the loop to its cap) |
//!
//! Uniform values (layer 4) follow WGSL's rule, conservatively: literals, the
//! uniform buffer `cfg`, `workgroup_id`, `num_workgroups`, the subgroup size
//! and count, read-only storage (`phi`) at a uniform index, the counter of a
//! loop whose bound is uniform, and a register or private array every
//! assignment of which is uniform under uniform control (a fixpoint).
//! Thread ids, workgroup memory, `h` (read-write) and subgroup operations
//! are not uniform.
//!
//! The worst shape is the cost estimator's ([`super::render::WORST_D`],
//! [`super::render::WORST_M`], [`super::render::WORST_ROWS`]); a value is
//! KNOWN there when every leaf is a literal, a binding, a thread id (its
//! range from the grid's dispatch at that shape; a subgroup lane within
//! WGSL's 4..=128) or a counter of a loop with a known bound, narrowed by
//! the u32 comparisons of the enclosing branch conditions over stateless
//! expressions. A register or a load is not known; arithmetic over it is not
//! judged for overflow, and an array index over it is judged by its mask.
//!
//! A wrap is illegal wherever it lands (Codex's review, 2026-10-12); where
//! it lands names the reason. It flows up through arithmetic rows to the
//! value they make, and stops at a load, an access, a register, a conversion,
//! or a clamped operand (either side of a `min` or an `&`; the value of
//! `clamp_to`, `clamp` and `%`; a row's or column's offset, clamped to the
//! block): in a store's cell index `store_cell_wraps` (layer 3), in a loop's
//! bound `loop_bound_wraps` (layer 6), anywhere else `index_wraps` (layer 3).
//! The kingdom cannot spell a difference that wraps (a raw difference is
//! `u32.indexw`, a uniform difference, product or shift `u32.uniformw`, and
//! no row takes either); a cell is the thread's row and column with clamped
//! offsets, and an expression bound's value is a `u32.xbound`, whose rows
//! cannot pass their operands. A sum, product or left shift of index values
//! is not typed by its range: this layer fails one that wraps.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use naga::valid::Capabilities;

use crate::geneframe::Ty;
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::render::{array_len, dual_ty, grid_of, parr_var, register_name, safe_base, snap, wgmem_var, workgroup_dim, workgroup_plan, Grid, Role, Tree, BLK, UNROLL, WORST_D, WORST_M, WORST_ROWS};
use crate::wgsl::target::Target;

/// The checker's layers, in the order they run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    Typed = 1,
    Grammar = 2,
    Range = 3,
    Uniformity = 4,
    Adapter = 5,
    Bounded = 6,
}

impl Layer {
    pub const ALL: [Layer; 6] = [Layer::Typed, Layer::Grammar, Layer::Range, Layer::Uniformity, Layer::Adapter, Layer::Bounded];

    pub fn number(self) -> u8 {
        self as u8
    }

    pub fn name(self) -> &'static str {
        match self {
            Layer::Typed => "typed tree",
            Layer::Grammar => "structural grammar",
            Layer::Range => "range",
            Layer::Uniformity => "barrier uniformity",
            Layer::Adapter => "adapter limits",
            Layer::Bounded => "bounded loops",
        }
    }
}

/// The first failure: its layer, a short reason name (for counting) and the
/// node or value it was found at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Illegal {
    pub layer: Layer,
    pub reason: &'static str,
    pub detail: String,
}

impl fmt::Display for Illegal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "layer {} ({}): {}: {}", self.layer.number(), self.layer.name(), self.reason, self.detail)
    }
}

fn illegal(layer: Layer, reason: &'static str, detail: String) -> Illegal {
    Illegal { layer, reason, detail }
}

/// Check `tree` against every layer, in order; the first failure is returned.
pub fn check(tree: &Tree, kingdom: &WgslKingdom, role: &Role, target: &Target) -> Result<(), Illegal> {
    Checker::new(kingdom, role, target).check(tree)
}

/// The checker with its row index built once, for checking many trees.
pub struct Checker<'a> {
    kingdom: &'a WgslKingdom,
    target: Target,
    /// Every row's signatures by name: the kingdom's and the role's.
    rows: HashMap<String, Vec<(Vec<Ty>, Ty)>>,
}

/// A kernel's workgroup arrays: variable name to element type and length.
type WgPlan = BTreeMap<String, (String, u32)>;

/// The largest u32.
const U32_MAX: u128 = u32::MAX as u128;

/// A closed interval of u32 values (u128 so sums and products of two u32
/// values never wrap in the checker itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Iv {
    lo: u128,
    hi: u128,
}

fn pt(v: u128) -> Iv {
    Iv { lo: v, hi: v }
}

/// A statement's walk mode: which layer the shared walker is checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Grammar,
    Range,
    Uniform,
    Bounded,
}

/// A loop's parts, from its row.
struct LoopParts<'t> {
    items: bool,
    cont: bool,
    /// The `loop.items`, `loop.items_unrolled` and `loop.tile` rows of the
    /// unsafe kingdom: not a bounded constructor of the safe grammar.
    legacy: bool,
    /// The bound (an expression, a binding or a literal count).
    bound: Option<&'t Tree>,
    /// Whether the bound is a literal count (`loop.tile*`, `loop.unroll`).
    literal_count: bool,
    /// The step per iteration: a stride or step literal, or the unroll
    /// factor of an unrolled item loop (`t + u <= bound`).
    step: Option<&'t Tree>,
    body: &'t Tree,
}

/// One tile loop's counter in the walk: its range and whether it is uniform.
#[derive(Debug, Clone, Copy)]
struct Counter {
    iv: Option<Iv>,
    uniform: bool,
}

/// Where a value lands: the u32 wrap rule depends on it (Andrew, 2026-10-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Site {
    /// A store's cell index: a wrapped cell can be another thread's (a data race).
    StoreCell,
    /// A loop's bound: a wrapped bound runs the loop to its cap.
    LoopBound,
    /// Anywhere else: every access is masked or bounds-checked, so a wrap only gives wrong values.
    Other,
}

/// The walker's state.
struct Cx {
    /// Where the value being ranged lands (its arithmetic flows into it).
    site: Site,
    mode: Mode,
    grid: Grid,
    plan: WgPlan,
    in_items: bool,
    t: Counter,
    /// Whether `t` is uniform where a later continued item loop starts it.
    t_uniform_after: bool,
    tiles: Vec<Counter>,
    in_if: usize,
    nonuniform_regs: BTreeSet<String>,
    nonuniform_parrs: BTreeSet<String>,
    changed: bool,
    tainted: bool,
    exits: Vec<bool>,
    /// What the enclosing branch conditions prove: an expression (by its
    /// s-expression) and the range it holds in this branch.
    facts: Vec<(String, Iv)>,
    err: Option<Illegal>,
}

impl Cx {
    fn new(mode: Mode, grid: Grid, plan: WgPlan) -> Cx {
        Cx {
            mode,
            grid,
            plan,
            in_items: false,
            t: Counter { iv: None, uniform: false },
            t_uniform_after: true,
            tiles: Vec::new(),
            in_if: 0,
            nonuniform_regs: BTreeSet::new(),
            nonuniform_parrs: BTreeSet::new(),
            changed: false,
            tainted: false,
            exits: Vec::new(),
            facts: Vec::new(),
            err: None,
            site: Site::Other,
        }
    }

    fn fail(&mut self, layer: Layer, reason: &'static str, detail: String) {
        if self.err.is_none() {
            self.err = Some(illegal(layer, reason, detail));
        }
    }

    /// The thread-id component index carrying the output's rows and columns.
    fn row_comp(&self) -> usize {
        if self.grid.rows_x { 0 } else { 1 }
    }

    fn col_comp(&self) -> usize {
        1 - self.row_comp()
    }

    /// A component of a builtin id vector at the worst shape: `lid`, `wid`,
    /// `gid` or `nwg`, component 0 (x), 1 (y) or 2 (z).
    fn comp(&self, id: &str, c: usize) -> Option<Iv> {
        let (nx, ny) = self.grid.dispatch(WORST_ROWS as u32, WORST_D as u32);
        let n = [u128::from(nx.max(1)), u128::from(ny.max(1)), 1];
        let w = [u128::from(self.grid.x.max(1)), u128::from(self.grid.y.max(1)), 1];
        let c = c.min(2);
        Some(match id {
            "lid" => Iv { lo: 0, hi: w[c] - 1 },
            "wid" => Iv { lo: 0, hi: n[c] - 1 },
            "gid" => Iv { lo: 0, hi: n[c] * w[c] - 1 },
            "nwg" => pt(n[c]),
            _ => return None,
        })
    }
}

/// The builtin id vector a leaf names (`gid`, `lid`, `wid`, `nwg`).
fn builtin_id(name: &str) -> Option<&'static str> {
    Some(match name {
        "builtin.global_invocation_id" => "gid",
        "builtin.local_invocation_id" => "lid",
        "builtin.workgroup_id" => "wid",
        "builtin.num_workgroups" => "nwg",
        _ => return None,
    })
}

/// The rows that put a barrier in the text (a reduction writes them too).
fn is_barrier(base: &str) -> bool {
    matches!(base, "stmt.barrier" | "stmt.storage_barrier" | "stmt.subgroup_barrier" | "wg.reduce_add" | "wg.reduce_max" | "wg.reduce_min")
}

/// The tile counter a leaf reads (1-based depth), or None.
fn tile_counter_depth(base: &str) -> Option<usize> {
    Some(match base {
        "safe.tile_counter" | "builtin.tile" => 1,
        "safe.tile_counter2" => 2,
        "safe.tile_counter3" => 3,
        "safe.tile_counter4" => 4,
        _ => return None,
    })
}

/// Whether a subtree is a row or a cell the thread owns along `axis` (`row`
/// or `col`): built from the thread's own block origin by the clamped
/// offsets of the safe rows. A fallback renders the thread's own origin
/// (`render.rs`, 2cd3c98), so it is owned.
fn owned_axis(t: &Tree, axis: &str) -> bool {
    let base = safe_base(&t.name);
    if base == "fallback" || base.starts_with(&format!("safe.thread_{axis}")) {
        return true;
    }
    if base.starts_with(&format!("safe.{axis}_plus")) {
        return t.kids.first().is_some_and(|k| owned_axis(k, axis));
    }
    false
}

/// Whether a row's value is arithmetic over its children (a wrap in a
/// child lands where the row's value lands), not a load, an array access, a
/// register or a conversion.
fn carries_value(name: &str) -> bool {
    let base = safe_base(name);
    name == "fallback" || ["index.", "count.", "bound.", "uniform.", "subgroup.", "safe.", "shape.", "axis.", "literal."].iter().any(|p| base.starts_with(p))
}

/// Whether a row clamps its child `i` to a bound of its own: either side
/// of `min` and `and`, the value of `clamp_to`, `clamp` and `mod`, and a row's or column's offset, which
/// the renderer clamps to the block (`min(o, blk - 1)`). A wrap in that child
/// gives a value no larger than the clamp: it does not reach the parent.
fn clamps_child(name: &str, i: usize) -> bool {
    let base = safe_base(name);
    let op = base.split_once('.').map_or(base, |(_, op)| op);
    let arith = ["index.", "count.", "uniform."].iter().any(|p| base.starts_with(p));
    (arith && (matches!(op, "min" | "and") || (matches!(op, "clamp_to" | "clamp" | "mod") && i == 0))) || ((base.starts_with("safe.row_plus") || base.starts_with("safe.col_plus")) && i >= 1)
}

fn owned_cell(t: &Tree) -> bool {
    t.name == "fallback" || (safe_base(&t.name) == "safe.cell_of" && t.kids.len() == 2 && owned_axis(&t.kids[0], "row") && owned_axis(&t.kids[1], "col"))
}

/// The parts of a loop row, or None for a statement that is not a loop.
fn loop_parts(s: &Tree) -> Option<LoopParts<'_>> {
    let k = s.kids.as_slice();
    let base = safe_base(&s.name);
    fn make<'t>(items: bool, cont: bool, legacy: bool, bound: Option<&'t Tree>, literal_count: bool, step: Option<&'t Tree>, body: Option<&'t Tree>) -> Option<LoopParts<'t>> {
        body.map(|body| LoopParts { items, cont, legacy, bound, literal_count, step, body })
    }
    match base {
        "loop.items_bounded" => make(true, false, false, k.first(), false, None, k.get(1)),
        "loop.items_unrolled_bounded" => make(true, false, false, k.get(1), false, k.first(), k.get(2)),
        "loop.items_strided_bounded" => make(true, false, false, k.first(), false, k.get(1), k.get(2)),
        "loop.items_cont_bounded" => make(true, true, false, k.first(), false, None, k.get(1)),
        "loop.items_cont_strided_bounded" => make(true, true, false, k.first(), false, k.get(1), k.get(2)),
        "loop.tilex_bounded" => make(false, false, false, k.first(), false, None, k.get(1)),
        "loop.tilex_strided_bounded" => make(false, false, false, k.first(), false, k.get(1), k.get(2)),
        "loop.tile_bounded" | "loop.unroll" => make(false, false, false, k.first(), true, None, k.get(1)),
        "loop.tile_strided_bounded" => make(false, false, false, k.first(), true, k.get(1), k.get(2)),
        "loop.items" => make(true, false, true, None, false, None, k.first()),
        "loop.items_unrolled" => make(true, false, true, None, false, k.first(), k.get(1)),
        "loop.tile" => make(false, false, true, k.first(), true, None, k.get(1)),
        _ => None,
    }
}

/// A subtree's s-expression, cut for a message.
fn brief(t: &Tree) -> String {
    let s = t.to_sexpr();
    if s.chars().count() <= 200 {
        s
    } else {
        format!("{}…", s.chars().take(200).collect::<String>())
    }
}

/// A bounded literal count as the renderer reads it (`lit`): 1 for a
/// fallback, the constant rounded into 1..=max.
fn lit_count(t: &Tree, max: u128) -> u128 {
    if t.name.starts_with("literal.") && t.konst.is_finite() {
        (f64::from(t.konst).round().max(1.0) as u128).min(max)
    } else {
        1
    }
}

/// A shift count as the renderer masks it (`& 31u`): exact for a known
/// count, any of 0..=31 for a range that reaches past 31.
fn shift_count(b: Iv) -> Iv {
    if b.lo == b.hi {
        pt(b.lo & 31)
    } else if b.hi <= 31 {
        b
    } else {
        Iv { lo: 0, hi: 31 }
    }
}

/// The smaller of an interval and a constant; an unknown value below a
/// constant is still known to be at most the constant.
fn min_c(a: Option<Iv>, c: u128) -> Iv {
    match a {
        Some(a) => Iv { lo: a.lo.min(c), hi: a.hi.min(c) },
        None => Iv { lo: 0, hi: c },
    }
}

/// The registers and private arrays layer 4 found non-uniform.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NonUniform {
    pub registers: BTreeSet<String>,
    pub private_arrays: BTreeSet<String>,
}

impl<'a> Checker<'a> {
    pub fn new(kingdom: &'a WgslKingdom, role: &Role, target: &Target) -> Checker<'a> {
        let mut rows: HashMap<String, Vec<(Vec<Ty>, Ty)>> = HashMap::new();
        for r in &kingdom.rows {
            rows.entry(r.name.clone()).or_default().push((r.inputs.clone(), r.output));
        }
        for p in &role.per_kernel {
            let ins: Option<Vec<Ty>> = p.inputs.iter().map(|d| dual_ty(kingdom, d).ok()).collect();
            if let (Some(ins), Ok(out)) = (ins, dual_ty(kingdom, &p.output)) {
                rows.entry(p.name.clone()).or_default().push((ins, out));
            }
        }
        Checker { kingdom, target: *target, rows }
    }

    /// Every layer in order; the first failure.
    pub fn check(&self, tree: &Tree) -> Result<(), Illegal> {
        self.typed(tree)?;
        self.grammar(tree)?;
        self.range(tree)?;
        let nonuniform = self.uniformity(tree)?;
        self.adapter(tree)?;
        self.bounded(tree, &nonuniform)
    }

    fn dual_name(&self, ty: Ty) -> String {
        self.kingdom.dual(ty).map_or_else(|| format!("{ty:?}"), |d| d.dual.clone())
    }

    /// A dual holding a thread's row, column or cell (thread-dependent).
    fn is_thread_dual(&self, t: &Tree) -> bool {
        self.kingdom.dual(t.ty).is_some_and(|d| d.form.starts_with("thread") || d.form == "cell")
    }

    fn is_stmt_level(&self, t: &Tree) -> bool {
        self.kingdom.dual(t.ty).is_some_and(|d| matches!(d.slot.as_str(), "stmt" | "block" | "kernel"))
    }

    // ---------------------------------------------------------------- layer 1

    /// Layer 1, types: every node a row at its signature, a literal of its
    /// dual with a finite constant, or a childless fallback.
    pub fn typed(&self, t: &Tree) -> Result<(), Illegal> {
        let fail = |reason: &'static str, detail: String| Err(illegal(Layer::Typed, reason, detail));
        if self.kingdom.dual(t.ty).is_none() {
            return fail("unknown_dual", format!("{} has type {:?}, not a dual of the kingdom", t.name, t.ty));
        }
        if t.name == "fallback" {
            if !t.kids.is_empty() {
                return fail("fallback_with_children", format!("a fallback of {} with {} children", self.dual_name(t.ty), t.kids.len()));
            }
            return Ok(());
        }
        if t.name.starts_with("literal.") {
            if !t.kids.is_empty() {
                return fail("literal_with_children", brief(t));
            }
            if !t.konst.is_finite() {
                return fail("nonfinite_literal", format!("{} {}", t.name, t.konst));
            }
            if !self.rows.get(&t.name).is_some_and(|sigs| sigs.iter().any(|(ins, out)| ins.is_empty() && *out == t.ty)) {
                return fail("literal_not_a_row", format!("{} of {}", t.name, self.dual_name(t.ty)));
            }
            return Ok(());
        }
        let Some(sigs) = self.rows.get(&t.name) else {
            return fail("unknown_row", format!("{} is not a row of the kingdom or the role", t.name));
        };
        let kid_tys: Vec<Ty> = t.kids.iter().map(|k| k.ty).collect();
        if !sigs.iter().any(|(ins, out)| *out == t.ty && *ins == kid_tys) {
            let have: Vec<String> = kid_tys.iter().map(|k| self.dual_name(*k)).collect();
            let want: Vec<String> = sigs.iter().map(|(ins, out)| format!("({}) -> {}", ins.iter().map(|i| self.dual_name(*i)).collect::<Vec<_>>().join(", "), self.dual_name(*out))).collect();
            return fail("signature_mismatch", format!("{} as ({}) -> {}; the row's signatures: {}", t.name, have.join(", "), self.dual_name(t.ty), want.join("; ")));
        }
        t.kids.iter().try_for_each(|k| self.typed(k))
    }

    // ---------------------------------------------------------------- layer 2

    /// Layer 2: the safe kernel's root, owned-cell stores and reads of `h`,
    /// barriers outside every `if`, loops from the bounded constructors.
    pub fn grammar(&self, tree: &Tree) -> Result<(), Illegal> {
        let root_dual = self.dual_name(tree.ty);
        let root_ok = tree.name == "kernel.entry_safe" || (tree.name == "fallback" && root_dual == "kernel.safe");
        if !root_ok {
            return Err(illegal(Layer::Grammar, "root_not_safe_kernel", format!("the root is {} ({root_dual}), not kernel.entry_safe", tree.name)));
        }
        self.walk_root(tree, Mode::Grammar, &NonUniform::default()).err.map_or(Ok(()), Err)
    }

    // ---------------------------------------------------------------- layer 3

    /// Layer 3: array indices in range and no index arithmetic overflow at
    /// the worst shape.
    pub fn range(&self, tree: &Tree) -> Result<(), Illegal> {
        self.walk_root(tree, Mode::Range, &NonUniform::default()).err.map_or(Ok(()), Err)
    }

    // ---------------------------------------------------------------- layer 4

    /// Layer 4: barriers only under uniform control. Returns the registers
    /// and private arrays found non-uniform (the fixpoint), for layer 6.
    pub fn uniformity(&self, tree: &Tree) -> Result<NonUniform, Illegal> {
        let (nonuniform, err) = self.uniform_fixpoint(tree);
        err.map_or(Ok(nonuniform), Err)
    }

    /// The non-uniform registers and private arrays, and the first barrier
    /// in non-uniform control flow if any.
    pub fn uniform_fixpoint(&self, tree: &Tree) -> (NonUniform, Option<Illegal>) {
        let mut nonuniform = NonUniform::default();
        // Each pass only adds to the non-uniform sets, which are finite.
        loop {
            let cx = self.walk_root(tree, Mode::Uniform, &nonuniform);
            if !cx.changed {
                return (nonuniform, cx.err);
            }
            nonuniform = NonUniform { registers: cx.nonuniform_regs, private_arrays: cx.nonuniform_parrs };
        }
    }

    // ---------------------------------------------------------------- layer 5

    /// Layer 5: the workgroup shape and memory within the target's limits,
    /// the features the tree's rows need present.
    pub fn adapter(&self, tree: &Tree) -> Result<(), Illegal> {
        let fail = |reason: &'static str, detail: String| Err(illegal(Layer::Adapter, reason, detail));
        let (grid, plan) = self.grid_and_plan(&self.resolved(tree)?)?;
        let t = &self.target;
        if grid.x > t.max_size[0] || grid.y > t.max_size[1] || u64::from(grid.x) * u64::from(grid.y) > u64::from(t.max_invocations) {
            return fail("workgroup_size_over_limit", format!("@workgroup_size({}, {}, 1) against the target's {:?} per axis and {} invocations", grid.x, grid.y, t.max_size, t.max_invocations));
        }
        let elem_bytes = |e: &str| -> u64 {
            if e == "vec4<f32>" {
                16
            } else if e == "vec2<f32>" {
                8
            } else {
                4
            }
        };
        let arrays: Vec<(String, u64)> = if plan.is_empty() {
            ["wg_a", "wg_b", "wg_va", "wg_vb"].iter().map(|v| (v.to_string(), u64::from(array_len(v)) * if v.starts_with("wg_v") { 16 } else { 4 })).collect()
        } else {
            plan.iter().map(|(v, (e, l))| (v.clone(), u64::from(*l) * elem_bytes(e))).collect()
        };
        // Each variable rounded up to 16 bytes, as Metal allocates it (`Target::check_limits`).
        let bytes: u64 = arrays.iter().map(|(_, b)| b.div_ceil(16) * 16).sum();
        if bytes > u64::from(t.max_workgroup_storage) {
            return fail("workgroup_memory_over_limit", format!("{bytes} bytes of workgroup arrays {arrays:?} against the target's {}", t.max_workgroup_storage));
        }
        match self.features(tree) {
            Some(detail) => fail("feature_missing", detail),
            None => Ok(()),
        }
    }

    /// The first feature a node needs that the target lacks.
    fn features(&self, t: &Tree) -> Option<String> {
        let caps = self.target.caps;
        let base = safe_base(&t.name);
        if (base.starts_with("subgroup.") || base.starts_with("builtin.subgroup_") || base == "builtin.num_subgroups") && !caps.contains(Capabilities::SUBGROUP) {
            return Some(format!("{} needs the subgroup feature", t.name));
        }
        if base == "stmt.subgroup_barrier" && !caps.contains(Capabilities::SUBGROUP_BARRIER) {
            return Some(format!("{} needs the subgroup barrier feature", t.name));
        }
        if let Some(d) = self.kingdom.dual(t.ty) {
            if d.slot.contains("f16") {
                // naga 0.20 has no f16 capability: no target admits it.
                return Some(format!("{} of {}: naga 0.20 has no f16", t.name, d.dual));
            }
            if d.slot.contains("f64") && !caps.contains(Capabilities::FLOAT64) {
                return Some(format!("{} of {} needs SHADER_F64", t.name, d.dual));
            }
        }
        t.kids.iter().find_map(|k| self.features(k))
    }

    // ---------------------------------------------------------------- layer 6

    /// Layer 6: every loop's trip count bounded, given the non-uniform
    /// registers and private arrays of layer 4.
    pub fn bounded(&self, tree: &Tree, nonuniform: &NonUniform) -> Result<(), Illegal> {
        self.walk_root(tree, Mode::Bounded, nonuniform).err.map_or(Ok(()), Err)
    }

    /// A loop's bound: a literal or a uniform value its body leaves alone,
    /// or a value known at the worst shape that the counter cannot step past
    /// `u32::MAX`.
    fn check_bound(&self, s: &Tree, lp: &LoopParts<'_>, bound_iv: Option<Iv>, step: u128, cx: &mut Cx) {
        let Some(b) = lp.bound else { return };
        match bound_iv {
            Some(iv) => {
                if iv.hi + step - 1 > U32_MAX {
                    cx.fail(Layer::Bounded, "loop_bound_wraps", format!("{}: the bound reaches {} and the counter steps by {step}: it wraps past u32::MAX before it reaches the bound", s.name, iv.hi));
                }
            }
            None => {
                if !self.uniform(b, cx) {
                    cx.fail(Layer::Bounded, "loop_bound_not_uniform", format!("{}: its bound {} is neither a literal, a binding, a value known at the worst shape, nor uniform", s.name, brief(b)));
                    return;
                }
                let (mut read_regs, mut read_parrs) = (BTreeSet::new(), BTreeSet::new());
                self.reads(b, &mut read_regs, &mut read_parrs);
                let (mut wrote_regs, mut wrote_parrs) = (BTreeSet::new(), BTreeSet::new());
                self.writes(lp.body, &mut wrote_regs, &mut wrote_parrs);
                let changed: Vec<&String> = read_regs.intersection(&wrote_regs).chain(read_parrs.intersection(&wrote_parrs)).collect();
                if !changed.is_empty() {
                    cx.fail(Layer::Bounded, "loop_bound_changes_in_body", format!("{}: its bound {} reads {changed:?}, which the loop's body writes", s.name, brief(b)));
                }
            }
        }
    }

    /// The registers and private arrays an expression reads.
    fn reads(&self, t: &Tree, regs: &mut BTreeSet<String>, parrs: &mut BTreeSet<String>) {
        if let Some(d) = self.kingdom.dual(t.ty) {
            if d.slot.starts_with("reg<") {
                if let Ok(r) = register_name(self.kingdom, t) {
                    regs.insert(r);
                }
            } else if d.slot.starts_with("parr<") {
                if let Ok(p) = parr_var(self.kingdom, t) {
                    parrs.insert(p.to_string());
                }
            }
        }
        t.kids.iter().for_each(|k| self.reads(k, regs, parrs));
    }

    /// The registers and private arrays a statement subtree assigns.
    fn writes(&self, t: &Tree, regs: &mut BTreeSet<String>, parrs: &mut BTreeSet<String>) {
        let base = safe_base(&t.name);
        if let Some(target) = t.kids.first() {
            if base.starts_with("reg.set_") {
                if let Ok(r) = register_name(self.kingdom, target) {
                    regs.insert(r);
                }
            } else if base.starts_with("parr.set_") {
                if let Ok(p) = parr_var(self.kingdom, target) {
                    parrs.insert(p.to_string());
                }
            }
        }
        t.kids.iter().for_each(|k| self.writes(k, regs, parrs));
    }

    // ------------------------------------------------------------ the walker

    /// The tree with every knob reference written as the value its kernel's
    /// declarations give (`params::resolve`), as the renderer reads it.
    fn resolved(&self, tree: &Tree) -> Result<Tree, Illegal> {
        crate::wgsl::params::resolve(tree).map_err(|e| illegal(Layer::Grammar, "knob_reference_unresolved", e))
    }

    /// The kernel's grid and workgroup arrays as the renderer plans them.
    fn grid_and_plan(&self, tree: &Tree) -> Result<(Grid, WgPlan), Illegal> {
        match (tree.name.as_str(), tree.kids.first()) {
            ("kernel.entry_safe", Some(g)) => {
                let grid = grid_of(g).map_err(|e| illegal(Layer::Grammar, "grid_not_in_grammar", e))?;
                let plan = workgroup_plan(self.kingdom, g, tree).map_err(|e| illegal(Layer::Grammar, "workgroup_array_not_in_grammar", e))?;
                Ok((grid, plan))
            }
            _ => Ok((Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, BTreeMap::new())),
        }
    }

    /// Walk the kernel's body, then its epilogue, in `mode`.
    fn walk_root(&self, tree: &Tree, mode: Mode, nonuniform: &NonUniform) -> Cx {
        let resolved = self.resolved(tree);
        let (grid, plan) = match resolved.as_ref().map_err(Clone::clone).and_then(|t| self.grid_and_plan(t)) {
            Ok(gp) => gp,
            Err(e) => {
                let mut cx = Cx::new(mode, Grid { x: 16, y: 16, bx: 1, by: 1, rows_x: false }, BTreeMap::new());
                cx.err = Some(e);
                return cx;
            }
        };
        let mut cx = Cx::new(mode, grid, plan);
        cx.nonuniform_regs = nonuniform.registers.clone();
        cx.nonuniform_parrs = nonuniform.private_arrays.clone();
        let Ok(tree) = resolved else { return cx };
        if tree.name == "kernel.entry_safe" {
            for k in tree.kids.iter().skip(1) {
                self.walk_block(k, &mut cx, true);
            }
        }
        cx
    }

    /// A block; `ctrl` is whether every condition the block depends on is
    /// uniform (layer 4's control dependence).
    fn walk_block(&self, b: &Tree, cx: &mut Cx, ctrl: bool) {
        let k = b.kids.as_slice();
        let in_loop = cx.in_items || !cx.tiles.is_empty();
        match safe_base(&b.name) {
            // An empty block, or the epilogue's store at the thread's first cell.
            "fallback" => {}
            "block.one" | "block.seq2" | "block.seq3" | "block.seq4" => k.iter().for_each(|s| self.walk_stmt(s, cx, ctrl)),
            "block.brk" if k.len() == 3 => {
                if cx.mode == Mode::Grammar && !in_loop {
                    cx.fail(Layer::Grammar, "break_outside_loop", brief(b));
                }
                self.walk_block(&k[0], cx, ctrl);
                self.value(&k[1], cx);
                let c = self.uniform(&k[1], cx);
                // A break under a non-uniform condition: the threads leave
                // the loop at different iterations.
                if cx.mode == Mode::Uniform && !(ctrl && c && !cx.tainted) {
                    if let Some(e) = cx.exits.last_mut() {
                        *e = true;
                    }
                }
                self.walk_block_given(&k[2], &k[1], false, cx, ctrl && c);
            }
            "block.cont" if k.len() == 2 => {
                if cx.mode == Mode::Grammar && !in_loop {
                    cx.fail(Layer::Grammar, "continue_outside_loop", brief(b));
                }
                self.value(&k[0], cx);
                let c = self.uniform(&k[0], cx);
                self.walk_block_given(&k[1], &k[0], false, cx, ctrl && c);
            }
            _ => {
                if cx.mode == Mode::Grammar {
                    cx.fail(Layer::Grammar, "block_not_in_safe_grammar", brief(b));
                }
            }
        }
    }

    fn walk_stmt(&self, s: &Tree, cx: &mut Cx, ctrl: bool) {
        if let Some(lp) = loop_parts(s) {
            self.walk_loop(s, lp, cx, ctrl);
            return;
        }
        let k = s.kids.as_slice();
        let base = safe_base(&s.name);
        let live = ctrl && !cx.tainted;
        match base {
            "fallback" => {}
            // stmt.cblock (legality item 5): conditional statements written in place.
            "stmt.block" | "stmt.cblock" if k.len() == 1 => self.walk_block(&k[0], cx, ctrl),
            "stmt.if" | "stmt.if_else" if k.len() >= 2 => {
                self.value(&k[0], cx);
                let c = self.uniform(&k[0], cx);
                cx.in_if += 1;
                for (i, b) in k[1..].iter().enumerate() {
                    self.walk_block_given(b, &k[0], i == 0, cx, ctrl && c);
                }
                cx.in_if -= 1;
            }
            "stmt.return" => {
                // A return under a non-uniform condition: every later
                // statement, and every statement of an enclosing loop, runs
                // in some threads only.
                if cx.mode == Mode::Uniform && !live {
                    cx.tainted = true;
                    cx.exits.iter_mut().for_each(|e| *e = true);
                }
            }
            b if is_barrier(b) => {
                self.values(s, cx);
                match cx.mode {
                    Mode::Grammar if cx.in_if > 0 => cx.fail(Layer::Grammar, "barrier_under_condition", format!("{} inside an if", s.name)),
                    Mode::Uniform if !live => {
                        let why = if cx.tainted { "it follows a return under a condition that is not workgroup-uniform" } else { "a condition or loop bound it depends on is not workgroup-uniform" };
                        cx.fail(Layer::Uniformity, "barrier_in_nonuniform_control_flow", format!("{}: {why}", s.name));
                    }
                    _ => {}
                }
            }
            "store.h_safe" => {
                if cx.mode == Mode::Grammar && !k.first().is_some_and(owned_cell) {
                    cx.fail(Layer::Grammar, "store_not_owned_cell", format!("{} at {}: not a cell the thread owns", s.name, k.first().map(brief).unwrap_or_default()));
                }
                if let Some(cell) = k.first() {
                    cx.site = Site::StoreCell;
                    self.value(cell, cx);
                    cx.site = Site::Other;
                }
                k.iter().skip(1).filter(|v| !self.is_stmt_level(v)).for_each(|v| self.value(v, cx));
            }
            "reg.set_f" | "reg.set_v" | "reg.set_i" | "reg.set_b" | "reg.set_n" | "reg.set_u" | "reg.set_x" | "reg.set_y" | "reg.set_z" | "reg.set_v_lane" => {
                self.values(s, cx);
                if cx.mode == Mode::Uniform {
                    if let Some(r) = k.first().and_then(|r| register_name(self.kingdom, r).ok()) {
                        let uniform = live && k.get(1..).unwrap_or(&[]).iter().all(|v| self.uniform(v, cx));
                        if !uniform && cx.nonuniform_regs.insert(r) {
                            cx.changed = true;
                        }
                    }
                }
            }
            "parr.set_safe" | "parr.set_v_lane" => {
                self.access(s, cx);
                self.values(s, cx);
                if cx.mode == Mode::Uniform {
                    if let Some(p) = k.first().and_then(|p| parr_var(self.kingdom, p).ok()) {
                        let uniform = live && k.get(1..).unwrap_or(&[]).iter().all(|v| self.uniform(v, cx));
                        if !uniform && cx.nonuniform_parrs.insert(p.to_string()) {
                            cx.changed = true;
                        }
                    }
                }
            }
            "wg.set_safe" | "wg.set_v_lane" | "wg.set" | "atomic.store_u" | "atomic.store_i" => {
                self.access(s, cx);
                self.values(s, cx);
            }
            "stmt.discard_u" | "stmt.discard_i" => self.values(s, cx),
            _ => {
                if cx.mode == Mode::Grammar {
                    if base.starts_with("store.") {
                        cx.fail(Layer::Grammar, "store_not_owned_cell", format!("{}: a store outside the owned-cell row", s.name));
                    } else {
                        cx.fail(Layer::Grammar, "statement_not_in_safe_grammar", brief(s));
                    }
                }
                self.values(s, cx);
            }
        }
    }

    fn walk_loop(&self, s: &Tree, lp: LoopParts<'_>, cx: &mut Cx, ctrl: bool) {
        if cx.mode == Mode::Grammar {
            if lp.legacy {
                cx.fail(Layer::Grammar, "loop_not_bounded_constructor", format!("{}: a loop row of the unsafe kingdom", s.name));
            } else if lp.items && cx.in_items {
                cx.fail(Layer::Grammar, "nested_item_loop", format!("{} inside an item loop", s.name));
            } else if !lp.items && cx.tiles.len() >= 4 {
                cx.fail(Layer::Grammar, "tile_depth_exceeded", format!("{} at tile depth {}", s.name, cx.tiles.len() + 1));
            }
        }
        // The header is evaluated outside the loop's scope.
        for kid in &s.kids {
            if !std::ptr::eq(kid, lp.body) && !self.is_stmt_level(kid) {
                if lp.bound.is_some_and(|b| std::ptr::eq(kid, b)) {
                    cx.site = Site::LoopBound;
                }
                self.value(kid, cx);
                cx.site = Site::Other;
            }
        }
        let bound_iv = match (lp.legacy, lp.literal_count, lp.bound) {
            // `loop.items`: the role's item count, cfg.m.
            (true, false, _) => Some(pt(WORST_M as u128)),
            (_, true, Some(b)) => Some(pt(lit_count(b, if lp.legacy { 32 } else { 64 }))),
            (_, false, Some(b)) => {
                cx.site = Site::LoopBound;
                let iv = self.iv(b, cx);
                cx.site = Site::Other;
                iv
            }
            _ => None,
        };
        let bound_uniform = lp.literal_count || lp.bound.is_none_or(|b| self.uniform(b, cx));
        let step = match lp.step {
            Some(t) if self.kingdom.dual(t.ty).is_some_and(|d| d.form == "unroll") => {
                if t.name.starts_with("literal.") {
                    u128::from(snap(&UNROLL, t.konst))
                } else {
                    u128::from(UNROLL[0])
                }
            }
            Some(t) => lit_count(t, if lp.items { 4096 } else { 64 }),
            None => 1,
        };
        if cx.mode == Mode::Bounded && !lp.legacy && !lp.literal_count {
            self.check_bound(s, &lp, bound_iv, step, cx);
        }
        let counter = Counter { iv: bound_iv.map(|b| Iv { lo: 0, hi: b.hi.max(1) - 1 }), uniform: bound_uniform && (!lp.cont || cx.t_uniform_after) };
        let body_ctrl = ctrl && counter.uniform;
        let saved = (cx.in_items, cx.t);
        if lp.items {
            cx.in_items = true;
            cx.t = counter;
        } else {
            cx.tiles.push(counter);
        }
        cx.exits.push(false);
        self.walk_block(lp.body, cx, body_ctrl);
        let exit = cx.exits.pop().unwrap_or(false);
        if exit && cx.mode == Mode::Uniform {
            // A thread may leave the loop while others iterate: every
            // statement of the body is under non-uniform control.
            cx.exits.push(false);
            self.walk_block(lp.body, cx, false);
            cx.exits.pop();
        }
        if lp.items {
            (cx.in_items, cx.t) = saved;
            cx.t_uniform_after = counter.uniform && !exit;
        } else {
            cx.tiles.pop();
        }
    }

    /// Whether an expression reads no state a statement can change
    /// (registers, private and workgroup arrays, `h`, atomics, subgroup
    /// exchanges): what a branch condition proves about it holds throughout
    /// the branch.
    fn stateless(&self, t: &Tree) -> bool {
        let slot_ok = self.kingdom.dual(t.ty).is_none_or(|d| !(d.slot.starts_with("reg<") || d.slot.starts_with("parr<") || d.slot.starts_with("wgmem<")));
        let base = safe_base(&t.name);
        let row_ok = !(base.starts_with("load.h") || base.starts_with("wg.") || base.starts_with("atomic.") || base.starts_with("subgroup."));
        slot_ok && row_ok && t.kids.iter().all(|k| self.stateless(k))
    }

    /// The ranges a u32 comparison proves in the branch where it is `truth`:
    /// `a < b` gives a at most b - 1 and b at least a + 1; its negation
    /// `a >= b` gives a at least b and b at most a. `&&` when true and `||`
    /// when false give both sides' facts; `!` flips.
    fn facts_of(&self, cond: &Tree, truth: bool, cx: &mut Cx) -> Vec<(String, Iv)> {
        let base = safe_base(&cond.name);
        let k = cond.kids.as_slice();
        match (base, k) {
            ("logic.and", [a, b]) if truth => [self.facts_of(a, true, cx), self.facts_of(b, true, cx)].concat(),
            ("logic.or", [a, b]) if !truth => [self.facts_of(a, false, cx), self.facts_of(b, false, cx)].concat(),
            ("logic.not", [a]) => self.facts_of(a, !truth, cx),
            ("compare.lt" | "compare.le" | "compare.gt" | "compare.ge", [a, b]) => {
                if self.kingdom.dual(a.ty).is_none_or(|d| d.slot != "u32") {
                    return Vec::new();
                }
                // As `x < y` (strict) or `x <= y`.
                let (x, y, strict) = match base {
                    "compare.lt" => (a, b, true),
                    "compare.le" => (a, b, false),
                    "compare.gt" => (b, a, true),
                    _ => (b, a, false),
                };
                // The negation of `x < y` is `y <= x`; of `x <= y`, `y < x`.
                let (x, y, strict) = if truth { (x, y, strict) } else { (y, x, !strict) };
                let gap = u128::from(strict);
                let (xi, yi) = (self.iv(x, cx), self.iv(y, cx));
                let mut out = Vec::new();
                if let Some(yi) = yi.filter(|_| self.stateless(x)) {
                    if yi.hi >= gap {
                        out.push((x.to_sexpr(), Iv { lo: 0, hi: yi.hi - gap }));
                    }
                }
                if let Some(xi) = xi.filter(|_| self.stateless(y)) {
                    out.push((y.to_sexpr(), Iv { lo: xi.lo + gap, hi: U32_MAX }));
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// Walk a block under the facts `cond` proves when it is `truth`.
    fn walk_block_given(&self, b: &Tree, cond: &Tree, truth: bool, cx: &mut Cx, ctrl: bool) {
        let facts = self.facts_of(cond, truth, cx);
        let depth = cx.facts.len();
        cx.facts.extend(facts);
        self.walk_block(b, cx, ctrl);
        cx.facts.truncate(depth);
    }

    /// Every value child of a statement, checked in the walk's mode.
    fn values(&self, s: &Tree, cx: &mut Cx) {
        for k in &s.kids {
            if !self.is_stmt_level(k) {
                self.value(k, cx);
            }
        }
    }

    fn value(&self, v: &Tree, cx: &mut Cx) {
        match cx.mode {
            Mode::Grammar => self.grammar_value(v, cx),
            Mode::Range => {
                self.iv(v, cx);
            }
            Mode::Uniform | Mode::Bounded => {}
        }
    }

    fn grammar_value(&self, v: &Tree, cx: &mut Cx) {
        match safe_base(&v.name) {
            "load.h" => cx.fail(Layer::Grammar, "read_of_unowned_cell", format!("{}: h read at any index (another thread's cell)", brief(v))),
            "load.h_safe" if !v.kids.first().is_some_and(owned_cell) => cx.fail(Layer::Grammar, "read_of_unowned_cell", format!("{}: h read at a cell the thread does not own", brief(v))),
            _ => {}
        }
        v.kids.iter().for_each(|k| self.grammar_value(k, cx));
    }

    /// Whether a value is workgroup-uniform (WGSL's rule, conservatively).
    fn uniform(&self, v: &Tree, cx: &Cx) -> bool {
        if let Some(d) = self.kingdom.dual(v.ty) {
            if d.slot.starts_with("reg<") {
                return register_name(self.kingdom, v).is_ok_and(|r| !cx.nonuniform_regs.contains(&r));
            }
            if d.slot.starts_with("parr<") {
                return parr_var(self.kingdom, v).is_ok_and(|p| !cx.nonuniform_parrs.contains(p));
            }
        }
        let base = safe_base(&v.name);
        match base {
            "fallback" => !self.is_thread_dual(v),
            b if b.starts_with("literal.") => true,
            "builtin.workgroup_id" | "builtin.num_workgroups" | "builtin.subgroup_size" | "builtin.num_subgroups" => true,
            "builtin.global_invocation_id" | "builtin.local_invocation_id" | "builtin.local_invocation_index" | "builtin.subgroup_invocation_id" | "builtin.subgroup_id" => false,
            // Outside its loop a counter is the constant 0.
            "safe.counter" | "builtin.item" => !cx.in_items || cx.t.uniform,
            b if tile_counter_depth(b).is_some() => tile_counter_depth(b).and_then(|d| cx.tiles.get(d - 1)).is_none_or(|c| c.uniform),
            b if b.starts_with("safe.thread_") => false,
            b if b.starts_with("axis.") => b.ends_with("_wid"),
            // h is read-write storage: not uniform. phi is read-only: uniform at a uniform index.
            "load.h_safe" | "load.h" => false,
            b if b.starts_with("wg.") || b.starts_with("atomic.") || b.starts_with("subgroup.") => false,
            _ => v.kids.iter().all(|k| self.uniform(k, cx)),
        }
    }

    /// A u32 literal's value as the renderer writes it.
    fn lit_u32(&self, t: &Tree) -> Option<Iv> {
        let d = self.kingdom.dual(t.ty)?;
        if d.slot != "u32" {
            return None;
        }
        let v = t.konst;
        Some(pt(match d.form.as_str() {
            "wgdim" => u128::from(workgroup_dim(v)),
            "lit64" => lit_count(t, 64),
            "lit4096" => lit_count(t, 4096),
            "blk" => u128::from(snap(&BLK, v)),
            "unroll" => u128::from(snap(&UNROLL, v)),
            _ => {
                if v.is_finite() {
                    v.clamp(0.0, 4_294_967_040.0) as u128
                } else {
                    0
                }
            }
        }))
    }

    /// Record an overflow: illegal where it lands in a store's cell index
    /// (layer 3) or a loop's bound (layer 6), a diagnostic anywhere else.
    fn overflow(&self, v: &Tree, cx: &mut Cx, what: String) {
        let detail = format!("{}: {what} at the worst shape (d {WORST_D}, m {WORST_M}, {WORST_ROWS} rows)", brief(v));
        match (cx.site, cx.mode) {
            (Site::StoreCell, Mode::Range) => cx.fail(Layer::Range, "store_cell_wraps", detail),
            (Site::LoopBound, Mode::Bounded) => cx.fail(Layer::Bounded, "loop_bound_wraps", detail),
            (Site::Other, Mode::Range) => cx.fail(Layer::Range, "index_wraps", detail),
            _ => {}
        }
    }

    fn add(&self, a: Option<Iv>, b: Option<Iv>, v: &Tree, cx: &mut Cx) -> Option<Iv> {
        let (a, b) = (a?, b?);
        let r = Iv { lo: a.lo + b.lo, hi: a.hi + b.hi };
        if r.hi > U32_MAX {
            self.overflow(v, cx, format!("a sum reaches {}", r.hi));
            return None;
        }
        Some(r)
    }

    fn mul(&self, a: Option<Iv>, b: Option<Iv>, v: &Tree, cx: &mut Cx) -> Option<Iv> {
        let (a, b) = (a?, b?);
        let r = Iv { lo: a.lo * b.lo, hi: a.hi * b.hi };
        if r.hi > U32_MAX {
            self.overflow(v, cx, format!("a product reaches {}", r.hi));
            return None;
        }
        Some(r)
    }

    /// The value of a u32 expression at the worst shape, when known; every
    /// child is evaluated (and every array access in it checked) either way.
    fn iv(&self, v: &Tree, cx: &mut Cx) -> Option<Iv> {
        let raw = self.iv_raw(v, cx);
        if cx.facts.is_empty() {
            return raw;
        }
        let key = v.to_sexpr();
        let mut out = raw;
        for (_, f) in cx.facts.iter().filter(|(k, _)| *k == key) {
            out = Some(match out {
                Some(r) if r.lo.max(f.lo) <= r.hi.min(f.hi) => Iv { lo: r.lo.max(f.lo), hi: r.hi.min(f.hi) },
                Some(r) => r,
                None => *f,
            });
        }
        out
    }

    fn iv_raw(&self, v: &Tree, cx: &mut Cx) -> Option<Iv> {
        // A load, an access or a conversion ends the flow: its children's
        // arithmetic lands in an index (masked or bounds-checked), not here.
        let site = cx.site;
        let carries = carries_value(&v.name);
        let kids: Vec<Option<Iv>> = v
            .kids
            .iter()
            .enumerate()
            .map(|(i, k)| {
                // A clamped child (a min, a mask, a row's clamped offset)
                // cannot carry a wrap past its clamp.
                cx.site = if carries && !clamps_child(&v.name, i) { site } else { Site::Other };
                self.iv(k, cx)
            })
            .collect();
        cx.site = site;
        self.access(v, cx);
        let k = |i: usize| kids.get(i).copied().flatten();
        let d = self.kingdom.dual(v.ty)?;
        if v.name == "fallback" {
            // A thread's row, column or cell renders the thread's own origin.
            return match d.form.as_str() {
                "threadrow" => self.iv_raw(&Tree::new("safe.thread_row", v.ty, Vec::new()), cx),
                "threadcol" => self.iv_raw(&Tree::new("safe.thread_col", v.ty, Vec::new()), cx),
                "threadrows" | "threadrowg" => self.iv_raw(&Tree::new("safe.thread_row_s", v.ty, Vec::new()), cx),
                "threadcols" | "threadcolg" => self.iv_raw(&Tree::new("safe.thread_col_s", v.ty, Vec::new()), cx),
                "cell" => {
                    let row = self.iv_raw(&Tree::new("safe.thread_row", v.ty, Vec::new()), cx);
                    let col = self.iv_raw(&Tree::new("safe.thread_col", v.ty, Vec::new()), cx);
                    let r = self.mul(row, Some(pt(WORST_D as u128)), v, cx);
                    self.add(r, col, v, cx)
                }
                _ => (d.slot == "u32").then(|| pt(u128::from(d.fallback_bits))),
            };
        }
        if v.name.starts_with("literal.") {
            return self.lit_u32(v);
        }
        // A uniform row (legality item 5) computes as its index row or load.
        let base = crate::wgsl::render::uniform_as_index(&v.name).unwrap_or_else(|| safe_base(&v.name));
        let g = cx.grid;
        let (rc, cc) = (cx.row_comp(), cx.col_comp());
        let wdims = [u128::from(g.x), u128::from(g.y)];
        // The block factor and workgroup extent along a safe row's axis.
        let axis = |b: &str| if b.starts_with("safe.row") || b.starts_with("safe.thread_row") { (u128::from(g.by), wdims[rc], rc) } else { (u128::from(g.bx), wdims[cc], cc) };
        match base {
            "load.cfg_d" | "bound.d" => Some(pt(WORST_D as u128)),
            "load.cfg_m" | "bound.m" => Some(pt(WORST_M as u128)),
            "load.cfg_r_rows" | "bound.rows" => Some(pt(WORST_ROWS as u128)),
            "load.cfg_r_off" => Some(pt(0)),
            "safe.counter" | "builtin.item" => if cx.in_items { cx.t.iv } else { Some(pt(0)) },
            b if tile_counter_depth(b).is_some() => match tile_counter_depth(b).and_then(|n| cx.tiles.get(n - 1)) {
                Some(c) => c.iv,
                None => Some(pt(0)),
            },
            "builtin.local_invocation_index" => Some(Iv { lo: 0, hi: wdims[0] * wdims[1] - 1 }),
            // WGSL bounds a subgroup to 4..=128 lanes.
            "builtin.subgroup_invocation_id" => Some(Iv { lo: 0, hi: 127 }),
            "builtin.subgroup_size" => Some(Iv { lo: 4, hi: 128 }),
            "builtin.subgroup_id" => Some(Iv { lo: 0, hi: wdims[0] * wdims[1] - 1 }),
            "builtin.num_subgroups" => Some(Iv { lo: 1, hi: wdims[0] * wdims[1] }),
            b if b.starts_with("axis.") && d.slot == "u32" => {
                let (c, id) = b.trim_start_matches("axis.").split_once('_')?;
                let comp = match c {
                    "y" => rc,
                    "x" => cc,
                    _ => 2,
                };
                cx.comp(id, comp)
            }
            b if b.starts_with("shape.access_") && d.slot == "u32" => {
                let i: usize = b.trim_start_matches("shape.access_").parse().ok()?;
                let src = v.kids.first()?;
                match builtin_id(&src.name) {
                    Some(id) => cx.comp(id, i),
                    None => k(0),
                }
            }
            // An expression bound carries its cap (min(e, MAX_LOOP_TRIP)): known even when e is not.
            "bound.of_index" | "bound.of_uniform" | "bound.x_of_uniform" => {
                let cap = u128::from(crate::wgsl::render::MAX_LOOP_TRIP);
                Some(k(0).map_or(Iv { lo: 0, hi: cap }, |iv| Iv { lo: iv.lo.min(cap), hi: iv.hi.min(cap) }))
            }
            "safe.counter_index" | "safe.tile_index" | "safe.tile_index2" | "safe.tile_index3" | "safe.tile_index4" | "safe.row_as_index" | "safe.col_as_index" | "index.of_count" | "index.of_row" | "index.of_col" | "index.of_lidx" if d.slot == "u32" => k(0),
            "index.of_uniform" | "uniform.of_lit" | "uniform.of_counter" | "uniform.of_tcounter" | "uniform.of_tcounter2" | "uniform.of_tcounter3" | "uniform.of_tcounter4" => k(0),
            "uniform.wid_x" | "uniform.wid_y" | "uniform.wid_z" | "uniform.nwg_x" | "uniform.nwg_y" | "uniform.nwg_z" => {
                let id = if base.starts_with("uniform.wid") { "wid" } else { "nwg" };
                cx.comp(id, "xyz".find(base.chars().last()?)?)
            }
            "uniform.subgroup_size" => Some(Iv { lo: 4, hi: 128 }),
            "uniform.num_subgroups" => Some(Iv { lo: 1, hi: wdims[0] * wdims[1] }),
            "index.add" | "count.add" => self.add(k(0), k(1), v, cx),
            "index.mul" | "count.mul" => self.mul(k(0), k(1), v, cx),
            "index.sub" | "count.sub" => {
                // The same stateless expression on both sides is zero, whatever its range.
                if v.kids.len() == 2 && v.kids[0] == v.kids[1] && self.stateless(&v.kids[0]) {
                    return Some(pt(0));
                }
                let (a, b) = (k(0)?, k(1)?);
                if a.lo < b.hi {
                    self.overflow(v, cx, format!("a difference wraps below zero ([{}, {}] - [{}, {}])", a.lo, a.hi, b.lo, b.hi));
                    return None;
                }
                Some(Iv { lo: a.lo - b.hi, hi: a.hi - b.lo })
            }
            "index.sub_floor" | "count.sub_floor" => {
                let (a, b) = (k(0)?, k(1)?);
                Some(Iv { lo: a.lo.saturating_sub(b.hi), hi: a.hi.saturating_sub(b.lo) })
            }
            "index.shl" => {
                let (a, b) = (k(0)?, k(1)?);
                let s = shift_count(b);
                let hi = a.hi << s.hi;
                if hi > U32_MAX {
                    self.overflow(v, cx, format!("a shift reaches {hi}"));
                    return None;
                }
                Some(Iv { lo: a.lo << s.lo, hi })
            }
            "index.shr" => {
                let (a, b) = (k(0)?, k(1)?);
                let s = shift_count(b);
                Some(Iv { lo: a.lo >> s.hi, hi: a.hi >> s.lo })
            }
            "index.div" | "index.band_of" => {
                let (a, b) = (k(0)?, k(1)?);
                Some(Iv { lo: a.lo / b.hi.max(1), hi: a.hi / b.lo.max(1) })
            }
            "index.mod" => {
                let (a, b) = (k(0)?, k(1)?);
                Some(Iv { lo: 0, hi: a.hi.min(b.hi.saturating_sub(1)) })
            }
            "index.min" | "count.min" => match (k(0), k(1)) {
                (Some(a), Some(b)) => Some(Iv { lo: a.lo.min(b.lo), hi: a.hi.min(b.hi) }),
                (Some(x), None) | (None, Some(x)) => Some(Iv { lo: 0, hi: x.hi }),
                (None, None) => None,
            },
            "index.max" | "count.max" => {
                let (a, b) = (k(0)?, k(1)?);
                Some(Iv { lo: a.lo.max(b.lo), hi: a.hi.max(b.hi) })
            }
            "index.clamp_to" => {
                let n = k(1)?;
                Some(min_c(k(0), n.hi.max(1) - 1))
            }
            "index.and" => match (k(0), k(1)) {
                (Some(a), Some(b)) => Some(Iv { lo: 0, hi: a.hi.min(b.hi) }),
                (Some(x), None) | (None, Some(x)) => Some(Iv { lo: 0, hi: x.hi }),
                (None, None) => None,
            },
            "index.or" => {
                let (a, b) = (k(0)?, k(1)?);
                Some(Iv { lo: a.lo.max(b.lo), hi: (a.hi.max(b.hi) + 1).next_power_of_two() - 1 })
            }
            "index.round_down" => Some(Iv { lo: 0, hi: k(0)?.hi }),
            "safe.thread_row" | "safe.thread_col" => {
                let (blk, _, c) = axis(base);
                let gid = cx.comp("gid", c);
                self.mul(gid, Some(pt(blk)), v, cx)
            }
            "safe.thread_row_s" | "safe.thread_col_s" => {
                let (blk, w, c) = axis(base);
                let (wid, lid) = (cx.comp("wid", c), cx.comp("lid", c));
                let origin = self.mul(wid, Some(pt(w * blk)), v, cx);
                self.add(origin, lid, v, cx)
            }
            "safe.thread_row_g" | "safe.thread_col_g" => {
                let (blk, w, c) = axis(base);
                let groups = v.kids.get(3).map_or(1, |t| lit_count(t, 64)).clamp(1, blk.max(1));
                let (wid, lid) = (cx.comp("wid", c), cx.comp("lid", c));
                let origin = self.mul(wid, Some(pt(w * blk)), v, cx);
                let lane = self.mul(lid, Some(pt(blk / groups)), v, cx);
                self.add(origin, lane, v, cx)
            }
            b if b.starts_with("safe.row_plus") || b.starts_with("safe.col_plus") => {
                let (blk, w, _) = axis(b);
                let last = blk.max(1) - 1;
                let rest = b.trim_start_matches("safe.row_plus").trim_start_matches("safe.col_plus");
                let strided = rest.ends_with("_s");
                let offset = if rest.starts_with("_tile") || rest.starts_with("_x") {
                    min_c(k(1), last)
                } else if rest == "_g" {
                    let groups = v.kids.get(1).map_or(1, |t| lit_count(t, 64)).clamp(1, blk.max(1));
                    let s = blk.max(1) / groups;
                    let g_off = self.mul(Some(min_c(k(2), groups - 1)), Some(pt(w * s)), v, cx);
                    let e_off = min_c(k(3), s.max(1) - 1);
                    let off = self.add(g_off, Some(e_off), v, cx)?;
                    return self.add(k(0), Some(off), v, cx);
                } else {
                    pt(v.kids.get(1).map_or(1, |t| lit_count(t, 64)).min(last))
                };
                let offset = if strided { self.mul(Some(offset), Some(pt(w)), v, cx)? } else { offset };
                self.add(k(0), Some(offset), v, cx)
            }
            "safe.cell_of" | "safe.phi_of" => {
                let row = self.mul(k(0), Some(pt(WORST_D as u128)), v, cx);
                self.add(row, k(1), v, cx)
            }
            "safe.wg_of" => Some(min_c(k(0), 2047)),
            "safe.wgv_of" => Some(min_c(k(0), 511)),
            "safe.wgv2_of" => Some(min_c(k(0), 1023)),
            "safe.pidx64" => Some(min_c(k(0), 63)),
            "safe.pidx16" => Some(min_c(k(0), 15)),
            _ => None,
        }
    }

    /// An array access (a read or a write of a workgroup or private array):
    /// its index in the array's range, by the mask its row writes or by its
    /// value at the worst shape (layer 3 only).
    fn access(&self, n: &Tree, cx: &mut Cx) {
        if cx.mode != Mode::Range {
            return;
        }
        let base = safe_base(&n.name);
        let (Some(array), Some(idx)) = (n.kids.first(), n.kids.get(1)) else { return };
        let wg = matches!(base, "wg.get_safe" | "wg.get_v_safe" | "wg.get_v2_safe" | "wg.get_v_lane" | "wg.set_safe" | "wg.set_v_lane" | "wg.get" | "wg.set") || base.starts_with("atomic.");
        let parr = matches!(base, "parr.get_f" | "parr.get_v" | "parr.get_i" | "parr.get_b" | "parr.get_t" | "parr.get_v_lane" | "parr.set_safe" | "parr.set_v_lane");
        let (var, len) = if wg {
            let Ok(var) = wgmem_var(self.kingdom, array) else { return };
            (var, cx.plan.get(var).map_or_else(|| array_len(var), |(_, l)| *l))
        } else if parr {
            let Ok(var) = parr_var(self.kingdom, array) else { return };
            (var, array_len(var))
        } else {
            return;
        };
        let len = u128::from(len);
        // The unsafe kingdom's workgroup rows mask every index to 2,047,
        // whatever the array's length in this kernel's plan.
        if matches!(base, "wg.get" | "wg.set") {
            return; // masked into the array's declared length by the renderer (a1d6614, legality item 6)
        }
        let ibase = safe_base(&idx.name);
        if matches!(ibase, "safe.wg_of" | "safe.wgv_of" | "safe.wgv2_of" | "safe.wga_of") && idx.kids.len() == 1 {
            return; // masked into the array's own length by the renderer
        }
        if matches!(ibase, "safe.pidx64" | "safe.pidx16") && idx.kids.len() == 1 {
            if idx.kids[0].name.starts_with("literal.") {
                return; // a literal is folded into the array on the host
            }
            let mask: u128 = if ibase == "safe.pidx64" { 63 } else { 15 };
            if mask >= len {
                cx.fail(Layer::Range, "index_out_of_range", format!("{}: {ibase} masks its index to {mask} but {var} has {len} elements", brief(n)));
            }
            return;
        }
        match self.iv(idx, cx) {
            Some(i) if i.hi < len => {}
            Some(i) => cx.fail(Layer::Range, "index_out_of_range", format!("{}: the index reaches {} but {var} has {len} elements", brief(n), i.hi)),
            None => cx.fail(Layer::Range, "index_out_of_range", format!("{}: the index {} is not masked and is not known at the worst shape", brief(n), brief(idx))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::render::{gram_role, gram_shipped_safe_tree, sample, worst_bound, Renderer};
    use crate::wgsl::safe_reader::read_safe;
    use crate::wgsl::table::FunctionTable;

    fn setup() -> (WgslKingdom, Role) {
        (WgslKingdom::load_safe(), gram_role())
    }

    fn ty(k: &WgslKingdom, n: &str) -> Ty {
        dual_ty(k, n).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The checker's worst shape is the cost estimator's.
    #[test]
    fn the_worst_shape_is_the_cost_estimators() {
        let (k, _) = setup();
        assert_eq!(worst_bound(&Tree::new("bound.m", ty(&k, "u32.boundm"), vec![])), WORST_M);
        assert_eq!(worst_bound(&Tree::new("bound.d", ty(&k, "u32.boundd"), vec![])), WORST_D);
        assert_eq!(worst_bound(&Tree::new("bound.rows", ty(&k, "u32.boundrows"), vec![])), WORST_ROWS);
    }

    /// The shipped kernel and every corpus kernel that reads pass every layer.
    #[test]
    fn the_shipped_kernel_and_every_corpus_kernel_pass() {
        let (k, role) = setup();
        let checker = Checker::new(&k, &role, &Target::recorded());
        checker.check(&gram_shipped_safe_tree(&k).unwrap()).unwrap_or_else(|e| panic!("the shipped kernel: {e}"));
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/safewgsl"));
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).expect("manifest");
        let (mut checked, mut failures) = (0usize, Vec::new());
        for line in manifest.lines().skip(1) {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.get(1) != Some(&"reads") {
                continue;
            }
            let src = std::fs::read_to_string(dir.join(format!("{}.wgsl", cols[0]))).unwrap_or_else(|e| panic!("{}: {e}", cols[0]));
            let tree = read_safe(&src, &k).unwrap_or_else(|e| panic!("{}: does not read: {e}", cols[0])).tree;
            checked += 1;
            if let Err(e) = checker.check(&tree) {
                failures.push(format!("{}: {e}", cols[0]));
            }
        }
        println!("corpus: {checked} kernels checked, {} fail", failures.len());
        assert!(checked >= 90, "only {checked} corpus kernels read");
        assert!(failures.is_empty(), "{} of {checked} corpus kernels fail the checker:\n{}", failures.len(), failures.join("\n"));
    }

    /// The shipped kernel with its body replaced by `body` (a block.L0).
    fn with_body(k: &WgslKingdom, body: Tree) -> Tree {
        let mut tree = gram_shipped_safe_tree(k).unwrap();
        tree.kids[1] = body;
        tree
    }

    fn thread_col(k: &WgslKingdom) -> Tree {
        let gid = Tree::new("builtin.global_invocation_id", ty(k, "vec3<u32>.gid"), vec![]);
        let col = Tree::new("safe.thread_col", ty(k, "u32.threadcol"), vec![gid, Tree::literal("literal.blk", ty(k, "u32.blk"), 1.0)]);
        Tree::new("safe.col_as_index", ty(k, "u32.index"), vec![col])
    }

    fn reg_i(k: &WgslKingdom, n: &str) -> Tree {
        Tree::new("reg.get_i", ty(k, "u32.index"), vec![Tree::new(n, ty(k, "reg<u32>.index"), vec![])])
    }

    fn layer_of(checker: &Checker, tree: &Tree) -> Illegal {
        match checker.check(tree) {
            Ok(()) => panic!("passed every layer: {}", tree.to_sexpr()),
            Err(e) => e,
        }
    }

    /// Hand-made illegal trees, each failing at the layer that owns its fault.
    #[test]
    fn each_illegal_tree_fails_at_its_layer() {
        let (k, role) = setup();
        let target = Target::recorded();
        let checker = Checker::new(&k, &role, &target);
        let one_l0 = |s: Tree| Tree::new("block.one_L0", ty(&k, "block.L0"), vec![s]);
        let lt = |a: Tree, b: Tree| Tree::new("compare.lt", ty(&k, "bool.flag"), vec![a, b]);
        let lit_i = |v: f32| Tree::literal("literal.index", ty(&k, "u32.index"), v);

        // A barrier under a thread-dependent if: the types forbid it (a C
        // rung has no barrier row), so layer 1 refuses it; the grammar's
        // position rule names it on its own.
        let barrier_l0 = Tree::new("stmt.barrier_L0", ty(&k, "stmt.L0"), vec![]);
        let under_if = with_body(&k, one_l0(Tree::new("stmt.if_L0", ty(&k, "stmt.L0"), vec![lt(thread_col(&k), lit_i(7.0)), Tree::new("block.one_C0", ty(&k, "block.C0"), vec![barrier_l0])])));
        let e = layer_of(&checker, &under_if);
        assert_eq!((e.layer, e.reason), (Layer::Typed, "signature_mismatch"), "{e}");
        let e = checker.grammar(&under_if).unwrap_err();
        assert_eq!((e.layer, e.reason), (Layer::Grammar, "barrier_under_condition"), "{e}");

        // A barrier in a loop whose trip count is the thread's column: since
        // legality item 5 a loop that may hold a barrier takes a uniform bound
        // (u32.boundu), so the tree is not even typed.
        let bound = Tree::new("bound.of_index", ty(&k, "u32.boundx"), vec![thread_col(&k)]);
        let looped = with_body(&k, one_l0(Tree::new("loop.tilex_L0", ty(&k, "stmt.L0"), vec![bound, Tree::new("block.one_L2", ty(&k, "block.L2"), vec![Tree::new("stmt.barrier_L2", ty(&k, "stmt.L2"), vec![])])])));
        let e = layer_of(&checker, &looped);
        assert_eq!((e.layer, e.reason), (Layer::Typed, "signature_mismatch"), "{e}");

        // A 64-mask into a private array of 16: since legality item 4 the
        // array takes u32.pindex16, so the tree is not typed.
        let parr = Tree::new("parr.get_v", ty(&k, "vec4<f32>.real"), vec![Tree::new("parr.v", ty(&k, "parr<vec4<f32>>.real"), vec![]), Tree::new("safe.pidx64", ty(&k, "u32.pindex"), vec![reg_i(&k, "reg.i0")])]);
        let out_of_range = with_body(&k, one_l0(Tree::new("reg.set_v_L0", ty(&k, "stmt.L0"), vec![Tree::new("reg.v0", ty(&k, "reg<vec4<f32>>.real"), vec![]), parr])));
        let e = layer_of(&checker, &out_of_range);
        assert_eq!((e.layer, e.reason), (Layer::Typed, "signature_mismatch"), "{e}");

        // An index product that overflows u32 at the worst shape: m * d * d.
        let cnt = |n: &str| Tree::new("index.of_count", ty(&k, "u32.index"), vec![Tree::new(n, ty(&k, "u32.count"), vec![])]);
        let mul = |a: Tree, b: Tree| Tree::new("index.mul", ty(&k, "u32.index"), vec![a, b]);
        let big = mul(mul(cnt("load.cfg_m"), cnt("load.cfg_d")), cnt("load.cfg_d"));
        let overflow = with_body(&k, one_l0(Tree::new("reg.set_i_L0", ty(&k, "stmt.L0"), vec![Tree::new("reg.i0", ty(&k, "reg<u32>.index"), vec![]), big])));
        // A wrap is illegal everywhere (Codex's review, 2026-10-12).
        let e = layer_of(&checker, &overflow);
        assert_eq!((e.layer, e.reason), (Layer::Range, "index_wraps"), "{e}");
        assert!(e.detail.contains("product"), "{e}");

        // A wrap in a loop bound is illegal (it runs the loop to its cap), and
        // the kingdom cannot spell it: a product of uniform values may wrap,
        // so it is u32.uniformw, which no bound row takes (layer 1); the
        // bounded-loops layer names the wrap on its own. m * m = 2^32 at the
        // worst shape.
        let u = |n: &str| Tree::new(n, ty(&k, "u32.uniform"), vec![]);
        let mm = Tree::new("uniform.mul", ty(&k, "u32.uniformw"), vec![u("uniform.cfg_m"), u("uniform.cfg_m")]);
        let wrapped = Tree::new("bound.x_of_uniform", ty(&k, "u32.boundx"), vec![mm.clone()]);
        let set_f1 = Tree::new("reg.set_f_C1", ty(&k, "stmt.C1"), vec![Tree::new("reg.f1", ty(&k, "reg<f32>.real"), vec![]), Tree::literal("literal.real", ty(&k, "f32.real"), 1.0)]);
        let wraps = with_body(&k, one_l0(Tree::new("stmt.cblock_L0", ty(&k, "stmt.L0"), vec![Tree::new("block.one_C0", ty(&k, "block.C0"), vec![Tree::new("loop.tilex_C0", ty(&k, "stmt.C0"), vec![wrapped, Tree::new("block.one_C1", ty(&k, "block.C1"), vec![set_f1.clone()])])])])));
        let e = layer_of(&checker, &wraps);
        assert_eq!((e.layer, e.reason), (Layer::Typed, "signature_mismatch"), "{e}");
        let (nonuniform, _) = checker.uniform_fixpoint(&wraps);
        let e = checker.bounded(&wraps, &nonuniform).unwrap_err();
        assert_eq!((e.layer, e.reason), (Layer::Bounded, "loop_bound_wraps"), "{e}");
        // Clamped by a min it is still a wrap (Codex's review): no row spells
        // the min of a value that may wrap, and the range layer names it.
        let clamped = Tree::new("uniform.min", ty(&k, "u32.uniform"), vec![Tree::new("uniform.of_lit", ty(&k, "u32.uniform"), vec![Tree::literal("literal.lit4096", ty(&k, "u32.lit4096"), 1024.0)]), mm]);
        let mut clamped_tree = wraps.clone();
        let lp = &mut clamped_tree.kids[1].kids[0].kids[0].kids[0];
        assert_eq!(lp.name, "loop.tilex_C0");
        lp.kids[0].kids[0] = clamped;
        assert_eq!(layer_of(&checker, &clamped_tree).layer, Layer::Typed);
        let e = checker.range(&clamped_tree).unwrap_err();
        assert_eq!((e.layer, e.reason), (Layer::Range, "index_wraps"), "{e}");

        // An unbounded loop: its bound is a register its body increments.
        let xreg = |n: &str| Tree::new("reg.get_i", ty(&k, "u32.xbound"), vec![Tree::new(n, ty(&k, "reg<u32>.index"), vec![])]);
        let bound = Tree::new("bound.of_index", ty(&k, "u32.boundx"), vec![xreg("reg.i1")]);
        // (At the conditional rung, written in place: a loop with any bound.)
        let in_place = |l: Tree| Tree::new("stmt.cblock_L0", ty(&k, "stmt.L0"), vec![Tree::new("block.one_C0", ty(&k, "block.C0"), vec![l])]);
        let bump = Tree::new("reg.set_i_C1", ty(&k, "stmt.C1"), vec![Tree::new("reg.i1", ty(&k, "reg<u32>.index"), vec![]), Tree::new("index.add", ty(&k, "u32.index"), vec![reg_i(&k, "reg.i1"), lit_i(1.0)])]);
        let unbounded = with_body(&k, one_l0(in_place(Tree::new("loop.tilex_C0", ty(&k, "stmt.C0"), vec![bound, Tree::new("block.one_C1", ty(&k, "block.C1"), vec![bump])]))));
        // Since the expression bound carries its cap (min(e, MAX_LOOP_TRIP)), the loop
        // is bounded whatever its body does to the register: it passes.
        checker.check(&unbounded).unwrap_or_else(|e| panic!("a capped loop bound: {e}"));

        // A loop bounded by a thread-dependent register: not uniform, not known.
        let set_i5 = Tree::new("reg.set_i_L0", ty(&k, "stmt.L0"), vec![Tree::new("reg.i5", ty(&k, "reg<u32>.index"), vec![]), thread_col(&k)]);
        let bound = Tree::new("bound.of_index", ty(&k, "u32.boundx"), vec![xreg("reg.i5")]);
        let set_f = Tree::new("reg.set_f_C1", ty(&k, "stmt.C1"), vec![Tree::new("reg.f1", ty(&k, "reg<f32>.real"), vec![]), Tree::literal("literal.real", ty(&k, "f32.real"), 1.0)]);
        let thread_bound = with_body(&k, Tree::new("block.seq2_L0", ty(&k, "block.L0"), vec![set_i5, in_place(Tree::new("loop.tilex_C0", ty(&k, "stmt.C0"), vec![bound, Tree::new("block.one_C1", ty(&k, "block.C1"), vec![set_f])]))]));
        // Bounded by the cap too (a thread's register, no barrier in the loop).
        checker.check(&thread_bound).unwrap_or_else(|e| panic!("a capped loop bound: {e}"));

        // A workgroup over the adapter's limit: 32 x 32 on a device of 256 invocations.
        let mut wide = gram_shipped_safe_tree(&k).unwrap();
        wide.kids[0].kids[0].konst = 32.0;
        wide.kids[0].kids[1].konst = 32.0;
        checker.check(&wide).unwrap_or_else(|e| panic!("32 x 32 on the M3 Max: {e}"));
        let small = Target { max_invocations: 256, max_size: [256, 256, 64], ..target };
        let e = Checker::new(&k, &role, &small).check(&wide).unwrap_err();
        assert_eq!((e.layer, e.reason), (Layer::Adapter, "workgroup_size_over_limit"), "{e}");
        // And a subgroup row on a device without subgroups.
        let sg = with_body(&k, one_l0(Tree::new("reg.set_i_L0", ty(&k, "stmt.L0"), vec![Tree::new("reg.i0", ty(&k, "reg<u32>.index"), vec![]), Tree::new("builtin.subgroup_invocation_id", ty(&k, "u32.index"), vec![])])));
        checker.check(&sg).unwrap_or_else(|e| panic!("subgroups on the M3 Max: {e}"));
        let bare = Target { caps: target.caps - Capabilities::SUBGROUP - Capabilities::SUBGROUP_BARRIER, ..target };
        let e = Checker::new(&k, &role, &bare).check(&sg).unwrap_err();
        assert_eq!((e.layer, e.reason), (Layer::Adapter, "feature_missing"), "{e}");

        // A fallback cell is the thread's own (it renders the thread's
        // block origin): legal.
        let mut own = gram_shipped_safe_tree(&k).unwrap();
        let store = &mut own.kids[2].kids[0].kids[1].kids[0];
        assert_eq!(store.name, "store.h_O0");
        store.kids[0] = Tree::fallback(ty(&k, "u32.cell"));
        checker.check(&own).unwrap_or_else(|e| panic!("a fallback cell: {e}"));

        // A store to a cell the thread does not own: the role's `store.h`
        // at any index. Its type is the unsafe kingdom's statement, so
        // layer 1 refuses it; the grammar names the store on its own.
        let mut shared = gram_shipped_safe_tree(&k).unwrap();
        shared.kids[2].kids[0].kids[1].kids[0] = Tree::new("store.h", ty(&k, "stmt.stmt"), vec![lit_i(0.0), Tree::literal("literal.real", ty(&k, "f32.real"), 1.0)]);
        let e = layer_of(&checker, &shared);
        assert_eq!((e.layer, e.reason), (Layer::Typed, "signature_mismatch"), "{e}");
        let e = checker.grammar(&shared).unwrap_err();
        assert_eq!((e.layer, e.reason), (Layer::Grammar, "store_not_owned_cell"), "{e}");

        // A read of another thread's cell: the role's `load.h` at any index
        // is typed in the safe kingdom (the sampler reaches it).
        let other = with_body(&k, one_l0(Tree::new("reg.set_f_L0", ty(&k, "stmt.L0"), vec![Tree::new("reg.f1", ty(&k, "reg<f32>.real"), vec![]), Tree::new("load.h", ty(&k, "f32.real"), vec![lit_i(0.0)])])));
        let e = layer_of(&checker, &other);
        assert_eq!((e.layer, e.reason), (Layer::Grammar, "read_of_unowned_cell"), "{e}");

        // The item counter read before the item loop is the constant 0, as
        // the rows say and the renderer writes: legal.
        let counter = Tree::new("safe.counter_index", ty(&k, "u32.index"), vec![Tree::new("safe.counter", ty(&k, "u32.counter"), vec![])]);
        let early = with_body(&k, one_l0(Tree::new("reg.set_i_L0", ty(&k, "stmt.L0"), vec![Tree::new("reg.i0", ty(&k, "reg<u32>.index"), vec![]), counter])));
        checker.check(&early).unwrap_or_else(|e| panic!("a counter outside its loop: {e}"));
    }

    /// THE FINDING: 1,000 random genes from the kingdom's sampler (the safe
    /// root, depth 7, seeds 0..1000) through the checker; the pass count and
    /// every failing layer and reason, with the first example of each. The
    /// kingdom should admit only legal trees: every failure is a kingdom bug.
    /// Every tree that passes renders.
    #[test]
    fn random_genes_from_the_sampler_by_layer() {
        random_genes(&gram_role(), false);
    }

    /// The same 1,000 seeds under the second interface (the cosine
    /// re-score, 2026-10-12): its own bindings and its second input's row
    /// in the sampler; every gene legal, rendered and valid by naga.
    #[test]
    fn random_genes_under_the_cosine_interface() {
        random_genes(&crate::wgsl::render::cosine_role(), true);
    }

    fn random_genes(role: &Role, validate: bool) {
        let k = WgslKingdom::load_safe();
        let checker = Checker::new(&k, role, &Target::recorded());
        let table = FunctionTable::safe();
        let rr = Renderer::new(&k, &table, role);
        let root = ty(&k, "kernel.safe");
        let n = 1000u64;
        let mut passed = 0usize;
        let mut by_reason: BTreeMap<(Layer, &'static str), (usize, u64, String)> = BTreeMap::new();
        // Each layer on its own (every tree through every layer), so a layer's
        // faults are counted even where an earlier layer fails first.
        let mut alone: BTreeMap<(Layer, &'static str), (usize, u64, String)> = BTreeMap::new();
        for seed in 0..n {
            let tree = sample(&k, role, root, 7, seed).unwrap();
            let (nonuniform, uniformity) = checker.uniform_fixpoint(&tree);
            let each = [checker.typed(&tree).err(), checker.grammar(&tree).err(), checker.range(&tree).err(), uniformity, checker.adapter(&tree).err(), checker.bounded(&tree, &nonuniform).err()];
            for e in each.into_iter().flatten() {
                alone.entry((e.layer, e.reason)).or_insert((0, seed, e.detail.clone())).0 += 1;
            }
            match checker.check(&tree) {
                Ok(()) => {
                    passed += 1;
                    let text = rr.kernel(&tree).unwrap_or_else(|e| panic!("seed {seed} passes the checker but does not render: {e}")).text;
                    if validate {
                        crate::wgsl::render::validate_text(&text).unwrap_or_else(|e| panic!("seed {seed} under {}: {e}\n{text}", role.name));
                    }
                }
                Err(e) => {
                    let entry = by_reason.entry((e.layer, e.reason)).or_insert((0, seed, e.detail.clone()));
                    entry.0 += 1;
                }
            }
        }
        let (name, failed) = (&role.name, n as usize - passed);
        println!("random genes under {name}: {n} sampled, {passed} pass every layer, {failed} fail");
        for layer in Layer::ALL {
            let count: usize = by_reason.iter().filter(|((l, _), _)| *l == layer).map(|(_, (c, _, _))| c).sum();
            println!("  layer {} {:<22} {count:>4}", layer.number(), layer.name());
        }
        for ((layer, reason), (count, seed, detail)) in &by_reason {
            println!("  layer {} {reason:<36} {count:>4}  first: seed {seed}: {}", layer.number(), detail.chars().take(240).collect::<String>());
        }
        println!("  each layer on its own (a tree counted once per layer it fails):");
        for ((layer, reason), (count, seed, detail)) in &alone {
            println!("  layer {} {reason:<36} {count:>4}  first: seed {seed}: {}", layer.number(), detail.chars().take(240).collect::<String>());
        }
        assert_eq!(passed + by_reason.values().map(|(c, _, _)| c).sum::<usize>(), n as usize);
        // The kingdom admits only legal genes (Andrew's rule: an illegal gene it
        // admits is a kingdom bug): 1,000 of 1,000 since 2026-10-12.
        assert_eq!(passed, n as usize, "random genes the kingdom admits that are not legal");
    }
}
