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
use crate::wgsl::render::{dual_ty, Tree, COMPOSITES};

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
    tile_counter4: Option<String>,
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
    /// Helper functions (see `Helper`); the function being walked's call
    /// results by handle → statement path, and its call arguments by
    /// statement path.
    helpers: &'a HashMap<String, Helper<'a>>,
    calls: &'a BTreeMap<u32, Vec<usize>>,
    arg_roots: &'a HashMap<Vec<usize>, Vec<(usize, Node)>>,
    module: &'a naga::Module,
    /// While a helper is inlined as statements (seed04's `ldv`, run A): the
    /// prefix its locals' registers are keyed under, and the register (row,
    /// kind) its return value is written to.
    local_prefix: String,
    inline_result: Option<(String, char)>,
    /// The arguments of a helper inlined as statements, already read into
    /// the safe grammar IN THE CALLER (its counters, registers and lets):
    /// read lazily inside the helper, the caller's item counter became an
    /// unset register (seed04's `ldv(t0 + lt, …)` rebuilt 109 of 128 cases).
    inline_values: Option<HashMap<String, Tree>>,
    /// While a helper body is being inlined: its arguments by `arg.<name>`,
    /// already resolved and typed in the caller.
    inline_args: Option<HashMap<String, Tree>>,
    /// Locals that are vectors (`var bv = vec4<f32>(0.0)`), by name with
    /// their arena index: a `.#` load or store on one is a lane of a vec4
    /// register, not a private array.
    local_lanes: HashMap<String, u32>,
    /// Locals that have been the counter of a finished item loop: a later
    /// item loop over one of them CONTINUES it (`for (; t < m; …)`, the
    /// remainder loop of k05 and k26) unless the kernel set it back to 0.
    used_counters: std::collections::BTreeSet<String>,
    /// Private u32 arrays that hold this thread's own rows or columns (k26:
    /// `lrs[i] = wid.y * 32u + i * 8u + lid.y`): local → (axis, strided, blk).
    /// A load `lrs[k]` with a tile counter k is then that block row.
    row_arrays: BTreeMap<String, (usize, bool, u32)>,
    /// What a scalar local last held, as the expression stored into it with
    /// earlier locals expanded (`i3 = i1 + lid.y` after `i1 = wid.y * 64u`),
    /// and the block depth of that store: the ownership proof of a cell sees
    /// through a row or column kept in a register (the three r6x kernels of
    /// run A). Leaving a block (an arm of an `if`, a loop body) forgets what
    /// was stored inside it; entering a loop forgets every local the loop
    /// stores (the second iteration reads the loop's own value). The
    /// rendered store recomputes the cell from the thread's ids, never from
    /// the register.
    axis_memo: BTreeMap<String, (Tree, usize)>,
    loop_depth: usize,
    /// The function whose body is being walked (its pointers resolve the
    /// locals a loop stores).
    naga_fn: &'a Function,
}

/// A helper function the grammar inlines. One of lets and one `return expr`
/// (k05's `prod`, the kingdom's `lit_u32`) is `pure`: its return expression
/// is inlined at each use. The kingdom's own `at_vecNx` (copied verbatim by
/// a model from a rendered kernel) is `composite`: a call is `shape.at`.
/// Any other (seed04's `ldv`: locals, lets, stores, ifs, one return at the
/// end) is inlined AS STATEMENTS at its call: its locals become registers
/// keyed under the helper's name, its return value is written to a register
/// named for the call and read where the call's result is used.
pub struct Helper<'a> {
    pub args: Vec<String>,
    pub body: Option<Node>,
    pub lets: HashMap<String, Node>,
    pub pure: bool,
    pub composite: bool,
    pub roots: HashMap<Vec<usize>, Vec<&'a Root>>,
    pub inits: Vec<&'a Root>,
    /// Locals without an initialiser, with their dual: set to the fallback
    /// at each inline (WGSL zero-initialises a `var`; the register may
    /// hold an earlier call's value).
    pub uninit: Vec<(String, String)>,
    pub local_lanes: HashMap<String, u32>,
    pub calls: BTreeMap<u32, Vec<usize>>,
    pub arg_roots: HashMap<Vec<usize>, Vec<(usize, Node)>>,
    pub block: &'a naga::Block,
    pub naga_fn: &'a Function,
    pub ret_dual: Option<String>,
}

/// The register dual of a naga type: the four kinds a register holds.
fn dual_of_type(module: &naga::Module, ty: Handle<naga::Type>) -> Option<String> {
    use naga::{ScalarKind, TypeInner, VectorSize};
    match &module.types[ty].inner {
        TypeInner::Scalar(sc) => match (sc.kind, sc.width) {
            (ScalarKind::Float, 4) => Some("f32.real".into()),
            (ScalarKind::Uint, 4) => Some("u32.index".into()),
            (ScalarKind::Bool, _) => Some("bool.flag".into()),
            _ => None,
        },
        TypeInner::Vector { size: VectorSize::Quad, scalar } if scalar.kind == ScalarKind::Float && scalar.width == 4 => Some("vec4<f32>.real".into()),
        _ => None,
    }
}

fn ty(ctx: &Ctx, name: &str) -> Result<Ty, String> {
    dual_ty(ctx.kingdom, name)
}

/// The reader's node as a typed tree, duals by the inference's text, lets inlined.
/// A local's name under the helper being inlined: `load.local.v@0@v2` →
/// `load.local.ldv/v@0@v2`, so the helper's locals (registers, counters,
/// arrays, memos) never collide with the caller's of the same name.
fn prefixed(prefix: &str, name: &str) -> String {
    match name.find("local.") {
        Some(i) if !prefix.is_empty() && (i == 0 || name.as_bytes()[i - 1] == b'.') => format!("{}local.{prefix}{}", &name[..i], &name[i + 6..]),
        _ => name.to_string(),
    }
}

fn tree_of(ctx: &mut Ctx, n: &Node) -> Result<Tree, String> {
    let text = n.to_sexpr();
    match n {
        Node::Leaf { name, .. } => {
            let name = &prefixed(&ctx.local_prefix, name);
            if let Some(def) = name.strip_prefix("let.").and_then(|k| ctx.lets.get(k)) {
                let def = def.clone();
                return tree_of(ctx, &def);
            }
            if name.starts_with("arg.") {
                let bound = ctx.inline_args.as_ref().and_then(|a| a.get(name)).cloned().ok_or_else(|| format!("{name} outside an inlined helper"))?;
                // A statement-inlined helper's argument: a leaf the value
                // step replaces by the caller's safe tree.
                if ctx.inline_values.is_some() {
                    return Ok(Tree::new(name, bound.ty, vec![]));
                }
                return Ok(bound);
            }
            // A helper's result: the helper's return expression with its
            // arguments substituted (its lets inlined on the way); the
            // kingdom's own `at_vecNx` as `shape.at`; a helper inlined as
            // statements by the register its return was written to.
            if let Some((callee, idx)) = name.strip_prefix("call.").and_then(|r| r.rsplit_once('@')) {
                let idx: u32 = idx.parse().map_err(|_| format!("{name}: no handle"))?;
                let helper = ctx.helpers.get(callee).ok_or_else(|| format!("a call of {callee}, which the grammar cannot inline"))?;
                if !helper.pure && !helper.composite {
                    let d = helper.ret_dual.as_deref().ok_or_else(|| format!("{callee} returns a value no register holds"))?;
                    return Ok(Tree::new(&format!("load.local.call.{callee}@{idx}"), ty(ctx, d)?, vec![]));
                }
                let bound = ctx.call_arguments(callee, helper, idx)?;
                if helper.composite {
                    let v = bound.get(&helper.args[0]).cloned().ok_or_else(|| format!("{callee}: no vector argument"))?;
                    let i = bound.get(&helper.args[1]).cloned().ok_or_else(|| format!("{callee}: no index argument"))?;
                    let vd = ctx.kingdom.dual(v.ty).ok_or_else(|| format!("{callee}: an untyped vector"))?;
                    let scalar = vd.scalar().ok_or_else(|| format!("{callee} over {}: not a vector", vd.dual))?;
                    let t = ty(ctx, &format!("{scalar}.{}", vd.form))?;
                    return Ok(Tree::new("shape.at", t, vec![v, i]));
                }
                let saved_lets = std::mem::replace(&mut ctx.lets, &helper.lets);
                let saved_args = ctx.inline_args.replace(bound);
                let body = helper.body.clone().ok_or_else(|| format!("{callee} returns nothing"))?;
                let out = tree_of(ctx, &body);
                ctx.inline_args = saved_args;
                ctx.lets = saved_lets;
                return out.map_err(|e| format!("inlining {callee}: {e}"));
            }
            let t = ctx.duals.get(&text).copied().flatten().ok_or_else(|| format!("untyped leaf {name}"))?;
            Ok(Tree::new(name, t, vec![]))
        }
        Node::Literal { form, value, .. } => {
            // An integer literal is carried as an f32 value: exact only below 2^24.
            if form != "real" && value.abs() > 16_777_216.0 {
                return Err(format!("the literal {value} exceeds the exact range of an integer literal (2^24); spell the mask as a rounding or a shift"));
            }
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
            // Two indices in a row (`sa[si][comp]`, seed08 of run A): the
            // element's index then the lane's; the store and load spell it.
            if name == "naga.NestedAccess" && kids.len() == 2 {
                let kids = kids.iter().map(|k| tree_of(ctx, k)).collect::<Result<Vec<_>, _>>()?;
                return Ok(Tree::new("naga.NestedAccess", ty(ctx, "u32.index")?, kids));
            }
            // An element of an array VALUE chosen at run time (`let accs =
            // array<vec4<f32>, 4>(...); accs[i]`, r3x_reinvent): a select
            // chain over the elements on the index.
            if name == "shape.access_dyn" && kids.len() == 2 {
                // The array is usually a `let`: its definition.
                let base = match &kids[0] {
                    Node::Leaf { name, .. } => name.strip_prefix("let.").and_then(|k| ctx.lets.get(k)).unwrap_or(&kids[0]),
                    other => other,
                };
                if let Node::App { name: cn, kids: elems, .. } = base {
                    if cn.starts_with("naga.Compose.type") && !elems.is_empty() {
                        let elems = elems.iter().map(|e| tree_of(ctx, e)).collect::<Result<Vec<_>, _>>()?;
                        let idx = tree_of(ctx, &kids[1])?;
                        let (flag, index_ty) = (ty(ctx, "bool.flag")?, ty(ctx, "u32.index")?);
                        let mut chain = elems[elems.len() - 1].clone();
                        for (i, e) in elems.iter().enumerate().rev().skip(1) {
                            let eq = Tree::new("compare.eq", flag, vec![idx.clone(), Tree::literal("literal.index", index_ty, i as f32)]);
                            chain = Tree::new("select.scalar_cond", e.ty, vec![e.clone(), chain, eq]);
                        }
                        return Ok(chain);
                    }
                }
            }
            if !known {
                return Err(format!("{name} has no row in the table"));
            }
            // `x & ~(k - 1)` with k a power of two (k16's `m & ~3u`): the mask
            // is not exact as an f32 literal; it is `x` rounded down to a
            // multiple of k.
            if (name == "bits.and" || name == "index.and") && kids.len() == 2 {
                for (x, m) in [(&kids[0], &kids[1]), (&kids[1], &kids[0])] {
                    if let Node::Literal { value, .. } = m {
                        let k = 4_294_967_296.0 - value;
                        if *value >= 2_147_483_648.0 && k >= 1.0 && k.log2().fract() == 0.0 {
                            let t = ctx.duals.get(&text).copied().flatten().ok_or_else(|| format!("untyped node {}", &text[..text.len().min(80)]))?;
                            let x = tree_of(ctx, x)?;
                            let kt = Tree::literal("literal.index", x.ty, k as f32);
                            return Ok(Tree::new("index.round_down", t, vec![x, kt]));
                        }
                    }
                }
            }
            let kids = kids.iter().map(|k| tree_of(ctx, k)).collect::<Result<Vec<_>, _>>()?;
            // A lane of a vector chosen at run time has no type from the
            // inference when naga refused the module for it: the vector's scalar.
            let lane_of_vector = || -> Option<Ty> {
                let d = ctx.kingdom.dual(kids.first()?.ty)?;
                (name == "shape.access_dyn" && d.lanes > 1 && d.slot.starts_with("vec")).then(|| ty(ctx, &format!("{}.{}", d.scalar()?, d.form)).ok()).flatten()
            };
            let t = ctx.duals.get(&text).copied().flatten().or_else(lane_of_vector).ok_or_else(|| format!("untyped node {}", &text[..text.len().min(80)]))?;
            let name = &prefixed(&ctx.local_prefix, name);
            // A lane of a vector built in place (`vec2<f32>(a, b).x`): naga
            // flattens `vec3<f32>(vec2<f32>(a, b), c)` into the lanes of the
            // vec2, so the kingdom's own compose chains read back doubled at
            // every level without this fold (a rendered kernel read twice was
            // 9 times its size, run A).
            if let Some(lane) = name.strip_prefix("shape.access_").and_then(|i| i.parse::<usize>().ok()) {
                if kids.len() == 1 {
                    if let Some(c) = component(&kids[0], lane, ctx.kingdom) {
                        return Ok(c);
                    }
                }
            }
            Ok(Tree::new(name, t, kids))
        }
    }
}

/// Lane `lane` of a vector built by a compose or a splat, as the expression
/// that lane holds (the parts of a compose may be scalars or vectors in any
/// order; their widths are their duals' lanes); `None` for anything else.
fn component(v: &Tree, lane: usize, kingdom: &WgslKingdom) -> Option<Tree> {
    let stem = v.name.split('.').take(2).collect::<Vec<_>>().join(".");
    let lanes = |t: &Tree| kingdom.dual(t.ty).map(|d| d.lanes.max(1) as usize);
    match stem.as_str() {
        "shape.splat2" | "shape.splat3" | "shape.splat4" if v.kids.len() == 1 && lanes(&v.kids[0]) == Some(1) => Some(v.kids[0].clone()),
        "shape.compose2" | "shape.compose3" | "shape.compose4" => {
            let mut offset = 0usize;
            for kid in &v.kids {
                let w = lanes(kid)?;
                if lane < offset + w {
                    return if w == 1 { Some(kid.clone()) } else { component(kid, lane - offset, kingdom) };
                }
                offset += w;
            }
            None
        }
        _ => None,
    }
}

/// An index already masked by the row's own mask (`i & 2047u` under
/// `safe.wg_of`, read back from a rendered kernel): the index without it,
/// since the row masks again.
fn unmasked(idx: &Tree, mask: f32) -> &Tree {
    if (idx.name == "index.and" || idx.name == "bits.and") && idx.kids.len() == 2 {
        for (x, m) in [(&idx.kids[0], &idx.kids[1]), (&idx.kids[1], &idx.kids[0])] {
            if literal_value(m) == Some(mask) {
                return x;
            }
        }
    }
    idx
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

/// A target indexed twice (`sa[si][comp]`: the name carries `.#.#`) with
/// its index tree a nested access → the element's index and the lane's.
fn nested_index<'t>(rest: &str, kids: &'t [Tree]) -> Option<(&'t Tree, &'t Tree)> {
    if !rest.contains(".#.#") {
        return None;
    }
    match kids {
        [ix, ..] if ix.name == "naga.NestedAccess" && ix.kids.len() == 2 => Some((&ix.kids[0], &ix.kids[1])),
        _ => None,
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
type Tiles<'t> = [&'t Option<String>; 4];

/// The kingdom's own spelling of a block offset: `min(k, B - 1)` with k a
/// tile counter (and for a strided block `min(k, B - 1) * W`); a kernel
/// rendered by the kingdom must read back, on the `gid` path as on the
/// `wid + lid` path (r4p1 and r4p1_g2 of run A, 2026-10-11).
fn tile_min(x: &Tree, tiles: Tiles) -> Option<u8> {
    if x.name == "index.min" && x.kids.len() == 2 {
        for (c, l) in [(&x.kids[0], &x.kids[1]), (&x.kids[1], &x.kids[0])] {
            if c.kids.is_empty() && literal_value(l).is_some() {
                return local_name(&c.name).and_then(|n| tile_of(&n, tiles));
            }
        }
    }
    None
}

/// Which tile counter (1, 2 or 3) a local is, if any.
fn tile_of(local: &str, tiles: Tiles) -> Option<u8> {
    tiles.iter().position(|c| c.as_deref() == Some(local)).map(|p| p as u8 + 1)
}

/// A row or column expression of this thread: `gid.axis`, `gid.axis * K`,
/// `gid.axis * K + off` (off a literal below K, or a tile counter), with K
/// the grid's block factor → `(blk, offset)`.
fn thread_axis_off_in(t: &Tree, axis: usize, workgroup: u32, tiles: Tiles, arrays: &BTreeMap<String, (usize, bool, u32)>) -> Option<(u32, Off)> {
    // A guarded value: `select(0u, X, ok)` or `select(X, 0u, !ok)` is X here;
    // the store is guarded by its own row's bounds check anyway.
    if t.name == "select.scalar_cond" && t.kids.len() == 3 {
        for (a, b) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
            if literal_value(a) == Some(0.0) {
                if let Some(x) = thread_axis_off_in(b, axis, workgroup, tiles, arrays) {
                    return Some(x);
                }
            }
        }
    }
    // A load of a row array at a tile counter: that block row.
    if let Some(rest) = t.name.strip_prefix("load.local.") {
        if rest.contains(".#") && t.kids.len() == 1 && t.kids[0].kids.is_empty() {
            let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
            if let Some(&(ax, strided, blk)) = arrays.get(&local) {
                if ax == axis {
                    if let Some(n) = local_name(&t.kids[0].name).and_then(|l| tile_of(&l, tiles)) {
                        return Some((blk, if strided { Off::StridedTile(n) } else { Off::Tile(n) }));
                    }
                }
            }
        }
    }
    if let Some((k, 0)) = thread_axis(t, axis, workgroup) {
        return Some((k, Off::Lit(0)));
    }
    if let Some(x) = axis_terms(t, axis, workgroup, tiles) {
        return Some(x);
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
                        if literal_value(p) == Some(workgroup as f32) {
                            if q.kids.is_empty() {
                                if let Some(n) = local_name(&q.name).and_then(|l| tile_of(&l, tiles)) {
                                    return Some((b, Off::StridedTile(n)));
                                }
                            }
                            if let Some(n) = tile_min(q, tiles) {
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
                if let Some(n) = tile_min(y, tiles) {
                    return Some((k, Off::Tile(n)));
                }
            }
        }
    }
    None
}

/// The terms of a sum, however it is nested.
fn flatten_add<'t>(t: &'t Tree, out: &mut Vec<&'t Tree>) {
    if t.name == "index.add" && t.kids.len() == 2 {
        flatten_add(&t.kids[0], out);
        flatten_add(&t.kids[1], out);
    } else {
        out.push(t);
    }
}

/// A row or column of this thread written as any sum of: `wid.axis * K`,
/// `lid.axis` or `lid.axis * B`, and at most one offset (a literal, a tile
/// counter, or a tile counter times the workgroup width for a strided
/// block), e.g. k26's `wid.y * 32u + i * 8u + lid.y`.
fn axis_terms(t: &Tree, axis: usize, workgroup: u32, tiles: Tiles) -> Option<(u32, Off)> {
    let mut terms = Vec::new();
    flatten_add(t, &mut terms);
    if terms.len() < 2 || terms.len() > 3 {
        return None;
    }
    let wid = |x: &Tree| scaled_id(x, axis, "builtin.work_group_id").or_else(|| scaled_id(x, axis, "builtin.workgroup_id"));
    let lid = |x: &Tree| if id_axis(x, axis, "builtin.local_invocation_id") { Some(1) } else { scaled_id(x, axis, "builtin.local_invocation_id") };
    let kw_i = terms.iter().position(|x| wid(x).is_some())?;
    let kl_i = terms.iter().position(|x| lid(x).is_some())?;
    if kw_i == kl_i {
        return None;
    }
    let (kw, kl) = (wid(terms[kw_i])?, lid(terms[kl_i])?);
    let rest: Vec<&Tree> = terms.iter().enumerate().filter(|(i, _)| *i != kw_i && *i != kl_i).map(|(_, x)| *x).collect();
    if workgroup == 0 {
        return None;
    }
    let tile_min = |x: &Tree| tile_min(x, tiles);
    if kw == workgroup * kl {
        // A contiguous block of B = kl cells: offsets below B.
        let b = kl;
        return match rest.as_slice() {
            [] => Some((b, Off::Lit(0))),
            [x] => {
                if let Some(v) = literal_value(x) {
                    return ((v as u32) < b).then_some((b, Off::Lit(v as u32)));
                }
                if x.kids.is_empty() {
                    return local_name(&x.name).and_then(|l| tile_of(&l, tiles)).map(|n| (b, Off::Tile(n)));
                }
                tile_min(x).map(|n| (b, Off::Tile(n)))
            }
            _ => None,
        };
    }
    if kl == 1 && kw.is_multiple_of(workgroup) && kw / workgroup > 1 {
        // A strided block of B rows W apart: offsets are multiples of W.
        let b = kw / workgroup;
        return match rest.as_slice() {
            [] => Some((b, Off::Strided(0))),
            [x] => {
                if let Some(v) = literal_value(x) {
                    let v = v as u32;
                    return (v.is_multiple_of(workgroup) && v / workgroup < b).then_some((b, Off::Strided(v / workgroup)));
                }
                if x.name == "index.mul" && x.kids.len() == 2 {
                    for (p, q) in [(&x.kids[0], &x.kids[1]), (&x.kids[1], &x.kids[0])] {
                        if literal_value(p) == Some(workgroup as f32) {
                            if q.kids.is_empty() {
                                return local_name(&q.name).and_then(|l| tile_of(&l, tiles)).map(|n| (b, Off::StridedTile(n)));
                            }
                            if let Some(n) = tile_min(q) {
                                return Some((b, Off::StridedTile(n)));
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        };
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
    /// A call's arguments, resolved and typed in the caller, by the
    /// helper's argument names.
    fn call_arguments(&mut self, callee: &str, helper: &Helper<'a>, idx: u32) -> Result<HashMap<String, Tree>, String> {
        let path = self.calls.get(&idx).ok_or_else(|| format!("call.{callee}@{idx}: no call statement"))?;
        let mut args: Vec<(usize, Node)> = self.arg_roots.get(path).cloned().unwrap_or_default();
        args.sort_by_key(|(p, _)| *p);
        if args.len() != helper.args.len() {
            return Err(format!("{callee}: {} arguments at the call, {} declared", args.len(), helper.args.len()));
        }
        let mut bound = HashMap::new();
        for ((_, node), arg) in args.into_iter().zip(&helper.args) {
            bound.insert(arg.clone(), tree_of(self, &node)?);
        }
        Ok(bound)
    }

    fn register(&mut self, local: &str, dual: &str) -> Result<String, String> {
        if let Some(r) = self.registers.get(local) {
            return Ok(r.clone());
        }
        // A u32 local the inference typed as bits or a count (`i1 = i0 >> 6u`,
        // r2x_union of run A) is the u32 register too; conform converts at
        // the uses.
        let (kind, max) = match dual {
            "f32.real" => ('f', 64),
            "vec4<f32>.real" => ('v', 32),
            "u32.index" | "u32.bits" | "u32.count" => ('i', 64),
            "bool.flag" => ('b', 16),
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
            if let Some(v) = self.inline_values.as_ref().and_then(|m| m.get(&t.name)) {
                return Ok(v.clone());
            }
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
                if self.tile_counter4.as_deref() == Some(local.as_str()) {
                    return Ok(Tree::new("safe.tile_index4", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter4", ty(self, "u32.tcounter4")?, vec![])]));
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
            if let Some((i, j)) = nested_index(rest, &t.kids) {
                // A lane of a vec4 element: `sa[si][comp]`; a literal lane
                // (`sa[si].x`, the kingdom's own rendering of a lane of a
                // loaded element) is the element loaded whole, then its lane.
                let which = self.array(arr, "vec4<f32>.real")?;
                let (of, idx_dual, arr_dual, get, _) = Self::wg_rows(which);
                let i = self.value(unmasked(i, 511.0))?;
                let wg = Tree::new(of, ty(self, idx_dual)?, vec![i]);
                let element = Tree::new(which, ty(self, arr_dual)?, vec![]);
                if let Some(lane) = literal_value(unmasked(j, 3.0)).filter(|l| (0.0..=3.0).contains(l)) {
                    let whole = Tree::new(get, ty(self, "vec4<f32>.real")?, vec![element, wg]);
                    return Ok(Tree::new(&format!("shape.access_{}", lane as u32), t.ty, vec![whole]));
                }
                let j = self.value(unmasked(j, 3.0))?;
                return Ok(Tree::new("wg.get_v_lane", t.ty, vec![element, wg, j]));
            }
            let which = self.array(arr, &d)?;
            let (of, idx_dual, arr_dual, get, _) = Self::wg_rows(which);
            let idx = self.value(unmasked(&t.kids[0], if which.starts_with("wg.v") { 511.0 } else { 2047.0 }))?;
            let wg = Tree::new(of, ty(self, idx_dual)?, vec![idx]);
            return Ok(Tree::new(get, t.ty, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), wg]));
        }
        if let Some(rest) = t.name.strip_prefix("load.local.") {
            if rest.contains(".#") {
                let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
                // A lane of a vector register: `load.local.bv@8.#@v2 (idx)`; the
                // register is the local's own (named without the `.#`).
                let base = local.trim_end_matches(".#").to_string();
                if let Some((i, j)) = nested_index(rest, &t.kids) {
                    // A lane of a vec4 element of a private array: `accs[i][j]`.
                    let (arr, kind, mask) = self.private_array(&format!("{base}.#"), "vec4<f32>.real")?;
                    if kind != 'v' {
                        return Err(format!("a lane of an element of {base}, which is not a vec4 array"));
                    }
                    let i = self.value(unmasked(i, 15.0))?;
                    let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![i]);
                    let array = Tree::new(arr, ty(self, parr_dual(kind))?, vec![]);
                    if let Some(lane) = literal_value(unmasked(j, 3.0)).filter(|l| (0.0..=3.0).contains(l)) {
                        let whole = Tree::new("parr.get_v", ty(self, "vec4<f32>.real")?, vec![array, pidx]);
                        return Ok(Tree::new(&format!("shape.access_{}", lane as u32), t.ty, vec![whole]));
                    }
                    let j = self.value(unmasked(j, 3.0))?;
                    return Ok(Tree::new("parr.get_v_lane", t.ty, vec![array, pidx, j]));
                }
                if self.local_lanes.get(&base) == Some(&4) {
                    let row = self.register(&base, "vec4<f32>.real")?;
                    let idx = self.value(unmasked(&t.kids[0], 3.0))?;
                    return Ok(Tree::new("reg.get_v_lane", t.ty, vec![Tree::new(&row, ty(self, Self::reg_dual('v'))?, vec![]), idx]));
                }
                // A private array: `load.local.acc@0.#@v3 (idx)`.
                let (arr, kind, mask) = self.private_array(&local, &d)?;
                let idx = self.value(unmasked(&t.kids[0], if mask == "safe.pidx64" { 63.0 } else { 15.0 }))?;
                let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![idx]);
                let arr_dual = parr_dual(kind);
                return Ok(Tree::new(&format!("parr.get_{kind}"), t.ty, vec![Tree::new(arr, ty(self, arr_dual)?, vec![]), pidx]));
            }
        }
        if t.name.starts_with("load.") {
            return Err(format!("load {} has no safe spelling", t.name));
        }
        // A vector component chosen at run time (`v[j]`): the kingdom's `shape.at`.
        if t.name == "shape.access_dyn" && t.kids.len() == 2 && self.kingdom.dual(t.kids[0].ty).is_some_and(|d| d.lanes > 1 && d.slot.starts_with("vec")) {
            let v = self.value(&t.kids[0])?;
            let i = self.value(&t.kids[1])?;
            return Ok(Tree::new("shape.at", t.ty, vec![v, i]));
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
            'v' => ["parr.v", "parr.w"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a third vec4 local array {local}"))?,
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

    /// `t` with every scalar local the memo knows replaced by what it holds.
    fn expand_memo(&self, t: &Tree) -> Tree {
        if t.kids.is_empty() {
            // A statement-inlined helper's argument: the caller's expression
            // (fresh48 of run A: `wrow(r, c0)` storing to h[r * d + c0 + j]).
            if let Some(raw) = self.inline_args.as_ref().and_then(|a| a.get(&t.name)).filter(|_| t.name.starts_with("arg.")) {
                return self.expand_memo(raw);
            }
            if let Some(local) = local_name(&t.name).filter(|_| !t.name.contains(".#")) {
                // A loop's counter is the loop's own: its `k = 0u` before the
                // loop is not what it holds inside.
                let counter = [&self.item_counter, &self.tile_counter, &self.tile_counter2, &self.tile_counter3, &self.tile_counter4].iter().any(|c| c.as_deref() == Some(local.as_str()));
                if let Some((held, _)) = self.axis_memo.get(&local).filter(|_| !counter) {
                    return held.clone();
                }
            }
            return t.clone();
        }
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| self.expand_memo(k)).collect() }
    }

    /// `(index.add (index.mul R D) C)` with R and C this thread's row and
    /// column → `safe.cell_of(row, col)`.
    fn cell(&mut self, idx: &Tree) -> Result<Tree, String> {
        let expanded = self.expand_memo(idx);
        let idx = &expanded;
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
                    let tiles = (self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone(), self.tile_counter4.clone());
                    let tiles: Tiles = [&tiles.0, &tiles.1, &tiles.2, &tiles.3];
                    let (Some((by, i)), Some((bx, j))) = (thread_axis_off_in(r, 1, self.grid.1, tiles, &self.row_arrays), thread_axis_off_in(col, 0, self.grid.0, tiles, &self.row_arrays)) else { continue };
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
        fail(&format!("row or column is not this thread's (tile counters {:?} / {:?} / {:?} / {:?})", self.tile_counter, self.tile_counter2, self.tile_counter3, self.tile_counter4))
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

    /// Whether `c` guarding exactly one store to h is the kingdom's own guard
    /// of that store's cell: `row < cfg.r_rows && col < cfg.d && cfg.r_off +
    /// row < cfg.d` over the row and column of `h[row * cfg.d + col]`.
    fn is_own_guard(&mut self, c: &Tree, accept: &naga::Block, reject: &naga::Block, here: &[usize]) -> Result<bool, String> {
        let stmts: Vec<&Statement> = accept.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        if stmts.len() != 1 || !matches!(stmts[0], Statement::Store { .. }) || reject.iter().any(|s| !matches!(s, Statement::Emit(_))) {
            return Ok(false);
        }
        let mut p = here.to_vec();
        p.push(0);
        p.push(0);
        let Some(root) = self.root_at(&p, |k| matches!(k, RootKind::Store { .. })) else { return Ok(false) };
        let Node::App { name, kids, .. } = &root.tree else { return Ok(false) };
        if !name.starts_with("store.buffer.h.#") || kids.len() != 2 {
            return Ok(false);
        }
        let idx = tree_of(self, &kids[0])?;
        let idx = self.expand_memo(&idx);
        // The cell: (R * d) + C in either order.
        let is_d = |t: &Tree| is_leaf(t, "load.uniform.cfg.d");
        let mut cell = None;
        if idx.name == "index.add" && idx.kids.len() == 2 {
            for (prod, col) in [(&idx.kids[0], &idx.kids[1]), (&idx.kids[1], &idx.kids[0])] {
                if prod.name == "index.mul" && prod.kids.len() == 2 {
                    for (r, d) in [(&prod.kids[0], &prod.kids[1]), (&prod.kids[1], &prod.kids[0])] {
                        if is_d(d) {
                            cell = Some((r.to_sexpr(), col.to_sexpr()));
                        }
                    }
                }
            }
        }
        let Some((row, col)) = cell else { return Ok(false) };
        // The guard: and(and(lt(R, rows), lt(C, d)), lt(add(r_off, R), d)).
        let c = self.expand_memo(c);
        let lt = |t: &Tree, bound: &str| -> Option<String> { (t.name == "compare.lt" && t.kids.len() == 2 && is_leaf(&t.kids[1], bound)).then(|| t.kids[0].to_sexpr()) };
        if c.name != "logic.and" || c.kids.len() != 2 || c.kids[0].name != "logic.and" || c.kids[0].kids.len() != 2 {
            return Ok(false);
        }
        let (a, b, g) = (&c.kids[0].kids[0], &c.kids[0].kids[1], &c.kids[1]);
        let off_row = (g.name == "compare.lt" && g.kids.len() == 2 && is_leaf(&g.kids[1], "load.uniform.cfg.d") && g.kids[0].name == "index.add" && g.kids[0].kids.len() == 2 && is_leaf(&g.kids[0].kids[0], "load.uniform.cfg.r_off")).then(|| g.kids[0].kids[1].to_sexpr());
        let own = lt(a, "load.uniform.cfg.r_rows") == Some(row.clone()) && lt(b, "load.uniform.cfg.d") == Some(col) && off_row == Some(row);
        if own {
            self.notes.push("the kingdom's own guard on a store to h was read as the store alone".into());
        }
        Ok(own)
    }

    /// A helper with statements, inlined at its call (seed04's `ldv`, run
    /// A): its arguments resolved in the caller, its locals registers under
    /// the helper's name, its body walked at the caller's rung, its return
    /// value written to the call's own register.
    fn inline_call(&mut self, callee: &str, helper: &'a Helper<'a>, result: Option<u32>, here: &[usize], rung: &str) -> Result<Option<Tree>, String> {
        if !self.local_prefix.is_empty() {
            return Err(format!("{callee}: a helper with statements called from a helper (not inlined)"));
        }
        let mut args: Vec<(usize, Node)> = self.arg_roots.get(here).cloned().unwrap_or_default();
        args.sort_by_key(|(p, _)| *p);
        if args.len() != helper.args.len() {
            return Err(format!("{callee}: {} arguments at the call, {} declared", args.len(), helper.args.len()));
        }
        let mut bound = HashMap::new();
        let mut values = HashMap::new();
        for ((_, node), arg) in args.into_iter().zip(&helper.args) {
            let raw = tree_of(self, &node)?;
            values.insert(arg.clone(), self.value(&raw)?);
            bound.insert(arg.clone(), raw);
        }
        let result_reg = match result {
            Some(idx) => {
                let d = helper.ret_dual.as_deref().ok_or_else(|| format!("{callee} returns a value no register holds"))?;
                let row = self.register(&format!("local.call.{callee}@{idx}"), d)?;
                let kind = row.chars().nth(4).unwrap_or('f');
                Some((row, kind))
            }
            None => None,
        };
        // Enter the helper: its own roots, lets, calls and vector locals; its
        // locals are named under the helper (`ldv/v@0`), so the caller's
        // counters, registers, memo and row arrays stay in scope: a loop in
        // the helper nests below the caller's loops, and a store to h in it
        // (fresh48's `wrow`) is proved over the caller's row.
        let saved_roots = std::mem::replace(&mut self.roots, &helper.roots);
        let saved_lets = std::mem::replace(&mut self.lets, &helper.lets);
        let saved_calls = std::mem::replace(&mut self.calls, &helper.calls);
        let saved_arg_roots = std::mem::replace(&mut self.arg_roots, &helper.arg_roots);
        let saved_fn = std::mem::replace(&mut self.naga_fn, helper.naga_fn);
        let saved_lanes = std::mem::replace(&mut self.local_lanes, helper.local_lanes.clone());
        let saved_args = self.inline_args.replace(bound);
        let saved_values = self.inline_values.replace(values);
        let saved_tail = std::mem::replace(&mut self.tail, true);
        let saved_result = std::mem::replace(&mut self.inline_result, result_reg);
        self.local_prefix = format!("{callee}/");
        let out = self.inline_body(helper, rung);
        self.local_prefix = String::new();
        self.inline_result = saved_result;
        self.tail = saved_tail;
        self.inline_args = saved_args;
        self.inline_values = saved_values;
        self.local_lanes = saved_lanes;
        self.naga_fn = saved_fn;
        self.arg_roots = saved_arg_roots;
        self.calls = saved_calls;
        self.lets = saved_lets;
        self.roots = saved_roots;
        let mut stmts = out.map_err(|e| format!("inlining {callee}: {e}"))?;
        let st = self.stmt_ty(rung)?;
        Ok(match stmts.len() {
            0 => None,
            1 => stmts.pop(),
            _ => Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(stmts, rung)?])),
        })
    }

    fn inline_body(&mut self, helper: &'a Helper<'a>, rung: &str) -> Result<Vec<Tree>, String> {
        let st = self.stmt_ty(rung)?;
        let mut stmts = Vec::new();
        for (local, dual) in &helper.uninit {
            let zero = match dual.as_str() {
                "f32.real" => Tree::literal("literal.real", ty(self, "f32.real")?, 0.0),
                "u32.index" => Tree::literal("literal.index", ty(self, "u32.index")?, 0.0),
                "bool.flag" => Tree::literal("literal.flag", ty(self, "bool.flag")?, 0.0),
                "vec4<f32>.real" => Tree::new("shape.splat4", ty(self, "vec4<f32>.real")?, vec![Tree::literal("literal.real", ty(self, "f32.real")?, 0.0)]),
                other => return Err(format!("the local {local} ({other}) has no initialiser and no register (not inlined)")),
            };
            let row = self.register(local, dual)?;
            let kind = row.chars().nth(4).unwrap_or('f');
            stmts.push(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), zero]));
        }
        for r in &helper.inits {
            let node = r.tree.clone();
            stmts.extend(self.store_root(&node, rung).map_err(|e| format!("at init {}: {e}", r.tree.to_sexpr()))?);
        }
        stmts.extend(self.walk(helper.block, &mut Vec::new(), rung, 0)?);
        Ok(stmts)
    }

    /// A block one level deeper: what its statements store into locals is
    /// forgotten by the memo when it ends.
    fn nested<T>(&mut self, walk: impl FnOnce(&mut Self, &mut Vec<usize>) -> T, path: &mut Vec<usize>) -> T {
        self.loop_depth += 1;
        let out = walk(self, path);
        self.loop_depth -= 1;
        let depth = self.loop_depth;
        self.axis_memo.retain(|_, (_, d)| *d <= depth);
        out
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
                        let node = root.tree.clone();
                        let mut stmts = self.store_root(&node, rung)?;
                        Ok(match stmts.len() {
                            0 => None,
                            1 => stmts.pop(),
                            _ => Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(stmts, rung)?])),
                        })
                    }
                    Statement::Call { function, result, .. } => {
                        // A pure helper's call: its result is inlined where it
                        // is read, the call statement itself has no effect. A
                        // helper with statements is inlined here.
                        let callee = self.module.functions[*function].name.clone().unwrap_or_else(|| format!("fn{}", function.index()));
                        let helper = self.helpers.get(&callee).ok_or_else(|| format!("a call of {callee}, which the grammar cannot inline"))?;
                        if helper.pure || helper.composite {
                            return Ok(None);
                        }
                        self.inline_call(&callee, helper, result.map(|r| r.index() as u32), &here, rung)
                    }
                    Statement::Return { value: Some(_) } => {
                        // A helper's return, inlined: its value to the call's register.
                        let (row, kind) = self.inline_result.clone().ok_or("a return of a value outside a helper")?;
                        if cond {
                            return Err("a return under a condition in a helper (not inlined)".into());
                        }
                        if here.len() != 1 {
                            return Err("a return inside a block of a helper (not inlined)".into());
                        }
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Return)).ok_or("no return root at this statement")?;
                        let Node::App { kids, .. } = &root.tree else { return Err("a return root without a value".into()) };
                        let t = tree_of(self, &kids[0])?;
                        let v = self.value(&t)?;
                        Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), v])))
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
                        // The kingdom's own guard around a store to h, read back
                        // from a rendered kernel: the store row renders that
                        // guard itself, so the `if` is dropped rather than
                        // doubled at every copy.
                        if self.is_own_guard(&cond_tree.kids[0], accept, reject, &here)? {
                            self.tail = stmt_tail;
                            path.push(0);
                            let a = self.walk(accept, path, rung, 0);
                            path.pop();
                            self.tail = block_tail;
                            let mut a = a?;
                            return Ok(match a.len() {
                                0 => None,
                                1 => a.pop(),
                                _ => Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(a, rung)?])),
                            });
                        }
                        let crung = format!("C{level}");
                        self.tail = stmt_tail;
                        path.push(0);
                        let a = self.nested(|ctx, path| ctx.walk(accept, path, &crung, 0), path);
                        path.pop();
                        path.push(1);
                        let r = self.nested(|ctx, path| ctx.walk(reject, path, &crung, 0), path);
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

    /// A store root as statements: a local array written whole from an array
    /// constructor (`var accs = array<vec4<f32>, 4>(a0, a1, a2, a3)`, k08)
    /// becomes one private-array store per element; anything else is the
    /// ordinary store.
    fn store_root(&mut self, node: &Node, rung: &str) -> Result<Vec<Tree>, String> {
        if let Node::App { name, kids, .. } = node {
            if let (Some(rest), [Node::App { name: vname, kids: elems, .. }]) = (name.strip_prefix("store.local."), kids.as_slice()) {
                if !rest.contains(".#") && vname.starts_with("naga.Compose.") {
                    let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
                    let values = elems.iter().map(|e| tree_of(self, e)).collect::<Result<Vec<_>, _>>()?;
                    let first = values.first().ok_or("an empty array constructor")?;
                    let d = self.kingdom.dual(first.ty).map(|d| d.dual.clone()).unwrap_or_default();
                    let (arr, kind, mask) = self.private_array(&local, &d)?;
                    let st = self.stmt_ty(rung)?;
                    let mut out = Vec::new();
                    for (i, v) in values.iter().enumerate() {
                        let v = self.value(v)?;
                        let idx = Tree::literal("literal.index", ty(self, "u32.index")?, i as f32);
                        let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![idx]);
                        out.push(Tree::new(&format!("parr.set_{kind}_{rung}"), st, vec![Tree::new(arr, ty(self, parr_dual(kind))?, vec![]), pidx, v]));
                    }
                    return Ok(out);
                }
            }
        }
        let tree = tree_of(self, node)?;
        Ok(self.store(&tree, rung)?.into_iter().collect())
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
            if let Some((i, j)) = nested_index(rest, &tree.kids[..1]) {
                // A lane of a vec4 element: `sa[si][comp] = x` (seed08).
                let which = self.array(arr, "vec4<f32>.real")?;
                let (of, idx_dual, arr_dual, _, _) = Self::wg_rows(which);
                let i = self.value(unmasked(i, 511.0))?;
                let j = self.value(unmasked(j, 3.0))?;
                let v = self.value(value)?;
                let wg = Tree::new(of, ty(self, idx_dual)?, vec![i]);
                return Ok(Some(Tree::new(&format!("wg.set_v_lane_{rung}"), st, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), wg, j, v])));
            }
            let d = self.kingdom.dual(value.ty).map(|d| d.dual.clone()).unwrap_or_default();
            let which = self.array(arr, &d)?;
            let (of, idx_dual, arr_dual, _, set) = Self::wg_rows(which);
            let i = self.value(unmasked(idx, if which.starts_with("wg.v") { 511.0 } else { 2047.0 }))?;
            let v = self.value(value)?;
            let wg = Tree::new(of, ty(self, idx_dual)?, vec![i]);
            return Ok(Some(Tree::new(&format!("{set}_{rung}"), st, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), wg, v])));
        }
        if let Some(rest) = name.strip_prefix("store.local.") {
            if rest.contains(".#") {
                let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
                let [idx, value] = tree.kids.as_slice() else { return Err("a store to a local array without an index".into()) };
                // A lane of a vector register: `bv.x = …`, `v[j] = …`.
                let base = local.trim_end_matches(".#").to_string();
                if let Some((i, j)) = nested_index(rest, &tree.kids[..1]) {
                    // A lane of a vec4 element of a private array: `accs[i][j] = x`.
                    let (arr, kind, mask) = self.private_array(&format!("{base}.#"), "vec4<f32>.real")?;
                    if kind != 'v' {
                        return Err(format!("a lane of an element of {base}, which is not a vec4 array"));
                    }
                    let i = self.value(unmasked(i, 15.0))?;
                    let j = self.value(unmasked(j, 3.0))?;
                    let v = self.value(value)?;
                    let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![i]);
                    return Ok(Some(Tree::new(&format!("parr.set_v_lane_{rung}"), st, vec![Tree::new(arr, ty(self, parr_dual(kind))?, vec![]), pidx, j, v])));
                }
                if self.local_lanes.get(&base) == Some(&4) {
                    let row = self.register(&base, "vec4<f32>.real")?;
                    let i = self.value(unmasked(idx, 3.0))?;
                    let v = self.value(value)?;
                    return Ok(Some(Tree::new(&format!("reg.set_v_lane_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual('v'))?, vec![]), i, v])));
                }
                // `lrs[i] = <this thread's i-th block row>` with i a tile counter:
                // the array holds the thread's rows; a later `lrs[k]` is a row.
                if idx.kids.is_empty() {
                    let tiles = (self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone(), self.tile_counter4.clone());
                    let tiles: Tiles = [&tiles.0, &tiles.1, &tiles.2, &tiles.3];
                    if let Some(n) = local_name(&idx.name).and_then(|l| tile_of(&l, tiles)) {
                        for (axis, wg) in [(1usize, self.grid.1), (0usize, self.grid.0)] {
                            match thread_axis_off_in(value, axis, wg, tiles, &self.row_arrays) {
                                Some((blk, Off::StridedTile(m))) if m == n => {
                                    self.row_arrays.insert(local.clone(), (axis, true, blk));
                                }
                                Some((blk, Off::Tile(m))) if m == n => {
                                    self.row_arrays.insert(local.clone(), (axis, false, blk));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                let d = self.kingdom.dual(value.ty).map(|d| d.dual.clone()).unwrap_or_default();
                let (arr, kind, mask) = self.private_array(&local, &d)?;
                let i = self.value(unmasked(idx, if mask == "safe.pidx64" { 63.0 } else { 15.0 }))?;
                let v = self.value(value)?;
                let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![i]);
                let arr_dual = parr_dual(kind);
                return Ok(Some(Tree::new(&format!("parr.set_{kind}_{rung}"), st, vec![Tree::new(arr, ty(self, arr_dual)?, vec![]), pidx, v])));
            }
        }
        if let Some(local) = local_name(name) {
            // A store to the loop counter inside its loop is the loop's own step.
            if [&self.item_counter, &self.tile_counter, &self.tile_counter2, &self.tile_counter3, &self.tile_counter4].iter().any(|c| c.as_deref() == Some(local.as_str())) {
                return Ok(None);
            }
            // A finished item loop's counter set back to 0: the next item loop
            // starts afresh; any other value has no spelling.
            if self.used_counters.contains(&local) {
                return match literal_value(&tree.kids[0]) {
                    Some(v) if v.to_bits() == 0 => {
                        self.used_counters.remove(&local);
                        Ok(None)
                    }
                    _ => Err(format!("the item counter {local} set to {} between loops (only 0 has a spelling)", tree.kids[0].to_sexpr())),
                };
            }
            let d = self.kingdom.dual(tree.kids[0].ty).map(|d| d.dual.clone()).unwrap_or_default();
            let row = self.register(&local, &d)?;
            let kind = row.chars().nth(4).unwrap_or('f');
            let v = self.value(&tree.kids[0])?;
            let held = self.expand_memo(&tree.kids[0]);
            if held.size() <= 512 {
                self.axis_memo.insert(local.clone(), (held, self.loop_depth));
            } else {
                self.axis_memo.remove(&local);
            }
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
        // The condition: the first non-Emit statement of the body is an If
        // whose one arm is a Break; calls of pure helpers may precede it (the
        // kingdom's own `lit_u32` in a rendered bound, which naga calls at
        // the top of the body).
        let body_stmts_all: Vec<&Statement> = body.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        let test_pos = body_stmts_all.iter().position(|s| matches!(s, Statement::If { .. })).ok_or("a loop without a test")?;
        for s in &body_stmts_all[..test_pos] {
            let Statement::Call { function, .. } = s else { return Err("a loop whose first statement is not its test".into()) };
            let callee = self.module.functions[*function].name.clone().unwrap_or_else(|| format!("fn{}", function.index()));
            if !self.helpers.get(&callee).is_some_and(|h| h.pure || h.composite) {
                return Err(format!("a loop whose test calls {callee}, a helper with statements"));
            }
        }
        let Statement::If { accept, reject, .. } = body_stmts_all[test_pos] else { return Err("a loop whose first statement is not its test".into()) };
        let is_break = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Break));
        let is_empty = |b: &naga::Block| b.iter().all(|s| matches!(s, Statement::Emit(_)));
        let negated = if is_break(reject) && is_empty(accept) { false } else if is_break(accept) && is_empty(reject) { true } else { return Err("a loop whose test does not break".into()) };
        path.push(0);
        path.push(test_pos);
        let root = self.root_at(path, |k| matches!(k, RootKind::Condition { .. }));
        path.pop();
        path.pop();
        let root = root.ok_or("no condition root at the loop's test")?;
        let test = tree_of(self, &root.tree)?;
        let mut c = test.kids[0].clone();
        if negated {
            // naga lowers `for (; t < m;)` into `if (t >= m) { break; }`; a
            // draft writes `if (t + 4u > m) { break; }`: the break condition
            // negated is the continue condition.
            if c.name == "logic.not" && c.kids.len() == 1 {
                c = c.kids[0].clone();
            } else {
                let flipped = match c.name.as_str() {
                    "compare.ge" => "compare.lt",
                    "compare.gt" => "compare.le",
                    "compare.le" => "compare.gt",
                    "compare.lt" => "compare.ge",
                    _ => return Err(format!("a loop test `if (cond) break` that is not a comparison of the counter with a bound: {}", c.to_sexpr())),
                };
                c = Tree::new(flipped, c.ty, c.kids.clone());
            }
        }
        // The test normalised to `counter < bound`: `bound > counter`,
        // `counter <= bound` (bound + 1), `counter + k < bound` and
        // `counter + k > bound` as `if (...) break` (bound - k, floored).
        let (counter, bound) = normalise_test(&c).ok_or_else(|| format!("a loop test that is not a counter against a bound: {}", c.to_sexpr()))?;
        let bound = &bound;
        let index_ty = ty(self, "u32.index")?;
        let (bound_tree, kind) = if is_leaf(bound, "load.uniform.cfg.m") {
            (Tree::new("bound.m", ty(self, "u32.boundm")?, vec![]), "boundm")
        } else if is_leaf(bound, "load.uniform.cfg.d") {
            (Tree::new("bound.d", ty(self, "u32.boundd")?, vec![]), "boundd")
        } else if is_leaf(bound, "load.uniform.cfg.r_rows") {
            (Tree::new("bound.rows", ty(self, "u32.boundrows")?, vec![]), "boundrows")
        } else if let Some(v) = literal_value(bound).filter(|v| *v >= 1.0 && *v <= 64.0) {
            (Tree::literal("literal.lit64", ty(self, "u32.lit64")?, v), "lit64")
        } else {
            // Any other expression: the bound as written (the complete kingdom,
            // step 6); the cost estimator values it at the worst shape.
            let expr = self.value(bound)?;
            let expr = Tree { ty: index_ty, ..expr };
            (Tree::new("bound.of_index", ty(self, "u32.boundx")?, vec![expr]), "boundx")
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
                            Some(x) if (1.0..=4096.0).contains(&x) => step = Some(x as u32),
                            _ => {
                                cpath.pop();
                                return Err(format!("a loop step that is not counter + a literal in 1..=4096: {}", v.to_sexpr()));
                            }
                        }
                    }
                }
            }
            cpath.pop();
        }
        // A `loop { if (t >= m) { break; } ... t = t + s; }` steps at the body's end.
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
                        if let Some(x) = lit.filter(|x| (1.0..=4096.0).contains(x)) {
                            step = Some(x as u32);
                            last_is_step = true;
                        }
                    }
                }
            }
        }
        let step = step.ok_or_else(|| format!("a loop whose continuing block or last statement does not step its counter {counter} by a literal in 1..=4096"))?;
        // The rung of the body and the loop row.
        // A tile loop's depth is the number of tile counters already in effect;
        // its body is one rung deeper (L0 and L1 both open at L2).
        let depth = [&self.tile_counter, &self.tile_counter2, &self.tile_counter3, &self.tile_counter4].iter().filter(|c| c.is_some()).count() as u8 + 1;
        let tile = if step == 1 { "tile" } else { "tile_strided" };
        // A literal-bounded loop is a tile loop at any rung; an expression-bounded
        // loop below the top level is a tilex loop (the counter of its depth,
        // the bound as written); a binding- or expression-bounded loop at the
        // top level is an item loop. A tile loop's step is at most 64.
        let tilex = if step == 1 { "tilex" } else { "tilex_strided" };
        let short = |t: &str, rung: &str| -> Result<String, String> {
            if step > 64 {
                return Err(format!("a short loop stepped by {step} (at most 64)"));
            }
            Ok(format!("loop.{t}_{rung}"))
        };
        let (body_rung, row_name, out_rung) = match (kind, rung) {
            ("lit64", "L0" | "L1") => ("L2".to_string(), short(tile, rung)?, rung),
            ("lit64", "L2") => ("L3".to_string(), short(tile, "L2")?, "L2"),
            ("lit64", "L3") => ("L4".to_string(), short(tile, "L3")?, "L3"),
            ("lit64", "L4") => ("L5".to_string(), short(tile, "L4")?, "L4"),
            ("lit64", "C0") => ("C1".to_string(), short(tile, "C0")?, "C0"),
            ("lit64", "C1") => ("C2".to_string(), short(tile, "C1")?, "C1"),
            ("lit64", "C2") => ("C3".to_string(), short(tile, "C2")?, "C2"),
            ("lit64", "C3") => ("C4".to_string(), short(tile, "C3")?, "C3"),
            ("lit64", "C4") => ("C5".to_string(), short(tile, "C4")?, "C4"),
            ("lit64", _) => return Err(format!("a tile loop at rung {rung} (too deep)")),
            (b, "L0") if self.used_counters.contains(&counter) && step == 1 => ("L1".to_string(), format!("loop.items_cont_{b}"), "L0"),
            (b, "L0") if self.used_counters.contains(&counter) => ("L1".to_string(), format!("loop.items_cont_strided_{b}"), "L0"),
            (b, "L0") if step == 1 => ("L1".to_string(), format!("loop.items_{b}"), "L0"),
            (b, "L0") => ("L1".to_string(), format!("loop.items_strided_{b}"), "L0"),
            ("boundx", "L1") => ("L2".to_string(), short(tilex, "L1")?, "L1"),
            ("boundx", "L2") => ("L3".to_string(), short(tilex, "L2")?, "L2"),
            ("boundx", "L3") => ("L4".to_string(), short(tilex, "L3")?, "L3"),
            ("boundx", "L4") => ("L5".to_string(), short(tilex, "L4")?, "L4"),
            ("boundx", "C0") => ("C1".to_string(), short(tilex, "C0")?, "C0"),
            ("boundx", "C1") => ("C2".to_string(), short(tilex, "C1")?, "C1"),
            ("boundx", "C2") => ("C3".to_string(), short(tilex, "C2")?, "C2"),
            ("boundx", "C3") => ("C4".to_string(), short(tilex, "C3")?, "C3"),
            ("boundx", "C4") => ("C5".to_string(), short(tilex, "C4")?, "C4"),
            ("boundx", _) => return Err(format!("an expression-bounded loop at rung {rung} (too deep)")),
            (_, _) => return Err(format!("a long loop bounded by a binding at rung {rung} (only at the top level; spell the bound as an expression to nest it)")),
        };
        let _ = cond;
        if (kind == "lit64" || kind == "boundx") && depth > 4 && body_rung != "L1" {
            return Err("a fifth nested tile loop".into());
        }
        let saved = (self.item_counter.clone(), self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone(), self.tile_counter4.clone());
        self.axis_memo.remove(&counter);
        if body_rung == "L1" {
            self.item_counter = Some(counter.clone());
        } else {
            match depth {
                1 => self.tile_counter = Some(counter.clone()),
                2 => self.tile_counter2 = Some(counter.clone()),
                3 => self.tile_counter3 = Some(counter.clone()),
                _ => self.tile_counter4 = Some(counter.clone()),
            }
        }
        // The body after its test, at the body rung: the test is statement
        // `test_pos` of the body block (path + [0]); the rest follow it.
        let mut rest: Vec<Statement> = body_stmts_all.iter().skip(test_pos + 1).map(|s| (*s).clone()).collect();
        if last_is_step {
            rest.pop();
        }
        let rest = naga::Block::from_vec(rest);
        path.push(0);
        for local in stored_locals(self.naga_fn, body).into_iter().chain(stored_locals(self.naga_fn, continuing)) {
            self.axis_memo.remove(&prefixed(&self.local_prefix, &local));
        }
        let body_stmts = self.nested(|ctx, path| ctx.walk(&rest, path, &body_rung, test_pos + 1), path);
        path.pop();
        self.item_counter = saved.0;
        self.tile_counter = saved.1;
        self.tile_counter2 = saved.2;
        self.tile_counter3 = saved.3;
        self.tile_counter4 = saved.4;
        if body_rung == "L1" {
            self.used_counters.insert(counter.clone());
        }
        let body_stmts = body_stmts?;
        let block = self.block(body_stmts, &body_rung)?;
        let st = self.stmt_ty(out_rung)?;
        let step_lit = if body_rung == "L1" { Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, step as f32) } else { Tree::literal("literal.lit64", ty(self, "u32.lit64")?, step as f32) };
        let kids = if step != 1 { vec![bound_tree, step_lit, block] } else { vec![bound_tree, block] };
        Ok(Some(Tree::new(&row_name, st, kids)))
    }

}

/// A loop test as `(counter local, bound tree)`: `c < b`, `b > c`, `c <= b`
/// (bound + 1), `c + k < b` (bound − k, floored at zero), and under the
/// reader's negation `c + k > b` has already become `c + k < b`... the
/// counter is a bare local on one side, possibly plus a literal.
fn normalise_test(c: &Tree) -> Option<(String, Tree)> {
    if c.kids.len() != 2 {
        return None;
    }
    let (l, r) = (&c.kids[0], &c.kids[1]);
    // Orient as `left < right` or `left <= right`.
    let (lhs, rhs, le) = match c.name.as_str() {
        "compare.lt" => (l, r, false),
        "compare.le" => (l, r, true),
        "compare.gt" => (r, l, false),
        "compare.ge" => (r, l, true),
        _ => return None,
    };
    let index_ty = lhs.ty;
    let bound = if le { Tree::new("index.add", index_ty, vec![rhs.clone(), Tree::literal("literal.index", index_ty, 1.0)]) } else { rhs.clone() };
    // The counter alone.
    if lhs.kids.is_empty() {
        return local_name(&lhs.name).map(|c| (c, bound));
    }
    // The counter plus a literal: counter < bound - k.
    if lhs.name == "index.add" && lhs.kids.len() == 2 {
        for (a, b) in [(&lhs.kids[0], &lhs.kids[1]), (&lhs.kids[1], &lhs.kids[0])] {
            if a.kids.is_empty() {
                if let (Some(counter), Some(k)) = (local_name(&a.name), literal_value(b)) {
                    return Some((counter, Tree::new("index.sub_floor", index_ty, vec![bound, Tree::literal("literal.index", index_ty, k)])));
                }
            }
        }
    }
    None
}

/// The locals a block stores into, anywhere inside it, as `local.<name>@<index>`.
fn stored_locals(f: &Function, block: &naga::Block) -> Vec<String> {
    fn base_local(f: &Function, mut e: Handle<Expression>) -> Option<String> {
        loop {
            match &f.expressions[e] {
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => e = *base,
                Expression::LocalVariable(l) => return Some(format!("local.{}@{}", f.local_variables[*l].name.clone().unwrap_or_else(|| "local".into()), l.index())),
                _ => return None,
            }
        }
    }
    fn walk(f: &Function, block: &naga::Block, out: &mut Vec<String>) {
        for s in block.iter() {
            match s {
                Statement::Store { pointer, .. } => out.extend(base_local(f, *pointer)),
                Statement::Block(b) => walk(f, b, out),
                Statement::If { accept, reject, .. } => {
                    walk(f, accept, out);
                    walk(f, reject, out);
                }
                Statement::Loop { body, continuing, .. } => {
                    walk(f, body, out);
                    walk(f, continuing, out);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| walk(f, &c.body, out)),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(f, block, &mut out);
    out
}

/// A zero value: the literal 0 of any form, or a vector, matrix or array
/// built from zeros (`vec4<f32>(0.0)`, `vec4<f32>()`).
fn is_zero(n: &Node) -> bool {
    match n {
        Node::Literal { value, .. } => *value == 0.0,
        Node::App { name, kids, .. } => (name == "naga.ZeroValue" || name.starts_with("shape.") || name.starts_with("naga.Compose") || name.starts_with("naga.Splat")) && kids.iter().all(is_zero) && (name == "naga.ZeroValue" || !kids.is_empty()),
        Node::Leaf { .. } => false,
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

/// The dual of a text the inference gave several candidates (a `u32` used
/// as an index here and a count there): the one the kingdom's arithmetic
/// prefers, in the order index, count, bits, real, int, flag; `conform`
/// inserts the conversion wherever a row demands the other. A text with one
/// candidate keeps it; one with none stays untyped (and is named).
fn choose_dual(set: &std::collections::BTreeSet<Option<Ty>>, kingdom: &WgslKingdom) -> Option<Ty> {
    let candidates: Vec<Ty> = set.iter().copied().flatten().collect();
    if candidates.len() <= 1 {
        return candidates.first().copied();
    }
    for form in ["index", "count", "bits", "real", "int", "flag"] {
        if let Some(t) = candidates.iter().find(|t| kingdom.dual(**t).is_some_and(|d| d.form == form)) {
            return Some(*t);
        }
    }
    candidates.first().copied()
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
    let f = kernel.functions.iter().find(|f| f.entry_point).ok_or("no entry point")?;
    let naga_fn: &Function = &kernel.module.entry_points.first().ok_or("no entry point")?.function;
    // Helper functions: pure ones (lets and one return) inlined as
    // expressions at their calls (k05), the kingdom's own `at_vecNx` read as
    // `shape.at`, any other inlined as statements at its call (seed04).
    let mut helpers: HashMap<String, Helper> = HashMap::new();
    let mut helper_duals: Vec<(String, Option<Ty>)> = Vec::new();
    for (handle, nf) in kernel.module.functions.iter() {
        let name = nf.name.clone().unwrap_or_else(|| format!("fn{}", handle.index()));
        let hf = kernel.functions.iter().find(|x| !x.entry_point && x.name == name).ok_or_else(|| format!("helper {name} was not read"))?;
        let mut lets = HashMap::new();
        let mut body = None;
        let mut roots: HashMap<Vec<usize>, Vec<&Root>> = HashMap::new();
        let mut inits: Vec<&Root> = Vec::new();
        let mut arg_roots: HashMap<Vec<usize>, Vec<(usize, Node)>> = HashMap::new();
        for r in &hf.roots {
            match (&r.kind, &r.tree) {
                (RootKind::Let { name }, Node::App { kids, .. }) => {
                    lets.insert(name.clone(), kids[0].clone());
                }
                (RootKind::Return, Node::App { kids, .. }) if body.is_none() => {
                    body = Some(kids[0].clone());
                    roots.entry(r.path.clone()).or_default().push(r);
                }
                (RootKind::Return, _) => return Err(format!("helper {name} returns twice (not inlined)")),
                (RootKind::Init { .. }, _) => inits.push(r),
                (RootKind::Argument { position, .. }, Node::App { kids, .. }) => arg_roots.entry(r.path.clone()).or_default().push((*position, kids[0].clone())),
                _ => {
                    roots.entry(r.path.clone()).or_default().push(r);
                }
            }
        }
        let pure = inits.is_empty() && arg_roots.is_empty() && roots.values().all(|rs| rs.iter().all(|r| matches!(r.kind, RootKind::Return)));
        let composite = name.starts_with("at_vec") && nf.arguments.len() == 2 && COMPOSITES.iter().any(|(n, text)| *n == name && source.contains(text));
        if body.is_none() && (pure || composite) {
            return Err(format!("helper {name} returns nothing"));
        }
        let args = nf.arguments.iter().enumerate().map(|(i, a)| format!("arg.{}", a.name.clone().unwrap_or_else(|| i.to_string()))).collect();
        let mut local_lanes = HashMap::new();
        let mut uninit = Vec::new();
        for (h, lv) in nf.local_variables.iter() {
            let local = format!("local.{name}/{}@{}", lv.name.clone().unwrap_or_else(|| "local".into()), h.index());
            if let naga::TypeInner::Vector { size, .. } = &kernel.module.types[lv.ty].inner {
                local_lanes.insert(local.clone(), *size as u32);
            }
            if lv.init.is_none() {
                uninit.push((local, dual_of_type(&kernel.module, lv.ty).unwrap_or_else(|| "no register".into())));
            }
        }
        let ret_dual = nf.result.as_ref().and_then(|r| dual_of_type(&kernel.module, r.ty));
        let types = infer_function(hf, kingdom);
        helper_duals.extend(types.by_text.iter().map(|(text, set)| (text.clone(), choose_dual(set, kingdom))));
        helpers.insert(name, Helper { args, body, lets, pure, composite, roots, inits, uninit, local_lanes, calls: hf.calls.clone(), arg_roots, block: &nf.body, naga_fn: nf, ret_dual });
    }
    // Roots by statement path; lets by name; duals by text (one form, else untyped).
    let mut roots: HashMap<Vec<usize>, Vec<&Root>> = HashMap::new();
    let mut lets: HashMap<String, Node> = HashMap::new();
    let mut inits: Vec<&Root> = Vec::new();
    let mut arg_roots: HashMap<Vec<usize>, Vec<(usize, Node)>> = HashMap::new();
    for r in &f.roots {
        match &r.kind {
            RootKind::Let { name } => {
                if let Node::App { kids, .. } = &r.tree {
                    lets.insert(name.clone(), kids[0].clone());
                }
            }
            RootKind::Init { .. } => inits.push(r),
            RootKind::Argument { position, .. } => {
                if let Node::App { kids, .. } = &r.tree {
                    arg_roots.entry(r.path.clone()).or_default().push((*position, kids[0].clone()));
                }
            }
            _ => roots.entry(r.path.clone()).or_default().push(r),
        }
    }
    let types = infer_function(f, kingdom);
    let mut duals: BTreeMap<String, Option<Ty>> = types.by_text.iter().map(|(text, set)| (text.clone(), choose_dual(set, kingdom))).collect();
    for (text, t) in helper_duals {
        duals.entry(text).or_insert(t);
    }
    let calls = f.calls.clone();
    // The vector locals, from naga's own types.
    let mut local_lanes: HashMap<String, u32> = HashMap::new();
    for (h, lv) in naga_fn.local_variables.iter() {
        if let naga::TypeInner::Vector { size, .. } = &kernel.module.types[lv.ty].inner {
            let name = lv.name.clone().unwrap_or_else(|| "local".into());
            local_lanes.insert(format!("local.{name}@{}", h.index()), *size as u32);
        }
    }
    let mut ctx = Ctx { kingdom, duals: &duals, lets: &lets, roots: &roots, item_counter: None, tile_counter: None, tile_counter2: None, tile_counter3: None, tile_counter4: None, registers: BTreeMap::new(), next_reg: BTreeMap::new(), guard: None, tail: true, arrays: BTreeMap::new(), parrs: BTreeMap::new(), grid, notes: Vec::new(), helpers: &helpers, calls: &calls, arg_roots: &arg_roots, inline_args: None, row_arrays: BTreeMap::new(), used_counters: std::collections::BTreeSet::new(), local_lanes, axis_memo: BTreeMap::new(), loop_depth: 0, naga_fn, module: &kernel.module, local_prefix: String::new(), inline_result: None, inline_values: None };
    // Register initialisations first, in root order. A local initialised to
    // its form's fallback (0.0, 0u, false, a zero vector) costs nothing: the
    // register's declaration already holds that value, and the register is
    // allocated at the local's first use. Five model kernels of run A copied
    // the kingdom's own declarations (every register and the four counters,
    // all zero) and were refused a 33rd u32 register for it.
    let mut stmts = Vec::new();
    for r in inits {
        if let Node::App { kids, .. } = &r.tree {
            if kids.len() == 1 && is_zero(&kids[0]) {
                continue;
            }
        }
        let node = r.tree.clone();
        stmts.extend(ctx.store_root(&node, "L0").map_err(|e| format!("at init {}: {e}", r.tree.to_sexpr()))?);
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

    /// THE CORPUS (Andrew, 2026-10-11: "I want a fully complete kingdom"):
    /// every draft and every model reply in `tests/corpus/safewgsl` does what
    /// its manifest line says — reads into the kingdom, is refused by a
    /// design rule (named), or is invalid WGSL that naga itself refuses. The
    /// table is printed; one mismatch against `expected` (today's truth) fails
    /// the test; `target` is what completeness demands and the unmet count
    /// is printed until it is zero.
    #[test]
    fn the_corpus_reads_as_its_manifest_says() {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/safewgsl"));
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).expect("manifest");
        let k = WgslKingdom::load_safe();
        let mut mismatches = Vec::new();
        let mut unmet = 0usize;
        let mut counts = std::collections::BTreeMap::new();
        for line in manifest.lines().skip(1) {
            let cols: Vec<&str> = line.split('\t').collect();
            let (name, expected, target) = (cols[0], cols[1], cols[2]);
            let src = std::fs::read_to_string(dir.join(format!("{name}.wgsl"))).unwrap_or_else(|e| panic!("{name}: {e}"));
            // Two outcomes: the kernel reads, or our software has a bug
            // (naga cannot parse the text our renderer led a model to, or
            // the reader has no spelling for a construct); the bug is named.
            let outcome = match read(&src) {
                Err(e) => ("bug".to_string(), format!("naga: {}", e.lines().next().unwrap_or("").chars().take(80).collect::<String>())),
                Ok(_) => match read_safe(&src, &k) {
                    Ok(r) => ("reads".to_string(), format!("{} nodes", r.tree.size())),
                    Err(e) => ("bug".to_string(), format!("reader: {}", e.chars().take(80).collect::<String>())),
                },
            };
            *counts.entry(outcome.0.clone()).or_insert(0usize) += 1;
            let ok = outcome.0 == expected;
            if outcome.0 != target {
                unmet += 1;
            }
            println!("{:<34} expected {:<9} target {:<9} got {:<8} {}{}", name, expected, target, outcome.0, if ok { "" } else { "MISMATCH " }, outcome.1);
            if !ok {
                mismatches.push(format!("{name}: expected {expected}, got {} ({})", outcome.0, outcome.1));
            }
        }
        println!("corpus: {counts:?}; targets unmet: {unmet}");
        assert!(mismatches.is_empty(), "{} mismatches:\n{}", mismatches.len(), mismatches.join("\n"));
    }

    /// THE KINGDOM'S OWN OUTPUT READS BACK (run A, 2026-10-11: models copy
    /// rendered kernels, and five copies were declined for the kingdom's own
    /// register declarations and its own `gid.y * 4u + min(k, 3u)`): every
    /// corpus kernel that reads is rendered, and the text reads again and
    /// renders again. The sizes are printed: a second reading inlines what
    /// the first rendered, so the tree grows; it is not required to be the
    /// same tree.
    #[test]
    fn every_rendered_corpus_kernel_reads_back() {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/safewgsl"));
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).expect("manifest");
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let mut failures = Vec::new();
        let mut checked = 0usize;
        for line in manifest.lines().skip(1) {
            let cols: Vec<&str> = line.split('\t').collect();
            let (name, expected) = (cols[0], cols[1]);
            if expected != "reads" {
                continue;
            }
            let src = std::fs::read_to_string(dir.join(format!("{name}.wgsl"))).unwrap_or_else(|e| panic!("{name}: {e}"));
            let first = match read_safe(&src, &k) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("{name}: does not read: {e}"));
                    continue;
                }
            };
            let text = match rr.kernel(&first.tree) {
                Ok(o) => format!("// grid {} {} {} {}\n{}", o.grid.x, o.grid.y, o.grid.bx, o.grid.by, o.text),
                Err(e) => {
                    failures.push(format!("{name}: does not render: {e}"));
                    continue;
                }
            };
            checked += 1;
            match read_safe(&text, &k) {
                Err(e) => failures.push(format!("{name}: the rendered text does not read back: {}", &e[..e.len().min(300)])),
                Ok(again) => match rr.kernel(&again.tree) {
                    Ok(o) => match validate_text(&o.text) {
                        Ok(()) => println!("{name:<40} {:>5} nodes, read back {:>5}", first.tree.size(), again.tree.size()),
                        Err(e) => failures.push(format!("{name}: read back and rendered, naga refuses the text: {}", e.lines().next().unwrap_or(""))),
                    },
                    Err(e) => failures.push(format!("{name}: read back, does not render: {e}")),
                },
            }
        }
        println!("{checked} rendered kernels read back");
        assert!(failures.is_empty(), "{} of {checked}:\n{}", failures.len(), failures.join("\n"));
    }

    /// THE POPULATION AND THE LIBRARY ENTER A RUN WHOLE (Andrew, 2026-10-11,
    /// after run A's r5 was costed at 3,023 per item against a bound of
    /// 2,560 and kept out of run A2): every kernel text in
    /// `logs/HGP/001/popA_g001/texts` and `logs/HGP/001/library_texts` reads,
    /// renders within the default cost bound, and the rendered text reads
    /// back. r5's per-item cost is pinned below the bound.
    #[test]
    fn the_population_and_the_library_read_render_within_the_bound_and_read_back() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let bound = crate::wgsl::safe_karva::default_bound(&k).unwrap();
        let mut failures = Vec::new();
        let mut checked = 0usize;
        let mut r5_per_item = None;
        for dir in ["logs/HGP/001/popA_g001/texts", "logs/HGP/001/library_texts"] {
            let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/")).join(dir);
            let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
            files.sort();
            for f in files {
                let name = f.file_name().unwrap().to_string_lossy().to_string();
                let src = std::fs::read_to_string(&f).unwrap();
                checked += 1;
                let tree = match read_safe(&src, &k) {
                    Ok(r) => r.tree,
                    Err(e) => {
                        failures.push(format!("{name}: our software has a bug, it does not read: {}", &e[..e.len().min(200)]));
                        continue;
                    }
                };
                let c = crate::wgsl::render::cost(&tree);
                if name.contains("g001_r5") {
                    r5_per_item = Some(c.per_item);
                }
                if c.per_item > bound.per_item || c.base > bound.base {
                    failures.push(format!("{name}: cost {:.0} per item, {:.0} base exceeds the bound {:.0}, {:.0}", c.per_item, c.base, bound.per_item, bound.base));
                }
                let text = match rr.kernel(&tree) {
                    Ok(o) => format!("// grid {} {} {} {}\n{}", o.grid.x, o.grid.y, o.grid.bx, o.grid.by, o.text),
                    Err(e) => {
                        failures.push(format!("{name}: does not render: {e}"));
                        continue;
                    }
                };
                if let Err(e) = validate_text(&text) {
                    failures.push(format!("{name}: naga on the rendered text: {}", e.lines().next().unwrap_or("")));
                }
                if let Err(e) = read_safe(&text, &k) {
                    failures.push(format!("{name}: the rendered text does not read back: {}", &e[..e.len().min(200)]));
                }
                println!("{name:<44} {:>5} nodes, {:>6.0} per item, {:>6.0} base", tree.size(), c.per_item, c.base);
            }
        }
        println!("{checked} kernels; bound {:.0} per item, {:.0} base", bound.per_item, bound.base);
        assert!(failures.is_empty(), "{} of {checked}:\n{}", failures.len(), failures.join("\n"));
        // r5 (run A: feasible, HFF 0.5213) was 3,023 per item when conform copied every compose part once per lane; 1,015 since.
        let r5 = r5_per_item.expect("r5 is in the population and the library");
        assert!(r5 < 1100.0 && r5 < bound.per_item, "r5 per item {r5}");
    }

    /// A kernel with a helper function reads: the helper is inlined at its
    /// calls (the complete kingdom, step 6; `load4`, with a local, an
    /// if-else and lane stores, as statements since run A), or the refusal
    /// names the helper and what in it the grammar cannot inline. The
    /// blocked kernel's own `continue` is what refuses it now, by name.
    #[test]
    fn a_kernel_with_a_helper_function_is_inlined_or_named() {
        let k = WgslKingdom::load_safe();
        match read_safe(BLOCKED, &k) {
            Ok(r) => assert!(r.tree.size() > 10),
            Err(e) => {
                assert!(!e.contains("helper") && !e.contains("load4"), "the helper is inlined, not named as the bug: {e}");
                assert!(e.contains("Continue has no safe spelling"), "{e}");
            }
        }
    }
}
