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
    /// Each mapped workgroup array's declared length, by its kingdom row.
    array_lens: BTreeMap<&'static str, u32>,
    /// Local arrays by name → a private array row, its kind and its mask row.
    parrs: BTreeMap<String, (&'static str, char, &'static str)>,
    grid: (u32, u32, u32, u32),
    /// The layout (2026-10-12): Some(true) when the output's rows are on
    /// thread x (the header's `rows-x`, or the first store's proof found
    /// them there), Some(false) on y, None until a store decides it.
    rows_x: Option<bool>,
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
    /// The main function's lets bound to a register at their definition
    /// (a use sits inside a loop that does not enclose the definition), by
    /// name, with the path of the statement each precedes.
    bound_lets: HashMap<String, Vec<usize>>,
    /// The bound lets already set, with their register row.
    let_regs: HashMap<String, String>,
    /// A subgroup or atomic result inlined into a loop's bound (`subgroup@<h>`,
    /// `atomic@<h>` to its expression): naga runs it as a statement ahead of
    /// the loop's test, the bound re-evaluates it every pass.
    inlined: HashMap<String, Tree>,
    /// The locals stored so far in reading order (a counter never stored
    /// enters its loop at WGSL's zero), and a tile loop's counter offset
    /// while its body is read (`for (var w = 2u; ...)`: the counter is
    /// 2 + k, the loop runs bound - 2 times).
    stored_seen: std::collections::BTreeSet<String>,
    counter_offsets: HashMap<String, f32>,
    /// A statement-inlined helper's pointer parameters: `arg.p` to the
    /// caller's register for the local `&x` it was given (row, kind).
    pointer_args: HashMap<String, (String, char)>,
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
    /// The tile counters' literal bounds, while their loops are open.
    tile_bounds: Bounds,
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
                // A let bound to a register at its definition reads the register.
                // (spelled as a load of the local `let.<name>`, which the value
                // step reads as the register that local holds).
                if let Some(k) = name.strip_prefix("let.").filter(|k| ctx.local_prefix.is_empty() && ctx.let_regs.contains_key(*k)) {
                    let local = format!("load.local.let.{k}");
                    let t = tree_of(ctx, &def)?;
                    return Ok(Tree::new(&local, t.ty, vec![]));
                }
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
            // A subgroup operation's result: the register its statement wrote.
            if let Some(t) = ctx.inlined.get(name.as_str()) {
                return Ok(Tree::new(INLINED, t.ty, vec![t.clone()]));
            }
            // A subgroup operation's or a workgroupUniformLoad's result: the
            // register its statement wrote.
            if let Some((what, idx)) = name.strip_prefix("subgroup@").map(|i| ("subgroup", i)).or_else(|| name.strip_prefix("uniformload@").map(|i| ("uniformload", i))) {
                let local = format!("local.{what}@{idx}");
                let row = ctx.registers.get(&local).cloned().ok_or_else(|| format!("{name} read before its subgroup operation"))?;
                let d = match row.chars().nth(4) {
                    Some('f') => "f32.real",
                    Some('v') => "vec4<f32>.real",
                    Some('b') => "bool.flag",
                    Some('n') => "i32.int",
                    Some('u') => "vec4<u32>.bits",
                    _ => "u32.index",
                };
                return Ok(Tree::new(&format!("load.{local}"), ty(ctx, d)?, vec![]));
            }
            // An atomic's old value: the register its statement wrote.
            if let Some(idx) = name.strip_prefix("atomic@") {
                // An atomic<i32>'s old value is in an i32 register.
                let local = format!("local.atomic@{idx}");
                let d = if ctx.registers.get(&local).is_some_and(|r| r.starts_with("reg.n")) { "i32.int" } else { "u32.index" };
                return Ok(Tree::new(&format!("load.{local}"), ty(ctx, d)?, vec![]));
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
                    // The kingdom's own `col_matCxRf`: a matrix's column at run time.
                    // (`CxR`: the column vector has R lanes.)
                    if let Some(rows) = callee.strip_prefix("col_mat").and_then(|r| r.strip_suffix('f')).and_then(|r| r.rsplit('x').next()) {
                        let t = ty(ctx, &format!("vec{rows}<f32>.real"))?;
                        return Ok(Tree::new("shape.col_at", t, vec![v, i]));
                    }
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
                // A naga node the WGSL table has no row for may have one in
                // the safe table (rows the safe kingdom added: insertBits,
                // dot on integer vectors, % on reals): resolved there by its
                // operand's scalar kind and arity.
                // A matrix of 3 or 4 columns (the WGSL table spells 2), and a
                // matrix's column by position.
                let slot = slot_of(n);
                if name.starts_with("naga.Compose.type") && slot.starts_with("mat") && (kids.len() == 3 || kids.len() == 4) {
                    return tree_of(ctx, &Node::App { name: format!("shape.compose_mat{}.{slot}", kids.len()), slot: slot.to_string(), known: true, kids: kids.clone() });
                }
                if let (Some(i), [k]) = (name.strip_prefix("naga.AccessIndex."), kids.as_slice()) {
                    if slot_of(k).starts_with("mat") {
                        return tree_of(ctx, &Node::App { name: format!("shape.col_{i}"), slot: slot.to_string(), known: true, kids: kids.clone() });
                    }
                }
                // A field of modf / frexp's struct (through a `let` of it), and
                // the length of a binding.
                if let (Some(i), [k]) = (name.strip_prefix("naga.AccessIndex."), kids.as_slice()) {
                    let base = match k {
                        Node::Leaf { name: l, .. } => l.strip_prefix("let.").and_then(|key| ctx.lets.get(key)).unwrap_or(k),
                        other => other,
                    };
                    if let Node::App { name: f, kids: args, .. } = base {
                        let row = match (f.as_str(), i) {
                            ("naga.Math.Modf", "0") => Some("arith.modf_fract"),
                            ("naga.Math.Modf", "1") => Some("arith.modf_whole"),
                            ("naga.Math.Frexp", "0") => Some("arith.frexp_fract"),
                            ("naga.Math.Frexp", "1") => Some("arith.frexp_exp"),
                            _ => None,
                        };
                        if let (Some(row), [x]) = (row, args.as_slice()) {
                            return tree_of(ctx, &Node::App { name: row.into(), slot: slot.to_string(), known: true, kids: vec![x.clone()] });
                        }
                    }
                }
                if let (true, [Node::Leaf { name: g, .. }]) = (name == "naga.ArrayLength", kids.as_slice()) {
                    let row = match g.as_str() {
                        "global.phi" => "load.phi_len",
                        "global.h" => "load.h_len",
                        other => return Err(format!("arrayLength of {other}: the Gram role's runtime-sized bindings are phi and h")),
                    };
                    return Ok(Tree::new(row, ty(ctx, "u32.count")?, vec![]));
                }
                if let Some(row) = safe_row_for(name, kids) {
                    return tree_of(ctx, &Node::App { name: row, slot: slot_of(n).to_string(), known: true, kids: kids.clone() });
                }
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
            // A row the inference does not know (one the safe table added):
            // its output with the node's own slot, among the rows of its name
            // and arity.
            let by_row = || -> Option<Ty> {
                let slot = slot_of(n);
                // A compose carries its slot in its name (`shape.compose_mat3.mat3x3<f32>`).
                let row = if name.starts_with("shape.compose") { name.split('.').take(2).collect::<Vec<_>>().join(".") } else { name.clone() };
                ctx.kingdom.rows.iter().filter(|r| r.name == row && r.inputs.len() == kids.len()).map(|r| r.output).find(|&o| ctx.kingdom.dual(o).is_some_and(|d| d.slot == slot))
            };
            let t = ctx.duals.get(&text).copied().flatten().or_else(lane_of_vector).or_else(by_row).ok_or_else(|| format!("untyped node {}", &text[..text.len().min(80)]))?;
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
        't' => "parr<vec2<f32>>.real",
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

/// The safe table's row for an unresolved `naga.<Node>.<Variant>` node, by
/// the scalar kind of its first operand and its arity.
fn safe_row_for(name: &str, kids: &[Node]) -> Option<String> {
    // The i32 crossings the WGSL table has no row for (naga's As by kinds).
    let cast = match name {
        "naga.As.uint_to_sint.convert" => Some("convert.index_to_int"),
        "naga.As.sint_to_uint.convert" => Some("convert.int_to_index"),
        "naga.As.sint_to_uint.bitcast" => Some("convert.bitcast_int_bits"),
        "naga.As.uint_to_sint.bitcast" => Some("convert.bitcast_bits_int"),
        "naga.As.float_to_sint.bitcast" => Some("convert.bitcast_real_int"),
        "naga.As.sint_to_float.bitcast" => Some("convert.bitcast_int_real"),
        _ => None,
    };
    if let Some(row) = cast {
        return Some(row.to_string());
    }
    static TABLE: std::sync::OnceLock<crate::wgsl::table::FunctionTable> = std::sync::OnceLock::new();
    let (node, variant) = name.strip_prefix("naga.")?.split_once('.')?;
    let kind = match slot_of(kids.first()?) {
        s if s.contains("f32") || s.contains("f16") => crate::wgsl::table::ScalarKind::Float,
        s if s.contains("i32") => crate::wgsl::table::ScalarKind::Sint,
        s if s.contains("u32") => crate::wgsl::table::ScalarKind::Uint,
        s if s.contains("bool") => crate::wgsl::table::ScalarKind::Bool,
        _ => return None,
    };
    TABLE.get_or_init(crate::wgsl::table::FunctionTable::safe).resolve(&format!("{node}::{variant}"), kind, kids.len()).map(|t| t.name())
}

/// A reader node's WGSL slot.
fn slot_of(n: &Node) -> &str {
    match n {
        Node::Leaf { slot, .. } | Node::Literal { slot, .. } | Node::App { slot, .. } => slot,
    }
}

/// Whether a loop body (after its test; through the block naga makes of a
/// `for` body) has `if (c) { break; }` at its top level.
fn has_break_guard(stmts: &[Statement]) -> bool {
    let only_break = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Break));
    let empty = |b: &naga::Block| b.iter().all(|s| matches!(s, Statement::Emit(_)));
    stmts.iter().any(|s| match s {
        Statement::Block(b) => has_break_guard(b),
        Statement::If { accept, reject, .. } => (only_break(accept) && empty(reject)) || (only_break(reject) && empty(accept)),
        _ => false,
    })
}

/// The break condition of a guard statement: `if (c) { break; }` is c,
/// `if (c) { } else { break; }` is !c; None for any other statement.
fn break_guard(t: &Tree, ctx: &Ctx) -> Option<Result<Tree, String>> {
    let only_break = |b: &Tree| {
        let mut inner = Vec::new();
        flatten_stmts(b.clone(), &mut inner);
        matches!(inner.as_slice(), [s] if s.name == "stmt.break")
    };
    let empty = |b: &Tree| b.name == "fallback" || {
        let mut inner = Vec::new();
        flatten_stmts(b.clone(), &mut inner);
        inner.is_empty()
    };
    match (rung_stem(&t.name), t.kids.as_slice()) {
        (Some("stmt.if"), [c, b]) if only_break(b) => Some(Ok(c.clone())),
        (Some("stmt.if_else"), [c, a, b]) if only_break(b) && empty(a) => Some(ty(ctx, "bool.flag").map(|f| Tree::new("logic.not", f, vec![c.clone()]))),
        (Some("stmt.if_else"), [c, a, b]) if only_break(a) && empty(b) => Some(Ok(c.clone())),
        _ => None,
    }
}

/// Whether a statement tree holds a break anywhere in it.
fn has_break(t: &Tree) -> bool {
    t.name == "stmt.break" || t.kids.iter().any(has_break)
}

/// A `var<private>` load or store as a function local's: a private variable
/// is the thread's own for the whole invocation, as a local of the entry
/// point is (`load.private.pg@v1` is `load.local.private_pg@g@v1`,
/// `store.private.pa.#` is `store.local.private_pa@g.#`). None for any
/// other node.
fn private_as_local(t: &Tree) -> Option<Tree> {
    let (op, rest) = t.name.strip_prefix("load.private.").map(|r| ("load", r)).or_else(|| t.name.strip_prefix("store.private.").map(|r| ("store", r)))?;
    let cut = rest.find(['.', '@']).unwrap_or(rest.len());
    let (var, tail) = rest.split_at(cut);
    Some(Tree { name: format!("{op}.local.private_{var}@g{tail}"), ty: t.ty, konst: t.konst, kids: t.kids.clone() })
}

/// A helper body of guards `if (c) { return a; }` and a last `return b;` (or
/// a last `if (c) { return a; } else { return b; }`) as one value,
/// select(select(b, a2, c2), a1, c1) built from the last return up; the
/// guards' condition roots are taken out of `roots` (they are read into the
/// value). None for any other body.
fn guard_chain<'r>(body: &naga::Block, returns: &[&'r Root], roots: &mut HashMap<Vec<usize>, Vec<&'r Root>>) -> Option<Node> {
    let stmts: Vec<&Statement> = body.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
    let value_at = |path: &[usize]| -> Option<Node> {
        returns.iter().find(|r| r.path == path).and_then(|r| match &r.tree {
            Node::App { kids, .. } => kids.first().cloned(),
            _ => None,
        })
    };
    let only_return = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Return { value: Some(_) }));
    let empty = |b: &naga::Block| b.iter().all(|s| matches!(s, Statement::Emit(_)));
    let cond_at = |path: Vec<usize>, roots: &mut HashMap<Vec<usize>, Vec<&'r Root>>| -> Option<Node> {
        let rs = roots.remove(&path)?;
        let c = rs.iter().find(|r| matches!(r.kind, RootKind::Condition { .. })).and_then(|r| match &r.tree {
            Node::App { kids, .. } => kids.first().cloned(),
            _ => None,
        });
        let others: Vec<&Root> = rs.into_iter().filter(|r| !matches!(r.kind, RootKind::Condition { .. })).collect();
        if !others.is_empty() {
            roots.insert(path, others);
        }
        c
    };
    let select = |accept: Node, reject: Node, c: Node| {
        let slot = match &accept {
            Node::Leaf { slot, .. } | Node::Literal { slot, .. } | Node::App { slot, .. } => slot.clone(),
        };
        Node::App { name: "select.scalar_cond".into(), slot, known: true, kids: vec![accept, reject, c] }
    };
    let last = stmts.len().checked_sub(1)?;
    let mut rest = match stmts[last] {
        Statement::Return { value: Some(_) } => value_at(&[last])?,
        Statement::If { accept, reject, .. } if only_return(accept) && only_return(reject) => {
            let a = value_at(&[last, 0, 0])?;
            let b = value_at(&[last, 1, 0])?;
            select(a, b, cond_at(vec![last], roots)?)
        }
        _ => return None,
    };
    for i in (0..last).rev() {
        let Statement::If { accept, reject, .. } = stmts[i] else { return None };
        if !only_return(accept) || !empty(reject) {
            return None;
        }
        let a = value_at(&[i, 0, 0])?;
        rest = select(a, rest, cond_at(vec![i], roots)?);
    }
    Some(rest)
}

/// The marker of a tree already in the safe grammar inside a raw one.
const INLINED: &str = "safe.inlined";

/// The increments of a read-add-write of h at the cell `idx`: `value` a sum
/// (arith.add, any nesting) with the cell's load as one term; the other
/// terms in the order the kernel's adds apply them to the load. None when
/// the load is not a term of the sum.
fn h_increments<'t>(value: &'t Tree, idx: &str) -> Option<Vec<&'t Tree>> {
    if value.name.starts_with("load.buffer.h.#") && value.kids.len() == 1 && value.kids[0].to_sexpr() == idx {
        return Some(Vec::new());
    }
    if value.name == "arith.add" && value.kids.len() == 2 {
        for (inner, other) in [(&value.kids[0], &value.kids[1]), (&value.kids[1], &value.kids[0])] {
            if let Some(mut xs) = h_increments(inner, idx) {
                xs.push(other);
                return Some(xs);
            }
        }
    }
    None
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
#[derive(Clone, PartialEq)]
enum Off {
    Lit(u32),
    Tile(u8),
    Strided(u32),
    StridedTile(u8),
    /// Any index expression the counters' bounds prove below the block
    /// factor (`2 * q + s`, fit_A2's r2x_reinvent); for a strided block, in
    /// units of the workgroup width. Rendered clamped into the block.
    Expr(Tree),
    StridedExpr(Tree),
    /// A GROUPED block (fit_A2's r6x_reinvent): the thread's B cells as G
    /// groups of B / G contiguous cells, the groups W * B / G apart; the
    /// group index and the lane index, both proved below their bounds and
    /// rendered clamped.
    Grouped(u32, Tree, Tree),
}

/// The tile counters' literal bounds by local name, for proving an offset.
type Bounds = BTreeMap<String, u32>;

/// The largest value an offset expression takes: a literal, a tile counter
/// (its bound less one), a product by a literal, a sum, a `min` by a
/// literal, a shift right or a mask by a literal; `None` for anything else.
fn offset_max(t: &Tree, tiles: Tiles, bounds: &Bounds) -> Option<f64> {
    if let Some(v) = literal_value(t) {
        return Some(f64::from(v));
    }
    if t.kids.is_empty() {
        let local = local_name(&t.name)?;
        tile_of(&local, tiles)?;
        return bounds.get(&local).map(|b| f64::from(b.saturating_sub(1)));
    }
    match (t.name.as_str(), t.kids.as_slice()) {
        ("index.add", [a, b]) => Some(offset_max(a, tiles, bounds)? + offset_max(b, tiles, bounds)?),
        ("index.mul", [a, b]) => Some(offset_max(a, tiles, bounds)? * offset_max(b, tiles, bounds)?),
        // `x >> s` with a literal s is at most max(x) / 2^s, and `x & l` with a
        // literal l is at most l: one counter split into a group and a lane
        // (fit_A2's r2x_reinvent: rows (r >> 2u) * 64u + lid.y * 4u + (r & 3u), r < 8).
        ("bits.shr" | "index.shr", [a, b]) => {
            let s = literal_value(b)?;
            Some((offset_max(a, tiles, bounds)? / 2f64.powf(f64::from(s).min(31.0))).floor())
        }
        ("bits.and" | "index.and", [a, b]) => match (literal_value(a), literal_value(b)) {
            (Some(l), _) | (_, Some(l)) => Some(f64::from(l).min(offset_max(if literal_value(a).is_some() { b } else { a }, tiles, bounds).unwrap_or(f64::INFINITY))),
            _ => Some(offset_max(a, tiles, bounds)?.min(offset_max(b, tiles, bounds)?)),
        },
        // `min(x, l)` with a literal l is at most l whatever x is (the kingdom's own clamp).
        ("index.min", [a, b]) => match (literal_value(a), literal_value(b)) {
            (Some(l), _) | (_, Some(l)) => Some(f64::from(l).min(offset_max(if literal_value(a).is_some() { b } else { a }, tiles, bounds).unwrap_or(f64::INFINITY))),
            _ => Some(offset_max(a, tiles, bounds)?.min(offset_max(b, tiles, bounds)?)),
        },
        _ => None,
    }
}

/// The kingdom's own clamp `min(x, B - 1)` read back: the offset x itself,
/// so the clamp is not nested at every reading.
fn unclamped(t: &Tree, b: u32) -> &Tree {
    if t.name == "index.min" && t.kids.len() == 2 {
        for (x, l) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
            if literal_value(l) == Some(b.saturating_sub(1) as f32) {
                return x;
            }
        }
    }
    t
}

/// A strided block's offset term in units of the workgroup width W: a
/// literal multiple of W, `W * x` or `x * W`, or `c * x` with c a multiple
/// of W.
fn strided_unit(t: &Tree, workgroup: u32) -> Option<Tree> {
    if let Some(v) = literal_value(t) {
        let v = v as u32;
        return v.is_multiple_of(workgroup).then(|| Tree::literal("literal.index", t.ty, (v / workgroup) as f32));
    }
    if t.name == "index.mul" && t.kids.len() == 2 {
        for (p, q) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
            if let Some(c) = literal_value(p) {
                let c = c as u32;
                if c == workgroup {
                    return Some(q.clone());
                }
                if c.is_multiple_of(workgroup) && c > 0 {
                    return Some(Tree::new("index.mul", t.ty, vec![Tree::literal("literal.index", p.ty, (c / workgroup) as f32), q.clone()]));
                }
            }
        }
    }
    None
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
fn thread_axis_off_in(t: &Tree, axis: usize, workgroup: u32, tiles: Tiles, arrays: &BTreeMap<String, (usize, bool, u32)>, bounds: &Bounds) -> Option<(u32, Off)> {
    // A guarded value: `select(0u, X, ok)` or `select(X, 0u, !ok)` is X here;
    // the store is guarded by its own row's bounds check anyway.
    if t.name == "select.scalar_cond" && t.kids.len() == 3 {
        for (a, b) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
            if literal_value(a) == Some(0.0) {
                if let Some(x) = thread_axis_off_in(b, axis, workgroup, tiles, arrays, bounds) {
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
    if let Some(x) = axis_terms(t, axis, workgroup, tiles, bounds) {
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
                // Any other offset the counters' bounds (or the kingdom's own
                // clamp) prove below the block factor.
                if offset_max(y, tiles, bounds).is_some_and(|m| m < f64::from(k)) {
                    return Some((k, Off::Expr(unclamped(y, k).clone())));
                }
            }
        }
    }
    // `gid.axis * K` plus SEVERAL offset terms, however the sum is nested
    // (fit_A2's r8x_reinvent: i15 = i14 + 4u with i14 = gid.x * 8u + k2, k2 < 4,
    // so the column is (gid.x * 8u + k2) + 4u): the offsets' sum proved below K
    // from the counters' bounds.
    let mut terms = Vec::new();
    flatten_add(t, &mut terms);
    if terms.len() > 2 {
        if let Some(at) = terms.iter().position(|x| matches!(thread_axis(x, axis, workgroup), Some((_, 0))) && (gid_axis(x, axis) || x.name == "index.mul")) {
            let (k, _) = thread_axis(terms[at], axis, workgroup)?;
            let sum = terms.iter().enumerate().filter(|(i, _)| *i != at).map(|(_, x)| (*x).clone()).reduce(|acc, x| Tree::new("index.add", acc.ty, vec![acc, x]))?;
            if offset_max(&sum, tiles, bounds).is_some_and(|m| m < f64::from(k)) {
                return Some((k, Off::Expr(unclamped(&sum, k).clone())));
            }
        }
    }
    None
}

/// An index expression with every literal factor distributed over the sums
/// it multiplies and literal products folded: `(wid * 16u + lid) * 4u` is
/// `wid * 64u + lid * 4u` (fit_A2 gen 2's fresh43 writes its block so), so
/// every algebraically equal spelling of a thread's block reaches the
/// ownership match in one form.
fn distribute_literals(t: &Tree) -> Tree {
    let t = Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(distribute_literals).collect() };
    if t.name != "index.mul" || t.kids.len() != 2 {
        return t;
    }
    let (lit, x) = match (literal_value(&t.kids[0]), literal_value(&t.kids[1])) {
        (Some(_), Some(_)) => {
            let mut folded = t.kids[0].clone();
            folded.konst = t.kids[0].konst * t.kids[1].konst;
            return folded;
        }
        (Some(_), None) => (&t.kids[0], &t.kids[1]),
        (None, Some(_)) => (&t.kids[1], &t.kids[0]),
        (None, None) => return t,
    };
    // A product of a non-literal by a literal stays as written.
    let sum_or_product = x.name == "index.add" || (x.name == "index.mul" && x.kids.iter().any(|k| literal_value(k).is_some()));
    if !sum_or_product {
        return t;
    }
    scale_by(x, lit, t.ty)
}

/// `x * lit` with x already normalised: over a sum, each term scaled; over a
/// product by a literal, the literals folded; otherwise the product as
/// written (`x * lit`, the literal second).
fn scale_by(x: &Tree, lit: &Tree, ty: Ty) -> Tree {
    if x.name == "index.add" && x.kids.len() == 2 {
        return Tree::new("index.add", x.ty, vec![scale_by(&x.kids[0], lit, ty), scale_by(&x.kids[1], lit, ty)]);
    }
    if let Some(v) = literal_value(x) {
        let mut folded = x.clone();
        folded.konst = v * lit.konst;
        return folded;
    }
    if x.name == "index.mul" && x.kids.len() == 2 {
        for (p, q) in [(&x.kids[0], &x.kids[1]), (&x.kids[1], &x.kids[0])] {
            if let Some(v) = literal_value(p) {
                let mut folded = p.clone();
                folded.konst = v * lit.konst;
                return Tree::new("index.mul", x.ty, vec![q.clone(), folded]);
            }
        }
    }
    Tree::new("index.mul", ty, vec![x.clone(), lit.clone()])
}

/// The statement at a reader path (naga's Emits skipped; an if's or a
/// loop's statement followed by 0 or 1 for its branch or body).
fn stmt_at<'b>(body: &'b naga::Block, path: &[usize]) -> Option<&'b Statement> {
    let nth = |b: &'b naga::Block, i: usize| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).nth(i);
    let mut block = body;
    let mut i = 0;
    loop {
        let s = nth(block, *path.get(i)?)?;
        if i + 1 == path.len() {
            return Some(s);
        }
        match s {
            Statement::Block(b) => {
                block = b;
                i += 1;
            }
            Statement::If { accept, reject, .. } => {
                block = if *path.get(i + 1)? == 0 { accept } else { reject };
                i += 2;
            }
            Statement::Loop { body, continuing, .. } => {
                block = if *path.get(i + 1)? == 0 { body } else { continuing };
                i += 2;
            }
            _ => return None,
        }
    }
}

/// Whether the statement at `used` sits inside a loop that does not enclose
/// a definition made in the block at `def_block`.
fn in_deeper_loop(body: &naga::Block, def_block: &[usize], used: &[usize]) -> bool {
    if used.len() <= def_block.len() || used[..def_block.len()] != *def_block {
        return false;
    }
    (def_block.len()..used.len() - 1).any(|i| matches!(stmt_at(body, &used[..=i]), Some(Statement::Loop { .. })) && used.len() > i + 1)
}

/// A loop's test (`if (!(k < n)) { break; }`, the reader's condition root):
/// evaluated as the loop's header, so a let it reads is used at the loop.
fn is_loop_test(body: &naga::Block, path: &[usize]) -> bool {
    let only_break = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Break));
    path.len() >= 3 && path[path.len() - 2] == 0 && matches!(stmt_at(body, &path[..path.len() - 2]), Some(Statement::Loop { .. })) && matches!(stmt_at(body, path), Some(Statement::If { accept, reject, .. }) if only_break(accept) || only_break(reject))
}

/// The lets of a function bound to a register at their definition: those
/// that read memory and have a use (followed through the lets that inline
/// them) inside a loop that does not enclose the definition (fit_A2's
/// r4x_reinvent: a vec4 of four workgroup reads per kk, used in the 4 x 4
/// loop inside, was reloaded at every multiply-add). A use in a loop's own
/// test counts as at the loop. Index arithmetic of ids, uniforms and
/// counters stays inlined: binding it too (measured on the corpus, 2026-10-11)
/// numbered each kernel's geometry by its own registers, and the drafts k15
/// and k07 shared 13 nodes instead of over 30. By name, with the path of the
/// statement each precedes.
fn lets_used_in_deeper_loops(roots: &[Root], body: &naga::Block) -> HashMap<String, Vec<usize>> {
    fn mentions(n: &Node, leaf: &str) -> bool {
        match n {
            Node::Leaf { name, .. } => name == leaf,
            Node::App { kids, .. } => kids.iter().any(|k| mentions(k, leaf)),
            _ => false,
        }
    }
    let defs: Vec<(&String, &Vec<usize>, &Node)> = roots.iter().filter_map(|r| if let RootKind::Let { name } = &r.kind { Some((name, &r.path, &r.tree)) } else { None }).collect();
    // A let's use sites: other roots that read it; a let that reads it
    // contributes its own use sites (it is inlined there) or, once bound,
    // its definition.
    let mut bound: HashMap<String, Vec<usize>> = HashMap::new();
    // Later lets first, so a reading let's binding is known before its reads are followed.
    let mut order: Vec<usize> = (0..defs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(defs[i].0.rsplit('@').next().and_then(|x| x.parse::<u32>().ok()).unwrap_or(0)));
    fn uses(name: &str, roots: &[Root], body: &naga::Block, bound: &HashMap<String, Vec<usize>>, depth: usize) -> Vec<Vec<usize>> {
        let leaf = format!("let.{name}");
        let mut out = Vec::new();
        if depth > 64 {
            return out;
        }
        for r in roots {
            if !mentions(&r.tree, &leaf) {
                continue;
            }
            match &r.kind {
                RootKind::Let { name: m } if m == name => {}
                RootKind::Let { name: m } => match bound.get(m) {
                    Some(p) => out.push(p.clone()),
                    None => out.extend(uses(m, roots, body, bound, depth + 1)),
                },
                _ if is_loop_test(body, &r.path) => out.push(r.path[..r.path.len() - 2].to_vec()),
                _ => out.push(r.path.clone()),
            }
        }
        out
    }
    // A read of memory (a buffer, a workgroup or private array): a uniform,
    // a builtin, a register and index arithmetic are not, and a let of only
    // those stays inlined (the ownership proof and shared sections see its
    // expression).
    fn reads_memory(n: &Node, defs: &[(&String, &Vec<usize>, &Node)]) -> bool {
        match n {
            Node::Leaf { name, .. } => match name.strip_prefix("let.") {
                Some(k) => defs.iter().find(|(m, _, _)| m.as_str() == k).is_some_and(|(_, _, t)| reads_memory(t, defs)),
                None => false,
            },
            Node::App { name, kids, .. } => {
                // A load other than a uniform's or a scalar local's (a register).
                let register = name.starts_with("load.local.") && !name.contains(".#");
                let memory = name.starts_with("load.") && !name.starts_with("load.uniform.") && !register;
                memory || kids.iter().any(|k| reads_memory(k, defs))
            }
            _ => false,
        }
    }
    for i in order {
        let (name, path, tree) = defs[i];
        let def_block = &path[..path.len().saturating_sub(1)];
        if reads_memory(tree, &defs) && uses(name, roots, body, &bound, 0).iter().any(|u| in_deeper_loop(body, def_block, u)) {
            bound.insert(name.clone(), path.clone());
        }
    }
    bound
}

/// A tree with every component of a thread id (`shape.access_k` of
/// `builtin.global_invocation_id`, `local_invocation_id`, `workgroup_id`)
/// spelled as the axis row it is: x the column axis, y the row axis, z a
/// plain index; with the rows on x (`rows_x`), x the row axis and y the
/// column axis.
fn by_axis(t: &Tree, rows_x: bool) -> Tree {
    let kids: Vec<Tree> = t.kids.iter().map(|k| by_axis(k, rows_x)).collect();
    if let (Some(k), [b]) = (t.name.strip_prefix("shape.access_"), kids.as_slice()) {
        let id = match b.name.as_str() {
            "builtin.global_invocation_id" => Some("gid"),
            "builtin.local_invocation_id" => Some("lid"),
            "builtin.workgroup_id" => Some("wid"),
            _ => None,
        };
        let component = match (k, rows_x) {
            ("0", false) | ("1", true) => Some("x"),
            ("1", false) | ("0", true) => Some("y"),
            ("2", _) => Some("z"),
            _ => None,
        };
        if let (Some(id), Some(component)) = (id, component) {
            return Tree { name: format!("axis.{component}_{id}"), ty: t.ty, konst: t.konst, kids };
        }
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids }
}

/// Why a continue outside the epilogue has no spelling.
const CONTINUE_OUTSIDE: &str = "a continue outside the kernel's epilogue: a later barrier of its loop would be reached by some threads only; the epilogue's tile loops, which hold no barrier, spell `if (c) { continue; }`";

/// Whether a statement tree holds a continue anywhere in it.
fn has_continue(t: &Tree) -> bool {
    t.name == "stmt.continue" || t.kids.iter().any(has_continue)
}

/// `if (c) { continue; }`: an if whose block is one continue.
fn is_continue_guard(t: &Tree) -> bool {
    if rung_stem(&t.name) != Some("stmt.if") || t.kids.len() != 2 {
        return false;
    }
    let mut inner = Vec::new();
    flatten_stmts(t.kids[1].clone(), &mut inner);
    matches!(inner.as_slice(), [c] if c.name == "stmt.continue")
}

/// Whether a statement tree stores h anywhere in it.
fn has_store(t: &Tree) -> bool {
    t.name.starts_with("store.h_") || t.kids.iter().any(has_store)
}

/// A row name's stem without its statement rung (`stmt.if_L2` is `stmt.if`);
/// `None` for a row with no rung (an expression).
fn rung_stem(name: &str) -> Option<&str> {
    let (stem, rung) = name.rsplit_once('_')?;
    let mut c = rung.chars();
    match (c.next(), c.next(), c.next()) {
        (Some('L' | 'C' | 'O'), Some(n), None) if n.is_ascii_digit() => Some(stem),
        _ => None,
    }
}

/// A statement's statements in order, a block statement and a block opened
/// in place (the reader nests the fifth statement on as a block statement).
fn flatten_stmts(s: Tree, out: &mut Vec<Tree>) {
    let open = |n: &str| rung_stem(n).is_some_and(|b| matches!(b, "stmt.block" | "block.one" | "block.seq2" | "block.seq3" | "block.seq4"));
    if open(&s.name) {
        for k in s.kids {
            flatten_stmts(k, out);
        }
    } else if s.name != "fallback" {
        out.push(s);
    }
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
fn axis_terms(t: &Tree, axis: usize, workgroup: u32, tiles: Tiles, bounds: &Bounds) -> Option<(u32, Off)> {
    let mut terms = Vec::new();
    flatten_add(t, &mut terms);
    if terms.len() < 2 {
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
    // Any other offset: a sum of terms the counters' bounds prove below the
    // block factor (`2 * q + s` with q < 4 and s < 2 against a block of 8).
    let sum_of = |terms: &[&Tree]| -> Option<Tree> {
        let mut sum: Option<Tree> = None;
        for x in terms {
            sum = Some(match sum {
                None => (*x).clone(),
                Some(acc) => Tree::new("index.add", acc.ty, vec![acc, (*x).clone()]),
            });
        }
        sum
    };
    if kw == workgroup * kl {
        // A contiguous block of B = kl cells: offsets below B.
        let b = kl;
        return match rest.as_slice() {
            [] => Some((b, Off::Lit(0))),
            [x] if literal_value(x).is_some() => literal_value(x).and_then(|v| ((v as u32) < b).then_some((b, Off::Lit(v as u32)))),
            [x] if x.kids.is_empty() => local_name(&x.name).and_then(|l| tile_of(&l, tiles)).map(|n| (b, Off::Tile(n))),
            [x] if tile_min(x).is_some() => tile_min(x).map(|n| (b, Off::Tile(n))),
            _ => {
                let sum = sum_of(&rest)?;
                (offset_max(&sum, tiles, bounds)? < f64::from(b)).then_some((b, Off::Expr(unclamped(&sum, b).clone())))
            }
        };
    }
    // A grouped block: lid scaled by s = B / G with G > 1 groups (the kernel
    // writes col = wid.x * 64u + hf * 32u + lid.x * 4u + e: W 8, B 8, s 4, G 2).
    if kl > 1 && kw.is_multiple_of(workgroup) && (kw / workgroup).is_multiple_of(kl) && kw / workgroup / kl > 1 {
        let (b, s) = (kw / workgroup, kl);
        let g_count = b / s;
        let stride = workgroup * s;
        let (mut groups, mut lanes) = (Vec::new(), Vec::new());
        for x in &rest {
            match strided_unit(x, stride) {
                Some(u) => groups.push(u),
                None => lanes.push((*x).clone()),
            }
        }
        let index_ty = t.ty;
        let zero = || Tree::literal("literal.index", index_ty, 0.0);
        let g = if groups.is_empty() { zero() } else { sum_of(&groups.iter().collect::<Vec<_>>())? };
        let e = if lanes.is_empty() { zero() } else { sum_of(&lanes.iter().collect::<Vec<_>>())? };
        if offset_max(&g, tiles, bounds)? < f64::from(g_count) && offset_max(&e, tiles, bounds)? < f64::from(s) {
            return Some((b, Off::Grouped(g_count, unclamped(&g, g_count).clone(), unclamped(&e, s).clone())));
        }
        return None;
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
                let unit = strided_unit(x, workgroup)?;
                (offset_max(&unit, tiles, bounds)? < f64::from(b)).then_some((b, Off::StridedExpr(unclamped(&unit, b).clone())))
            }
            _ => {
                let units = rest.iter().map(|x| strided_unit(x, workgroup)).collect::<Option<Vec<_>>>()?;
                let sum = sum_of(&units.iter().collect::<Vec<_>>())?;
                (offset_max(&sum, tiles, bounds)? < f64::from(b)).then_some((b, Off::StridedExpr(sum)))
            }
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
            "i32.int" => ('n', 16),
            "vec4<u32>.bits" | "vec4<u32>.index" | "vec4<u32>.count" => ('u', 8),
            "mat2x2<f32>.real" => ('x', 2),
            "mat3x3<f32>.real" => ('y', 2),
            "mat4x4<f32>.real" => ('z', 2),
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

    /// The scalar value of a module `const.NAME` or `override.NAME` leaf
    /// (an override at its default); None for any other leaf.
    fn module_constant(&self, leaf: &str) -> Result<Option<f64>, String> {
        let init = if let Some(name) = leaf.strip_prefix("const.") {
            self.module.constants.iter().find(|(_, c)| c.name.as_deref() == Some(name)).map(|(_, c)| c.init)
        } else if let Some(name) = leaf.strip_prefix("override.") {
            let o = self.module.overrides.iter().find(|(_, o)| o.name.as_deref() == Some(name)).map(|(_, o)| o);
            match o {
                Some(o) => Some(o.init.ok_or_else(|| format!("the override {name} has no default (its value is the pipeline's)"))?),
                None => None,
            }
        } else {
            return Ok(None);
        };
        let Some(init) = init else { return Err(format!("{leaf}: no such module constant")) };
        let v = match self.module.global_expressions[init] {
            Expression::Literal(l) => match l {
                naga::Literal::F32(x) => f64::from(x),
                naga::Literal::U32(x) => f64::from(x),
                naga::Literal::I32(x) => f64::from(x),
                naga::Literal::Bool(b) => f64::from(u8::from(b)),
                naga::Literal::AbstractInt(x) => x as f64,
                naga::Literal::AbstractFloat(x) => x,
                other => return Err(format!("{leaf} = {other:?}: no literal of that kind")),
            },
            ref other => return Err(format!("{leaf} is not a scalar literal ({other:?})")),
        };
        Ok(Some(v))
    }

    /// The dual of a workgroup variable's element (the variable itself for
    /// a scalar or a vector): f32, vec2 and vec4 of f32, u32.
    fn workgroup_elem_dual(&self, var: &str) -> Result<String, String> {
        let (_, g) = self.module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(var) && g.space == naga::AddressSpace::WorkGroup).ok_or_else(|| format!("no workgroup variable {var}"))?;
        let inner = match &self.module.types[g.ty].inner {
            naga::TypeInner::Array { base, .. } => &self.module.types[*base].inner,
            other => other,
        };
        Ok(match inner {
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, .. }) => "f32.real".into(),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, .. }) => "u32.index".into(),
            naga::TypeInner::Vector { size, scalar: naga::Scalar { kind: naga::ScalarKind::Float, .. } } => format!("vec{}<f32>.real", *size as u32),
            other => return Err(format!("the workgroup variable {var} of {other:?}: no dual reads it")),
        })
    }

    /// A matrix local's columns and rows (`local.<name>@<h>`), from naga.
    fn local_mat(&self, local: &str) -> Option<(u32, u32)> {
        let h: usize = local.rsplit_once('@')?.1.parse().ok()?;
        let (_, lv) = self.naga_fn.local_variables.iter().nth(h)?;
        match self.module.types[lv.ty].inner {
            naga::TypeInner::Matrix { columns, rows, .. } => Some((columns as u32, rows as u32)),
            _ => None,
        }
    }

    /// The register kind of a square matrix (`x` 2x2, `y` 3x3, `z` 4x4).
    fn mat_kind(cols: u32, rows: u32) -> Result<char, String> {
        match (cols, rows) {
            (2, 2) => Ok('x'),
            (3, 3) => Ok('y'),
            (4, 4) => Ok('z'),
            _ => Err(format!("a mat{cols}x{rows} local (the kingdom has square matrix registers)")),
        }
    }

    /// A matrix local's value: its register read.
    fn mat_register_value(&mut self, base: &str, cols: u32, rows: u32) -> Result<Tree, String> {
        let kind = Self::mat_kind(cols, rows)?;
        let dual = format!("mat{cols}x{rows}<f32>.real");
        let row = self.register(base, &dual)?;
        Ok(Tree::new(&format!("reg.get_{kind}"), ty(self, &dual)?, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![])]))
    }

    /// Whether a local (`local.<name>@<h>`) is a vector of u32.
    fn local_vec_uint(&self, local: &str) -> bool {
        let Some(h) = local.rsplit_once('@').and_then(|(_, h)| h.parse::<usize>().ok()) else { return false };
        self.naga_fn.local_variables.iter().nth(h).is_some_and(|(_, lv)| matches!(self.module.types[lv.ty].inner, naga::TypeInner::Vector { scalar: naga::Scalar { kind: naga::ScalarKind::Uint, .. }, .. }))
    }

    /// The lanes of a local array's element vector (`var a: array<vec2<f32>, 4>`
    /// is 2), from naga's declaration of the local named `local.<name>@<h>`.
    fn local_elem_lanes(&self, local: &str) -> Option<u32> {
        let h: usize = local.rsplit_once('@')?.1.parse().ok()?;
        let (_, lv) = self.naga_fn.local_variables.iter().nth(h)?;
        match &self.module.types[lv.ty].inner {
            naga::TypeInner::Array { base, .. } => match &self.module.types[*base].inner {
                naga::TypeInner::Vector { size, .. } => Some(*size as u32),
                _ => None,
            },
            _ => None,
        }
    }

    /// A tile counter's read: its index, plus the loop's entry value when
    /// the loop did not start at zero.
    fn offset_counter(&self, local: &str, k: Tree) -> Result<Tree, String> {
        match self.counter_offsets.get(local) {
            Some(&a) if a != 0.0 => {
                let index = ty(self, "u32.index")?;
                Ok(Tree::new("index.add", index, vec![Tree::literal("literal.index", index, a), k]))
            }
            _ => Ok(k),
        }
    }

    fn reg_dual(kind: char) -> &'static str {
        match kind {
            'f' => "reg<f32>.real",
            'v' => "reg<vec4<f32>>.real",
            'i' => "reg<u32>.index",
            'n' => "reg<i32>.int",
            'u' => "reg<vec4<u32>>.bits",
            'x' => "reg<mat2x2<f32>>.real",
            'y' => "reg<mat3x3<f32>>.real",
            'z' => "reg<mat4x4<f32>>.real",
            _ => "reg<bool>.flag",
        }
    }

    /// A value tree in the safe grammar.
    fn value(&mut self, t: &Tree) -> Result<Tree, String> {
        if let Some(local) = private_as_local(t) {
            return self.value(&local);
        }
        // `*p` for a pointer parameter: the caller's register.
        if let Some((row, kind)) = t.name.strip_prefix("load.").and_then(|r| r.split('@').next()).and_then(|a| self.pointer_args.get(a)).cloned() {
            return Ok(Tree::new(&format!("reg.get_{kind}"), t.ty, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![])]));
        }
        // An expression already in the safe grammar (a subgroup or atomic
        // result inlined into a loop's bound): as it is.
        if t.name == INLINED && t.kids.len() == 1 {
            return Ok(t.kids[0].clone());
        }
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
                    let k = Tree::new("safe.tile_index", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter", ty(self, "u32.tcounter")?, vec![])]);
                    return self.offset_counter(&local, k);
                }
                if self.tile_counter2.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.tile_index2", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter2", ty(self, "u32.tcounter2")?, vec![])]);
                    return self.offset_counter(&local, k);
                }
                if self.tile_counter3.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.tile_index3", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter3", ty(self, "u32.tcounter3")?, vec![])]);
                    return self.offset_counter(&local, k);
                }
                if self.tile_counter4.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.tile_index4", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter4", ty(self, "u32.tcounter4")?, vec![])]);
                    return self.offset_counter(&local, k);
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
            if t.name == "builtin.num_work_groups" {
                // naga's name; the table's row is `builtin.num_workgroups`.
                return Ok(Tree::new("builtin.num_workgroups", t.ty, vec![]));
            }
            if t.name.starts_with("builtin.") || t.name.starts_with("literal.") || t.name == "load.phi_len" || t.name == "load.h_len" {
                return Ok(t.clone());
            }
            // A workgroup scalar: element 0 of a workgroup array of one.
            if let Some(rest) = t.name.strip_prefix("load.workgroup.") {
                let var = rest.split('@').next().unwrap_or(rest);
                let zero = Tree::literal("literal.index", ty(self, "u32.index")?, 0.0);
                return self.value(&Tree::new(&format!("load.workgroup.{var}.#"), t.ty, vec![zero]));
            }
            // A module `const`, and an `override` at its default (no pipeline
            // of the kingdom sets one): its literal value.
            if let Some(v) = self.module_constant(&t.name)? {
                let form = self.kingdom.dual(t.ty).map(|d| d.form.clone()).unwrap_or_default();
                let row = format!("literal.{form}");
                if !self.kingdom.rows.iter().any(|r| r.name == row) {
                    return Err(format!("{} = {v}: a literal of form {form} has no row", t.name));
                }
                return Ok(Tree::literal(&row, t.ty, v as f32));
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
                let i = self.value(self.wg_unmasked(i, which))?;
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
            let idx = self.value(self.wg_unmasked(&t.kids[0], which))?;
            let wg = Tree::new(of, ty(self, idx_dual)?, vec![idx]);
            return Ok(Tree::new(get, t.ty, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), wg]));
        }
        if let Some(rest) = t.name.strip_prefix("load.local.") {
            if rest.contains(".#") {
                let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
                // A lane of a vector register: `load.local.bv@8.#@v2 (idx)`; the
                // register is the local's own (named without the `.#`).
                let base = local.trim_end_matches(".#").to_string();
                if let Some((cols, rows)) = self.local_mat(&base) {
                    // A column of a matrix local (`m[1]`), or a lane of one (`m[1][0]`).
                    let (col, lane) = match nested_index(rest, &t.kids) {
                        Some((i, j)) => (i, Some(j)),
                        None => (&t.kids[0], None),
                    };
                    let whole = self.mat_register_value(&base, cols, rows)?;
                    let vty = ty(self, &format!("vec{rows}<f32>.real"))?;
                    let column = match literal_value(col).filter(|c| *c >= 0.0 && (*c as u32) < cols) {
                        Some(c) => Tree::new(&format!("shape.col_{}", c as u32), vty, vec![whole]),
                        None => Tree::new("shape.col_at", vty, vec![whole, self.value(col)?]),
                    };
                    return Ok(match lane {
                        None => column,
                        Some(j) => match literal_value(j).filter(|l| *l >= 0.0 && (*l as u32) < rows) {
                            Some(l) => Tree::new(&format!("shape.access_{}", l as u32), t.ty, vec![column]),
                            None => Tree::new("shape.at", t.ty, vec![column, self.value(j)?]),
                        },
                    });
                }
                if let Some((i, j)) = nested_index(rest, &t.kids) {
                    // A lane of a vec4 (or vec2) element of a private array: `accs[i][j]`.
                    let elem = if self.local_elem_lanes(&base) == Some(2) { "vec2<f32>.real" } else { "vec4<f32>.real" };
                    let (arr, kind, mask) = self.private_array(&format!("{base}.#"), elem)?;
                    if kind == 't' {
                        let i = self.value(unmasked(i, 15.0))?;
                        let pidx = Tree::new(mask, ty(self, "u32.pindex")?, vec![i]);
                        let whole = Tree::new("parr.get_t", ty(self, "vec2<f32>.real")?, vec![Tree::new(arr, ty(self, parr_dual(kind))?, vec![]), pidx]);
                        if let Some(lane) = literal_value(unmasked(j, 1.0)).filter(|l| (0.0..=1.0).contains(l)) {
                            return Ok(Tree::new(&format!("shape.access_{}", lane as u32), t.ty, vec![whole]));
                        }
                        let j = self.value(unmasked(j, 1.0))?;
                        return Ok(Tree::new("shape.at", t.ty, vec![whole, j]));
                    }
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
                if self.local_lanes.get(&base) == Some(&4) && self.local_vec_uint(&base) {
                    // A lane of a vec4<u32> register (subgroupBallot's): the
                    // register's value, then its lane.
                    let row = self.register(&base, "vec4<u32>.bits")?;
                    let whole = Tree::new("reg.get_u", ty(self, "vec4<u32>.bits")?, vec![Tree::new(&row, ty(self, Self::reg_dual('u'))?, vec![])]);
                    if let Some(lane) = literal_value(unmasked(&t.kids[0], 3.0)).filter(|l| (0.0..=3.0).contains(l)) {
                        return Ok(Tree::new(&format!("shape.access_{}", lane as u32), t.ty, vec![whole]));
                    }
                    let idx = self.value(unmasked(&t.kids[0], 3.0))?;
                    return Ok(Tree::new("shape.at", t.ty, vec![whole, idx]));
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
        // A swizzle (`v.zyx`, `v.xy`): its lanes composed, `vec3(v.z, v.y,
        // v.x)` (the kingdom has no row per pattern; the value is the same).
        if let Some(pat) = t.name.strip_prefix("shape.swizzle.") {
            if t.kids.len() == 1 && (1..=4).contains(&pat.len()) {
                let v = self.value(&t.kids[0])?;
                let d = self.kingdom.dual(v.ty).ok_or("a swizzle of an untyped vector")?;
                let lane = ty(self, &format!("{}.{}", d.scalar().ok_or("a swizzle of a non-vector")?, d.form))?;
                let lanes = pat
                    .chars()
                    .map(|c| {
                        let i = "xyzw".find(c).ok_or_else(|| format!("swizzle lane {c}"))?;
                        Ok(Tree::new(&format!("shape.access_{i}"), lane, vec![v.clone()]))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                if lanes.len() == 1 {
                    return Ok(lanes.into_iter().next().expect("one lane"));
                }
                // Typed by the pattern's width over the source's lanes (the
                // inference can give the swizzle another width).
                let out = ty(self, &format!("vec{}<{}>.{}", lanes.len(), d.scalar().unwrap_or("f32"), d.form))?;
                return Ok(Tree::new(&format!("shape.compose{}", lanes.len()), out, lanes));
            }
        }
        // Matrix arithmetic: naga's Multiply / Add / Subtract / Negate resolve
        // by kind alone; the operands' shapes choose the row.
        let is_mat = |ctx: &Self, k: &Tree| ctx.kingdom.dual(k.ty).is_some_and(|d| d.slot.starts_with("mat"));
        let is_vec = |ctx: &Self, k: &Tree| ctx.kingdom.dual(k.ty).is_some_and(|d| d.slot.starts_with("vec"));
        if matches!(t.name.as_str(), "arith.mul" | "arith.add" | "arith.sub" | "arith.neg") && t.kids.iter().any(|k| is_mat(self, k)) {
            let row = match (t.name.as_str(), t.kids.as_slice()) {
                ("arith.mul", [a, b]) if is_mat(self, a) && is_mat(self, b) => "arith.mul_mat_mat",
                ("arith.mul", [a, b]) if is_mat(self, a) && is_vec(self, b) => "arith.mul_mat_vec",
                ("arith.mul", [a, b]) if is_vec(self, a) && is_mat(self, b) => "arith.mul_vec_mat",
                ("arith.mul", [a, _]) if is_mat(self, a) => "arith.mul_mat_scalar",
                ("arith.mul", [_, _]) => "arith.mul_scalar_mat",
                ("arith.add", _) => "arith.add_mat",
                ("arith.sub", _) => "arith.sub_mat",
                _ => "arith.neg_mat",
            };
            let kids = t.kids.iter().map(|k| self.value(k)).collect::<Result<Vec<_>, _>>()?;
            return Ok(Tree { name: row.into(), ty: t.ty, konst: t.konst, kids });
        }
        if t.name == "shape.access_dyn" && t.kids.len() == 2 && is_mat(self, &t.kids[0]) {
            let kids = t.kids.iter().map(|k| self.value(k)).collect::<Result<Vec<_>, _>>()?;
            return Ok(Tree { name: "shape.col_at".into(), ty: t.ty, konst: t.konst, kids });
        }
        // `select(a, b, c)` with a vector of bools for c chooses per lane: the
        // kingdom's select.lane_cond (naga's Select resolves by kind alone).
        if t.name == "select.scalar_cond" && t.kids.len() == 3 && self.kingdom.dual(t.kids[2].ty).is_some_and(|d| d.lanes > 1 && d.slot.starts_with("vec")) {
            let kids = t.kids.iter().map(|k| self.value(k)).collect::<Result<Vec<_>, _>>()?;
            return Ok(Tree { name: "select.lane_cond".into(), ty: t.ty, konst: t.konst, kids });
        }
        // A vector component chosen at run time (`v[j]`): the kingdom's `shape.at`.
        if t.name == "shape.access_dyn" && t.kids.len() == 2 && self.kingdom.dual(t.kids[0].ty).is_some_and(|d| d.lanes > 1 && d.slot.starts_with("vec")) {
            let v = self.value(&t.kids[0])?;
            let i = self.value(&t.kids[1])?;
            return Ok(Tree::new("shape.at", t.ty, vec![v, i]));
        }
        // `ldexp(e, i32(x))`: WGSL's exponent is an integer per lane, and the
        // kingdom's row takes the real it is converted from (the renderer
        // writes the conversion; read back, it is the row's own argument).
        if t.name == "arith.ldexp" && t.kids.len() == 2 && t.kids[1].name == "convert.f32_to_int" && t.kids[1].kids.len() == 1 {
            let (e, x) = (self.value(&t.kids[0])?, self.value(&t.kids[1].kids[0])?);
            return Ok(Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: vec![e, x] });
        }
        let kids = t.kids.iter().map(|k| self.value(k)).collect::<Result<Vec<_>, _>>()?;
        // The reader suffixes a compose with its slot (`shape.compose4.vec4<f32>`);
        // the table's row is the bare name, the tree's dual carries the slot.
        let name = if t.name.starts_with("shape.compose") { t.name.split('.').take(2).collect::<Vec<_>>().join(".") } else { t.name.clone() };
        Ok(Tree { name, ty: t.ty, konst: t.konst, kids })
    }

    /// A local array as a private array row: (`parr.a|b|v|i`, kind, mask row).
    /// One array is one private array: its indexed accesses name it
    /// `local.s0@7.#`, a whole-array constructor `local.s0@7` (fit_A2's
    /// r2x_union, whose two constructed vec4 arrays took four private arrays).
    fn private_array(&mut self, local: &str, elem_dual: &str) -> Result<(&'static str, char, &'static str), String> {
        let local = local.trim_end_matches(".#");
        if let Some(a) = self.parrs.get(local) {
            return Ok(*a);
        }
        let (kind, mask) = match elem_dual {
            "f32.real" => ('f', "safe.pidx64"),
            "vec4<f32>.real" => ('v', "safe.pidx16"),
            "u32.index" => ('i', "safe.pidx16"),
            "bool.flag" => ('b', "safe.pidx16"),
            "vec2<f32>.real" => ('t', "safe.pidx16"),
            other => return Err(format!("a local array {local} of {other} has no private array")),
        };
        let used: Vec<&'static str> = self.parrs.values().map(|a| a.0).collect();
        let arr = match kind {
            'f' => ["parr.a", "parr.b", "parr.c", "parr.d", "parr.e", "parr.f"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a seventh f32 local array {local}"))?,
            'v' => ["parr.v", "parr.w", "parr.x", "parr.y"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a fifth vec4 local array {local}"))?,
            'b' => ["parr.m", "parr.n"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a third bool local array {local}"))?,
            't' => ["parr.q", "parr.r"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a third vec2 local array {local}"))?,
            _ => ["parr.i", "parr.j", "parr.k"].iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("a fourth u32 local array {local}"))?,
        };
        self.parrs.insert(local.to_string(), (arr, kind, mask));
        Ok((arr, kind, mask))
    }

    /// A workgroup array by name and element dual: four arrays of f32
    /// (`wg.a`..`wg.d`), of vec2<f32> (`wg.v2a`..`wg.v2d`) and of vec4<f32>
    /// (`wg.va`..`wg.vd`); its declared length is recorded (the kernel's
    /// declaration on the grid, 2026-10-12).
    fn array(&mut self, name: &str, elem_dual: &str) -> Result<&'static str, String> {
        if let Some(w) = self.arrays.get(name) {
            return Ok(w);
        }
        // An atomic array is one by its declared type, whatever a use of it reads.
        let atomic = self.module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(name) && g.space == naga::AddressSpace::WorkGroup).and_then(|(_, g)| match &self.module.types[g.ty].inner {
            naga::TypeInner::Array { base, .. } => match self.module.types[*base].inner {
                naga::TypeInner::Atomic(sc) => Some(sc.kind),
                _ => None,
            },
            _ => None,
        });
        let elem_dual = match atomic {
            Some(naga::ScalarKind::Uint) => "atomic<u32>",
            Some(naga::ScalarKind::Sint) => "atomic<i32>",
            Some(other) => return Err(format!("a workgroup array {name} of atomic {other:?} has no safe spelling")),
            None => elem_dual,
        };
        let candidates: &[&'static str] = match elem_dual {
            "atomic<u32>" => &["wg.au_a", "wg.au_b"],
            "atomic<i32>" => &["wg.ai_a", "wg.ai_b"],
            "f32.real" => &["wg.a", "wg.b", "wg.c", "wg.d"],
            "vec2<f32>.real" => &["wg.v2a", "wg.v2b", "wg.v2c", "wg.v2d"],
            "vec4<f32>.real" => &["wg.va", "wg.vb", "wg.vc", "wg.vd"],
            other => return Err(format!("a workgroup array {name} of {other} has no safe spelling")),
        };
        let used: Vec<&'static str> = self.arrays.values().copied().collect();
        let which = candidates.iter().copied().find(|c| !used.contains(c)).ok_or_else(|| format!("one more workgroup array {name} of {elem_dual} than the grammar's {}", candidates.len()))?;
        let len = self
            .module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(name) && g.space == naga::AddressSpace::WorkGroup)
            .and_then(|(_, g)| match self.module.types[g.ty].inner {
                naga::TypeInner::Array { size: naga::ArraySize::Constant(n), .. } => Some(n.get()),
                // A workgroup scalar or vector: an array of one.
                naga::TypeInner::Scalar(_) | naga::TypeInner::Vector { .. } => Some(1),
                _ => None,
            })
            .ok_or_else(|| format!("the workgroup array {name} has no constant length"))?;
        if len > 8192 {
            return Err(format!("the workgroup array {name} of {len} elements is past the kingdom's 8,192"));
        }
        self.arrays.insert(name.to_string(), which);
        self.array_lens.insert(which, len);
        Ok(which)
    }

    /// A subgroup operation: `subgroup.<fun>(x[, lane])` into a register of
    /// its result's dual, at the statement's own point (a collective runs
    /// where the kernel runs it). A ballot (vec4<u32>, no register), a
    /// broadcast whose lane is not a literal, and a result no register holds
    /// are refused by name.
    fn subgroup_statement(&mut self, root: &Root, result: u32, rung: &str) -> Result<Tree, String> {
        let (expr, d) = self.subgroup_expr(root)?;
        let kind = match d.as_str() {
            "f32.real" => 'f',
            "vec4<f32>.real" => 'v',
            "u32.index" | "u32.bits" | "u32.count" => 'i',
            "bool.flag" => 'b',
            "i32.int" => 'n',
            "vec4<u32>.bits" => 'u',
            other => return Err(format!("a subgroup {} of {other}: no register holds its result", expr.name)),
        };
        let row = self.register(&format!("local.subgroup@{result}"), &d)?;
        let st = self.stmt_ty(rung)?;
        Ok(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), expr]))
    }

    /// A subgroup operation's expression `subgroup.<fun>(x[, lane])` and its
    /// argument's dual.
    fn subgroup_expr(&mut self, root: &Root) -> Result<(Tree, String), String> {
        let RootKind::Subgroup { fun } = &root.kind else { return Err("not a subgroup root".into()) };
        if fun == "ballot" {
            // subgroupBallot(b) (no argument: every active lane): a vec4<u32> of lane bits.
            let Node::App { kids, .. } = &root.tree else { return Err("a subgroup root without its argument".into()) };
            let flag = ty(self, "bool.flag")?;
            let b = match kids.first() {
                Some(k) => {
                    let t = tree_of(self, k)?;
                    self.value(&t)?
                }
                None => Tree::literal("literal.flag", flag, 1.0),
            };
            return Ok((Tree::new("subgroup.ballot", ty(self, "vec4<u32>.bits")?, vec![b]), "vec4<u32>.bits".to_string()));
        }
        let Node::App { kids, .. } = &root.tree else { return Err("a subgroup root without its argument".into()) };
        let trees = kids.iter().map(|k| tree_of(self, k)).collect::<Result<Vec<_>, _>>()?;
        let x = self.value(&trees[0])?;
        let mut args = vec![x.clone()];
        if let Some(l) = trees.get(1) {
            if fun == "broadcast" {
                let lane = literal_value(l).ok_or("subgroupBroadcast's lane is not a literal (WGSL demands a constant)")?;
                args.push(Tree::literal("literal.lit64", ty(self, "u32.lit64")?, lane));
            } else {
                args.push(self.value(l)?);
            }
        }
        let d = self.kingdom.dual(x.ty).map(|d| d.dual.clone()).unwrap_or_default();
        Ok((Tree::new(&format!("subgroup.{fun}"), x.ty, args), d))
    }

    /// An atomic read-modify-write statement: `atomic.<fun>_<u|i>(array,
    /// index, value)` (a compare-exchange `atomic.cas_<u|i>(array, index,
    /// compare, value)`); its old value to a register when a later expression
    /// reads it, else kept for its effect only (`_ = ...`).
    fn atomic_statement(&mut self, root: &Root, result: u32, rung: &str) -> Result<Tree, String> {
        let (expr, unsigned) = self.atomic_expr(root)?;
        let suffix = if unsigned { "u" } else { "i" };
        let st = self.stmt_ty(rung)?;
        let leaf = format!("atomic@{result}");
        fn reads(n: &Node, leaf: &str) -> bool {
            match n {
                Node::Leaf { name, .. } => name == leaf,
                Node::App { kids, .. } => kids.iter().any(|k| reads(k, leaf)),
                _ => false,
            }
        }
        let used = self.roots.values().flatten().any(|r| reads(&r.tree, &leaf)) || self.lets.values().any(|t| reads(t, &leaf));
        if !used {
            return Ok(Tree::new(&format!("stmt.discard_{suffix}_{rung}"), st, vec![expr]));
        }
        // The old value read later: to a u32 register, or for an atomic<i32> an i32 one.
        let (dual, kind) = if unsigned { ("u32.index", 'i') } else { ("i32.int", 'n') };
        let row = self.register(&format!("local.atomic@{result}"), dual)?;
        Ok(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), expr]))
    }

    /// An atomic's expression `atomic.<fun>_<u|i>(array, index, value)` and
    /// whether its array is atomic<u32>.
    fn atomic_expr(&mut self, root: &Root) -> Result<(Tree, bool), String> {
        let RootKind::Atomic { target, fun } = &root.kind else { return Err("not an atomic root".into()) };
        let Node::App { kids, .. } = &root.tree else { return Err("an atomic root without operands".into()) };
        let arr = target.strip_prefix("workgroup.").and_then(|r| r.strip_suffix(".#")).ok_or_else(|| format!("an atomic on {target}: the safe kingdom's atomics are workgroup arrays (a storage atomic needs a binding the Gram role does not have)"))?;
        let trees = kids.iter().map(|k| tree_of(self, k)).collect::<Result<Vec<_>, _>>()?;
        let which = self.array(arr, "atomic")?;
        let (of, idx_dual, arr_dual, _, _) = Self::wg_rows(which);
        let unsigned = which.starts_with("wg.au");
        let suffix = if unsigned { "u" } else { "i" };
        let idx = self.value(self.wg_unmasked(&trees[0], which))?;
        let at = Tree::new(of, ty(self, idx_dual)?, vec![idx]);
        let array = Tree::new(which, ty(self, arr_dual)?, vec![]);
        let out_ty = ty(self, if unsigned { "u32.index" } else { "i32.int" })?;
        if fun == "cas" {
            return Err("atomicCompareExchangeWeak has no safe spelling: naga 0.20's Metal backend does not implement it (MSL: FeatureNotImplemented)".into());
        }
        let value = self.value(&trees[1])?;
        Ok((Tree::new(&format!("atomic.{fun}_{suffix}"), out_ty, vec![array, at, value]), unsigned))
    }

    /// A workgroup array's index without its mask into the array: `i & (n - 1)`
    /// for a length n that is a power of two, `min(i, n - 1)` otherwise.
    fn wg_unmasked<'t>(&self, idx: &'t Tree, which: &str) -> &'t Tree {
        let n = self.array_lens.get(which).copied().unwrap_or(2048);
        let last = n.saturating_sub(1) as f32;
        if n.is_power_of_two() {
            return unmasked(idx, last);
        }
        if idx.name == "index.min" && idx.kids.len() == 2 {
            for (x, m) in [(&idx.kids[0], &idx.kids[1]), (&idx.kids[1], &idx.kids[0])] {
                if literal_value(m) == Some(last) {
                    return x;
                }
            }
        }
        idx
    }

    /// The index and array duals, the get row and the set stem of a workgroup array row.
    fn wg_rows(which: &str) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
        if which.starts_with("wg.au") {
            return ("safe.wga_of", "u32.wgaindex", "wgmem<atomic<u32>>.index", "atomic.load_u", "atomic.store_u");
        }
        if which.starts_with("wg.ai") {
            return ("safe.wga_of", "u32.wgaindex", "wgmem<atomic<i32>>.int", "atomic.load_i", "atomic.store_i");
        }
        if which.starts_with("wg.v2") {
            return ("safe.wgv2_of", "u32.wgv2index", "wgmem<vec2<f32>>.real", "wg.get_v2_safe", "wg.set_v2");
        }
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
        // The index as a sum of terms, however it is nested: the one term that
        // is `row * cfg.d`, and the column as the sum of every other term
        // (`row * d + c0 + e`, fit_A2's r6x_reinvent; `(row * d + col) + 1u`).
        let mut terms = Vec::new();
        flatten_add(idx, &mut terms);
        let is_prod = |t: &Tree| t.name == "index.mul" && t.kids.len() == 2 && t.kids.iter().any(|k| is_leaf(k, "load.uniform.cfg.d"));
        let idx = match terms.iter().position(|t| is_prod(t)) {
            Some(p) if terms.len() >= 2 => {
                let rest: Vec<Tree> = terms.iter().enumerate().filter(|(i, _)| *i != p).map(|(_, t)| (*t).clone()).collect();
                let col = rest.into_iter().reduce(|acc, t| Tree::new("index.add", idx.ty, vec![acc, t])).expect("a column term");
                Tree::new("index.add", idx.ty, vec![terms[p].clone(), col])
            }
            _ => idx.clone(),
        };
        let idx = &idx;
        for (prod, col) in [(&idx.kids[0], &idx.kids[1]), (&idx.kids[1], &idx.kids[0])] {
            if prod.name == "index.mul" && prod.kids.len() == 2 {
                for (r, dd) in [(&prod.kids[0], &prod.kids[1]), (&prod.kids[1], &prod.kids[0])] {
                    if !is_leaf(dd, "load.uniform.cfg.d") {
                        continue;
                    }
                    let tiles = (self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone(), self.tile_counter4.clone());
                    let tiles: Tiles = [&tiles.0, &tiles.1, &tiles.2, &tiles.3];
                    // The row and column with literal factors distributed, so
                    // `(wid * W + lid) * B + k` is proved as `wid * (W * B) + lid * B + k`.
                    let (r, col) = (&distribute_literals(r), &distribute_literals(col));
                    // The rows on thread y (grid.of) or on x (grid.of_rows_x):
                    // the layout the header or an earlier store fixed, else
                    // the one this store proves, y first.
                    let layouts: &[bool] = match self.rows_x {
                        Some(true) => &[true],
                        Some(false) => &[false],
                        None => &[false, true],
                    };
                    let proved = layouts.iter().find_map(|&rows_x| {
                        let ((ra, rw), (ca, cw)) = if rows_x { ((0, self.grid.0), (1, self.grid.1)) } else { ((1, self.grid.1), (0, self.grid.0)) };
                        Some((rows_x, thread_axis_off_in(r, ra, rw, tiles, &self.row_arrays, &self.tile_bounds)?, thread_axis_off_in(col, ca, cw, tiles, &self.row_arrays, &self.tile_bounds)?))
                    });
                    let Some((rows_x, (by, i), (bx, j))) = proved else { continue };
                    self.rows_x = Some(rows_x);
                    if by != self.grid.3 || bx != self.grid.2 {
                        return fail(&format!("block {bx} x {by} in the index, the grid says {} x {}", self.grid.2, self.grid.3));
                    }
                    let gid = Tree::new("builtin.global_invocation_id", ty(self, "vec3<u32>.gid")?, vec![]);
                    let blk = |v: u32| Tree::literal("literal.blk", ty(self, "u32.blk").expect("dual"), v as f32);
                    let wid = Tree::new("builtin.workgroup_id", ty(self, "vec3<u32>.wid")?, vec![]);
                    let lid = Tree::new("builtin.local_invocation_id", ty(self, "vec3<u32>.lid")?, vec![]);
                    let strided = |o: &Off| matches!(o, Off::Strided(_) | Off::StridedTile(_) | Off::StridedExpr(_));
                    let lit64 = |ctx: &Self, v: u32| Tree::literal("literal.lit64", ty(ctx, "u32.lit64").expect("dual"), v as f32);
                    let base_row = match &i {
                        Off::Grouped(g, _, _) => Tree::new("safe.thread_row_g", ty(self, "u32.threadrow")?, vec![wid.clone(), lid.clone(), blk(by), lit64(self, *g)]),
                        o if strided(o) => Tree::new("safe.thread_row_s", ty(self, "u32.threadrow")?, vec![wid.clone(), lid.clone(), blk(by)]),
                        _ => Tree::new("safe.thread_row", ty(self, "u32.threadrow")?, vec![gid.clone(), blk(by)]),
                    };
                    let base_col = match &j {
                        Off::Grouped(g, _, _) => Tree::new("safe.thread_col_g", ty(self, "u32.threadcol")?, vec![wid, lid, blk(bx), lit64(self, *g)]),
                        o if strided(o) => Tree::new("safe.thread_col_s", ty(self, "u32.threadcol")?, vec![wid, lid, blk(bx)]),
                        _ => Tree::new("safe.thread_col", ty(self, "u32.threadcol")?, vec![gid, blk(bx)]),
                    };
                    let row = self.plus(base_row, i, "row")?;
                    let c = self.plus(base_col, j, "col")?;
                    return Ok(Tree::new("safe.cell_of", ty(self, "u32.cell")?, vec![row, c]));
                }
            }
        }
        fail(&format!("row or column is not this thread's (tile counters {:?} / {:?} / {:?} / {:?})", self.tile_counter, self.tile_counter2, self.tile_counter3, self.tile_counter4))
    }

    /// A thread's row or column plus an offset within its block, as the row that proves it.
    fn plus(&mut self, base: Tree, off: Off, axis: &str) -> Result<Tree, String> {
        let dual = ty(self, if axis == "row" { "u32.threadrow" } else { "u32.threadcol" })?;
        let suffix = |n: u8| if n == 1 { String::new() } else { n.to_string() };
        let counter = |ctx: &Self, n: u8| -> Result<Tree, String> { Ok(Tree::new(&format!("safe.tile_counter{}", suffix(n)), ty(ctx, &format!("u32.tcounter{}", suffix(n)))?, vec![])) };
        let index_ty = ty(self, "u32.index")?;
        Ok(match off {
            Off::Lit(0) | Off::Strided(0) => base,
            Off::Strided(v) => Tree::new(&format!("safe.{axis}_plus_s"), dual, vec![base, Tree::literal("literal.lit64", ty(self, "u32.lit64")?, v as f32)]),
            Off::Lit(v) => Tree::new(&format!("safe.{axis}_plus"), dual, vec![base, Tree::literal("literal.lit64", ty(self, "u32.lit64")?, v as f32)]),
            Off::Tile(n) => Tree::new(&format!("safe.{axis}_plus_tile{}", suffix(n)), dual, vec![base, counter(self, n)?]),
            Off::StridedTile(n) => Tree::new(&format!("safe.{axis}_plus_tile{}_s", suffix(n)), dual, vec![base, counter(self, n)?]),
            Off::Expr(e) => {
                let e = Tree { ty: index_ty, ..self.value(&e)? };
                Tree::new(&format!("safe.{axis}_plus_x"), dual, vec![base, e])
            }
            Off::StridedExpr(e) => {
                let e = Tree { ty: index_ty, ..self.value(&e)? };
                Tree::new(&format!("safe.{axis}_plus_x_s"), dual, vec![base, e])
            }
            Off::Grouped(g_count, g, e) => {
                let g = Tree { ty: index_ty, ..self.value(&g)? };
                let e = Tree { ty: index_ty, ..self.value(&e)? };
                Tree::new(&format!("safe.{axis}_plus_g"), dual, vec![base, Tree::literal("literal.lit64", ty(self, "u32.lit64")?, g_count as f32), g, e])
            }
        })
    }

    fn stmt_ty(&self, rung: &str) -> Result<Ty, String> {
        ty(self, &format!("stmt.{rung}"))
    }

    /// The top-level statements split into the body and the epilogue: the
    /// last statement that stores h (with every guard and tile loop around
    /// its store) as an `O0` block, the statements before it as the body. A
    /// kernel that never writes h, or that has a statement after its last
    /// store, has no spelling: the kingdom's root demands that every kernel
    /// ends in the store of its thread's cell.
    fn split_epilogue(&self, stmts: Vec<Tree>) -> Result<(Vec<Tree>, Tree), String> {
        // Blocks opened in place first (a `{ ... }` around the whole kernel,
        // a helper's statements), so the split is at the store's own level.
        let mut flat = Vec::new();
        for s in stmts {
            flatten_stmts(s, &mut flat);
        }
        let mut stmts = flat;
        // A kernel that never writes h (a gene whose body returns at the
        // top level, rendered without its epilogue; the reader drops the
        // trailing return as main's implicit one): the body, an explicit
        // return, and the epilogue's fallback, which the renderer does not
        // write after a body that returns.
        if !stmts.iter().any(has_store) {
            if stmts.last().is_none_or(|s| rung_stem(&s.name) != Some("stmt.return")) {
                stmts.push(Tree::new("stmt.return_L0", self.stmt_ty("L0")?, vec![]));
            }
            return Ok((stmts, Tree::fallback(ty(self, "block.O0")?)));
        }
        let p = stmts.iter().rposition(has_store).ok_or("the kernel never writes h: the kingdom's root ends every kernel in the store of its thread's cell")?;
        let after: Vec<Tree> = stmts.split_off(p + 1);
        if !after.is_empty() {
            return Err(format!("{} top-level statement(s) after the last store of h (the first: {}); the kernel's epilogue ends in that store", after.len(), after[0].name));
        }
        let last = stmts.pop().expect("the statement at p");
        if stmts.iter().any(has_continue) {
            return Err(CONTINUE_OUTSIDE.into());
        }
        Ok((stmts, self.epilogue_block(vec![last], 0)?))
    }

    /// An epilogue block at tile depth d: the statements (flattened through
    /// nested blocks) up to the last that stores h; those before it are
    /// conditional statements of the depth (`C<d>`), the last an epilogue
    /// statement (`O<d>`).
    fn epilogue_block(&self, items: Vec<Tree>, d: usize) -> Result<Tree, String> {
        let mut list = Vec::new();
        for s in items {
            flatten_stmts(s, &mut list);
        }
        let p = list.iter().rposition(has_store).ok_or("an epilogue block without a store of h")?;
        if p + 1 < list.len() {
            return Err(format!("{} statement(s) after the last store of h inside the epilogue (the first: {})", list.len() - p - 1, list[p + 1].name));
        }
        // `if (c) { continue; }` before the store: the guard, then the rest
        // of the block, as one epilogue block (block.cont_O<d>); only inside
        // an epilogue tile loop (depth 1 and deeper).
        if let Some(q) = list[..p].iter().position(is_continue_guard) {
            if d == 0 {
                return Err("a continue in the kernel's epilogue outside its tile loops".into());
            }
            let rest: Vec<Tree> = list[q + 1..].to_vec();
            let cond = list[q].kids[0].clone();
            let o = format!("O{d}");
            let cont = Tree::new(&format!("block.cont_{o}"), ty(self, &format!("block.{o}"))?, vec![cond, self.epilogue_block(rest, d)?]);
            if q == 0 {
                return Ok(cont);
            }
            let last = Tree::new(&format!("stmt.block_{o}"), self.stmt_ty(&o)?, vec![cont]);
            return self.epilogue_assemble(&list[..q], last, d);
        }
        let last = self.epilogue_stmt(&list[p], d)?;
        self.epilogue_assemble(&list[..p], last, d)
    }

    /// An epilogue block at depth d from the statements before its last
    /// (moved to the conditional rung C<d>) and its last epilogue statement.
    fn epilogue_assemble(&self, before: &[Tree], last: Tree, d: usize) -> Result<Tree, String> {
        let mut prefix = before.iter().map(|s| self.rerung(s, d)).collect::<Result<Vec<_>, _>>()?;
        let c = format!("C{d}");
        if prefix.len() > 3 {
            // At most three conditional statements before the last: the
            // leading ones nested as one block statement.
            let keep = prefix.split_off(prefix.len() - 2);
            let lead = Tree::new(&format!("stmt.block_{c}"), self.stmt_ty(&c)?, vec![self.block(prefix, &c)?]);
            prefix = std::iter::once(lead).chain(keep).collect();
        }
        let o = format!("O{d}");
        let name = match prefix.len() {
            0 => "one",
            1 => "seq2",
            2 => "seq3",
            _ => "seq4",
        };
        prefix.push(last);
        Ok(Tree::new(&format!("block.{name}_{o}"), ty(self, &format!("block.{o}"))?, prefix))
    }

    /// The epilogue statement at depth d that holds the last store of h: the
    /// store itself, an if or an if-else (both branches storing) around an
    /// epilogue block, a block, or a tile loop whose body is the epilogue of
    /// the next depth.
    fn epilogue_stmt(&self, s: &Tree, d: usize) -> Result<Tree, String> {
        let o = format!("O{d}");
        let st = self.stmt_ty(&o)?;
        let stem = rung_stem(&s.name).unwrap_or(&s.name);
        let kids = s.kids.as_slice();
        match (stem, kids) {
            ("store.h", [cell, v]) => Ok(Tree::new(&format!("store.h_{o}"), st, vec![cell.clone(), v.clone()])),
            ("stmt.if", [c, b]) => Ok(Tree::new(&format!("stmt.if_{o}"), st, vec![c.clone(), self.epilogue_block(vec![b.clone()], d)?])),
            ("stmt.if_else", [c, a, b]) => {
                if !has_store(a) || !has_store(b) {
                    return Err("the last store of h is in one branch of an if-else only; the kernel's epilogue stores on every path".into());
                }
                Ok(Tree::new(&format!("stmt.if_else_{o}"), st, vec![c.clone(), self.epilogue_block(vec![a.clone()], d)?, self.epilogue_block(vec![b.clone()], d)?]))
            }
            ("stmt.block", [b]) => Ok(Tree::new(&format!("stmt.block_{o}"), st, vec![self.epilogue_block(vec![b.clone()], d)?])),
            ("loop.tile" | "loop.tile_strided" | "loop.tilex" | "loop.tilex_strided", [lead @ .., body]) => {
                if d + 1 > 4 {
                    return Err("the last store of h is inside five nested tile loops (the epilogue has four)".into());
                }
                let mut k: Vec<Tree> = lead.to_vec();
                k.push(self.epilogue_block(vec![body.clone()], d + 1)?);
                Ok(Tree::new(&format!("{stem}_{o}"), st, k))
            }
            _ => Err(format!("the last store of h is inside {}, which has no epilogue spelling", s.name)),
        }
    }

    /// A statement before the epilogue's last, moved to the conditional rung
    /// of its tile depth (`C<d>`; a tile loop's body one deeper). A barrier,
    /// a return or an item loop has no conditional spelling.
    fn rerung(&self, t: &Tree, d: usize) -> Result<Tree, String> {
        if t.name == "stmt.continue" {
            return Err("a continue that is not a guard `if (c) { continue; }` where only conditional statements are spelled (the rest of a loop body after a continue, or the epilogue before its store)".into());
        }
        let Some(stem) = rung_stem(&t.name) else { return Ok(t.clone()) };
        if stem == "stmt.barrier" || stem == "stmt.storage_barrier" || stem == "stmt.subgroup_barrier" || stem == "stmt.return" || stem.starts_with("loop.items") {
            return Err(format!("{} where only conditional statements are spelled (after a loop's continue guard, or before the epilogue's last store of h)", t.name));
        }
        let tile = matches!(stem, "loop.tile" | "loop.tile_strided" | "loop.tilex" | "loop.tilex_strided");
        if tile && d + 1 > 5 {
            return Err("a tile loop six deep inside the kernel's epilogue".into());
        }
        let last = t.kids.len().saturating_sub(1);
        let kids = t.kids.iter().enumerate().map(|(i, k)| if tile && i == last { self.rerung(k, d + 1) } else { self.rerung(k, d) }).collect::<Result<Vec<_>, _>>()?;
        let c = format!("C{d}");
        let ty_name = if stem.starts_with("block.") { format!("block.{c}") } else { format!("stmt.{c}") };
        Ok(Tree::new(&format!("{stem}_{c}"), ty(self, &ty_name)?, kids))
    }

    /// A list of statements as a block of the rung.
    fn block(&self, stmts: Vec<Tree>, rung: &str) -> Result<Tree, String> {
        let bt = ty(self, &format!("block.{rung}"))?;
        let st = self.stmt_ty(rung)?;
        if stmts.is_empty() {
            return Ok(Tree::fallback(bt));
        }
        // `if (c) { continue; }` inside a loop (an L or C rung below the top):
        // the guard, then the rest of the block at the conditional rung of its
        // depth (no barrier after a continue; every thread still reaches the
        // next iteration).
        let looped_rung = matches!(rung.as_bytes(), [b'L' | b'C', d] if (b'1'..=b'5').contains(d));
        // A continue whose rest stores h is the epilogue's (epilogue_block
        // reads it into block.cont_O<d>).
        if looped_rung && !stmts.iter().any(has_store) {
            if let Some(at) = stmts.iter().position(is_continue_guard) {
                let mut before = stmts;
                let rest: Vec<Tree> = before.split_off(at + 1);
                let guard = before.pop().expect("the guard at `at`");
                let d: usize = rung[1..].parse().unwrap_or(1);
                let rest = rest.iter().map(|t| self.rerung(t, d)).collect::<Result<Vec<_>, _>>()?;
                let cont = Tree::new(&format!("block.cont_{rung}"), bt, vec![guard.kids[0].clone(), self.block(rest, &format!("C{d}"))?]);
                if before.is_empty() {
                    return Ok(cont);
                }
                before.push(Tree::new(&format!("stmt.block_{rung}"), st, vec![cont]));
                return self.block(before, rung);
            }
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
        let mut pointers = HashMap::new();
        for (i, ((_, node), arg)) in args.into_iter().zip(&helper.args).enumerate() {
            // A pointer parameter given `&x`: the helper reads and writes the
            // caller's register for x.
            if let naga::TypeInner::Pointer { base, .. } = self.module.types[helper.naga_fn.arguments[i].ty].inner {
                let Node::Leaf { name: local, .. } = &node else { return Err(format!("{callee}: pointer argument {i} is not a local (&x)")) };
                if !local.starts_with("local.") || local.contains(".#") {
                    return Err(format!("{callee}: pointer argument {i} is {local}, not a local held in a register"));
                }
                let dual = dual_of_type(self.module, base).ok_or_else(|| format!("{callee}: pointer argument {i} to a type no register holds"))?;
                let row = self.register(local, &dual)?;
                let kind = row.chars().nth(4).unwrap_or('f');
                pointers.insert(arg.clone(), (row, kind));
                continue;
            }
            let raw = tree_of(self, &node)?;
            values.insert(arg.clone(), self.value(&raw)?);
            bound.insert(arg.clone(), raw);
        }
        let saved_pointers = std::mem::replace(&mut self.pointer_args, pointers);
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
        self.pointer_args = saved_pointers;
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
            // The lets defined just before this statement whose uses sit in a
            // deeper loop: set once here, as the source computes them.
            if self.local_prefix.is_empty() {
                let mut here_lets: Vec<String> = self.bound_lets.iter().filter(|(_, p)| **p == here).map(|(n, _)| n.clone()).collect();
                here_lets.sort_by_key(|n| n.rsplit('@').next().and_then(|i| i.parse::<u32>().ok()).unwrap_or(0));
                for n in here_lets {
                    let def = self.lets.get(&n).cloned().ok_or_else(|| format!("let {n} has no definition"))?;
                    // An array let (`let accs = array<vec4<f32>, 4>(...)`) has no
                    // single value: its indexed uses are read in place, as before.
                    if matches!(&def, Node::App { name, .. } if name.starts_with("naga.Compose.type")) {
                        self.notes.push(format!("let {n} is an array used in a deeper loop: its elements are read in place"));
                        continue;
                    }
                    let t = tree_of(self, &def)?;
                    let d = self.kingdom.dual(t.ty).map(|d| d.dual.clone()).unwrap_or_default();
                    if !matches!(d.as_str(), "f32.real" | "vec4<f32>.real" | "u32.index" | "u32.bits" | "u32.count" | "bool.flag") {
                        // No register of this dual (a vec2, a vec3, a matrix): the
                        // let stays inlined at its uses, and the note says so.
                        self.notes.push(format!("let {n} ({d}) is used in a deeper loop but has no register kind: inlined at its uses"));
                        continue;
                    }
                    let v = self.value(&t)?;
                    let row = self.register(&format!("local.let.{n}"), &d).map_err(|e| format!("let {n}, used in a deeper loop, bound to a register: {e}"))?;
                    let kind = row.chars().nth(4).unwrap_or('f');
                    out.push(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), v]));
                    // The ownership proof sees through the register to what it
                    // holds, as for a local (a let never changes in its scope).
                    let held = self.expand_memo(&t);
                    if held.size() <= 512 {
                        self.axis_memo.insert(format!("local.let.{n}"), (held, self.loop_depth));
                    }
                    self.let_regs.insert(n, row);
                }
            }
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
                    Statement::SubgroupCollectiveOperation { result, .. } | Statement::SubgroupGather { result, .. } | Statement::SubgroupBallot { result, .. } => {
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Subgroup { .. })).ok_or("no subgroup root at this statement")?;
                        self.subgroup_statement(root, result.index() as u32, rung).map(Some)
                    }
                    Statement::WorkGroupUniformLoad { result, .. } => {
                        // naga's lowering of workgroupUniformLoad(&p): a barrier,
                        // the load (to a register), a barrier; in uniform control
                        // flow, as a barrier is.
                        if cond {
                            return Err("a workgroupUniformLoad under a condition".into());
                        }
                        let root = self.root_at(&here, |k| matches!(k, RootKind::UniformLoad { .. })).ok_or("no uniform-load root at this statement")?;
                        // Typed by the workgroup variable's element (the inference
                        // does not see this load).
                        let RootKind::UniformLoad { target } = &root.kind else { return Err("not a uniform-load root".into()) };
                        let var = target.strip_prefix("workgroup.").and_then(|r| r.split('.').next()).ok_or_else(|| format!("workgroupUniformLoad of {target}, not a workgroup variable"))?;
                        let elem = self.workgroup_elem_dual(var)?;
                        let lty = ty(self, &elem)?;
                        let loaded = match &root.tree {
                            Node::App { name, kids, .. } => {
                                let idx = kids.first().map(|k| tree_of(self, k)).transpose()?.ok_or("an indexed uniform load without its index")?;
                                Tree::new(name, lty, vec![idx])
                            }
                            Node::Leaf { name, .. } => Tree::new(name, lty, vec![]),
                            Node::Literal { .. } => return Err("a uniform load of a literal".into()),
                        };
                        let v = self.value(&loaded)?;
                        let d = self.kingdom.dual(v.ty).map(|d| d.dual.clone()).unwrap_or_default();
                        let row = self.register(&format!("local.uniformload@{}", result.index()), &d)?;
                        let kind = row.chars().nth(4).unwrap_or('f');
                        let barrier = || Tree::new(&format!("stmt.barrier_{rung}"), st, vec![]);
                        let set = Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), v]);
                        Ok(Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(vec![barrier(), set, barrier()], rung)?])))
                    }
                    Statement::Atomic { result, .. } => {
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Atomic { .. })).ok_or("no atomic root at this statement")?;
                        self.atomic_statement(root, result.index() as u32, rung).map(Some)
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
                    Statement::Switch { cases, .. } => {
                        // A switch is its if-else chain (WGSL has no fallthrough
                        // and a selector is a value): each case group tests its
                        // values in order, the group with `default` is the last
                        // else. A case list (`case 1u, 2u:`) is naga's
                        // fall-through cases with empty bodies.
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at this switch")?;
                        let sel_tree = tree_of(self, &root.tree)?;
                        let sel = self.value(&sel_tree.kids[0])?;
                        let flag = ty(self, "bool.flag")?;
                        let mut groups: Vec<(Vec<Tree>, usize)> = Vec::new();
                        let mut default_at: Option<usize> = None;
                        let mut values: Vec<Tree> = Vec::new();
                        let mut has_default = false;
                        for (ci, case) in cases.iter().enumerate() {
                            match case.value {
                                naga::SwitchValue::I32(v) => values.push(Tree::literal("literal.int", ty(self, "i32.int")?, v as f32)),
                                naga::SwitchValue::U32(v) => values.push(Tree::literal("literal.index", ty(self, "u32.index")?, v as f32)),
                                naga::SwitchValue::Default => has_default = true,
                            }
                            if case.fall_through {
                                if case.body.iter().any(|s| !matches!(s, Statement::Emit(_))) {
                                    return Err("a switch case that falls through after statements (WGSL has no fallthrough)".into());
                                }
                                continue;
                            }
                            if has_default {
                                default_at = Some(ci);
                            } else {
                                let eqs: Vec<Tree> = values.iter().map(|v| Tree::new("compare.eq", flag, vec![sel.clone(), v.clone()])).collect();
                                let cond = eqs.into_iter().reduce(|a, b| Tree::new("logic.or", flag, vec![a, b])).ok_or("a switch case with no value")?;
                                groups.push((vec![cond], ci));
                            }
                            values.clear();
                            has_default = false;
                        }
                        // A case's body without its trailing break (it ends the case
                        // anyway); a break anywhere else in a case leaves the switch,
                        // which an if-else chain cannot say.
                        let body_of = |ci: usize| -> Result<naga::Block, String> {
                            let mut stmts: Vec<Statement> = cases[ci].body.iter().cloned().collect();
                            if let Some(at) = stmts.iter().rposition(|s| !matches!(s, Statement::Emit(_))) {
                                if matches!(stmts[at], Statement::Break) {
                                    stmts.remove(at);
                                }
                            }
                            fn has_break(b: &[Statement]) -> bool {
                                b.iter().any(|s| match s {
                                    Statement::Break => true,
                                    Statement::Block(b) => has_break(b),
                                    Statement::If { accept, reject, .. } => has_break(accept) || has_break(reject),
                                    _ => false,
                                })
                            }
                            if has_break(&stmts) {
                                return Err("a break inside a switch case before its end (it leaves the switch; the kingdom spells a switch as its if-else chain)".into());
                            }
                            Ok(naga::Block::from_vec(stmts))
                        };
                        let crung = format!("C{level}");
                        let walk_case = |ctx: &mut Self, ci: usize, path: &mut Vec<usize>, r: &str| -> Result<Vec<Tree>, String> {
                            let b = body_of(ci)?;
                            path.push(ci);
                            let out = ctx.nested(|ctx, path| ctx.walk(&b, path, r, 0), path);
                            path.pop();
                            out
                        };
                        self.tail = stmt_tail;
                        if groups.is_empty() {
                            // Only a default: its statements, unconditionally.
                            let d = match default_at {
                                Some(ci) => walk_case(self, ci, path, rung),
                                None => Ok(Vec::new()),
                            };
                            self.tail = block_tail;
                            let d = d?;
                            return Ok(if d.is_empty() { None } else { Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(d, rung)?])) });
                        }
                        let mut bodies = Vec::new();
                        for (_, ci) in &groups {
                            match walk_case(self, *ci, path, &crung) {
                                Ok(b) => bodies.push(b),
                                Err(e) => {
                                    self.tail = block_tail;
                                    return Err(e);
                                }
                            }
                        }
                        let default = match default_at {
                            Some(ci) => walk_case(self, ci, path, &crung),
                            None => Ok(Vec::new()),
                        };
                        self.tail = block_tail;
                        let default = default?;
                        // The chain from its last test up: each else holds the next if.
                        let cst = self.stmt_ty(&crung)?;
                        let mut rest: Vec<Tree> = default;
                        for (i, ((conds, _), body)) in groups.iter().zip(bodies).enumerate().rev() {
                            let (r, t) = if i == 0 { (rung.to_string(), st) } else { (crung.clone(), cst) };
                            let then = self.block(body, &crung)?;
                            let stmt = if rest.is_empty() {
                                Tree::new(&format!("stmt.if_{r}"), t, vec![conds[0].clone(), then])
                            } else {
                                Tree::new(&format!("stmt.if_else_{r}"), t, vec![conds[0].clone(), then, self.block(rest, &crung)?])
                            };
                            rest = vec![stmt];
                        }
                        Ok(rest.pop())
                    }
                    Statement::Loop { body, continuing, break_if } => {
                        self.tail = false;
                        let looped = self.looped(body, continuing, *break_if, path, rung, cond);
                        self.tail = block_tail;
                        looped
                    }
                    Statement::Barrier(flags) => {
                        if cond {
                            return Err("a barrier under a condition".into());
                        }
                        // subgroupBarrier() is the subgroup flag alone, storageBarrier()
                        // the storage flag alone, workgroupBarrier() the workgroup flag.
                        let row = match *flags {
                            naga::Barrier::SUB_GROUP => "stmt.subgroup_barrier",
                            naga::Barrier::STORAGE => "stmt.storage_barrier",
                            naga::Barrier::WORK_GROUP => "stmt.barrier",
                            other => return Err(format!("a barrier with the flags {other:?}, which no WGSL barrier writes")),
                        };
                        Ok(Some(Tree::new(&format!("{row}_{rung}"), st, vec![])))
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
                    // A continue is a marker here: the epilogue spells it (a guard
                    // then the rest of an epilogue tile loop's body); anywhere else
                    // it is refused by name when the kernel is split.
                    Statement::Continue => Ok(Some(Tree::new("stmt.continue", st, vec![]))),
                    // A break: the guard `if (c) { break; }` of a loop that breaks
                    // (looped reads it into block.brk_B<d>); anywhere else it is named.
                    Statement::Break => Ok(Some(Tree::new("stmt.break", st, vec![]))),
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
        if let Some(local) = private_as_local(tree) {
            return self.store(&local, rung);
        }
        // `*p = v` for a pointer parameter: the caller's register set.
        if let (Some((row, kind)), [v]) = (tree.name.strip_prefix("store.").and_then(|a| self.pointer_args.get(a)).cloned(), tree.kids.as_slice()) {
            let v = self.value(v)?;
            let st = self.stmt_ty(rung)?;
            return Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), v])));
        }
        let st = self.stmt_ty(rung)?;
        let name = tree.name.as_str();
        if name.starts_with("store.buffer.h.#") {
            let [idx, value] = tree.kids.as_slice() else { return Err("store to h without an index".into()) };
            // read-add-write: value is a sum with the cell's own load as one
            // term, `h[c] + x`, `h[c] + a + b` (left-nested), `a + (h[c] + b)`.
            // Each add is commutative bit for bit, so `(h[c] + a) + b` is
            // `h[c] += a; h[c] += b` exactly: one store.h per increment, in
            // the order the kernel's own adds apply them.
            let xs = h_increments(value, &idx.to_sexpr()).ok_or("a store to h is not read-add-write of the same cell")?;
            if xs.is_empty() {
                return Err("a store of h[c] to itself (no increment)".into());
            }
            let cell = self.cell(idx)?;
            let vs = xs.into_iter().map(|x| self.value(x)).collect::<Result<Vec<_>, _>>()?;
            let stores = |ctx: &Self, r: &str| -> Result<Tree, String> {
                let list = vs.iter().map(|v| Ok(Tree::new(&format!("store.h_{r}"), ctx.stmt_ty(r)?, vec![cell.clone(), v.clone()]))).collect::<Result<Vec<_>, String>>()?;
                if list.len() == 1 {
                    return Ok(list.into_iter().next().expect("one store"));
                }
                Ok(Tree::new(&format!("stmt.block_{r}"), ctx.stmt_ty(r)?, vec![ctx.block(list, r)?]))
            };
            return match self.guard.clone() {
                Some(g) if !rung.starts_with('C') => {
                    let level = &rung[1..];
                    let crung = format!("C{level}");
                    let inner = self.block(vec![stores(self, &crung)?], &crung)?;
                    Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![g, inner])))
                }
                _ => Ok(Some(stores(self, rung)?)),
            };
        }
        if let (Some(rest), [value]) = (name.strip_prefix("store.workgroup."), tree.kids.as_slice()) {
            // A workgroup scalar: element 0 of a workgroup array of one.
            let zero = Tree::literal("literal.index", ty(self, "u32.index")?, 0.0);
            return self.store(&Tree::new(&format!("store.workgroup.{rest}.#"), tree.ty, vec![zero, value.clone()]), rung);
        }
        if let Some(rest) = name.strip_prefix("store.workgroup.") {
            let arr = rest.split('.').next().unwrap_or("");
            let [idx, value] = tree.kids.as_slice() else { return Err("store to a workgroup array without an index".into()) };
            if let Some((i, j)) = nested_index(rest, &tree.kids[..1]) {
                // A lane of a vec4 element: `sa[si][comp] = x` (seed08).
                let which = self.array(arr, "vec4<f32>.real")?;
                let (of, idx_dual, arr_dual, _, _) = Self::wg_rows(which);
                let i = self.value(self.wg_unmasked(i, which))?;
                let j = self.value(unmasked(j, 3.0))?;
                let v = self.value(value)?;
                let wg = Tree::new(of, ty(self, idx_dual)?, vec![i]);
                return Ok(Some(Tree::new(&format!("wg.set_v_lane_{rung}"), st, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), wg, j, v])));
            }
            let d = self.kingdom.dual(value.ty).map(|d| d.dual.clone()).unwrap_or_default();
            let which = self.array(arr, &d)?;
            let (of, idx_dual, arr_dual, _, set) = Self::wg_rows(which);
            let i = self.value(self.wg_unmasked(idx, which))?;
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
                if let Some((cols, rows)) = self.local_mat(&base) {
                    // `m[i] = v`: the matrix rebuilt from its columns with column i
                    // replaced (the kingdom writes a matrix register whole).
                    let Some(c) = literal_value(idx).filter(|c| *c >= 0.0 && (*c as u32) < cols).map(|c| c as u32) else {
                        return Err(format!("a store to a column of the matrix {base} chosen at run time (a column store has a literal column)"));
                    };
                    if nested_index(rest, &tree.kids[..1]).is_some() {
                        return Err(format!("a store to one lane of a column of the matrix {base} (the kingdom writes a matrix register whole)"));
                    }
                    let vty = ty(self, &format!("vec{rows}<f32>.real"))?;
                    let v = self.value(value)?;
                    let whole = self.mat_register_value(&base, cols, rows)?;
                    let columns: Vec<Tree> = (0..cols).map(|i| if i == c { v.clone() } else { Tree::new(&format!("shape.col_{i}"), vty, vec![whole.clone()]) }).collect();
                    let mty = ty(self, &format!("mat{cols}x{rows}<f32>.real"))?;
                    let compose = if cols == 2 { "shape.compose_mat".to_string() } else { format!("shape.compose_mat{cols}") };
                    let kind = Self::mat_kind(cols, rows)?;
                    let row = self.register(&base, &format!("mat{cols}x{rows}<f32>.real"))?;
                    return Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, ty(self, Self::reg_dual(kind))?, vec![]), Tree::new(&compose, mty, columns)])));
                }
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
                if self.local_lanes.get(&base) == Some(&4) && self.local_vec_uint(&base) {
                    return Err(format!("a store to one lane of the vec4<u32> local {base} (the kingdom writes a vec4<u32> register whole)"));
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
                            match thread_axis_off_in(value, axis, wg, tiles, &self.row_arrays, &self.tile_bounds) {
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
                        // Its value is known again: 0, for the loop that follows.
                        self.stored_seen.insert(local.clone());
                        let index = ty(self, "u32.index")?;
                        self.axis_memo.insert(local.clone(), (Tree::literal("literal.index", index, 0.0), self.loop_depth));
                        Ok(None)
                    }
                    _ => Err(format!("the item counter {local} set to {} between loops (only 0 has a spelling)", tree.kids[0].to_sexpr())),
                };
            }
            self.stored_seen.insert(local.clone());
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
    /// A TREE REDUCTION WRITTEN AS A HALVING LOOP (2026-10-12, a reader bug
    /// by Andrew's rule: models write it so): `for (var s = n / 2; s > 0u;
    /// s = s >> 1u) { if (lidx < s) { a[lidx] = a[lidx] op a[lidx + s]; }
    /// workgroupBarrier(); }`, s starting at a power of two (its declaration's
    /// literal), stepped by `>> 1u` or `/ 2u`, op `+`, max or min, is
    /// `wg.reduce_<op>(a, n)`. `None` when the test is not `s > 0` (the loop
    /// is some other loop); an error when it is and the rest does not match.
    fn halving(&mut self, c: &Tree, continuing: &naga::Block, stmts: &[&Statement], path: &[usize], rung: &str, cond: bool) -> Result<Option<Tree>, String> {
        let zero = |t: &Tree| literal_value(t) == Some(0.0);
        let one = |t: &Tree| literal_value(t) == Some(1.0);
        let counter = match (c.name.as_str(), c.kids.as_slice()) {
            ("compare.gt" | "compare.ne", [x, z]) if zero(z) => local_name(&x.name),
            ("compare.ge", [x, o]) if one(o) => local_name(&x.name),
            ("compare.lt" | "compare.ne", [z, x]) if zero(z) => local_name(&x.name),
            _ => None,
        };
        let Some(counter) = counter.filter(|_| c.kids.iter().all(|k| k.kids.is_empty())) else { return Ok(None) };
        let fail = |why: &str| Err(format!("a halving loop over {counter} that is not a tree reduction: {why}"));
        if cond || !rung.starts_with('L') {
            return fail("under a condition (a reduction's barriers need uniform control flow)");
        }
        // The start: the counter's declaration's literal, a power of two.
        let index: Option<usize> = counter.rsplit('@').next().and_then(|i| i.parse().ok());
        let start = self.naga_fn.local_variables.iter().find(|(h, _)| Some(h.index()) == index).and_then(|(_, lv)| lv.init).and_then(|e| match self.naga_fn.expressions[e] {
            Expression::Literal(naga::Literal::U32(v)) => Some(v),
            Expression::Literal(naga::Literal::I32(v)) => u32::try_from(v).ok(),
            _ => None,
        });
        let Some(start) = start.filter(|v| v.is_power_of_two() && *v <= 2048) else { return fail("its start is not a literal power of two in its declaration (at most 2,048)") };
        // The step: `s = s >> 1u` or `s = s / 2u`, in the continuing block or
        // as the body's last statement.
        let halves = |t: &Tree| -> bool {
            local_name(&t.name).as_deref() == Some(counter.as_str())
                && t.kids.len() == 1
                && matches!((t.kids[0].name.as_str(), t.kids[0].kids.as_slice()), (n, [x, k]) if local_name(&x.name).as_deref() == Some(counter.as_str()) && ((n.ends_with(".shr") && literal_value(k) == Some(1.0)) || (n.ends_with(".div") && literal_value(k) == Some(2.0))))
        };
        let step_at = |ctx: &mut Self, p: Vec<usize>| -> Result<bool, String> {
            match ctx.root_at(&p, |k| matches!(k, RootKind::Store { .. })) {
                Some(r) => Ok(halves(&tree_of(ctx, &r.tree)?)),
                None => Ok(false),
            }
        };
        let cont: Vec<&Statement> = continuing.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        let test_pos = stmts.iter().position(|s| matches!(s, Statement::If { .. })).ok_or("a halving loop without a test")?;
        let mut rest: Vec<(usize, &Statement)> = stmts.iter().copied().enumerate().skip(test_pos + 1).collect();
        let stepped = if cont.len() == 1 {
            step_at(self, [path, &[1, 0]].concat())?
        } else if cont.is_empty() && matches!(rest.last(), Some((_, Statement::Store { .. }))) {
            let at = rest.last().map(|(i, _)| *i).unwrap_or_default();
            let ok = step_at(self, [path, &[0, at]].concat())?;
            if ok {
                rest.pop();
            }
            ok
        } else {
            false
        };
        if !stepped {
            return fail("its step is not `s >> 1u` or `s / 2u`");
        }
        // The body: `if (lidx < s) { a[lidx] = a[lidx] op a[lidx + s]; }` then
        // a workgroup barrier (in the block naga makes of a `for` body).
        let mut prefix = [path, &[0]].concat();
        if let [(at, Statement::Block(b))] = rest.as_slice() {
            prefix.push(*at);
            rest = b.iter().filter(|s| !matches!(s, Statement::Emit(_))).enumerate().collect();
        }
        let [(if_at, Statement::If { accept, reject, .. }), (_, Statement::Barrier(flags))] = rest.as_slice() else { return fail("its body is not one `if` and a barrier") };
        if !flags.contains(naga::Barrier::WORK_GROUP) || !reject.iter().all(|s| matches!(s, Statement::Emit(_))) || accept.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() != 1 {
            return fail("its body is not `if (lidx < s) { one store }` then workgroupBarrier()");
        }
        let flat = self.grid.1 == 1;
        let lane = |t: &Tree| t.name == "builtin.local_invocation_index" || (flat && t.name == "shape.access_0" && t.kids.len() == 1 && t.kids[0].name == "builtin.local_invocation_id");
        let is_counter = |t: &Tree| local_name(&t.name).as_deref() == Some(counter.as_str());
        let test = self.root_at(&[&prefix[..], &[*if_at]].concat(), |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at the halving loop's step")?;
        let test = tree_of(self, &test.tree)?;
        let guard = &test.kids[0];
        let guarded = match (guard.name.as_str(), guard.kids.as_slice()) {
            ("compare.lt", [l, s]) | ("compare.gt", [s, l]) => lane(l) && is_counter(s),
            _ => false,
        };
        if !guarded {
            return fail(&format!("its step is not under `lidx < s`: {}", guard.to_sexpr()));
        }
        let store = self.root_at(&[&prefix[..], &[*if_at, 0, 0]].concat(), |k| matches!(k, RootKind::Store { .. })).ok_or("no store under the halving loop's test")?;
        let store = tree_of(self, &store.tree)?;
        let Some(arr) = store.name.strip_prefix("store.workgroup.").and_then(|r| r.strip_suffix(".#")) else { return fail("its step does not store a workgroup array") };
        let [at, value] = store.kids.as_slice() else { return fail("its store has no index") };
        let load = |t: &Tree, i: &dyn Fn(&Tree) -> bool| t.name.starts_with(&format!("load.workgroup.{arr}.#")) && t.kids.len() == 1 && i(&t.kids[0]);
        let partner = |t: &Tree| t.name == "index.add" && t.kids.len() == 2 && ((lane(&t.kids[0]) && is_counter(&t.kids[1])) || (is_counter(&t.kids[0]) && lane(&t.kids[1])));
        let op = match (value.name.as_str(), value.kids.as_slice()) {
            (n, [x, y]) if (load(x, &lane) && load(y, &partner)) || (load(y, &lane) && load(x, &partner)) => match n {
                "arith.add" => "add",
                "arith.max" => "max",
                "arith.min" => "min",
                _ => return fail(&format!("it combines by {n}, not +, max or min")),
            },
            _ => return fail(&format!("its step is not a[lidx] op a[lidx + s]: {}", value.to_sexpr())),
        };
        if !lane(at) {
            return fail("its store is not at a[lidx]");
        }
        let d = self.kingdom.dual(value.ty).map(|d| d.dual.clone()).unwrap_or_default();
        let which = self.array(arr, &d)?;
        let (_, _, arr_dual, _, _) = Self::wg_rows(which);
        let n = Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, (start * 2) as f32);
        Ok(Some(Tree::new(&format!("wg.reduce_{op}_{rung}"), self.stmt_ty(rung)?, vec![Tree::new(which, ty(self, arr_dual)?, vec![]), n])))
    }

    fn looped(&mut self, body: &naga::Block, continuing: &naga::Block, break_if: Option<Handle<Expression>>, path: &mut Vec<usize>, rung: &str, cond: bool) -> Result<Option<Tree>, String> {
        // The condition: the first non-Emit statement of the body is an If
        // whose one arm is a Break; calls of pure helpers may precede it (the
        // kingdom's own `lit_u32` in a rendered bound, which naga calls at
        // the top of the body).
        let body_stmts_all: Vec<&Statement> = body.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        // `loop { ...; continuing { step; break if c; } }` (a do-while): no
        // test at the top; the break condition is the root at the loop's path.
        let do_while = break_if.is_some();
        let test_pos = if do_while { 0 } else { body_stmts_all.iter().position(|s| matches!(s, Statement::If { .. })).ok_or("a loop without a test")? };
        for (i, s) in body_stmts_all[..test_pos].iter().enumerate().filter(|_| !do_while) {
            // A subgroup operation or an atomic in the bound: naga runs it
            // ahead of the test; the bound holds its expression inline.
            if let Statement::SubgroupCollectiveOperation { result, .. } | Statement::SubgroupGather { result, .. } | Statement::Atomic { result, .. } = s {
                let at = [&path[..], &[0, i]].concat();
                let subgroup = !matches!(s, Statement::Atomic { .. });
                let root = if subgroup { self.root_at(&at, |k| matches!(k, RootKind::Subgroup { .. })) } else { self.root_at(&at, |k| matches!(k, RootKind::Atomic { .. })) };
                let root = root.ok_or("no root at a statement ahead of the loop's test")?;
                let (expr, leaf) = if subgroup { (self.subgroup_expr(root)?.0, format!("subgroup@{}", result.index())) } else { (self.atomic_expr(root)?.0, format!("atomic@{}", result.index())) };
                self.inlined.insert(leaf, expr);
                continue;
            }
            let Statement::Call { function, .. } = s else { return Err("a loop whose first statement is not its test".into()) };
            let callee = self.module.functions[*function].name.clone().unwrap_or_else(|| format!("fn{}", function.index()));
            if !self.helpers.get(&callee).is_some_and(|h| h.pure || h.composite) {
                return Err(format!("a loop whose test calls {callee}, a helper with statements"));
            }
        }
        let (negated, root) = if do_while {
            (true, self.root_at(path, |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at the loop's break if")?)
        } else {
            let Statement::If { accept, reject, .. } = body_stmts_all[test_pos] else { return Err("a loop whose first statement is not its test".into()) };
            let is_break = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Break));
            let is_empty = |b: &naga::Block| b.iter().all(|s| matches!(s, Statement::Emit(_)));
            let negated = if is_break(reject) && is_empty(accept) { false } else if is_break(accept) && is_empty(reject) { true } else { return Err("a loop whose test does not break".into()) };
            path.push(0);
            path.push(test_pos);
            let root = self.root_at(path, |k| matches!(k, RootKind::Condition { .. }));
            path.pop();
            path.pop();
            (negated, root.ok_or("no condition root at the loop's test")?)
        };
        // The test is the loop's header: a let it reads is its value, inlined
        // (a let never changes in its scope), so the bound stays an expression
        // the cost estimator can value, not a register it cannot see.
        let saved_regs = std::mem::take(&mut self.let_regs);
        let test = tree_of(self, &root.tree);
        self.let_regs = saved_regs;
        let test = test?;
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
        // A tree reduction written as a halving loop is the reduction row.
        if let Some(t) = if do_while { None } else { self.halving(&c, continuing, &body_stmts_all, path, rung, cond)? } {
            return Ok(Some(t));
        }
        // The test normalised to `counter < bound`: `bound > counter`,
        // `counter <= bound` (bound + 1), `counter + k < bound` and
        // `counter + k > bound` as `if (...) break` (bound - k, floored).
        let (counter, bound) = normalise_test(&c).ok_or_else(|| format!("a loop test that is not a counter against a bound: {}", c.to_sexpr()))?;
        // The bound is evaluated once, so nothing it reads may change in the
        // loop's body (fresh28's `t0` is the enclosing loop's counter).
        let stored = stored_locals(self.naga_fn, body);
        let mut read_locals = Vec::new();
        locals_in(&bound, &mut read_locals);
        if let Some(l) = read_locals.iter().find(|l| **l != counter && stored.contains(l)) {
            return Err(format!("a loop bound that reads {l}, which the loop's body stores"));
        }
        let bound = &bound;
        let index_ty = ty(self, "u32.index")?;
        // A binding bound (cfg.m, cfg.d, cfg.r_rows) is the item loop's at the
        // top level; nested (under a guard, inside a loop: fit_A2 gen 6's r12p1
        // puts its whole body under `if (in range)`), it is spelled as the
        // expression it is, a uniform, and the loop is an expression-bounded one.
        let top = rung == "L0";
        let (bound_tree, kind) = if top && is_leaf(bound, "load.uniform.cfg.m") {
            (Tree::new("bound.m", ty(self, "u32.boundm")?, vec![]), "boundm")
        } else if top && is_leaf(bound, "load.uniform.cfg.d") {
            (Tree::new("bound.d", ty(self, "u32.boundd")?, vec![]), "boundd")
        } else if top && is_leaf(bound, "load.uniform.cfg.r_rows") {
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
        // A `while (t < n) { ...; t = t + s; }`: naga writes the body after
        // the test as one block, so the step is that block's last statement.
        let mut block_last_is_step = false;
        if step.is_none() {
            if let Some(Statement::Block(b)) = body_stmts_all.last() {
                let inner: Vec<&Statement> = b.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
                if let Some(Statement::Store { .. }) = inner.last() {
                    let lpath = [&path[..], &[0, body_stmts_all.len() - 1, inner.len() - 1]].concat();
                    if let Some(r) = self.root_at(&lpath, |k| matches!(k, RootKind::Store { .. })) {
                        let t = tree_of(self, &r.tree)?;
                        if local_name(&t.name).as_deref() == Some(counter.as_str()) {
                            let v = &t.kids[0];
                            let lit = if v.name == "index.add" && v.kids.len() == 2 && v.kids.iter().any(|k| local_name(&k.name).as_deref() == Some(counter.as_str())) { v.kids.iter().find_map(literal_value) } else { None };
                            if let Some(x) = lit.filter(|x| (1.0..=4096.0).contains(x)) {
                                step = Some(x as u32);
                                block_last_is_step = true;
                            }
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
        if (kind == "lit64" || kind == "boundx") && depth > 4 && body_rung != "L1" {
            return Err("a fifth nested tile loop".into());
        }
        // The counter's value as the loop starts: WGSL's zero for a local not
        // stored yet, else the literal last stored (a finished tile loop
        // leaves its exit value); a tile loop that starts at a reads its
        // counter as a + k and runs bound - a times.
        let entry = if !self.stored_seen.contains(&counter) { Some(0.0) } else { self.axis_memo.get(&counter).and_then(|(t, _)| literal_value(t)) };
        let item_loop = !(kind == "lit64" || kind == "boundx");
        let start = match entry {
            Some(v) => v,
            None if item_loop => 0.0,
            None => return Err(format!("a loop whose counter {counter} enters with a value our reader does not know (only a literal or zero has a spelling)")),
        };
        let (bound_tree, start) = if start == 0.0 || item_loop {
            (bound_tree, 0.0)
        } else if kind == "lit64" {
            let n = literal_value(bound).unwrap_or(0.0);
            if n <= start {
                if do_while {
                    return Err(format!("a loop with `break if` entered at {start}, at or past its bound {n} (it runs once; the counted loop would not)"));
                }
                // It never runs: the counter keeps its value.
                return Ok(None);
            }
            (Tree::literal("literal.lit64", ty(self, "u32.lit64")?, n - start), start)
        } else {
            let index = ty(self, "u32.index")?;
            let expr = Tree::new("index.sub_floor", index, vec![bound_tree.kids[0].clone(), Tree::literal("literal.index", index, start)]);
            (Tree::new("bound.of_index", bound_tree.ty, vec![expr]), start)
        };
        if item_loop && entry.is_some_and(|v| v != 0.0) && !self.used_counters.contains(&counter) {
            return Err(format!("the item loop's counter {counter} starts at {} (an item loop starts at 0)", entry.unwrap_or_default()));
        }
        // A do-while runs its body once before its first test: it is the
        // counted loop when it enters below a literal bound (a loop entered
        // at or past its bound returned above, which a do-while may not).
        if do_while && (kind != "lit64" || entry.is_none()) {
            return Err(format!("a loop with `break if` whose bound is not a literal or whose counter's entry value our reader does not know ({counter})"));
        }
        let exit = (kind == "lit64").then(|| literal_value(bound).map(|n| start + ((n - start) / step as f32).ceil() * step as f32)).flatten();
        self.counter_offsets.insert(counter.clone(), start);
        let saved = (self.item_counter.clone(), self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone(), self.tile_counter4.clone());
        self.axis_memo.remove(&counter);
        if kind == "lit64" {
            if let Some(v) = literal_value(bound) {
                self.tile_bounds.insert(counter.clone(), v as u32);
            }
        }
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
        let first = if do_while { 0 } else { test_pos + 1 };
        let mut rest: Vec<Statement> = body_stmts_all.iter().skip(first).map(|s| (*s).clone()).collect();
        // A loop that breaks (`if (c) { break; }` at the top of its body): its
        // body is read at the conditional rung of its depth, no barrier in it.
        let breaks = has_break_guard(&rest);
        if breaks && (body_rung == "L1" || step != 1) {
            self.item_counter = saved.0.clone();
            self.tile_counter = saved.1.clone();
            self.tile_counter2 = saved.2.clone();
            self.tile_counter3 = saved.3.clone();
            self.tile_counter4 = saved.4.clone();
            return Err(if body_rung == "L1" { "the item loop breaks (a loop that breaks is a tile loop: its body holds no barrier)".into() } else { format!("a loop stepped by {step} that breaks (a loop that breaks steps by 1)") });
        }
        let walk_rung = if breaks { format!("C{}", &body_rung[1..]) } else { body_rung.clone() };
        if last_is_step {
            rest.pop();
        }
        if block_last_is_step {
            // The step out of the body's block (its last statement; the
            // block keeps its place, so every earlier path is unchanged).
            if let Some(Statement::Block(b)) = rest.pop() {
                let mut inner: Vec<Statement> = b.iter().cloned().collect();
                if let Some(at) = inner.iter().rposition(|s| !matches!(s, Statement::Emit(_))) {
                    inner.remove(at);
                }
                rest.push(Statement::Block(naga::Block::from_vec(inner)));
            }
        }
        let rest = naga::Block::from_vec(rest);
        path.push(0);
        for local in stored_locals(self.naga_fn, body).into_iter().chain(stored_locals(self.naga_fn, continuing)) {
            self.axis_memo.remove(&prefixed(&self.local_prefix, &local));
        }
        let body_stmts = self.nested(|ctx, path| ctx.walk(&rest, path, &walk_rung, first), path);
        path.pop();
        self.item_counter = saved.0;
        self.tile_counter = saved.1;
        self.tile_counter2 = saved.2;
        self.tile_counter3 = saved.3;
        self.tile_counter4 = saved.4;
        self.tile_bounds.remove(&counter);
        // The counter's value after the loop: its literal exit, for a later
        // loop that does not set it again.
        self.counter_offsets.remove(&counter);
        self.stored_seen.insert(counter.clone());
        match exit.filter(|_| !breaks) {
            Some(x) => {
                let index = ty(self, "u32.index")?;
                self.axis_memo.insert(counter.clone(), (Tree::literal("literal.index", index, x), self.loop_depth));
            }
            None => {
                self.axis_memo.remove(&counter);
            }
        }
        if body_rung == "L1" {
            self.used_counters.insert(counter.clone());
        }
        let body_stmts = body_stmts?;
        let st = self.stmt_ty(out_rung)?;
        if breaks {
            // before; if (c) { break; }; after, each conditional.
            let mut flat = Vec::new();
            for t in body_stmts {
                flatten_stmts(t, &mut flat);
            }
            let at = flat.iter().position(|t| break_guard(t, self).is_some()).ok_or("a loop that breaks, its guard not at the top of its body")?;
            let c = break_guard(&flat[at], self).expect("the guard at `at`")?;
            let after: Vec<Tree> = flat.split_off(at + 1);
            flat.pop();
            if flat.iter().chain(&after).any(has_break) {
                return Err("a second break in one loop body, or a break below the top of its body".into());
            }
            let d = &body_rung[1..];
            let brk = Tree::new(&format!("block.brk_B{d}"), ty(self, &format!("block.B{d}"))?, vec![self.block(flat, &walk_rung)?, c, self.block(after, &walk_rung)?]);
            let row = format!("loop.{}_brk_{out_rung}", if kind == "lit64" { "tile" } else { "tilex" });
            return Ok(Some(Tree::new(&row, st, vec![bound_tree, brk])));
        }
        let block = self.block(body_stmts, &body_rung)?;
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
    normalise_test_for(c, None)
}

/// [`normalise_test`] with the counter known (from the other side of a
/// conjunction), and two more forms: `x && y` over one counter is the
/// smaller of the two bounds (fresh28 of fit_A2: `b < 16u && (t0 + b * 64u)
/// < m`), and `a + k * c < M` with a literal c (both thresholds rise with the
/// counter) is `k < (M - a, floored, + c - 1) / c`.
fn normalise_test_for(c: &Tree, hint: Option<&str>) -> Option<(String, Tree)> {
    if c.name == "logic.and" && c.kids.len() == 2 {
        let (x, y) = (&c.kids[0], &c.kids[1]);
        let (counter, bx, by) = match normalise_test_for(x, hint) {
            Some((k, bx)) => {
                let (k2, by) = normalise_test_for(y, Some(&k))?;
                (k.clone(), bx, if k2 == k { by } else { return None })
            }
            None => {
                let (k, by) = normalise_test_for(y, hint)?;
                let (k2, bx) = normalise_test_for(x, Some(&k))?;
                (k.clone(), if k2 == k { bx } else { return None }, by)
            }
        };
        let ty = bx.ty;
        return Some((counter, Tree::new("index.min", ty, vec![bx, by])));
    }
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
                    if hint.is_none_or(|h| h == counter) {
                        return Some((counter, Tree::new("index.sub_floor", index_ty, vec![bound, Tree::literal("literal.index", index_ty, k)])));
                    }
                }
            }
        }
    }
    // `a + k * c < M`: one term the counter (alone or times a literal c > 0),
    // the others free of it.
    let mut terms = Vec::new();
    flatten_add(lhs, &mut terms);
    let counter_term = |t: &Tree| -> Option<(String, f32)> {
        if t.kids.is_empty() {
            return local_name(&t.name).map(|n| (n, 1.0));
        }
        if t.name == "index.mul" && t.kids.len() == 2 {
            for (x, l) in [(&t.kids[0], &t.kids[1]), (&t.kids[1], &t.kids[0])] {
                if let (true, Some(n), Some(v)) = (x.kids.is_empty(), local_name(&x.name), literal_value(l)) {
                    if v >= 1.0 {
                        return Some((n, v));
                    }
                }
            }
        }
        None
    };
    let candidates: Vec<(usize, String, f32)> = terms.iter().enumerate().filter_map(|(i, t)| counter_term(t).map(|(n, v)| (i, n, v))).filter(|(_, n, _)| hint.is_none_or(|h| h == n)).collect();
    let [(at, counter, step)] = candidates.as_slice() else { return None };
    let mentions = |t: &Tree| -> bool {
        fn any(t: &Tree, counter: &str) -> bool {
            local_name(&t.name).as_deref() == Some(counter) || t.kids.iter().any(|k| any(k, counter))
        }
        any(t, counter)
    };
    let rest: Vec<Tree> = terms.iter().enumerate().filter(|(i, _)| i != at).map(|(_, t)| (*t).clone()).collect();
    if rest.iter().any(mentions) {
        return None;
    }
    let lit = |v: f32| Tree::literal("literal.index", index_ty, v);
    let room = match rest.into_iter().reduce(|acc, t| Tree::new("index.add", index_ty, vec![acc, t])) {
        Some(a) => Tree::new("index.sub_floor", index_ty, vec![bound, a]),
        None => bound,
    };
    let bound = if *step == 1.0 { room } else { Tree::new("index.div", index_ty, vec![Tree::new("index.add", index_ty, vec![room, lit(step - 1.0)]), lit(*step)]) };
    Some((counter.clone(), bound))
}

/// The locals a tree loads, as `local.<name>@<index>`.
fn locals_in(t: &Tree, out: &mut Vec<String>) {
    if let Some(l) = local_name(&t.name) {
        out.push(l);
    }
    t.kids.iter().for_each(|k| locals_in(k, out));
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

/// The `// grid x y bx by [rows-x]` header's four numbers.
pub(crate) fn grid_header(source: &str) -> Option<(u32, u32, u32, u32)> {
    header_fields(source).map(|(g, _)| g)
}

/// Whether the header says `rows-x` (the layout knob, `grid.of_rows_x`).
pub(crate) fn grid_rows_x(source: &str) -> bool {
    header_fields(source).is_some_and(|(_, r)| r)
}

fn header_fields(source: &str) -> Option<((u32, u32, u32, u32), bool)> {
    source.lines().map(str::trim).find_map(|l| {
        let rest = l.strip_prefix("// grid ")?;
        let mut tokens: Vec<&str> = rest.split_whitespace().collect();
        let rows_x = tokens.last() == Some(&"rows-x");
        if rows_x {
            tokens.pop();
        }
        let v: Vec<u32> = tokens.iter().map(|t| t.parse().ok()).collect::<Option<Vec<u32>>>()?;
        (v.len() == 4).then_some(((v[0], v[1], v[2], v[3]), rows_x))
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

/// Read `source` into the safe grammar, or say why not: the tree with its
/// knobs as parameters (`params::assign`: every position that depends on a
/// block factor, the workgroup's extent or the tile depth written as a
/// reference to it); it renders the same text as [`read_safe_literal`]'s.
pub fn read_safe(source: &str, kingdom: &WgslKingdom) -> Result<SafeRead, String> {
    let r = read_safe_literal(source, kingdom)?;
    let tree = crate::wgsl::params::assign(&r.tree, kingdom)?;
    Ok(SafeRead { tree, ..r })
}

/// Read `source` into the safe grammar with every knob-dependent position
/// left as the literal the kernel wrote.
pub fn read_safe_literal(source: &str, kingdom: &WgslKingdom) -> Result<SafeRead, String> {
    let grid = grid_header(source).ok_or("no `// grid x y bx by` header")?;
    let kernel = read(source)?;
    // The workgroup the header names is the one the kernel declares: x and y
    // each 1..=1,024 with x * y <= 1,024 (the M3 Max's limits), z 1 (a thread
    // owns cells by x and y; two threads differing only in z would share them).
    let declared = kernel.module.entry_points.first().ok_or("no entry point")?.workgroup_size;
    if declared != [grid.0, grid.1, 1] {
        return Err(format!("the header's workgroup {} x {} is not the kernel's @workgroup_size({}, {}, {})", grid.0, grid.1, declared[0], declared[1], declared[2]));
    }
    if grid.0 == 0 || grid.1 == 0 || grid.0 * grid.1 > crate::wgsl::render::MAX_THREADS {
        return Err(format!("workgroup {} x {} is outside 1..=1,024 invocations", grid.0, grid.1));
    }
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
        let mut returns: Vec<&Root> = Vec::new();
        for r in &hf.roots {
            match (&r.kind, &r.tree) {
                (RootKind::Let { name }, Node::App { kids, .. }) => {
                    lets.insert(name.clone(), kids[0].clone());
                }
                (RootKind::Return, Node::App { .. }) => returns.push(r),
                (RootKind::Init { .. }, _) => inits.push(r),
                (RootKind::Argument { position, .. }, Node::App { kids, .. }) => arg_roots.entry(r.path.clone()).or_default().push((*position, kids[0].clone())),
                _ => {
                    roots.entry(r.path.clone()).or_default().push(r);
                }
            }
        }
        // One return: the helper's value. Several: `if (c) { return a; }` ...
        // `return b;` (or a last if-else of two returns) is the select chain
        // select(.., a, c), its guards' conditions read into it.
        match returns.as_slice() {
            [] => {}
            [r] => {
                if let Node::App { kids, .. } = &r.tree {
                    body = Some(kids[0].clone());
                }
                roots.entry(r.path.clone()).or_default().push(r);
            }
            several => {
                let chain = guard_chain(&nf.body, several, &mut roots).ok_or_else(|| format!("helper {name} returns twice, not as guards `if (c) {{ return a; }}` before its last return (not inlined)"))?;
                for r in several {
                    roots.entry(r.path.clone()).or_default().push(r);
                }
                body = Some(chain);
            }
        }
        let pure = inits.is_empty() && arg_roots.is_empty() && roots.values().all(|rs| rs.iter().all(|r| matches!(r.kind, RootKind::Return)));
        let composite = (name.starts_with("at_vec") || name.starts_with("col_mat")) && nf.arguments.len() == 2 && COMPOSITES.iter().any(|(n, text)| *n == name && source.contains(text));
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
    let bound_lets = lets_used_in_deeper_loops(&f.roots, &naga_fn.body);
    let mut ctx = Ctx { kingdom, duals: &duals, lets: &lets, roots: &roots, item_counter: None, tile_counter: None, tile_counter2: None, tile_counter3: None, tile_counter4: None, registers: BTreeMap::new(), next_reg: BTreeMap::new(), guard: None, tail: true, arrays: BTreeMap::new(), array_lens: BTreeMap::new(), parrs: BTreeMap::new(), grid, rows_x: grid_rows_x(source).then_some(true), notes: Vec::new(), helpers: &helpers, calls: &calls, arg_roots: &arg_roots, inline_args: None, row_arrays: BTreeMap::new(), used_counters: std::collections::BTreeSet::new(), local_lanes, axis_memo: BTreeMap::new(), loop_depth: 0, tile_bounds: BTreeMap::new(), naga_fn, module: &kernel.module, local_prefix: String::new(), inline_result: None, inline_values: None, bound_lets, let_regs: HashMap::new(), inlined: HashMap::new(), stored_seen: std::collections::BTreeSet::new(), counter_offsets: HashMap::new(), pointer_args: HashMap::new() };
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
    // A `var<private>` with an initializer: its value stored at the start.
    for (_, gv) in kernel.module.global_variables.iter().filter(|(_, g)| g.space == naga::AddressSpace::Private) {
        let Some(init) = gv.init else { continue };
        let name = gv.name.clone().unwrap_or_default();
        let (form, dual) = match kernel.module.types[gv.ty].inner {
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, .. }) => ("real", "f32.real"),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, .. }) => ("index", "u32.index"),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, .. }) => ("int", "i32.int"),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Bool, .. }) => ("flag", "bool.flag"),
            ref other => return Err(format!("the private variable {name} of {other:?} with an initializer (a scalar's has a spelling)")),
        };
        let v = match kernel.module.global_expressions[init] {
            Expression::Literal(naga::Literal::F32(x)) => f64::from(x),
            Expression::Literal(naga::Literal::U32(x)) => f64::from(x),
            Expression::Literal(naga::Literal::I32(x)) => f64::from(x),
            Expression::Literal(naga::Literal::Bool(b)) => f64::from(u8::from(b)),
            Expression::ZeroValue(_) => 0.0,
            ref other => return Err(format!("the private variable {name}'s initializer {other:?} is not a literal")),
        };
        if v == 0.0 {
            continue;
        }
        let t = dual_ty(kingdom, dual)?;
        let lit = Tree::literal(&format!("literal.{form}"), t, v as f32);
        stmts.extend(ctx.store(&Tree::new(&format!("store.private.{name}"), t, vec![lit]), "L0")?);
    }
    let body = ctx.walk(&naga_fn.body, &mut Vec::new(), "L0", 0)?;
    stmts.extend(body);
    // The kernel ends in its epilogue: the last top-level statement that
    // stores h, with what wraps the store; everything before it is the body.
    let (stmts, epilogue) = ctx.split_epilogue(stmts)?;
    let block = ctx.block(stmts, "L0")?;
    let g = |n: &str, v: u32| Tree::literal(n, dual_ty(kingdom, if n == "literal.wgdim" { "u32.wgdim" } else { "u32.blk" }).expect("dual"), v as f32);
    let rows_x = ctx.rows_x == Some(true);
    let grid_tree = Tree::new(if rows_x { "grid.of_rows_x" } else { "grid.of" }, dual_ty(kingdom, "grid.grid")?, vec![g("literal.wgdim", grid.0), g("literal.wgdim", grid.1), g("literal.blk", grid.2), g("literal.blk", grid.3)]);
    // The kernel's workgroup arrays, each at its declared length (2026-10-12):
    // the grid carries them as a list of declarations.
    let grid_tree = if ctx.array_lens.is_empty() {
        grid_tree
    } else {
        let decl_ty = dual_ty(kingdom, "wg.decl")?;
        let decls_ty = dual_ty(kingdom, "wg.decls")?;
        let decls: Vec<Tree> = ctx
            .array_lens
            .iter()
            .map(|(which, len)| -> Result<Tree, String> {
                let (row, arr_dual) = if which.starts_with("wg.au") {
                    ("wg.decl_au", "wgmem<atomic<u32>>.index")
                } else if which.starts_with("wg.ai") {
                    ("wg.decl_ai", "wgmem<atomic<i32>>.int")
                } else if which.starts_with("wg.v2") {
                    ("wg.decl_v2", "wgmem<vec2<f32>>.real")
                } else if which.starts_with("wg.v") {
                    ("wg.decl_v4", "wgmem<vec4<f32>>.real")
                } else {
                    ("wg.decl_f", "wgmem<f32>.real")
                };
                Ok(Tree::new(row, decl_ty, vec![Tree::new(which, dual_ty(kingdom, arr_dual)?, vec![]), Tree::literal("literal.wglen", dual_ty(kingdom, "u32.wglen")?, *len as f32)]))
            })
            .collect::<Result<_, _>>()?;
        let mut list: Option<Tree> = None;
        for d in decls.into_iter().rev() {
            list = Some(match list {
                None => Tree::new("wg.decls_one", decls_ty, vec![d]),
                Some(rest) => Tree::new("wg.decls_cons", decls_ty, vec![d, rest]),
            });
        }
        Tree::new("grid.with_wg", dual_ty(kingdom, "grid.grid")?, vec![grid_tree, list.expect("one declaration at least")])
    };
    let tree = Tree::new("kernel.entry_safe", dual_ty(kingdom, "kernel.safe")?, vec![grid_tree, block, epilogue]);
    // The thread's axes: a component of a thread id is the axis row it is
    // (`lid.y` is `axis.y_lid`, a row-axis index); renamed here, after the
    // ownership proof, which reads the components as naga wrote them.
    let tree = by_axis(&tree, rows_x);
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
        assert!(sx.contains("(block.one_O0 (stmt.if_O0") && sx.contains("store.h_O0 (safe.cell_of (safe.thread_row"), "the guarded store is the epilogue: {sx}");
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
                Ok(o) => format!("{}\n{}", o.grid.header(), o.text),
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
        for dir in ["logs/HGP/001/popA_g001/texts", "logs/HGP/001/popA2_g001b/texts", "logs/HGP/001/library_texts"] {
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
                    Ok(o) => format!("{}\n{}", o.grid.header(), o.text),
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

    /// A THREAD'S BLOCK IN ANY ALGEBRAIC SPELLING (fit_A2 gen 2's fresh43:
    /// `c0 = (wid.x * 16u + lid.x) * 4u`, workgroup 16, block 4): the
    /// factored form, the literal first, and the expanded form
    /// `wid.x * 64u + lid.x * 4u` all prove the same owned cell, rendered on
    /// the kingdom's own thread rows.
    #[test]
    fn a_factored_block_index_is_proved_as_the_expanded_one() {
        let k = WgslKingdom::load_safe();
        let factored = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g002_fresh43_sonnet.wgsl");
        let (c0, r0) = ("let c0 = (wid.x * 16u + lid.x) * 4u;", "let r0 = (wid.y * 16u + lid.y) * 4u;");
        assert!(factored.contains(c0) && factored.contains(r0));
        let literal_first = factored.replace(c0, "let c0 = 4u * (lid.x + 16u * wid.x);").replace(r0, "let r0 = 4u * (lid.y + wid.y * 16u);");
        let expanded = factored.replace(c0, "let c0 = wid.x * 64u + lid.x * 4u;").replace(r0, "let r0 = wid.y * 64u + lid.y * 4u;");
        let cell = |src: &str| {
            let sx = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree.to_sexpr();
            let at = sx.find("(safe.cell_of").unwrap_or_else(|| panic!("no owned cell: {sx}"));
            sx[at..at + 160.min(sx.len() - at)].to_string()
        };
        let want = cell(&expanded);
        assert!(want.contains("safe.thread_row") && want.contains("safe.thread_col"), "{want}");
        assert_eq!(cell(factored), want);
        assert_eq!(cell(&literal_first), want);
    }

    /// AN ITEM LOOP OVER GROUPS OF ITEMS IS PRICED BY ITS PASSES (fit_A2 gen
    /// 2's fresh43_g12, `t0 < (m + 15u) / 16u`, 16 items a pass; 128/128 on
    /// the GPU and 8x faster than shipped): the estimate floored the passes
    /// at one and priced it at 3,413 per item against the bound of 2,560, so
    /// the run never dispatched it. It is 213 per item now.
    #[test]
    fn an_item_loop_over_groups_of_items_is_priced_by_its_passes() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g002_fresh43_g12_opus.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let c = crate::wgsl::render::cost(&tree);
        let bound = crate::wgsl::safe_karva::default_bound(&k).unwrap();
        assert!(c.per_item < 250.0 && c.per_item < bound.per_item / 10.0, "per item {} against the bound {}", c.per_item, bound.per_item);
        assert!(c.base <= bound.base, "base {} against the bound {}", c.base, bound.base);
    }

    /// ONE ARRAY IS ONE PRIVATE ARRAY (fit_A2 gen 1's r2x_union): o0, o1,
    /// s0 and s1, each built whole by an array constructor and then indexed,
    /// take the four vec4 private arrays, one each.
    #[test]
    fn a_constructed_and_indexed_array_takes_one_private_array() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g001_r2x_union_sonnet.wgsl");
        let sx = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree.to_sexpr();
        for arr in ["parr.v", "parr.w", "parr.x", "parr.y"] {
            assert!(sx.contains(arr), "{arr} unused");
        }
    }

    /// A GROUPED BLOCK FROM ONE COUNTER (fit_A2 gen 2's r2x_reinvent, grid
    /// 16 x 16, block 4 x 8): the row wid.y * 128u + (r >> 2u) * 64u +
    /// lid.y * 4u + (r & 3u) with r < 8 is the thread's 8 rows as 2 groups
    /// of 4, 64 apart; the shift and the mask prove the group below 2 and
    /// the lane below 4, and the store is on the kingdom's grouped rows.
    #[test]
    fn a_grouped_block_from_one_counter_split_by_shift_and_mask_is_proved() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g002_r2x_reinvent_onecounter_sonnet.wgsl");
        assert!(src.contains("(r >> 2u) * 64u") && src.contains("(r & 3u)"));
        let sx = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree.to_sexpr();
        assert!(sx.contains("(safe.row_plus_g (safe.thread_row_g"), "{sx}");
        // A mask or shift that leaves the block is not proved: r & 7u reaches 7, past a lane of 4.
        let wide = src.replace("(r & 3u)", "(r & 7u)");
        assert!(read_safe(&wide, &k).is_err(), "r & 7u is not a lane of 4");
    }

    /// A LOOP TESTED BY TWO CONDITIONS (fit_A2's fresh28: `b < 16u && (t0 +
    /// b * 64u) < m`): the conjunction over one counter is the smaller bound,
    /// and the affine test is `b < (sub_floor(m, t0) + 63u) / 64u`; the
    /// kernel reads as a tile loop bounded by that expression. A bound that
    /// reads a local the body stores is not a bound.
    #[test]
    fn a_loop_tested_by_a_conjunction_over_one_counter_reads_as_the_smaller_bound() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g002_fresh28_opus.wgsl");
        let test = "b < 16u && (t0 + b * 64u) < m";
        assert!(src.contains(test));
        let sx = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree.to_sexpr();
        assert!(sx.contains("(bound.of_index (index.min (literal.index 16) (index.div (index.add (index.sub_floor"), "{sx}");
        // The bound read once must not change in the body: t0 stored inside the b loop is refused by name.
        let moved = src.replacen("var q0 = vec4<f32>(0.0);", "t0 = t0 + 0u; var q0 = vec4<f32>(0.0);", 1);
        let Err(e) = read_safe(&moved, &k) else { panic!("t0 changes in the body, and the kernel read") };
        assert!(e.contains("which the loop's body stores"), "{e}");
    }

    /// A LET THAT READS MEMORY, USED IN A DEEPER LOOP, IS SET ONCE WHERE IT IS
    /// DEFINED (fit_A2's r4x_reinvent): `let a` and `let b` (four workgroup reads each) per kk,
    /// used in the 4 x 4 loop inside, are registers set before that loop, so
    /// the multiply-add reads registers; `let idx`, used in its own loop
    /// body only, stays inlined. The per-item cost is pinned: 3,259 when
    /// every let was inlined, below 1,000 now.
    #[test]
    fn a_let_used_in_a_deeper_loop_is_a_register_set_at_its_definition() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g002_r4x_reinvent_sonnet.wgsl");
        assert!(src.contains("let a = vec4<f32>(wg_a[") && src.contains("let idx = lidx + 128u * k;"));
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let (t, role) = (FunctionTable::safe(), gram_role());
        let text = Renderer::new(&k, &t, &role).kernel(&tree).unwrap().text;
        let fma = text.lines().find(|l| l.contains("pb[") && l.contains(" * ")).unwrap_or_else(|| panic!("no multiply-add\n{text}"));
        assert!(!fma.contains("wg_a[") && !fma.contains("wg_b["), "the multiply-add reads registers: {fma}");
        assert_eq!(text.matches("wg_a[((((k * 32u)").count(), 1, "a's first read once, before the inner loops\n{text}");
        // (the mask is the source's own array length, 1,024, since the workgroup arrays are the kernel's)
        assert!(text.contains("wg_a[(((lidx + (128u * k))) & 1023u)]"), "idx, used in its own loop body, is inlined\n{text}");
        let c = crate::wgsl::render::cost(&tree);
        assert!(c.per_item < 1000.0, "per item {}", c.per_item);
    }

    /// fit_A2 gen 4's r3p1 (Opus): `let cv0` and `let cv1`, four phi reads
    /// each per pass, used in the 8 x 4 loop inside, are registers set once
    /// a pass; priced at 4,552 per item when they were inlined at every
    /// multiply-add (bound 2,560), below 1,500 now.
    #[test]
    fn g004_r3p1_reads_its_vectors_once_a_pass() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g004_r3p1_opus.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let c = crate::wgsl::render::cost(&tree);
        let bound = crate::wgsl::safe_karva::default_bound(&k).unwrap();
        assert!(c.per_item < 1500.0 && c.per_item < bound.per_item && c.base < bound.base, "per item {}, base {}", c.per_item, c.base);
    }

    /// AN ITEM LOOP UNDER THE THREAD'S GUARD (fit_A2 gen 6's r12p1: the
    /// shipped kernel with its whole body under `if (in range)`, so `for (tt
    /// < cfg.m)` is nested): the binding bound is spelled as the expression
    /// it is, an expression-bounded loop, and the estimator prices that loop
    /// as item work (its trip count grows with m), not m times into the base.
    #[test]
    fn a_binding_bounded_loop_under_a_guard_reads_as_an_expression_bound_priced_per_item() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g006_r12p1_sonnet.wgsl");
        assert!(src.contains("for (var tt: u32 = 0u; tt < cfg.m; tt = tt + 1u)"));
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let sx = tree.to_sexpr();
        assert!(sx.contains("(loop.tilex_C0 (bound.of_index (index.of_count load.cfg_m))"), "{sx}");
        let c = crate::wgsl::render::cost(&tree);
        let shipped = crate::wgsl::render::cost(&crate::wgsl::render::gram_shipped_safe_tree(&k).unwrap());
        assert!(c.per_item < 2.0 * shipped.per_item && c.base < 1000.0, "per item {} (shipped {}), base {}", c.per_item, shipped.per_item, c.base);
    }

    /// A COLUMN FROM A REGISTER DERIVED FROM ANOTHER (fit_A2's r8x_reinvent,
    /// grid 16 x 16, block 8 x 4): i15 = i14 + 4u with i14 = gid.x * 8u + k2
    /// and k2 < 4 is the column (gid.x * 8u + k2) + 4u, inside the thread's 8;
    /// it is proved from k2's bound and stored on the kingdom's own rows. The
    /// same with 5u, which reaches column 8, is not the thread's.
    #[test]
    fn gid_times_k_plus_several_offsets_is_proved_from_the_counters_bounds() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reply_fitA2_g001_r8x_reinvent_opus.wgsl");
        assert!(src.contains("i15 = (i14 + 4u);"));
        let sx = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree.to_sexpr();
        assert!(sx.contains("safe.col_plus_x (safe.thread_col"), "{sx}");
        let past = src.replace("i15 = (i14 + 4u);", "i15 = (i14 + 5u);");
        let Err(e) = read_safe(&past, &k) else { panic!("k2 + 5 < 8 read as the thread's column") };
        assert!(e.contains("not at this thread's cell"), "{e}");
    }

    /// THE FULL WORKGROUP RANGE (2026-10-12; {4, 8, 16, 32} per axis before):
    /// the shipped kernel, one cell per global id, at 1-D and 2-D shapes up to
    /// the M3 Max's 1,024 invocations reads, renders its own
    /// `@workgroup_size`, validates and reads back; a header that names
    /// another workgroup than the kernel declares is refused by name, and a
    /// grid row of 64 x 32 renders as 64 x 16 (the shape the device runs).
    #[test]
    fn workgroup_shapes_over_the_full_legal_range_read_render_and_read_back() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        assert!(SHIPPED.contains("@workgroup_size(16, 16)") || SHIPPED.contains("@workgroup_size(16, 16, 1)"));
        let declared = if SHIPPED.contains("@workgroup_size(16, 16, 1)") { "@workgroup_size(16, 16, 1)" } else { "@workgroup_size(16, 16)" };
        for (x, y) in [(256u32, 1u32), (64, 4), (1024, 1), (1, 1024), (32, 32), (48, 3)] {
            let src = format!("// grid {x} {y} 1 1\n{}", SHIPPED.replace(declared, &format!("@workgroup_size({x}, {y}, 1)")));
            let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{x} x {y}: {e}"));
            let out = rr.kernel(&r.tree).unwrap_or_else(|e| panic!("{x} x {y}: {e}"));
            assert!(out.text.contains(&format!("@workgroup_size({x}, {y}, 1)")) && (out.grid.x, out.grid.y) == (x, y), "{x} x {y}: {}", out.text);
            validate_text(&out.text).unwrap_or_else(|e| panic!("{x} x {y}: {e}"));
            read_safe(&format!("// grid {x} {y} 1 1\n{}", out.text), &k).unwrap_or_else(|e| panic!("{x} x {y} read back: {e}"));
        }
        let mismatch = format!("// grid 64 4 1 1\n{}", SHIPPED.replace(declared, "@workgroup_size(256, 1, 1)"));
        let Err(e) = read_safe(&mismatch, &k) else { panic!("a header of 64 x 4 over @workgroup_size(256, 1, 1) read") };
        assert!(e.contains("is not the kernel's @workgroup_size"), "{e}");
        let ty = |n: &str| crate::wgsl::render::dual_ty(&k, n).unwrap();
        let mut wide = read_safe(&format!("// grid 16 16 1 1\n{SHIPPED}"), &k).unwrap().tree;
        wide.kids[0].kids[0] = Tree::literal("literal.wgdim", ty("u32.wgdim"), 64.0);
        wide.kids[0].kids[1] = Tree::literal("literal.wgdim", ty("u32.wgdim"), 32.0);
        let out = rr.kernel(&wide).unwrap();
        assert_eq!((out.grid.x, out.grid.y), (64, 16));
        validate_text(&out.text).unwrap();
    }

    /// WORKGROUP ARRAYS AS KNOBS (2026-10-12): a kernel's arrays, their
    /// element types and lengths are its own: a vec2 array of 1,056 (masked
    /// by `min`, the length not a power of two) and an f32 array of 256 read,
    /// render at those lengths and read back the same; a declaration over the
    /// device's 32 KB (8,192 vec2) renders halved until the arrays fit.
    #[test]
    fn workgroup_arrays_are_declared_at_the_kernels_own_lengths_within_32_kb() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let src = include_str!("../../tests/corpus/safewgsl/draft_wg_knobs.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let out = rr.kernel(&tree).unwrap();
        assert!(out.text.contains("var<workgroup> wg_v2a: array<vec2<f32>, 1056>;") && out.text.contains("var<workgroup> wg_a: array<f32, 256>;"), "{}", out.text);
        assert!(!out.text.contains("wg_va") && out.text.contains("min((lidx), 1055u)"), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let again = read_safe(&format!("// grid 16 16 1 1\n{}", out.text), &k).unwrap_or_else(|e| panic!("{e}"));
        // Read back (a second reading inlines what the first rendered, so the text
        // need not be the same): the same declarations, valid.
        let back = rr.kernel(&again.tree).unwrap().text;
        assert!(back.contains("var<workgroup> wg_v2a: array<vec2<f32>, 1056>;") && back.contains("var<workgroup> wg_a: array<f32, 256>;"), "{back}");
        validate_text(&back).unwrap_or_else(|e| panic!("{e}\n{back}"));
        // 8,192 vec2 (64 KB) beside 256 f32: the vec2 array halves to 2,048 (16 KB + 1 KB).
        fn set_len(t: &mut Tree, to: f32) {
            if t.name == "wg.decl_v2" {
                t.kids[1].konst = to;
            }
            t.kids.iter_mut().for_each(|c| set_len(c, to));
        }
        let mut big = tree.clone();
        set_len(&mut big, 8192.0);
        let out = rr.kernel(&big).unwrap();
        assert!(out.text.contains("var<workgroup> wg_v2a: array<vec2<f32>, 2048>;"), "{}", out.text);
        assert!(out.text.contains("& 2047u)]"), "the mask follows the halved length: {}", out.text);
        validate_text(&out.text).unwrap();
    }

    /// FULL UNROLLING (2026-10-12): every tile loop of the draft k07 turned
    /// into the unroll row of its signature (one symbol each) renders as
    /// straight-line copies, the counter a literal in each, validates and
    /// reads back; the cost is the tile loop's.
    #[test]
    fn a_tile_loop_unrolled_in_full_renders_straight_line_and_reads_back() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/logs/HGP/001/gen0/k07.wgsl"));
        let Ok(src) = std::fs::read_to_string(dir) else { return };
        let tree = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        fn unroll(t: &mut Tree, n: &mut usize) {
            if let Some(r) = t.name.strip_prefix("loop.tile_") {
                if !r.contains("strided") && t.kids.len() == 2 {
                    t.name = format!("loop.unroll_{r}");
                    *n += 1;
                }
            }
            t.kids.iter_mut().for_each(|c| unroll(c, n));
        }
        let mut unrolled = tree.clone();
        let mut n = 0;
        unroll(&mut unrolled, &mut n);
        assert!(n > 0, "k07 has tile loops");
        let before = rr.kernel(&tree).unwrap();
        let out = rr.kernel(&unrolled).unwrap_or_else(|e| panic!("{e}"));
        assert!(before.text.contains("for (k = 0u;") && !out.text.contains("for (k = 0u;") && out.text.contains("lit_u32(0u)"), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        read_safe(&format!("{}\n{}", out.grid.header(), out.text), &k).unwrap_or_else(|e| panic!("read back: {e}"));
        let (c0, c1) = (crate::wgsl::render::cost(&tree), crate::wgsl::render::cost(&unrolled));
        assert!((c0.per_item - c1.per_item).abs() < 1.0 && (c0.base - c1.base).abs() < 1.0, "{c0:?} vs {c1:?}");
    }

    /// A TREE REDUCTION WRITTEN AS A HALVING LOOP reads as the reduction row
    /// (2026-10-12): `>> 1u` or `/ 2u`, `+` or max, the start's power of two
    /// times two the element count; it renders, validates and reads back, and
    /// a halving loop that is not a reduction is refused by name.
    #[test]
    fn a_halving_loop_reads_as_the_tree_reduction_row() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let src = include_str!("../../tests/corpus/safewgsl/draft_halving_loop.wgsl");
        for (from, to, row) in [("", "", "wg.reduce_add_L0"), ("s = s >> 1u", "s = s / 2u", "wg.reduce_add_L0"), ("part[lidx] + part[lidx + s]", "max(part[lidx], part[lidx + s])", "wg.reduce_max_L0")] {
            let text = if from.is_empty() { src.to_string() } else { src.replacen(from, to, 1) };
            assert!(from.is_empty() || text != src, "{from} not in the draft");
            let tree = read_safe(&text, &k).unwrap_or_else(|e| panic!("{to}: {e}")).tree;
            let sx = tree.to_sexpr();
            assert!(sx.contains(&format!("({row} wg.a (literal.lit4096 256))")), "{to}: {sx}");
            assert_eq!(sx.matches("(loop.").count(), 1, "{to}: the halving loop read as a loop: {sx}");
            let out = rr.kernel(&tree).unwrap();
            validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
            read_safe(&format!("// grid 256 1 1 1\n{}", out.text), &k).unwrap_or_else(|e| panic!("read back: {e}"));
        }
        let Err(e) = read_safe(&src.replacen("if (lidx < s)", "if (lidx <= s)", 1), &k) else { panic!("lidx <= s read as a reduction") };
        assert!(e.contains("halving loop over local.s@") && e.contains("not under `lidx < s`"), "{e}");
    }

    /// TREE REDUCTION IN WORKGROUP MEMORY (2026-10-12): one statement row
    /// (array, n, add/max/min) renders as the leading barrier and log2(n)
    /// halving steps `if (lidx < s) { a[lidx] = a[lidx] + a[lidx + s]; }`,
    /// each followed by a barrier, in uniform control flow (its rows are L
    /// statements only; after a return the renderer masks it as it masks a
    /// barrier); the shipped
    /// kernel with a reduction of 256 f32 renders, validates and reads back,
    /// and the reduction costs its steps.
    #[test]
    fn a_tree_reduction_renders_its_halving_steps_and_reads_back() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let ty = |n: &str| crate::wgsl::render::dual_ty(&k, n).unwrap();
        let mut tree = read_safe(&format!("// grid 16 16 1 1\n{SHIPPED}"), &k).unwrap().tree;
        let reduce = Tree::new("wg.reduce_add_L0", ty("stmt.L0"), vec![Tree::new("wg.a", ty("wgmem<f32>.real"), vec![]), Tree::literal("literal.lit4096", ty("u32.lit4096"), 256.0)]);
        let body = tree.kids[1].clone();
        tree.kids[1] = Tree::new("block.seq2_L0", ty("block.L0"), vec![Tree::new("stmt.block_L0", ty("stmt.L0"), vec![body]), reduce]);
        let out = rr.kernel(&tree).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out.text.matches("workgroupBarrier();").count(), 9, "{}", out.text);
        assert!(out.text.contains("if (lidx < 128u) { wg_a[((lidx) & 2047u)] = (wg_a[((lidx) & 2047u)] + wg_a[((lidx + 128u) & 2047u)]); }"), "{}", out.text);
        assert!(out.text.contains("if (lidx < 1u)"), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        read_safe(&format!("// grid 16 16 1 1\n{}", out.text), &k).unwrap_or_else(|e| panic!("read back: {e}"));
        let plain = read_safe(&format!("// grid 16 16 1 1\n{SHIPPED}"), &k).unwrap().tree;
        assert!(crate::wgsl::render::cost(&tree).base > crate::wgsl::render::cost(&plain).base + 8.0 * 50.0);
        // Only where a barrier stands: every reduction row is an L statement (none under a condition).
        assert!(k.rows.iter().filter(|r| r.name.starts_with("wg.reduce_")).all(|r| k.dual(r.output).is_some_and(|d| d.dual.starts_with("stmt.L"))));
    }

    /// INTEGER ATOMICS IN WORKGROUP MEMORY (2026-10-12): the draft's every
    /// form reads, renders (atomicStore, an atomicAdd whose old value is read
    /// into a register, `_ = atomicOr(..)`, the i32 add, atomicLoad),
    /// validates and reads back. A compare-exchange is refused by name: naga
    /// 0.20's Metal backend does not implement it, so no gene could compile it.
    #[test]
    fn workgroup_atomics_render_validate_and_read_back() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let src = include_str!("../../tests/corpus/safewgsl/draft_atomics.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let out = rr.kernel(&tree).unwrap();
        for want in ["var<workgroup> wg_au_a: array<atomic<u32>, 256>;", "var<workgroup> wg_ai_a: array<atomic<i32>, 64>;", "atomicStore(&wg_au_a[", "= atomicAdd(&wg_au_a[", "_ = atomicOr(&wg_au_a[", "_ = atomicAdd(&wg_ai_a[", "atomicLoad(&wg_au_a["] {
            assert!(out.text.contains(want), "{want}\n{}", out.text);
        }
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        read_safe(&format!("// grid 16 16 1 1\n{}", out.text), &k).unwrap_or_else(|e| panic!("read back: {e}"));
        let cas = src.replacen("    n = atomicLoad(&hits[lidx]) + n - n;", "    n = atomicCompareExchangeWeak(&hits[lidx], n, n + 1u).old_value;", 1);
        assert_ne!(cas, src);
        let Err(e) = read_safe(&cas, &k) else { panic!("a compare-exchange read") };
        assert!(e.contains("atomicCompareExchangeWeak has no safe spelling"), "{e}");
    }

    /// SUBGROUPS (2026-10-12): the built-in values, a reduction, the scans,
    /// the gathers and subgroupBarrier read, render (the values declared on
    /// main only when read), validate and read back.
    #[test]
    fn subgroups_render_validate_and_read_back() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let rr = Renderer::new(&k, &t, &role);
        let src = include_str!("../../tests/corpus/safewgsl/draft_subgroups.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let out = rr.kernel(&tree).unwrap();
        for want in [
            "@builtin(subgroup_invocation_id) sg_lane: u32",
            "@builtin(subgroup_size) sg_size: u32",
            "@builtin(num_subgroups) sg_count: u32",
            "@builtin(subgroup_id) sg_index: u32",
            "subgroupAdd(",
            "subgroupInclusiveAdd(",
            "subgroupExclusiveAdd(",
            "subgroupMax(",
            "subgroupMin(",
            "subgroupBroadcastFirst(",
            "subgroupBroadcast(",
            "subgroupShuffle(",
            "subgroupShuffleXor(",
            "subgroupShuffleDown(",
            "subgroupShuffleUp(",
            "subgroupBarrier();",
        ] {
            assert!(out.text.contains(want), "{want}\n{}", out.text);
        }
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        read_safe(&format!("// grid 256 1 1 1\n{}", out.text), &k).unwrap_or_else(|e| panic!("read back: {e}"));
        // On a 2-D workgroup naga refuses the lane and subgroup-number
        // built-ins: they are computed at entry instead, and the kernel
        // validates and reads back.
        fn widen(t: &mut Tree) {
            if t.name == "grid.of" {
                t.kids[0].konst = 16.0;
                t.kids[1].konst = 16.0;
            }
            t.kids.iter_mut().for_each(widen);
        }
        let mut square = tree.clone();
        widen(&mut square);
        let out2 = rr.kernel(&square).unwrap();
        assert!(out2.text.contains("@workgroup_size(16, 16, 1)"), "{}", out2.text);
        assert!(out2.text.contains("let sg_lane: u32 = subgroupExclusiveAdd(1u);") && out2.text.contains("let sg_index: u32 = lidx / sg_size;"), "{}", out2.text);
        assert!(!out2.text.contains("subgroup_invocation_id") && !out2.text.contains("@builtin(subgroup_id)"), "{}", out2.text);
        validate_text(&out2.text).unwrap_or_else(|e| panic!("{e}\n{}", out2.text));
        read_safe(&format!("// grid 16 16 1 1\n{}", out2.text), &k).unwrap_or_else(|e| panic!("read back 2-D: {e}"));
        // A kernel that reads no subgroup value declares none.
        let plain = rr.kernel(&read_safe(include_str!("../../tests/corpus/safewgsl/draft_atomics.wgsl"), &k).unwrap().tree).unwrap().text;
        assert!(!plain.contains("subgroup"), "{plain}");
    }

    /// A STORE TO H AS AN ADD CHAIN (2026-10-12): `h[c] = h[c] + a + b` is
    /// `h[c] += a; h[c] += b`, two stores in that order; the kernel renders,
    /// validates and reads back. A store whose sum holds no load of its own
    /// cell is still not a read-add-write.
    #[test]
    fn a_store_to_h_as_an_add_chain_is_one_store_per_increment() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let src = include_str!("../../tests/corpus/safewgsl/draft_add_chain.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let sx = tree.to_sexpr();
        assert_eq!(sx.matches("(store.h_").count(), 2, "{sx}");
        let first = sx.find("(store.h_").unwrap();
        assert!(sx[first..].find("arith.mul").unwrap() > sx[first..][1..].find("(store.h_").unwrap(), "f0 is added before f1 * 0.0: {sx}");
        let text = Renderer::new(&k, &t, &role).kernel(&tree).unwrap().text;
        validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        read_safe(&format!("// grid 16 16 1 1\n{text}"), &k).unwrap_or_else(|e| panic!("read back: {e}"));
        let other = src.replacen("= h[gid.y * cfg.d + gid.x] + f0 + f1 * 0.0;", "= h[gid.y * cfg.d + gid.x + 1u] + f0 + f1 * 0.0;", 1);
        assert_ne!(other, src);
        let Err(e) = read_safe(&other, &k) else { panic!("a store of another cell's sum read") };
        assert!(e.contains("not read-add-write of the same cell"), "{e}");
    }

    /// A SUBGROUP OPERATION IN A LOOP'S BOUND (2026-10-12; a single-symbol
    /// mutant wrote `t < subgroupShuffleUp(cfg.m, 2u)`): naga runs it as a
    /// statement ahead of the loop's test; the bound holds it inline, renders,
    /// validates and reads back.
    #[test]
    fn a_subgroup_operation_in_a_loop_bound_reads_inline() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let src = include_str!("../../tests/corpus/safewgsl/draft_subgroups.wgsl").replacen("t < cfg.m;", "t < subgroupMax(cfg.m);", 1);
        assert!(src.contains("subgroupMax(cfg.m)"));
        let tree = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        assert!(tree.to_sexpr().contains("(subgroup.max (index.of_count load.cfg_m))"), "{}", tree.to_sexpr());
        let r = Renderer::new(&k, &t, &role).kernel(&tree).unwrap();
        let text = r.text;
        assert!(text.contains("subgroupMax(cfg.m)"), "{text}");
        validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        let g = r.grid;
        read_safe(&format!("{}\n{text}", g.header()), &k).unwrap_or_else(|e| panic!("read back: {e}"));
    }

    /// THE LAYOUT KNOB (2026-10-12): a kernel with the output's rows on
    /// thread x reads as grid.of_rows_x, with or without the header's
    /// `rows-x` (the store's proof finds the rows on x), to the same tree as
    /// its rows-on-y twin but for the grid row; it renders with the rows on
    /// gid.x and the header `rows-x`, validates and reads back with its rows still on x; the
    /// dispatch swaps its two counts.
    #[test]
    fn the_rows_on_thread_x_read_render_and_read_back() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let src = include_str!("../../tests/corpus/safewgsl/draft_rows_x.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        assert!(tree.to_sexpr().starts_with("(kernel.entry_safe (grid.of_rows_x "), "{}", tree.to_sexpr());
        let untagged = read_safe(&src.replacen(" rows-x\n", "\n", 1), &k).unwrap_or_else(|e| panic!("untagged: {e}")).tree;
        assert_eq!(untagged.to_sexpr(), tree.to_sexpr());
        let twin = src.replacen("// grid 64 4 2 2 rows-x", "// grid 4 64 2 2", 1).replacen("@workgroup_size(64, 4, 1)", "@workgroup_size(4, 64, 1)", 1).replacen("let r0 = gid.x * 2u;", "let r0 = gid.y * 2u;", 1).replacen("let c0 = gid.y * 2u;", "let c0 = gid.x * 2u;", 1);
        let twin = read_safe(&twin, &k).unwrap_or_else(|e| panic!("twin: {e}")).tree;
        let body = |t: &Tree| t.kids[1..].iter().map(Tree::to_sexpr).collect::<Vec<_>>();
        assert_eq!(body(&twin), body(&tree), "the same gene but for its grid");
        let out = Renderer::new(&k, &t, &role).kernel(&tree).unwrap();
        assert!(out.grid.rows_x && out.grid.header() == "// grid 64 4 2 2 rows-x", "{:?}", out.grid);
        assert!(out.text.contains("(gid.x * 2u)") && out.text.contains("(gid.y * 2u)") && out.text.contains("@workgroup_size(64, 4, 1)"), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&format!("{}\n{}", out.grid.header(), out.text), &k).unwrap_or_else(|e| panic!("read back: {e}")).tree;
        assert!(back.to_sexpr().starts_with("(kernel.entry_safe (grid.of_rows_x "), "{}", back.to_sexpr());
        let again = Renderer::new(&k, &t, &role).kernel(&back).unwrap();
        assert!(again.grid == out.grid && again.text.contains("(gid.x * 2u)"), "{}", again.text);
        // 1,000 rows of 300 columns: rows over x (64 threads x 2), columns over y (4 x 2).
        assert_eq!(out.grid.dispatch(1000, 300), (8, 38));
        assert_eq!(Renderer::new(&k, &t, &role).kernel(&twin).unwrap().grid.dispatch(1000, 300), (38, 8));
    }

    /// A kernel with a helper function reads: the helper is inlined at its
    /// calls (the complete kingdom, step 6; `load4`, with a local, an
    /// if-else and lane stores, as statements since run A). The blocked
    /// kernel's own `if (lr >= cfg.r_rows) { continue; }`, in its epilogue
    /// loop, is the epilogue's continue (block.cont_O1).
    #[test]
    fn a_kernel_with_a_helper_function_is_inlined_or_named() {
        let k = WgslKingdom::load_safe();
        let r = read_safe(BLOCKED, &k).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.tree.to_sexpr().contains("(block.cont_O1 "), "{}", r.tree.to_sexpr());
        let (t, role) = (FunctionTable::safe(), gram_role());
        let text = Renderer::new(&k, &t, &role).kernel(&r.tree).unwrap().text;
        validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert!(text.contains(") >= cfg.r_rows)) { continue; }"), "{text}");
    }

    /// A LOOP THAT BREAKS (2026-10-12, WGSL_COVERAGE.md gap 7): a thread that
    /// breaks skips every barrier the others reach in later passes, so a
    /// loop that breaks holds no barrier: draft_control_flow's breaking loop
    /// reads as loop.tile_brk with block.brk_B<d>, renders the guard and
    /// validates; the same loop with a barrier in it is refused by name.
    #[test]
    fn a_loop_that_breaks_reads_and_holds_no_barrier() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let src = include_str!("../../tests/corpus/safewgsl/draft_control_flow.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let sx = tree.to_sexpr();
        assert!(sx.contains("(loop.tile_brk_L0 ") && sx.contains("(block.brk_B2 "), "{sx}");
        let text = Renderer::new(&k, &t, &role).kernel(&tree).unwrap().text;
        validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert!(text.contains("{ break; }"), "{text}");
        let barrier = src.replacen("{ break; } g = g + 1.0; }", "{ break; } g = g + 1.0; workgroupBarrier(); }", 1);
        assert_ne!(barrier, src);
        let Err(e) = read_safe(&barrier, &k) else { panic!("a barrier in a loop that breaks read") };
        assert!(e.contains("a barrier under a condition"), "{e}");
    }

    /// A CONTINUE IN A LOOP (2026-10-12, WGSL_COVERAGE.md gap 7): the guard,
    /// then the rest of the loop body at the conditional rung (block.cont_L1
    /// in the item loop), so no barrier follows it; every thread still
    /// reaches the next iteration. The blocked kernel with a continue in its
    /// item loop reads, renders and validates; the shipped kernel with a
    /// barrier after a continue is refused by name (a barrier some threads
    /// skip).
    #[test]
    fn a_continue_in_a_loop_reads_unless_a_barrier_follows_it() {
        let k = WgslKingdom::load_safe();
        let moved = BLOCKED.replacen("        let rv = load4(", "        if (t == 7u) { continue; }\n        let rv = load4(", 1);
        assert_ne!(moved, BLOCKED);
        let tree = read_safe(&moved, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        assert!(tree.to_sexpr().contains("(block.cont_L1 "), "{}", tree.to_sexpr());
        let (t, role) = (FunctionTable::safe(), gram_role());
        let text = Renderer::new(&k, &t, &role).kernel(&tree).unwrap().text;
        validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert!(text.contains("== 7u)) { continue; }"), "{text}");
        let barrier = format!("// grid 16 16 1 1\n{SHIPPED}").replacen("    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {\n", "    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {\n        if (t == 7u) { continue; }\n        workgroupBarrier();\n", 1);
        assert!(barrier.contains("workgroupBarrier"));
        let Err(e) = read_safe(&barrier, &k) else { panic!("a barrier after a continue read") };
        assert!(e.contains("where only conditional statements are spelled"), "{e}");
    }
}
