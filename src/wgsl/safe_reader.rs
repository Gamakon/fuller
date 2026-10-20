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

use std::collections::{BTreeMap, BTreeSet, HashMap};

use naga::{Expression, Function, Handle, Statement};

use crate::geneframe::Ty;
use crate::wgsl::infer::infer_function;
use crate::wgsl::loader::WgslKingdom;
use crate::wgsl::reader::{read, Node, Root, RootKind};
use crate::wgsl::render::{dual_ty, Tree, COMPOSITES};

/// The conversion's result: the safe tree and what was noted on the way.
pub struct SafeRead {
    /// The kingdom the tree's duals are in when the module declared aggregate types
    /// ([`WgslKingdom::extended`]); `None`: the caller's own.
    pub kingdom: Option<WgslKingdom>,
    pub tree: Tree,
    pub notes: Vec<String>,
    pub registers: BTreeMap<String, String>,
    /// The interface the kernel declared (`// interface <name>`), else the one its module gives (`Role::inferred`).
    pub role: crate::wgsl::render::Role,
}

struct Ctx<'a> {
    kingdom: &'a WgslKingdom,
    /// The interface's per-kernel rows (a second input's load among them).
    per_kernel_rows: Vec<crate::wgsl::safe_karva::PerKernelRow>,
    /// The interface's uniform variable (its fields' rows are
    /// `load.<var>_<field>`), and the lanes of a vector block (`x`..`w`).
    uniform_var: String,
    uniform_lanes: Vec<String>,
    /// Whether main's last statement is a `return` (a body that returns, the
    /// epilogue not written after it).
    main_returns: bool,
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
    /// Stores to a loop's counter inside its body (not its step), as
    /// (counter, value): the loop that owns the counter reads them.
    counter_stores: Vec<CounterStore>,
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
    /// The rows of the workgroup arrays this entry point uses (a module's
    /// entry points share `arrays`; each declares its own).
    arrays_used: BTreeSet<&'static str>,
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
    /// Of those, the lets bound because a store changes a local they read
    /// before a use: one with no register kind has no spelling.
    stale_lets: BTreeSet<String>,
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
    /// A loop counter's EXPRESSION entry value (a grid-stride loop,
    /// `for (var i = lid.x; i < n; i += s)`): its reads are `e + k`.
    counter_exprs: HashMap<String, Tree>,
    /// A statement-inlined helper's pointer parameters: `arg.p` to the
    /// caller's register for the local `&x` it was given (row, kind).
    pointer_args: HashMap<String, (String, char)>,
    /// A pointer parameter to a local array, given `&a`: the caller's array
    /// (`arg.p` is `local.a@0`), read and written as the caller's own.
    array_args: HashMap<String, String>,
    /// While a helper body is being inlined: its arguments by `arg.<name>`,
    /// already resolved and typed in the caller.
    inline_args: Option<HashMap<String, Tree>>,
    /// The path of the return that ends a branch read as `if (c) { .. }
    /// else { rest }` ([`Ctx::early_return_branch`]), while its branch is read.
    early_return_at: Option<Vec<usize>>,
    /// How many early-return branches the walk is inside (their bodies and
    /// the rest of the block as their else): a lone `return` there ends a
    /// branch, never the thread guard.
    branch_depth: usize,
    /// Statements the walk passes over (by path): a loop's leading break
    /// guard folded into its bound.
    skip_stmts: BTreeSet<Vec<usize>>,
    /// Nodes the reader has built (trees read, expansions for the ownership
    /// proof): past [`READ_NODE_BOUND`] the read stops, named, instead of
    /// running on (run 1's 03:15Z hang, 2026-10-17: a helper argument
    /// expanded into itself without end).
    built: std::cell::Cell<usize>,
    /// The arguments of the helpers being inlined around the one in
    /// `inline_args`, innermost last (a helper inlined inside another).
    outer_args: Vec<Option<HashMap<String, Tree>>>,
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
    /// The text says it writes the whole H (a mirror store reads as one).
    mode_full: bool,
    /// An index read while the layout was open that is a thread's row or
    /// column under one of the layouts ([`Ctx::thread_index`]): the entry
    /// point is read again under the layout its stores decide.
    open_thread_index: bool,
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
    /// A kingdom helper written verbatim (`render::COMPOSITE_ROWS`): its row,
    /// a call read back as it.
    pub row: Option<&'static str>,
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
    /// Its lets set in a register where they are defined (read after a store
    /// or barrier that changes what they read, or in a deeper loop), as main's.
    pub bound_lets: HashMap<String, Vec<usize>>,
    pub stale_lets: BTreeSet<String>,
}

/// The register dual of a naga type: the four kinds a register holds.
fn dual_of_type(module: &naga::Module, ty: Handle<naga::Type>) -> Option<String> {
    use naga::{ScalarKind, TypeInner, VectorSize};
    match &module.types[ty].inner {
        TypeInner::Scalar(sc) => match (sc.kind, sc.width) {
            (ScalarKind::Float, 4) => Some("f32.real".into()),
            (ScalarKind::Uint, 4) => Some("u32.index".into()),
            (ScalarKind::Bool, _) => Some("bool.flag".into()),
            (ScalarKind::Sint, 4) => Some("i32.int".into()),
            (ScalarKind::Float, 2) => Some("f16.real".into()),
            _ => None,
        },
        TypeInner::Vector { size: VectorSize::Quad, scalar } if scalar.kind == ScalarKind::Float && scalar.width == 4 => Some("vec4<f32>.real".into()),
        // (Two sums returned together, `vec2<f32>(s, n)`: v24 run 1's seed11.)
        TypeInner::Vector { size: VectorSize::Bi, scalar } if scalar.kind == ScalarKind::Float && scalar.width == 4 => Some("vec2<f32>.real".into()),
        // The other vectors a register holds (EXTRA_REGISTERS), and the square matrices.
        TypeInner::Vector { .. } | TypeInner::Matrix { .. } => {
            // (A register of any other type is a kind the module's extended kingdom adds: `register` finds it.)
            crate::wgsl::render::elem_dual(&crate::wgsl::interface::type_text(module, ty))
        }
        // An array of a known size or a struct the module declares is a value ([`WgslKingdom::extended`]).
        TypeInner::Array { size: naga::ArraySize::Constant(_), .. } | TypeInner::Struct { .. } if crate::wgsl::aggregate::is_value(module, ty, 0) && !crate::wgsl::interface::type_text(module, ty).is_empty() => Some(format!("{}.value", crate::wgsl::interface::type_text(module, ty))),
        _ => None,
    }
}

/// The kingdom's dual of any naga type a binding's field may hold (scalars,
/// vectors, float matrices, atomics), by its WGSL spelling.
fn field_dual(module: &naga::Module, ty: Handle<naga::Type>) -> Option<String> {
    let text = crate::wgsl::interface::type_text(module, ty);
    match text.as_str() {
        "atomic<u32>" => Some("u32.index".into()),
        "atomic<i32>" => Some("i32.int".into()),
        other => crate::wgsl::render::elem_dual(other),
    }
}

/// The type a buffer place has: the binding `binding`'s type, then each
/// segment of its path (`#` an array's element, else a struct member by name).
fn place_type(module: &naga::Module, binding: &str, segs: &[&str]) -> Option<Handle<naga::Type>> {
    let mut ty = module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(binding))?.1.ty;
    for seg in segs {
        ty = match (&module.types[ty].inner, *seg) {
            (naga::TypeInner::Array { base, .. }, "#") => *base,
            (naga::TypeInner::Struct { members, .. }, name) => members.iter().find(|m| m.name.as_deref() == Some(name))?.ty,
            _ => return None,
        };
    }
    Some(ty)
}

/// A place's name with a member appended before the lineage mark (`load.buffer.args.#@v0`
/// and `x` give `load.buffer.args.#.x@v0`).
fn member_place(place: &str, member: &str) -> String {
    match place.find('@') {
        Some(i) => format!("{}.{member}{}", &place[..i], &place[i..]),
        None => format!("{place}.{member}"),
    }
}

/// What a struct written whole is made of.
enum StructSource {
    /// The operands of its constructor, in member order.
    Compose(Vec<Node>),
    /// The struct place it is loaded from: the load's name and index operands.
    Place(String, Vec<Node>),
}

fn ty(ctx: &Ctx, name: &str) -> Result<Ty, String> {
    dual_ty(ctx.kingdom, name)
}

/// The dual of a register leaf of this kind letter: the shipped kinds', or the input of the module's own `reg.get_<kind>`.
fn reg_ty(ctx: &Ctx, kind: char) -> Result<Ty, String> {
    if kind.is_ascii_uppercase() && !crate::wgsl::render::EXTRA_REGISTERS.iter().any(|x| x.letter == kind) {
        let get = format!("reg.get_{kind}");
        return ctx.kingdom.rows.iter().find(|r| r.name == get).and_then(|r| r.inputs.first().copied()).ok_or_else(|| format!("no register kind {kind} in the module's kingdom"));
    }
    ty(ctx, Ctx::reg_dual(kind))
}

use crate::wgsl::aggregate::module_register_of_value;

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

/// The most nodes one reading of a kernel builds (trees and expansions),
/// far above any kernel the corpora hold (every corpus test reads under it).
pub const READ_NODE_BOUND: usize = 20_000_000;

fn tree_of(ctx: &mut Ctx, n: &Node) -> Result<Tree, String> {
    ctx.built.set(ctx.built.get() + 1);
    if ctx.built.get() > READ_NODE_BOUND {
        return Err(format!("our software has a bug: reading this kernel built more than {READ_NODE_BOUND} nodes (an expansion without end); the read is stopped"));
    }
    let text = n.to_sexpr();
    match n {
        Node::Leaf { name, .. } if name.starts_with("tex.") => ctx.texture_row(name, &[]),
        Node::Leaf { name, .. } => {
            let name = &prefixed(&ctx.local_prefix, name);
            if let Some(def) = name.strip_prefix("let.").and_then(|k| ctx.lets.get(k)) {
                let def = def.clone();
                // The kingdom's subgroup lane and index on a 2-D workgroup
                // (`let sg_lane: u32 = subgroupExclusiveAdd(1u);`, `let
                // sg_index: u32 = lidx / sg_size;`): the builtins they stand
                // for (2026-10-13, AST to AST: read as their definitions).
                let base = name.strip_prefix("let.").and_then(|k| k.split('@').next()).unwrap_or("");
                let builtin = match (base, &def) {
                    ("sg_lane", Node::App { name: d, .. }) if d.contains("Subgroup") || d.starts_with("subgroup.") => Some("builtin.subgroup_invocation_id"),
                    ("sg_index", Node::App { name: d, .. }) if d.contains("Divide") || d.ends_with(".div") => Some("builtin.subgroup_id"),
                    _ => None,
                };
                if let Some(row) = builtin.and_then(|b| ctx.kingdom.rows.iter().find(|r| r.name == b)) {
                    return Ok(Tree::new(&row.name, row.output, vec![]));
                }
                // A let bound to a register at its definition reads the register.
                // (spelled as a load of the local `let.<name>`, which the value
                // step reads as the register that local holds).
                if let Some(k) = name.strip_prefix("let.").filter(|k| ctx.let_regs.contains_key(*k)) {
                    // (A helper's own, named under it.)
                    let local = format!("load.local.{}let.{k}", ctx.local_prefix);
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
                    Some('w') => "vec2<f32>.real",
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
                if let Some(row) = helper.row {
                    let bound = ctx.call_arguments(callee, helper, idx)?;
                    let args = helper.args.iter().map(|a| bound.get(a).cloned().ok_or_else(|| format!("{callee}: no argument {a}"))).collect::<Result<Vec<_>, _>>()?;
                    let t = ctx.duals.get(&text).copied().flatten().or_else(|| ctx.kingdom.rows.iter().find(|r| r.name == row && r.inputs.len() == args.len() && r.inputs.iter().zip(&args).all(|(i, a)| *i == a.ty)).map(|r| r.output)).ok_or_else(|| format!("a call of {callee}: untyped"))?;
                    return Ok(Tree::new(row, t, args));
                }
                if !helper.pure && !helper.composite {
                    let d = helper.ret_dual.as_deref().ok_or_else(|| format!("{callee} returns a value no register holds"))?;
                    // (Named under the helper being inlined, as its call's register is.)
                    return Ok(Tree::new(&format!("load.local.call.{}{callee}@{idx}", ctx.local_prefix), ty(ctx, d)?, vec![]));
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
                // Its arguments are the caller's trees, substituted whole (a
                // statement-inlined caller's own arguments in them stay its
                // leaves, read in its scope): not the caller's argument values
                // (`do_row` calling the pure `safe_tanh(x)`, k1119 of the saved
                // model kernels: `arg.x` looked up among do_row's, 2026-10-17).
                let saved_values = ctx.inline_values.take();
                let body = helper.body.clone().ok_or_else(|| format!("{callee} returns nothing"))?;
                let out = tree_of(ctx, &body);
                ctx.inline_values = saved_values;
                ctx.inline_args = saved_args;
                ctx.lets = saved_lets;
                return out.map_err(|e| format!("inlining {callee}: {e}"));
            }
            let t = ctx.duals.get(&text).copied().flatten().ok_or_else(|| format!("untyped leaf {name}"))?;
            Ok(Tree::new(name, t, vec![]))
        }
        Node::Literal { form, value, slot } => {
            // A literal's value is carried as an f64: exact for every u32 and i32 (and every f32), so a hash's
            // multiplier or key (CubeCL's 2246822507u), a mask (0xFFFFFFFFu) or i32::MIN reads as the value it is.
            let t = match ctx.duals.get(&text).copied().flatten() {
                Some(t) => t,
                None => {
                    // Two forms in the inference (an index here, a count there): the form's own scalar dual.
                    let dual = match form.as_str() { "real" => "f32.real", "int" => "i32.int", "flag" => "bool.flag", "count" => "u32.count", "bits" => "u32.bits", _ => "u32.index" };
                    ty(ctx, dual)?
                }
            };
            // (The zero of an aggregate type, `S()`, is the dual's terminal `agg.zero`.)
            if *value == 0.0 {
                if let Some(d) = ctx.kingdom.duals.iter().find(|d| d.slot == *slot && ctx.kingdom.shapes.contains_key(&d.index)) {
                    return Ok(Tree::new("agg.zero", d.ty(), vec![]));
                }
            }
            // (A real literal is an f32's value; an integer one is exact as an f64.)
            Ok(Tree::literal(&format!("literal.{form}"), t, if form == "real" { f64::from(*value as f32) } else { *value }))
        }
        Node::App { name, kids, .. } if name.starts_with("tex.") => ctx.texture_row(name, kids),
        Node::App { name, kids, known, .. } => {
            // Two indices in a row (`sa[si][comp]`, seed08 of run A): the
            // element's index then the lane's; the store and load spell it.
            if name == "naga.NestedAccess" && kids.len() == 2 {
                let kids = kids.iter().map(|k| tree_of(ctx, k)).collect::<Result<Vec<_>, _>>()?;
                return Ok(Tree::new("naga.NestedAccess", ty(ctx, "u32.index")?, kids));
            }
            // A member of a struct read from a storage binding (`s.joints` for
            // `let s = skin[i]`): the field's load.
            if let Some(t) = ctx.struct_field_load(n, &text)? {
                return Ok(t);
            }
            // An element of an array VALUE chosen at run time (`let accs =
            // array<vec4<f32>, 4>(...); accs[i]`, r3x_reinvent): a select
            // chain over the elements on the index.
            if (name == "shape.access_dyn" || name == "agg.at") && kids.len() == 2 {
                // The array is usually a `let`: its definition.
                let base = match &kids[0] {
                    Node::Leaf { name, .. } => name.strip_prefix("let.").and_then(|k| ctx.lets.get(k)).unwrap_or(&kids[0]),
                    other => other,
                };
                if let Node::App { name: cn, kids: elems, .. } = base {
                    if cn.starts_with("naga.Compose.type") && !elems.is_empty() {
                        let elems = elems.iter().map(|e| tree_of(ctx, e)).collect::<Result<Vec<_>, _>>()?;
                        // (A constant index picks the element.)
                        if let Some(i) = (if let Node::Literal { value, .. } = &kids[1] { Some(*value) } else { None }).filter(|l| *l >= 0.0 && l.fract() == 0.0).map(|l| l as usize).filter(|i| *i < elems.len()) {
                            return Ok(elems[i].clone());
                        }
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
                    // The input and the output the kingdom's rows name (phi, h); any
                    // other runtime-sized array by the interface's row `len.<name>`.
                    let row = match g.as_str() {
                        "global.phi" => "load.phi_len".to_string(),
                        "global.h" => "load.h_len".to_string(),
                        other => {
                            let row = format!("len.{}", other.strip_prefix("global.").unwrap_or(other));
                            if !ctx.per_kernel_rows.iter().any(|p| p.0 == row) {
                                return Err(format!("arrayLength of {other}: the interface has no runtime-sized array of that name (row {row})"));
                            }
                            row
                        }
                    };
                    return Ok(Tree::new(&row, ty(ctx, "u32.count")?, vec![]));
                }
                if let Some(row) = safe_row_for(name, kids) {
                    return tree_of(ctx, &Node::App { name: row, slot: slot_of(n).to_string(), known: true, kids: kids.clone() });
                }
                // A member of a struct VALUE the module declares (a helper's struct argument, a struct a helper returns): the
                // aggregate's row `agg.member_<m>` over the struct's value (docs/PLAN_wgsl_terminals.md, "Aggregates").
                if let (Some(i), [k]) = (name.strip_prefix("naga.AccessIndex.").and_then(|i| i.parse::<usize>().ok()), kids.as_slice()) {
                    let member = ctx.kingdom.duals.iter().position(|d| d.form == "value" && d.slot == slot_of(k)).and_then(|d| ctx.kingdom.shapes.get(&(d as u16))).and_then(|sh| if let crate::wgsl::loader::Shape::Struct { members } = sh { members.get(i).map(|(m, _)| m.clone()) } else { None });
                    if let Some(m) = member {
                        return tree_of(ctx, &Node::App { name: format!("agg.member_{m}"), slot: slot_of(n).to_string(), known: true, kids: kids.clone() });
                    }
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
            let mut kids = kids.iter().map(|k| tree_of(ctx, k)).collect::<Result<Vec<_>, _>>()?;
            // `x / max(b, 1u)`, `x % max(b, 1u)`: how the kingdom writes a
            // divisor that is not a literal (no zero divisor, 2026-10-12;
            // the remainder 2026-10-13), read back as x / b, x % b.
            // (A vector's divisor: `max(b, vecN<u32>(1u))`, the 1 splatted;
            // 2026-10-13, AST to AST: 35 of 1,000 random trees.)
            let one = |t: &Tree| literal_value(t) == Some(1.0) || (t.name.starts_with("shape.splat") && t.kids.len() == 1 && literal_value(&t.kids[0]) == Some(1.0));
            if (name == "index.div" || name == "index.mod") && kids.len() == 2 && kids[1].name == "index.max" && kids[1].kids.len() == 2 && one(&kids[1].kids[1]) {
                let b = kids[1].kids[0].clone();
                kids[1] = b;
            }
            // The kingdom's floored difference over pass-through literals
            // (`select((lit_u32(23u)) - (lit_u32(4u)), 0u, (lit_u32(4u)) > ...)`):
            // each `lit_u32` is its own call, so the node pattern (floor_differences)
            // cannot see the sides are equal; the trees can.
            if name == "select.scalar_cond" && kids.len() == 3 && literal_value(&kids[0]) == Some(0.0) && kids[1].name == "index.sub_floor" && kids[1].kids.len() == 2 && kids[2].kids.len() == 2 {
                let (dk, ck) = (&kids[1].kids, &kids[2].kids);
                let same = |a: &Tree, b: &Tree| a == b || (literal_value(a).is_some() && literal_value(a) == literal_value(b));
                if (kids[2].name == "compare.gt" && same(&ck[0], &dk[1]) && same(&ck[1], &dk[0])) || (kids[2].name == "compare.lt" && same(&ck[0], &dk[0]) && same(&ck[1], &dk[1])) {
                    return Ok(kids[1].clone());
                }
            }
            // `x << (s & 31u)` (a vector's count splatted, `vecN<u32>((s) & 31u)`):
            // how the kingdom writes a shift's count, masked below the width;
            // read back as the count (2026-10-12: 36 of 1,000 random trees).
            if (name.ends_with(".shl") || name.ends_with(".shr")) && kids.len() == 2 {
                let strip = |t: &Tree| -> Option<Tree> {
                    match (t.name.as_str(), t.kids.as_slice()) {
                        ("bits.and" | "index.and", [x, m]) if m.name.starts_with("literal.") && m.konst == 31.0 => Some(x.clone()),
                        _ => None,
                    }
                };
                // (A vector's count is splatted by the renderer; the row's
                // count is the scalar.)
                let masked = |t: &Tree| -> Option<Tree> {
                    if t.name.starts_with("shape.splat") && t.kids.len() == 1 {
                        // naga folds a literal count's mask (`(40u) & 31u` is 8u).
                        let x = &t.kids[0];
                        return strip(x).or_else(|| literal_value(x).map(|_| x.clone()));
                    }
                    strip(t)
                };
                if let Some(x) = masked(&kids[1]) {
                    kids[1] = x;
                }
            }
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
pub(crate) fn component(v: &Tree, lane: usize, kingdom: &WgslKingdom) -> Option<Tree> {
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

/// One step of a workgroup reduction as the emitter unrolls it, read back:
/// `if (lidx < s) { a[lidx] = op(a[lidx], a[lidx + s]); }` as
/// (the array, the op, s).
fn reduction_step(t: &Tree) -> Option<(Tree, &'static str, f32, Option<Tree>)> {
    let [c, b] = t.kids.as_slice() else { return None };
    if rung_stem(&t.name) != Some("stmt.if") || c.name != "compare.lt" || rung_stem(&b.name) != Some("block.one") {
        return None;
    }
    let [lidx, s] = c.kids.as_slice() else { return None };
    // (Read before conform, the index is the builtin itself; after, its conversion.)
    let s = literal_value(s).filter(|_| matches!(lidx.name.as_str(), "builtin.local_invocation_index" | "index.of_lidx"))?;
    let set = b.kids.first()?;
    if let Some((a, op, i)) = argmax_step(set, lidx, &c.kids[1]) {
        return Some((a, op, s, Some(i)));
    }
    let [arr, idx, val] = set.kids.as_slice() else { return None };
    // (Compared by their rows: read before conform, a node's dual may not
    // be its final one yet.)
    let sx = |t: &Tree| t.to_sexpr();
    if !set.name.starts_with("wg.set") || !arr.name.starts_with("wg.") || !arr.kids.is_empty() || idx.kids.first().map(sx) != Some(sx(lidx)) {
        return None;
    }
    let op = ["add", "min", "max"].into_iter().find(|op| val.name == format!("arith.{op}"))?;
    let [x, y] = val.kids.as_slice() else { return None };
    let at = |g: &Tree, i: String| g.name.starts_with("wg.get") && g.kids.first().map(sx) == Some(sx(arr)) && g.kids.get(1).map(sx) == Some(i);
    let plus = format!("({} (index.add {} {}))", idx.name, sx(lidx), sx(&c.kids[1]));
    (at(x, sx(idx)) && at(y, plus)).then(|| (arr.clone(), op, s, None))
}

/// A workgroup array row's element: f32, u32, vec2 or vec4.
fn wg_elem(which: &str) -> &'static str {
    if which.starts_with("wg.au") || which.starts_with("wg.ai") {
        "atomic"
    } else if which.starts_with("wg.v2") {
        "vec2"
    } else if which.starts_with("wg.u") {
        "u32"
    } else if which.starts_with("wg.v") {
        "vec4"
    } else {
        "f32"
    }
}

/// An argmax (argmin) reduction's step body: `if (v[lidx + s] > v[lidx]) {
/// v[lidx] = v[lidx + s]; i[lidx] = i[lidx + s]; }` (`<` for argmin), the
/// two arrays and the operation; None for any other statement.
fn argmax_step(t: &Tree, lidx: &Tree, s: &Tree) -> Option<(Tree, &'static str, Tree)> {
    let [c, b] = t.kids.as_slice() else { return None };
    if rung_stem(&t.name) != Some("stmt.if") {
        return None;
    }
    let op = match c.name.as_str() {
        "compare.gt" => "argmax",
        "compare.lt" => "argmin",
        _ => return None,
    };
    let sx = |t: &Tree| t.to_sexpr();
    let mut sets = Vec::new();
    flatten_stmts(b.clone(), &mut sets);
    let [sv, si] = sets.as_slice() else { return None };
    let get = |g: &Tree, arr: &Tree, plus: bool| -> bool {
        let Some(idx) = g.kids.get(1) else { return false };
        let want = if plus { format!("({} (index.add {} {}))", idx.name, sx(lidx), sx(s)) } else { format!("({} {})", idx.name, sx(lidx)) };
        g.name.starts_with("wg.get") && g.kids.first().map(sx) == Some(sx(arr)) && sx(idx) == want
    };
    let set_ok = |set: &Tree| -> Option<Tree> {
        let [arr, idx, val] = set.kids.as_slice() else { return None };
        (set.name.starts_with("wg.set") && arr.name.starts_with("wg.") && arr.kids.is_empty() && idx.kids.first().map(sx) == Some(sx(lidx)) && get(val, arr, true)).then(|| arr.clone())
    };
    let (v, i) = (set_ok(sv)?, set_ok(si)?);
    let [x, y] = c.kids.as_slice() else { return None };
    (get(x, &v, true) && get(y, &v, false)).then_some((v, op, i))
}

/// A store to a loop's counter inside its body: (counter, value).
type CounterStore = (String, Tree);

/// The unrolled item loop's main loop, read, before the loop after it
/// that finishes the items joins it into loop.items_unrolled_<bound>.
const UNROLLED_MAIN: &str = "read.items_unrolled_main";

fn local_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("load.").or_else(|| name.strip_prefix("store."))?;
    let rest = rest.strip_prefix("local.")?;
    Some(format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@")))
}

/// The index dual of a private array's mask row: a 16-element array's
/// (safe.pidx16) is u32.pindex16, a 64-element array's u32.pindex.
fn pindex_dual(mask: &str) -> &'static str {
    match mask {
        "safe.pidx16" => "u32.pindex16",
        "safe.pidx128" => "u32.pindex128",
        _ => "u32.pindex",
    }
}

/// The largest index a private array's mask row writes (`i & mask`).
fn pindex_mask(mask: &str) -> f32 {
    match mask {
        "safe.pidx16" => 15.0,
        "safe.pidx128" => 127.0,
        _ => 63.0,
    }
}

/// The private array dual of a kind letter.
fn parr_dual_of(arr: &str, kind: char) -> &'static str {
    // (The 64-element u32 arrays, the 128-element ones and the 64-element vec4
    // arrays: their rows name their dual.)
    if let Some(d) = crate::wgsl::render::array_dual_of_row(arr).filter(|d| d.starts_with("parr64<") || d.starts_with("parr128<")) {
        return d;
    }
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

/// A place in a storage binding that the interface's rows spell by name: the
/// binding, the path after its element index (members by name, `#` an index)
/// and the number of indexes the place takes. `<binding>.#.<path>` over an
/// array of structs, `<binding>.<path>` over a struct, and the whole of a
/// binding that is one scalar, vector or matrix (an empty path). None for a
/// place the older rows spell: an array's element or lane.
struct BufferPlace {
    binding: String,
    path: String,
    indexes: usize,
}

fn buffer_place(module: &naga::Module, rest: &str) -> Option<BufferPlace> {
    // (A load carries its version after the path: `.value.#@v0`.)
    let rest = rest.split('@').next().unwrap_or(rest);
    let mut it = rest.split('.');
    let binding = it.next()?;
    let mut segs: Vec<&str> = it.collect();
    let array = matches!(module.types[place_type(module, binding, &[])?].inner, naga::TypeInner::Array { .. });
    let mut indexes = 0;
    if array {
        if segs.first() != Some(&"#") {
            return None;
        }
        segs.remove(0);
        indexes = 1;
    }
    indexes += segs.iter().filter(|s| **s == "#").count();
    if segs.iter().all(|s| *s == "#") && (array || !segs.is_empty()) {
        return None;
    }
    Some(BufferPlace { binding: binding.to_string(), path: segs.join("."), indexes })
}

/// The index operands of a place: one operand, or the nest of several
/// (`naga.NestedAccess`, outer first).
fn flat_indexes(t: &Tree, out: &mut Vec<Tree>) {
    if t.name == "naga.NestedAccess" && t.kids.len() == 2 {
        flat_indexes(&t.kids[0], out);
        flat_indexes(&t.kids[1], out);
    } else {
        out.push(t.clone());
    }
}

fn literal_value(t: &Tree) -> Option<f32> {
    t.name.starts_with("literal.").then_some(t.konst as f32)
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
/// Whether a slot is a struct's: `struct`, or the name of a declared struct (the slots of a value are scalars, vectors,
/// matrices and the `<>` types; an image, a sampler and the like are named by their first word).
fn is_struct_slot(slot: &str) -> bool {
    slot == "struct" || (!slot.contains('<') && !matches!(slot, "f32" | "f16" | "i32" | "u32" | "bool" | "store" | "image" | "sampler" | "bindingarray" | "accelerationstructure" | "rayquery" | "cooperativematrix" | "valuepointer" | "pointer"))
}

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

/// Whether statements hold a break of their own loop: a break anywhere in
/// them, in blocks and either branch of an if, not one inside a nested loop
/// or switch (which ends that statement, not this loop).
fn has_own_break(stmts: &[Statement]) -> bool {
    stmts.iter().any(|s| match s {
        Statement::Break => true,
        Statement::Block(b) => has_own_break(b),
        Statement::If { accept, reject, .. } => has_own_break(accept) || has_own_break(reject),
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

/// A name of the base reader with the interface's buffers, uniform and
/// fields as the kingdom's canonical roles: `load.buffer.<input>` is
/// `load.buffer.phi`, `<output>` is `h`, `load.uniform.<var>.<field>` is
/// `load.uniform.cfg.<role>`, `global.<input>` is `global.phi`.
fn canonical_name(name: &str, role: &crate::wgsl::render::Role) -> String {
    let swap = |name: &str, from: &str, to: &str| -> Option<String> {
        let rest = name.strip_prefix(from)?;
        (rest.is_empty() || rest.starts_with(['.', '@'])).then(|| format!("{to}{rest}"))
    };
    let prefixes = ["load.buffer.", "store.buffer.", "buffer.", "global."];
    for p in prefixes {
        if let Some(n) = swap(name, &format!("{p}{}", role.input), &format!("{p}phi")) {
            return n;
        }
        if let Some(n) = swap(name, &format!("{p}{}", role.output), &format!("{p}h")) {
            return n;
        }
    }
    if let Some(rest) = name.strip_prefix(&format!("load.uniform.{}.", role.uniform_var)) {
        let (field, tail) = rest.split_at(rest.find(['.', '@']).unwrap_or(rest.len()));
        let canon = role.fields.iter().find(|f| f.name == field && !f.role.is_empty()).map_or(field, |f| f.role.as_str());
        return format!("load.uniform.cfg.{canon}{tail}");
    }
    name.to_string()
}

/// [`canonical_name`] over a whole node.
fn canonical_node(n: &Node, role: &crate::wgsl::render::Role) -> Node {
    match n {
        Node::Leaf { name, slot } => Node::Leaf { name: canonical_name(name, role), slot: slot.clone() },
        Node::Literal { .. } => n.clone(),
        Node::App { name, slot, known, kids } => Node::App { name: canonical_name(name, role), slot: slot.clone(), known: *known, kids: kids.iter().map(|k| canonical_node(k, role)).collect() },
    }
}

/// Every raw u32 difference `x - y` as the kingdom's floored difference, and a
/// left shift by a literal k as a product by 2^k,
/// (`index.sub_floor`, 2026-10-12, Codex's review: a wrap is illegal
/// everywhere, and a raw difference is `u32.indexw`, which no row takes): the
/// same value wherever it does not wrap. The kingdom's own floored difference,
/// `select(x - y, 0u, y > x)`, reads back as itself.
fn floor_differences(n: &Node) -> Node {
    let Node::App { name, slot, known, kids } = n else { return n.clone() };
    // `min(a, max(n, 1u) - 1u)`, the difference unfloored: how the kingdom
    // writes a clamp below a count (index.clamp_to, and uniform.clamp_to
    // written as it; a floored difference is written `select(..)`), read
    // back as the clamp (2026-10-17: 45 of the 879 random trees that missed
    // the AST drawn read a min of a difference).
    if let ("index.min", "u32", [a, Node::App { name: sub, kids: sk, .. }]) = (name.as_str(), slot.as_str(), kids.as_slice()) {
        if let ("index.sub", [Node::App { name: mx, kids: mk, .. }, Node::Literal { value: one, .. }]) = (sub.as_str(), sk.as_slice()) {
            if let ("index.max", [m, Node::Literal { value: one2, .. }]) = (mx.as_str(), mk.as_slice()) {
                if *one == 1.0 && *one2 == 1.0 {
                    return Node::App { name: "index.clamp_to".into(), slot: slot.clone(), known: true, kids: vec![floor_differences(a), floor_differences(m)] };
                }
            }
        }
    }
    let kids: Vec<Node> = kids.iter().map(floor_differences).collect();
    if name == "index.sub" && kids.len() == 2 && slot == "u32" {
        return Node::App { name: "index.sub_floor".into(), slot: slot.clone(), known: true, kids };
    }
    // `x << k`, k a literal: `x * 2^k` (a left shift is u32.indexw too).
    if let ("index.shl", "u32", [x, Node::Literal { form, slot: ks, value }]) = (name.as_str(), slot.as_str(), kids.as_slice()) {
        if (0.0..32.0).contains(value) && value.fract() == 0.0 {
            let k = Node::Literal { form: form.clone(), slot: ks.clone(), value: 2f64.powi(*value as i32) };
            return Node::App { name: "index.mul".into(), slot: slot.clone(), known: true, kids: vec![x.clone(), k] };
        }
    }
    if name == "select.scalar_cond" {
        if let [Node::Literal { value, .. }, diff @ Node::App { name: d, kids: dk, .. }, Node::App { name: c, kids: ck, .. }] = kids.as_slice() {
            let floored = *value == 0.0 && d == "index.sub_floor" && dk.len() == 2 && ck.len() == 2;
            // (Equal up to a literal's form: the inference may type `4u` a
            // count on one side and an index on the other.)
            fn same(a: &Node, b: &Node) -> bool {
                match (a, b) {
                    (Node::Literal { value: x, .. }, Node::Literal { value: y, .. }) => x == y,
                    _ => a == b,
                }
            }
            if floored && ((c == "compare.gt" && same(&ck[0], &dk[1]) && same(&ck[1], &dk[0])) || (c == "compare.lt" && same(&ck[0], &dk[0]) && same(&ck[1], &dk[1]))) {
                return diff.clone();
            }
        }
    }
    Node::App { name: name.clone(), slot: slot.clone(), known: *known, kids }
}

/// A loop bound without the kingdom's cap (`min(e, MAX_LOOP_TRIP)`, which
/// the renderer writes on every computed bound), wherever it stands in it.
fn strip_cap(t: &Tree) -> Tree {
    let cap = crate::wgsl::render::MAX_LOOP_TRIP as f32;
    if let ("index.min", [a, b]) = (t.name.as_str(), t.kids.as_slice()) {
        if literal_value(b) == Some(cap) {
            return strip_cap(a);
        }
        if literal_value(a) == Some(cap) {
            return strip_cap(b);
        }
    }
    Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(strip_cap).collect() }
}

/// The marker of a tree already in the safe grammar inside a raw one.
const INLINED: &str = "safe.inlined";

/// The increments of a read-add-write of h at the cell `idx`: `value` a sum
/// (arith.add, any nesting) with the cell's load as one term; the other
/// terms in the order the kernel's adds apply them to the load. None when
/// the load is not a term of the sum.
fn h_increments<'t>(value: &'t Tree, idx: &str, key: &dyn Fn(&Tree) -> String) -> Option<Vec<&'t Tree>> {
    if value.name.starts_with("load.buffer.h.#") && value.kids.len() == 1 && key(&value.kids[0]) == idx {
        return Some(Vec::new());
    }
    if value.name == "arith.add" && value.kids.len() == 2 {
        for (inner, other) in [(&value.kids[0], &value.kids[1]), (&value.kids[1], &value.kids[0])] {
            if let Some(mut xs) = h_increments(inner, idx, key) {
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
        folded.konst = f64::from(v) * lit.konst;
        return folded;
    }
    if x.name == "index.mul" && x.kids.len() == 2 {
        for (p, q) in [(&x.kids[0], &x.kids[1]), (&x.kids[1], &x.kids[0])] {
            if let Some(v) = literal_value(p) {
                let mut folded = p.clone();
                folded.konst = f64::from(v) * lit.konst;
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
///
/// Also bound: a let STALE at a use, one that reads a local (by the version
/// its load names) that a store changes before the use reads it (a Kahan sum's
/// `let s = sum + y; c = (s - sum) - y; sum = s;`: inlined, `sum = s` read the
/// compensation already updated; 2026-10-12). Its value is the one at its
/// definition, so it is set there, whatever it reads; the second map names them.
fn lets_used_in_deeper_loops(roots: &[Root], body: &naga::Block, points: &BTreeMap<Vec<usize>, BTreeMap<String, u32>>) -> (HashMap<String, Vec<usize>>, BTreeSet<String>) {
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
    // (Both memoised by let: a chain of lets each read twice, `let a2 = a1 +
    // a1`, was walked 2^n times; 2026-10-17.)
    fn reads_memory(n: &Node, defs: &[(&String, &Vec<usize>, &Node)], memo: &mut HashMap<String, bool>) -> bool {
        match n {
            Node::Leaf { name, .. } => match name.strip_prefix("let.") {
                Some(k) => {
                    if let Some(&v) = memo.get(k) {
                        return v;
                    }
                    memo.insert(k.to_string(), false);
                    let v = defs.iter().find(|(m, _, _)| m.as_str() == k).is_some_and(|(_, _, t)| reads_memory(t, defs, memo));
                    memo.insert(k.to_string(), v);
                    v
                }
                None => false,
            },
            Node::App { name, kids, .. } => {
                // A load other than a uniform's or a scalar local's (a register).
                let register = name.starts_with("load.local.") && !name.contains(".#");
                let memory = name.starts_with("load.") && !name.starts_with("load.uniform.") && !register;
                memory || kids.iter().any(|k| reads_memory(k, defs, memo))
            }
            _ => false,
        }
    }
    // The versioned local loads a let's value reads (through the lets it
    // inlines): `load.local.<name>@<i>@v<n>` is the location `local.<name>@<i>` at version n.
    fn local_loads(n: &Node, defs: &[(&String, &Vec<usize>, &Node)], out: &mut BTreeSet<(String, u32)>, depth: usize, memo: &mut HashMap<String, BTreeSet<(String, u32)>>) {
        if depth > 64 {
            return;
        }
        let name = match n {
            Node::Leaf { name, .. } | Node::App { name, .. } => name.as_str(),
            Node::Literal { .. } => return,
        };
        if let Some(k) = name.strip_prefix("let.") {
            if let Some(set) = memo.get(k) {
                out.extend(set.iter().cloned());
            } else if let Some((_, _, t)) = defs.iter().find(|(m, _, _)| m.as_str() == k) {
                let mut set = BTreeSet::new();
                local_loads(t, defs, &mut set, depth + 1, memo);
                out.extend(set.iter().cloned());
                memo.insert(k.to_string(), set);
            }
        } else if let Some((loc, v)) = name.strip_prefix("load.").filter(|l| l.starts_with("local.") || l.starts_with("workgroup.")).and_then(|l| l.rsplit_once("@v")) {
            // (An element of a local or workgroup array too, at the array's
            // version, which every store to an element bumps: `let ts = best[j];
            // best[j] = cs; cs = ts;`, the v24 reference's insertion, read ts
            // after the store when inlined, 2026-10-17.)
            if let Ok(v) = v.parse::<u32>() {
                out.insert((loc.replace(".#", ""), v));
            }
        }
        if let Node::App { kids, .. } = n {
            kids.iter().for_each(|k| local_loads(k, defs, out, depth, memo));
        }
    }
    // Where a let's value is evaluated: the points of the roots that read it,
    // through the lets that inline it (a bound one at its own definition).
    fn use_points(name: &str, roots: &[Root], bound: &HashMap<String, Vec<usize>>, depth: usize) -> Vec<Vec<usize>> {
        let leaf = format!("let.{name}");
        let mut out = Vec::new();
        if depth > 64 {
            return out;
        }
        for r in roots.iter().filter(|r| mentions(&r.tree, &leaf)) {
            match &r.kind {
                RootKind::Let { name: m } if m == name => {}
                RootKind::Let { name: m } if bound.contains_key(m) => out.push(r.point.clone()),
                RootKind::Let { name: m } => out.extend(use_points(m, roots, bound, depth + 1)),
                _ => out.push(r.point.clone()),
            }
        }
        out
    }
    let at = |p: &Vec<usize>, loc: &str| points.get(p).and_then(|m| m.get(loc)).copied().unwrap_or(0);
    let mut stale = BTreeSet::new();
    let (mut loads_memo, mut memory_memo) = (HashMap::new(), HashMap::new());
    for i in order {
        let (name, path, tree) = defs[i];
        // A struct held whole has no register: each use reads its field.
        if matches!(tree, Node::App { kids, .. } if kids.first().is_some_and(|k| is_struct_slot(slot_of(k)))) {
            continue;
        }
        let def_block = &path[..path.len().saturating_sub(1)];
        let mut loads = BTreeSet::new();
        local_loads(tree, &defs, &mut loads, 0, &mut loads_memo);
        if !loads.is_empty() && use_points(name, roots, &bound, 0).iter().any(|p| loads.iter().any(|(loc, v)| at(p, loc) != *v)) {
            stale.insert(name.clone());
            bound.insert(name.clone(), path.clone());
        } else if reads_memory(tree, &defs, &mut memory_memo) && {
            let u = uses(name, roots, body, &bound, 0);
            // Read more than once: inlined, each use reloads it (nluca's
            // generated_step: a gathered state value read three times a slot,
            // 1.91 ms against the original's 1.26 at 65,536 rows; bound, 1.27;
            // 2026-10-14).
            u.len() > 1 || u.iter().any(|u| in_deeper_loop(body, def_block, u))
        } {
            bound.insert(name.clone(), path.clone());
        }
    }
    (bound, stale)
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
    rung_stem(&t.name).is_some_and(|s| s.starts_with("store.")) || t.kids.iter().any(has_store)
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
    // A block holding a continue guard (block.cont, block.brk) stays one
    // statement: opened, its guard would be a block among statements.
    let guarded = |t: &Tree| t.kids.first().and_then(|b| rung_stem(&b.name)).is_some_and(|b| b == "block.cont" || b == "block.brk");
    if open(&s.name) && !(rung_stem(&s.name) == Some("stmt.block") && guarded(&s)) {
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
    // (s = 1 too: B groups of one cell, the lane offset clamped to 0; a
    // grouped row the kingdom writes with G = B, 2026-10-12.)
    if kl >= 1 && kw.is_multiple_of(workgroup) && (kw / workgroup).is_multiple_of(kl) && kw / workgroup / kl > 1 {
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
        // With one-cell groups and no lane term it is the strided block, read below.
        let strided_after = kl == 1 && lanes.is_empty();
        if strided_after {
            groups.clear();
        }
        if !strided_after {
        let g = if groups.is_empty() { zero() } else { sum_of(&groups.iter().collect::<Vec<_>>())? };
        let e = if lanes.is_empty() { zero() } else { sum_of(&lanes.iter().collect::<Vec<_>>())? };
        if offset_max(&g, tiles, bounds)? < f64::from(g_count) && offset_max(&e, tiles, bounds)? < f64::from(s) {
            return Some((b, Off::Grouped(g_count, unclamped(&g, g_count).clone(), unclamped(&e, s).clone())));
        }
        return None;
        }
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
            // (An opaque value of a slot is that slot's register: a value
            // carried through unchanged, a random module of 2026-10-17.)
            "u32.index" | "u32.bits" | "u32.count" | "u32.opaque" => ('i', 64),
            "bool.flag" | "bool.opaque" => ('b', 16),
            "i32.int" | "i32.opaque" => ('n', 16),
            // (Any form of vec4<u32>, opaque too: insertBits over a vector,
            // 2026-10-13, a random kernel.)
            "vec4<u32>.bits" | "vec4<u32>.index" | "vec4<u32>.count" | "vec4<u32>.opaque" => ('u', 16),
            "vec2<f32>.real" | "vec2<f32>.opaque" => ('w', 16),
            "mat2x2<f32>.real" => ('x', 4),
            "mat3x3<f32>.real" => ('y', 4),
            "mat4x4<f32>.real" => ('z', 4),
            other => match crate::wgsl::render::extra_register_of_value(other) {
                Some(x) => (x.letter, x.count as u32),
                None => match module_register_of_value(self.kingdom, other) {
                    Some((letter, count)) => (letter, count as u32),
                    None => return Err(format!("local {local} has dual {other}, which has no register kind")),
                },
            },
        };
        // A kernel the kingdom wrote names its registers as the rows do
        // (`f17` is reg.f17; the renderer writes reg.f16, reg.f32 and
        // reg.i32 as f16r, f32r and i32r): that row, when free; any other
        // name, the lowest free (2026-10-13, AST to AST: registers were
        // numbered by first use, so a tree read back named them anew).
        let taken: std::collections::BTreeSet<&String> = self.registers.values().collect();
        let base = local.strip_prefix("local.").and_then(|r| r.split('@').next()).unwrap_or("");
        let base = base.strip_suffix('r').filter(|b| matches!(*b, "f16" | "f32" | "i32")).unwrap_or(base);
        let own = base.strip_prefix(kind).and_then(|n| n.parse::<u32>().ok()).filter(|n| *n < max && base == format!("{kind}{n}")).map(|n| format!("reg.{kind}{n}")).filter(|r| !taken.contains(r));
        let row = match own {
            Some(r) => r,
            None => (0..max).map(|n| format!("reg.{kind}{n}")).find(|r| !taken.contains(r)).ok_or_else(|| format!("more than {max} {kind} registers (local {local})"))?,
        };
        self.registers.insert(local.to_string(), row.clone());
        Ok(row)
    }

    /// The scalar value of a module `const.NAME` or `override.NAME` leaf
    /// (an override at its default); None for any other leaf.
    /// The step of a loop's counter stored as `v`: `counter + k` with k a
    /// literal, or a module `const` (its value: `r = r + THREADS`, the v24
    /// reference's, 2026-10-17).
    fn step_of(&self, v: &Tree, counter: &str) -> Option<f32> {
        if !matches!(v.name.as_str(), "index.add" | "int.add") || v.kids.len() != 2 || !v.kids.iter().any(|k| local_name(&k.name).as_deref() == Some(counter)) {
            return None;
        }
        v.kids.iter().find_map(|k| literal_value(k).or_else(|| if k.kids.is_empty() { self.module_constant(&k.name).ok().flatten().map(|x| x as f32) } else { None }))
    }

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

    /// A bound expression (in the safe grammar) as the uniform values, or
    /// None when it reads anything a thread or a subgroup holds alone (a
    /// thread id, a register, a load): the cfg fields, literals, the
    /// workgroup's id, the grid's size, the subgroup size and count, and the
    /// index arithmetic of them.
    /// [`Ctx::walk`]'s branch that ends in a return with statements after it:
    /// `if (c) { a; return; } rest` as `if (c) { a } else { rest }` (None for
    /// any other statement). `here` is the if's path.
    fn early_return_branch(&mut self, at: &BlockAt, here: &[usize], rung: &str) -> Result<Option<Tree>, String> {
        let BlockAt { block, position, last, next } = *at;
        let Some(Statement::If { accept, reject, .. }) = block.get(position) else { return Ok(None) };
        let acc: Vec<&Statement> = accept.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        // `return;` after statements (a return alone at the top level is the
        // thread guard; below it, inside the else an earlier branch made, it is
        // this branch too: v24 run 1's seed24), or a helper's `return v;` (its
        // value to the call's register, then the else).
        let void = matches!(acc.last(), Some(Statement::Return { value: None })) && (acc.len() >= 2 || !(rung == "L0" && here.len() == 1 && self.branch_depth == 0));
        let valued = matches!(acc.last(), Some(Statement::Return { value: Some(_) })) && self.inline_result.is_some();
        if !(void || valued) || !reject.iter().all(|s| matches!(s, Statement::Emit(_))) || Some(position) == last {
            return Ok(None);
        }
        let ret = accept.iter().rposition(|s| matches!(s, Statement::Return { .. })).expect("a return");
        let trimmed = naga::Block::from_vec(accept.iter().take(if valued { ret + 1 } else { ret }).cloned().collect());
        let rest = naga::Block::from_vec(block.iter().skip(position + 1).cloned().collect());
        let root = self.root_at(here, |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at this if")?;
        let cond_tree = tree_of(self, &root.tree)?;
        let c = self.value(&cond_tree.kids[0])?;
        let st = self.stmt_ty(rung)?;
        let level = &rung[1..];
        let needed = holds_barrier(self.module, &trimmed) || holds_barrier(self.module, &rest) || holds_item_loop(self.naga_fn, &trimmed) || holds_item_loop(self.naga_fn, &rest);
        // As an if-else is read: a simple test of two uniform values first,
        // then any uniform predicate, so the text written reads back so.
        let simple = match self.simple_test(&c)?.filter(|_| rung.starts_with('L') && needed) {
            Some((op, x, y)) => match (self.uniform_bound_of(&x)?, self.uniform_bound_of(&y)?) {
                (Some(ux), Some(uy)) => Some((op, ux, uy)),
                _ => None,
            },
            None => None,
        };
        let flag = if rung.starts_with('L') && needed && simple.is_none() { Some(self.uniform_flag_checked(&c)?) } else { None };
        let body_rung = if flag.is_some() || simple.is_some() { rung.to_string() } else { format!("C{level}") };
        let mut p = here.to_vec();
        p.push(0);
        let saved = self.early_return_at.replace([&p[..], &[acc.len() - 1]].concat());
        self.branch_depth += 1;
        let a = self.nested(|ctx, path| ctx.walk(&trimmed, path, &body_rung, 0), &mut p);
        self.early_return_at = saved;
        let mut parent = here[..here.len() - 1].to_vec();
        let b = self.nested(|ctx, path| ctx.walk(&rest, path, &body_rung, next), &mut parent);
        self.branch_depth -= 1;
        let (a, b) = (a?, b?);
        Ok(Some(match (simple, flag) {
            (Some((op, ux, uy)), _) => Tree::new(&format!("stmt.if_else_{op}_{rung}"), st, vec![ux, uy, self.block(a, &body_rung)?, self.block(b, &body_rung)?]),
            (None, Some(f)) => Tree::new(&format!("stmt.if_else_uflag_{rung}"), st, vec![f, self.block(a, &body_rung)?, self.block(b, &body_rung)?]),
            (None, None) => Tree::new(&format!("stmt.if_else_{rung}"), st, vec![c, self.block(a, &body_rung)?, self.block(b, &body_rung)?]),
        }))
    }

    /// A comparison as the operands of a simple uniform test row
    /// (`stmt.if_ule` and its kin): the op and its two sides, before they are
    /// checked uniform; None for anything else.
    fn simple_test(&self, c: &Tree) -> Result<Option<(&'static str, Tree, Tree)>, String> {
        let zero = |t: &Tree| literal_value(t) == Some(0.0);
        let one = Tree::literal("literal.index", ty(self, "u32.index")?, 1.0);
        Ok(match (c.name.as_str(), c.kids.as_slice()) {
            ("compare.eq", [x, z]) | ("compare.eq", [z, x]) if zero(z) => Some(("ult", x.clone(), one)),
            ("compare.ne" | "compare.gt", [x, z]) if zero(z) => Some(("ule", one, x.clone())),
            ("compare.ne" | "compare.lt", [z, x]) if zero(z) => Some(("ule", one, x.clone())),
            ("compare.le", [x, y]) => Some(("ule", x.clone(), y.clone())),
            ("compare.ge", [x, y]) => Some(("ule", y.clone(), x.clone())),
            ("compare.eq", [x, y]) => Some(("ueq", x.clone(), y.clone())),
            ("compare.ne", [x, y]) => Some(("une", x.clone(), y.clone())),
            ("compare.lt", [x, y]) => Some(("ult", x.clone(), y.clone())),
            ("compare.gt", [x, y]) => Some(("ult", y.clone(), x.clone())),
            _ => None,
        })
    }

    /// A condition as a workgroup-uniform predicate (`bool.uflag`): a
    /// comparison of two uniform values, and the and, or and not of such;
    /// None for anything a thread's value or memory can change.
    fn uniform_flag(&self, c: &Tree) -> Result<Option<Tree>, String> {
        let uf = ty(self, "bool.uflag")?;
        Ok(match (c.name.as_str(), c.kids.as_slice()) {
            (n, [a, b]) if n.starts_with("compare.") && ["eq", "ne", "lt", "le", "gt", "ge"].contains(&&n[8..]) => match (self.uniform_bound_of(a)?, self.uniform_bound_of(b)?) {
                (Some(a), Some(b)) => Some(Tree::new(&format!("uflag.{}", &n[8..]), uf, vec![a, b])),
                _ => None,
            },
            ("logic.and" | "logic.or", [a, b]) => match (self.uniform_flag(a)?, self.uniform_flag(b)?) {
                (Some(a), Some(b)) => Some(Tree::new(&format!("uflag.{}", &c.name[6..]), uf, vec![a, b])),
                _ => None,
            },
            ("logic.not", [a]) => self.uniform_flag(a)?.map(|a| Tree::new("uflag.not", uf, vec![a])),
            _ => None,
        })
    }

    /// A test over a body that holds a barrier, as a workgroup-uniform
    /// predicate: of the uniform values when it is one, else the flag itself
    /// as `uflag.of_flag`, which the checker's layer 4 proves uniform (a
    /// register set by a workgroupUniformLoad, a flag set under a uniform
    /// test) or names as a barrier in non-uniform control.
    fn uniform_flag_checked(&self, c: &Tree) -> Result<Tree, String> {
        match self.uniform_flag(c)? {
            Some(f) => Ok(f),
            None => Ok(Tree::new("uflag.of_flag", ty(self, "bool.uflag")?, vec![c.clone()])),
        }
    }

    /// A uniform predicate as the plain flag it computes (a test at a
    /// conditional rung): comparisons of the uniform values as indices.
    fn flag_of_uniform(&self, f: &Tree) -> Result<Tree, String> {
        let flag = ty(self, "bool.flag")?;
        let index = |u: &Tree| -> Result<Tree, String> { Ok(Tree::new("index.of_uniform", ty(self, "u32.index")?, vec![u.clone()])) };
        match (f.name.as_str(), f.kids.as_slice()) {
            ("uflag.and" | "uflag.or", [a, b]) => Ok(Tree::new(&format!("logic.{}", &f.name[6..]), flag, vec![self.flag_of_uniform(a)?, self.flag_of_uniform(b)?])),
            ("uflag.not", [a]) => Ok(Tree::new("logic.not", flag, vec![self.flag_of_uniform(a)?])),
            ("uflag.of_flag", [a]) => Ok(a.clone()),
            (n, [a, b]) if n.starts_with("uflag.") => Ok(Tree::new(&format!("compare.{}", &n[6..]), flag, vec![index(a)?, index(b)?])),
            _ => Err(format!("{} is not a uniform predicate", f.name)),
        }
    }

    fn uniform_of(&self, t: &Tree) -> Result<Option<Tree>, String> {
        let u = ty(self, "u32.uniform")?;
        // A product or a shift may wrap (u32.uniformw): no row takes it, so a
        // uniform value that holds one is not a uniform bound (2026-10-12).
        let uw = ty(self, "u32.uniformw")?;
        let plain = |x: &Option<Tree>| x.as_ref().is_some_and(|x| x.ty == u);
        let leaf = |name: &str| Tree::new(name, u, vec![]);
        let comp = |k: &Tree| -> Option<char> { t.name.strip_prefix("shape.access_").and_then(|i| i.parse::<usize>().ok()).and_then(|i| "xyz".chars().nth(i)).filter(|_| k.kids.is_empty()) };
        Ok(match (t.name.as_str(), t.kids.as_slice()) {
            ("load.cfg_m", []) => Some(leaf("uniform.cfg_m")),
            ("load.cfg_d", []) => Some(leaf("uniform.cfg_d")),
            ("load.cfg_r_rows", []) => Some(leaf("uniform.cfg_r_rows")),
            ("load.cfg_r_off", []) => Some(leaf("uniform.cfg_r_off")),
            // A field of the interface's uniform without a kingdom role: its
            // uniform row (`Role::inferred`).
            (n, []) if n.starts_with("load.") && self.per_kernel_rows.iter().any(|p| p.0 == format!("uniform.{}", &n[5..]) && p.2 == u) => Some(leaf(&format!("uniform.{}", &n[5..]))),
            ("builtin.subgroup_size", []) => Some(leaf("uniform.subgroup_size")),
            ("builtin.num_subgroups", []) => Some(leaf("uniform.num_subgroups")),
            ("index.of_count", [k]) | ("index.of_uniform", [k]) => self.uniform_of(k)?,
            // A loop counter (its loop's bound is uniform wherever an L-rung loop reads it).
            ("safe.counter_index" | "safe.tile_index" | "safe.tile_index2" | "safe.tile_index3" | "safe.tile_index4", [c]) => {
                let row = match c.name.as_str() {
                    "safe.counter" => "uniform.of_counter",
                    "safe.tile_counter" => "uniform.of_tcounter",
                    "safe.tile_counter2" => "uniform.of_tcounter2",
                    "safe.tile_counter3" => "uniform.of_tcounter3",
                    "safe.tile_counter4" => "uniform.of_tcounter4",
                    _ => return Ok(None),
                };
                Some(Tree::new(row, u, vec![c.clone()]))
            }
            (n, []) if n.starts_with("literal.") => {
                let v = t.konst;
                // 0 is the uniform dual's fallback, which renders `0u` (2026-10-12).
                if v == 0.0 {
                    return Ok(Some(Tree::fallback(u)));
                }
                (v.fract() == 0.0 && (1.0..=4096.0).contains(&v)).then(|| Tree::new("uniform.of_lit", u, vec![Tree::literal("literal.lit4096", ty(self, "u32.lit4096").expect("dual"), v)]))
            }
            (n, [k]) if n.starts_with("shape.access_") => match (k.name.as_str(), comp(k)) {
                ("builtin.workgroup_id" | "builtin.work_group_id", Some(c)) => Some(Tree::new(&format!("uniform.wid_{c}"), u, vec![Tree::new("builtin.workgroup_id", ty(self, "vec3<u32>.wid")?, vec![])])),
                ("builtin.num_workgroups" | "builtin.num_work_groups", Some(c)) => Some(Tree::new(&format!("uniform.nwg_{c}"), u, vec![])),
                _ => None,
            },
            ("index.clamp", [a, b, c]) => match (self.uniform_of(a)?, self.uniform_of(b)?, self.uniform_of(c)?) {
                (a, b, c) if plain(&a) && plain(&b) && plain(&c) => Some(Tree::new("uniform.clamp", u, vec![a.expect("plain"), b.expect("plain"), c.expect("plain")])),
                _ => None,
            },
            (n, [a, b]) => {
                let op = match n {
                    "index.mod" => "mod",
                    "index.round_down" => "round_down",
                    "index.band_of" => "band_of",
                    "index.clamp_to" => "clamp_to",
                    "index.add" | "count.add" => "add",
                    "index.sub_floor" | "count.sub_floor" => "sub_floor",
                    "index.mul" => "mul",
                    "index.div" => "div",
                    "index.min" | "count.min" => "min",
                    "index.max" | "count.max" => "max",
                    // The reader names u32 bit operations bits.*: the kingdom's
                    // index and uniform rows render the same operators.
                    "index.shl" | "bits.shl" => "shl",
                    "index.shr" | "bits.shr" => "shr",
                    "index.and" | "bits.and" => "and",
                    "index.or" | "bits.or" => "or",
                    _ => return Ok(None),
                };
                match (self.uniform_of(a)?, self.uniform_of(b)?) {
                    (Some(a), Some(b)) => match (op, a.ty == uw || b.ty == uw) {
                        (_, true) => None,
                        ("mul" | "shl", _) => Some(Tree::new(&format!("uniform.{op}"), uw, vec![a, b])),
                        _ => Some(Tree::new(&format!("uniform.{op}"), u, vec![a, b])),
                    },
                    _ => None,
                }
            }
            _ => None,
        })
    }

    /// A loop bound as uniform values that cannot wrap (`u32.uniform`), or
    /// None.
    fn uniform_bound_of(&self, t: &Tree) -> Result<Option<Tree>, String> {
        let u = ty(self, "u32.uniform")?;
        Ok(self.uniform_of(t)?.filter(|x| x.ty == u))
    }

    /// An expression loop bound's value in the rows that cannot wrap
    /// (`u32.xbound`, 2026-10-12): a register, a uniform value, the
    /// arithmetic of them that does not grow past its operands, or a min with
    /// any index.
    fn xbound_of(&self, t: &Tree) -> Result<Tree, String> {
        let xb = ty(self, "u32.xbound")?;
        let index = ty(self, "u32.index")?;
        let count = ty(self, "u32.count")?;
        // (A second operand of any u32 dual the inference gave it,
        // `x - u32(pack2x16unorm(..))`, u32 bits or an opaque u32: conform
        // makes it the index the row takes. 3 of 1,000 random trees,
        // 2026-10-14.)
        let unsigned = |b: &Tree| self.kingdom.dual(b.ty).is_some_and(|d| d.slot == "u32");
        if let Some(u) = self.uniform_bound_of(t)? {
            return Ok(Tree::new("index.of_uniform", xb, vec![u]));
        }
        let no_row = || format!("a loop bound {} with a product, a shift or a sum over lanes that may wrap: no row of u32.xbound spells it", t.to_sexpr());
        let at = |name: &str, kids: Vec<Tree>| Tree::new(name, xb, kids);
        Ok(match (t.name.as_str(), t.kids.as_slice()) {
            ("reg.get_i", [r]) => at("reg.get_i", vec![r.clone()]),
            // A literal of any value (a uniform literal is 1..=4096).
            (n, []) if n.starts_with("literal.") => Tree::literal("literal.index", xb, t.konst),
            ("index.add" | "index.max" | "index.or", [a, b]) => at(&t.name, vec![self.xbound_of(a)?, self.xbound_of(b)?]),
            ("bits.or", [a, b]) => at("index.or", vec![self.xbound_of(a)?, self.xbound_of(b)?]),
            ("index.sub_floor" | "index.div" | "index.mod" | "index.shr" | "index.and" | "index.round_down", [a, b]) if unsigned(b) => at(&t.name, vec![self.xbound_of(a)?, b.clone()]),
            // The reader names `>>` and `&` on u32 bits.*: the index rows render them.
            ("bits.shr" | "bits.and", [a, b]) if b.name.starts_with("literal.") => at(&t.name.replacen("bits.", "index.", 1), vec![self.xbound_of(a)?, Tree::literal("literal.index", index, b.konst)]),
            // Any other second operand: an index (conform takes a bit row's
            // index twin, 2026-10-12).
            ("bits.shr" | "bits.and", [a, b]) => at(&t.name.replacen("bits.", "index.", 1), vec![self.xbound_of(a)?, b.clone()]),
            // A min with any index: one side a bound that cannot wrap, the
            // other any value (conform makes it an index; read before conform,
            // its dual may still be the inference's, a bit field's:
            // 2026-10-13, a random kernel).
            ("index.min", [a, b]) => match self.xbound_of(a) {
                Ok(a) => at("index.min", vec![a, b.clone()]),
                Err(_) => at("index.min", vec![a.clone(), self.xbound_of(b)?]),
            },
            // (A count of any u32 dual the inference gave it: conform makes
            // it the count the row takes.)
            ("index.clamp_to", [a, n]) if n.ty == count || unsigned(n) => at("index.clamp_to", vec![self.xbound_of(a)?, n.clone()]),
            // A u32 read from workgroup memory (a list's length a thread staged):
            // its value (the loop that reads it is at a conditional rung or
            // uniform by its own proof).
            ("wg.get_u_safe", [a, i]) => at("wg.get_u_safe", vec![a.clone(), i.clone()]),
            // A u32 read from an input binding (the interface's load row at
            // u32.xbound): its value, `for (j < n[0u])`.
            (name, [i]) if self.per_kernel_rows.iter().any(|p| p.0 == name && p.2 == xb) => at(name, vec![i.clone()]),
            // A lane of a u32 vector read from memory (a struct's `morph.y`): the
            // interface's lane row at u32.xbound.
            (lane, [v]) if lane.starts_with("shape.access_") && self.per_kernel_rows.iter().any(|p| p.0 == format!("{}.lane{}", v.name, &lane["shape.access_".len()..]) && p.2 == xb) => {
                let name = format!("{}.lane{}", v.name, &lane["shape.access_".len()..]);
                at(&name, v.kids.clone())
            }
            ("subgroup.min" | "subgroup.max" | "subgroup.broadcast_first", [x]) => at(&t.name, vec![self.xbound_of(x)?]),
            ("subgroup.broadcast" | "subgroup.shuffle" | "subgroup.shuffle_down" | "subgroup.shuffle_up" | "subgroup.shuffle_xor", [x, i]) => at(&t.name, vec![self.xbound_of(x)?, i.clone()]),
            _ => return Err(no_row()),
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
        Ok(Tree::new(&format!("reg.get_{kind}"), ty(self, &dual)?, vec![Tree::new(&row, reg_ty(self, kind)?, vec![])]))
    }

    /// Whether a local (`local.<name>@<h>`) is a vector of u32.
    /// Whether a local is a vector of f32 lanes.
    fn local_vec_float(&self, local: &str) -> bool {
        let Some(h) = local.rsplit_once('@').and_then(|(_, h)| h.parse::<usize>().ok()) else { return false };
        self.naga_fn.local_variables.iter().nth(h).is_some_and(|(_, lv)| matches!(self.module.types[lv.ty].inner, naga::TypeInner::Vector { scalar: naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }, .. }))
    }

    /// A vec2<f32> local's register value, `reg.get_w(w<n>)`.
    fn vec2_register_value(&mut self, local: &str) -> Result<Tree, String> {
        let row = self.register(local, "vec2<f32>.real")?;
        Ok(Tree::new("reg.get_w", ty(self, "vec2<f32>.real")?, vec![Tree::new(&row, reg_ty(self, 'w')?, vec![])]))
    }

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
    /// A loop counter's value where the kernel reads it as an i32 (a loop
    /// `for (var j = 0; j < 6; j++)`): the counter, a u32 here, converted.
    fn counter_as(&self, read: &Tree, counter: Tree) -> Result<Tree, String> {
        if self.kingdom.dual(read.ty).is_some_and(|d| d.slot == "i32") {
            return Ok(Tree::new("convert.index_to_int", read.ty, vec![counter]));
        }
        Ok(counter)
    }

    fn offset_counter(&self, local: &str, k: Tree) -> Result<Tree, String> {
        if let Some(e) = self.counter_exprs.get(local) {
            return Ok(Tree::new("index.add", ty(self, "u32.index")?, vec![e.clone(), k]));
        }
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
            'w' => "reg<vec2<f32>>.real",
            'x' => "reg<mat2x2<f32>>.real",
            'y' => "reg<mat3x3<f32>>.real",
            'z' => "reg<mat4x4<f32>>.real",
            other => crate::wgsl::render::EXTRA_REGISTERS.iter().find(|x| x.letter == other).map_or("reg<bool>.flag", |x| x.dual),
        }
    }

    /// A load or store through a pointer parameter to a local array
    /// (`load.arg.p.#`) as the caller's array's (`load.local.acc@0.#`); None
    /// for any other node.
    fn through_array_arg(&self, t: &Tree) -> Option<Tree> {
        let (op, rest) = t.name.strip_prefix("load.").map(|r| ("load", r)).or_else(|| t.name.strip_prefix("store.").map(|r| ("store", r)))?;
        let (arg, local) = self.array_args.iter().find(|(a, _)| rest.strip_prefix(a.as_str()).is_some_and(|tail| tail.is_empty() || tail.starts_with(['.', '@'])))?;
        Some(Tree { name: format!("{op}.{local}{}", &rest[arg.len()..]), ty: t.ty, konst: t.konst, kids: t.kids.clone() })
    }

    /// A value tree in the safe grammar.
    fn value(&mut self, t: &Tree) -> Result<Tree, String> {
        if let Some(local) = private_as_local(t) {
            return self.value(&local);
        }
        if let Some(x) = t.name.strip_prefix("load.local.").map(|rest| self.aggregate_load(t, rest)).transpose()?.flatten() {
            return Ok(x);
        }
        if let Some(own) = self.through_array_arg(t) {
            return self.value(&own);
        }
        // `*p` for a pointer parameter: the caller's register.
        if let Some((row, kind)) = t.name.strip_prefix("load.").and_then(|r| r.split('@').next()).and_then(|a| self.pointer_args.get(a)).cloned() {
            return Ok(Tree::new(&format!("reg.get_{kind}"), t.ty, vec![Tree::new(&row, reg_ty(self, kind)?, vec![])]));
        }
        // An expression already in the safe grammar (a subgroup or atomic
        // result inlined into a loop's bound): as it is.
        if t.name == INLINED && t.kids.len() == 1 {
            return Ok(t.kids[0].clone());
        }
        let d = self.kingdom.dual(t.ty).map(|d| d.dual.clone()).unwrap_or_default();
        // (Any scalar u32 the inference chose: the same text spelled as the
        // emitter writes a strided row, `wid.y * 408u + lid.y`, was inferred
        // a dual other than the index, 4 random trees.)
        if self.kingdom.dual(t.ty).is_some_and(|x| x.slot == "u32" && x.lanes == 1) {
            if let Some(x) = self.thread_index(t)? {
                return Ok(x);
            }
        }
        // Leaves.
        if t.kids.is_empty() {
            if let Some(v) = self.inline_values.as_ref().and_then(|m| m.get(&t.name)) {
                return Ok(v.clone());
            }
            if let Some(local) = local_name(&t.name) {
                if self.item_counter.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.counter_index", ty(self, "u32.index")?, vec![Tree::new("safe.counter", ty(self, "u32.counter")?, vec![])]);
                    return self.counter_as(t, self.offset_counter(&local, k)?);
                }
                if self.tile_counter.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.tile_index", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter", ty(self, "u32.tcounter")?, vec![])]);
                    return self.counter_as(t, self.offset_counter(&local, k)?);
                }
                if self.tile_counter2.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.tile_index2", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter2", ty(self, "u32.tcounter2")?, vec![])]);
                    return self.counter_as(t, self.offset_counter(&local, k)?);
                }
                if self.tile_counter3.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.tile_index3", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter3", ty(self, "u32.tcounter3")?, vec![])]);
                    return self.counter_as(t, self.offset_counter(&local, k)?);
                }
                if self.tile_counter4.as_deref() == Some(local.as_str()) {
                    let k = Tree::new("safe.tile_index4", ty(self, "u32.index")?, vec![Tree::new("safe.tile_counter4", ty(self, "u32.tcounter4")?, vec![])]);
                    return self.counter_as(t, self.offset_counter(&local, k)?);
                }
                let row = self.register(&local, &d)?;
                let kind = row.chars().nth(4).unwrap_or('f');
                return Ok(Tree::new(&format!("reg.get_{kind}"), t.ty, vec![Tree::new(&row, reg_ty(self, kind)?, vec![])]));
            }
            if let Some(x) = self.other_uniform(t)? {
                return Ok(x);
            }
            if let Some(x) = self.buffer_field(t)? {
                return Ok(x);
            }
            if let Some(field) = t.name.strip_prefix("load.uniform.cfg.") {
                return match field {
                    "d" | "m" | "r_rows" => Ok(Tree::new(&format!("load.cfg_{field}"), ty(self, "u32.count")?, vec![])),
                    "r_off" => Ok(Tree::new("load.cfg_r_off", ty(self, "u32.index")?, vec![])),
                    other => self.uniform_field(other),
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
            if t.name.starts_with("builtin.") || t.name.starts_with("literal.") || t.name == "agg.zero" || t.name == "load.phi_len" || t.name == "load.h_len" || t.name.starts_with("len.") || t.name.starts_with("tex.") {
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
                return Ok(Tree::literal(&row, t.ty, if form == "real" { f64::from(v as f32) } else { v }));
            }
            return Err(format!("leaf {} has no safe spelling", t.name));
        }
        // Loads with an index.
        if let Some(x) = self.other_uniform(t)? {
            return Ok(x);
        }
        if let Some(x) = self.buffer_field(t)? {
            return Ok(x);
        }
        if t.name.starts_with("load.buffer.phi.#") {
            // Any index: a read past the buffer is the buffer's last element (the device's bounds policy, naga Restrict).
            let idx = self.value(&t.kids[0])?;
            return Ok(Tree::new("load.phi_safe", ty(self, "f32.real")?, vec![idx]));
        }
        // A read of the interface's other inputs: its per-kernel row
        // (`load.<buffer>`), at any index (a read past it is the buffer's last element).
        if let Some(buf) = t.name.strip_prefix("load.buffer.").and_then(|r| r.split(['.', '@']).next()).filter(|b| *b != "phi" && *b != "h") {
            if let (Some(p), [i]) = (self.per_kernel_rows.iter().find(|p| p.0 == format!("load.{buf}")), t.kids.as_slice()) {
                let (name, out) = (p.0.clone(), p.2);
                // A lane of a vector element chosen at run time (`keys[r][d & 3u]`
                // over `array<vec4<f32>>`, v24 run 1's seed02): the element
                // loaded whole, then `shape.at` its lane.
                if let Some((i, j)) = t.name.strip_prefix("load.buffer.").and_then(|rest| nested_index(rest, &t.kids)) {
                    let whole = Tree::new(&name, out, vec![self.value(i)?]);
                    return Ok(Tree::new("shape.at", t.ty, vec![whole, self.value(j)?]));
                }
                let idx = self.value(i)?;
                return Ok(Tree::new(&name, out, vec![idx]));
            }
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
                        let pidx = Tree::new(mask, ty(self, pindex_dual(mask))?, vec![i]);
                        let whole = Tree::new("parr.get_t", ty(self, "vec2<f32>.real")?, vec![Tree::new(arr, ty(self, parr_dual_of(arr, kind))?, vec![]), pidx]);
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
                    let pidx = Tree::new(mask, ty(self, pindex_dual(mask))?, vec![i]);
                    let array = Tree::new(arr, ty(self, parr_dual_of(arr, kind))?, vec![]);
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
                    let whole = Tree::new("reg.get_u", ty(self, "vec4<u32>.bits")?, vec![Tree::new(&row, reg_ty(self, 'u')?, vec![])]);
                    if let Some(lane) = literal_value(unmasked(&t.kids[0], 3.0)).filter(|l| (0.0..=3.0).contains(l)) {
                        return Ok(Tree::new(&format!("shape.access_{}", lane as u32), t.ty, vec![whole]));
                    }
                    let idx = self.value(unmasked(&t.kids[0], 3.0))?;
                    return Ok(Tree::new("shape.at", t.ty, vec![whole, idx]));
                }
                if self.local_lanes.get(&base) == Some(&4) {
                    let row = self.register(&base, "vec4<f32>.real")?;
                    let idx = self.value(unmasked(&t.kids[0], 3.0))?;
                    return Ok(Tree::new("reg.get_v_lane", t.ty, vec![Tree::new(&row, reg_ty(self, 'v')?, vec![]), idx]));
                }
                // A lane of a vec2<f32> local: its register (reg.w*, the one its
                // whole value is written to), then the lane.
                if self.local_lanes.get(&base) == Some(&2) && self.local_vec_float(&base) {
                    let whole = self.vec2_register_value(&base)?;
                    if let Some(lane) = literal_value(unmasked(&t.kids[0], 1.0)).filter(|l| (0.0..=1.0).contains(l)) {
                        return Ok(Tree::new(&format!("shape.access_{}", lane as u32), t.ty, vec![whole]));
                    }
                    let idx = self.value(unmasked(&t.kids[0], 1.0))?;
                    return Ok(Tree::new("shape.at", t.ty, vec![whole, idx]));
                }
                // A private array: `load.local.acc@0.#@v3 (idx)`.
                let (arr, kind, mask) = self.private_array(&local, &d)?;
                let idx = self.value(unmasked(&t.kids[0], pindex_mask(mask)))?;
                let pidx = Tree::new(mask, ty(self, pindex_dual(mask))?, vec![idx]);
                let arr_dual = parr_dual_of(arr, kind);
                return Ok(Tree::new(&format!("parr.get_{kind}"), t.ty, vec![Tree::new(arr, ty(self, arr_dual)?, vec![]), pidx]));
            }
        }
        // A lane of a vector uniform block (`params.x`, naga's constant
        // index into the vector): the field of that lane.
        if t.name == "load.uniform.cfg.#" && !self.uniform_lanes.is_empty() {
            if let [k] = t.kids.as_slice() {
                if let Some(lane) = literal_value(k).filter(|v| v.fract() == 0.0 && *v >= 0.0).and_then(|v| self.uniform_lanes.get(v as usize)).cloned() {
                    return self.uniform_field(&lane);
                }
            }
            return Err(format!("a lane of the uniform {} at an index that is not a literal lane", self.uniform_var));
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
        // The local's declared length: an array longer than the kingdom's
        // private array of its kind has no spelling (it was masked into a
        // 16-element one until 2026-10-17: the v24 reference's `var rows:
        // array<u32, 64>` rendered `array<u32, 16>` indexed `& 15u`).
        let declared: Option<u32> = local.rsplit('@').next().and_then(|h| h.parse::<usize>().ok()).and_then(|h| self.naga_fn.local_variables.iter().nth(h)).and_then(|(_, lv)| match self.module.types[lv.ty].inner {
            naga::TypeInner::Array { size: naga::ArraySize::Constant(n), .. } => Some(n.get()),
            _ => None,
        });
        let any_u32 = matches!(elem_dual, "u32.index" | "u32.bits" | "u32.count" | "u32.opaque");
        let huge_u32 = any_u32 && declared.is_some_and(|n| n > 64);
        let long_u32 = any_u32 && !huge_u32 && declared.is_some_and(|n| n > 16);
        let long_v4 = elem_dual == "vec4<f32>.real" && declared.is_some_and(|n| n > 16);
        let (kind, mask) = match elem_dual {
            _ if huge_u32 => ('i', "safe.pidx128"),
            _ if long_u32 => ('i', "safe.pidx64"),
            _ if long_v4 => ('v', "safe.pidx64"),
            "f32.real" => ('f', "safe.pidx64"),
            "vec4<f32>.real" => ('v', "safe.pidx16"),
            // The kingdom's u32 private arrays (pi, pj, pk), whatever form the
            // inference gives an element (an index, bits, an opaque value
            // carried through: a uniform load's register stored into one,
            // 2026-10-17, random modules).
            "u32.index" | "u32.bits" | "u32.count" | "u32.opaque" => ('i', "safe.pidx16"),
            "bool.flag" | "bool.opaque" => ('b', "safe.pidx16"),
            "vec2<f32>.real" => ('t', "safe.pidx16"),
            other => return Err(format!("a local array {local} of {other} has no private array")),
        };
        let used: Vec<&'static str> = self.parrs.values().map(|a| a.0).collect();
        // A kernel the kingdom wrote names its private arrays as the rows do
        // (`pk` is parr.k): that row, when free and of the kind.
        let own = format!("parr.{}", local.strip_prefix("local.").and_then(|r| r.split('@').next()).and_then(|n| n.strip_prefix('p')).unwrap_or(""));
        let capacity = pindex_mask(mask) as u32 + 1;
        if let Some(n) = declared.filter(|n| *n > capacity) {
            return Err(format!("a local array {local} of {n} elements is longer than the kingdom's private arrays of its kind ({capacity})"));
        }
        let family = match kind {
            _ if huge_u32 => "parr128<u32>.index",
            _ if long_u32 => "parr64<u32>.index",
            _ if long_v4 => "parr64<vec4<f32>>.real",
            'f' => "parr<f32>.real",
            'v' => "parr<vec4<f32>>.real",
            'b' => "parr<bool>.flag",
            't' => "parr<vec2<f32>>.real",
            _ => "parr<u32>.index",
        };
        let rows: &[&'static str] = crate::wgsl::render::array_rows(family);
        if let Some(arr) = rows.iter().copied().find(|a| **a == own && !used.contains(a)) {
            self.parrs.insert(local.to_string(), (arr, kind, mask));
            return Ok((arr, kind, mask));
        }
        let arr = rows.iter().copied().find(|a| !used.contains(a)).ok_or_else(|| format!("one more local array {local} of {family} than the grammar's {}", rows.len()))?;
        self.parrs.insert(local.to_string(), (arr, kind, mask));
        Ok((arr, kind, mask))
    }

    /// A workgroup array by name and element dual: four arrays of f32
    /// (`wg.a`..`wg.d`), of vec2<f32> (`wg.v2a`..`wg.v2d`) and of vec4<f32>
    /// (`wg.va`..`wg.vd`); its declared length is recorded (the kernel's
    /// declaration on the grid, 2026-10-12).
    fn array(&mut self, name: &str, elem_dual: &str) -> Result<&'static str, String> {
        if let Some(w) = self.arrays.get(name).copied() {
            self.arrays_used.insert(w);
            return Ok(w);
        }
        // An atomic array is one by its declared type, whatever a use of it reads.
        let atomic = self.module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(name) && g.space == naga::AddressSpace::WorkGroup).and_then(|(_, g)| match &self.module.types[g.ty].inner {
            naga::TypeInner::Array { base, .. } => match self.module.types[*base].inner {
                naga::TypeInner::Atomic(sc) => Some(sc.kind),
                _ => None,
            },
            // (A workgroup atomic scalar, `var<workgroup> wcnt: atomic<u32>;`, is
            // an atomic array of one: v24 run 1's seed00 counting its candidates.)
            naga::TypeInner::Atomic(sc) => Some(sc.kind),
            _ => None,
        });
        // An array of u32 is an index array by its declared type too (a use
        // may read its element as bits or an opaque value; the get row's
        // index converts to what the use demands).
        // (A workgroup u32 scalar is an array of one of them, `var<workgroup>
        // wcode: u32;` read as bits: a seed of run 1, 2026-10-17.)
        let declared_u32 = self.module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(name) && g.space == naga::AddressSpace::WorkGroup).is_some_and(|(_, g)| match &self.module.types[g.ty].inner {
            naga::TypeInner::Array { base, .. } => matches!(self.module.types[*base].inner, naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 })),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 }) => true,
            _ => false,
        });
        let elem_dual = match atomic {
            Some(naga::ScalarKind::Uint) => "atomic<u32>",
            Some(naga::ScalarKind::Sint) => "atomic<i32>",
            Some(other) => return Err(format!("a workgroup array {name} of atomic {other:?} has no safe spelling")),
            None if declared_u32 => "u32.index",
            None => elem_dual,
        };
        let family = match elem_dual {
            "atomic<u32>" => "wgmem<atomic<u32>>.index",
            "atomic<i32>" => "wgmem<atomic<i32>>.int",
            "f32.real" => "wgmem<f32>.real",
            "vec2<f32>.real" => "wgmem<vec2<f32>>.real",
            "i32.int" => "wgmem<i32>.int",
            "u32.index" => "wgmem<u32>.index",
            "vec4<f32>.real" => "wgmem<vec4<f32>>.real",
            other => return Err(format!("a workgroup array {name} of {other} has no safe spelling")),
        };
        let candidates: &[&'static str] = crate::wgsl::render::array_rows(family);
        let used: Vec<&'static str> = self.arrays.values().copied().collect();
        // A kernel the kingdom wrote names its arrays as the rows do
        // (`wg_d` is wg.d): that row, when free; any other name, the first free.
        let own = format!("wg.{}", name.strip_prefix("wg_").unwrap_or(""));
        let which = candidates.iter().copied().find(|c| **c == own && !used.contains(c)).or_else(|| candidates.iter().copied().find(|c| !used.contains(c))).ok_or_else(|| format!("one more workgroup array {name} of {elem_dual} than the grammar's {}", candidates.len()))?;
        let len = self
            .module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(name) && g.space == naga::AddressSpace::WorkGroup)
            .and_then(|(_, g)| match self.module.types[g.ty].inner {
                naga::TypeInner::Array { size: naga::ArraySize::Constant(n), .. } => Some(n.get()),
                // A workgroup scalar or vector: an array of one.
                naga::TypeInner::Scalar(_) | naga::TypeInner::Vector { .. } | naga::TypeInner::Atomic(_) => Some(1),
                _ => None,
            })
            .ok_or_else(|| format!("the workgroup array {name} has no constant length"))?;
        if len > 8192 {
            return Err(format!("the workgroup array {name} of {len} elements is past the kingdom's 8,192"));
        }
        self.arrays.insert(name.to_string(), which);
        self.array_lens.insert(which, len);
        self.arrays_used.insert(which);
        Ok(which)
    }

    /// A subgroup operation: `subgroup.<fun>(x[, lane])` into a register of
    /// its result's dual, at the statement's own point (a collective runs
    /// where the kernel runs it). A ballot (vec4<u32>, no register), a
    /// broadcast whose lane is not a literal, and a result no register holds
    /// are refused by name.
    fn subgroup_statement(&mut self, root: &Root, result: u32, rung: &str) -> Result<Option<Tree>, String> {
        let (expr, d) = self.subgroup_expr(root)?;
        // A result of a dual no register holds (a vec2 or vec3, the kingdom's
        // renderer writes the operation inline at its use): read inline at
        // its use, where naga's statement for it stands (2026-10-12: 80 of
        // 1,000 random trees did not read back).
        // So too a result one statement reads (the kingdom writes every
        // subgroup operation inline at its use; a register each ran out of
        // registers in a fully unrolled loop, 2026-10-12).
        let leaf = format!("subgroup@{result}");
        fn count(n: &Node, leaf: &str) -> usize {
            match n {
                Node::Leaf { name, .. } => usize::from(name == leaf),
                Node::App { kids, .. } => kids.iter().map(|k| count(k, leaf)).sum(),
                _ => 0,
            }
        }
        // (Read by one statement, however often: a subgroup operation of the
        // same operands in one statement is one value, as a store's guard
        // and its cell read it.)
        // (A call's arguments are roots of their own, by the call's statement.)
        let readers = self.roots.values().flatten().filter(|r| count(&r.tree, &leaf) > 0).count() + self.arg_roots.values().filter(|args| args.iter().any(|(_, n)| count(n, &leaf) > 0)).count();
        let in_let = self.lets.values().any(|t| count(t, &leaf) > 0);
        // The emitter's lane on a 2-D workgroup, `let sg_lane: u32 =
        // subgroupExclusiveAdd(1u);`, read by nothing but that let: its uses
        // read as the builtin (tree_of), so the collective itself is no
        // statement (2026-10-17: read as a register set at the kernel's top,
        // the first AST difference of 1 in 10 random trees).
        // (Named by the emitter's let: naga's named expression of the
        // collective's result.)
        let named_lane = self.naga_fn.named_expressions.iter().any(|(h, n)| h.index() as u32 == result && n == "sg_lane");
        let one_lane = expr.name.starts_with("subgroup.exclusive_add") && expr.kids.first().is_some_and(|k| literal_value(k) == Some(1.0));
        if named_lane && one_lane {
            if let Some(row) = self.kingdom.rows.iter().find(|r| r.name == "builtin.subgroup_invocation_id") {
                self.inlined.insert(leaf, Tree::new(&row.name, row.output, vec![]));
                return Ok(None);
            }
        }
        if matches!(d.as_str(), "vec2<f32>.real" | "vec3<f32>.real" | "vec2<u32>.index" | "vec3<u32>.index" | "vec2<i32>.int" | "vec3<i32>.int") || (readers == 1 && !in_let) {
            self.inlined.insert(leaf, expr);
            return Ok(None);
        }
        // Read by nothing (`_ = subgroupBroadcast(x, 1u) >> s`: naga keeps the
        // collective, drops the rest): the collective, discarded.
        if readers == 0 && !in_let {
            let st = self.stmt_ty(rung)?;
            match d.as_str() {
                "u32.index" | "u32.bits" | "u32.count" => return Ok(Some(Tree::new(&format!("stmt.discard_u_{rung}"), st, vec![Tree { ty: ty(self, "u32.index")?, ..expr }]))),
                "i32.int" => return Ok(Some(Tree::new(&format!("stmt.discard_i_{rung}"), st, vec![expr]))),
                _ => {}
            }
        }
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
        // What the register holds, for the cell proof (as for a let): the
        // same operation of the same operands is the same value.
        let held = self.expand_memo(&expr);
        if held.size() <= 512 {
            self.axis_memo.insert(format!("local.subgroup@{result}"), (held, self.loop_depth));
        }
        let st = self.stmt_ty(rung)?;
        Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), expr])))
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
            if fun == "broadcast" || fun == "quad_broadcast" {
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
    fn atomic_statement(&mut self, root: &Root, result: u32, rung: &str) -> Result<Option<Tree>, String> {
        // A storage atomic after the thread guard (a leading `if (c) {
        // return; }`, whose return the grammar drops): its effect reaches
        // other invocations, so it runs only under the guard, as a store of
        // an output does, its old value (when read) in a register set there.
        let storage = matches!(&root.kind, RootKind::Atomic { target, .. } if target.starts_with("buffer."));
        if let (true, Some(g), Some(level)) = (storage, self.guard.clone(), rung.get(1..)) {
            let crung = format!("C{level}");
            let Some(inner) = self.atomic_statement_at(root, result, &crung, false)? else { return Err("our software has a bug: a guarded atomic without its statement".into()) };
            let st = self.stmt_ty(rung)?;
            return Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![g, self.block(vec![inner], &crung)?])));
        }
        self.atomic_statement_at(root, result, rung, true)
    }

    /// [`atomic_statement`] at `rung`; `inline` false keeps the old value in
    /// a register even when one statement reads it.
    fn atomic_statement_at(&mut self, root: &Root, result: u32, rung: &str, inline: bool) -> Result<Option<Tree>, String> {
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
        fn count(n: &Node, leaf: &str) -> usize {
            match n {
                Node::Leaf { name, .. } => usize::from(name == leaf),
                Node::App { kids, .. } => kids.iter().map(|k| count(k, leaf)).sum(),
                _ => 0,
            }
        }
        let uses: usize = self.roots.values().flatten().map(|r| count(&r.tree, &leaf)).sum::<usize>() + self.lets.values().map(|t| count(t, &leaf)).sum::<usize>() + self.arg_roots.values().flatten().map(|(_, n)| count(n, &leaf)).sum::<usize>();
        if uses == 0 {
            return Ok(Some(Tree::new(&format!("stmt.discard_{suffix}_{rung}"), st, vec![expr])));
        }
        // Read once, by the statement naga wrote it ahead of: inline there, as
        // the kingdom's renderer writes an atomic (2026-10-12: a fully
        // unrolled loop of atomics took a register each, past 64).
        if inline && uses == 1 && !self.lets.values().any(|t| reads(t, &leaf)) {
            self.inlined.insert(leaf, expr);
            return Ok(None);
        }
        // The old value read later: to a u32 register, or for an atomic<i32> an i32 one.
        let (dual, kind) = if unsigned { ("u32.index", 'i') } else { ("i32.int", 'n') };
        let row = self.register(&format!("local.atomic@{result}"), dual)?;
        Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), expr])))
    }

    /// A node followed through the lets that name it.
    fn resolve_lets<'n>(&'n self, n: &'n Node) -> &'n Node {
        let mut n = n;
        while let Node::Leaf { name, .. } = n {
            match name.strip_prefix("let.").and_then(|k| self.lets.get(k)) {
                Some(def) => n = def,
                None => break,
            }
        }
        n
    }

    /// The struct a node holds in place, read from a storage binding: the load
    /// `load.buffer.<binding>[.#][.<member>...]` (through lets) or a member of
    /// one (`naga.AccessIndex.k` of it) whose type is a struct. As (the place's
    /// load name, its index operands, the struct's type). None for any other node.
    fn struct_place(&self, n: &Node) -> Option<(String, Vec<Node>, Handle<naga::Type>)> {
        match self.resolve_lets(n) {
            Node::App { name, kids, .. } if name.starts_with("load.buffer.") => {
                let base = name.split('@').next().unwrap_or(name);
                let mut segs = base.strip_prefix("load.buffer.")?.split('.');
                let b = segs.next()?;
                let segs: Vec<&str> = segs.collect();
                let sty = place_type(self.module, b, &segs)?;
                matches!(self.module.types[sty].inner, naga::TypeInner::Struct { .. }).then(|| (name.clone(), kids.clone(), sty))
            }
            Node::App { name, kids, .. } if name.starts_with("naga.AccessIndex.") && kids.len() == 1 => {
                let k: usize = name.strip_prefix("naga.AccessIndex.")?.parse().ok()?;
                let (pname, pkids, sty) = self.struct_place(&kids[0])?;
                let naga::TypeInner::Struct { members, .. } = &self.module.types[sty].inner else { return None };
                let m = members.get(k)?;
                matches!(self.module.types[m.ty].inner, naga::TypeInner::Struct { .. }).then(|| (member_place(&pname, m.name.as_deref().unwrap_or("")), pkids, m.ty))
            }
            _ => None,
        }
    }

    /// A member of a struct held in place: its field load, typed by the
    /// inference of `at` or else by the member's type. None when `n` is not
    /// the field of a struct place.
    fn struct_field_load(&mut self, n: &Node, at: &str) -> Result<Option<Tree>, String> {
        let Node::App { name, kids, .. } = n else { return Ok(None) };
        let (Some(i), [base]) = (name.strip_prefix("naga.AccessIndex.").and_then(|i| i.parse::<usize>().ok()), kids.as_slice()) else { return Ok(None) };
        let Some((pname, pkids, sty)) = self.struct_place(base) else { return Ok(None) };
        let module = self.module;
        let naga::TypeInner::Struct { members, .. } = &module.types[sty].inner else { return Ok(None) };
        let m = members.get(i).ok_or_else(|| format!("{name}: the struct has {} members", members.len()))?;
        if matches!(module.types[m.ty].inner, naga::TypeInner::Struct { .. }) {
            return Ok(None);
        }
        let dual = self.duals.get(at).copied().flatten().or_else(|| field_dual(module, m.ty).and_then(|d| ty(self, &d).ok()));
        let t = dual.ok_or_else(|| format!("{pname}.{}: a field of type {} the kingdom has no dual for", m.name.as_deref().unwrap_or(""), crate::wgsl::interface::type_text(module, m.ty)))?;
        let kids = pkids.iter().map(|k| tree_of(self, k)).collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Tree::new(&member_place(&pname, m.name.as_deref().unwrap_or("")), t, kids)))
    }

    /// A load of a field of a struct binding (`load.buffer.infos.num_keys`) or
    /// of a struct-element array's element (`load.buffer.instances.#.data`):
    /// the interface's row `load.<binding>.<path>`, over the index when the
    /// binding is an array. None for any other name.
    fn buffer_field(&mut self, t: &Tree) -> Result<Option<Tree>, String> {
        let Some(place) = t.name.strip_prefix("load.buffer.").and_then(|r| buffer_place(self.module, r)) else { return Ok(None) };
        let row = if place.path.is_empty() { format!("load.{}", place.binding) } else { format!("load.{}.{}", place.binding, place.path) };
        // A column or a lane of a vector or matrix member (`morph[1]`, `transform[1][2]`):
        // the member's row, then the indices into it.
        if !self.per_kernel_rows.iter().any(|p| p.0 == row && p.1.len() == place.indexes) {
            let mut base = place.path.as_str();
            let mut extra = 0;
            while extra < 2 && base.ends_with('#') {
                base = base[..base.len() - 1].trim_end_matches('.');
                extra += 1;
            }
            if extra > 0 {
                let base_row = if base.is_empty() { format!("load.{}", place.binding) } else { format!("load.{}.{base}", place.binding) };
                if let Some((name, inputs, out)) = self.per_kernel_rows.iter().find(|p| p.0 == base_row && p.1.len() + extra == place.indexes && self.kingdom.dual(p.2).is_some_and(|d| d.lanes > 1)).cloned() {
                    let mut idx = Vec::new();
                    for k in &t.kids {
                        flat_indexes(k, &mut idx);
                    }
                    if idx.len() == inputs.len() + extra {
                        let index = ty(self, "u32.index")?;
                        let kids = idx[..inputs.len()].iter().map(|i| Ok(Tree { ty: index, ..self.value(i)? })).collect::<Result<Vec<_>, String>>()?;
                        let whole = Tree::new(&name, out, kids);
                        return Ok(Some(self.index_into(whole, &idx[inputs.len()..], t.ty)?));
                    }
                }
            }
        }
        let Some((name, inputs, out)) = self.per_kernel_rows.iter().find(|p| p.0 == row && p.1.len() == place.indexes).cloned() else {
            return Err(format!("{} has no safe spelling: the interface has no row {row} over {} indexes (a place of a type the kingdom does not spell)", t.name, place.indexes));
        };
        let mut idx = Vec::new();
        for k in &t.kids {
            flat_indexes(k, &mut idx);
        }
        if idx.len() != inputs.len() {
            return Err(format!("{}: {} index operands for the row {row}, which takes {}", t.name, idx.len(), inputs.len()));
        }
        let index = ty(self, "u32.index")?;
        let kids = idx.iter().map(|i| Ok(Tree { ty: index, ..self.value(i)? })).collect::<Result<Vec<_>, String>>()?;
        Ok(Some(Tree::new(&name, out, kids)))
    }

    /// What a struct value stored whole is made of: the members of a
    /// constructor (through lets) or the struct place it is loaded from.
    fn struct_source(&self, value: &Node, sty: Handle<naga::Type>) -> Result<StructSource, String> {
        match self.resolve_lets(value) {
            Node::App { name, kids, .. } if name.starts_with("naga.Compose.type") => {
                let naga::TypeInner::Struct { members, .. } = &self.module.types[sty].inner else { return Err("not a struct".into()) };
                if members.len() != kids.len() {
                    return Err(format!("{name}: {} operands for a struct of {} members", kids.len(), members.len()));
                }
                Ok(StructSource::Compose(kids.clone()))
            }
            other => match self.struct_place(other) {
                Some((pname, pkids, _)) => Ok(StructSource::Place(pname, pkids)),
                None => Err(format!("a struct written whole from {}: the reader splits a constructor or a struct loaded from a storage binding", other.to_sexpr().chars().take(80).collect::<String>())),
            },
        }
    }

    /// The stores of a struct's members to the place `path` (`buffer.dst.#`) at
    /// `idx`, a nested struct's members in turn.
    fn store_struct(&mut self, path: &str, idx: Option<&Tree>, sty: Handle<naga::Type>, src: &StructSource, rung: &str) -> Result<Vec<Tree>, String> {
        let module = self.module;
        let naga::TypeInner::Struct { members, .. } = &module.types[sty].inner else { return Err("not a struct".into()) };
        let mut out = Vec::new();
        for (k, m) in members.iter().enumerate() {
            let mn = m.name.clone().unwrap_or_else(|| format!("f{k}"));
            let member_path = format!("{path}.{mn}");
            let nested = matches!(module.types[m.ty].inner, naga::TypeInner::Struct { .. });
            let (sub, value): (Option<StructSource>, Option<Tree>) = match src {
                StructSource::Compose(kids) => {
                    if nested {
                        (Some(self.struct_source(&kids[k], m.ty)?), None)
                    } else {
                        (None, Some(tree_of(self, &kids[k])?))
                    }
                }
                StructSource::Place(pname, pkids) => {
                    let place = member_place(pname, &mn);
                    if nested {
                        (Some(StructSource::Place(place, pkids.clone())), None)
                    } else {
                        let dual = field_dual(module, m.ty).ok_or_else(|| format!("{member_path}: a member of type {} the kingdom has no dual for", crate::wgsl::interface::type_text(module, m.ty)))?;
                        let kids = pkids.iter().map(|k| tree_of(self, k)).collect::<Result<Vec<_>, _>>()?;
                        (None, Some(Tree::new(&place, ty(self, &dual)?, kids)))
                    }
                }
            };
            if let Some(sub) = sub {
                out.extend(self.store_struct(&member_path, idx, m.ty, &sub, rung)?);
            } else if let Some(v) = value {
                let kids: Vec<Tree> = idx.into_iter().cloned().chain([v]).collect();
                let t = Tree::new(&format!("store.{member_path}"), self.stmt_ty(rung)?, kids);
                out.extend(self.store(&t, rung)?);
            }
        }
        Ok(out)
    }

    /// A texture or sampler operation (`tex.load.<b>.<sig>`, `tex.dim.<b>`,
    /// `tex.sample.<b>.<s>.<mode>...`): the interface's row of the same name
    /// over the operands.
    fn texture_row(&mut self, name: &str, kids: &[Node]) -> Result<Tree, String> {
        let Some((row, inputs, out)) = self.per_kernel_rows.iter().find(|p| p.0 == name).cloned() else {
            return Err(format!("{name}: the interface has no such row (a texture operation of operand types the kingdom does not spell)"));
        };
        if inputs.len() != kids.len() {
            return Err(format!("{name}: {} operands for a row of {}", kids.len(), inputs.len()));
        }
        let kids = kids.iter().map(|k| tree_of(self, k)).collect::<Result<Vec<_>, _>>()?;
        Ok(Tree::new(&row, out, kids))
    }

    /// A field of a uniform block other than the first (or of any block, when
    /// the field has no kingdom role): the interface's row `load.<var>_<field>`.
    /// A leaf `load.uniform.<var>.<field>` is the field's value; with an index
    /// (`load.uniform.<var>.<field>.#`) a lane of a vector field (a literal
    /// lane `shape.access_N`, else `shape.at`) or the element of an array
    /// field (the row takes the index). None for any other name.
    fn other_uniform(&mut self, t: &Tree) -> Result<Option<Tree>, String> {
        let Some(rest) = t.name.strip_prefix("load.uniform.") else { return Ok(None) };
        // The first block's fields without a kingdom role are named under its canonical `cfg`.
        let renamed;
        let rest = match rest.strip_prefix("cfg.") {
            Some(f) if !["d", "m", "r_rows", "r_off", "#"].contains(&f.split(['.', '@']).next().unwrap_or(f)) && !self.uniform_var.is_empty() => {
                renamed = format!("{}.{f}", self.uniform_var);
                renamed.as_str()
            }
            Some(_) => return Ok(None),
            None => rest,
        };
        // A uniform that is no struct (`load.uniform.u`, `load.uniform.u.#`):
        // the variable's own row.
        let whole = rest.split(['.', '@']).next().unwrap_or(rest);
        if self.module.global_variables.iter().any(|(_, g)| g.name.as_deref() == Some(whole) && g.space == naga::AddressSpace::Uniform) && self.per_kernel_rows.iter().any(|p| p.0 == format!("load.{whole}")) && rest[whole.len()..].split('.').all(|seg| seg.is_empty() || seg == "#") {
            let indexes = rest[whole.len()..].matches("#").count();
            return self.uniform_value(t, whole, indexes);
        }
        let Some((var, tail)) = rest.split_once('.') else { return Ok(None) };
        let (field, indexes) = match tail.find(".#") {
            Some(i) => (&tail[..i], tail[i..].matches('#').count()),
            None => (tail, 0),
        };
        if field.contains(['.', '#', '@']) || tail[field.len()..] != ".#".repeat(indexes) {
            // (A place inside a uniform that is an aggregate value: its whole value, then each step.)
            if let Some(x) = self.aggregate_binding_load(t, rest)? {
                return Ok(Some(x));
            }
            return Err(format!("{} has no safe spelling: a uniform field read through a path the reader does not spell", t.name));
        }
        let row = format!("load.{var}_{field}");
        let Some((name, inputs, out)) = self.per_kernel_rows.iter().find(|p| p.0 == row).cloned() else {
            return Err(format!("uniform field {var}.{field}: the interface has no row {row}"));
        };
        let lane = |ctx: &mut Self, whole: Tree, j: &Tree| -> Result<Tree, String> {
            Ok(match literal_value(j) {
                Some(l) if l >= 0.0 && l.fract() == 0.0 => Tree::new(&format!("shape.access_{}", l as u32), t.ty, vec![whole]),
                _ => Tree::new("shape.at", t.ty, vec![whole, ctx.value(j)?]),
            })
        };
        match (inputs.len(), indexes, t.kids.as_slice()) {
            (0, 0, []) => Ok(Some(Tree::new(&name, out, vec![]))),
            (1, 1, [i]) => Ok(Some(Tree::new(&name, out, vec![Tree { ty: ty(self, "u32.index")?, ..self.value(i)? }]))),
            (0, 1, [j]) => Ok(Some(lane(self, Tree::new(&name, out, vec![]), j)?)),
            (1, 2, [n]) if n.name == "naga.NestedAccess" && n.kids.len() == 2 => {
                let whole = Tree::new(&name, out, vec![Tree { ty: ty(self, "u32.index")?, ..self.value(&n.kids[0])? }]);
                Ok(Some(lane(self, whole, &n.kids[1])?))
            }
            _ => Err(format!("{} has no safe spelling: the row {row} takes {} inputs", t.name, inputs.len())),
        }
    }

    /// A value indexed by the operands `ix` in turn (a matrix's column, then a
    /// lane; a vector's lane), the last giving `result`'s type. A literal index
    /// is the row of that column or lane, any other the dynamic one.
    fn index_into(&mut self, whole: Tree, ix: &[Tree], result: Ty) -> Result<Tree, String> {
        let mut whole = whole;
        for (n, j) in ix.iter().enumerate() {
            let last = n + 1 == ix.len();
            let matrix_rows = self.kingdom.dual(whole.ty).and_then(|d| d.slot.strip_prefix("mat").map(|m| m[2..3].to_string()));
            let to = |ctx: &Self, rows: Option<&str>| -> Result<Ty, String> {
                if last {
                    return Ok(result);
                }
                match rows {
                    Some(r) => ty(ctx, &format!("vec{r}<f32>.real")),
                    None => Ok(whole.ty),
                }
            };
            whole = match (literal_value(j), matrix_rows) {
                (Some(l), Some(rows)) if l >= 0.0 && l.fract() == 0.0 => Tree::new(&format!("shape.col_{}", l as u32), to(self, Some(&rows))?, vec![whole.clone()]),
                (None, Some(rows)) => Tree::new("shape.col_at", to(self, Some(&rows))?, vec![whole.clone(), self.value(j)?]),
                (Some(l), None) if l >= 0.0 && l.fract() == 0.0 => Tree::new(&format!("shape.access_{}", l as u32), to(self, None)?, vec![whole.clone()]),
                _ => Tree::new("shape.at", to(self, None)?, vec![whole.clone(), self.value(j)?]),
            };
        }
        Ok(whole)
    }

    /// A uniform variable that is no struct, read whole or at an index (an
    /// array's element, a vector's or a matrix's lane or column): its row
    /// `load.<var>`.
    fn uniform_value(&mut self, t: &Tree, var: &str, indexes: usize) -> Result<Option<Tree>, String> {
        let row = format!("load.{var}");
        if let Some(x) = self.aggregate_binding_load(t, &t.name["load.uniform.".len()..])? {
            return Ok(Some(x));
        }
        // (The row of the most inputs the indexes give: an array's element over its index, else the whole value.)
        let Some((name, inputs, out)) = self.per_kernel_rows.iter().filter(|p| p.0 == row && p.1.len() <= indexes).max_by_key(|p| p.1.len()).cloned() else {
            return Ok(None);
        };
        let mut idx = Vec::new();
        for k in &t.kids {
            flat_indexes(k, &mut idx);
        }
        let index = ty(self, "u32.index")?;
        // The row's own inputs first (an array's element), the rest are lanes.
        if idx.len() < inputs.len() || idx.len() != indexes {
            return Err(format!("{} has no safe spelling: {} index operands over the row {row}", t.name, idx.len()));
        }
        let whole = Tree::new(&name, out, idx[..inputs.len()].iter().map(|i| Ok(Tree { ty: index, ..self.value(i)? })).collect::<Result<Vec<_>, String>>()?);
        let whole = self.index_into(whole, &idx[inputs.len()..], t.ty)?;
        Ok(Some(whole))
    }

    /// A uniform field without a kingdom role: the interface's row of its
    /// value, `load.<var>_<field>` (`Role::inferred`).
    fn uniform_field(&self, field: &str) -> Result<Tree, String> {
        let row = format!("load.{}_{field}", self.uniform_var);
        match self.per_kernel_rows.iter().find(|p| p.0 == row && p.1.is_empty()) {
            Some((name, _, out)) => Ok(Tree::new(name, *out, vec![])),
            None => Err(format!("uniform field {field}: the interface has no row {row}")),
        }
    }

    /// An atomic's expression `atomic.<fun>_<u|i>(array, index, value)` and
    /// whether its array is atomic<u32>.
    fn atomic_expr(&mut self, root: &Root) -> Result<(Tree, bool), String> {
        let RootKind::Atomic { target, fun } = &root.kind else { return Err("not an atomic root".into()) };
        let Node::App { kids, .. } = &root.tree else { return Err("an atomic root without operands".into()) };
        let trees = kids.iter().map(|k| tree_of(self, k)).collect::<Result<Vec<_>, _>>()?;
        // An atomic leaf of a struct binding or of a struct-element array's
        // element, or of an array in one: the row `atomic.<b>.<path>_<fun>`
        // over the place's indexes and the operand.
        if let Some(place) = target.strip_prefix("buffer.").and_then(|r| buffer_place(self.module, r)).filter(|p| !p.path.is_empty()) {
            if fun == "cas" {
                return Err("atomicCompareExchangeWeak has no safe spelling: the kingdom has no compare-exchange row".into());
            }
            let row = format!("atomic.{}.{}_{fun}", place.binding, place.path);
            let Some((_, _, out)) = self.per_kernel_rows.iter().find(|p| p.0 == row && p.1.len() == place.indexes + 1).cloned() else {
                return Err(format!("an atomic {fun} on {target}: the interface has no row {row} over {} indexes", place.indexes));
            };
            let Some((value, index_kids)) = trees.split_last() else { return Err(format!("an atomic {fun} on {target} without an operand")) };
            let mut idx = Vec::new();
            for k in index_kids {
                flat_indexes(k, &mut idx);
            }
            if idx.len() != place.indexes {
                return Err(format!("an atomic {fun} on {target} with {} index operands, {} expected", idx.len(), place.indexes));
            }
            let index = ty(self, "u32.index")?;
            let mut kids = idx.iter().map(|i| Ok(Tree { ty: index, ..self.value(i)? })).collect::<Result<Vec<_>, String>>()?;
            kids.push(self.value(value)?);
            let unsigned = self.kingdom.dual(out).is_some_and(|d| d.slot == "u32");
            return Ok((Tree::new(&row, out, kids), unsigned));
        }
        // A storage atomic the interface declares: its per-kernel row
        // `atomic.<name>_<fun>` over the operand.
        if let Some(name) = target.strip_prefix("buffer.").filter(|n| !n.contains('.')) {
            let row = format!("atomic.{name}_{fun}");
            let Some((_, _, out)) = self.per_kernel_rows.iter().find(|p| p.0 == row).cloned() else {
                return Err(format!("an atomic {fun} on the storage buffer {name}: the kernel's interface declares no atomic {name}"));
            };
            let value = trees.last().ok_or("a storage atomic without an operand")?;
            let value = self.value(value)?;
            let unsigned = self.kingdom.dual(out).is_some_and(|d| d.slot == "u32");
            return Ok((Tree::new(&row, out, vec![value]), unsigned));
        }
        // An element of a storage array of atomics the interface declares
        // (`atomicAdd(&a[i], v)`): its per-kernel row `atomic.<name>_<fun>`
        // over the index (the row clamps it to the binding) and the operand.
        if let Some(name) = target.strip_prefix("buffer.").and_then(|r| r.strip_suffix(".#")).filter(|n| !n.contains('.')) {
            if fun == "cas" {
                return Err("atomicCompareExchangeWeak has no safe spelling: the kingdom has no compare-exchange row (naga 0.20's Metal backend did not implement it; naga 30's does, and the kingdom has not added the row)".into());
            }
            let row = format!("atomic.{name}_{fun}");
            let Some((_, _, out)) = self.per_kernel_rows.iter().find(|p| p.0 == row && p.1.len() == 2).cloned() else {
                return Err(format!("an atomic {fun} on an element of the storage buffer {name}: the kernel's interface declares no array of atomics {name}"));
            };
            let [idx, value] = trees.as_slice() else { return Err(format!("an atomic {fun} on {name} with {} operands, an index and a value expected", trees.len())) };
            let index = ty(self, "u32.index")?;
            let idx = Tree { ty: index, ..self.value(idx)? };
            let value = self.value(value)?;
            let unsigned = self.kingdom.dual(out).is_some_and(|d| d.slot == "u32");
            return Ok((Tree::new(&row, out, vec![idx, value]), unsigned));
        }
        // (A workgroup atomic scalar, `atomicAdd(&wcnt, 1u)`: element 0 of an
        // atomic array of one, as a workgroup scalar's store is.)
        let (arr, trees) = match target.strip_prefix("workgroup.") {
            Some(r) if r.ends_with(".#") => (&r[..r.len() - 2], trees),
            Some(r) if !r.contains('.') => (r, std::iter::once(Tree::literal("literal.index", ty(self, "u32.index")?, 0.0)).chain(trees).collect()),
            _ => return Err(format!("an atomic on {target}: the safe kingdom's atomics are workgroup arrays or a storage atomic the interface declares")),
        };
        let which = self.array(arr, "atomic")?;
        let (of, idx_dual, arr_dual, _, _) = Self::wg_rows(which);
        let unsigned = which.starts_with("wg.au");
        let suffix = if unsigned { "u" } else { "i" };
        let idx = self.value(self.wg_unmasked(&trees[0], which))?;
        let at = Tree::new(of, ty(self, idx_dual)?, vec![idx]);
        let array = Tree::new(which, ty(self, arr_dual)?, vec![]);
        let out_ty = ty(self, if unsigned { "u32.index" } else { "i32.int" })?;
        if fun == "cas" {
            return Err("atomicCompareExchangeWeak has no safe spelling: the kingdom has no compare-exchange row (naga 0.20's Metal backend did not implement it; naga 30's does, and the kingdom has not added the row)".into());
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
        if which.starts_with("wg.u") {
            return ("safe.wg_of", "u32.wgindex", "wgmem<u32>.index", "wg.get_u_safe", "wg.set_u");
        }
        if which.starts_with("wg.n") {
            return ("safe.wg_of", "u32.wgindex", "wgmem<i32>.int", "wg.get_n_safe", "wg.set_n");
        }
        if which.starts_with("wg.v") {
            ("safe.wgv_of", "u32.wgvindex", "wgmem<vec4<f32>>.real", "wg.get_v_safe", "wg.set_v")
        } else {
            ("safe.wg_of", "u32.wgindex", "wgmem<f32>.real", "wg.get_safe", "wg.set")
        }
    }

    /// `t` with every scalar local the memo knows replaced by what it holds.
    fn expand_memo(&self, t: &Tree) -> Tree {
        self.expand_memo_at(t, 0)
    }

    /// [`expand_memo`] with the arguments of the helper `depth` levels out
    /// from the one being inlined (0: its own): an argument is the caller's
    /// expression, expanded in the caller's scope, whose own arguments are
    /// the next level out's (a helper inlined inside another and given the
    /// outer's argument of the same name, `f(x)` calling `g(x)`, expanded
    /// `arg.x` into itself without end: two seeds of v24's run 1, 2026-10-17).
    fn expand_memo_at(&self, t: &Tree, depth: usize) -> Tree {
        self.built.set(self.built.get() + 1);
        if self.built.get() > READ_NODE_BOUND {
            // (Stopped: the walk names it at the statement, see `walk`.)
            return t.clone();
        }
        if t.kids.is_empty() {
            // A statement-inlined helper's argument: the caller's expression
            // (fresh48 of run A: `wrow(r, c0)` storing to h[r * d + c0 + j]).
            if t.name.starts_with("arg.") {
                let args = if depth == 0 { self.inline_args.as_ref() } else { self.outer_args.len().checked_sub(depth).and_then(|i| self.outer_args[i].as_ref()) };
                if let Some(raw) = args.and_then(|a| a.get(&t.name)) {
                    return self.expand_memo_at(raw, depth + 1);
                }
                return t.clone();
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
        Tree { name: t.name.clone(), ty: t.ty, konst: t.konst, kids: t.kids.iter().map(|k| self.expand_memo_at(k, depth)).collect() }
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
                    let base_row = self.thread_base("row", &i, by)?;
                    let base_col = self.thread_base("col", &j, bx)?;
                    let row = self.plus(base_row, i, "row")?;
                    let c = self.plus(base_col, j, "col")?;
                    return Ok(Tree::new("safe.cell_of", ty(self, "u32.cell")?, vec![row, c]));
                }
            }
        }
        fail(&format!("row or column is not this thread's (tile counters {:?} / {:?} / {:?} / {:?})", self.tile_counter, self.tile_counter2, self.tile_counter3, self.tile_counter4))
    }

    /// The origin of this thread's row (or column) of block factor `b` that
    /// the offset `off` was proved against: the strided or grouped origin
    /// for a strided or grouped offset, else `gid * b`.
    fn thread_base(&self, axis: &str, off: &Off, b: u32) -> Result<Tree, String> {
        let dual = ty(self, if axis == "row" { "u32.threadrow" } else { "u32.threadcol" })?;
        let gid = Tree::new("builtin.global_invocation_id", ty(self, "vec3<u32>.gid")?, vec![]);
        let wid = Tree::new("builtin.workgroup_id", ty(self, "vec3<u32>.wid")?, vec![]);
        let lid = Tree::new("builtin.local_invocation_id", ty(self, "vec3<u32>.lid")?, vec![]);
        let blk = Tree::literal("literal.blk", ty(self, "u32.blk")?, b as f32);
        Ok(match off {
            Off::Grouped(g, _, _) => Tree::new(&format!("safe.thread_{axis}_g"), dual, vec![wid, lid, blk, Tree::literal("literal.lit64", ty(self, "u32.lit64")?, *g as f32)]),
            Off::Strided(_) | Off::StridedTile(_) | Off::StridedExpr(_) => Tree::new(&format!("safe.thread_{axis}_s"), dual, vec![wid, lid, blk]),
            _ => Tree::new(&format!("safe.thread_{axis}"), dual, vec![gid, blk]),
        })
    }

    /// An index that is this thread's row or column, `gid.y * B + min(k, B -
    /// 1u)` and the like, as the emitter writes `safe.row_as_index` and
    /// `safe.col_as_index` (2026-10-17: read as index arithmetic, 74 of the
    /// 915 random trees that missed the AST drawn): the row's axis is the
    /// layout's (an open layout reads it again once decided, rows on y when
    /// no store decides), the block factor the grid's, and a factor of 1 is
    /// left as the arithmetic it is (`gid.y` itself).
    fn thread_index(&mut self, t: &Tree) -> Result<Option<Tree>, String> {
        if !matches!(t.name.as_str(), "index.add" | "index.mul") || t.kids.len() != 2 {
            return Ok(None);
        }
        let t = distribute_literals(&self.expand_memo(t));
        let tiles = (self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone(), self.tile_counter4.clone());
        let tiles: Tiles = [&tiles.0, &tiles.1, &tiles.2, &tiles.3];
        let layouts: &[bool] = match self.rows_x {
            Some(x) => if x { &[true] } else { &[false] },
            None => &[false, true],
        };
        for &rows_x in layouts {
            let (row, col) = if rows_x { ((0, self.grid.0, self.grid.3), (1, self.grid.1, self.grid.2)) } else { ((1, self.grid.1, self.grid.3), (0, self.grid.0, self.grid.2)) };
            for (axis, (a, w, blk)) in [("row", row), ("col", col)] {
                if blk < 2 {
                    continue;
                }
                let Some((b, off)) = thread_axis_off_in(&t, a, w, tiles, &self.row_arrays, &self.tile_bounds) else { continue };
                if b != blk {
                    continue;
                }
                // Open: the axis is a row under one layout and a column
                // under the other, so it waits for the read under the
                // layout the stores decide.
                if self.rows_x.is_none() {
                    self.open_thread_index = true;
                    return Ok(None);
                }
                let base = self.thread_base(axis, &off, b)?;
                let x = self.plus(base, off, axis)?;
                return Ok(Some(Tree::new(&format!("safe.{axis}_as_index"), ty(self, "u32.index")?, vec![x])));
            }
        }
        Ok(None)
    }

    /// `C * cfg.d + R` with R and C this thread's row and column: the mirror
    /// of its cell (`safe.mirror_of(row, col)`), proved as the cell R * d + C.
    fn mirror(&mut self, idx: &Tree) -> Option<Tree> {
        let expanded = self.expand_memo(idx);
        let mut terms = Vec::new();
        flatten_add(&expanded, &mut terms);
        let is_prod = |t: &Tree| t.name == "index.mul" && t.kids.len() == 2 && t.kids.iter().any(|k| is_leaf(k, "load.uniform.cfg.d"));
        let p = terms.iter().position(|t| is_prod(t))?;
        if terms.len() < 2 {
            return None;
        }
        let prod = terms[p];
        let (factor, d) = if is_leaf(&prod.kids[1], "load.uniform.cfg.d") { (&prod.kids[0], &prod.kids[1]) } else { (&prod.kids[1], &prod.kids[0]) };
        let rest: Vec<Tree> = terms.iter().enumerate().filter(|(i, _)| *i != p).map(|(_, t)| (*t).clone()).collect();
        let row = rest.into_iter().reduce(|acc, t| Tree::new("index.add", idx.ty, vec![acc, t]))?;
        let swapped = Tree::new("index.add", idx.ty, vec![Tree::new("index.mul", prod.ty, vec![row, d.clone()]), factor.clone()]);
        let saved = self.rows_x;
        match self.cell(&swapped) {
            Ok(c) if c.name == "safe.cell_of" => Some(Tree { name: "safe.mirror_of".into(), ty: ty(self, "u32.mirror").ok()?, konst: 0.0, kids: c.kids }),
            _ => {
                self.rows_x = saved;
                None
            }
        }
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

    /// A loop body's breaks below its top level (in an if, after a store,
    /// or more than one) as a stop flag: a bool register set false first,
    /// each `break` setting it true, the statements after a statement that
    /// may break run under `if (!stop)`; the flag is the body's break
    /// condition, its guard the body's last statement. The same values:
    /// nothing after a break runs, and the loop ends at the guard.
    fn lower_breaks(&mut self, flat: Vec<Tree>, rung: &str, path: &[usize]) -> Result<(Vec<Tree>, Tree), String> {
        let local = format!("local.break_stop@{}", path.iter().map(|p| p.to_string()).collect::<Vec<_>>().join("_"));
        let row = self.register(&local, "bool.flag")?;
        let reg = Tree::new(&row, reg_ty(self, 'b')?, vec![]);
        let flag = ty(self, "bool.flag")?;
        let set_false = Tree::new(&format!("reg.set_b_{rung}"), self.stmt_ty(rung)?, vec![reg.clone(), Tree::literal("literal.flag", flag, 0.0)]);
        let mut pre = vec![set_false];
        pre.extend(self.lower_break_list(flat, rung, &reg)?);
        Ok((pre, Tree::new("reg.get_b", flag, vec![reg])))
    }

    /// [`Ctx::lower_breaks`] over statements at `rung`.
    fn lower_break_list(&self, list: Vec<Tree>, rung: &str, reg: &Tree) -> Result<Vec<Tree>, String> {
        let mut flat = Vec::new();
        for t in list {
            flatten_stmts(t, &mut flat);
        }
        let mut out = Vec::new();
        let mut it = flat.into_iter();
        while let Some(t) = it.next() {
            if !has_break(&t) {
                out.push(t);
                continue;
            }
            out.push(self.lower_break_stmt(t, rung, reg)?);
            let rest: Vec<Tree> = it.collect();
            if !rest.is_empty() {
                let flag = ty(self, "bool.flag")?;
                let go = Tree::new("logic.not", flag, vec![Tree::new("reg.get_b", flag, vec![reg.clone()])]);
                let body = self.lower_break_list(rest, rung, reg)?;
                out.push(Tree::new(&format!("stmt.if_{rung}"), self.stmt_ty(rung)?, vec![go, self.block(body, rung)?]));
            }
            break;
        }
        Ok(out)
    }

    /// One statement holding a break of the loop being lowered.
    fn lower_break_stmt(&self, t: Tree, rung: &str, reg: &Tree) -> Result<Tree, String> {
        if t.name == "stmt.break" {
            let flag = ty(self, "bool.flag")?;
            return Ok(Tree::new(&format!("reg.set_b_{rung}"), self.stmt_ty(rung)?, vec![reg.clone(), Tree::literal("literal.flag", flag, 1.0)]));
        }
        let inner = |b: Tree| -> Result<Tree, String> {
            let r = b.name.rsplit_once('_').map(|(_, r)| r.to_string()).filter(|r| rung_stem(&format!("x_{r}")).is_some()).unwrap_or_else(|| rung.to_string());
            self.block(self.lower_break_list(vec![b], &r, reg)?, &r)
        };
        match (rung_stem(&t.name), t.kids.len()) {
            (Some("stmt.if"), 2) | (Some("stmt.if_else"), 3) => {
                let mut kids = t.kids.into_iter();
                let mut out = vec![kids.next().expect("the condition")];
                for b in kids {
                    out.push(inner(b)?);
                }
                Ok(Tree { kids: out, ..t })
            }
            _ => Err(format!("a break inside {}", t.name)),
        }
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
        // A body that returns (main ends in `return;`): the epilogue was not
        // written (the renderer stops at the return, and writes none at the
        // end of an epilogue).
        let unspelled = |stmts: &[Tree]| -> bool {
            match stmts.iter().rposition(has_store) {
                Some(p) => p + 1 < stmts.len() || self.epilogue_block(vec![stmts[p].clone()], 0).is_err(),
                None => true,
            }
        };
        if self.main_returns && unspelled(&stmts) {
            stmts.push(Tree::new("stmt.return_L0", self.stmt_ty("L0")?, vec![]));
            return Ok((stmts, Tree::fallback(ty(self, "block.O0")?)));
        }
        let p = stmts.iter().rposition(has_store).ok_or("the kernel never writes h: the kingdom's root ends every kernel in the store of its thread's cell")?;
        // An interface with no output cell (no `h`: v24's index arena and
        // candidate lists, 2026-10-17) does not end in one store: statements
        // after its last store (atomics, more stores) are the body's, which
        // ends in a return, the epilogue its fallback (not written after a
        // body that returns), as for a kernel that writes no h at all.
        if p + 1 < stmts.len() && !self.per_kernel_rows.iter().any(|r| r.0 == "store.h") {
            stmts.push(Tree::new("stmt.return_L0", self.stmt_ty("L0")?, vec![]));
            return Ok((stmts, Tree::fallback(ty(self, "block.O0")?)));
        }
        let after: Vec<Tree> = stmts.split_off(p + 1);
        if !after.is_empty() {
            return Err(format!("{} top-level statement(s) after the last store of h (the first: {}); the kernel's epilogue ends in that store", after.len(), after[0].name));
        }
        let last = stmts.pop().expect("the statement at p");
        // A continue at the top level has no loop. One inside a body loop is
        // the loop's own guard, read at its block (block.cont: the rest at the
        // conditional rung, no barrier after it); an epilogue loop's continue
        // before an earlier store is such a loop's (2026-10-12: 14 of 1,000).
        if stmts.iter().any(|s| s.name == "stmt.continue" || is_continue_guard(s)) {
            return Err(CONTINUE_OUTSIDE.into());
        }
        match self.epilogue_block(vec![last.clone()], 0) {
            Ok(e) => Ok((stmts, e)),
            // An interface with no output cell (no h) needs no epilogue: a last
            // store the epilogue cannot spell (inside an item loop, a loop that
            // breaks: five of the saved model kernels, 2026-10-17) stays in the
            // body, which ends in a return, the epilogue its fallback.
            Err(_) if !self.per_kernel_rows.iter().any(|r| r.0 == "store.h") => {
                let mut stmts = stmts;
                stmts.push(last);
                stmts.push(Tree::new("stmt.return_L0", self.stmt_ty("L0")?, vec![]));
                Ok((stmts, Tree::fallback(ty(self, "block.O0")?)))
            }
            Err(e) => Err(e),
        }
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
        // A continue its loop's block folded (block.cont, kept whole by the
        // flattening) that holds the store: the epilogue's own continue block,
        // as the unfolded guard reads (2026-10-12).
        let list: Vec<Tree> = list.into_iter().map(|t| if rung_stem(&t.name) == Some("stmt.block") && t.kids.first().is_some_and(|b| rung_stem(&b.name) == Some("block.cont")) { t.kids[0].clone() } else { t }).collect();
        if let [only] = list.as_slice() {
            if rung_stem(&only.name) == Some("block.cont") && has_store(only) {
                if d == 0 {
                    return Err("a continue in the kernel's epilogue outside its tile loops".into());
                }
                let o = format!("O{d}");
                return Ok(Tree::new(&format!("block.cont_{o}"), ty(self, &format!("block.{o}"))?, vec![only.kids[0].clone(), self.epilogue_block(vec![only.kids[1].clone()], d)?]));
            }
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
        // One statement that is a block of this rung is that block (as in
        // `block`; 2026-10-13, AST to AST).
        if prefix.is_empty() && last.name == format!("stmt.block_{o}") && last.kids.len() == 1 && last.kids[0].ty == ty(self, &format!("block.{o}"))? {
            return Ok(last.kids[0].clone());
        }
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
            // An output's store at any index (an inferred interface's row).
            (store, [i, v]) if store.starts_with("store.") => Ok(Tree::new(&format!("{store}_{o}"), st, vec![i.clone(), v.clone()])),
            ("stmt.if", [c, b]) => Ok(Tree::new(&format!("stmt.if_{o}"), st, vec![c.clone(), self.epilogue_block(vec![b.clone()], d)?])),
            ("stmt.if_else", [c, a, b]) => {
                if !has_store(a) || !has_store(b) {
                    return Err("the last store of h is in one branch of an if-else only; the kernel's epilogue stores on every path".into());
                }
                Ok(Tree::new(&format!("stmt.if_else_{o}"), st, vec![c.clone(), self.epilogue_block(vec![a.clone()], d)?, self.epilogue_block(vec![b.clone()], d)?]))
            }
            ("stmt.block", [b]) => Ok(Tree::new(&format!("stmt.block_{o}"), st, vec![self.epilogue_block(vec![b.clone()], d)?])),
            // A continue guard folded at its loop's block: the guard, then the
            // rest as an epilogue block (only inside an epilogue tile loop).
            ("block.cont", [c, rest]) => {
                if d == 0 {
                    return Err("a continue in the kernel's epilogue outside its tile loops".into());
                }
                let cont = Tree::new(&format!("block.cont_{o}"), ty(self, &format!("block.{o}"))?, vec![c.clone(), self.epilogue_block(vec![rest.clone()], d)?]);
                Ok(Tree::new(&format!("stmt.block_{o}"), st, vec![cont]))
            }
            // Conditional statements written in place (a loop with no barrier
            // read at the conditional rung): in the epilogue, a block like any.
            ("stmt.cblock", [b]) => Ok(Tree::new(&format!("stmt.block_{o}"), st, vec![self.epilogue_block(vec![b.clone()], d)?])),
            ("loop.tile" | "loop.tile_strided" | "loop.tilex" | "loop.tilex_strided", [lead @ .., body]) => {
                if d + 1 > 4 {
                    return Err("the last store of h is inside five nested tile loops (the epilogue has four)".into());
                }
                let mut k: Vec<Tree> = lead.iter().map(|b| self.any_bound(b)).collect::<Result<_, _>>()?;
                k.push(self.epilogue_block(vec![body.clone()], d + 1)?);
                Ok(Tree::new(&format!("{stem}_{o}"), st, k))
            }
            _ => Err(format!("the last store of h is inside {}, which has no epilogue spelling", s.name)),
        }
    }

    /// A statement before the epilogue's last, moved to the conditional rung
    /// of its tile depth (`C<d>`; a tile loop's body one deeper). A barrier,
    /// a return or an item loop has no conditional spelling.
    /// A loop's bound where any bound is taken (a C or O rung): a uniform
    /// bound read at an L rung (`bound.of_uniform`, u32.boundu) is its
    /// uncapped twin there (`bound.x_of_uniform`, u32.boundx); the same text.
    fn any_bound(&self, b: &Tree) -> Result<Tree, String> {
        if b.name == "bound.of_uniform" {
            return Ok(Tree::new("bound.x_of_uniform", ty(self, "u32.boundx")?, b.kids.clone()));
        }
        Ok(b.clone())
    }

    fn rerung(&self, t: &Tree, d: usize) -> Result<Tree, String> {
        if t.name == "bound.of_uniform" {
            return self.any_bound(t);
        }
        if t.name == "stmt.continue" {
            return Err("a continue that is not a guard `if (c) { continue; }` where only conditional statements are spelled (the rest of a loop body after a continue, or the epilogue before its store)".into());
        }
        let Some(stem) = rung_stem(&t.name) else { return Ok(t.clone()) };
        // (A reduction holds barriers: it is never conditional either.)
        if stem == "stmt.barrier" || stem == "stmt.storage_barrier" || stem == "stmt.subgroup_barrier" || stem == "stmt.return" || stem.starts_with("loop.items") || stem.starts_with("wg.reduce_") {
            return Err(format!("{} where only conditional statements are spelled (after a loop's continue guard, or before the epilogue's last store of h)", t.name));
        }
        // A test of two uniform values (stmt.if_ule has L rows only) is, at
        // a conditional rung, the if of their comparison (the rest of a loop
        // body after a continue guard, an epilogue loop's statements: 10 of
        // 1,000 random trees, 2026-10-14).
        if stem == "stmt.if_uflag" || stem == "stmt.if_else_uflag" {
            let [f, bodies @ ..] = t.kids.as_slice() else { return Err(format!("{} with no children", t.name)) };
            let c = format!("C{d}");
            let row = if bodies.len() == 2 { "if_else" } else { "if" };
            let mut kids = vec![self.flag_of_uniform(f)?];
            for b in bodies {
                kids.push(self.rerung(b, d)?);
            }
            return Ok(Tree::new(&format!("stmt.{row}_{c}"), ty(self, &format!("stmt.{c}"))?, kids));
        }
        if let Some(cmp) = crate::wgsl::render::uniform_test_compare(stem) {
            let [x, y, bodies @ ..] = t.kids.as_slice() else { return Err(format!("{} with {} children", t.name, t.kids.len())) };
            let index = |u: &Tree| -> Result<Tree, String> { Ok(Tree::new("index.of_uniform", ty(self, "u32.index")?, vec![u.clone()])) };
            let le = Tree::new(cmp, ty(self, "bool.flag")?, vec![index(x)?, index(y)?]);
            let c = format!("C{d}");
            let row = if bodies.len() == 2 { "if_else" } else { "if" };
            let mut kids = vec![le];
            for b in bodies {
                kids.push(self.rerung(b, d)?);
            }
            return Ok(Tree::new(&format!("stmt.{row}_{c}"), ty(self, &format!("stmt.{c}"))?, kids));
        }
        // Conditional statements written in place at an L rung are, at a
        // conditional rung, a block like any (stmt.cblock has L rows only).
        if stem == "stmt.cblock" {
            let [b] = t.kids.as_slice() else { return Err(format!("{} with {} children", t.name, t.kids.len())) };
            let c = format!("C{d}");
            return Ok(Tree::new(&format!("stmt.block_{c}"), ty(self, &format!("stmt.{c}"))?, vec![self.rerung(b, d)?]));
        }
        // A loop that breaks: its body, the break block B<d+1> (conditional
        // statements of the next depth, the guard, conditional statements),
        // moved with it (it kept its old depth's B and C before; 2026-10-12).
        // (And the stepped one, its stride kept: 12 of 1,000 random trees, the
        // first time the sampler drew it, kept the old depth's B, 2026-10-17.)
        if matches!(stem, "loop.tile_brk" | "loop.tilex_brk" | "loop.tilex_strided_brk") {
            if d + 1 > 5 {
                return Err("a loop that breaks six deep".into());
            }
            let (bound, stride, body) = match t.kids.as_slice() {
                [bound, body] => (bound, None, body),
                [bound, stride, body] => (bound, Some(stride), body),
                _ => return Err(format!("{} with {} children", t.name, t.kids.len())),
            };
            let [before, cond, after] = body.kids.as_slice() else { return Err(format!("{} with {} children", body.name, body.kids.len())) };
            let b = format!("B{}", d + 1);
            let brk = Tree::new(&format!("block.brk_{b}"), ty(self, &format!("block.{b}"))?, vec![self.rerung(before, d + 1)?, cond.clone(), self.rerung(after, d + 1)?]);
            let c = format!("C{d}");
            let mut kids = vec![self.any_bound(bound)?];
            kids.extend(stride.cloned());
            kids.push(brk);
            return Ok(Tree::new(&format!("{stem}_{c}"), ty(self, &format!("stmt.{c}"))?, kids));
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
    /// The emitter's unrolled workgroup reductions read back as the rows
    /// (2026-10-13, AST to AST: 47 of 1,000 random trees read a reduction
    /// back as a barrier and its steps): a barrier, then `if (lidx < s)
    /// { a[lidx] = op(a[lidx], a[lidx + s]); }` and a barrier for s halving
    /// to 1, over one array and one op, is wg.reduce_<op>(a, 2 s).
    fn fold_reductions(&self, stmts: Vec<Tree>, rung: &str) -> Result<Vec<Tree>, String> {
        let barrier = format!("stmt.barrier_{rung}");
        let mut out = Vec::new();
        let mut i = 0;
        while i < stmts.len() {
            let mut steps: Vec<(Tree, &str, f32, Option<Tree>)> = Vec::new();
            if stmts[i].name == barrier {
                let mut j = i + 1;
                while let (Some(step), Some(b)) = (stmts.get(j).and_then(reduction_step), stmts.get(j + 1)) {
                    let follows = steps.last().is_none_or(|(a, op, s, ix)| a.name == step.0.name && *op == step.1 && step.2 * 2.0 == *s && ix.as_ref().map(|x| &x.name) == step.3.as_ref().map(|x| &x.name));
                    if b.name != barrier || !follows {
                        break;
                    }
                    steps.push(step);
                    j += 2;
                }
            }
            let row = steps.first().map(|(a, op, _, ix)| (format!("wg.reduce_{op}_{rung}"), a.clone(), ix.clone()));
            let ok = steps.last().is_some_and(|(_, _, s, _)| *s == 1.0) && row.as_ref().is_some_and(|(r, _, _)| self.kingdom.rows.iter().any(|x| x.name == *r));
            match row.filter(|_| ok) {
                Some((r, a, ix)) => {
                    let n = Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, steps[0].2 * 2.0);
                    out.push(Tree::new(&r, self.stmt_ty(rung)?, std::iter::once(a).chain(ix).chain(std::iter::once(n)).collect()));
                    i += 1 + 2 * steps.len();
                }
                None => {
                    out.push(stmts[i].clone());
                    i += 1;
                }
            }
        }
        Ok(out)
    }

    /// The emitter's workgroup puts read back as the rows (2026-10-14): a
    /// barrier, `if (i < len) { a[i] = v; }` with i this thread's element
    /// (lidx + k * threads, k a literal or a tile counter) and len the
    /// array's, then a barrier, is wg.put<kind>(a, own, v).
    fn fold_puts(&self, stmts: Vec<Tree>, rung: &str) -> Result<Vec<Tree>, String> {
        let barrier = format!("stmt.barrier_{rung}");
        let threads = self.grid.0 * self.grid.1;
        // (The builtin itself before the axes are renamed, index.of_lidx after.)
        let lidx = |t: &Tree| (t.name == "builtin.local_invocation_index" && t.kids.is_empty()) || (t.name == "index.of_lidx" && t.kids.len() == 1);
        // A constant: a literal, or a sum or product of constants (`lit_u32(0u)
        // * 370u`, a tile counter written outside its loop).
        fn constant(t: &Tree) -> Option<f32> {
            match (t.name.as_str(), t.kids.as_slice()) {
                ("index.add", [a, b]) => Some(constant(a)? + constant(b)?),
                ("index.mul", [a, b]) => Some(constant(a)? * constant(b)?),
                _ => literal_value(t),
            }
        }
        // The own element an index is, or None.
        let own = |ctx: &Self, i: &Tree, len: u32| -> Option<Tree> {
            let wgown = ty(ctx, "u32.wgown").ok()?;
            let lit = |c: u32| -> Option<Tree> { Some(Tree::new("safe.wg_own", wgown, vec![Tree::literal("literal.lit64", ty(ctx, "u32.lit64").ok()?, c as f32)])) };
            // (`lidx + 0u` read as lidx.)
            if lidx(i) {
                return lit(0);
            }
            let ([l, k], "index.add") = (i.kids.as_slice(), i.name.as_str()) else { return None };
            let k = if lidx(l) { k } else if lidx(k) { l } else { return None };
            if let Some(c) = constant(k) {
                let c = c as u32;
                if threads == 0 || !c.is_multiple_of(threads) || c / threads >= 64 {
                    return None;
                }
                return lit(c / threads);
            }
            let ("index.mul", [a, b]) = (k.name.as_str(), k.kids.as_slice()) else { return None };
            let (counter, t) = if literal_value(b).is_some() { (a, b) } else { (b, a) };
            // (The emitter's min(k, ceil(len / threads)), which keeps the
            // product from wrapping: the counter itself.)
            let counter = match (counter.name.as_str(), counter.kids.as_slice()) {
                ("index.min", [x, c]) if literal_value(c) == Some(len.div_ceil(threads.max(1)) as f32) => x,
                _ => counter,
            };
            if literal_value(t) != Some(threads as f32) || counter.kids.len() != 1 {
                return None;
            }
            let row = match (counter.name.as_str(), counter.kids[0].name.as_str()) {
                ("safe.tile_index", "safe.tile_counter") => "safe.wg_own_tile",
                ("safe.tile_index2", "safe.tile_counter2") => "safe.wg_own_tile2",
                _ => return None,
            };
            Some(Tree::new(row, wgown, vec![counter.kids[0].clone()]))
        };
        let put = |ctx: &Self, s: &Tree| -> Option<Tree> {
            let [c, b] = s.kids.as_slice() else { return None };
            if rung_stem(&s.name) != Some("stmt.if") || c.name != "compare.lt" || c.kids.len() != 2 {
                return None;
            }
            let mut inner = Vec::new();
            flatten_stmts(b.clone(), &mut inner);
            let [set] = inner.as_slice() else { return None };
            let row = match rung_stem(&set.name)? {
                "wg.set" => "wg.put",
                "wg.set_v" => "wg.put_v",
                "wg.set_v2" => "wg.put_v2",
                "wg.set_u" => "wg.put_u",
                "wg.set_n" => "wg.put_n",
                _ => return None,
            };
            let [arr, idx, v] = set.kids.as_slice() else { return None };
            let i = idx.kids.first()?;
            let len = self.array_lens.get(arr.name.as_str()).copied().unwrap_or_else(|| crate::wgsl::render::array_len(&arr.name.replace('.', "_")));
            if *i != c.kids[0] || literal_value(&c.kids[1])? as u32 != len {
                return None;
            }
            let own = own(ctx, i, len)?;
            Some(Tree::new(&format!("{row}_{rung}"), ctx.stmt_ty(rung).ok()?, vec![arr.clone(), own, v.clone()]))
        };
        let mut out = Vec::new();
        let mut i = 0;
        while i < stmts.len() {
            if stmts[i].name == barrier && stmts.get(i + 2).is_some_and(|b| b.name == barrier) {
                if let Some(p) = put(self, &stmts[i + 1]) {
                    out.push(p);
                    i += 3;
                    continue;
                }
            }
            out.push(stmts[i].clone());
            i += 1;
        }
        Ok(out)
    }

    /// An unrolled item loop's main loop and the loop after it that
    /// finishes the items (`for (; t < bound; t = t + 1u)`, the same body)
    /// as loop.items_unrolled_<bound> (2026-10-13: the parser dropped the
    /// counter's steps between the copies and read the copies all at t).
    fn fold_unrolled(&self, stmts: Vec<Tree>) -> Result<Vec<Tree>, String> {
        let mut out = Vec::new();
        let mut it = stmts.into_iter().peekable();
        while let Some(t) = it.next() {
            if t.name != UNROLLED_MAIN {
                out.push(t);
                continue;
            }
            let rest = it.next().ok_or("an unrolled item loop without the loop that finishes its items")?;
            let kind = rest.name.strip_prefix("loop.items_cont_").filter(|k| k.starts_with("bound")).ok_or_else(|| format!("an unrolled item loop followed by {}, not the loop that finishes its items", rest.name))?;
            let ([unroll, main_body], [bound, body]) = (t.kids.as_slice(), rest.kids.as_slice()) else { return Err("an unrolled item loop of the wrong shape".into()) };
            let (mut a, mut b) = (Vec::new(), Vec::new());
            flatten_stmts(main_body.clone(), &mut a);
            flatten_stmts(body.clone(), &mut b);
            if !a.iter().map(Tree::to_sexpr).eq(b.iter().map(Tree::to_sexpr)) {
                return Err("an unrolled item loop whose copies are not the body of the loop that finishes its items".into());
            }
            out.push(Tree::new(&format!("loop.items_unrolled_{kind}"), t.ty, vec![unroll.clone(), bound.clone(), body.clone()]));
        }
        Ok(out)
    }

    fn block(&self, stmts: Vec<Tree>, rung: &str) -> Result<Tree, String> {
        let bt = ty(self, &format!("block.{rung}"))?;
        let st = self.stmt_ty(rung)?;
        let stmts = if rung.starts_with('L') { self.fold_puts(self.fold_reductions(stmts, rung)?, rung)? } else { stmts };
        let stmts = self.fold_unrolled(stmts)?;
        if stmts.is_empty() {
            return Ok(Tree::fallback(bt));
        }
        // `if (c) { continue; }` inside a loop (an L or C rung below the top):
        // the guard, then the rest of the block at the conditional rung of its
        // depth (no barrier after a continue; every thread still reaches the
        // next iteration).
        let looped_rung = matches!(rung.as_bytes(), [b'L' | b'C', d] if (b'1'..=b'5').contains(d));
        // (A continue whose rest stores h too: a body loop's store is a
        // conditional statement after it; the epilogue reads the folded block
        // into block.cont_O<d>, 2026-10-12.)
        if looped_rung {
            // (Not over a break guard after it: the loop's body splits at the
            // break first, then each side's continue folds; 2026-10-12.)
            let breaks_after = |at: usize| stmts[at + 1..].iter().any(|t| break_guard(t, self).is_some());
            if let Some(at) = stmts.iter().position(is_continue_guard).filter(|&at| !breaks_after(at)) {
                let mut before = stmts;
                let rest: Vec<Tree> = before.split_off(at + 1);
                let guard = before.pop().expect("the guard at `at`");
                let d: usize = rung[1..].parse().unwrap_or(1);
                // A later continue guard in the rest is the rest's own (the
                // conditional rung's block reads it: block.cont_C<d>).
                // The rest the conditional rung spells; from the first statement
                // it does not (a barrier, a reduction: the kingdom's continue in
                // a conditional block before them, under a uniform condition),
                // the block goes on at its own rung after the continue's block
                // (the same text: the continue skips both; 2026-10-12).
                let mut spelled = Vec::new();
                let mut tail: Vec<Tree> = Vec::new();
                for (i, t) in rest.iter().enumerate() {
                    let r = if is_continue_guard(t) { Ok(t.clone()) } else { self.rerung(t, d) };
                    match r {
                        Ok(x) => spelled.push(x),
                        Err(_) => {
                            tail = rest[i..].to_vec();
                            break;
                        }
                    }
                }
                let cont = Tree::new(&format!("block.cont_{rung}"), bt, vec![guard.kids[0].clone(), self.block(spelled, &format!("C{d}"))?]);
                if before.is_empty() && tail.is_empty() {
                    return Ok(cont);
                }
                before.push(Tree::new(&format!("stmt.block_{rung}"), st, vec![cont]));
                before.extend(tail);
                return self.block(before, rung);
            }
        }
        let mut items = stmts;
        // A block of one statement that is a block (a continue's, kept whole
        // when a list was flattened) is that block (2026-10-13, AST to AST:
        // block.one(stmt.block(block.cont ..)) was read where the block
        // itself was written).
        if let [one] = items.as_slice() {
            if one.name == format!("stmt.block_{rung}") && one.kids.len() == 1 && one.kids[0].ty == bt {
                return Ok(one.kids[0].clone());
            }
        }
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
        // One store, after the calls and subgroup operations its value reads
        // (naga writes each as a statement of its own: the kingdom's helpers,
        // lit_f32 and the rest, and a subgroup operation; the emitter writes
        // the store's value inside the guard, and the parser reads a result
        // one statement reads inline in it, so it stays there), and the
        // `return` naga writes at the end of a kernel's last branch
        // (2026-10-13, AST to AST: 170 of 1,000 random trees).
        let mut stmts: Vec<&Statement> = accept.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        if matches!(stmts.last(), Some(Statement::Return { value: None })) {
            stmts.pop();
        }
        let Some((last, before)) = stmts.split_last() else { return Ok(false) };
        // (And an atomic whose result the store's value reads, `h[c] = h[c] +
        // phi[atomicXor(&a[0], x)]`: the emitter writes it inside the guard
        // too, and read as an if its guard was written again at every
        // rendering, 2026-10-17: 20 of the 83 random trees that missed the
        // fixpoint.)
        let read_inline = |s: &Statement| matches!(s, Statement::Call { result: Some(_), .. } | Statement::SubgroupCollectiveOperation { .. } | Statement::SubgroupBallot { .. } | Statement::SubgroupGather { .. } | Statement::Atomic { .. });
        // (The else branch empty, or only that `return`.)
        let reject_empty = reject.iter().all(|s| matches!(s, Statement::Emit(_) | Statement::Return { value: None }));
        if !matches!(last, Statement::Store { .. }) || !before.iter().all(|s| read_inline(s)) || !reject_empty {
            return Ok(false);
        }
        let mut p = here.to_vec();
        p.push(0);
        p.push(before.len());
        let Some(root) = self.root_at(&p, |k| matches!(k, RootKind::Store { .. })) else { return Ok(false) };
        let Node::App { name, kids, .. } = &root.tree else { return Ok(false) };
        if !name.starts_with("store.buffer.h.#") || kids.len() != 2 {
            return Ok(false);
        }
        // A cell that reads a subgroup result made inside the guard (or a
        // call made there over one, as an argument) is not the cell the
        // guard tests (naga made it again there): an if. Calls over other
        // values are pure and read the same inside as before.
        let mut inside: Vec<usize> = Vec::new();
        for s in before {
            match s {
                Statement::SubgroupCollectiveOperation { result, .. } | Statement::SubgroupGather { result, .. } | Statement::SubgroupBallot { result, .. } | Statement::Atomic { result: Some(result), .. } => inside.push(result.index()),
                Statement::Call { result: Some(r), arguments, .. } if arguments.iter().any(|a| inside.contains(&a.index())) => inside.push(r.index()),
                _ => {}
            }
        }
        fn mentions(n: &Node, inside: &[usize]) -> bool {
            match n {
                Node::Leaf { name, .. } => (name.starts_with("subgroup@") || name.starts_with("call.")) && name.rsplit_once('@').and_then(|(_, i)| i.parse::<usize>().ok()).is_some_and(|i| inside.contains(&i)),
                Node::Literal { .. } => false,
                Node::App { kids, .. } => kids.iter().any(|k| mentions(k, inside)),
            }
        }
        if mentions(&kids[0], &inside) {
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
        let own = lt(a, "load.uniform.cfg.r_rows") == Some(row.clone()) && lt(b, "load.uniform.cfg.d") == Some(col.clone()) && off_row == Some(row.clone());
        // The mirror's store (h[C * d + R]) under its own cell's guard.
        let own = own || (lt(a, "load.uniform.cfg.r_rows") == Some(col.clone()) && lt(b, "load.uniform.cfg.d") == Some(row) && off_row == Some(col));
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
        // A helper called from a helper being inlined (13 of the saved model
        // kernels' `do_row` calling `fmix`, 2026-10-17): inlined in turn, its
        // locals and its result named under both (`do_row/fmix/h@0`); WGSL
        // has no recursion, so the nesting ends.
        let mut args: Vec<(usize, Node)> = self.arg_roots.get(here).cloned().unwrap_or_default();
        args.sort_by_key(|(p, _)| *p);
        if args.len() != helper.args.len() {
            return Err(format!("{callee}: {} arguments at the call, {} declared", args.len(), helper.args.len()));
        }
        let mut bound = HashMap::new();
        let mut values = HashMap::new();
        let mut pointers = HashMap::new();
        let mut array_args = HashMap::new();
        for (i, ((_, node), arg)) in args.into_iter().zip(&helper.args).enumerate() {
            // A pointer parameter given `&x`: the helper reads and writes the
            // caller's register for x.
            if let naga::TypeInner::Pointer { base, .. } = self.module.types[helper.naga_fn.arguments[i].ty].inner {
                let Node::Leaf { name: local, .. } = &node else { return Err(format!("{callee}: pointer argument {i} is not a local (&x)")) };
                if !local.starts_with("local.") || local.contains(".#") {
                    return Err(format!("{callee}: pointer argument {i} is {local}, not a local held in a register"));
                }
                // (To a local array, `ptr<function, array<f32, 16>>` given `&acc`:
                // the caller's array itself, v24 run 1's seeds 00 and 31.)
                if matches!(self.module.types[base].inner, naga::TypeInner::Array { .. }) {
                    array_args.insert(arg.clone(), local.clone());
                    continue;
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
        let saved_array_args = std::mem::replace(&mut self.array_args, array_args);
        let result_reg = match result {
            Some(idx) => {
                let d = helper.ret_dual.as_deref().ok_or_else(|| format!("{callee} returns a value no register holds"))?;
                let row = self.register(&format!("local.call.{}{callee}@{idx}", self.local_prefix), d)?;
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
        let saved_bound = std::mem::replace(&mut self.bound_lets, helper.bound_lets.clone());
        let saved_stale = std::mem::replace(&mut self.stale_lets, helper.stale_lets.clone());
        let saved_let_regs = std::mem::take(&mut self.let_regs);
        let saved_calls = std::mem::replace(&mut self.calls, &helper.calls);
        let saved_arg_roots = std::mem::replace(&mut self.arg_roots, &helper.arg_roots);
        let saved_fn = std::mem::replace(&mut self.naga_fn, helper.naga_fn);
        let saved_lanes = std::mem::replace(&mut self.local_lanes, helper.local_lanes.clone());
        let saved_args = self.inline_args.replace(bound);
        self.outer_args.push(saved_args.clone());
        let saved_values = self.inline_values.replace(values);
        let saved_tail = std::mem::replace(&mut self.tail, true);
        let saved_result = std::mem::replace(&mut self.inline_result, result_reg);
        let nested = format!("{}{callee}/", self.local_prefix);
        let saved_prefix = std::mem::replace(&mut self.local_prefix, nested);
        let out = self.inline_body(helper, rung);
        self.local_prefix = saved_prefix;
        self.pointer_args = saved_pointers;
        self.array_args = saved_array_args;
        self.inline_result = saved_result;
        self.tail = saved_tail;
        self.outer_args.pop();
        self.inline_args = saved_args;
        self.inline_values = saved_values;
        self.local_lanes = saved_lanes;
        self.naga_fn = saved_fn;
        self.arg_roots = saved_arg_roots;
        self.calls = saved_calls;
        self.lets = saved_lets;
        self.bound_lets = saved_bound;
        self.stale_lets = saved_stale;
        self.let_regs = saved_let_regs;
        self.roots = saved_roots;
        let mut stmts = out.map_err(|e| format!("inlining {callee}: {e}"))?;
        let st = self.stmt_ty(rung)?;
        Ok(match stmts.len() {
            0 => None,
            1 => stmts.pop(),
            _ => Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(stmts, rung)?])),
        })
    }

    /// The local array `local` of `n` elements of `elem` zeroed element by
    /// element (WGSL's zero for a local declared without an initialiser).
    fn zero_array(&mut self, local: &str, elem: &str, n: usize, rung: &str) -> Result<Vec<Tree>, String> {
        let st = self.stmt_ty(rung)?;
        let Some(zero) = self.zero_of_dual(elem)? else {
            return Err(format!("the local array {local} of {elem} has no initialiser and no zero"));
        };
        let (arr, kind, mask) = self.private_array(local, elem)?;
        let mut stmts = Vec::new();
        for i in 0..n {
            let idx = Tree::literal("literal.index", ty(self, "u32.index")?, i as f32);
            let pidx = Tree::new(mask, ty(self, pindex_dual(mask))?, vec![idx]);
            stmts.push(Tree::new(&format!("parr.set_{kind}_{rung}"), st, vec![Tree::new(arr, ty(self, parr_dual_of(arr, kind))?, vec![]), pidx, zero.clone()]));
        }
        Ok(stmts)
    }

    /// The zero of a value dual: a literal of a scalar's form, a splat of one for
    /// a vector's. None for a dual with no zero (a matrix, an opaque form).
    fn zero_of_dual(&self, dual: &str) -> Result<Option<Tree>, String> {
        let Some((slot, form)) = dual.split_once('.') else { return Ok(None) };
        let (lanes, scalar) = match slot.strip_prefix("vec") {
            Some(r) => (r[..1].parse::<usize>().ok(), r[2..r.len() - 1].to_string()),
            None => (None, slot.to_string()),
        };
        let (lit, scalar_dual) = match scalar.as_str() {
            "f32" => ("literal.real", "f32.real"),
            "i32" => ("literal.int", "i32.int"),
            "u32" => (if form == "count" { "literal.count" } else if form == "bits" { "literal.bits" } else { "literal.index" }, if form == "count" { "u32.count" } else if form == "bits" { "u32.bits" } else { "u32.index" }),
            "bool" => ("literal.flag", "bool.flag"),
            _ => return Ok(None),
        };
        let zero = Tree::literal(lit, ty(self, scalar_dual)?, 0.0);
        Ok(Some(match lanes {
            None => zero,
            Some(n @ 2..=4) => Tree::new(&format!("shape.splat{n}"), ty(self, dual)?, vec![zero]),
            Some(_) => return Ok(None),
        }))
    }

    fn inline_body(&mut self, helper: &'a Helper<'a>, rung: &str) -> Result<Vec<Tree>, String> {
        let st = self.stmt_ty(rung)?;
        let mut stmts = Vec::new();
        for (local, dual) in &helper.uninit {
            if let Some((elem, n)) = dual.strip_prefix("array:").and_then(|r| r.rsplit_once(':')) {
                let n: usize = n.parse().map_err(|e| format!("the local array {local}: {e}"))?;
                stmts.extend(self.zero_array(local, elem, n, rung).map_err(|e| format!("{e} (not inlined)"))?);
                continue;
            }
            let Some(zero) = self.zero_of_dual(dual)? else {
                return Err(format!("the local {local} ({dual}) has no initialiser and no register (not inlined)"));
            };
            let row = self.register(local, dual)?;
            let kind = row.chars().nth(4).unwrap_or('f');
            stmts.push(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), zero]));
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
            if self.skip_stmts.remove(&here) {
                path.pop();
                continue;
            }
            // The lets defined just before this statement whose uses sit in a
            // deeper loop: set once here, as the source computes them.
            // (A helper inlined as statements binds its own, 2026-10-17: `let v =
            // wred[0]; workgroupBarrier(); return v;` read wred[0] after the
            // barrier when inlined at the return, run 1's seeds' reductions.)
            {
                let mut here_lets: Vec<String> = self.bound_lets.iter().filter(|(_, p)| **p == here).map(|(n, _)| n.clone()).collect();
                here_lets.sort_by_key(|n| n.rsplit('@').next().and_then(|i| i.parse::<u32>().ok()).unwrap_or(0));
                for n in here_lets {
                    let def = self.lets.get(&n).cloned().ok_or_else(|| format!("let {n} has no definition"))?;
                    // An array let (`let accs = array<vec4<f32>, 4>(...)`) has no
                    // single value: its indexed uses are read in place, as before.
                    if matches!(&def, Node::App { name, .. } if name.starts_with("naga.Compose.type")) {
                        if self.stale_lets.contains(&n) {
                            return Err(format!("let {n} is an array that reads a local a store changes before its use: no register holds it"));
                        }
                        self.notes.push(format!("let {n} is an array used in a deeper loop: its elements are read in place"));
                        continue;
                    }
                    let t = tree_of(self, &def)?;
                    let d = self.kingdom.dual(t.ty).map(|d| d.dual.clone()).unwrap_or_default();
                    if !matches!(d.as_str(), "f32.real" | "vec4<f32>.real" | "u32.index" | "u32.bits" | "u32.count" | "bool.flag") {
                        if self.stale_lets.contains(&n) {
                            return Err(format!("let {n} ({d}) reads a local a store changes before its use, and no register of its dual holds its value"));
                        }
                        // No register of this dual (a vec2, a vec3, a matrix): the
                        // let stays inlined at its uses, and the note says so.
                        self.notes.push(format!("let {n} ({d}) is used in a deeper loop but has no register kind: inlined at its uses"));
                        continue;
                    }
                    let v = self.value(&t)?;
                    let row = self.register(&format!("local.{}let.{n}", self.local_prefix), &d).map_err(|e| format!("let {n}, used in a deeper loop, bound to a register: {e}"))?;
                    let kind = row.chars().nth(4).unwrap_or('f');
                    out.push(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), v]));
                    // The ownership proof sees through the register to what it
                    // holds, as for a local (a let never changes in its scope).
                    let held = self.expand_memo(&t);
                    if held.size() <= 512 {
                        self.axis_memo.insert(format!("local.{}let.{n}", self.local_prefix), (held, self.loop_depth));
                    }
                    self.let_regs.insert(n, row);
                }
            }
            // A branch that ends in a return, statements after it (`if (cfg.pass_index
            // == 0u) { ..; return; } ..`, 6 of run 1's seeds, 2026-10-17): the rest
            // of the block is its else, `if (c) { .. } else { .. }`, the same
            // statements on every path. A workgroup-uniform test over a barrier
            // keeps its rung (the uniform predicate); any other reads its bodies
            // at the conditional rung. (A return alone is the thread guard.)
            if let Some(t) = self.early_return_branch(&BlockAt { block, position, last, next: index }, &here, rung)? {
                out.push(t);
                path.pop();
                return Ok(out);
            }
            if self.built.get() > READ_NODE_BOUND {
                return Err(format!("our software has a bug: reading this kernel built more than {READ_NODE_BOUND} nodes (an expansion without end); the read is stopped"));
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
                    Statement::Store { .. } | Statement::ImageStore { .. } => {
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
                        self.subgroup_statement(root, result.index() as u32, rung)
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
                        // The result stored into a local at once and read by that
                        // store only (`i0 = workgroupUniformLoad(&wg_ua[0u])`, the
                        // kingdom's own rendering): that local's register, the
                        // store passed over, so the text reads back the same.
                        let mut held = format!("local.uniformload@{}", result.index());
                        let next = block.iter().enumerate().skip(position + 1).find(|(_, s)| !matches!(s, Statement::Emit(_)));
                        if let Some((_, Statement::Store { pointer, value })) = next {
                            if let (Expression::LocalVariable(l), true) = (&self.naga_fn.expressions[*pointer], value == result) {
                                let name = format!("local.{}@{}", self.naga_fn.local_variables[*l].name.clone().unwrap_or_else(|| "local".into()), l.index());
                                fn count(n: &Node, leaf: &str) -> usize {
                                    match n {
                                        Node::Leaf { name, .. } => usize::from(name == leaf),
                                        Node::App { kids, .. } => kids.iter().map(|k| count(k, leaf)).sum(),
                                        _ => 0,
                                    }
                                }
                                let leaf = format!("uniformload@{}", result.index());
                                let readers = self.roots.values().flatten().filter(|r| count(&r.tree, &leaf) > 0).count() + self.arg_roots.values().filter(|args| args.iter().any(|(_, n)| count(n, &leaf) > 0)).count();
                                if readers == 1 && !self.lets.values().any(|t| count(t, &leaf) > 0) {
                                    let mut at = here.clone();
                                    if let Some(last) = at.last_mut() {
                                        *last += 1;
                                    }
                                    self.skip_stmts.insert(at);
                                    held = prefixed(&self.local_prefix, &name);
                                }
                            }
                        }
                        let row = self.register(&held, &d)?;
                        self.registers.insert(format!("local.uniformload@{}", result.index()), row.clone());
                        let kind = row.chars().nth(4).unwrap_or('f');
                        // Of a u32 or f32 workgroup array's element: the kingdom's
                        // own uniform load (one statement, its register uniform
                        // when the index is, as the checker proves).
                        if let ("wg.get_u_safe" | "wg.get_safe", [arr, at], 'i' | 'f') = (v.name.as_str(), v.kids.as_slice(), kind) {
                            if v.ty == ty(self, if kind == 'i' { "u32.index" } else { "f32.real" })? {
                                return Ok(Some(Tree::new(&format!("wg.uload_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), arr.clone(), at.clone()])));
                            }
                        }
                        let barrier = || Tree::new(&format!("stmt.barrier_{rung}"), st, vec![]);
                        let set = Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), v]);
                        Ok(Some(Tree::new(&format!("stmt.block_{rung}"), st, vec![self.block(vec![barrier(), set, barrier()], rung)?])))
                    }
                    Statement::Atomic { result: Some(result), .. } => {
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Atomic { .. })).ok_or("no atomic root at this statement")?;
                        self.atomic_statement(root, result.index() as u32, rung)
                    }
                    Statement::Atomic { result: None, .. } => Err("an atomic without a result (a 64-bit min or max) has no row".into()),
                    Statement::Call { function, result, .. } => {
                        // A pure helper's call: its result is inlined where it
                        // is read, the call statement itself has no effect. A
                        // helper with statements is inlined here.
                        let callee = self.module.functions[*function].name.clone().unwrap_or_else(|| format!("fn{}", function.index()));
                        let helper = self.helpers.get(&callee).ok_or_else(|| format!("a call of {callee}, which the grammar cannot inline"))?;
                        if helper.pure || helper.composite || helper.row.is_some() {
                            return Ok(None);
                        }
                        self.inline_call(&callee, helper, result.map(|r| r.index() as u32), &here, rung)
                    }
                    Statement::Return { value: Some(_) } => {
                        // A helper's return, inlined: its value to the call's register.
                        let (row, kind) = self.inline_result.clone().ok_or("a return of a value outside a helper")?;
                        // The helper's own statements are walked from its body
                        // (paths from there): a return at its top level ends it
                        // on every path, however conditional the CALL is (78 of
                        // the 1,798 saved model kernels called `fmix` under an
                        // if and were refused it until 2026-10-17); one inside
                        // the helper's own if, loop or block is not inlined.
                        if here.len() != 1 && self.early_return_at.as_deref() != Some(here.as_slice()) {
                            return Err(format!("a return {} of a helper (not inlined)", if cond { "under a condition" } else { "inside a block" }));
                        }
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Return)).ok_or("no return root at this statement")?;
                        let Node::App { kids, .. } = &root.tree else { return Err("a return root without a value".into()) };
                        let t = tree_of(self, &kids[0])?;
                        let v = self.value(&t)?;
                        Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), v])))
                    }
                    Statement::If { accept, reject, .. } => {
                        let root = self.root_at(&here, |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at this if")?;
                        let cond_tree = tree_of(self, &root.tree)?;
                        let c = self.value(&cond_tree.kids[0])?;
                        // The thread guard: `if (c) { return; }` at L0 with nothing else,
                        // at the entry point's top level. (One inside a branch guards
                        // that branch only: seed02 of v24 run 1 returned from its pass
                        // 0 branch, read as a guard on the stores of pass 1 too.)
                        let is_return_only = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Return { value: None }));
                        if rung == "L0" && here.len() == 1 && self.branch_depth == 0 && is_return_only(accept) && reject.iter().all(|s| matches!(s, Statement::Emit(_))) {
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
                        // A test of two workgroup-uniform values at an L rung (the
                        // symmetric half's `if (wid_row <= wid_col)`): its body keeps
                        // the rung, a barrier in it in uniform control.
                        // (And ==, != and <, > and >= by their operands swapped,
                        // 2026-10-17: v24 seeds' `if (cfg.pass_index == 0u)`. A
                        // literal 0 is no uniform value: `x == 0u` is `x < 1u`,
                        // `x != 0u` and `x > 0u` are `1u <= x`.)
                        let uniform_le = self.simple_test(&c)?;
                        // (Not a continue's or a break's guard, `if (cfg.d >= nwg.x)
                        // { continue; }`: no row continues at an L rung; read as the
                        // conditional it is. A sketch draw, 2026-10-14.)
                        let exits = accept.iter().any(|s| matches!(s, Statement::Continue | Statement::Break));
                        let exits_else = reject.iter().any(|s| matches!(s, Statement::Continue | Statement::Break));
                        // (Only <= is read so without a barrier, as before; the
                        // others only when a body needs its rung, a barrier in
                        // it: the text `if (a < b)` is otherwise the plain if,
                        // as the kingdom writes one: 2 of 1,000 random trees.)
                        // (Or an item loop, the kingdom's loop over t, which only the
                        // top rung spells: drawn in a uniform test's body, it read at
                        // the conditional rung as a store to its counter, 2026-10-17.)
                        let needed = holds_barrier(self.module, accept) || holds_barrier(self.module, reject) || holds_item_loop(self.naga_fn, accept) || holds_item_loop(self.naga_fn, reject);
                        let was_le = matches!(c.name.as_str(), "compare.le" | "compare.ge");
                        let has_else = !reject.iter().all(|s| matches!(s, Statement::Emit(_)));
                        if let Some((op, x, y)) = uniform_le.filter(|(op, _, _)| rung.starts_with('L') && !exits && !exits_else && (needed || (*op == "ule" && was_le && !has_else))) {
                            if let (Some(ux), Some(uy)) = (self.uniform_bound_of(&x)?, self.uniform_bound_of(&y)?) {
                                self.tail = stmt_tail;
                                path.push(0);
                                let a = self.nested(|ctx, path| ctx.walk(accept, path, rung, 0), path);
                                path.pop();
                                let b = if has_else {
                                    path.push(1);
                                    let b = self.nested(|ctx, path| ctx.walk(reject, path, rung, 0), path);
                                    path.pop();
                                    Some(b)
                                } else {
                                    None
                                };
                                self.tail = block_tail;
                                let a = a?;
                                // (An else: both bodies keep the rung, `if (cfg.pass_index ==
                                // 0u) { .. } else { .. }`, 12 of run 1's 29 seeds, 2026-10-17.)
                                return Ok(Some(match b {
                                    None => Tree::new(&format!("stmt.if_{op}_{rung}"), st, vec![ux, uy, self.block(a, rung)?]),
                                    Some(b) => Tree::new(&format!("stmt.if_else_{op}_{rung}"), st, vec![ux, uy, self.block(a, rung)?, self.block(b?, rung)?]),
                                }));
                            }
                        }
                        // Any other test of workgroup-uniform values (`cfg.stage == 1u &&
                        // wid.x < NC`, `p == 1u || p == 3u`, run 1's seeds) over a body
                        // that holds a barrier: the uniform predicate, its bodies at the rung.
                        if rung.starts_with('L') && !exits && !exits_else && needed {
                            if let Some(flag) = Some(self.uniform_flag_checked(&c)?) {
                                self.tail = stmt_tail;
                                path.push(0);
                                let a = self.nested(|ctx, path| ctx.walk(accept, path, rung, 0), path);
                                path.pop();
                                let b = if has_else {
                                    path.push(1);
                                    let b = self.nested(|ctx, path| ctx.walk(reject, path, rung, 0), path);
                                    path.pop();
                                    Some(b)
                                } else {
                                    None
                                };
                                self.tail = block_tail;
                                let a = a?;
                                return Ok(Some(match b {
                                    None => Tree::new(&format!("stmt.if_uflag_{rung}"), st, vec![flag, self.block(a, rung)?]),
                                    Some(b) => Tree::new(&format!("stmt.if_else_uflag_{rung}"), st, vec![flag, self.block(a, rung)?, self.block(b?, rung)?]),
                                }));
                            }
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
                    Statement::MemoryBarrier(_) => Err("our software has a bug: naga's MemoryBarrier (a memory barrier without control; the WGSL front end never writes one)".into()),
                    Statement::ControlBarrier(flags) => {
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
            // A struct written whole to a storage binding (`dst[i] = Out(..)`,
            // `dst[i] = src[i]`): one store of each member.
            if let Some(rest) = name.strip_prefix("store.buffer.").filter(|r| !r.contains('@')) {
                let mut it = rest.split('.');
                let b = it.next().unwrap_or("");
                let segs: Vec<&str> = it.collect();
                let module = self.module;
                if let Some(sty) = place_type(module, b, &segs).filter(|t| matches!(module.types[*t].inner, naga::TypeInner::Struct { .. })) {
                    let (idx, value) = match kids.as_slice() {
                        [i, v] => (Some(i), v),
                        [v] => (None, v),
                        _ => return Err(format!("{name}: {} operands", kids.len())),
                    };
                    let idx = match idx {
                        Some(i) => Some(Tree { ty: ty(self, "u32.index")?, ..tree_of(self, i)? }),
                        None => None,
                    };
                    let src = self.struct_source(value, sty)?;
                    return self.store_struct(&format!("buffer.{rest}"), idx.as_ref(), sty, &src, rung);
                }
            }
            // A local array zeroed whole (naga 30 zeroes `var a: array<T, N>;`
            // declared in a loop at every pass, as WGSL does; naga 0.20 left the
            // store out): each element set to its zero, as a helper's local
            // array is at each call (2026-10-19, the naga/wgpu 30 upgrade).
            if let (Some(rest), [Node::Literal { form, value, .. }]) = (name.strip_prefix("store.local."), kids.as_slice()) {
                let array = rest.split('@').nth(1).and_then(|h| h.parse::<usize>().ok()).and_then(|h| self.naga_fn.local_variables.iter().nth(h)).and_then(|(_, l)| match self.module.types[l.ty].inner {
                    naga::TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => Some((base, n.get())),
                    _ => None,
                });
                if let (Some((base, n)), "opaque", 0.0) = (array, form.as_str(), *value) {
                    let local = format!("local.{}", rest.split('@').take(2).collect::<Vec<_>>().join("@"));
                    let elem = dual_of_type(self.module, base).ok_or_else(|| format!("the local array {local}: its element has no dual"))?;
                    return self.zero_array(&local, &elem, n as usize, rung);
                }
            }
            if let (Some(rest), [Node::App { name: vname, kids: elems, .. }]) = (name.strip_prefix("store.local."), kids.as_slice()) {
                // Of an array local only: a matrix written whole from its
                // columns (`y1 = mat3x3<f32>(...)`) is the ordinary store.
                let array_local = rest.split('@').nth(1).and_then(|h| h.parse::<usize>().ok()).and_then(|h| self.naga_fn.local_variables.iter().nth(h)).is_some_and(|(_, l)| matches!(self.module.types[l.ty].inner, naga::TypeInner::Array { .. }));
                if !rest.contains(".#") && vname.starts_with("naga.Compose.") && array_local {
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
                        let pidx = Tree::new(mask, ty(self, pindex_dual(mask))?, vec![idx]);
                        out.push(Tree::new(&format!("parr.set_{kind}_{rung}"), st, vec![Tree::new(arr, ty(self, parr_dual_of(arr, kind))?, vec![]), pidx, v]));
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
        if let Some(x) = self.aggregate_store(tree, rung)? {
            return Ok(Some(x));
        }
        if let Some(own) = self.through_array_arg(tree) {
            return self.store(&own, rung);
        }
        // `*p = v` for a pointer parameter: the caller's register set.
        if let (Some((row, kind)), [v]) = (tree.name.strip_prefix("store.").and_then(|a| self.pointer_args.get(a)).cloned(), tree.kids.as_slice()) {
            let v = self.value(v)?;
            let st = self.stmt_ty(rung)?;
            return Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), v])));
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
            // The cell compared as what it computes: two subgroup results of
            // the same operation (a store's index and its load's, each its own
            // statement) are the same cell.
            let key = |t: &Tree| self.expand_memo(t).to_sexpr();
            let xs = h_increments(value, &key(idx), &key).ok_or("a store to h is not read-add-write of the same cell")?;
            if xs.is_empty() {
                return Err("a store of h[c] to itself (no increment)".into());
            }
            // The thread's own cell, or (in a kernel that says `// mode full`)
            // its mirror, col * d + row: the symmetric half's second write.
            // There, with the layout still open, rows on y first for both
            // readings: a mirror read as a cell would turn the layout.
            let (cell, row_name) = if !self.mode_full {
                (self.cell(idx)?, "store.h")
            } else {
                let open = self.rows_x.is_none();
                if open {
                    self.rows_x = Some(false);
                }
                let first = match self.cell(idx) {
                    Ok(c) => Some((c, "store.h")),
                    Err(_) => self.mirror(idx).map(|m| (m, "store.h_mirror")),
                };
                match first {
                    Some(x) => x,
                    None => {
                        if open {
                            self.rows_x = None;
                        }
                        match self.cell(idx) {
                            Ok(c) => (c, "store.h"),
                            Err(e) => match self.mirror(idx) {
                                Some(m) => (m, "store.h_mirror"),
                                None => return Err(e),
                            },
                        }
                    }
                }
            };
            let vs = xs.into_iter().map(|x| self.value(x)).collect::<Result<Vec<_>, _>>()?;
            let stores = |ctx: &Self, r: &str| -> Result<Tree, String> {
                let list = vs.iter().map(|v| Ok(Tree::new(&format!("{row_name}_{r}"), ctx.stmt_ty(r)?, vec![cell.clone(), v.clone()]))).collect::<Result<Vec<_>, String>>()?;
                if list.len() == 1 {
                    return Ok(list.into_iter().next().expect("one store"));
                }
                Ok(Tree::new(&format!("stmt.block_{r}"), ctx.stmt_ty(r)?, vec![ctx.block(list, r)?]))
            };
            // (Under a condition too: a store in `if (t == 0u) { .. }` after the
            // guard ran in the threads the guard returned, 9 of run 1's seeds'
            // renderings wrote rows not arrived, 2026-10-17.)
            return match self.guard.clone() {
                Some(g) => {
                    let crung = format!("C{}", &rung[1..]);
                    let inner = self.block(vec![stores(self, &crung)?], &crung)?;
                    Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![g, inner])))
                }
                None => Ok(Some(stores(self, rung)?)),
            };
        }
        // A store of an output at any index (an inferred interface's
        // `store.<binding>_<rung>`; layer 7 of the checker proves its write
        // set), under the thread guard of a leading return as h's is.
        // A field of a struct-element array's element or of a struct binding
        // (`store.buffer.instances.#.transform`, `store.buffer.infos.even_pass`)
        // is the row `store.<binding>.<path>_<rung>`, over the index when the
        // binding is an array.
        // A texture store: the row `store.image.<b>.<sig>_<rung>`.
        if name.starts_with("store.image.") {
            let row = format!("{name}_{rung}");
            let Some((_, inputs, _)) = self.per_kernel_rows.iter().find(|p| p.0 == row).cloned() else {
                return Err(format!("{name}: the interface has no row {row} (a texture the kernel cannot write, or operand types the kingdom does not spell)"));
            };
            if inputs.len() != tree.kids.len() {
                return Err(format!("{name}: {} operands for a row of {}", tree.kids.len(), inputs.len()));
            }
            let kids = tree.kids.iter().map(|k| self.value(k)).collect::<Result<Vec<_>, _>>()?;
            let store = |ctx: &Self, r: &str| -> Result<Tree, String> { Ok(Tree::new(&format!("{name}_{r}"), ctx.stmt_ty(r)?, kids.clone())) };
            return match self.guard.clone() {
                Some(g) => {
                    let crung = format!("C{}", &rung[1..]);
                    let inner = self.block(vec![store(self, &crung)?], &crung)?;
                    Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![g, inner])))
                }
                None => Ok(Some(store(self, rung)?)),
            };
        }
        if let Some(place) = name.strip_prefix("store.buffer.").and_then(|r| buffer_place(self.module, r)) {
            let row = if place.path.is_empty() { format!("store.{}", place.binding) } else { format!("store.{}.{}", place.binding, place.path) };
            if self.per_kernel_rows.iter().any(|(n, i, _)| *n == format!("{row}_{rung}") && i.len() == place.indexes + 1) {
                let Some((value, index_kids)) = tree.kids.split_last() else { return Err(format!("a store to {name} without a value")) };
                let mut idx = Vec::new();
                for k in index_kids {
                    flat_indexes(k, &mut idx);
                }
                if idx.len() != place.indexes {
                    return Err(format!("a store to {name} with {} index operands, {} expected", idx.len(), place.indexes));
                }
                let index = ty(self, "u32.index")?;
                let mut kids = idx.iter().map(|i| Ok(Tree { ty: index, ..self.value(i)? })).collect::<Result<Vec<_>, String>>()?;
                kids.push(self.value(value)?);
                let store = |ctx: &Self, r: &str| -> Result<Tree, String> { Ok(Tree::new(&format!("{row}_{r}"), ctx.stmt_ty(r)?, kids.clone())) };
                return match self.guard.clone() {
                    Some(g) => {
                        let crung = format!("C{}", &rung[1..]);
                        let inner = self.block(vec![store(self, &crung)?], &crung)?;
                        Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![g, inner])))
                    }
                    None => Ok(Some(store(self, rung)?)),
                };
            }
            return Err(format!("store {name} has no safe spelling: the interface has no row {row}_{rung} over {} indexes (a place of a type the kingdom does not spell, or a binding the kernel does not write)", place.indexes));
        }
        if let Some(b) = name.strip_prefix("store.buffer.").and_then(|r| r.split('.').next()) {
            if self.per_kernel_rows.iter().any(|(n, _, _)| *n == format!("store.{b}_{rung}")) {
                let [idx, value] = tree.kids.as_slice() else { return Err(format!("a store to {b} without an index")) };
                let index = ty(self, "u32.index")?;
                let i = Tree { ty: index, ..self.value(idx)? };
                let v = self.value(value)?;
                let store = |ctx: &Self, r: &str| -> Result<Tree, String> { Ok(Tree::new(&format!("store.{b}_{r}"), ctx.stmt_ty(r)?, vec![i.clone(), v.clone()])) };
                return match self.guard.clone() {
                    Some(g) => {
                        let crung = format!("C{}", &rung[1..]);
                        let inner = self.block(vec![store(self, &crung)?], &crung)?;
                        Ok(Some(Tree::new(&format!("stmt.if_{rung}"), st, vec![g, inner])))
                    }
                    None => Ok(Some(store(self, rung)?)),
                };
            }
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
                    return Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), Tree::new(&compose, mty, columns)])));
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
                    let pidx = Tree::new(mask, ty(self, pindex_dual(mask))?, vec![i]);
                    return Ok(Some(Tree::new(&format!("parr.set_v_lane_{rung}"), st, vec![Tree::new(arr, ty(self, parr_dual_of(arr, kind))?, vec![]), pidx, j, v])));
                }
                if self.local_lanes.get(&base) == Some(&4) && self.local_vec_uint(&base) {
                    return Err(format!("a store to one lane of the vec4<u32> local {base} (the kingdom writes a vec4<u32> register whole)"));
                }
                if self.local_lanes.get(&base) == Some(&4) {
                    let row = self.register(&base, "vec4<f32>.real")?;
                    let i = self.value(unmasked(idx, 3.0))?;
                    let v = self.value(value)?;
                    return Ok(Some(Tree::new(&format!("reg.set_v_lane_{rung}"), st, vec![Tree::new(&row, reg_ty(self, 'v')?, vec![]), i, v])));
                }
                // One lane of a vec2<f32> local (`v.x = a`): its register set
                // whole, the new lane composed with the other's value.
                if self.local_lanes.get(&base) == Some(&2) && self.local_vec_float(&base) {
                    let lane = literal_value(unmasked(idx, 1.0)).filter(|l| (0.0..=1.0).contains(l)).ok_or_else(|| format!("a store to a lane of the vec2<f32> local {base} at an index that is not a literal lane"))? as usize;
                    let whole = self.vec2_register_value(&base)?;
                    let v = self.value(value)?;
                    let f = ty(self, "f32.real")?;
                    let lanes: Vec<Tree> = (0..2).map(|i| if i == lane { v.clone() } else { Tree::new(&format!("shape.access_{i}"), f, vec![whole.clone()]) }).collect();
                    let row = self.register(&base, "vec2<f32>.real")?;
                    return Ok(Some(Tree::new(&format!("reg.set_w_{rung}"), st, vec![Tree::new(&row, reg_ty(self, 'w')?, vec![]), Tree::new("shape.compose2", ty(self, "vec2<f32>.real")?, lanes)])));
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
                let i = self.value(unmasked(idx, pindex_mask(mask)))?;
                let v = self.value(value)?;
                let pidx = Tree::new(mask, ty(self, pindex_dual(mask))?, vec![i]);
                let arr_dual = parr_dual_of(arr, kind);
                return Ok(Some(Tree::new(&format!("parr.set_{kind}_{rung}"), st, vec![Tree::new(arr, ty(self, arr_dual)?, vec![]), pidx, v])));
            }
        }
        if let Some(local) = local_name(name) {
            // A store to a loop's counter inside its body: the loop reads it
            // (the unrolled item loop's copies step it; anything else has no
            // spelling, 2026-10-13).
            if [&self.item_counter, &self.tile_counter, &self.tile_counter2, &self.tile_counter3, &self.tile_counter4].iter().any(|c| c.as_deref() == Some(local.as_str())) {
                self.counter_stores.push((local.clone(), tree.kids[0].clone()));
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
            // A literal stored to a loop counter of the kingdom's (t, k, k2,
            // k3, k4: no register row spells them) is where its next loop
            // starts, as `for (k = 0u; ..)` writes it, not a register: the
            // loop reads its entry from here (2026-10-13, AST to AST: every
            // tile loop read back with an extra `i0 = 0u` before it).
            let counter_named = local.strip_prefix("local.").and_then(|r| r.split('@').next()).is_some_and(|n| matches!(n, "t" | "k" | "k2" | "k3" | "k4"));
            if counter_named {
                if let Some(v) = literal_value(&tree.kids[0]) {
                    self.stored_seen.insert(local.clone());
                    let index = ty(self, "u32.index")?;
                    self.axis_memo.insert(local.clone(), (Tree::literal("literal.index", index, v), self.loop_depth));
                    return Ok(None);
                }
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
            return Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![Tree::new(&row, reg_ty(self, kind)?, vec![]), v])));
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
        // A loop tested `s > 0` that does not halve its counter is some other
        // loop (v24's insertion step counts down by one, 2026-10-17).
        if !stepped {
            return Ok(None);
        }
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
        // (Or the literal stored just before the loop: a `for (var s = 32u; ..)`
        // inside another loop is a store each pass, v24 run 1's seed04.)
        let start = start.or_else(|| self.axis_memo.get(&counter).and_then(|(t, _)| literal_value(t)).filter(|v| v.fract() == 0.0 && *v >= 1.0).map(|v| v as u32));
        let Some(start) = start.filter(|v| v.is_power_of_two() && *v <= 2048) else { return fail("its start is not a literal power of two in its declaration (at most 2,048)") };
        // The body: `if (lidx < s) { a[lidx] = a[lidx] op a[lidx + s]; }` then
        // a workgroup barrier (in the block naga makes of a `for` body).
        let mut prefix = [path, &[0]].concat();
        if let [(at, Statement::Block(b))] = rest.as_slice() {
            prefix.push(*at);
            rest = b.iter().filter(|s| !matches!(s, Statement::Emit(_))).enumerate().collect();
        }
        let [(if_at, Statement::If { accept, reject, .. }), (_, Statement::ControlBarrier(flags))] = rest.as_slice() else { return fail("its body is not one `if` and a barrier") };
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
        // AN ARGMAX (ARGMIN) STEP: `if (v[lidx + s] > v[lidx]) { v[lidx] =
        // v[lidx + s]; i[lidx] = i[lidx + s]; }` under the test (v24 run 1's
        // seed04): wg.reduce_argmax over v and i.
        if let Some(Statement::If { accept: inner, reject: none, .. }) = accept.iter().find(|s| !matches!(s, Statement::Emit(_))) {
            let at = [&prefix[..], &[*if_at, 0, 0]].concat();
            let mut stores: Vec<Tree> = Vec::new();
            for i in 0..inner.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() {
                if let Some(r) = self.root_at(&[&at[..], &[0, i]].concat(), |k| matches!(k, RootKind::Store { .. })) {
                    let node = r.tree.clone();
                    stores.push(tree_of(self, &node)?);
                }
            }
            let cond = self.root_at(&at, |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at the argmax step")?;
            let cond = tree_of(self, &cond.tree)?.kids[0].clone();
            let partner = |t: &Tree| t.name == "index.add" && t.kids.len() == 2 && ((lane(&t.kids[0]) && is_counter(&t.kids[1])) || (is_counter(&t.kids[0]) && lane(&t.kids[1])));
            let load_of = |t: &Tree, arr: &str, i: &dyn Fn(&Tree) -> bool| t.name.starts_with(&format!("load.workgroup.{arr}.#")) && t.kids.len() == 1 && i(&t.kids[0]);
            let moved = |st: &Tree| -> Option<String> {
                let arr = st.name.strip_prefix("store.workgroup.").and_then(|r| r.strip_suffix(".#"))?;
                let [at, v] = st.kids.as_slice() else { return None };
                (lane(at) && load_of(v, arr, &partner)).then(|| arr.to_string())
            };
            let op = match cond.name.as_str() {
                "compare.gt" => "argmax",
                "compare.lt" => "argmin",
                _ => return fail(&format!("its step compares by {}, not > or <", cond.name)),
            };
            match (stores.as_slice(), cond.kids.as_slice(), none.iter().all(|s| matches!(s, Statement::Emit(_)))) {
                ([sv, si], [x, y], true) => {
                    let (Some(v), Some(i)) = (moved(sv), moved(si)) else { return fail("its step does not move v[lidx + s] and i[lidx + s] down") };
                    if !(load_of(x, &v, &partner) && load_of(y, &v, &lane)) {
                        return fail(&format!("its step's test is not v[lidx + s] {op} v[lidx]: {}", cond.to_sexpr()));
                    }
                    let fd = self.kingdom.dual(x.ty).map(|d| d.dual.clone()).unwrap_or_default();
                    let wv = self.array(&v, &fd)?;
                    let wi = self.array(&i, "u32.index")?;
                    if wg_elem(wv) != "f32" || wg_elem(wi) != "u32" {
                        return fail("its arrays are not an f32 value array and a u32 index array");
                    }
                    let (_, _, vd, _, _) = Self::wg_rows(wv);
                    let (_, _, id, _, _) = Self::wg_rows(wi);
                    let n = Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, (start * 2) as f32);
                    return Ok(Some(Tree::new(&format!("wg.reduce_{op}_{rung}"), self.stmt_ty(rung)?, vec![Tree::new(wv, ty(self, vd)?, vec![]), Tree::new(wi, ty(self, id)?, vec![]), n])));
                }
                _ => return fail("its step is not one test moving a value and its index"),
            }
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
            if let Statement::SubgroupCollectiveOperation { result, .. } | Statement::SubgroupGather { result, .. } | Statement::Atomic { result: Some(result), .. } = s {
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
            if !self.helpers.get(&callee).is_some_and(|h| h.pure || h.composite || h.row.is_some()) {
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
                // (Any other break condition is negated as it stands, `!(j ==
                // 0u)`: a loop with no counter, v24's insertion step `loop {
                // if (j == 0u) { break; } ... j = j - 1u; }`, 2026-10-17.)
                c = match c.name.as_str() {
                    "compare.ge" => Tree::new("compare.lt", c.ty, c.kids.clone()),
                    "compare.gt" => Tree::new("compare.le", c.ty, c.kids.clone()),
                    "compare.le" => Tree::new("compare.gt", c.ty, c.kids.clone()),
                    "compare.lt" => Tree::new("compare.ge", c.ty, c.kids.clone()),
                    "compare.eq" => Tree::new("compare.ne", c.ty, c.kids.clone()),
                    "compare.ne" => Tree::new("compare.eq", c.ty, c.kids.clone()),
                    _ => Tree::new("logic.not", c.ty, vec![c.clone()]),
                };
            }
        }
        // A tree reduction written as a halving loop is the reduction row.
        if let Some(t) = if do_while { None } else { self.halving(&c, continuing, &body_stmts_all, path, rung, cond)? } {
            return Ok(Some(t));
        }
        // The test normalised to `counter < bound`: `bound > counter`,
        // `counter <= bound` (bound + 1), `counter + k < bound` and
        // `counter + k > bound` as `if (...) break` (bound - k, floored).
        // `loop { if (!c) { break; } ... }` with no counter (CubeCL's
        // rejection sampling: a hash redrawn until it lands in range): a loop
        // that breaks, capped at the kingdom's MAX_LOOP_TRIP passes (every
        // loop is bounded); the same computation wherever it ends sooner.
        if normalise_test(&c).is_none() && !do_while && continuing.iter().all(|s| matches!(s, Statement::Emit(_))) {
            return self.capped_loop(c, &body_stmts_all, test_pos, path, rung).map(Some);
        }
        let (counter, bound) = normalise_test(&c).ok_or_else(|| format!("a loop test that is not a counter against a bound: {}", c.to_sexpr()))?;
        // The bound is evaluated once, so nothing it reads may change in the
        // loop's body (fresh28's `t0` is the enclosing loop's counter).
        let stored = stored_locals(self.naga_fn, body);
        let mut read_locals = Vec::new();
        locals_in(&bound, &mut read_locals);
        if let Some(l) = read_locals.iter().find(|l| **l != counter && stored.contains(l)) {
            return Err(format!("a loop bound that reads {l}, which the loop's body stores"));
        }
        // A guard `if (counter >= x) { break; }` right after the test (a scan
        // in chunks of a count it was given, v24 run 1's seed21 over a
        // barrier): the loop runs while counter < min(bound, x), the same
        // passes, and holds no break.
        let mut bound = bound;
        let mut folded_guard: Option<Vec<usize>> = None;
        if !do_while {
            // (The guard is the body's first statement, or the first of the
            // block naga makes of a `for` body.)
            let at: Option<(Vec<usize>, &Statement)> = match body_stmts_all.get(test_pos + 1) {
                Some(s @ Statement::If { .. }) => Some(([&path[..], &[0, test_pos + 1]].concat(), *s)),
                Some(Statement::Block(b)) => b.iter().find(|s| !matches!(s, Statement::Emit(_))).filter(|s| matches!(s, Statement::If { .. })).map(|s| ([&path[..], &[0, test_pos + 1, 0]].concat(), s)),
                _ => None,
            };
            if let Some((at, Statement::If { accept, reject, .. })) = at {
                let only_break = |b: &naga::Block| b.iter().filter(|s| !matches!(s, Statement::Emit(_))).count() == 1 && b.iter().any(|s| matches!(s, Statement::Break));
                if only_break(accept) && reject.iter().all(|s| matches!(s, Statement::Emit(_))) {
                    if let Some(root) = self.root_at(&at, |k| matches!(k, RootKind::Condition { .. })) {
                        let g = tree_of(self, &root.tree)?.kids[0].clone();
                        // (The counter itself against x, nothing added to it: a
                        // guard over an expression of the counter, `r >= n` with
                        // r = 4 gid.x + j, stays the break it is, its bound the
                        // literal layer 7 proves the stores by.)
                        let is_counter = |t: &Tree| t.kids.is_empty() && local_name(&t.name).as_deref() == Some(counter.as_str());
                        let x = match (g.name.as_str(), g.kids.as_slice()) {
                            ("compare.ge", [a, x]) | ("compare.le", [x, a]) if is_counter(a) => Some(x.clone()),
                            _ => None,
                        };
                        if let Some(x) = x {
                            let mut xs = Vec::new();
                            locals_in(&x, &mut xs);
                            if !xs.iter().any(|l| stored.contains(l) || *l == counter) {
                                bound = Tree::new("index.min", ty(self, "u32.index")?, vec![bound, x]);
                                folded_guard = Some(at);
                            }
                        }
                    }
                }
            }
        }
        // The body as the break detection sees it: without the folded guard.
        let without_guard = |stmts: &[Statement], first: usize| -> Vec<Statement> {
            let mut v: Vec<Statement> = stmts.to_vec();
            if folded_guard.is_some() {
                let at = test_pos + 1 - first;
                match v.get_mut(at) {
                    Some(Statement::Block(b)) => {
                        let mut inner: Vec<Statement> = b.iter().cloned().collect();
                        if let Some(i) = inner.iter().position(|s| !matches!(s, Statement::Emit(_))) {
                            inner[i] = Statement::Block(naga::Block::new());
                        }
                        *b = naga::Block::from_vec(inner);
                    }
                    Some(s) => *s = Statement::Block(naga::Block::new()),
                    None => {}
                }
            }
            v
        };
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
            // The kingdom's own cap (bound.of_index renders min(e, MAX_LOOP_TRIP)),
            // read back: the bound is e.
            // Anywhere in it: an unrolled item loop's test `t + u <= min(e, cap)`
            // is the floored bound over the capped one (2026-10-12).
            let expr = strip_cap(&expr);
            let expr = Tree { ty: index_ty, ..expr };
            (Tree::new("bound.of_index", ty(self, "u32.boundx")?, vec![expr]), "boundx")
        };
        // The step: the continuing block's store to the counter is counter + s, s a literal in 1..=64.
        // A loop that may hold a barrier (at an L rung, not one that breaks)
        // has a uniform bound: written as the uniform values when it is one;
        // otherwise the loop is read at the conditional rung of its depth (no
        // barrier in it) and written in place (stmt.cblock_L<d>).
        let mut bound_tree = bound_tree;
        // The item counter t, after an item loop, against a literal at the top
        // level (`for (; t < 1u; t = t + 1u)`, the items continued to a bound
        // whose fallback writes 1u, drawn in a uniform test's body): the items
        // continued, bounded by that literal as a uniform value, not a tile
        // loop over t (whose counter is k), 2026-10-17.
        let item_named = counter.strip_prefix("local.").and_then(|r| r.split('@').next()) == Some("t");
        let kind = if kind == "lit64" && rung == "L0" && item_named && self.used_counters.contains(&counter) {
            bound_tree = Tree::new("bound.of_index", ty(self, "u32.boundx")?, vec![Tree::literal("literal.index", index_ty, bound_tree.konst)]);
            "boundx"
        } else {
            kind
        };
        let first0 = if do_while { 0 } else { test_pos + 1 };
        let body0: Vec<Statement> = body_stmts_all.iter().skip(first0).map(|s| (*s).clone()).collect();
        let body0 = if do_while { body0 } else { without_guard(&body0, first0) };
        let breaks0 = has_break_guard(&body0) || has_own_break(&body0);
        if kind == "boundx" && rung.starts_with('L') && !breaks0 {
            match self.uniform_bound_of(&bound_tree.kids[0])? {
                Some(u) => bound_tree = Tree::new("bound.of_uniform", ty(self, "u32.boundu")?, vec![u]),
                // A bound over a body that holds a barrier: the index as a
                // uniform value the checker's layer 4 proves so (a helper's
                // argument from a workgroupUniformLoad) or names.
                None if holds_barrier(self.module, body) => {
                    let u = Tree::new("uniform.of_checked", ty(self, "u32.uniform")?, vec![bound_tree.kids[0].clone()]);
                    bound_tree = Tree::new("bound.of_uniform", ty(self, "u32.boundu")?, vec![u]);
                }
                None => {
                    let crung = format!("C{}", &rung[1..]);
                    let st = self.stmt_ty(rung)?;
                    let inner = self.looped(body, continuing, break_if, path, &crung, true)?;
                    return match inner {
                        Some(t) => Ok(Some(Tree::new(&format!("stmt.cblock_{rung}"), st, vec![self.block(vec![t], &crung)?]))),
                        None => Ok(None),
                    };
                }
            }
        } else if kind == "boundx" {
            // A loop without barriers (C and O rungs, a loop that breaks): a
            // uniform bound uncapped, any other with bound.of_index's cap.
            if let Some(u) = self.uniform_bound_of(&bound_tree.kids[0])? {
                bound_tree = Tree::new("bound.x_of_uniform", bound_tree.ty, vec![u]);
            }
        }
        // Any other expression bound: its value in the rows that cannot wrap.
        if bound_tree.name == "bound.of_index" {
            bound_tree.kids[0] = self.xbound_of(&bound_tree.kids[0])?;
        }
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
                        let lit = self.step_of(v, &counter);
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
                        let lit = self.step_of(v, &counter);
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
                            let lit = self.step_of(v, &counter);
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
        // A literal bound of at most 64 stepped wider than 64 (it runs once)
        // is the expression-bounded loop, whose step reaches 4,096.
        // Its bound takes the forms an expression bound takes above: uniform
        // (a literal is) at an L rung of a loop that may hold a barrier, else
        // the uncapped uniform bound.
        let (bound_tree, kind) = if kind == "lit64" && step > 64 {
            let expr = self.value(bound)?;
            let tree = match (self.uniform_bound_of(&expr)?, rung.starts_with('L') && !breaks0) {
                (Some(u), true) => Tree::new("bound.of_uniform", ty(self, "u32.boundu")?, vec![u]),
                (Some(u), false) => Tree::new("bound.x_of_uniform", ty(self, "u32.boundx")?, vec![u]),
                (None, _) => Tree::new("bound.of_index", ty(self, "u32.boundx")?, vec![expr]),
            };
            (tree, "boundx")
        } else {
            (bound_tree, kind)
        };
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
        // A literal-bounded tile loop (bound at most 64) steps at most 64; an
        // expression-bounded one (tilex) steps up to 4,096 (`u32.lit4096`,
        // 2026-10-16: 62 of 1,798 saved model kernels stepped by 128 or 256).
        let short = |t: &str, rung: &str| -> Result<String, String> {
            let most = if t.starts_with("tilex") { 4096 } else { 64 };
            if step > most {
                return Err(format!("a short loop stepped by {step} (at most {most})"));
            }
            Ok(format!("loop.{t}_{rung}"))
        };
        // At the top level an expression-bounded loop over a kingdom's tile
        // counter `k` is a tile loop (loop.tilex_L0, the renderer's spelling),
        // not the item loop (2026-10-12: 45 of 1,000 random trees, a tilex_brk_L0
        // read as an item loop that breaks).
        // (k2, k3, k4 too: a loop inside a fully unrolled one is written on the
        // next counter, the unrolled loop being no loop in the text.)
        let tile_named = counter.strip_prefix("local.").and_then(|r| r.split('@').next()).is_some_and(|n| matches!(n, "k" | "k2" | "k3" | "k4"));
        let (body_rung, row_name, out_rung) = match (kind, rung) {
            ("boundx", "L0") if tile_named => ("L2".to_string(), short(tilex, "L0")?, "L0"),
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
        // The item loop is the one whose body is at L1 (an expression-bounded
        // loop over t at the top level is one: loop.items_boundx, 2026-10-12).
        let item_loop = body_rung == "L1";
        // A loop whose counter enters with a value the reader cannot spell
        // (`for (var i = lidx; i < 1024u; i += 128u)`, a thread-strided
        // loop: nluca_s2's Sonnet seeds staged their codebook so) has no row:
        // read from 0 it would be another kernel (each thread staging the
        // same eight entries), so it is refused by name. Only an item loop
        // that continues the items an earlier loop left counts on from where
        // that loop stopped.
        // THE GRID-STRIDE LOOP (2026-10-16: 68 of the 1,798 saved model
        // kernels): `for (var i = e; i < n; i += s) { body(i) }` with e a
        // thread's expression (its ids, uniforms, literals: nothing the loop
        // could change) is read as `for (k = 0; k < n; k += s) { if (e + k
        // < n) { body(e + k) } }` — the same values in the same order, since
        // e >= 0 makes e + k < n imply k < n — over rows that already exist,
        // the loop's trip count uniform, the thread's part in its guard.
        let grid_stride: Option<Tree> = if entry.is_none() && !(item_loop && self.used_counters.contains(&counter)) {
            let e = self.axis_memo.get(&counter).map(|(t, _)| t.clone());
            fn fixed(t: &Tree) -> bool {
                !(t.name.starts_with("reg.") || t.name.starts_with("parr.") || t.name.starts_with("wg.") || t.name.starts_with("load.") || t.name.starts_with("safe.counter") || t.name.starts_with("safe.tile_")) && t.kids.iter().all(fixed)
            }
            match e {
                Some(e) if fixed(&e) && !do_while && e.ty == ty(self, "u32.index")? => Some(e),
                _ => return Err(format!("a loop whose counter {counter} enters with a value our reader does not know (a literal, zero, or a thread's expression of its ids, uniforms and literals has a spelling)")),
            }
        } else {
            None
        };
        let entry = if grid_stride.is_some() { Some(0.0) } else { entry };
        let start = match entry {
            Some(v) => v,
            None if item_loop => 0.0,
            None => return Err(format!("a loop whose counter {counter} enters with a value our reader does not know (only a literal or zero has a spelling)")),
        };
        // (An item loop bounded by an expression may start at a literal too, as
        // a tile loop does: `for (var kk = 1u; kk < n; kk++)`, 8 of run 1's
        // seeds, 2026-10-17; its bound is n - 1, its counter read as 1 + t.)
        let offset_items = item_loop && kind == "boundx" && !self.used_counters.contains(&counter);
        let (bound_tree, start) = if start == 0.0 || (item_loop && !offset_items) {
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
        } else if bound_tree.name == "bound.of_uniform" || bound_tree.name == "bound.x_of_uniform" {
            // A uniform bound stays uniform: bound - start.
            let u = ty(self, "u32.uniform")?;
            if !(1.0..=4096.0).contains(&start) {
                return Err(format!("a loop entered at {start} with a uniform bound (a uniform literal is 1..=4096)"));
            }
            let lit = Tree::new("uniform.of_lit", u, vec![Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, start)]);
            let expr = Tree::new("uniform.sub_floor", u, vec![bound_tree.kids[0].clone(), lit]);
            (Tree::new(&bound_tree.name, bound_tree.ty, vec![expr]), start)
        } else {
            let index = ty(self, "u32.index")?;
            let expr = Tree::new("index.sub_floor", ty(self, "u32.xbound")?, vec![bound_tree.kids[0].clone(), Tree::literal("literal.index", index, start)]);
            (Tree::new("bound.of_index", bound_tree.ty, vec![expr]), start)
        };
        if item_loop && !offset_items && entry.is_some_and(|v| v != 0.0) && !self.used_counters.contains(&counter) {
            return Err(format!("the item loop's counter {counter} starts at {} (an item loop starts at 0)", entry.unwrap_or_default()));
        }
        // A do-while runs its body once before its first test: it is the
        // counted loop when it enters below a literal bound (a loop entered
        // at or past its bound returned above, which a do-while may not).
        if do_while && (kind != "lit64" || entry.is_none()) {
            return Err(format!("a loop with `break if` whose bound is not a literal or whose counter's entry value our reader does not know ({counter})"));
        }
        // (A grid-stride loop's exit value is the thread's own: no literal.)
        let exit = (kind == "lit64" && grid_stride.is_none()).then(|| literal_value(bound).map(|n| start + ((n - start) / step as f32).ceil() * step as f32)).flatten();
        self.counter_offsets.insert(counter.clone(), start);
        if let Some(e) = &grid_stride {
            self.counter_exprs.insert(counter.clone(), e.clone());
        }
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
        // (A break below the top, inside an if, or a second one: the same
        // loop, its breaks lowered to a stop flag; v24 run 1's seeds 11, 17, 20.)
        let rest_seen = if do_while { rest.clone() } else { without_guard(&rest, first) };
        let breaks = has_break_guard(&rest_seen) || has_own_break(&rest_seen);
        if breaks && body_rung == "L1" {
            self.item_counter = saved.0.clone();
            self.tile_counter = saved.1.clone();
            self.tile_counter2 = saved.2.clone();
            self.tile_counter3 = saved.3.clone();
            self.tile_counter4 = saved.4.clone();
            return Err("the item loop breaks (a loop that breaks is a tile loop: its body holds no barrier)".into());
        }
        if breaks && grid_stride.is_some() {
            self.counter_exprs.remove(&counter);
            self.item_counter = saved.0.clone();
            self.tile_counter = saved.1.clone();
            self.tile_counter2 = saved.2.clone();
            self.tile_counter3 = saved.3.clone();
            self.tile_counter4 = saved.4.clone();
            return Err(format!("a grid-stride loop over {counter} that breaks"));
        }
        // A grid-stride body is guarded by the thread's own test: read at
        // the conditional rung of its depth (no barrier under the guard).
        let walk_rung = if breaks || grid_stride.is_some() { format!("C{}", &body_rung[1..]) } else { body_rung.clone() };
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
        let outer_stores = std::mem::take(&mut self.counter_stores);
        if let Some(at) = &folded_guard {
            self.skip_stmts.insert(at.clone());
        }
        let body_stmts = self.nested(|ctx, path| ctx.walk(&rest, path, &walk_rung, first), path);
        if let Some(at) = &folded_guard {
            self.skip_stmts.remove(at);
        }
        let (own, others): (Vec<CounterStore>, Vec<CounterStore>) = std::mem::replace(&mut self.counter_stores, outer_stores).into_iter().partition(|(c, _)| *c == counter);
        self.counter_stores.extend(others);
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
        self.counter_exprs.remove(&counter);
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
            // One guard at the top level and no other break: the guard
            // splits the body. Any other breaks: lowered to a stop flag, the
            // guard on it the body's last statement.
            let single = flat.iter().position(|t| break_guard(t, self).is_some()).filter(|&at| !flat[..at].iter().chain(&flat[at + 1..]).any(has_break));
            let (flat, c, after) = match single {
                Some(at) => {
                    let c = break_guard(&flat[at], self).expect("the guard at `at`")?;
                    let after: Vec<Tree> = flat.split_off(at + 1);
                    flat.pop();
                    (flat, c, after)
                }
                None => {
                    let (pre, c) = self.lower_breaks(flat, &walk_rung, path)?;
                    (pre, c, Vec::new())
                }
            };
            let d = &body_rung[1..];
            let brk = Tree::new(&format!("block.brk_B{d}"), ty(self, &format!("block.B{d}"))?, vec![self.block(flat, &walk_rung)?, c, self.block(after, &walk_rung)?]);
            // (Stepped by a literal: the expression-bounded loop with its stride,
            // run 1's search scanning candidates 256 at a time, 2026-10-17.)
            if step != 1 {
                // (A literal bound as the uniform value it is, as its text reads back.)
                let bound = if kind == "lit64" {
                    let u = ty(self, "u32.uniform")?;
                    Tree::new("bound.x_of_uniform", ty(self, "u32.boundx")?, vec![Tree::new("uniform.of_lit", u, vec![Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, bound_tree.konst)])])
                } else {
                    bound_tree
                };
                let stride = Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, step as f32);
                return Ok(Some(Tree::new(&format!("loop.tilex_strided_brk_{out_rung}"), st, vec![bound, stride, brk])));
            }
            let row = format!("loop.{}_brk_{out_rung}", if kind == "lit64" { "tile" } else { "tilex" });
            return Ok(Some(Tree::new(&row, st, vec![bound_tree, brk])));
        }
        // The unrolled item loop's main loop (the emitter's loop.items_unrolled):
        // u copies of the body, the counter stepped by 1 between them and set
        // back by u - 1 after the last, the loop stepping by u. Read as one
        // copy, to be joined with the loop after it that finishes the items.
        if !own.is_empty() {
            let delta = |v: &Tree| -> Option<i64> {
                let [x, k] = v.kids.as_slice() else { return None };
                let k = literal_value(k)? as i64;
                (local_name(&x.name).as_deref() == Some(counter.as_str())).then_some(())?;
                match v.name.as_str() {
                    n if n.ends_with(".add") => Some(k),
                    n if n.ends_with(".sub") || n.ends_with(".sub_floor") => Some(-k),
                    _ => None,
                }
            };
            let u = step as usize;
            let deltas: Vec<Option<i64>> = own.iter().map(|(_, v)| delta(v)).collect();
            let pattern = body_rung == "L1" && u >= 2 && deltas.len() == u && deltas[..u - 1].iter().all(|d| *d == Some(1)) && deltas[u - 1] == Some(-(u as i64 - 1));
            let mut flat = Vec::new();
            for t in body_stmts {
                flatten_stmts(t, &mut flat);
            }
            let n = flat.len() / u.max(1);
            let copies_equal = pattern && flat.len() == n * u && (1..u).all(|j| flat[j * n..(j + 1) * n].iter().map(Tree::to_sexpr).eq(flat[..n].iter().map(Tree::to_sexpr)));
            if !copies_equal {
                return Err(format!("a store to the loop counter {counter} inside its loop's body ({}): only the loop's step steps it", own[0].1.to_sexpr()));
            }
            flat.truncate(n);
            let unroll = Tree::literal("literal.unroll", ty(self, "u32.unroll")?, step as f32);
            return Ok(Some(Tree::new(UNROLLED_MAIN, st, vec![unroll, self.block(flat, &body_rung)?])));
        }
        let block = match &grid_stride {
            None => self.block(body_stmts, &body_rung)?,
            Some(e) => {
                // The guard: e + k < n, the counter k of this loop's depth.
                let index = ty(self, "u32.index")?;
                let k = if item_loop {
                    Tree::new("safe.counter_index", index, vec![Tree::new("safe.counter", ty(self, "u32.counter")?, vec![])])
                } else {
                    let (ix, c, cty) = match depth {
                        1 => ("safe.tile_index", "safe.tile_counter", "u32.tcounter"),
                        2 => ("safe.tile_index2", "safe.tile_counter2", "u32.tcounter2"),
                        3 => ("safe.tile_index3", "safe.tile_counter3", "u32.tcounter3"),
                        _ => ("safe.tile_index4", "safe.tile_counter4", "u32.tcounter4"),
                    };
                    Tree::new(ix, index, vec![Tree::new(c, ty(self, cty)?, vec![])])
                };
                let n = match (bound_tree.name.as_str(), bound_tree.kids.as_slice()) {
                    ("literal.lit64", []) => Tree::literal("literal.index", index, bound_tree.konst),
                    ("bound.of_index", [x]) => x.clone(),
                    // (A literal uniform is the index literal its text reads as.)
                    ("bound.of_uniform" | "bound.x_of_uniform", [u]) => match (u.name.as_str(), u.kids.as_slice()) {
                        ("uniform.of_lit", [l]) if l.name.starts_with("literal.") => Tree::literal("literal.index", index, l.konst),
                        _ => Tree::new("index.of_uniform", index, vec![u.clone()]),
                    },
                    (other, _) => return Err(format!("a grid-stride loop bounded by {other} (a literal, an expression or a uniform bound has a guard)")),
                };
                let cond = Tree::new("compare.lt", ty(self, "bool.flag")?, vec![Tree::new("index.add", index, vec![e.clone(), k]), n]);
                let crung = format!("C{}", &body_rung[1..]);
                let inner = self.block(body_stmts, &crung)?;
                let guard = Tree::new(&format!("stmt.if_{body_rung}"), self.stmt_ty(&body_rung)?, vec![cond, inner]);
                self.block(vec![guard], &body_rung)?
            }
        };
        let wide = body_rung == "L1" || row_name.starts_with("loop.tilex_strided");
        let step_lit = if wide { Tree::literal("literal.lit4096", ty(self, "u32.lit4096")?, step as f32) } else { Tree::literal("literal.lit64", ty(self, "u32.lit64")?, step as f32) };
        let kids = if step != 1 { vec![bound_tree, step_lit, block] } else { vec![bound_tree, block] };
        Ok(Some(Tree::new(&row_name, st, kids)))
    }

    /// A loop with no counter whose first statement is its break guard
    /// (`c` the condition it goes on under): the tile loop of the next depth
    /// bounded by MAX_LOOP_TRIP, its body the break block `if (!c) { break; }`
    /// then the rest, conditional (no barrier in a loop that breaks).
    fn capped_loop(&mut self, c: Tree, body: &[&Statement], test_pos: usize, path: &mut Vec<usize>, rung: &str) -> Result<Tree, String> {
        let depth = [&self.tile_counter, &self.tile_counter2, &self.tile_counter3, &self.tile_counter4].iter().filter(|c| c.is_some()).count() + 1;
        if depth > 4 {
            return Err("a loop that breaks inside four tile loops (a fifth has no counter)".into());
        }
        let d: usize = rung[1..].parse().map_err(|_| format!("a loop at rung {rung}"))?;
        // The break block's depth: an L0 or L1 loop's body opens at 2, any
        // other one deeper than its rung (the table's loop.tilex_brk rows).
        let b = if rung.starts_with('L') { (d + 1).max(2) } else { d + 1 };
        if b > 5 {
            return Err(format!("a loop that breaks at rung {rung} (too deep)"));
        }
        let body_rung = format!("C{b}");
        let out_rung = if rung.starts_with('O') { return Err("a loop with no counter in the epilogue".into()) } else { rung.to_string() };
        let cap = Tree::literal("literal.index", ty(self, "u32.xbound")?, crate::wgsl::render::MAX_LOOP_TRIP as f32);
        let bound = Tree::new("bound.of_index", ty(self, "u32.boundx")?, vec![cap]);
        // Further break guards straight after the test (`if (j == 0u) { break; }
        // if (best[j - 1u] >= v) { break; }`, v24's insertion step): one stop
        // condition, `a || b` evaluating them in order as the ifs do.
        let is_break_guard = |s: &Statement| matches!(s, Statement::If { accept, reject, .. } if accept.iter().filter(|x| !matches!(x, Statement::Emit(_))).count() == 1 && accept.iter().any(|x| matches!(x, Statement::Break)) && reject.iter().all(|x| matches!(x, Statement::Emit(_))));
        let guards = body.iter().skip(test_pos + 1).take_while(|s| is_break_guard(s)).count();
        let mut more_stops = Vec::new();
        for i in 0..guards {
            let at = [&path[..], &[0, test_pos + 1 + i]].concat();
            let root = self.root_at(&at, |k| matches!(k, RootKind::Condition { .. })).ok_or("no condition root at a loop's second break")?;
            let t = tree_of(self, &root.tree)?;
            more_stops.push(self.value(&t.kids[0])?);
        }
        let first = test_pos + 1 + guards;
        let rest: Vec<Statement> = body.iter().skip(first).map(|s| (*s).clone()).collect();
        let rest = naga::Block::from_vec(rest);
        // The loop takes the tile counter of its depth (its body may hold tile loops).
        let saved = (self.tile_counter.clone(), self.tile_counter2.clone(), self.tile_counter3.clone(), self.tile_counter4.clone());
        let mine = Some(format!("local.capped@{}", path.iter().map(usize::to_string).collect::<Vec<_>>().join(".")));
        match depth {
            1 => self.tile_counter = mine,
            2 => self.tile_counter2 = mine,
            3 => self.tile_counter3 = mine,
            _ => self.tile_counter4 = mine,
        }
        path.push(0);
        let stmts = self.nested(|ctx, path| ctx.walk(&rest, path, &body_rung, first), path);
        path.pop();
        (self.tile_counter, self.tile_counter2, self.tile_counter3, self.tile_counter4) = saved;
        let after = stmts?;
        if after.iter().any(has_break) {
            return Err("a second break in a loop with no counter".into());
        }
        let flag = ty(self, "bool.flag")?;
        let stop = more_stops.into_iter().fold(Tree::new("logic.not", flag, vec![self.value(&c)?]), |a, b| Tree::new("logic.or", flag, vec![a, b]));
        let brk = Tree::new(&format!("block.brk_B{b}"), ty(self, &format!("block.B{b}"))?, vec![self.block(Vec::new(), &body_rung)?, stop, self.block(after, &body_rung)?]);
        Ok(Tree::new(&format!("loop.tilex_brk_{out_rung}"), self.stmt_ty(&out_rung)?, vec![bound, brk]))
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

/// Whether a block holds a barrier (workgroup, storage or subgroup, or a
/// workgroupUniformLoad), in it or in a function it calls.
/// Whether a block holds the kingdom's item loop: a loop over the local
/// `t` (the emitter's item counter), at any depth of blocks and branches.
fn holds_item_loop(f: &naga::Function, b: &naga::Block) -> bool {
    let over_t = |body: &naga::Block| {
        body.iter().any(|s| match s {
            Statement::If { condition, .. } => {
                fn reads_t(f: &naga::Function, e: Handle<Expression>, depth: usize) -> bool {
                    depth < 8 && match &f.expressions[e] {
                        Expression::Load { pointer } => matches!(&f.expressions[*pointer], Expression::LocalVariable(l) if f.local_variables[*l].name.as_deref() == Some("t")),
                        Expression::Binary { left, right, .. } => reads_t(f, *left, depth + 1) || reads_t(f, *right, depth + 1),
                        Expression::Unary { expr, .. } => reads_t(f, *expr, depth + 1),
                        _ => false,
                    }
                }
                reads_t(f, *condition, 0)
            }
            _ => false,
        })
    };
    b.iter().any(|s| match s {
        Statement::Loop { body, .. } => over_t(body),
        Statement::Block(x) => holds_item_loop(f, x),
        Statement::If { accept, reject, .. } => holds_item_loop(f, accept) || holds_item_loop(f, reject),
        _ => false,
    })
}

fn holds_barrier(module: &naga::Module, b: &naga::Block) -> bool {
    b.iter().any(|s| match s {
        Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) | Statement::WorkGroupUniformLoad { .. } => true,
        Statement::Block(x) => holds_barrier(module, x),
        Statement::If { accept, reject, .. } => holds_barrier(module, accept) || holds_barrier(module, reject),
        Statement::Switch { cases, .. } => cases.iter().any(|c| holds_barrier(module, &c.body)),
        Statement::Loop { body, continuing, .. } => holds_barrier(module, body) || holds_barrier(module, continuing),
        Statement::Call { function, .. } => holds_barrier(module, &module.functions[*function].body),
        _ => false,
    })
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
        Statement::ControlBarrier(_) => "ControlBarrier",
        Statement::MemoryBarrier(_) => "MemoryBarrier",
        Statement::Store { .. } => "Store",
        Statement::ImageStore { .. } => "ImageStore",
        Statement::Atomic { .. } => "Atomic",
        Statement::WorkGroupUniformLoad { .. } => "WorkGroupUniformLoad",
        Statement::Call { .. } => "Call",
        Statement::RayQuery { .. } => "RayQuery",
        Statement::SubgroupBallot { .. } | Statement::SubgroupGather { .. } | Statement::SubgroupCollectiveOperation { .. } => "Subgroup",
        Statement::ImageAtomic { .. } => "ImageAtomic",
        Statement::RayPipelineFunction(_) => "RayPipelineFunction",
        Statement::CooperativeStore { .. } => "CooperativeStore",
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
        // (Checked before indexing: `then_some` builds its value first, and a
        // model's three-number header, `// grid 256 1 1`, panicked the fit,
        // CR_A_s3 2026-10-16.)
        match v.as_slice() {
            &[x, y, bx, by] => Some(((x, y, bx, by), rows_x)),
            _ => None,
        }
    })
}

/// The entry function's statements by path, one line each (for reading a
/// refusal); in a module of several entry points every entry point's, each
/// line led by its entry point's name.
pub fn statement_outline(source: &str) -> Result<Vec<String>, String> {
    let kernel = read(source)?;
    if kernel.module.entry_points.is_empty() {
        return Err("no entry point".into());
    }
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
    let several = kernel.module.entry_points.len() > 1;
    let mut out = Vec::new();
    for ep in &kernel.module.entry_points {
        let mut lines = Vec::new();
        walk(&ep.function.body, &mut Vec::new(), &mut lines);
        out.extend(lines.into_iter().map(|l| if several { format!("{} {l}", ep.name) } else { l }));
    }
    Ok(out)
}

/// Read `source` into the safe grammar, or say why not: the tree with its
/// knobs as parameters (`params::assign`: every position that depends on a
/// block factor, the workgroup's extent or the tile depth written as a
/// reference to it); it renders the same text as [`read_safe_literal`]'s.
pub fn read_safe(source: &str, kingdom: &WgslKingdom) -> Result<SafeRead, String> {
    let r = read_safe_literal(source, kingdom)?;
    let kingdom = r.kingdom.as_ref().unwrap_or(kingdom);
    // (A module's entry points each by its own knobs.)
    let tree = if r.tree.name.starts_with("kernel.module_") {
        let entries = crate::wgsl::render::module_entries(&r.tree).iter().map(|k| crate::wgsl::params::assign(k, kingdom)).collect::<Result<Vec<_>, _>>()?;
        crate::wgsl::render::module_tree(kingdom, entries)?
    } else {
        crate::wgsl::params::assign(&r.tree, kingdom)?
    };
    Ok(SafeRead { tree, ..r })
}

/// The value an override with no default and no header line starts at: 64 for a size (the usual workgroup width), 1 for any
/// other integer, 1.0 for a float, `false` for a bool. A constant of its type either way: evolution moves it.
pub fn override_seed(name: &str, ty: &str) -> String {
    let lower = name.to_lowercase();
    let size = ["workgroup", "thread", "group", "size", "block", "tile", "width"].iter().any(|w| lower.contains(w));
    match ty {
        "u32" => format!("{}u", if size { 64 } else { 1 }),
        "i32" => format!("{}i", if size { 64 } else { 1 }),
        "f32" => "1.0f".to_string(),
        "bool" => "false".to_string(),
        _ => "1".to_string(),
    }
}

/// A kernel's pipeline-overridable constants given their values by header
/// lines, `// override <name> <value>`: each `override <name>: <type>;` with no
/// default is declared with that value (`override <name>: <type> = <value>;`).
/// The kingdom has no pipeline to set one, so a kernel reads at the values
/// its text names.
pub fn with_overrides(source: &str) -> String {
    let given: Vec<(&str, &str)> = source.lines().filter_map(|l| l.trim().strip_prefix("// override ")).filter_map(|r| r.split_once(' ')).map(|(n, v)| (n.trim(), v.trim())).collect();
    if given.is_empty() && !source.contains("override ") {
        return source.to_string();
    }
    let mut out = String::with_capacity(source.len() + 32);
    let mut rest = source;
    while let Some(i) = rest.find("override ") {
        let (head, tail) = rest.split_at(i);
        let line = head.rsplit('\n').next().unwrap_or(head);
        let word_start = head.chars().next_back().is_none_or(|c| !(c.is_alphanumeric() || c == '_')) && !line.contains("//");
        out.push_str(head);
        let after = &tail["override ".len()..];
        let name: String = after.trim_start().chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        let end = after.find(';').unwrap_or(after.len());
        // An override with no default and no header line is a constant the pipeline sets and the kingdom has no pipeline
        // for: it reads as a constant of its type (a literal, so a gene holds it as a `C_T` entry and evolution chooses it),
        // started at a seed value, `override_seed`.
        let seed = (word_start && !after[..end].contains('=') && end < after.len()).then(|| after[..end].split_once(':').map(|(_, ty)| override_seed(&name, ty.trim()))).flatten();
        match given.iter().find(|(n, _)| *n == name).map(|(_, v)| (*v).to_string()).or(seed) {
            Some(v) if word_start && !after[..end].contains('=') && end < after.len() => {
                out.push_str("override ");
                out.push_str(&after[..end]);
                out.push_str(&format!(" = {v}"));
                rest = &after[end..];
            }
            _ => {
                out.push_str("override ");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// The workgroup arrays read so far: a module variable's row (`wg_x` is
/// `wg.ua`) and each row's declared length. A module's entry points share
/// its variables, so each is read with the arrays of those before it: one
/// variable is one row in every entry point that uses it.
type WgArrays = (BTreeMap<String, &'static str>, BTreeMap<&'static str, u32>);

/// One entry point read: its tree, notes, registers and the arrays after it.
type EntryRead = (Tree, Vec<String>, BTreeMap<String, String>, WgArrays);

/// An entry point's grid line in a module of several (`// entry <name>
/// grid x y bx by [rows-x]`, [`crate::wgsl::render::Grid::entry_header`]).
pub(crate) fn entry_grid_header(source: &str, entry: &str) -> Option<((u32, u32, u32, u32), bool)> {
    source.lines().map(str::trim).find_map(|l| {
        let rest = l.strip_prefix("// entry ")?.strip_prefix(entry)?.strip_prefix(" grid ")?;
        let mut tokens: Vec<&str> = rest.split_whitespace().collect();
        let rows_x = tokens.last() == Some(&"rows-x");
        if rows_x {
            tokens.pop();
        }
        match tokens.iter().map(|t| t.parse().ok()).collect::<Option<Vec<u32>>>()?.as_slice() {
            &[x, y, bx, by] => Some(((x, y, bx, by), rows_x)),
            _ => None,
        }
    })
}

/// Read `source` into the safe grammar with every knob-dependent position
/// left as the literal the kernel wrote.
pub fn read_safe_literal(source: &str, kingdom: &WgslKingdom) -> Result<SafeRead, String> {
    let source = &with_overrides(source);
    // A kernel is read as it is written; one that does not read, and has a
    // return inside a loop (the kingdom spells a return at the end of a
    // block), is read again with those returns lowered (`wgsl::lower`).
    match read_safe_literal_of(source, kingdom, false) {
        Ok(r) => Ok(r),
        Err(first) => {
            let changes = crate::wgsl::naga_shape::parse(source).map(|mut m| crate::wgsl::lower::lower_returns_in_loops(&mut m)).unwrap_or(0);
            if changes == 0 {
                return Err(first);
            }
            read_safe_literal_of(source, kingdom, true)
        }
    }
}

fn read_safe_literal_of(source: &str, kingdom: &WgslKingdom, lower: bool) -> Result<SafeRead, String> {
    let role = crate::wgsl::render::interface_of(source)?;
    let mut kernel = crate::wgsl::reader::read_with(source, lower)?;
    // The interface's names as the kingdom's canonical roles (its input is
    // phi, its output h, its uniform cfg with the fields d, m, r_rows, r_off),
    // so the ownership proof and the safe rows read every interface alike.
    for f in &mut kernel.functions {
        for r in &mut f.roots {
            r.tree = floor_differences(&canonical_node(&r.tree, &role));
            match &mut r.kind {
                RootKind::Store { target } | RootKind::Atomic { target, .. } => *target = canonical_name(target, &role),
                _ => {}
            }
        }
    }
    let extended = kingdom.extended(&kernel.module);
    let used = (extended.duals.len() != kingdom.duals.len()).then(|| extended.clone());
    let kingdom = &extended;
    let position = |name: &str| kernel.module.entry_points.iter().position(|e| e.name == name).ok_or_else(|| format!("no entry point {name}"));
    if role.entries.len() <= 1 {
        // The grid: the `// grid x y bx by` header when the text has one (its
        // block factors), else the module's own workgroup with 1 x 1 blocks.
        let ep = position(&role.entry)?;
        let declared = crate::wgsl::interface::workgroup_of(&kernel.module, &kernel.module.entry_points[ep]);
        let grid = grid_header(source).unwrap_or((declared[0], declared[1], 1, 1));
        let at = Reading { source, kingdom, kernel: &kernel, role: &role };
        let (tree, notes, registers, _) = read_entry(&at, ep, (grid, grid_rows_x(source)), (BTreeMap::new(), BTreeMap::new()))?;
        return Ok(SafeRead { kingdom: used, tree, notes, registers, role });
    }
    // SEVERAL ENTRY POINTS (2026-10-17, B6 of PLAN_v24_ingest_evolution): each
    // read as a safe kernel in the interface's order, each grid from its own
    // line (`// entry <name> grid ..`) else its workgroup at 1 x 1 blocks, the
    // module's workgroup variables one row each across them.
    let (mut kids, mut notes, mut registers) = (Vec::new(), Vec::new(), BTreeMap::new());
    let mut arrays: WgArrays = (BTreeMap::new(), BTreeMap::new());
    let at = Reading { source, kingdom, kernel: &kernel, role: &role };
    for name in &role.entries {
        let ep = position(name)?;
        let declared = crate::wgsl::interface::workgroup_of(&kernel.module, &kernel.module.entry_points[ep]);
        // (A grid line whose workgroup is not the entry point's own is no line of
        // this kernel's: a model copied the kingdom's text, `// entry ingest grid
        // 16 16 1 1` over @workgroup_size(64, 1, 1), 3 of run 1's seeds; the
        // entry point's own workgroup at 1 x 1 blocks.)
        let grid = entry_grid_header(source, name).filter(|((x, y, _, _), _)| [*x, *y, 1] == declared).unwrap_or(((declared[0], declared[1], 1, 1), false));
        let (tree, n, regs, after) = read_entry(&at, ep, grid, arrays).map_err(|e| format!("entry point {name}: {e}"))?;
        arrays = after;
        notes.extend(n.into_iter().map(|n| format!("{name}: {n}")));
        registers.extend(regs.into_iter().map(|(k, v)| (format!("{name}/{k}"), v)));
        kids.push(tree);
    }
    let tree = crate::wgsl::render::module_tree(kingdom, kids)?;
    Ok(SafeRead { kingdom: used, tree, notes, registers, role })
}

/// A statement of a block being walked: the block, the statement's position
/// in it (Emits counted), the last non-Emit position, and the number the
/// next statement takes.
#[derive(Clone, Copy)]
struct BlockAt<'b> {
    block: &'b naga::Block,
    position: usize,
    last: Option<usize>,
    next: usize,
}

/// A kernel text being read: the text, the kingdom, naga's reading of it
/// (its roots made canonical) and its interface.
struct Reading<'a> {
    source: &'a str,
    kingdom: &'a WgslKingdom,
    kernel: &'a crate::wgsl::reader::Kernel,
    role: &'a crate::wgsl::render::Role,
}

/// One entry point of a read kernel (`ep`, its index among the module's)
/// as a safe kernel tree, at its grid (and whether a header put the rows on
/// x), its workgroup variables' rows starting from `arrays`.
fn read_entry(at: &Reading, ep: usize, (grid, rows_x): ((u32, u32, u32, u32), bool), arrays: WgArrays) -> Result<EntryRead, String> {
    let (read, open, rows_x) = read_entry_under(at, ep, grid, rows_x.then_some(true), arrays.clone())?;
    if open {
        return read_entry_under(at, ep, grid, Some(rows_x), arrays).map(|(r, _, _)| r);
    }
    Ok(read)
}

/// [`read_entry`] with the layout preset (`None`: open until a store
/// decides it); with the read, whether a thread's row or column index was
/// read while the layout was open, and the layout decided.
fn read_entry_under(at: &Reading, ep: usize, grid: (u32, u32, u32, u32), preset: Option<bool>, arrays: WgArrays) -> Result<(EntryRead, bool, bool), String> {
    let Reading { source, kingdom, kernel, role } = *at;
    let entry = &kernel.module.entry_points[ep];
    // The workgroup the header names is the one the kernel declares: x and y
    // each 1..=1,024 with x * y <= 1,024 (the M3 Max's limits), z 1 (a thread
    // owns cells by x and y; two threads differing only in z would share them).
    let declared = crate::wgsl::interface::workgroup_of(&kernel.module, entry);
    if declared != [grid.0, grid.1, 1] {
        return Err(format!("the header's workgroup {} x {} is not the kernel's @workgroup_size({}, {}, {})", grid.0, grid.1, declared[0], declared[1], declared[2]));
    }
    if grid.0 == 0 || grid.1 == 0 || grid.0 * grid.1 > crate::wgsl::render::MAX_THREADS {
        return Err(format!("workgroup {} x {} is outside 1..=1,024 invocations", grid.0, grid.1));
    }
    let f = kernel.functions.iter().find(|f| f.entry_point && f.name == entry.name).ok_or_else(|| format!("entry point {} was not read", entry.name))?;
    let naga_fn: &Function = &entry.function;
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
        // A kingdom helper, written verbatim: its row (a call is the row), its
        // body not read (2026-10-12: hash32_to_index returns twice, fx_add_sat
        // holds literals past 2^24; 100 of 1,000 random trees; inverse_4 of
        // inverse_4, inlined, was a million nodes).
        let row = crate::wgsl::render::helper_row(&name).filter(|_| COMPOSITES.iter().any(|(n, text)| *n == name && source.contains(text)));
        // One return: the helper's value. Several: `if (c) { return a; }` ...
        // `return b;` (or a last if-else of two returns) is the select chain
        // select(.., a, c), its guards' conditions read into it.
        match returns.as_slice() {
            _ if row.is_some() => {}
            [] => {}
            [r] => {
                if let Node::App { kids, .. } = &r.tree {
                    body = Some(kids[0].clone());
                }
                roots.entry(r.path.clone()).or_default().push(r);
            }
            several => {
                // Guards over its returns alone: a select chain. Otherwise (a
                // helper with statements before or between its guards, `sgn`'s
                // hash then `if (c) { return 1.0; } return -1.0;`, 8 of run 1's
                // seeds, 2026-10-17) it is inlined as statements, each guard's
                // return ending its branch, the rest of the helper the else.
                let mut tried = roots.clone();
                if let Some(chain) = guard_chain(&nf.body, several, &mut tried) {
                    roots = tried;
                    body = Some(chain);
                }
                for r in several {
                    roots.entry(r.path.clone()).or_default().push(r);
                }
            }
        }
        let pure = inits.is_empty() && arg_roots.is_empty() && roots.values().all(|rs| rs.iter().all(|r| matches!(r.kind, RootKind::Return)));
        let composite = (name.starts_with("at_vec") || name.starts_with("col_mat")) && nf.arguments.len() == 2 && COMPOSITES.iter().any(|(n, text)| *n == name && source.contains(text));
        if body.is_none() && (pure || composite) && row.is_none() && nf.result.is_some() {
            return Err(format!("helper {name} returns nothing"));
        }
        let args = nf.arguments.iter().enumerate().map(|(i, a)| format!("arg.{}", a.name.clone().unwrap_or_else(|| i.to_string()))).collect();
        let mut local_lanes = HashMap::new();
        let mut uninit = Vec::new();
        for (h, lv) in nf.local_variables.iter() {
            // (A folded short circuit's local holds nothing: no register, no zero.)
            if lv.name.is_none() && crate::wgsl::naga_shape::is_dead_local(nf, h) {
                continue;
            }
            let local = format!("local.{name}/{}@{}", lv.name.clone().unwrap_or_else(|| "local".into()), h.index());
            if let naga::TypeInner::Vector { size, .. } = &kernel.module.types[lv.ty].inner {
                local_lanes.insert(local.clone(), *size as u32);
            }
            if lv.init.is_none() {
                // (A local array, `var x0: array<vec4<f32>, 8>;`: its element's
                // dual and length, `array:<dual>:<n>`, zeroed element by element
                // at each call, as WGSL zeroes it: v24 run 1's seed29.)
                let d = match &kernel.module.types[lv.ty].inner {
                    naga::TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => dual_of_type(&kernel.module, *base).map(|d| format!("array:{d}:{n}")),
                    _ => dual_of_type(&kernel.module, lv.ty),
                };
                uninit.push((local, d.unwrap_or_else(|| "no register".into())));
            }
        }
        let ret_dual = nf.result.as_ref().and_then(|r| dual_of_type(&kernel.module, r.ty));
        let types = infer_function(hf, kingdom);
        helper_duals.extend(types.by_text.iter().map(|(text, set)| (text.clone(), choose_dual(set, kingdom))));
        let (bound_lets, stale_lets) = lets_used_in_deeper_loops(&hf.roots, &nf.body, &hf.points);
        helpers.insert(name, Helper { args, body, lets, pure, composite, row, roots, inits, uninit, local_lanes, calls: hf.calls.clone(), arg_roots, block: &nf.body, naga_fn: nf, ret_dual, bound_lets, stale_lets });
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
    let (bound_lets, stale_lets) = lets_used_in_deeper_loops(&f.roots, &naga_fn.body, &f.points);
    // Main's text ends in `return;`: the renderer writes it after a body
    // that returns, with no epilogue after it (naga's own trailing return is
    // in every kernel's IR, so the text decides).
    let main_returns = source.find(&format!("fn {}(", entry.name)).map(|i| &source[i..]).and_then(|m| m.rfind('}').map(|j| &m[..j])).is_some_and(|m| m.trim_end().ends_with("return;"));
    let uniform_lanes = if role.uniform_struct.is_empty() { role.fields.iter().map(|f| f.name.clone()).collect() } else { Vec::new() };
    let (arrays, array_lens) = arrays;
    let mut ctx = Ctx { kingdom, main_returns, per_kernel_rows: crate::wgsl::safe_karva::per_kernel_rows(kingdom, role), uniform_var: role.uniform_var.clone(), uniform_lanes, duals: &duals, lets: &lets, roots: &roots, item_counter: None, tile_counter: None, tile_counter2: None, tile_counter3: None, tile_counter4: None, registers: BTreeMap::new(), counter_stores: Vec::new(), guard: None, tail: true, arrays, array_lens, arrays_used: BTreeSet::new(), parrs: BTreeMap::new(), grid, rows_x: preset, notes: Vec::new(), helpers: &helpers, calls: &calls, arg_roots: &arg_roots, inline_args: None, built: std::cell::Cell::new(0), early_return_at: None, branch_depth: 0, skip_stmts: BTreeSet::new(), outer_args: Vec::new(), row_arrays: BTreeMap::new(), used_counters: std::collections::BTreeSet::new(), local_lanes, axis_memo: BTreeMap::new(), loop_depth: 0, tile_bounds: BTreeMap::new(), naga_fn, module: &kernel.module, local_prefix: String::new(), inline_result: None, inline_values: None, bound_lets, stale_lets, let_regs: HashMap::new(), inlined: HashMap::new(), stored_seen: std::collections::BTreeSet::new(), counter_offsets: HashMap::new(), counter_exprs: HashMap::new(), pointer_args: HashMap::new(), array_args: HashMap::new(), mode_full: source.lines().any(|l| l.trim() == "// mode full"), open_thread_index: false };
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
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }) => ("real", "f32.real"),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, .. }) => ("index", "u32.index"),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Sint, .. }) => ("int", "i32.int"),
            naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Bool, .. }) => ("flag", "bool.flag"),
            // Any other type (an f16, a vector, a matrix, an array, a struct): its constant value, built from literals.
            _ => {
                let value = ConstValue::of(&kernel.module, init, gv.ty).map_err(|e| format!("the private variable {name}'s initializer: {e}"))?;
                if value.is_zero() {
                    continue;
                }
                let tree = ctx.const_tree(&value, gv.ty).map_err(|e| format!("the private variable {name}'s initializer: {e}"))?;
                let dual = dual_of_type(&kernel.module, gv.ty).ok_or_else(|| format!("the private variable {name} has no dual"))?;
                stmts.extend(ctx.store(&Tree::new(&format!("store.private.{name}"), dual_ty(kingdom, &dual)?, vec![tree]), "L0")?);
                continue;
            }
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
    // the grid carries them as a list of declarations (of the arrays this
    // entry point uses: a module's others are its other entry points').
    let grid_tree = if ctx.arrays_used.is_empty() {
        grid_tree
    } else {
        let decl_ty = dual_ty(kingdom, "wg.decl")?;
        let decls_ty = dual_ty(kingdom, "wg.decls")?;
        let decls: Vec<Tree> = ctx
            .array_lens
            .iter()
            .filter(|(which, _)| ctx.arrays_used.contains(*which))
            .map(|(which, len)| -> Result<Tree, String> {
                let (row, arr_dual) = if which.starts_with("wg.au") {
                    ("wg.decl_au", "wgmem<atomic<u32>>.index")
                } else if which.starts_with("wg.ai") {
                    ("wg.decl_ai", "wgmem<atomic<i32>>.int")
                } else if which.starts_with("wg.v2") {
                    ("wg.decl_v2", "wgmem<vec2<f32>>.real")
                } else if which.starts_with("wg.u") {
                    ("wg.decl_u", "wgmem<u32>.index")
                } else if which.starts_with("wg.n") {
                    ("wg.decl_n", "wgmem<i32>.int")
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
    let tree = crate::wgsl::safe_karva::conform(&tree, tree.ty, kingdom, &ctx.per_kernel_rows)?;
    Ok(((tree, ctx.notes, ctx.registers, (ctx.arrays, ctx.array_lens)), ctx.open_thread_index && preset.is_none(), rows_x))
}

/// One step of a place inside an aggregate local: a field by name, or an element, column or lane by an index tree.
enum Step {
    Field(String),
    Index(Tree),
}

/// A place inside an aggregate local: the local's key (`local.private_p@g`), its naga type and the steps.
type AggregatePlace = (String, Handle<naga::Type>, Vec<Step>);

impl<'a> Ctx<'a> {
    /// The aggregate local a load or store of `rest` (`private_p@g.#.m.#@v2`, the name after `load.local.` or
    /// `store.local.`) is on, when its type is a value a register of the module's kingdom holds and the place goes
    /// into it (or is all of it): the local's key (`local.private_p@g`), its naga type and the steps.
    fn aggregate_place(&self, rest: &str, kids: &[Tree]) -> Result<Option<AggregatePlace>, String> {
        let Some((name, after)) = rest.split_once('@') else { return Ok(None) };
        let end = after.find(['.', '@']).unwrap_or(after.len());
        let (idx, tail) = after.split_at(end);
        let ty_handle = if idx == "g" {
            let Some(var) = name.strip_prefix("private_") else { return Ok(None) };
            let Some((_, gv)) = self.module.global_variables.iter().find(|(_, g)| g.space == naga::AddressSpace::Private && g.name.as_deref() == Some(var)) else { return Ok(None) };
            gv.ty
        } else {
            let Some(lv) = idx.parse::<usize>().ok().and_then(|h| self.naga_fn.local_variables.iter().nth(h)).map(|(_, l)| l) else { return Ok(None) };
            lv.ty
        };
        let Some(dual) = dual_of_type(self.module, ty_handle) else { return Ok(None) };
        if module_register_of_value(self.kingdom, &dual).is_none() {
            return Ok(None);
        }
        // (An array of an element the private arrays hold is one of them.)
        if matches!(&self.module.types[ty_handle].inner, naga::TypeInner::Array { base, .. } if crate::wgsl::aggregate::has_private_array(&crate::wgsl::interface::type_text(self.module, *base))) {
            return Ok(None);
        }
        let mut indexes = Vec::new();
        for k in kids {
            flat_indexes(k, &mut indexes);
        }
        let mut indexes = indexes.into_iter();
        let mut steps = Vec::new();
        for seg in tail.split('@').next().unwrap_or("").split('.').filter(|s| !s.is_empty()) {
            steps.push(if seg == "#" { Step::Index(indexes.next().ok_or_else(|| format!("{rest}: a place with fewer indexes than steps"))?) } else { Step::Field(seg.to_string()) });
        }
        Ok(Some((format!("local.{name}@{idx}"), ty_handle, steps)))
    }

    /// The type reached by a step, and the row that reads it from a value of the type before.
    fn step_into(&mut self, here: &naga::TypeInner, step: &Step, whole: Tree) -> Result<(naga::TypeInner, Tree), String> {
        use naga::TypeInner;
        let text = |sc: &naga::Scalar| match (sc.kind, sc.width) {
            (naga::ScalarKind::Float, 2) => "f16",
            (naga::ScalarKind::Float, _) => "f32",
            (naga::ScalarKind::Sint, _) => "i32",
            (naga::ScalarKind::Uint, _) => "u32",
            _ => "bool",
        };
        match (here, step) {
            (TypeInner::Array { base, .. }, Step::Index(i)) => {
                let d = dual_of_type(self.module, *base).ok_or_else(|| "an element of an aggregate with no dual".to_string())?;
                let i = self.value(i)?;
                Ok((self.module.types[*base].inner.clone(), Tree::new("agg.at", ty(self, &d)?, vec![whole, i])))
            }
            (TypeInner::Struct { members, .. }, Step::Field(f)) => {
                let m = members.iter().find(|m| m.name.as_deref() == Some(f.as_str())).ok_or_else(|| format!("no member {f}"))?;
                let d = dual_of_type(self.module, m.ty).ok_or_else(|| format!("the member {f} has no dual"))?;
                Ok((self.module.types[m.ty].inner.clone(), Tree::new(&format!("agg.member_{f}"), ty(self, &d)?, vec![whole])))
            }
            (TypeInner::Matrix { rows, scalar, .. }, Step::Index(i)) => {
                let d = crate::wgsl::render::elem_dual(&format!("vec{}<{}>", *rows as u32, text(scalar))).ok_or_else(|| "a column with no dual".to_string())?;
                Ok((TypeInner::Vector { size: *rows, scalar: *scalar }, match literal_value(i) {
                    Some(c) => Tree::new(&format!("shape.col_{}", c as u32), ty(self, &d)?, vec![whole]),
                    None => Tree::new("shape.col_at", ty(self, &d)?, vec![whole, self.value(i)?]),
                }))
            }
            (TypeInner::Vector { scalar, .. }, Step::Index(i)) => {
                let d = crate::wgsl::render::elem_dual(text(scalar)).ok_or_else(|| "a lane with no dual".to_string())?;
                Ok((TypeInner::Scalar(*scalar), match literal_value(i) {
                    Some(l) => Tree::new(&format!("shape.access_{}", l as u32), ty(self, &d)?, vec![whole]),
                    None => Tree::new("shape.at", ty(self, &d)?, vec![whole, self.value(i)?]),
                }))
            }
            (other, _) => Err(format!("a step into {other:?} of an aggregate local is not a place the kingdom reads")),
        }
    }

    /// A load of an aggregate local or of a place inside one: the register read, then each step's row.
    fn aggregate_load(&mut self, t: &Tree, rest: &str) -> Result<Option<Tree>, String> {
        let Some((local, top, steps)) = self.aggregate_place(rest, &t.kids)? else { return Ok(None) };
        let dual = dual_of_type(self.module, top).ok_or("an aggregate local with no dual")?;
        let row = self.register(&local, &dual)?;
        let kind = row.chars().nth(4).unwrap_or('f');
        let mut cur = Tree::new(&format!("reg.get_{kind}"), ty(self, &dual)?, vec![Tree::new(&row, reg_ty(self, kind)?, vec![])]);
        let mut here = self.module.types[top].inner.clone();
        for step in &steps {
            let (next, tree) = self.step_into(&here, step, cur)?;
            (here, cur) = (next, tree);
        }
        cur.ty = t.ty;
        Ok(Some(cur))
    }

    /// A store to an aggregate local or to a place inside one: the register set to the value with that place replaced,
    /// the replacement a chain of `agg.with_*` rows over the reads of the parts on the way.
    fn aggregate_store(&mut self, tree: &Tree, rung: &str) -> Result<Option<Tree>, String> {
        let Some(rest) = tree.name.strip_prefix("store.local.") else { return Ok(None) };
        let Some((value, index_kids)) = tree.kids.split_last() else { return Ok(None) };
        let Some((local, top, steps)) = self.aggregate_place(rest, index_kids)? else { return Ok(None) };
        let dual = dual_of_type(self.module, top).ok_or("an aggregate local with no dual")?;
        let row = self.register(&local, &dual)?;
        let kind = row.chars().nth(4).unwrap_or('f');
        let reg = Tree::new(&row, reg_ty(self, kind)?, vec![]);
        let whole = Tree::new(&format!("reg.get_{kind}"), ty(self, &dual)?, vec![reg.clone()]);
        let v = self.value(value)?;
        let here = self.module.types[top].inner.clone();
        let new = self.update_place(&here, whole, &steps, v)?;
        let st = self.stmt_ty(rung)?;
        Ok(Some(Tree::new(&format!("reg.set_{kind}_{rung}"), st, vec![reg, new])))
    }

    /// `whole` of type `here` with the part `steps` reach replaced by `v`.
    fn update_place(&mut self, here: &naga::TypeInner, whole: Tree, steps: &[Step], v: Tree) -> Result<Tree, String> {
        use naga::TypeInner;
        let Some((step, more)) = steps.split_first() else { return Ok(v) };
        let (part_ty, part) = self.step_into(here, step, whole.clone())?;
        let inner = self.update_place(&part_ty, part, more, v)?;
        let out = whole.ty;
        match (here, step) {
            (TypeInner::Array { .. } | TypeInner::Matrix { .. } | TypeInner::Vector { .. }, Step::Index(i)) => {
                let i = self.value(i)?;
                Ok(Tree::new("agg.with_at", out, vec![whole, i, inner]))
            }
            (TypeInner::Struct { .. }, Step::Field(f)) => Ok(Tree::new(&format!("agg.with_{f}"), out, vec![whole, inner])),
            (other, _) => Err(format!("a write into {other:?} inside an aggregate local has no row")),
        }
    }

    /// A place inside a uniform binding that is an aggregate value (`u[2].m` over `u : array<S, 4>`): the binding's whole
    /// value (its row `load.<var>`), then each step's row.
    fn aggregate_binding_load(&mut self, t: &Tree, rest: &str) -> Result<Option<Tree>, String> {
        let var = rest.split(['.', '@']).next().unwrap_or(rest);
        let tail = &rest[var.len()..];
        let tail = tail.split('@').next().unwrap_or("");
        if tail.is_empty() {
            return Ok(None);
        }
        let Some(top) = self.module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(var) && g.space == naga::AddressSpace::Uniform).map(|(_, g)| g.ty) else { return Ok(None) };
        let mut here = self.module.types[top].inner.clone();
        let Some(dual) = dual_of_type(self.module, top) else { return Ok(None) };
        let row = format!("load.{var}");
        // (An element the interface has a row for is that row's: `load.u` over an index.)
        if self.per_kernel_rows.iter().any(|p| p.0 == row && !p.1.is_empty()) {
            return Ok(None);
        }
        let Some((name, out)) = self.per_kernel_rows.iter().find(|p| p.0 == row && p.1.is_empty()).map(|p| (p.0.clone(), p.2)) else { return Ok(None) };
        if self.kingdom.dual(out).map(|d| d.dual.as_str()) != Some(dual.as_str()) {
            return Ok(None);
        }
        let mut indexes = Vec::new();
        for k in &t.kids {
            flat_indexes(k, &mut indexes);
        }
        let mut indexes = indexes.into_iter();
        let mut cur = Tree::new(&name, out, vec![]);
        for seg in tail.split('.').filter(|s| !s.is_empty()) {
            let step = if seg == "#" { Step::Index(indexes.next().ok_or_else(|| format!("{rest}: a place with fewer indexes than steps"))?) } else { Step::Field(seg.to_string()) };
            let (next, tree) = self.step_into(&here, &step, cur)?;
            (here, cur) = (next, tree);
        }
        cur.ty = t.ty;
        Ok(Some(cur))
    }
}


/// A constant value of a type, evaluated from a module's global expressions (a `var<private>` initializer): scalars as
/// numbers, aggregates as their parts.
enum ConstValue {
    Scalar(f64),
    Parts(Vec<ConstValue>),
}

impl ConstValue {
    fn of(module: &naga::Module, expr: Handle<naga::Expression>, ty: Handle<naga::Type>) -> Result<ConstValue, String> {
        use naga::{Expression as E, Literal as L, TypeInner as T};
        Ok(match &module.global_expressions[expr] {
            E::Literal(l) => ConstValue::Scalar(match *l {
                L::F32(x) => f64::from(x),
                L::F16(x) => f64::from(f32::from(x)),
                L::F64(x) => x,
                L::U32(x) => f64::from(x),
                L::I32(x) => f64::from(x),
                L::U64(x) => x as f64,
                L::I64(x) => x as f64,
                L::Bool(b) => f64::from(u8::from(b)),
                L::AbstractInt(x) => x as f64,
                L::AbstractFloat(x) => x,
                L::U16(x) => f64::from(x),
                L::I16(x) => f64::from(x),
            }),
            E::ZeroValue(zt) => Self::zero(module, *zt)?,
            E::Constant(c) => Self::of(module, module.constants[*c].init, module.constants[*c].ty)?,
            E::Splat { size, value } => {
                let v = Self::of(module, *value, ty)?;
                ConstValue::Parts((0..*size as usize).map(|_| v.clone()).collect())
            }
            E::Compose { components, .. } => {
                // (A vector composed of vectors and scalars is its lanes in order.)
                let mut parts = Vec::new();
                for c in components {
                    match Self::of(module, *c, ty)? {
                        ConstValue::Parts(p) if matches!(module.types[ty].inner, T::Vector { .. }) => parts.extend(p),
                        v => parts.push(v),
                    }
                }
                ConstValue::Parts(parts)
            }
            other => return Err(format!("the expression {other:?} is not a constant the reader evaluates")),
        })
    }

    fn zero(module: &naga::Module, ty: Handle<naga::Type>) -> Result<ConstValue, String> {
        use naga::TypeInner as T;
        Ok(match &module.types[ty].inner {
            T::Scalar(_) => ConstValue::Scalar(0.0),
            T::Vector { size, .. } => ConstValue::Parts((0..*size as usize).map(|_| ConstValue::Scalar(0.0)).collect()),
            T::Matrix { columns, rows, .. } => ConstValue::Parts((0..*columns as usize).map(|_| ConstValue::Parts((0..*rows as usize).map(|_| ConstValue::Scalar(0.0)).collect())).collect()),
            T::Array { base, size: naga::ArraySize::Constant(n), .. } => ConstValue::Parts((0..n.get() as usize).map(|_| Self::zero(module, *base)).collect::<Result<_, _>>()?),
            T::Struct { members, .. } => ConstValue::Parts(members.iter().map(|m| Self::zero(module, m.ty)).collect::<Result<_, _>>()?),
            other => return Err(format!("the type {other:?} has no zero value")),
        })
    }

    fn is_zero(&self) -> bool {
        match self {
            ConstValue::Scalar(v) => *v == 0.0 && v.is_sign_positive(),
            ConstValue::Parts(p) => p.iter().all(ConstValue::is_zero),
        }
    }
}

impl Clone for ConstValue {
    fn clone(&self) -> Self {
        match self {
            ConstValue::Scalar(v) => ConstValue::Scalar(*v),
            ConstValue::Parts(p) => ConstValue::Parts(p.clone()),
        }
    }
}

impl<'a> Ctx<'a> {
    /// The tree of a constant value of type `ty`: literals composed into a vector or a matrix, an aggregate's nonzero
    /// parts written into its zero.
    fn const_tree(&mut self, value: &ConstValue, ty_handle: Handle<naga::Type>) -> Result<Tree, String> {
        use naga::TypeInner as T;
        let inner = self.module.types[ty_handle].inner.clone();
        let dual = dual_of_type(self.module, ty_handle).ok_or("a constant of a type with no dual")?;
        match (&inner, value) {
            (T::Scalar(sc), ConstValue::Scalar(v)) => {
                let form = match sc.kind {
                    naga::ScalarKind::Float => "real",
                    naga::ScalarKind::Sint => "int",
                    naga::ScalarKind::Uint => "index",
                    _ => "flag",
                };
                let v = if form == "real" { f64::from(*v as f32) } else { *v };
                Ok(Tree::literal(&format!("literal.{form}"), ty(self, &dual)?, v))
            }
            (T::Vector { .. }, ConstValue::Parts(_)) => self.const_vector(value, &inner),
            (T::Matrix { columns, rows, scalar }, ConstValue::Parts(cols)) => {
                let col_ty = T::Vector { size: *rows, scalar: *scalar };
                // (A column's vector type may not be declared by the module: build it from its parts.)
                let mut columns_tree = Vec::new();
                for c in cols {
                    columns_tree.push(self.const_vector(c, &col_ty)?);
                }
                if scalar.width == 4 {
                    let name = match *columns as u32 {
                        2 => "shape.compose_mat",
                        3 => "shape.compose_mat3",
                        _ => "shape.compose_mat4",
                    };
                    return Ok(Tree::new(name, ty(self, &dual)?, columns_tree));
                }
                let mut acc = Tree::new("agg.zero", ty(self, &dual)?, vec![]);
                for (i, c) in columns_tree.into_iter().enumerate() {
                    let index = Tree::literal("literal.index", ty(self, "u32.index")?, i as f64);
                    acc = Tree::new("agg.with_at", ty(self, &dual)?, vec![acc, index, c]);
                }
                Ok(acc)
            }
            (T::Array { base, .. }, ConstValue::Parts(parts)) => {
                let mut acc = Tree::new("agg.zero", ty(self, &dual)?, vec![]);
                for (i, p) in parts.iter().enumerate() {
                    if p.is_zero() {
                        continue;
                    }
                    let element = self.const_tree(p, *base)?;
                    let index = Tree::literal("literal.index", ty(self, "u32.index")?, i as f64);
                    acc = Tree::new("agg.with_at", ty(self, &dual)?, vec![acc, index, element]);
                }
                Ok(acc)
            }
            (T::Struct { members, .. }, ConstValue::Parts(parts)) => {
                let mut acc = Tree::new("agg.zero", ty(self, &dual)?, vec![]);
                for (m, p) in members.iter().zip(parts) {
                    if p.is_zero() {
                        continue;
                    }
                    let name = m.name.clone().ok_or("a member with no name")?;
                    let part = self.const_tree(p, m.ty)?;
                    acc = Tree::new(&format!("agg.with_{name}"), ty(self, &dual)?, vec![acc, part]);
                }
                Ok(acc)
            }
            (other, _) => Err(format!("a constant of {other:?} has no tree")),
        }
    }

    /// A constant vector of the type `vector` (which the module may not declare as a type of its own).
    fn const_vector(&mut self, value: &ConstValue, vector: &naga::TypeInner) -> Result<Tree, String> {
        use naga::TypeInner as T;
        let (T::Vector { size, scalar }, ConstValue::Parts(lanes)) = (vector, value) else { return Err("a column that is not a vector of numbers".into()) };
        let lane_name = match (scalar.kind, scalar.width) {
            (naga::ScalarKind::Float, 2) => "f16",
            (naga::ScalarKind::Float, _) => "f32",
            (naga::ScalarKind::Sint, _) => "i32",
            (naga::ScalarKind::Uint, _) => "u32",
            _ => "bool",
        };
        let lane_dual = crate::wgsl::render::elem_dual(lane_name).ok_or("a lane with no dual")?;
        let lit = |ctx: &mut Self, v: &ConstValue| -> Result<Tree, String> {
            let ConstValue::Scalar(x) = v else { return Err("a lane that is not a number".into()) };
            let x = if lane_name.starts_with('f') { f64::from(*x as f32) } else { *x };
            let form = lane_dual.split('.').nth(1).unwrap_or("real");
            Ok(Tree::literal(&format!("literal.{form}"), ty(ctx, &lane_dual)?, x))
        };
        let slot = |k: usize| crate::wgsl::render::elem_dual(&format!("vec{k}<{lane_name}>"));
        let n = *size as usize;
        let mut acc = Tree::new("shape.compose2", ty(self, &slot(2).ok_or("no vec2 dual")?)?, vec![lit(self, &lanes[0])?, lit(self, &lanes[1])?]);
        for k in 3..=n {
            acc = Tree::new(&format!("shape.compose{k}"), ty(self, &slot(k).ok_or("no vector dual")?)?, vec![acc, lit(self, &lanes[k - 1])?]);
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::render::{gram_role, validate_text, Renderer};
    use crate::wgsl::table::FunctionTable;

    const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");
    const BLOCKED: &str = include_str!("../../kingdoms/wgsl/gram/blocked.wgsl");

    /// A grid header of other than four numbers is no header (a model wrote
    /// `// grid 256 1 1`; reading it panicked a fit, 2026-10-16).
    #[test]
    fn a_grid_header_of_three_numbers_is_no_header() {
        assert_eq!(grid_header("// grid 256 1 1\nfn f() {}"), None);
        assert!(!grid_rows_x("// grid 256 1 rows-x\n"));
        assert_eq!(grid_header("// grid 7 17 8 8\n"), Some((7, 17, 8, 8)));
        assert!(grid_rows_x("// grid 7 17 8 8 rows-x\n"));
    }

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
    /// THE CUNEUS KERNELS (2026-10-19, the naga/wgpu 30 upgrade): the four
    /// compute kernels of cuneus 0.6.4 (written for naga 30) that bind no
    /// image or sampler, tests/corpus/cuneus. Each parses with naga 30
    /// (naga 0.20 parsed none of the radix sort's function-local consts) and
    /// is read; how many read is printed with each one's reason, no bar.
    #[test]
    fn the_cuneus_kernels_parse_and_are_read() {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/cuneus"));
        let k = WgslKingdom::load_safe();
        let (mut parsed, mut validated, mut read_n) = (0usize, 0usize, 0usize);
        let names = ["instances", "meshlab_swarm", "radix_sort", "skin"];
        for name in names {
            let src = std::fs::read_to_string(dir.join(format!("{name}.wgsl"))).unwrap_or_else(|e| panic!("{name}: {e}"));
            let module = crate::wgsl::naga_shape::parse(&src).unwrap_or_else(|e| panic!("{name}: naga 30 parses it: {e}"));
            parsed += 1;
            if let Err(e) = crate::wgsl::target::Target::recorded().validate(&module, &src) {
                println!("cuneus {name:<14} validates: no ({})", e.lines().find(|l| l.contains('=')).unwrap_or("").trim());
                continue;
            }
            validated += 1;
            match read_safe(&src, &k) {
                Ok(r) => {
                    read_n += 1;
                    println!("cuneus {name:<14} reads: {} nodes", r.tree.size());
                }
                Err(e) => println!("cuneus {name:<14} reads: no ({})", e.chars().take(160).collect::<String>()),
            }
        }
        println!("cuneus: {parsed} of {} parse with naga 30, {validated} validate, {read_n} read", names.len());
        assert_eq!(parsed, names.len());
        // (2026-10-19: instances, meshlab_swarm and skin read; the radix sort's loops step by a
        // dispatch-sized stride, no literal.)
        for name in ["instances", "meshlab_swarm", "skin"] {
            let src = std::fs::read_to_string(dir.join(format!("{name}.wgsl"))).unwrap();
            let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{name}: {e}"));
            // (A module that declares aggregate types reads in the kingdom extended by them.)
            let kr = r.kingdom.as_ref().unwrap_or(&k);
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, kr, &r.role).unwrap_or_else(|e| panic!("{name}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{name}: {e}"));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("{name}: read back: {e}"));
            let kb = back.kingdom.as_ref().unwrap_or(&k);
            let again = read_safe(&crate::wgsl::safe_karva::render_tree_with(&back.tree, kb, &back.role).unwrap().text, &k).unwrap();
            assert!(again.tree == back.tree, "{name}: a fixpoint: {:?}", crate::wgsl::render::first_difference(&back.tree, &again.tree));
        }
    }

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
        // (A uniform bound since legality item 5: the loop may hold a barrier.)
        assert!(sx.contains("(bound.of_uniform (uniform.min (uniform.of_lit (literal.lit4096 16)) (uniform.div (uniform.add (uniform.sub_floor"), "{sx}");
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
        // (The source's `pc` keeps its own name, parr.c, since 2026-10-13.)
        let fma = text.lines().find(|l| l.contains("pc[") && l.contains(" * ")).unwrap_or_else(|| panic!("no multiply-add\n{text}"));
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
        // (A uniform bound at a C rung is uncapped: bound.x_of_uniform.)
        assert!(sx.contains("(loop.tilex_C0 (bound.x_of_uniform uniform.cfg_m)"), "{sx}");
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
        fn set_len(t: &mut Tree, to: f64) {
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
    /// validates and reads back. A compare-exchange is refused by name: the
    /// kingdom has no row for it (naga 0.20's Metal backend did not implement
    /// it; naga 30's does).
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

    /// A SECOND INTERFACE (PLAN_constrained_mutation.md section 2.4,
    /// 2026-10-12): UltraDim's exact re-score, a dense batched cosine score of
    /// queries against candidates with a Kahan-summed dot product, each thread
    /// owning one (query, candidate) cell. It reads under its declared
    /// interface (`// interface cosine`), its second input through `load.qry`;
    /// it renders with its own names, validates, reads back under the same
    /// interface and renders valid again, and every layer of the checker
    /// passes it.
    #[test]
    fn the_cosine_interface_reads_and_renders_a_rescore_kernel() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/interfaces/cosine_rescore.wgsl");
        let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.role.name, "cosine");
        let sx = r.tree.to_sexpr();
        assert!(sx.contains("load.qry") && sx.contains("load.phi_safe") && sx.contains("store.h_O0"), "{sx}");
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        for want in ["// interface cosine", "struct Params { nc: u32, dim: u32, nq: u32, q_off: u32, };", "var<storage, read> qry: array<f32>;", "t < p.dim", "< p.nq) & ", "score[", "qry[", "cand["] {
            assert!(out.text.contains(want), "{want}:\n{}", out.text);
        }
        for gram in ["phi[", "h[", "cfg."] {
            assert!(!out.text.contains(gram), "{gram}:\n{}", out.text);
        }
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert_eq!(back.role.name, "cosine");
        // (Not the same text: a re-read numbers registers by first use and
        // reads the store's own guard as an if, as for a Gram kernel.)
        let again = crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &back.role).unwrap().text;
        validate_text(&again).unwrap_or_else(|e| panic!("{e}\n{again}"));
        crate::wgsl::check::check(&r.tree, &k, &r.role, &crate::wgsl::target::Target::recorded()).unwrap_or_else(|e| panic!("{e}"));
    }

    /// A STORAGE ATOMIC (the coverage probe `atomic_storage`, 2026-10-12):
    /// the Gram's interface with `counter: atomic<u32>` declared
    /// (`// interface gram_counter`); `atomicAdd(&counter, 1u)` reads as its
    /// per-kernel row, renders, validates and passes every layer.
    #[test]
    fn a_storage_atomic_the_interface_declares_reads() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/interfaces/atomic_storage.wgsl");
        let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.role.name, "gram_counter");
        assert!(r.tree.to_sexpr().contains("(atomic.counter_add (literal.index 1))"), "{}", r.tree.to_sexpr());
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        assert!(out.text.contains("var<storage, read_write> counter: atomic<u32>;") && out.text.contains("atomicAdd(&counter, "), "{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        crate::wgsl::check::check(&r.tree, &k, &r.role, &crate::wgsl::target::Target::recorded()).unwrap_or_else(|e| panic!("{e}"));
        // Without the interface line the module's own interface is inferred:
        // the same one (2026-10-14; it was a refusal under the Gram's assumed).
        let inferred = read_safe(&src.replace("// interface gram_counter\n", ""), &k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((inferred.role.name.as_str(), &inferred.tree), ("gram_counter", &r.tree));
    }

    /// The v24 contract's bindings (fuller docs/PLAN_v24_ingest_evolution.md
    /// section 1), the index arena an array of atomics, and an i32 atomic
    /// array beside it: the declarations of the storage-atomic fixtures.
    const V24_BINDINGS: &str = "struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
@group(0) @binding(6) var<storage, read_write> score: array<atomic<i32>>;
";

    /// An ingest pass over the v24 bindings: a bucket from a key's bits,
    /// a slot from atomicAdd, the slot stored with atomicStore, atomicMax,
    /// atomicLoad, a discarded atomicOr and an i32 atomicAdd, after the
    /// thread guard of the arrived rows.
    const V24_INGEST: &str = "@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = cfg.arrived_begin + gid.x;
    if (row >= cfg.arrived_end) { return; }
    let bucket = bitcast<u32>(keys[row * 2048u]) & 1023u;
    let slot = atomicAdd(&index_u32[bucket], 1u);
    atomicStore(&index_u32[1024u + row], bucket * 64u + slot);
    _ = atomicMax(&index_u32[2048u + bucket], row);
    let seen = atomicLoad(&index_u32[3072u]);
    _ = atomicOr(&index_u32[3073u], seen);
    _ = atomicAdd(&score[bucket], -1);
}
";

    /// STORAGE ATOMICS ON AN ARRAY (B6 of PLAN_v24_ingest_evolution,
    /// 2026-10-17): an index arena declared `array<atomic<u32>>` (and an
    /// `array<atomic<i32>>`) reads as the interface's per-kernel rows, every
    /// atomic after the thread guard under it; the text renders each index
    /// as written, naga validates it, it reads back to the same
    /// tree, and the checker passes every layer. A compare-exchange is named.
    #[test]
    fn storage_atomics_on_an_array_read_render_and_read_back_to_the_same_tree() {
        let k = WgslKingdom::load_safe();
        let src = format!("{V24_BINDINGS}{V24_INGEST}");
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((r.role.name.as_str(), r.role.entry.as_str()), (crate::wgsl::render::INFERRED, "ingest"));
        let sx = r.tree.to_sexpr();
        for want in ["(atomic.index_u32_add ", "(store.index_u32_C0 ", "(atomic.index_u32_max ", "(load.index_u32 ", "(atomic.index_u32_or ", "(atomic.score_add ", "(stmt.if_L0 (logic.not "] {
            assert!(sx.contains(want), "{want}: {sx}");
        }
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        for want in ["var<storage, read_write> index_u32: array<atomic<u32>>;", "atomicAdd(&index_u32[", "atomicStore(&index_u32[", "atomicLoad(&index_u32[", "atomicAdd(&score[", "fn ingest("] {
            assert!(out.text.contains(want), "{want}:\n{}", out.text);
        }
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        crate::wgsl::check::check(&r.tree, &k, &r.role, &crate::wgsl::target::Target::recorded()).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        // The kingdom has no compare-exchange row: no spelling.
        let cas = src.replace("let slot = atomicAdd(&index_u32[bucket], 1u);", "let slot = atomicCompareExchangeWeak(&index_u32[bucket], 0u, 1u).old_value;");
        assert_ne!(cas, src);
        let Err(e) = read_safe(&cas, &k) else { panic!("a compare-exchange read") };
        assert!(e.contains("atomicCompareExchangeWeak has no safe spelling"), "{e}");
    }

    /// SEVERAL ENTRY POINTS IN ONE MODULE (B6 of PLAN_v24_ingest_evolution,
    /// 2026-10-17): v24's `ingest` (64 threads, atomics on the index arena)
    /// and `search` (32 threads, a store of each query's candidate) over the
    /// shared bindings, a workgroup variable both use and one each uses
    /// alone. The interface is both entry points' (`cands` an output though
    /// only `search` writes it); the tree is `kernel.module_2`, one safe
    /// kernel per entry point; the text renders both functions and every
    /// shared declaration once, naga validates it, it reads back to the same
    /// tree, and each entry point passes every layer of the checker.
    #[test]
    fn a_module_of_two_entry_points_reads_renders_and_reads_back_to_the_same_tree() {
        let k = WgslKingdom::load_safe();
        let src = format!(
            "{V24_BINDINGS}var<workgroup> tile: array<u32, 64>;
var<workgroup> best: array<f32, 128>;
var<workgroup> spill: array<u32, 256>;
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    tile[lid.x] = gid.x;
    spill[lid.x] = gid.x * 3u;
    workgroupBarrier();
    let row = cfg.arrived_begin + tile[(lid.x + 1u) & 63u] + spill[(lid.x + 2u) & 255u];
    if (row >= cfg.arrived_end) {{ return; }}
    let bucket = bitcast<u32>(keys[row * 2048u]) & 1023u;
    let slot = atomicAdd(&index_u32[bucket], 1u);
    atomicStore(&index_u32[1024u + row], bucket * 64u + slot);
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    tile[lid.x] = gid.x * 2u;
    best[lid.x] = queries[gid.x * 2048u];
    workgroupBarrier();
    let q = gid.x;
    if (q >= cfg.n_queries) {{ return; }}
    let b = tile[(lid.x + 1u) & 63u] & 1023u;
    cands[q] = atomicLoad(&index_u32[b]) + u32(best[lid.x & 127u] > 0.0);
}}
"
        );
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.role.entries, vec!["ingest".to_string(), "search".to_string()]);
        assert!(r.role.per_kernel.iter().any(|p| p.name == "store.cands_O0"), "cands is an output of the module");
        assert_eq!(r.tree.name, "kernel.module_2");
        assert!(r.tree.kids.iter().all(|e| e.name == "kernel.entry_safe"), "{}", r.tree.to_sexpr());
        // `tile` is one row in both entry points; `spill` and `best` each one entry point's.
        let decls = |e: &Tree| -> Vec<String> {
            fn walk(t: &Tree, out: &mut Vec<String>) {
                if t.name.starts_with("wg.decl_") {
                    out.push(format!("{} {}", t.kids[0].name, t.kids[1].konst));
                }
                t.kids.iter().for_each(|k| walk(k, out));
            }
            let mut out = Vec::new();
            walk(&e.kids[0], &mut out);
            out
        };
        let (di, ds) = (decls(&r.tree.kids[0]), decls(&r.tree.kids[1]));
        assert_eq!(di.len(), 2, "{di:?}");
        assert_eq!(ds.len(), 2, "{ds:?}");
        assert_eq!(di.iter().filter(|d| ds.contains(d)).count(), 1, "one shared variable: {di:?} {ds:?}");
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        for want in ["// entry ingest grid 64 1 1 1", "// entry search grid 32 1 1 1", "@workgroup_size(64, 1, 1)\nfn ingest(", "@workgroup_size(32, 1, 1)\nfn search(", "atomicAdd(&index_u32[", "atomicLoad(&index_u32[", "cands["] {
            assert!(out.text.contains(want), "{want}:\n{}", out.text);
        }
        for once in ["struct Cfg", "var<storage, read_write> index_u32", "var<workgroup> wg_ua: array<u32, 64>;"] {
            assert_eq!(out.text.matches(once).count(), 1, "{once} once:\n{}", out.text);
        }
        assert_eq!(out.entries.iter().map(|(n, g)| (n.as_str(), g.x)).collect::<Vec<_>>(), vec![("ingest", 64), ("search", 32)]);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        assert_eq!(back.role.entries, r.role.entries);
        // Under a one-dimensional dispatch (the v24 problem's: a workgroup per
        // 64 rows or 32 queries along x), each entry point is proved.
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        // The genome: encoded, decoded and projected under the module's root, the tree.
        let h = 128usize;
        let slots = crate::wgsl::safe_karva::split(&r.tree, h).unwrap().tail.len().div_ceil(8) * 8;
        let c = crate::wgsl::safe_karva::encode(&r.tree, h, slots, 17).unwrap_or_else(|e| panic!("{e}"));
        let pset = crate::homeotic::pset_with_hrefs(&c.pset, slots);
        let decoded: Vec<String> = c.genes.iter().map(|(hd, tl)| crate::karva::karva_to_terms_generic(hd, tl, &pset)).collect::<Result<_, _>>().unwrap();
        let rebuilt = crate::wgsl::safe_karva::build(&crate::wgsl::safe_karva::unfold_decoded(&decoded, c.folded.filled).unwrap(), &k, &r.role).unwrap();
        assert!(rebuilt == r.tree, "through the genome: {:?}", crate::wgsl::render::first_difference(&r.tree, &rebuilt));
        // The outline names each entry point's statements.
        let outline = statement_outline(&src).unwrap();
        assert!(outline.iter().any(|l| l.starts_with("ingest [")) && outline.iter().any(|l| l.starts_with("search [")), "{outline:?}");
    }

    /// RANDOM MODULES (2026-10-17, for phylu's v24 generation 0 and
    /// mutation): the sampler draws a `kernel.module_2` under the v24
    /// interface, one safe kernel per entry point; every drawn module of
    /// 1,000 renders both entry points, naga validates it and it reads back
    /// under the same interface, as a module of the same entry points (1,000
    /// of 1,000, asserted). Measured beside it, with floors: the modules that
    /// parse back to the tree drawn in its canonical spelling (AST to AST),
    /// the ones whose text read back is a fixpoint, and the ones that pass
    /// every layer under a 1-D contract (a drawn store at an arbitrary index
    /// of `cands` is rightly refused by layer 7).
    #[test]
    fn random_module_trees_render_validate_and_read_back() {
        let k = WgslKingdom::load_safe();
        let src = format!("{V24_BINDINGS}@compute @workgroup_size(64, 1, 1)\nfn ingest(@builtin(global_invocation_id) gid: vec3<u32>) {{ _ = atomicAdd(&index_u32[gid.x], 1u); index_f32[gid.x] = keys[gid.x]; }}\n@compute @workgroup_size(32, 1, 1)\nfn search(@builtin(global_invocation_id) gid: vec3<u32>) {{ cands[gid.x] = atomicLoad(&index_u32[gid.x]) + u32(queries[gid.x] > 0.0); }}\n");
        let role = crate::wgsl::render::interface_of(&src).unwrap();
        assert_eq!(role.entries, ["ingest", "search"]);
        let root = dual_ty(&k, "kernel.module").unwrap();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let checker = crate::wgsl::check::Checker::with_contract(&k, &role, &crate::wgsl::target::Target::recorded(), contract);
        let n = 1000u64;
        let (mut read_back, mut same, mut fixpoint, mut legal) = (0usize, 0usize, 0usize, 0usize);
        let mut illegal: BTreeMap<String, usize> = BTreeMap::new();
        let mut failures = Vec::new();
        for seed in 0..n {
            let tree = crate::wgsl::render::sample(&k, &role, root, 6, seed).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            assert_eq!((tree.name.as_str(), tree.kids.len()), ("kernel.module_2", 2), "seed {seed}");
            let verdict = (|| -> Result<(Tree, bool), String> {
                let out = crate::wgsl::safe_karva::render_tree_with(&tree, &k, &role).map_err(|e| format!("render: {e}"))?;
                validate_text(&out.text).map_err(|e| format!("naga: {}", e.lines().next().unwrap_or("")))?;
                let back = read_safe(&out.text, &k).map_err(|e| format!("read: {}", e.chars().take(200).collect::<String>()))?;
                if back.role.entries != role.entries || back.tree.name != "kernel.module_2" {
                    return Err(format!("read back as {} under {:?}", back.tree.name, back.role.entries));
                }
                let again = crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &role).map_err(|e| format!("re-render: {e}"))?.text;
                let third = read_safe(&again, &k).ok().and_then(|b| crate::wgsl::safe_karva::render_tree_with(&b.tree, &k, &role).ok()).map(|r| r.text);
                Ok((back.tree, third.as_deref() == Some(again.as_str())))
            })();
            match verdict {
                Ok((back, fix)) => {
                    read_back += 1;
                    same += usize::from(back == crate::wgsl::canonical::canonical(&tree, &k, &role));
                    fixpoint += usize::from(fix);
                    match checker.check(&tree) {
                        Ok(()) => legal += 1,
                        Err(e) => *illegal.entry(format!("{} {}", e.layer.number(), e.reason)).or_insert(0usize) += 1,
                    }
                }
                Err(e) => failures.push(format!("seed {seed}: {e}")),
            }
        }
        println!("random modules: {read_back} of {n} read back; {same} to the tree drawn (canonical); {fixpoint} to a fixpoint; {legal} pass every layer under a 1-D contract");
        println!("the layers the others fail (layer, reason): {illegal:?}");
        assert!(failures.is_empty(), "{} of {n}:\n{}", failures.len(), failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n"));
        assert!(same >= MODULE_AST_FLOOR && fixpoint >= MODULE_FIXPOINT_FLOOR && legal >= MODULE_LEGAL_FLOOR, "below the floors reached: AST {same} (floor {MODULE_AST_FLOOR}), fixpoint {fixpoint} ({MODULE_FIXPOINT_FLOOR}), legal {legal} ({MODULE_LEGAL_FLOOR})");
    }

    /// A kernel's rendering run beside the kernel on the CPU executor, entry
    /// point by entry point from the same inputs (`roundtrip::inputs`): Ok
    /// when every storage buffer holds the same bits after each, else the
    /// first entry point, buffer and word that differ; Ok(false) when the
    /// CPU executor cannot run the source itself (nothing to compare).
    /// (`roundtrip::same_values` runs a module's first entry point only.)
    fn same_values_every_entry(src: &str, k: &WgslKingdom, workgroups: [u32; 3]) -> Result<bool, String> {
        let read = read_safe(src, k)?;
        let rendered = crate::wgsl::safe_karva::render_tree_with(&read.tree, k, &read.role)?.text;
        let a = crate::wgsl::naga_shape::parse(src)?;
        let b = crate::wgsl::naga_shape::parse(&rendered).map_err(|e| format!("{e}\n{rendered}"))?;
        let t = crate::wgsl::roundtrip::Trial::default();
        let inputs = crate::wgsl::roundtrip::inputs(&a, &t)?;
        let storage: Vec<String> = a.global_variables.iter().filter(|(_, g)| matches!(g.space, naga::AddressSpace::Storage { .. })).filter_map(|(_, g)| g.name.clone()).collect();
        for entry in crate::wgsl::interface::compute_entries(&a) {
            let Ok(ra) = crate::wgsl::cpu::run_entry(&a, &entry, inputs.clone(), workgroups, t.max_steps) else { return Ok(false) };
            let rb = crate::wgsl::cpu::run_entry(&b, &entry, inputs.clone(), workgroups, t.max_steps).map_err(|e| format!("{entry}: the rendering does not run: {e}"))?;
            for name in &storage {
                let (mut wa, mut wb) = (Vec::new(), Vec::new());
                if let Some(v) = ra.buffers.get(name) {
                    v.words(&mut wa);
                }
                if let Some(v) = rb.buffers.get(name) {
                    v.words(&mut wb);
                }
                if let Some(i) = (0..wa.len().max(wb.len())).find(|&i| wa.get(i) != wb.get(i)) {
                    return Err(format!("{entry}: {name}[{i}] is {:?} in the source, {:?} in the rendering\n{rendered}", wa.get(i), wb.get(i)));
                }
            }
        }
        Ok(true)
    }

    /// THE EMITTER COMPUTES WHAT THE KERNEL COMPUTES (2026-10-17, the
    /// harness agent's repro: run 1's rendering of the v24 reference gave ~2
    /// candidates a query against the text's 100). Three faults, each in a
    /// kernel run beside its rendering on the CPU executor: a 64-element u32
    /// local array (it was read into a 16-element private array, `& 15u`); a
    /// swap through lets of array elements stored before the lets' uses
    /// (`let ts = best[j]; best[j] = cs; cs = ts;`: the lets were inlined
    /// after the stores); a u32 constant past 2^24 (`const PAD: u32 =
    /// 0xFFFFFFFFu`, rendered 0u). And the v24 reference itself.
    #[test]
    fn the_rendering_computes_the_kernels_values() {
        let k = WgslKingdom::load_safe();
        let src = "const PAD: u32 = 0xFFFFFFFFu;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_s: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_r: array<u32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var best: array<f32, 64>;
    var rows: array<u32, 64>;
    for (var j = 0u; j < 64u; j++) { best[j] = -1.0; rows[j] = PAD; }
    for (var i = 0u; i < 40u; i++) {
        var cs = x[(gid.x * 40u + i) & 63u];
        var cr = i;
        for (var j = 0u; j < 64u; j++) {
            if (cs > best[j]) {
                let ts = best[j];
                let tr = rows[j];
                best[j] = cs;
                rows[j] = cr;
                cs = ts;
                cr = tr;
            }
        }
    }
    for (var j = 0u; j < 64u; j++) {
        out_s[gid.x * 64u + j] = best[j];
        out_r[gid.x * 64u + j] = select(rows[j], 7u, rows[j] == PAD);
    }
}
";
        assert!(same_values_every_entry(src, &k, [1, 1, 1]).unwrap_or_else(|e| panic!("{e}")), "the source runs on the CPU");
        let reference = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("problems/v24_ingest_small/reference.wgsl")).expect("the v24 reference");
        assert!(same_values_every_entry(&reference, &k, [2, 1, 1]).unwrap_or_else(|e| panic!("the v24 reference: {e}")), "the reference runs on the CPU");
    }

    /// A READ NEVER RUNS UNBOUNDED (2026-10-17, after run 1's 03:15Z hang):
    /// a kernel whose lets double at each step (`let a1 = c + c;`, then
    /// `let a2 = a1 + a1;` and on, over a uniform, inlined: 2^40 nodes) is
    /// stopped at READ_NODE_BOUND, by name, within a minute.
    #[test]
    fn a_read_that_would_expand_without_end_is_stopped_by_name() {
        let k = WgslKingdom::load_safe();
        let mut body = String::from("    let a0 = cfg.m + 1u;\n");
        for i in 1..=40 {
            body.push_str(&format!("    let a{i} = a{} + a{};\n", i - 1, i - 1));
        }
        let src = format!("struct Cfg {{ d: u32, m: u32, r_rows: u32, r_off: u32, }};\n@group(0) @binding(0) var<storage, read> phi: array<f32>;\n@group(0) @binding(1) var<storage, read_write> out: array<u32>;\n@group(0) @binding(2) var<uniform> cfg: Cfg;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n{body}    out[gid.x] = a40;\n}}\n");
        let t0 = std::time::Instant::now();
        let Err(e) = read_safe(&src, &k) else { panic!("a 2^40-node read finished") };
        assert!(e.contains("built more than") && e.contains("the read is stopped"), "{e}");
        assert!(t0.elapsed().as_secs() < 60, "stopped after {:?}", t0.elapsed());
        println!("stopped by name after {:?}", t0.elapsed());
    }

    /// RUN 1'S SEEDS RENDER TO THEIR OWN VALUES (2026-10-17): every raw model
    /// seed of logs/V24/run1/raw that the reader reads is run beside its
    /// rendering on the CPU executor, each entry point from the same inputs;
    /// every one the CPU executor runs computes the same storage buffers.
    /// (Run from the rendering, a seed is the kernel the evolution sees.)
    #[test]
    fn run_ones_readable_seeds_render_to_their_own_values() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs/V24/run1/raw");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        files.sort();
        let (mut read, mut same, mut not_runnable) = (0usize, 0usize, 0usize);
        let mut differ = Vec::new();
        for f in &files {
            let src = std::fs::read_to_string(f).unwrap();
            if read_safe(&src, &k).is_err() {
                continue;
            }
            read += 1;
            match same_values_every_entry(&src, &k, [2, 1, 1]) {
                Ok(true) => same += 1,
                Ok(false) => {
                    not_runnable += 1;
                    println!("not runnable on the CPU: {}", f.file_name().unwrap().to_string_lossy());
                }
                Err(e) => differ.push(format!("{}: {}", f.file_name().unwrap().to_string_lossy(), e.lines().next().unwrap_or(""))),
            }
        }
        println!("run 1's seeds: {} files, {read} read, {same} render to their own values, {not_runnable} not runnable on the CPU", files.len());
        assert!(differ.is_empty(), "{} of {read} differ:\n{}", differ.len(), differ.join("\n"));
    }

    /// A LOOP THAT BREAKS, STEPPED BY A LITERAL (2026-10-17, 4 of run 1's
    /// seeds: a search scanning candidates a stride at a time until a budget
    /// is met): with a literal bound and with an expression bound, it reads
    /// as the strided expression-bounded loop whose body breaks, renders,
    /// validates, reads back to the same tree and passes every layer.
    #[test]
    fn a_stepped_loop_that_breaks_reads() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        for bound in ["64u", "cfg.n_queries"] {
            let src = format!(
                "{V24_BINDINGS}@compute @workgroup_size(32, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>) {{
    _ = atomicAdd(&index_u32[lid.x], 1u);
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {{
    var got = 0u;
    if (gid.x < 4096u) {{
        for (var i = 0u; i < {bound}; i += 4u) {{
            if (got >= 3u) {{ break; }}
            got = got + (atomicLoad(&index_u32[i & 1023u]) & 1u);
        }}
        cands[gid.x] = got;
    }}
}}
"
            );
            let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{bound}: {e}"));
            assert!(r.tree.to_sexpr().contains("(loop.tilex_strided_brk_C0 "), "{bound}: {}", r.tree.to_sexpr());
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{bound}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{bound}: {e}\n{}", out.text));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("{bound}: read back: {e}\n{}", out.text));
            assert!(back.tree == r.tree, "{bound}: the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
            let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract.clone());
            checker.check(&r.tree).unwrap_or_else(|e| panic!("{bound}: {e}\n{}", out.text));
        }
    }

    /// BREAKS BELOW THE TOP OF A LOOP'S BODY (2026-10-17, v24 run 1's seeds
    /// 11, 17 and 20: a break after a store inside an if, a second break
    /// guard after statements, a rank counted until it is enough): the loop
    /// reads with its breaks lowered to a stop flag, renders, validates,
    /// reads back to the same tree, passes every layer and computes the
    /// source's values on the CPU executor. Beside it the thread id used as
    /// bits and as a shift count (`(lid.x >> 5u) << 5u`, `1u << lid.x`,
    /// seeds 20, 26 and 30) and twelve u32 workgroup arrays (seed20's nine).
    #[test]
    fn breaks_below_the_top_of_a_loop_body_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let arrays: String = (0..12).map(|i| format!("var<workgroup> w{i}: array<u32, 32>;\n")).collect();
        let src = format!(
            "{V24_BINDINGS}{arrays}@compute @workgroup_size(32, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>) {{
    _ = atomicAdd(&index_u32[lid.x], 1u);
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
{fill}    workgroupBarrier();
    var got = 0u;
    var rank = 0u;
    if (gid.x < 4096u) {{
        for (var i = 0u; i < cfg.n_queries; i = i + 1u) {{
            let v = atomicLoad(&index_u32[i & 1023u]);
            if ((v & 1u) == 1u) {{
                got = got + 1u;
                if (got >= 3u) {{
                    rank = rank + 7u;
                    break;
                }}
            }}
            rank = rank + 1u;
            if (rank >= 40u) {{ break; }}
            rank = rank + (w11[i & 31u] & 1u);
        }}
        cands[gid.x] = got + rank + (((lid.x >> 2u) << 2u) ^ lid.x) + (1u << lid.x);
    }}
}}
",
            fill = (0..12).map(|i| format!("    w{i}[lid.x] = lid.x + {i}u;\n")).collect::<String>()
        );
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        let sx = r.tree.to_sexpr();
        assert!(sx.contains("(loop.tilex_brk_C0 ") && sx.contains("(block.brk_B1 "), "{sx}");
        assert!(sx.contains("(wg.ul)") || sx.contains("wg.ul"), "the twelfth u32 workgroup array: {sx}");
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        assert_eq!(same_values_every_entry(&src, &k, [2, 1, 1]), Ok(true), "the rendering computes the source's values\n{}", out.text);
    }

    /// RUN 1'S OTHER GAPS (2026-10-17, v24 run 1's seeds 00, 02, 11, 24 and
    /// 31): a helper given `&acc` of a local array, a helper returning
    /// `vec2<f32>`, a workgroup atomic scalar, a lane of a storage vector
    /// element chosen at run time, a pass-0 branch that returns from inside
    /// it (its returns guard that branch, not the stores of the other pass)
    /// and a lone return below the top. The module reads, renders, validates,
    /// reads back to the same tree and computes the source's values on the
    /// CPU executor, entry point by entry point.
    #[test]
    fn run_ones_other_gaps_read() {
        let k = WgslKingdom::load_safe();
        let src = "struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wcnt: atomic<u32>;
fn addp(p: ptr<function, array<f32, 16>>, x: f32) {
    for (var i = 0u; i < 16u; i += 1u) {
        (*p)[i] = (*p)[i] + x * f32(i);
    }
}
fn two(r: u32) -> vec2<f32> {
    var s = 0.0;
    var n = 0.0;
    for (var d = 0u; d < 4u; d = d + 1u) {
        let x = keys[(r + d) & 63u][d & 3u];
        s = s + x;
        n = n + x * x;
    }
    return vec2<f32>(s, n);
}
@compute @workgroup_size(32, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    if (cfg.pass_index == 0u) {
        if (cfg.stage != 1u) {
            return;
        }
        index_f32[(wid.x * 32u + lid.x) & 1023u] = keys[lid.x & 63u][lid.x & 3u];
        return;
    }
    if (lid.x >= cfg.arrived_end) {
        return;
    }
    _ = atomicAdd(&index_u32[lid.x & 1023u], 1u);
}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        atomicStore(&wcnt, 0u);
    }
    workgroupBarrier();
    var acc: array<f32, 16>;
    if (gid.x < 4096u) {
        addp(&acc, queries[gid.x]);
        addp(&acc, queries[gid.x + 1u]);
        let v = two(gid.x);
        var s = v.x + v.y;
        for (var i = 0u; i < 16u; i += 1u) {
            s = s + acc[i];
        }
        let slot = atomicAdd(&wcnt, 1u);
        cands[gid.x] = u32(abs(s)) + slot;
    }
}
";
        let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}"));
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        assert!(!r.notes.iter().any(|n| n.contains("thread guard")) || out.text.matches("cfg.stage").count() <= 2, "a return inside the pass-0 branch guards the other pass's stores:\n{}", out.text);
        assert_eq!(same_values_every_entry(src, &k, [2, 1, 1]), Ok(true), "the rendering computes the source's values\n{}", out.text);
    }

    /// BARRIERS UNDER TESTS THE CHECKER PROVES UNIFORM (2026-10-17, v24 run
    /// 1's seeds 06, 21 and 29): a test of a count from workgroupUniformLoad
    /// over a loop that breaks when the count runs out (the guard folded into
    /// the bound) and holds a barrier, and a flag set under a uniform test
    /// over a barrier, read as `uflag.of_flag` and `uniform.of_checked`: each
    /// reads, renders, validates, reads back to the same tree, passes every
    /// layer and computes the source's values. The same count read from
    /// workgroup memory with no uniform load reads too, and layer 4 names its
    /// barrier in non-uniform control.
    #[test]
    fn barriers_under_tests_the_checker_proves_uniform_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let module = |load: &str| format!("{V24_BINDINGS}var<workgroup> wu2: u32;
var<workgroup> sv: array<f32, 256>;
@compute @workgroup_size(256)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    if (t == 0u) {{
        wu2 = cfg.stage;
    }}
    workgroupBarrier();
    let n = {load};
    if (n != 0u) {{
        for (var i = 0u; i < 1024u; i = i + 256u) {{
            if (i >= n) {{
                break;
            }}
            let j = i + t;
            var row = 0.0;
            if (j < n) {{
                row = keys[j];
            }}
            workgroupBarrier();
            sv[t] = row;
        }}
    }}
    var dc = false;
    if (cfg.pass_index == 0u && wid.x < 4u) {{
        dc = true;
    }}
    if (dc) {{
        workgroupBarrier();
        sv[t] = sv[t] + 1.0;
    }}
    index_f32[wid.x * 256u + t] = sv[t];
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {{
    cands[gid.x] = atomicLoad(&index_u32[gid.x]);
}}
");
        let src = module("workgroupUniformLoad(&wu2)");
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        let sx = r.tree.to_sexpr();
        assert!(sx.contains("uflag.of_flag") && sx.contains("uniform.of_checked") && !sx.contains("_brk"), "{sx}");
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract.clone());
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        assert_eq!(same_values_every_entry(&src, &k, [2, 1, 1]), Ok(true), "the rendering computes the source's values\n{}", out.text);
        // The count from workgroup memory, no uniform load: read, and named.
        let plain = module("wu2");
        let r = read_safe(&plain, &k).unwrap_or_else(|e| panic!("{e}"));
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        let e = checker.check(&r.tree).expect_err("a barrier under a test of a workgroup load passes the checker");
        assert_eq!((e.layer.number(), e.reason), (4, "barrier_in_nonuniform_control_flow"), "{e}");
    }

    /// AN ARGMAX REDUCTION (2026-10-17, v24 run 1's seed04): the best score
    /// and its index in two workgroup arrays, reduced by a halving loop
    /// whose step moves the larger value and its index down. It reads as
    /// wg.reduce_argmax, renders, validates, reads back to the same tree and
    /// passes every layer; the same with `<` reads as wg.reduce_argmin.
    #[test]
    fn an_argmax_reduction_reads_as_its_row() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        for (cmp, row) in [(">", "wg.reduce_argmax_L0"), ("<", "wg.reduce_argmin_L0")] {
            let src = format!(
                "{V24_BINDINGS}var<workgroup> wbest: array<f32, 64>;
var<workgroup> widx: array<u32, 64>;
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    wbest[t] = keys[wid.x * 64u + t];
    widx[t] = t;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {{
        if (t < st) {{
            if (wbest[t + st] {cmp} wbest[t]) {{
                wbest[t] = wbest[t + st];
                widx[t] = widx[t + st];
            }}
        }}
        workgroupBarrier();
    }}
    if (t == 0u) {{ cands[wid.x] = widx[0]; }}
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {{
    _ = atomicLoad(&index_u32[gid.x & 1023u]);
}}
"
            );
            let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{cmp}: {e}"));
            assert!(r.tree.to_sexpr().contains(&format!("({row} ")), "{cmp}: {}", r.tree.to_sexpr());
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
            assert!(back.tree == r.tree, "{cmp}: the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
            let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract.clone());
            checker.check(&r.tree).unwrap_or_else(|e| panic!("{cmp}: {e}\n{}", out.text));
        }
    }

    /// ITEM LOOPS IN A UNIFORM TEST'S BODY (2026-10-17, random trees drawn
    /// with the uniform tests): the kingdom's renderings of an unrolled item
    /// loop under `if (nwg.y != 0u)` (seed 24 of that draw) and of items
    /// continued to a literal bound (`for (; t < 1u; ..)`, seed 11) read, the
    /// reading reading back to the same tree; they read as a store to the
    /// loop counter and as a counter of unknown entry before.
    #[test]
    fn item_loops_in_a_uniform_tests_body_read() {
        let k = WgslKingdom::load_safe();
        for (name, src) in [("unrolled", include_str!("../../tests/fixtures/uniform_test_unrolled_items.wgsl")), ("continued", include_str!("../../tests/fixtures/uniform_test_continued_items.wgsl"))] {
            let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{name}: {e}"));
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{name}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{name}: {e}"));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("{name}: read back: {e}"));
            assert!(back.tree == r.tree, "{name}: {:?}", crate::wgsl::render::first_difference(&r.tree, &back.tree));
        }
    }

    /// A SEARCH'S WORKGROUP STATE (2026-10-17, 3 of run 1's seeds): five f32
    /// score lists, a workgroup u32 scalar read as bits (`var<workgroup>
    /// wcode: u32;`), and a loop over the candidates a thread staged, bounded
    /// by a workgroup u32 (`for (var i = 0u; i < wn[0]; i++)`). The module
    /// reads, renders, validates, reads back to the same tree and passes
    /// every layer.
    #[test]
    fn a_searchs_workgroup_lists_and_a_staged_count_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let src = format!(
            "{V24_BINDINGS}var<workgroup> s1: array<f32, 32>;
var<workgroup> s2: array<f32, 32>;
var<workgroup> s3: array<f32, 32>;
var<workgroup> s4: array<f32, 32>;
var<workgroup> s5: array<f32, 32>;
var<workgroup> wcode: u32;
var<workgroup> wn: array<u32, 1>;
@compute @workgroup_size(32, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>) {{
    _ = atomicAdd(&index_u32[lid.x], 1u);
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    s1[lid.x] = 1.0; s2[lid.x] = 2.0; s3[lid.x] = 3.0; s4[lid.x] = 4.0; s5[lid.x] = 5.0;
    if (lid.x == 0u) {{ wcode = atomicLoad(&index_u32[wid.x]) & 255u; wn[0] = atomicLoad(&index_u32[wid.x + 1u]) & 31u; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        var best = 0u;
        for (var i = 0u; i < wn[0]; i++) {{
            if (s1[i] + s2[i] + s3[i] + s4[i] + s5[i] > 10.0) {{ best = i ^ (wcode >> 4u); }}
        }}
        cands[wid.x] = best;
    }}
}}
"
        );
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        let sx = r.tree.to_sexpr();
        assert!(sx.contains("wg.e") && sx.contains("(wg.get_u_safe wg.u"), "{sx}");
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
    }

    /// A BRANCH THAT RETURNS EARLY, AND A HELPER'S GUARDED RETURN AFTER ITS
    /// STATEMENTS (2026-10-17, 8 of run 1's seeds): `if (cfg.pass_index ==
    /// 0u) { ..barrier..; return; } ..barrier..` is `if (..) { .. } else { ..
    /// }`, a uniform test over its barriers; `sgn` hashes into a local,
    /// then `if ((h & 1u) == 0u) { return 1.0; } return -1.0;` is inlined as
    /// statements, the guard's return its branch, the rest the else. The
    /// module reads, renders, validates, reads back to the tree read
    /// (canonical), a fixpoint, and passes every layer.
    #[test]
    fn an_early_return_branch_and_a_guarded_helper_return_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\nparam cfg.arrived_begin 0 1000000\n").unwrap();
        let src = format!(
            "{V24_BINDINGS}var<workgroup> red: array<f32, 64>;
fn sgn(j: u32) -> f32 {{
    var h = j * 40503u;
    h = h ^ (h >> 15u);
    if ((h & 1u) == 0u) {{ return 1.0; }}
    return -1.0;
}}
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = cfg.arrived_begin + wid.x;
    if (cfg.pass_index == 0u) {{
        red[lid.x] = keys[(row & 1023u) * 64u + lid.x] * sgn(lid.x);
        workgroupBarrier();
        if (lid.x == 0u) {{ index_f32[row] = red[0]; }}
        return;
    }}
    workgroupBarrier();
    if (lid.x == 0u) {{ _ = atomicAdd(&index_u32[row & 1023u], 1u); }}
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x < 4096u) {{ cands[gid.x] = atomicLoad(&index_u32[gid.x & 1023u]); }}
}}
"
        );
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.tree.to_sexpr().contains("(stmt.if_else_ult_L0 "), "{}", r.tree.to_sexpr());
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        assert!(!out.text.contains("fn sgn"), "inlined:\n{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        let canon = crate::wgsl::canonical::canonical(&r.tree, &k, &r.role);
        assert!(canon == back.tree || back.tree == r.tree, "the rendered text reads back to the tree read: {:?}\n{}", crate::wgsl::render::first_difference(&canon, &back.tree), out.text);
        let again = read_safe(&crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &back.role).unwrap().text, &k).unwrap();
        assert!(again.tree == back.tree, "a fixpoint: {:?}", crate::wgsl::render::first_difference(&back.tree, &again.tree));
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
    }

    /// MORE OF RUN 1'S SEEDS (2026-10-17, logs/V24/run1/kingdom_gaps.tsv):
    /// a uniform predicate over a barrier (`cfg.stage == 1u && wid.x < 4u`,
    /// `else if (p == 1u || p == 3u)`); an item loop that starts at 1 (`for
    /// (var kk = 1u; kk < n; kk++)`, its bound n - 1, its counter 1 + t); a
    /// shift by a loop's counter (`b = b | (1u << i)`, an index read as a count);
    /// six u32 workgroup arrays; a copied `// entry` grid line that is not the
    /// entry point's workgroup (ignored). The module reads, renders, naga
    /// validates it, reads back to the same tree and passes every layer.
    #[test]
    fn run_ones_seed_shapes_read_and_read_back() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\nparam cfg.arrived_begin 0 1000000\nparam cfg.n_queries 0 4096\n").unwrap();
        let src = format!(
            "// entry ingest grid 16 16 1 1
{V24_BINDINGS}var<workgroup> red: array<f32, 64>;
var<workgroup> w1: array<u32, 8>;
var<workgroup> w2: array<u32, 8>;
var<workgroup> w3: array<u32, 8>;
var<workgroup> w4: array<u32, 8>;
var<workgroup> w5: array<u32, 8>;
var<workgroup> w6: array<u32, 8>;
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let p = cfg.pass_index;
    if (cfg.stage == 1u && wid.x < 4u) {{
        red[lid.x] = keys[wid.x * 64u + lid.x];
        workgroupBarrier();
        if (lid.x == 0u) {{ _ = atomicAdd(&index_u32[wid.x], u32(red[1])); }}
    }} else if (p == 1u || p == 3u) {{
        workgroupBarrier();
    }}
    if (lid.x < 8u) {{
        w1[lid.x] = 1u; w2[lid.x] = 2u; w3[lid.x] = 3u; w4[lid.x] = 4u; w5[lid.x] = 5u; w6[lid.x] = 6u;
    }}
    workgroupBarrier();
    let h = w1[1] + w2[2] + w3[3] + w4[4] + w5[5] + w6[6] + bitcast<u32>(red[0]);
    for (var kk = 1u; kk < cfg.n_queries; kk++) {{
        if (lid.x == 0u) {{
            var b = 0u;
            for (var i = 0u; i < 4u; i++) {{
                if (red[i] > 0.0) {{ b = b | (1u << i); }}
            }}
            _ = atomicOr(&index_u32[4096u + (kk & 1023u)], b & h);
        }}
    }}
}}
@compute @workgroup_size(32, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x < cfg.n_queries) {{ cands[gid.x] = atomicLoad(&index_u32[gid.x & 1023u]); }}
}}
"
        );
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        let sx = r.tree.to_sexpr();
        for want in ["(stmt.if_else_uflag_L0 (uflag.and ", "(stmt.if_uflag_L0 (uflag.or ", "convert.index_to_count", "wg.uf", "loop.items_boundx"] {
            assert!(sx.contains(want), "{want}: {sx}");
        }
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        assert!(out.text.contains("// entry ingest grid 64 1 1 1"), "the entry point's own grid:\n{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
    }

    /// ONE PASS'S WORK UNDER A UNIFORM TEST (2026-10-17, 13 of the 29 model
    /// seeds of v24's run 1): `if (cfg.pass_index == 0u) { ..barrier.. } else
    /// if (cfg.pass_index != 2u) { ..barrier.. } else { .. }` and `if
    /// (cfg.stage > 1u)`: tests of workgroup-uniform values whose bodies keep
    /// their rung, so a barrier in them is in uniform control (== 0u is < 1u).
    /// And a helper inlined inside a helper that passes it an argument of the
    /// same name (`norm(x)` calling `sq(x)`: two seeds never finished reading).
    /// The kernel reads, renders, naga validates it, reads back to the tree
    /// read (canonical), a fixpoint, and passes every layer; a test of a
    /// thread's value over a barrier reads (`uflag.of_flag`, since
    /// 2026-10-17) and the checker's layer 4 names its barrier.
    #[test]
    fn uniform_tests_keep_their_rung_and_nested_helpers_share_an_argument_name() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\nparam cfg.arrived_begin 0 1000000\n").unwrap();
        let src = format!(
            "{V24_BINDINGS}var<workgroup> red: array<f32, 64>;
fn sq(x: f32) -> f32 {{
    var y = x;
    y = y * x;
    return y;
}}
fn norm(x: f32) -> f32 {{
    var s = sq(x);
    s = s + 1.0;
    return s;
}}
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = cfg.arrived_begin + wid.x;
    if (cfg.pass_index == 0u) {{
        red[lid.x] = norm(keys[(row & 1023u) * 64u + lid.x]);
        workgroupBarrier();
        if (lid.x == 0u) {{ index_f32[row] = red[0]; }}
    }} else if (cfg.pass_index != 2u) {{
        workgroupBarrier();
        if (lid.x == 0u && cfg.stage > 1u) {{ _ = atomicAdd(&index_u32[row & 1023u], 1u); }}
    }} else {{
        if (lid.x == 0u) {{ _ = atomicMax(&index_u32[2048u + (row & 1023u)], row); }}
    }}
}}
"
        );
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        let sx = r.tree.to_sexpr();
        assert!(sx.contains("(stmt.if_else_ult_L0 ") && sx.contains("(stmt.if_else_une_L0 "), "{sx}");
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        // (The inlined helpers' statements are one block statement as read,
        // written in place: the same tree in its canonical spelling, then a fixpoint.)
        let canon = crate::wgsl::canonical::canonical(&r.tree, &k, &r.role);
        assert!(canon == back.tree, "the rendered text reads back to the tree read, canonical: {:?}\n{}", crate::wgsl::render::first_difference(&canon, &back.tree), out.text);
        let again = read_safe(&crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &back.role).unwrap().text, &k).unwrap();
        assert!(again.tree == back.tree, "a fixpoint: {:?}", crate::wgsl::render::first_difference(&back.tree, &again.tree));
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let thread = src.replace("if (cfg.pass_index == 0u) {", "if (lid.x == 0u) {");
        let t = read_safe(&thread, &k).unwrap_or_else(|e| panic!("a barrier under a thread's test: {e}"));
        let checker = crate::wgsl::check::Checker::with_contract(&k, &t.role, &crate::wgsl::target::Target::recorded(), crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap());
        let e = checker.check(&t.tree).expect_err("a barrier under a thread's test passes the checker");
        assert_eq!((e.layer.number(), e.reason), (4, "barrier_in_nonuniform_control_flow"), "{e}");
    }

    /// A LAST STORE THE EPILOGUE CANNOT SPELL, UNDER AN INTERFACE WITH NO h
    /// (2026-10-17; five saved model kernels, nluca's shape): the kernel's
    /// last store inside a loop bounded by a binding, and inside a loop that
    /// breaks. With no output cell there is no epilogue to end in: the store
    /// stays in the body, which ends in a return. Each reads, renders, naga
    /// validates it, reads back to the same tree and passes every layer.
    #[test]
    fn a_last_store_inside_a_loop_reads_under_an_interface_without_h() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let pre = "@group(0) @binding(0) var<storage, read> n: array<u32>;\n@group(0) @binding(1) var<storage, read> x: array<f32>;\n@group(0) @binding(2) var<storage, read_write> y: array<f32>;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
        for (what, body) in [
            ("item loop", "    let rows = min(n[0u], 4u);\n    for (var j = 0u; j < rows; j += 1u) {\n        let r = gid.x * 4u + j;\n        if (r < n[1u]) { y[r] = x[r] * 2.0; }\n    }\n}\n"),
            ("a loop that breaks", "    for (var j = 0u; j < 4u; j += 1u) {\n        let r = gid.x * 4u + j;\n        if (r >= n[0u]) { break; }\n        y[r] = x[r] + 1.0;\n    }\n}\n"),
        ] {
            let src = format!("{pre}{body}");
            let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{what}: {e}"));
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("{what}: read back: {e}\n{}", out.text));
            assert!(back.tree == r.tree, "{what}: the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
            let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract.clone());
            checker.check(&r.tree).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
        }
    }

    /// THE LENGTH OF ANY RUNTIME-SIZED BINDING (2026-10-19; cuneus's
    /// `instances` and `skin` kernels read `arrayLength(&src)`): the interface
    /// has the row `len.<binding>` for each, so none of them is the Gram's
    /// phi or h. Reads, renders, naga validates it, reads back to the same
    /// tree and passes every layer.
    #[test]
    fn the_length_of_any_runtime_sized_binding_reads() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let src = "@group(0) @binding(0) var<storage, read> src: array<f32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<f32>;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    if (gid.x >= arrayLength(&src) || gid.x >= arrayLength(&dst)) { return; }\n    dst[gid.x] = src[gid.x] * 2.0;\n}\n";
        let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.tree.to_sexpr().contains("len.src") && r.tree.to_sexpr().contains("len.dst"), "{}", r.tree.to_sexpr());
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        assert!(out.text.contains("arrayLength(&src)") && out.text.contains("arrayLength(&dst)"), "{}", out.text);
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "{:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
    }

    /// UNIFORM FIELDS OF ANY STRUCT, AND STRUCT-ELEMENT BINDINGS (2026-10-19;
    /// cuneus's `meshlab_swarm`, `instances` and `skin`): two uniform blocks
    /// of fields of any type (a `vec3<f32>` lane, an array element), a
    /// struct-element storage array written a field at a time and as a whole
    /// from a constructor or from another element, and read through a `let`
    /// of the element. Reads, renders, naga validates it, reads back to the
    /// same tree and passes every layer.
    #[test]
    fn uniform_blocks_and_struct_element_bindings_of_any_shape_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let src = "struct Time { t: f32, n: u32 };
@group(0) @binding(0) var<uniform> u_time: Time;
struct Swarm { count: u32, spread: f32, center: vec3<f32>, planes: array<vec4<f32>, 6> };
@group(1) @binding(1) var<uniform> swarm: Swarm;
struct Item { m: mat4x4<f32>, d: vec4<f32>, id: u32 };
@group(2) @binding(0) var<storage, read> src: array<Item>;
@group(3) @binding(0) var<storage, read_write> dst: array<Item>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= swarm.count || i >= arrayLength(&src) { return; }
    let s = src[i];
    let e = Item(s.m, s.d * u_time.t + vec4<f32>(swarm.center, swarm.planes[2u].w), s.id + u_time.n);
    if i % 2u == 0u { dst[i] = e; } else { dst[i] = src[i]; }
    dst[i].d = vec4<f32>(swarm.spread * swarm.center.y);
}
";
        let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}"));
        let sx = r.tree.to_sexpr();
        for want in ["load.swarm_count", "load.swarm_center", "load.swarm_planes", "load.u_time_t", "len.src", "load.src.m", "store.dst.d_", "store.dst.id_"] {
            assert!(sx.contains(want), "{want} missing from {sx}");
        }
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "{:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
    }

    /// PLACES OF ANY DEPTH (2026-10-19; the harvested corpus's litmus tests
    /// and dawn's buffer tests): a binding that is one vector, written whole; a
    /// struct binding with an array member, an array of structs inside a
    /// struct, and atomic leaves at both; a uniform matrix; a pipeline
    /// constant given its value by a `// override` header. Each reads,
    /// renders, naga validates it, reads back to the same tree and passes
    /// every layer.
    #[test]
    fn places_of_any_depth_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let sources = [
            ("a vector binding written whole", "@group(0) @binding(0) var<uniform> u: mat2x2<f32>;\n@group(0) @binding(1) var<storage, read_write> out: vec4<f32>;\n@compute @workgroup_size(1)\nfn main() {\n    out = vec4<f32>(u[1], u[0].x, u[1][1]);\n}\n"),
            ("a struct with an array member and atomics", "struct Cell { r0: u32, r1: atomic<u32> };\nstruct Results { seq: atomic<u32>, value: array<Cell, 8> };\nstruct Locs { value: array<atomic<u32>> };\n@group(0) @binding(0) var<storage, read_write> results: Results;\n@group(0) @binding(1) var<storage, read_write> locs: Locs;\n@compute @workgroup_size(32)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    if (gid.x < 8u) {\n        let old = atomicAdd(&locs.value[gid.x], 2u);\n        results.value[gid.x].r0 = old + 1u;\n        atomicStore(&results.value[gid.x].r1, old);\n    }\n    atomicMax(&results.seq, gid.x);\n}\n"),
            ("a pipeline constant", "// override size 64\noverride size: u32;\n@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    if (gid.x < size) { out[gid.x] = size; }\n}\n"),
        ];
        for (what, src) in sources {
            let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{what}: {e}"));
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("{what}: read back: {e}\n{}", out.text));
            assert!(back.tree == r.tree, "{what}: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
            // (The first kernel is one thread writing one cell: dispatched once.)
            let c = if what.starts_with("a vector") { crate::wgsl::safety::Contract::parse("workgroups x 1 1\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap() } else { contract.clone() };
            let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), c);
            checker.check(&r.tree).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
        }
    }

    /// TEXTURES AND SAMPLERS (2026-10-19; 3,802 of the harvested corpus's
    /// failures and every cuneus image kernel): sampled, storage, arrayed and
    /// depth textures, loads with i32 or u32 coordinates, the queries, a
    /// sample at a level, a gradient and a comparison, and a store. Each reads,
    /// renders, naga validates it, reads back to the same tree and passes
    /// every layer.
    #[test]
    fn textures_and_samplers_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1024\nworkgroups z 1 1\n").unwrap();
        let sources = [
            ("sample, load and store", "@group(0) @binding(0) var tex: texture_2d<f32>;\n@group(0) @binding(1) var smp: sampler;\n@group(0) @binding(2) var out: texture_storage_2d<rgba8unorm, write>;\n@compute @workgroup_size(8, 8)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let d = textureDimensions(tex);\n    if (gid.x >= d.x || gid.y >= d.y) { return; }\n    let a = textureLoad(tex, vec2<i32>(gid.xy), 0);\n    let b = textureSampleLevel(tex, smp, (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(d), 0.0);\n    textureStore(out, vec2<i32>(gid.xy), a + b);\n}\n"),
            ("arrayed storage, u32 coordinates", "@group(0) @binding(0) var src: texture_storage_2d_array<rgba32uint, read>;\n@group(0) @binding(1) var dst: texture_storage_2d_array<rgba32uint, write>;\n@compute @workgroup_size(8, 8, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let n = textureNumLayers(src);\n    if (gid.z < n) { textureStore(dst, gid.xy, gid.z, textureLoad(src, gid.xy, gid.z) + vec4<u32>(1u)); }\n}\n"),
            ("depth, gradient and comparison", "@group(0) @binding(0) var shadow: texture_depth_2d;\n@group(0) @binding(1) var cmp: sampler_comparison;\n@group(0) @binding(2) var color: texture_2d<f32>;\n@group(0) @binding(3) var lin: sampler;\n@group(0) @binding(4) var<storage, read_write> out: array<f32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let uv = vec2<f32>(f32(gid.x) / 64.0, 0.5);\n    let s = textureSampleCompareLevel(shadow, cmp, uv, 0.5);\n    let g = textureSampleGrad(color, lin, uv, vec2<f32>(0.01, 0.0), vec2<f32>(0.0, 0.01));\n    out[gid.x] = s + g.x + textureSampleLevel(shadow, lin, uv, 0);\n}\n"),
        ];
        for (what, src) in sources {
            let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{what}: {e}"));
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("{what}: read back: {e}\n{}", out.text));
            assert!(back.tree == r.tree, "{what}: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
            // (The third writes a row of cells: one workgroup row.)
            let c = if what.starts_with("depth") { crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap() } else { contract.clone() };
            let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), c);
            checker.check(&r.tree).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
        }
    }

    /// A MODULE OF ANY NUMBER OF ENTRY POINTS (2026-10-19; cuneus's
    /// `blackhole` has eight): a node holds four children at most, so five or
    /// more entry points are a `kernel.module_ext<k>` chain over a module of
    /// four. Six entry points read, render, read back to the same tree and
    /// pass every layer under a contract of one workgroup each.
    #[test]
    fn a_module_of_six_entry_points_reads_as_a_chain() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let mut src = String::from("@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n");
        for e in 0..6 {
            src.push_str(&format!("@compute @workgroup_size(32)\nfn e{e}(@builtin(local_invocation_id) lid: vec3<u32>) {{\n    out[{} + lid.x] = f32(lid.x + {e}u);\n}}\n", e * 32));
        }
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.tree.name, "kernel.module_ext2", "{}", r.tree.to_sexpr().chars().take(200).collect::<String>());
        assert_eq!(crate::wgsl::render::module_entries(&r.tree).len(), 6);
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        for e in 0..6 {
            assert!(out.text.contains(&format!("fn e{e}(")), "{}", out.text);
        }
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        assert!(back.tree == r.tree, "{:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        // A node of four children at most: the chain is a genome.
        crate::wgsl::safe_karva::encode(&r.tree, 32, 128, 7).unwrap_or_else(|e| panic!("encode: {e}"));
    }

    /// AN i32 LOOP COUNTER AND A LOOP BOUND FROM A STRUCT'S U32 LANE
    /// (2026-10-19; cuneus's `instances` and `skin`): `for (var j = 0; j < 6;
    /// j++)` over an array of a uniform, and `for (k < s.morph.y)` over a
    /// vec4<u32> member of a storage struct, the second field of a non-first
    /// uniform block read by a non-role name. Each reads, renders, naga
    /// validates it, reads back to the same tree and passes every layer.
    #[test]
    fn an_i32_counter_and_a_u32_lane_bound_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        let src = "struct Params { count: u32, planes: array<vec4<f32>, 6> };\nstruct Vert { morph: vec4<u32>, w: vec4<f32> };\n@group(0) @binding(0) var<uniform> params: Params;\n@group(0) @binding(1) var<storage, read> verts: array<Vert>;\n@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    if gid.x >= params.count { return; }\n    let v = verts[gid.x];\n    var acc = 0.0;\n    for (var j = 0; j < 6; j++) { acc += params.planes[j].w * f32(j); }\n    for (var k = 0u; k < v.morph.y; k++) { acc += v.w.x; }\n    out[gid.x] = acc;\n}\n";
        let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}"));
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        // (The index of an i32 counter reads back as the other spelling of the same
        // conversion, u32(i) for index.of_int: the text then reads to a fixpoint.)
        let again = read_safe(&crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &back.role).unwrap().text, &k).unwrap();
        assert!(again.tree == back.tree, "a fixpoint: {:?}", crate::wgsl::render::first_difference(&back.tree, &again.tree));
        let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract);
        checker.check(&r.tree).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
    }

    /// MORE ARRAYS OF A KIND THAN THE KINGDOM GREW WITH (2026-10-19, the slot
    /// counts): six atomic<u32> workgroup arrays (the first four are named
    /// rows, the others generated), four i32 ones, three 128-element u32
    /// private arrays, and five registers of a vec3. Each reads, renders,
    /// naga validates it and reads back to the same tree.
    #[test]
    fn a_kernel_with_more_arrays_than_the_first_names_reads() {
        let k = WgslKingdom::load_safe();
        let mut src = String::from("@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n");
        for i in 0..6 {
            src.push_str(&format!("var<workgroup> au{i}: array<atomic<u32>, 64>;\n"));
        }
        for i in 0..4 {
            src.push_str(&format!("var<workgroup> ni{i}: array<i32, 64>;\n"));
        }
        src.push_str("@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {\n");
        for i in 0..3 {
            src.push_str(&format!("    var c{i}: array<u32, 100>;\n"));
        }
        src.push_str("    var acc = 0u;\n    for (var j = 0u; j < 100u; j++) { c0[j] = j; c1[j] = j * 2u; c2[j] = j * 3u; }\n");
        for i in 0..6 {
            src.push_str(&format!("    atomicStore(&au{i}[lid.x], lid.x + {i}u);\n"));
        }
        for i in 0..4 {
            src.push_str(&format!("    ni{i}[lid.x] = i32(lid.x) - {i};\n"));
        }
        src.push_str("    workgroupBarrier();\n");
        for i in 0..6 {
            src.push_str(&format!("    acc += atomicLoad(&au{i}[(lid.x + 1u) & 63u]);\n"));
        }
        for i in 0..4 {
            src.push_str(&format!("    acc += u32(ni{i}[(lid.x + 1u) & 63u]);\n"));
        }
        src.push_str("    acc += c0[lid.x] + c1[lid.x] + c2[lid.x];\n");
        for i in 0..5 {
            src.push_str(&format!("    var v{i} = vec3<f32>(f32(lid.x), {i}.0, 1.0); acc += u32(dot(v{i}, vec3<f32>(1.0)));\n"));
        }
        src.push_str("    out[gid.x] = acc;\n}\n");
        let r = read_safe(&src, &k).unwrap_or_else(|e| panic!("{e}"));
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        for want in ["wg_au_4", "wg_au_5", "wg_n_2", "wg_n_3", "pux2"] {
            assert!(out.text.contains(want), "{want} missing\n{}", out.text);
        }
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        let again = read_safe(&crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &back.role).unwrap().text, &k).unwrap();
        assert!(again.tree == back.tree, "a fixpoint: {:?}", crate::wgsl::render::first_difference(&back.tree, &again.tree));
    }

    /// TWO LOOPS OF THE V24 REFERENCE (2026-10-17, the harness agent's
    /// `problems/v24_ingest_small/reference.wgsl` worked around both): a
    /// thread-strided loop stepped by a module `const` (`r = r + THREADS`),
    /// and an insertion step counting down with two breaks, `loop { if (j ==
    /// 0u) { break; } if (best[j - 1u] >= v) { break; } ... j = j - 1u; }`
    /// (a loop with no counter, its stop the two guards in order). Each reads,
    /// renders, naga validates it, reads back to the same tree and passes
    /// every layer.
    #[test]
    fn a_const_stepped_loop_and_a_downward_insertion_loop_read() {
        let k = WgslKingdom::load_safe();
        let contract = crate::wgsl::safety::Contract::parse("workgroups x 1 1024\nworkgroups y 1 1\nworkgroups z 1 1\n").unwrap();
        for (what, src) in [
            ("const step", "@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
const THREADS = 16u;
var<workgroup> tile: array<f32, 256>;
@compute @workgroup_size(16, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    for (var r = lid.x; r < 256u; r = r + THREADS) {
        tile[r] = keys[wid.x * 256u + r];
    }
    workgroupBarrier();
    out[wid.x * 16u + lid.x] = tile[lid.x * 16u];
}
"),
            ("insertion", "@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var best: array<f32, 8>;
    let v = keys[gid.x];
    var j = 7u;
    loop {
        if (j == 0u) { break; }
        if (best[j - 1u] >= v) { break; }
        best[j] = best[j - 1u];
        j = j - 1u;
    }
    best[j] = v;
    out[gid.x] = best[0];
}
"),
        ] {
            let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{what}: {e}"));
            let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate_text(&out.text).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
            let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("{what}: read back: {e}\n{}", out.text));
            assert!(back.tree == r.tree, "{what}: the rendered text reads back to the same tree: {:?}\n{}", crate::wgsl::render::first_difference(&r.tree, &back.tree), out.text);
            let checker = crate::wgsl::check::Checker::with_contract(&k, &r.role, &crate::wgsl::target::Target::recorded(), contract.clone());
            checker.check(&r.tree).unwrap_or_else(|e| panic!("{what}: {e}\n{}", out.text));
            if what == "insertion" {
                assert!(out.text.contains(") | ("), "{what}: one stop of both guards, in order (the eager `|`):\n{}", out.text);
            }
        }
    }

    /// HELPERS INLINED AS STATEMENTS, CALLED UNDER A CONDITION AND FROM EACH
    /// OTHER (B6, 2026-10-17; 78 and 13 of the 1,798 saved model kernels):
    /// `jitter` (locals, two calls of `fmix`, a return) called under an `if`,
    /// `fmix` (a local, a return) called from it, and the pure `capped` (a
    /// guarded return, a select) over jitter's own argument. Their returns at their top
    /// level end them on every path; each is inlined with its locals under
    /// its own names; the kernel renders, naga validates it, it reads back to
    /// the same tree and passes every layer. A guarded return after the
    /// helper's statements reads (its branch, the rest the else); one inside
    /// a loop of the helper is named.
    #[test]
    fn helpers_called_under_a_condition_and_from_a_helper_are_inlined() {
        let k = WgslKingdom::load_safe();
        let src = "// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
fn fmix(h0: u32) -> u32 {
    var x = h0;
    x = x ^ (x >> 16u);
    x = x * 40503u;
    x = x ^ (x >> 13u);
    return x;
}
fn capped(x: f32) -> f32 {
    if (x > 0.5) {
        return 0.5;
    }
    return x;
}
fn jitter(i: u32, j: u32) -> f32 {
    var s = fmix(i * 31u + j);
    s = s ^ fmix(j);
    return capped(f32(s & 255u) * 0.0001 + f32(j) * 0.001);
}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    var acc = 0.0;
    for (var t = 0u; t < cfg.m; t = t + 1u) {
        acc = acc + phi[t * cfg.d + cfg.r_off + row] * phi[t * cfg.d + col];
    }
    if (acc > 0.0) {
        acc = acc + jitter(row, col);
    }
    if (row < cfg.r_rows && col < cfg.d && cfg.r_off + row < cfg.d) {
        h[row * cfg.d + col] = h[row * cfg.d + col] + acc;
    }
}
";
        let r = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}"));
        // Three helper results, each its own register (the two calls of fmix and jitter's).
        let calls: BTreeSet<&String> = r.registers.keys().filter(|l| l.starts_with("local.call.")).collect();
        assert_eq!(calls.len(), 3, "{:?}", r.registers);
        assert!(calls.iter().filter(|l| l.starts_with("local.call.jitter/fmix@")).count() == 2, "{calls:?}");
        let out = crate::wgsl::safe_karva::render_tree_with(&r.tree, &k, &r.role).unwrap_or_else(|e| panic!("{e}"));
        assert!(!out.text.contains("fn fmix") && !out.text.contains("fn jitter") && !out.text.contains("fn capped"), "inlined:\n{}", out.text);
        validate_text(&out.text).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        let back = read_safe(&out.text, &k).unwrap_or_else(|e| panic!("read back: {e}\n{}", out.text));
        // The same tree in its canonical spelling: an inlined helper's
        // statements are one block statement as read, written in place, and
        // read back as the statements of the block around them (rule 1 of
        // wgsl::canonical, a block's grouping); read again, the same tree.
        let canon = crate::wgsl::canonical::canonical(&r.tree, &k, &r.role);
        assert!(canon == back.tree, "the rendered text reads back to the tree read, canonical: {:?}\n{}", crate::wgsl::render::first_difference(&canon, &back.tree), out.text);
        let again = read_safe(&crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &back.role).unwrap().text, &k).unwrap();
        assert!(again.tree == back.tree, "a fixpoint: {:?}", crate::wgsl::render::first_difference(&back.tree, &again.tree));
        crate::wgsl::check::check(&r.tree, &k, &r.role, &crate::wgsl::target::Target::recorded()).unwrap_or_else(|e| panic!("{e}\n{}", out.text));
        // A guarded return after the helper's statements is its branch, the
        // rest of the helper the else (2026-10-17): it reads and validates.
        let inner = src.replace("    x = x ^ (x >> 13u);\n    return x;", "    if (x > 7u) { return x; }\n    return x + 1u;");
        assert_ne!(inner, src);
        let r2 = read_safe(&inner, &k).unwrap_or_else(|e| panic!("a guarded helper return: {e}"));
        validate_text(&crate::wgsl::safe_karva::render_tree_with(&r2.tree, &k, &r2.role).unwrap().text).unwrap();
        // A return inside a loop of the helper reads with the return lowered to a
        // flag and a break (`wgsl::lower`, 2026-10-19), and the kernel validates.
        let looped = src.replace("    x = x ^ (x >> 13u);\n    return x;", "    for (var i = 0u; i < 4u; i++) { if (x > i) { return x; } }\n    return x + 1u;");
        assert_ne!(looped, src);
        let r3 = read_safe(&looped, &k).unwrap_or_else(|e| panic!("a return inside a helper's loop: {e}"));
        validate_text(&crate::wgsl::safe_karva::render_tree_with(&r3.tree, &k, &r3.role).unwrap().text).unwrap();
    }

    /// What the kingdom's missing compare-exchange rested on, measured on the
    /// naga it builds with. naga 0.20's Metal backend wrote an atomicAdd on a
    /// storage array and refused atomicCompareExchangeWeak; naga 30's writes
    /// both (2026-10-19, the naga/wgpu 30 upgrade). The kingdom still has no
    /// compare-exchange row, so the reader still refuses one by name: our
    /// device policy now, not a backend gap (the refusal's words say so).
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    #[test]
    fn nagas_metal_backend_writes_storage_atomics_and_not_compare_exchange() {
        let msl = |text: &str| -> Result<String, String> {
            let module = crate::wgsl::naga_shape::parse(text)?;
            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| format!("{e:?}"))?;
            let options = naga::back::msl::Options::default();
            let pipeline = naga::back::msl::PipelineOptions::default();
            naga::back::msl::write_string(&module, &info, &options, &pipeline).map(|(s, _)| s).map_err(|e| format!("{e:?}"))
        };
        let kernel = |body: &str| format!("@group(0) @binding(0) var<storage, read_write> a: array<atomic<u32>>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {body} }}\n");
        let add = msl(&kernel("_ = atomicAdd(&a[min(gid.x, arrayLength(&a) - 1u)], 1u);")).unwrap_or_else(|e| panic!("{e}"));
        assert!(add.contains("atomic_fetch_add_explicit"), "{add}");
        let cas = msl(&kernel("_ = atomicCompareExchangeWeak(&a[gid.x], 0u, 1u);")).unwrap_or_else(|e| panic!("naga 30 writes a compare-exchange to Metal: {e}"));
        assert!(cas.contains("atomic_compare_exchange_weak_explicit"), "{cas}");
        let k = WgslKingdom::load_safe();
        let src = "// grid 64 1 1 1\nstruct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };\n@group(0) @binding(0) var<storage, read> phi: array<f32>;\n@group(0) @binding(1) var<storage, read_write> h: array<f32>;\n@group(0) @binding(2) var<uniform> cfg: Cfg;\nvar<workgroup> wg_au_a: array<atomic<u32>, 1024>;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {\n    let o = atomicCompareExchangeWeak(&wg_au_a[lidx], 0u, 1u).old_value;\n    if (gid.x < cfg.r_rows) { h[gid.x * cfg.d] = h[gid.x * cfg.d] + f32(o); }\n}\n";
        let Err(e) = read_safe(src, &k) else { panic!("a compare-exchange read") };
        assert!(e.contains("atomicCompareExchangeWeak has no safe spelling"), "{e}");
    }

    /// A LET STALE AT ITS USE (2026-10-12, found reading the cosine
    /// re-score): Kahan's `let s = sum + y; c = (s - sum) - y; sum = s;`.
    /// Inlined at `sum = s`, `s` read the compensation after its store; the
    /// let is set in a register at its definition, so the sum is a copy of it.
    #[test]
    fn a_let_a_store_changes_before_its_use_is_set_at_its_definition() {
        let k = WgslKingdom::load_safe();
        let (t, role) = (FunctionTable::safe(), gram_role());
        let src = include_str!("../../tests/corpus/safewgsl/draft_kahan.wgsl");
        let tree = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let text = Renderer::new(&k, &t, &role).kernel(&tree).unwrap().text;
        let copy = |l: &str| -> bool {
            let reg = |x: &str| x.strip_prefix('f').is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
            l.trim().strip_suffix(';').and_then(|l| l.split_once(" = ")).is_some_and(|(a, b)| reg(a) && reg(b))
        };
        assert!(text.lines().any(copy), "the sum is not a copy of the let's register:\n{text}");
        validate_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
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
        // (An expression bound's value is u32.xbound since 2026-10-12: cfg.m as its uniform value.)
        assert!(tree.to_sexpr().contains("(subgroup.max (index.of_uniform uniform.cfg_m))"), "{}", tree.to_sexpr());
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

    /// A BARRIER IN A UNIFORM-BOUNDED LOOP READS (legality item 5's own
    /// rule, 2026-10-12): reserve2's g003 r12 holds barriers inside
    /// `for (k = 0u; k < (min(1024u, (cfg.m - t))); k = k + 32u)`, bounded by
    /// cfg.m, the item counter and literals; the loop reads with a uniform
    /// bound, not as conditional statements.
    #[test]
    fn a_barrier_in_a_uniform_bounded_loop_reads() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/reserve2_g003_r12_uniform_bound.wgsl");
        assert!(src.contains("k < (min(1024u, (cfg.m - t)))"));
        let sx = read_safe(src, &k).unwrap_or_else(|e| panic!("{e}")).tree.to_sexpr();
        assert!(sx.contains("(bound.of_uniform (uniform.min") && sx.contains("uniform.sub"), "{sx}");
        assert!(sx.contains("stmt.barrier_"), "{sx}");
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
    /// barrier after a continue reads, and the checker refuses it when the
    /// continue's condition is a thread's own (a barrier some threads skip).
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
        // A barrier after a continue reads since 2026-10-12 (the continue's
        // block, then the barrier at the loop's rung: the same text); the
        // checker decides it. Under a uniform condition (the item counter)
        // every thread skips the barrier together: legal. Under a thread's
        // own value some threads reach it: refused by the uniformity layer.
        let checker = crate::wgsl::check::Checker::new(&k, &role, &crate::wgsl::target::Target::recorded());
        let uniform = read_safe(&barrier, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        checker.check(&uniform).unwrap_or_else(|e| panic!("a uniform continue before a barrier: {e}"));
        let divergent = barrier.replacen("if (t == 7u) { continue; }", "if (gid.x == 7u) { continue; }", 1);
        let tree = read_safe(&divergent, &k).unwrap_or_else(|e| panic!("{e}")).tree;
        let e = checker.check(&tree).unwrap_err();
        assert_eq!((e.layer, e.reason), (crate::wgsl::check::Layer::Uniformity, "barrier_in_nonuniform_control_flow"), "{e}");
    }

    /// Whether a module's text negates a matrix with a unary minus anywhere
    /// (a helper or the entry point): naga 0.20 writes it to Metal as a unary
    /// minus on a `metal::floatCxR`, which Metal does not define.
    fn negates_a_matrix(text: &str) -> bool {
        let m = crate::wgsl::naga_shape::parse(text).unwrap_or_else(|e| panic!("{e}"));
        let info = crate::wgsl::target::Target::recorded().validate(&m, text).unwrap_or_else(|e| panic!("{e}"));
        let hits = |f: &naga::Function, fi: &naga::valid::FunctionInfo| {
            f.expressions.iter().any(|(_, e)| matches!(e, naga::Expression::Unary { op: naga::UnaryOperator::Negate, expr } if matches!(fi[*expr].ty.inner_with(&m.types), naga::TypeInner::Matrix { .. })))
        };
        m.functions.iter().any(|(h, f)| hits(f, &info[h])) || m.entry_points.iter().enumerate().any(|(i, ep)| hits(&ep.function, info.get_entry_point(i)))
    }

    /// A MATRIX NEGATED (2026-10-12, fit_SF_random: six random kernels failed
    /// Metal compilation, "invalid argument type 'metal::float4x4' to unary
    /// expression"): the renderer writes it as the matrix times -1.0, the
    /// same value, which reads back as that product; no later pass negates one.
    /// (2026-10-19, naga 30: a unary minus on a matrix is not WGSL by the spec,
    /// and naga 30 refuses it, so the draft that wrote `-m2` no longer reads,
    /// its manifest entry a bug; the drawn trees that negate a matrix are the
    /// ones rendered here.)
    #[test]
    fn a_negated_matrix_renders_as_a_product_metal_compiles() {
        let k = WgslKingdom::load_safe();
        let src = include_str!("../../tests/corpus/safewgsl/draft_matrices.wgsl");
        assert!(negates_a_matrix_text(src), "the draft negates a matrix");
        let Err(e) = read_safe(src, &k) else { panic!("naga 30 refuses a unary minus on a matrix") };
        assert!(e.contains("InvalidUnaryOperandType(Negate"), "{e}");
        let role = gram_role();
        let root = dual_ty(&k, "kernel.safe").unwrap();
        let mut seen = 0usize;
        for seed in 0..2000u64 {
            let tree = crate::wgsl::render::sample(&k, &role, root, 7, seed).unwrap();
            if !tree.to_sexpr().contains("(arith.neg_mat") {
                continue;
            }
            seen += 1;
            let text = crate::wgsl::safe_karva::render_tree_with(&tree, &k, &crate::wgsl::render::gram_role()).unwrap_or_else(|e| panic!("{e}")).text;
            assert!(!negates_a_matrix(&text), "{text}");
            assert!(text.contains(" * -1.0)"), "{text}");
            let again = crate::wgsl::safe_karva::render_tree_with(&read_safe(&text, &k).unwrap_or_else(|e| panic!("{e}")).tree, &k, &role).unwrap_or_else(|e| panic!("{e}")).text;
            assert!(!negates_a_matrix(&again), "{again}");
            if seen == 5 {
                break;
            }
        }
        assert!(seen >= 1, "no drawn tree negates a matrix");
    }

    /// Whether `text` writes a unary minus on a matrix (by its spelling: naga 30
    /// refuses to parse such text into a valid module, so the IR test cannot).
    fn negates_a_matrix_text(text: &str) -> bool {
        text.contains("(-m2)")
    }

    /// The counts of [`random_sampler_trees_read_back_to_their_own_text`]
    /// reached so far: to a fixpoint after the first read, and to the AST
    /// drawn (canonical). Every tree reads back (1,000 of 1,000, asserted
    /// exactly). Their own text is printed, not a bar (Andrew, 2026-10-13).
    /// 930 since 2026-10-17 12:35Z (the rows drawn then, AST_FLOOR's note);
    /// 928 since 15:30Z (the 64-element vec4 arrays, AST_FLOOR's note);
    /// 917 since 16:05Z (the argmax reductions, AST_FLOOR's note); 959
    /// since 19:08Z (an atomic read inside a guarded store's value).
    const FIXPOINT_FLOOR: usize = 959;
    // (Both reset 2026-10-13 when the sampler drew its stores from an input:
    // other trees, 253 and 930 on the earlier draws; again when conditions
    // read a variable: 110 then 79; 59 when they vary by thread or data;
    // 2026-10-14, 54 (fixpoint 922) when the kingdom drew only genes layer 7
    // proves: the workgroup put, one cell kind a kernel, literal offsets,
    // reads of h in a store's value; a put through a tile counter outside
    // its loop reads back as the put of element 0, its value; 2026-10-16,
    // 42 (fixpoint 914) when the kingdom drew u32 workgroup arrays: the
    // first differences fall in the classes of the earlier draws, the u32
    // put's own the tile counter outside its loop that every put has;
    // 2026-10-17, 35 (fixpoint 918) when the kingdom drew the stepped loops
    // that break (loop.tilex_strided_brk, read from v24 seeds; undrawable at
    // first, 14 of 1,000 drawn trees did not read back: 12 kept the old
    // depth's break block where an epilogue moved them, fixed in rerung; 2
    // were workgroup puts through a tile counter of a loop bounded near 2^24,
    // whose lidx + k * threads wrapped, fixed by min(k, ceil(len / threads))):
    // the first differences fall in the earlier classes, a barrier or a loop
    // grouped against a register set, a literal bound against its uniform
    // spelling, which tilex_brk and tilex loops have too.)
    // (Re-measured 2026-10-17 11:44Z, 31, when the kingdom drew its uniform
    // loads wg.uload_i/f: other draws (40 the commit before, with them
    // undrawn); 4 of the 969 others differ first at a uniform load, the
    // subgroup operation set into a register at its statement, the earlier
    // class; fixpoint 930.) (12:35Z, 46 and 930, when the kingdom drew every
    // row but the uniform tests and the reader's two checked rows: the wider
    // workgroup arrays, the 64-element u32 arrays, an index as a count, a
    // workgroup u32 as a loop bound, the uniform loads; 1,000 of 1,000 read
    // back.) (15:30Z, 37 and 928, when the kingdom gained the 64-element vec4
    // private arrays: other draws, a table of other rows; with those arrays
    // undrawn the same commit measures 34 and 934, so the drop is the draws,
    // not the rows; 1,000 of 1,000 read back.) (16:05Z, 43 and 917 with the
    // argmax and argmin reductions drawn: other draws; of the 83 that miss
    // the fixpoint none differs at a reduction, the earlier classes all.)
    // (17:40Z, 85: the emitter's sg_lane let read as the builtin, not a
    // register set at the kernel's top, the largest class.)
    // (19:54Z, 106: a thread's row or column in an index read as the row
    // it is, safe.row_as_index / col_as_index, the largest class.)
    // (21:22Z, 111: an index bound that is a uniform value read as it,
    // canonical rule 8; the class went from 32 to 2.)
    // (21:30Z, 121: a literal offset in a thread's block snapped below the
    // block factor, as the emitter writes it.)
    // (22:11Z, 128: `min(a, max(n, 1u) - 1u)` read as index.clamp_to.)
    const AST_FLOOR: usize = 128;

    /// The floors of [`random_module_trees_render_validate_and_read_back`]
    /// (1,000 v24 modules, depth 6; measured 2026-10-17 at its first run:
    /// 4 parse back to the tree drawn, canonical, 955 to a fixpoint, 151 pass
    /// every layer under a 1-D contract; the canonical spelling is the Gram
    /// role's, so few module trees meet it; a drawn store of `cands` at an
    /// arbitrary index is refused by layer 7, as it must be. Re-measured the
    /// same day, 3, 948 and 119, when the kingdom drew the stepped loops that
    /// break and an opaque local took its slot's register: other draws; every
    /// module the checker turns away fails layer 7's write sets, 265
    /// store_index_not_affine and 616 write_sets_overlap, random stores).
    /// Re-measured 2026-10-17 10:05Z, 1, 965 and 143, when the kingdom drew
    /// the vec2<f32> registers (reg.w0..w7): other draws (with those rows
    /// left undrawn the same commit measures 3, 948 and 119 exactly; 310 of
    /// the 1,000 draw one, and 17 of those 310 miss the fixpoint, the rate
    /// of the rest); the fixpoint gain is the divisor guard seeing through
    /// `uniform.of_lit` and a fallback (`x / max(0u, 1u)`, which naga folds to
    /// a division by 1, read back as `max(1u, 1u)`). The first AST
    /// differences are the earlier classes: a subgroup operation the reader
    /// sets into a register at its statement, a barrier against a register
    /// set, a literal bound against its uniform spelling; 266
    /// store_index_not_affine and 591 write_sets_overlap at layer 7.
    /// Re-measured 11:30Z, 1, 958 and 127, when the kingdom drew its uniform
    /// loads (wg.uload_i, wg.uload_f): other draws again (with them undrawn
    /// this commit measures 1, 965 and 143 exactly; 142 of the 1,000 draw one
    /// and 4 of those miss the fixpoint, for the earlier reasons: an i32
    /// vector compared as f32, a select nested once more; 259
    /// store_index_not_affine and 614 write_sets_overlap at layer 7, no
    /// layer 4).
    /// 12:35Z, 5, 962 and 130, with every row drawn but the uniform tests and
    /// the reader's two checked rows (a module's workgroup arrays now at one
    /// length for all its entry points: one over 32 KB did not validate).
    /// 15:30Z, 8, 968 and 139 with the 64-element vec4 arrays (risen);
    /// 16:05Z, 1, 970 and 134 with the argmax reductions drawn (other draws).
    /// 17:40Z, AST 4 with the emitter's sg_lane read as the builtin.
    /// 19:54Z, AST 7 with a thread's row or column in an index read as the
    /// row it is. 22:11Z, AST 11 with the clamp below a count read as it.
    const MODULE_AST_FLOOR: usize = 11;
    const MODULE_FIXPOINT_FLOOR: usize = 970;
    const MODULE_LEGAL_FLOOR: usize = 134;


    /// THE KINGDOM'S COMPLETENESS BAR (2026-10-12): 1,000 random trees of
    /// the kingdom's typed sampler (the safe root, depth 7, seeds 0..1000)
    /// render, pass every layer of the legality checker, read back, and the
    /// tree read back renders the same text. A kernel the kingdom writes that
    /// the reader cannot read is a bug in our software (fit_random1: 64 of
    /// 104 random kernels did not read back). Failures are counted by reason.
    /// THE GRID-STRIDE LOOP (2026-10-16): nluca's tuned f1 with its codebook
    /// preload written thread-strided, `for (k = lidx; k < 1024u; k += 256u)`,
    /// reads (as `for (k = 0; k < 1024; k += 256) { if (lidx + k < 1024)
    /// { .. } }`), renders, reads back to the same tree, and the safety layer
    /// proves it race-free under nluca's contract; a start read from memory
    /// is still refused.
    #[test]
    fn a_grid_stride_loop_reads_as_a_guarded_loop_and_is_proved_race_free() {
        let k = WgslKingdom::load_safe();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("problems/nluca_step");
        let f1 = std::fs::read_to_string(dir.join("tuned_f1.wgsl")).expect("tuned_f1.wgsl");
        let old = "    for (k = 0u; k < 4u; k = k + 1u) {\n    wg_a[((((k * 256u) + lidx)) & 1023u)] = buffer_2_global[(((k * 256u) + lidx))];\n    }";
        assert!(f1.contains(old), "the preload as f1 writes it");
        let strided = f1.replace(old, "    for (k = lidx; k < 1024u; k = k + 256u) {\n    wg_a[(k & 1023u)] = buffer_2_global[k];\n    }");
        let read = read_safe(&strided, &k).expect("the grid-stride kernel reads");
        let sx = read.tree.to_sexpr();
        assert!(sx.contains("stmt.if_") && sx.contains("compare.lt"), "the body is guarded: {sx}");
        let text = crate::wgsl::safe_karva::render_tree_with(&read.tree, &k, &read.role).expect("renders").text;
        let back = read_safe(&text, &k).expect("its text reads back");
        assert!(back.tree == read.tree, "the rendered text reads back to the same tree: {:?}", crate::wgsl::render::first_difference(&read.tree, &back.tree));
        let contract = crate::wgsl::safety::Contract::parse(&std::fs::read_to_string(dir.join("safety.contract")).expect("contract")).expect("parses");
        let under = crate::wgsl::knob::Under { role: read.role.clone(), contract: Some(contract) };
        crate::wgsl::knob::check_legal_under(&read.tree, &k, &under).expect("proved race-free and legal");
        // A start read from memory is not a fixed expression the guard may
        // re-read: refused by name.
        let loaded = f1.replace(old, "    for (k = buffer_6_global[0u]; k < 1024u; k = k + 256u) {\n    wg_a[(k & 1023u)] = buffer_2_global[k];\n    }");
        let e = read_safe(&loaded, &k).err().expect("a start read from memory is refused");
        assert!(e.contains("enters with a value our reader does not know"), "{e}");
    }

    #[test]
    fn random_sampler_trees_read_back_to_their_own_text() {
        let k = WgslKingdom::load_safe();
        let role = gram_role();
        let checker = crate::wgsl::check::Checker::new(&k, &role, &crate::wgsl::target::Target::recorded());
        let root = crate::wgsl::render::dual_ty(&k, "kernel.safe").unwrap();
        let n = 1000u64;
        let mut by_reason: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        let mut passed = 0usize;
        // Measured beside it: the trees that read back at all, and the ones
        // whose text read back is a fixpoint (read and rendered again, the same).
        let (mut read_back, mut fixpoint, mut ast_equal) = (0usize, 0usize, 0usize);
        let mut by_ast: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for seed in 0..n {
            let tree = crate::wgsl::render::sample(&k, &role, root, 7, seed).unwrap();
            let verdict: Result<(), String> = (|| {
                let text = crate::wgsl::safe_karva::render_tree_with(&tree, &k, &crate::wgsl::render::gram_role()).map_err(|e| format!("render: {e}"))?.text;
                checker.check(&tree).map_err(|e| format!("check: {} {}", e.layer.number(), e.reason))?;
                if negates_a_matrix(&text) {
                    return Err("render: a unary minus on a matrix (Metal does not compile it)".to_string());
                }
                // naga 0.20 writes degrees and radians to Metal in bare
                // parentheses that can parse as a cast; the renderer writes
                // the products instead (random seed 673, 2026-10-13).
                if text.contains("degrees(") || text.contains("radians(") {
                    return Err("render: degrees or radians as a call (Metal can misparse naga's text)".to_string());
                }
                if std::env::var("ROUNDTRIP_SHOW").ok().is_some_and(|v| v.split(',').any(|x| x.parse::<u64>().ok() == Some(seed))) {
                    println!("SHOW drawn: {}\nSHOW text:\n{text}", tree.to_sexpr());
                }
                let back = read_safe(&text, &k).map_err(|e| format!("read: {e}"))?;
                let again = crate::wgsl::safe_karva::render_tree_with(&back.tree, &k, &crate::wgsl::render::gram_role()).map_err(|e| format!("re-render: {e}"))?.text;
                read_back += 1;
                let canon = crate::wgsl::canonical::canonical(&tree, &k, &role);
                // ROUNDTRIP_SHOW=<seed>[,<seed>...]: those trees' canonical and parsed ASTs.
                if std::env::var("ROUNDTRIP_SHOW").ok().is_some_and(|v| v.split(',').any(|x| x.parse::<u64>().ok() == Some(seed))) {
                    println!("SHOW canonical: {}\nSHOW parsed:    {}", canon.to_sexpr(), back.tree.to_sexpr());
                }
                // THE BAR (Andrew, 2026-10-13): AST to AST. The tree parsed
                // from the emitted text is the tree drawn, node for node, in
                // its canonical spelling (wgsl::canonical: the spellings the
                // text cannot tell apart, as the parser reads them).
                match crate::wgsl::render::first_difference(&canon, &back.tree) {
                    None => ast_equal += 1,
                    // (Tallied against the parsed tree in its canonical
                    // spelling too, so a difference of grouping early in a
                    // block does not hide the first difference of substance.)
                    Some(d) => {
                        let d = crate::wgsl::render::first_difference(&canon, &crate::wgsl::canonical::canonical(&back.tree, &k, &role)).unwrap_or(d);
                        if std::env::var_os("ROUNDTRIP_ALL").is_some() {
                            let mut keys: Vec<String> = crate::wgsl::render::all_differences(&canon, &crate::wgsl::canonical::canonical(&back.tree, &k, &role)).into_iter().map(|k| k.chars().map(|c| if c.is_ascii_digit() { 'N' } else { c }).collect()).collect();
                            keys.sort();
                            keys.dedup();
                            for key in keys {
                                println!("ALL {key} @{seed}");
                            }
                        }
                        let key: String = d.chars().map(|c| if c.is_ascii_digit() { 'N' } else { c }).take(140).collect();
                        by_ast.entry(key).or_insert((0, seed)).0 += 1;
                    }
                }
                let third = read_safe(&again, &k).map_err(|e| e.chars().take(200).collect::<String>()).and_then(|b| crate::wgsl::safe_karva::render_tree_with(&b.tree, &k, &crate::wgsl::render::gram_role()).map(|r| r.text));
                if third.as_ref().is_ok_and(|r| *r == again) {
                    fixpoint += 1;
                } else if std::env::var_os("ROUNDTRIP_FIXPOINT").is_some() {
                    let why = match &third {
                        Err(e) => format!("read again: {e}"),
                        // (Around the first character that differs: a long line's start
                        // is often the same.)
                        Ok(r) => r.lines().zip(again.lines()).find(|(a, b)| a != b).map(|(a, b)| {
                            let at = a.chars().zip(b.chars()).position(|(x, y)| x != y).unwrap_or(0).saturating_sub(40);
                            format!("{} / {}", b.chars().skip(at).take(120).collect::<String>(), a.chars().skip(at).take(120).collect::<String>())
                        }).unwrap_or_else(|| "length".into()),
                    };
                    println!("FIX seed {seed}: {why}");
                }
                if again != text {
                    // The first differing statement (declarations and helpers set aside), as drawn.
                    let body = |t: &str| -> Vec<String> { t.lines().map(str::trim).filter(|l| !(l.starts_with("var") || l.starts_with("fn ") || l.starts_with("let ") || l.starts_with('@') || l.starts_with("//") || l.starts_with("struct "))).map(String::from).collect() };
                    let (a, b) = (body(&text), body(&again));
                    let at = a.iter().zip(&b).position(|(x, y)| x != y);
                    let what = match at {
                        Some(i) => format!("{} / {}", a[i].chars().take(40).collect::<String>(), b[i].chars().take(40).collect::<String>()),
                        None => format!("declarations or length ({} / {} statements)", a.len(), b.len()),
                    };
                    return Err(format!("differs: {what}"));
                }
                Ok(())
            })();
            match verdict {
                Ok(()) => passed += 1,
                Err(e) => {
                    // ROUNDTRIP_SHOW=<seed>: that tree's whole reason.
                    if std::env::var("ROUNDTRIP_SHOW").ok().is_some_and(|v| v.split(',').any(|x| x.parse::<u64>().ok() == Some(seed))) {
                        println!("SHOW reason: {e}");
                    }
                    // The reason without its statement path and numbers.
                    let mut r = e.clone();
                    while let Some(i) = r.find("at statement [") {
                        let j = r[i..].find("]: ").map_or(r.len(), |j| i + j + 3);
                        r.replace_range(i..j, "");
                    }
                    let r: String = r.chars().map(|c| if c.is_ascii_digit() { 'N' } else { c }).take(110).collect();
                    by_reason.entry(r).or_insert((0, seed)).0 += 1;
                }
            }
        }
        let mut reasons: Vec<_> = by_reason.into_iter().collect();
        reasons.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        println!("random trees: {passed} of {n} read back to their own text; {read_back} read back; {fixpoint} read back to a fixpoint (the text read back reads back to itself)");
        let mut ast_reasons: Vec<_> = by_ast.into_iter().collect();
        ast_reasons.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        println!("random trees: {ast_equal} of {n} parse back to the AST drawn, in its canonical spelling (AST to AST); the first difference of the others, canonical / parsed:");
        for (r, (c, seed)) in &ast_reasons {
            println!("  {c:>4}  seed {seed:>4}  {r}");
        }
        if std::env::var_os("ROUNDTRIP_TEXT_REASONS").is_some() {
            for (r, (c, seed)) in &reasons {
                println!("  {c:>4}  seed {seed:>4}  {r}");
            }
        }
        // A ratchet while the groups are fixed one commit at a time (2026-10-12):
        // the count only rises, to 1,000 of 1,000.
        assert_eq!(read_back as u64, n, "random trees that read back: {read_back} of {n}");
        assert!(ast_equal >= AST_FLOOR, "random trees that parse back to the AST drawn: {ast_equal}, fewer than the {AST_FLOOR} already reached");
        assert!(fixpoint >= FIXPOINT_FLOOR, "random trees that read back to a fixpoint: {fixpoint}, fewer than the {FIXPOINT_FLOOR} already reached");
    }
}
