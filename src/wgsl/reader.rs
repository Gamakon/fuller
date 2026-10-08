//! A kernel into roots and trees.
//!
//! naga parses and validates the source. Then every function and entry point
//! is walked statement by statement: a `Store` yields a root whose tree is
//! `(store.<target> [index] value)`, an `If` / `Loop` / `Switch` condition a
//! `(store.branch cond)` root, a `Return` value a `(store.return v)` root and
//! a `Call` argument a `(store.arg.<callee>.<i> v)` root. Each root's value
//! handle is expanded into a tree through the expression arena, stopping at
//! the typed leaves: literals, loads, function arguments, builtins, constants
//! and call results. A handle reached from several roots is expanded in each
//! and counted, so the fold can recover it later; this reader does not fold.
//!
//! Every interior node is named from the function table by its naga node and
//! the scalar kind of the slot it computes; a node the table has no row for
//! is rendered as `naga.<Variant>` and counted, never silently dropped.

use std::collections::{BTreeMap, BTreeSet};

use naga::{Expression, Handle, Module, ScalarKind as NagaKind, Statement, TypeInner, UniqueArena};

use super::table::{FunctionTable, ScalarKind};

/// A tree as the reader produces it.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// A terminal: a load, an argument, a builtin, a constant, a call result.
    /// `name` is the per-kernel terminal name (`load.scores`, `arg.base`,
    /// `builtin.global_invocation_id`); `slot` the WGSL slot it carries.
    Leaf { name: String, slot: String },
    /// A literal with its default form: `(literal.<form> (Num v))`.
    Literal { form: String, slot: String, value: f64 },
    /// `(name child …)` where `name` is `class.instance` or, with `known`
    /// false, `naga.<Variant>` for a node the table has no row for.
    App { name: String, slot: String, known: bool, kids: Vec<Node> },
}

impl Node {
    /// The s-expression the generic Karva encoder reads.
    pub fn to_sexpr(&self) -> String {
        match self {
            Node::Leaf { name, .. } => format!("(Var \"{name}\")"),
            Node::Literal { form, value, .. } => format!("(literal.{form} (Num {}))", fmt_num(*value)),
            Node::App { name, kids, .. } => {
                let parts: Vec<String> = kids.iter().map(Node::to_sexpr).collect();
                if parts.is_empty() {
                    format!("({name})")
                } else {
                    format!("({name} {})", parts.join(" "))
                }
            }
        }
    }

    /// Nodes in the tree, leaves included (a literal counts as two: the
    /// `literal.<form>` node and its `Num`, which is how the encoder sees it).
    pub fn size(&self) -> usize {
        match self {
            Node::Leaf { .. } => 1,
            Node::Literal { .. } => 2,
            Node::App { kids, .. } => 1 + kids.iter().map(Node::size).sum::<usize>(),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Node::Leaf { .. } => 1,
            Node::Literal { .. } => 2,
            Node::App { kids, .. } => 1 + kids.iter().map(Node::depth).max().unwrap_or(0),
        }
    }

    /// The slot this node computes.
    pub fn slot(&self) -> &str {
        match self {
            Node::Leaf { slot, .. } | Node::Literal { slot, .. } | Node::App { slot, .. } => slot,
        }
    }

    /// Every node of the tree, pre-order.
    pub fn walk<'a>(&'a self, out: &mut Vec<&'a Node>) {
        out.push(self);
        if let Node::App { kids, .. } = self {
            for k in kids {
                k.walk(out);
            }
        }
    }
}

fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// What a root is in the scaffold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootKind {
    /// `Statement::Store`; `target` is the pointer's base name.
    Store { target: String },
    /// A local variable's initialiser (`var s = 0.0;`), computed before the
    /// body runs; `path` is empty.
    Init { local: String },
    /// An `if` condition, a `loop`'s `break_if`, or a `switch` selector.
    Condition { statement: &'static str },
    /// `Statement::Return` with a value.
    Return,
    /// One argument of a `Statement::Call`.
    Argument { callee: String, position: usize },
}

/// One root: a hole in the scaffold and the tree that fills it.
#[derive(Debug, Clone)]
pub struct Root {
    pub kind: RootKind,
    /// The path of statement indices from the function body to the holed
    /// statement (nested blocks, then-branches as `accept`, loops as `body`),
    /// counting every statement but naga's `Emit`s.
    pub path: Vec<usize>,
    pub tree: Node,
}

/// One function or entry point, read.
#[derive(Debug, Clone)]
pub struct KernelFunction {
    pub name: String,
    pub entry_point: bool,
    pub roots: Vec<Root>,
    /// Expression handles reached from more than one root, with the count:
    /// the raw repeats the fold would see before the load-sharing rule.
    pub shared_handles: BTreeMap<u32, usize>,
    /// Of those, how many are `Load`s (the sharing rule has not been applied
    /// to them; they are reported, not merged).
    pub shared_loads: usize,
    /// Statements the kingdom does not read (atomics, barriers, image
    /// stores, …), by name.
    pub unread_statements: BTreeMap<String, usize>,
}

/// A kernel read: the naga module and its validation info, kept for the
/// scaffold rebuild, and every function's roots.
pub struct Kernel {
    pub module: Module,
    pub info: naga::valid::ModuleInfo,
    pub functions: Vec<KernelFunction>,
}

impl Kernel {
    /// The per-kernel terminal names every root uses (loads, arguments,
    /// builtins, constants, call results), the pset's variables.
    pub fn terminals(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for f in &self.functions {
            for r in &f.roots {
                let mut nodes = Vec::new();
                r.tree.walk(&mut nodes);
                for n in nodes {
                    if let Node::Leaf { name, .. } = n {
                        out.insert(name.clone());
                    }
                }
            }
        }
        out
    }

    /// Every `class.instance` (or `naga.<Variant>`) name used, with counts.
    pub fn functions_used(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for f in &self.functions {
            for r in &f.roots {
                let mut nodes = Vec::new();
                r.tree.walk(&mut nodes);
                for n in nodes {
                    match n {
                        Node::App { name, .. } => *out.entry(name.clone()).or_default() += 1,
                        Node::Literal { form, .. } => *out.entry(format!("literal.{form}")).or_default() += 1,
                        Node::Leaf { .. } => {}
                    }
                }
            }
        }
        out
    }
}

/// Parse, validate and read a kernel.
pub fn read(source: &str) -> Result<Kernel, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| format!("validation: {e:?}"))?;
    let table = FunctionTable::shipped();
    let mut functions = Vec::new();
    for (handle, f) in module.functions.iter() {
        let fi = &info[handle];
        functions.push(read_function(&module, &table, f, fi, f.name.clone().unwrap_or_else(|| format!("fn{}", handle.index())), false));
    }
    for (i, ep) in module.entry_points.iter().enumerate() {
        let fi = info.get_entry_point(i);
        functions.push(read_function(&module, &table, &ep.function, fi, ep.name.clone(), true));
    }
    Ok(Kernel { module, info, functions })
}

struct Ctx<'a> {
    module: &'a Module,
    table: &'a FunctionTable,
    func: &'a naga::Function,
    info: &'a naga::valid::FunctionInfo,
    /// How many roots reached each handle.
    reached: BTreeMap<u32, usize>,
}

fn read_function(
    module: &Module,
    table: &FunctionTable,
    func: &naga::Function,
    info: &naga::valid::FunctionInfo,
    name: String,
    entry_point: bool,
) -> KernelFunction {
    let mut ctx = Ctx { module, table, func, info, reached: BTreeMap::new() };
    let mut roots = Vec::new();
    let mut unread = BTreeMap::new();
    for (h, local) in func.local_variables.iter() {
        if let Some(init) = local.init {
            let name = local.name.clone().unwrap_or_else(|| format!("local{}", h.index()));
            let tree = expand(&mut ctx, init, true);
            roots.push(Root {
                kind: RootKind::Init { local: format!("local.{name}") },
                path: Vec::new(),
                tree: Node::App { name: format!("store.local.{name}"), slot: "store".into(), known: true, kids: vec![tree] },
            });
        }
    }
    walk_block(&mut ctx, &func.body, &mut Vec::new(), &mut roots, &mut unread);
    let shared_handles: BTreeMap<u32, usize> = ctx.reached.iter().filter(|(_, &n)| n > 1).map(|(&h, &n)| (h, n)).collect();
    let shared_loads = func
        .expressions
        .iter()
        .filter(|(h, e)| shared_handles.contains_key(&(h.index() as u32)) && matches!(e, Expression::Load { .. }))
        .count();
    KernelFunction { name, entry_point, roots, shared_handles, shared_loads, unread_statements: unread }
}

fn walk_block(
    ctx: &mut Ctx,
    block: &naga::Block,
    path: &mut Vec<usize>,
    roots: &mut Vec<Root>,
    unread: &mut BTreeMap<String, usize>,
) {
    // `path` indexes statements with naga's Emits skipped: a rebuilt body
    // places its Emits differently, and the paths must still agree.
    let mut index = 0usize;
    for stmt in block.iter() {
        if matches!(stmt, Statement::Emit(_)) {
            continue;
        }
        path.push(index);
        index += 1;
        match stmt {
            Statement::Emit(_) | Statement::Break | Statement::Continue | Statement::Kill => {}
            Statement::Block(b) => walk_block(ctx, b, path, roots, unread),
            Statement::Store { pointer, value } => {
                let (target, index) = pointer_target(ctx, *pointer);
                let value_tree = expand(ctx, *value, true);
                let mut kids = Vec::new();
                if let Some(ix) = index {
                    kids.push(ix);
                }
                kids.push(value_tree);
                roots.push(Root {
                    kind: RootKind::Store { target: target.clone() },
                    path: path.clone(),
                    tree: Node::App { name: format!("store.{target}"), slot: "store".into(), known: true, kids },
                });
            }
            Statement::If { condition, accept, reject } => {
                roots.push(condition_root(ctx, *condition, "if", path));
                path.push(0);
                walk_block(ctx, accept, path, roots, unread);
                path.pop();
                path.push(1);
                walk_block(ctx, reject, path, roots, unread);
                path.pop();
            }
            Statement::Switch { selector, cases } => {
                roots.push(condition_root(ctx, *selector, "switch", path));
                for (c, case) in cases.iter().enumerate() {
                    path.push(c);
                    walk_block(ctx, &case.body, path, roots, unread);
                    path.pop();
                }
            }
            Statement::Loop { body, continuing, break_if } => {
                path.push(0);
                walk_block(ctx, body, path, roots, unread);
                path.pop();
                path.push(1);
                walk_block(ctx, continuing, path, roots, unread);
                path.pop();
                if let Some(b) = break_if {
                    roots.push(condition_root(ctx, *b, "loop", path));
                }
            }
            Statement::Return { value } => {
                if let Some(v) = value {
                    let tree = expand(ctx, *v, true);
                    roots.push(Root {
                        kind: RootKind::Return,
                        path: path.clone(),
                        tree: Node::App { name: "store.return".into(), slot: "store".into(), known: true, kids: vec![tree] },
                    });
                }
            }
            Statement::Call { function, arguments, .. } => {
                let callee = ctx.module.functions[*function].name.clone().unwrap_or_else(|| format!("fn{}", function.index()));
                for (p, a) in arguments.iter().enumerate() {
                    let tree = expand(ctx, *a, true);
                    roots.push(Root {
                        kind: RootKind::Argument { callee: callee.clone(), position: p },
                        path: path.clone(),
                        tree: Node::App { name: format!("store.arg.{callee}.{p}"), slot: "store".into(), known: true, kids: vec![tree] },
                    });
                }
            }
            other => {
                let name = format!("{other:?}");
                let name = name.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?").to_string();
                *unread.entry(name).or_default() += 1;
            }
        }
        path.pop();
    }
}

fn condition_root(ctx: &mut Ctx, cond: Handle<Expression>, statement: &'static str, path: &[usize]) -> Root {
    let tree = expand(ctx, cond, true);
    Root {
        kind: RootKind::Condition { statement },
        path: path.to_vec(),
        tree: Node::App { name: "store.branch".into(), slot: "store".into(), known: true, kids: vec![tree] },
    }
}

/// The store target: the pointer's base name and, for an indexed access, the
/// index tree. `best_omr2[row * 3u]` → (`best_omr2`, Some(tree of `row * 3u`));
/// a local `win` → (`win`, None); a uniform field `hp.rows` → (`hp.rows`, None).
fn pointer_target(ctx: &mut Ctx, pointer: Handle<Expression>) -> (String, Option<Node>) {
    match &ctx.func.expressions[pointer] {
        Expression::Access { base, index } => {
            let (name, inner) = pointer_target(ctx, *base);
            let ix = expand(ctx, *index, true);
            match inner {
                None => (name, Some(ix)),
                // A nested index (array of arrays): compose the two.
                Some(outer) => (name, Some(Node::App { name: "naga.NestedAccess".into(), slot: "u32".into(), known: false, kids: vec![outer, ix] })),
            }
        }
        Expression::AccessIndex { base, index } => {
            let (name, inner) = pointer_target(ctx, *base);
            if points_at_struct(ctx, *base) {
                (format!("{name}.{}", field_name(ctx, *base, *index)), inner)
            } else {
                // A constant index into an array, vector or matrix: the same
                // index child a dynamic access has, as a literal.
                let ix = Node::Literal { form: "index".into(), slot: "u32".into(), value: f64::from(*index) };
                match inner {
                    None => (name, Some(ix)),
                    Some(outer) => (name, Some(Node::App { name: "naga.NestedAccess".into(), slot: "u32".into(), known: false, kids: vec![outer, ix] })),
                }
            }
        }
        // The target carries its kind, as the table's templates do
        // (`load.buffer`, `load.uniform`, `load.local`, `arg`), so a local and
        // a buffer of the same name are two targets.
        Expression::GlobalVariable(g) => {
            let gv = &ctx.module.global_variables[*g];
            let name = gv.name.clone().unwrap_or_else(|| format!("global{}", g.index()));
            let kind = match gv.space {
                naga::AddressSpace::Storage { .. } => "buffer",
                naga::AddressSpace::Uniform => "uniform",
                naga::AddressSpace::WorkGroup => "workgroup",
                naga::AddressSpace::Private => "private",
                naga::AddressSpace::PushConstant => "push",
                naga::AddressSpace::Handle | naga::AddressSpace::Function => "global",
            };
            (format!("{kind}.{name}"), None)
        }
        Expression::LocalVariable(l) => (format!("local.{}", ctx.func.local_variables[*l].name.clone().unwrap_or_else(|| format!("local{}", l.index()))), None),
        Expression::FunctionArgument(i) => (format!("arg.{}", ctx.func.arguments[*i as usize].name.clone().unwrap_or_else(|| format!("{i}"))), None),
        other => (format!("naga.{}", variant_name(other)), None),
    }
}

/// Whether `base` is, or points at, a struct (so an `AccessIndex` on it is a
/// field, not an element).
fn points_at_struct(ctx: &Ctx, base: Handle<Expression>) -> bool {
    let inner = ctx.info[base].ty.inner_with(&ctx.module.types);
    let through = match inner {
        TypeInner::Pointer { base, .. } => &ctx.module.types[*base].inner,
        other => other,
    };
    matches!(through, TypeInner::Struct { .. })
}

/// The name of field `index` of the struct `base` points at or holds, or the
/// index itself for a vector lane.
fn field_name(ctx: &Ctx, base: Handle<Expression>, index: u32) -> String {
    let inner = ctx.info[base].ty.inner_with(&ctx.module.types);
    let through = match inner {
        TypeInner::Pointer { base, .. } => &ctx.module.types[*base].inner,
        other => other,
    };
    if let TypeInner::Struct { members, .. } = through {
        if let Some(m) = members.get(index as usize) {
            return m.name.clone().unwrap_or_else(|| index.to_string());
        }
    }
    index.to_string()
}

fn variant_name(e: &Expression) -> String {
    let dbg = format!("{e:?}");
    dbg.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?").to_string()
}

/// The slot a type is, as WGSL spells it.
pub fn slot_name(types: &UniqueArena<naga::Type>, inner: &TypeInner) -> String {
    fn scalar(s: naga::Scalar) -> String {
        match (s.kind, s.width) {
            (NagaKind::Float, 4) => "f32".into(),
            (NagaKind::Float, 2) => "f16".into(),
            (NagaKind::Float, 8) => "f64".into(),
            (NagaKind::Sint, 4) => "i32".into(),
            (NagaKind::Uint, 4) => "u32".into(),
            (NagaKind::Sint, 8) => "i64".into(),
            (NagaKind::Uint, 8) => "u64".into(),
            (NagaKind::Bool, _) => "bool".into(),
            (k, w) => format!("{k:?}{}", w * 8),
        }
    }
    match inner {
        TypeInner::Scalar(s) => scalar(*s),
        TypeInner::Vector { size, scalar: s } => format!("vec{}<{}>", *size as u32, scalar(*s)),
        TypeInner::Matrix { columns, rows, scalar: s } => format!("mat{}x{}<{}>", *columns as u32, *rows as u32, scalar(*s)),
        TypeInner::Atomic(s) => format!("atomic<{}>", scalar(*s)),
        TypeInner::Pointer { base, .. } => format!("ptr<{}>", slot_name(types, &types[*base].inner)),
        TypeInner::ValuePointer { size, scalar: s, .. } => match size {
            Some(n) => format!("ptr<vec{}<{}>>", *n as u32, scalar(*s)),
            None => format!("ptr<{}>", scalar(*s)),
        },
        TypeInner::Array { base, .. } => format!("array<{}>", slot_name(types, &types[*base].inner)),
        TypeInner::Struct { .. } => "struct".into(),
        other => {
            let dbg = format!("{other:?}");
            dbg.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?").to_lowercase()
        }
    }
}

fn kind_of(inner: &TypeInner) -> Option<ScalarKind> {
    let s = match inner {
        TypeInner::Scalar(s) | TypeInner::Vector { scalar: s, .. } | TypeInner::Matrix { scalar: s, .. } | TypeInner::ValuePointer { scalar: s, .. } => *s,
        _ => return None,
    };
    Some(match s.kind {
        NagaKind::Float | NagaKind::AbstractFloat => ScalarKind::Float,
        NagaKind::Sint | NagaKind::AbstractInt => ScalarKind::Sint,
        NagaKind::Uint => ScalarKind::Uint,
        NagaKind::Bool => ScalarKind::Bool,
    })
}

/// The default form of a slot before any inference by use.
fn default_form(kind: Option<ScalarKind>) -> &'static str {
    match kind {
        Some(ScalarKind::Float) => "real",
        Some(ScalarKind::Sint) => "int",
        Some(ScalarKind::Uint) => "index",
        Some(ScalarKind::Bool) => "flag",
        None => "opaque",
    }
}

/// Expand an expression handle into a tree. `count` marks the handle as
/// reached by the current root (shared-handle statistics).
fn expand(ctx: &mut Ctx, h: Handle<Expression>, count: bool) -> Node {
    if count {
        *ctx.reached.entry(h.index() as u32).or_default() += 1;
    }
    let inner = ctx.info[h].ty.inner_with(&ctx.module.types).clone();
    let slot = slot_name(&ctx.module.types, &inner);
    let kind = kind_of(&inner);
    let expr = ctx.func.expressions[h].clone();
    match expr {
        Expression::Literal(lit) => {
            let (form, value) = match lit {
                naga::Literal::F32(v) => ("real", f64::from(v)),
                naga::Literal::F64(v) => ("real", v),
                naga::Literal::U32(v) => ("index", f64::from(v)),
                naga::Literal::I32(v) => ("int", f64::from(v)),
                naga::Literal::U64(v) => ("index", v as f64),
                naga::Literal::I64(v) => ("int", v as f64),
                naga::Literal::Bool(b) => ("flag", if b { 1.0 } else { 0.0 }),
                naga::Literal::AbstractInt(v) => ("int", v as f64),
                naga::Literal::AbstractFloat(v) => ("real", v),
            };
            Node::Literal { form: form.into(), slot, value }
        }
        Expression::ZeroValue(_) => Node::Literal { form: default_form(kind).into(), slot, value: 0.0 },
        Expression::Constant(c) => {
            let name = ctx.module.constants[c].name.clone().unwrap_or_else(|| format!("const{}", c.index()));
            Node::Leaf { name: format!("const.{name}"), slot }
        }
        Expression::Override(o) => {
            let name = ctx.module.overrides[o].name.clone().unwrap_or_else(|| format!("override{}", o.index()));
            Node::Leaf { name: format!("override.{name}"), slot }
        }
        Expression::FunctionArgument(i) => {
            let arg = &ctx.func.arguments[i as usize];
            let name = match &arg.binding {
                Some(naga::Binding::BuiltIn(b)) => format!("builtin.{}", snake(&format!("{b:?}"))),
                _ => format!("arg.{}", arg.name.clone().unwrap_or_else(|| format!("{i}"))),
            };
            Node::Leaf { name, slot }
        }
        Expression::GlobalVariable(g) => {
            // A global used as a value (a handle-space resource); as a pointer
            // it is reached through Load, below.
            let name = ctx.module.global_variables[g].name.clone().unwrap_or_else(|| format!("global{}", g.index()));
            Node::Leaf { name: format!("global.{name}"), slot }
        }
        Expression::LocalVariable(l) => {
            let name = ctx.func.local_variables[l].name.clone().unwrap_or_else(|| format!("local{}", l.index()));
            Node::Leaf { name: format!("local.{name}"), slot }
        }
        Expression::Load { pointer } => {
            let (target, index) = pointer_target(ctx, pointer);
            match index {
                Some(ix) => Node::App { name: format!("load.{target}"), slot, known: true, kids: vec![ix] },
                None => Node::Leaf { name: format!("load.{target}"), slot },
            }
        }
        Expression::CallResult(f) => {
            // Named by the RESULT's handle index too: a function called twice
            // has two live results, and the rebuild must put each leaf on the
            // CallResult its own Call statement created.
            let name = ctx.module.functions[f].name.clone().unwrap_or_else(|| format!("fn{}", f.index()));
            Node::Leaf { name: format!("call.{name}@{}", h.index()), slot }
        }
        Expression::Unary { op, expr } => {
            let kid = expand(ctx, expr, count);
            app(ctx, &format!("Unary::{op:?}"), kind, slot, vec![kid])
        }
        Expression::Binary { op, left, right } => {
            let l = expand(ctx, left, count);
            let r = expand(ctx, right, count);
            // A comparison's slot is bool; the row is chosen by its OPERANDS.
            let operand_kind = kind_of(ctx.info[left].ty.inner_with(&ctx.module.types));
            let row_kind = match op {
                naga::BinaryOperator::Equal
                | naga::BinaryOperator::NotEqual
                | naga::BinaryOperator::Less
                | naga::BinaryOperator::LessEqual
                | naga::BinaryOperator::Greater
                | naga::BinaryOperator::GreaterEqual => operand_kind,
                _ => kind,
            };
            app(ctx, &format!("Binary::{op:?}"), row_kind, slot, vec![l, r])
        }
        Expression::Select { condition, accept, reject } => {
            // The table's order is (value, value, cond): accept, reject, condition.
            let a = expand(ctx, accept, count);
            let r = expand(ctx, reject, count);
            let c = expand(ctx, condition, count);
            app(ctx, "Select", kind, slot, vec![a, r, c])
        }
        Expression::Relational { fun, argument } => {
            let kid = expand(ctx, argument, count);
            let operand_kind = kind_of(ctx.info[argument].ty.inner_with(&ctx.module.types));
            app(ctx, &format!("Relational::{fun:?}"), operand_kind, slot, vec![kid])
        }
        Expression::Math { fun, arg, arg1, arg2, arg3 } => {
            let mut kids = vec![expand(ctx, arg, count)];
            for a in [arg1, arg2, arg3].into_iter().flatten() {
                kids.push(expand(ctx, a, count));
            }
            let operand_kind = kind_of(ctx.info[arg].ty.inner_with(&ctx.module.types));
            app(ctx, &format!("Math::{fun:?}"), operand_kind.or(kind), slot, kids)
        }
        Expression::As { expr, kind: to_kind, convert } => {
            let kid = expand(ctx, expr, count);
            let from = kind_of(ctx.info[expr].ty.inner_with(&ctx.module.types));
            let name = convert_name(from, to_kind, convert.is_some());
            match name {
                Some(n) => Node::App { name: n.into(), slot, known: true, kids: vec![kid] },
                None => {
                    let from = from.map_or("none".to_string(), |k| format!("{k:?}").to_lowercase());
                    let to = format!("{to_kind:?}").to_lowercase();
                    Node::App { name: format!("naga.As.{from}_to_{to}.{}", if convert.is_some() { "convert" } else { "bitcast" }), slot, known: false, kids: vec![kid] }
                }
            }
        }
        Expression::Access { base, index } => {
            let b = expand(ctx, base, count);
            let i = expand(ctx, index, count);
            Node::App { name: "shape.access_dyn".into(), slot, known: true, kids: vec![b, i] }
        }
        Expression::AccessIndex { base, index } => {
            let b = expand(ctx, base, count);
            let base_inner = ctx.info[base].ty.inner_with(&ctx.module.types);
            let known = matches!(base_inner, TypeInner::Vector { .. }) && index < 4;
            let name = if known { format!("shape.access_{index}") } else { format!("naga.AccessIndex.{}", field_name(ctx, base, index)) };
            Node::App { name, slot, known, kids: vec![b] }
        }
        Expression::Splat { size, value } => {
            let v = expand(ctx, value, count);
            Node::App { name: format!("shape.splat{}", size as u32), slot, known: true, kids: vec![v] }
        }
        Expression::Swizzle { size, vector, pattern } => {
            let v = expand(ctx, vector, count);
            let pat: String = pattern.iter().take(size as usize).map(|c| match c {
                naga::SwizzleComponent::X => 'x',
                naga::SwizzleComponent::Y => 'y',
                naga::SwizzleComponent::Z => 'z',
                naga::SwizzleComponent::W => 'w',
            }).collect();
            Node::App { name: format!("shape.swizzle.{pat}"), slot, known: true, kids: vec![v] }
        }
        Expression::Compose { components, .. } => {
            let kids: Vec<Node> = components.iter().map(|c| expand(ctx, *c, count)).collect();
            let n = kids.len();
            let (name, known) = match (&inner, n) {
                (TypeInner::Vector { .. }, 2..=4) => (format!("shape.compose{n}"), true),
                (TypeInner::Matrix { .. }, 2) => ("shape.compose_mat".to_string(), true),
                _ => (format!("naga.Compose{n}"), false),
            };
            Node::App { name, slot, known, kids }
        }
        Expression::ArrayLength(e) => {
            let kid = expand(ctx, e, count);
            Node::App { name: "naga.ArrayLength".into(), slot, known: false, kids: vec![kid] }
        }
        other => Node::Leaf { name: format!("naga.{}", variant_name(&other)), slot },
    }
}

/// An application named from the table, or `naga.<node>` when it has no row.
fn app(ctx: &Ctx, naga: &str, kind: Option<ScalarKind>, slot: String, kids: Vec<Node>) -> Node {
    let row = kind.and_then(|k| ctx.table.resolve(naga, k, kids.len()));
    match row {
        Some(t) => Node::App { name: t.name(), slot, known: true, kids },
        None => Node::App { name: format!("naga.{}", naga.replace("::", ".")), slot, known: false, kids },
    }
}

/// The `convert` row for a cast, by the kinds it crosses; `None` when the
/// table has no row for that crossing.
fn convert_name(from: Option<ScalarKind>, to: NagaKind, convert: bool) -> Option<&'static str> {
    Some(match (from?, to, convert) {
        (ScalarKind::Sint, NagaKind::Float, true) => "convert.int_to_f32",
        (ScalarKind::Float, NagaKind::Sint, true) => "convert.f32_to_int",
        (ScalarKind::Uint, NagaKind::Float, true) => "convert.index_to_f32",
        (ScalarKind::Float, NagaKind::Uint, true) => "convert.f32_to_count",
        (ScalarKind::Float, NagaKind::Uint, false) => "convert.bitcast_f32_bits",
        (ScalarKind::Uint, NagaKind::Float, false) => "convert.bitcast_bits_f32",
        _ => return None,
    })
}

fn snake(camel: &str) -> String {
    let mut out = String::new();
    for (i, c) in camel.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_STORE: &str = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = xs[i] * 2.0 + 1.0;
}
"#;

    #[test]
    fn a_one_store_kernel_reads_to_one_root_over_the_kingdoms_names() {
        let k = read(ONE_STORE).unwrap();
        assert_eq!(k.functions.len(), 1);
        let f = &k.functions[0];
        assert_eq!((f.name.as_str(), f.entry_point), ("main", true));
        assert_eq!(f.roots.len(), 1);
        let r = &f.roots[0];
        assert_eq!(r.kind, RootKind::Store { target: "buffer.out".into() });
        assert_eq!(
            r.tree.to_sexpr(),
            "(store.buffer.out (shape.access_0 (Var \"builtin.global_invocation_id\")) (arith.add (arith.mul (load.buffer.xs (shape.access_0 (Var \"builtin.global_invocation_id\"))) (literal.real (Num 2.0))) (literal.real (Num 1.0))))"
        );
        assert!(f.unread_statements.is_empty());
        // gid.x is reached twice (the store index and the load index).
        assert_eq!(f.shared_handles.values().copied().max(), Some(2));
        assert_eq!(f.shared_loads, 0);
        assert_eq!(k.terminals(), ["builtin.global_invocation_id".to_string()].into_iter().collect());
    }

    #[test]
    fn a_branch_yields_a_condition_root_and_a_local_store_has_no_index() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var acc = 0u;
    if (gid.x > 4u) {
        acc = gid.x & 3u;
    }
    out[0] = acc;
}
"#;
        let k = read(src).unwrap();
        let f = &k.functions[0];
        let kinds: Vec<&RootKind> = f.roots.iter().map(|r| &r.kind).collect();
        assert_eq!(
            kinds,
            vec![
                &RootKind::Init { local: "local.acc".into() },
                &RootKind::Condition { statement: "if" },
                &RootKind::Store { target: "local.acc".into() },
                &RootKind::Store { target: "buffer.out".into() },
            ]
        );
        assert_eq!(f.roots[0].tree.to_sexpr(), "(store.local.acc (literal.index (Num 0.0)))");
        assert!(f.roots[0].path.is_empty());
        assert_eq!(f.roots[1].tree.to_sexpr(), "(store.branch (compare.gt (shape.access_0 (Var \"builtin.global_invocation_id\")) (literal.index (Num 4.0))))");
        assert_eq!(f.roots[2].tree.to_sexpr(), "(store.local.acc (bits.and (shape.access_0 (Var \"builtin.global_invocation_id\")) (literal.index (Num 3.0))))");
        // Emit-free paths: the if is the body's first non-Emit statement, the
        // store the first statement of its accept block.
        assert_eq!(f.roots[1].path, vec![0]);
        assert_eq!(f.roots[2].path, vec![0, 0, 0]);
        assert_eq!(f.roots[3].path, vec![1]);
        assert_eq!(f.roots[3].tree.to_sexpr(), "(store.buffer.out (literal.index (Num 0.0)) (Var \"load.local.acc\"))");
    }

    #[test]
    fn a_loop_a_select_and_a_cast_read_to_their_rows_and_an_unrowed_node_is_named_not_dropped() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        s = s + select(1.0, f32(i), i > 1u);
    }
    out[0] = fract(s) + modf(s).fract;
}
"#;
        let k = read(src).unwrap();
        let f = &k.functions[0];
        let sexprs: Vec<String> = f.roots.iter().map(|r| r.tree.to_sexpr()).collect();
        // select(f, t, c) renders (accept t, reject f, cond c), the table's (value, value, cond).
        assert!(sexprs.iter().any(|s| s.contains("(select.scalar_cond (convert.index_to_f32 (Var \"load.local.i\")) (literal.real (Num 1.0)) (compare.gt (Var \"load.local.i\") (literal.index (Num 1.0))))")), "{sexprs:?}");
        assert!(sexprs.iter().any(|s| s == "(store.branch (compare.lt (Var \"load.local.i\") (literal.index (Num 4.0))))"), "the for condition is an if root inside the loop: {sexprs:?}");
        assert!(sexprs.iter().any(|s| s == "(store.local.i (index.add (Var \"load.local.i\") (literal.index (Num 1.0))))"), "{sexprs:?}");
        assert!(sexprs.iter().any(|s| s.starts_with("(store.buffer.out (literal.index (Num 0.0)) ")), "a constant index is an index child: {sexprs:?}");
        let used = k.functions_used();
        assert!(used.contains_key("arith.fract"));
        assert!(used.keys().any(|n| n.starts_with("naga.Math.Modf")), "modf has no row and is named: {used:?}");
    }

    #[test]
    fn a_call_argument_and_a_return_are_roots() {
        let src = r#"
fn twice(x: f32) -> f32 { return x * 2.0; }
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = twice(3.0);
}
"#;
        let k = read(src).unwrap();
        let twice = k.functions.iter().find(|f| f.name == "twice").unwrap();
        assert_eq!(twice.roots.len(), 1);
        assert_eq!(twice.roots[0].kind, RootKind::Return);
        assert_eq!(twice.roots[0].tree.to_sexpr(), "(store.return (arith.mul (Var \"arg.x\") (literal.real (Num 2.0))))");
        let main = k.functions.iter().find(|f| f.name == "main").unwrap();
        let kinds: Vec<&RootKind> = main.roots.iter().map(|r| &r.kind).collect();
        assert_eq!(kinds[0], &RootKind::Argument { callee: "twice".into(), position: 0 });
        assert!(main.roots[1].tree.to_sexpr().contains("(Var \"call.twice@"), "{}", main.roots[1].tree.to_sexpr());
    }

    #[test]
    fn an_invalid_kernel_is_an_error_not_a_partial_read() {
        assert!(read("fn main() { let x: f32 = 1u; }").is_err());
    }
}
