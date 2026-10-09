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
//!
//! **Versions.** A load of a location the function ever bumps (stores,
//! atomics, barriers, calls that store; `versions`) is named with the
//! version current at its program point: `load.local.i@1@v3`,
//! `load.buffer.q.#@v2`. A load of a location the function never bumps
//! carries no version (its version is 0 everywhere, so the bare text already
//! says so). Equal text therefore means equal lineage: two loads with the
//! same name read the same value, whatever lies between them.

use std::collections::{BTreeMap, BTreeSet};

use naga::{Expression, Handle, Module, ScalarKind as NagaKind, Statement, TypeInner, UniqueArena};

use super::table::{FunctionTable, ScalarKind};
use super::versions::{self, Counters, Effects, State};

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

    /// The lineage of this subtree: every versioned load name and every
    /// `let` binding it reads (`docs/PLAN_wgsl_lineage.md` §1). Two subtrees
    /// with equal text have equal lineage; this is what a pointer `let`'s
    /// index is checked against at each use.
    pub fn lineage(&self) -> BTreeSet<String> {
        let mut nodes = Vec::new();
        self.walk(&mut nodes);
        nodes
            .into_iter()
            .filter_map(|n| match n {
                Node::Leaf { name, .. } if name.starts_with("load.") || name.starts_with("let.") => Some(name.clone()),
                Node::App { name, .. } if name.starts_with("load.") => Some(name.clone()),
                _ => None,
            })
            .collect()
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
    /// A source `let` (a naga named expression): its value is bound ONCE
    /// where naga emits it and read later by name, which matters when an
    /// operand's target is stored between the binding and a use
    /// (`let pos = q_pos[head_i]; head_i += 1u;`). Inlining it at each use
    /// would re-read after the store; the rebuilt decode kernel refused 19
    /// of 27 sample genes that way (2026-10-08). `name` is `<name>@<idx>`.
    Let { name: String },
}

/// One root: a hole in the scaffold and the tree that fills it.
#[derive(Debug, Clone)]
pub struct Root {
    pub kind: RootKind,
    /// The path of statement indices from the function body to the holed
    /// statement (nested blocks, then-branches as `accept`, loops as `body`),
    /// counting every statement but naga's `Emit`s.
    pub path: Vec<usize>,
    /// Where the tree is EVALUATED: `path` for every kind but a loop's
    /// `break_if`, which naga evaluates at the end of `continuing`
    /// (`loop path ++ [1, statements in continuing]`). The versions at this
    /// point are the ones the tree's loads carry.
    pub point: Vec<usize>,
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
    /// Versions created per bumped location (stores, phis, barriers, calls):
    /// the locations whose loads carry `@v<n>`.
    pub versions: BTreeMap<String, u32>,
    /// The version of every bumped location before every statement (keyed
    /// by the statement's path) and at every `break_if` point: what
    /// `legality::decide` reads to place a shared definition. A location
    /// absent from a map is at version 0 there.
    pub points: BTreeMap<Vec<usize>, BTreeMap<String, u32>>,
    /// Every `Call` statement with a result: the result's handle index (the
    /// `call.<fn>@<idx>` leaf) → the statement's path. A call result is a
    /// value bound once, like a `let`, available after its statement.
    pub calls: BTreeMap<u32, Vec<usize>>,
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
    let effects = versions::module_effects(&module);
    let mut functions = Vec::new();
    for (handle, f) in module.functions.iter() {
        let ctx = Ctx::new(&module, &table, &effects, handle.index(), f, &info[handle]);
        functions.push(read_function(ctx, f.name.clone().unwrap_or_else(|| format!("fn{}", handle.index())), false)?);
    }
    for (i, ep) in module.entry_points.iter().enumerate() {
        let ctx = Ctx::new(&module, &table, &effects, module.functions.len() + i, &ep.function, info.get_entry_point(i));
        functions.push(read_function(ctx, ep.name.clone(), true)?);
    }
    Ok(Kernel { module, info, functions })
}

/// The states collected for a loop's joins while its body is walked.
#[derive(Default)]
struct LoopFrame {
    breaks: Vec<State>,
    continues: Vec<State>,
}

struct Ctx<'a> {
    module: &'a Module,
    table: &'a FunctionTable,
    func: &'a naga::Function,
    info: &'a naga::valid::FunctionInfo,
    /// Every function's effects, by the reader's function index.
    effects: &'a [Effects],
    /// Locations this function bumps anywhere: their loads carry a version.
    bumped: BTreeSet<String>,
    state: State,
    counters: Counters,
    loops: Vec<LoopFrame>,
    /// How many roots reached each handle.
    reached: BTreeMap<u32, usize>,
    /// Handles already bound as `let` roots: a later use is a leaf.
    lets: BTreeSet<u32>,
    /// Pointer-typed `let`s (`let p = &a[i]`): not roots; their index is
    /// re-expanded at each use, which is exact only while the index's
    /// lineage is unchanged since the binding. Handle → lineage at binding.
    pointer_lets: BTreeMap<u32, BTreeSet<String>>,
    errors: Vec<String>,
    points: BTreeMap<Vec<usize>, BTreeMap<String, u32>>,
    calls: BTreeMap<u32, Vec<usize>>,
}

impl<'a> Ctx<'a> {
    fn new(module: &'a Module, table: &'a FunctionTable, effects: &'a [Effects], index: usize, func: &'a naga::Function, info: &'a naga::valid::FunctionInfo) -> Self {
        Ctx {
            module,
            table,
            func,
            info,
            effects,
            bumped: versions::bumped_locations(func, &effects[index]),
            state: State::entry(),
            counters: Counters::default(),
            loops: Vec::new(),
            reached: BTreeMap::new(),
            lets: BTreeSet::new(),
            pointer_lets: BTreeMap::new(),
            errors: Vec::new(),
            points: BTreeMap::new(),
            calls: BTreeMap::new(),
        }
    }

    /// Record the versions current before the statement at `path`.
    fn record_point(&mut self, path: &[usize]) {
        self.points.insert(path.to_vec(), self.state.current.clone());
    }

    /// A new version of `loc` at the current point.
    fn bump(&mut self, loc: &str) {
        versions::bump(&mut self.state, &mut self.counters, loc);
    }

    /// The location a pointer expression stores to or loads from, with a
    /// pointer parameter named as `pointer_target` names it.
    fn location(&self, pointer: Handle<Expression>) -> String {
        let loc = versions::location_of(self.module, self.func, pointer);
        match loc.strip_prefix("arg#") {
            Some(i) => versions::param_location(self.func, i.parse().expect("arg# carries the index")),
            None => loc,
        }
    }

    /// A call's effects at this site: the callee's globals, and its stored
    /// pointer parameters substituted by the arguments passed here.
    fn apply_call(&mut self, function: Handle<naga::Function>, arguments: &[Handle<Expression>]) {
        let callee = self.effects[function.index()].clone();
        for loc in &callee.locations {
            if !loc.starts_with("local.") {
                self.bump(loc);
            }
        }
        for &p in &callee.params {
            if let Some(&a) = arguments.get(p as usize) {
                let loc = self.location(a);
                self.bump(&loc);
            }
        }
    }

    fn barrier(&mut self, flags: naga::Barrier) {
        let mut locs = BTreeSet::new();
        versions::barrier_locations(self.module, flags, &mut locs);
        for loc in locs {
            self.bump(&loc);
        }
    }
}

fn read_function(mut ctx: Ctx, name: String, entry_point: bool) -> Result<KernelFunction, String> {
    let func = ctx.func;
    let mut roots = Vec::new();
    let mut unread = BTreeMap::new();
    for (h, local) in func.local_variables.iter() {
        if let Some(init) = local.init {
            let name = format!("{}@{}", local.name.clone().unwrap_or_else(|| "local".into()), h.index());
            let tree = expand(&mut ctx, init, true);
            roots.push(Root {
                kind: RootKind::Init { local: format!("local.{name}") },
                path: Vec::new(),
                point: Vec::new(),
                tree: Node::App { name: format!("store.local.{name}"), slot: "store".into(), known: true, kids: vec![tree] },
            });
        }
    }
    walk_block(&mut ctx, &func.body, &mut Vec::new(), &mut roots, &mut unread);
    if let Some(e) = ctx.errors.first() {
        return Err(format!("{name}: {e}"));
    }
    let shared_handles: BTreeMap<u32, usize> = ctx.reached.iter().filter(|(_, &n)| n > 1).map(|(&h, &n)| (h, n)).collect();
    let shared_loads = func
        .expressions
        .iter()
        .filter(|(h, e)| shared_handles.contains_key(&(h.index() as u32)) && matches!(e, Expression::Load { .. }))
        .count();
    Ok(KernelFunction { name, entry_point, roots, shared_handles, shared_loads, unread_statements: unread, versions: ctx.counters.created().clone(), points: ctx.points, calls: ctx.calls })
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
        if let Statement::Emit(range) = stmt {
            // A named expression in this Emit is a `let`: a root at the
            // position of the statement that follows (the path the rebuild
            // places it before), in handle order.
            for h in range.clone() {
                if let Some(name) = ctx.func.named_expressions.get(&h).filter(|n| !is_bake(n)) {
                    if matches!(ctx.info[h].ty.inner_with(&ctx.module.types), TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. }) {
                        // A pointer `let` binds a place, not a value: no root.
                        // Its index is re-expanded at each use; the lineage
                        // at the binding is what each use is checked against.
                        let lineage = pointer_lineage(ctx, h);
                        ctx.pointer_lets.insert(h.index() as u32, lineage);
                        continue;
                    }
                    let name = format!("{name}@{}", h.index());
                    ctx.lets.insert(h.index() as u32);
                    let tree = expand_let(ctx, h);
                    let mut p = path.clone();
                    p.push(index);
                    ctx.record_point(&p);
                    roots.push(Root {
                        kind: RootKind::Let { name: name.clone() },
                        path: p.clone(),
                        point: p,
                        tree: Node::App { name: format!("store.let.{name}"), slot: "store".into(), known: true, kids: vec![tree] },
                    });
                }
            }
            continue;
        }
        path.push(index);
        index += 1;
        ctx.record_point(path);
        match stmt {
            Statement::Emit(_) => {}
            Statement::Break => {
                let s = ctx.state.clone();
                if let Some(frame) = ctx.loops.last_mut() {
                    frame.breaks.push(s);
                }
                ctx.state.live = false;
            }
            Statement::Continue => {
                let s = ctx.state.clone();
                if let Some(frame) = ctx.loops.last_mut() {
                    frame.continues.push(s);
                }
                ctx.state.live = false;
            }
            Statement::Kill => ctx.state.live = false,
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
                    point: path.clone(),
                    tree: Node::App { name: format!("store.{target}"), slot: "store".into(), known: true, kids },
                });
                let loc = ctx.location(*pointer);
                ctx.bump(&loc);
            }
            Statement::If { condition, accept, reject } => {
                roots.push(condition_root(ctx, *condition, "if", path));
                let entry = ctx.state.clone();
                path.push(0);
                walk_block(ctx, accept, path, roots, unread);
                path.pop();
                let after_accept = std::mem::replace(&mut ctx.state, entry);
                path.push(1);
                walk_block(ctx, reject, path, roots, unread);
                path.pop();
                let after_reject = ctx.state.clone();
                ctx.state = versions::merge(&[after_accept, after_reject], &mut ctx.counters);
            }
            Statement::Switch { selector, cases } => {
                roots.push(condition_root(ctx, *selector, "switch", path));
                let entry = ctx.state.clone();
                let mut ends: Vec<State> = Vec::new();
                let mut carry: Option<State> = None;
                for (c, case) in cases.iter().enumerate() {
                    // A fall-through case's end flows into the next case.
                    ctx.state = match carry.take() {
                        Some(from_above) => versions::merge(&[entry.clone(), from_above], &mut ctx.counters),
                        None => entry.clone(),
                    };
                    path.push(c);
                    walk_block(ctx, &case.body, path, roots, unread);
                    path.pop();
                    if case.fall_through {
                        carry = Some(ctx.state.clone());
                    } else {
                        ends.push(ctx.state.clone());
                    }
                }
                if let Some(s) = carry {
                    ends.push(s);
                }
                if !cases.iter().any(|c| matches!(c.value, naga::SwitchValue::Default)) {
                    ends.push(entry);
                }
                ctx.state = versions::merge(&ends, &mut ctx.counters);
            }
            Statement::Loop { body, continuing, break_if } => {
                // Header phi: every location the loop stores takes a fresh
                // version before the body, distinct from the pre-loop one.
                let stored = versions::loop_stores(ctx.module, ctx.func, ctx.effects, body, continuing);
                for loc in &stored {
                    ctx.bump(loc);
                }
                ctx.loops.push(LoopFrame::default());
                path.push(0);
                walk_block(ctx, body, path, roots, unread);
                path.pop();
                // The back edge: the end of the body and every `continue`
                // flow into `continuing`.
                let body_end = ctx.state.clone();
                let mut into_continuing = vec![body_end];
                into_continuing.extend(std::mem::take(&mut ctx.loops.last_mut().expect("the loop's frame").continues));
                ctx.state = versions::merge(&into_continuing, &mut ctx.counters);
                path.push(1);
                walk_block(ctx, continuing, path, roots, unread);
                if let Some(b) = break_if {
                    // Evaluated at the end of `continuing`, under its versions.
                    let n = continuing.iter().filter(|s| !matches!(s, Statement::Emit(_))).count();
                    path.push(n);
                    ctx.record_point(path);
                    let mut root = condition_root(ctx, *b, "loop", path);
                    path.pop();
                    path.pop();
                    root.path = path.clone();
                    roots.push(root);
                } else {
                    path.pop();
                }
                // Exit phi: every `break` and the `break_if` join; a location
                // the loop stores takes a fresh version after it.
                let frame = ctx.loops.pop().expect("the loop's frame");
                let mut exits = frame.breaks;
                if break_if.is_some() {
                    exits.push(ctx.state.clone());
                }
                if exits.is_empty() {
                    // No way out: what follows is dead.
                    exits.push(State { current: ctx.state.current.clone(), live: false });
                }
                ctx.state = versions::merge(&exits, &mut ctx.counters);
                for loc in &stored {
                    ctx.bump(loc);
                }
            }
            Statement::Return { value } => {
                if let Some(v) = value {
                    let tree = expand(ctx, *v, true);
                    roots.push(Root {
                        kind: RootKind::Return,
                        path: path.clone(),
                        point: path.clone(),
                        tree: Node::App { name: "store.return".into(), slot: "store".into(), known: true, kids: vec![tree] },
                    });
                }
                ctx.state.live = false;
            }
            Statement::Call { function, arguments, result } => {
                let callee = ctx.module.functions[*function].name.clone().unwrap_or_else(|| format!("fn{}", function.index()));
                if let Some(r) = result {
                    ctx.calls.insert(r.index() as u32, path.clone());
                }
                for (p, a) in arguments.iter().enumerate() {
                    let tree = expand(ctx, *a, true);
                    roots.push(Root {
                        kind: RootKind::Argument { callee: callee.clone(), position: p },
                        path: path.clone(),
                        point: path.clone(),
                        tree: Node::App { name: format!("store.callarg.{callee}.{p}"), slot: "store".into(), known: true, kids: vec![tree] },
                    });
                }
                ctx.apply_call(*function, arguments);
            }
            other => {
                let name = format!("{other:?}");
                let name = name.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?").to_string();
                *unread.entry(name).or_default() += 1;
                // Unread, but not without effect on what follows.
                match other {
                    Statement::Barrier(flags) => ctx.barrier(*flags),
                    Statement::WorkGroupUniformLoad { .. } => ctx.barrier(naga::Barrier::WORK_GROUP),
                    Statement::Atomic { pointer, .. } | Statement::ImageStore { image: pointer, .. } => {
                        let loc = ctx.location(*pointer);
                        ctx.bump(&loc);
                    }
                    _ => {}
                }
            }
        }
        path.pop();
    }
}

/// The lineage of a pointer `let`'s index expressions at its binding.
fn pointer_lineage(ctx: &mut Ctx, h: Handle<Expression>) -> BTreeSet<String> {
    let (_, index) = pointer_target(ctx, h);
    index.map(|n| n.lineage()).unwrap_or_default()
}

fn condition_root(ctx: &mut Ctx, cond: Handle<Expression>, statement: &'static str, path: &[usize]) -> Root {
    let tree = expand(ctx, cond, true);
    Root {
        kind: RootKind::Condition { statement },
        path: path.to_vec(),
        point: path.to_vec(),
        tree: Node::App { name: "store.branch".into(), slot: "store".into(), known: true, kids: vec![tree] },
    }
}

/// The store target: the pointer's base name and, for an indexed access, the
/// index tree. `best_omr2[row * 3u]` → (`best_omr2`, Some(tree of `row * 3u`));
/// a local `win` → (`win`, None); a uniform field `hp.rows` → (`hp.rows`, None).
fn pointer_target(ctx: &mut Ctx, pointer: Handle<Expression>) -> (String, Option<Node>) {
    let (name, index) = pointer_steps(ctx, pointer);
    // A use of a pointer `let` re-expands its index here; that is the
    // kernel's value only while nothing in the index's lineage was bumped
    // since the binding.
    if let Some(at_binding) = ctx.pointer_lets.get(&(pointer.index() as u32)) {
        let now = index.as_ref().map(Node::lineage).unwrap_or_default();
        if now != *at_binding {
            let let_name = ctx.func.named_expressions.get(&pointer).cloned().unwrap_or_default();
            ctx.errors.push(format!(
                "pointer let `{let_name}` is used after a store to its index's lineage (bound over {at_binding:?}, used over {now:?}); the kingdom cannot yet bind a pointer once"
            ));
        }
    }
    (name, index)
}

fn pointer_steps(ctx: &mut Ctx, pointer: Handle<Expression>) -> (String, Option<Node>) {
    match &ctx.func.expressions[pointer] {
        // The target names each step in order: a field by name, an index as
        // `#` (its tree is the index child; several nest as `naga.NestedAccess`
        // outer-first), so `buf[i].hi` is `buffer.buf.#.hi` and the rebuild
        // applies the steps in the order the kernel did.
        Expression::Access { base, index } => {
            let (name, inner) = pointer_target(ctx, *base);
            let ix = expand(ctx, *index, true);
            (format!("{name}.#"), Some(nest(inner, ix)))
        }
        Expression::AccessIndex { base, index } => {
            let (name, inner) = pointer_target(ctx, *base);
            if points_at_struct(ctx, *base) {
                (format!("{name}.{}", field_name(ctx, *base, *index)), inner)
            } else {
                // A constant index into an array, vector or matrix: the same
                // index child a dynamic access has, as a literal.
                let ix = Node::Literal { form: "index".into(), slot: "u32".into(), value: f64::from(*index) };
                (format!("{name}.#"), Some(nest(inner, ix)))
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
        // A local carries its arena index: two locals in different scopes may
        // share a name, and the index is what the rebuild selects by.
        Expression::LocalVariable(l) => (format!("local.{}@{}", ctx.func.local_variables[*l].name.clone().unwrap_or_else(|| "local".into()), l.index()), None),
        Expression::FunctionArgument(i) => (format!("arg.{}", ctx.func.arguments[*i as usize].name.clone().unwrap_or_else(|| format!("{i}"))), None),
        other => (format!("naga.{}", variant_name(other)), None),
    }
}

/// Chain a further index after an earlier one, outer first.
fn nest(inner: Option<Node>, ix: Node) -> Node {
    match inner {
        None => ix,
        Some(outer) => Node::App { name: "naga.NestedAccess".into(), slot: "u32".into(), known: false, kids: vec![outer, ix] },
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

pub(crate) fn kind_of(inner: &TypeInner) -> Option<ScalarKind> {
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
/// A name naga's writer invented for a multiply-used expression (`_e12`),
/// not a source `let`: reading a written text back must not turn those
/// into roots, or the original and its round trip would disagree.
pub(crate) fn is_bake(name: &str) -> bool {
    name.strip_prefix("_e").is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

/// Expand the tree of a `let` root itself: its own handle is not a leaf,
/// but every earlier let it reaches is.
fn expand_let(ctx: &mut Ctx, h: Handle<Expression>) -> Node {
    ctx.lets.remove(&(h.index() as u32));
    let tree = expand(ctx, h, true);
    ctx.lets.insert(h.index() as u32);
    tree
}

fn expand(ctx: &mut Ctx, h: Handle<Expression>, count: bool) -> Node {
    if count {
        *ctx.reached.entry(h.index() as u32).or_default() += 1;
    }
    let inner = ctx.info[h].ty.inner_with(&ctx.module.types).clone();
    let slot = slot_name(&ctx.module.types, &inner);
    if ctx.lets.contains(&(h.index() as u32)) {
        let name = ctx.func.named_expressions.get(&h).cloned().unwrap_or_else(|| "let".into());
        return Node::Leaf { name: format!("let.{name}@{}", h.index()), slot };
    }
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
                Some(naga::Binding::BuiltIn(b)) => format!("builtin.{}", super::naga_names::builtin_name(*b)),
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
            let name = ctx.func.local_variables[l].name.clone().unwrap_or_else(|| "local".into());
            Node::Leaf { name: format!("local.{name}@{}", l.index()), slot }
        }
        Expression::Load { pointer } => {
            let (target, index) = pointer_target(ctx, pointer);
            // A load of a location this function bumps anywhere carries the
            // version current here; one it never bumps is version 0 always
            // and says so by having no suffix.
            let loc = ctx.location(pointer);
            let version = if ctx.bumped.contains(&loc) { format!("@v{}", ctx.state.version(&loc)) } else { String::new() };
            match index {
                Some(ix) => Node::App { name: format!("load.{target}{version}"), slot, known: true, kids: vec![ix] },
                None => Node::Leaf { name: format!("load.{target}{version}"), slot },
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
            // A struct field read from a VALUE is named by member position:
            // structural, and immune to the writer renaming a field.
            let known = matches!(base_inner, TypeInner::Vector { .. }) && index < 4;
            let name = if known { format!("shape.access_{index}") } else { format!("naga.AccessIndex.{index}") };
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
        Expression::Compose { ty, components } => {
            // The composed TYPE rides in the name (the gene text has no other
            // place for it, and the rebuild needs the exact type handle): the
            // slot for a vector or matrix, the struct's name otherwise.
            let kids: Vec<Node> = components.iter().map(|c| expand(ctx, *c, count)).collect();
            let n = kids.len();
            let (name, known) = match (&inner, n) {
                (TypeInner::Vector { .. }, 2..=4) => (format!("shape.compose{n}.{slot}"), true),
                (TypeInner::Matrix { .. }, 2) => (format!("shape.compose_mat.{slot}"), true),
                // Any other composed type (a struct, an array) by its handle
                // index, which a cloned module keeps.
                _ => (format!("naga.Compose.type{}", ty.index()), false),
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
        // `let i = gid.x;` is a root of its own; the store reads it by name.
        assert_eq!(f.roots.len(), 2);
        let (l, r) = (&f.roots[0], &f.roots[1]);
        assert!(matches!(&l.kind, RootKind::Let { name } if name.starts_with("i@")), "{:?}", l.kind);
        assert!(l.tree.to_sexpr().starts_with("(store.let.i@"), "{}", l.tree.to_sexpr());
        assert!(l.tree.to_sexpr().ends_with(" (shape.access_0 (Var \"builtin.global_invocation_id\")))"));
        assert_eq!(r.kind, RootKind::Store { target: "buffer.out.#".into() });
        let s = r.tree.to_sexpr();
        assert!(s.starts_with("(store.buffer.out.# (Var \"let.i@"), "{s}");
        assert!(s.contains("(arith.add (arith.mul (load.buffer.xs.# (Var \"let.i@"), "{s}");
        assert!(s.ends_with("(literal.real (Num 2.0))) (literal.real (Num 1.0))))"), "{s}");
        assert!(f.unread_statements.is_empty());
        assert_eq!(f.shared_loads, 0);
        let terms = k.terminals();
        assert!(terms.contains("builtin.global_invocation_id") && terms.iter().any(|t| t.starts_with("let.i@")), "{terms:?}");
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
                &RootKind::Init { local: "local.acc@0".into() },
                &RootKind::Condition { statement: "if" },
                &RootKind::Store { target: "local.acc@0".into() },
                &RootKind::Store { target: "buffer.out.#".into() },
            ]
        );
        assert_eq!(f.roots[0].tree.to_sexpr(), "(store.local.acc@0 (literal.index (Num 0.0)))");
        assert!(f.roots[0].path.is_empty());
        assert_eq!(f.roots[1].tree.to_sexpr(), "(store.branch (compare.gt (shape.access_0 (Var \"builtin.global_invocation_id\")) (literal.index (Num 4.0))))");
        assert_eq!(f.roots[2].tree.to_sexpr(), "(store.local.acc@0 (bits.and (shape.access_0 (Var \"builtin.global_invocation_id\")) (literal.index (Num 3.0))))");
        // Emit-free paths: the if is the body's first non-Emit statement, the
        // store the first statement of its accept block.
        assert_eq!(f.roots[1].path, vec![0]);
        assert_eq!(f.roots[2].path, vec![0, 0, 0]);
        assert_eq!(f.roots[3].path, vec![1]);
        assert!(f.roots.iter().all(|r| !matches!(r.kind, RootKind::Let { .. })), "no let in this kernel");
        // acc is stored in one arm only: the read after the join is the phi (v2).
        assert_eq!(f.roots[3].tree.to_sexpr(), "(store.buffer.out.# (literal.index (Num 0.0)) (Var \"load.local.acc@0@v2\"))");
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
        // Inside the loop `i` reads its header version (v1); `s` after the loop its exit version.
        assert!(sexprs.iter().any(|s| s.contains("(select.scalar_cond (convert.index_to_f32 (Var \"load.local.i@1@v1\")) (literal.real (Num 1.0)) (compare.gt (Var \"load.local.i@1@v1\") (literal.index (Num 1.0))))")), "{sexprs:?}");
        assert!(sexprs.iter().any(|s| s == "(store.branch (compare.lt (Var \"load.local.i@1@v1\") (literal.index (Num 4.0))))"), "the for condition is an if root inside the loop: {sexprs:?}");
        assert!(sexprs.iter().any(|s| s == "(store.local.i@1 (index.add (Var \"load.local.i@1@v1\") (literal.index (Num 1.0))))"), "{sexprs:?}");
        assert!(sexprs.iter().any(|s| s.contains("(arith.fract (Var \"load.local.s@0@v3\"))")), "{sexprs:?}");
        assert!(sexprs.iter().any(|s| s.starts_with("(store.buffer.out.# (literal.index (Num 0.0)) ")), "a constant index is an index child: {sexprs:?}");
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

    // ---- memory versions (docs/PLAN_wgsl_lineage.md §6) ----

    fn sexprs(src: &str, function: &str) -> (Vec<String>, BTreeMap<String, u32>) {
        let k = read(src).unwrap_or_else(|e| panic!("{e}"));
        let f = k.functions.iter().find(|f| f.name == function).unwrap_or_else(|| panic!("no function {function}"));
        (f.roots.iter().map(|r| r.tree.to_sexpr()).collect(), f.versions.clone())
    }

    #[test]
    fn a_store_bumps_its_location_and_no_other_and_a_uniform_never_bumps() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<uniform> u: f32;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    b[i] = a[i] * u;
    a[i] = a[i] + u;
    b[i] = a[i] * u;
}
"#;
        let (s, versions) = sexprs(src, "main");
        assert!(s[1].contains("(load.buffer.a.#@v0 ") && s[1].contains("(Var \"load.uniform.u\")"), "{}", s[1]);
        assert!(s[2].contains("(load.buffer.a.#@v0 "), "the store's own value reads the version before it: {}", s[2]);
        assert!(s[3].contains("(load.buffer.a.#@v1 "), "after the store to a: {}", s[3]);
        assert!(!s[3].contains("@v2"), "the store to b bumps b, not a: {}", s[3]);
        assert_eq!(versions, BTreeMap::from([("buffer.a".to_string(), 1), ("buffer.b".to_string(), 2)]), "u is never bumped");
    }

    #[test]
    fn a_call_bumps_what_the_callee_stores() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
fn poke() { a[0] = 1.0; }
@compute @workgroup_size(1)
fn main() {
    let x = a[1];
    poke();
    a[2] = a[1] + x;
}
"#;
        let (s, versions) = sexprs(src, "main");
        assert!(s[0].contains("(load.buffer.a.#@v0 "), "{}", s[0]);
        assert!(s[1].contains("(load.buffer.a.#@v1 ") && s[1].contains("(Var \"let.x@"), "{}", s[1]);
        assert_eq!(versions.get("buffer.a"), Some(&2), "the call and the store");
    }

    #[test]
    fn a_loop_gives_a_stored_local_a_header_version_a_post_store_version_and_an_exit_version() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 0.0;
    var i = 0u;
    out[0] = s + xs[0];
    loop {
        if (i >= 4u) { break; }
        out[1] = s + xs[0];
        s = s + xs[i];
        out[2] = s * xs[0];
        i = i + 1u;
    }
    out[3] = s + xs[0];
}
"#;
        let (s, versions) = sexprs(src, "main");
        let store = |n: usize| s.iter().find(|t| t.starts_with(&format!("(store.buffer.out.# (literal.index (Num {n}.0))"))).cloned().unwrap_or_else(|| panic!("no store to out[{n}]: {s:?}"));
        assert!(store(0).contains("(Var \"load.local.s@0@v0\")"), "before the loop: {}", store(0));
        assert!(store(1).contains("(Var \"load.local.s@0@v1\")"), "the header version: {}", store(1));
        assert!(store(2).contains("(Var \"load.local.s@0@v2\")"), "after the store in the body: {}", store(2));
        assert!(store(3).contains("(Var \"load.local.s@0@v3\")"), "the exit version, a third one: {}", store(3));
        // A location the loop never stores has the same text inside and after it.
        assert!(s.iter().filter(|t| t.contains("(load.buffer.xs.# ")).count() >= 4, "xs carries no version: {s:?}");
        assert!(!s.iter().any(|t| t.contains("xs.#@v")), "{s:?}");
        assert_eq!(versions.get("local.s@0"), Some(&3));
        assert_eq!(versions.get("local.i@1"), Some(&3), "header, the increment, exit");
    }

    #[test]
    fn a_store_in_one_arm_bumps_after_the_join_for_if_and_for_switch() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var s = 1.0;
    if (gid.x > 0u) { s = 2.0; }
    out[0] = s;
    switch (gid.x) {
        case 0u: { s = 4.0; }
        default: {}
    }
    out[1] = s;
    if (gid.x > 2u) { out[2] = s; } else { out[3] = s; }
    out[4] = s;
}
"#;
        let (s, versions) = sexprs(src, "main");
        let store = |n: usize| s.iter().find(|t| t.starts_with(&format!("(store.buffer.out.# (literal.index (Num {n}.0))"))).cloned().unwrap();
        assert!(store(0).ends_with("(Var \"load.local.s@0@v2\"))"), "phi after the if: {}", store(0));
        assert!(store(1).ends_with("(Var \"load.local.s@0@v4\"))"), "phi after the switch: {}", store(1));
        // No store in either arm: no phi, the same version in both arms and after.
        assert!(store(2).ends_with("@v4\"))") && store(3).ends_with("@v4\"))") && store(4).ends_with("@v4\"))"), "{s:?}");
        assert_eq!(versions.get("local.s@0"), Some(&4));
    }

    #[test]
    fn a_barrier_bumps_every_workgroup_location_and_an_atomic_bumps_its_own() {
        let src = r#"
var<workgroup> w: array<f32, 64>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read_write> cnt: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    w[lid.x] = f32(lid.x);
    workgroupBarrier();
    out[0] = w[0];
    workgroupBarrier();
    out[1] = w[0];
    atomicAdd(&cnt, 1u);
    out[2] = f32(atomicLoad(&cnt));
}
"#;
        let k = read(src).unwrap_or_else(|e| panic!("{e}"));
        let f = &k.functions[0];
        let s: Vec<String> = f.roots.iter().map(|r| r.tree.to_sexpr()).collect();
        let store = |n: usize| s.iter().find(|t| t.starts_with(&format!("(store.buffer.out.# (literal.index (Num {n}.0))"))).cloned().unwrap();
        assert!(store(0).contains("(load.workgroup.w.#@v2 "), "stored once, then the barrier: {}", store(0));
        assert!(store(1).contains("(load.workgroup.w.#@v3 "), "the second barrier: {}", store(1));
        assert!(store(2).contains("cnt@v1"), "the atomic bumped cnt: {}", store(2));
        assert_eq!(f.versions.get("workgroup.w"), Some(&3));
        assert_eq!(f.versions.get("buffer.cnt"), Some(&1));
        assert_eq!(f.unread_statements.get("Barrier"), Some(&2));
        assert_eq!(f.unread_statements.get("Atomic"), Some(&1));
    }

    #[test]
    fn a_pointer_let_store_bumps_its_base_and_a_pointer_parameter_bumps_the_callers_argument_at_that_site() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
fn poke(p: ptr<function, f32>) { *p = 1.0; }
@compute @workgroup_size(1)
fn main() {
    var x = 0.0;
    var y = 0.0;
    let before = x + y;
    poke(&x);
    out[0] = x + y;
    poke(&y);
    out[1] = x + y;
    let p = &a[2];
    out[2] = a[2];
    *p = 5.0;
    out[3] = a[2];
}
"#;
        let k = read(src).unwrap_or_else(|e| panic!("{e}"));
        let poke = k.functions.iter().find(|f| f.name == "poke").unwrap();
        assert_eq!(poke.versions, BTreeMap::from([("arg.p".to_string(), 1)]));
        assert_eq!(poke.roots[0].kind, RootKind::Store { target: "arg.p".into() });
        let main = k.functions.iter().find(|f| f.name == "main").unwrap();
        let s: Vec<String> = main.roots.iter().map(|r| r.tree.to_sexpr()).collect();
        let store = |n: usize| s.iter().find(|t| t.starts_with(&format!("(store.buffer.out.# (literal.index (Num {n}.0))"))).cloned().unwrap();
        let before = main.roots.iter().find(|r| matches!(&r.kind, RootKind::Let { name } if name.starts_with("before@"))).unwrap().tree.to_sexpr();
        assert!(before.contains("load.local.x@0@v0") && before.contains("load.local.y@1@v0"), "{before}");
        assert!(store(0).contains("load.local.x@0@v1") && store(0).contains("load.local.y@1@v0"), "poke(&x) bumped x only: {}", store(0));
        assert!(store(1).contains("load.local.x@0@v1") && store(1).contains("load.local.y@1@v1"), "poke(&y) bumped y only: {}", store(1));
        assert!(store(2).contains("(load.buffer.a.#@v0 "), "{}", store(2));
        assert!(store(3).contains("(load.buffer.a.#@v1 "), "the store through p bumped a: {}", store(3));
        assert!(!main.roots.iter().any(|r| matches!(&r.kind, RootKind::Let { name } if name.starts_with("p@"))), "a pointer let is not a root");
        assert!(main.roots.iter().any(|r| matches!(&r.kind, RootKind::Let { name } if name.starts_with("before@"))));
    }

    #[test]
    fn a_pointer_let_whose_index_lineage_changes_before_a_use_is_refused() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var i = 0u;
    let p = &a[i];
    i = 1u;
    *p = 2.0;
}
"#;
        let e = match read(src) {
            Err(e) => e,
            Ok(_) => panic!("read accepted a pointer let used after a store to its index"),
        };
        assert!(e.contains("pointer let `p`") && e.contains("local.i@0@v0") && e.contains("local.i@0@v1"), "{e}");
    }

    #[test]
    fn a_let_use_after_a_store_carries_the_bindings_lineage() {
        // The step-5 fault: `let pos = q[head]; head += 1u;` read after the store.
        let src = r#"
@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    var head = 0u;
    let pos = q[head];
    head = head + 1u;
    out[0] = pos + q[head];
}
"#;
        let k = read(src).unwrap_or_else(|e| panic!("{e}"));
        let f = &k.functions[0];
        let binding = &f.roots[1];
        assert!(matches!(&binding.kind, RootKind::Let { name } if name.starts_with("pos@")));
        assert_eq!(binding.tree.lineage(), BTreeSet::from(["load.buffer.q.#".to_string(), "load.local.head@0@v0".to_string()]), "q is never stored: no version");
        let store = f.roots.iter().find(|r| matches!(r.kind, RootKind::Store { .. }) && r.tree.to_sexpr().starts_with("(store.buffer.out")).unwrap();
        let lineage = store.tree.lineage();
        assert!(lineage.iter().any(|n| n.starts_with("let.pos@")), "{lineage:?}");
        assert!(lineage.contains("load.local.head@0@v1"), "{lineage:?}");
        assert!(!lineage.contains("load.local.head@0@v0"), "the use carries the binding, not the text re-read: {lineage:?}");
    }
}
