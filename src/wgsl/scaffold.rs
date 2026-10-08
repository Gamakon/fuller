//! The scaffold: a kernel's statements with one hole per root, and the
//! rebuild that fills the holes from gene text and gives naga a module again.
//!
//! The rebuild walks the ORIGINAL function's statements in the reader's
//! order and makes a new expression arena and body from scratch: every
//! original `Emit` is dropped; every other statement is copied with its
//! holes filled from the next root's tree. A hole's tree is appended in two
//! passes, as naga's validator requires: first its `needs_pre_emit` leaves
//! (literals, arguments, globals, locals, constants), then its computed
//! nodes in post-order as one contiguous range, emitted by one
//! `Statement::Emit` immediately before the holed statement (a `Loop`'s
//! `break_if` is emitted at the end of `continuing`, where naga scopes it).
//! A `Call` appends its `CallResult` after its argument emits, outside any
//! range, and a later `call.<fn>@<idx>` leaf resolves to it. Loads are
//! re-emitted at every site; no handle is ever borrowed across blocks.
//!
//! Three losses the gene text cannot carry are refused with a message rather
//! than papered over: an `f16` literal (the text says `literal.real`, which
//! rebuilds as `f32`), a `ZeroValue` of a composite (read as a scalar `0.0`),
//! and a name the rebuild has no rule for.
//!
//! The gate ([`round_trip`]) is structural: rebuild, validate, write WGSL,
//! parse and validate the text again, read it, and compare every function's
//! roots (kind, tree, Emit-free path) with the original read.

use std::collections::BTreeMap;

use naga::{
    front::Typifier, proc::ResolveContext, Arena, Block, Expression, Function, Handle, Module, Span, Statement, TypeInner,
};

use crate::karva::{parse_math, MathNode};

use super::loader::WgslKingdom;
use super::naga_names;
use super::reader::{read, slot_name, Kernel};
use super::table::FunctionTable;

/// A rebuilt kernel: the module, its validation info and the WGSL text.
pub struct Rebuilt {
    pub module: Module,
    pub info: naga::valid::ModuleInfo,
    pub wgsl: String,
}

/// Rebuild `kernel` with every function's roots replaced by the trees in
/// `roots` (one `Vec` per function, in `kernel.functions` order, each tree an
/// s-expression as the reader renders it, references unfolded).
pub fn rebuild(kernel: &Kernel, roots: &[Vec<String>]) -> Result<Rebuilt, String> {
    if roots.len() != kernel.functions.len() {
        return Err(format!("{} functions, {} root lists", kernel.functions.len(), roots.len()));
    }
    let table = FunctionTable::shipped();
    let kingdom = WgslKingdom::load();
    let mut module = kernel.module.clone();
    let n_fns = kernel.module.functions.len();
    let mut rebuilt_fns: Vec<Function> = Vec::new();
    for (i, f) in kernel.module.functions.iter().map(|(_, f)| f).chain(kernel.module.entry_points.iter().map(|ep| &ep.function)).enumerate() {
        let trees: Vec<MathNode> = roots[i].iter().map(|s| parse_math(s)).collect::<Result<_, _>>()?;
        let name = kernel.functions[i].name.clone();
        let mut b = Builder::new(&kernel.module, &table, &kingdom, f, &trees, &name);
        let rebuilt = b.function()?;
        rebuilt_fns.push(rebuilt);
    }
    let mut it = rebuilt_fns.into_iter();
    for (_, f) in module.functions.iter_mut() {
        *f = it.next().expect("one rebuilt function per original");
    }
    for ep in module.entry_points.iter_mut() {
        ep.function = it.next().expect("one rebuilt function per entry point");
    }
    debug_assert_eq!(n_fns + module.entry_points.len(), kernel.functions.len());
    let info = validate(&module)?;
    let wgsl = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty()).map_err(|e| format!("write: {e}"))?;
    Ok(Rebuilt { module, info, wgsl })
}

pub fn validate(module: &Module) -> Result<naga::valid::ModuleInfo, String> {
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(module)
        .map_err(|e| format!("validation: {}", e.emit_to_string("")))
}

/// The structural gate: rebuild from `roots`, write, parse and validate the
/// text, read it, and compare with `original` function by function. `Ok` is
/// `ROUND_TRIP ok`; `Err` names the first difference.
pub fn round_trip(original: &Kernel, roots: &[Vec<String>]) -> Result<Rebuilt, String> {
    let rebuilt = rebuild(original, roots)?;
    let again = read(&rebuilt.wgsl).map_err(|e| format!("the written text does not read back: {e}\n{}", rebuilt.wgsl))?;
    if again.functions.len() != original.functions.len() {
        return Err(format!("{} functions read back, {} originally", again.functions.len(), original.functions.len()));
    }
    for (i_fn, (a, b)) in original.functions.iter().zip(&again.functions).enumerate() {
        // The writer may rename a function (`mul32x32_64` came back as
        // `mul32x32_64_`); position is the identity, the name is recorded.
        if a.name != b.name && !b.name.starts_with(&a.name) {
            return Err(format!("function {} read back as {}", a.name, b.name));
        }
        if a.entry_point != b.entry_point {
            return Err(format!("function {} changed entry-point status", a.name));
        }
        if a.roots.len() != b.roots.len() {
            return Err(format!("{}: {} roots originally, {} after the round trip", a.name, a.roots.len(), b.roots.len()));
        }
        for (i, (ra, rb)) in a.roots.iter().zip(&b.roots).enumerate() {
            let (fa, fb) = (function_at(&original.module, i_fn), function_at(&again.module, i_fn));
            if kind_key(&ra.kind, &original.module, fa)? != kind_key(&rb.kind, &again.module, fb)? {
                return Err(format!("{} root {i}: kind {:?} became {:?}", a.name, ra.kind, rb.kind));
            }
            if ra.path != rb.path {
                return Err(format!("{} root {i}: path {:?} became {:?}", a.name, ra.path, rb.path));
            }
            let (sa, sb) = (canonical(&ra.tree.to_sexpr(), &original.module, fa)?, canonical(&rb.tree.to_sexpr(), &again.module, fb)?);
            if sa != sb {
                return Err(format!("{} root {i} differs after the round trip:\n  {sa}\n  {sb}", a.name));
            }
        }
    }
    Ok(rebuilt)
}

/// The `i`-th function as the reader numbers them: module functions, then
/// entry points.
fn function_at(module: &Module, i: usize) -> &Function {
    let n = module.functions.len();
    if i < n {
        module.functions.iter().nth(i).map(|(_, f)| f).expect("index within the arena")
    } else {
        &module.entry_points[i - n].function
    }
}

/// The member position of field `name` reached by `prefix` (a target's
/// steps from its base: `[kind, name, ...fields/#]`), or `None` when the
/// prefix does not resolve to a struct.
fn field_position(module: &Module, f: &Function, prefix: &[&str], name: &str) -> Option<usize> {
    let types = &module.types;
    let mut i = 0;
    let base_ty = loop {
        if i + 1 >= prefix.len() {
            return None;
        }
        let (kind, raw) = (prefix[i], prefix[i + 1]);
        let found = match kind {
            "buffer" | "uniform" | "workgroup" | "private" | "push" | "global" => module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(raw)).map(|(_, g)| g.ty),
            "local" => raw.rsplit_once('@').and_then(|(_, k)| k.parse::<usize>().ok()).and_then(|k| f.local_variables.iter().nth(k)).map(|(_, l)| l.ty),
            "arg" => f.arguments.iter().find(|a| a.name.as_deref() == Some(raw)).map(|a| a.ty),
            _ => None,
        };
        match found {
            Some(t) => {
                i += 2;
                break t;
            }
            None => i += 1,
        }
    };
    let mut inner = &types[base_ty].inner;
    for step in &prefix[i..] {
        inner = match (inner, *step) {
            (TypeInner::Pointer { base, .. }, _) => &types[*base].inner,
            _ => inner,
        };
        inner = match (inner, *step) {
            (TypeInner::Array { base, .. }, "#") => &types[*base].inner,
            (TypeInner::Struct { members, .. }, field) => {
                let m = members.iter().find(|m| m.name.as_deref() == Some(field))?;
                &types[m.ty].inner
            }
            _ => return None,
        };
    }
    if let TypeInner::Pointer { base, .. } = inner {
        inner = &types[*base].inner;
    }
    match inner {
        TypeInner::Struct { members, .. } => members.iter().position(|m| m.name.as_deref() == Some(name)),
        _ => None,
    }
}

/// A root kind with its names by position (see [`canonical`]).
fn kind_key(kind: &super::reader::RootKind, module: &Module, f: &Function) -> Result<String, String> {
    use super::reader::RootKind;
    let name = match kind {
        RootKind::Store { target } => format!("store:{}", canonical(&format!("(x.{target})"), module, f)?),
        RootKind::Init { local } => format!("init:{}", canonical(&format!("(x.{local})"), module, f)?),
        RootKind::Condition { statement } => format!("cond:{statement}"),
        RootKind::Return => "return".to_string(),
        RootKind::Argument { callee, position } => format!("arg:{}:{position}", canonical(&format!("(call.{callee})"), module, f)?),
    };
    Ok(name)
}

/// One rendering for both sides of a comparison: `{:?}` floats; call results
/// without their handle index (a rebuild renumbers); arguments and locals by
/// POSITION, because naga's writer renames a name that clashes elsewhere in
/// the module (`base` → `base_1`) and the position is what the kernel means.
fn canonical(sexpr: &str, module: &Module, f: &Function) -> Result<String, String> {
    let args: Vec<String> = f.arguments.iter().enumerate().map(|(i, a)| a.name.clone().unwrap_or_else(|| i.to_string())).collect();
    let locals: Vec<String> = f.local_variables.iter().enumerate().map(|(i, (_, l))| l.name.clone().unwrap_or_else(|| format!("local{i}"))).collect();
    let globals: Vec<String> = module.global_variables.iter().enumerate().map(|(i, (_, g))| g.name.clone().unwrap_or_else(|| format!("global{i}"))).collect();
    let functions: Vec<String> = module.functions.iter().enumerate().map(|(i, (_, g))| g.name.clone().unwrap_or_else(|| format!("fn{i}"))).collect();
    let constants: Vec<String> = module.constants.iter().enumerate().map(|(i, (_, c))| c.name.clone().unwrap_or_else(|| format!("const{i}"))).collect();
    let by_position = |name: &str| -> String {
        // Every named thing by its position in its arena: `arg.<n>`,
        // `local.<n>`, `buffer.<n>` (and the other global kinds),
        // `call.<fn>`, `store.arg.<fn>.<p>`.
        let parts: Vec<&str> = name.split('.').collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < parts.len() {
            let p = parts[i];
            let pool: Option<&Vec<String>> = match p {
                "arg" => Some(&args),
                "callarg" => Some(&functions),
                "local" => Some(&locals),
                "buffer" | "uniform" | "workgroup" | "private" | "push" | "global" => Some(&globals),
                "call" => Some(&functions),
                "const" => Some(&constants),
                _ => None,
            };
            // A struct field after a target: by member position (the writer
            // renames a field ending in a digit, `arg0` → `arg0_`).
            if i > 0 && pool.is_none() && p != "#" {
                if let Some(k) = field_position(module, f, &parts[..i], p) {
                    out.push(format!("#{k}"));
                    i += 1;
                    continue;
                }
            }
            match pool {
                Some(pool) if i + 1 < parts.len() => {
                    // `local.<name>@<index>` is already by position; the rest by lookup.
                    let (raw, at) = parts[i + 1].split_once('@').map(|(r, a)| (r, Some(a))).unwrap_or((parts[i + 1], None));
                    let idx = match (p, at) {
                        ("local", Some(k)) => format!("#{k}"),
                        _ => pool.iter().position(|n| n == raw).map(|k| format!("#{k}")).unwrap_or_else(|| raw.to_string()),
                    };
                    out.push(p.to_string());
                    out.push(idx);
                    i += 2;
                }
                _ => {
                    out.push(p.to_string());
                    i += 1;
                }
            }
        }
        out.join(".")
    };
    fn go(n: &MathNode, by_position: &dyn Fn(&str) -> String) -> String {
        match n {
            MathNode::Num(v) => format!("(Num {v:?})"),
            MathNode::Var(name) => format!("(Var {:?})", by_position(name)),
            MathNode::App(ctor, kids) => {
                let parts: Vec<String> = kids.iter().map(|k| go(k, by_position)).collect();
                let ctor = by_position(ctor);
                if parts.is_empty() {
                    format!("({ctor})")
                } else {
                    format!("({ctor} {})", parts.join(" "))
                }
            }
        }
    }
    Ok(go(&parse_math(sexpr)?, &by_position))
}

struct Builder<'a> {
    module: &'a Module,
    table: &'a FunctionTable,
    kingdom: &'a WgslKingdom,
    original: &'a Function,
    name: &'a str,
    trees: &'a [MathNode],
    next_root: usize,
    arena: Arena<Expression>,
    typifier: Typifier,
    /// Original call-result handle index → the rebuilt `CallResult`.
    call_results: BTreeMap<u32, Handle<Expression>>,
}

impl<'a> Builder<'a> {
    fn new(module: &'a Module, table: &'a FunctionTable, kingdom: &'a WgslKingdom, original: &'a Function, trees: &'a [MathNode], name: &'a str) -> Self {
        Builder { module, table, kingdom, original, name, trees, next_root: 0, arena: Arena::new(), typifier: Typifier::new(), call_results: BTreeMap::new() }
    }

    fn err<T>(&self, msg: impl std::fmt::Display) -> Result<T, String> {
        Err(format!("{}: {msg}", self.name))
    }

    fn take_root(&mut self, what: &str) -> Result<&'a MathNode, String> {
        let t = self.trees.get(self.next_root).ok_or_else(|| format!("{}: ran out of roots at {what} (root {})", self.name, self.next_root))?;
        self.next_root += 1;
        Ok(t)
    }

    fn append(&mut self, e: Expression) -> Result<Handle<Expression>, String> {
        let h = self.arena.append(e, Span::UNDEFINED);
        let ctx = ResolveContext::with_locals(self.module, &self.original.local_variables, &self.original.arguments);
        self.typifier.grow(h, &self.arena, &ctx).map_err(|e| format!("{}: type of expression {}: {e:?}", self.name, h.index()))?;
        Ok(h)
    }

    fn inner(&self, h: Handle<Expression>) -> &TypeInner {
        self.typifier.get(h, &self.module.types)
    }

    fn function(&mut self) -> Result<Function, String> {
        let mut f = Function {
            name: self.original.name.clone(),
            arguments: self.original.arguments.clone(),
            result: self.original.result.clone(),
            local_variables: self.original.local_variables.clone(),
            expressions: Arena::new(),
            named_expressions: Default::default(),
            body: Block::new(),
        };
        // Local initialisers first, as the reader produced them: a
        // constant leaf, appended outside any Emit (naga needs it const).
        let locals: Vec<(Handle<naga::LocalVariable>, bool)> = self.original.local_variables.iter().map(|(h, l)| (h, l.init.is_some())).collect();
        for (h, has_init) in locals {
            if has_init {
                let tree = self.take_root("a local initialiser")?;
                let (_, value) = self.unwrap_store(tree)?;
                let init = self.leaf_only(value)?;
                f.local_variables[h].init = Some(init);
            }
        }
        let body = self.block(&self.original.body)?;
        if self.next_root != self.trees.len() {
            return self.err(format!("{} roots given, {} used", self.trees.len(), self.next_root));
        }
        f.expressions = std::mem::replace(&mut self.arena, Arena::new());
        f.body = body;
        Ok(f)
    }

    /// `(store.<target> [index] value)` → (index tree, value tree).
    fn unwrap_store<'t>(&self, tree: &'t MathNode) -> Result<(Option<&'t MathNode>, &'t MathNode), String> {
        match tree {
            MathNode::App(ctor, kids) if ctor.starts_with("store.") => match kids.as_slice() {
                [value] => Ok((None, value)),
                [index, value] => Ok((Some(index), value)),
                _ => self.err(format!("{ctor} with {} children", kids.len())),
            },
            other => self.err(format!("a root must be a store node, got {}", render(other))),
        }
    }

    /// A local initialiser: built outside any Emit (naga scopes it as a
    /// constant expression, and rejects it if it is not one).
    fn leaf_only(&mut self, tree: &MathNode) -> Result<Handle<Expression>, String> {
        let mut leaves = BTreeMap::new();
        self.emit_leaves(tree, &mut leaves)?;
        self.build(tree, &leaves)
    }

    fn block(&mut self, original: &Block) -> Result<Block, String> {
        let mut out = Block::new();
        for stmt in original.iter() {
            match stmt {
                Statement::Emit(_) => {}
                Statement::Break => out.push(Statement::Break, Span::UNDEFINED),
                Statement::Continue => out.push(Statement::Continue, Span::UNDEFINED),
                Statement::Kill => out.push(Statement::Kill, Span::UNDEFINED),
                Statement::Barrier(b) => out.push(Statement::Barrier(*b), Span::UNDEFINED),
                Statement::Block(b) => {
                    let inner = self.block(b)?;
                    out.push(Statement::Block(inner), Span::UNDEFINED);
                }
                Statement::Store { .. } => {
                    let tree = self.take_root("a store")?;
                    let (index, value) = self.unwrap_store(tree)?;
                    let target = match tree {
                        MathNode::App(ctor, _) => ctor["store.".len()..].to_string(),
                        _ => unreachable!("unwrap_store accepted it"),
                    };
                    let mut leaves = BTreeMap::new();
                    self.emit_base(&target, &mut leaves)?;
                    if let Some(ix) = index {
                        self.emit_leaves(ix, &mut leaves)?;
                    }
                    self.emit_leaves(value, &mut leaves)?;
                    let start = self.arena.len();
                    let pointer = self.pointer(&target, index, &leaves)?;
                    let v = self.build(value, &leaves)?;
                    self.emit(&mut out, start);
                    out.push(Statement::Store { pointer, value: v }, Span::UNDEFINED);
                }
                Statement::If { accept, reject, .. } => {
                    let condition = self.hole(&mut out, "an if condition")?;
                    let accept = self.block(accept)?;
                    let reject = self.block(reject)?;
                    out.push(Statement::If { condition, accept, reject }, Span::UNDEFINED);
                }
                Statement::Switch { cases, .. } => {
                    let selector = self.hole(&mut out, "a switch selector")?;
                    let mut new_cases = Vec::with_capacity(cases.len());
                    for c in cases {
                        let body = self.block(&c.body)?;
                        new_cases.push(naga::SwitchCase { value: c.value, body, fall_through: c.fall_through });
                    }
                    out.push(Statement::Switch { selector, cases: new_cases }, Span::UNDEFINED);
                }
                Statement::Loop { body, continuing, break_if } => {
                    let body = self.block(body)?;
                    let mut continuing = self.block(continuing)?;
                    let break_if = match break_if {
                        Some(_) => Some(self.hole(&mut continuing, "a loop break_if")?),
                        None => None,
                    };
                    out.push(Statement::Loop { body, continuing, break_if }, Span::UNDEFINED);
                }
                Statement::Return { value } => {
                    let value = match value {
                        Some(_) => Some(self.hole(&mut out, "a return value")?),
                        None => None,
                    };
                    out.push(Statement::Return { value }, Span::UNDEFINED);
                }
                Statement::Call { function, arguments, result } => {
                    let mut args = Vec::with_capacity(arguments.len());
                    for _ in arguments {
                        args.push(self.hole(&mut out, "a call argument")?);
                    }
                    let result = match result {
                        Some(orig) => {
                            let h = self.append(Expression::CallResult(*function))?;
                            self.call_results.insert(orig.index() as u32, h);
                            Some(h)
                        }
                        None => None,
                    };
                    out.push(Statement::Call { function: *function, arguments: args, result }, Span::UNDEFINED);
                }
                other => return self.err(format!("statement {:?} is outside the kingdom", variant(other))),
            }
        }
        Ok(out)
    }

    /// Fill one single-value hole: emit the next root's value tree into
    /// `out` and return its handle.
    fn hole(&mut self, out: &mut Block, what: &str) -> Result<Handle<Expression>, String> {
        let tree = self.take_root(what)?;
        let (index, value) = self.unwrap_store(tree)?;
        if index.is_some() {
            return self.err(format!("{what} has an index child"));
        }
        let mut leaves = BTreeMap::new();
        self.emit_leaves(value, &mut leaves)?;
        let start = self.arena.len();
        let h = self.build(value, &leaves)?;
        self.emit(out, start);
        Ok(h)
    }

    fn emit(&mut self, out: &mut Block, start: usize) {
        if self.arena.len() > start {
            out.push(Statement::Emit(self.arena.range_from(start)), Span::UNDEFINED);
        }
    }

    // ---- leaves (pre-emit), appended before a hole's range ----

    fn emit_leaves(&mut self, n: &MathNode, leaves: &mut BTreeMap<String, Handle<Expression>>) -> Result<(), String> {
        match n {
            MathNode::Var(name) => {
                if name.starts_with("call.") {
                    return Ok(());
                }
                if let Some(target) = name.strip_prefix("load.") {
                    return self.emit_base(target, leaves);
                }
                if !leaves.contains_key(name) {
                    let h = self.leaf(name)?;
                    leaves.insert(name.clone(), h);
                }
            }
            MathNode::Num(_) => {}
            MathNode::App(ctor, kids) => {
                if let Some(form) = ctor.strip_prefix("literal.") {
                    let key = render(n);
                    if let std::collections::btree_map::Entry::Vacant(slot) = leaves.entry(key) {
                        let h = self.literal(form, kids)?;
                        slot.insert(h);
                    }
                } else {
                    if let Some(target) = ctor.strip_prefix("load.") {
                        self.emit_base(target, leaves)?;
                    }
                    for k in kids {
                        self.emit_leaves(k, leaves)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn literal(&mut self, form: &str, kids: &[MathNode]) -> Result<Handle<Expression>, String> {
        let v = match kids {
            [MathNode::Num(v)] => *v,
            _ => return self.err(format!("literal.{form} must hold one Num")),
        };
        let lit = match form {
            "real" => naga::Literal::F32(v as f32),
            "index" | "count" | "bits" | "hash32" | "code4" | "code8" | "code16" | "code1024" => naga::Literal::U32(v as u32),
            "int" | "q15_16" => naga::Literal::I32(v as i32),
            "flag" | "sign" => naga::Literal::Bool(v != 0.0),
            other => return self.err(format!("literal.{other} has no naga literal")),
        };
        self.append(Expression::Literal(lit))
    }

    fn leaf(&mut self, name: &str) -> Result<Handle<Expression>, String> {
        let (kind, rest) = name.split_once('.').ok_or_else(|| format!("{}: leaf {name:?} has no kind", self.name))?;
        match kind {
            "builtin" => {
                let want = naga_names::builtin(rest).ok_or_else(|| format!("{}: unknown builtin {rest}", self.name))?;
                let i = self
                    .original
                    .arguments
                    .iter()
                    .position(|a| matches!(a.binding, Some(naga::Binding::BuiltIn(b)) if b == want))
                    .ok_or_else(|| format!("{}: no argument carries builtin {rest}", self.name))?;
                self.append(Expression::FunctionArgument(i as u32))
            }
            "arg" => {
                let i = self.argument_index(rest)?;
                self.append(Expression::FunctionArgument(i))
            }
            "const" => {
                let (h, _) = self.module.constants.iter().find(|(_, c)| c.name.as_deref() == Some(rest)).ok_or_else(|| format!("{}: no constant {rest}", self.name))?;
                self.append(Expression::Constant(h))
            }
            "override" => {
                let (h, _) = self.module.overrides.iter().find(|(_, c)| c.name.as_deref() == Some(rest)).ok_or_else(|| format!("{}: no override {rest}", self.name))?;
                self.append(Expression::Override(h))
            }
            "global" | "buffer" | "uniform" | "workgroup" | "private" | "push" => {
                let h = self.global(rest)?;
                self.append(Expression::GlobalVariable(h))
            }
            "local" => {
                let h = self.local(rest)?;
                self.append(Expression::LocalVariable(h))
            }
            _ => self.err(format!("leaf {name:?} has no rule")),
        }
    }

    fn argument_index(&self, name: &str) -> Result<u32, String> {
        if let Ok(i) = name.parse::<u32>() {
            return Ok(i);
        }
        self.original
            .arguments
            .iter()
            .position(|a| a.name.as_deref() == Some(name))
            .map(|i| i as u32)
            .ok_or_else(|| format!("{}: no argument {name}", self.name))
    }

    fn global(&self, name: &str) -> Result<Handle<naga::GlobalVariable>, String> {
        self.module.global_variables.iter().find(|(_, g)| g.name.as_deref() == Some(name)).map(|(h, _)| h).ok_or_else(|| format!("{}: no global {name}", self.name))
    }

    /// A local by its `name@index` (the index selects; the name is checked).
    fn local(&self, name: &str) -> Result<Handle<naga::LocalVariable>, String> {
        let (_, idx) = name.rsplit_once('@').ok_or_else(|| format!("{}: local {name:?} carries no index", self.name))?;
        let idx: usize = idx.parse().map_err(|_| format!("{}: local {name:?} index", self.name))?;
        self.original.local_variables.iter().nth(idx).map(|(h, _)| h).ok_or_else(|| format!("{}: no local at index {idx} ({name})", self.name))
    }

    // ---- pointers ----

    /// Pre-emit the base of a pointer target (`kind.name`): a global, local
    /// or argument, which naga scopes without an Emit and so must sit
    /// outside every range.
    fn emit_base(&mut self, target: &str, leaves: &mut BTreeMap<String, Handle<Expression>>) -> Result<(), String> {
        let mut parts = target.split('.');
        let kind = parts.next().unwrap_or("");
        let name = parts.next().ok_or_else(|| format!("{}: target {target:?} has no name", self.name))?;
        let key = format!("base:{kind}.{name}");
        if leaves.contains_key(&key) {
            return Ok(());
        }
        let h = match kind {
            "buffer" | "uniform" | "workgroup" | "private" | "push" | "global" => {
                let g = self.global(name)?;
                self.append(Expression::GlobalVariable(g))?
            }
            "local" => {
                let l = self.local(name)?;
                self.append(Expression::LocalVariable(l))?
            }
            "arg" => {
                let i = self.argument_index(name)?;
                self.append(Expression::FunctionArgument(i))?
            }
            _ => return self.err(format!("target kind {kind:?} has no rule")),
        };
        leaves.insert(key, h);
        Ok(())
    }

    /// The pointer for a target `kind.name[.field…]` with an optional index
    /// tree (`a[i]`, or nested `naga.NestedAccess`).
    fn pointer(&mut self, target: &str, index: Option<&MathNode>, leaves: &BTreeMap<String, Handle<Expression>>) -> Result<Handle<Expression>, String> {
        let mut parts = target.split('.');
        let kind = parts.next().unwrap_or("");
        let name = parts.next().ok_or_else(|| format!("{}: target {target:?} has no name", self.name))?;
        let mut base = *leaves.get(&format!("base:{kind}.{name}")).ok_or_else(|| format!("{}: pointer base {kind}.{name} was not pre-emitted", self.name))?;
        // The index trees in application order: a `naga.NestedAccess` chain
        // is (outer, inner), flattened.
        let mut indices: Vec<&MathNode> = Vec::new();
        fn flatten<'t>(n: &'t MathNode, out: &mut Vec<&'t MathNode>) {
            match n {
                MathNode::App(ctor, kids) if ctor == "naga.NestedAccess" && kids.len() == 2 => {
                    flatten(&kids[0], out);
                    flatten(&kids[1], out);
                }
                other => out.push(other),
            }
        }
        if let Some(ix) = index {
            flatten(ix, &mut indices);
        }
        let mut next_index = indices.into_iter();
        for seg in parts {
            if seg == "#" {
                let ix = next_index.next().ok_or_else(|| format!("{}: target {target:?} has more `#` than index trees", self.name))?;
                base = self.indexed(base, ix, leaves)?;
            } else {
                base = self.field_access(base, seg)?;
            }
        }
        if next_index.next().is_some() {
            return self.err(format!("target {target:?} has fewer `#` than index trees"));
        }
        Ok(base)
    }

    fn indexed(&mut self, base: Handle<Expression>, ix: &MathNode, leaves: &BTreeMap<String, Handle<Expression>>) -> Result<Handle<Expression>, String> {
        match ix {
            // A constant index was read as a literal; rebuild it as naga had it,
            // a constant `AccessIndex`, so the text reads back the same way.
            MathNode::App(ctor, kids) if ctor == "literal.index" => match kids.as_slice() {
                [MathNode::Num(v)] => self.append(Expression::AccessIndex { base, index: *v as u32 }),
                _ => self.err("literal.index must hold one Num"),
            },
            other => {
                let i = self.build(other, leaves)?;
                self.append(Expression::Access { base, index: i })
            }
        }
    }

    /// `AccessIndex` on a struct (by member name) or a vector lane / array
    /// element (by number).
    fn field_access(&mut self, base: Handle<Expression>, field: &str) -> Result<Handle<Expression>, String> {
        if let Ok(n) = field.parse::<u32>() {
            return self.append(Expression::AccessIndex { base, index: n });
        }
        let inner = self.inner(base).clone();
        let through = match &inner {
            TypeInner::Pointer { base, .. } => self.module.types[*base].inner.clone(),
            other => other.clone(),
        };
        let index = match through {
            TypeInner::Struct { members, .. } => members.iter().position(|m| m.name.as_deref() == Some(field)),
            _ => None,
        }
        .ok_or_else(|| format!("{}: no field {field} on {}", self.name, slot_name(&self.module.types, &inner)))?;
        self.append(Expression::AccessIndex { base, index: index as u32 })
    }

    // ---- computed nodes, post-order ----

    fn build(&mut self, n: &MathNode, leaves: &BTreeMap<String, Handle<Expression>>) -> Result<Handle<Expression>, String> {
        match n {
            MathNode::Num(v) => self.err(format!("a bare number {v} outside a literal node")),
            MathNode::Var(name) => {
                if let Some(h) = leaves.get(name) {
                    return Ok(*h);
                }
                if let Some(rest) = name.strip_prefix("call.") {
                    let idx: u32 = rest.rsplit_once('@').and_then(|(_, i)| i.parse().ok()).ok_or_else(|| format!("{}: call leaf {name} has no handle index", self.name))?;
                    return self.call_results.get(&idx).copied().ok_or_else(|| format!("{}: call result {name} before its call", self.name));
                }
                if let Some(target) = name.strip_prefix("load.") {
                    let p = self.pointer(target, None, leaves)?;
                    return self.append(Expression::Load { pointer: p });
                }
                self.err(format!("leaf {name:?} was not pre-emitted"))
            }
            MathNode::App(ctor, kids) => {
                if ctor.starts_with("literal.") {
                    return leaves.get(&render(n)).copied().ok_or_else(|| format!("{}: literal {} was not pre-emitted", self.name, render(n)));
                }
                if let Some(target) = ctor.strip_prefix("load.") {
                    let ix = match kids.as_slice() {
                        [ix] => ix,
                        _ => return self.err(format!("{ctor} with {} children", kids.len())),
                    };
                    let p = self.pointer(target, Some(ix), leaves)?;
                    return self.append(Expression::Load { pointer: p });
                }
                if let Some(rest) = ctor.strip_prefix("naga.") {
                    return self.unrowed(rest, kids, leaves);
                }
                if let Some(rest) = ctor.strip_prefix("shape.") {
                    return self.shape(rest, kids, leaves);
                }
                // A table row: its naga column says which node.
                let template = self.table.named(ctor).ok_or_else(|| format!("{}: {ctor} is not a row of the function table", self.name))?.clone();
                let args: Vec<Handle<Expression>> = kids.iter().map(|k| self.build(k, leaves)).collect::<Result<_, _>>()?;
                self.node_of_column(&template.naga, ctor, &args)
            }
        }
    }

    fn node_of_column(&mut self, column: &str, ctor: &str, args: &[Handle<Expression>]) -> Result<Handle<Expression>, String> {
        let one = |b: &Self, n: usize| -> Result<(), String> { if args.len() == n { Ok(()) } else { b.err(format!("{ctor} takes {n} children, {} given", args.len())) } };
        if let Some(op) = column.strip_prefix("Binary::") {
            one(self, 2)?;
            let op = naga_names::binary_operator(op).ok_or_else(|| format!("{}: unknown binary operator {op}", self.name))?;
            return self.append(Expression::Binary { op, left: args[0], right: args[1] });
        }
        if let Some(op) = column.strip_prefix("Unary::") {
            one(self, 1)?;
            let op = naga_names::unary_operator(op).ok_or_else(|| format!("{}: unknown unary operator {op}", self.name))?;
            return self.append(Expression::Unary { op, expr: args[0] });
        }
        if let Some(fun) = column.strip_prefix("Relational::") {
            one(self, 1)?;
            let fun = naga_names::relational_function(fun).ok_or_else(|| format!("{}: unknown relational function {fun}", self.name))?;
            return self.append(Expression::Relational { fun, argument: args[0] });
        }
        if let Some(fun) = column.strip_prefix("Math::") {
            let fun = naga_names::math_function(fun).ok_or_else(|| format!("{}: unknown math function {fun}", self.name))?;
            if args.is_empty() || args.len() > 4 {
                return self.err(format!("{ctor} with {} children", args.len()));
            }
            return self.append(Expression::Math { fun, arg: args[0], arg1: args.get(1).copied(), arg2: args.get(2).copied(), arg3: args.get(3).copied() });
        }
        if column == "Select" {
            one(self, 3)?;
            return self.append(Expression::Select { condition: args[2], accept: args[0], reject: args[1] });
        }
        if column == "As" || column == "As(bitcast)" {
            one(self, 1)?;
            // The row's out dual says the target kind and width.
            let row = self.kingdom.rows.iter().find(|r| r.name == ctor).ok_or_else(|| format!("{}: {ctor} has no instantiated row", self.name))?;
            let out = self.kingdom.dual(row.output).ok_or_else(|| format!("{}: {ctor} output is not a dual", self.name))?;
            let (kind, width) = scalar_of(out.scalar().unwrap_or(""))?;
            let convert = if column == "As" { Some(width) } else { None };
            return self.append(Expression::As { expr: args[0], kind, convert });
        }
        self.err(format!("{ctor}: naga column {column:?} has no rebuild rule"))
    }

    fn shape(&mut self, rest: &str, kids: &[MathNode], leaves: &BTreeMap<String, Handle<Expression>>) -> Result<Handle<Expression>, String> {
        let args: Vec<Handle<Expression>> = kids.iter().map(|k| self.build(k, leaves)).collect::<Result<_, _>>()?;
        if let Some(n) = rest.strip_prefix("access_") {
            if n == "dyn" {
                return self.append(Expression::Access { base: args[0], index: args[1] });
            }
            let index: u32 = n.parse().map_err(|_| format!("{}: shape.access_{n}", self.name))?;
            return self.append(Expression::AccessIndex { base: args[0], index });
        }
        if let Some(n) = rest.strip_prefix("splat") {
            let size = vector_size(n.parse().map_err(|_| format!("{}: shape.splat{n}", self.name))?)?;
            return self.append(Expression::Splat { size, value: args[0] });
        }
        if let Some(pat) = rest.strip_prefix("swizzle.") {
            let size = vector_size(pat.len() as u32)?;
            let mut pattern = [naga::SwizzleComponent::X; 4];
            for (i, c) in pat.chars().enumerate() {
                pattern[i] = match c {
                    'x' => naga::SwizzleComponent::X,
                    'y' => naga::SwizzleComponent::Y,
                    'z' => naga::SwizzleComponent::Z,
                    'w' => naga::SwizzleComponent::W,
                    other => return self.err(format!("swizzle component {other}")),
                };
            }
            return self.append(Expression::Swizzle { size, vector: args[0], pattern });
        }
        if let Some(suffix) = rest.strip_prefix("compose") {
            // `compose2.vec2<f32>`, `compose_mat.mat2x3<f32>`: the type by slot.
            let slot = suffix.split_once('.').map(|(_, s)| s).ok_or_else(|| format!("{}: shape.{rest} carries no type", self.name))?;
            return self.compose_as(&args, |types, t| slot_name(types, &t.inner) == slot, slot);
        }
        self.err(format!("shape.{rest} has no rebuild rule"))
    }

    /// `Compose` of `args` into the module type `pick` selects (the one the
    /// original kernel composed; every type it used is in the module).
    fn compose_as(&mut self, args: &[Handle<Expression>], pick: impl Fn(&naga::UniqueArena<naga::Type>, &naga::Type) -> bool, what: &str) -> Result<Handle<Expression>, String> {
        let ty = self
            .module
            .types
            .iter()
            .find(|(_, t)| pick(&self.module.types, t))
            .map(|(h, _)| h)
            .ok_or_else(|| format!("{}: the module has no type {what}", self.name))?;
        self.append(Expression::Compose { ty, components: args.to_vec() })
    }

    /// `naga.<Variant>…` nodes the table has no row for.
    fn unrowed(&mut self, rest: &str, kids: &[MathNode], leaves: &BTreeMap<String, Handle<Expression>>) -> Result<Handle<Expression>, String> {
        let args: Vec<Handle<Expression>> = kids.iter().map(|k| self.build(k, leaves)).collect::<Result<_, _>>()?;
        if let Some(index) = rest.strip_prefix("AccessIndex.") {
            let index: u32 = index.parse().map_err(|_| format!("{}: naga.AccessIndex.{index} is not a member position", self.name))?;
            return self.append(Expression::AccessIndex { base: args[0], index });
        }
        if rest == "ArrayLength" {
            return self.append(Expression::ArrayLength(args[0]));
        }
        if let Some(type_ref) = rest.strip_prefix("Compose.type") {
            let idx: usize = type_ref.parse().map_err(|_| format!("{}: naga.Compose.type{type_ref}", self.name))?;
            let ty = self.module.types.iter().nth(idx).map(|(h, _)| h).ok_or_else(|| format!("{}: no type at index {idx}", self.name))?;
            return self.append(Expression::Compose { ty, components: args });
        }
        if let Some(cast) = rest.strip_prefix("As.") {
            // `<from>_to_<to>.<convert|bitcast>`
            let (kinds, mode) = cast.split_once('.').ok_or_else(|| format!("{}: naga.As.{cast}", self.name))?;
            let (_, to) = kinds.split_once("_to_").ok_or_else(|| format!("{}: naga.As.{cast}", self.name))?;
            let kind = match to {
                "float" => naga::ScalarKind::Float,
                "sint" => naga::ScalarKind::Sint,
                "uint" => naga::ScalarKind::Uint,
                "bool" => naga::ScalarKind::Bool,
                other => return self.err(format!("cast to {other}")),
            };
            let convert = if mode == "convert" { Some(4) } else { None };
            return self.append(Expression::As { expr: args[0], kind, convert });
        }
        let column = rest.replacen('.', "::", 1);
        self.node_of_column(&column, &format!("naga.{rest}"), &args)
    }
}

fn render(n: &MathNode) -> String {
    match n {
        MathNode::Num(v) => format!("(Num {v:?})"),
        MathNode::Var(name) => format!("(Var {name:?})"),
        MathNode::App(ctor, kids) => {
            let parts: Vec<String> = kids.iter().map(render).collect();
            if parts.is_empty() {
                format!("({ctor})")
            } else {
                format!("({ctor} {})", parts.join(" "))
            }
        }
    }
}

fn variant(s: &Statement) -> String {
    let dbg = format!("{s:?}");
    dbg.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?").to_string()
}

fn vector_size(n: u32) -> Result<naga::VectorSize, String> {
    Ok(match n {
        2 => naga::VectorSize::Bi,
        3 => naga::VectorSize::Tri,
        4 => naga::VectorSize::Quad,
        other => return Err(format!("a vector of {other} lanes")),
    })
}

fn scalar_of(slot: &str) -> Result<(naga::ScalarKind, u8), String> {
    Ok(match slot {
        "f32" => (naga::ScalarKind::Float, 4),
        "f16" => (naga::ScalarKind::Float, 2),
        "i32" => (naga::ScalarKind::Sint, 4),
        "u32" => (naga::ScalarKind::Uint, 4),
        "bool" => (naga::ScalarKind::Bool, 1),
        other => return Err(format!("scalar {other:?}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::chromosome::{chromosome, ChromosomeOptions};
    use crate::homeotic;

    fn roots_of(k: &Kernel) -> Vec<Vec<String>> {
        k.functions.iter().map(|f| f.roots.iter().map(|r| r.tree.to_sexpr()).collect()).collect()
    }

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
    fn a_one_store_kernel_rebuilds_validates_and_reads_back_equal() {
        let k = read(ONE_STORE).unwrap();
        let r = round_trip(&k, &roots_of(&k)).unwrap();
        assert!(r.wgsl.contains("fn main("), "{}", r.wgsl);
        assert_eq!(r.module.entry_points.len(), 1);
    }

    #[test]
    fn a_kernel_with_a_branch_a_loop_a_call_and_a_struct_uniform_round_trips() {
        let src = r#"
struct Params { n: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
fn twice(x: f32) -> f32 { return x * 2.0; }
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var s = 0.0;
    for (var i = 0u; i < p.n; i = i + 1u) {
        if (i > 1u) {
            s = s + twice(f32(i)) * p.scale;
        } else {
            s = s - 1.0;
        }
    }
    let v = vec2<f32>(s, 1.0);
    out[gid.x] = select(s, v.y, s > 10.0) + length(v);
}
"#;
        let k = read(src).unwrap();
        let r = round_trip(&k, &roots_of(&k)).unwrap();
        assert!(r.wgsl.contains("twice("), "{}", r.wgsl);
    }

    #[test]
    fn two_stores_sharing_a_subtree_fold_encode_decode_and_rebuild_equal() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * 2u + 1u;
    let v = xs[i] * xs[i];
    a[i] = v + 1.0;
    b[i] = v - 1.0;
}
"#;
        let k = read(src).unwrap();
        let c = chromosome(&k.functions[0], &ChromosomeOptions::default()).unwrap();
        assert!(c.folded.filled >= 1);
        // From the DECODED genes: decode, unfold, rebuild.
        let (_, genes) = c.genes.as_ref().unwrap();
        let decoded: Vec<String> = genes.iter().map(|(h, t)| crate::karva::karva_to_terms_generic(h, t, &c.pset).unwrap()).collect();
        let folded = homeotic::FoldedChromosome { head: decoded[..c.folded.head.len()].to_vec(), tail: decoded[c.folded.head.len()..].to_vec(), filled: c.folded.filled, skipped: c.folded.skipped };
        let roots = homeotic::unfold(&folded).unwrap();
        round_trip(&k, &[roots]).unwrap();
    }

    #[test]
    fn a_wrong_root_count_and_a_wrong_root_shape_are_errors() {
        let k = read(ONE_STORE).unwrap();
        let err = |r: Result<Rebuilt, String>| r.err().expect("an error");
        assert!(err(rebuild(&k, &[vec![]])).contains("ran out of roots"));
        assert!(err(rebuild(&k, &[vec!["(arith.add (Num 1.0) (Num 2.0))".into()]])).contains("must be a store node"));
    }

    #[test]
    fn naga_rejects_a_rebuilt_body_whose_emit_was_dropped() {
        let k = read(ONE_STORE).unwrap();
        let mut r = rebuild(&k, &roots_of(&k)).unwrap();
        let body = &mut r.module.entry_points[0].function.body;
        let n = body.len();
        // Every Emit is removed: the Store then uses expressions never brought into scope.
        let kept: Vec<Statement> = body.iter().filter(|s| !matches!(s, Statement::Emit(_))).cloned().collect();
        assert!(kept.len() < n);
        *body = Block::from_vec(kept);
        assert!(validate(&r.module).is_err(), "naga must reject expressions used outside any Emit");
    }

    #[test]
    fn naga_rejects_a_gene_that_borrows_an_operand_emitted_in_a_sibling_block() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 1u) {
        out[0] = f32(gid.x) * 2.0;
    } else {
        out[1] = f32(gid.x) * 3.0;
    }
}
"#;
        let k = read(src).unwrap();
        let mut r = rebuild(&k, &roots_of(&k)).unwrap();
        let body = &mut r.module.entry_points[0].function.body;
        let mut borrowed = false;
        for (stmt, _) in body.span_iter_mut() {
            if let Statement::If { accept, reject, .. } = stmt {
                let accept_value = accept.iter().find_map(|s| match s {
                    Statement::Store { value, .. } => Some(*value),
                    _ => None,
                });
                for (s, _) in reject.span_iter_mut() {
                    if let (Statement::Store { value, .. }, Some(v)) = (s, accept_value) {
                        *value = v;
                        borrowed = true;
                    }
                }
            }
        }
        assert!(borrowed, "the test kernel has an if with a store in each branch");
        assert!(validate(&r.module).is_err(), "naga must reject a handle emitted in the sibling block");
    }
}
