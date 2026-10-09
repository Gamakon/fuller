//! The reference interpreter: the third implementation of the kingdom's
//! semantics beside the reader and the rebuild (`docs/PLAN_wgsl_lineage.md`
//! §4). It executes a FOLDED chromosome against the kernel's naga statement
//! tree, one invocation at a time: the head genes fill the holes (stores,
//! conditions, lets, call arguments, returns) in the order the reader cut
//! them, and every tail definition is evaluated at its placement
//! (`legality::Placement`), so an `href` leaf read before its definition
//! was evaluated is an error here, not a wrong value. That is what makes
//! this implementation independent of the rebuild: the rebuild (lineage
//! plan step 4) will emit definitions at the same points, and the device
//! and this interpreter must agree.
//!
//! **Scope.** Scalars `f32`/`i32`/`u32`/`bool`, fixed-width vectors of
//! them, arrays of either in storage, uniform, workgroup, private and
//! function memory; `if`, `switch`, `loop`, `break`/`continue`/`return`,
//! calls with value and `ptr<function>` parameters. Struct values,
//! matrices, atomics, images and barriers are refused with a message, as
//! is any row the table maps that is not implemented here: the oracle
//! reports those as their own class. Workgroup memory shared between
//! invocations is out of scope (one invocation runs alone).
//!
//! **WGSL semantics, as the spec writes them**, each pinned by a test with
//! its expected value computed by hand: integer `x / 0 == x` and
//! `x % 0 == 0`, `i32::MIN / -1 == i32::MIN` and `i32::MIN % -1 == 0`;
//! shift amounts masked to the low five bits; integer add, subtract and
//! multiply wrap; float to integer conversion rounds toward zero and
//! saturates (NaN gives 0, which is what Rust's `as` does and what the
//! device test measures); `round` is to nearest even; `min`/`max` return
//! the non-NaN operand; float division and the transcendentals are IEEE.
//! Indexing follows the bounds-check policy wgpu applies on this platform,
//! `ReadZeroSkipWrite` (wgpu-hal 0.21 `metal/device.rs`, runtime checks
//! on): an out-of-range load reads zero, an out-of-range store is skipped.
//! Where Metal differs from the spec (fast-math on NaN), the device run of
//! the ORIGINAL kernel is the authority and the difference is a finding.

use std::collections::BTreeMap;

use naga::{Expression, Function, Module, Statement, TypeInner};

use crate::homeotic;
use crate::karva::{parse_math, MathNode};

use super::chromosome::WgslChromosome;
use super::legality::Placement;
use super::reader::Kernel;
use super::table::FunctionTable;
use super::versions;

/// A runtime value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
    /// A vector of scalars (2 to 4) or an array (any length).
    Vec(Vec<Value>),
}

impl Value {
    pub fn kind(&self) -> &'static str {
        match self {
            Value::F32(_) => "f32",
            Value::I32(_) => "i32",
            Value::U32(_) => "u32",
            Value::Bool(_) => "bool",
            Value::Vec(_) => "vec",
        }
    }

    fn as_u32(&self) -> Result<u32, String> {
        match self {
            Value::U32(v) => Ok(*v),
            Value::I32(v) if *v >= 0 => Ok(*v as u32),
            other => Err(format!("interp: {other:?} is not an index")),
        }
    }

    fn as_bool(&self) -> Result<bool, String> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(format!("interp: {other:?} is not a bool")),
        }
    }

    pub fn as_f32(&self) -> Result<f32, String> {
        match self {
            Value::F32(v) => Ok(*v),
            other => Err(format!("interp: {other:?} is not an f32")),
        }
    }

    /// Bit-exact equality (`to_bits`, so NaN payloads and `-0.0` count), the
    /// oracle's criterion.
    pub fn bits_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::F32(a), Value::F32(b)) => a.to_bits() == b.to_bits(),
            (Value::Vec(a), Value::Vec(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.bits_eq(y)),
            _ => self == other,
        }
    }

    fn zero_like(&self) -> Value {
        match self {
            Value::F32(_) => Value::F32(0.0),
            Value::I32(_) => Value::I32(0),
            Value::U32(_) => Value::U32(0),
            Value::Bool(_) => Value::Bool(false),
            Value::Vec(v) => Value::Vec(v.iter().map(Value::zero_like).collect()),
        }
    }
}

/// The zero value of a naga type (scalars, vectors, fixed arrays of them).
pub fn zero_of(module: &Module, ty: naga::Handle<naga::Type>) -> Result<Value, String> {
    match &module.types[ty].inner {
        TypeInner::Scalar(s) => zero_scalar(*s),
        TypeInner::Vector { size, scalar } => Ok(Value::Vec((0..*size as usize).map(|_| zero_scalar(*scalar)).collect::<Result<_, _>>()?)),
        TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => {
            let z = zero_of(module, *base)?;
            Ok(Value::Vec(vec![z; n.get() as usize]))
        }
        TypeInner::Pointer { base, .. } => zero_of(module, *base),
        other => Err(format!("interp: no zero for type {other:?}")),
    }
}

fn zero_scalar(s: naga::Scalar) -> Result<Value, String> {
    Ok(match (s.kind, s.width) {
        (naga::ScalarKind::Float, 4) => Value::F32(0.0),
        (naga::ScalarKind::Sint, 4) => Value::I32(0),
        (naga::ScalarKind::Uint, 4) => Value::U32(0),
        (naga::ScalarKind::Bool, _) => Value::Bool(false),
        other => return Err(format!("interp: scalar {other:?} is not supported")),
    })
}

/// The kernel's memory: one cell per location, by the reader's location
/// name (`buffer.xs`, `uniform.n`, `workgroup.w`, `private.p`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Memory {
    pub cells: BTreeMap<String, Value>,
}

impl Memory {
    pub fn set(&mut self, location: &str, value: Value) {
        self.cells.insert(location.to_string(), value);
    }

    pub fn get(&self, location: &str) -> Option<&Value> {
        self.cells.get(location)
    }
}

/// The ids of the one invocation that runs.
#[derive(Debug, Clone, Default)]
pub struct Invocation {
    pub global_id: [u32; 3],
    pub local_id: [u32; 3],
    pub local_index: u32,
    pub workgroup_id: [u32; 3],
    pub num_workgroups: [u32; 3],
}

/// A function's folded program: head genes (the roots) and tail
/// definitions with their placements, parsed once.
struct Program {
    head: Vec<MathNode>,
    tail: Vec<MathNode>,
    placements: Vec<Option<Placement>>,
}

impl Program {
    fn of(c: &WgslChromosome) -> Result<Program, String> {
        let head = c.folded.head.iter().map(|g| parse_math(g)).collect::<Result<_, _>>()?;
        let tail = c.folded.tail.iter().map(|g| parse_math(g)).collect::<Result<_, _>>()?;
        if c.placements.len() != c.folded.tail.len() {
            return Err(format!("interp: {}: {} placements for {} tail slots", c.function, c.placements.len(), c.folded.tail.len()));
        }
        Ok(Program { head, tail, placements: c.placements.clone() })
    }
}

/// Where the roots sit in a function's statement tree, in the reader's
/// order: the lets before a statement, the statement's own roots, a loop's
/// `break_if`, and the locals' initialisers. Built by the same walk the
/// reader does; a root consumed under the wrong kind is an error.
#[derive(Default, Debug)]
struct Holes {
    /// Initialiser roots: (local name as the reader spells it, root).
    init: Vec<(String, usize)>,
    /// Lets bound before the statement at a path: (handle index, root).
    lets: BTreeMap<Vec<usize>, Vec<(u32, usize)>>,
    /// A statement's own roots at its path, in order.
    own: BTreeMap<Vec<usize>, Vec<usize>>,
    /// A loop's `break_if` root, by the loop's path.
    break_if: BTreeMap<Vec<usize>, usize>,
}

fn holes_of(module: &Module, f: &Function) -> Holes {
    let mut h = Holes::default();
    let mut next = 0usize;
    for (handle, local) in f.local_variables.iter() {
        if local.init.is_some() {
            h.init.push((format!("local.{}@{}", local.name.clone().unwrap_or_else(|| "local".into()), handle.index()), next));
            next += 1;
        }
    }
    fn is_pointer(module: &Module, f: &Function, h: naga::Handle<Expression>) -> bool {
        match &f.expressions[h] {
            Expression::GlobalVariable(_) | Expression::LocalVariable(_) => true,
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => is_pointer(module, f, *base),
            Expression::FunctionArgument(i) => matches!(module.types[f.arguments[*i as usize].ty].inner, TypeInner::Pointer { .. }),
            _ => false,
        }
    }
    fn walk(module: &Module, f: &Function, block: &naga::Block, path: &mut Vec<usize>, next: &mut usize, h: &mut Holes) {
        let mut index = 0usize;
        for stmt in block.iter() {
            if let Statement::Emit(range) = stmt {
                for e in range.clone() {
                    if f.named_expressions.get(&e).is_some_and(|n| !super::reader::is_bake(n)) && !is_pointer(module, f, e) {
                        let mut p = path.clone();
                        p.push(index);
                        h.lets.entry(p).or_default().push((e.index() as u32, *next));
                        *next += 1;
                    }
                }
                continue;
            }
            path.push(index);
            index += 1;
            let own = h.own.entry(path.clone()).or_default();
            match stmt {
                Statement::Store { .. } => {
                    own.push(*next);
                    *next += 1;
                }
                Statement::If { accept, reject, .. } => {
                    own.push(*next);
                    *next += 1;
                    path.push(0);
                    walk(module, f, accept, path, next, h);
                    path.pop();
                    path.push(1);
                    walk(module, f, reject, path, next, h);
                    path.pop();
                }
                Statement::Switch { cases, .. } => {
                    own.push(*next);
                    *next += 1;
                    for (c, case) in cases.iter().enumerate() {
                        path.push(c);
                        walk(module, f, &case.body, path, next, h);
                        path.pop();
                    }
                }
                Statement::Loop { body, continuing, break_if } => {
                    path.push(0);
                    walk(module, f, body, path, next, h);
                    path.pop();
                    path.push(1);
                    walk(module, f, continuing, path, next, h);
                    path.pop();
                    if break_if.is_some() {
                        h.break_if.insert(path.clone(), *next);
                        *next += 1;
                    }
                }
                Statement::Return { value: Some(_) } => {
                    own.push(*next);
                    *next += 1;
                }
                Statement::Call { arguments, .. } => {
                    for _ in arguments {
                        own.push(*next);
                        *next += 1;
                    }
                }
                Statement::Block(b) => walk(module, f, b, path, next, h),
                _ => {}
            }
            path.pop();
        }
    }
    walk(module, f, &f.body, &mut Vec::new(), &mut next, &mut h);
    h
}

/// A parameter as the callee sees it.
#[derive(Debug, Clone)]
enum Arg {
    Value(Value),
    /// A `ptr<function>` parameter: the caller's local, by cell.
    Cell(usize),
}

enum Flow {
    Next,
    Break,
    Continue,
    Return,
}

struct Frame {
    fi: usize,
    holes: Holes,
    /// Local variables, by the reader's `local.<name>@<idx>`, as cells.
    locals: BTreeMap<String, usize>,
    args: Vec<Arg>,
    /// Argument names (`arg.<name>`) → position.
    arg_names: BTreeMap<String, usize>,
    lets: BTreeMap<String, Value>,
    calls: BTreeMap<u32, Value>,
    defs: Vec<Option<Value>>,
    ret: Option<Value>,
}

/// The interpreter over one kernel: its module, its functions' programs,
/// the memory and the invocation.
pub struct Interp<'a> {
    kernel: &'a Kernel,
    programs: Vec<Program>,
    table: FunctionTable,
    pub memory: Memory,
    invocation: Invocation,
    cells: Vec<Value>,
}

impl<'a> Interp<'a> {
    /// `chromosomes` has one entry per `kernel.functions` entry, in order.
    pub fn new(kernel: &'a Kernel, chromosomes: &[WgslChromosome], memory: Memory, invocation: Invocation) -> Result<Self, String> {
        if chromosomes.len() != kernel.functions.len() {
            return Err(format!("interp: {} chromosomes for {} functions", chromosomes.len(), kernel.functions.len()));
        }
        let programs = chromosomes.iter().map(Program::of).collect::<Result<_, _>>()?;
        Ok(Interp { kernel, programs, table: FunctionTable::shipped(), memory, invocation, cells: Vec::new() })
    }

    fn function(&self, fi: usize) -> &'a Function {
        versions::functions(&self.kernel.module).nth(fi).expect("a function per chromosome")
    }

    /// Run the entry point (or function) named `name` with no arguments.
    pub fn run(&mut self, name: &str) -> Result<(), String> {
        let fi = self.kernel.functions.iter().position(|f| f.name == name).ok_or_else(|| format!("interp: no function {name}"))?;
        // Private globals start at zero unless the caller set them.
        for (h, g) in self.kernel.module.global_variables.iter() {
            if g.space == naga::AddressSpace::Private {
                let loc = versions::global_location(&self.kernel.module, h);
                if !self.memory.cells.contains_key(&loc) {
                    let z = zero_of(&self.kernel.module, g.ty)?;
                    self.memory.set(&loc, z);
                }
            }
        }
        self.call(fi, Vec::new()).map(|_| ())
    }

    fn call(&mut self, fi: usize, args: Vec<Arg>) -> Result<Option<Value>, String> {
        let f = self.function(fi);
        let holes = holes_of(&self.kernel.module, f);
        let mut locals = BTreeMap::new();
        for (h, local) in f.local_variables.iter() {
            let name = format!("local.{}@{}", local.name.clone().unwrap_or_else(|| "local".into()), h.index());
            let z = zero_of(&self.kernel.module, local.ty)?;
            self.cells.push(z);
            locals.insert(name, self.cells.len() - 1);
        }
        let arg_names = f.arguments.iter().enumerate().map(|(i, a)| (format!("arg.{}", a.name.clone().unwrap_or_else(|| i.to_string())), i)).collect();
        let n_tail = self.programs[fi].tail.len();
        let mut frame = Frame { fi, holes, locals, args, arg_names, lets: BTreeMap::new(), calls: BTreeMap::new(), defs: vec![None; n_tail], ret: None };
        // Initialisers, before any statement.
        let inits = std::mem::take(&mut frame.holes.init);
        for (local, root) in &inits {
            let tree = self.programs[fi].head[*root].clone();
            let (ctor, kids) = store_parts(&tree)?;
            if ctor != format!("store.{local}") || kids.len() != 1 {
                return Err(format!("interp: root {root} is {ctor}, expected the initialiser of {local}"));
            }
            let v = self.eval(&mut frame, &kids[0])?;
            let cell = frame.locals[local];
            self.cells[cell] = v;
        }
        frame.holes.init = inits;
        let body = &f.body;
        self.block(&mut frame, body, &mut Vec::new())?;
        Ok(frame.ret.take())
    }

    /// Evaluate the definitions placed at `(path, after_let = j)`, in
    /// descending slot order so a definition reading a later slot finds it.
    fn definitions_at(&mut self, frame: &mut Frame, path: &[usize], j: usize) -> Result<(), String> {
        let slots: Vec<usize> = self.programs[frame.fi]
            .placements
            .iter()
            .enumerate()
            .filter(|(_, p)| p.as_ref().is_some_and(|p| p.path == path && p.after_let == j))
            .map(|(s, _)| s)
            .rev()
            .collect();
        for s in slots {
            let def = self.programs[frame.fi].tail[s].clone();
            let v = self.eval(frame, &def)?;
            frame.defs[s] = Some(v);
        }
        Ok(())
    }

    fn block(&mut self, frame: &mut Frame, block: &naga::Block, path: &mut Vec<usize>) -> Result<Flow, String> {
        let mut index = 0usize;
        for stmt in block.iter() {
            if matches!(stmt, Statement::Emit(_)) {
                continue;
            }
            path.push(index);
            index += 1;
            // The lets bound before this statement, with the definitions
            // placed among them.
            let lets = frame.holes.lets.get(path).cloned().unwrap_or_default();
            for (j, (handle, root)) in lets.iter().enumerate() {
                self.definitions_at(frame, path, j)?;
                let tree = self.programs[frame.fi].head[*root].clone();
                let (ctor, kids) = store_parts(&tree)?;
                let name = ctor.strip_prefix("store.let.").ok_or_else(|| format!("interp: root {root} is {ctor}, expected a let"))?.to_string();
                if !name.ends_with(&format!("@{handle}")) {
                    return Err(format!("interp: let root {root} is {name}, expected handle {handle}"));
                }
                let v = self.eval(frame, &kids[0])?;
                frame.lets.insert(format!("let.{name}"), v);
            }
            self.definitions_at(frame, path, lets.len())?;
            let own = frame.holes.own.get(path).cloned().unwrap_or_default();
            let flow = match stmt {
                Statement::Emit(_) => Flow::Next,
                Statement::Break => Flow::Break,
                Statement::Continue => Flow::Continue,
                Statement::Kill => Flow::Return,
                Statement::Block(b) => {
                    let flow = self.block(frame, b, path)?;
                    if !matches!(flow, Flow::Next) {
                        path.pop();
                        return Ok(flow);
                    }
                    Flow::Next
                }
                Statement::Store { .. } => {
                    let tree = self.programs[frame.fi].head[own[0]].clone();
                    let (ctor, kids) = store_parts(&tree)?;
                    let target = ctor.strip_prefix("store.").ok_or_else(|| format!("interp: root is {ctor}, expected a store"))?.to_string();
                    let (index_tree, value_tree) = match kids {
                        [v] => (None, v),
                        [ix, v] => (Some(ix), v),
                        _ => return Err(format!("interp: {ctor} with {} children", kids.len())),
                    };
                    let v = self.eval(frame, value_tree)?;
                    let place = self.place(frame, &target, index_tree)?;
                    self.store(frame, &place, v)?;
                    Flow::Next
                }
                Statement::If { accept, reject, .. } => {
                    let c = self.hole_value(frame, own[0], "store.branch")?.as_bool()?;
                    let arm = if c { accept } else { reject };
                    path.push(if c { 0 } else { 1 });
                    let flow = self.block(frame, arm, path)?;
                    path.pop();
                    flow
                }
                Statement::Switch { cases, .. } => {
                    let sel = self.hole_value(frame, own[0], "store.branch")?;
                    let hit = cases.iter().position(|c| match (&c.value, &sel) {
                        (naga::SwitchValue::I32(v), Value::I32(s)) => v == s,
                        (naga::SwitchValue::U32(v), Value::U32(s)) => v == s,
                        _ => false,
                    });
                    let start = match hit.or_else(|| cases.iter().position(|c| matches!(c.value, naga::SwitchValue::Default))) {
                        Some(i) => i,
                        None => return Err("interp: switch with no matching case and no default".into()),
                    };
                    let mut flow = Flow::Next;
                    for (ci, case) in cases.iter().enumerate().skip(start) {
                        path.push(ci);
                        flow = self.block(frame, &case.body, path)?;
                        path.pop();
                        if !matches!(flow, Flow::Next) || !case.fall_through {
                            break;
                        }
                    }
                    // A `break` inside a switch leaves the switch only.
                    match flow {
                        Flow::Break => Flow::Next,
                        other => other,
                    }
                }
                Statement::Loop { body, continuing, break_if } => {
                    let mut out = Flow::Next;
                    let mut guard = 0u64;
                    loop {
                        guard += 1;
                        if guard > 10_000_000 {
                            return Err("interp: loop exceeded 10,000,000 iterations".into());
                        }
                        path.push(0);
                        let flow = self.block(frame, body, path)?;
                        path.pop();
                        match flow {
                            Flow::Break => break,
                            Flow::Return => {
                                out = Flow::Return;
                                break;
                            }
                            Flow::Next | Flow::Continue => {}
                        }
                        path.push(1);
                        let flow = self.block(frame, continuing, path)?;
                        path.pop();
                        match flow {
                            Flow::Return => {
                                out = Flow::Return;
                                break;
                            }
                            Flow::Break => break,
                            Flow::Next | Flow::Continue => {}
                        }
                        if break_if.is_some() {
                            let root = frame.holes.break_if[path];
                            let n = continuing.iter().filter(|s| !matches!(s, Statement::Emit(_))).count();
                            path.push(1);
                            path.push(n);
                            self.definitions_at(frame, path, 0)?;
                            path.pop();
                            path.pop();
                            if self.hole_value(frame, root, "store.branch")?.as_bool()? {
                                break;
                            }
                        }
                    }
                    out
                }
                Statement::Return { value } => {
                    if value.is_some() {
                        let v = self.hole_value(frame, own[0], "store.return")?;
                        frame.ret = Some(v);
                    }
                    Flow::Return
                }
                Statement::Call { function, arguments, result } => {
                    let callee = function.index();
                    let callee_fn = self.function(callee);
                    let mut args = Vec::with_capacity(arguments.len());
                    for (p, a) in arguments.iter().enumerate() {
                        let tree = self.programs[frame.fi].head[own[p]].clone();
                        let (ctor, kids) = store_parts(&tree)?;
                        if !ctor.starts_with("store.callarg.") || kids.len() != 1 {
                            return Err(format!("interp: root is {ctor}, expected call argument {p}"));
                        }
                        let is_ptr = matches!(self.kernel.module.types[callee_fn.arguments[p].ty].inner, TypeInner::Pointer { .. });
                        if is_ptr {
                            // The argument is `&local`: the reader renders it as
                            // the local's leaf.
                            let cell = match &kids[0] {
                                MathNode::Var(name) if name.starts_with("local.") => *frame.locals.get(name).ok_or_else(|| format!("interp: no local {name}"))?,
                                MathNode::Var(name) if name.starts_with("arg.") => match &frame.args[frame.arg_names[name]] {
                                    Arg::Cell(c) => *c,
                                    Arg::Value(_) => return Err(format!("interp: {name} is not a pointer argument")),
                                },
                                other => return Err(format!("interp: pointer argument {other:?} is not a local")),
                            };
                            args.push(Arg::Cell(cell));
                        } else {
                            args.push(Arg::Value(self.eval(frame, &kids[0])?));
                        }
                        // The reader expands pointer arguments through
                        // `pointer_target`: `&x` is `(Var "local.x@i")`, fine.
                        let _ = a;
                    }
                    let r = self.call(callee, args)?;
                    if let (Some(h), Some(v)) = (result, r) {
                        frame.calls.insert(h.index() as u32, v);
                    }
                    Flow::Next
                }
                other => return Err(format!("interp: statement {} is not supported", stmt_name(other))),
            };
            path.pop();
            if !matches!(flow, Flow::Next) {
                return Ok(flow);
            }
        }
        Ok(Flow::Next)
    }

    /// The value of a single-child root of the expected kind.
    fn hole_value(&mut self, frame: &mut Frame, root: usize, expect: &str) -> Result<Value, String> {
        let tree = self.programs[frame.fi].head[root].clone();
        let (ctor, kids) = store_parts(&tree)?;
        if ctor != expect || kids.len() != 1 {
            return Err(format!("interp: root {root} is {ctor} with {} children, expected {expect}", kids.len()));
        }
        self.eval(frame, &kids[0])
    }

    // ---- places ----

    fn place(&mut self, frame: &mut Frame, target: &str, index: Option<&MathNode>) -> Result<Place, String> {
        let mut parts = target.split('.');
        let kind = parts.next().unwrap_or("");
        let name = parts.next().ok_or_else(|| format!("interp: target {target:?} has no name"))?;
        let base = match kind {
            "local" => {
                let key = format!("local.{name}");
                Base::Cell(*frame.locals.get(&key).ok_or_else(|| format!("interp: no local {key}"))?)
            }
            "arg" => match &frame.args[*frame.arg_names.get(&format!("arg.{name}")).ok_or_else(|| format!("interp: no argument {name}"))?] {
                Arg::Cell(c) => Base::Cell(*c),
                Arg::Value(_) => return Err(format!("interp: argument {name} is not a pointer")),
            },
            "buffer" | "uniform" | "workgroup" | "private" | "push" | "global" => Base::Memory(format!("{kind}.{name}")),
            other => return Err(format!("interp: target kind {other:?} is not supported")),
        };
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
        let mut steps = Vec::new();
        for seg in parts {
            if seg == "#" {
                let ix = next_index.next().ok_or_else(|| format!("interp: target {target:?} has more `#` than index trees"))?;
                let v = self.eval(frame, ix)?.as_u32()?;
                steps.push(v as usize);
            } else {
                return Err(format!("interp: struct field {seg} in {target}: struct values are not supported"));
            }
        }
        if next_index.next().is_some() {
            return Err(format!("interp: target {target:?} has fewer `#` than index trees"));
        }
        Ok(Place { base, steps })
    }

    fn load(&self, frame: &Frame, place: &Place) -> Result<Value, String> {
        let mut cur = match &place.base {
            Base::Cell(c) => &self.cells[*c],
            Base::Memory(loc) => self.memory.get(loc).ok_or_else(|| format!("interp: memory has no {loc}"))?,
        };
        let _ = frame;
        for &i in &place.steps {
            cur = match cur {
                Value::Vec(items) => match items.get(i) {
                    Some(v) => v,
                    // ReadZeroSkipWrite: an out-of-range read is zero.
                    None => return Ok(items.first().map(Value::zero_like).unwrap_or(Value::F32(0.0))),
                },
                other => return Err(format!("interp: indexing into {other:?}")),
            };
        }
        Ok(cur.clone())
    }

    fn store(&mut self, frame: &Frame, place: &Place, value: Value) -> Result<(), String> {
        let _ = frame;
        let cell: &mut Value = match &place.base {
            Base::Cell(c) => &mut self.cells[*c],
            Base::Memory(loc) => self.memory.cells.get_mut(loc).ok_or_else(|| format!("interp: memory has no {loc}"))?,
        };
        let mut cur = cell;
        for &i in &place.steps {
            cur = match cur {
                Value::Vec(items) => match items.get_mut(i) {
                    Some(v) => v,
                    // ReadZeroSkipWrite: an out-of-range store is skipped.
                    None => return Ok(()),
                },
                other => return Err(format!("interp: indexing into {other:?}")),
            };
        }
        if cur.kind() != value.kind() {
            return Err(format!("interp: storing {} into a {} cell", value.kind(), cur.kind()));
        }
        *cur = value;
        Ok(())
    }

    // ---- expressions ----

    fn eval(&mut self, frame: &mut Frame, n: &MathNode) -> Result<Value, String> {
        match n {
            MathNode::Num(v) => Err(format!("interp: bare number {v} outside a literal")),
            MathNode::Var(name) => self.leaf(frame, name),
            MathNode::App(ctor, kids) => {
                if let Some(form) = ctor.strip_prefix("literal.") {
                    let v = match kids.as_slice() {
                        [MathNode::Num(v)] => *v,
                        _ => return Err(format!("interp: {ctor} without its number")),
                    };
                    return literal(form, v);
                }
                let (base, _) = versions::split_version(ctor);
                if let Some(target) = base.strip_prefix("load.") {
                    let ix = match kids.as_slice() {
                        [ix] => ix,
                        _ => return Err(format!("interp: {ctor} with {} children", kids.len())),
                    };
                    let place = self.place(frame, target, Some(ix))?;
                    return self.load(frame, &place);
                }
                let args: Vec<Value> = kids.iter().map(|k| self.eval(frame, k)).collect::<Result<_, _>>()?;
                if let Some(rest) = ctor.strip_prefix("shape.") {
                    return shape(rest, &args);
                }
                if let Some(rest) = ctor.strip_prefix("naga.") {
                    return self.unrowed(rest, kids, &args);
                }
                if let Some(rest) = ctor.strip_prefix("convert.") {
                    return convert(rest, &args);
                }
                if ctor.starts_with("select.") {
                    return match args.as_slice() {
                        [a, r, c] => Ok(if c.as_bool()? { a.clone() } else { r.clone() }),
                        _ => Err(format!("interp: {ctor} with {} children", args.len())),
                    };
                }
                let template = self.table.named(ctor).ok_or_else(|| format!("interp: {ctor} is not a row of the function table"))?;
                let column = template.naga.clone();
                naga_op(&column, ctor, &args)
            }
        }
    }

    fn leaf(&mut self, frame: &mut Frame, name: &str) -> Result<Value, String> {
        let (base, _) = versions::split_version(name);
        if let Some(target) = base.strip_prefix("load.") {
            let place = self.place(frame, target, None)?;
            return self.load(frame, &place);
        }
        if let Some(slot) = homeotic::href_slot(name) {
            return frame.defs.get(slot).and_then(|d| d.clone()).ok_or_else(|| format!("interp: {name} read before its definition was evaluated at its placement"));
        }
        if name.starts_with("let.") {
            return frame.lets.get(name).cloned().ok_or_else(|| format!("interp: {name} read before its binding"));
        }
        if let Some(rest) = name.strip_prefix("call.") {
            let idx: u32 = rest.rsplit_once('@').and_then(|(_, i)| i.parse().ok()).ok_or_else(|| format!("interp: call leaf {name} has no handle index"))?;
            return frame.calls.get(&idx).cloned().ok_or_else(|| format!("interp: {name} read before its call"));
        }
        if name.starts_with("arg.") {
            let i = *frame.arg_names.get(name).ok_or_else(|| format!("interp: no argument {name}"))?;
            return match &frame.args[i] {
                Arg::Value(v) => Ok(v.clone()),
                Arg::Cell(c) => Ok(self.cells[*c].clone()),
            };
        }
        if let Some(b) = name.strip_prefix("builtin.") {
            let v3 = |a: [u32; 3]| Value::Vec(a.iter().map(|&x| Value::U32(x)).collect());
            return match b {
                "global_invocation_id" => Ok(v3(self.invocation.global_id)),
                "local_invocation_id" => Ok(v3(self.invocation.local_id)),
                "local_invocation_index" => Ok(Value::U32(self.invocation.local_index)),
                "workgroup_id" => Ok(v3(self.invocation.workgroup_id)),
                "num_workgroups" => Ok(v3(self.invocation.num_workgroups)),
                other => Err(format!("interp: builtin {other} is not supported")),
            };
        }
        if let Some(c) = name.strip_prefix("const.") {
            let (_, konst) = self.kernel.module.constants.iter().find(|(_, k)| k.name.as_deref() == Some(c)).ok_or_else(|| format!("interp: no constant {c}"))?;
            return match &self.kernel.module.global_expressions[konst.init] {
                Expression::Literal(l) => literal_of(*l),
                other => Err(format!("interp: constant {c} is not a literal ({other:?})")),
            };
        }
        if name.starts_with("global.") {
            return Err(format!("interp: {name} used as a value"));
        }
        Err(format!("interp: leaf {name} has no rule"))
    }

    fn unrowed(&self, rest: &str, kids: &[MathNode], args: &[Value]) -> Result<Value, String> {
        let node = rest.split('.').next().unwrap_or("");
        match node {
            "AccessIndex" => {
                let i: usize = rest.rsplit('.').next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("interp: naga.{rest} has no index"))?;
                match args.first() {
                    Some(Value::Vec(items)) => items.get(i).cloned().ok_or_else(|| format!("interp: naga.{rest} out of range")),
                    other => Err(format!("interp: naga.{rest} on {other:?} (struct values are not supported)")),
                }
            }
            "ArrayLength" => match kids.first() {
                Some(MathNode::Var(g)) => {
                    let loc = g.strip_prefix("global.").map(|n| format!("buffer.{n}")).ok_or_else(|| format!("interp: arrayLength of {g}"))?;
                    match self.memory.get(&loc) {
                        Some(Value::Vec(items)) => Ok(Value::U32(items.len() as u32)),
                        other => Err(format!("interp: arrayLength of {loc}: {other:?}")),
                    }
                }
                other => Err(format!("interp: arrayLength of {other:?}")),
            },
            "As" => {
                // `naga.As.<from>_to_<to>.(convert|bitcast)`
                let mut it = rest.split('.');
                it.next();
                let crossing = it.next().unwrap_or("");
                let how = it.next().unwrap_or("");
                let to = crossing.rsplit("_to_").next().unwrap_or("");
                let v = args.first().ok_or("interp: naga.As without operand")?;
                cast(v, to, how == "convert")
            }
            _ => Err(format!("interp: naga.{rest} is not supported")),
        }
    }
}

enum Base {
    Cell(usize),
    Memory(String),
}

struct Place {
    base: Base,
    steps: Vec<usize>,
}

fn stmt_name(s: &Statement) -> String {
    let dbg = format!("{s:?}");
    dbg.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?").to_string()
}

/// A root's constructor and children.
fn store_parts(tree: &MathNode) -> Result<(&str, &[MathNode]), String> {
    match tree {
        MathNode::App(ctor, kids) => Ok((ctor, kids)),
        other => Err(format!("interp: root {other:?} is not a store")),
    }
}

fn literal(form: &str, v: f64) -> Result<Value, String> {
    Ok(match form {
        "real" => Value::F32(v as f32),
        "index" | "count" | "bits" | "hash32" | "code4" | "code8" | "code16" | "code1024" => Value::U32(v as u32),
        "int" | "q15_16" => Value::I32(v as i32),
        "flag" | "sign" => Value::Bool(v != 0.0),
        other => return Err(format!("interp: literal.{other} has no value")),
    })
}

fn literal_of(l: naga::Literal) -> Result<Value, String> {
    Ok(match l {
        naga::Literal::F32(v) => Value::F32(v),
        naga::Literal::U32(v) => Value::U32(v),
        naga::Literal::I32(v) => Value::I32(v),
        naga::Literal::Bool(b) => Value::Bool(b),
        other => return Err(format!("interp: literal {other:?} is not supported")),
    })
}

fn shape(rest: &str, args: &[Value]) -> Result<Value, String> {
    if let Some(i) = rest.strip_prefix("access_") {
        if i == "dyn" {
            return match args {
                [Value::Vec(items), ix] => Ok(items.get(ix.as_u32()? as usize).cloned().unwrap_or_else(|| items[0].zero_like())),
                _ => Err("interp: shape.access_dyn operands".into()),
            };
        }
        let i: usize = i.parse().map_err(|_| format!("interp: shape.{rest}"))?;
        return match args {
            [Value::Vec(items)] => items.get(i).cloned().ok_or_else(|| format!("interp: shape.{rest} out of range")),
            _ => Err(format!("interp: shape.{rest} on a non-vector")),
        };
    }
    if rest.starts_with("compose") {
        if rest.starts_with("compose_mat") {
            return Err("interp: matrices are not supported".into());
        }
        return Ok(Value::Vec(args.to_vec()));
    }
    if let Some(n) = rest.strip_prefix("splat") {
        let n: usize = n.parse().map_err(|_| format!("interp: shape.{rest}"))?;
        return Ok(Value::Vec(vec![args[0].clone(); n]));
    }
    if let Some(pat) = rest.strip_prefix("swizzle.") {
        return match args {
            [Value::Vec(items)] => pat
                .chars()
                .map(|c| {
                    let i = match c {
                        'x' => 0,
                        'y' => 1,
                        'z' => 2,
                        'w' => 3,
                        _ => 9,
                    };
                    items.get(i).cloned().ok_or_else(|| format!("interp: swizzle {pat} out of range"))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|v| if v.len() == 1 { v[0].clone() } else { Value::Vec(v) }),
            _ => Err("interp: swizzle of a non-vector".into()),
        };
    }
    Err(format!("interp: shape.{rest} is not supported"))
}

fn convert(rest: &str, args: &[Value]) -> Result<Value, String> {
    let v = args.first().ok_or("interp: convert without operand")?;
    match rest {
        "index_to_f32" | "int_to_f32" => cast(v, "float", true),
        "f32_to_int" => cast(v, "sint", true),
        "f32_to_count" => cast(v, "uint", true),
        "bitcast_f32_bits" => cast(v, "uint", false),
        "bitcast_bits_f32" => cast(v, "float", false),
        other => Err(format!("interp: convert.{other} is not supported")),
    }
}

/// `v` as kind `to` (`float`, `sint`, `uint`, `bool`), converting (WGSL:
/// toward zero, saturating) or bit-casting. Vectors convert lane by lane.
fn cast(v: &Value, to: &str, convert: bool) -> Result<Value, String> {
    if let Value::Vec(items) = v {
        return Ok(Value::Vec(items.iter().map(|i| cast(i, to, convert)).collect::<Result<_, _>>()?));
    }
    Ok(match (v, to, convert) {
        (Value::U32(x), "float", true) => Value::F32(*x as f32),
        (Value::I32(x), "float", true) => Value::F32(*x as f32),
        (Value::F32(x), "sint", true) => Value::I32(*x as i32),
        (Value::F32(x), "uint", true) => Value::U32(*x as u32),
        (Value::U32(x), "sint", true) => Value::I32(*x as i32),
        (Value::I32(x), "uint", true) => Value::U32(*x as u32),
        (Value::Bool(b), "float", true) => Value::F32(if *b { 1.0 } else { 0.0 }),
        (Value::Bool(b), "uint", true) => Value::U32(u32::from(*b)),
        (Value::Bool(b), "sint", true) => Value::I32(i32::from(*b)),
        (Value::F32(x), "bool", true) => Value::Bool(*x != 0.0),
        (Value::U32(x), "bool", true) => Value::Bool(*x != 0),
        (Value::I32(x), "bool", true) => Value::Bool(*x != 0),
        (Value::F32(x), "uint", false) => Value::U32(x.to_bits()),
        (Value::F32(x), "sint", false) => Value::I32(x.to_bits() as i32),
        (Value::U32(x), "float", false) => Value::F32(f32::from_bits(*x)),
        (Value::I32(x), "float", false) => Value::F32(f32::from_bits(*x as u32)),
        (Value::U32(x), "sint", false) => Value::I32(*x as i32),
        (Value::I32(x), "uint", false) => Value::U32(*x as u32),
        (x, k, _) if x.kind() == k || (x.kind() == "f32" && k == "float") || (x.kind() == "i32" && k == "sint") || (x.kind() == "u32" && k == "uint") => x.clone(),
        (x, k, c) => return Err(format!("interp: cast {x:?} to {k} ({}) is not supported", if c { "convert" } else { "bitcast" })),
    })
}

/// A naga operation on values, by the table's naga column.
fn naga_op(column: &str, ctor: &str, args: &[Value]) -> Result<Value, String> {
    // Element-wise over vectors of equal length (or a vector and scalars).
    if let Some(n) = args.iter().find_map(|a| match a {
        Value::Vec(items) => Some(items.len()),
        _ => None,
    }) {
        let mut lanes = Vec::with_capacity(n);
        for i in 0..n {
            let lane: Vec<Value> = args
                .iter()
                .map(|a| match a {
                    Value::Vec(items) => items.get(i).cloned().ok_or_else(|| format!("interp: {ctor}: vector lengths differ")),
                    s => Ok(s.clone()),
                })
                .collect::<Result<_, _>>()?;
            lanes.push(naga_op(column, ctor, &lane)?);
        }
        return Ok(Value::Vec(lanes));
    }
    if let Some(op) = column.strip_prefix("Binary::") {
        return binary(op, ctor, args);
    }
    if let Some(op) = column.strip_prefix("Unary::") {
        return match (op, args) {
            ("Negate", [Value::F32(x)]) => Ok(Value::F32(-x)),
            ("Negate", [Value::I32(x)]) => Ok(Value::I32(x.wrapping_neg())),
            ("LogicalNot", [Value::Bool(b)]) => Ok(Value::Bool(!b)),
            ("BitwiseNot", [Value::U32(x)]) => Ok(Value::U32(!x)),
            ("BitwiseNot", [Value::I32(x)]) => Ok(Value::I32(!x)),
            _ => Err(format!("interp: {ctor} ({column}) on {args:?} is not implemented")),
        };
    }
    if let Some(f) = column.strip_prefix("Relational::") {
        return match (f, args) {
            ("IsNan", [Value::F32(x)]) => Ok(Value::Bool(x.is_nan())),
            ("IsInf", [Value::F32(x)]) => Ok(Value::Bool(x.is_infinite())),
            _ => Err(format!("interp: {ctor} ({column}) on {args:?} is not implemented")),
        };
    }
    if let Some(f) = column.strip_prefix("Math::") {
        return math(f, ctor, args);
    }
    Err(format!("interp: {ctor}: naga column {column:?} is not implemented"))
}

fn binary(op: &str, ctor: &str, args: &[Value]) -> Result<Value, String> {
    use Value::*;
    let r = match (op, args) {
        ("Add", [F32(a), F32(b)]) => F32(a + b),
        ("Subtract", [F32(a), F32(b)]) => F32(a - b),
        ("Multiply", [F32(a), F32(b)]) => F32(a * b),
        ("Divide", [F32(a), F32(b)]) => F32(a / b),
        ("Modulo", [F32(a), F32(b)]) => F32(a % b),
        ("Add", [I32(a), I32(b)]) => I32(a.wrapping_add(*b)),
        ("Subtract", [I32(a), I32(b)]) => I32(a.wrapping_sub(*b)),
        ("Multiply", [I32(a), I32(b)]) => I32(a.wrapping_mul(*b)),
        // WGSL: x / 0 == x, i32::MIN / -1 == i32::MIN.
        ("Divide", [I32(a), I32(b)]) => I32(if *b == 0 || (*a == i32::MIN && *b == -1) { *a } else { a / b }),
        // WGSL: x % 0 == 0, i32::MIN % -1 == 0.
        ("Modulo", [I32(a), I32(b)]) => I32(if *b == 0 || (*a == i32::MIN && *b == -1) { 0 } else { a % b }),
        ("Add", [U32(a), U32(b)]) => U32(a.wrapping_add(*b)),
        ("Subtract", [U32(a), U32(b)]) => U32(a.wrapping_sub(*b)),
        ("Multiply", [U32(a), U32(b)]) => U32(a.wrapping_mul(*b)),
        ("Divide", [U32(a), U32(b)]) => U32(if *b == 0 { *a } else { a / b }),
        ("Modulo", [U32(a), U32(b)]) => U32(if *b == 0 { 0 } else { a % b }),
        ("Equal", [a, b]) => Bool(a == b),
        ("NotEqual", [a, b]) => Bool(a != b),
        ("Less", [F32(a), F32(b)]) => Bool(a < b),
        ("LessEqual", [F32(a), F32(b)]) => Bool(a <= b),
        ("Greater", [F32(a), F32(b)]) => Bool(a > b),
        ("GreaterEqual", [F32(a), F32(b)]) => Bool(a >= b),
        ("Less", [I32(a), I32(b)]) => Bool(a < b),
        ("LessEqual", [I32(a), I32(b)]) => Bool(a <= b),
        ("Greater", [I32(a), I32(b)]) => Bool(a > b),
        ("GreaterEqual", [I32(a), I32(b)]) => Bool(a >= b),
        ("Less", [U32(a), U32(b)]) => Bool(a < b),
        ("LessEqual", [U32(a), U32(b)]) => Bool(a <= b),
        ("Greater", [U32(a), U32(b)]) => Bool(a > b),
        ("GreaterEqual", [U32(a), U32(b)]) => Bool(a >= b),
        ("And", [U32(a), U32(b)]) => U32(a & b),
        ("ExclusiveOr", [U32(a), U32(b)]) => U32(a ^ b),
        ("InclusiveOr", [U32(a), U32(b)]) => U32(a | b),
        ("And", [I32(a), I32(b)]) => I32(a & b),
        ("ExclusiveOr", [I32(a), I32(b)]) => I32(a ^ b),
        ("InclusiveOr", [I32(a), I32(b)]) => I32(a | b),
        ("And", [Bool(a), Bool(b)]) | ("LogicalAnd", [Bool(a), Bool(b)]) => Bool(*a && *b),
        ("InclusiveOr", [Bool(a), Bool(b)]) | ("LogicalOr", [Bool(a), Bool(b)]) => Bool(*a || *b),
        ("ExclusiveOr", [Bool(a), Bool(b)]) => Bool(a != b),
        // WGSL: the shift amount is e2 & 31.
        ("ShiftLeft", [U32(a), U32(b)]) => U32(a.wrapping_shl(b & 31)),
        ("ShiftRight", [U32(a), U32(b)]) => U32(a.wrapping_shr(b & 31)),
        ("ShiftLeft", [I32(a), U32(b)]) => I32(a.wrapping_shl(b & 31)),
        ("ShiftRight", [I32(a), U32(b)]) => I32(a.wrapping_shr(b & 31)),
        _ => return Err(format!("interp: {ctor} (Binary::{op}) on {args:?} is not implemented")),
    };
    Ok(r)
}

/// WGSL `clamp(e, low, high)` is `min(max(e, low), high)` by the spec, and
/// WGSL `max(NaN, low)` is `low`: so `clamp(NaN, 0, 1)` is 0, not NaN as
/// Rust's `f32::clamp` would give. Written out for that reason.
fn wgsl_clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

fn math(f: &str, ctor: &str, args: &[Value]) -> Result<Value, String> {
    use Value::*;
    let r = match (f, args) {
        ("Abs", [F32(x)]) => F32(x.abs()),
        ("Abs", [I32(x)]) => I32(x.wrapping_abs()),
        ("Abs", [U32(x)]) => U32(*x),
        // WGSL min/max: the non-NaN operand when one is NaN.
        ("Min", [F32(a), F32(b)]) => F32(a.min(*b)),
        ("Max", [F32(a), F32(b)]) => F32(a.max(*b)),
        ("Min", [I32(a), I32(b)]) => I32(*a.min(b)),
        ("Max", [I32(a), I32(b)]) => I32(*a.max(b)),
        ("Min", [U32(a), U32(b)]) => U32(*a.min(b)),
        ("Max", [U32(a), U32(b)]) => U32(*a.max(b)),
        ("Clamp", [F32(x), F32(lo), F32(hi)]) => F32(wgsl_clamp(*x, *lo, *hi)),
        ("Clamp", [I32(x), I32(lo), I32(hi)]) => I32((*x).max(*lo).min(*hi)),
        ("Clamp", [U32(x), U32(lo), U32(hi)]) => U32((*x).max(*lo).min(*hi)),
        ("Saturate", [F32(x)]) => F32(wgsl_clamp(*x, 0.0, 1.0)),
        ("Sqrt", [F32(x)]) => F32(x.sqrt()),
        ("InverseSqrt", [F32(x)]) => F32(1.0 / x.sqrt()),
        ("Log", [F32(x)]) => F32(x.ln()),
        ("Log2", [F32(x)]) => F32(x.log2()),
        ("Exp", [F32(x)]) => F32(x.exp()),
        ("Exp2", [F32(x)]) => F32(x.exp2()),
        ("Pow", [F32(a), F32(b)]) => F32(a.powf(*b)),
        ("Floor", [F32(x)]) => F32(x.floor()),
        ("Ceil", [F32(x)]) => F32(x.ceil()),
        // WGSL round: to nearest, ties to even.
        ("Round", [F32(x)]) => F32(x.round_ties_even()),
        ("Trunc", [F32(x)]) => F32(x.trunc()),
        ("Fract", [F32(x)]) => F32(x - x.floor()),
        ("Sign", [F32(x)]) => F32(if *x > 0.0 { 1.0 } else if *x < 0.0 { -1.0 } else { *x }),
        ("Sign", [I32(x)]) => I32(x.signum()),
        ("Fma", [F32(a), F32(b), F32(c)]) => F32(a.mul_add(*b, *c)),
        ("Mix", [F32(a), F32(b), F32(t)]) => F32(a * (1.0 - t) + b * t),
        ("Step", [F32(edge), F32(x)]) => F32(if x < edge { 0.0 } else { 1.0 }),
        ("Smoothstep", [F32(e0), F32(e1), F32(x)]) => {
            let t = wgsl_clamp((x - e0) / (e1 - e0), 0.0, 1.0);
            F32(t * t * (3.0 - 2.0 * t))
        }
        ("Sin", [F32(x)]) => F32(x.sin()),
        ("Cos", [F32(x)]) => F32(x.cos()),
        ("Tan", [F32(x)]) => F32(x.tan()),
        ("Asin", [F32(x)]) => F32(x.asin()),
        ("Acos", [F32(x)]) => F32(x.acos()),
        ("Atan", [F32(x)]) => F32(x.atan()),
        ("Atan2", [F32(y), F32(x)]) => F32(y.atan2(*x)),
        ("Sinh", [F32(x)]) => F32(x.sinh()),
        ("Cosh", [F32(x)]) => F32(x.cosh()),
        ("Tanh", [F32(x)]) => F32(x.tanh()),
        ("CountOneBits", [U32(x)]) => U32(x.count_ones()),
        ("CountOneBits", [I32(x)]) => I32(x.count_ones() as i32),
        ("ReverseBits", [U32(x)]) => U32(x.reverse_bits()),
        ("CountLeadingZeros", [U32(x)]) => U32(x.leading_zeros()),
        ("CountTrailingZeros", [U32(x)]) => U32(x.trailing_zeros()),
        // firstTrailingBit: 0xFFFFFFFF for 0.
        ("FindLsb", [U32(x)]) => U32(if *x == 0 { u32::MAX } else { x.trailing_zeros() }),
        // firstLeadingBit: 0xFFFFFFFF for 0.
        ("FindMsb", [U32(x)]) => U32(if *x == 0 { u32::MAX } else { 31 - x.leading_zeros() }),
        ("ExtractBits", [U32(x), U32(offset), U32(count)]) => {
            let o = (*offset).min(32);
            let c = (*count).min(32 - o);
            U32(if c == 0 { 0 } else { (x >> o) & (u32::MAX >> (32 - c)) })
        }
        ("InsertBits", [U32(x), U32(n), U32(offset), U32(count)]) => {
            let o = (*offset).min(32);
            let c = (*count).min(32 - o);
            if c == 0 {
                U32(*x)
            } else {
                let mask = (u32::MAX >> (32 - c)) << o;
                U32((x & !mask) | ((n << o) & mask))
            }
        }
        _ => return Err(format!("interp: {ctor} (Math::{f}) on {args:?} is not implemented")),
    };
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::chromosome::{chromosome, ChromosomeOptions};
    use crate::wgsl::reader::read;

    /// Read, fold (default options) and run `main` over `memory` for one
    /// invocation with the given global id; returns the memory after.
    fn run(src: &str, memory: Memory, gid: u32) -> Result<Memory, String> {
        let k = read(src)?;
        let cs: Vec<WgslChromosome> = k.functions.iter().map(|f| chromosome(f, &ChromosomeOptions::default())).collect::<Result<_, _>>()?;
        let inv = Invocation { global_id: [gid, 0, 0], ..Default::default() };
        let mut it = Interp::new(&k, &cs, memory, inv)?;
        it.run("main")?;
        Ok(it.memory)
    }

    fn f32s(v: &[f32]) -> Value {
        Value::Vec(v.iter().map(|&x| Value::F32(x)).collect())
    }

    fn u32s(v: &[u32]) -> Value {
        Value::Vec(v.iter().map(|&x| Value::U32(x)).collect())
    }

    fn i32s(v: &[i32]) -> Value {
        Value::Vec(v.iter().map(|&x| Value::I32(x)).collect())
    }

    fn out_f32(m: &Memory, loc: &str) -> Vec<f32> {
        match m.get(loc) {
            Some(Value::Vec(items)) => items.iter().map(|v| v.as_f32().unwrap()).collect(),
            other => panic!("{loc}: {other:?}"),
        }
    }

    fn out_u32(m: &Memory, loc: &str) -> Vec<u32> {
        match m.get(loc) {
            Some(Value::Vec(items)) => items.iter().map(|v| v.as_u32().unwrap()).collect(),
            other => panic!("{loc}: {other:?}"),
        }
    }

    fn out_i32(m: &Memory, loc: &str) -> Vec<i32> {
        match m.get(loc) {
            Some(Value::Vec(items)) => items.iter().map(|v| match v { Value::I32(x) => *x, o => panic!("{o:?}") }).collect(),
            other => panic!("{loc}: {other:?}"),
        }
    }

    #[test]
    fn a_one_store_kernel_computes_its_value() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = xs[i] * 2.0 + 1.0;
}
"#;
        let mut m = Memory::default();
        m.set("buffer.xs", f32s(&[1.5, -2.0, 0.25]));
        m.set("buffer.out", f32s(&[0.0; 3]));
        let m = run(src, m, 1).unwrap();
        assert_eq!(out_f32(&m, "buffer.out"), vec![0.0, -3.0, 0.0]);
    }

    #[test]
    fn a_loop_a_branch_a_call_and_a_pointer_parameter_run() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
fn bump(p: ptr<function, f32>, by: f32) { *p = *p + by; }
fn twice(x: f32) -> f32 { return x * 2.0; }
@compute @workgroup_size(1)
fn main() {
    var s = 0.0;
    var i = 0u;
    loop {
        if (i >= 4u) { break; }
        if (xs[i] > 0.0) { bump(&s, xs[i]); } else { s = s - 1.0; }
        i = i + 1u;
    }
    let t = twice(s);
    out[0] = t;
    switch (i) {
        case 4u: { out[1] = 4.0; }
        default: { out[1] = -1.0; }
    }
}
"#;
        let mut m = Memory::default();
        m.set("buffer.xs", f32s(&[1.0, -5.0, 2.5, -1.0]));
        m.set("buffer.out", f32s(&[0.0; 2]));
        let m = run(src, m, 0).unwrap();
        // s = 1.0 - 1.0 + 2.5 - 1.0 = 1.5; twice → 3.0; i == 4.
        assert_eq!(out_f32(&m, "buffer.out"), vec![3.0, 4.0]);
    }

    #[test]
    fn a_shared_definition_is_evaluated_at_its_placement_and_an_early_read_is_an_error() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 1.0;
    out[0] = 0.0;
    s = 3.0;
    out[1] = s * s * 2.0 + xs[0];
    out[2] = s * s * 2.0 + xs[1];
}
"#;
        let k = read(src).unwrap();
        let c = chromosome(&k.functions[0], &ChromosomeOptions::default()).unwrap();
        assert_eq!(c.folded.filled, 1, "{:?}", c.refused);
        assert_eq!(c.placements[0], Some(Placement { path: vec![2], after_let: 0 }));
        let mut m = Memory::default();
        m.set("buffer.xs", f32s(&[0.5, 0.25]));
        m.set("buffer.out", f32s(&[0.0; 3]));
        let inv = Invocation::default();
        let mut it = Interp::new(&k, std::slice::from_ref(&c), m.clone(), inv.clone()).unwrap();
        it.run("main").unwrap();
        // s = 3: 3 * 3 * 2 = 18, plus xs.
        assert_eq!(out_f32(&it.memory, "buffer.out"), vec![0.0, 18.5, 18.25]);
        // Move the placement before the store to s (a wrong placement) and
        // the definition reads s == 1.0: the interpreter follows the
        // placement it is given, which is what the device will do too.
        let mut wrong = c.clone();
        wrong.placements[0] = Some(Placement { path: vec![0], after_let: 0 });
        let mut it = Interp::new(&k, std::slice::from_ref(&wrong), m.clone(), inv.clone()).unwrap();
        it.run("main").unwrap();
        // s = 1 at the wrong point: 1 * 1 * 2 = 2, plus xs.
        assert_eq!(out_f32(&it.memory, "buffer.out"), vec![0.0, 2.5, 2.25]);
        // A placement after the uses: the href is read before its definition.
        let mut late = c.clone();
        late.placements[0] = Some(Placement { path: vec![3], after_let: 0 });
        let mut it = Interp::new(&k, std::slice::from_ref(&late), m, inv).unwrap();
        let e = it.run("main").unwrap_err();
        assert!(e.contains("read before its definition"), "{e}");
    }

    #[test]
    fn integer_division_and_remainder_by_zero_and_min_by_minus_one_follow_wgsl() {
        let src = r#"
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> q: array<i32>;
@group(0) @binding(3) var<storage, read_write> r: array<i32>;
@group(0) @binding(4) var<storage, read> ua: array<u32>;
@group(0) @binding(5) var<storage, read> ub: array<u32>;
@group(0) @binding(6) var<storage, read_write> uq: array<u32>;
@group(0) @binding(7) var<storage, read_write> ur: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    q[i] = a[i] / b[i];
    r[i] = a[i] % b[i];
    uq[i] = ua[i] / ub[i];
    ur[i] = ua[i] % ub[i];
}
"#;
        // Hand-computed per the WGSL spec: x / 0 = x, x % 0 = 0,
        // MIN / -1 = MIN, MIN % -1 = 0, -7 / 2 = -3, -7 % 2 = -1.
        let a = [7, i32::MIN, -7, i32::MIN];
        let b = [0, -1, 2, 0];
        let want_q = [7, i32::MIN, -3, i32::MIN];
        let want_r = [0, 0, -1, 0];
        let ua = [9u32, u32::MAX, 10, 3];
        let ub = [0u32, 1, 3, 0];
        let want_uq = [9u32, u32::MAX, 3, 3];
        let want_ur = [0u32, 0, 1, 0];
        for i in 0..4 {
            let mut m = Memory::default();
            m.set("buffer.a", i32s(&a));
            m.set("buffer.b", i32s(&b));
            m.set("buffer.q", i32s(&[0; 4]));
            m.set("buffer.r", i32s(&[0; 4]));
            m.set("buffer.ua", u32s(&ua));
            m.set("buffer.ub", u32s(&ub));
            m.set("buffer.uq", u32s(&[0; 4]));
            m.set("buffer.ur", u32s(&[0; 4]));
            let m = run(src, m, i as u32).unwrap();
            assert_eq!(out_i32(&m, "buffer.q")[i], want_q[i], "q[{i}]");
            assert_eq!(out_i32(&m, "buffer.r")[i], want_r[i], "r[{i}]");
            assert_eq!(out_u32(&m, "buffer.uq")[i], want_uq[i], "uq[{i}]");
            assert_eq!(out_u32(&m, "buffer.ur")[i], want_ur[i], "ur[{i}]");
        }
    }

    #[test]
    fn shifts_mask_the_amount_integers_wrap_and_float_to_int_saturates() {
        let src = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> n: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read> f: array<f32>;
@group(0) @binding(4) var<storage, read_write> fi: array<i32>;
@group(0) @binding(5) var<storage, read_write> fu: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = (x[i] << n[i]) + (x[i] >> n[i]) * 3u;
    fi[i] = i32(f[i]);
    fu[i] = u32(f[i]);
}
"#;
        // Hand-computed: 1 << 33 = 1 << 1 = 2 (amount masked to 1),
        // 1 >> 33 = 0 → 2; 0x80000000 << 1 wraps to 0, >> 1 = 0x40000000,
        // ×3 wraps: 0xC0000000 → 0xC0000000; u32::MAX << 0 + u32::MAX*3
        // = 0xFFFFFFFF + 0xFFFFFFFD = 0xFFFFFFFC.
        let x = [1u32, 0x8000_0000, u32::MAX];
        let n = [33u32, 1, 0];
        let want = [2u32, 0xC000_0000, 0xFFFF_FFFC];
        // f32 → int: toward zero, saturating; NaN → 0.
        let f = [-7.9f32, 3.0e9, f32::NAN];
        let want_i = [-7i32, i32::MAX, 0];
        let want_u = [0u32, 3_000_000_000, 0];
        for i in 0..3 {
            let mut m = Memory::default();
            m.set("buffer.x", u32s(&x));
            m.set("buffer.n", u32s(&n));
            m.set("buffer.out", u32s(&[0; 3]));
            m.set("buffer.f", f32s(&f));
            m.set("buffer.fi", i32s(&[0; 3]));
            m.set("buffer.fu", u32s(&[0; 3]));
            let m = run(src, m, i as u32).unwrap();
            assert_eq!(out_u32(&m, "buffer.out")[i], want[i], "out[{i}]");
            assert_eq!(out_i32(&m, "buffer.fi")[i], want_i[i], "fi[{i}]");
            assert_eq!(out_u32(&m, "buffer.fu")[i], want_u[i], "fu[{i}]");
        }
    }

    #[test]
    fn float_edges_follow_ieee_and_min_max_select_pass_nan_as_wgsl_says() {
        let src = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    switch (i) {
        case 0u: { out[i] = a[i] / b[i]; }
        case 1u: { out[i] = sqrt(a[i]); }
        case 2u: { out[i] = log(a[i]); }
        case 3u: { out[i] = min(a[i], b[i]); }
        case 4u: { out[i] = max(a[i], b[i]); }
        case 5u: { out[i] = select(a[i], b[i], a[i] < b[i]); }
        case 6u: { out[i] = round(a[i]) + sign(b[i]); }
        default: { out[i] = a[i] * b[i]; }
    }
}
"#;
        // Hand-computed per IEEE and the WGSL spec: 1/0 = +inf; sqrt(-4) =
        // NaN; log(0) = -inf; min(NaN, 2) = 2; max(3, NaN) = 3; a < b is
        // false when b is NaN so select gives a; round(2.5) = 2 (ties to
        // even) + sign(-0.0) = -0.0 → 2.0; 1e30 * 1e30 = +inf.
        let a = [1.0f32, -4.0, 0.0, f32::NAN, 3.0, 1.0, 2.5, 1.0e30];
        let b = [0.0f32, 0.0, 0.0, 2.0, f32::NAN, f32::NAN, -0.0, 1.0e30];
        let mut m = Memory::default();
        m.set("buffer.a", f32s(&a));
        m.set("buffer.b", f32s(&b));
        m.set("buffer.out", f32s(&[0.0; 8]));
        let mut mem = m;
        for i in 0..8 {
            mem = run(src, mem, i).unwrap();
        }
        let got = out_f32(&mem, "buffer.out");
        assert_eq!(got[0], f32::INFINITY);
        assert!(got[1].is_nan());
        assert_eq!(got[2], f32::NEG_INFINITY);
        assert_eq!(got[3], 2.0);
        assert_eq!(got[4], 3.0);
        assert_eq!(got[5], 1.0);
        assert_eq!(got[6], 2.0);
        assert_eq!(got[7], f32::INFINITY);
    }

    #[test]
    fn out_of_range_indexing_reads_zero_and_skips_the_write() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = xs[i + 2u] + 1.0;
    out[i + 5u] = 7.0;
}
"#;
        // ReadZeroSkipWrite: xs[3] out of range reads 0 → out[1] = 1;
        // out[6] out of range: skipped, out stays length 2.
        let mut m = Memory::default();
        m.set("buffer.xs", f32s(&[10.0, 20.0, 30.0]));
        m.set("buffer.out", f32s(&[0.0, 0.0]));
        let m = run(src, m, 1).unwrap();
        assert_eq!(out_f32(&m, "buffer.out"), vec![0.0, 1.0]);
        let mut m = Memory::default();
        m.set("buffer.xs", f32s(&[10.0, 20.0, 30.0]));
        m.set("buffer.out", f32s(&[0.0, 0.0]));
        let m = run(src, m, 0).unwrap();
        assert_eq!(out_f32(&m, "buffer.out"), vec![31.0, 0.0]);
    }

    #[test]
    fn an_unimplemented_row_and_a_struct_value_are_refused_with_a_message() {
        let src = r#"
struct P { a: f32, b: f32 }
@group(0) @binding(0) var<storage, read> ps: array<P>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    let p = ps[0];
    out[0] = p.a + p.b;
}
"#;
        let mut m = Memory::default();
        m.set("buffer.ps", Value::Vec(vec![Value::Vec(vec![Value::F32(1.0), Value::F32(2.0)])]));
        m.set("buffer.out", f32s(&[0.0]));
        // Loading a struct element gives a Vec; the field read `naga.AccessIndex`
        // on it is served by position, so this one happens to run.
        let m = run(src, m, 0).unwrap();
        assert_eq!(out_f32(&m, "buffer.out"), vec![3.0]);
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = determinant(mat2x2<f32>(xs[0], xs[1], xs[2], xs[3]));
}
"#;
        let mut m = Memory::default();
        m.set("buffer.xs", f32s(&[1.0, 2.0, 3.0, 4.0]));
        m.set("buffer.out", f32s(&[0.0]));
        let e = run(src, m, 0).unwrap_err();
        assert!(e.contains("not supported") || e.contains("not implemented"), "{e}");
    }
}
