//! A CPU EXECUTOR OF A WHOLE DISPATCH (2026-10-14): every workgroup of a
//! small dispatch, its invocations run in lockstep between workgroup
//! barriers (each runs to its next barrier, then the next invocation; all
//! resume together), so workgroup memory is shared as on the device. It runs
//! the kernel's naga module directly, with WGSL's value semantics (integer
//! arithmetic wraps, `x / 0 == x`, `x % 0 == 0`, shift counts masked to five
//! bits, floats IEEE) and wgpu's bounds policy (`ReadZeroSkipWrite`: a load
//! past a buffer reads zero, a store past it is skipped). It records every
//! store and atomic to a storage buffer by invocation, so a test can
//! enumerate each invocation's writes (the exhaustive small-domain check of
//! the write-set proof, `safety.rs`) and compare outputs exactly (the
//! symmetric-half rule's H against its parent's).
//!
//! The interpreter beside it (`interp.rs`) runs one invocation of a folded
//! chromosome alone; this one runs the rendered text. Not here: images,
//! ray queries, subgroup operations (a kernel using one is refused by name),
//! f16, 64-bit values.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use naga::{AddressSpace, BinaryOperator, Block, Expression, Function, Handle, MathFunction, Module, ScalarKind, Statement, TypeInner};

/// A value.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    F(f32),
    U(u32),
    I(i32),
    B(bool),
    /// A vector, an array or a struct (its members in order).
    V(Vec<Val>),
    /// A matrix: its columns.
    M(Vec<Vec<f32>>),
    P(Ptr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Base {
    Global(Handle<naga::GlobalVariable>),
    /// A local of the frame with this id.
    Local(usize, Handle<naga::LocalVariable>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ptr {
    base: Base,
    path: Vec<u32>,
}

impl Val {
    fn u(&self) -> Result<u32, String> {
        match self {
            Val::U(v) => Ok(*v),
            Val::I(v) => Ok(*v as u32),
            Val::B(b) => Ok(u32::from(*b)),
            other => Err(format!("cpu: {other:?} is not an integer")),
        }
    }

    fn b(&self) -> Result<bool, String> {
        match self {
            Val::B(b) => Ok(*b),
            other => Err(format!("cpu: {other:?} is not a bool")),
        }
    }

    fn f(&self) -> Result<f32, String> {
        match self {
            Val::F(v) => Ok(*v),
            other => Err(format!("cpu: {other:?} is not an f32")),
        }
    }

    /// The value's 32-bit words (a buffer's contents).
    pub fn words(&self, out: &mut Vec<u32>) {
        match self {
            Val::F(v) => out.push(v.to_bits()),
            Val::U(v) => out.push(*v),
            Val::I(v) => out.push(*v as u32),
            Val::B(b) => out.push(u32::from(*b)),
            Val::V(v) => v.iter().for_each(|x| x.words(out)),
            Val::M(c) => c.iter().flatten().for_each(|x| out.push(x.to_bits())),
            Val::P(_) => {}
        }
    }
}

/// The zero value of a type.
pub fn zero(module: &Module, ty: Handle<naga::Type>) -> Result<Val, String> {
    Ok(match &module.types[ty].inner {
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => scalar_zero(s.kind)?,
        TypeInner::Vector { size, scalar } => Val::V(vec![scalar_zero(scalar.kind)?; *size as usize]),
        TypeInner::Matrix { columns, rows, .. } => Val::M(vec![vec![0.0; *rows as usize]; *columns as usize]),
        TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => Val::V(vec![zero(module, *base)?; n.get() as usize]),
        TypeInner::Array { size: naga::ArraySize::Dynamic, .. } => Val::V(Vec::new()),
        TypeInner::Struct { members, .. } => Val::V(members.iter().map(|m| zero(module, m.ty)).collect::<Result<_, _>>()?),
        other => return Err(format!("cpu: no zero of {other:?}")),
    })
}

fn scalar_zero(k: ScalarKind) -> Result<Val, String> {
    Ok(match k {
        ScalarKind::Float => Val::F(0.0),
        ScalarKind::Uint => Val::U(0),
        ScalarKind::Sint => Val::I(0),
        ScalarKind::Bool => Val::B(false),
        other => return Err(format!("cpu: no zero of {other:?}")),
    })
}

/// A buffer's value from its words: `elem` the array's element type.
pub fn from_words(module: &Module, elem: Handle<naga::Type>, words: &[u32]) -> Result<Val, String> {
    let z = zero(module, elem)?;
    let mut w = Vec::new();
    z.words(&mut w);
    let n = w.len().max(1);
    let fill = |words: &[u32], z: &Val| -> Val {
        fn go(z: &Val, it: &mut std::slice::Iter<u32>) -> Val {
            match z {
                Val::F(_) => Val::F(f32::from_bits(*it.next().unwrap_or(&0))),
                Val::U(_) => Val::U(*it.next().unwrap_or(&0)),
                Val::I(_) => Val::I(*it.next().unwrap_or(&0) as i32),
                Val::B(_) => Val::B(*it.next().unwrap_or(&0) != 0),
                Val::V(v) => Val::V(v.iter().map(|x| go(x, it)).collect()),
                Val::M(c) => Val::M(c.iter().map(|col| col.iter().map(|_| f32::from_bits(*it.next().unwrap_or(&0))).collect()).collect()),
                Val::P(p) => Val::P(p.clone()),
            }
        }
        go(z, &mut words.iter())
    };
    Ok(Val::V(words.chunks(n).map(|c| fill(c, &z)).collect()))
}

/// A write an invocation made to a storage buffer: the binding, the element
/// index path, and the invocation's global id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Write {
    pub binding: String,
    pub path: Vec<u32>,
    pub gid: [u32; 3],
}

/// The bindings' contents (by name, as values) and the writes made.
pub struct Dispatch {
    pub buffers: HashMap<String, Val>,
    pub writes: Vec<Write>,
}

enum Cont<'m> {
    Block { block: &'m Block, pc: usize },
    Loop { body: &'m Block, continuing: &'m Block, break_if: Option<Handle<Expression>>, in_continuing: bool },
    /// A switch: a break inside leaves it.
    Switch,
}

struct Frame<'m> {
    id: usize,
    func: &'m Function,
    args: Vec<Val>,
    vals: HashMap<Handle<Expression>, Val>,
    conts: Vec<Cont<'m>>,
    /// The caller's expression the call's result goes to.
    result: Option<Handle<Expression>>,
    /// The function's expressions only a short-circuit operator's right
    /// operand uses (`short_circuit_only`): skipped at their Emit and
    /// computed only when the operator needs its right side.
    lazy: Rc<HashSet<Handle<Expression>>>,
}

struct Inv<'m> {
    gid: [u32; 3],
    lid: [u32; 3],
    wid: [u32; 3],
    frames: Vec<Frame<'m>>,
    locals: HashMap<(usize, Handle<naga::LocalVariable>), Val>,
    privates: HashMap<Handle<naga::GlobalVariable>, Val>,
    done: bool,
    /// A subgroup operation this invocation waits at: its operation, its
    /// value, a lane argument, and the expression its result goes to.
    pending: Option<(Pend, Val, Option<u32>, Handle<Expression>)>,
}

#[derive(Clone, Copy, PartialEq)]
enum Pend {
    Ballot,
    Collective(naga::SubgroupOperation, naga::CollectiveOperation),
    BroadcastFirst,
    Broadcast,
    Shuffle,
    ShuffleDown,
    ShuffleUp,
    ShuffleXor,
}

/// The subgroup width the executor runs (lanes by local invocation index).
pub const SUBGROUP: u32 = 32;

fn lidx(inv: &Inv, wg: [u32; 3]) -> u32 {
    inv.lid[0] + wg[0] * (inv.lid[1] + wg[1] * inv.lid[2])
}

enum Status {
    /// At a subgroup operation: the workgroup's invocations meet there.
    Collective,
    Barrier,
    Done,
}

struct Machine<'m> {
    module: &'m Module,
    globals: HashMap<Handle<naga::GlobalVariable>, Val>,
    shared: HashMap<Handle<naga::GlobalVariable>, Val>,
    nwg: [u32; 3],
    wg: [u32; 3],
    writes: Vec<Write>,
    next_frame: usize,
    steps: u64,
    max_steps: u64,
    /// `short_circuit_only` per function, computed once a dispatch.
    lazy: HashMap<*const Function, Rc<HashSet<Handle<Expression>>>>,
}

/// Run `module`'s entry point over `workgroups` with the bindings' contents
/// (`buffers`, by binding name: storage arrays and the uniform struct as
/// values). `max_steps` caps the statements executed.
pub fn run(module: &Module, buffers: HashMap<String, Val>, workgroups: [u32; 3], max_steps: u64) -> Result<Dispatch, String> {
    let ep = module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Compute).ok_or("cpu: no compute entry point")?;
    run_on(module, ep, buffers, workgroups, max_steps)
}

/// [`run`] of the compute entry point named `entry` (a module of several
/// entry points: the harness's multi-entry dispatch).
pub fn run_entry(module: &Module, entry: &str, buffers: HashMap<String, Val>, workgroups: [u32; 3], max_steps: u64) -> Result<Dispatch, String> {
    let ep = module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Compute && e.name == entry).ok_or_else(|| format!("cpu: no compute entry point {entry}"))?;
    run_on(module, ep, buffers, workgroups, max_steps)
}

fn run_on(module: &Module, ep: &naga::EntryPoint, buffers: HashMap<String, Val>, workgroups: [u32; 3], max_steps: u64) -> Result<Dispatch, String> {
    let mut globals = HashMap::new();
    for (h, g) in module.global_variables.iter() {
        if g.binding.is_some() {
            let name = g.name.clone().unwrap_or_default();
            let v = match buffers.get(&name) {
                Some(v) => v.clone(),
                None => zero(module, g.ty)?,
            };
            globals.insert(h, v);
        }
    }
    let mut m = Machine { module, globals, shared: HashMap::new(), nwg: workgroups, wg: ep.workgroup_size, writes: Vec::new(), next_frame: 0, steps: 0, max_steps, lazy: HashMap::new() };
    for wz in 0..workgroups[2] {
        for wy in 0..workgroups[1] {
            for wx in 0..workgroups[0] {
                m.workgroup([wx, wy, wz], &ep.function)?;
            }
        }
    }
    let mut out = HashMap::new();
    for (h, g) in module.global_variables.iter() {
        if g.binding.is_some() {
            out.insert(g.name.clone().unwrap_or_default(), m.globals[&h].clone());
        }
    }
    Ok(Dispatch { buffers: out, writes: m.writes })
}

/// [`run`] over WGSL text.
pub fn run_text(source: &str, buffers: HashMap<String, Val>, workgroups: [u32; 3], max_steps: u64) -> Result<Dispatch, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    run(&module, buffers, workgroups, max_steps)
}

/// An expression's operands; None for one this executor cannot list (it
/// then defers nothing in the function).
fn operands(e: &Expression) -> Option<Vec<Handle<Expression>>> {
    use Expression as E;
    Some(match e {
        E::Literal(_) | E::Constant(_) | E::Override(_) | E::ZeroValue(_) | E::FunctionArgument(_) | E::GlobalVariable(_) | E::LocalVariable(_) | E::CallResult(_) | E::AtomicResult { .. } | E::WorkGroupUniformLoadResult { .. } | E::SubgroupBallotResult | E::SubgroupOperationResult { .. } => Vec::new(),
        E::Compose { components, .. } => components.clone(),
        E::Access { base, index } => vec![*base, *index],
        E::AccessIndex { base, .. } => vec![*base],
        E::Splat { value, .. } => vec![*value],
        E::Swizzle { vector, .. } => vec![*vector],
        E::Load { pointer } => vec![*pointer],
        E::Unary { expr, .. } | E::Derivative { expr, .. } | E::As { expr, .. } | E::ArrayLength(expr) => vec![*expr],
        E::Binary { left, right, .. } => vec![*left, *right],
        E::Select { condition, accept, reject } => vec![*condition, *accept, *reject],
        E::Relational { argument, .. } => vec![*argument],
        E::Math { arg, arg1, arg2, arg3, .. } => [Some(*arg), *arg1, *arg2, *arg3].into_iter().flatten().collect(),
        _ => return None,
    })
}

/// The expressions a block's statements name directly (not through Emit);
/// false when a statement is one this executor cannot list.
fn statement_uses(block: &Block, out: &mut HashSet<Handle<Expression>>) -> bool {
    for st in block.iter() {
        let ok = match st {
            Statement::Emit(_) | Statement::Break | Statement::Continue | Statement::Kill | Statement::Barrier(_) => true,
            Statement::Block(b) => statement_uses(b, out),
            Statement::If { condition, accept, reject } => {
                out.insert(*condition);
                statement_uses(accept, out) && statement_uses(reject, out)
            }
            Statement::Switch { selector, cases } => {
                out.insert(*selector);
                cases.iter().all(|c| statement_uses(&c.body, out))
            }
            Statement::Loop { body, continuing, break_if } => {
                out.extend(break_if.iter().copied());
                statement_uses(body, out) && statement_uses(continuing, out)
            }
            Statement::Return { value } => {
                out.extend(value.iter().copied());
                true
            }
            Statement::Store { pointer, value } => {
                out.extend([*pointer, *value]);
                true
            }
            Statement::Atomic { pointer, fun, value, result } => {
                out.extend([*pointer, *value, *result]);
                if let naga::AtomicFunction::Exchange { compare: Some(c) } = fun {
                    out.insert(*c);
                }
                true
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                out.extend([*pointer, *result]);
                true
            }
            Statement::Call { arguments, result, .. } => {
                out.extend(arguments.iter().copied());
                out.extend(result.iter().copied());
                true
            }
            Statement::SubgroupBallot { result, predicate } => {
                out.insert(*result);
                out.extend(predicate.iter().copied());
                true
            }
            Statement::SubgroupCollectiveOperation { argument, result, .. } => {
                out.extend([*argument, *result]);
                true
            }
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// THE EXPRESSIONS ONLY A SHORT-CIRCUIT OPERATOR'S RIGHT SIDE USES: for
/// every `||` and `&&` of the function, the expressions of its right
/// operand's tree whose every user is in that tree or the operator, named
/// by no `let` and used by no statement. naga emits them before the
/// operator; the CPU executor skips them at their Emit and computes them
/// only when the left side does not decide, as WGSL's short circuit does.
/// Empty when the function holds an expression or statement it cannot list.
fn short_circuit_only(func: &Function) -> HashSet<Handle<Expression>> {
    let mut users: HashMap<Handle<Expression>, Vec<Handle<Expression>>> = HashMap::new();
    for (h, e) in func.expressions.iter() {
        let Some(ops) = operands(e) else { return HashSet::new() };
        for o in ops {
            users.entry(o).or_default().push(h);
        }
    }
    let mut named_or_stated: HashSet<Handle<Expression>> = func.named_expressions.keys().copied().collect();
    if !statement_uses(&func.body, &mut named_or_stated) {
        return HashSet::new();
    }
    let mut lazy = HashSet::new();
    for (b, e) in func.expressions.iter() {
        let Expression::Binary { op: BinaryOperator::LogicalOr | BinaryOperator::LogicalAnd, right, .. } = e else { continue };
        let mut tree = HashSet::new();
        let mut stack = vec![*right];
        while let Some(h) = stack.pop() {
            if tree.insert(h) {
                stack.extend(operands(&func.expressions[h]).unwrap_or_default());
            }
        }
        for h in &tree {
            let only_here = users.get(h).is_none_or(|us| us.iter().all(|u| *u == b || tree.contains(u)));
            if only_here && !named_or_stated.contains(h) {
                lazy.insert(*h);
            }
        }
    }
    lazy
}

fn bin_u(op: BinaryOperator, a: u32, b: u32) -> Result<Val, String> {
    use BinaryOperator as B;
    Ok(match op {
        B::Add => Val::U(a.wrapping_add(b)),
        B::Subtract => Val::U(a.wrapping_sub(b)),
        B::Multiply => Val::U(a.wrapping_mul(b)),
        B::Divide => Val::U(if b == 0 { a } else { a / b }),
        B::Modulo => Val::U(if b == 0 { 0 } else { a % b }),
        B::And => Val::U(a & b),
        B::InclusiveOr => Val::U(a | b),
        B::ExclusiveOr => Val::U(a ^ b),
        B::ShiftLeft => Val::U(a << (b & 31)),
        B::ShiftRight => Val::U(a >> (b & 31)),
        B::Equal => Val::B(a == b),
        B::NotEqual => Val::B(a != b),
        B::Less => Val::B(a < b),
        B::LessEqual => Val::B(a <= b),
        B::Greater => Val::B(a > b),
        B::GreaterEqual => Val::B(a >= b),
        other => return Err(format!("cpu: u32 {other:?}")),
    })
}

fn bin_i(op: BinaryOperator, a: i32, b: i32, b_u: u32) -> Result<Val, String> {
    use BinaryOperator as B;
    Ok(match op {
        B::Add => Val::I(a.wrapping_add(b)),
        B::Subtract => Val::I(a.wrapping_sub(b)),
        B::Multiply => Val::I(a.wrapping_mul(b)),
        B::Divide => Val::I(if b == 0 || (a == i32::MIN && b == -1) { a } else { a / b }),
        B::Modulo => Val::I(if b == 0 || (a == i32::MIN && b == -1) { 0 } else { a % b }),
        B::And => Val::I(a & b),
        B::InclusiveOr => Val::I(a | b),
        B::ExclusiveOr => Val::I(a ^ b),
        B::ShiftLeft => Val::I(a.wrapping_shl(b_u & 31)),
        B::ShiftRight => Val::I(a >> (b_u & 31)),
        B::Equal => Val::B(a == b),
        B::NotEqual => Val::B(a != b),
        B::Less => Val::B(a < b),
        B::LessEqual => Val::B(a <= b),
        B::Greater => Val::B(a > b),
        B::GreaterEqual => Val::B(a >= b),
        other => return Err(format!("cpu: i32 {other:?}")),
    })
}

fn bin_f(op: BinaryOperator, a: f32, b: f32) -> Result<Val, String> {
    use BinaryOperator as B;
    Ok(match op {
        B::Add => Val::F(a + b),
        B::Subtract => Val::F(a - b),
        B::Multiply => Val::F(a * b),
        B::Divide => Val::F(a / b),
        B::Modulo => Val::F(a % b),
        B::Equal => Val::B(a == b),
        B::NotEqual => Val::B(a != b),
        B::Less => Val::B(a < b),
        B::LessEqual => Val::B(a <= b),
        B::Greater => Val::B(a > b),
        B::GreaterEqual => Val::B(a >= b),
        other => return Err(format!("cpu: f32 {other:?}")),
    })
}

fn binary(op: BinaryOperator, a: &Val, b: &Val) -> Result<Val, String> {
    use BinaryOperator as B;
    match (a, b) {
        (Val::U(x), Val::U(y)) => bin_u(op, *x, *y),
        (Val::I(x), Val::U(y)) => bin_i(op, *x, *y as i32, *y),
        (Val::I(x), Val::I(y)) => bin_i(op, *x, *y, *y as u32),
        (Val::F(x), Val::F(y)) => bin_f(op, *x, *y),
        (Val::B(x), Val::B(y)) => Ok(match op {
            B::LogicalAnd | B::And => Val::B(*x && *y),
            B::LogicalOr | B::InclusiveOr => Val::B(*x || *y),
            B::Equal => Val::B(x == y),
            B::NotEqual => Val::B(x != y),
            other => return Err(format!("cpu: bool {other:?}")),
        }),
        (Val::V(x), Val::V(y)) if x.len() == y.len() => Ok(Val::V(x.iter().zip(y).map(|(p, q)| binary(op, p, q)).collect::<Result<_, _>>()?)),
        (Val::V(x), s) if !matches!(s, Val::V(_) | Val::M(_)) => Ok(Val::V(x.iter().map(|p| binary(op, p, s)).collect::<Result<_, _>>()?)),
        (s, Val::V(y)) if !matches!(s, Val::V(_) | Val::M(_)) => Ok(Val::V(y.iter().map(|q| binary(op, s, q)).collect::<Result<_, _>>()?)),
        (Val::M(m), Val::V(v)) if op == B::Multiply => {
            let rows = m.first().map_or(0, Vec::len);
            let mut out = vec![0.0f32; rows];
            for (c, col) in m.iter().enumerate() {
                let s = v.get(c).ok_or("cpu: matrix times a short vector")?.f()?;
                for (r, x) in col.iter().enumerate() {
                    out[r] += x * s;
                }
            }
            Ok(Val::V(out.into_iter().map(Val::F).collect()))
        }
        (Val::V(v), Val::M(m)) if op == B::Multiply => Ok(Val::V(m.iter().map(|col| col.iter().zip(v).try_fold(0.0f32, |acc, (x, y)| Ok::<f32, String>(acc + x * y.f()?)).map(Val::F)).collect::<Result<_, _>>()?)),
        (Val::M(a), Val::M(b)) if op == B::Multiply => {
            let rows = a.first().map_or(0, Vec::len);
            Ok(Val::M(b.iter().map(|bc| (0..rows).map(|r| a.iter().zip(bc).map(|(ac, y)| ac[r] * y).sum()).collect()).collect()))
        }
        (Val::M(a), Val::M(b)) if matches!(op, B::Add | B::Subtract) => Ok(Val::M(a.iter().zip(b).map(|(x, y)| x.iter().zip(y).map(|(p, q)| if op == B::Add { p + q } else { p - q }).collect()).collect())),
        (Val::M(a), Val::F(s)) | (Val::F(s), Val::M(a)) if op == B::Multiply => Ok(Val::M(a.iter().map(|c| c.iter().map(|x| x * s).collect()).collect())),
        (x, y) => Err(format!("cpu: {op:?} of {x:?} and {y:?}")),
    }
}

fn round_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }
}

/// f32 to the bits of the nearest f16 (ties to even), as WGSL's pack2x16float.
fn f16_bits(x: f32) -> u32 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r;
    }
    let rem = man & 0x1fff;
    let mut r = ((e as u32) << 10) | (man >> 13);
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r
}

/// The f32 an f16's bits stand for.
fn f16_value(h: u32) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        0x1f => if man == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        e => sign * (1.0 + man / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

fn floats(v: &[Val]) -> Result<Vec<f32>, String> {
    v.iter().map(Val::f).collect()
}

fn det(m: &[Vec<f32>]) -> f32 {
    match m.len() {
        1 => m[0][0],
        2 => m[0][0] * m[1][1] - m[1][0] * m[0][1],
        n => (0..n)
            .map(|c| {
                let minor: Vec<Vec<f32>> = (0..n).filter(|&k| k != c).map(|k| m[k][1..].to_vec()).collect();
                let s = if c % 2 == 0 { 1.0 } else { -1.0 };
                s * m[c][0] * det(&minor)
            })
            .sum(),
    }
}

fn math1(fun: MathFunction, a: &Val) -> Result<Val, String> {
    use MathFunction as M;
    let unorm = |x: f32, s: f32| (x.clamp(0.0, 1.0) * s).round_ties_even() as u32;
    let snorm = |x: f32, s: f32| ((x.clamp(-1.0, 1.0) * s).round_ties_even() as i32) as u32;
    match (fun, a) {
        (M::Pack4x8unorm, Val::V(v)) => return Ok(Val::U(floats(v)?.iter().enumerate().fold(0, |acc, (i, x)| acc | (unorm(*x, 255.0) << (8 * i))))),
        (M::Pack4x8snorm, Val::V(v)) => return Ok(Val::U(floats(v)?.iter().enumerate().fold(0, |acc, (i, x)| acc | ((snorm(*x, 127.0) & 0xff) << (8 * i))))),
        (M::Pack2x16unorm, Val::V(v)) => return Ok(Val::U(floats(v)?.iter().enumerate().fold(0, |acc, (i, x)| acc | (unorm(*x, 65535.0) << (16 * i))))),
        (M::Pack2x16snorm, Val::V(v)) => return Ok(Val::U(floats(v)?.iter().enumerate().fold(0, |acc, (i, x)| acc | ((snorm(*x, 32767.0) & 0xffff) << (16 * i))))),
        (M::Pack2x16float, Val::V(v)) => return Ok(Val::U(floats(v)?.iter().enumerate().fold(0, |acc, (i, x)| acc | (f16_bits(*x) << (16 * i))))),
        (M::Unpack4x8unorm, Val::U(x)) => return Ok(Val::V((0..4).map(|i| Val::F(((x >> (8 * i)) & 0xff) as f32 / 255.0)).collect())),
        (M::Unpack4x8snorm, Val::U(x)) => return Ok(Val::V((0..4).map(|i| Val::F(((((x >> (8 * i)) & 0xff) as u8 as i8) as f32 / 127.0).max(-1.0))).collect())),
        (M::Unpack2x16unorm, Val::U(x)) => return Ok(Val::V((0..2).map(|i| Val::F(((x >> (16 * i)) & 0xffff) as f32 / 65535.0)).collect())),
        (M::Unpack2x16snorm, Val::U(x)) => return Ok(Val::V((0..2).map(|i| Val::F(((((x >> (16 * i)) & 0xffff) as u16 as i16) as f32 / 32767.0).max(-1.0))).collect())),
        (M::Unpack2x16float, Val::U(x)) => return Ok(Val::V((0..2).map(|i| Val::F(f16_value((x >> (16 * i)) & 0xffff))).collect())),
        (M::Determinant, Val::M(m)) => return Ok(Val::F(det(m))),
        (M::Transpose, Val::M(m)) => {
            let rows = m.first().map_or(0, Vec::len);
            return Ok(Val::M((0..rows).map(|r| m.iter().map(|c| c[r]).collect()).collect()));
        }
        (M::Modf, Val::F(x)) => return Ok(Val::V(vec![Val::F(x.fract()), Val::F(x.trunc())])),
        (M::Frexp, Val::F(x)) => {
            if *x == 0.0 || !x.is_finite() {
                return Ok(Val::V(vec![Val::F(*x), Val::I(0)]));
            }
            let e = x.abs().log2().floor() as i32 + 1;
            let mut f = x / 2f32.powi(e);
            let mut e = e;
            if f.abs() >= 1.0 {
                f /= 2.0;
                e += 1;
            }
            if f.abs() < 0.5 {
                f *= 2.0;
                e -= 1;
            }
            return Ok(Val::V(vec![Val::F(f), Val::I(e)]));
        }
        (M::Modf | M::Frexp, Val::V(v)) => {
            let parts: Vec<Val> = v.iter().map(|x| math1(fun, x)).collect::<Result<_, _>>()?;
            let pick = |i: usize| Val::V(parts.iter().map(|p| get_path_val(p, &[i as u32]).unwrap_or(Val::F(0.0))).collect());
            return Ok(Val::V(vec![pick(0), pick(1)]));
        }
        _ => {}
    }
    if let Val::V(v) = a {
        match fun {
            M::Length => return Ok(Val::F(v.iter().map(|x| x.f().map(|f| f * f)).sum::<Result<f32, _>>()?.sqrt())),
            M::Normalize => {
                let l = v.iter().map(|x| x.f().map(|f| f * f)).sum::<Result<f32, _>>()?.sqrt();
                return Ok(Val::V(v.iter().map(|x| x.f().map(|f| Val::F(f / l))).collect::<Result<_, _>>()?));
            }
            _ => return Ok(Val::V(v.iter().map(|x| math1(fun, x)).collect::<Result<_, _>>()?)),
        }
    }
    Ok(match (fun, a) {
        (M::Abs, Val::F(x)) => Val::F(x.abs()),
        (M::Abs, Val::I(x)) => Val::I(x.wrapping_abs()),
        (M::Abs, Val::U(x)) => Val::U(*x),
        (M::Sign, Val::F(x)) => Val::F(if *x > 0.0 { 1.0 } else if *x < 0.0 { -1.0 } else { 0.0 }),
        (M::Sign, Val::I(x)) => Val::I(x.signum()),
        (M::CountOneBits, Val::I(x)) => Val::I(x.count_ones() as i32),
        (M::CountLeadingZeros, Val::I(x)) => Val::I(x.leading_zeros() as i32),
        (M::CountTrailingZeros, Val::I(x)) => Val::I(x.trailing_zeros() as i32),
        (M::ReverseBits, Val::I(x)) => Val::I(x.reverse_bits()),
        (M::FindLsb, Val::I(x)) => Val::I(if *x == 0 { -1 } else { x.trailing_zeros() as i32 }),
        // firstLeadingBit of an i32: the most significant bit that differs from the sign.
        (M::FindMsb, Val::I(x)) => {
            let y = if *x < 0 { !*x } else { *x };
            Val::I(if y == 0 { -1 } else { 31 - y.leading_zeros() as i32 })
        }
        (M::CountOneBits, x) => Val::U(x.u()?.count_ones()),
        (M::CountLeadingZeros, x) => Val::U(x.u()?.leading_zeros()),
        (M::CountTrailingZeros, x) => Val::U(x.u()?.trailing_zeros()),
        (M::ReverseBits, Val::U(x)) => Val::U(x.reverse_bits()),
        (M::FindLsb, x) => {
            let v = x.u()?;
            Val::U(if v == 0 { u32::MAX } else { v.trailing_zeros() })
        }
        (M::FindMsb, Val::U(v)) => Val::U(if *v == 0 { u32::MAX } else { 31 - v.leading_zeros() }),
        (f, Val::F(x)) => Val::F(match f {
            M::Floor => x.floor(),
            M::Ceil => x.ceil(),
            M::Round => round_even(*x),
            M::Fract => x - x.floor(),
            M::Trunc => x.trunc(),
            M::Sqrt => x.sqrt(),
            M::InverseSqrt => 1.0 / x.sqrt(),
            M::Exp => x.exp(),
            M::Exp2 => x.exp2(),
            M::Log => x.ln(),
            M::Log2 => x.log2(),
            M::Sin => x.sin(),
            M::Cos => x.cos(),
            M::Tan => x.tan(),
            M::Sinh => x.sinh(),
            M::Cosh => x.cosh(),
            M::Tanh => x.tanh(),
            M::Asin => x.asin(),
            M::Acos => x.acos(),
            M::Atan => x.atan(),
            M::Asinh => x.asinh(),
            M::Acosh => x.acosh(),
            M::Atanh => x.atanh(),
            M::Saturate => x.clamp(0.0, 1.0),
            M::Radians => x.to_radians(),
            M::Degrees => x.to_degrees(),
            M::Length => x.abs(),
            M::Normalize => x.signum(),
            other => return Err(format!("cpu: {other:?} of a float")),
        }),
        (f, x) => return Err(format!("cpu: {f:?} of {x:?}")),
    })
}

fn min_max(fun: MathFunction, a: &Val, b: &Val) -> Result<Val, String> {
    let max = fun == MathFunction::Max;
    Ok(match (a, b) {
        (Val::U(x), Val::U(y)) => Val::U(if max { *x.max(y) } else { *x.min(y) }),
        (Val::I(x), Val::I(y)) => Val::I(if max { *x.max(y) } else { *x.min(y) }),
        (Val::F(x), Val::F(y)) => Val::F(if max { x.max(*y) } else { x.min(*y) }),
        (Val::V(x), Val::V(y)) => Val::V(x.iter().zip(y).map(|(p, q)| min_max(fun, p, q)).collect::<Result<_, _>>()?),
        (x, y) => return Err(format!("cpu: {fun:?} of {x:?}, {y:?}")),
    })
}

fn math(fun: MathFunction, args: &[Val]) -> Result<Val, String> {
    use MathFunction as M;
    match (fun, args) {
        (_, [a]) => math1(fun, a),
        (M::Min | M::Max, [a, b]) => min_max(fun, a, b),
        (M::Clamp, [x, lo, hi]) => min_max(M::Min, &min_max(M::Max, x, lo)?, hi),
        (M::Pow, [Val::F(a), Val::F(b)]) => Ok(Val::F(a.powf(*b))),
        (M::Atan2, [Val::F(a), Val::F(b)]) => Ok(Val::F(a.atan2(*b))),
        (M::Step, [Val::F(e), Val::F(x)]) => Ok(Val::F(if *x < *e { 0.0 } else { 1.0 })),
        (M::Ldexp, [Val::F(a), Val::I(e)]) => Ok(Val::F(a * 2f32.powi(*e))),
        (M::Dot, [Val::V(a), Val::V(b)]) => {
            let mut acc = binary(BinaryOperator::Multiply, &a[0], &b[0])?;
            for (x, y) in a.iter().zip(b).skip(1) {
                acc = binary(BinaryOperator::Add, &acc, &binary(BinaryOperator::Multiply, x, y)?)?;
            }
            Ok(acc)
        }
        (M::Distance, [a, b]) => math1(M::Length, &binary(BinaryOperator::Subtract, a, b)?),
        (M::Reflect, [e1, e2]) => {
            let d = math(M::Dot, &[e2.clone(), e1.clone()])?;
            let two_d = binary(BinaryOperator::Multiply, &Val::F(2.0), &d)?;
            binary(BinaryOperator::Subtract, e1, &binary(BinaryOperator::Multiply, &two_d, e2)?)
        }
        (M::FaceForward, [n, i, r]) => {
            let d = math(M::Dot, &[r.clone(), i.clone()])?.f()?;
            Ok(if d < 0.0 { n.clone() } else { unary(naga::UnaryOperator::Negate, n)? })
        }
        (M::Refract, [i, n, Val::F(eta)]) => {
            let d = math(M::Dot, &[n.clone(), i.clone()])?.f()?;
            let k = 1.0 - eta * eta * (1.0 - d * d);
            if k < 0.0 {
                return Ok(Val::V(vec![Val::F(0.0); match i { Val::V(v) => v.len(), _ => 1 }]));
            }
            let a = binary(BinaryOperator::Multiply, &Val::F(*eta), i)?;
            let b = binary(BinaryOperator::Multiply, &Val::F(eta * d + k.sqrt()), n)?;
            binary(BinaryOperator::Subtract, &a, &b)
        }
        (M::Outer, [Val::V(a), Val::V(b)]) => {
            let (a, b) = (floats(a)?, floats(b)?);
            Ok(Val::M(b.iter().map(|y| a.iter().map(|x| x * y).collect()).collect()))
        }
        (M::Cross, [Val::V(a), Val::V(b)]) if a.len() == 3 => {
            let (a, b): (Vec<f32>, Vec<f32>) = (a.iter().map(Val::f).collect::<Result<_, _>>()?, b.iter().map(Val::f).collect::<Result<_, _>>()?);
            Ok(Val::V(vec![Val::F(a[1] * b[2] - a[2] * b[1]), Val::F(a[2] * b[0] - a[0] * b[2]), Val::F(a[0] * b[1] - a[1] * b[0])]))
        }
        (M::Fma, [Val::F(a), Val::F(b), Val::F(c)]) => Ok(Val::F(a.mul_add(*b, *c))),
        (M::Fma | M::Mix | M::SmoothStep | M::Step | M::Pow | M::Atan2, [Val::V(_), ..]) => {
            let n = match &args[0] {
                Val::V(v) => v.len(),
                _ => 0,
            };
            let lane = |v: &Val, i: usize| match v {
                Val::V(x) => x[i].clone(),
                s => s.clone(),
            };
            Ok(Val::V((0..n).map(|i| math(fun, &args.iter().map(|a| lane(a, i)).collect::<Vec<_>>())).collect::<Result<_, _>>()?))
        }
        (M::Mix, [Val::F(a), Val::F(b), Val::F(t)]) => Ok(Val::F(a * (1.0 - t) + b * t)),
        (M::SmoothStep, [Val::F(e0), Val::F(e1), Val::F(x)]) => {
            let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
            Ok(Val::F(t * t * (3.0 - 2.0 * t)))
        }
        (M::ExtractBits, [e, Val::U(off), Val::U(cnt)]) => {
            let (o, c) = ((*off).min(32), (*cnt).min(32 - (*off).min(32)));
            Ok(match e {
                Val::U(x) => Val::U(if c == 0 { 0 } else { (x >> o) & (u32::MAX >> (32 - c)) }),
                Val::I(x) => Val::I(if c == 0 { 0 } else { (x << (32 - o - c)) >> (32 - c) }),
                other => return Err(format!("cpu: extractBits of {other:?}")),
            })
        }
        (M::InsertBits, [e, n, Val::U(off), Val::U(cnt)]) => {
            let (o, c) = ((*off).min(32), (*cnt).min(32 - (*off).min(32)));
            let mask = if c == 0 { 0 } else { (u32::MAX >> (32 - c)) << o };
            let v = (e.u()? & !mask) | ((n.u()? << o) & mask);
            Ok(if matches!(e, Val::I(_)) { Val::I(v as i32) } else { Val::U(v) })
        }
        (f, a) => Err(format!("cpu: {f:?} of {a:?}")),
    }
}

fn convert(v: &Val, kind: ScalarKind, bitcast: bool) -> Result<Val, String> {
    if let Val::V(x) = v {
        return Ok(Val::V(x.iter().map(|e| convert(e, kind, bitcast)).collect::<Result<_, _>>()?));
    }
    Ok(match (v, kind, bitcast) {
        (Val::F(x), ScalarKind::Uint, true) => Val::U(x.to_bits()),
        (Val::F(x), ScalarKind::Sint, true) => Val::I(x.to_bits() as i32),
        (Val::U(x), ScalarKind::Float, true) => Val::F(f32::from_bits(*x)),
        (Val::I(x), ScalarKind::Float, true) => Val::F(f32::from_bits(*x as u32)),
        (Val::U(x), ScalarKind::Sint, _) => Val::I(*x as i32),
        (Val::I(x), ScalarKind::Uint, _) => Val::U(*x as u32),
        (Val::U(x), ScalarKind::Uint, _) => Val::U(*x),
        (Val::I(x), ScalarKind::Sint, _) => Val::I(*x),
        (Val::F(x), ScalarKind::Float, _) => Val::F(*x),
        (Val::F(x), ScalarKind::Uint, false) => Val::U(if x.is_nan() { 0 } else { x.clamp(0.0, 4_294_967_040.0) as u32 }),
        (Val::F(x), ScalarKind::Sint, false) => Val::I(if x.is_nan() { 0 } else { x.clamp(-2_147_483_648.0, 2_147_483_520.0) as i32 }),
        (Val::U(x), ScalarKind::Float, false) => Val::F(*x as f32),
        (Val::I(x), ScalarKind::Float, false) => Val::F(*x as f32),
        (Val::B(b), ScalarKind::Uint, _) => Val::U(u32::from(*b)),
        (Val::B(b), ScalarKind::Sint, _) => Val::I(i32::from(*b)),
        (Val::B(b), ScalarKind::Float, _) => Val::F(f32::from(u8::from(*b))),
        (Val::U(x), ScalarKind::Bool, _) => Val::B(*x != 0),
        (Val::I(x), ScalarKind::Bool, _) => Val::B(*x != 0),
        (Val::F(x), ScalarKind::Bool, _) => Val::B(*x != 0.0),
        (Val::B(b), ScalarKind::Bool, _) => Val::B(*b),
        (x, k, _) => return Err(format!("cpu: {x:?} as {k:?}")),
    })
}

/// An element of a value at a path (a matrix's column; out of range: None).
fn get_path_val(v: &Val, path: &[u32]) -> Option<Val> {
    match (v, path.split_first()) {
        (_, None) => Some(v.clone()),
        (Val::M(cols), Some((c, rest))) => {
            let col = Val::V(cols.get(*c as usize)?.iter().map(|x| Val::F(*x)).collect());
            get_path_val(&col, rest)
        }
        (Val::V(x), Some((i, rest))) => get_path_val(x.get(*i as usize)?, rest),
        _ => None,
    }
}

/// Store at a path; false when the path leaves the value (skipped).
fn set_path(v: &mut Val, path: &[u32], new: Val) -> bool {
    match (v, path.split_first()) {
        (v, None) => {
            *v = new;
            true
        }
        (Val::M(cols), Some((c, rest))) => {
            let Some(col) = cols.get_mut(*c as usize) else { return false };
            match (rest, new) {
                ([], Val::V(nv)) => {
                    for (x, y) in col.iter_mut().zip(nv) {
                        if let Val::F(f) = y {
                            *x = f;
                        }
                    }
                    true
                }
                ([r], Val::F(f)) => col.get_mut(*r as usize).map(|x| *x = f).is_some(),
                _ => false,
            }
        }
        (Val::V(x), Some((i, rest))) => match x.get_mut(*i as usize) {
            Some(e) => set_path(e, rest, new),
            None => false,
        },
        _ => false,
    }
}

impl<'m> Machine<'m> {
    fn workgroup(&mut self, wid: [u32; 3], entry: &'m Function) -> Result<(), String> {
        // Workgroup memory, zero for each workgroup.
        self.shared.clear();
        for (h, g) in self.module.global_variables.iter() {
            if g.space == AddressSpace::WorkGroup {
                self.shared.insert(h, zero(self.module, g.ty)?);
            }
        }
        let [wx, wy, wz] = self.wg;
        let mut invs = Vec::new();
        for lz in 0..wz {
            for ly in 0..wy {
                for lx in 0..wx {
                    let lid = [lx, ly, lz];
                    let gid = [wid[0] * wx + lx, wid[1] * wy + ly, wid[2] * wz + lz];
                    let mut inv = Inv { gid, lid, wid, frames: Vec::new(), locals: HashMap::new(), privates: HashMap::new(), done: false, pending: None };
                    for (h, g) in self.module.global_variables.iter() {
                        if g.space == AddressSpace::Private {
                            let v = match g.init {
                                Some(e) => self.global_expr(e)?,
                                None => zero(self.module, g.ty)?,
                            };
                            inv.privates.insert(h, v);
                        }
                    }
                    let args = self.entry_args(entry, &inv)?;
                    self.push_frame(&mut inv, entry, args, None)?;
                    invs.push(inv);
                }
            }
        }
        // Each invocation to its next barrier or subgroup operation (or its
        // end), then all again; a subgroup operation is computed over its
        // subgroup's lanes once every invocation waits at one.
        loop {
            let (mut at_barrier, mut at_collective) = (0usize, 0usize);
            for inv in invs.iter_mut().filter(|i| !i.done) {
                match self.step_until_barrier(inv)? {
                    Status::Barrier => at_barrier += 1,
                    Status::Collective => at_collective += 1,
                    Status::Done => inv.done = true,
                }
            }
            if at_collective > 0 {
                if at_barrier > 0 {
                    return Err("cpu: a subgroup operation and a barrier met in one pass (non-uniform control)".into());
                }
                Self::collective(&mut invs, self.wg)?;
                continue;
            }
            if at_barrier == 0 {
                return Ok(());
            }
        }
    }

    fn entry_args(&self, f: &Function, inv: &Inv) -> Result<Vec<Val>, String> {
        let v3 = |a: [u32; 3]| Val::V(a.iter().map(|x| Val::U(*x)).collect());
        f.arguments
            .iter()
            .map(|a| match &a.binding {
                Some(naga::Binding::BuiltIn(b)) => Ok(match b {
                    naga::BuiltIn::GlobalInvocationId => v3(inv.gid),
                    naga::BuiltIn::LocalInvocationId => v3(inv.lid),
                    naga::BuiltIn::WorkGroupId => v3(inv.wid),
                    naga::BuiltIn::NumWorkGroups => v3(self.nwg),
                    naga::BuiltIn::WorkGroupSize => v3(self.wg),
                    naga::BuiltIn::LocalInvocationIndex => Val::U(lidx(inv, self.wg)),
                    // Subgroups of 32 by local index (the M3 Max's SIMD width).
                    naga::BuiltIn::SubgroupSize => Val::U(SUBGROUP),
                    naga::BuiltIn::SubgroupInvocationId => Val::U(lidx(inv, self.wg) % SUBGROUP),
                    naga::BuiltIn::SubgroupId => Val::U(lidx(inv, self.wg) / SUBGROUP),
                    naga::BuiltIn::NumSubgroups => Val::U((self.wg[0] * self.wg[1] * self.wg[2]).div_ceil(SUBGROUP)),
                    other => return Err(format!("cpu: the builtin {other:?}")),
                }),
                _ => Err("cpu: an entry argument that is not a builtin".into()),
            })
            .collect()
    }

    fn push_frame(&mut self, inv: &mut Inv<'m>, func: &'m Function, args: Vec<Val>, result: Option<Handle<Expression>>) -> Result<(), String> {
        self.next_frame += 1;
        let id = self.next_frame;
        let lazy = self.lazy.entry(func as *const Function).or_insert_with(|| Rc::new(short_circuit_only(func))).clone();
        let mut frame = Frame { id, func, args, vals: HashMap::new(), conts: vec![Cont::Block { block: &func.body, pc: 0 }], result, lazy };
        for (h, l) in func.local_variables.iter() {
            let v = match l.init {
                Some(e) => self.eval(inv, &mut frame, e)?,
                None => zero(self.module, l.ty)?,
            };
            inv.locals.insert((id, h), v);
        }
        inv.frames.push(frame);
        Ok(())
    }

    fn global_expr(&self, h: Handle<Expression>) -> Result<Val, String> {
        Ok(match &self.module.global_expressions[h] {
            Expression::Literal(l) => literal(l)?,
            Expression::ZeroValue(t) => zero(self.module, *t)?,
            Expression::Constant(c) => self.global_expr(self.module.constants[*c].init)?,
            Expression::Compose { ty, components } => self.compose(*ty, components.iter().map(|c| self.global_expr(*c)).collect::<Result<_, _>>()?)?,
            Expression::Splat { size, value } => Val::V(vec![self.global_expr(*value)?; *size as usize]),
            other => return Err(format!("cpu: the constant expression {other:?}")),
        })
    }

    fn compose(&self, ty: Handle<naga::Type>, parts: Vec<Val>) -> Result<Val, String> {
        Ok(match &self.module.types[ty].inner {
            TypeInner::Vector { .. } => {
                let mut out = Vec::new();
                for p in parts {
                    match p {
                        Val::V(v) => out.extend(v),
                        x => out.push(x),
                    }
                }
                Val::V(out)
            }
            TypeInner::Matrix { .. } => Val::M(parts.iter().map(|c| match c {
                Val::V(v) => v.iter().map(Val::f).collect::<Result<Vec<f32>, String>>(),
                other => Err(format!("cpu: a matrix column {other:?}")),
            }).collect::<Result<_, _>>()?),
            _ => Val::V(parts),
        })
    }

    fn ptr_get(&self, inv: &Inv, p: &Ptr) -> Result<Val, String> {
        let root = match &p.base {
            Base::Local(id, h) => inv.locals.get(&(*id, *h)).ok_or("cpu: a dead local")?,
            Base::Global(g) => match self.module.global_variables[*g].space {
                AddressSpace::WorkGroup => &self.shared[g],
                AddressSpace::Private => &inv.privates[g],
                _ => &self.globals[g],
            },
        };
        // Past the end of an array: zero of the element (ReadZeroSkipWrite).
        match get_path_val(root, &p.path) {
            Some(v) => Ok(v),
            None => {
                let ty = match &p.base {
                    Base::Global(g) => self.module.global_variables[*g].ty,
                    Base::Local(_, _) => return Err("cpu: a local indexed out of range".into()),
                };
                self.zero_at(ty, &p.path)
            }
        }
    }

    /// The zero of the type reached by a path into `ty`.
    fn zero_at(&self, mut ty: Handle<naga::Type>, path: &[u32]) -> Result<Val, String> {
        for &i in path {
            ty = match &self.module.types[ty].inner {
                TypeInner::Array { base, .. } => *base,
                TypeInner::Struct { members, .. } => members.get(i as usize).ok_or("cpu: a struct member out of range")?.ty,
                TypeInner::Vector { scalar, .. } => return scalar_zero(scalar.kind),
                TypeInner::Matrix { rows, .. } => return Ok(Val::V(vec![Val::F(0.0); *rows as usize])),
                other => return Err(format!("cpu: a path into {other:?}")),
            };
        }
        zero(self.module, ty)
    }

    fn ptr_set(&mut self, inv: &mut Inv, p: &Ptr, v: Val) -> Result<(), String> {
        match &p.base {
            Base::Local(id, h) => {
                let root = inv.locals.get_mut(&(*id, *h)).ok_or("cpu: a dead local")?;
                set_path(root, &p.path, v);
            }
            Base::Global(g) => {
                let gv = &self.module.global_variables[*g];
                match gv.space {
                    AddressSpace::WorkGroup => {
                        let root = self.shared.get_mut(g).ok_or("cpu: workgroup memory")?;
                        set_path(root, &p.path, v);
                    }
                    AddressSpace::Private => {
                        let root = inv.privates.get_mut(g).ok_or("cpu: private memory")?;
                        set_path(root, &p.path, v);
                    }
                    _ => {
                        let root = self.globals.get_mut(g).ok_or("cpu: a binding")?;
                        if set_path(root, &p.path, v) {
                            self.writes.push(Write { binding: gv.name.clone().unwrap_or_default(), path: p.path.clone(), gid: inv.gid });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn eval(&self, inv: &Inv, fr: &mut Frame<'m>, h: Handle<Expression>) -> Result<Val, String> {
        if let Some(v) = fr.vals.get(&h) {
            return Ok(v.clone());
        }
        let v = self.compute(inv, fr, h)?;
        fr.vals.insert(h, v.clone());
        Ok(v)
    }

    fn compute(&self, inv: &Inv, fr: &mut Frame<'m>, h: Handle<Expression>) -> Result<Val, String> {
        let e = &fr.func.expressions[h];
        Ok(match e {
            Expression::Literal(l) => literal(l)?,
            Expression::Constant(c) => self.global_expr(self.module.constants[*c].init)?,
            Expression::ZeroValue(t) => zero(self.module, *t)?,
            Expression::Compose { ty, components } => {
                let parts = components.iter().map(|c| self.eval(inv, fr, *c)).collect::<Result<_, _>>()?;
                self.compose(*ty, parts)?
            }
            Expression::Splat { size, value } => Val::V(vec![self.eval(inv, fr, *value)?; *size as usize]),
            Expression::Swizzle { size, vector, pattern } => {
                let v = self.eval(inv, fr, *vector)?;
                let Val::V(x) = v else { return Err("cpu: a swizzle of a scalar".into()) };
                Val::V(pattern[..*size as usize].iter().map(|c| x[*c as usize].clone()).collect())
            }
            Expression::Access { base, index } => {
                let b = self.eval(inv, fr, *base)?;
                let i = self.eval(inv, fr, *index)?.u()?;
                match b {
                    Val::P(mut p) => {
                        p.path.push(i);
                        Val::P(p)
                    }
                    v => get_path_val(&v, &[i]).ok_or("cpu: an index past a value")?,
                }
            }
            Expression::AccessIndex { base, index } => {
                let b = self.eval(inv, fr, *base)?;
                match b {
                    Val::P(mut p) => {
                        p.path.push(*index);
                        Val::P(p)
                    }
                    v => get_path_val(&v, &[*index]).ok_or("cpu: an index past a value")?,
                }
            }
            Expression::FunctionArgument(i) => fr.args.get(*i as usize).cloned().ok_or("cpu: an argument")?,
            Expression::GlobalVariable(g) => Val::P(Ptr { base: Base::Global(*g), path: Vec::new() }),
            Expression::LocalVariable(l) => Val::P(Ptr { base: Base::Local(fr.id, *l), path: Vec::new() }),
            Expression::Load { pointer } => {
                let Val::P(p) = self.eval(inv, fr, *pointer)? else { return Err("cpu: a load of a value".into()) };
                // A uniform-load result is set by its statement; a plain load reads memory.
                self.ptr_get(inv, &p)?
            }
            Expression::Unary { op, expr } => {
                let v = self.eval(inv, fr, *expr)?;
                unary(*op, &v)?
            }
            // WGSL's `||` and `&&` short-circuit: the right side is
            // evaluated only when the left does not decide (`j == 0u ||
            // best[j - 1u] < s` reads no best[j - 1u] at j 0).
            Expression::Binary { op: op @ (BinaryOperator::LogicalOr | BinaryOperator::LogicalAnd), left, right } => {
                let a = self.eval(inv, fr, *left)?.b()?;
                if a == (*op == BinaryOperator::LogicalOr) {
                    Val::B(a)
                } else {
                    Val::B(self.eval(inv, fr, *right)?.b()?)
                }
            }
            Expression::Binary { op, left, right } => {
                let (a, b) = (self.eval(inv, fr, *left)?, self.eval(inv, fr, *right)?);
                binary(*op, &a, &b)?
            }
            Expression::Select { condition, accept, reject } => {
                let c = self.eval(inv, fr, *condition)?;
                let (a, r) = (self.eval(inv, fr, *accept)?, self.eval(inv, fr, *reject)?);
                select(&c, &a, &r)?
            }
            Expression::Math { fun, arg, arg1, arg2, arg3 } => {
                let args = [Some(*arg), *arg1, *arg2, *arg3].into_iter().flatten().map(|a| self.eval(inv, fr, a)).collect::<Result<Vec<_>, _>>()?;
                math(*fun, &args)?
            }
            Expression::As { expr, kind, convert: c } => {
                let v = self.eval(inv, fr, *expr)?;
                convert(&v, *kind, c.is_none())?
            }
            Expression::Relational { fun, argument } => {
                let v = self.eval(inv, fr, *argument)?;
                let lanes: Vec<Val> = match v {
                    Val::V(x) => x,
                    s => vec![s],
                };
                match fun {
                    naga::RelationalFunction::All => Val::B(lanes.iter().map(Val::b).collect::<Result<Vec<_>, _>>()?.iter().all(|b| *b)),
                    naga::RelationalFunction::Any => Val::B(lanes.iter().map(Val::b).collect::<Result<Vec<_>, _>>()?.iter().any(|b| *b)),
                    naga::RelationalFunction::IsNan => Val::B(lanes.iter().map(Val::f).collect::<Result<Vec<_>, _>>()?.iter().any(|f| f.is_nan())),
                    naga::RelationalFunction::IsInf => Val::B(lanes.iter().map(Val::f).collect::<Result<Vec<_>, _>>()?.iter().any(|f| f.is_infinite())),
                }
            }
            Expression::ArrayLength(p) => {
                let Val::P(p) = self.eval(inv, fr, *p)? else { return Err("cpu: arrayLength of a value".into()) };
                match self.ptr_get(inv, &p)? {
                    Val::V(x) => Val::U(x.len() as u32),
                    _ => return Err("cpu: arrayLength of a scalar".into()),
                }
            }
            other => return Err(format!("cpu: the expression {other:?} read before its statement, or not run here")),
        })
    }

    /// Run until a workgroup barrier or the end of the entry point.
    fn step_until_barrier(&mut self, inv: &mut Inv<'m>) -> Result<Status, String> {
        loop {
            self.steps += 1;
            if self.steps > self.max_steps {
                return Err(format!("cpu: more than {} statements", self.max_steps));
            }
            let Some(frame) = inv.frames.last_mut() else { return Ok(Status::Done) };
            let Some(cont) = frame.conts.last_mut() else {
                // The function ended without a return.
                self.ret(inv, None)?;
                continue;
            };
            match cont {
                Cont::Block { block, pc } => {
                    let block: &'m Block = block;
                    if *pc >= block.len() {
                        frame.conts.pop();
                        // A loop's body or continuing block ended.
                        if let Some(Cont::Loop { continuing, in_continuing, break_if, .. }) = frame.conts.last_mut() {
                            if !*in_continuing {
                                *in_continuing = true;
                                let c: &'m Block = continuing;
                                frame.conts.push(Cont::Block { block: c, pc: 0 });
                            } else {
                                let bi = *break_if;
                                let stop = match bi {
                                    Some(e) => {
                                        let mut fr = inv.frames.pop().expect("a frame");
                                        let v = self.eval(inv, &mut fr, e);
                                        inv.frames.push(fr);
                                        v?.b()?
                                    }
                                    None => false,
                                };
                                let frame = inv.frames.last_mut().expect("a frame");
                                if stop {
                                    frame.conts.pop();
                                } else if let Some(Cont::Loop { body, in_continuing, .. }) = frame.conts.last_mut() {
                                    *in_continuing = false;
                                    // A new pass (its Emits evaluate the body's expressions again).
                                    let b: &'m Block = body;
                                    frame.conts.push(Cont::Block { block: b, pc: 0 });
                                }
                            }
                        } else if let Some(Cont::Switch) = frame.conts.last() {
                            frame.conts.pop();
                        }
                        continue;
                    }
                    let stmt = &block[*pc];
                    *pc += 1;
                    if let Some(s) = self.stmt(inv, stmt)? {
                        return Ok(s);
                    }
                }
                Cont::Loop { .. } | Cont::Switch => {
                    // A loop or switch with no block above it: ended.
                    frame.conts.pop();
                }
            }
        }
    }

    /// Every invocation waits at a subgroup operation: each subgroup's
    /// results, written to the invocations' frames.
    fn collective(invs: &mut [Inv<'m>], wg: [u32; 3]) -> Result<(), String> {
        use naga::{CollectiveOperation as C, SubgroupOperation as S};
        let mut groups: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
        for (i, inv) in invs.iter().enumerate().filter(|(_, i)| i.pending.is_some()) {
            groups.entry(lidx(inv, wg) / SUBGROUP).or_default().push(i);
        }
        for members in groups.values() {
            let pend: Vec<(Pend, Val, Option<u32>, Handle<Expression>)> = members.iter().map(|&i| invs[i].pending.clone().expect("pending")).collect();
            let kind = pend[0].0;
            if pend.iter().any(|p| p.0 != kind) {
                return Err("cpu: two subgroup operations met in one subgroup (non-uniform control)".into());
            }
            let vals: Vec<&Val> = pend.iter().map(|p| &p.1).collect();
            let lane_of = |i: usize| (lidx(&invs[members[i]], wg) % SUBGROUP) as usize;
            let at = |lane: usize, own: usize| -> Val { members.iter().position(|&m| lidx(&invs[m], wg) % SUBGROUP == lane as u32).map_or_else(|| vals[own].clone(), |p| vals[p].clone()) };
            let fold = |op: S, a: &Val, b: &Val| -> Result<Val, String> {
                match op {
                    S::Add => binary(BinaryOperator::Add, a, b),
                    S::Mul => binary(BinaryOperator::Multiply, a, b),
                    S::Min => min_max(MathFunction::Min, a, b),
                    S::Max => min_max(MathFunction::Max, a, b),
                    S::And | S::All => binary(BinaryOperator::And, a, b),
                    S::Or | S::Any => binary(BinaryOperator::InclusiveOr, a, b),
                    S::Xor => binary(BinaryOperator::ExclusiveOr, a, b),
                }
            };
            let identity = |op: S, like: &Val| -> Val {
                fn go(op: S, like: &Val) -> Val {
                    match like {
                        Val::V(v) => Val::V(v.iter().map(|x| go(op, x)).collect()),
                        Val::F(_) => Val::F(match op {
                            S::Mul => 1.0,
                            S::Min => f32::INFINITY,
                            S::Max => f32::NEG_INFINITY,
                            _ => 0.0,
                        }),
                        Val::U(_) => Val::U(match op {
                            S::Mul => 1,
                            S::Min | S::And => u32::MAX,
                            _ => 0,
                        }),
                        Val::I(_) => Val::I(match op {
                            S::Mul => 1,
                            S::Min => i32::MAX,
                            S::Max => i32::MIN,
                            S::And => -1,
                            _ => 0,
                        }),
                        Val::B(_) => Val::B(matches!(op, S::And | S::All)),
                        other => other.clone(),
                    }
                }
                go(op, like)
            };
            let mut order: Vec<usize> = (0..members.len()).collect();
            order.sort_by_key(|&i| lane_of(i));
            let mut results: Vec<Val> = vec![Val::U(0); members.len()];
            match kind {
                Pend::Ballot => {
                    let mut mask = [0u32; 4];
                    for (i, v) in vals.iter().enumerate() {
                        if v.b()? {
                            let l = lane_of(i);
                            mask[l / 32] |= 1 << (l % 32);
                        }
                    }
                    let m = Val::V(mask.iter().map(|x| Val::U(*x)).collect());
                    results.iter_mut().for_each(|r| *r = m.clone());
                }
                Pend::Collective(op, coll) => {
                    let mut acc = identity(op, vals[0]);
                    for &i in &order {
                        let before = acc.clone();
                        acc = fold(op, &acc, vals[i])?;
                        results[i] = match coll {
                            C::ExclusiveScan => before,
                            _ => acc.clone(),
                        };
                    }
                    if coll == C::Reduce {
                        results.iter_mut().for_each(|r| *r = acc.clone());
                    }
                }
                Pend::BroadcastFirst => {
                    let first = vals[order[0]].clone();
                    results.iter_mut().for_each(|r| *r = first.clone());
                }
                g => {
                    for i in 0..members.len() {
                        let k = pend[i].2.unwrap_or(0) as usize;
                        let me = lane_of(i);
                        let src = match g {
                            Pend::Broadcast | Pend::Shuffle => k,
                            Pend::ShuffleDown => me + k,
                            Pend::ShuffleUp => me.wrapping_sub(k),
                            _ => me ^ k,
                        };
                        results[i] = at(src, i);
                    }
                }
            }
            for (i, &m) in members.iter().enumerate() {
                let (_, _, _, h) = invs[m].pending.take().expect("pending");
                invs[m].frames.last_mut().ok_or("cpu: a subgroup operation with no frame")?.vals.insert(h, results[i].clone());
            }
        }
        Ok(())
    }

    fn ret(&mut self, inv: &mut Inv<'m>, value: Option<Val>) -> Result<(), String> {
        let frame = inv.frames.pop().ok_or("cpu: a return with no frame")?;
        inv.locals.retain(|(id, _), _| *id != frame.id);
        if let (Some(r), Some(caller)) = (frame.result, inv.frames.last_mut()) {
            caller.vals.insert(r, value.unwrap_or(Val::U(0)));
        }
        Ok(())
    }

    fn stmt(&mut self, inv: &mut Inv<'m>, s: &'m Statement) -> Result<Option<Status>, String> {
        let mut fr = inv.frames.pop().ok_or("cpu: no frame")?;
        let out = self.stmt_in(inv, &mut fr, s);
        // A call pushed its callee above: keep the caller below it.
        let callee = if !inv.frames.is_empty() && matches!(s, Statement::Call { .. }) && out.as_ref().is_ok_and(|o| matches!(o, Flow::Called)) { inv.frames.pop() } else { None };
        inv.frames.push(fr);
        if let Some(c) = callee {
            inv.frames.push(c);
        }
        match out? {
            Flow::Next | Flow::Called => Ok(None),
            Flow::Barrier => Ok(Some(Status::Barrier)),
            Flow::Collective => Ok(Some(Status::Collective)),
            Flow::Return(v) => {
                self.ret(inv, v)?;
                Ok(if inv.frames.is_empty() { Some(Status::Done) } else { None })
            }
        }
    }

    fn stmt_in(&mut self, inv: &mut Inv<'m>, fr: &mut Frame<'m>, s: &'m Statement) -> Result<Flow, String> {
        match s {
            Statement::Emit(range) => {
                for h in range.clone() {
                    // A short-circuit operator's right side waits for the
                    // operator; a value from an earlier pass of a loop is
                    // dropped so it is computed afresh.
                    if fr.lazy.contains(&h) {
                        fr.vals.remove(&h);
                        continue;
                    }
                    let v = self.compute(inv, fr, h)?;
                    fr.vals.insert(h, v);
                }
            }
            Statement::Block(b) => fr.conts.push(Cont::Block { block: b, pc: 0 }),
            Statement::If { condition, accept, reject } => {
                let c = self.eval(inv, fr, *condition)?.b()?;
                fr.conts.push(Cont::Block { block: if c { accept } else { reject }, pc: 0 });
            }
            Statement::Switch { selector, cases } => {
                let v = self.eval(inv, fr, *selector)?;
                let sel = v.u()?;
                let mut start = cases.iter().position(|c| match c.value {
                    naga::SwitchValue::I32(x) => x as u32 == sel,
                    naga::SwitchValue::U32(x) => x == sel,
                    naga::SwitchValue::Default => false,
                });
                if start.is_none() {
                    start = cases.iter().position(|c| matches!(c.value, naga::SwitchValue::Default));
                }
                if let Some(i) = start {
                    // The case and every case it falls through to, as one run.
                    let mut end = i;
                    while cases[end].fall_through && end + 1 < cases.len() {
                        end += 1;
                    }
                    fr.conts.push(Cont::Switch);
                    for c in cases[i..=end].iter().rev() {
                        fr.conts.push(Cont::Block { block: &c.body, pc: 0 });
                    }
                }
            }
            Statement::Loop { body, continuing, break_if } => {
                fr.conts.push(Cont::Loop { body, continuing, break_if: *break_if, in_continuing: false });
                fr.conts.push(Cont::Block { block: body, pc: 0 });
            }
            Statement::Break => {
                while let Some(c) = fr.conts.pop() {
                    if matches!(c, Cont::Loop { .. } | Cont::Switch) {
                        break;
                    }
                }
            }
            Statement::Continue => {
                while let Some(c) = fr.conts.last() {
                    if matches!(c, Cont::Loop { .. }) {
                        break;
                    }
                    fr.conts.pop();
                }
                if let Some(Cont::Loop { continuing, in_continuing, .. }) = fr.conts.last_mut() {
                    *in_continuing = true;
                    let c: &'m Block = continuing;
                    fr.conts.push(Cont::Block { block: c, pc: 0 });
                }
            }
            Statement::Return { value } => {
                let v = match value {
                    Some(e) => Some(self.eval(inv, fr, *e)?),
                    None => None,
                };
                return Ok(Flow::Return(v));
            }
            Statement::Kill => return Ok(Flow::Return(None)),
            Statement::Barrier(flags) => {
                if flags.contains(naga::Barrier::WORK_GROUP) {
                    return Ok(Flow::Barrier);
                }
            }
            Statement::Store { pointer, value } => {
                let Val::P(p) = self.eval(inv, fr, *pointer)? else { return Err("cpu: a store through a value".into()) };
                let v = self.eval(inv, fr, *value)?;
                self.ptr_set(inv, &p, v)?;
            }
            Statement::Atomic { pointer, fun, value, result } => {
                let Val::P(p) = self.eval(inv, fr, *pointer)? else { return Err("cpu: an atomic through a value".into()) };
                let old = self.ptr_get(inv, &p)?;
                let v = self.eval(inv, fr, *value)?;
                let new = match fun {
                    naga::AtomicFunction::Add => binary(BinaryOperator::Add, &old, &v)?,
                    naga::AtomicFunction::Subtract => binary(BinaryOperator::Subtract, &old, &v)?,
                    naga::AtomicFunction::And => binary(BinaryOperator::And, &old, &v)?,
                    naga::AtomicFunction::InclusiveOr => binary(BinaryOperator::InclusiveOr, &old, &v)?,
                    naga::AtomicFunction::ExclusiveOr => binary(BinaryOperator::ExclusiveOr, &old, &v)?,
                    naga::AtomicFunction::Min => min_max(MathFunction::Min, &old, &v)?,
                    naga::AtomicFunction::Max => min_max(MathFunction::Max, &old, &v)?,
                    naga::AtomicFunction::Exchange { compare: None } => v,
                    naga::AtomicFunction::Exchange { compare: Some(_) } => return Err("cpu: atomicCompareExchangeWeak".into()),
                };
                self.ptr_set(inv, &p, new)?;
                fr.vals.insert(*result, old);
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                let Val::P(p) = self.eval(inv, fr, *pointer)? else { return Err("cpu: a uniform load of a value".into()) };
                let v = self.ptr_get(inv, &p)?;
                fr.vals.insert(*result, v);
            }
            Statement::Call { function, arguments, result } => {
                let args = arguments.iter().map(|a| self.eval(inv, fr, *a)).collect::<Result<Vec<_>, _>>()?;
                let callee = &self.module.functions[*function];
                // The callee's frame goes above the caller's (stmt re-stacks them).
                self.push_frame(inv, callee, args, *result)?;
                return Ok(Flow::Called);
            }
            Statement::SubgroupBallot { result, predicate } => {
                let p = match predicate {
                    Some(e) => self.eval(inv, fr, *e)?,
                    None => Val::B(true),
                };
                inv.pending = Some((Pend::Ballot, p, None, *result));
                return Ok(Flow::Collective);
            }
            Statement::SubgroupCollectiveOperation { op, collective_op, argument, result } => {
                let v = self.eval(inv, fr, *argument)?;
                inv.pending = Some((Pend::Collective(*op, *collective_op), v, None, *result));
                return Ok(Flow::Collective);
            }
            Statement::SubgroupGather { mode, argument, result } => {
                let v = self.eval(inv, fr, *argument)?;
                let (p, i) = match mode {
                    naga::GatherMode::BroadcastFirst => (Pend::BroadcastFirst, None),
                    naga::GatherMode::Broadcast(e) => (Pend::Broadcast, Some(*e)),
                    naga::GatherMode::Shuffle(e) => (Pend::Shuffle, Some(*e)),
                    naga::GatherMode::ShuffleDown(e) => (Pend::ShuffleDown, Some(*e)),
                    naga::GatherMode::ShuffleUp(e) => (Pend::ShuffleUp, Some(*e)),
                    naga::GatherMode::ShuffleXor(e) => (Pend::ShuffleXor, Some(*e)),
                };
                let lane = match i {
                    Some(e) => Some(self.eval(inv, fr, e)?.u()?),
                    None => None,
                };
                inv.pending = Some((p, v, lane, *result));
                return Ok(Flow::Collective);
            }
            Statement::ImageStore { .. } | Statement::RayQuery { .. } => return Err("cpu: an image store or a ray query".into()),
        }
        Ok(Flow::Next)
    }
}

enum Flow {
    Next,
    Called,
    Barrier,
    Collective,
    Return(Option<Val>),
}

fn literal(l: &naga::Literal) -> Result<Val, String> {
    Ok(match l {
        naga::Literal::F32(x) => Val::F(*x),
        naga::Literal::U32(x) => Val::U(*x),
        naga::Literal::I32(x) => Val::I(*x),
        naga::Literal::Bool(b) => Val::B(*b),
        naga::Literal::AbstractInt(x) => Val::I(*x as i32),
        naga::Literal::AbstractFloat(x) => Val::F(*x as f32),
        other => return Err(format!("cpu: the literal {other:?}")),
    })
}

fn unary(op: naga::UnaryOperator, v: &Val) -> Result<Val, String> {
    use naga::UnaryOperator as U;
    Ok(match (op, v) {
        (_, Val::V(x)) => Val::V(x.iter().map(|e| unary(op, e)).collect::<Result<_, _>>()?),
        (U::Negate, Val::F(x)) => Val::F(-x),
        (U::Negate, Val::I(x)) => Val::I(x.wrapping_neg()),
        (U::Negate, Val::M(m)) => Val::M(m.iter().map(|c| c.iter().map(|x| -x).collect()).collect()),
        (U::LogicalNot, Val::B(b)) => Val::B(!b),
        (U::BitwiseNot, Val::U(x)) => Val::U(!x),
        (U::BitwiseNot, Val::I(x)) => Val::I(!x),
        (op, v) => return Err(format!("cpu: {op:?} of {v:?}")),
    })
}

fn select(c: &Val, a: &Val, r: &Val) -> Result<Val, String> {
    // WGSL's select(f, t, c): naga's accept is t, reject is f.
    Ok(match (c, a, r) {
        (Val::B(b), a, r) => if *b { a.clone() } else { r.clone() },
        (Val::V(cs), Val::V(x), Val::V(y)) => Val::V(cs.iter().zip(x).zip(y).map(|((c, a), r)| select(c, a, r)).collect::<Result<_, _>>()?),
        (c, _, _) => return Err(format!("cpu: select on {c:?}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");
    const TILED: &str = include_str!("../../kingdoms/wgsl/gram/tiled.wgsl");

    /// The Gram's buffers at a small shape: phi (m x d), the uniform, h zero.
    pub(crate) fn gram_buffers(d: u32, m: u32, r_rows: u32, r_off: u32, phi: &[f32]) -> HashMap<String, Val> {
        let mut b = HashMap::new();
        b.insert("phi".to_string(), Val::V(phi.iter().map(|x| Val::F(*x)).collect()));
        b.insert("h".to_string(), Val::V(vec![Val::F(0.0); (r_rows * d) as usize]));
        b.insert("cfg".to_string(), Val::V(vec![Val::U(d), Val::U(m), Val::U(r_rows), Val::U(r_off)]));
        b
    }

    fn phi(d: u32, m: u32) -> Vec<f32> {
        (0..d * m).map(|i| ((i * 7919 % 97) as f32) / 31.0 - 1.5).collect()
    }

    fn h_of(out: &Dispatch) -> Vec<f32> {
        let mut w = Vec::new();
        out.buffers["h"].words(&mut w);
        w.into_iter().map(f32::from_bits).collect()
    }

    /// The shipped and the tiled kernels (workgroup memory, barriers) give
    /// the Gram H = Phi^T Phi of a reference loop, bit for bit where the
    /// summation order is the same (the shipped kernel's: t ascending).
    #[test]
    fn the_shipped_and_tiled_kernels_compute_the_gram() {
        let (d, m) = (37u32, 41u32);
        let p = phi(d, m);
        let mut want = vec![0.0f32; (d * d) as usize];
        for r in 0..d {
            for c in 0..d {
                let mut acc = 0.0f32;
                for t in 0..m {
                    acc += p[(t * d + r) as usize] * p[(t * d + c) as usize];
                }
                want[(r * d + c) as usize] = acc;
            }
        }
        let module = naga::front::wgsl::parse_str(SHIPPED).unwrap();
        let out = run(&module, gram_buffers(d, m, d, 0, &p), [d.div_ceil(16), d.div_ceil(16), 1], 100_000_000).unwrap();
        assert_eq!(h_of(&out), want);
        // Every cell written once, by the thread that owns it.
        assert_eq!(out.writes.len(), (d * d) as usize);
        let tiled = naga::front::wgsl::parse_str(TILED).unwrap();
        let out = run(&tiled, gram_buffers(d, m, d, 0, &p), [d.div_ceil(64), d.div_ceil(64), 1], 100_000_000).unwrap();
        let got = h_of(&out);
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "tiled against the reference: {worst}");
    }

    /// WGSL'S SHORT CIRCUIT: `j == 0u || best[j - 1u] < 2.0` reads no
    /// best[j - 1u] at j 0 (an index past the local array is an error here),
    /// and its dual `j != 0u && best[j - 1u] < 2.0` neither; in a loop the
    /// right side is computed afresh each pass. best = [1, 3, 1, 3]: the
    /// `||` gives 1 1 0 1, the `&&` 0 1 0 1.
    #[test]
    fn logical_or_and_and_short_circuit() {
        let text = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    var best: array<f32, 4> = array<f32, 4>(1.0, 3.0, 1.0, 3.0);
    for (var j = 0u; j < 4u; j = j + 1u) {
        out[j] = select(0u, 1u, j == 0u || best[j - 1u] < 2.0);
        out[4u + j] = select(0u, 1u, j != 0u && best[j - 1u] < 2.0);
    }
}
";
        let mut bufs = HashMap::new();
        bufs.insert("out".to_string(), Val::V(vec![Val::U(9); 8]));
        let d = run_text(text, bufs, [1, 1, 1], 1_000_000).unwrap();
        let mut w = Vec::new();
        d.buffers["out"].words(&mut w);
        assert_eq!(w, vec![1, 1, 0, 1, 0, 1, 0, 1]);
        // The guard is what keeps the read in range: without it the same
        // read at j 0 is the out-of-range error.
        let unguarded = text.replace("j == 0u || ", "").replace("j != 0u && ", "");
        let mut bufs = HashMap::new();
        bufs.insert("out".to_string(), Val::V(vec![Val::U(9); 8]));
        match run_text(&unguarded, bufs, [1, 1, 1], 1_000_000) {
            Err(e) => assert!(e.contains("out of range"), "{e}"),
            Ok(_) => panic!("the unguarded read at j 0 ran"),
        }
    }
}
