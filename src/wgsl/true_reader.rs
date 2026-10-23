//! UI1 (s9 batch 8): a kernel's own WGSL read exactly into the rows of the true kingdom, for a module interface (v24).
//!
//! The reader is the parser side of the round trip (the renderer, [`TrueKingdom::render`], is the emitter). naga reads
//! and validates the text; each entry point is walked statement by statement into an abstract statement list (phase 1:
//! user functions inlined, a `let` that reads mutable state stored once in a register, every local variable a register,
//! every counted loop the kingdom's counted loop), then each statement and value is spelled by the rows whose types fit
//! (phase 2: a value is matched to the rows with its own spelling, `({0} + {1})`, `min({0}, {1})`, whose output is the
//! type its slot asks for; the identity rows `{0}` are the only coercions). Nothing is guessed: a construct no row spells
//! is a table gap, [`ReadError::Gap`], our bug, named by the construct; a kernel that is illegal under its contract (an
//! unbounded loop, a barrier the kingdom's types cannot prove uniform) is [`ReadError::Out`] with its reason.
//!
//! Spellings that differ from the source without changing any value, each the kingdom's own form of the construct:
//!
//! * a counted loop `for (var i = a; i < n; i += s)` is the kingdom's loop over `k = 0, s, 2s, ...` below the bound
//!   `n - a` (floored), with `i` read as `a + k` (`k` when `a` is 0); a halving loop `for (var s = C; s > 0u; s >>= 1u)`
//!   is the counted loop of its trip count with `s` read as `C >> k`; after the loop the seed's variable holds the final
//!   value it would hold;
//! * `if (c) { ...; return; } rest` is `if (c) { ... } else { rest }`; a helper's `return v` stores `v` in the call's
//!   result register;
//! * a break inside a conditional (`if (c) { X; break; }`) is `b = c; if (b) { X } if (b) { break; }`, the kingdom's one
//!   break point of a loop body; a `continue` is the `else` of the rest of the body;
//! * a value of a vector and a scalar is the vector op over the scalar splatted (every lane the same operation);
//!   `clamp(x, lo, hi)` with literal `lo <= hi` is `min(max(x, lo), hi)` (WGSL's definition);
//! * a plain store to or load from the atomic arena is `atomicStore` / `atomicLoad`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use naga::{AddressSpace, BinaryOperator, Block, BuiltIn, Expression, Function, Handle, Literal, LocalVariable, MathFunction, Module, ScalarKind, Statement, TypeInner, UnaryOperator};

use crate::typed_gep::Tree;
use crate::wgsl::true_kingdom::{RowKind, TrueKingdom};

/// Why a kernel was not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// The table lacks a row (or a constant) the kernel needs: our bug.
    Gap(String),
    /// The kernel is illegal under its contract, with the reason.
    Out(String),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Gap(s) => write!(f, "our software has a bug: the table lacks {s}"),
            ReadError::Out(s) => write!(f, "outside the kingdom: {s}"),
        }
    }
}

type R<T> = Result<T, ReadError>;

fn gap<T>(s: impl Into<String>) -> R<T> {
    Err(ReadError::Gap(s.into()))
}

/// The ranges of the interface's uniform fields and dispatch (the problem's `safety.contract`): what bounds a loop.
#[derive(Debug, Clone)]
pub struct Contract {
    /// `param cfg.<field> lo hi`, by field index of the uniform block.
    pub fields: Vec<(String, u64)>,
    /// The most workgroups of any dispatch along x.
    pub workgroups_x: u64,
}

impl Contract {
    /// `param cfg.<f> lo hi` and `workgroups x lo hi` lines of a safety contract, for the fields in `order`.
    pub fn parse(text: &str, order: &[&str]) -> Result<Contract, String> {
        let mut max: BTreeMap<String, u64> = BTreeMap::new();
        let mut wx = None;
        for l in text.lines() {
            let w: Vec<&str> = l.split_whitespace().collect();
            match w.as_slice() {
                ["param", name, _, hi] if name.starts_with("cfg.") => {
                    max.insert(name[4..].to_string(), hi.parse().map_err(|e| format!("safety contract: {l}: {e}"))?);
                }
                ["workgroups", "x", _, hi] => wx = Some(hi.parse().map_err(|e| format!("safety contract: {l}: {e}"))?),
                _ => {}
            }
        }
        let fields = order.iter().map(|f| max.get(*f).map(|m| (f.to_string(), *m)).ok_or_else(|| format!("the safety contract gives no range for cfg.{f}"))).collect::<Result<_, _>>()?;
        Ok(Contract { fields, workgroups_x: wx.ok_or("the safety contract gives no workgroups x range")? })
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// WGSL types of values
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Sc {
    F32,
    F16,
    U32,
    I32,
    Bool,
}

/// A value's WGSL type: a scalar or a vector of 2..=4 lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Wt {
    s: Sc,
    lanes: u8,
}

impl Wt {
    fn scalar(s: Sc) -> Wt {
        Wt { s, lanes: 1 }
    }
    fn name(self) -> String {
        let s = match self.s {
            Sc::F32 => "f32",
            Sc::F16 => "f16",
            Sc::U32 => "u32",
            Sc::I32 => "i32",
            Sc::Bool => "bool",
        };
        if self.lanes == 1 {
            s.to_string()
        } else {
            format!("vec{}<{s}>", self.lanes)
        }
    }
    fn of_scalar(s: naga::Scalar) -> Option<Sc> {
        Some(match (s.kind, s.width) {
            (ScalarKind::Float, 4) => Sc::F32,
            (ScalarKind::Float, 2) => Sc::F16,
            (ScalarKind::Uint, 4) => Sc::U32,
            (ScalarKind::Sint, 4) => Sc::I32,
            (ScalarKind::Bool, _) => Sc::Bool,
            _ => return None,
        })
    }
    fn of_inner(t: &TypeInner) -> Option<Wt> {
        match t {
            TypeInner::Scalar(s) => Some(Wt::scalar(Wt::of_scalar(*s)?)),
            TypeInner::Vector { size, scalar } => Some(Wt { s: Wt::of_scalar(*scalar)?, lanes: *size as u8 }),
            TypeInner::Atomic(s) => Some(Wt::scalar(Wt::of_scalar(*s)?)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Phase 1: the abstract kernel
// ---------------------------------------------------------------------------------------------------------------------

/// A register of the kingdom's register file: its key (`f`, `i`, `b`, `v4`, ...), its number, its value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Reg {
    kind: usize,
    n: usize,
}

/// A private array of the kingdom: its kind and number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Parr {
    kind: usize,
    n: usize,
    /// The seed's own length of the array: an index past it reads its last element, as the device restricts it.
    len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bi {
    Gid,
    Lid,
    Wid,
    Nwg,
    Lidx,
}

#[derive(Debug, Clone, PartialEq)]
enum Lit {
    U(u32),
    I(i32),
    F(f32),
    B(bool),
}

/// A value, every user function inlined and every `let` that reads mutable state a register.
#[derive(Debug, Clone, PartialEq)]
enum E {
    Lit(Lit),
    Zero(Wt),
    Bi(Bi),
    Field(usize),
    Reg(Reg, Wt),
    /// The trip of loop `id` inside its body (and its final value after it), the kingdom's counter.
    Counter(usize),
    Parr(Parr, Box<E>, Wt),
    /// `binding[i]` of an f32 / u32 input or arena (an atomic arena through `atomicLoad`).
    Load(String, Box<E>, Wt),
    /// Element `j` of the vec4 view of an f32 binding.
    LoadV4(String, Box<E>),
    /// `arrayLength(&b)`, of the vec4 view when `.1`.
    Len(String, bool),
    Bin(BinaryOperator, Box<E>, Box<E>, Wt),
    Un(UnaryOperator, Box<E>, Wt),
    /// `select(f, t, c)` as WGSL spells it: (false value, true value, condition).
    Sel(Box<E>, Box<E>, Box<E>, Wt),
    /// A built-in function by its WGSL name.
    Call(&'static str, Vec<E>, Wt),
    /// `T(x)`, a numeric conversion.
    Cast(Wt, Box<E>),
    Bitcast(Wt, Box<E>),
    Compose(Wt, Vec<E>),
    Splat(Wt, Box<E>),
    Lane(Box<E>, u8, Wt),
    Swz(Box<E>, String, Wt),
    DynLane(Box<E>, Box<E>, Wt),
    /// An atomic read-modify-write of an arena cell: (function, binding, index, value); its value is the old value.
    Rmw(&'static str, String, Box<E>, Box<E>),
    /// Element `i` of workgroup array `v` (the seed's index), a foreign read outside a phase.
    Wg(usize, Box<E>, Wt),
    /// The invocation's own element `lidx + m * T` of workgroup array `v` (inside a write or pair phase).
    WgOwn(usize, Box<E>, Wt),
    /// An atomic load of element `i` of atomic workgroup array `v`.
    WgA(usize, Box<E>),
    /// An atomic read-modify-write of a workgroup atomic: (function, array, index, value).
    WgRmw(&'static str, usize, Box<E>, Box<E>),
    /// A uniform register `U<n>` (a `workgroupUniformLoad` result).
    UReg(usize),
}

impl E {
    fn wt(&self) -> Wt {
        match self {
            E::Lit(Lit::U(_)) => Wt::scalar(Sc::U32),
            E::Lit(Lit::I(_)) => Wt::scalar(Sc::I32),
            E::Lit(Lit::F(_)) => Wt::scalar(Sc::F32),
            E::Lit(Lit::B(_)) => Wt::scalar(Sc::Bool),
            E::Zero(w) | E::Reg(_, w) | E::Parr(_, _, w) | E::Load(_, _, w) | E::Bin(_, _, _, w) | E::Un(_, _, w) | E::Sel(_, _, _, w) | E::Call(_, _, w) | E::Cast(w, _) | E::Bitcast(w, _) | E::Compose(w, _) | E::Splat(w, _) | E::Lane(_, _, w) | E::Swz(_, _, w) | E::DynLane(_, _, w) => *w,
            E::Bi(Bi::Lidx) | E::Field(_) | E::Counter(_) | E::Len(..) | E::Rmw(..) | E::WgA(..) | E::WgRmw(..) | E::UReg(_) => Wt::scalar(Sc::U32),
            E::Wg(_, _, w) | E::WgOwn(_, _, w) => *w,
            E::Bi(_) => Wt { s: Sc::U32, lanes: 3 },
            E::LoadV4(..) => Wt { s: Sc::F32, lanes: 4 },
        }
    }
}

/// A statement of the abstract kernel.
#[derive(Debug, Clone, PartialEq)]
enum St {
    Set(Reg, E),
    SetLane(Reg, E, E),
    ParrSet(Parr, E, E),
    /// A store into an arena or a row output: (binding, index, value).
    Store(String, E, E),
    /// A read-modify-write whose old value nothing reads.
    Discard(E),
    If(E, Vec<St>, Vec<St>),
    Loop(Box<LoopSt>),
    /// The one break point of a loop body: leave the loop when the value holds.
    BreakIf(E),
    Barrier,
    StorageBarrier,
    /// A plain store into workgroup array `v` at the seed's index (before the phases are found).
    WgSet(usize, E, E),
    /// An atomic store into atomic workgroup array `v`.
    WgAStore(usize, E, E),
    /// `U<n> = workgroupUniformLoad(&v[i])`.
    ULoad(usize, usize, E),
    /// A write phase: the statements (W), then the rest of the block.
    Phase(Vec<St>),
    /// A pair phase over the uniform offset: the statements (P), then the rest of the block.
    Pair(E, Vec<St>),
    /// A solo phase: invocation 0's statements (S), then the rest of the block.
    Solo(Vec<St>),
    /// An own write `v[lidx + m * T] = value` (W, P).
    WgOwnSet(usize, E, E),
    /// The solo invocation's write at any element (S).
    WgPut(usize, E, E),
    /// The partner element of array `v` into the register (P).
    PLoad(Reg, usize),
    /// A split phase: the arrays it reads (placed at or above the split), the statements (X), then the rest.
    Split(Vec<usize>, Vec<St>),
    /// Element `i` of array `v`, at or above the split, into the register (X).
    HLoad(Reg, usize, E),
}

#[derive(Debug, Clone, PartialEq)]
struct LoopSt {
    id: usize,
    /// The trip bound `n`: the counter runs `0, s, 2s, ...` while below it.
    bound: E,
    stride: u32,
    body: Vec<St>,
}

/// The register kinds of the kingdom's file (`kingdoms/wgsl_true/regs.py`): key, row prefix, value type.
const REG_KINDS: &[(&str, &str, Sc, u8)] = &[
    ("f", "rf", Sc::F32, 1),
    ("i", "ru", Sc::U32, 1),
    ("b", "rb", Sc::Bool, 1),
    ("n", "rn", Sc::I32, 1),
    ("h", "rh", Sc::F16, 1),
    ("v4", "v", Sc::F32, 4),
    ("v2", "w", Sc::F32, 2),
    ("v3", "C", Sc::F32, 3),
    ("uv2", "rvu2_", Sc::U32, 2),
    ("uv3", "rvu3_", Sc::U32, 3),
    ("uv4", "rvu4_", Sc::U32, 4),
    ("iv2", "rvi2_", Sc::I32, 2),
    ("iv3", "rvi3_", Sc::I32, 3),
    ("iv4", "rvi4_", Sc::I32, 4),
    ("bv2", "rvb2_", Sc::Bool, 2),
    ("bv3", "rvb3_", Sc::Bool, 3),
    ("bv4", "rvb4_", Sc::Bool, 4),
];

fn reg_kind_of(w: Wt) -> Option<usize> {
    REG_KINDS.iter().position(|(_, _, s, l)| *s == w.s && *l == w.lanes)
}

/// One function being walked: its naga function, its argument values and its local variables' meanings.
struct Frame<'a> {
    f: &'a Function,
    info: &'a naga::valid::FunctionInfo,
    args: Vec<E>,
    /// Pointer arguments: the caller's local each names.
    ptr_args: HashMap<u32, Local>,
    locals: HashMap<Handle<LocalVariable>, Local>,
    /// Values of expressions computed by a statement (a call's result, an atomic's old value) or stored in a register at
    /// their emit point (a `let` of mutable state).
    values: HashMap<Handle<Expression>, E>,
    /// The named expressions (`let`s).
    named: BTreeSet<Handle<Expression>>,
    ret: Option<(Reg, Wt)>,
}

#[derive(Debug, Clone)]
enum Local {
    Reg(Reg, Wt),
    Parr(Parr, Wt),
    /// A loop's counter variable inside its loop: `start + k` (`k` when start is `None`), or `start >> k` for a halving
    /// loop, `start << k` for a doubling one.
    Counter { loop_id: usize, start: Option<E>, op: CounterOp },
    /// A temporary naga wrote for a short-circuit `||` / `&&` whose value is stable: the value itself.
    Known(E),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CounterOp {
    Add,
    Shr,
    Shl,
}

/// The private-array kinds of the kingdom (`regs.py` PARRS): key, row prefix, element type, length, index type.
fn parr_kinds(k: &TrueKingdom) -> Vec<(String, String, Wt, u32, String)> {
    let mut out = Vec::new();
    for r in &k.rows {
        // `parr.get_<key>`: (parr<elem,len>.form, u32.pidx_j) -> value
        let Some(key) = r.name.strip_prefix("parr.get_") else { continue };
        if key.ends_with("_lane") || key.ends_with("_w") || r.args.len() != 2 {
            continue;
        }
        let at = k.type_name(r.args[0]);
        let Some(inner) = at.strip_prefix("parr<").and_then(|s| s.split_once(">.")).map(|(a, _)| a) else { continue };
        let Some((elem, len)) = inner.rsplit_once(',') else { continue };
        let Ok(len) = len.parse::<u32>() else { continue };
        let wt = match elem {
            "f32" => Wt::scalar(Sc::F32),
            "u32" => Wt::scalar(Sc::U32),
            "i32" => Wt::scalar(Sc::I32),
            "bool" => Wt::scalar(Sc::Bool),
            "vec4<f32>" => Wt { s: Sc::F32, lanes: 4 },
            "vec2<f32>" => Wt { s: Sc::F32, lanes: 2 },
            _ => continue,
        };
        let prefix = k.rows.iter().find(|q| q.kind == RowKind::Terminal && q.out == r.args[0]).map(|q| q.spelling.trim_end_matches(|c: char| c.is_ascii_digit()).to_string()).unwrap_or_default();
        out.push((key.to_string(), prefix, wt, len, k.type_name(r.args[1])));
    }
    out
}

/// A workgroup variable of the seed: its element type, its length, whether it is atomic.
#[derive(Debug, Clone)]
struct WgVar {
    name: String,
    elem: Wt,
    len: u32,
    atomic: bool,
}

/// Where a workgroup array lives (s9 batch 9): a region of the plain pool (base and, for a vec4 array, the lane stride,
/// in blocks of the workgroup size; the seed's block layout transposed when `block` is `Some(C)`), or of the atomic pool
/// (its first word).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    Plain { base: u32, lane: u32, block: Option<u32> },
    Atomic { word: u32 },
}

/// The kernel being read: the module, the kingdom, the register allocation.
pub struct Reader<'a> {
    k: &'a TrueKingdom,
    module: &'a Module,
    info: &'a naga::valid::ModuleInfo,
    contract: &'a Contract,
    /// The uniform field names in block order.
    fields: Vec<String>,
    parrs: Vec<(String, String, Wt, u32, String)>,
    next_reg: Vec<usize>,
    reg_limit: Vec<usize>,
    next_parr: Vec<usize>,
    parr_limit: Vec<usize>,
    next_loop: usize,
    /// Each loop's trip bound and stride, for the counter's range.
    loop_bounds: HashMap<usize, (Option<u64>, u32)>,
    /// The entry being read: its workgroup size.
    wg: [u32; 3],
    /// Registers that hold one `let`'s value: its upper bound.
    reg_max: HashMap<Reg, u64>,
    /// Registers written once, by the statement that computes a `let`, a call's result or an atomic's old value.
    ssa: std::collections::HashSet<Reg>,
    /// The value a register is known to hold at the current point of the walk (its last store at this level).
    last_set: HashMap<Reg, E>,
    wgvars: Vec<WgVar>,
    used: Vec<usize>,
    blocks: BTreeMap<usize, u32>,
    /// Every split phase's (written, read) arrays: the placement's constraints.
    splits: Vec<(Vec<usize>, Vec<usize>)>,
    wg_of: HashMap<Handle<naga::GlobalVariable>, usize>,
    regions: Vec<Region>,
    next_ureg: usize,
    /// The bound `lidx < c` a condition being read under gives (the write phase's own-element proof).
    lidx_cap: std::cell::Cell<Option<u64>>,
    /// Under a guard `lidx == 0` (inside a write or split phase): an element `m * T` is invocation 0's own.
    lidx_zero: std::cell::Cell<bool>,
    // phase 2
    rows_by_name: HashMap<String, u32>,
    by_spelling: HashMap<String, Vec<u32>>,
    coercions: HashMap<u32, Vec<(u32, crate::geneframe::Ty)>>,
    /// The masking rows of the index types (`safe.pidx_<j>`, `safe.wg_of_<j>`), by output type.
    wrappers: HashMap<u32, Vec<(u32, crate::geneframe::Ty)>>,
    constants: HashMap<String, u32>,
    types_by_name: HashMap<String, crate::geneframe::Ty>,
    loop_ctx: HashMap<usize, (u8, bool)>,
    memo: HashMap<(usize, u32), Option<Tree>>,
    /// Values made during spelling (a composition regrouped, a scalar splatted): kept alive so their addresses stay
    /// unique keys of the memo.
    temps: Vec<std::rc::Rc<E>>,
    /// The most specific thing no row spells, for the gap's message.
    missing: Option<String>,
}

const MAX_INLINE_DEPTH: usize = 16;

/// The plain pool's words and the atomic pool's (`kingdoms/wgsl_true/pool.py`).
pub const P_WORDS: u64 = 6080;
pub const A_WORDS: u64 = 2112;

impl<'a> Reader<'a> {
    fn new(k: &'a TrueKingdom, module: &'a Module, info: &'a naga::valid::ModuleInfo, contract: &'a Contract) -> Reader<'a> {
        let reg_limit = REG_KINDS
            .iter()
            .map(|(_, p, _, _)| k.rows.iter().filter(|r| r.kind == RowKind::Terminal && r.name.starts_with("reg.") && r.spelling.starts_with(p) && r.spelling[p.len()..].bytes().all(|c| c.is_ascii_digit()) && r.spelling.len() > p.len()).count())
            .collect();
        let parrs = parr_kinds(k);
        let parr_limit = parrs.iter().map(|(_, p, _, _, _)| k.rows.iter().filter(|r| r.kind == RowKind::Terminal && r.name.starts_with("parr.") && r.spelling.starts_with(p.as_str()) && r.spelling[p.len()..].bytes().all(|c| c.is_ascii_digit()) && r.spelling.len() > p.len()).count()).collect();
        let mut rows_by_name = HashMap::new();
        let mut by_spelling: HashMap<String, Vec<u32>> = HashMap::new();
        let mut coercions: HashMap<u32, Vec<(u32, crate::geneframe::Ty)>> = HashMap::new();
        let mut wrappers: HashMap<u32, Vec<(u32, crate::geneframe::Ty)>> = HashMap::new();
        let mut constants = HashMap::new();
        for (i, r) in k.rows.iter().enumerate() {
            rows_by_name.insert(r.name.clone(), i as u32);
            if r.name.contains("_bare") {
                continue;
            }
            match r.kind {
                RowKind::Constant => {
                    constants.insert(k.type_name(r.out), i as u32);
                }
                RowKind::Function => {
                    let value_arg = r.args.len() == 1 && { let a = k.type_name(r.args[0]); !a.starts_with("reg<") && !a.starts_with("parr<") && !a.starts_with("wgmem<") && !a.starts_with("ureg<") };
                    if r.spelling == "{0}" && value_arg {
                        coercions.entry(r.out.code()).or_default().push((i as u32, r.args[0]));
                    } else if (r.name.starts_with("safe.pidx_") || r.name.starts_with("safe.wg_of_")) && value_arg {
                        wrappers.entry(r.out.code()).or_default().push((i as u32, r.args[0]));
                    } else {
                        by_spelling.entry(r.spelling.clone()).or_default().push(i as u32);
                    }
                }
                RowKind::Terminal => {
                    by_spelling.entry(r.spelling.clone()).or_default().push(i as u32);
                }
            }
        }
        let types_by_name = k.types.iter().map(|d| (d.dual.clone(), d.ty())).collect();
        Reader {
            k,
            module,
            info,
            contract,
            fields: contract.fields.iter().map(|(f, _)| f.clone()).collect(),
            parr_limit,
            next_parr: vec![0; parrs.len()],
            parrs,
            next_reg: vec![0; REG_KINDS.len()],
            reg_limit,
            next_loop: 0,
            loop_bounds: HashMap::new(),
            wg: [1, 1, 1],
            reg_max: HashMap::new(),
            ssa: std::collections::HashSet::new(),
            last_set: HashMap::new(),
            wgvars: Vec::new(),
            used: Vec::new(),
            blocks: BTreeMap::new(),
            splits: Vec::new(),
            wg_of: HashMap::new(),
            regions: Vec::new(),
            next_ureg: 0,
            lidx_cap: std::cell::Cell::new(None),
            lidx_zero: std::cell::Cell::new(false),
            rows_by_name,
            by_spelling,
            coercions,
            wrappers,
            constants,
            types_by_name,
            loop_ctx: HashMap::new(),
            memo: HashMap::new(),
            temps: Vec::new(),
            missing: None,
        }
    }

    fn reg(&mut self, w: Wt) -> R<Reg> {
        let kind = reg_kind_of(w).ok_or_else(|| ReadError::Gap(format!("a register of {}", w.name())))?;
        let n = self.next_reg[kind];
        if n >= self.reg_limit[kind] {
            return gap(format!("a register {}{n}: the kernel needs more than the {} {} registers", REG_KINDS[kind].1, self.reg_limit[kind], w.name()));
        }
        self.next_reg[kind] += 1;
        Ok(Reg { kind, n })
    }

    fn parr(&mut self, elem: Wt, len: u32) -> R<Parr> {
        let best = self.parrs.iter().enumerate().filter(|(_, (_, _, w, l, _))| *w == elem && *l >= len).min_by_key(|(_, (_, _, _, l, _))| *l).map(|(i, _)| i);
        let kind = best.ok_or_else(|| ReadError::Gap(format!("a private array of {len} {}", elem.name())))?;
        let n = self.next_parr[kind];
        if n >= self.parr_limit[kind] {
            return gap(format!("a private array {}{n}: the kernel needs more than the {} arrays of {} {}", self.parrs[kind].1, self.parr_limit[kind], self.parrs[kind].3, elem.name()));
        }
        self.next_parr[kind] += 1;
        Ok(Parr { kind, n, len })
    }

    fn ty_of(&self, fr: &Frame<'a>, h: Handle<Expression>) -> &'a TypeInner {
        let info: &'a naga::valid::FunctionInfo = fr.info;
        info[h].ty.inner_with(&self.module.types)
    }

    fn wt_of(&self, fr: &Frame<'a>, h: Handle<Expression>) -> R<Wt> {
        let t = self.ty_of(fr, h);
        Wt::of_inner(t).ok_or_else(|| ReadError::Gap(format!("a value of type {t:?}")))
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The interface's bindings, as the seed declares them
// ---------------------------------------------------------------------------------------------------------------------

/// How a seed's global names one of the interface's bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Bound {
    /// An f32 or u32 storage binding read or written element by element (an atomic arena included).
    Plain(String),
    /// The vec4 view of an f32 binding.
    Vec4(String),
    /// The uniform block.
    Cfg,
}

/// The interface's group 0 bindings (contract.md's table): binding number, name, what the kingdom declares.
const V24_BINDINGS: &[(u32, &str)] = &[(0, "keys"), (1, "queries"), (2, "index_u32"), (3, "index_f32"), (4, "cands"), (5, "cfg")];

/// What a binding of the interface is: an input (read-only), an arena, the row output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Input,
    Arena,
    Rows,
}

fn role(name: &str) -> Role {
    match name {
        "keys" | "queries" => Role::Input,
        "cands" => Role::Rows,
        _ => Role::Arena,
    }
}

fn atomic_fun(f: &naga::AtomicFunction) -> R<&'static str> {
    Ok(match f {
        naga::AtomicFunction::Add => "add",
        naga::AtomicFunction::Subtract => "sub",
        naga::AtomicFunction::And => "and",
        naga::AtomicFunction::ExclusiveOr => "xor",
        naga::AtomicFunction::InclusiveOr => "or",
        naga::AtomicFunction::Min => "min",
        naga::AtomicFunction::Max => "max",
        naga::AtomicFunction::Exchange { compare: None } => "exchange",
        naga::AtomicFunction::Exchange { compare: Some(_) } => return Err(ReadError::Out("atomicCompareExchangeWeak, which the contract does not allow (naga's Metal backend has none)".into())),
    })
}

fn math_name(f: MathFunction) -> Option<&'static str> {
    use MathFunction as M;
    Some(match f {
        M::Abs => "abs",
        M::Min => "min",
        M::Max => "max",
        M::Saturate => "saturate",
        M::Cos => "cos",
        M::Sin => "sin",
        M::Tan => "tan",
        M::Tanh => "tanh",
        M::Acos => "acos",
        M::Asin => "asin",
        M::Atan => "atan",
        M::Atan2 => "atan2",
        M::Ceil => "ceil",
        M::Floor => "floor",
        M::Round => "round",
        M::Fract => "fract",
        M::Trunc => "trunc",
        M::Exp => "exp",
        M::Exp2 => "exp2",
        M::Log => "log",
        M::Log2 => "log2",
        M::Pow => "pow",
        M::Dot => "dot",
        M::Cross => "cross",
        M::Distance => "distance",
        M::Length => "length",
        M::Normalize => "normalize",
        M::FaceForward => "faceForward",
        M::Reflect => "reflect",
        M::Refract => "refract",
        M::Sign => "sign",
        M::Fma => "fma",
        M::Mix => "mix",
        M::Step => "step",
        M::SmoothStep => "smoothstep",
        M::Sqrt => "sqrt",
        M::InverseSqrt => "inverseSqrt",
        M::CountTrailingZeros => "countTrailingZeros",
        M::CountLeadingZeros => "countLeadingZeros",
        M::CountOneBits => "countOneBits",
        M::ReverseBits => "reverseBits",
        M::ExtractBits => "extractBits",
        M::InsertBits => "insertBits",
        M::FirstTrailingBit => "firstTrailingBit",
        M::FirstLeadingBit => "firstLeadingBit",
        M::Pack4x8snorm => "pack4x8snorm",
        M::Pack4x8unorm => "pack4x8unorm",
        M::Pack2x16snorm => "pack2x16snorm",
        M::Pack2x16unorm => "pack2x16unorm",
        M::Pack2x16float => "pack2x16float",
        M::Unpack4x8snorm => "unpack4x8snorm",
        M::Unpack4x8unorm => "unpack4x8unorm",
        M::Unpack2x16snorm => "unpack2x16snorm",
        M::Unpack2x16unorm => "unpack2x16unorm",
        M::Unpack2x16float => "unpack2x16float",
        _ => return None,
    })
}

/// How a walked statement list ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Normal,
    Returned,
}

impl<'a> Reader<'a> {
    // ---------------------------------------------------------------- values

    fn global_lit(&self, h: Handle<Expression>) -> R<E> {
        match &self.module.global_expressions[h] {
            Expression::Literal(l) => lit_of(l),
            Expression::Constant(c) => self.global_lit(self.module.constants[*c].init),
            Expression::ZeroValue(t) => zero_of(&self.module.types[*t].inner),
            o => gap(format!("the module constant {o:?}")),
        }
    }

    /// The binding a global names.
    fn binding_of(&self, g: Handle<naga::GlobalVariable>) -> R<Bound> {
        let gv = &self.module.global_variables[g];
        let Some(rb) = &gv.binding else { return gap(format!("the module-scope variable {:?} (workgroup memory)", gv.name.as_deref().unwrap_or("?"))) };
        let name = V24_BINDINGS.iter().find(|(b, _)| rb.group == 0 && *b == rb.binding).map(|(_, n)| *n).ok_or_else(|| ReadError::Out(format!("binding ({}, {}), which the contract's table does not list", rb.group, rb.binding)))?;
        if name == "cfg" {
            return Ok(Bound::Cfg);
        }
        let inner = &self.module.types[gv.ty].inner;
        let TypeInner::Array { base, .. } = inner else { return gap(format!("binding {name} of type {inner:?}")) };
        match &self.module.types[*base].inner {
            TypeInner::Vector { size: naga::VectorSize::Quad, scalar } if scalar.kind == ScalarKind::Float => Ok(Bound::Vec4(name.to_string())),
            TypeInner::Scalar(_) | TypeInner::Atomic(_) => Ok(Bound::Plain(name.to_string())),
            o => gap(format!("binding {name} with elements {o:?}")),
        }
    }

    /// The value of expression `h` in frame `fr`.
    fn ex(&mut self, fr: &Frame<'a>, h: Handle<Expression>) -> R<E> {
        if let Some(v) = fr.values.get(&h) {
            return Ok(v.clone());
        }
        let expr = &fr.f.expressions[h];
        Ok(match expr {
            Expression::Literal(l) => lit_of(l)?,
            Expression::Constant(c) => self.global_lit(self.module.constants[*c].init)?,
            Expression::ZeroValue(t) => zero_of(&self.module.types[*t].inner)?,
            Expression::FunctionArgument(i) => fr.args.get(*i as usize).cloned().ok_or_else(|| ReadError::Gap(format!("argument {i}")))?,
            Expression::Compose { components, .. } => {
                let w = self.wt_of(fr, h)?;
                let parts = components.iter().map(|c| self.ex(fr, *c)).collect::<R<Vec<_>>>()?;
                E::Compose(w, parts)
            }
            Expression::Splat { value, .. } => E::Splat(self.wt_of(fr, h)?, Box::new(self.ex(fr, *value)?)),
            Expression::Swizzle { size, vector, pattern } => {
                let n = *size as usize;
                let pat: String = pattern[..n].iter().map(|c| ['x', 'y', 'z', 'w'][*c as usize]).collect();
                E::Swz(Box::new(self.ex(fr, *vector)?), pat, self.wt_of(fr, h)?)
            }
            Expression::AccessIndex { base, index } => {
                if matches!(self.ty_of(fr, *base), TypeInner::Vector { .. }) {
                    E::Lane(Box::new(self.ex(fr, *base)?), *index as u8, self.wt_of(fr, h)?)
                } else {
                    return gap(format!("an access of {:?}", self.ty_of(fr, *base)));
                }
            }
            Expression::Access { base, index } => {
                if matches!(self.ty_of(fr, *base), TypeInner::Vector { .. }) {
                    E::DynLane(Box::new(self.ex(fr, *base)?), Box::new(self.ex(fr, *index)?), self.wt_of(fr, h)?)
                } else {
                    return gap(format!("a dynamic access of {:?}", self.ty_of(fr, *base)));
                }
            }
            Expression::Load { pointer } => self.load(fr, *pointer)?,
            Expression::Unary { op, expr } => E::Un(*op, Box::new(self.ex(fr, *expr)?), self.wt_of(fr, h)?),
            Expression::Binary { op, left, right } => E::Bin(*op, Box::new(self.ex(fr, *left)?), Box::new(self.ex(fr, *right)?), self.wt_of(fr, h)?),
            Expression::Select { condition, accept, reject } => E::Sel(Box::new(self.ex(fr, *reject)?), Box::new(self.ex(fr, *accept)?), Box::new(self.ex(fr, *condition)?), self.wt_of(fr, h)?),
            Expression::Relational { fun, argument } => {
                let name = match fun {
                    naga::RelationalFunction::All => "all",
                    naga::RelationalFunction::Any => "any",
                    o => return gap(format!("the relational function {o:?}")),
                };
                E::Call(name, vec![self.ex(fr, *argument)?], self.wt_of(fr, h)?)
            }
            Expression::Math { fun, arg, arg1, arg2, arg3 } => {
                let mut args = vec![self.ex(fr, *arg)?];
                for a in [arg1, arg2, arg3].into_iter().flatten() {
                    args.push(self.ex(fr, *a)?);
                }
                let w = self.wt_of(fr, h)?;
                if *fun == MathFunction::Clamp {
                    // WGSL: clamp(e, low, high) = min(max(e, low), high); read so only when low <= high is provable
                    let (lo, hi) = (args[1].clone(), args[2].clone());
                    let ordered = match (&lo, &hi) {
                        (E::Lit(Lit::F(a)), E::Lit(Lit::F(b))) => a <= b,
                        (E::Lit(Lit::U(a)), E::Lit(Lit::U(b))) => a <= b,
                        (E::Lit(Lit::I(a)), E::Lit(Lit::I(b))) => a <= b,
                        (E::Splat(_, a), E::Splat(_, b)) => match (a.as_ref(), b.as_ref()) {
                            (E::Lit(Lit::F(a)), E::Lit(Lit::F(b))) => a <= b,
                            (E::Lit(Lit::U(a)), E::Lit(Lit::U(b))) => a <= b,
                            (E::Lit(Lit::I(a)), E::Lit(Lit::I(b))) => a <= b,
                            _ => false,
                        },
                        _ => self.max_of(&lo).is_some_and(|m| self.min_of(&hi).is_some_and(|n| m <= n)),
                    };
                    if !ordered {
                        return gap(format!("clamp with bounds not provably ordered: clamp(_, {lo:?}, {hi:?})"));
                    }
                    let x = args.remove(0);
                    return Ok(E::Call("min", vec![E::Call("max", vec![x, lo], w), hi], w));
                }
                let name = math_name(*fun).ok_or_else(|| ReadError::Gap(format!("the built-in {fun:?}")))?;
                E::Call(name, args, w)
            }
            Expression::As { expr, kind: _, convert } => {
                let w = self.wt_of(fr, h)?;
                let x = Box::new(self.ex(fr, *expr)?);
                let int = |t: Wt| matches!(t.s, Sc::U32 | Sc::I32);
                // a bitcast between u32 and i32 is the conversion `u32(x)` / `i32(x)`: both reinterpret the bits
                if convert.is_some() || (int(w) && int(x.wt()) && w.lanes == x.wt().lanes) {
                    E::Cast(w, x)
                } else {
                    E::Bitcast(w, x)
                }
            }
            Expression::ArrayLength(p) => {
                let g = match &fr.f.expressions[*p] {
                    Expression::GlobalVariable(g) => *g,
                    o => return gap(format!("arrayLength of {o:?}")),
                };
                match self.binding_of(g)? {
                    Bound::Plain(n) => E::Len(n, false),
                    Bound::Vec4(n) => E::Len(n, true),
                    Bound::Cfg => return gap("arrayLength of the uniform block"),
                }
            }
            Expression::CallResult(_) | Expression::AtomicResult { .. } | Expression::WorkGroupUniformLoadResult { .. } => {
                return Err(ReadError::Gap(format!("our software has a bug: the result {h:?} read before its statement")))
            }
            o => return gap(format!("the expression {o:?}")),
        })
    }

    /// The workgroup variable a global is, registered on first sight.
    fn wg_var(&mut self, g: Handle<naga::GlobalVariable>) -> R<Option<usize>> {
        let gv = &self.module.global_variables[g];
        if gv.space != AddressSpace::WorkGroup {
            return Ok(None);
        }
        if let Some(v) = self.wg_of.get(&g) {
            return Ok(Some(*v));
        }
        let name = gv.name.clone().unwrap_or_default();
        let inner = &self.module.types[gv.ty].inner;
        let (elem_inner, len) = match inner {
            TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => (&self.module.types[*base].inner, n.get()),
            o => (o, 1),
        };
        let atomic = matches!(elem_inner, TypeInner::Atomic(_));
        let elem = Wt::of_inner(elem_inner).ok_or_else(|| ReadError::Gap(format!("a workgroup array {name} of {elem_inner:?}")))?;
        let ok = if atomic { elem == Wt::scalar(Sc::U32) } else { matches!((elem.s, elem.lanes), (Sc::U32 | Sc::F32 | Sc::I32, 1) | (Sc::F32, 4)) };
        if !ok {
            return gap(format!("a workgroup array {name} of {}{}", if atomic { "atomic " } else { "" }, elem.name()));
        }
        let v = self.wgvars.len();
        self.wgvars.push(WgVar { name, elem, len, atomic });
        self.wg_of.insert(g, v);
        Ok(Some(v))
    }

    /// The workgroup element a pointer names: (variable, the seed's index).
    fn wg_place(&mut self, fr: &Frame<'a>, p: Handle<Expression>) -> R<Option<(usize, E)>> {
        match &fr.f.expressions[p] {
            Expression::GlobalVariable(g) => Ok(self.wg_var(*g)?.map(|v| (v, E::Lit(Lit::U(0))))),
            Expression::Access { base, index } => match &fr.f.expressions[*base] {
                Expression::GlobalVariable(g) => match self.wg_var(*g)? {
                    Some(v) => Ok(Some((v, self.ex(fr, *index)?))),
                    None => Ok(None),
                },
                _ => Ok(None),
            },
            Expression::AccessIndex { base, index } => match &fr.f.expressions[*base] {
                Expression::GlobalVariable(g) => Ok(self.wg_var(*g)?.map(|v| (v, E::Lit(Lit::U(*index))))),
                _ => Ok(None),
            },
            _ => Ok(None),
        }
    }

    /// The local a pointer expression names: a local variable, or a pointer argument naming the caller's.
    fn local_of_ptr(&self, fr: &Frame<'a>, h: Handle<Expression>) -> Option<Local> {
        match &fr.f.expressions[h] {
            Expression::LocalVariable(l) => fr.locals.get(l).cloned(),
            Expression::FunctionArgument(i) => fr.ptr_args.get(i).cloned(),
            _ => None,
        }
    }

    /// A load through a pointer.
    fn load(&mut self, fr: &Frame<'a>, p: Handle<Expression>) -> R<E> {
        if let Some((v, idx)) = self.wg_place(fr, p)? {
            let w = self.wgvars[v].elem;
            return Ok(if self.wgvars[v].atomic { E::WgA(v, Box::new(idx)) } else { E::Wg(v, Box::new(idx), w) });
        }
        if let Some(l) = self.local_of_ptr(fr, p) {
            return match l {
                Local::Reg(r, w) => Ok(E::Reg(r, w)),
                Local::Counter { loop_id, start, op } => Ok(counter_value(loop_id, start, op)),
                Local::Known(e) => Ok(e),
                Local::Parr(..) => gap("a load of a whole private array"),
            };
        }
        match &fr.f.expressions[p] {
            Expression::LocalVariable(l) => match fr.locals.get(l) {
                Some(Local::Reg(r, w)) => Ok(E::Reg(*r, *w)),
                Some(Local::Counter { loop_id, start, op, .. }) => Ok(counter_value(*loop_id, start.clone(), *op)),
                Some(Local::Parr(..)) => gap("a load of a whole private array"),
                Some(Local::Known(e)) => Ok(e.clone()),
                None => gap("our software has a bug: a local with no register"),
            },
            Expression::Access { base, index } if self.local_of_ptr(fr, *base).is_some() => {
                let idx = self.ex(fr, *index)?;
                match self.local_of_ptr(fr, *base) {
                    Some(Local::Parr(pa, w)) => Ok(E::Parr(pa, Box::new(idx), w)),
                    Some(Local::Reg(r, w)) => {
                        let lane = Wt::scalar(w.s);
                        Ok(E::DynLane(Box::new(E::Reg(r, w)), Box::new(idx), lane))
                    }
                    _ => gap("a dynamic access of a loop counter"),
                }
            }
            Expression::Access { base, index } if matches!(&fr.f.expressions[*base], Expression::Access { .. } | Expression::AccessIndex { .. }) => {
                // a lane of an element: `keys4[i][l]`, `arr[i][l]`
                let elem = self.load(fr, *base)?;
                let w = elem.wt();
                if w.lanes == 1 {
                    return gap("a dynamic access of a scalar element");
                }
                let idx = self.ex(fr, *index)?;
                Ok(E::DynLane(Box::new(elem), Box::new(idx), Wt::scalar(w.s)))
            }
            Expression::Access { base, index } => match &fr.f.expressions[*base] {
                Expression::GlobalVariable(g) => {
                    let idx = self.ex(fr, *index)?;
                    match self.binding_of(*g)? {
                        Bound::Plain(n) => {
                            if role(&n) == Role::Rows {
                                return gap(format!("a read of the row output {n}"));
                            }
                            Ok(E::Load(n.clone(), Box::new(idx), if n == "index_u32" { Wt::scalar(Sc::U32) } else { Wt::scalar(Sc::F32) }))
                        }
                        Bound::Vec4(n) => Ok(E::LoadV4(n, Box::new(idx))),
                        Bound::Cfg => gap("a dynamic access of the uniform block"),
                    }
                }
                o => gap(format!("a load through {o:?}")),
            },
            Expression::AccessIndex { base, index } => match &fr.f.expressions[*base] {
                Expression::GlobalVariable(g) => match self.binding_of(*g)? {
                    Bound::Cfg => {
                        if (*index as usize) < self.fields.len() {
                            Ok(E::Field(*index as usize))
                        } else {
                            gap(format!("uniform field {index}"))
                        }
                    }
                    Bound::Plain(n) => Ok(E::Load(n.clone(), Box::new(E::Lit(Lit::U(*index))), if n == "index_u32" { Wt::scalar(Sc::U32) } else { Wt::scalar(Sc::F32) })),
                    Bound::Vec4(n) => Ok(E::LoadV4(n, Box::new(E::Lit(Lit::U(*index))))),
                },
                Expression::LocalVariable(_) | Expression::FunctionArgument(_) => match self.local_of_ptr(fr, *base) {
                    Some(Local::Reg(r, w)) => Ok(E::Lane(Box::new(E::Reg(r, w)), *index as u8, Wt::scalar(w.s))),
                    Some(Local::Parr(pa, w)) => Ok(E::Parr(pa, Box::new(E::Lit(Lit::U(*index))), w)),
                    _ => gap("a lane of a loop counter"),
                },
                Expression::Access { base: b2, index: i2 } => {
                    // a lane of an element: arr[i].x of a private vec array, keys4[i].x of a vec4 view
                    let elem = self.load(fr, *base)?;
                    let _ = (b2, i2);
                    let w = elem.wt();
                    Ok(E::Lane(Box::new(elem), *index as u8, Wt::scalar(w.s)))
                }
                o => gap(format!("a load through an index of {o:?}")),
            },
            Expression::GlobalVariable(g) => {
                let gv = &self.module.global_variables[*g];
                if gv.space == AddressSpace::WorkGroup {
                    return gap(format!("workgroup memory (var<workgroup> {})", gv.name.as_deref().unwrap_or("?")));
                }
                gap(format!("a load of the whole binding {:?}", gv.name))
            }
            o => gap(format!("a load through {o:?}")),
        }
    }

    // ---------------------------------------------------------------- value properties

    /// Whether a value is the same at every later point of its frame (a `let` of it may be read where it is used).
    fn stable(&self, e: &E) -> bool {
        match e {
            E::Lit(_) | E::Zero(_) | E::Bi(_) | E::Field(_) | E::Counter(_) | E::Len(..) => true,
            E::Reg(r, _) => self.ssa.contains(r),
            E::Parr(..) | E::Rmw(..) | E::Wg(..) | E::WgOwn(..) | E::WgA(..) | E::WgRmw(..) => false,
            E::UReg(_) => true,
            E::Load(n, i, _) => role(n) == Role::Input && self.stable(i),
            E::LoadV4(n, i) => role(n) == Role::Input && self.stable(i),
            E::Bin(_, a, b, _) => self.stable(a) && self.stable(b),
            E::Un(_, a, _) | E::Cast(_, a) | E::Bitcast(_, a) | E::Splat(_, a) | E::Lane(a, _, _) | E::Swz(a, _, _) => self.stable(a),
            E::DynLane(a, b, _) => self.stable(a) && self.stable(b),
            E::Sel(a, b, c, _) => self.stable(a) && self.stable(b) && self.stable(c),
            E::Call(_, v, _) | E::Compose(_, v) => v.iter().all(|x| self.stable(x)),
        }
    }

    /// An upper bound of a u32 value, from the contract's ranges, the workgroup size and literals; `None` when unknown.
    fn max_of(&self, e: &E) -> Option<u64> {
        const M: u64 = u32::MAX as u64;
        let r = match e {
            E::Lit(Lit::U(v)) => *v as u64,
            E::Field(i) => self.contract.fields.get(*i)?.1,
            E::Bi(Bi::Lidx) => ((self.wg[0] * self.wg[1] * self.wg[2]) as u64 - 1).min(self.lidx_cap.get().unwrap_or(u64::MAX)),
            E::Lane(b, i, _) => match b.as_ref() {
                E::Bi(Bi::Lid) if *i == 0 && self.wg[1] == 1 && self.wg[2] == 1 => (self.wg[0] as u64 - 1).min(self.lidx_cap.get().unwrap_or(u64::MAX)),
                E::Bi(Bi::Lid) => self.wg[*i as usize] as u64 - 1,
                E::Bi(Bi::Wid) => if *i == 0 { self.contract.workgroups_x - 1 } else { 0 },
                E::Bi(Bi::Nwg) => if *i == 0 { self.contract.workgroups_x } else { 1 },
                E::Bi(Bi::Gid) => if *i == 0 { self.contract.workgroups_x * self.wg[0] as u64 - 1 } else { self.wg[*i as usize] as u64 - 1 },
                _ => return None,
            },
            E::Reg(r, _) => *self.reg_max.get(r)?,
            E::Counter(id) => {
                let (n, s) = self.loop_bounds.get(id)?;
                (*n)?.saturating_sub(1) / (*s as u64).max(1) * (*s as u64).max(1)
            }
            E::Len(_, _) => return None,
            E::Bin(op, a, b, w) if w.s == Sc::U32 && w.lanes == 1 => {
                let (x, y) = (self.max_of(a), self.max_of(b));
                match op {
                    BinaryOperator::Add => x? + y?,
                    BinaryOperator::Multiply => x?.checked_mul(y?)?,
                    BinaryOperator::Subtract => x?,
                    BinaryOperator::Divide => x?,
                    BinaryOperator::Modulo => y?.saturating_sub(1).min(x.unwrap_or(M)),
                    BinaryOperator::ShiftRight => x?,
                    BinaryOperator::And => x.unwrap_or(M).min(y.unwrap_or(M)),
                    BinaryOperator::ShiftLeft => {
                        let s = self.max_of(b)?;
                        if s >= 32 {
                            return None;
                        }
                        x?.checked_shl(s as u32)?
                    }
                    _ => return None,
                }
            }
            E::Call("min", v, w) if w.s == Sc::U32 && w.lanes == 1 => v.iter().filter_map(|x| self.max_of(x)).min()?,
            E::Call("max", v, w) if w.s == Sc::U32 && w.lanes == 1 => {
                let m: Vec<Option<u64>> = v.iter().map(|x| self.max_of(x)).collect();
                m.into_iter().collect::<Option<Vec<u64>>>()?.into_iter().max()?
            }
            E::Call("countOneBits", _, _) => 32,
            E::Sel(a, b, _, w) if w.s == Sc::U32 && w.lanes == 1 => self.max_of(a)?.max(self.max_of(b)?),
            _ => return None,
        };
        if r > M {
            return None;
        }
        Some(r)
    }

    fn min_of(&self, e: &E) -> Option<u64> {
        match e {
            E::Lit(Lit::U(v)) => Some(*v as u64),
            _ => Some(0).filter(|_| matches!(e.wt(), Wt { s: Sc::U32, lanes: 1 })),
        }
    }

    // ---------------------------------------------------------------- statements

    /// Store `e` in a fresh register that only this statement writes, and return its value.
    fn ssa(&mut self, e: E, out: &mut Vec<St>) -> R<E> {
        let w = e.wt();
        let r = self.reg(w)?;
        if let Some(m) = self.max_of(&e) {
            self.reg_max.insert(r, m);
        }
        self.ssa.insert(r);
        self.push(out, St::Set(r, e));
        Ok(E::Reg(r, w))
    }

    /// The locals of a function: a register (or private array) each; their initial values set as the function begins.
    fn locals(&mut self, fr: &mut Frame<'a>, out: &mut Vec<St>, entry: bool) -> R<()> {
        for (h, lv) in fr.f.local_variables.iter() {
            let inner = &self.module.types[lv.ty].inner;
            match inner {
                TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => {
                    let w = Wt::of_inner(&self.module.types[*base].inner).ok_or_else(|| ReadError::Gap(format!("a private array of {:?}", self.module.types[*base].inner)))?;
                    let p = self.parr(w, n.get())?;
                    match lv.init {
                        Some(i) => {
                            // a constant initialiser: each element stored as the function begins
                            let parts = match &fr.f.expressions[i] {
                                Expression::Compose { components, .. } => components.clone(),
                                o => return gap(format!("a private array initialised by {o:?}")),
                            };
                            for (j, c) in parts.iter().enumerate() {
                                let v = self.ex(fr, *c)?;
                                self.push(out, St::ParrSet(p, E::Lit(Lit::U(j as u32)), v));
                            }
                        }
                        None if !entry => {
                            // a helper's array is zero at every call: a loop over its elements stores zero
                            let id = self.next_loop;
                            self.next_loop += 1;
                            self.loop_bounds.insert(id, (Some(n.get() as u64), 1));
                            let body = vec![St::ParrSet(p, E::Counter(id), zero_of_wt(w))];
                            self.push(out, St::Loop(Box::new(LoopSt { id, bound: E::Lit(Lit::U(n.get())), stride: 1, body })));
                        }
                        None => {}
                    }
                    fr.locals.insert(h, Local::Parr(p, w));
                }
                _ => {
                    let w = Wt::of_inner(inner).ok_or_else(|| ReadError::Gap(format!("a local of type {inner:?}")))?;
                    let r = self.reg(w)?;
                    match lv.init {
                        Some(i) => {
                            let v = self.ex(fr, i)?;
                            let zero = matches!(v, E::Lit(Lit::U(0)) | E::Lit(Lit::I(0)) | E::Lit(Lit::B(false))) || matches!(v, E::Lit(Lit::F(f)) if f.to_bits() == 0) || matches!(v, E::Zero(_));
                            if !(entry && zero) {
                                self.push(out, St::Set(r, v));
                            } else {
                                self.last_set.insert(r, v);
                            }
                        }
                        None if !entry => self.push(out, St::Set(r, zero_of_wt(w))),
                        None => {
                            self.last_set.insert(r, zero_of_wt(w));
                        }
                    }
                    fr.locals.insert(h, Local::Reg(r, w));
                }
            }
        }
        Ok(())
    }

    /// Walk a statement list. `in_loop`: the statements are a loop body's own (a `break` or `continue` may stand here).
    fn walk(&mut self, fr: &mut Frame<'a>, stmts: &'a [Statement], out: &mut Vec<St>, depth: usize) -> R<Flow> {
        let mut i = 0;
        while i < stmts.len() {
            let s = &stmts[i];
            match s {
                Statement::Emit(range) => {
                    for h in range.clone() {
                        if fr.named.contains(&h) && !fr.values.contains_key(&h) {
                            let v = self.ex(fr, h)?;
                            let v = if self.stable(&v) { v } else { self.ssa(v, out)? };
                            fr.values.insert(h, v);
                        }
                    }
                }
                Statement::Block(b) => {
                    if self.walk(fr, b, out, depth)? == Flow::Returned {
                        return Ok(Flow::Returned);
                    }
                }
                Statement::If { condition, accept, reject } if self.short_circuit(fr, *condition, accept, reject)?.is_some() => {
                    // naga's lowering of `a || b` / `a && b`: a temporary set in both branches; it is the value `a || b`
                    if let Some((l, e)) = self.short_circuit(fr, *condition, accept, reject)? {
                        if self.stable(&e) {
                            fr.locals.insert(l, Local::Known(e));
                        } else {
                            let Some(Local::Reg(r, _)) = fr.locals.get(&l).cloned() else { return gap("our software has a bug: a short-circuit temporary with no register") };
                            self.ssa.remove(&r);
                            self.push(out, St::Set(r, e));
                        }
                    }
                }
                Statement::If { condition, accept, reject } => {
                    let c = self.ex(fr, *condition)?;
                    let saved = self.last_set.clone();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    let fa = self.walk(fr, accept, &mut a, depth)?;
                    let after_a = std::mem::replace(&mut self.last_set, saved.clone());
                    let fb = self.walk(fr, reject, &mut b, depth)?;
                    if fa == Flow::Returned && fb == Flow::Returned {
                        self.last_set = saved;
                        self.push(out, St::If(c, a, b));
                        return Ok(Flow::Returned);
                    }
                    if fa == Flow::Returned || fb == Flow::Returned {
                        // `if (c) { A; return; } rest` is `if (c) { A } else { B; rest }`
                        if fb == Flow::Returned {
                            self.last_set = after_a;
                        }
                        let rest = &stmts[i + 1..];
                        let tail = if fa == Flow::Returned { &mut b } else { &mut a };
                        let fr2 = self.walk(fr, rest, tail, depth)?;
                        self.last_set = saved;
                        self.push(out, St::If(c, a, b));
                        return Ok(if fr2 == Flow::Returned { Flow::Returned } else { Flow::Normal });
                    }
                    self.last_set = saved;
                    self.push(out, St::If(c, a, b));
                }
                Statement::Loop { body, continuing, break_if } => {
                    if break_if.is_some() {
                        return gap("a loop with a break-if continuing block");
                    }
                    self.counted_loop(fr, body, continuing, out, depth)?;
                }
                Statement::Break | Statement::Continue => return gap("a break or continue outside a loop body's own statements"),
                Statement::Return { value } => {
                    if let Some(v) = value {
                        let e = self.ex(fr, *v)?;
                        let (r, _) = fr.ret.ok_or_else(|| ReadError::Gap("a return value with no result register".into()))?;
                        self.push(out, St::Set(r, e));
                    }
                    return Ok(Flow::Returned);
                }
                Statement::Store { pointer, value } => {
                    let v = self.ex(fr, *value)?;
                    self.store(fr, *pointer, v, out)?;
                }
                Statement::Call { function, arguments, result } => {
                    if depth >= MAX_INLINE_DEPTH {
                        return gap("calls nested deeper than the inliner's limit");
                    }
                    let callee = &self.module.functions[*function];
                    let mut args = Vec::new();
                    let mut ptr_args = HashMap::new();
                    for (ai, a) in arguments.iter().enumerate() {
                        if matches!(self.ty_of(fr, *a), TypeInner::Pointer { .. }) {
                            let l = self.local_of_ptr(fr, *a).ok_or_else(|| ReadError::Gap("a pointer argument that is not a local variable".into()))?;
                            ptr_args.insert(ai as u32, l);
                            args.push(E::Lit(Lit::U(0)));
                            continue;
                        }
                        let e = self.ex(fr, *a)?;
                        args.push(if self.stable(&e) { e } else { self.ssa(e, out)? });
                    }
                    let ret = match result {
                        Some(h) => {
                            let w = self.wt_of(fr, *h)?;
                            Some((self.reg(w)?, w))
                        }
                        None => None,
                    };
                    let info: &'a naga::valid::ModuleInfo = self.info;
                    let mut cf = Frame { f: callee, info: &info[*function], args, ptr_args, locals: HashMap::new(), values: HashMap::new(), named: callee.named_expressions.keys().copied().collect(), ret };
                    self.locals(&mut cf, out, false)?;
                    self.walk(&mut cf, &callee.body, out, depth + 1)?;
                    if let (Some(h), Some((r, w))) = (result, ret) {
                        // the result register is written at every return of the callee: one value a call
                        fr.values.insert(*h, E::Reg(r, w));
                    }
                }
                Statement::Atomic { pointer, fun, value, result } => {
                    let f = atomic_fun(fun)?;
                    let rmw = match self.wg_place(fr, *pointer)? {
                        Some((var, idx)) => {
                            let v = self.ex(fr, *value)?;
                            E::WgRmw(f, var, Box::new(idx), Box::new(v))
                        }
                        None => {
                            let (b, idx) = self.arena_cell(fr, *pointer)?;
                            let v = self.ex(fr, *value)?;
                            E::Rmw(f, b, Box::new(idx), Box::new(v))
                        }
                    };
                    match result {
                        Some(h) => {
                            let r = self.reg(Wt::scalar(Sc::U32))?;
                            self.ssa.insert(r);
                            self.push(out, St::Set(r, rmw));
                            fr.values.insert(*h, E::Reg(r, Wt::scalar(Sc::U32)));
                        }
                        None => out.push(St::Discard(rmw)),
                    }
                }
                Statement::ControlBarrier(b) => {
                    if b.contains(naga::Barrier::WORK_GROUP) {
                        out.push(St::Barrier);
                    }
                    if b.contains(naga::Barrier::STORAGE) {
                        out.push(St::StorageBarrier);
                    }
                    if b.contains(naga::Barrier::SUB_GROUP) || b.contains(naga::Barrier::TEXTURE) {
                        return gap(format!("the barrier {b:?}"));
                    }
                }
                Statement::MemoryBarrier(b) => return gap(format!("the memory barrier {b:?}")),
                Statement::WorkGroupUniformLoad { pointer, result } => {
                    let (var, idx) = self.wg_place(fr, *pointer)?.ok_or_else(|| ReadError::Gap("workgroupUniformLoad of a pointer that is not a workgroup element".into()))?;
                    if self.wgvars[var].atomic || self.wgvars[var].elem != Wt::scalar(Sc::U32) {
                        return gap(format!("workgroupUniformLoad of {}", self.wgvars[var].name));
                    }
                    let n = self.next_ureg;
                    if n >= 8 {
                        return gap("a ninth uniform register");
                    }
                    self.next_ureg += 1;
                    self.push(out, St::ULoad(n, var, idx));
                    fr.values.insert(*result, E::UReg(n));
                }
                o => return gap(format!("the statement {o:?}")),
            }
            i += 1;
        }
        Ok(Flow::Normal)
    }

    /// naga's lowering of `a || b` (`if (a) { t = true; } else { t = b; }`) and `a && b` (`if (a) { t = b; } else { t =
    /// false; }`) into an unnamed temporary: the temporary and its value.
    fn short_circuit(&mut self, fr: &Frame<'a>, c: Handle<Expression>, accept: &'a Block, reject: &'a Block) -> R<Option<(Handle<LocalVariable>, E)>> {
        let store = |b: &Block| -> Option<(Handle<LocalVariable>, Handle<Expression>)> {
            let real: Vec<&Statement> = b.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
            match real.as_slice() {
                [Statement::Store { pointer, value }] => match fr.f.expressions[*pointer] {
                    Expression::LocalVariable(l) if fr.f.local_variables[l].name.is_none() => Some((l, *value)),
                    _ => None,
                },
                _ => None,
            }
        };
        let (Some((la, va)), Some((lb, vb))) = (store(accept), store(reject)) else { return Ok(None) };
        if la != lb || !matches!(fr.locals.get(&la), Some(Local::Reg(..))) {
            return Ok(None);
        }
        let lit = |h: Handle<Expression>| match fr.f.expressions[h] {
            Expression::Literal(Literal::Bool(b)) => Some(b),
            _ => None,
        };
        let bw = Wt::scalar(Sc::Bool);
        let not = |e: E| E::Un(UnaryOperator::LogicalNot, Box::new(e), bw);
        let e = match (lit(va), lit(vb)) {
            (Some(true), _) => E::Bin(BinaryOperator::LogicalOr, Box::new(self.ex(fr, c)?), Box::new(self.ex(fr, vb)?), bw),
            (_, Some(false)) => E::Bin(BinaryOperator::LogicalAnd, Box::new(self.ex(fr, c)?), Box::new(self.ex(fr, va)?), bw),
            (_, Some(true)) => E::Bin(BinaryOperator::LogicalOr, Box::new(not(self.ex(fr, c)?)), Box::new(self.ex(fr, va)?), bw),
            (Some(false), _) => E::Bin(BinaryOperator::LogicalAnd, Box::new(not(self.ex(fr, c)?)), Box::new(self.ex(fr, vb)?), bw),
            _ => return Ok(None),
        };
        Ok(Some((la, e)))
    }

    /// The arena cell an atomic names.
    fn arena_cell(&mut self, fr: &Frame<'a>, p: Handle<Expression>) -> R<(String, E)> {
        match &fr.f.expressions[p] {
            Expression::Access { base, index } => match &fr.f.expressions[*base] {
                Expression::GlobalVariable(g) => match self.binding_of(*g)? {
                    Bound::Plain(n) if n == "index_u32" => Ok((n, self.ex(fr, *index)?)),
                    o => gap(format!("an atomic on {o:?}")),
                },
                o => gap(format!("an atomic through {o:?}")),
            },
            Expression::AccessIndex { base, index } => match &fr.f.expressions[*base] {
                Expression::GlobalVariable(g) => match self.binding_of(*g)? {
                    Bound::Plain(n) if n == "index_u32" => Ok((n, E::Lit(Lit::U(*index)))),
                    o => gap(format!("an atomic on {o:?}")),
                },
                o => gap(format!("an atomic through {o:?}")),
            },
            Expression::GlobalVariable(g) => {
                let gv = &self.module.global_variables[*g];
                gap(format!("an atomic on workgroup memory (var<workgroup> {})", gv.name.as_deref().unwrap_or("?")))
            }
            o => gap(format!("an atomic through {o:?}")),
        }
    }

    fn store(&mut self, fr: &mut Frame<'a>, p: Handle<Expression>, v: E, out: &mut Vec<St>) -> R<()> {
        if let Some((var, idx)) = self.wg_place(fr, p)? {
            let st = if self.wgvars[var].atomic { St::WgAStore(var, idx, v) } else { St::WgSet(var, idx, v) };
            self.push(out, st);
            return Ok(());
        }
        if let Some(l) = self.local_of_ptr(fr, p) {
            return match l {
                Local::Reg(r, _) => {
                    self.ssa.remove(&r);
                    self.reg_max.remove(&r);
                    self.push(out, St::Set(r, v));
                    Ok(())
                }
                Local::Counter { .. } => Err(ReadError::Out("a store to a loop's counter inside its body (the loop's trip count is not the bound's)".into())),
                Local::Known(_) => gap("a second store to a short-circuit temporary"),
                Local::Parr(..) => gap("a store of a whole private array"),
            };
        }
        let base_idx = match &fr.f.expressions[p] {
            Expression::Access { base, index } => Some((*base, Some(*index), 0)),
            Expression::AccessIndex { base, index } => Some((*base, None, *index)),
            _ => None,
        };
        if let Some((base, dyn_index, const_index)) = base_idx {
            if let Some(l) = self.local_of_ptr(fr, base) {
                let idx = match dyn_index {
                    Some(h) => self.ex(fr, h)?,
                    None => E::Lit(Lit::U(const_index)),
                };
                return match l {
                    Local::Parr(pa, _) => {
                        self.push(out, St::ParrSet(pa, idx, v));
                        Ok(())
                    }
                    Local::Reg(r, _) => {
                        self.ssa.remove(&r);
                        self.push(out, St::SetLane(r, idx, v));
                        Ok(())
                    }
                    Local::Counter { .. } | Local::Known(_) => gap("a store into a loop counter or a short-circuit temporary"),
                };
            }
        }
        match &fr.f.expressions[p] {
            Expression::LocalVariable(l) => match fr.locals.get(l) {
                Some(Local::Reg(r, _)) => {
                    let r = *r;
                    self.ssa.remove(&r);
                    self.reg_max.remove(&r);
                    self.push(out, St::Set(r, v));
                    Ok(())
                }
                Some(Local::Counter { .. }) => Err(ReadError::Out("a store to a loop's counter inside its body (the loop's trip count is not the bound's)".into())),
                _ => gap("a store of a whole private array"),
            },
            Expression::Access { base, index } => match &fr.f.expressions[*base] {
                Expression::LocalVariable(l) => {
                    let idx = self.ex(fr, *index)?;
                    match fr.locals.get(l) {
                        Some(Local::Parr(pa, _)) => {
                            out.push(St::ParrSet(*pa, idx, v));
                            Ok(())
                        }
                        Some(Local::Reg(r, _)) => {
                            let r = *r;
                            self.ssa.remove(&r);
                            self.push(out, St::SetLane(r, idx, v));
                            Ok(())
                        }
                        _ => gap("a store into a loop counter"),
                    }
                }
                Expression::GlobalVariable(g) => {
                    let idx = self.ex(fr, *index)?;
                    match self.binding_of(*g)? {
                        Bound::Plain(n) => {
                            if role(&n) == Role::Input {
                                return Err(ReadError::Out(format!("a store into the read-only binding {n}")));
                            }
                            out.push(St::Store(n, idx, v));
                            Ok(())
                        }
                        Bound::Vec4(n) => gap(format!("a vec4 store into {n}")),
                        Bound::Cfg => Err(ReadError::Out("a store into the uniform block".into())),
                    }
                }
                o => gap(format!("a store through {o:?}")),
            },
            Expression::AccessIndex { base, index } => match &fr.f.expressions[*base] {
                Expression::LocalVariable(l) => match fr.locals.get(l) {
                    Some(Local::Reg(r, _)) => {
                        let r = *r;
                        self.ssa.remove(&r);
                        self.push(out, St::SetLane(r, E::Lit(Lit::U(*index)), v));
                        Ok(())
                    }
                    Some(Local::Parr(pa, _)) => {
                        out.push(St::ParrSet(*pa, E::Lit(Lit::U(*index)), v));
                        Ok(())
                    }
                    _ => gap("a store into a lane of a loop counter"),
                },
                Expression::GlobalVariable(g) => match self.binding_of(*g)? {
                    Bound::Plain(n) if role(&n) != Role::Input => {
                        out.push(St::Store(n, E::Lit(Lit::U(*index)), v));
                        Ok(())
                    }
                    o => gap(format!("a store into {o:?}")),
                },
                o => gap(format!("a store through an index of {o:?}")),
            },
            Expression::GlobalVariable(g) => {
                let gv = &self.module.global_variables[*g];
                gap(format!("workgroup memory (var<workgroup> {})", gv.name.as_deref().unwrap_or("?")))
            }
            o => gap(format!("a store through {o:?}")),
        }
    }
}

fn lit_of(l: &Literal) -> R<E> {
    Ok(E::Lit(match l {
        Literal::U32(v) => Lit::U(*v),
        Literal::I32(v) => Lit::I(*v),
        Literal::F32(v) => Lit::F(*v),
        Literal::Bool(v) => Lit::B(*v),
        Literal::AbstractInt(v) => Lit::I(i32::try_from(*v).map_err(|_| ReadError::Gap(format!("the abstract integer {v}")))?),
        Literal::AbstractFloat(v) => Lit::F(*v as f32),
        o => return gap(format!("the literal {o:?}")),
    }))
}

fn zero_of_wt(w: Wt) -> E {
    if w.lanes > 1 {
        return E::Zero(w);
    }
    E::Lit(match w.s {
        Sc::F32 | Sc::F16 => Lit::F(0.0),
        Sc::U32 => Lit::U(0),
        Sc::I32 => Lit::I(0),
        Sc::Bool => Lit::B(false),
    })
}

fn zero_of(t: &TypeInner) -> R<E> {
    let w = Wt::of_inner(t).ok_or_else(|| ReadError::Gap(format!("the zero value of {t:?}")))?;
    Ok(zero_of_wt(w))
}

/// The seed's loop variable inside (and right after) its loop.
fn counter_value(loop_id: usize, start: Option<E>, op: CounterOp) -> E {
    let k = E::Counter(loop_id);
    let u = Wt::scalar(Sc::U32);
    match (start, op) {
        (None, _) => k,
        (Some(s), CounterOp::Add) => E::Bin(BinaryOperator::Add, Box::new(s), Box::new(k), u),
        (Some(s), CounterOp::Shr) => E::Bin(BinaryOperator::ShiftRight, Box::new(s), Box::new(k), u),
        (Some(s), CounterOp::Shl) => E::Bin(BinaryOperator::ShiftLeft, Box::new(s), Box::new(k), u),
    }
}

/// A statement list with its nested blocks flattened (a `{ ... }` block is its statements).
fn flat(b: &Block) -> Vec<&Statement> {
    let mut out = Vec::new();
    for s in b.iter() {
        match s {
            Statement::Block(inner) => out.extend(flat(inner)),
            o => out.push(o),
        }
    }
    out
}

fn has_jump(b: &Block) -> bool {
    b.iter().any(|s| match s {
        Statement::Break | Statement::Continue => true,
        Statement::Block(x) => has_jump(x),
        Statement::If { accept, reject, .. } => has_jump(accept) || has_jump(reject),
        _ => false,
    })
}

fn negate(op: BinaryOperator) -> Option<BinaryOperator> {
    use BinaryOperator as B;
    Some(match op {
        B::Less => B::GreaterEqual,
        B::GreaterEqual => B::Less,
        B::LessEqual => B::Greater,
        B::Greater => B::LessEqual,
        B::Equal => B::NotEqual,
        B::NotEqual => B::Equal,
        _ => return None,
    })
}

fn flip(op: BinaryOperator) -> Option<BinaryOperator> {
    use BinaryOperator as B;
    Some(match op {
        B::Less => B::Greater,
        B::Greater => B::Less,
        B::LessEqual => B::GreaterEqual,
        B::GreaterEqual => B::LessEqual,
        B::Equal => B::Equal,
        B::NotEqual => B::NotEqual,
        _ => return None,
    })
}

/// The registers a statement list writes.
fn written(sts: &[St], out: &mut BTreeSet<(usize, usize)>) {
    for s in sts {
        match s {
            St::Set(r, _) | St::SetLane(r, _, _) => {
                out.insert((r.kind, r.n));
            }
            St::If(_, a, b) => {
                written(a, out);
                written(b, out);
            }
            St::Loop(l) => written(&l.body, out),
            _ => {}
        }
    }
}

fn lit_u(e: &E) -> Option<u32> {
    match e {
        E::Lit(Lit::U(v)) => Some(*v),
        E::Lit(Lit::I(v)) if *v >= 0 => Some(*v as u32),
        _ => None,
    }
}

impl<'a> Reader<'a> {
    /// Forget what is known of every register the statements write (after a branch or a loop).
    fn forget(&mut self, sts: &[St]) {
        let mut w = BTreeSet::new();
        written(sts, &mut w);
        self.last_set.retain(|r, _| !w.contains(&(r.kind, r.n)));
    }

    fn push(&mut self, out: &mut Vec<St>, s: St) {
        match &s {
            St::Set(r, e) => {
                if self.stable(e) {
                    self.last_set.insert(*r, e.clone());
                } else {
                    self.last_set.remove(r);
                }
            }
            St::SetLane(r, _, _) => {
                self.last_set.remove(r);
            }
            St::If(_, a, b) => {
                self.forget(a);
                self.forget(b);
            }
            St::Loop(l) => {
                let b = l.body.clone();
                self.forget(&b);
            }
            _ => {}
        }
        out.push(s);
    }

    /// A counted loop: `for (var i = a; i OP n; i = i STEP s)` (or the `while` form with the step last), read as the
    /// kingdom's counted loop over its trip bound.
    fn counted_loop(&mut self, fr: &mut Frame<'a>, body: &'a Block, continuing: &'a Block, out: &mut Vec<St>, depth: usize) -> R<()> {
        let stmts = flat(body);
        let mut j = 0;
        while matches!(stmts.get(j), Some(Statement::Emit(_))) {
            j += 1;
        }
        let (cond_h, negated) = match stmts.get(j) {
            Some(Statement::If { condition, accept, reject }) if accept.is_empty() && matches!(&reject[..], [Statement::Break]) => (*condition, false),
            Some(Statement::If { condition, accept, reject }) if reject.is_empty() && matches!(&accept[..], [Statement::Break]) => (*condition, true),
            _ => return Err(ReadError::Out("a loop with no trip condition at its head: not bounded by cfg values or constants (contract: use loops bounded by cfg values or by constants)".into())),
        };
        let local_of = |h: Handle<Expression>| match &fr.f.expressions[h] {
            Expression::Load { pointer } => match &fr.f.expressions[*pointer] {
                Expression::LocalVariable(v) => Some(*v),
                _ => None,
            },
            _ => None,
        };
        let (op, l, r) = match &fr.f.expressions[cond_h] {
            Expression::Binary { op, left, right } => (*op, *left, *right),
            o => return gap(format!("a loop condition that is not a comparison: {o:?}")),
        };
        let (var, limit_h, op) = if let Some(v) = local_of(l) {
            (v, r, op)
        } else if let Some(v) = local_of(r) {
            (v, l, flip(op).ok_or_else(|| ReadError::Gap(format!("the loop condition {op:?}")))?)
        } else {
            return gap("a loop condition that compares no local variable");
        };
        let op = if negated { negate(op).ok_or_else(|| ReadError::Gap(format!("the loop condition {op:?}")))? } else { op };
        // the step: the continuing block's store of the variable, or the body's last statement
        let step_of = |s: &Statement| -> Option<(BinaryOperator, Handle<Expression>)> {
            match s {
                Statement::Store { pointer, value } if matches!(fr.f.expressions[*pointer], Expression::LocalVariable(v) if v == var) => match &fr.f.expressions[*value] {
                    Expression::Binary { op, left, right } if local_of(*left) == Some(var) => Some((*op, *right)),
                    _ => None,
                },
                _ => None,
            }
        };
        let cont = flat(continuing);
        let cont_real: Vec<&&Statement> = cont.iter().filter(|s| !matches!(s, Statement::Emit(_))).collect();
        let (step_op, step_h, body_end) = if cont_real.len() == 1 {
            let (o, h) = step_of(cont_real[0]).ok_or_else(|| ReadError::Gap("a loop whose continuing block is not its counter's step".into()))?;
            (o, h, stmts.len())
        } else if cont_real.is_empty() {
            let last = stmts.iter().rposition(|s| !matches!(s, Statement::Emit(_))).ok_or_else(|| ReadError::Gap("an empty loop".into()))?;
            let (o, h) = step_of(stmts[last]).ok_or_else(|| ReadError::Out("a loop whose counter has no step: not bounded by cfg values or constants".into()))?;
            if stmts[j + 1..last].iter().any(|s| matches!(s, Statement::Continue) || matches!(s, Statement::If { accept, reject, .. } if has_jump(accept) || has_jump(reject)) && !matches!(s, Statement::If { .. } if false)) {
                // a continue would skip the seed's step, the kingdom's loop steps anyway
                if stmts[j + 1..last].iter().any(|s| matches!(s, Statement::Continue)) {
                    return gap("a continue in a loop whose step is the body's last statement");
                }
            }
            (o, h, last)
        } else {
            return gap("a loop whose continuing block does more than step its counter");
        };
        let step = self.ex(fr, step_h)?;
        let Some(Local::Reg(reg, w)) = fr.locals.get(&var).cloned() else { return gap("a loop counter that is not a register") };
        if w != Wt::scalar(Sc::U32) && w != Wt::scalar(Sc::I32) {
            return gap(format!("a loop counter of type {}", w.name()));
        }
        if w.s == Sc::I32 {
            return gap("an i32 loop counter");
        }
        let start = self.last_set.get(&reg).cloned().unwrap_or(E::Reg(reg, w));
        let limit = self.ex(fr, limit_h)?;
        if !self.stable(&limit) {
            return gap("a loop bound that is not the same on every trip (it reads mutable state)");
        }
        let u = Wt::scalar(Sc::U32);
        let (bound, stride, cstart, cop) = match (step_op, op, lit_u(&step)) {
            (BinaryOperator::Add, BinaryOperator::Less, Some(s)) if s >= 1 => {
                let (bound, cstart) = match (lit_u(&start), lit_u(&limit)) {
                    (Some(0), _) => (limit.clone(), None),
                    (Some(a), Some(n)) => (E::Lit(Lit::U(n.saturating_sub(a))), Some(start.clone())),
                    _ => (E::Call("sub_floor", vec![limit.clone(), start.clone()], u), Some(start.clone())),
                };
                (bound, s, cstart, CounterOp::Add)
            }
            (BinaryOperator::Add, BinaryOperator::LessEqual, Some(s)) if s >= 1 => match (lit_u(&start), lit_u(&limit)) {
                (Some(a), Some(n)) if n < u32::MAX => (E::Lit(Lit::U((n + 1).saturating_sub(a))), s, if a == 0 { None } else { Some(start.clone()) }, CounterOp::Add),
                _ => return gap("a loop `i <= n` whose bounds are not literals"),
            },
            (BinaryOperator::ShiftRight, _, Some(1)) | (BinaryOperator::Divide, _, Some(2)) => {
                let (Some(c), Some(lim)) = (lit_u(&start), lit_u(&limit)) else { return gap("a halving loop whose start or limit is not a literal") };
                let keep = |v: u32| match op {
                    BinaryOperator::Greater => v > lim,
                    BinaryOperator::GreaterEqual => v >= lim,
                    BinaryOperator::NotEqual => v != lim,
                    _ => false,
                };
                if !matches!(op, BinaryOperator::Greater | BinaryOperator::GreaterEqual | BinaryOperator::NotEqual) {
                    return gap(format!("a halving loop with the condition {op:?}"));
                }
                let mut t = 0u32;
                while t < 32 && keep(c >> t) {
                    t += 1;
                }
                if t >= 32 {
                    return Err(ReadError::Out("a halving loop that never ends".into()));
                }
                (E::Lit(Lit::U(t)), 1, Some(start.clone()), CounterOp::Shr)
            }
            (BinaryOperator::ShiftLeft, _, Some(1)) | (BinaryOperator::Multiply, _, Some(2)) => {
                let (Some(c), Some(lim)) = (lit_u(&start), lit_u(&limit)) else { return gap("a doubling loop whose start or limit is not a literal") };
                if c == 0 || !matches!(op, BinaryOperator::Less | BinaryOperator::LessEqual) {
                    return gap(format!("a doubling loop from {c} with the condition {op:?}"));
                }
                let mut t = 0u32;
                while t < 32 && (c as u64) << t <= u32::MAX as u64 && (if op == BinaryOperator::Less { ((c as u64) << t) < lim as u64 } else { ((c as u64) << t) <= lim as u64 }) {
                    t += 1;
                }
                (E::Lit(Lit::U(t)), 1, Some(start.clone()), CounterOp::Shl)
            }
            o => return gap(format!("a counted loop of the form {o:?}")),
        };
        let id = self.next_loop;
        self.next_loop += 1;
        self.loop_bounds.insert(id, (self.max_of(&bound), stride));
        fr.locals.insert(var, Local::Counter { loop_id: id, start: cstart.clone(), op: cop });
        let mut b = Vec::new();
        let saved = self.last_set.clone();
        let res = self.loop_body(fr, &stmts[j + 1..body_end], &mut b, depth);
        self.last_set = saved;
        fr.locals.insert(var, Local::Reg(reg, w));
        res?;
        self.push(out, St::Loop(Box::new(LoopSt { id, bound, stride, body: b })));
        // after the loop the seed's variable holds its final value
        self.ssa.remove(&reg);
        self.reg_max.remove(&reg);
        self.push(out, St::Set(reg, counter_value(id, cstart, cop)));
        Ok(())
    }

    /// A loop body's own statements: one `break` point (`if (c) { X; break; }`) and `continue`s at this level.
    fn loop_body(&mut self, fr: &mut Frame<'a>, stmts: &[&'a Statement], out: &mut Vec<St>, depth: usize) -> R<()> {
        let mut i = 0;
        let mut broke = false;
        let bool_w = Wt::scalar(Sc::Bool);
        while i < stmts.len() {
            match stmts[i] {
                Statement::If { condition, accept, reject } if has_jump(accept) || has_jump(reject) => {
                    let jump_last = |b: &Block| matches!(b.last(), Some(Statement::Break | Statement::Continue)) && !b[..b.len() - 1].iter().any(|s| matches!(s, Statement::Break | Statement::Continue) || matches!(s, Statement::If { accept, reject, .. } if has_jump(accept) || has_jump(reject)));
                    let c = self.ex(fr, *condition)?;
                    let (taken, other, cond) = if jump_last(accept) && !has_jump(reject) {
                        (accept, reject, c)
                    } else if jump_last(reject) && !has_jump(accept) {
                        (reject, accept, E::Un(UnaryOperator::LogicalNot, Box::new(c), bool_w))
                    } else {
                        return gap("a break or continue nested inside conditionals of a loop body");
                    };
                    let is_break = matches!(taken.last(), Some(Statement::Break));
                    let flag = self.reg(bool_w)?;
                    self.push(out, St::Set(flag, cond));
                    self.ssa.insert(flag);
                    let fv = E::Reg(flag, bool_w);
                    let saved = self.last_set.clone();
                    let mut a = Vec::new();
                    if self.walk(fr, &taken[..taken.len() - 1], &mut a, depth)? == Flow::Returned {
                        return gap("a return inside a loop");
                    }
                    self.last_set = saved.clone();
                    let mut o = Vec::new();
                    if self.walk(fr, other, &mut o, depth)? == Flow::Returned {
                        return gap("a return inside a loop");
                    }
                    if is_break {
                        self.last_set = saved;
                        if broke {
                            // a second break point: `A; b1 = c1; if (b1) {X1}; B; b2 = c2; if (b2) {X2}; break if b1 || b2`
                            // is `A; b1 = c1; if (b1) {X1}; b2 = false; if (!b1) { B; b2 = c2; if (b2) {X2} }; break if
                            // b1 || b2`: B and the second test run only when the first did not break
                            let p = out.iter().rposition(|x| matches!(x, St::BreakIf(_))).ok_or_else(|| ReadError::Gap("our software has a bug: the first break point is lost".into()))?;
                            let tail = out.split_off(p + 1);
                            let Some(St::BreakIf(e1)) = out.pop() else { return gap("our software has a bug: the first break point is lost") };
                            // the second flag is written twice, so it is not a let's register
                            self.ssa.remove(&flag);
                            // the tail ends in the second flag's `b2 = c2`
                            let mut inner = tail;
                            inner.push(St::If(fv.clone(), a, o));
                            out.push(St::Set(flag, E::Lit(Lit::B(false))));
                            out.push(St::If(E::Un(UnaryOperator::LogicalNot, Box::new(e1.clone()), bool_w), inner, Vec::new()));
                            self.last_set.remove(&flag);
                            out.push(St::BreakIf(E::Bin(BinaryOperator::LogicalOr, Box::new(e1), Box::new(fv), bool_w)));
                        } else {
                            self.push(out, St::If(fv.clone(), a, o));
                            broke = true;
                            out.push(St::BreakIf(fv));
                        }
                    } else {
                        // `if (c) { X; continue; } rest` is `if (c) { X } else { O; rest }`
                        self.loop_body(fr, &stmts[i + 1..], &mut o, depth)?;
                        if o.iter().any(|s| matches!(s, St::BreakIf(_))) {
                            return gap("a break after a continue in a loop body");
                        }
                        self.last_set = saved;
                        self.push(out, St::If(fv, a, o));
                        return Ok(());
                    }
                }
                Statement::Break => {
                    if broke {
                        return gap("a loop body with two break points");
                    }
                    out.push(St::BreakIf(E::Lit(Lit::B(true))));
                    return Ok(());
                }
                Statement::Continue => return Ok(()),
                s => {
                    if self.walk(fr, std::slice::from_ref(s), out, depth)? == Flow::Returned {
                        return gap("a return inside a loop");
                    }
                }
            }
            i += 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Phase 2: spelling the abstract kernel in the kingdom's rows
// ---------------------------------------------------------------------------------------------------------------------

/// Whether a constant text of `C_T` is the literal's value.
fn lit_text_is(text: &str, l: &Lit) -> bool {
    let t = text.trim();
    match l {
        Lit::U(v) => t.trim_end_matches('u').parse::<u64>().ok() == Some(*v as u64),
        Lit::I(v) => {
            if t == "(-2147483647i - 1i)" {
                return *v == i32::MIN;
            }
            t.trim_end_matches('i').parse::<i64>().ok() == Some(*v as i64)
        }
        Lit::F(v) => t.trim_end_matches('f').parse::<f64>().ok().map(|x| (x as f32).to_bits()) == Some(v.to_bits()),
        Lit::B(b) => t == if *b { "true" } else { "false" },
    }
}

/// WGSL text of a literal, spelled as the interface's constants are (`extra_constants`).
fn lit_text(l: &Lit) -> String {
    match l {
        Lit::U(v) => v.to_string(),
        Lit::I(v) => format!("{v}i"),
        Lit::F(v) => {
            let s = format!("{v:?}");
            s
        }
        Lit::B(b) => b.to_string(),
    }
}

fn bin_key(op: BinaryOperator, scalar_bool: bool) -> &'static str {
    use BinaryOperator as B;
    match op {
        B::Add => "({0} + {1})",
        B::Subtract => "({0} - {1})",
        B::Multiply => "({0} * {1})",
        B::Divide => "({0} / {1})",
        B::Modulo => "({0} % {1})",
        B::Equal => "({0} == {1})",
        B::NotEqual => "({0} != {1})",
        B::Less => "({0} < {1})",
        B::LessEqual => "({0} <= {1})",
        B::Greater => "({0} > {1})",
        B::GreaterEqual => "({0} >= {1})",
        B::And if scalar_bool => "({0} && {1})",
        B::And => "({0} & {1})",
        B::ExclusiveOr => "({0} ^ {1})",
        B::InclusiveOr if scalar_bool => "({0} || {1})",
        B::InclusiveOr => "({0} | {1})",
        B::LogicalAnd => "({0} && {1})",
        B::LogicalOr => "({0} || {1})",
        B::ShiftLeft => "({0} << {1})",
        B::ShiftRight => "({0} >> {1})",
    }
}

fn sc_name(s: Sc) -> &'static str {
    match s {
        Sc::F32 => "f32",
        Sc::F16 => "f16",
        Sc::U32 => "u32",
        Sc::I32 => "i32",
        Sc::Bool => "bool",
    }
}

/// The spellings of a value's own operation, as the rows that spell it write them (a lane `.x` is also the axis helper
/// `col_axis` of the cols-x grid, `.y` `row_axis`); empty for a value spelled by name.
fn keys_of(e: &E) -> Vec<String> {
    if let E::Lane(_, i, _) = e {
        let c = ["x", "y", "z", "w"][*i as usize];
        let mut v = vec![format!("{{0}}[{i}]"), format!("{{0}}.{c}")];
        match i {
            0 => v.push("col_axis({0})".into()),
            1 => v.push("row_axis({0})".into()),
            _ => {}
        }
        return v;
    }
    key_of(e).into_iter().collect()
}

fn key_of(e: &E) -> Option<String> {
    Some(match e {
        E::Bin(op, a, _, _) => bin_key(*op, a.wt() == Wt::scalar(Sc::Bool)).to_string(),
        E::Un(op, _, _) => match op {
            UnaryOperator::Negate => "(-{0})",
            UnaryOperator::LogicalNot => "(!{0})",
            UnaryOperator::BitwiseNot => "(~{0})",
        }
        .to_string(),
        E::Sel(..) => "select({0}, {1}, {2})".to_string(),
        E::Call(name, args, _) => format!("{name}({})", (0..args.len()).map(|i| format!("{{{i}}}")).collect::<Vec<_>>().join(", ")),
        E::Cast(w, _) | E::Splat(w, _) => format!("{}({{0}})", w.name()),
        E::Bitcast(w, _) => format!("bitcast<{}>({{0}})", w.name()),
        E::Lane(_, i, _) => format!("{{0}}[{i}]"),
        E::Swz(_, p, _) => format!("{{0}}.{p}"),
        E::DynLane(b, _, _) => {
            let l = b.wt().lanes;
            let s = sc_name(b.wt().s);
            if l == 3 {
                format!("at3_{s}({{0}}, ({{1}}) % 3u)")
            } else {
                format!("at{l}_{s}({{0}}, ({{1}}) & {}u)", l - 1)
            }
        }
        E::Compose(w, parts) if parts.len() == 2 => format!("{}({{0}}, {{1}})", w.name()),
        _ => return None,
    })
}

fn kids_of(e: &E) -> Vec<&E> {
    match e {
        E::Bin(_, a, b, _) => vec![a, b],
        E::Un(_, a, _) | E::Cast(_, a) | E::Bitcast(_, a) | E::Splat(_, a) | E::Lane(a, _, _) | E::Swz(a, _, _) => vec![a],
        E::DynLane(a, b, _) => vec![a, b],
        E::Sel(a, b, c, _) => vec![a, b, c],
        E::Call(_, v, _) | E::Compose(_, v) => v.iter().collect(),
        _ => Vec::new(),
    }
}

impl<'a> Reader<'a> {
    fn ty(&self, name: &str) -> R<crate::geneframe::Ty> {
        self.types_by_name.get(name).copied().ok_or_else(|| ReadError::Gap(format!("the type {name}")))
    }

    fn row(&self, name: &str) -> R<u32> {
        self.rows_by_name.get(name).copied().ok_or_else(|| ReadError::Gap(format!("the row {name}")))
    }

    fn leaf(&self, name: &str) -> R<Tree> {
        Ok(Tree { symbol: self.row(name)?, const_index: None, kids: Vec::new() })
    }

    fn note(&mut self, what: String) {
        // the first thing found missing is the deepest: the search fails bottom-up
        if self.missing.is_none() {
            self.missing = Some(what);
        }
    }

    /// The constant `?_T` of type `ty_name` holding the literal, if `C_T` has it.
    fn konst(&mut self, ty_name: &str, l: &Lit) -> Option<Tree> {
        let row = *self.constants.get(ty_name)?;
        let ty = self.k.rows[row as usize].out;
        let texts = self.k.constants.get(&ty)?;
        match texts.iter().position(|t| lit_text_is(t, l)) {
            Some(i) => Some(Tree { symbol: row, const_index: Some(i as u32), kids: Vec::new() }),
            None => {
                self.note(format!("the constant {} in C_T of {ty_name}", lit_text(l)));
                None
            }
        }
    }

    /// The value of a value made during spelling: the box is kept until the memo is cleared, so no later value can
    /// take its address (the memo's key).
    fn val_temp(&mut self, e: E, want: crate::geneframe::Ty) -> Option<Tree> {
        let b = std::rc::Rc::new(e);
        let r = self.val(&b, want);
        self.temps.push(b);
        r
    }

    /// The value `e` as type `want`, or `None` when no row spells it so.
    fn val(&mut self, e: &E, want: crate::geneframe::Ty) -> Option<Tree> {
        let key = (e as *const E as usize, want.code());
        if let Some(r) = self.memo.get(&key) {
            return r.clone();
        }
        self.memo.insert(key, None);
        let tn = self.k.type_name(want);
        let slot = tn.rsplit_once('.').map(|(s, _)| s.to_string()).unwrap_or_default();
        let r = if slot != e.wt().name() { None } else { self.val_direct(e, want, &tn).or_else(|| self.val_coerce(e, want)) };
        self.memo.insert(key, r.clone());
        r
    }

    fn val_coerce(&mut self, e: &E, want: crate::geneframe::Ty) -> Option<Tree> {
        let mut cands = self.coercions.get(&want.code()).cloned().unwrap_or_default();
        cands.extend(self.wrappers.get(&want.code()).cloned().unwrap_or_default());
        for (row, arg) in cands {
            if let Some(t) = self.val(e, arg) {
                return Some(Tree { symbol: row, const_index: None, kids: vec![t] });
            }
        }
        None
    }

    /// The rows named `names` (in order) whose output is `want`, each with its argument types.
    fn named_rows(&self, names: &[String], want: crate::geneframe::Ty) -> Vec<(u32, Vec<crate::geneframe::Ty>)> {
        names.iter().filter_map(|n| self.rows_by_name.get(n)).filter(|r| self.k.rows[**r as usize].out == want).map(|r| (*r, self.k.rows[*r as usize].args.clone())).collect()
    }

    fn with_kids(&mut self, row: u32, args: &[crate::geneframe::Ty], kids: &[&E]) -> Option<Tree> {
        if args.len() != kids.len() {
            return None;
        }
        let mut ts = Vec::new();
        for (a, e) in args.iter().zip(kids) {
            ts.push(self.val(e, *a)?);
        }
        Some(Tree { symbol: row, const_index: None, kids: ts })
    }

    fn val_direct(&mut self, e: &E, want: crate::geneframe::Ty, tn: &str) -> Option<Tree> {
        match e {
            E::Lit(l) => return self.konst(tn, l),
            E::Zero(w) => {
                let sp = format!("zero_vec{}_{}()", w.lanes, sc_name(w.s));
                return self.by_spelling.get(&sp).cloned().unwrap_or_default().into_iter().find(|r| self.k.rows[*r as usize].out == want).map(|r| Tree { symbol: r, const_index: None, kids: Vec::new() });
            }
            E::Bi(b) => {
                let sp = match b {
                    Bi::Gid => "gid",
                    Bi::Lid => "lid",
                    Bi::Wid => "wid",
                    Bi::Nwg => "nwg",
                    Bi::Lidx => "lidx",
                };
                return self.by_spelling.get(sp).cloned().unwrap_or_default().into_iter().find(|r| self.k.rows[*r as usize].out == want && self.k.rows[*r as usize].args.is_empty()).map(|r| Tree { symbol: r, const_index: None, kids: Vec::new() });
            }
            E::Field(i) => {
                let sp = format!("cfg.{}", self.fields[*i]);
                return self.by_spelling.get(&sp).cloned().unwrap_or_default().into_iter().find(|r| self.k.rows[*r as usize].out == want && self.k.rows[*r as usize].args.is_empty()).map(|r| Tree { symbol: r, const_index: None, kids: Vec::new() });
            }
            E::Reg(r, _) => {
                let (key, prefix, _, _) = REG_KINDS[r.kind];
                let names = [format!("reg.get_{key}"), format!("reg.get_{key}_w")];
                let reg_leaf = self.leaf(&format!("reg.{prefix}{}", r.n)).ok()?;
                return self.named_rows(&names, want).first().map(|(row, _)| Tree { symbol: *row, const_index: None, kids: vec![reg_leaf] });
            }
            E::Counter(id) => {
                let (d, div) = *self.loop_ctx.get(id)?;
                let name = if div { format!("safe.xtile_counter{d}") } else { format!("safe.tile_counter{d}") };
                let row = *self.rows_by_name.get(&name)?;
                return (self.k.rows[row as usize].out == want).then_some(Tree { symbol: row, const_index: None, kids: Vec::new() });
            }
            E::Parr(p, idx, _) => {
                let (key, prefix, _, _, _) = self.parrs[p.kind].clone();
                let names = [format!("parr.get_{key}"), format!("parr.get_{key}_w")];
                let leaf = self.leaf(&format!("parr.{prefix}{}", p.n)).ok()?;
                for (row, args) in self.named_rows(&names, want) {
                    if let Some(i) = self.parr_index(idx, args[1], p.len) {
                        return Some(Tree { symbol: row, const_index: None, kids: vec![leaf.clone(), i] });
                    }
                }
                return None;
            }
            E::Load(n, idx, _) => {
                let names = [format!("load.{n}"), format!("load.{n}_w")];
                for (row, args) in self.named_rows(&names, want) {
                    if let Some(t) = self.with_kids(row, &args, &[idx]) {
                        return Some(t);
                    }
                }
                return None;
            }
            E::LoadV4(n, idx) => {
                let names = [format!("load.{n}_v4"), format!("load.{n}_v4_w")];
                for (row, args) in self.named_rows(&names, want) {
                    if let Some(t) = self.with_kids(row, &args, &[idx]) {
                        return Some(t);
                    }
                }
                return None;
            }
            E::Len(n, v4) => {
                let base = if *v4 { format!("load.{n}_len_v4") } else { format!("load.{n}_len") };
                let names = [base.clone(), format!("{base}_w")];
                return self.named_rows(&names, want).first().map(|(row, _)| Tree { symbol: *row, const_index: None, kids: Vec::new() });
            }
            E::Wg(v, i, w) => {
                let Region::Plain { base, lane, block } = *self.regions.get(*v)? else { return None };
                let key = elem_key(*w)?;
                let idx = match block {
                    None => (**i).clone(),
                    Some(c) => {
                        let t = self.threads();
                        // the transposed layout: element i at (i % C) * T + i / C
                        E::Bin(BinaryOperator::Add, u(E::Bin(BinaryOperator::Multiply, u(E::Bin(BinaryOperator::Modulo, i.clone(), u(E::Lit(Lit::U(c))), uw())), u(E::Lit(Lit::U(t))), uw())), u(E::Bin(BinaryOperator::Divide, i.clone(), u(E::Lit(Lit::U(c))), uw())), uw())
                    }
                };
                let row = *self.rows_by_name.get(&format!("wg.get_{key}"))?;
                if self.k.rows[row as usize].out != want {
                    return None;
                }
                let mut kids = vec![self.wg_base(base)?];
                if key == "v4" {
                    kids.push(self.wg_base(lane)?);
                }
                let b = std::rc::Rc::new(idx);
                let it = self.val(&b, self.ty("u32.index").ok()?);
                self.temps.push(b);
                kids.push(it?);
                return Some(Tree { symbol: row, const_index: None, kids });
            }
            E::WgOwn(v, m, w) => {
                let Region::Plain { base, lane, .. } = *self.regions.get(*v)? else { return None };
                let key = elem_key(Wt { s: w.s, lanes: w.lanes })?;
                let names = [format!("wg.get_own_{key}"), format!("wg.get_own_plain_{key}")];
                let rows = self.named_rows(&names, want);
                let (row, _) = rows.first()?;
                let mut kids = vec![self.wg_base(base)?];
                if key == "v4" {
                    kids.push(self.wg_base(lane)?);
                }
                kids.push(self.wg_own(m)?);
                return Some(Tree { symbol: *row, const_index: None, kids });
            }
            E::WgA(v, i) => {
                // a relaxed load of the atomic pool, plain or its `w` twin (atomics stand in a phase)
                let Region::Atomic { word } = *self.regions.get(*v)? else { return None };
                for (row, args) in self.named_rows(&["atomic.wg_load".to_string(), "atomic.wg_load_w".to_string()], want) {
                    let Some(base) = self.konst(&self.k.type_name(args[0]), &Lit::U(word)) else { continue };
                    if let Some(it) = self.val(i, args[1]) {
                        return Some(Tree { symbol: row, const_index: None, kids: vec![base, it] });
                    }
                }
                return None;
            }
            E::WgRmw(f, v, i, x) => {
                let Region::Atomic { word } = *self.regions.get(*v)? else { return None };
                for (row, args) in self.named_rows(&[format!("atomic.wg_{f}"), format!("atomic.wg_{f}_w")], want) {
                    let Some(base) = self.konst(&self.k.type_name(args[0]), &Lit::U(word)) else { continue };
                    if let (Some(it), Some(xt)) = (self.val(i, args[1]), self.val(x, args[2])) {
                        return Some(Tree { symbol: row, const_index: None, kids: vec![base, it, xt] });
                    }
                }
                return None;
            }
            E::UReg(n) => {
                let row = *self.rows_by_name.get("uniform.get_U")?;
                if self.k.rows[row as usize].out != want {
                    return None;
                }
                let leaf = self.leaf(&format!("ureg.U{n}")).ok()?;
                return Some(Tree { symbol: row, const_index: None, kids: vec![leaf] });
            }
            E::Rmw(f, b, i, v) => {
                let names = [format!("atomic.{b}_{f}")];
                for (row, args) in self.named_rows(&names, want) {
                    if let Some(t) = self.with_kids(row, &args, &[i, v]) {
                        return Some(t);
                    }
                }
                return None;
            }
            E::DynLane(b, i, _) => {
                if let E::Reg(r, _) = b.as_ref() {
                    let (key, prefix, _, _) = REG_KINDS[r.kind];
                    let names = [format!("reg.get_{key}_lane"), format!("reg.get_{key}_lane_w")];
                    let leaf = self.leaf(&format!("reg.{prefix}{}", r.n)).ok()?;
                    for (row, args) in self.named_rows(&names, want) {
                        if let Some(it) = self.val(i, args[1]) {
                            return Some(Tree { symbol: row, const_index: None, kids: vec![leaf.clone(), it] });
                        }
                    }
                }
            }
            E::Compose(w, parts) => {
                // the kingdom composes vec2(s, s), vec3(vec2, s), vec4(vec3, s)
                let regrouped = match (w.lanes, parts.len()) {
                    (_, 2) => None,
                    (3, 3) => Some(vec![E::Compose(Wt { s: w.s, lanes: 2 }, parts[..2].to_vec()), parts[2].clone()]),
                    (4, 4) => Some(vec![E::Compose(Wt { s: w.s, lanes: 3 }, vec![E::Compose(Wt { s: w.s, lanes: 2 }, parts[..2].to_vec()), parts[2].clone()]), parts[3].clone()]),
                    (4, 3) if parts[0].wt().lanes == 2 => Some(vec![E::Compose(Wt { s: w.s, lanes: 3 }, parts[..2].to_vec()), parts[2].clone()]),
                    _ => {
                        self.note(format!("a {} composed of {} parts", w.name(), parts.len()));
                        return None;
                    }
                };
                if let Some(g) = regrouped {
                    return self.val_temp(E::Compose(*w, g), want);
                }
            }
            _ => {}
        }
        let keys = keys_of(e);
        if keys.is_empty() {
            return None;
        }
        let cands: Vec<u32> = keys.iter().flat_map(|k| self.by_spelling.get(k).cloned().unwrap_or_default()).collect();
        if cands.is_empty() {
            self.note(format!("a row spelled {} (for {})", keys[0], e.wt().name()));
        }
        let kids = kids_of(e);
        for row in cands {
            let r = &self.k.rows[row as usize];
            if r.out != want || r.kind != RowKind::Function {
                continue;
            }
            let args = r.args.clone();
            if let Some(t) = self.with_kids(row, &args, &kids) {
                return Some(t);
            }
        }
        // a vector op over a scalar: the scalar splatted, every lane the same operation
        if let E::Bin(op, a, b, w) = e {
            if w.lanes > 1 && (a.wt().lanes == 1) != (b.wt().lanes == 1) {
                let lift = |x: &E| if x.wt().lanes == 1 { E::Splat(Wt { s: x.wt().s, lanes: w.lanes }, Box::new(x.clone())) } else { x.clone() };
                return self.val_temp(E::Bin(*op, Box::new(lift(a)), Box::new(lift(b)), *w), want);
            }
        }
        None
    }

    /// A private array's index as the kingdom's index type: `min(i, len - 1)`, the device's own restriction.
    fn parr_index(&mut self, idx: &E, pidx: crate::geneframe::Ty, len: u32) -> Option<Tree> {
        let tn = self.k.type_name(pidx);
        let w = tn.ends_with('w');
        let j = tn.trim_start_matches("u32.pidx_").trim_end_matches('w');
        let row = *self.rows_by_name.get(&format!("safe.pidx_c_{j}{}", if w { "_w" } else { "" }))?;
        let last = self.konst(&format!("u32.plast_{j}"), &Lit::U(len.saturating_sub(1)))?;
        let it = self.val(idx, self.ty(if w { "u32.indexw" } else { "u32.index" }).ok()?)?;
        Some(Tree { symbol: row, const_index: None, kids: vec![it, last] })
    }

    fn wg_base(&mut self, blocks: u32) -> Option<Tree> {
        let lit = self.konst("u32.wgblk", &Lit::U(blocks))?;
        Some(Tree { symbol: *self.rows_by_name.get("wg.base")?, const_index: None, kids: vec![lit] })
    }

    fn wg_own(&mut self, m: &E) -> Option<Tree> {
        if lit_u(m) == Some(0) {
            return self.leaf("own.u32_wgown").ok();
        }
        let mt = self.val(m, self.ty("u32.indexw").ok()?)?;
        Some(Tree { symbol: *self.rows_by_name.get("wg.own_x")?, const_index: None, kids: vec![mt] })
    }

    /// `e` as `want`, or the gap naming the most specific thing no row spells.
    fn must(&mut self, e: &E, want_name: &str) -> R<Tree> {
        let want = self.ty(want_name)?;
        self.missing = None;
        // a fresh memo for each value: its keys are addresses, valid while the values (and the kept boxes) live
        self.memo.clear();
        self.temps.clear();
        match self.val(e, want) {
            Some(t) => Ok(t),
            None => {
                let m = self.missing.take().unwrap_or_else(|| format!("a row that spells the value as {want_name}"));
                let mut d = format!("{e:?}");
                if d.len() > 200 {
                    d.truncate(200);
                    d.push_str("...");
                }
                gap(format!("{m} [value {d} as {want_name}]"))
            }
        }
    }

    // ---------------------------------------------------------------- statements

    /// The fuel class and literal bound of a loop: a literal bound in 1..=64 is class 1; any other bound is capped at the
    /// least `2^c` above its upper bound (`c` in 6, 12, 18, 24), class `c / 6`.
    fn class_of(&self, lp: &LoopSt) -> R<(u32, Option<u32>, u32)> {
        if let Some(n) = lit_u(&lp.bound) {
            if (1..=64).contains(&n) {
                return Ok((1, Some(n), 6));
            }
        }
        let m = self.max_of(&lp.bound).ok_or_else(|| ReadError::Out(format!("a loop whose trip bound is not bounded by the contract's cfg ranges or constants: {:?}", lp.bound)))?;
        for c in [6u32, 12, 18, 24] {
            if m <= 1u64 << c {
                return Ok((c / 6, None, c));
            }
        }
        Err(ReadError::Out(format!("a loop of up to {m} trips, over the kingdom's largest bound 2^24")))
    }

    fn need(&self, st: &St) -> R<u32> {
        Ok(match st {
            St::If(_, a, b) => self.need_all(a)?.max(self.need_all(b)?),
            St::Phase(b) | St::Pair(_, b) | St::Solo(b) | St::Split(_, b) => self.need_all(b)?,
            St::Loop(l) => self.need_all(&l.body)? + self.class_of(l)?.0,
            _ => 0,
        })
    }

    fn need_all(&self, sts: &[St]) -> R<u32> {
        let mut m = 0;
        for s in sts {
            m = m.max(self.need(s)?);
        }
        Ok(m)
    }

    fn rung(x: char, d: u8, k: u32) -> String {
        format!("{x}{d}_f{k}")
    }

    fn block(&mut self, sts: &[St], x: char, d: u8, k: u32) -> R<Tree> {
        let mut trees = Vec::new();
        for (i, s) in sts.iter().enumerate() {
            match s {
                St::BreakIf(_) => return gap("our software has a bug: a break point outside its loop body"),
                St::Phase(_) | St::Pair(..) | St::Solo(_) | St::Split(..) => {
                    if x != 'L' {
                        return gap("a workgroup phase under divergent control");
                    }
                    // a phase holds the rest of its block
                    let rest = self.block(&sts[i + 1..], 'L', d, k)?;
                    let r = Self::rung('L', d, k);
                    let t = match s {
                        St::Phase(w) => {
                            let wb = self.block(w, 'W', d, k)?;
                            Tree { symbol: self.row(&format!("wg.phase_{r}"))?, const_index: None, kids: vec![wb, rest] }
                        }
                        St::Pair(sv, b) => {
                            let st = self.must(sv, "u32.uniform")?;
                            let pb = self.block(b, 'P', d, k)?;
                            Tree { symbol: self.row(&format!("wg.pair_{r}"))?, const_index: None, kids: vec![st, pb, rest] }
                        }
                        St::Solo(b) => {
                            let sb = self.block(b, 'S', d, k)?;
                            Tree { symbol: self.row(&format!("wg.solo_{r}"))?, const_index: None, kids: vec![sb, rest] }
                        }
                        St::Split(rd, b) => {
                            let sp = rd.iter().filter_map(|v| match self.regions.get(*v) {
                                Some(Region::Plain { base, .. }) => Some(*base),
                                _ => None,
                            }).min().ok_or_else(|| ReadError::Gap("our software has a bug: a split phase reading no plain array".into()))?;
                            let spt = self.wg_base(sp).ok_or_else(|| ReadError::Gap(format!("the split base {sp}")))?;
                            let xb = self.block(b, 'X', d, k)?;
                            Tree { symbol: self.row(&format!("wg.split_{r}"))?, const_index: None, kids: vec![spt, xb, rest] }
                        }
                        _ => unreachable_phase()?,
                    };
                    trees.push(t);
                    break;
                }
                _ => trees.push(self.stmt(s, x, d, k)?),
            }
        }
        self.block_of(trees, x, d, k)
    }

    fn block_of(&mut self, mut trees: Vec<Tree>, x: char, d: u8, k: u32) -> R<Tree> {
        let r = Self::rung(x, d, k);
        if trees.is_empty() {
            return self.leaf(&format!("own.block_{r}"));
        }
        if trees.len() > 4 {
            let rest = trees.split_off(3);
            let inner = self.block_of(rest, x, d, k)?;
            trees.push(Tree { symbol: self.row(&format!("stmt.block_{r}"))?, const_index: None, kids: vec![inner] });
        }
        let name = ["one", "seq2", "seq3", "seq4"][trees.len() - 1];
        Ok(Tree { symbol: self.row(&format!("block.{name}_{r}"))?, const_index: None, kids: trees })
    }

    /// A statement at exactly stratum `k`: spelled at its own stratum, then lifted.
    fn stmt(&mut self, st: &St, x: char, d: u8, k: u32) -> R<Tree> {
        let n = self.need(st)?;
        if n > k {
            return gap(format!("our software has a bug: a statement of stratum {n} in a block of stratum {k}"));
        }
        let mut t = self.stmt_at(st, x, d, n)?;
        for j in n + 1..=k {
            t = Tree { symbol: self.row(&format!("stmt.lift_{}", Self::rung(x, d, j)))?, const_index: None, kids: vec![t] };
        }
        Ok(t)
    }

    /// The statement row `name` with its first arguments `leading` given and the rest the values `vals`.
    fn stmt_row(&mut self, name: &str, leading: Vec<Tree>, vals: &[&E]) -> R<Tree> {
        let row = self.row(name)?;
        let args = self.k.rows[row as usize].args.clone();
        if args.len() != leading.len() + vals.len() {
            return gap(format!("our software has a bug: {name} takes {} arguments", args.len()));
        }
        let mut kids = leading;
        for v in vals {
            let an = self.k.type_name(args[kids.len()]);
            kids.push(self.must(v, &an)?);
        }
        Ok(Tree { symbol: row, const_index: None, kids })
    }

    fn stmt_at(&mut self, st: &St, x: char, d: u8, k: u32) -> R<Tree> {
        let r = Self::rung(x, d, k);
        match st {
            St::Set(reg, e) => {
                let (key, prefix, _, _) = REG_KINDS[reg.kind];
                let leaf = self.leaf(&format!("reg.{prefix}{}", reg.n))?;
                self.stmt_row(&format!("reg.set_{key}_{r}"), vec![leaf], &[e])
            }
            St::SetLane(reg, i, e) => {
                let (key, prefix, _, _) = REG_KINDS[reg.kind];
                let leaf = self.leaf(&format!("reg.{prefix}{}", reg.n))?;
                self.stmt_row(&format!("reg.set_{key}_lane_{r}"), vec![leaf], &[i, e])
            }
            St::ParrSet(p, i, e) => {
                let (key, prefix, _, _, _) = self.parrs[p.kind].clone();
                let leaf = self.leaf(&format!("parr.{prefix}{}", p.n))?;
                let name = format!("parr.set_{key}_{r}");
                let row = self.row(&name)?;
                let pidx = self.k.rows[row as usize].args[1];
                self.memo.clear();
                let it = self.parr_index(i, pidx, p.len).ok_or_else(|| ReadError::Gap(format!("the private array index {i:?}")))?;
                self.stmt_row(&name, vec![leaf, it], &[e])
            }
            St::Store(b, i, e) => self.stmt_row(&format!("store.{b}_{r}"), vec![], &[i, e]),
            St::Discard(e) => {
                self.stmt_row(&format!("stmt.discard_u_{r}"), vec![], &[e])
            }
            St::Barrier | St::StorageBarrier => {
                if x != 'L' {
                    return gap("a uniform proof of the control around a barrier (naga validated it; the reader's uniformity is weaker)");
                }
                let name = if matches!(st, St::Barrier) { "barrier" } else { "storage_barrier" };
                self.leaf(&format!("stmt.{name}_{r}"))
            }
            St::BreakIf(_) => gap("our software has a bug: a break point outside its loop body"),
            St::Phase(_) | St::Pair(..) | St::Solo(_) | St::Split(..) => gap("our software has a bug: a phase spelled as a statement"),
            St::HLoad(reg, v, i) => {
                let Some(Region::Plain { base, lane, block }) = self.regions.get(*v).copied() else { return gap("our software has a bug: a workgroup array with no region") };
                let key = elem_key(self.wgvars[*v].elem).ok_or_else(|| ReadError::Gap(format!("a workgroup array of {}", self.wgvars[*v].elem.name())))?;
                let (_, prefix, _, _) = REG_KINDS[reg.kind];
                let mut lead = vec![self.leaf(&format!("reg.{prefix}{}", reg.n))?, self.wg_base(base).ok_or_else(|| ReadError::Gap(format!("the region base {base}")))?];
                if key == "v4" {
                    lead.push(self.wg_base(lane).ok_or_else(|| ReadError::Gap(format!("the lane stride {lane}")))?);
                }
                let idx = match block {
                    None => i.clone(),
                    Some(c) => {
                        let t = self.threads();
                        E::Bin(BinaryOperator::Add, u(E::Bin(BinaryOperator::Multiply, u(E::Bin(BinaryOperator::Modulo, u(i.clone()), u(E::Lit(Lit::U(c))), uw())), u(E::Lit(Lit::U(t))), uw())), u(E::Bin(BinaryOperator::Divide, u(i.clone()), u(E::Lit(Lit::U(c))), uw())), uw())
                    }
                };
                self.stmt_row(&format!("wg.hload_{key}_{r}"), lead, &[&idx])
            }
            St::WgSet(v, i, _) => gap(format!("a write of {}[{i:?}] outside any phase this kingdom spells", self.wgvars[*v].name)),
            St::WgOwnSet(v, m, e) => {
                let Some(Region::Plain { base, lane, .. }) = self.regions.get(*v).copied() else { return gap("our software has a bug: a workgroup array with no region") };
                let key = elem_key(self.wgvars[*v].elem).ok_or_else(|| ReadError::Gap(format!("a workgroup array of {}", self.wgvars[*v].elem.name())))?;
                let mut lead = vec![self.wg_base(base).ok_or_else(|| ReadError::Gap(format!("the region base {base}")))?];
                if key == "v4" {
                    lead.push(self.wg_base(lane).ok_or_else(|| ReadError::Gap(format!("the lane stride {lane}")))?);
                }
                self.memo.clear();
                lead.push(self.wg_own(m).ok_or_else(|| ReadError::Gap(format!("an own element of m = {m:?}")))?);
                self.stmt_row(&format!("wg.set_{key}_{r}"), lead, &[e])
            }
            St::WgPut(v, i, e) => {
                let Some(Region::Plain { base, lane, block }) = self.regions.get(*v).copied() else { return gap("our software has a bug: a workgroup array with no region") };
                let key = elem_key(self.wgvars[*v].elem).ok_or_else(|| ReadError::Gap(format!("a workgroup array of {}", self.wgvars[*v].elem.name())))?;
                let mut lead = vec![self.wg_base(base).ok_or_else(|| ReadError::Gap(format!("the region base {base}")))?];
                if key == "v4" {
                    lead.push(self.wg_base(lane).ok_or_else(|| ReadError::Gap(format!("the lane stride {lane}")))?);
                }
                let idx = match block {
                    None => i.clone(),
                    Some(c) => {
                        let t = self.threads();
                        E::Bin(BinaryOperator::Add, u(E::Bin(BinaryOperator::Multiply, u(E::Bin(BinaryOperator::Modulo, u(i.clone()), u(E::Lit(Lit::U(c))), uw())), u(E::Lit(Lit::U(t))), uw())), u(E::Bin(BinaryOperator::Divide, u(i.clone()), u(E::Lit(Lit::U(c))), uw())), uw())
                    }
                };
                self.stmt_row(&format!("wg.put_{key}_{r}"), lead, &[&idx, e])
            }
            St::WgAStore(v, i, e) => {
                let Some(Region::Atomic { word }) = self.regions.get(*v).copied() else { return gap("our software has a bug: an atomic array with no region") };
                let base = self.konst("u32.wgabase", &Lit::U(word)).ok_or_else(|| ReadError::Gap(format!("the atomic word {word}")))?;
                self.stmt_row(&format!("atomic.wg_store_{r}"), vec![base], &[i, e])
            }
            St::ULoad(n, v, i) => {
                let Some(Region::Plain { base, block: None, .. }) = self.regions.get(*v).copied() else { return gap("workgroupUniformLoad of a transposed array") };
                let leaf = self.leaf(&format!("ureg.U{n}"))?;
                let b = self.wg_base(base).ok_or_else(|| ReadError::Gap(format!("the region base {base}")))?;
                self.stmt_row(&format!("wg.uload_{r}"), vec![leaf, b], &[i])
            }
            St::PLoad(reg, v) => {
                let Some(Region::Plain { base, lane, .. }) = self.regions.get(*v).copied() else { return gap("our software has a bug: a workgroup array with no region") };
                let key = elem_key(self.wgvars[*v].elem).ok_or_else(|| ReadError::Gap(format!("a workgroup array of {}", self.wgvars[*v].elem.name())))?;
                let (_, prefix, _, _) = REG_KINDS[reg.kind];
                let mut kids = vec![self.leaf(&format!("reg.{prefix}{}", reg.n))?, self.wg_base(base).ok_or_else(|| ReadError::Gap(format!("the region base {base}")))?];
                if key == "v4" {
                    kids.push(self.wg_base(lane).ok_or_else(|| ReadError::Gap(format!("the lane stride {lane}")))?);
                }
                Ok(Tree { symbol: self.row(&format!("wg.pload_{key}_{r}"))?, const_index: None, kids })
            }
            St::If(c, a, b) => {
                let nl = needs_l(a) || needs_l(b);
                match x {
                    'L' if nl => {
                        let cond = self.must(c, "bool.uflag").map_err(|e| match e {
                            ReadError::Gap(m) => ReadError::Gap(format!("a uniform proof of a condition around a barrier: {m}")),
                            o => o,
                        })?;
                        let ta = self.block(a, 'L', d, k)?;
                        if b.is_empty() {
                            Ok(Tree { symbol: self.row(&format!("stmt.if_uflag_{r}"))?, const_index: None, kids: vec![cond, ta] })
                        } else {
                            let tb = self.block(b, 'L', d, k)?;
                            Ok(Tree { symbol: self.row(&format!("stmt.if_else_uflag_{r}"))?, const_index: None, kids: vec![cond, ta, tb] })
                        }
                    }
                    'L' | 'C' => {
                        if nl {
                            return gap("a uniform proof of the control around a barrier (naga validated it; the reader's uniformity is weaker)");
                        }
                        let cond = self.must(c, "bool.flag")?;
                        let ta = self.block(a, 'C', d, k)?;
                        if b.is_empty() {
                            Ok(Tree { symbol: self.row(&format!("stmt.if_{r}"))?, const_index: None, kids: vec![cond, ta] })
                        } else {
                            let tb = self.block(b, 'C', d, k)?;
                            Ok(Tree { symbol: self.row(&format!("stmt.if_else_{r}"))?, const_index: None, kids: vec![cond, ta, tb] })
                        }
                    }
                    'S' => {
                        if nl {
                            return gap("a barrier inside invocation 0's code");
                        }
                        let cond = self.must(c, "bool.flag")?;
                        let ta = self.block(a, 'S', d, k)?;
                        if b.is_empty() {
                            Ok(Tree { symbol: self.row(&format!("stmt.if_{r}"))?, const_index: None, kids: vec![cond, ta] })
                        } else {
                            let tb = self.block(b, 'S', d, k)?;
                            Ok(Tree { symbol: self.row(&format!("stmt.if_else_{r}"))?, const_index: None, kids: vec![cond, ta, tb] })
                        }
                    }
                    _ => {
                        let cond = self.must(c, "bool.flagw")?;
                        let ta = self.block(a, x, d, k)?;
                        if b.is_empty() {
                            Ok(Tree { symbol: self.row(&format!("stmt.if_{r}"))?, const_index: None, kids: vec![cond, ta] })
                        } else {
                            let tb = self.block(b, x, d, k)?;
                            Ok(Tree { symbol: self.row(&format!("stmt.if_else_{r}"))?, const_index: None, kids: vec![cond, ta, tb] })
                        }
                    }
                }
            }
            St::Loop(lp) => self.loop_at(lp, x, d, k),
        }
    }

    fn loop_at(&mut self, lp: &LoopSt, x: char, d: u8, k: u32) -> R<Tree> {
        let (class, lit, c) = self.class_of(lp)?;
        let kb = self.need_all(&lp.body)?;
        if kb + class != k {
            return gap(format!("our software has a bug: a loop of stratum {} spelled at {k}", kb + class));
        }
        let d1 = d + 1;
        if d1 > 5 || (x == 'O' && d1 > 4) {
            return gap(format!("a loop nested {d1} deep (the kingdom's ladder holds 5)"));
        }
        let brk = lp.body.iter().position(|s| matches!(s, St::BreakIf(_)));
        let nl = needs_l(&lp.body);
        let stride = lp.stride;
        let lit64 = |s: &mut Self, n: u32| s.konst("u32.lit64", &Lit::U(n)).ok_or_else(|| ReadError::Gap(format!("the constant {n} in C_T of u32.lit64")));
        let lit4096 = |s: &mut Self, n: u32| s.konst("u32.lit4096", &Lit::U(n)).ok_or_else(|| ReadError::Gap(format!("the constant {n} in C_T of u32.lit4096")));
        if brk.is_some() && stride != 1 {
            return gap("a strided loop that breaks");
        }
        if brk.is_some() && nl {
            return gap("a loop that breaks and holds a barrier");
        }
        // the loop's row, its body's rung and whether its counter is divergent
        let uniform_bound = if lit.is_none() && x == 'L' { self.must(&lp.bound, "u32.uniform").ok() } else { None };
        if x == 'L' && nl && lit.is_none() && uniform_bound.is_none() {
            return gap(format!("a uniform proof of the bound of a loop holding a barrier: {:?}", lp.bound));
        }
        let (row, body_x, div, bound_kids): (String, char, bool, Vec<Tree>) = match (x, lit, brk.is_some()) {
            ('W' | 'X', Some(n), false) if stride == 1 => (format!("loop.tile_{x}{d}_f{kb}"), x, true, vec![lit64(self, n)?]),
            ('W' | 'X', None, false) if stride == 1 => {
                let b = self.must(&lp.bound, "u32.indexw")?;
                (format!("loop.tilex_{c}_{x}{d}_f{kb}"), x, true, vec![Tree { symbol: self.row(&format!("bound.of_index_{c}_w"))?, const_index: None, kids: vec![b] }])
            }
            ('W' | 'X', Some(n), true) => (format!("loop.tile_brk_{x}{d}_f{kb}"), x, true, vec![lit64(self, n)?]),
            ('W' | 'X', None, true) => {
                let b = self.must(&lp.bound, "u32.indexw")?;
                (format!("loop.tilex_brk_{c}_{x}{d}_f{kb}"), x, true, vec![Tree { symbol: self.row(&format!("bound.of_index_{c}_w"))?, const_index: None, kids: vec![b] }])
            }
            ('W' | 'X', Some(n), false) => (format!("loop.tile_strided_{x}{d}_f{kb}"), x, true, vec![lit64(self, n)?, lit4096(self, stride)?]),
            ('W' | 'X', None, false) => {
                let b = self.must(&lp.bound, "u32.indexw")?;
                (format!("loop.tilex_strided_{c}_{x}{d}_f{kb}"), x, true, vec![Tree { symbol: self.row(&format!("bound.of_index_{c}_w"))?, const_index: None, kids: vec![b] }, lit4096(self, stride)?])
            }
            ('P', _, _) => return gap("a loop inside a pair phase"),
            ('S', Some(n), true) => (format!("loop.tile_brk_S{d}_f{kb}"), 'S', true, vec![lit64(self, n)?]),
            ('S', None, true) => {
                let b = self.must(&lp.bound, "u32.index")?;
                (format!("loop.tilex_brk_{c}_S{d}_f{kb}"), 'S', true, vec![Tree { symbol: self.row(&format!("bound.of_index_{c}"))?, const_index: None, kids: vec![b] }])
            }
            ('S', Some(n), false) => {
                if stride == 1 {
                    (format!("loop.tile_S{d}_f{kb}"), 'S', true, vec![lit64(self, n)?])
                } else {
                    (format!("loop.tile_strided_S{d}_f{kb}"), 'S', true, vec![lit64(self, n)?, lit4096(self, stride)?])
                }
            }
            ('S', None, false) => {
                let b = self.must(&lp.bound, "u32.index")?;
                let b = Tree { symbol: self.row(&format!("bound.of_index_{c}"))?, const_index: None, kids: vec![b] };
                if stride == 1 {
                    (format!("loop.tilex_{c}_S{d}_f{kb}"), 'S', true, vec![b])
                } else {
                    (format!("loop.tilex_strided_{c}_S{d}_f{kb}"), 'S', true, vec![b, lit4096(self, stride)?])
                }
            }
            (_, Some(n), true) => (format!("loop.tile_brk_{x}{d}_f{kb}"), 'C', true, vec![lit64(self, n)?]),
            (_, None, true) => {
                let b = self.must(&lp.bound, "u32.index")?;
                (format!("loop.tilex_brk_{c}_{x}{d}_f{kb}"), 'C', true, vec![Tree { symbol: self.row(&format!("bound.of_index_{c}"))?, const_index: None, kids: vec![b] }])
            }
            ('L', Some(n), false) => {
                if stride == 1 {
                    (format!("loop.tile_L{d}_f{kb}"), 'L', false, vec![lit64(self, n)?])
                } else {
                    (format!("loop.tile_strided_L{d}_f{kb}"), 'L', false, vec![lit64(self, n)?, lit4096(self, stride)?])
                }
            }
            ('L', None, false) if uniform_bound.is_some() => {
                let b = Tree { symbol: self.row(&format!("bound.of_uniform_{c}"))?, const_index: None, kids: vec![uniform_bound.clone().unwrap_or(Tree { symbol: 0, const_index: None, kids: vec![] })] };
                if stride == 1 {
                    (format!("loop.tileu_{c}_L{d}_f{kb}"), 'L', false, vec![b])
                } else {
                    (format!("loop.tileu_strided_{c}_L{d}_f{kb}"), 'L', false, vec![b, lit4096(self, stride)?])
                }
            }
            (_, Some(n), false) => {
                if stride == 1 {
                    (format!("loop.tile_{x}{d}_f{kb}"), 'C', true, vec![lit64(self, n)?])
                } else {
                    (format!("loop.tile_strided_{x}{d}_f{kb}"), 'C', true, vec![lit64(self, n)?, lit4096(self, stride)?])
                }
            }
            (_, None, false) => {
                let b = self.must(&lp.bound, "u32.index")?;
                let b = Tree { symbol: self.row(&format!("bound.of_index_{c}"))?, const_index: None, kids: vec![b] };
                if stride == 1 {
                    (format!("loop.tilex_{c}_{x}{d}_f{kb}"), 'C', true, vec![b])
                } else {
                    (format!("loop.tilex_strided_{c}_{x}{d}_f{kb}"), 'C', true, vec![b, lit4096(self, stride)?])
                }
            }
        };
        if nl && body_x != 'L' {
            return gap("a uniform proof of the control around a barrier (naga validated it; the reader's uniformity is weaker)");
        }
        self.loop_ctx.insert(lp.id, (d1, div));
        let body = match brk {
            None => self.block(&lp.body, body_x, d1, kb)?,
            Some(at) => {
                let St::BreakIf(flag) = &lp.body[at] else { return gap("our software has a bug: the break point moved") };
                let (bx, fl) = match x {
                    'W' | 'X' => (x, "bool.flagw"),
                    'S' => ('S', "bool.flag"),
                    _ => ('C', "bool.flag"),
                };
                let a = self.block(&lp.body[..at], bx, d1, kb)?;
                let f = self.must(flag, fl)?;
                let c2 = self.block(&lp.body[at + 1..], bx, d1, kb)?;
                let name = match x {
                    'W' => format!("block.brk_WB{d1}_f{kb}"),
                    'X' => format!("block.brk_XB{d1}_f{kb}"),
                    'S' => format!("block.brk_SB{d1}_f{kb}"),
                    _ => format!("block.brk_B{d1}_f{kb}"),
                };
                Tree { symbol: self.row(&name)?, const_index: None, kids: vec![a, f, c2] }
            }
        };
        let mut kids = bound_kids;
        kids.push(body);
        let symbol = self.row(&row)?;
        Ok(Tree { symbol, const_index: None, kids })
    }

    /// Every runtime literal in a slot whose row has a bare twin there becomes the bare literal (s9 batch 7), the first
    /// such slot of each node.
    fn bare(&self, t: &mut Tree) {
        for kid in t.kids.iter_mut() {
            self.bare(kid);
        }
        let name = &self.k.rows[t.symbol as usize].name;
        for (i, kid) in t.kids.iter().enumerate() {
            let kn = &self.k.rows[kid.symbol as usize].name;
            let fam = match kn.trim_end_matches("_w") {
                "literal.real" => "bare.real",
                "literal.index" => "bare.index",
                "literal.count" => "bare.count",
                "literal.int" => "bare.int",
                "literal.flag" => "bare.flag",
                "literal.f16" => "bare.h",
                _ => continue,
            };
            let (Some(twin), Some(b)) = (self.rows_by_name.get(&format!("{name}_bare{i}")), self.rows_by_name.get(fam)) else { continue };
            let c = kid.const_index;
            t.symbol = *twin;
            t.kids[i] = Tree { symbol: *b, const_index: c, kids: Vec::new() };
            return;
        }
    }
}

/// Whether the statements begin with a barrier (a phase begins with its leading one) on every path.
fn leads_barrier(sts: &[St]) -> bool {
    match sts.first() {
        Some(St::Barrier | St::Phase(_) | St::Pair(..) | St::Solo(_) | St::Split(..)) => true,
        Some(St::Loop(l)) => leads_barrier(&l.body),
        Some(St::If(_, a, b)) => leads_barrier(a) && leads_barrier(b),
        _ => false,
    }
}

/// Whether the statements end with a barrier (a phase as the last statement ends with its middle one) on every path.
fn ends_barrier(sts: &[St]) -> bool {
    match sts.last() {
        Some(St::Barrier | St::Phase(_) | St::Pair(..) | St::Solo(_) | St::Split(..)) => true,
        Some(St::Loop(l)) => ends_barrier(&l.body) && !l.body.iter().any(|x| matches!(x, St::BreakIf(_))),
        Some(St::If(_, a, b)) => ends_barrier(a) && ends_barrier(b),
        _ => false,
    }
}

fn needs_l(sts: &[St]) -> bool {
    sts.iter().any(|s| match s {
        St::Barrier | St::StorageBarrier | St::Phase(_) | St::Pair(..) | St::Solo(_) | St::Split(..) | St::ULoad(..) => true,
        St::If(_, a, b) => needs_l(a) || needs_l(b),
        St::Loop(l) => needs_l(&l.body),
        _ => false,
    })
}

impl<'a> Reader<'a> {
    /// One entry point read into the tree of its gene: `kernel.entry_<e>(grid, [passes,] body)`.
    fn entry(&mut self, ep: &'a naga::EntryPoint, info: &'a naga::valid::FunctionInfo, passes: u32) -> R<Tree> {
        let mut args = Vec::new();
        for a in &ep.function.arguments {
            args.push(match &a.binding {
                Some(naga::Binding::BuiltIn(b)) => E::Bi(match b {
                    BuiltIn::GlobalInvocationId => Bi::Gid,
                    BuiltIn::LocalInvocationId => Bi::Lid,
                    BuiltIn::WorkGroupId => Bi::Wid,
                    BuiltIn::NumWorkGroups => Bi::Nwg,
                    BuiltIn::LocalInvocationIndex => Bi::Lidx,
                    o => return gap(format!("the builtin {o:?}")),
                }),
                o => return gap(format!("an entry argument bound as {o:?}")),
            });
        }
        let mut fr = Frame { f: &ep.function, info, args, ptr_args: HashMap::new(), locals: HashMap::new(), values: HashMap::new(), named: ep.function.named_expressions.keys().copied().collect(), ret: None };
        let mut out = Vec::new();
        self.locals(&mut fr, &mut out, true)?;
        self.walk(&mut fr, &ep.function.body, &mut out, 0)?;
        self.allocate(&out)?;
        let out = self.structure(out)?;
        if !self.splits.is_empty() {
            let pairs: Vec<(usize, usize)> = self.splits.iter().flat_map(|(w, r)| w.iter().flat_map(move |a| r.iter().map(move |b| (*a, *b)))).collect();
            self.place(&pairs)?;
        }
        let k = self.k.fuel_budget;
        let need = self.need_all(&out)?;
        if need > k {
            return gap(format!("fuel: the entry {} needs stratum {need}, the table's budget is K = {k}", ep.name));
        }
        let mut body = self.block(&out, 'L', 0, k)?;
        let [x, y, z] = ep.workgroup_size;
        if z != 1 {
            return gap(format!("a workgroup of depth {z}"));
        }
        let shape = self.konst_text("grid.shape", &format!("{x} {y}"))?;
        let one = |s: &mut Self, t: &str| s.konst_text(t, "1");
        let grid = Tree { symbol: self.row("grid.of")?, const_index: None, kids: vec![shape, one(self, "u32.blk_col")?, one(self, "u32.blk_row")?, one(self, "u32.ktile")?] };
        self.bare(&mut body);
        let name = format!("kernel.entry_{}", ep.name);
        let row = self.row(&name)?;
        let kids = if self.k.rows[row as usize].args.len() == 3 { vec![grid, self.konst_text("u32.passes", &passes.to_string())?, body] } else { vec![grid, body] };
        Ok(Tree { symbol: row, const_index: None, kids })
    }

    /// The constant of `ty_name` whose `C_T` text is `text`.
    fn konst_text(&self, ty_name: &str, text: &str) -> R<Tree> {
        let row = *self.constants.get(ty_name).ok_or_else(|| ReadError::Gap(format!("a constant of {ty_name}")))?;
        let ty = self.k.rows[row as usize].out;
        let texts = self.k.constants.get(&ty).ok_or_else(|| ReadError::Gap(format!("C_T of {ty_name}")))?;
        let i = texts.iter().position(|t| t == text).ok_or_else(|| ReadError::Gap(format!("the constant {text} in C_T of {ty_name}")))?;
        Ok(Tree { symbol: row, const_index: Some(i as u32), kids: Vec::new() })
    }
}

/// Read a module's text: one tree per entry point of the kingdom's module interface, in gene order.
pub fn read_module(k: &TrueKingdom, src: &str, contract: &Contract) -> R<Vec<Tree>> {
    let module = crate::wgsl::naga_shape::parse_unscanned(src).map_err(|e| ReadError::Out(format!("naga does not read the text: {e}")))?;
    let info = crate::wgsl::target::Target::current().validate(&module, src).map_err(|e| ReadError::Out(format!("naga does not validate the text: {e}")))?;
    let passes = crate::wgsl::harness::passes_header(src).map_err(ReadError::Out)?;
    let mut entries = Vec::new();
    for root in &k.gene_roots {
        let tn = k.type_name(*root);
        let name = tn.strip_prefix("kernel.").ok_or_else(|| ReadError::Gap(format!("our software has a bug: the gene root {tn}")))?;
        let (i, ep) = module.entry_points.iter().enumerate().find(|(_, e)| e.name == name).ok_or_else(|| ReadError::Out(format!("no entry point {name}")))?;
        if ep.stage != naga::ShaderStage::Compute {
            return Err(ReadError::Out(format!("the entry point {name} is not a compute shader")));
        }
        let mut rd = Reader::new(k, &module, &info, contract);
        rd.wg = ep.workgroup_size;
        entries.push(rd.entry(ep, info.get_entry_point(i), passes)?);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn kingdom() -> TrueKingdom {
        TrueKingdom::load(&crate::wgsl::true_kingdom::default_dir().join("mod").join("v24")).expect("the v24 table")
    }

    pub(crate) fn contract() -> Contract {
        let text = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("problems/v24_ingest5b/safety.contract")).expect("safety contract");
        Contract::parse(&text, &["seed_slot", "stage", "arrived_begin", "arrived_end", "n_queries", "pass_index", "n_passes", "c_max"]).expect("contract")
    }

    /// Every seed of UI1: (set, name, text).
    pub(crate) fn seeds() -> Vec<(String, String, String)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut out = Vec::new();
        let dir = root.join("logs/V24/seeds5b");
        let mut names: Vec<_> = std::fs::read_dir(&dir).expect("seeds5b").flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect();
        names.sort();
        for p in names {
            out.push(("seeds5b".to_string(), p.file_name().unwrap_or_default().to_string_lossy().to_string(), std::fs::read_to_string(&p).expect("seed")));
        }
        for (n, t) in crate::wgsl::fixture_texts::set("v24_run1_seeds") {
            out.push(("run1".to_string(), n, t));
        }
        for (set, rel) in [("reference", "problems/v24_ingest_small/reference.wgsl"), ("run4", "logs/V24/run4/best.wgsl"), ("run5", "logs/V24/run5/best.wgsl")] {
            out.push((set.to_string(), rel.rsplit('/').next().unwrap_or(rel).to_string(), std::fs::read_to_string(root.join(rel)).expect("kernel")));
        }
        out
    }

    /// The functions of a tree (its program size).
    pub(crate) fn functions(t: &Tree, k: &TrueKingdom) -> usize {
        usize::from(!k.rows[t.symbol as usize].args.is_empty()) + t.kids.iter().map(|c| functions(c, k)).sum::<usize>()
    }

    /// Every seed that reads is a module the write-back puts in two genes (at the program size each entry needs), the
    /// typed walk expresses back exactly, and naga reads and validates.
    #[test]
    fn every_read_seed_round_trips_through_its_genes_and_validates() {
        let k = kingdom();
        let c = contract();
        let mut n = 0;
        for (set, name, text) in seeds() {
            let Ok(trees) = read_module(&k, &text, &c) else { continue };
            let sizes: Vec<usize> = trees.iter().map(|t| functions(t, &k)).collect();
            let m = crate::wgsl::true_module::TrueModule::new(&k, &sizes).unwrap_or_else(|e| panic!("{set}/{name}: {e}"));
            let prev = m.random(1, &mut crate::wgsl::true_kingdom::Rng(5), crate::wgsl::true_kingdom::Share::Half, true).expect("random");
            let ch = m.encode(&trees, &prev).unwrap_or_else(|e| panic!("{set}/{name}: {e}"));
            assert_eq!(m.entry_trees(&ch).expect("express"), trees, "{set}/{name}: the genes express another tree");
            let src = m.render(&ch).expect("render");
            let module = crate::wgsl::naga_shape::parse_unscanned(&src).unwrap_or_else(|e| panic!("{set}/{name}: naga parse: {e}\n{src}"));
            crate::wgsl::target::Target::current().validate(&module, &src).unwrap_or_else(|e| panic!("{set}/{name}: naga validate: {e}\n{src}"));
            n += 1;
        }
        assert!(n > 0, "no seed read");
        println!("{n} seeds round-trip and validate");
    }

    /// UI1's census: every seed and its status, compared with the recorded table `tests/fixtures/v24_true_read_status.tsv`
    /// (`set/name`, `reads` or `gap` (our bug: the table lacks a row) or `out` (illegal under its contract), the reason).
    /// A seed whose status changes fails the test loudly; regenerate the table with `UI1_WRITE_STATUS=1`.
    #[test]
    fn every_v24_seed_reads_or_is_named_with_its_reason() {
        let k = kingdom();
        let c = contract();
        // the kernels named outside the kingdom by the evidence of their own text (a race), each with it
        let outside_text = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v24_true_outside.tsv")).expect("v24_true_outside.tsv");
        let outside: BTreeMap<String, String> = outside_text.lines().skip(1).filter_map(|l| l.split_once('\t')).map(|(a, b)| (a.to_string(), b.to_string())).collect();
        let mut table = String::from("seed\tstatus\treason\n");
        for (set, name, text) in seeds() {
            let r = read_module(&k, &text, &c);
            if let Some(ev) = outside.get(&format!("{set}/{name}")) {
                assert!(r.is_err(), "{set}/{name} is named outside the kingdom ({ev}) but reads: our software has a bug, a race read as legal");
                let why: String = ev.chars().take(160).collect();
                table.push_str(&format!("{set}/{name}\tout\t{why}\n"));
                continue;
            }
            let (st, why) = match r {
                Ok(_) => ("reads", String::new()),
                Err(ReadError::Gap(g)) => ("gap", g.split(" [value").next().unwrap_or(&g).replace(['\t', '\n'], " ")),
                Err(ReadError::Out(o)) => ("out", o.replace(['\t', '\n'], " ")),
            };
            let why: String = why.chars().take(160).collect();
            table.push_str(&format!("{set}/{name}\t{st}\t{why}\n"));
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v24_true_read_status.tsv");
        if std::env::var("UI1_WRITE_STATUS").is_ok() {
            std::fs::write(&path, &table).expect("write the status table");
        }
        let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e} (run once with UI1_WRITE_STATUS=1)", path.display()));
        assert_eq!(table, want, "the seeds' statuses changed");
        let reads = table.lines().filter(|l| l.split('\t').nth(1) == Some("reads")).count();
        println!("{reads} of {} seeds read", table.lines().count() - 1);
    }

    /// The split dispatch's text (`harness::tiled_text`) of every seed that reads, and of its expressed module, is a
    /// kernel naga reads and validates, with every entry's workgroup, global and workgroup-count ids rebound from the
    /// tile uniform; a tiny kernel's text is checked word for word.
    #[test]
    fn the_split_dispatch_text_of_every_read_seed_validates() {
        let tiny = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n@compute @workgroup_size(8, 2)\nfn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {\n    o[g.x] = wid.x + n.x + lidx;\n}\n";
        let t = crate::wgsl::harness::tiled_text(tiny).expect("tiled");
        assert!(t.contains("fn main(@builtin(workgroup_id) wid_htile: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(global_invocation_id) g_htile: vec3<u32>, @builtin(num_workgroups) n_htile: vec3<u32>) {"), "{t}");
        assert!(t.contains("let wid: vec3<u32> = wid_htile + harness_tile.origin.xyz;") && t.contains("let g: vec3<u32> = g_htile + harness_tile.origin.xyz * vec3<u32>(8u, 2u, 1u);") && t.contains("let n: vec3<u32> = harness_tile.grid.xyz;"), "{t}");
        assert!(t.contains("    o[g.x] = wid.x + n.x + lidx;"), "the body is the text's own: {t}");
        assert!(crate::wgsl::harness::tiled_text(&t).is_err(), "a text that already names the tile uniform");
        let k = kingdom();
        let c = contract();
        let mut n = 0;
        for (set, name, text) in seeds() {
            let Ok(trees) = read_module(&k, &text, &c) else { continue };
            let m = crate::wgsl::true_module::TrueModule::new(&k, &trees.iter().map(|t| functions(t, &k)).collect::<Vec<_>>()).expect("module");
            let rendered = k.render(&m.module_tree(trees).expect("tree")).expect("render");
            for (what, src) in [("text", &text), ("expressed", &rendered)] {
                let t = crate::wgsl::harness::tiled_text(src).unwrap_or_else(|e| panic!("{set}/{name} {what}: {e}"));
                let module = crate::wgsl::naga_shape::parse_unscanned(&t).unwrap_or_else(|e| panic!("{set}/{name} {what}: naga parse: {e}\n{t}"));
                crate::wgsl::target::Target::current().validate(&module, &t).unwrap_or_else(|e| panic!("{set}/{name} {what}: naga validate: {e}\n{t}"));
                assert!(t.contains("harness_tile.origin.xyz"), "{set}/{name} {what}: no id rebound");
            }
            n += 1;
        }
        assert!(n >= 27, "{n} seeds read");
    }

    /// One workgroup's bound counts a solo phase's body once: run1 seed14's search (its beam's insertion runs on
    /// invocation 0 alone) is bounded by 7.3 M node executions an invocation, 234 M for 32 invocations counted each,
    /// over the 170 M per-submit cap; counted as the workgroup runs it, one workgroup is under the cap. A tree with no
    /// solo phase is bounded by its invocations times the invocation's bound.
    #[test]
    fn a_solo_phase_counts_once_a_workgroup() {
        let k = kingdom();
        let c = contract();
        let fields: BTreeMap<String, u64> = [("seed_slot", 0), ("stage", 2), ("arrived_begin", 32), ("arrived_end", 64), ("n_queries", 16), ("pass_index", 0), ("n_passes", 1), ("c_max", 64)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
        let (_, _, text) = seeds().into_iter().find(|(s, n, _)| s == "run1" && n == "seed14.wgsl").expect("run1 seed14");
        let trees = read_module(&k, &text, &c).expect("reads");
        let search = &trees[1];
        let per = k.tree_work_bound_in(search, &fields).expect("bound");
        let wg = k.tree_work_bound_workgroup_in(search, &fields, 32).expect("bound");
        let cap = crate::wgsl::work_guard::CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT;
        assert!(per * 32 > cap && wg <= cap && wg < per * 32 && wg >= per, "per invocation {per}, a workgroup {wg}, cap {cap}");
        assert_eq!(k.tree_work_bound_workgroup_in(search, &fields, 1).expect("bound"), per);
        let mut plain = 0;
        for (_, _, text) in seeds() {
            let Ok(trees) = read_module(&k, &text, &c) else { continue };
            for t in &trees {
                if !t.level_order().iter().any(|(n, _)| k.rows[n.symbol as usize].name.starts_with("wg.solo_")) {
                    assert_eq!(k.tree_work_bound_workgroup_in(t, &fields, 64).expect("bound"), 64 * k.tree_work_bound_in(t, &fields).expect("bound"));
                    plain += 1;
                }
            }
        }
        assert!(plain > 0, "no entry without a solo phase");
        println!("seed14 search: {per} an invocation, {wg} a workgroup of 32; {plain} entries without a solo phase");
    }

    /// `UI1_DUMP=<set/name>`: print that seed's expressed module (the text its genes render), for reading by eye.
    #[test]
    fn dump_one_seed_when_asked() {
        let Ok(want) = std::env::var("UI1_DUMP") else { return };
        let k = kingdom();
        let c = contract();
        for (set, name, text) in seeds() {
            if format!("{set}/{name}") == want {
                let trees = read_module(&k, &text, &c).expect("reads");
                let m = crate::wgsl::true_module::TrueModule::new(&k, &trees.iter().map(|t| functions(t, &k)).collect::<Vec<_>>()).expect("module");
                let tree = m.module_tree(trees).expect("tree");
                println!("{}", k.render(&tree).expect("render"));
            }
        }
    }

    #[test]
    fn print_every_v24_seed_status() {
        let k = kingdom();
        let c = contract();
        let mut reads = 0;
        let mut tally: BTreeMap<String, usize> = BTreeMap::new();
        for (set, name, text) in seeds() {
            let r = read_module(&k, &text, &c);
            match &r {
                Ok(t) => {
                    reads += 1;
                    let n: usize = t.iter().map(|x| x.level_order().len()).sum();
                    println!("{set}/{name}: reads ({n} nodes)");
                }
                Err(e) => {
                    let s = e.to_string();
                    println!("{set}/{name}: {s}");
                    let key: String = s.split(" [value").next().unwrap_or(&s).chars().take(90).collect();
                    *tally.entry(key).or_default() += 1;
                }
            }
        }
        println!("reads {reads}");
        let mut v: Vec<_> = tally.into_iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (k, n) in v {
            println!("{n:3} {k}");
        }
    }
}

#[cfg(test)]
mod literal_census {
    use super::tests::seeds;
    use std::collections::BTreeSet;

    /// The literals the seeds spell, by family, as `interface.py` lists the v24 interface's `extra_constants` (run with
    /// --nocapture when a seed set changes; the interface is append-only).
    #[test]
    fn print_the_seeds_literals() {
        let (mut u, mut i, mut f) = (BTreeSet::new(), BTreeSet::new(), Vec::<f32>::new());
        for (_, _, text) in seeds() {
            let Ok(m) = crate::wgsl::naga_shape::parse_unscanned(&text) else { continue };
            let mut visit = |e: &naga::Expression| {
                if let naga::Expression::Literal(l) = e {
                    match l {
                        naga::Literal::U32(v) => {
                            u.insert(*v);
                        }
                        naga::Literal::I32(v) => {
                            i.insert(*v);
                        }
                        naga::Literal::F32(v) => {
                            if !f.iter().any(|x| x.to_bits() == v.to_bits()) {
                                f.push(*v);
                            }
                        }
                        naga::Literal::AbstractInt(v) => {
                            if let Ok(x) = u32::try_from(*v) {
                                u.insert(x);
                            }
                            if let Ok(x) = i32::try_from(*v) {
                                i.insert(x);
                            }
                        }
                        naga::Literal::AbstractFloat(v) => {
                            let x = *v as f32;
                            if !f.iter().any(|y| y.to_bits() == x.to_bits()) {
                                f.push(x);
                            }
                        }
                        _ => {}
                    }
                }
            };
            for (_, e) in m.global_expressions.iter() {
                visit(e);
            }
            for (_, func) in m.functions.iter() {
                for (_, e) in func.expressions.iter() {
                    visit(e);
                }
            }
            for ep in &m.entry_points {
                for (_, e) in ep.function.expressions.iter() {
                    visit(e);
                }
            }
        }
        // a literal also reaches the kernel negated (a unary minus over it) and as its own integer forms
        let neg: Vec<f32> = f.iter().map(|x| -x).collect();
        for x in neg {
            if !f.iter().any(|y| y.to_bits() == x.to_bits()) {
                f.push(x);
            }
        }
        f.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        println!("U32 {:?}", u.iter().map(|v| v.to_string()).collect::<Vec<_>>());
        println!("I32 {:?}", i.iter().map(|v| if *v == i32::MIN { "(-2147483647i - 1i)".to_string() } else { format!("{v}i") }).collect::<Vec<_>>());
        println!("F32 {:?}", f.iter().filter(|x| x.is_finite()).map(|v| format!("{v:?}")).collect::<Vec<_>>());
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Workgroup memory (s9 batch 9): regions of the two pools, and the phases
// ---------------------------------------------------------------------------------------------------------------------

/// Whether `e` is this invocation's local index (`lidx`, or `lid.x` of a one-dimensional workgroup).
fn is_lidx(e: &E, wg: [u32; 3]) -> bool {
    match e {
        E::Bi(Bi::Lidx) => true,
        E::Lane(b, 0, _) => matches!(b.as_ref(), E::Bi(Bi::Lid)) && wg[1] == 1 && wg[2] == 1,
        _ => false,
    }
}

fn u(e: E) -> Box<E> {
    Box::new(e)
}

fn uw() -> Wt {
    Wt::scalar(Sc::U32)
}

/// Whether a condition holds only for invocation 0 (`lidx == 0u`), and what else it asks.
fn solo_cond(c: &E, wg: [u32; 3]) -> Option<Option<E>> {
    match c {
        E::Bin(BinaryOperator::Equal, a, b, _) if (is_lidx(a, wg) && lit_u(b) == Some(0)) || (is_lidx(b, wg) && lit_u(a) == Some(0)) => Some(None),
        E::Bin(BinaryOperator::LogicalAnd | BinaryOperator::And, a, b, w) if *w == Wt::scalar(Sc::Bool) => {
            if let Some(None) = solo_cond(a, wg) {
                return Some(Some((**b).clone()));
            }
            if let Some(None) = solo_cond(b, wg) {
                return Some(Some((**a).clone()));
            }
            None
        }
        _ => None,
    }
}

/// Whether a condition is `lidx < s`, and the offset `s` (and what else it asks).
fn pair_cond(c: &E, wg: [u32; 3]) -> Option<(E, Option<E>)> {
    match c {
        E::Bin(BinaryOperator::Less, a, b, _) if is_lidx(a, wg) => Some(((**b).clone(), None)),
        E::Bin(BinaryOperator::Greater, a, b, _) if is_lidx(b, wg) => Some(((**a).clone(), None)),
        E::Bin(BinaryOperator::LogicalAnd | BinaryOperator::And, a, b, w) if *w == Wt::scalar(Sc::Bool) => {
            if let Some((s, None)) = pair_cond(a, wg) {
                return Some((s, Some((**b).clone())));
            }
            if let Some((s, None)) = pair_cond(b, wg) {
                return Some((s, Some((**a).clone())));
            }
            None
        }
        _ => None,
    }
}

/// Whether the statements write a workgroup array (a plain store), at any depth.
fn writes_wg(sts: &[St]) -> bool {
    sts.iter().any(|s| match s {
        St::WgSet(..) => true,
        St::If(_, a, b) => writes_wg(a) || writes_wg(b),
        St::Loop(l) => writes_wg(&l.body),
        _ => false,
    })
}

/// The registers an expression reads.
fn e_regs(e: &E, out: &mut Vec<Reg>) {
    match e {
        E::Reg(r, _) => out.push(*r),
        E::Wg(_, i, _) | E::WgOwn(_, i, _) | E::WgA(_, i) | E::Parr(_, i, _) | E::Load(_, i, _) | E::LoadV4(_, i) => e_regs(i, out),
        E::WgRmw(_, _, i, x) | E::Rmw(_, _, i, x) => {
            e_regs(i, out);
            e_regs(x, out);
        }
        o => kids_of(o).into_iter().for_each(|k| e_regs(k, out)),
    }
}

/// The registers statements write.
fn regs_written(sts: &[St], out: &mut Vec<Reg>) {
    for s in sts {
        match s {
            St::Set(r, _) | St::SetLane(r, _, _) | St::PLoad(r, _) | St::HLoad(r, _, _) => out.push(*r),
            St::If(_, a, b) => {
                regs_written(a, out);
                regs_written(b, out);
            }
            St::Loop(l) => regs_written(&l.body, out),
            _ => {}
        }
    }
}

/// `if (c) { A B .. }` (no else, a workgroup write inside) as `if (c) { A } if (c) { B } ..` when `c` reads no register
/// the body writes and no workgroup element: each statement then runs under the same value of `c`.
fn distribute_ifs(seg: Vec<St>) -> Vec<St> {
    let mut out = Vec::new();
    for s in seg {
        match s {
            St::If(c, a, b) if b.is_empty() && a.len() > 1 && writes_wg(&a) => {
                let (mut read, mut written, mut wg) = (Vec::new(), Vec::new(), Vec::new());
                e_regs(&c, &mut read);
                regs_written(&a, &mut written);
                wg_uses_e(&c, &mut wg);
                if wg.is_empty() && !read.iter().any(|r| written.contains(r)) {
                    out.extend(distribute_ifs(a).into_iter().map(|x| St::If(c.clone(), vec![x], Vec::new())));
                } else {
                    out.push(St::If(c, a, b));
                }
            }
            o => out.push(o),
        }
    }
    out
}

/// Every workgroup access of the statements: (array, its index), `None` for an access at no index the reader can
/// name (a pair phase's partner).
fn wg_accesses<'s>(sts: &'s [St], out: &mut Vec<(usize, Option<&'s E>)>) {
    fn ex<'s>(e: &'s E, out: &mut Vec<(usize, Option<&'s E>)>) {
        match e {
            E::Wg(v, i, _) | E::WgA(v, i) => {
                out.push((*v, Some(i)));
                ex(i, out);
            }
            E::WgOwn(v, _, _) => out.push((*v, None)),
            E::WgRmw(_, v, i, x) => {
                out.push((*v, Some(i)));
                ex(i, out);
                ex(x, out);
            }
            E::Parr(_, i, _) | E::Load(_, i, _) | E::LoadV4(_, i) => ex(i, out),
            E::Rmw(_, _, i, x) => {
                ex(i, out);
                ex(x, out);
            }
            o => kids_of(o).into_iter().for_each(|k| ex(k, out)),
        }
    }
    for s in sts {
        match s {
            St::Set(_, e) | St::Discard(e) | St::BreakIf(e) => ex(e, out),
            St::SetLane(_, a, b) | St::ParrSet(_, a, b) | St::Store(_, a, b) => {
                ex(a, out);
                ex(b, out);
            }
            St::WgSet(v, i, x) | St::WgAStore(v, i, x) | St::WgPut(v, i, x) => {
                out.push((*v, Some(i)));
                ex(i, out);
                ex(x, out);
            }
            St::HLoad(_, v, i) => {
                out.push((*v, Some(i)));
                ex(i, out);
            }
            // an own write `v[lidx + m * T]` and an own read were proven own when they were made
            St::WgOwnSet(v, _, x) => {
                out.push((*v, None));
                ex(x, out);
            }
            St::PLoad(_, v) => out.push((*v, None)),
            St::ULoad(_, v, i) => {
                out.push((*v, Some(i)));
                ex(i, out);
            }
            St::If(c, a, b) => {
                ex(c, out);
                wg_accesses(a, out);
                wg_accesses(b, out);
            }
            St::Loop(l) => {
                ex(&l.bound, out);
                wg_accesses(&l.body, out);
            }
            St::Phase(b) | St::Solo(b) | St::Split(_, b) | St::Pair(_, b) => wg_accesses(b, out),
            St::Barrier | St::StorageBarrier => {}
        }
    }
}

/// The plain workgroup arrays statements write, at any depth and in any phase.
fn plain_written(sts: &[St], out: &mut Vec<usize>) {
    for s in sts {
        match s {
            St::WgSet(v, _, _) | St::WgOwnSet(v, _, _) | St::WgPut(v, _, _) => out.push(*v),
            St::If(_, a, b) => {
                plain_written(a, out);
                plain_written(b, out);
            }
            St::Loop(l) => plain_written(&l.body, out),
            St::Phase(b) | St::Solo(b) | St::Split(_, b) | St::Pair(_, b) => plain_written(b, out),
            _ => {}
        }
    }
}

/// Every workgroup write's (array, index, inside a solo condition).
fn wg_writes<'s>(sts: &'s [St], solo: bool, wg: [u32; 3], out: &mut Vec<(usize, &'s E, bool)>) {
    for s in sts {
        match s {
            St::WgSet(v, i, _) => out.push((*v, i, solo)),
            St::If(c, a, b) => {
                let so = solo || solo_cond(c, wg).is_some();
                wg_writes(a, so, wg, out);
                wg_writes(b, solo, wg, out);
            }
            St::Loop(l) => wg_writes(&l.body, solo, wg, out),
            _ => {}
        }
    }
}

/// Every workgroup array an expression or statement list names, in first-use order.
fn wg_uses_e(e: &E, out: &mut Vec<usize>) {
    match e {
        E::Wg(v, i, _) | E::WgOwn(v, i, _) | E::WgA(v, i) => {
            if !out.contains(v) {
                out.push(*v);
            }
            wg_uses_e(i, out);
        }
        E::WgRmw(_, v, i, x) => {
            if !out.contains(v) {
                out.push(*v);
            }
            wg_uses_e(i, out);
            wg_uses_e(x, out);
        }
        E::Rmw(_, _, a, b) => {
            wg_uses_e(a, out);
            wg_uses_e(b, out);
        }
        E::Parr(_, i, _) | E::Load(_, i, _) | E::LoadV4(_, i) => wg_uses_e(i, out),
        o => {
            for k in kids_of(o) {
                wg_uses_e(k, out);
            }
        }
    }
}

fn wg_uses(sts: &[St], out: &mut Vec<usize>) {
    for s in sts {
        match s {
            St::Set(_, e) | St::Discard(e) | St::BreakIf(e) => wg_uses_e(e, out),
            St::SetLane(_, a, b) | St::ParrSet(_, a, b) | St::Store(_, a, b) => {
                wg_uses_e(a, out);
                wg_uses_e(b, out);
            }
            St::WgSet(v, a, b) | St::WgAStore(v, a, b) | St::WgOwnSet(v, a, b) | St::WgPut(v, a, b) => {
                if !out.contains(v) {
                    out.push(*v);
                }
                wg_uses_e(a, out);
                wg_uses_e(b, out);
            }
            St::ULoad(_, v, a) => {
                if !out.contains(v) {
                    out.push(*v);
                }
                wg_uses_e(a, out);
            }
            St::If(c, a, b) => {
                wg_uses_e(c, out);
                wg_uses(a, out);
                wg_uses(b, out);
            }
            St::Loop(l) => {
                wg_uses_e(&l.bound, out);
                wg_uses(&l.body, out);
            }
            St::Pair(e, b) => {
                wg_uses_e(e, out);
                wg_uses(b, out);
            }
            St::Phase(b) | St::Solo(b) | St::Split(_, b) => wg_uses(b, out),
            St::HLoad(_, v, a) => {
                if !out.contains(v) {
                    out.push(*v);
                }
                wg_uses_e(a, out);
            }
            St::PLoad(_, v) => {
                if !out.contains(v) {
                    out.push(*v);
                }
            }
            St::Barrier | St::StorageBarrier => {}
        }
    }
}

impl<'a> Reader<'a> {
    fn threads(&self) -> u32 {
        self.wg[0] * self.wg[1] * self.wg[2]
    }

    /// `e` with every mask that cannot change it removed: `x & (2^j - 1)` when `x < 2^j`.
    fn unmask(&self, e: &E) -> E {
        let e = &self.map_kids(e, &mut |s, k| Ok(s.unmask(k))).unwrap_or_else(|_| e.clone());
        if let E::Bin(BinaryOperator::And, x, m, w) = e {
            if let Some(m) = lit_u(m) {
                if m.wrapping_add(1).is_power_of_two() && self.max_of(x).is_some_and(|v| v <= m as u64) {
                    return self.unmask(x);
                }
            }
            if let Some(mm) = lit_u(x) {
                if mm.wrapping_add(1).is_power_of_two() && self.max_of(m).is_some_and(|v| v <= mm as u64) {
                    return self.unmask(m);
                }
            }
            let _ = w;
        }
        e.clone()
    }

    /// `r / T` when `r` is provably a multiple of the workgroup size `T`.
    fn div_t(&self, r: &E) -> Option<E> {
        let t = self.threads();
        match r {
            E::Lit(Lit::U(c)) if c.is_multiple_of(t) => Some(E::Lit(Lit::U(c / t))),
            E::Bin(BinaryOperator::Multiply, a, b, _) => {
                if let Some(c) = lit_u(b).filter(|c| c.is_multiple_of(t)) {
                    return Some(if c == t { (**a).clone() } else { E::Bin(BinaryOperator::Multiply, a.clone(), u(E::Lit(Lit::U(c / t))), uw()) });
                }
                if let Some(c) = lit_u(a).filter(|c| c.is_multiple_of(t)) {
                    return Some(if c == t { (**b).clone() } else { E::Bin(BinaryOperator::Multiply, b.clone(), u(E::Lit(Lit::U(c / t))), uw()) });
                }
                None
            }
            E::Bin(BinaryOperator::ShiftLeft, a, b, _) => {
                let sft = lit_u(b)?;
                if sft < 32 && (1u64 << sft).is_multiple_of(t as u64) {
                    let c = ((1u64 << sft) / t as u64) as u32;
                    return Some(if c == 1 { (**a).clone() } else { E::Bin(BinaryOperator::Multiply, a.clone(), u(E::Lit(Lit::U(c))), uw()) });
                }
                None
            }
            E::Bin(BinaryOperator::Add, a, b, _) => {
                let (x, y) = (self.div_t(a)?, self.div_t(b)?);
                Some(E::Bin(BinaryOperator::Add, u(x), u(y), uw()))
            }
            E::Counter(id) => {
                let (_, s) = self.loop_bounds.get(id)?;
                if s.is_multiple_of(t) {
                    Some(E::Bin(BinaryOperator::Divide, u(r.clone()), u(E::Lit(Lit::U(t))), uw()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Split `lidx + rest` into `rest`.
    fn minus_lidx(&self, e: &E) -> Option<E> {
        if is_lidx(e, self.wg) {
            return Some(E::Lit(Lit::U(0)));
        }
        if let E::Bin(BinaryOperator::Add, a, b, _) = e {
            if is_lidx(a, self.wg) {
                return Some((**b).clone());
            }
            if is_lidx(b, self.wg) {
                return Some((**a).clone());
            }
            if let Some(r) = self.minus_lidx(a) {
                return Some(if lit_u(&r) == Some(0) { (**b).clone() } else { E::Bin(BinaryOperator::Add, u(r), b.clone(), uw()) });
            }
            if let Some(r) = self.minus_lidx(b) {
                return Some(if lit_u(&r) == Some(0) { (**a).clone() } else { E::Bin(BinaryOperator::Add, a.clone(), u(r), uw()) });
            }
        }
        None
    }

    /// The seed's block form `lidx * C + j` (`j < C`): `(C, j)`.
    fn block_form(&self, e: &E) -> Option<(u32, E)> {
        let mul_c = |x: &E| -> Option<u32> {
            match x {
                E::Bin(BinaryOperator::Multiply, a, b, _) if is_lidx(a, self.wg) => lit_u(b),
                E::Bin(BinaryOperator::Multiply, a, b, _) if is_lidx(b, self.wg) => lit_u(a),
                E::Bin(BinaryOperator::ShiftLeft, a, b, _) if is_lidx(a, self.wg) => lit_u(b).filter(|s| *s < 32).map(|s| 1u32 << s),
                _ => None,
            }
        };
        if let Some(c) = mul_c(e) {
            return Some((c, E::Lit(Lit::U(0))));
        }
        if let E::Bin(BinaryOperator::Add, a, b, _) = e {
            for (x, y) in [(a, b), (b, a)] {
                if let Some(c) = mul_c(x) {
                    if self.max_of(y).is_some_and(|m| m < c as u64) {
                        return Some((c, (**y).clone()));
                    }
                }
            }
        }
        None
    }

    /// The `m` of an own element `lidx + m * T` of array `v`, in its region's layout.
    fn own_m(&self, v: usize, idx: &E) -> Option<E> {
        let i = self.unmask(idx);
        // under `lidx == 0`, a literal element at invocation 0's residue is its own: `lidx + m * T` with lidx 0
        if self.lidx_zero.get() {
            if let Some(c) = lit_u(&i) {
                let t = self.threads();
                match self.regions.get(v) {
                    Some(Region::Plain { block: Some(b), .. }) if c < *b => return Some(E::Lit(Lit::U(c))),
                    Some(Region::Plain { block: None, .. }) if c.is_multiple_of(t) => return Some(E::Lit(Lit::U(c / t))),
                    _ => {}
                }
            }
        }
        match self.regions.get(v) {
            Some(Region::Plain { block: Some(c), .. }) => {
                let (c2, j) = self.block_form(&i)?;
                (c2 == *c).then_some(j)
            }
            _ => {
                let rest = self.minus_lidx(&i)?;
                if lit_u(&rest) == Some(0) {
                    return Some(E::Lit(Lit::U(0)));
                }
                self.div_t(&rest)
            }
        }
    }

    /// Lay out and place every workgroup array the entry uses: an array the entry writes block by block (`lidx * C +
    /// j` outside a solo phase) is stored transposed; each region of the plain pool begins at a multiple of the
    /// workgroup size.
    fn allocate(&mut self, sts: &[St]) -> R<()> {
        let mut used = Vec::new();
        wg_uses(sts, &mut used);
        let mut writes = Vec::new();
        wg_writes(sts, false, self.wg, &mut writes);
        let mut block: BTreeMap<usize, u32> = BTreeMap::new();
        for (v, i, solo) in &writes {
            if *solo {
                continue;
            }
            let i = self.unmask(i);
            if let Some((c, _)) = self.block_form(&i) {
                if c > 1 {
                    if let Some(old) = block.insert(*v, c) {
                        if old != c {
                            return gap(format!("workgroup array {} written in two block layouts ({old} and {c})", self.wgvars[*v].name));
                        }
                    }
                }
            }
        }
        self.used = used;
        self.blocks = block;
        self.place(&[])
    }

    /// Place the regions in first-use order, every array a split phase writes before every array it reads.
    fn place(&mut self, before: &[(usize, usize)]) -> R<()> {
        let block = self.blocks.clone();
        let mut order: Vec<usize> = Vec::new();
        let mut left: Vec<usize> = self.used.clone();
        while !left.is_empty() {
            let pick = left.iter().position(|v| !before.iter().any(|(a, b)| b == v && left.contains(a) && a != v)).ok_or_else(|| ReadError::Gap("split phases whose written and read arrays cannot be ordered in the pool".into()))?;
            order.push(left.remove(pick));
        }
        let t = self.threads();
        let pool_blocks = (crate::wgsl::true_reader::P_WORDS / t as u64) as u32;
        self.regions = vec![Region::Atomic { word: 0 }; self.wgvars.len()];
        let (mut next_blk, mut next_word) = (0u32, 0u32);
        for v in order {
            let w = &self.wgvars[v];
            if w.atomic {
                self.regions[v] = Region::Atomic { word: next_word };
                next_word += w.len;
                continue;
            }
            let blocks = |words: u32| words.div_ceil(t);
            let (reg, size) = match block.get(&v) {
                Some(c) => {
                    if w.len > c * t {
                        return gap(format!("workgroup array {} of {} written in blocks of {c} by {t} invocations", w.name, w.len));
                    }
                    (Region::Plain { base: next_blk, lane: 0, block: Some(*c) }, *c * w.elem.lanes as u32)
                }
                None => {
                    let lane = blocks(w.len);
                    (Region::Plain { base: next_blk, lane, block: None }, lane * w.elem.lanes as u32)
                }
            };
            self.regions[v] = reg;
            next_blk += size;
        }
        if next_blk > pool_blocks {
            return gap(format!("{} blocks of {t} words of workgroup memory, over the pool's {pool_blocks}", next_blk));
        }
        if next_word as u64 > crate::wgsl::true_reader::A_WORDS {
            return gap(format!("{next_word} words of workgroup atomics, over the atomic pool's {}", crate::wgsl::true_reader::A_WORDS));
        }
        Ok(())
    }

    // ---------------------------------------------------------------- phases

    /// The phases of an L-level statement list: each barrier-delimited segment that writes workgroup memory becomes a
    /// solo, a pair or a write phase.
    fn structure(&mut self, sts: Vec<St>) -> R<Vec<St>> {
        // nested uniform bodies first
        let mut inner = Vec::new();
        for s in sts {
            inner.push(match s {
                St::Loop(mut l) if needs_l(&l.body) => {
                    l.body = self.structure(l.body)?;
                    St::Loop(l)
                }
                St::If(c, a, b) if needs_l(&a) || needs_l(&b) => St::If(c, self.structure(a)?, self.structure(b)?),
                o => o,
            });
        }
        let mut out = Vec::new();
        let mut seg: Vec<St> = Vec::new();
        for s in inner {
            if matches!(s, St::Barrier) {
                self.close(std::mem::take(&mut seg), true, &mut out)?;
            } else {
                seg.push(s);
            }
        }
        self.close(seg, false, &mut out)?;
        Ok(out)
    }

    /// One segment (the statements between two barriers), its barrier consumed by its phase when it has one. A
    /// segment that is not one phase may be several: a cut between two runs of its statements that touch no common
    /// workgroup array (none writes an array the other reads or writes), each run one phase. A barrier added between
    /// them changes no value: every conflicting pair of accesses is inside one run, so every execution of the phases is
    /// an execution of the segment. An `if (c) { A B }` whose condition reads no register its body writes is first
    /// `if (c) { A } if (c) { B }`, the same statements in the same order.
    fn close(&mut self, seg: Vec<St>, barrier: bool, out: &mut Vec<St>) -> R<()> {
        let mut first = Vec::new();
        let splits = self.splits.len();
        let e = match self.close_one(seg.clone(), barrier, &mut first) {
            Ok(()) => {
                out.extend(first);
                return Ok(());
            }
            Err(e) => e,
        };
        self.splits.truncate(splits);
        let seg = distribute_ifs(seg);
        // a statement that already holds barriers (a loop or uniform branch of phases) between two runs it shares no
        // written array with: the runs close on either side of it
        if let Some(i) = seg.iter().position(|x| needs_l(std::slice::from_ref(x))) {
            let (a, rest) = seg.split_at(i);
            let (l, b) = rest.split_at(1);
            // a side the statement's own barrier already orders needs no proof; a loop that may run no trip leaves the
            // two runs unordered with each other
            let once = !matches!(&l[0], St::Loop(lp) if lit_u(&lp.bound).is_none_or(|n| n == 0));
            if (leads_barrier(l) || self.independent(a, l)) && (ends_barrier(l) || self.independent(l, b)) && (once || self.independent(a, b)) {
                let mut tried = Vec::new();
                let at = self.splits.len();
                let ra = if writes_wg(a) { self.close(a.to_vec(), false, &mut tried) } else {
                    tried.extend(a.iter().cloned());
                    Ok(())
                };
                if ra.is_ok() {
                    tried.extend(l.iter().cloned());
                    if self.close(b.to_vec(), barrier, &mut tried).is_ok() {
                        out.extend(tried);
                        return Ok(());
                    }
                }
                self.splits.truncate(at);
            }
        }
        for cut in 1..seg.len() {
            let (a, b) = seg.split_at(cut);
            if !writes_wg(a) || !writes_wg(b) || !self.independent(a, b) {
                continue;
            }
            let mut tried = Vec::new();
            let at = self.splits.len();
            if self.close_one(a.to_vec(), false, &mut tried).is_ok() && self.close(b.to_vec(), barrier, &mut tried).is_ok() {
                out.extend(tried);
                return Ok(());
            }
            self.splits.truncate(at);
        }
        Err(e)
    }

    /// Whether two runs of statements may run unordered without a race, so a barrier between them changes no value:
    /// every plain array one writes and the other touches is touched in both only at this invocation's own elements
    /// (`lidx + m * T`), each the same invocation's in program order. Atomics never race.
    fn independent(&self, a: &[St], b: &[St]) -> bool {
        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        plain_written(a, &mut wa);
        plain_written(b, &mut wb);
        let (mut xa, mut xb) = (Vec::new(), Vec::new());
        wg_accesses(a, &mut xa);
        wg_accesses(b, &mut xb);
        let own = |v: usize, i: Option<&E>| match i {
            None => true,
            Some(i) => self.own_m(v, i).is_some(),
        };
        let shared: Vec<usize> = wa.iter().filter(|v| xb.iter().any(|(u, _)| u == *v)).chain(wb.iter().filter(|v| xa.iter().any(|(u, _)| u == *v))).copied().collect();
        shared.iter().all(|v| xa.iter().chain(xb.iter()).filter(|(u, _)| u == v).all(|(u, i)| own(*u, *i)))
    }

    fn close_one(&mut self, seg: Vec<St>, barrier: bool, out: &mut Vec<St>) -> R<()> {
        if !writes_wg(&seg) {
            out.extend(seg);
            if barrier {
                out.push(St::Barrier);
            }
            return Ok(());
        }
        let first = seg.iter().position(|s| writes_wg(std::slice::from_ref(s))).unwrap_or(0);
        let mut pre: Vec<St> = seg[..first].to_vec();
        let mut part: Vec<St> = seg[first..].to_vec();
        // `if (c) { if (lidx == 0u) { B } }` is `if (lidx == 0u) { if (c) { B } }` (a condition has no effect)
        if let St::If(c, a, b) = &part[0] {
            if b.is_empty() && a.len() == 1 {
                if let St::If(c2, a2, b2) = &a[0] {
                    if b2.is_empty() && solo_cond(c2, self.wg).is_some() {
                        part[0] = St::If(c2.clone(), vec![St::If(c.clone(), a2.clone(), Vec::new())], Vec::new());
                    }
                }
            }
        }
        // a solo phase: `if (lidx == 0u) { ... }`, the segment's only writer
        if let St::If(c, a, b) = &part[0] {
            if let Some(rest) = solo_cond(c, self.wg) {
                if b.is_empty() && !writes_wg(&part[1..]) {
                    let body = match rest {
                        None => a.clone(),
                        Some(r) => vec![St::If(r, a.clone(), Vec::new())],
                    };
                    let body = self.solo_body(body)?;
                    out.extend(pre);
                    out.push(St::Solo(body));
                    out.extend(part.drain(1..));
                    return Ok(());
                }
            }
            if let Some((sv, rest)) = pair_cond(c, self.wg) {
                if b.is_empty() && part.len() == 1 && self.stable(&sv) && self.reads_partner(&sv, a) {
                    let t = self.threads() as u64;
                    if self.max_of(&sv).is_some_and(|m| m <= t / 2) {
                        let body = match rest {
                            None => a.clone(),
                            Some(r) => vec![St::If(r, a.clone(), Vec::new())],
                        };
                        let body = self.pair_body(&sv, body)?;
                        out.extend(pre);
                        out.push(St::Pair(sv, body));
                        return Ok(());
                    }
                }
            }
        }
        // a split phase: the arrays the segment writes are not the arrays it reads foreign elements of
        let split_gap = match self.try_split(&part) {
            Ok(Some(body)) => {
                out.extend(pre.clone());
                out.push(body);
                return Ok(());
            }
            Ok(None) => None,
            Err(ReadError::Gap(g)) => Some(g),
            Err(ReadError::Out(o)) => Some(o),
        };
        // a write phase's gap names the split phase's too, so the reason is the whole of what the table lacks
        let both = |e: ReadError| match (e, &split_gap) {
            (ReadError::Gap(g), Some(s)) => ReadError::Gap(format!("{g}; as a split phase: {s}")),
            (e, _) => e,
        };
        // a write phase: its statements read no workgroup element another invocation may write
        let mut w = Vec::new();
        for s in part.drain(..) {
            match self.write_stmt(&s) {
                Ok(x) => w.push(x),
                Err(e) if w.is_empty() => match self.split_if(&s)? {
                    // `if (c) { A; writes }`: `f = c; if (f) { A }` before the phase, `if (f) { writes }` in it
                    Some((set, before, inside)) => match self.write_stmt(&inside) {
                        Ok(x) => {
                            pre.push(set);
                            pre.push(before);
                            w.push(x);
                        }
                        Err(_) => return Err(both(e)),
                    },
                    None => return Err(both(e)),
                },
                Err(e) => {
                    // a statement with no workgroup write that reads foreign elements moves before the phase when nothing
                    // in the phase so far shares a register with it
                    if !writes_wg(std::slice::from_ref(&s)) && w.is_empty() {
                        pre.push(s);
                        continue;
                    }
                    return Err(both(e));
                }
            }
        }
        let _ = barrier;
        out.extend(pre);
        out.push(St::Phase(w));
        Ok(())
    }

    /// A segment as a split phase: every write an own write of an array the segment never reads a foreign element of;
    /// every foreign read hoisted into a register just before the statement that reads it.
    fn try_split(&mut self, part: &[St]) -> R<Option<St>> {
        let mut wr = Vec::new();
        let mut ws = Vec::new();
        wg_writes(part, false, self.wg, &mut ws);
        for (v, _, _) in &ws {
            if !wr.contains(v) {
                wr.push(*v);
            }
        }
        let mut rd = Vec::new();
        let mut body = Vec::new();
        for s in part {
            body.extend(self.x_stmt(s, &wr, &mut rd)?);
        }
        if rd.is_empty() || rd.iter().any(|v| wr.contains(v)) {
            return Ok(None);
        }
        self.splits.push((wr, rd.clone()));
        Ok(Some(St::Split(rd, body)))
    }

    /// `e` in a split phase: reads of written arrays own reads, foreign reads of other arrays hoisted into `pre`.
    fn x_e(&mut self, e: &E, wr: &[usize], rd: &mut Vec<usize>, pre: &mut Vec<St>) -> R<E> {
        Ok(match e {
            E::Wg(v, i, w) => {
                let i2 = self.x_e(i, wr, rd, pre)?;
                if wr.contains(v) {
                    match self.own_m(*v, &i2) {
                        Some(m) => E::WgOwn(*v, u(m), *w),
                        None => return gap(format!("a split phase reading {}[{i2:?}], an element of an array it writes that is not its own", self.wgvars[*v].name)),
                    }
                } else {
                    if !rd.contains(v) {
                        rd.push(*v);
                    }
                    let r = self.reg(*w)?;
                    pre.push(St::HLoad(r, *v, i2));
                    E::Reg(r, *w)
                }
            }
            o => {
                let mut err = None;
                let mut kids = Vec::new();
                for k in kids_of(o) {
                    match self.x_e(k, wr, rd, pre) {
                        Ok(x) => kids.push(x),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                if let Some(e) = err {
                    return Err(e);
                }
                if let E::Parr(p, i, w) = o {
                    let i2 = self.x_e(i, wr, rd, pre)?;
                    return Ok(E::Parr(*p, u(i2), *w));
                }
                if let E::Load(n, i, w) = o {
                    let i2 = self.x_e(i, wr, rd, pre)?;
                    return Ok(E::Load(n.clone(), u(i2), *w));
                }
                let mut it = kids.into_iter();
                self.map_kids(o, &mut |_, _| Ok(it.next().unwrap_or(E::Lit(Lit::U(0)))))?
            }
        })
    }

    /// A statement of a split phase, with the hoisted loads it needs before it.
    fn x_stmt(&mut self, s: &St, wr: &[usize], rd: &mut Vec<usize>) -> R<Vec<St>> {
        let mut pre = Vec::new();
        let st = match s {
            St::WgSet(v, i, x) => {
                let i2 = self.x_e(i, wr, rd, &mut pre)?;
                let x2 = self.x_e(x, wr, rd, &mut pre)?;
                match self.own_m(*v, &i2) {
                    Some(m) => St::WgOwnSet(*v, m, x2),
                    None => return gap(format!("a write of {}[{i2:?}], an element another invocation may own", self.wgvars[*v].name)),
                }
            }
            St::Set(r, e) => St::Set(*r, self.x_e(e, wr, rd, &mut pre)?),
            St::SetLane(r, a, b) => St::SetLane(*r, self.x_e(a, wr, rd, &mut pre)?, self.x_e(b, wr, rd, &mut pre)?),
            St::ParrSet(p, a, b) => St::ParrSet(*p, self.x_e(a, wr, rd, &mut pre)?, self.x_e(b, wr, rd, &mut pre)?),
            St::Store(n, a, b) => St::Store(n.clone(), self.x_e(a, wr, rd, &mut pre)?, self.x_e(b, wr, rd, &mut pre)?),
            St::WgAStore(v, a, b) => St::WgAStore(*v, self.x_e(a, wr, rd, &mut pre)?, self.x_e(b, wr, rd, &mut pre)?),
            St::Discard(e) => St::Discard(self.x_e(e, wr, rd, &mut pre)?),
            St::BreakIf(e) => St::BreakIf(self.x_e(e, wr, rd, &mut pre)?),
            St::If(c, a, b) => {
                let c2 = self.x_e(c, wr, rd, &mut pre)?;
                let zero = self.lidx_zero.get();
                if solo_cond(c, self.wg).is_some() {
                    self.lidx_zero.set(true);
                }
                let mut a2 = Vec::new();
                let ra: R<()> = a.iter().try_for_each(|x| self.x_stmt(x, wr, rd).map(|v| a2.extend(v)));
                self.lidx_zero.set(zero);
                ra?;
                let mut b2 = Vec::new();
                for x in b {
                    b2.extend(self.x_stmt(x, wr, rd)?);
                }
                St::If(c2, a2, b2)
            }
            St::Loop(l) => {
                let mut l2 = (**l).clone();
                let b2 = self.x_e(&l.bound, wr, rd, &mut pre)?;
                if !matches!(b2, E::Lit(_)) && b2 != l.bound {
                    return gap("a split phase loop bounded by a workgroup element");
                }
                let mut body = Vec::new();
                for x in &l.body {
                    // a break point stays at the body's own level: its hoisted loads go just before it
                    body.extend(self.x_stmt(x, wr, rd)?);
                }
                l2.body = body;
                St::Loop(Box::new(l2))
            }
            o => return gap(format!("the statement {o:?} inside a split phase")),
        };
        pre.push(st);
        Ok(pre)
    }

    /// Whether a pair body reads some array's partner element `lidx + s`.
    fn reads_partner(&self, sv: &E, body: &[St]) -> bool {
        let mut found = false;
        let mut visit = |e: &E| {
            let mut stack = vec![e];
            while let Some(x) = stack.pop() {
                if let E::Wg(_, i, _) = x {
                    if self.minus_lidx(&self.pair_unmask(i)).is_some_and(|r| self.unmask(&r) == self.unmask(sv)) {
                        found = true;
                    }
                }
                stack.extend(kids_of(x));
                if let E::Wg(_, i, _) | E::Parr(_, i, _) | E::Load(_, i, _) = x {
                    stack.push(i);
                }
            }
        };
        fn walk_st(sts: &[St], f: &mut dyn FnMut(&E)) {
            for s in sts {
                match s {
                    St::Set(_, e) => f(e),
                    St::WgSet(_, i, e) | St::ParrSet(_, i, e) | St::SetLane(_, i, e) => {
                        f(i);
                        f(e)
                    }
                    St::If(c, a, b) => {
                        f(c);
                        walk_st(a, f);
                        walk_st(b, f)
                    }
                    _ => {}
                }
            }
        }
        walk_st(body, &mut visit);
        found
    }

    /// `if (c) { A; W }` with `W` the statements from the first workgroup write on: `(f = c, if (f) { A }, if (f) { W })`.
    fn split_if(&mut self, s: &St) -> R<Option<(St, St, St)>> {
        let St::If(c, a, b) = s else { return Ok(None) };
        if !b.is_empty() {
            return Ok(None);
        }
        let Some(at) = a.iter().position(|x| writes_wg(std::slice::from_ref(x))) else { return Ok(None) };
        if at == 0 {
            return Ok(None);
        }
        let w = Wt::scalar(Sc::Bool);
        let f = self.reg(w)?;
        self.ssa.insert(f);
        let fv = E::Reg(f, w);
        Ok(Some((St::Set(f, c.clone()), St::If(fv.clone(), a[..at].to_vec(), Vec::new()), St::If(fv, a[at..].to_vec(), Vec::new()))))
    }

    fn own_e(&self, e: &E, ctx: &str) -> R<E> {
        Ok(match e {
            E::Wg(v, i, w) => match self.own_m(*v, i) {
                Some(m) => E::WgOwn(*v, u(m), *w),
                None => return gap(format!("{ctx}: a read of {}[{:?}], an element another invocation may write in the phase", self.wgvars[*v].name, i)),
            },
            o => self.map_kids(o, &mut |s, k| s.own_e(k, ctx))?,
        })
    }

    /// `e` with each child replaced by `f(child)`.
    fn map_kids(&self, e: &E, f: &mut dyn FnMut(&Self, &E) -> R<E>) -> R<E> {
        Ok(match e {
            E::Bin(op, a, b, w) => E::Bin(*op, u(f(self, a)?), u(f(self, b)?), *w),
            E::Un(op, a, w) => E::Un(*op, u(f(self, a)?), *w),
            E::Sel(a, b, c, w) => E::Sel(u(f(self, a)?), u(f(self, b)?), u(f(self, c)?), *w),
            E::Call(n, v, w) => E::Call(n, v.iter().map(|x| f(self, x)).collect::<R<Vec<_>>>()?, *w),
            E::Cast(w, a) => E::Cast(*w, u(f(self, a)?)),
            E::Bitcast(w, a) => E::Bitcast(*w, u(f(self, a)?)),
            E::Compose(w, v) => E::Compose(*w, v.iter().map(|x| f(self, x)).collect::<R<Vec<_>>>()?),
            E::Splat(w, a) => E::Splat(*w, u(f(self, a)?)),
            E::Lane(a, i, w) => E::Lane(u(f(self, a)?), *i, *w),
            E::Swz(a, p, w) => E::Swz(u(f(self, a)?), p.clone(), *w),
            E::DynLane(a, b, w) => E::DynLane(u(f(self, a)?), u(f(self, b)?), *w),
            E::Parr(p, i, w) => E::Parr(*p, u(f(self, i)?), *w),
            E::Load(n, i, w) => E::Load(n.clone(), u(f(self, i)?), *w),
            E::LoadV4(n, i) => E::LoadV4(n.clone(), u(f(self, i)?)),
            E::Wg(v, i, w) => E::Wg(*v, u(f(self, i)?), *w),
            E::WgA(v, i) => E::WgA(*v, u(f(self, i)?)),
            E::WgRmw(fun, v, i, x) => E::WgRmw(fun, *v, u(f(self, i)?), u(f(self, x)?)),
            E::Rmw(fun, b, i, x) => E::Rmw(fun, b.clone(), u(f(self, i)?), u(f(self, x)?)),
            o => o.clone(),
        })
    }

    /// A statement of a write phase: own writes, own reads, no atomic read-modify-write.
    fn write_stmt(&self, s: &St) -> R<St> {
        let c = "a write phase";
        Ok(match s {
            St::WgSet(v, i, x) => match self.own_m(*v, i) {
                Some(m) => St::WgOwnSet(*v, m, self.own_e(x, c)?),
                None => return gap(format!("a write of {}[{:?}], an element another invocation may own", self.wgvars[*v].name, i)),
            },
            St::WgAStore(v, i, x) => St::WgAStore(*v, self.own_e(i, c)?, self.own_e(x, c)?),
            St::Set(r, e) => St::Set(*r, self.own_e(e, c)?),
            St::SetLane(r, a, b) => St::SetLane(*r, self.own_e(a, c)?, self.own_e(b, c)?),
            St::ParrSet(p, a, b) => St::ParrSet(*p, self.own_e(a, c)?, self.own_e(b, c)?),
            St::Store(n, a, b) => St::Store(n.clone(), self.own_e(a, c)?, self.own_e(b, c)?),
            St::If(e, a, b) => {
                let saved = self.lidx_cap.get();
                if let Some((sv, None)) = pair_cond(e, self.wg) {
                    if let Some(m) = lit_u(&sv) {
                        self.lidx_cap.set(Some(saved.unwrap_or(u64::MAX).min(m.saturating_sub(1) as u64)));
                    }
                }
                let zero = self.lidx_zero.get();
                if solo_cond(e, self.wg).is_some() {
                    self.lidx_zero.set(true);
                }
                let ra = a.iter().map(|x| self.write_stmt(x)).collect::<R<Vec<_>>>();
                self.lidx_zero.set(zero);
                self.lidx_cap.set(saved);
                St::If(self.own_e(e, c)?, ra?, b.iter().map(|x| self.write_stmt(x)).collect::<R<_>>()?)
            }
            St::Loop(l) => {
                let mut l2 = (**l).clone();
                l2.bound = self.own_e(&l.bound, c)?;
                l2.body = l.body.iter().map(|x| self.write_stmt(x)).collect::<R<_>>()?;
                St::Loop(Box::new(l2))
            }
            St::BreakIf(e) => St::BreakIf(self.own_e(e, c)?),
            St::Discard(e) => St::Discard(self.own_e(e, c)?),
            o => return gap(format!("the statement {o:?} inside a write phase")),
        })
    }

    /// A pair phase's body: partner reads `A[lidx + s]` become registers loaded first, own reads and writes `A[lidx]`.
    fn pair_body(&mut self, sv: &E, body: Vec<St>) -> R<Vec<St>> {
        let mut partners: Vec<(usize, Reg)> = Vec::new();
        let mut out = Vec::new();
        let body2 = body.iter().map(|s| self.p_stmt(sv, s, &mut partners)).collect::<R<Vec<_>>>()?;
        for (v, r) in &partners {
            out.push(St::PLoad(*r, *v));
        }
        out.extend(body2);
        Ok(out)
    }

    /// Under the pair guard `lidx < s <= T / 2`, `lidx + s < T`: a mask `& m` with `m + 1 >= T` changes nothing.
    fn pair_unmask(&self, i: &E) -> E {
        let i2 = self.unmask(i);
        if let E::Bin(BinaryOperator::And, x, m, _) = &i2 {
            if let Some(m) = lit_u(m) {
                if m.wrapping_add(1).is_power_of_two() && m as u64 + 1 >= self.threads() as u64 {
                    return (**x).clone();
                }
            }
        }
        i2
    }

    fn p_e(&mut self, sv: &E, e: &E, partners: &mut Vec<(usize, Reg)>) -> R<E> {
        Ok(match e {
            E::Wg(v, i, w) => {
                let i2 = self.pair_unmask(i);
                if is_lidx(&i2, self.wg) {
                    E::WgOwn(*v, u(E::Lit(Lit::U(0))), *w)
                } else if self.minus_lidx(&i2).is_some_and(|r| self.unmask(&r) == self.unmask(sv)) {
                    let r = match partners.iter().find(|(pv, _)| pv == v) {
                        Some((_, r)) => *r,
                        None => {
                            let r = self.reg(*w)?;
                            partners.push((*v, r));
                            r
                        }
                    };
                    E::Reg(r, *w)
                } else {
                    return gap(format!("a pair phase reading {}[{:?}], neither the own element nor the partner", self.wgvars[*v].name, i));
                }
            }
            E::WgA(..) | E::WgRmw(..) | E::Rmw(..) => return gap("an atomic inside a pair phase"),
            o => {
                let mut err = None;
                let mut kids = Vec::new();
                for k in kids_of(o) {
                    match self.p_e(sv, k, partners) {
                        Ok(x) => kids.push(x),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                if let Some(e) = err {
                    return Err(e);
                }
                let mut it = kids.into_iter();
                self.map_kids(o, &mut |_, _| Ok(it.next().unwrap_or(E::Lit(Lit::U(0)))))?
            }
        })
    }

    fn p_stmt(&mut self, sv: &E, s: &St, partners: &mut Vec<(usize, Reg)>) -> R<St> {
        Ok(match s {
            St::WgSet(v, i, x) => {
                if !is_lidx(&self.pair_unmask(i), self.wg) {
                    return gap(format!("a pair phase writing {}[{:?}], not the own element", self.wgvars[*v].name, i));
                }
                St::WgOwnSet(*v, E::Lit(Lit::U(0)), self.p_e(sv, x, partners)?)
            }
            St::Set(r, e) => St::Set(*r, self.p_e(sv, e, partners)?),
            St::ParrSet(p, a, b) => St::ParrSet(*p, self.p_e(sv, a, partners)?, self.p_e(sv, b, partners)?),
            St::SetLane(r, a, b) => St::SetLane(*r, self.p_e(sv, a, partners)?, self.p_e(sv, b, partners)?),
            St::If(c, a, b) => {
                let c2 = self.p_e(sv, c, partners)?;
                let a2 = a.iter().map(|x| self.p_stmt(sv, x, partners)).collect::<R<Vec<_>>>()?;
                let b2 = b.iter().map(|x| self.p_stmt(sv, x, partners)).collect::<R<Vec<_>>>()?;
                St::If(c2, a2, b2)
            }
            o => return gap(format!("the statement {o:?} inside a pair phase")),
        })
    }

    /// A solo phase's body: invocation 0 writes any element.
    fn solo_body(&self, body: Vec<St>) -> R<Vec<St>> {
        body.into_iter().map(|s| self.s_stmt(s)).collect()
    }

    fn s_stmt(&self, s: St) -> R<St> {
        Ok(match s {
            St::WgSet(v, i, x) => St::WgPut(v, i, x),
            St::If(c, a, b) => St::If(c, self.solo_body(a)?, self.solo_body(b)?),
            St::Loop(mut l) => {
                l.body = self.solo_body(l.body)?;
                St::Loop(l)
            }
            St::Barrier | St::StorageBarrier => return gap("a barrier inside invocation 0's code"),
            o => o,
        })
    }
}

/// The element family of a workgroup array's rows (`pool.py` ELEMS).
fn elem_key(w: Wt) -> Option<&'static str> {
    Some(match (w.s, w.lanes) {
        (Sc::U32, 1) => "u",
        (Sc::F32, 1) => "f",
        (Sc::I32, 1) => "n",
        (Sc::F32, 4) => "v4",
        _ => return None,
    })
}

fn unreachable_phase() -> R<Tree> {
    gap("our software has a bug: a phase of no kind")
}

