//! GPU batch evaluator: many expressions, one dispatch, data resident.
//!
//! # Why this shape
//!
//! An e-graph saturation turns one gene into a whole equivalence class, and a
//! population turns that into a lot of expressions:
//!
//! ```text
//! 100 individuals x 3 genes x ~20 e-class variants x 3 wrappers = ~18,000
//! ```
//!
//! Every one of those is the same interpreter over a different token array,
//! evaluated against the same rows. No communication between them. That is the
//! shape a GPU is for.
//!
//! The dataset is uploaded ONCE and stays resident: subsequent generations
//! write only a token buffer (a few hundred KB) and dispatch. What comes back
//! is a reduction — per-expression error terms — not predictions, so the return
//! trip is ~6 floats per expression rather than rows x expressions.
//!
//! # Why an interpreter, not generated shaders
//!
//! Compiling WGSL per chromosome costs milliseconds; evaluating 800 rows costs
//! microseconds. So there is ONE kernel, compiled once, and the population
//! arrives as data. Karva is already a flat, arity-bounded linearisation, so a
//! gene is a token array and needs no parsing.
//!
//! # Why the control flow does not diverge
//!
//! Karva is level-order, so a node's children sit at positions determined by
//! the arities before it — computable from the TOKENS ALONE, independent of the
//! data. Child offsets are resolved once on the host into an index table, and
//! the kernel then walks the array in reverse: every node's children are at
//! higher indices, so one backward linear scan yields the root. Threads in a
//! warp run the same op sequence over different rows, which is the property
//! that makes this worth doing at all.
//!
//! # Measured sizes (why a fixed scratch works)
//!
//! Real k-expressions are short — median 1-2 tokens, p90 6-7, max 34 across
//! head=16 and head=48 populations — because decoding stops as soon as arity is
//! satisfied. So [`MAX_NODES`] scratch floats per thread fits in registers and
//! occupancy is not the constraint I expected it to be.
//!
//! # Precision
//!
//! f32 on device. That is fine for ranking and search, and NOT fine for an
//! exact-recovery gate: a recovered Lorentz form passed at 1.6e-13 relative
//! error, which f32 cannot represent. Use this to shortlist, then re-score
//! finalists on the CPU f64 evaluator in [`crate::eval`], which stays the
//! correctness reference.

/// Longest expression this evaluator handles — THE ONE NODE LIMIT. Every shader
/// in fuller and in phylu's decoder sizes its per-thread scratch from this
/// through [`with_limits`], so raising it here raises it everywhere. It was 64,
/// chosen when a k-expression's p90 was 6-7 nodes and the max 34; at 5000+5000
/// populations with three-gene models judged as one flat expression, 64 sat
/// below the laws (the snap guard skipped 287 models in one trial for size,
/// and a gene of 43+ nodes could not express at all). 128 is the measured next
/// step; see `docs/` for the throughput cost.
pub const MAX_NODES: usize = 128;

/// Nodes of work area a shader that grafts a template onto a slot needs: the
/// slot plus the largest template (15) and a `Neg` over it.
pub const WORK_NODES: usize = MAX_NODES + 16;

/// A shader's source with its fixed-size scratch sized from [`MAX_NODES`]. The
/// WGSL files are written against the historical 64 (and 80 for the work area)
/// so they read as plain WGSL; this is the only place those literals are
/// rewritten, and every `ShaderSource::Wgsl` in the crate goes through it.
pub fn with_limits(src: &str) -> String {
    src.replace("const MAX_NODES: u32 = 64u;", &format!("const MAX_NODES: u32 = {MAX_NODES}u;"))
        .replace("const SLOT: u32 = 64u;", &format!("const SLOT: u32 = {MAX_NODES}u;"))
        .replace("const WORK: u32 = 80u;", &format!("const WORK: u32 = {WORK_NODES}u;"))
        .replace("array<f32, 64>", &format!("array<f32, {MAX_NODES}>"))
        .replace("array<u32, 64>", &format!("array<u32, {MAX_NODES}>"))
        .replace("array<u32, 80>", &format!("array<u32, {WORK_NODES}>"))
        .replace("array<Node, 64>", &format!("array<Node, {MAX_NODES}>"))
        .replace("array<Node, 80>", &format!("array<Node, {WORK_NODES}>"))
        .replace("array<u32, 128>", &format!("array<u32, {}>", 2 * MAX_NODES))
}

/// wgpu's per-dimension workgroup cap (`max_compute_workgroups_per_dimension`).
pub const MAX_GROUPS_PER_DIM: u32 = 65535;

/// Opcodes. These MUST match `crate::eval` semantics exactly — in particular
/// the `Protected*` variants, which are not "raw op plus a guard" but distinct
/// functions with the engine's own behaviour on negatives, zero and overflow.
/// A divergence here silently breaks fuller's soundness claims, so the parity
/// test against the CPU evaluator is the gate on this whole module.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Leaf: read variable `arg0` from the resident data buffer.
    Var = 0,
    /// Leaf: literal, value carried in the token's `konst` field.
    Num = 1,
    Add = 2,
    Sub = 3,
    Mul = 4,
    Div = 5,
    Neg = 6,
    Abs = 7,
    Sqrt = 8,
    Log = 9,
    Exp = 10,
    Sin = 11,
    Cos = 12,
    Tan = 13,
    Tanh = 14,
    Pow = 15,
    Pow2 = 16,
    Pow3 = 17,
    Inv = 18,
    ProtectedDiv = 19,
    ProtectedSqrt = 20,
    ProtectedLog = 21,
    ProtectedExp = 22,
    ProtectedInv = 23,
    Asin = 24,
    Acos = 25,
    ProtectedAsin = 26,
    ProtectedAcos = 27,
}

impl Op {
    /// Map a `Math` constructor name to an opcode.
    pub fn from_math(name: &str) -> Option<Op> {
        Some(match name {
            "Var" => Op::Var,
            "Num" => Op::Num,
            "Add" => Op::Add,
            "Sub" => Op::Sub,
            "Mul" => Op::Mul,
            "Div" => Op::Div,
            "Neg" => Op::Neg,
            "Abs" => Op::Abs,
            "Sqrt" => Op::Sqrt,
            "Log" => Op::Log,
            "Exp" => Op::Exp,
            "Sin" => Op::Sin,
            "Cos" => Op::Cos,
            "Tan" => Op::Tan,
            "Tanh" => Op::Tanh,
            "Pow" => Op::Pow,
            "Pow2" => Op::Pow2,
            "Pow3" => Op::Pow3,
            "Inv" => Op::Inv,
            "ProtectedDiv" => Op::ProtectedDiv,
            "ProtectedSqrt" => Op::ProtectedSqrt,
            "ProtectedLog" => Op::ProtectedLog,
            "ProtectedExp" => Op::ProtectedExp,
            "ProtectedInv" => Op::ProtectedInv,
            "Asin" => Op::Asin,
            "Acos" => Op::Acos,
            "ProtectedAsin" => Op::ProtectedAsin,
            "ProtectedAcos" => Op::ProtectedAcos,
            _ => return None,
        })
    }

    pub fn arity(self) -> usize {
        match self {
            Op::Var | Op::Num => 0,
            Op::Neg
            | Op::Abs
            | Op::Sqrt
            | Op::Log
            | Op::Exp
            | Op::Sin
            | Op::Cos
            | Op::Tan
            | Op::Tanh
            | Op::Pow2
            | Op::Pow3
            | Op::Inv
            | Op::ProtectedSqrt
            | Op::ProtectedLog
            | Op::ProtectedExp
            | Op::ProtectedInv
            | Op::Asin
            | Op::Acos
            | Op::ProtectedAsin
            | Op::ProtectedAcos => 1,
            _ => 2,
        }
    }
}

/// One node, as uploaded. 16 bytes, so a 64-node expression is 1 KiB and a
/// 18,000-expression population is ~18 MiB — well inside a 128 MiB Metal
/// binding cap, no banding needed at these sizes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuNode {
    /// [`Op`] as u32.
    pub op: u32,
    /// `Var`: column index into the resident data buffer. Otherwise: index of
    /// the first child, resolved on the host.
    pub arg0: u32,
    /// Index of the second child for binary ops; unused otherwise.
    pub arg1: u32,
    /// Literal value for `Num`; unused otherwise.
    pub konst: f32,
}

/// The WGSL interpreter.
///
/// One thread per `(expression, row)`. Each walks its own node slice backwards,
/// filling a scratch array, and writes the root value. Control flow is uniform
/// across the rows of one expression: the `switch` selects on the opcode, which
/// is a property of the token array, not of the data.
///
/// Protected-op semantics are written out explicitly rather than reusing the
/// raw ops, because they are different functions. `ProtectedExp` is uncapped
/// and returns +inf on overflow, matching the engine.
pub const EVAL_WGSL: &str = r#"
struct Node {
    op: u32,
    arg0: u32,
    arg1: u32,
    konst: f32,
};

struct Meta {
    n_expr: u32,
    n_rows: u32,
    n_vars: u32,
    // Invocations per dispatch ROW (groups_x * 64). One dimension caps at
    // 65535 workgroups = 4.19M invocations, and the population x e-class
    // cross-product is far past that, so the dispatch is 2-D.
    stride: u32,
    // 1 = also write every subtree's value to `partials`, laid out
    // partials[(expr * MAX_NODES + k) * n_rows + row] = scratch[k]. 0 = leave
    // `partials` untouched (the ordinary eval, byte-identical to before this
    // field existed). Every subtree value is ALREADY computed in `scratch`; this
    // only decides whether it is emitted.
    emit_partials: u32,
};

@group(0) @binding(0) var<storage, read>       nodes:   array<Node>;
// expr i owns nodes[offsets[i] .. offsets[i] + lengths[i])
@group(0) @binding(1) var<storage, read>       offsets: array<u32>;
@group(0) @binding(2) var<storage, read>       lengths: array<u32>;
// row-major: data[row * n_vars + col]. Uploaded once, never moved.
@group(0) @binding(3) var<storage, read>       data:    array<f32>;
@group(0) @binding(4) var<storage, read_write> out:     array<f32>;
@group(0) @binding(5) var<uniform>             cfg:     Meta;
// Per-subtree values, written only when cfg.emit_partials == 1u. When off it is
// bound to a 1-element dummy and never touched.
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;

const MAX_NODES: u32 = 64u;
fn nan() -> f32 { return bitcast<f32>(0x7fc00000u); }
fn pos_inf() -> f32 { return bitcast<f32>(0x7f800000u); }
// NaN fails every comparison, so this is false for NaN and for +-inf.
fn is_finite(x: f32) -> bool { return abs(x) <= 3.4028235e38; }
fn is_inf(x: f32) -> bool { return abs(x) > 3.4028235e38; }
// f32 cannot carry the PHASE of an angle this large: at 1e7 adjacent floats are
// 1.0 apart, a sixth of a period. Metal's sin/cos are accurate to 1e-8 up to
// 3e6 (measured) and return exactly 0.0 from 1e7 on — a wrong number that
// looks valid. Beyond this the expression is refused, not answered.
const TRIG_MAX_ARG: f32 = 1.0e7;
fn trig_ok(x: f32) -> bool { return abs(x) < TRIG_MAX_ARG; }

@compute @workgroup_size(64, 1, 1)
fn eval_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * cfg.stride + gid.x;
    let total = cfg.n_expr * cfg.n_rows;
    if (idx >= total) { return; }

    let expr = idx / cfg.n_rows;
    let row  = idx % cfg.n_rows;

    let base = offsets[expr];
    let n    = lengths[expr];
    if (n == 0u || n > MAX_NODES) {
        out[idx] = bitcast<f32>(0x7fc00000u);   // NaN: caller falls back to CPU
        return;
    }

    var scratch: array<f32, 64>;
    // Set when the ENGINE would raise rather than return: Python's math.sin /
    // math.cos raise ValueError on +-inf, and the engine rejects the whole
    // individual. A NaN is not enough to say that — ProtectedSqrt maps
    // non-finite to 0.0 and would swallow it — so it is carried separately.
    var poison: bool = false;

    // Backward scan. Karva is level-order, so every child index is greater
    // than its parent's — one reverse pass resolves the whole tree with no
    // stack and no recursion.
    var k: u32 = n;
    loop {
        if (k == 0u) { break; }
        k = k - 1u;
        let nd = nodes[base + k];
        var v: f32 = 0.0;
        switch (nd.op) {
            case 0u: { v = data[row * cfg.n_vars + nd.arg0]; }   // Var
            case 1u: { v = nd.konst; }                            // Num
            case 2u: { v = scratch[nd.arg0] + scratch[nd.arg1]; }
            case 3u: { v = scratch[nd.arg0] - scratch[nd.arg1]; }
            case 4u: { v = scratch[nd.arg0] * scratch[nd.arg1]; }
            // Raw ops carry the crate's div0 contract: division by zero is
            // NaN, NOT IEEE infinity. Bare `/` gives inf, which diverges from
            // eval.rs and would put a +inf member and a NaN member in the same
            // e-class (Pow(x,-1) and Inv(x) are equivalent under the rules).
            case 5u: {                                             // Div
                let d = scratch[nd.arg1];
                if (d == 0.0) { v = nan(); } else { v = scratch[nd.arg0] / d; }
            }
            case 6u: { v = -scratch[nd.arg0]; }
            case 7u: { v = abs(scratch[nd.arg0]); }
            case 8u: {                                             // Sqrt
                let a = scratch[nd.arg0];
                if (a < 0.0) { v = nan(); } else { v = sqrt(a); }
            }
            case 9u: {                                             // Log
                let a = scratch[nd.arg0];
                if (a <= 0.0) { v = nan(); } else { v = log(a); }
            }
            case 10u: { v = exp(scratch[nd.arg0]); }
            // WGSL leaves trig of a non-finite input UNDEFINED, and Metal
            // returns a finite number for cos(inf): an individual the engine
            // rejects was being scored. Non-finite in -> NaN out, and +-inf
            // poisons the expression (see `poison`).
            case 11u: {
                let a = scratch[nd.arg0];
                if (trig_ok(a)) { v = sin(a); } else { v = nan(); poison = poison || !(a != a); }
            }
            case 12u: {
                let a = scratch[nd.arg0];
                if (trig_ok(a)) { v = cos(a); } else { v = nan(); poison = poison || !(a != a); }
            }
            case 13u: {                                            // Tan
                let a = scratch[nd.arg0];
                if (trig_ok(a)) {
                    let c = cos(a);
                    if (c == 0.0) { v = nan(); } else { v = sin(a) / c; }
                } else { v = nan(); poison = poison || !(a != a); }
            }
            // Tanh: Metal's tanh is 0 at 44 and NaN past it (exp(2x) overflows
            // inside it), where the engine's math.tanh is +-1. Past 20 the f32
            // answer IS +-1 exactly (tanh(20) = 1 - 8e-18), so saturate there;
            // a NaN input stays NaN, as the engine's does.
            case 14u: {
                let a = scratch[nd.arg0];
                if (a != a) { v = a; } else if (a > 20.0) { v = 1.0; } else if (a < -20.0) { v = -1.0; } else { v = tanh(a); }
            }
            case 15u: {                                            // Pow
                let a = scratch[nd.arg0];
                let b = scratch[nd.arg1];
                if (a == 0.0 && b < 0.0) { v = nan(); } else { v = pow(a, b); }
            }
            case 16u: { let a = scratch[nd.arg0]; v = a * a; }
            case 17u: { let a = scratch[nd.arg0]; v = a * a * a; }
            case 18u: {                                            // Inv
                let a = scratch[nd.arg0];
                if (a == 0.0) { v = nan(); } else { v = 1.0 / a; }
            }
            // Protected ops: distinct functions, not guarded raw ops. Each
            // mirrors the ENGINE's primitive exactly, including what it does
            // with a non-finite input — see eval.rs for the definitions.
            // ProtectedDiv: 0.0 when |b| < 1e-6 (protected_div_zero), not
            // only when b == 0: b = sin(omega * -2.3e-8) = -1e-7 gave 0 on the
            // engine and c / -1e-7 here.
            case 19u: {
                let d = scratch[nd.arg1];
                if (abs(d) < 1e-6) { v = 0.0; } else { v = scratch[nd.arg0] / d; }
            }
            case 20u: {                                            // ProtectedSqrt
                let a = scratch[nd.arg0];
                if (is_finite(a)) { v = sqrt(abs(a)); } else { v = 0.0; }
            }
            case 21u: {                                            // ProtectedLog
                let a = scratch[nd.arg0];
                if (!is_finite(a) || a == 0.0) { v = pos_inf(); } else { v = log(abs(a)); }
            }
            case 22u: {                                            // ProtectedExp
                let a = scratch[nd.arg0];
                if (is_finite(a)) { v = exp(a); } else { v = pos_inf(); }
            }
            case 23u: {                                            // ProtectedInv
                let a = scratch[nd.arg0];
                if (a == 0.0) { v = 1.0; } else { v = 1.0 / a; }
            }
            // Inverse trig. WGSL leaves asin/acos outside [-1, 1] UNDEFINED, so
            // they are never called there: the raw ops are NaN outside the
            // domain (a NaN input fails the comparison, so NaN in -> NaN out),
            // the protected ones clamp, and are 0.0 on a non-finite input like
            // ProtectedSqrt.
            case 24u: {                                            // Asin
                let a = scratch[nd.arg0];
                if (abs(a) <= 1.0) { v = asin(a); } else { v = nan(); }
            }
            case 25u: {                                            // Acos
                let a = scratch[nd.arg0];
                if (abs(a) <= 1.0) { v = acos(a); } else { v = nan(); }
            }
            case 26u: {                                            // ProtectedAsin
                let a = scratch[nd.arg0];
                if (is_finite(a)) { v = asin(clamp(a, -1.0, 1.0)); } else { v = 0.0; }
            }
            case 27u: {                                            // ProtectedAcos
                let a = scratch[nd.arg0];
                if (is_finite(a)) { v = acos(clamp(a, -1.0, 1.0)); } else { v = 0.0; }
            }
            default: { v = bitcast<f32>(0x7fc00000u); }
        }
        scratch[k] = v;
    }

    if (poison) { out[idx] = nan(); } else { out[idx] = scratch[0]; }

    // EMIT PARTIALS: every subtree's value, already in scratch. Off by default,
    // so the ordinary eval writes nothing here and is unchanged.
    if (cfg.emit_partials == 1u) {
        var j: u32 = 0u;
        loop {
            if (j >= n) { break; }
            var pv: f32 = scratch[j];
            if (poison) { pv = nan(); }
            partials[(expr * MAX_NODES + j) * cfg.n_rows + row] = pv;
            j = j + 1u;
        }
    }
}
"#;

/// A flattened population, ready to upload.
#[derive(Debug, Default, Clone)]
pub struct ExprBatch {
    pub nodes: Vec<GpuNode>,
    pub offsets: Vec<u32>,
    pub lengths: Vec<u32>,
}

impl ExprBatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one expression's nodes. Returns its index in the batch.
    ///
    /// `nodes` must already be in karva level order with child indices
    /// resolved LOCALLY (relative to this expression), which is what the
    /// kernel's `scratch` indexing expects.
    pub fn push(&mut self, nodes: &[GpuNode]) -> usize {
        let idx = self.lengths.len();
        self.offsets.push(self.nodes.len() as u32);
        self.lengths.push(nodes.len() as u32);
        self.nodes.extend_from_slice(nodes);
        idx
    }

    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    /// Expressions too long for the kernel's fixed scratch. The caller must
    /// evaluate these on the CPU — the kernel marks them NaN rather than
    /// truncating, but knowing up front avoids a wasted round trip.
    pub fn oversized(&self) -> Vec<usize> {
        self.lengths
            .iter()
            .enumerate()
            .filter(|(_, &l)| l as usize > MAX_NODES || l == 0)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Parse a `Math` s-expression into level-order [`GpuNode`]s.
///
/// This is the piece that makes the kernel usable: without it the evaluator
/// can only be driven by hand-built node arrays, which tests nothing about
/// real chromosomes.
///
/// Emits BREADTH-FIRST, because the kernel's backward scan relies on every
/// child sitting at a higher index than its parent. A depth-first emit would
/// break that invariant and silently compute nonsense.
///
/// `vars` maps a variable name to its column in the resident data buffer. A
/// name not in `vars` is an error rather than a default, because silently
/// reading column 0 would produce a plausible wrong answer.
pub fn math_to_nodes(expr: &str, vars: &[String]) -> Result<Vec<GpuNode>, String> {
    let toks = tokenize(expr);
    let mut pos = 0usize;
    let tree = parse(&toks, &mut pos)?;
    if pos != toks.len() {
        return Err(format!("trailing tokens at {pos} in {expr:?}"));
    }
    flatten(&tree, vars)
}

/// Karva tokens straight to GPU nodes.
///
/// This is the engine-facing entry point: the engine holds chromosomes as
/// karva, not as Math s-expressions. Composes the existing
/// [`crate::karva::karva_to_terms`] decoder with [`math_to_nodes`], so the
/// karva semantics (head/tail split, level-order child consumption, arity
/// bounding) stay defined in exactly one place rather than being reimplemented
/// here and drifting.
pub fn karva_to_nodes(
    head: &[crate::karva::Token],
    tail: &[crate::karva::Token],
    pset: &crate::karva::PsetSpec,
    vars: &[String],
) -> Result<Vec<GpuNode>, String> {
    let math = crate::karva::karva_to_terms(head, tail, pset)?;
    math_to_nodes(&math, vars)
}

#[derive(Debug, Clone)]
enum Tree {
    Num(f64),
    Var(String),
    App(String, Vec<Tree>),
}

fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_str = false;
    for c in s.chars() {
        match c {
            '"' => {
                in_str = !in_str;
                cur.push(c);
                if !in_str {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ if in_str => cur.push(c),
            '(' | ')' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(c.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn parse(toks: &[String], pos: &mut usize) -> Result<Tree, String> {
    let t = toks.get(*pos).ok_or("unexpected end of input")?;
    if t != "(" {
        return Err(format!("expected '(' at {pos}, found {t:?}"));
    }
    *pos += 1;
    let head = toks.get(*pos).ok_or("missing constructor")?.clone();
    *pos += 1;

    let node = match head.as_str() {
        "Num" => {
            let v = toks.get(*pos).ok_or("Num missing value")?;
            *pos += 1;
            Tree::Num(v.parse::<f64>().map_err(|e| format!("bad Num {v:?}: {e}"))?)
        }
        "Var" => {
            let v = toks.get(*pos).ok_or("Var missing name")?;
            *pos += 1;
            Tree::Var(v.trim_matches('"').to_string())
        }
        _ => {
            let op = Op::from_math(&head)
                .ok_or_else(|| format!("unknown Math constructor {head:?}"))?;
            let mut kids = Vec::new();
            for _ in 0..op.arity() {
                kids.push(parse(toks, pos)?);
            }
            Tree::App(head.clone(), kids)
        }
    };

    match toks.get(*pos) {
        Some(t) if t == ")" => {
            *pos += 1;
            Ok(node)
        }
        other => Err(format!("expected ')' closing {head}, found {other:?}")),
    }
}

/// Breadth-first flatten with child indices resolved.
fn flatten(root: &Tree, vars: &[String]) -> Result<Vec<GpuNode>, String> {
    let mut out: Vec<GpuNode> = Vec::new();
    // (tree, index of the slot already reserved in `out`)
    let mut queue: std::collections::VecDeque<(&Tree, usize)> =
        std::collections::VecDeque::new();

    out.push(GpuNode { op: 0, arg0: 0, arg1: 0, konst: 0.0 });
    queue.push_back((root, 0));

    while let Some((tree, slot)) = queue.pop_front() {
        match tree {
            Tree::Num(v) => {
                out[slot] = GpuNode {
                    op: Op::Num as u32,
                    arg0: 0,
                    arg1: 0,
                    konst: *v as f32,
                };
            }
            Tree::Var(name) => {
                let col = vars
                    .iter()
                    .position(|v| v == name)
                    .ok_or_else(|| format!("variable {name:?} not in {vars:?}"))?;
                out[slot] = GpuNode {
                    op: Op::Var as u32,
                    arg0: col as u32,
                    arg1: 0,
                    konst: 0.0,
                };
            }
            Tree::App(head, kids) => {
                let op = Op::from_math(head)
                    .ok_or_else(|| format!("unknown constructor {head:?}"))?;
                // Reserve the children's slots now so their indices are known.
                let first = out.len();
                for _ in kids {
                    out.push(GpuNode { op: 0, arg0: 0, arg1: 0, konst: 0.0 });
                }
                out[slot] = GpuNode {
                    op: op as u32,
                    arg0: first as u32,
                    arg1: if kids.len() > 1 { first as u32 + 1 } else { 0 },
                    konst: 0.0,
                };
                for (i, k) in kids.iter().enumerate() {
                    queue.push_back((k, first + i));
                }
            }
        }
        if out.len() > MAX_NODES {
            return Err(format!("expression exceeds MAX_NODES ({MAX_NODES})"));
        }
    }
    Ok(out)
}

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    /// Holds the device and the RESIDENT dataset. Built once per fit; every
    /// generation reuses it, uploading only tokens.
    pub struct GpuEvaluator {
        // ONE DEVICE for the whole fit. The evolution engine's own kernels
        // (phylu's decoder, sampler, variation, scorer, typer, HFF) run on this
        // device too, so a buffer one of them writes is a buffer the next one
        // binds — a bind group cannot take a buffer from another device. The
        // handles are shared, not cloned (wgpu 0.20's are not `Clone`).
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        data_buf: wgpu::Buffer,
        // A persistent 1-element storage buffer bound at binding 6 when
        // emit_partials is 0. It must OUTLIVE every dispatch that references it —
        // `eval_pass` only RECORDS the pass and the caller submits later, so a
        // per-call dummy destroyed before that submit is a use-after-free
        // ("Buffer is destroyed" on Queue::submit). Owning it here ties its life
        // to the evaluator's instead.
        dummy_partials: wgpu::Buffer,
        n_rows: u32,
        n_vars: u32,
    }

    impl GpuEvaluator {
        /// Upload the dataset. `rows` is row-major, `n_vars` wide.
        pub fn new(rows: &[f32], n_vars: usize) -> Result<Self, String> {
            pollster::block_on(Self::new_async(rows, n_vars))
        }

        async fn new_async(rows: &[f32], n_vars: usize) -> Result<Self, String> {
            if n_vars == 0 || !rows.len().is_multiple_of(n_vars) {
                return Err(format!(
                    "rows ({}) not divisible by n_vars ({n_vars})",
                    rows.len()
                ));
            }
            let n_rows = (rows.len() / n_vars) as u32;

            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok_or("no GPU adapter")?;
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("fuller-eval-device"),
                        required_features: wgpu::Features::empty(),
                        // The adapter's real limits, not wgpu's portable
                        // defaults: the default 128 MiB storage binding is
                        // smaller than one generation's prediction buffer
                        // once e-class variants are in the batch.
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("request_device: {e}"))?;
            let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fuller-eval"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(with_limits(EVAL_WGSL))),
            });

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fuller-eval-layout"),
                entries: &(0..7)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: if i == 5 {
                            wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            }
                        } else {
                            wgpu::BindingType::Buffer {
                                // binding 4 (out) and 6 (partials) are written.
                                ty: wgpu::BufferBindingType::Storage {
                                    read_only: i != 4 && i != 6,
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            }
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fuller-eval-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "eval_main",
                compilation_options: Default::default(),
            });

            // THE resident buffer. Uploaded once; every later dispatch reuses it.
            let data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fuller-data-resident"),
                contents: bytemuck::cast_slice(rows),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // The persistent binding-6 dummy for emit_partials == 0 dispatches.
            // Lives as long as the evaluator, so a recorded-then-later-submitted
            // pass never references a destroyed buffer.
            let dummy_partials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fuller-partials-dummy"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE,
            });

            Ok(Self {
                device,
                queue,
                pipeline,
                layout,
                data_buf,
                dummy_partials,
                n_rows,
                n_vars: n_vars as u32,
            })
        }

        pub fn n_rows(&self) -> u32 {
            self.n_rows
        }

        /// The device and queue, for a kernel that consumes this evaluator's
        /// predictions where they are (`evolve::score`).
        pub fn device(&self) -> &wgpu::Device {
            &self.device
        }

        pub fn queue(&self) -> &wgpu::Queue {
            &self.queue
        }

        /// The device and queue, SHARED: what the evolution engine builds its
        /// own resident buffers and kernels on, so they and the evaluator's
        /// live on one device and bind each other's buffers.
        pub fn shared_device(&self) -> std::sync::Arc<wgpu::Device> {
            std::sync::Arc::clone(&self.device)
        }

        pub fn shared_queue(&self) -> std::sync::Arc<wgpu::Queue> {
            std::sync::Arc::clone(&self.queue)
        }

        /// Record the evaluation of `n_expr` expressions that are ALREADY on this
        /// device onto a caller's encoder — `buffers` is (nodes, offsets,
        /// lengths, out), in `eval_resident`'s layouts, `out` taking `n_expr *
        /// n_rows` f32 — so a kernel that built the expressions on the device
        /// (the snap graft) and one that reads the predictions (the snap guard)
        /// share one command buffer with it. The uniform buffer returned is the
        /// caller's to destroy after the submit.
        pub fn eval_pass(
            &self,
            enc: &mut wgpu::CommandEncoder,
            buffers: [&wgpu::Buffer; 4],
            n_expr: u32,
        ) -> Result<wgpu::Buffer, String> {
            let total = (n_expr as u64) * (self.n_rows as u64);
            if n_expr == 0 || total > u32::MAX as u64 {
                return Err(format!(
                    "batch of {n_expr} expressions x {} rows = {total} \
                     invocations: empty, or past the kernel's u32 index",
                    self.n_rows
                ));
            }
            let [nodes_buf, offs_buf, lens_buf, out_buf] = buffers;
            let groups = total.div_ceil(64) as u32;
            let groups_x = groups.min(MAX_GROUPS_PER_DIM);
            let groups_y = groups.div_ceil(groups_x);
            let meta = [n_expr, self.n_rows, self.n_vars, groups_x * 64, 0];
            let meta_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("meta"),
                    contents: bytemuck::cast_slice(&meta),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            // emit_partials is 0 here, so `partials` is never written; bind the
            // evaluator's persistent 1-element dummy (it must outlive this
            // recorded pass, which the caller submits later).
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: nodes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: offs_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: lens_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: meta_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.dummy_partials.as_entire_binding() },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            Ok(meta_buf)
        }

        /// [`eval_pass`] over a RANGE of expressions already on the device. The
        /// same kernel; `offsets` and `lengths` are bound from entry `expr_base`
        /// on, so a caller holding the WHOLE population's offsets and lengths
        /// resident evaluates one block of `n_expr` of them into `out`
        /// (`n_expr * n_rows` f32) without re-uploading anything. `offsets`
        /// stay GLOBAL into `nodes`, which is bound whole. This is what the
        /// device decoder's outputs are evaluated through, block by block.
        ///
        /// `expr_base * 4` must be a multiple of the device's
        /// `min_storage_buffer_offset_alignment` — the caller chooses its block
        /// size so every block start is; an unaligned base is an error, not a
        /// silent mis-read.
        pub fn eval_pass_range(
            &self,
            enc: &mut wgpu::CommandEncoder,
            buffers: [&wgpu::Buffer; 4],
            n_expr: u32,
            expr_base: u32,
        ) -> Result<wgpu::Buffer, String> {
            self.eval_pass_range_with(enc, buffers, None, n_expr, expr_base)
        }

        /// [`Self::eval_pass_range`] that ALSO writes every subtree's value into
        /// `partials` when one is given, at `partials[(expr * MAX_NODES + k) * n_rows + row]`
        /// for `expr` in `0..n_expr` (batch coordinates, the base already
        /// applied) — the resident form of [`Self::eval_with_partials`]. A
        /// poisoned row (an infinite trig argument) is NaN in every one of its
        /// partials, as it is in `out`. Entries `k >= lengths[expr]` are NOT
        /// written and hold whatever the buffer held: a consumer reads the length.
        pub fn eval_pass_range_with(
            &self,
            enc: &mut wgpu::CommandEncoder,
            buffers: [&wgpu::Buffer; 4],
            partials: Option<&wgpu::Buffer>,
            n_expr: u32,
            expr_base: u32,
        ) -> Result<wgpu::Buffer, String> {
            let align = u64::from(self.device.limits().min_storage_buffer_offset_alignment);
            let offset = u64::from(expr_base) * 4;
            if offset % align != 0 {
                return Err(format!(
                    "eval_pass_range: expression base {expr_base} is byte offset {offset}, \
                     not a multiple of the device's storage offset alignment {align}"
                ));
            }
            let total = (n_expr as u64) * (self.n_rows as u64);
            if n_expr == 0 || total > u32::MAX as u64 {
                return Err(format!(
                    "batch of {n_expr} expressions x {} rows = {total} \
                     invocations: empty, or past the kernel's u32 index",
                    self.n_rows
                ));
            }
            let [nodes_buf, offs_buf, lens_buf, out_buf] = buffers;
            let groups = total.div_ceil(64) as u32;
            let groups_x = groups.min(MAX_GROUPS_PER_DIM);
            let groups_y = groups.div_ceil(groups_x);
            let meta = [n_expr, self.n_rows, self.n_vars, groups_x * 64, u32::from(partials.is_some())];
            let meta_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("meta"),
                    contents: bytemuck::cast_slice(&meta),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let offs_from = wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: offs_buf, offset, size: None });
            let lens_from = wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: lens_buf, offset, size: None });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: nodes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: offs_from },
                    wgpu::BindGroupEntry { binding: 2, resource: lens_from },
                    wgpu::BindGroupEntry { binding: 3, resource: self.data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: meta_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: partials.unwrap_or(&self.dummy_partials).as_entire_binding() },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            Ok(meta_buf)
        }

        /// Evaluate `batch` and LEAVE the predictions on the device: a buffer of
        /// `batch.len() * n_rows` f32, expression-major, usable as a storage
        /// binding and as a copy source. The caller destroys it. `eval` is this
        /// plus a read-back; an oversized expression is NaN in both.
        pub fn eval_resident(&self, batch: &ExprBatch) -> Result<wgpu::Buffer, String> {
            if batch.is_empty() {
                return Err("eval_resident: an empty batch has no predictions".to_string());
            }
            let n_expr = batch.len() as u32;
            let total = (n_expr as u64) * (self.n_rows as u64);
            if total > u32::MAX as u64 {
                return Err(format!(
                    "batch of {n_expr} expressions x {} rows = {total} \
                     invocations overflows the kernel's u32 index",
                    self.n_rows
                ));
            }
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            if batch.nodes.is_empty() {
                // Every expression failed to convert: the same all-NaN answer
                // `eval` gives, without a zero-sized binding.
                let nan = vec![f32::NAN; total as usize];
                return Ok(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("out"),
                    contents: bytemuck::cast_slice(&nan),
                    usage,
                }));
            }
            let node_bytes: Vec<u32> = batch
                .nodes
                .iter()
                .flat_map(|n| [n.op, n.arg0, n.arg1, n.konst.to_bits()])
                .collect();
            let nodes_buf = self.storage(&node_bytes, "nodes");
            let offs_buf = self.storage(&batch.offsets, "offsets");
            let lens_buf = self.storage(&batch.lengths, "lengths");
            let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size: total * 4,
                usage,
                mapped_at_creation: false,
            });
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let meta_buf = self.eval_pass(&mut enc, [&nodes_buf, &offs_buf, &lens_buf, &out_buf], n_expr)?;
            self.queue.submit(Some(enc.finish()));
            // Released once the submitted work that uses them has finished.
            nodes_buf.destroy();
            offs_buf.destroy();
            lens_buf.destroy();
            meta_buf.destroy();
            Ok(out_buf)
        }

        /// Evaluate every expression in `batch` over every resident row.
        ///
        /// Returns `n_expr * n_rows` values, expression-major. Oversized
        /// expressions come back NaN — see [`ExprBatch::oversized`].
        pub fn eval(&self, batch: &ExprBatch) -> Result<Vec<f32>, String> {
            if batch.is_empty() {
                return Ok(Vec::new());
            }
            // wgpu rejects a zero-sized binding, so a batch in which EVERY
            // expression failed to convert would panic inside create_bind_group
            // rather than returning. That is reachable in a real run: a
            // generation whose genes all carry unresolved RNC placeholders
            // produces exactly this. Return NaNs for the whole batch instead,
            // which is the same signal an individually-undecodable expression
            // gives.
            if batch.nodes.is_empty() {
                let total = batch.len() * self.n_rows as usize;
                return Ok(vec![f32::NAN; total]);
            }
            let n_expr = batch.len() as u32;
            let total = (n_expr as u64) * (self.n_rows as u64);

            let node_bytes: Vec<u32> = batch
                .nodes
                .iter()
                .flat_map(|n| [n.op, n.arg0, n.arg1, n.konst.to_bits()])
                .collect();

            let nodes_buf = self.storage(&node_bytes, "nodes");
            let offs_buf = self.storage(&batch.offsets, "offsets");
            let lens_buf = self.storage(&batch.lengths, "lengths");

            let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size: total * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: total * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            if total > u32::MAX as u64 {
                return Err(format!(
                    "batch of {n_expr} expressions x {} rows = {total} \
                     invocations overflows the kernel's u32 index",
                    self.n_rows
                ));
            }
            // 2-D dispatch: a single dimension caps at 65535 workgroups
            // (4.19M invocations); the e-class cross-product exceeds it.
            let groups = total.div_ceil(64) as u32;
            let groups_x = groups.min(MAX_GROUPS_PER_DIM);
            let groups_y = groups.div_ceil(groups_x);
            let meta = [n_expr, self.n_rows, self.n_vars, groups_x * 64, 0];
            let meta_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("meta"),
                    contents: bytemuck::cast_slice(&meta),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            // emit_partials 0: the evaluator's persistent dummy at binding 6.
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: nodes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: offs_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: lens_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: meta_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.dummy_partials.as_entire_binding() },
                ],
            });

            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, total * 4);
            self.queue.submit(Some(enc.finish()));

            let slice = read_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|e| format!("map_async channel: {e}"))?
                .map_err(|e| format!("map_async: {e}"))?;
            let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
            read_buf.unmap();

            // Release every per-dispatch buffer explicitly. Dropping the Rust
            // handle does NOT free the GPU allocation promptly — wgpu defers
            // it — so a long fit accumulates them and Metal reports
            // "Context leak detected, CoreAnalytics returned false" once per
            // dispatch. Measured 4 per generation row before this.
            //
            // The resident data buffer is NOT destroyed: it belongs to the
            // session and is the whole point of keeping one.
            nodes_buf.destroy();
            offs_buf.destroy();
            lens_buf.destroy();
            meta_buf.destroy();
            out_buf.destroy();
            read_buf.destroy();

            // Drain wgpu's deferred-release queue. submit() hands wgpu a
            // command buffer and staging allocations that it holds until the
            // device is polled; the poll inside map_async above WAITS for the
            // copy but does not run the maintain pass that actually frees
            // them. Without this the process grows ~23KB per dispatch —
            // measured 108MB -> 177MB over 300 dispatches, still climbing —
            // and Metal reports "Context leak detected" once per dispatch.
            self.device.poll(wgpu::Maintain::Poll);
            Ok(out)
        }

        /// Evaluate `batch` AND emit every subtree's value.
        ///
        /// Returns `(roots, partials)`:
        /// - `roots`: `n_expr * n_rows`, expression-major, as [`Self::eval`].
        /// - `partials`: `n_expr * MAX_NODES * n_rows`, indexed
        ///   `partials[(expr * MAX_NODES + k) * n_rows + row]` = the value of the
        ///   subtree rooted at node `k` of expression `expr` on that row. Slots
        ///   `k >= lengths[expr]` are untouched (leave them 0); the caller reads
        ///   only `k < lengths[expr]`.
        ///
        /// This is [`Self::eval`] with `emit_partials = 1` and a real partials
        /// buffer. It reuses the SAME scan and shader — the subtree values are
        /// already computed there; this only writes them out. The partials buffer
        /// is `n_expr * MAX_NODES * n_rows * 4` bytes, so pass a SMALL batch (the
        /// selected near-miss genes), not the whole population.
        pub fn eval_with_partials(&self, batch: &ExprBatch) -> Result<(Vec<f32>, Vec<f32>), String> {
            if batch.is_empty() || batch.nodes.is_empty() {
                let total = batch.len() * self.n_rows as usize;
                return Ok((vec![f32::NAN; total], vec![0.0; total * MAX_NODES]));
            }
            let n_expr = batch.len() as u32;
            let total = (n_expr as u64) * (self.n_rows as u64);
            let part_total = total * MAX_NODES as u64;
            if part_total > u32::MAX as u64 {
                return Err(format!(
                    "partials of {n_expr} expressions x {MAX_NODES} nodes x {} rows overflows the \
                     kernel's u32 index; pass fewer genes",
                    self.n_rows
                ));
            }
            let node_bytes: Vec<u32> = batch.nodes.iter().flat_map(|n| [n.op, n.arg0, n.arg1, n.konst.to_bits()]).collect();
            let nodes_buf = self.storage(&node_bytes, "nodes");
            let offs_buf = self.storage(&batch.offsets, "offsets");
            let lens_buf = self.storage(&batch.lengths, "lengths");
            let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size: total * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let part_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("partials"),
                size: part_total * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let part_read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("partials-readback"),
                size: part_total * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: total * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let groups = total.div_ceil(64) as u32;
            let groups_x = groups.min(MAX_GROUPS_PER_DIM);
            let groups_y = groups.div_ceil(groups_x);
            let meta = [n_expr, self.n_rows, self.n_vars, groups_x * 64, 1];
            let meta_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("meta"),
                contents: bytemuck::cast_slice(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: nodes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: offs_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: lens_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: meta_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: part_buf.as_entire_binding() },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, total * 4);
            enc.copy_buffer_to_buffer(&part_buf, 0, &part_read, 0, part_total * 4);
            self.queue.submit(Some(enc.finish()));

            let read = |buf: &wgpu::Buffer| -> Result<Vec<f32>, String> {
                let slice = buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                self.device.poll(wgpu::Maintain::Wait);
                rx.recv().map_err(|e| format!("map_async channel: {e}"))?.map_err(|e| format!("map_async: {e}"))?;
                let v = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
                buf.unmap();
                Ok(v)
            };
            let out = read(&read_buf)?;
            let partials = read(&part_read)?;

            nodes_buf.destroy();
            offs_buf.destroy();
            lens_buf.destroy();
            meta_buf.destroy();
            out_buf.destroy();
            read_buf.destroy();
            part_buf.destroy();
            part_read.destroy();
            self.device.poll(wgpu::Maintain::Poll);
            Ok((out, partials))
        }

        fn storage<T: bytemuck::Pod>(&self, v: &[T], label: &str) -> wgpu::Buffer {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(v),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        }
    }

    /// THE SUBTREE STATISTICS KERNEL: one invocation per (expression, node),
    /// reducing that subtree's partials over rows `row_lo..row_hi` to its
    /// min, max, sum and a finite flag — the numbers a fold judges flatness
    /// by, without a partial ever leaving the device. A node past the
    /// expression's length is (NaN, NaN, 0, 0). One thread walks one node's
    /// rows in order, so the f32 sum is the same every time.
    pub const STATS_WGSL: &str = r#"
struct Meta {
    n_expr: u32,
    n_rows: u32,
    row_lo: u32,
    row_hi: u32,
};
@group(0) @binding(0) var<storage, read>       partials: array<f32>;
@group(0) @binding(1) var<storage, read>       lengths:  array<u32>;
@group(0) @binding(2) var<storage, read_write> stats:    array<f32>;
@group(0) @binding(3) var<uniform>             cfg:      Meta;

const MAX_NODES: u32 = 64u;
fn nan() -> f32 { return bitcast<f32>(0x7fc00000u); }
fn is_finite(x: f32) -> bool { return abs(x) <= 3.4028235e38; }

@compute @workgroup_size(64, 1, 1)
fn stats_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cfg.n_expr * MAX_NODES) { return; }
    let expr = idx / MAX_NODES;
    let k = idx % MAX_NODES;
    let o = idx * 4u;
    if (k >= lengths[expr]) {
        stats[o] = nan(); stats[o + 1u] = nan(); stats[o + 2u] = 0.0; stats[o + 3u] = 0.0;
        return;
    }
    var lo: f32 = 3.4028235e38;
    var hi: f32 = -3.4028235e38;
    var sum: f32 = 0.0;
    var finite: f32 = 1.0;
    var row: u32 = cfg.row_lo;
    loop {
        if (row >= cfg.row_hi) { break; }
        let v = partials[idx * cfg.n_rows + row];
        if (!is_finite(v)) { finite = 0.0; break; }
        lo = min(lo, v);
        hi = max(hi, v);
        sum = sum + v;
        row = row + 1u;
    }
    stats[o] = lo; stats[o + 1u] = hi; stats[o + 2u] = sum; stats[o + 3u] = finite;
}
"#;

    /// One subtree's reduction over the rows asked for.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct SubtreeStat {
        pub min: f32,
        pub max: f32,
        pub sum: f32,
        /// Every row finite. False on any NaN or infinity — including a
        /// poisoned expression, whose partials are all NaN.
        pub finite: bool,
    }

    impl SubtreeStat {
        pub fn mean(&self, n_rows: usize) -> f64 {
            f64::from(self.sum) / n_rows as f64
        }
    }

    /// The most partial bytes one block may hold: banding keeps a wide dataset
    /// from asking the device for one enormous buffer.
    const PARTIALS_BYTES_CAP: u64 = 64 * 1024 * 1024;

    /// SUBTREE STATISTICS FOR A BATCH, on the device: the batch's node arrays
    /// go up, `eval_main` writes every subtree's value into a resident
    /// partials buffer, `stats_main` reduces them, and only the reductions
    /// (16 bytes a node) come back. Allocated once at construction for the
    /// block size the device's binding limit allows; a bigger batch runs in
    /// blocks. This is what a fold, on the search's beat or in the finishing
    /// kitchen sink, judges flatness from.
    pub struct SubtreeStats {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        nodes_buf: wgpu::Buffer,
        offsets_buf: wgpu::Buffer,
        lengths_buf: wgpu::Buffer,
        partials_buf: wgpu::Buffer,
        out_buf: wgpu::Buffer,
        stats_buf: wgpu::Buffer,
        staging: wgpu::Buffer,
        /// Expressions a block holds.
        block: usize,
        n_rows: u32,
    }

    impl SubtreeStats {
        pub fn new(evaluator: &GpuEvaluator) -> Result<Self, String> {
            let device = evaluator.device();
            let n_rows = evaluator.n_rows();
            let limits = device.limits();
            let per_expr = MAX_NODES as u64 * u64::from(n_rows) * 4;
            let cap = PARTIALS_BYTES_CAP.min(u64::from(limits.max_storage_buffer_binding_size));
            // A workgroup of 64 threads reduces 64 nodes, so an expression takes
            // MAX_NODES / 64 workgroups, and the dispatch is one-dimensional: a
            // block is at most one dimension's worth of workgroups.
            let block = ((cap / per_expr) as usize).min(MAX_GROUPS_PER_DIM as usize * 64 / MAX_NODES);
            if block == 0 {
                return Err(format!("subtree stats: one expression's {MAX_NODES} x {n_rows} partials do not fit the device's binding limit"));
            }
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fuller-subtree-stats"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(crate::gpu_eval::with_limits(STATS_WGSL))),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fuller-subtree-stats-layout"),
                entries: &(0..4)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: if i == 3 {
                            wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }
                        } else {
                            wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: i != 2 }, has_dynamic_offset: false, min_binding_size: None }
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fuller-subtree-stats-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "stats_main",
                compilation_options: Default::default(),
            });
            let storage = |bytes: u64, label: &str| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes.max(4),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            let stats_bytes = block as u64 * MAX_NODES as u64 * 16;
            Ok(Self {
                pipeline,
                layout,
                nodes_buf: storage(block as u64 * MAX_NODES as u64 * 16, "subtree stats nodes"),
                offsets_buf: storage(block as u64 * 4, "subtree stats offsets"),
                lengths_buf: storage(block as u64 * 4, "subtree stats lengths"),
                partials_buf: storage(block as u64 * per_expr, "subtree stats partials"),
                out_buf: storage(block as u64 * u64::from(n_rows) * 4, "subtree stats roots"),
                stats_buf: storage(stats_bytes, "subtree stats"),
                staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("subtree stats staging"),
                    size: stats_bytes.max(4),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                block,
                n_rows,
            })
        }

        /// Expressions one block holds.
        pub fn block(&self) -> usize {
            self.block
        }

        /// Every subtree's statistics over rows `row_lo..row_hi`, for every
        /// expression of `batch`: `MAX_NODES` entries an expression, in batch
        /// order, entries past an expression's length marked not finite. An
        /// expression the batch flagged oversized (length 0) is all not finite.
        pub fn run(&self, evaluator: &GpuEvaluator, batch: &ExprBatch, row_lo: u32, row_hi: u32) -> Result<Vec<SubtreeStat>, String> {
            if row_hi > self.n_rows || row_lo >= row_hi {
                return Err(format!("subtree stats: rows {row_lo}..{row_hi} of {}", self.n_rows));
            }
            let device = evaluator.device();
            let queue = evaluator.queue();
            let mut out: Vec<SubtreeStat> = Vec::with_capacity(batch.len() * MAX_NODES);
            let mut e0 = 0usize;
            while e0 < batch.len() {
                let n = self.block.min(batch.len() - e0);
                // The block's node arrays, re-based to the block, up in one write each.
                let node_lo = batch.offsets[e0] as usize;
                let node_hi = (e0 + n..batch.len()).next().map_or(batch.nodes.len(), |e| batch.offsets[e] as usize);
                let node_words: Vec<u32> = batch.nodes[node_lo..node_hi].iter().flat_map(|nd| [nd.op, nd.arg0, nd.arg1, nd.konst.to_bits()]).collect();
                let offsets: Vec<u32> = batch.offsets[e0..e0 + n].iter().map(|o| o - node_lo as u32).collect();
                let lengths = &batch.lengths[e0..e0 + n];
                if !node_words.is_empty() {
                    queue.write_buffer(&self.nodes_buf, 0, bytemuck::cast_slice(&node_words));
                }
                queue.write_buffer(&self.offsets_buf, 0, bytemuck::cast_slice(&offsets));
                queue.write_buffer(&self.lengths_buf, 0, bytemuck::cast_slice(lengths));
                let meta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("subtree stats meta"),
                    contents: bytemuck::cast_slice(&[n as u32, self.n_rows, row_lo, row_hi]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.partials_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.lengths_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.stats_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: meta.as_entire_binding() },
                    ],
                });
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("subtree stats") });
                // The root predictions are a by-product here, written to their own
                // buffer: two writable bindings may not alias.
                let eval_meta = evaluator.eval_pass_range_with(&mut enc, [&self.nodes_buf, &self.offsets_buf, &self.lengths_buf, &self.out_buf], Some(&self.partials_buf), n as u32, 0)?;
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(((n * MAX_NODES) as u32).div_ceil(64), 1, 1);
                }
                let stats_bytes = (n * MAX_NODES * 16) as u64;
                enc.copy_buffer_to_buffer(&self.stats_buf, 0, &self.staging, 0, stats_bytes);
                queue.submit(Some(enc.finish()));
                let slice = self.staging.slice(..stats_bytes);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                device.poll(wgpu::Maintain::Wait);
                rx.recv().map_err(|e| format!("map_async channel: {e}"))?.map_err(|e| format!("map_async: {e}"))?;
                {
                    let words = slice.get_mapped_range();
                    let f: &[f32] = bytemuck::cast_slice(&words);
                    out.extend(f.chunks_exact(4).map(|c| SubtreeStat { min: c[0], max: c[1], sum: c[2], finite: c[3] == 1.0 }));
                }
                self.staging.unmap();
                eval_meta.destroy();
                meta.destroy();
                device.poll(wgpu::Maintain::Poll);
                e0 += n;
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "gpu")]
pub use device::{GpuEvaluator, SubtreeStat, SubtreeStats, STATS_WGSL};

/// Per-expression error terms, reduced on the host from the device's
/// predictions. These are the columns an HFF objective vector is built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExprScore {
    pub mse: f64,
    pub mae: f64,
    pub max_err: f64,
    /// Rows that produced NaN or inf. A non-zero count means the expression is
    /// not total on this data, which a caller usually wants to rank last
    /// rather than average away.
    pub n_nonfinite: usize,
}

/// Reduce raw device predictions into per-expression scores.
///
/// Separated from the dispatch so it is testable without a GPU, and so the
/// same reduction applies whether predictions came from the device or the CPU
/// evaluator.
///
/// `preds` is expression-major: `preds[e * n_rows + r]`.
pub fn score_predictions(preds: &[f32], targets: &[f32], n_expr: usize) -> Vec<ExprScore> {
    let n_rows = targets.len();
    let mut out = Vec::with_capacity(n_expr);
    for e in 0..n_expr {
        let (mut se, mut ae, mut mx, mut bad) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for r in 0..n_rows {
            let p = preds[e * n_rows + r];
            if !p.is_finite() {
                bad += 1;
                continue;
            }
            let d = (p - targets[r]) as f64;
            se += d * d;
            ae += d.abs();
            mx = mx.max(d.abs());
        }
        let good = (n_rows - bad).max(1) as f64;
        out.push(ExprScore {
            mse: se / good,
            mae: ae / good,
            max_err: mx,
            n_nonfinite: bad,
        });
    }
    out
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcodes_cover_the_math_alphabet() {
        // Every Math constructor the evaluator can meet must have an opcode,
        // or an expression silently evaluates to NaN on device.
        for name in [
            "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Sqrt", "Log", "Exp", "Sin", "Cos",
            "Tan", "Tanh", "Pow", "Pow2", "Pow3", "Inv", "Num", "Var", "ProtectedDiv",
            "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "ProtectedInv", "Asin", "Acos",
            "ProtectedAsin", "ProtectedAcos",
        ] {
            assert!(Op::from_math(name).is_some(), "no opcode for {name}");
        }
    }

    #[test]
    fn arities_match_the_math_datatype() {
        assert_eq!(Op::Var.arity(), 0);
        assert_eq!(Op::Num.arity(), 0);
        assert_eq!(Op::Neg.arity(), 1);
        assert_eq!(Op::Sqrt.arity(), 1);
        assert_eq!(Op::ProtectedExp.arity(), 1);
        assert_eq!(Op::Asin.arity(), 1);
        assert_eq!(Op::Acos.arity(), 1);
        assert_eq!(Op::ProtectedAsin.arity(), 1);
        assert_eq!(Op::ProtectedAcos.arity(), 1);
        assert_eq!(Op::Add.arity(), 2);
        assert_eq!(Op::Pow.arity(), 2);
    }

    #[test]
    fn batch_tracks_offsets_and_flags_oversized() {
        let leaf = GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 };
        let mut b = ExprBatch::new();
        assert_eq!(b.push(&[leaf]), 0);
        assert_eq!(b.push(&[leaf, leaf, leaf]), 1);
        assert_eq!(b.offsets, vec![0, 1]);
        assert_eq!(b.lengths, vec![1, 3]);
        assert_eq!(b.len(), 2);
        assert!(b.oversized().is_empty());

        let too_long = vec![leaf; MAX_NODES + 1];
        b.push(&too_long);
        assert_eq!(b.oversized(), vec![2]);
    }

    #[test]
    fn node_is_sixteen_bytes() {
        // The size the buffer budget in the module docs assumes.
        assert_eq!(std::mem::size_of::<GpuNode>(), 16);
    }

    #[test]
    fn wgsl_switch_covers_every_opcode() {
        // A missing `case Nu:` is a silent NaN on device, so check the shader
        // source mentions each discriminant.
        for op in 0u32..=23 {
            assert!(
                EVAL_WGSL.contains(&format!("case {op}u:")),
                "WGSL has no case for opcode {op}"
            );
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod device_tests {
    use super::*;

    /// The gate on this whole module: device results must match the CPU
    /// evaluator. f32 vs f64 means we compare at f32 tolerance, not bitwise —
    /// which is exactly why the docs say to re-score finalists on CPU.
    #[test]
    fn matches_cpu_on_a_simple_expression() {
        // data: 4 rows, 2 vars (a, b), row-major
        let rows: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                return;
            }
        };

        // Mul(Var a, Var b) in level order: node0 = Mul(children 1,2)
        let nodes = [
            GpuNode { op: Op::Mul as u32, arg0: 1, arg1: 2, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 1, arg1: 0, konst: 0.0 },
        ];
        let mut batch = ExprBatch::new();
        batch.push(&nodes);

        let got = ev.eval(&batch).expect("eval");
        assert_eq!(got.len(), 4, "one value per row");
        for (i, chunk) in rows.chunks(2).enumerate() {
            let want = chunk[0] * chunk[1];
            assert!(
                (got[i] - want).abs() <= 1e-5 * want.abs().max(1.0),
                "row {i}: got {} want {want}",
                got[i]
            );
        }
    }

    #[test]
    fn partials_emit_every_subtree_value() {
        // 4 rows, 2 vars (a, b). Mul(a, b): node0 = a*b, node1 = a, node2 = b.
        let rows: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                return;
            }
        };
        let nodes = [
            GpuNode { op: Op::Mul as u32, arg0: 1, arg1: 2, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 1, arg1: 0, konst: 0.0 },
        ];
        let mut batch = ExprBatch::new();
        batch.push(&nodes);

        let (roots, partials) = ev.eval_with_partials(&batch).expect("eval_with_partials");
        let n_rows = 4usize;
        assert_eq!(roots.len(), n_rows, "one root per row");
        assert_eq!(partials.len(), MAX_NODES * n_rows, "one expr x MAX_NODES x rows");
        let at = |k: usize, row: usize| partials[(k) * n_rows + row]; // expr 0
        for (row, chunk) in rows.chunks(2).enumerate() {
            let (a, b) = (chunk[0], chunk[1]);
            // node0 = a*b (also the root), node1 = a, node2 = b — each subtree's value.
            assert!((at(0, row) - a * b).abs() <= 1e-5 * (a * b).abs().max(1.0), "subtree0 (a*b) row {row}: {}", at(0, row));
            assert!((at(1, row) - a).abs() <= 1e-5 * a.abs().max(1.0), "subtree1 (a) row {row}: {}", at(1, row));
            assert!((at(2, row) - b).abs() <= 1e-5 * b.abs().max(1.0), "subtree2 (b) row {row}: {}", at(2, row));
            // the root partial equals the ordinary eval root.
            assert!((at(0, row) - roots[row]).abs() <= 1e-6, "root partial == eval root");
        }
    }

    #[test]
    fn eval_is_unchanged_by_the_partials_field() {
        // The ordinary eval (emit_partials 0) must be byte-identical to before:
        // compare against a hand value on a mixed expression.
        let rows: Vec<f32> = vec![2.0, 3.0, 5.0, 7.0];
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(_) => return,
        };
        // Add(Mul(a,b), a): node0 Add(1,2), node1 Mul(3,4), node2 Var a, node3 Var a, node4 Var b
        let nodes = [
            GpuNode { op: Op::Add as u32, arg0: 1, arg1: 2, konst: 0.0 },
            GpuNode { op: Op::Mul as u32, arg0: 3, arg1: 4, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 1, arg1: 0, konst: 0.0 },
        ];
        let mut batch = ExprBatch::new();
        batch.push(&nodes);
        let got = ev.eval(&batch).expect("eval");
        for (row, chunk) in rows.chunks(2).enumerate() {
            let (a, b) = (chunk[0], chunk[1]);
            let want = a * b + a;
            assert!((got[row] - want).abs() <= 1e-5 * want.abs().max(1.0), "row {row}: got {} want {want}", got[row]);
        }
    }

    #[test]
    fn oversized_expressions_come_back_nan_not_truncated() {
        let rows: Vec<f32> = vec![1.0, 2.0];
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(_) => return,
        };
        let leaf = GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 };
        let mut batch = ExprBatch::new();
        batch.push(&vec![leaf; MAX_NODES + 1]);
        let got = ev.eval(&batch).expect("eval");
        assert!(got[0].is_nan(), "oversized must be NaN, got {}", got[0]);
    }

    #[test]
    fn many_expressions_one_dispatch() {
        // The point of the design: a population in a single pass.
        let rows: Vec<f32> = (0..200).map(|i| 1.0 + (i % 7) as f32).collect();
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(_) => return,
        };
        let mut batch = ExprBatch::new();
        for k in 0..500u32 {
            batch.push(&[
                GpuNode { op: Op::Add as u32, arg0: 1, arg1: 2, konst: 0.0 },
                GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: k as f32 },
            ]);
        }
        let got = ev.eval(&batch).expect("eval");
        assert_eq!(got.len(), 500 * 100, "500 expressions x 100 rows");
        // expression k, row 0 = data[0] + k
        for k in 0..500usize {
            let want = rows[0] + k as f32;
            let g = got[k * 100];
            assert!((g - want).abs() <= 1e-4 * want.abs().max(1.0), "expr {k}: {g} vs {want}");
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod subtree_stats_tests {
    use super::*;

    fn var(i: u32) -> GpuNode {
        GpuNode { op: Op::Var as u32, arg0: i, arg1: 0, konst: 0.0 }
    }
    fn num(v: f32) -> GpuNode {
        GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: v }
    }
    fn bin(op: Op, a: u32, b: u32) -> GpuNode {
        GpuNode { op: op as u32, arg0: a, arg1: b, konst: 0.0 }
    }
    fn un(op: Op, a: u32) -> GpuNode {
        GpuNode { op: op as u32, arg0: a, arg1: 0, konst: 0.0 }
    }

    /// EVERY SUBTREE'S MIN, MAX AND SUM, on the device, equal the host's; a
    /// subtree that does not compute is not finite; a poisoned expression
    /// (an infinite trig argument) is not finite in every node, its finite
    /// leaves included, exactly as `eval_main` scores the row; entries past
    /// an expression's length are not finite; and a row range reduces only
    /// those rows.
    #[test]
    fn subtree_statistics_match_the_host_and_carry_the_kernels_poison() {
        // 4 rows, 2 vars: a = 1,3,5,7  b = 2,4,6,8
        let rows: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                return;
            }
        };
        let stats = SubtreeStats::new(&ev).expect("stats");
        let mut batch = ExprBatch::new();
        batch.push(&[bin(Op::Mul, 1, 2), var(0), var(1)]); // a*b
        batch.push(&[un(Op::Log, 1), bin(Op::Sub, 2, 3), var(0), var(1)]); // log(a-b): a-b < 0
        batch.push(&[un(Op::Sin, 1), un(Op::Exp, 2), num(1000.0)]); // sin(exp(1000)) = sin(inf): poison
        let s = stats.run(&ev, &batch, 0, 4).expect("run");
        assert_eq!(s.len(), 3 * MAX_NODES);
        let at = |e: usize, k: usize| s[e * MAX_NODES + k];
        // a*b on every row: 2, 12, 30, 56.
        assert!(at(0, 0).finite);
        assert_eq!((at(0, 0).min, at(0, 0).max, at(0, 0).sum), (2.0, 56.0, 100.0));
        assert_eq!((at(0, 1).min, at(0, 1).max, at(0, 1).sum), (1.0, 7.0, 16.0), "leaf a");
        assert_eq!((at(0, 2).min, at(0, 2).max, at(0, 2).sum), (2.0, 8.0, 20.0), "leaf b");
        assert!((at(0, 0).mean(4) - 25.0).abs() < 1e-9);
        assert!(!at(0, 3).finite, "past the length is not finite");
        // log of a negative: the root does not compute, its child does.
        assert!(!at(1, 0).finite);
        assert!(at(1, 1).finite);
        assert_eq!((at(1, 1).min, at(1, 1).max), (-1.0, -1.0));
        // Poison: every node of the expression, the finite leaf 1000 included.
        assert!(!at(2, 0).finite);
        assert!(!at(2, 1).finite);
        assert!(!at(2, 2).finite, "a poisoned row is NaN in every partial, so the leaf reads not finite");
        // Rows 1..3 only: a*b = 12, 30.
        let s = stats.run(&ev, &batch, 1, 3).expect("run");
        assert_eq!((s[0].min, s[0].max, s[0].sum), (12.0, 30.0, 42.0));
        // An empty range or one past the data is refused.
        assert!(stats.run(&ev, &batch, 2, 2).is_err());
        assert!(stats.run(&ev, &batch, 0, 5).is_err());
    }

    /// A batch larger than one block runs in blocks and comes back whole, in order.
    #[test]
    fn a_batch_past_the_block_runs_in_blocks() {
        let rows: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let ev = match GpuEvaluator::new(&rows, 1) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                return;
            }
        };
        let stats = SubtreeStats::new(&ev).expect("stats");
        let n = stats.block() + 3;
        let mut batch = ExprBatch::new();
        for i in 0..n {
            batch.push(&[bin(Op::Add, 1, 2), var(0), num(i as f32)]); // a + i
        }
        let s = stats.run(&ev, &batch, 0, 4).expect("run");
        assert_eq!(s.len(), n * MAX_NODES);
        for i in 0..n {
            let r = s[i * MAX_NODES];
            assert!(r.finite, "expr {i}");
            assert_eq!((r.min, r.max), (1.0 + i as f32, 4.0 + i as f32), "expr {i}");
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod protected_parity_tests {
    use super::*;

    /// The CPU contract, transcribed from `crate::eval`. Spelled out rather
    /// than invoked so the test pins the SEMANTICS: if someone changes eval.rs,
    /// this fails and forces a deliberate decision, instead of both
    /// implementations drifting together and the parity test still passing.
    fn cpu(op: Op, a: f64, b: f64) -> f64 {
        match op {
            // The ENGINE's primitives, from hff_sr_engine.py:
            //   protected_div_zero: 0 if |b| < 1e-6 else a/b
            //   protected_sqrt:     sqrt(|x|) if isfinite(x) else 0.0
            //   protected_log:      +inf if not isfinite(x) or x == 0 else ln|x|
            //   protected_exp:      +inf if not isfinite(x) else exp(x)
            Op::ProtectedDiv => if b.abs() < 1e-6 { 0.0 } else { a / b },
            Op::ProtectedInv => if a == 0.0 { 1.0 } else { 1.0 / a },
            Op::ProtectedSqrt => if a.is_finite() { a.abs().sqrt() } else { 0.0 },
            Op::ProtectedLog => {
                if !a.is_finite() || a == 0.0 { f64::INFINITY } else { a.abs().ln() }
            }
            Op::ProtectedExp => if a.is_finite() { a.exp() } else { f64::INFINITY },
            // fuller's own (no engine counterpart), on protected_sqrt's
            // non-finite convention:
            //   protected_asin:     asin(clamp(x, -1, 1)) if isfinite(x) else 0.0
            //   protected_acos:     acos(clamp(x, -1, 1)) if isfinite(x) else 0.0
            Op::ProtectedAsin => if a.is_finite() { a.clamp(-1.0, 1.0).asin() } else { 0.0 },
            Op::ProtectedAcos => if a.is_finite() { a.clamp(-1.0, 1.0).acos() } else { 0.0 },
            // raw, real-domain: NaN outside [-1, 1]
            Op::Asin => if a.abs() <= 1.0 { a.asin() } else { f64::NAN },
            Op::Acos => if a.abs() <= 1.0 { a.acos() } else { f64::NAN },
            Op::Div => a / b,
            Op::Sqrt => a.sqrt(),
            other => panic!("no CPU reference for {other:?}"),
        }
    }

    /// THE DEVICE'S TANH SATURATES LIKE THE ENGINE'S. Measured before the fix
    /// on Apple silicon: tanh(44) = 0, tanh(45) = NaN, because Metal's tanh
    /// overflows exp(2x) inside. The engine (Python's math.tanh) is +-1 there,
    /// and a fold judging `Tanh(x + 40)` flat needs 1 on every row.
    #[test]
    fn tanh_saturates_to_one_past_twenty_instead_of_overflowing() {
        let xs = [0.5f32, 10.0, 19.9, 20.5, 44.0, 45.0, 100.0, 1e30, -44.0, -100.0, -1e30];
        let Some(v) = gpu_unary(Op::Tanh, &xs) else {
            eprintln!("no GPU available; skipping");
            return;
        };
        for (x, got) in xs.iter().zip(v.iter()) {
            let want = f64::from(*x).tanh() as f32;
            assert!(got.is_finite(), "tanh({x}) = {got}");
            assert!((got - want).abs() <= 1e-6, "tanh({x}) = {got}, want {want}");
        }
        // Non-finite in: infinity saturates as the engine's math.tanh(inf) = 1.0; NaN stays NaN.
        let v = gpu_unary(Op::Tanh, &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]).expect("gpu");
        assert_eq!((v[0], v[1]), (1.0, -1.0));
        assert!(v[2].is_nan());
    }

    /// Run a one-arg op on the GPU over `xs`, one row per value.
    fn gpu_unary(op: Op, xs: &[f32]) -> Option<Vec<f32>> {
        let ev = GpuEvaluator::new(xs, 1).ok()?;
        let mut b = ExprBatch::new();
        b.push(&[
            GpuNode { op: op as u32, arg0: 1, arg1: 0, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
        ]);
        ev.eval(&b).ok()
    }

    /// Run a two-arg op with `a` fixed and `b` varying by row.
    fn gpu_binary(op: Op, a: f32, bs: &[f32]) -> Option<Vec<f32>> {
        let ev = GpuEvaluator::new(bs, 1).ok()?;
        let mut batch = ExprBatch::new();
        batch.push(&[
            GpuNode { op: op as u32, arg0: 1, arg1: 2, konst: 0.0 },
            GpuNode { op: Op::Num as u32, arg0: 0, arg1: 0, konst: a },
            GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
        ]);
        ev.eval(&batch).ok()
    }

    fn agree(got: f32, want: f64, ctx: &str) {
        let w = want as f32;
        if w.is_nan() {
            assert!(got.is_nan(), "{ctx}: cpu NaN, gpu {got}");
        } else if w.is_infinite() {
            assert!(
                got.is_infinite() && got.signum() == w.signum(),
                "{ctx}: cpu {w}, gpu {got}"
            );
        } else {
            assert!(
                (got - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{ctx}: cpu {w}, gpu {got}"
            );
        }
    }

    /// ProtectedDiv(a, b) is 0.0 whenever |b| < 1e-6 — the engine's
    /// protected_div_zero — NOT only at b == 0, and NOT 1.0. With a `b == 0`
    /// guard, b = -1e-7 (a real value: sin(omega * -2.3e-8)) gave c / -1e-7
    /// here and 0 on the engine, and the two scored different functions.
    #[test]
    fn protected_div_by_zero_is_zero() {
        let bs: [f32; 8] = [0.0, 2.0, -4.0, 1e-30, -0.0, -1e-7, 9e-7, 2e-6];
        let Some(got) = gpu_binary(Op::ProtectedDiv, 5.0, &bs) else { return };
        for (i, &b) in bs.iter().enumerate() {
            let want = cpu(Op::ProtectedDiv, 5.0, b as f64);
            agree(got[i], want, &format!("ProtectedDiv(5, {b})"));
        }
        assert_eq!(got[0], 0.0, "5/0 must be 0.0");
        assert_eq!(got[4], 0.0, "5/-0.0 must be 0.0");
        assert_eq!(got[5], 0.0, "5/-1e-7 is under the threshold: 0.0");
        assert_eq!(got[6], 0.0, "5/9e-7 is under the threshold: 0.0");
        assert!(got[7] > 1e6, "5/2e-6 is over the threshold: a real quotient");
    }

    /// ProtectedLog(0) is +inf — the engine returns float("inf") at zero, not
    /// ln|0| = -inf. The sign matters: protected_exp(-inf) would be read as a
    /// finite 0 downstream, where the engine has +inf and rejects.
    #[test]
    fn protected_log_of_zero_is_pos_inf() {
        let xs: [f32; 5] = [0.0, 1.0, -4.0, std::f32::consts::E, -1.0];
        let Some(got) = gpu_unary(Op::ProtectedLog, &xs) else { return };
        for (i, &x) in xs.iter().enumerate() {
            let want = cpu(Op::ProtectedLog, x as f64, 0.0);
            agree(got[i], want, &format!("ProtectedLog({x})"));
        }
        assert!(got[0].is_infinite() && got[0] > 0.0, "protected_log(0) must be +inf");
    }

    #[test]
    fn protected_inv_of_zero_is_one() {
        let xs: [f32; 4] = [0.0, 2.0, -0.5, -0.0];
        let Some(got) = gpu_unary(Op::ProtectedInv, &xs) else { return };
        for (i, &x) in xs.iter().enumerate() {
            let want = cpu(Op::ProtectedInv, x as f64, 0.0);
            agree(got[i], want, &format!("ProtectedInv({x})"));
        }
        assert_eq!(got[0], 1.0, "1/0 must be 1.0 for ProtectedInv");
    }

    #[test]
    fn protected_sqrt_takes_abs_not_nan() {
        let xs: [f32; 4] = [-4.0, 4.0, 0.0, -1e-8];
        let Some(got) = gpu_unary(Op::ProtectedSqrt, &xs) else { return };
        for (i, &x) in xs.iter().enumerate() {
            let want = cpu(Op::ProtectedSqrt, x as f64, 0.0);
            agree(got[i], want, &format!("ProtectedSqrt({x})"));
        }
        assert_eq!(got[0], 2.0, "sqrt|-4| must be 2, not NaN");
    }

    /// ProtectedExp is UNCAPPED: +inf on overflow, and exp(-inf) = 0. Not
    /// exp(min(x, 700)), which would return a large finite value where the
    /// engine returns inf.
    #[test]
    fn protected_exp_is_uncapped() {
        // f32 overflows around 88, well before f64's ~709, so compare against
        // the CPU only where f32 can represent the answer; check the overflow
        // behaviour separately.
        let xs: [f32; 3] = [1.0, 0.0, -50.0];
        let Some(got) = gpu_unary(Op::ProtectedExp, &xs) else { return };
        for (i, &x) in xs.iter().enumerate() {
            let want = cpu(Op::ProtectedExp, x as f64, 0.0);
            agree(got[i], want, &format!("ProtectedExp({x})"));
        }
        // Overflow must go to +inf, not to a large finite value.
        let Some(big) = gpu_unary(Op::ProtectedExp, &[1000.0f32]) else { return };
        assert!(
            big[0].is_infinite() && big[0] > 0.0,
            "ProtectedExp(1000) must be +inf, got {}",
            big[0]
        );
    }

    /// What each protected op does with a NON-FINITE input, which is where the
    /// device and the engine disagreed: sqrt -> 0.0, log -> +inf, exp -> +inf
    /// (including exp(-inf), which IEEE makes 0). Inputs are built on the
    /// device — exp(1000) = +inf, -exp(1000) = -inf — because a non-finite
    /// value cannot be uploaded as data and compared.
    #[test]
    fn protected_ops_on_non_finite_input_match_the_engine() {
        let Ok(ev) = GpuEvaluator::new(&[1000.0f32], 1) else { return };
        let inf = |neg: bool| -> Vec<GpuNode> {
            // [op, (Neg,) Exp, Var]
            let mut v = vec![GpuNode { op: 0, arg0: 0, arg1: 0, konst: 0.0 }];
            if neg {
                v.push(GpuNode { op: Op::Neg as u32, arg0: 2, arg1: 0, konst: 0.0 });
                v.push(GpuNode { op: Op::Exp as u32, arg0: 3, arg1: 0, konst: 0.0 });
            } else {
                v.push(GpuNode { op: Op::Exp as u32, arg0: 2, arg1: 0, konst: 0.0 });
            }
            v.push(GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 });
            v
        };
        let mut b = ExprBatch::new();
        for op in [Op::ProtectedSqrt, Op::ProtectedLog, Op::ProtectedExp] {
            for neg in [false, true] {
                let mut nodes = inf(neg);
                nodes[0] = GpuNode { op: op as u32, arg0: 1, arg1: 0, konst: 0.0 };
                b.push(&nodes);
            }
        }
        let got = ev.eval(&b).expect("eval");
        assert_eq!(&got[0..2], &[0.0, 0.0], "protected_sqrt(+-inf) = 0.0");
        for (i, name) in [(2, "log(+inf)"), (3, "log(-inf)"), (4, "exp(+inf)"), (5, "exp(-inf)")] {
            assert!(got[i].is_infinite() && got[i] > 0.0, "protected_{name} must be +inf, got {}", got[i]);
        }
    }

    /// Asin / Acos are NaN outside [-1, 1]; the protected forms clamp there and
    /// are the raw function inside. One column spanning [-2, 2], the domain's
    /// two ends included.
    #[test]
    fn inverse_trig_matches_the_cpu_inside_and_outside_the_domain() {
        let xs: Vec<f32> = (0..=40).map(|i| -2.0 + 0.1 * i as f32).chain([-1.0, 1.0, 0.5, -0.0]).collect();
        for op in [Op::Asin, Op::Acos, Op::ProtectedAsin, Op::ProtectedAcos] {
            let Some(got) = gpu_unary(op, &xs) else { return };
            for (i, &x) in xs.iter().enumerate() {
                agree(got[i], cpu(op, x as f64, 0.0), &format!("{op:?}({x})"));
            }
        }
        let Some(clamped) = gpu_unary(Op::ProtectedAcos, &[-3.0, 2.0]) else { return };
        agree(clamped[0], std::f64::consts::PI, "ProtectedAcos(-3)");
        assert_eq!(clamped[1], 0.0, "ProtectedAcos(2) = acos(1) = 0");
    }

    /// A non-finite argument: 0.0 from the protected forms (ProtectedSqrt's
    /// convention, +inf and -inf alike), NaN from the raw ones. Built on the
    /// device, as above.
    #[test]
    fn inverse_trig_on_non_finite_input() {
        let Ok(ev) = GpuEvaluator::new(&[1000.0f32], 1) else { return };
        let mut b = ExprBatch::new();
        let ops = [Op::ProtectedAsin, Op::ProtectedAcos, Op::Asin, Op::Acos];
        for op in ops {
            // op(Exp(x)) = op(+inf), op(Neg(Exp(x))) = op(-inf)
            b.push(&[
                GpuNode { op: op as u32, arg0: 1, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Exp as u32, arg0: 2, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            ]);
            b.push(&[
                GpuNode { op: op as u32, arg0: 1, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Neg as u32, arg0: 2, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Exp as u32, arg0: 3, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            ]);
        }
        let got = ev.eval(&b).expect("eval");
        assert_eq!(&got[0..4], &[0.0, 0.0, 0.0, 0.0], "protected asin/acos of +-inf = 0.0");
        assert!(got[4..8].iter().all(|v| v.is_nan()), "raw asin/acos of +-inf is NaN, got {:?}", &got[4..8]);
    }

    /// The engine RAISES on sin/cos of +-inf (Python math.sin -> ValueError)
    /// and rejects the individual. A NaN alone cannot say that, because
    /// ProtectedSqrt maps non-finite to 0.0 and would swallow it: the
    /// expression is poisoned and comes back NaN whatever wraps the trig.
    /// Metal returns a FINITE value for cos(inf), so without this an
    /// individual the engine rejects was scored.
    #[test]
    fn trig_of_infinity_poisons_the_expression() {
        let Ok(ev) = GpuEvaluator::new(&[1000.0f32, 1.0f32], 1) else { return };
        let mut b = ExprBatch::new();
        for trig in [Op::Sin, Op::Cos] {
            // ProtectedSqrt(trig(Exp(x)))
            b.push(&[
                GpuNode { op: Op::ProtectedSqrt as u32, arg0: 1, arg1: 0, konst: 0.0 },
                GpuNode { op: trig as u32, arg0: 2, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Exp as u32, arg0: 3, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            ]);
        }
        let got = ev.eval(&b).expect("eval");
        for e in 0..2 {
            assert!(got[e * 2].is_nan(), "row x=1000: trig(inf) must poison, got {}", got[e * 2]);
            assert!(got[e * 2 + 1].is_finite(), "row x=1: finite input must evaluate");
        }
    }

    /// Metal's sin/cos return exactly 0.0 from |x| = 1e7 — where f32 has lost
    /// the phase — and are accurate below 3e6. A silent 0.0 is a wrong number
    /// that looks valid (on the power plant data, cos(AP**3) with AP ~ 1010
    /// came back as the constant 0 while the CPU scored a real function), so
    /// beyond 1e7 the expression is REFUSED: NaN, poisoned, not swallowed.
    #[test]
    fn trig_beyond_f32_phase_resolution_is_refused() {
        let xs: [f32; 4] = [1000.5, 3.0e6, 1.0e7, 1.0e9];
        let Ok(ev) = GpuEvaluator::new(&xs, 1) else { return };
        let mut b = ExprBatch::new();
        for trig in [Op::Sin, Op::Cos] {
            b.push(&[
                GpuNode { op: Op::ProtectedSqrt as u32, arg0: 1, arg1: 0, konst: 0.0 },
                GpuNode { op: trig as u32, arg0: 2, arg1: 0, konst: 0.0 },
                GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            ]);
        }
        let got = ev.eval(&b).expect("eval");
        for e in 0..2 {
            let r = &got[e * 4..e * 4 + 4];
            assert!(r[0].is_finite() && r[1].is_finite(), "in range must evaluate: {r:?}");
            assert!(r[2].is_nan() && r[3].is_nan(), "beyond 1e7 must be refused: {r:?}");
        }
        // and it IS accurate where it answers
        let want = (1000.5f32 as f64).sin().abs().sqrt() as f32;
        assert!((got[0] - want).abs() < 1e-5, "{} vs {want}", got[0]);
    }

    /// Raw ops must NOT behave like their protected namesakes: raw div by zero
    /// is inf/NaN, raw sqrt of a negative is NaN. Mapping a protected op to a
    /// raw one (or vice versa) is unsound on exactly these inputs.
    #[test]
    fn raw_ops_stay_raw() {
        let Some(div) = gpu_binary(Op::Div, 5.0, &[0.0f32]) else { return };
        assert!(
            div[0].is_infinite() || div[0].is_nan(),
            "raw 5/0 must not be the protected 0.0, got {}",
            div[0]
        );
        let Some(sq) = gpu_unary(Op::Sqrt, &[-4.0f32]) else { return };
        assert!(sq[0].is_nan(), "raw sqrt(-4) must be NaN, got {}", sq[0]);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod real_expression_tests {
    use super::*;

    /// CPU reference: evaluate a Math s-expr at one row, in f64, via the
    /// crate's own evaluator. This is the path the engine trusts.
    fn cpu_eval(expr: &str, vars: &[String], row: &[f32]) -> Option<f64> {
        use crate::eval::eval_term;
        use crate::expr::MATH_DATATYPE;
        use egglog::prelude::exprs;
        use egglog::EGraph;
        let mut eg = EGraph::default();
        eg.parse_and_run_program(None, MATH_DATATYPE).ok()?;
        eg.parse_and_run_program(None, &format!("(let __g {expr})")).ok()?;
        let (sort, value) = eg.eval_expr(&exprs::var("__g")).ok()?;
        let (termdag, term, _) = eg.extract_value(&sort, value).ok()?;
        eval_term(&termdag, term, &|name: &str| {
            vars.iter().position(|v| v == name).map(|i| row[i] as f64)
        })
        .ok()
    }

    /// The integration test: REAL Math expressions, parsed by the converter,
    /// evaluated on the GPU, compared against the CPU f64 evaluator row by
    /// row. Nothing here is a hand-built node array.
    #[test]
    fn gpu_matches_cpu_on_real_math_expressions() {
        let vars: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        // 8 rows x 3 vars, deliberately including a zero and a negative so the
        // protected ops are actually exercised on real data.
        let rows: Vec<f32> = vec![
            1.0, 2.0, 3.0, 2.0, 4.0, 0.5, 0.5, -1.0, 2.0, 3.0, 0.0, 1.5, -2.0, 1.0,
            4.0, 0.25, 8.0, -0.5, 5.0, 0.1, 2.5, 1.0, 1.0, 1.0,
        ];

        let exprs = [
            r#"(Mul (Var "a") (Var "b"))"#,
            r#"(Add (Mul (Var "a") (Var "b")) (Var "c"))"#,
            r#"(Div (Var "a") (Var "b"))"#,
            r#"(Sub (Pow2 (Var "a")) (Var "c"))"#,
            r#"(ProtectedDiv (Var "a") (Var "c"))"#,
            r#"(ProtectedSqrt (Var "b"))"#,
            r#"(ProtectedLog (Var "a"))"#,
            r#"(ProtectedInv (Var "c"))"#,
            r#"(Neg (Add (Var "a") (Var "b")))"#,
            r#"(Abs (Sub (Var "a") (Var "b")))"#,
            r#"(Mul (Num 3.5) (Var "a"))"#,
            r#"(Sin (Var "a"))"#,
            r#"(Tanh (Var "b"))"#,
            r#"(Pow3 (Var "c"))"#,
            r#"(Div (Mul (Num 2.0) (Var "a")) (Add (Var "b") (Num 1.0)))"#,
        ];

        let ev = match GpuEvaluator::new(&rows, 3) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("no GPU adapter ({e}); skipping");
                return;
            }
        };

        // ONE batch, ONE dispatch, every expression.
        let mut batch = ExprBatch::new();
        for e in &exprs {
            let nodes = math_to_nodes(e, &vars).unwrap_or_else(|err| panic!("{e}: {err}"));
            batch.push(&nodes);
        }
        let got = ev.eval(&batch).expect("gpu eval");
        assert_eq!(got.len(), exprs.len() * 8);

        let mut checked = 0;
        for (ei, e) in exprs.iter().enumerate() {
            for r in 0..8usize {
                let row = &rows[r * 3..r * 3 + 3];
                let want = cpu_eval(e, &vars, row)
                    .unwrap_or_else(|| panic!("cpu eval failed for {e}"));
                let g = got[ei * 8 + r];
                let w = want as f32;
                if w.is_nan() {
                    assert!(g.is_nan(), "{e} row {r}: cpu NaN, gpu {g}");
                } else if w.is_infinite() {
                    assert!(
                        g.is_infinite() && g.signum() == w.signum(),
                        "{e} row {r}: cpu {w}, gpu {g}"
                    );
                } else {
                    // f32 device vs f64 host: compare at f32 tolerance.
                    assert!(
                        (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                        "{e} row {r}: cpu {w}, gpu {g}"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, exprs.len() * 8);
        eprintln!("{checked} (expression, row) pairs matched CPU f64");
    }

    /// tan and the inverse-trig ops, raw and protected, as REAL Math expressions
    /// through the converter, one batch, over a column spanning [-2, 2] — the
    /// shapes of the laws that need them (feynman I.26.2 is asin(n sin t)).
    #[test]
    fn gpu_matches_cpu_on_tan_and_inverse_trig() {
        let vars: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let n_rows = 41usize;
        let rows: Vec<f32> = (0..n_rows).flat_map(|i| [-2.0 + 0.1 * i as f32, 0.3 + 0.02 * i as f32]).collect();
        let exprs = [
            r#"(Asin (Var "a"))"#,
            r#"(Acos (Var "a"))"#,
            r#"(ProtectedAsin (Var "a"))"#,
            r#"(ProtectedAcos (Var "a"))"#,
            r#"(Tan (Var "a"))"#,
            r#"(ProtectedAsin (Mul (Var "b") (Sin (Var "a"))))"#,
            r#"(Asin (Mul (Var "b") (Sin (Var "a"))))"#,
            r#"(ProtectedAcos (Div (Var "a") (Add (Var "b") (Num 1.0))))"#,
            r#"(Mul (Cos (Var "a")) (Inv (Tan (Var "b"))))"#,
            r#"(Sin (Asin (Var "a")))"#,
            r#"(Cos (Acos (Var "a")))"#,
        ];
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("no GPU adapter ({e}); skipping");
                return;
            }
        };
        let mut batch = ExprBatch::new();
        for e in &exprs {
            let nodes = math_to_nodes(e, &vars).unwrap_or_else(|err| panic!("{e}: {err}"));
            batch.push(&nodes);
        }
        let got = ev.eval(&batch).expect("gpu eval");
        assert_eq!(got.len(), exprs.len() * n_rows);
        let (mut finite, mut nan) = (0, 0);
        for (ei, e) in exprs.iter().enumerate() {
            for r in 0..n_rows {
                let row = &rows[r * 2..r * 2 + 2];
                let want = cpu_eval(e, &vars, row).unwrap_or_else(|| panic!("cpu eval failed for {e}"));
                let (g, w) = (got[ei * n_rows + r], want as f32);
                if w.is_nan() {
                    assert!(g.is_nan(), "{e} row {r} ({row:?}): cpu NaN, gpu {g}");
                    nan += 1;
                } else {
                    // f32 device vs f64 host: compare at f32 tolerance.
                    assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{e} row {r} ({row:?}): cpu {w}, gpu {g}");
                    finite += 1;
                }
            }
        }
        // Both sides of the domain edge were really met.
        assert!(finite > 0 && nan > 0, "{finite} finite, {nan} NaN");
    }

    #[test]
    fn converter_emits_breadth_first_so_children_follow_parents() {
        // The kernel's backward scan REQUIRES child index > parent index.
        let vars: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let nodes = math_to_nodes(
            r#"(Add (Mul (Var "a") (Var "b")) (Sub (Var "a") (Var "b")))"#,
            &vars,
        )
        .expect("convert");
        for (i, n) in nodes.iter().enumerate() {
            let op_is_leaf = n.op == Op::Var as u32 || n.op == Op::Num as u32;
            if op_is_leaf {
                continue;
            }
            assert!(n.arg0 as usize > i, "node {i}: child arg0={} not after it", n.arg0);
            let arity = match n.op {
                x if x == Op::Add as u32 || x == Op::Sub as u32 || x == Op::Mul as u32 => 2,
                _ => 1,
            };
            if arity == 2 {
                assert!(n.arg1 as usize > i, "node {i}: child arg1={} not after it", n.arg1);
            }
        }
    }

    #[test]
    fn unknown_variable_is_an_error_not_column_zero() {
        let vars: Vec<String> = vec!["a".to_string()];
        let r = math_to_nodes(r#"(Var "zzz")"#, &vars);
        assert!(r.is_err(), "unknown variable must fail loudly, got {r:?}");
    }

    #[test]
    fn unknown_constructor_is_an_error() {
        let vars: Vec<String> = vec!["a".to_string()];
        assert!(math_to_nodes(r#"(Bogus (Var "a"))"#, &vars).is_err());
    }
}

#[cfg(test)]
mod opcode_source_of_truth_tests {
    use super::*;

    /// Every opcode must name a constructor the CPU evaluator actually
    /// implements. An earlier version of this table carried `DiffSq`, which
    /// does not exist in Math at all — it was invented from a karva token
    /// name, and nothing caught it until a real expression was evaluated.
    ///
    /// Reading eval.rs at test time keeps the two in step: add an op there and
    /// forget it here (or invent one here) and this fails.
    #[test]
    fn every_opcode_exists_in_the_cpu_evaluator() {
        let eval_src = include_str!("eval.rs");
        for name in [
            "Var", "Num", "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Sqrt", "Log",
            "Exp", "Sin", "Cos", "Tan", "Tanh", "Pow", "Pow2", "Pow3", "Inv",
            "ProtectedDiv", "ProtectedSqrt", "ProtectedLog", "ProtectedExp",
            "ProtectedInv", "Asin", "Acos", "ProtectedAsin", "ProtectedAcos",
        ] {
            assert!(
                Op::from_math(name).is_some(),
                "{name} is in eval.rs but has no opcode"
            );
            // Var/Num are leaves handled by the parser, not by a match arm of
            // `apply_op`, which dispatches on the opcode itself.
            if name != "Var" && name != "Num" {
                assert!(
                    eval_src.contains(&format!("(Op::{name}, ")),
                    "opcode {name} has no match arm in eval.rs — is it invented?"
                );
            }
        }
    }

    /// The reverse direction: nothing in the opcode table may be absent from
    /// eval.rs.
    #[test]
    fn no_opcode_is_invented() {
        let eval_src = include_str!("eval.rs");
        for name in [
            "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Sqrt", "Log", "Exp", "Sin",
            "Cos", "Tan", "Tanh", "Pow", "Pow2", "Pow3", "Inv", "ProtectedDiv",
            "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "ProtectedInv", "Asin", "Acos",
            "ProtectedAsin", "ProtectedAcos",
        ] {
            assert!(
                eval_src.contains(&format!("(Op::{name}, ")),
                "opcode {name} does not exist in eval.rs"
            );
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod undecodable_batch_tests {
    use super::*;

    /// A batch where every expression failed to convert must not panic.
    ///
    /// wgpu refuses a zero-sized binding, so an all-empty batch used to abort
    /// inside create_bind_group. That is reachable in a real run: a generation
    /// whose genes all carry unresolved RNC placeholders converts to nothing.
    #[test]
    fn all_undecodable_batch_returns_nan_not_a_panic() {
        let rows: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let Ok(ev) = GpuEvaluator::new(&rows, 2) else { return };
        let mut b = ExprBatch::new();
        b.push(&[]);
        b.push(&[]);
        let got = ev.eval(&b).expect("must not panic");
        assert_eq!(got.len(), 4, "2 expressions x 2 rows");
        assert!(got.iter().all(|v| v.is_nan()), "all slots must be NaN");
    }

    /// An unresolved RNC placeholder is not a variable — it is an index into
    /// the gene's Dc array, and the CALLER must resolve it before conversion.
    /// Converting it as a variable would read whatever column happened to be
    /// named "?" or, worse, silently pick column 0.
    #[test]
    fn rnc_placeholder_is_rejected_not_guessed() {
        let vars: Vec<String> = vec!["a".to_string(), "b".to_string()];
        let r = math_to_nodes(r#"(Var "?")"#, &vars);
        assert!(r.is_err(), "unresolved '?' must fail, got {r:?}");
    }

    /// A batch past one dispatch dimension (65535 groups x 64 = 4.19M
    /// invocations) must still evaluate EVERY expression. With a 1-D dispatch
    /// wgpu rejects the call outright; with a wrong stride the tail of the
    /// batch reads another expression's slot. 6000 x 1000 = 6M invocations,
    /// and expression i is `x + i`, so any misindexing shows as a wrong sum.
    #[cfg(feature = "gpu")]
    #[test]
    fn batch_larger_than_one_dispatch_dimension() {
        let n_rows = 1000usize;
        let n_expr = 6000usize;
        let xs: Vec<f32> = (0..n_rows).map(|r| r as f32).collect();
        let Ok(ev) = GpuEvaluator::new(&xs, 1) else {
            return;
        };
        let mut b = ExprBatch::new();
        for i in 0..n_expr {
            b.push(&[
                GpuNode {
                    op: Op::Add as u32,
                    arg0: 1,
                    arg1: 2,
                    konst: 0.0,
                },
                GpuNode {
                    op: Op::Var as u32,
                    arg0: 0,
                    arg1: 0,
                    konst: 0.0,
                },
                GpuNode {
                    op: Op::Num as u32,
                    arg0: 0,
                    arg1: 0,
                    konst: i as f32,
                },
            ]);
        }
        let out = ev.eval(&b).expect("eval");
        assert_eq!(out.len(), n_expr * n_rows);
        for &i in &[0usize, 1, 4193, 4194, 4195, n_expr - 1] {
            for &r in &[0usize, 1, n_rows - 1] {
                let got = out[i * n_rows + r];
                assert_eq!(got, (r + i) as f32, "expr {i} row {r}");
            }
        }
    }
}
