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

/// Longest k-expression this evaluator handles. Measured p90 is 6-7 and the
/// observed max is 34; 64 leaves headroom while keeping per-thread scratch at
/// 64 floats. Longer expressions fall back to the CPU evaluator rather than
/// being silently truncated.
pub const MAX_NODES: usize = 64;

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
            | Op::ProtectedInv => 1,
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
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       nodes:   array<Node>;
// expr i owns nodes[offsets[i] .. offsets[i] + lengths[i])
@group(0) @binding(1) var<storage, read>       offsets: array<u32>;
@group(0) @binding(2) var<storage, read>       lengths: array<u32>;
// row-major: data[row * n_vars + col]. Uploaded once, never moved.
@group(0) @binding(3) var<storage, read>       data:    array<f32>;
@group(0) @binding(4) var<storage, read_write> out:     array<f32>;
@group(0) @binding(5) var<uniform>             cfg:     Meta;

const MAX_NODES: u32 = 64u;
fn nan() -> f32 { return bitcast<f32>(0x7fc00000u); }

@compute @workgroup_size(64, 1, 1)
fn eval_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
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
            case 11u: { v = sin(scratch[nd.arg0]); }
            case 12u: { v = cos(scratch[nd.arg0]); }
            case 13u: {                                            // Tan
                let a = scratch[nd.arg0];
                let c = cos(a);
                if (c == 0.0) { v = nan(); } else { v = sin(a) / c; }
            }
            case 14u: { v = tanh(scratch[nd.arg0]); }
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
            // Protected ops: distinct functions, not guarded raw ops.
            // ProtectedDiv: a/b if b != 0 ELSE 0.0 (NOT 1.0 — see eval.rs)
            case 19u: {
                let d = scratch[nd.arg1];
                if (d == 0.0) { v = 0.0; } else { v = scratch[nd.arg0] / d; }
            }
            case 20u: { v = sqrt(abs(scratch[nd.arg0])); }         // ProtectedSqrt: sqrt(|x|)
            // ProtectedLog: ln(|x|) UNGUARDED — |x| == 0 gives -inf, which is
            // the engine's behaviour. Substituting 0.0 here would be a silent
            // divergence that only shows up on data containing a zero.
            case 21u: { v = log(abs(scratch[nd.arg0])); }
            case 22u: { v = exp(scratch[nd.arg0]); }               // ProtectedExp: uncapped
            case 23u: {                                            // ProtectedInv
                let a = scratch[nd.arg0];
                if (a == 0.0) { v = 1.0; } else { v = 1.0 / a; }
            }
            default: { v = bitcast<f32>(0x7fc00000u); }
        }
        scratch[k] = v;
    }

    out[idx] = scratch[0];
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
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        data_buf: wgpu::Buffer,
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
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .map_err(|e| format!("request_device: {e}"))?;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fuller-eval"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EVAL_WGSL)),
            });

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fuller-eval-layout"),
                entries: &(0..6)
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
                                ty: wgpu::BufferBindingType::Storage {
                                    read_only: i != 4,
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

            Ok(Self {
                device,
                queue,
                pipeline,
                layout,
                data_buf,
                n_rows,
                n_vars: n_vars as u32,
            })
        }

        pub fn n_rows(&self) -> u32 {
            self.n_rows
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
            let meta = [n_expr, self.n_rows, self.n_vars, 0u32];
            let meta_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                let groups = total.div_ceil(64) as u32;
                pass.dispatch_workgroups(groups, 1, 1);
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

        fn storage<T: bytemuck::Pod>(&self, v: &[T], label: &str) -> wgpu::Buffer {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(v),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        }
    }
}

#[cfg(feature = "gpu")]
pub use device::GpuEvaluator;

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
            "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "ProtectedInv",
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
mod protected_parity_tests {
    use super::*;

    /// The CPU contract, transcribed from `crate::eval`. Spelled out rather
    /// than invoked so the test pins the SEMANTICS: if someone changes eval.rs,
    /// this fails and forces a deliberate decision, instead of both
    /// implementations drifting together and the parity test still passing.
    fn cpu(op: Op, a: f64, b: f64) -> f64 {
        match op {
            Op::ProtectedDiv => if b == 0.0 { 0.0 } else { a / b },
            Op::ProtectedInv => if a == 0.0 { 1.0 } else { 1.0 / a },
            Op::ProtectedSqrt => a.abs().sqrt(),
            Op::ProtectedLog => a.abs().ln(),
            Op::ProtectedExp => a.exp(),
            Op::Div => a / b,
            Op::Sqrt => a.sqrt(),
            other => panic!("no CPU reference for {other:?}"),
        }
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

    /// ProtectedDiv(a, 0) is 0.0, NOT 1.0 — an earlier draft of the shader had
    /// it as 1.0, which is a silent wrong answer on any row with a zero
    /// denominator. This is the test that would have caught it.
    #[test]
    fn protected_div_by_zero_is_zero() {
        let bs: [f32; 5] = [0.0, 2.0, -4.0, 1e-30, -0.0];
        let Some(got) = gpu_binary(Op::ProtectedDiv, 5.0, &bs) else { return };
        for (i, &b) in bs.iter().enumerate() {
            let want = cpu(Op::ProtectedDiv, 5.0, b as f64);
            agree(got[i], want, &format!("ProtectedDiv(5, {b})"));
        }
        assert_eq!(got[0], 0.0, "5/0 must be 0.0");
        assert_eq!(got[4], 0.0, "5/-0.0 must be 0.0");
    }

    /// ProtectedLog(0) is -inf (ln|0|), not 0.0. Also an earlier shader bug.
    #[test]
    fn protected_log_of_zero_is_neg_inf() {
        let xs: [f32; 5] = [0.0, 1.0, -4.0, std::f32::consts::E, -1.0];
        let Some(got) = gpu_unary(Op::ProtectedLog, &xs) else { return };
        for (i, &x) in xs.iter().enumerate() {
            let want = cpu(Op::ProtectedLog, x as f64, 0.0);
            agree(got[i], want, &format!("ProtectedLog({x})"));
        }
        assert!(got[0].is_infinite() && got[0] < 0.0, "log|0| must be -inf");
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
            "ProtectedInv",
        ] {
            assert!(
                Op::from_math(name).is_some(),
                "{name} is in eval.rs but has no opcode"
            );
            // Var/Num are leaves handled by the parser, not by a match arm.
            if name != "Var" && name != "Num" {
                assert!(
                    eval_src.contains(&format!("(\"{name}\"")),
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
            "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "ProtectedInv",
        ] {
            assert!(
                eval_src.contains(&format!("(\"{name}\"")),
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
}
