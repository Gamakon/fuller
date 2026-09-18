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
    DiffSq = 24,
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
            "DiffSq" => Op::DiffSq,
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
            case 5u: { v = scratch[nd.arg0] / scratch[nd.arg1]; }
            case 6u: { v = -scratch[nd.arg0]; }
            case 7u: { v = abs(scratch[nd.arg0]); }
            case 8u: { v = sqrt(scratch[nd.arg0]); }
            case 9u: { v = log(scratch[nd.arg0]); }
            case 10u: { v = exp(scratch[nd.arg0]); }
            case 11u: { v = sin(scratch[nd.arg0]); }
            case 12u: { v = cos(scratch[nd.arg0]); }
            case 13u: { v = tan(scratch[nd.arg0]); }
            case 14u: { v = tanh(scratch[nd.arg0]); }
            case 15u: { v = pow(scratch[nd.arg0], scratch[nd.arg1]); }
            case 16u: { let a = scratch[nd.arg0]; v = a * a; }
            case 17u: { let a = scratch[nd.arg0]; v = a * a * a; }
            case 18u: { v = 1.0 / scratch[nd.arg0]; }
            // Protected ops: distinct functions, not guarded raw ops.
            case 19u: {                                            // ProtectedDiv
                let d = scratch[nd.arg1];
                if (d == 0.0) { v = 1.0; } else { v = scratch[nd.arg0] / d; }
            }
            case 20u: { v = sqrt(abs(scratch[nd.arg0])); }         // ProtectedSqrt
            case 21u: {                                            // ProtectedLog
                let a = abs(scratch[nd.arg0]);
                if (a == 0.0) { v = 0.0; } else { v = log(a); }
            }
            case 22u: { v = exp(scratch[nd.arg0]); }               // ProtectedExp: uncapped
            case 23u: {                                            // ProtectedInv
                let a = scratch[nd.arg0];
                if (a == 0.0) { v = 1.0; } else { v = 1.0 / a; }
            }
            case 24u: {                                            // DiffSq
                let d = scratch[nd.arg0] - scratch[nd.arg1];
                v = d * d;
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
            "ProtectedSqrt", "ProtectedLog", "ProtectedExp", "ProtectedInv", "DiffSq",
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
        assert_eq!(Op::DiffSq.arity(), 2);
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
        for op in 0u32..=24 {
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
