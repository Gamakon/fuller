//! The linter kernel on the device (`kernel.wgsl`), and the host side of it.
//!
//! The host's part is everything that must be decided in f64: each literal's
//! class id and predicate bits, and a literal ID per distinct f64 value, so
//! the kernel compares integers only. The device's part is the greedy rounds.
//!
//! The result of a dispatch is, per expression, a canonical K-expression in
//! the same node layout the evaluator reads, plus how many rewrites were
//! applied and the weakest exactness used.

use std::collections::BTreeMap;

use super::flat::{Flat, LNode};
use super::pack::{pack_guards, LiteralCodes, PackedRule, GUARD_STRIDE, RULE_STRIDE};
use super::tables::{Exactness, GuardRule};
use crate::gpu_eval::Op;

/// Nodes per expression on the device, in and out. Matches `gpu_eval::MAX_NODES`.
pub const SLOT: usize = crate::gpu_eval::MAX_NODES;
/// Rewrites the kernel will apply to one expression. The longest chain met on
/// 113,444 live expressions was 11.
pub const DEFAULT_ROUNDS: u32 = 16;

pub const LINT_WGSL: &str = include_str!("kernel.wgsl");

/// Everything the kernel reads that does not change between dispatches.
#[derive(Debug, Clone)]
pub struct KernelTables {
    /// Rule rows sorted by (pattern root opcode, rule id), `RULE_STRIDE` words each.
    pub rule_words: Vec<u32>,
    /// `bucket[op] .. bucket[op + 1]` is the range of rules rooted at `op`.
    pub bucket: Vec<u32>,
    pub guard_words: Vec<u32>,
    pub n_guards: u32,
    pub codes: LiteralCodes,
}

impl KernelTables {
    pub fn new(rules: &[PackedRule], mut codes: LiteralCodes, guards: &[GuardRule]) -> Result<KernelTables, String> {
        let guard_words = pack_guards(guards, &mut codes)?;
        if codes.classes.len() > 255 {
            return Err("more than 255 literal classes: the class id is 8 bits".to_string());
        }
        if codes.preds.len() > 24 {
            return Err("more than 24 literal predicates: the predicate word is 24 bits".to_string());
        }
        let mut sorted: Vec<&PackedRule> = rules.iter().collect();
        sorted.sort_by_key(|r| (r.bucket(), r.rule_id));
        let n_ops = 24usize;
        let mut bucket = vec![0u32; n_ops + 1];
        for r in &sorted {
            bucket[r.bucket() as usize + 1] += 1;
        }
        for i in 0..n_ops {
            bucket[i + 1] += bucket[i];
        }
        let rule_words = sorted.iter().flat_map(|r| r.words()).collect::<Vec<u32>>();
        debug_assert_eq!(rule_words.len(), sorted.len() * RULE_STRIDE);
        Ok(KernelTables {
            rule_words,
            bucket,
            n_guards: (guard_words.len() / GUARD_STRIDE) as u32,
            guard_words,
            codes,
        })
    }

    fn aux(&self, v: f64) -> u32 {
        self.codes.class_of(v) | (self.codes.pred_bits(v) << 8)
    }
}

/// One dispatch's worth of expressions, encoded for the kernel.
pub struct Encoded {
    /// `SLOT` nodes per expression: (op, arg0, arg1, konst bits).
    pub nodes: Vec<u32>,
    pub aux: Vec<u32>,
    pub lengths: Vec<u32>,
    /// (konst bits, literal id, aux) per pool entry.
    pub pool: Vec<u32>,
    /// literal id -> the f64 it stands for.
    pub literals: Vec<f64>,
}

/// Encode expressions whose `vars` table is already the GLOBAL column list.
/// An expression longer than `SLOT` gets length 0: the kernel refuses it and
/// the host keeps its own form.
pub fn encode(exprs: &[Flat], tables: &KernelTables) -> Encoded {
    let mut ids: BTreeMap<u64, u32> = BTreeMap::new();
    let mut literals: Vec<f64> = Vec::new();
    let mut id_of = |v: f64, literals: &mut Vec<f64>| -> u32 {
        *ids.entry(v.to_bits()).or_insert_with(|| {
            literals.push(v);
            (literals.len() - 1) as u32
        })
    };
    let mut pool = Vec::with_capacity(tables.codes.pool.len() * 3);
    for v in &tables.codes.pool {
        let id = id_of(*v, &mut literals);
        pool.extend_from_slice(&[(*v as f32).to_bits(), id, tables.aux(*v)]);
    }
    let mut nodes = vec![0u32; exprs.len() * SLOT * 4];
    let mut aux = vec![0u32; exprs.len() * SLOT];
    let mut lengths = Vec::with_capacity(exprs.len());
    for (e, f) in exprs.iter().enumerate() {
        if f.nodes.len() > SLOT {
            lengths.push(0);
            continue;
        }
        lengths.push(f.nodes.len() as u32);
        for (i, n) in f.nodes.iter().enumerate() {
            let at = (e * SLOT + i) * 4;
            let is_num = n.op == Op::Num as u32;
            nodes[at] = n.op;
            nodes[at + 1] = n.arg0;
            nodes[at + 2] = if is_num { id_of(n.lit, &mut literals) } else { n.arg1 };
            nodes[at + 3] = (n.lit as f32).to_bits();
            aux[e * SLOT + i] = if is_num { tables.aux(n.lit) } else { 0 };
        }
    }
    Encoded { nodes, aux, lengths, pool, literals }
}

/// What came back for one expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Linted {
    /// `None` when the kernel refused the expression (too long): the caller
    /// keeps the form it already has.
    pub form: Option<Flat>,
    pub steps: u32,
    pub level: Exactness,
}

/// Decode the kernel's output. Literal ids become f64 values again, so the
/// result is exact in f64 although the device never held one.
pub fn decode(nodes_out: &[u32], info: &[u32], literals: &[f64], vars: &[String]) -> Vec<Linted> {
    info.chunks_exact(4)
        .enumerate()
        .map(|(e, i)| {
            let (len, steps, level, refused) = (i[0] as usize, i[1], i[2], i[3]);
            let level = [Exactness::Bit, Exactness::Rounding, Exactness::Finite][level.min(2) as usize];
            if refused != 0 {
                return Linted { form: None, steps: 0, level };
            }
            let nodes = (0..len)
                .map(|k| {
                    let at = (e * SLOT + k) * 4;
                    let op = nodes_out[at];
                    if op == Op::Num as u32 {
                        LNode { op, arg0: 0, arg1: 0, lit: literals[nodes_out[at + 2] as usize] }
                    } else {
                        LNode { op, arg0: nodes_out[at + 1], arg1: nodes_out[at + 2], lit: 0.0 }
                    }
                })
                .collect();
            Linted { form: Some(Flat { nodes, vars: vars.to_vec() }), steps, level }
        })
        .collect()
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    const MAX_GROUPS_PER_DIM: u32 = crate::gpu_eval::MAX_GROUPS_PER_DIM;

    pub struct LintKernel {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        rules_buf: wgpu::Buffer,
        bucket_buf: wgpu::Buffer,
        guards_buf: wgpu::Buffer,
        tables: KernelTables,
    }

    impl LintKernel {
        pub fn new(tables: KernelTables) -> Result<Self, String> {
            pollster::block_on(Self::new_async(tables))
        }

        async fn new_async(tables: KernelTables) -> Result<Self, String> {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok_or("no GPU adapter")?;
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("fuller-lint-device"),
                        required_features: wgpu::Features::empty(),
                        // The kernel binds ten storage buffers; the portable
                        // default allows eight.
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("request_device: {e}"))?;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fuller-lint"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LINT_WGSL)),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fuller-lint-layout"),
                entries: &(0..11)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if i == 10 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage { read_only: i < 8 }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fuller-lint-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "lint_main",
                compilation_options: Default::default(),
            });
            let resident = |label: &str, words: &[u32]| {
                // wgpu rejects a zero-sized binding.
                let padded: Vec<u32> = if words.is_empty() { vec![0] } else { words.to_vec() };
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&padded),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let rules_buf = resident("lint-rules", &tables.rule_words);
            let bucket_buf = resident("lint-bucket", &tables.bucket);
            let guards_buf = resident("lint-guards", &tables.guard_words);
            Ok(Self { device, queue, pipeline, layout, rules_buf, bucket_buf, guards_buf, tables })
        }

        pub fn tables(&self) -> &KernelTables {
            &self.tables
        }

        /// Lint every expression in one dispatch. `exprs` share the column
        /// list `vars`; `var_facts[c]` holds the caller's facts for column `c`.
        pub fn run(
            &self,
            exprs: &[Flat],
            vars: &[String],
            var_facts: &[u32],
            admit: Exactness,
            rounds: u32,
        ) -> Result<Vec<Linted>, String> {
            if exprs.is_empty() {
                return Ok(Vec::new());
            }
            let enc = encode(exprs, &self.tables);
            let n_expr = exprs.len() as u32;
            let storage = |label: &str, words: &[u32]| {
                let padded: Vec<u32> = if words.is_empty() { vec![0] } else { words.to_vec() };
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&padded),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let nodes_buf = storage("lint-nodes-in", &enc.nodes);
            let aux_buf = storage("lint-aux-in", &enc.aux);
            let len_buf = storage("lint-len-in", &enc.lengths);
            let pool_buf = storage("lint-pool", &enc.pool);
            let facts_buf = storage("lint-var-facts", var_facts);
            let out_bytes = (exprs.len() * SLOT * 16) as u64;
            let info_bytes = (exprs.len() * 16) as u64;
            let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let out_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let read_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
            let nodes_out = make("lint-nodes-out", out_bytes, out_usage);
            let info_out = make("lint-info-out", info_bytes, out_usage);
            let nodes_read = make("lint-nodes-read", out_bytes, read_usage);
            let info_read = make("lint-info-read", info_bytes, read_usage);

            let groups = n_expr.div_ceil(64);
            let groups_x = groups.min(MAX_GROUPS_PER_DIM);
            let groups_y = groups.div_ceil(groups_x);
            let cfg = [n_expr, self.tables.n_guards, admit as u32, rounds, groups_x * 64, 0, 0, 0];
            let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lint-cfg"),
                contents: bytemuck::cast_slice(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let buffers = [
                &nodes_buf, &aux_buf, &len_buf, &self.rules_buf, &self.bucket_buf, &self.guards_buf,
                &pool_buf, &facts_buf, &nodes_out, &info_out, &cfg_buf,
            ];
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &entries,
            });
            let mut enc_cmd = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc_cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            enc_cmd.copy_buffer_to_buffer(&nodes_out, 0, &nodes_read, 0, out_bytes);
            enc_cmd.copy_buffer_to_buffer(&info_out, 0, &info_read, 0, info_bytes);
            self.queue.submit(Some(enc_cmd.finish()));

            let read = |buf: &wgpu::Buffer| -> Result<Vec<u32>, String> {
                let slice = buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                self.device.poll(wgpu::Maintain::Wait);
                rx.recv()
                    .map_err(|e| format!("map_async channel: {e}"))?
                    .map_err(|e| format!("map_async: {e}"))?;
                let words = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
                buf.unmap();
                Ok(words)
            };
            let nodes_words = read(&nodes_read)?;
            let info_words = read(&info_read)?;

            // Release per-dispatch buffers explicitly and drain wgpu's deferred
            // queue — see gpu_eval::GpuEvaluator::eval for what happens otherwise.
            for b in [
                &nodes_buf, &aux_buf, &len_buf, &pool_buf, &facts_buf, &nodes_out, &info_out,
                &nodes_read, &info_read, &cfg_buf,
            ] {
                b.destroy();
            }
            self.device.poll(wgpu::Maintain::Poll);
            Ok(decode(&nodes_words, &info_words, &enc.literals, vars))
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::LintKernel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lint::node::Tree;
    use crate::lint::pack::pack;
    use crate::lint::tables::{Rule, Tables};

    fn kernel_tables() -> (Tables, KernelTables) {
        let tables = Tables::standard().unwrap();
        let rules: Vec<&Rule> = tables.rules.iter().collect();
        let p = pack(&rules);
        let kt = KernelTables::new(&p.rules, p.codes, &tables.guards).unwrap();
        (tables, kt)
    }

    #[test]
    fn rule_words_and_buckets_line_up() {
        let (_, kt) = kernel_tables();
        assert_eq!(kt.rule_words.len() % RULE_STRIDE, 0);
        let n_rules = (kt.rule_words.len() / RULE_STRIDE) as u32;
        assert_eq!(*kt.bucket.last().unwrap(), n_rules);
        // Every rule sits in the bucket of its own root opcode.
        for op in 0..24usize {
            for r in kt.bucket[op]..kt.bucket[op + 1] {
                let h = r as usize * RULE_STRIDE;
                let (root_kind, root_op) = (kt.rule_words[h + 1], kt.rule_words[h + 2]);
                let filed = if root_kind == crate::lint::pack::KIND_NUM { Op::Num as u32 } else { root_op };
                assert_eq!(filed as usize, op);
            }
        }
    }

    /// The WGSL's constants must be the ones the host packs with.
    #[test]
    fn wgsl_constants_match_the_host() {
        for needle in [
            format!("const RULE_STRIDE: u32 = {RULE_STRIDE}u;"),
            format!("const GUARD_STRIDE: u32 = {GUARD_STRIDE}u;"),
            format!("const SLOT: u32 = {SLOT}u;"),
            format!("const HEAP: u32 = {}u;", crate::lint::pack::HEAP_SLOTS),
        ] {
            assert!(LINT_WGSL.contains(&needle), "kernel.wgsl lacks `{needle}`");
        }
    }

    /// The kernel's arity table, read out of its source, agrees with `Op::arity`
    /// for every opcode — a wrong entry would mis-walk every tree.
    #[test]
    fn wgsl_arity_matches_the_opcodes() {
        let unary_line = LINT_WGSL
            .lines()
            .find(|l| l.contains("{ return 1u; }"))
            .expect("the unary case");
        let unary: Vec<u32> = unary_line
            .split("case")
            .nth(1)
            .unwrap()
            .split(':')
            .next()
            .unwrap()
            .split(',')
            .map(|t| t.trim().trim_end_matches('u').parse().unwrap())
            .collect();
        for code in 0..24u32 {
            let want = crate::lint::pack::arity(code);
            let got = if code <= 1 { 0 } else if unary.contains(&code) { 1 } else { 2 };
            assert_eq!(got, want, "opcode {code}");
        }
    }

    #[test]
    fn encode_then_decode_is_the_identity_in_f64() {
        let (_, kt) = kernel_tables();
        let vars = vec!["a".to_string(), "b".to_string()];
        let t = Tree::parse(r#"(Add (Mul (Num 1.0000000001) (Var "a")) (Sub (Var "b") (Num 1.0000000001)))"#).unwrap();
        let f = Flat::from_tree_in(&t, &vars).unwrap();
        let enc = encode(std::slice::from_ref(&f), &kt);
        // Same f64 value, same literal id; the f32 konst could not tell them
        // from 1.0, the id can.
        let lit_ids: Vec<u32> = (0..f.nodes.len())
            .filter(|i| f.nodes[*i].op == Op::Num as u32)
            .map(|i| enc.nodes[i * 4 + 2])
            .collect();
        assert_eq!(lit_ids[0], lit_ids[1]);
        let info = [f.nodes.len() as u32, 0, 0, 0];
        let back = decode(&enc.nodes, &info, &enc.literals, &vars);
        assert_eq!(back[0].form.as_ref().unwrap().to_tree(), t);
    }

    /// The device against the kernel-shaped CPU engine: same form, same number
    /// of rewrites, same exactness label, at every level.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_agrees_with_the_cpu_engine() {
        use crate::lint::engine::CallerFacts;
        use crate::lint::flat::run_greedy;
        let (tables, kt) = kernel_tables();
        let rules: Vec<&Rule> = tables.rules.iter().collect();
        let p = pack(&rules);
        let vars: Vec<String> = ["a", "b", "c", "x", "m"].iter().map(|s| s.to_string()).collect();
        let cases = [
            r#"(Pow2 (Sub (Neg (Var "a")) (Var "b")))"#,
            r#"(Sub (Neg (Mul (Var "a") (Num -1.0))) (Neg (Abs (Pow2 (Var "b")))))"#,
            r#"(ProtectedDiv (Sin (Neg (ProtectedInv (Abs (Exp (Var "x")))))) (Mul (Num -1.0) (Var "m")))"#,
            r#"(Sub (Add (Var "a") (Var "b")) (Var "b"))"#,
            r#"(Mul (Mul (Var "c") (Var "c")) (Inv (Var "c")))"#,
            r#"(Abs (ProtectedInv (ProtectedInv (Pow2 (Var "x")))))"#,
            r#"(Add (Var "x") (Mul (Sin (Var "x")) (Num 0.0)))"#,
            r#"(ProtectedDiv (Var "x") (Num 0.0000001))"#,
            r#"(Var "x")"#,
        ];
        let exprs: Vec<Flat> = cases
            .iter()
            .map(|c| Flat::from_tree_in(&Tree::parse(c).unwrap(), &vars).unwrap())
            .collect();
        let kernel = LintKernel::new(kt).expect("a GPU adapter");
        let var_facts = vec![0u32; vars.len()];
        let caller = CallerFacts::default();
        for admit in [Exactness::Bit, Exactness::Rounding, Exactness::Finite] {
            let got = kernel.run(&exprs, &vars, &var_facts, admit, DEFAULT_ROUNDS).unwrap();
            for ((case, start), dev) in cases.iter().zip(&exprs).zip(&got) {
                let want = run_greedy(start, &p.rules, &p.codes, &tables.guards, &caller, admit, DEFAULT_ROUNDS as usize);
                let form = dev.form.as_ref().expect("not refused");
                assert_eq!(form.to_tree().to_math(), want.form.to_tree().to_math(), "{case} at {admit:?}");
                assert_eq!(dev.steps as usize, want.steps, "{case} at {admit:?}: rewrites");
                assert_eq!(dev.level, want.level, "{case} at {admit:?}: level");
            }
        }
    }
}
