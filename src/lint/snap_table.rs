//! Snap on the device, stage 1: the constant lattice as a sorted table, and
//! the match — a numeric literal to the lattice entry it is within tolerance
//! of, or none.
//!
//! Snap searches the NUMBERS evolution and the least-squares wrap produce
//! against the lattice of known constant forms (`snap_karva::lattice`); named
//! constants are never planted as drawable terminals. This file is the table
//! and the search. The graft, the R² guard and the write-back into the gene
//! are later stages and build on the layout fixed here:
//!
//!   values[n]       f32   |value| of each entry, strictly ascending
//!   info[n × 4]     u32   (template offset in nodes, node count, flags, 0)
//!   templates[..]   u32   the entries' `math` forms, 4 words per node in the
//!                         linter's node layout: (op, arg0, arg1, f32 bits)
//!
//! An entry id is the entry's index in `values`, and indexes `info`.
//!
//! The rule is `snap.rs::best_match`'s: a literal `c` matches a constant `t`
//! iff `|c - t| / |t| <= rel_tol` (against `t == 0`, iff `|c| <= rel_tol`), the
//! smallest relative error wins, and every constant is tried as itself and
//! negated. A constant of the opposite sign has a relative error of at least
//! 1, so under the `rel_tol <= 0.5` this module accepts it never qualifies:
//! the table holds magnitudes and the sign travels as one bit.
//!
//! The device proposes in f32. Nothing it proposes leaves the device as a
//! constant until `confirm_f64` has re-derived the decision in f64.

use std::collections::BTreeSet;

use super::engine::LitMode;
use super::flat::{Flat, LNode};
use super::node::Tree;
use crate::gpu_eval::Op;
use crate::snap_karva::ConstEntry;

pub const SNAP_WGSL: &str = include_str!("snap.wgsl");

/// "No entry" on the device.
pub const NONE: u32 = u32::MAX;
/// Largest template the table carries. The linter's slot is 64 nodes and its
/// work area 80, which leaves room for 15 added nodes.
pub const TEMPLATE_MAX: usize = 15;
/// Words per entry in `info`.
pub const INFO_STRIDE: usize = 4;
/// `info` flag: the entry's template evaluates to MINUS its table value (the
/// lattice entry was negative and had no cheaper positive twin).
pub const FLAG_NEGATED: u32 = 1;

/// A match: which entry, and whether the form to graft is the entry's
/// template negated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub entry: u32,
    pub negative: bool,
}

/// What the build left out, counted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dropped {
    /// Value is NaN or infinite.
    pub non_finite: usize,
    /// Labels whose |value| is not a normal f32 (overflows to inf, or underflows to a
    /// subnormal or to zero): the device cannot hold it. A genuine zero is
    /// kept.
    pub out_of_f32_range: Vec<String>,
    /// (label, node count) of templates larger than `TEMPLATE_MAX`.
    pub oversize: Vec<(String, usize)>,
    /// Labels of the losers with the same |value| in f64 as the entry kept (this is where a negative entry
    /// goes when its positive twin exists).
    pub exact_duplicates: Vec<String>,
    /// Labels of the losers different in f64, the same f32: the device could not tell them apart,
    /// and the f32 and f64 searches would split on them.
    pub f32_collisions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SnapTable {
    /// |value| per entry, strictly ascending (in f64 and in f32).
    pub values: Vec<f64>,
    pub values_f32: Vec<f32>,
    pub labels: Vec<String>,
    pub maths: Vec<String>,
    /// `INFO_STRIDE` words per entry.
    pub info: Vec<u32>,
    /// Every template, 4 words per node, `Var` nodes indexing `names`.
    pub templates: Vec<u32>,
    /// The constant names the templates' `Var` nodes refer to, sorted.
    pub names: Vec<String>,
    /// Template sizes: `sizes[k]` entries have a `k`-node template.
    pub sizes: Vec<usize>,
    pub entries_in: usize,
    pub dropped: Dropped,
}

struct Candidate<'a> {
    magnitude: f64,
    entry: &'a ConstEntry,
    tree: Tree,
    nodes: usize,
}

fn var_names(tree: &Tree, out: &mut BTreeSet<String>) {
    match tree {
        Tree::Num(_) => {}
        Tree::Var(name) => {
            out.insert(name.clone());
        }
        Tree::App(_, kids) => kids.iter().for_each(|k| var_names(k, out)),
    }
}

/// The f64 choice: `snap.rs::best_match`'s arithmetic, on the two entries that
/// can win.
fn pick_f64(table: &[f64], x: f64, tol: f64) -> Option<usize> {
    let rel_err = |t: f64| if t == 0.0 { x } else { (x - t).abs() / t };
    let at = table.partition_point(|t| *t < x);
    let mut best: Option<(usize, f64)> = None;
    if at > 0 {
        let e = rel_err(table[at - 1]);
        if e <= tol {
            best = Some((at - 1, e));
        }
    }
    if at < table.len() {
        let e = rel_err(table[at]);
        let closer = match best {
            Some((_, b)) => e < b,
            None => true,
        };
        if e <= tol && closer {
            best = Some((at, e));
        }
    }
    best.map(|(i, _)| i)
}

const F32_MANTISSA: u32 = 0x007f_ffff;
/// The exponent the literal is moved to before any arithmetic: [4, 8).
const FRAME_EXPONENT: u32 = 129;

/// One candidate as the pair (distance, scale): its relative error is
/// `distance / scale`, and nobody divides. `x` and `t` are f32 bit patterns of
/// magnitudes. `None`: the two are more than a binade apart, so the relative
/// error is over 1/2 and the entry cannot qualify.
///
/// Both are first moved, by one exact power of two, to where the literal lies
/// in [4, 8). A device's f32 is not the host's at the ends of the range — this
/// kernel's first form, `abs(x - t) / t`, matched `f32::MAX` and
/// `f32::MIN_POSITIVE` to entries the host refused — and in this frame no
/// difference, product or tolerance is subnormal or overflows.
fn frame_f32(x: u32, t: u32) -> Option<(f32, f32)> {
    if t == 0 {
        // Against a zero constant the error is the literal's own magnitude.
        return Some((f32::from_bits(x), 1.0));
    }
    let (x_exp, t_exp) = (x >> 23, t >> 23);
    if x_exp == 0 || t_exp + 1 < x_exp || t_exp > x_exp + 1 {
        return None;
    }
    let xf = f32::from_bits((x & F32_MANTISSA) | (FRAME_EXPONENT << 23));
    let tf = f32::from_bits((t & F32_MANTISSA) | ((t_exp + FRAME_EXPONENT - x_exp) << 23));
    Some(((xf - tf).abs(), tf))
}

/// The f32 choice, the code `snap.wgsl` transcribes: an integer search (the
/// bit patterns of non-negative floats order as the floats do), then one
/// multiply per comparison.
fn pick_f32(table: &[f32], x: u32, tol: f32) -> Option<usize> {
    let at = table.partition_point(|t| t.to_bits() < x);
    let mut best: Option<(usize, f32, f32)> = None;
    if at > 0 {
        if let Some((d, scale)) = frame_f32(x, table[at - 1].to_bits()) {
            if d <= tol * scale {
                best = Some((at - 1, d, scale));
            }
        }
    }
    if at < table.len() {
        if let Some((d, scale)) = frame_f32(x, table[at].to_bits()) {
            // d / scale < best_d / best_scale, cross-multiplied.
            let closer = match best {
                Some((_, best_d, best_scale)) => d * best_scale < best_d * scale,
                None => true,
            };
            if d <= tol * scale && closer {
                best = Some((at, d, scale));
            }
        }
    }
    best.map(|(i, _, _)| i)
}

impl SnapTable {
    /// The table of the crate's own lattice.
    pub fn standard() -> Result<SnapTable, String> {
        SnapTable::build(&crate::snap_karva::lattice())
    }

    /// Order: ascending |value|, ties by label, then by math. Of entries that
    /// share an f32 value the one with the fewest template nodes is kept; on a
    /// tie a positive entry before a negative one, then by label, then math.
    pub fn build(entries: &[ConstEntry]) -> Result<SnapTable, String> {
        let mut dropped = Dropped::default();
        let mut cands: Vec<Candidate> = Vec::new();
        for e in entries {
            if !e.value.is_finite() {
                dropped.non_finite += 1;
                continue;
            }
            let magnitude = e.value.abs();
            if magnitude != 0.0 && !(magnitude as f32).is_normal() {
                dropped.out_of_f32_range.push(e.label.clone());
                continue;
            }
            let tree = Tree::parse(&e.math).map_err(|err| format!("lattice entry {:?}: {err}", e.label))?;
            let nodes = tree.node_count();
            if nodes > TEMPLATE_MAX {
                dropped.oversize.push((e.label.clone(), nodes));
                continue;
            }
            cands.push(Candidate { magnitude, entry: e, tree, nodes });
        }
        cands.sort_by(|a, b| {
            a.magnitude
                .total_cmp(&b.magnitude)
                .then_with(|| a.entry.label.cmp(&b.entry.label))
                .then_with(|| a.entry.math.cmp(&b.entry.math))
        });

        // The f32 cast is monotone, so entries that collide in f32 are
        // neighbours in the f64 order.
        let mut kept: Vec<Candidate> = Vec::new();
        for c in cands {
            let Some(last) = kept.last_mut() else {
                kept.push(c);
                continue;
            };
            if (last.magnitude as f32).to_bits() != (c.magnitude as f32).to_bits() {
                kept.push(c);
                continue;
            }
            // A negative entry and its positive twin meet here too.
            let exact = last.magnitude == c.magnitude;
            // Fewest nodes; then a positive entry before a negative one; then
            // the sort order (label, math), which `last` already leads.
            let loser = if (c.nodes, c.entry.value < 0.0) < (last.nodes, last.entry.value < 0.0) {
                std::mem::replace(last, c)
            } else {
                c
            };
            if exact {
                dropped.exact_duplicates.push(loser.entry.label.clone());
            } else {
                dropped.f32_collisions.push(loser.entry.label.clone());
            }
        }

        let mut name_set = BTreeSet::new();
        kept.iter().for_each(|c| var_names(&c.tree, &mut name_set));
        let names: Vec<String> = name_set.into_iter().collect();

        let mut table = SnapTable {
            values: Vec::with_capacity(kept.len()),
            values_f32: Vec::with_capacity(kept.len()),
            labels: Vec::with_capacity(kept.len()),
            maths: Vec::with_capacity(kept.len()),
            info: Vec::with_capacity(kept.len() * INFO_STRIDE),
            templates: Vec::new(),
            names,
            sizes: vec![0; TEMPLATE_MAX + 1],
            entries_in: entries.len(),
            dropped,
        };
        for c in &kept {
            let flat = Flat::from_tree_in(&c.tree, &table.names)?;
            let offset = (table.templates.len() / 4) as u32;
            for n in &flat.nodes {
                // A template's literals ride as f32 bits and nothing else, so
                // they have to be exact in f32.
                if n.op == Op::Num as u32 && f64::from(n.lit as f32) != n.lit {
                    return Err(format!("lattice entry {:?}: literal {} is not exact in f32", c.entry.label, n.lit));
                }
                table.templates.extend_from_slice(&[n.op, n.arg0, n.arg1, (n.lit as f32).to_bits()]);
            }
            let flags = if c.entry.value < 0.0 { FLAG_NEGATED } else { 0 };
            table.info.extend_from_slice(&[offset, flat.nodes.len() as u32, flags, 0]);
            table.sizes[flat.nodes.len()] += 1;
            table.values.push(c.magnitude);
            table.values_f32.push(c.magnitude as f32);
            table.labels.push(c.entry.label.clone());
            table.maths.push(c.entry.math.clone());
        }
        Ok(table)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Bytes resident on the device: values, info, templates.
    pub fn device_bytes(&self) -> (usize, usize, usize) {
        (self.values_f32.len() * 4, self.info.len() * 4, self.templates.len() * 4)
    }

    /// Steps the kernel's fixed binary search runs: `ceil(log2(n)) + 1`.
    pub fn search_steps(&self) -> u32 {
        self.len().next_power_of_two().trailing_zeros() + 1
    }

    /// An entry's template, read back out of the device block.
    pub fn template(&self, entry: u32) -> Flat {
        let at = entry as usize * INFO_STRIDE;
        let (offset, count) = (self.info[at] as usize, self.info[at + 1] as usize);
        let nodes = self.templates[offset * 4..(offset + count) * 4]
            .chunks_exact(4)
            .map(|w| LNode {
                op: w[0],
                arg0: w[1],
                arg1: w[2],
                lit: if w[0] == Op::Num as u32 { f64::from(f32::from_bits(w[3])) } else { 0.0 },
            })
            .collect();
        Flat { nodes, vars: self.names.clone() }
    }

    /// True when the entry's template evaluates to minus its table value.
    pub fn negated(&self, entry: u32) -> bool {
        self.info[entry as usize * INFO_STRIDE + 2] & FLAG_NEGATED != 0
    }

    /// The entry `v` snaps to, or `None`. `F32` is the device's arithmetic:
    /// the literal, the table and the tolerance are all rounded to f32 first.
    ///
    /// The closest entry by RELATIVE error wins (relative to the entry, as in
    /// `snap.rs`), and on an exact tie the lower entry. Only the entry below
    /// the literal and the one at or above it can win: below, the error falls
    /// as the entry rises; above, it rises.
    pub fn nearest(&self, v: f64, rel_tol: f64, mode: LitMode) -> Option<Hit> {
        assert!((0.0..=0.5).contains(&rel_tol), "rel_tol {rel_tol}: outside 0 ..= 0.5");
        let (entry, literal_negative) = match mode {
            LitMode::F64 => {
                if !v.is_finite() {
                    return None;
                }
                (pick_f64(&self.values, v.abs(), rel_tol)?, v.is_sign_negative())
            }
            LitMode::F32 => {
                let x = (v as f32).abs();
                // The kernel refuses these on the bit pattern.
                if !(x.is_normal() || x == 0.0) {
                    return None;
                }
                (pick_f32(&self.values_f32, x.to_bits(), rel_tol as f32)?, (v as f32).is_sign_negative())
            }
        };
        let entry = entry as u32;
        // Against a zero constant there is no sign: snap.rs tries +0 first.
        let negative = self.values[entry as usize] != 0.0 && (literal_negative != self.negated(entry));
        Some(Hit { entry, negative })
    }

    /// The host's word on a device proposal: true iff the f64 search makes
    /// exactly this decision for the f64 literal. Nothing the device matched
    /// may be reported or written back as a constant without it.
    pub fn confirm_f64(&self, value: f64, hit: Hit, rel_tol: f64) -> bool {
        self.nearest(value, rel_tol, LitMode::F64) == Some(hit)
    }

    /// The signed constant a hit stands for.
    pub fn signed_value(&self, hit: Hit) -> f64 {
        let v = self.values[hit.entry as usize];
        if hit.negative != self.negated(hit.entry) { -v } else { v }
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    const MAX_GROUPS_PER_DIM: u32 = crate::gpu_eval::MAX_GROUPS_PER_DIM;

    pub struct SnapKernel {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        values_buf: wgpu::Buffer,
        info_buf: wgpu::Buffer,
        templates_buf: wgpu::Buffer,
        table: SnapTable,
    }

    impl SnapKernel {
        pub fn new(table: SnapTable) -> Result<Self, String> {
            pollster::block_on(Self::new_async(table))
        }

        async fn new_async(table: SnapTable) -> Result<Self, String> {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok_or("no GPU adapter")?;
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("fuller-snap-device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("request_device: {e}"))?;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fuller-snap"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SNAP_WGSL)),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fuller-snap-layout"),
                entries: &(0..5)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if i == 4 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage { read_only: i < 3 }
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
                label: Some("fuller-snap-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "snap_main",
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
            let value_bits: Vec<u32> = table.values_f32.iter().map(|v| v.to_bits()).collect();
            let values_buf = resident("snap-values", &value_bits);
            let info_buf = resident("snap-info", &table.info);
            let templates_buf = resident("snap-templates", &table.templates);
            Ok(Self { device, queue, pipeline, layout, values_buf, info_buf, templates_buf, table })
        }

        pub fn table(&self) -> &SnapTable {
            &self.table
        }

        /// The resident template block, for the graft kernel to bind (the
        /// match kernel does not read it).
        pub fn templates_buffer(&self) -> &wgpu::Buffer {
            &self.templates_buf
        }

        /// The resident per-entry (offset, count, flags, 0) rows.
        pub fn info_buffer(&self) -> &wgpu::Buffer {
            &self.info_buf
        }

        /// Match every literal in one dispatch.
        pub fn run(&self, literals: &[f32], rel_tol: f64) -> Result<Vec<Option<Hit>>, String> {
            assert!((0.0..=0.5).contains(&rel_tol), "rel_tol {rel_tol}: outside 0 ..= 0.5");
            if literals.is_empty() {
                return Ok(Vec::new());
            }
            let n_lit = u32::try_from(literals.len()).map_err(|_| "more than u32::MAX literals".to_string())?;
            let bits: Vec<u32> = literals.iter().map(|v| v.to_bits()).collect();
            let lits_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("snap-literals"),
                contents: bytemuck::cast_slice(&bits),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let out_bytes = (literals.len() * 8) as u64;
            let hits_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snap-hits"),
                size: out_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snap-hits-read"),
                size: out_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let groups = n_lit.div_ceil(64);
            let groups_x = groups.min(MAX_GROUPS_PER_DIM);
            let groups_y = groups.div_ceil(groups_x);
            let cfg = [
                n_lit,
                self.table.len() as u32,
                groups_x * 64,
                self.table.search_steps(),
                (rel_tol as f32).to_bits(),
                0,
                0,
                0,
            ];
            let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("snap-cfg"),
                contents: bytemuck::cast_slice(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let buffers = [&lits_buf, &self.values_buf, &self.info_buf, &hits_buf, &cfg_buf];
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
            enc.copy_buffer_to_buffer(&hits_buf, 0, &read_buf, 0, out_bytes);
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
            let words = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
            read_buf.unmap();

            // Release per-dispatch buffers explicitly and drain wgpu's deferred
            // queue — see gpu_eval::GpuEvaluator::eval for what happens otherwise.
            for b in [&lits_buf, &hits_buf, &read_buf, &cfg_buf] {
                b.destroy();
            }
            self.device.poll(wgpu::Maintain::Poll);
            Ok(words
                .chunks_exact(2)
                .map(|w| (w[0] != NONE).then_some(Hit { entry: w[0], negative: w[1] != 0 }))
                .collect())
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::SnapKernel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolve::draw;
    use crate::lint::pack::arity;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-3;

    /// The constants as SRBench's laws PRINT them (3 significant figures) —
    /// text, because a number typed as 3.14 is a rounded constant on purpose.
    const SRBENCH: [&str; 23] = [
        "0.013", "0.016", "0.04", "0.048", "0.05", "0.053", "0.075", "0.08", "0.1", "0.101", "0.119", "0.125",
        "0.159", "0.239", "0.318", "0.399", "0.5", "2.89", "3.14", "4.19", "6.28", "12.6", "25.1",
    ];

    fn printed(text: &str) -> f64 {
        text.parse().expect("a number")
    }

    fn table() -> SnapTable {
        SnapTable::standard().expect("the lattice builds")
    }

    /// 10,000 literals log-uniform over 1e-6..1e6, a fair coin for the sign,
    /// then SRBench's 23. Rounded to f32 — what a gene's constant is — so the
    /// f32 and f64 searches start from the same number.
    fn literals() -> Vec<f32> {
        let mut out: Vec<f32> = (0..10_000u32)
            .map(|row| {
                let u = (draw(7, 0, row, 0, 0) >> 11) as f64 / (1u64 << 53) as f64;
                let magnitude = 10f64.powf(-6.0 + 12.0 * u);
                let v = if crate::evolve::coin(draw(7, 0, row, 1, 0)) { -magnitude } else { magnitude };
                v as f32
            })
            .collect();
        out.extend(SRBENCH.iter().map(|v| printed(v) as f32));
        out
    }

    /// `snap.rs::best_match`, transcribed over the table: every entry, as
    /// itself and negated, strict `<`.
    fn brute_force(t: &SnapTable, v: f64, rel_tol: f64) -> Option<f64> {
        if !v.is_finite() {
            return None;
        }
        let mut best: Option<(f64, f64)> = None;
        for &val in &t.values {
            for signed in [val, -val] {
                let rel_err = if signed == 0.0 { v.abs() } else { (v - signed).abs() / signed.abs() };
                if rel_err <= rel_tol && best.as_ref().map(|b| rel_err < b.1).unwrap_or(true) {
                    best = Some((signed, rel_err));
                }
            }
        }
        best.map(|b| b.0)
    }

    fn entry(value: f64, math: &str, label: &str) -> ConstEntry {
        ConstEntry { value, math: math.to_string(), label: label.to_string() }
    }

    #[test]
    fn the_table_is_strictly_ascending_and_its_counts_are_these() {
        let t = table();
        assert!(t.values.windows(2).all(|w| w[0] < w[1]), "f64 values");
        assert!(t.values_f32.windows(2).all(|w| w[0] < w[1]), "f32 values");
        assert!(t.values_f32.iter().all(|v| v.is_normal()));
        let (values, info, templates) = t.device_bytes();
        eprintln!(
            "snap table: {} lattice entries -> {} on the device; out of f32 range {} (first {:?}); exact duplicates {:?}; \
             f32 collisions {:?}; template sizes {:?}; \
             {} names; bytes: values {values}, info {info}, templates {templates}; search steps {}",
            t.entries_in,
            t.len(),
            t.dropped.out_of_f32_range.len(),
            &t.dropped.out_of_f32_range[..6],
            t.dropped.exact_duplicates,
            t.dropped.f32_collisions,
            t.sizes,
            t.names.len(),
            t.search_steps()
        );
        // A snapshot: a change to the lattice has to show up here.
        assert_eq!(t.entries_in, 7051);
        assert_eq!(t.dropped.non_finite, 0);
        assert_eq!(t.dropped.oversize, Vec::<(String, usize)>::new());
        assert_eq!(t.dropped.out_of_f32_range.len(), 605);
        // 37 = the 17 negative constants and the 20 negative small rationals,
        // each folded onto its positive twin.
        assert_eq!(t.dropped.exact_duplicates.len(), 37);
        assert!(t.dropped.exact_duplicates.iter().all(|l| l.starts_with('-')), "{:?}", t.dropped.exact_duplicates);
        assert_eq!(t.dropped.f32_collisions.len(), 6);
        assert_eq!(t.len(), 6403);
        assert_eq!(t.sizes[..8], [0, 29, 17, 990, 695, 1177, 511, 2984]);
        assert!(t.info.chunks_exact(INFO_STRIDE).all(|w| w[2] == 0), "no negated entry survives in the standard table");
        assert_eq!(t.info.len(), t.len() * INFO_STRIDE);
        assert_eq!(t.sizes.iter().sum::<usize>(), t.len());
        assert_eq!(
            t.entries_in,
            t.len() + t.dropped.out_of_f32_range.len() + t.dropped.exact_duplicates.len() + t.dropped.f32_collisions.len()
        );
    }

    #[test]
    fn every_template_fits_and_evaluates_to_its_entry() {
        let t = table();
        let mut row: Vec<(String, f64)> =
            crate::snap_karva::constant_values().iter().map(|(k, v)| (k.clone(), *v)).collect();
        row.sort_by(|a, b| a.0.cmp(&b.0));
        for id in 0..t.len() as u32 {
            let flat = t.template(id);
            assert!(flat.nodes.len() <= TEMPLATE_MAX, "{}", t.labels[id as usize]);
            let tree = flat.to_tree();
            assert_eq!(tree.to_math(), Tree::parse(&t.maths[id as usize]).unwrap().to_math());
            let got = tree.eval(&row).unwrap();
            let want = if t.negated(id) { -t.values[id as usize] } else { t.values[id as usize] };
            assert!(
                ((got - want) / want).abs() <= 1e-12,
                "{}: template gives {got}, entry says {want}",
                t.labels[id as usize]
            );
            // Children after their parent, inside the template.
            for (k, n) in flat.nodes.iter().enumerate() {
                for a in [n.arg0, n.arg1].iter().take(arity(n.op)) {
                    assert!((*a as usize) > k && (*a as usize) < flat.nodes.len());
                }
            }
        }
    }

    #[test]
    fn known_constants_match() {
        let t = table();
        let value_of = |v: f64| t.nearest(v, TOL, LitMode::F64).map(|h| t.signed_value(h));
        let pi = t.nearest(PI, TOL, LitMode::F64).expect("pi");
        assert_eq!(t.labels[pi.entry as usize], "pi");
        assert!(!pi.negative);
        assert_eq!(value_of(0.0796), Some(1.0 / (4.0 * PI)));
        assert_eq!(value_of(printed("6.2831")), Some(2.0 * PI));
        let half_turn = value_of(0.159155).expect("1/(2 pi)");
        // Several forms share this value to within an f32; whichever was kept,
        // it is 1/(2 pi) to the last digit a gene's f32 can hold.
        assert!((half_turn * 2.0 * PI - 1.0).abs() <= 1e-7, "{half_turn}");
        // Negative literals carry the sign and match the same entry.
        let minus_pi = t.nearest(-printed("3.1416"), TOL, LitMode::F64).expect("-pi");
        assert_eq!(minus_pi, Hit { entry: pi.entry, negative: true });
        assert_eq!(t.signed_value(minus_pi), -PI);
    }

    #[test]
    fn far_from_everything_and_not_a_number_are_none() {
        let t = table();
        // Midway between two neighbours that are at least 4e-3 apart.
        let gap = t.values.windows(2).find(|w| w[0] > 1.0 && w[1] / w[0] > 1.01).expect("a gap");
        let lonely = (gap[0] * gap[1]).sqrt();
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(lonely, 2e-3, mode), None);
            assert_eq!(t.nearest(0.0, TOL, mode), None);
            assert_eq!(t.nearest(-0.0, TOL, mode), None);
            assert_eq!(t.nearest(f64::NAN, TOL, mode), None);
            assert_eq!(t.nearest(f64::INFINITY, TOL, mode), None);
            assert_eq!(t.nearest(f64::NEG_INFINITY, TOL, mode), None);
        }
        // Finite in f64, not in f32.
        assert_eq!(t.nearest(1e300, TOL, LitMode::F32), None);
        assert_eq!(t.nearest(1e-42, TOL, LitMode::F32), None);
    }

    #[test]
    fn the_tolerance_band_has_a_hard_edge() {
        let t = table();
        // An entry with 1% of clear air either side: no other band reaches
        // its edges. (pi is not one: `3307` sits 1e-3 below it.)
        let alone = (1..t.len() - 1)
            .find(|i| t.values[*i] > 1.0 && t.values[*i] / t.values[i - 1] > 1.01 && t.values[i + 1] / t.values[*i] > 1.01)
            .expect("an isolated entry");
        let (value, hit) = (t.values[alone], Hit { entry: alone as u32, negative: false });
        for side in [-1.0, 1.0] {
            let inside = value * (1.0 + side * TOL * (1.0 - 1e-9));
            let outside = value * (1.0 + side * TOL * (1.0 + 1e-9));
            assert_eq!(t.nearest(inside, TOL, LitMode::F64), Some(hit), "inside, side {side}");
            assert_eq!(t.nearest(outside, TOL, LitMode::F64), None, "outside, side {side}");
            assert!(t.confirm_f64(inside, hit, TOL));
            assert!(!t.confirm_f64(outside, hit, TOL));
            assert_eq!(t.nearest(-inside, TOL, LitMode::F64), Some(Hit { negative: true, ..hit }));
        }
        let pi = t.nearest(PI, TOL, LitMode::F64).unwrap();
        // rel_tol 0 is exact equality.
        assert_eq!(t.nearest(PI, 0.0, LitMode::F64), Some(pi));
        assert_eq!(t.nearest(PI * (1.0 + 1e-15), 0.0, LitMode::F64), None);
    }

    #[test]
    fn ties_keep_the_lower_entry_and_relative_error_decides() {
        let t = SnapTable::build(&[entry(1.0, "(Num 1.0)", "1"), entry(3.0, "(Num 3.0)", "3")]).unwrap();
        // 1.5: relative error exactly 0.5 against both. The lower entry keeps it.
        let lower = Some(Hit { entry: 0, negative: false });
        let upper = Some(Hit { entry: 1, negative: false });
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(1.5, 0.5, mode), lower);
            // 1.25 is nearer 1 on the number line and nearer 1 by ratio too;
            // 2.0 is the same distance from both, but 1/3 of 3 and 1/1 of 1.
            assert_eq!(t.nearest(1.25, 0.5, mode), lower);
            assert_eq!(t.nearest(2.0, 0.5, mode), upper);
        }
        assert_eq!(brute_force(&t, 1.5, 0.5), Some(1.0));
        // Same distance, both inside the tolerance: the smaller RATIO wins.
        let t = SnapTable::build(&[entry(2.0, "(Num 2.0)", "2"), entry(4.0, "(Num 4.0)", "4")]).unwrap();
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(3.0, 0.5, mode), upper);
        }
        assert_eq!(brute_force(&t, 3.0, 0.5), Some(4.0));
    }

    #[test]
    fn a_zero_entry_uses_the_absolute_rule_and_has_no_sign() {
        let t = SnapTable::build(&[entry(0.0, "(Num 0.0)", "0"), entry(1.0, "(Num 1.0)", "1")]).unwrap();
        let zero = Some(Hit { entry: 0, negative: false });
        for mode in [LitMode::F64, LitMode::F32] {
            assert_eq!(t.nearest(0.0, TOL, mode), zero);
            assert_eq!(t.nearest(-0.0, TOL, mode), zero);
            assert_eq!(t.nearest(9.9e-4, TOL, mode), zero);
            assert_eq!(t.nearest(-9.9e-4, TOL, mode), zero, "snap.rs tries +0 first: no sign");
            assert_eq!(t.nearest(1.1e-3, TOL, mode), None);
            assert_eq!(t.nearest(1.0005, TOL, mode), Some(Hit { entry: 1, negative: false }));
        }
        assert_eq!(brute_force(&t, -9.9e-4, TOL), Some(0.0));
    }

    #[test]
    fn duplicates_negatives_and_the_f32_range_are_handled() {
        let t = SnapTable::build(&[
            entry(-2.0, r#"(Neg (Var "two"))"#, "-two"),
            entry(2.0, r#"(Mul (Var "one") (Var "two"))"#, "b"),
            entry(2.0, r#"(Var "two")"#, "two"),
            entry(2.0 + 4e-16, r#"(Add (Var "two") (Var "eps"))"#, "two+eps"),
            entry(-5.0, r#"(Neg (Var "five"))"#, "-five"),
            entry(1e-60, r#"(Var "tiny")"#, "tiny"),
            entry(1e60, r#"(Var "huge")"#, "huge"),
            entry(f64::NAN, r#"(Var "nan")"#, "nan"),
        ])
        .unwrap();
        assert_eq!(t.labels, ["two", "-five"]);
        assert_eq!(t.names, ["five", "two"]);
        let labels = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let want = Dropped {
            non_finite: 1,
            out_of_f32_range: labels(&["tiny", "huge"]),
            oversize: vec![],
            exact_duplicates: labels(&["b", "-two"]),
            f32_collisions: labels(&["two+eps"]),
        };
        assert_eq!(t.dropped, want);
        // A negative entry with no positive twin: its template is already the
        // negative form, so a NEGATIVE literal takes it as it is.
        assert_eq!(t.nearest(-5.0, TOL, LitMode::F64), Some(Hit { entry: 1, negative: false }));
        assert_eq!(t.nearest(5.0, TOL, LitMode::F64), Some(Hit { entry: 1, negative: true }));
        // Sixteen nodes is one too many.
        let mut big = r#"(Var "x")"#.to_string();
        for _ in 0..15 {
            big = format!("(Neg {big})");
        }
        let t = SnapTable::build(&[entry(1.0, &big, "big")]).unwrap();
        assert_eq!(t.dropped.oversize, [("big".to_string(), 16)]);
        assert!(t.is_empty());
        assert_eq!(t.nearest(1.0, TOL, LitMode::F64), None);
        // A template literal the device's f32 cannot hold exactly is refused.
        assert!(SnapTable::build(&[entry(0.1, "(Num 0.1)", "tenth")]).is_err());
    }

    #[test]
    fn the_binary_search_agrees_with_a_linear_scan() {
        let t = table();
        let lits = literals();
        let mut hits = 0usize;
        for rel_tol in [TOL, 5e-3] {
            for v in &lits {
                let v = f64::from(*v);
                let got = t.nearest(v, rel_tol, LitMode::F64).map(|h| t.signed_value(h));
                assert_eq!(got, brute_force(&t, v, rel_tol), "literal {v} at {rel_tol}");
                hits += usize::from(got.is_some());
            }
        }
        assert!(hits > 0);
        // SRBench's printed constants, as findings.
        for rel_tol in [TOL, 5e-3] {
            let matched: Vec<String> = SRBENCH
                .iter()
                .filter_map(|v| t.nearest(printed(v), rel_tol, LitMode::F64).map(|h| format!("{v}->{}", t.labels[h.entry as usize])))
                .collect();
            eprintln!("SRBench constants matched at {rel_tol}: {} of 23: {}", matched.len(), matched.join(", "));
            // A snapshot of the finding, not a target: the laws print three
            // significant figures, and the closest form is often not the law's.
            assert_eq!(matched.len(), if rel_tol == TOL { 13 } else { 19 });
        }
    }

    /// Literals a hair either side of every entry's band edge, in f32.
    fn edge_literals(t: &SnapTable) -> Vec<f32> {
        let mut out = Vec::new();
        for v in &t.values {
            for side in [-1.0, 1.0] {
                let edge = (v * (1.0 + side * TOL)) as f32;
                for step in [-2i32, -1, 0, 1, 2] {
                    out.push(f32::from_bits((edge.to_bits() as i32 + step) as u32));
                }
            }
        }
        out
    }

    /// Where F32 and F64 part ways, the literal sits on an edge: a candidate's
    /// relative error within 1e-6 of the tolerance, or the two candidates'
    /// errors within 1e-6 of each other.
    fn on_an_edge(t: &SnapTable, v: f64) -> bool {
        let x = v.abs();
        let at = t.values.partition_point(|e| *e < x);
        let errs: Vec<f64> = [at.checked_sub(1), (at < t.len()).then_some(at)]
            .iter()
            .flatten()
            .map(|i| (x - t.values[*i]).abs() / t.values[*i])
            .collect();
        errs.iter().any(|e| (e - TOL).abs() <= 1e-6) || (errs.len() == 2 && (errs[0] - errs[1]).abs() <= 1e-6)
    }

    #[test]
    fn f32_and_f64_part_ways_only_on_an_edge() {
        let t = table();
        for (name, lits) in [("random + SRBench", literals()), ("band edges", edge_literals(&t))] {
            let mut differ = 0usize;
            let mut unconfirmed = 0usize;
            for v in &lits {
                let v = f64::from(*v);
                let narrow = t.nearest(v, TOL, LitMode::F32);
                if narrow != t.nearest(v, TOL, LitMode::F64) {
                    differ += 1;
                    assert!(on_an_edge(&t, v), "{v}: F32 and F64 differ away from any edge");
                }
                // The hook the later stages call: it turns down exactly the
                // proposals f64 would not have made.
                if let Some(hit) = narrow {
                    let ok = t.confirm_f64(v, hit, TOL);
                    assert_eq!(ok, t.nearest(v, TOL, LitMode::F64) == Some(hit));
                    unconfirmed += usize::from(!ok);
                }
            }
            eprintln!("F32 vs F64 on {} {name} literals: {differ} differ, {unconfirmed} F32 proposals refused by confirm_f64", lits.len());
            // The edge set is built to disagree; the random one must hardly ever.
            if name != "band edges" {
                assert!(differ * 1000 <= lits.len(), "{name}: {differ} of {}", lits.len());
            }
        }
    }

    #[test]
    fn two_builds_and_two_searches_are_identical() {
        let (a, b) = (table(), table());
        assert_eq!(a.values_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.values_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!((&a.info, &a.templates, &a.names, &a.labels), (&b.info, &b.templates, &b.names, &b.labels));
        // Input order does not matter either.
        let mut reversed = crate::snap_karva::lattice();
        reversed.reverse();
        let c = SnapTable::build(&reversed).unwrap();
        assert_eq!((&a.info, &a.templates, &a.labels), (&c.info, &c.templates, &c.labels));
        let run = |t: &SnapTable| literals().iter().map(|v| t.nearest(f64::from(*v), TOL, LitMode::F32)).collect::<Vec<_>>();
        assert_eq!(run(&a), run(&c));
    }

    /// The kernel's layout constants are the host's.
    #[test]
    fn wgsl_constants_match_the_host() {
        for needle in [
            format!("const INFO_STRIDE: u32 = {INFO_STRIDE}u;"),
            format!("const FLAG_NEGATED: u32 = {FLAG_NEGATED}u;"),
            "const NONE: u32 = 0xffffffffu;".to_string(),
            format!("const MANTISSA: u32 = {F32_MANTISSA:#010x}u;"),
            format!("const FRAME_EXPONENT: u32 = {FRAME_EXPONENT}u;"),
        ] {
            assert!(SNAP_WGSL.contains(&needle), "snap.wgsl lacks `{needle}`");
        }
    }

    #[cfg(feature = "gpu")]
    fn twin(t: &SnapTable, lits: &[f32]) -> Vec<Option<Hit>> {
        lits.iter().map(|v| t.nearest(f64::from(*v), TOL, LitMode::F32)).collect()
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn device_agrees_with_the_f32_twin() {
        let kernel = SnapKernel::new(table()).expect("a GPU adapter");
        let t = kernel.table().clone();
        assert_eq!(kernel.run(&[], TOL).unwrap(), Vec::new());
        let one = [printed("3.1416") as f32];
        assert_eq!(kernel.run(&one, TOL).unwrap(), twin(&t, &one));
        assert!(kernel.run(&one, TOL).unwrap()[0].is_some());

        let specials = [0.0f32, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MIN_POSITIVE / 2.0, f32::MAX, f32::MIN_POSITIVE];
        assert_eq!(kernel.run(&specials, TOL).unwrap(), twin(&t, &specials));

        for (name, lits) in [("random + SRBench", literals()), ("band edges", edge_literals(&t))] {
            let start = std::time::Instant::now();
            let got = kernel.run(&lits, TOL).unwrap();
            let took = start.elapsed();
            let again = kernel.run(&lits, TOL).unwrap();
            let want = twin(&t, &lits);
            let differ = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            let hits = got.iter().flatten().count();
            eprintln!("device snap, {} {name} literals: {hits} hits, {differ} differ from the F32 twin, {took:?}", lits.len());
            assert_eq!(got, want, "{name}");
            assert_eq!(got, again, "{name}: two dispatches");
        }
    }

    /// A zero entry and a negated entry on the device.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_handles_zero_and_negated_entries() {
        let t = SnapTable::build(&[
            entry(0.0, "(Num 0.0)", "0"),
            entry(1.0, "(Num 1.0)", "1"),
            entry(-5.0, r#"(Neg (Var "five"))"#, "-five"),
        ])
        .unwrap();
        let kernel = SnapKernel::new(t.clone()).expect("a GPU adapter");
        let lits = [0.0f32, -0.0, 9.9e-4, -9.9e-4, 1.1e-3, 1.0005, -1.0005, 5.0, -5.0, 3.0];
        let got = kernel.run(&lits, TOL).unwrap();
        assert_eq!(got, twin(&t, &lits));
        assert_eq!(got[3], Some(Hit { entry: 0, negative: false }));
        assert_eq!(got[7], Some(Hit { entry: 2, negative: true }));
        assert_eq!(got[8], Some(Hit { entry: 2, negative: false }));
        // An empty table matches nothing.
        let empty = SnapKernel::new(SnapTable::build(&[]).unwrap()).expect("a GPU adapter");
        assert_eq!(empty.run(&lits, TOL).unwrap(), vec![None; lits.len()]);
    }

    /// 200,000 literals in one row of groups; 4,200,000 wrap into a second.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_large_batches_match() {
        let kernel = SnapKernel::new(table()).expect("a GPU adapter");
        let t = kernel.table().clone();
        let base = literals();
        for n in [200_000usize, 4_200_000] {
            let lits: Vec<f32> = (0..n).map(|i| base[i % base.len()] * (1.0 + (i / base.len()) as f32 * 1e-4)).collect();
            assert_eq!(n.div_ceil(64) > crate::gpu_eval::MAX_GROUPS_PER_DIM as usize, n > 4_194_240);
            let start = std::time::Instant::now();
            let got = kernel.run(&lits, TOL).unwrap();
            let device = start.elapsed();
            let start = std::time::Instant::now();
            let want = twin(&t, &lits);
            let cpu = start.elapsed();
            eprintln!("device snap, {n} literals: {} hits, device {device:?}, CPU F32 twin {cpu:?}", got.iter().flatten().count());
            assert_eq!(got.len(), n);
            assert!(got == want, "{n} literals: device and F32 twin differ");
        }
    }
}
