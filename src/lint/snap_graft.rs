//! Snap on the device, stage 2: the graft — a matched literal replaced by its
//! lattice entry's symbolic form, giving the SNAPPED VARIANTS of an expression.
//!
//! Stage 1 (`snap_table.rs`) says which entry a literal is within tolerance of.
//! This file splices the entry's template in where the literal was, with the
//! linter's own relayout (`flat.rs::splice`, `splice.wgsl`), so a variant is a
//! canonical K-expression like any other. The R² guard and the write-back into
//! the gene are later stages and read the blocks laid out here as they are.
//!
//! THE VARIANTS. An ATOM is a distinct literal (by f64 bit pattern — on the
//! device, by literal id) that has a hit; atoms are numbered by the first node
//! that holds them. This is `snap_karva::snap_variants`' policy at a fixed
//! width: it returns the original, one candidate per snapped atom with every
//! site of that atom replaced, then all atoms at once. Here the original is
//! the caller's already, slots `0 .. VARIANTS-2` are the first atoms' single
//! snaps and the last slot is every atom at once (all of them, not only the
//! first `VARIANTS-1`), filled only when there are two or more. A variant is
//! whole or not at all: if its grafts do not all fit in `SLOT` nodes it comes
//! back unchanged as `NodeOversize` — a half-grafted form would be a candidate
//! nobody asked for.
//!
//! A NAMED CONSTANT IN A VARIANT. In a `Flat` it is what it is everywhere else
//! in the crate: a `Var` named "pi", here indexing past the data columns
//! (`vars` = the columns, then the table's names), evaluated through
//! `snap_karva::constant_values`. On the device it is a `Num` whose `konst` is
//! the constant's f32 value and whose `arg0` is 1 + the name's index. No
//! reader of a `Num` looks at `arg0` — not the evaluator (`gpu_eval`: `konst`
//! only), not the linter (`same_subtree`: `arg1`), not `encode`/`decode` — and
//! the relayout carries a leaf's `arg0` through, so the evaluator scores the
//! variant block untouched and the name is still on the node for the
//! write-back. A grafted `Num` has `arg1 = LIT_EXACT` instead of a literal id:
//! its `konst` is its value, exactly (the table refuses a template literal
//! that f32 cannot hold).
//!
//! A literal is matched IN ITS CONTEXT ([`contexts`]: cyclic, algebraic or
//! exponential, from its ancestors), one code per node slot beside the literal
//! ([`slot_contexts`]), so one value at two sites may take two forms; it is
//! still one atom.
//!
//! The device proposes in f32. [`confirmed`] is the host's f64 word on a
//! variant; stage 3 calls it on a variant that passed the guard, before
//! anything is reported or written into a gene. It is never a filter here.

use super::device::{Encoded, SLOT};
use super::engine::LitMode;
use super::flat::{splice, Flat, LNode};
use super::pack::arity;
use super::snap_table::{Context, Hit, SnapTable, INFO_STRIDE, NONE, TEMPLATE_MAX};
use crate::gpu_eval::Op;

pub const SNAP_GRAFT_WGSL: &str = concat!(include_str!("splice.wgsl"), include_str!("snap_graft.wgsl"));

/// Variants per expression: `VARIANTS - 1` single-atom snaps, then all atoms.
pub const VARIANTS: usize = 4;
/// Words per variant in the info block: (length, status, sites, atoms).
pub const VINFO_STRIDE: usize = 4;
/// `arg1` of a grafted `Num` on the device: no literal id, `konst` is exact.
pub const LIT_EXACT: u32 = NONE;
/// The literal the match kernel is handed for a node that is not one.
pub const NOT_A_LITERAL: u32 = 0x7fc0_0000;
/// The work area: a full slot, the largest template, and the `Neg` over it.
const WORK: usize = 80;
const _: () = assert!(SLOT + TEMPLATE_MAX < WORK);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Nothing to graft in this slot: the form is the expression, unchanged.
    NoHit = 0,
    Grafted = 1,
    /// The grafts would not fit in `SLOT` nodes: unchanged, and counted.
    NodeOversize = 2,
    /// The expression itself is empty or longer than `SLOT`: no form.
    Refused = 3,
}

impl Status {
    fn from_code(code: u32) -> Result<Status, String> {
        [Status::NoHit, Status::Grafted, Status::NodeOversize, Status::Refused]
            .get(code as usize)
            .copied()
            .ok_or_else(|| format!("status {code} is not one the graft kernel writes"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// `None` only when `Refused`. `vars` is the columns, then the table's names.
    pub form: Option<Flat>,
    pub status: Status,
    /// Literal sites replaced.
    pub sites: u32,
    /// Per node: a `Num` that came out of a template (exact in f32).
    pub from_template: Vec<bool>,
}

/// One expression's share of a batch.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapped {
    /// Per node of the expression; empty when it was refused.
    pub hits: Vec<Option<Hit>>,
    /// `VARIANTS` of them.
    pub variants: Vec<Variant>,
}

/// A node's context from its parent's: the NEAREST enclosing Sin / Cos / Tan
/// or Exp / ProtectedExp / Log / ProtectedLog decides — `sin(exp(3.14 x))` is
/// exponential, `exp(sin(3.14 x))` cyclic — and every other operator (the
/// inverse trig functions and tanh among them) passes its own context down.
fn below(op: u32, own: Context) -> Context {
    const CYCLIC: [Op; 3] = [Op::Sin, Op::Cos, Op::Tan];
    const EXPONENTIAL: [Op; 4] = [Op::Exp, Op::ProtectedExp, Op::Log, Op::ProtectedLog];
    if CYCLIC.iter().any(|o| *o as u32 == op) {
        Context::Cyclic
    } else if EXPONENTIAL.iter().any(|o| *o as u32 == op) {
        Context::Exponential
    } else {
        own
    }
}

/// One forward pass over `(op, arg0, arg1)` nodes: a child's index is greater
/// than its parent's, so a node's context is settled before it is reached.
fn contexts_of(nodes: impl ExactSizeIterator<Item = (u32, u32, u32)>) -> Vec<Context> {
    let mut out = vec![Context::Algebraic; nodes.len()];
    for (i, (op, arg0, arg1)) in nodes.enumerate() {
        let passed = below(op, out[i]);
        for child in [arg0, arg1].iter().take(arity(op)) {
            if let Some(slot) = out.get_mut(*child as usize).filter(|_| *child as usize > i) {
                *slot = passed;
            }
        }
    }
    out
}

/// The context of every node of an expression (the root is algebraic).
pub fn contexts(expr: &Flat) -> Vec<Context> {
    contexts_of(expr.nodes.iter().map(|n| (n.op, n.arg0, n.arg1)))
}

/// Stage 1 over one expression: a hit or `None` per node, each literal matched
/// in its context.
pub fn match_literals(expr: &Flat, table: &SnapTable, rel_tol: f64, mode: LitMode) -> Vec<Option<Hit>> {
    expr.nodes
        .iter()
        .zip(contexts(expr))
        .map(|(n, context)| if n.op == Op::Num as u32 { table.nearest_in(n.lit, context, rel_tol, mode) } else { None })
        .collect()
}

/// The atoms, in order of first node: each is the ascending sites that hold it.
pub fn atoms(expr: &Flat, hits: &[Option<Hit>]) -> Vec<Vec<usize>> {
    let mut out: Vec<(u64, Vec<usize>)> = Vec::new();
    for (i, n) in expr.nodes.iter().enumerate() {
        if n.op != Op::Num as u32 || hits[i].is_none() {
            continue;
        }
        let bits = n.lit.to_bits();
        match out.iter_mut().find(|(b, _)| *b == bits) {
            Some((_, sites)) => sites.push(i),
            None => out.push((bits, vec![i])),
        }
    }
    out.into_iter().map(|(_, sites)| sites).collect()
}

/// The sites variant `v` grafts, ascending; empty when the slot has nothing.
pub fn variant_sites(expr: &Flat, hits: &[Option<Hit>], v: usize) -> Vec<usize> {
    let atoms = atoms(expr, hits);
    if v + 1 < VARIANTS {
        return atoms.get(v).cloned().unwrap_or_default();
    }
    if atoms.len() < 2 {
        return Vec::new();
    }
    let mut all: Vec<usize> = atoms.into_iter().flatten().collect();
    all.sort_unstable();
    all
}

/// The host's f64 word on variant `v`: every literal it replaced snaps, in
/// f64 and in its context, to exactly the entry and sign the graft used. Stage
/// 3 asks this of a variant that passed the R² guard, before it is reported or
/// written back.
pub fn confirmed(expr: &Flat, hits: &[Option<Hit>], v: usize, table: &SnapTable, rel_tol: f64) -> bool {
    let context = contexts(expr);
    variant_sites(expr, hits, v).iter().all(|i| match hits[*i] {
        Some(hit) => table.confirm_f64_in(expr.nodes[*i].lit, context[*i], hit, rel_tol),
        None => false,
    })
}

/// A template `Num` on the work copy, until the last relayout is done. (A
/// leaf's `arg0` is the one word of a node the relayout carries untouched.)
const FROM_TEMPLATE: u32 = u32::MAX;

/// The CPU reference: the `VARIANTS` snapped variants of ONE expression, whose
/// `vars` is the column list. `hits` is per node (`match_literals`).
pub fn snap_graft(expr: &Flat, hits: &[Option<Hit>], table: &SnapTable) -> Vec<Variant> {
    assert_eq!(hits.len(), expr.nodes.len(), "one hit slot per node");
    if expr.nodes.is_empty() || expr.nodes.len() > SLOT {
        return vec![Variant { form: None, status: Status::Refused, sites: 0, from_template: Vec::new() }; VARIANTS];
    }
    assert!(
        expr.nodes.iter().all(|n| n.op != Op::Num as u32 || n.arg0 == 0),
        "a literal's arg0 must be 0: on the device that word names a constant"
    );
    let n_cols = expr.vars.len() as u32;
    let vars: Vec<String> = expr.vars.iter().chain(&table.names).cloned().collect();
    let unchanged = |status: Status| Variant {
        form: Some(Flat { nodes: expr.nodes.clone(), vars: vars.clone() }),
        status,
        sites: 0,
        from_template: vec![false; expr.nodes.len()],
    };
    (0..VARIANTS)
        .map(|v| {
            let sites = variant_sites(expr, hits, v);
            if sites.is_empty() {
                return unchanged(Status::NoHit);
            }
            let grown = sites.iter().fold(expr.nodes.len(), |n, i| {
                let hit = hits[*i].expect("a site has a hit");
                n + table.info[hit.entry as usize * INFO_STRIDE + 1] as usize + usize::from(hit.negative) - 1
            });
            if grown > SLOT {
                return unchanged(Status::NodeOversize);
            }
            // The site's own index + 1 rides in its arg0 through the relayouts.
            let mut nodes = expr.nodes.clone();
            for i in &sites {
                nodes[*i].arg0 = *i as u32 + 1;
            }
            for _ in &sites {
                let pos = nodes
                    .iter()
                    .position(|n| n.op == Op::Num as u32 && n.arg0 != 0 && n.arg0 != FROM_TEMPLATE)
                    .expect("a site is left");
                let hit = hits[nodes[pos].arg0 as usize - 1].expect("a site has a hit");
                nodes = graft(nodes, pos as u32, hit, table, n_cols);
            }
            let from_template: Vec<bool> =
                nodes.iter().map(|n| n.op == Op::Num as u32 && n.arg0 == FROM_TEMPLATE).collect();
            for n in nodes.iter_mut().filter(|n| n.op == Op::Num as u32) {
                n.arg0 = 0;
            }
            Variant {
                form: Some(Flat { nodes, vars: vars.clone() }),
                status: Status::Grafted,
                sites: sites.len() as u32,
                from_template,
            }
        })
        .collect()
}

/// Replace the literal at `pos` with the entry's form, under a `Neg` if the
/// hit is negative: append, then the linter's relayout.
fn graft(mut nodes: Vec<LNode>, pos: u32, hit: Hit, table: &SnapTable, n_cols: u32) -> Vec<LNode> {
    let live = nodes.len();
    let mut first = live as u32;
    if hit.negative {
        nodes.push(LNode { op: Op::Neg as u32, arg0: first + 1, arg1: 0, lit: 0.0 });
        first += 1;
    }
    for t in &table.template(hit.entry).nodes {
        nodes.push(if t.op == Op::Var as u32 {
            LNode { arg0: n_cols + t.arg0, ..*t }
        } else if t.op == Op::Num as u32 {
            LNode { arg0: FROM_TEMPLATE, ..*t }
        } else {
            // Unary: arg1 is 0 in the template and the relayout rewrites it.
            LNode { arg0: first + t.arg0, arg1: first + t.arg1, ..*t }
        });
    }
    splice(nodes, live, pos, live as u32)
}

/// A batch on the CPU, as the device returns it.
pub fn snap_batch(exprs: &[Flat], table: &SnapTable, rel_tol: f64, mode: LitMode) -> Vec<Snapped> {
    exprs
        .iter()
        .map(|f| {
            if f.nodes.is_empty() || f.nodes.len() > SLOT {
                return Snapped { hits: Vec::new(), variants: snap_graft(f, &vec![None; f.nodes.len()], table) };
            }
            let hits = match_literals(f, table, rel_tol, mode);
            let variants = snap_graft(f, &hits, table);
            Snapped { hits, variants }
        })
        .collect()
}

/// The f32 value of every name a template refers to, as the evaluator will see
/// it. A name the crate has no value for, or one f32 cannot hold, is an error.
pub fn name_values(table: &SnapTable) -> Result<Vec<f32>, String> {
    let known = crate::snap_karva::constant_values();
    table
        .names
        .iter()
        .map(|name| {
            let v = *known.get(name).ok_or_else(|| format!("{name}: a template names it, constant_values does not"))?;
            if (v as f32).is_normal() { Ok(v as f32) } else { Err(format!("{name} = {v}: not a normal f32")) }
        })
        .collect()
}

/// The literals the match kernel reads, one per node slot: a `Num`'s `konst`
/// bits — `encode` already carries every literal's VALUE there, as f32 —
/// and `NOT_A_LITERAL` (a NaN, which the match refuses on the bit pattern)
/// everywhere else.
pub fn slot_literals(enc: &Encoded) -> Vec<u32> {
    let mut lits = vec![NOT_A_LITERAL; enc.lengths.len() * SLOT];
    for (e, len) in enc.lengths.iter().enumerate() {
        let at = e * SLOT;
        let nodes = enc.nodes[at * 4..].chunks_exact(4);
        for (lit, node) in lits[at..at + *len as usize].iter_mut().zip(nodes) {
            if node[0] == Op::Num as u32 {
                *lit = node[3];
            }
        }
    }
    lits
}

/// The context code the match kernel reads, one per node slot, beside
/// `slot_literals` (algebraic where there is no node).
pub fn slot_contexts(enc: &Encoded) -> Vec<u32> {
    let mut out = vec![Context::Algebraic as u32; enc.lengths.len() * SLOT];
    for (e, len) in enc.lengths.iter().enumerate() {
        let at = e * SLOT;
        let nodes = enc.nodes[at * 4..(at + *len as usize) * 4].chunks_exact(4).map(|w| (w[0], w[1], w[2]));
        for (slot, context) in out[at..].iter_mut().zip(contexts_of(nodes)) {
            *slot = context as u32;
        }
    }
    out
}

/// The words the device writes for a variant: the CPU reference in the device's
/// own encoding, for a bit-for-bit comparison. `lit_id` is `encode`'s id of an
/// f64 literal.
pub fn device_words(variant: &Variant, n_cols: u32, name_vals: &[f32], lit_id: &dyn Fn(f64) -> u32) -> Vec<u32> {
    let Some(form) = &variant.form else {
        return Vec::new();
    };
    form.nodes
        .iter()
        .zip(&variant.from_template)
        .flat_map(|(n, from_template)| {
            if n.op == Op::Var as u32 && n.arg0 >= n_cols {
                let name = n.arg0 - n_cols;
                [Op::Num as u32, name + 1, LIT_EXACT, name_vals[name as usize].to_bits()]
            } else if n.op == Op::Num as u32 {
                let id = if *from_template { LIT_EXACT } else { lit_id(n.lit) };
                [n.op, 0, id, (n.lit as f32).to_bits()]
            } else {
                [n.op, n.arg0, n.arg1, 0]
            }
        })
        .collect()
}

/// Decode the device's three blocks. An original literal is its f64 again
/// (through its id); a grafted one is its `konst`, which is exact; a named
/// constant is a `Var` past the columns.
pub fn decode(
    hit_words: &[u32],
    nodes_out: &[u32],
    vinfo: &[u32],
    enc: &Encoded,
    cols: &[String],
    names: &[String],
) -> Result<Vec<Snapped>, String> {
    let vars: Vec<String> = cols.iter().chain(names).cloned().collect();
    let mut out = Vec::with_capacity(enc.lengths.len());
    for (e, len) in enc.lengths.iter().enumerate() {
        let hits: Vec<Option<Hit>> = (e * SLOT..e * SLOT + *len as usize)
            .map(|i| (hit_words[i * 2] != NONE).then_some(Hit { entry: hit_words[i * 2], negative: hit_words[i * 2 + 1] != 0 }))
            .collect();
        let mut variants = Vec::with_capacity(VARIANTS);
        for t in e * VARIANTS..(e + 1) * VARIANTS {
            let info = &vinfo[t * VINFO_STRIDE..(t + 1) * VINFO_STRIDE];
            let status = Status::from_code(info[1])?;
            if status == Status::Refused {
                variants.push(Variant { form: None, status, sites: 0, from_template: Vec::new() });
                continue;
            }
            let mut nodes = Vec::with_capacity(info[0] as usize);
            let mut from_template = Vec::with_capacity(info[0] as usize);
            for k in 0..info[0] as usize {
                let w = &nodes_out[(t * SLOT + k) * 4..(t * SLOT + k) * 4 + 4];
                let is_num = w[0] == Op::Num as u32;
                from_template.push(is_num && w[1] == 0 && w[2] == LIT_EXACT);
                nodes.push(if is_num && w[1] != 0 {
                    LNode { op: Op::Var as u32, arg0: cols.len() as u32 + w[1] - 1, arg1: 0, lit: 0.0 }
                } else if is_num && w[2] == LIT_EXACT {
                    LNode { op: w[0], arg0: 0, arg1: 0, lit: f64::from(f32::from_bits(w[3])) }
                } else if is_num {
                    LNode { op: w[0], arg0: 0, arg1: 0, lit: enc.literals[w[2] as usize] }
                } else {
                    LNode { op: w[0], arg0: w[1], arg1: w[2], lit: 0.0 }
                });
            }
            variants.push(Variant { form: Some(Flat { nodes, vars: vars.clone() }), status, sites: info[2], from_template });
        }
        out.push(Snapped { hits, variants });
    }
    Ok(out)
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use crate::lint::device::{encode, KernelTables};
    use crate::lint::snap_table::SnapKernel;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    const MAX_GROUPS_PER_DIM: u32 = crate::gpu_eval::MAX_GROUPS_PER_DIM;

    /// What a dispatch left behind, raw: the words `decode` reads.
    pub struct GraftWords {
        pub enc: Encoded,
        pub hits: Vec<u32>,
        pub nodes: Vec<u32>,
        pub vinfo: Vec<u32>,
    }

    /// The graft kernel, on a `SnapKernel`'s device: it binds that kernel's
    /// resident template and info blocks.
    pub struct SnapGraft {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        names_buf: wgpu::Buffer,
    }

    impl SnapGraft {
        pub fn new(kernel: &SnapKernel) -> Result<Self, String> {
            if kernel.table().len() >= 1 << 24 {
                return Err("more than 2^24 entries: the entry id rides in 24 bits".to_string());
            }
            let name_bits: Vec<u32> = name_values(kernel.table())?.iter().map(|v| v.to_bits()).collect();
            let device = kernel.device();
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fuller-snap-graft"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SNAP_GRAFT_WGSL)),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fuller-snap-graft-layout"),
                entries: &(0..9)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if i == 8 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage { read_only: i < 6 }
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
                label: Some("fuller-snap-graft-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "graft_main",
                compilation_options: Default::default(),
            });
            // wgpu rejects a zero-sized binding.
            let padded: Vec<u32> = if name_bits.is_empty() { vec![0] } else { name_bits };
            let names_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("snap-name-values"),
                contents: bytemuck::cast_slice(&padded),
                usage: wgpu::BufferUsages::STORAGE,
            });
            Ok(Self { pipeline, layout, names_buf })
        }

        /// Record the graft on `enc`, after the match pass that fills
        /// `hits_buf`. `nodes_out` takes `n_expr * VARIANTS * SLOT` nodes in
        /// the evaluator's layout, `vinfo_out` `VINFO_STRIDE` words per
        /// variant; both stay on the device for whoever scores them. The
        /// uniform buffer returned is the caller's to destroy after the submit.
        pub fn graft_pass(
            &self,
            kernel: &SnapKernel,
            enc: &mut wgpu::CommandEncoder,
            buffers: [&wgpu::Buffer; 5],
            n_expr: u32,
            max_groups: u32,
        ) -> wgpu::Buffer {
            assert!(n_expr > 0 && (1..=MAX_GROUPS_PER_DIM).contains(&max_groups));
            let [nodes_in, len_in, hits_buf, nodes_out, vinfo_out] = buffers;
            let device = kernel.device();
            let groups = (n_expr * VARIANTS as u32).div_ceil(64);
            let groups_x = groups.min(max_groups);
            let groups_y = groups.div_ceil(groups_x);
            let cfg = [n_expr, groups_x * 64, 0, 0];
            let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("snap-graft-cfg"),
                contents: bytemuck::cast_slice(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bound = [
                nodes_in, len_in, hits_buf, kernel.info_buffer(), kernel.templates_buffer(), &self.names_buf,
                nodes_out, vinfo_out, &cfg_buf,
            ];
            let entries: Vec<wgpu::BindGroupEntry> = bound
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &entries,
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
            cfg_buf
        }

        /// Match and graft a batch in ONE command buffer: the match pass writes
        /// the hits the graft pass reads, and nothing crosses to the host in
        /// between. `exprs` share the column list `cols`.
        pub fn run(
            &self,
            kernel: &SnapKernel,
            exprs: &[Flat],
            cols: &[String],
            tables: &KernelTables,
            rel_tol: f64,
        ) -> Result<Vec<Snapped>, String> {
            let words = self.run_words(kernel, exprs, tables, rel_tol, MAX_GROUPS_PER_DIM)?;
            decode(&words.hits, &words.nodes, &words.vinfo, &words.enc, cols, &kernel.table().names)
        }

        /// [`SnapGraft::run`] before the decode, with the dispatch row width
        /// exposed. At the full width the match pass (a thread per node slot)
        /// wraps into a second row past 65,535 expressions, the graft pass (a
        /// thread per variant) only past 1,048,560 — 5 GiB of node blocks,
        /// more than a test should allocate; a narrow row wraps both.
        pub fn run_words(
            &self,
            kernel: &SnapKernel,
            exprs: &[Flat],
            tables: &KernelTables,
            rel_tol: f64,
            max_groups: u32,
        ) -> Result<GraftWords, String> {
            if let Some(f) = exprs.iter().find(|f| f.nodes.iter().any(|n| n.op == Op::Num as u32 && n.arg0 != 0)) {
                return Err(format!("a literal's arg0 must be 0 (it names a constant on the device): {:?}", f.nodes));
            }
            let enc = encode(exprs, tables);
            if exprs.is_empty() {
                return Ok(GraftWords { enc, hits: Vec::new(), nodes: Vec::new(), vinfo: Vec::new() });
            }
            let n_expr = u32::try_from(exprs.len()).map_err(|_| "more than u32::MAX expressions".to_string())?;
            let n_lit = n_expr
                .checked_mul(SLOT as u32)
                .filter(|n| n.checked_mul(8).is_some())
                .ok_or("the batch's node slots do not index in a u32")?;
            let device = kernel.device();
            let storage = |label: &str, words: &[u32]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(words),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let nodes_buf = storage("graft-nodes-in", &enc.nodes);
            let len_buf = storage("graft-len-in", &enc.lengths);
            let lits_buf = storage("graft-literals", &slot_literals(&enc));
            let contexts_buf = storage("graft-contexts", &slot_contexts(&enc));
            let hit_bytes = u64::from(n_lit) * 8;
            let out_bytes = (exprs.len() * VARIANTS * SLOT * 16) as u64;
            let vinfo_bytes = (exprs.len() * VARIANTS * VINFO_STRIDE * 4) as u64;
            let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
            };
            let out_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let read_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
            let hits_buf = make("graft-hits", hit_bytes, out_usage);
            let nodes_out = make("graft-nodes-out", out_bytes, out_usage);
            let vinfo_out = make("graft-vinfo-out", vinfo_bytes, out_usage);
            let hits_read = make("graft-hits-read", hit_bytes, read_usage);
            let nodes_read = make("graft-nodes-read", out_bytes, read_usage);
            let vinfo_read = make("graft-vinfo-read", vinfo_bytes, read_usage);

            let mut cmd = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let match_cfg = kernel.match_pass(&mut cmd, [&lits_buf, &contexts_buf, &hits_buf], n_lit, rel_tol, max_groups);
            let graft_cfg =
                self.graft_pass(kernel, &mut cmd, [&nodes_buf, &len_buf, &hits_buf, &nodes_out, &vinfo_out], n_expr, max_groups);
            cmd.copy_buffer_to_buffer(&hits_buf, 0, &hits_read, 0, hit_bytes);
            cmd.copy_buffer_to_buffer(&nodes_out, 0, &nodes_read, 0, out_bytes);
            cmd.copy_buffer_to_buffer(&vinfo_out, 0, &vinfo_read, 0, vinfo_bytes);
            kernel.queue().submit(Some(cmd.finish()));

            let read = |buf: &wgpu::Buffer| -> Result<Vec<u32>, String> {
                let slice = buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                device.poll(wgpu::Maintain::Wait);
                rx.recv()
                    .map_err(|e| format!("map_async channel: {e}"))?
                    .map_err(|e| format!("map_async: {e}"))?;
                let words = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
                buf.unmap();
                Ok(words)
            };
            let hits = read(&hits_read)?;
            let nodes = read(&nodes_read)?;
            let vinfo = read(&vinfo_read)?;

            // Release per-dispatch buffers explicitly and drain wgpu's deferred
            // queue — see gpu_eval::GpuEvaluator::eval for what happens otherwise.
            for b in [
                &nodes_buf, &len_buf, &lits_buf, &contexts_buf, &hits_buf, &nodes_out, &vinfo_out, &hits_read,
                &nodes_read, &vinfo_read, &match_cfg, &graft_cfg,
            ] {
                b.destroy();
            }
            device.poll(wgpu::Maintain::Poll);
            Ok(GraftWords { enc, hits, nodes, vinfo })
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::{GraftWords, SnapGraft};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolve::{below, coin, draw};
    use crate::lint::flat::is_canonical;
    use crate::lint::node::Tree;
    use crate::snap_karva::snap_variants;

    const TOL: f64 = 1e-3;
    const COLS: [&str; 3] = ["x", "y", "z"];

    fn table() -> SnapTable {
        SnapTable::standard().expect("the lattice builds")
    }

    fn cols() -> Vec<String> {
        COLS.iter().map(|s| s.to_string()).collect()
    }

    fn flat(math: &str) -> Flat {
        Flat::from_tree_in(&Tree::parse(math).unwrap(), &cols()).unwrap()
    }

    fn snapped(math: &str, t: &SnapTable) -> (Flat, Snapped) {
        let f = flat(math);
        let s = snap_batch(std::slice::from_ref(&f), t, TOL, LitMode::F64).remove(0);
        (f, s)
    }

    fn math_of(v: &Variant) -> String {
        v.form.as_ref().expect("a form").to_tree().to_math()
    }

    /// An entry's own form, as the crate prints it.
    fn form_of(t: &SnapTable, entry: u32) -> String {
        Tree::parse(&t.maths[entry as usize]).unwrap().to_math()
    }

    /// Probe rows: three columns in 0.5 .. 2.5.
    fn probe_rows() -> Vec<Vec<(String, f64)>> {
        (0..4u32)
            .map(|row| {
                COLS.iter()
                    .enumerate()
                    .map(|(c, name)| {
                        let u = (draw(23, 0, row, c as u32, 0) >> 11) as f64 / (1u64 << 53) as f64;
                        (name.to_string(), 0.5 + 2.0 * u)
                    })
                    .collect()
            })
            .collect()
    }

    fn values(tree: &Tree) -> Vec<f64> {
        probe_rows().iter().map(|row| tree.eval(row).expect("evaluates")).collect()
    }

    fn close(a: &[f64], b: &[f64], rel: f64) -> bool {
        a.iter().zip(b).all(|(p, q)| (!p.is_finite() && !q.is_finite()) || (p - q).abs() <= rel * q.abs().max(1e-300))
    }

    /// Root at 0, children after their parent, contiguous, nothing dead: the
    /// check the lint splice tests use, on every form a batch returns.
    fn assert_layout(batch: &[Snapped]) {
        for s in batch {
            assert_eq!(s.variants.len(), VARIANTS);
            for v in &s.variants {
                match &v.form {
                    Some(f) => {
                        assert!(is_canonical(f), "not canonical: {:?}", f.nodes);
                        assert!(!f.nodes.is_empty() && f.nodes.len() <= SLOT);
                        assert_eq!(f.nodes.len(), v.from_template.len());
                        assert_eq!(v.status == Status::Grafted, v.sites > 0);
                    }
                    None => assert_eq!(v.status, Status::Refused),
                }
            }
        }
    }

    #[test]
    fn a_coefficient_becomes_its_constant_form() {
        let t = table();
        let (f, s) = snapped(r#"(Mul (Num 0.0796) (Var "x"))"#, &t);
        let hit = s.hits[1].expect("0.0796 matches");
        assert_eq!(t.labels[hit.entry as usize], "1/(4*pi)");
        let v = &s.variants[0];
        assert_eq!((v.status, v.sites), (Status::Grafted, 1));
        assert_eq!(math_of(v), format!(r#"(Mul {} (Var "x"))"#, form_of(&t, hit.entry)));
        assert!(math_of(v).contains(r#"(Var "pi")"#), "{}", math_of(v));
        let (before, after) = (values(&f.to_tree()), values(&v.form.as_ref().unwrap().to_tree()));
        assert!(close(&after, &before, TOL), "{after:?} vs {before:?}");
        assert!(!close(&after, &before, 1e-9), "the snap moved nothing");
        assert!(confirmed(&f, &s.hits, 0, &t, TOL));
        // One atom: no other single, and no "all atoms" beside it.
        for other in &s.variants[1..] {
            assert_eq!((other.status, other.sites), (Status::NoHit, 0));
            assert_eq!(other.form.as_ref().unwrap().nodes, f.nodes);
        }
        assert_layout(std::slice::from_ref(&s));
    }

    /// The one literal of `math`: its context, and the label it snaps to.
    fn the_literal(math: &str, t: &SnapTable) -> (Context, Option<String>) {
        let (f, s) = snapped(math, t);
        let at = f.nodes.iter().position(|n| n.op == Op::Num as u32).expect("a literal");
        (contexts(&f)[at], s.hits[at].map(|h| t.labels[h.entry as usize].clone()))
    }

    #[test]
    fn a_literal_is_matched_in_its_context() {
        let t = table();
        let some = |l: &str| Some(l.to_string());
        assert_eq!(the_literal(r#"(Sin (Mul (Num 3.14159) (Var "x")))"#, &t), (Context::Cyclic, some("pi")));
        assert_eq!(the_literal(r#"(Sin (Add (Var "x") (Num 1.5708)))"#, &t), (Context::Cyclic, some("pi/2")));
        assert_eq!(the_literal(r#"(Cos (Mul (Mul (Num 6.2832) (Var "x")) (Var "y")))"#, &t), (Context::Cyclic, some("2*pi")));
        // Algebraic: no rational is within 1e-3 of 3.14159, and pi is one node.
        let polynomial = the_literal(r#"(Add (Mul (Num 3.14159) (Pow2 (Var "x"))) (Var "x"))"#, &t);
        assert_eq!(polynomial, (Context::Algebraic, some("pi")));
        // The NEAREST enclosing of the two kinds decides; other operators pass
        // their own context down.
        assert_eq!(the_literal(r#"(Exp (Sin (Mul (Num 3.14159) (Var "x"))))"#, &t).0, Context::Cyclic);
        assert_eq!(the_literal(r#"(Sin (Exp (Mul (Num 3.14159) (Var "x"))))"#, &t).0, Context::Exponential);
        assert_eq!(the_literal(r#"(ProtectedLog (Tanh (Div (Var "x") (Num 3.14159))))"#, &t).0, Context::Exponential);
        assert_eq!(the_literal(r#"(Tanh (Asin (Mul (Num 3.14159) (Var "x"))))"#, &t).0, Context::Algebraic);
        assert_eq!(the_literal("(Num 3.14159)", &t).0, Context::Algebraic);

        // A literal between 2/3 and pi/(e*sqrt3), nearer the pi form: the pi
        // form inside a sine, the rational in a polynomial, and under an
        // exponential — every family at home — whatever nearness and size say.
        let v = 2.0 / 3.0 * (1.0 + 0.63e-3);
        let inside = |op: &str| the_literal(&format!(r#"({op} (Mul (Num {v}) (Var "x")))"#), &t).1;
        assert_eq!(inside("Sin"), some("pi/(e*sqrt3)"));
        assert_eq!(inside("Neg"), some("2/3"));
        let energy = |label: &str| {
            let entry = t.labels.iter().position(|l| l == label).unwrap() as u32;
            let x3 = t.objectives(v, entry, Context::Exponential, TOL);
            assert_eq!(x3[2], 0.0, "{label}: at home under an exponential");
            x3[0] * x3[0] + x3[1] * x3[1]
        };
        let under_exp = inside("Exp").expect("a form");
        eprintln!("{v} under exp -> {under_exp}: energy 2/3 {:.4}, pi/(e*sqrt3) {:.4}", energy("2/3"), energy("pi/(e*sqrt3)"));
        assert_eq!(under_exp, if energy("2/3") < energy("pi/(e*sqrt3)") { "2/3" } else { "pi/(e*sqrt3)" });
        // One value at two sites in two contexts is one atom, and each site
        // takes its own context's form.
        let (f, s) = snapped(&format!(r#"(Add (Mul (Num {v}) (Var "x")) (Sin (Num {v})))"#), &t);
        assert_eq!(atoms(&f, &s.hits).len(), 1);
        let labels: Vec<&str> = s.hits.iter().flatten().map(|h| t.labels[h.entry as usize].as_str()).collect();
        assert_eq!(labels, ["2/3", "pi/(e*sqrt3)"], "level order: the product's literal, then the sine's");
        assert_eq!((s.variants[0].status, s.variants[0].sites), (Status::Grafted, 2));
        assert!(confirmed(&f, &s.hits, 0, &t, TOL));
    }

    #[test]
    fn a_negative_literal_grafts_the_form_under_a_neg() {
        let t = table();
        let (f, s) = snapped(r#"(Add (Var "x") (Num -3.1416))"#, &t);
        // pi as a gene prints it (text: a number typed as 3.1416 is rounded on purpose).
        let rounded: f64 = "3.1416".parse().expect("a number");
        assert_eq!(s.hits[2], t.nearest(rounded, TOL, LitMode::F64).map(|h| Hit { negative: true, ..h }));
        assert_eq!(math_of(&s.variants[0]), r#"(Add (Var "x") (Neg (Var "pi")))"#);
        assert!(close(&values(&s.variants[0].form.as_ref().unwrap().to_tree()), &values(&f.to_tree()), TOL));
        // At the root too.
        let (_, s) = snapped("(Num -3.1416)", &t);
        assert_eq!(math_of(&s.variants[0]), r#"(Neg (Var "pi"))"#);
        assert_layout(std::slice::from_ref(&s));
    }

    #[test]
    fn two_atoms_give_two_singles_and_the_pair() {
        let t = table();
        let (_, s) = snapped(r#"(Add (Mul (Num 3.1416) (Var "x")) (Num 0.0796))"#, &t);
        // Level order: Add, Mul, 0.0796, 3.1416, x — so 0.0796 is atom 0.
        let quarter = form_of(&t, s.hits[2].unwrap().entry);
        assert_eq!(math_of(&s.variants[0]), format!(r#"(Add (Mul (Num 3.1416) (Var "x")) {quarter})"#));
        assert_eq!(math_of(&s.variants[1]), r#"(Add (Mul (Var "pi") (Var "x")) (Num 0.0796))"#);
        assert_eq!(s.variants[2].status, Status::NoHit);
        assert_eq!(math_of(&s.variants[3]), format!(r#"(Add (Mul (Var "pi") (Var "x")) {quarter})"#));
        assert_eq!(s.variants.iter().map(|v| v.sites).collect::<Vec<_>>(), [1, 1, 0, 2]);
        assert_layout(std::slice::from_ref(&s));

        // The same value at two sites is ONE atom, replaced at both.
        let (_, s) = snapped(r#"(Add (Mul (Num 3.1416) (Var "x")) (Sin (Num 3.1416)))"#, &t);
        assert_eq!(math_of(&s.variants[0]), r#"(Add (Mul (Var "pi") (Var "x")) (Sin (Var "pi")))"#);
        assert_eq!(s.variants.iter().map(|v| (v.status, v.sites)).collect::<Vec<_>>(), [
            (Status::Grafted, 2),
            (Status::NoHit, 0),
            (Status::NoHit, 0),
            (Status::NoHit, 0)
        ]);

        // Five atoms: the first three have a single, the last slot takes all five.
        let five = r#"(Add (Add (Num 3.1416) (Num 6.2832)) (Add (Add (Num 0.0796) (Num 2.7183)) (Add (Num 1.4142) (Var "x"))))"#;
        let (f, s) = snapped(five, &t);
        assert_eq!(atoms(&f, &s.hits).len(), 5);
        assert_eq!(s.variants.iter().map(|v| v.sites).collect::<Vec<_>>(), [1, 1, 1, 5]);
        assert!(!math_of(&s.variants[3]).contains("(Num 3.1416)") && !math_of(&s.variants[3]).contains("(Num 1.4142)"));
        assert_layout(std::slice::from_ref(&s));
    }

    /// A 7-node entry between 1 and 10 that its own value snaps to (no shorter
    /// form in its band takes the literal from it).
    fn seven_nodes(t: &SnapTable) -> usize {
        (0..t.len())
            .find(|i| {
                t.info[i * INFO_STRIDE + 1] == 7
                    && (1.0..10.0).contains(&t.values[*i])
                    && t.nearest(t.values[*i], TOL, LitMode::F64).map(|h| h.entry as usize) == Some(*i)
            })
            .expect("a 7-node entry that wins its own band")
    }

    /// `Neg` over a chain of `adds` Adds ending in the literal: 2 * adds + 2 nodes.
    fn chain(adds: usize, literal: f64) -> String {
        let mut s = format!("(Num {literal})");
        for _ in 0..adds {
            s = format!(r#"(Add (Var "x") {s})"#);
        }
        format!("(Neg {s})")
    }

    #[test]
    fn a_graft_past_the_slot_is_skipped_and_counted() {
        let t = table();
        // A 7-node template: 58 - 1 + 7 = 64 fits, 60 - 1 + 7 = 66 does not.
        let seven = seven_nodes(&t);
        let literal = t.values[seven];
        let (f, s) = snapped(&chain(28, literal), &t);
        assert_eq!(f.nodes.len(), 58);
        assert_eq!(s.variants[0].status, Status::Grafted);
        assert_eq!(s.variants[0].form.as_ref().unwrap().nodes.len(), SLOT);
        let (f, s) = snapped(&chain(29, literal), &t);
        assert_eq!(f.nodes.len(), 60);
        assert_eq!((s.variants[0].status, s.variants[0].sites), (Status::NodeOversize, 0));
        assert_eq!(s.variants[0].form.as_ref().unwrap().nodes, f.nodes);
        assert_eq!(s.variants[0].from_template, vec![false; 60]);
        assert_layout(std::slice::from_ref(&s));
        // Longer than the slot: no form at all.
        let (f, s) = snapped(&chain(31, literal), &t);
        assert_eq!(f.nodes.len(), 64);
        assert_eq!(s.variants[0].status, Status::NodeOversize);
        let (_, s) = snapped(&chain(32, literal), &t);
        assert!(s.hits.is_empty());
        assert!(s.variants.iter().all(|v| v.status == Status::Refused && v.form.is_none()));
    }

    #[test]
    fn nothing_to_snap_comes_back_identical() {
        let t = table();
        let gap = t.values.windows(2).find(|w| w[0] > 1.0 && w[1] / w[0] > 1.01).expect("a gap");
        let lonely = (gap[0] * gap[1]).sqrt();
        for math in [r#"(Mul (Sin (Var "x")) (Var "y"))"#.to_string(), format!(r#"(Mul (Num {lonely}) (Var "x"))"#)] {
            let (f, s) = snapped(&math, &t);
            assert!(s.hits.iter().all(Option::is_none));
            for v in &s.variants {
                assert_eq!((v.status, v.sites), (Status::NoHit, 0));
                assert_eq!(v.form.as_ref().unwrap().nodes, f.nodes);
                assert_eq!(math_of(v), math);
            }
        }
    }

    /// A deterministic expression: depth <= 5 (at most 63 nodes), literals mostly within 0.8 of
    /// the tolerance of an entry between 0.01 and 100, rounded through f32 —
    /// what a gene's constant is.
    fn grow(tame: &[f64], row: u32, slot: &mut u32, depth: u32) -> String {
        let mut next = || {
            *slot += 1;
            draw(11, 0, row, *slot, 0)
        };
        if depth == 0 || (depth < 5 && below(next(), 6) == 0) {
            if coin(next()) {
                return format!(r#"(Var "{}")"#, COLS[below(next(), 3) as usize]);
            }
            let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
            let magnitude = if below(next(), 5) == 0 {
                0.1 + 9.9 * u
            } else {
                tame[below(next(), tame.len() as u32) as usize] * (1.0 + (u - 0.5) * 1.6 * TOL)
            };
            let v = if below(next(), 4) == 0 { -magnitude } else { magnitude };
            return Tree::Num(f64::from(v as f32)).to_math();
        }
        let op = ["Add", "Sub", "Mul", "Add", "Sub", "Mul", "Sin", "Cos", "Neg", "Pow2", "Tanh"][below(next(), 11) as usize];
        let a = grow(tame, row, slot, depth - 1);
        if ["Add", "Sub", "Mul"].contains(&op) {
            let b = grow(tame, row, slot, depth - 1);
            format!("({op} {a} {b})")
        } else {
            format!("({op} {a})")
        }
    }

    fn corpus(t: &SnapTable, n: u32) -> Vec<Flat> {
        let tame: Vec<f64> = t.values.iter().copied().filter(|v| (0.01..100.0).contains(v)).collect();
        (0..n).map(|row| flat(&grow(&tame, row, &mut 0, 5))).collect()
    }

    #[test]
    fn every_output_is_a_k_expression_and_two_runs_are_identical() {
        let t = table();
        let exprs = corpus(&t, 200);
        for mode in [LitMode::F64, LitMode::F32] {
            let batch = snap_batch(&exprs, &t, TOL, mode);
            assert_layout(&batch);
            assert_eq!(batch, snap_batch(&exprs, &t, TOL, mode));
            let mut counts = [0usize; 4];
            batch.iter().flat_map(|s| &s.variants).for_each(|v| counts[v.status as usize] += 1);
            eprintln!("snap graft {mode:?}, 200 expressions: no_hit {}, grafted {}, node_oversize {}, refused {}", counts[0], counts[1], counts[2], counts[3]);
            assert!(counts[1] > 200 && counts[2] > 0, "{counts:?}");
        }
    }

    /// The values `snap_variants` proposes for an input, per candidate.
    fn proposed(math: &str) -> Vec<Vec<f64>> {
        snap_variants(math, 64, TOL).unwrap().iter().map(|c| values(&Tree::parse(&c.expr).unwrap())).collect()
    }

    /// Every variant the graft produces against `snap_karva::snap_variants`'
    /// candidates for the same input, by value on the probe rows to 1e-12.
    #[test]
    fn variants_are_among_snap_variants_candidates() {
        let t = table();
        let exprs = corpus(&t, 200);
        let lattice = crate::snap_karva::lattice();
        let (mut agree, mut differ_entry, mut differ_extra) = (0usize, [0usize; 5], 0usize);
        let mut examples: [Vec<String>; 5] = Default::default();
        for f in &exprs {
            let math = f.to_tree().to_math();
            let hits = match_literals(f, &t, TOL, LitMode::F64);
            let theirs = proposed(&math);
            for (v, variant) in snap_graft(f, &hits, &t).iter().enumerate() {
                if variant.status != Status::Grafted {
                    continue;
                }
                let mine = values(&variant.form.as_ref().unwrap().to_tree());
                if theirs.iter().any(|c| close(c, &mine, 1e-12)) {
                    agree += 1;
                    continue;
                }
                // Why not. (1) For one of this variant's literals, snap_variants
                // does not propose the table's entry: asked about the literal
                // alone, no candidate has the entry's value.
                let sites = variant_sites(f, &hits, v);
                let other_entry = sites.iter().find(|i| {
                    let lit = f.nodes[**i].lit;
                    let value = t.signed_value(hits[**i].unwrap());
                    !proposed(&Tree::Num(lit).to_math()).iter().any(|c| close(c, &[value; 4], 1e-12))
                });
                // (2) It is the all-atoms slot, and snap_variants snapped a
                // literal the table has no hit for.
                let extra = f.nodes.iter().zip(&hits).find(|(n, h)| {
                    n.op == Op::Num as u32 && h.is_none() && proposed(&Tree::Num(n.lit).to_math()).len() > 1
                });
                if let Some(i) = other_entry {
                    let lit = f.nodes[*i].lit;
                    let hit = hits[*i].unwrap();
                    // snap_variants takes the SHORTEST `math` text within
                    // tolerance that its e-graph confirms, whatever the
                    // context; the table takes the smallest HFF angle over
                    // (nearness, size, fit to the context). And it tries an
                    // entry as itself only: a negative literal snaps only where
                    // the lattice lists the negative form.
                    let theirs = proposed(&Tree::Num(lit).to_math());
                    let their_entry = theirs.get(1).and_then(|c| lattice.iter().find(|e| close(&[e.value; 4], c, 1e-12)));
                    let their_family = their_entry
                        .and_then(|e| t.values.iter().position(|x| *x == e.value.abs()))
                        .map(|at| t.family(at as u32));
                    let within = lattice.iter().any(|e| ((lit - e.value) / e.value).abs() < TOL);
                    let cause = match (their_entry, their_family, within) {
                        (Some(_), Some(crate::lint::snap_table::Family::Physical), _) => 0,
                        (Some(_), Some(_), _) => 1,
                        (Some(_), None, _) => 2,
                        (None, _, false) => 3,
                        (None, _, true) => 4,
                    };
                    differ_entry[cause] += 1;
                    assert!(cause != 3 || lit < 0.0, "{lit}: the lattice has nothing within tolerance, the table has");
                    if examples[cause].len() < 3 {
                        examples[cause].push(format!(
                            "{lit} in {:?}: table -> {} ({}), snap_variants -> {}",
                            contexts(f)[*i],
                            t.labels[hit.entry as usize],
                            t.signed_value(hit),
                            their_entry.map(|e| format!("{} ({})", e.label, e.value)).unwrap_or("none".to_string())
                        ));
                    }
                } else if v == VARIANTS - 1 && extra.is_some() {
                    differ_extra += 1;
                } else {
                    panic!("variant {v} of {math} is not among snap_variants' candidates, and nothing explains it");
                }
            }
        }
        eprintln!(
            "snap graft vs snap_variants, 200 expressions: {agree} variants agree; for one of the literals \
             snap_variants proposes ANOTHER entry — {} a PHYSICAL form (the affinity table puts those last), {} a \
             non-physical form (shortest text against smallest angle: size, nearness and context trade off), {} a \
             form the device table does not hold (out of f32 range, or folded onto a twin); {} differ because the \
             literal is negative and the lattice lists no negative form within tolerance (snap_variants never \
             negates an entry); {} because snap_variants proposes nothing although the lattice has a form within \
             tolerance (its e-graph check refused); {differ_extra} all-atom variants differ because snap_variants \
             snapped a literal the table did not",
            differ_entry[0], differ_entry[1], differ_entry[2], differ_entry[3], differ_entry[4]
        );
        for e in examples.iter().flatten() {
            eprintln!("  {e}");
        }
        assert!(agree > 0);
    }

    /// The kernel's layout constants are the host's.
    #[test]
    fn wgsl_constants_match_the_host() {
        for needle in [
            format!("const VARIANTS: u32 = {VARIANTS}u;"),
            format!("const INFO_STRIDE: u32 = {INFO_STRIDE}u;"),
            format!("const SLOT: u32 = {SLOT}u;"),
            format!("const WORK: u32 = {WORK}u;"),
            format!("const OP_NEG: u32 = {}u;", Op::Neg as u32),
            format!("const NO_HIT: u32 = {}u;", Status::NoHit as u32),
            format!("const GRAFTED: u32 = {}u;", Status::Grafted as u32),
            format!("const NODE_OVERSIZE: u32 = {}u;", Status::NodeOversize as u32),
            format!("const REFUSED: u32 = {}u;", Status::Refused as u32),
            "const NONE: u32 = 0xffffffffu;".to_string(),
        ] {
            assert!(SNAP_GRAFT_WGSL.contains(&needle), "snap_graft.wgsl lacks `{needle}`");
        }
        assert_eq!(f32::from_bits(NOT_A_LITERAL).to_bits() & 0x7f80_0000, 0x7f80_0000, "a NaN: the match refuses it");
    }

    #[cfg(feature = "gpu")]
    mod device {
        use super::*;
        use crate::lint::device::KernelTables;
        use crate::lint::pack::pack;
        use crate::lint::snap_table::SnapKernel;
        use crate::lint::tables::{Rule, Tables};
        use std::collections::BTreeMap;

        fn kernel_tables() -> KernelTables {
            let tables = Tables::standard().unwrap();
            let rules: Vec<&Rule> = tables.rules.iter().collect();
            let p = pack(&rules);
            KernelTables::new(&p.rules, p.codes, &tables.guards).unwrap()
        }

        /// Device against the F32 reference: the decoded batch, and the raw
        /// node words of every variant.
        fn assert_parity(graft: &SnapGraft, kernel: &SnapKernel, kt: &KernelTables, exprs: &[Flat], max_groups: u32, name: &str) {
            let t = kernel.table();
            let start = std::time::Instant::now();
            let words = graft.run_words(kernel, exprs, kt, TOL, max_groups).unwrap();
            let device = start.elapsed();
            let got = decode(&words.hits, &words.nodes, &words.vinfo, &words.enc, &cols(), &t.names).unwrap();
            let start = std::time::Instant::now();
            let want = snap_batch(exprs, t, TOL, LitMode::F32);
            let cpu = start.elapsed();
            let grafted = want.iter().flat_map(|s| &s.variants).filter(|v| v.status == Status::Grafted).count();
            eprintln!(
                "device snap graft, {name}: {} expressions x {VARIANTS} variants, {grafted} grafted; match + graft + readback {device:?}, CPU F32 reference {cpu:?}",
                exprs.len()
            );
            assert_eq!(got.len(), want.len(), "{name}");
            for (e, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{name}: expression {e}");
            }
            let ids: BTreeMap<u64, u32> = words.enc.literals.iter().enumerate().map(|(i, v)| (v.to_bits(), i as u32)).collect();
            let name_vals = name_values(t).unwrap();
            let lit_id = |v: f64| ids[&v.to_bits()];
            for (e, s) in want.iter().enumerate() {
                for (v, variant) in s.variants.iter().enumerate() {
                    let at = (e * VARIANTS + v) * SLOT * 4;
                    let reference = device_words(variant, COLS.len() as u32, &name_vals, &lit_id);
                    assert_eq!(words.nodes[at..at + reference.len()], reference[..], "{name}: expression {e} variant {v}, node words");
                }
            }
            assert_layout(&got);
        }

        #[test]
        fn device_agrees_with_the_f32_reference() {
            let kernel = SnapKernel::new(table()).expect("a GPU adapter");
            let graft = SnapGraft::new(&kernel).unwrap();
            let kt = kernel_tables();
            let t = kernel.table().clone();
            assert_eq!(graft.run(&kernel, &[], &cols(), &kt, TOL).unwrap(), Vec::new());

            let one = [flat(r#"(Mul (Num 0.0796) (Var "x"))"#)];
            let got = graft.run(&kernel, &one, &cols(), &kt, TOL).unwrap();
            assert_eq!(got, snap_batch(&one, &t, TOL, LitMode::F32));
            assert_eq!(math_of(&got[0].variants[0]), format!(r#"(Mul {} (Var "x"))"#, form_of(&t, got[0].hits[1].unwrap().entry)));
            assert!(confirmed(&one[0], &got[0].hits, 0, &t, TOL));
            assert_parity(&graft, &kernel, &kt, &one, crate::gpu_eval::MAX_GROUPS_PER_DIM, "a batch of 1");

            // The 200, then the hand cases: a negative hit, a repeated atom,
            // five atoms, the oversize pair, a refused expression.
            let seven = seven_nodes(&t);
            let mut exprs = corpus(&t, 200);
            exprs.extend(
                [
                    r#"(Add (Var "x") (Num -3.1416))"#.to_string(),
                    "(Num -3.1416)".to_string(),
                    r#"(Add (Mul (Num 3.1416) (Var "x")) (Sin (Num 3.1416)))"#.to_string(),
                    r#"(Add (Add (Num 3.1416) (Num 6.2832)) (Add (Add (Num 0.0796) (Num 2.7183)) (Add (Num 1.4142) (Var "x"))))"#.to_string(),
                    chain(28, t.values[seven]),
                    chain(29, t.values[seven]),
                    chain(32, t.values[seven]),
                    r#"(Mul (Sin (Var "x")) (Var "y"))"#.to_string(),
                ]
                .iter()
                .map(|m| flat(m)),
            );
            assert_parity(&graft, &kernel, &kt, &exprs, crate::gpu_eval::MAX_GROUPS_PER_DIM, "the 200 and the hand cases");
            let again = graft.run(&kernel, &exprs, &cols(), &kt, TOL).unwrap();
            assert_eq!(again, graft.run(&kernel, &exprs, &cols(), &kt, TOL).unwrap(), "two dispatches");
        }

        /// 20,000 expressions; the same with a dispatch row of 3 groups, so both
        /// passes wrap into hundreds of rows (`gid.y * stride + gid.x`); then
        /// 70,000 at the full width.
        #[test]
        fn device_large_and_two_dimensional_batches_match() {
            let kernel = SnapKernel::new(table()).expect("a GPU adapter");
            let graft = SnapGraft::new(&kernel).unwrap();
            let kt = kernel_tables();
            let exprs = corpus(kernel.table(), 20_000);
            assert_parity(&graft, &kernel, &kt, &exprs, crate::gpu_eval::MAX_GROUPS_PER_DIM, "20,000 in one row");
            assert!((exprs.len() * VARIANTS).div_ceil(64) > 3);
            assert_parity(&graft, &kernel, &kt, &exprs, 3, "20,000 in rows of 3 groups");
            // At the full width: 70,000 node-slot groups are more than one row holds.
            let exprs = corpus(kernel.table(), 70_000);
            assert!(exprs.len() > crate::gpu_eval::MAX_GROUPS_PER_DIM as usize);
            assert_parity(&graft, &kernel, &kt, &exprs, crate::gpu_eval::MAX_GROUPS_PER_DIM, "70,000: the match pass wraps");
        }

        /// A negated entry and a zero entry, through the device.
        #[test]
        fn device_grafts_a_negated_entry() {
            let entry = |value: f64, math: &str, label: &str| crate::snap_karva::ConstEntry {
                value,
                math: math.to_string(),
                label: label.to_string(),
            };
            let t = SnapTable::build(&[
                entry(0.0, "(Num 0.0)", "0"),
                entry(-3.0, r#"(Neg (Var "pi"))"#, "-three-ish"),
                entry(2.0, r#"(Div (Var "e") (Var "pi"))"#, "two-ish"),
            ])
            .unwrap();
            let kernel = SnapKernel::new(t.clone()).expect("a GPU adapter");
            let graft = SnapGraft::new(&kernel).unwrap();
            let exprs = [flat(r#"(Add (Mul (Num 3.0) (Var "x")) (Sub (Num -3.0) (Mul (Num 0.0005) (Num -2.0))))"#)];
            let got = graft.run(&kernel, &exprs, &cols(), &kernel_tables(), TOL).unwrap();
            assert_eq!(got, snap_batch(&exprs, &t, TOL, LitMode::F32));
            assert_eq!(
                math_of(&got[0].variants[3]),
                r#"(Add (Mul (Neg (Neg (Var "pi"))) (Var "x")) (Sub (Neg (Var "pi")) (Mul (Num 0.0) (Neg (Div (Var "e") (Var "pi"))))))"#
            );
            // A name the crate has no value for cannot be grafted on the device.
            let nameless = SnapTable::build(&[entry(5.0, r#"(Var "five")"#, "five")]).unwrap();
            assert!(SnapGraft::new(&SnapKernel::new(nameless).expect("a GPU adapter")).is_err());
        }
    }
}
