//! Snap on the device, stage 3: the guard. "Snap proposes, the R² guard
//! disposes" (`hff/notebooks/_snap_op.py::snap_individual`): stage 1 matched
//! the literals, stage 2 grafted the `VARIANTS` snapped variants, and here each
//! variant is evaluated and scored beside its original, and kept only if its
//! R² drops by no more than `R2_DROP_TOL`.
//!
//! THE RULE, per expression. A variant is ACCEPTABLE if its status is
//! `Grafted`, its predictions are finite on every guarded row, and
//! `R²_variant >= R²_original - r2_drop_tol`. Among the acceptable the one with
//! the fewest nodes wins; ties go to the higher R², then to the lower slot. If
//! none is acceptable the original stands, and the verdict says why:
//! `NoVariant` (nothing was grafted), `R2Dropped` (a grafted variant was finite
//! and fell too far), `NotFinite` (every grafted variant overflowed or left its
//! domain; on the device also a residual whose scaled square leaves f32). An expression with no form is `Refused`; one whose own predictions
//! are not finite on the guarded rows, or whose R² is not, is `NoBaseline`
//! (`_snap_op.py`'s `baseline_r2 is None`) and nothing is tried against it.
//!
//! WHICH ROWS. `_apply_snap_to_winners` hands `snap_individual` the TRAIN rows
//! as its "holdout", so `GuardData::train` guards on rows `0 .. n_train`: the
//! validation and edge rows select models and must not also steer what is
//! written into the genes. `GuardData::rows` is a field; any range will do.
//!
//! R² is of the expression AS IT IS: no scale or offset is re-fitted here. The
//! caller that writes a kept form back re-fits a and b as it always does.
//!
//! THE PRECISION RULE. The device decides in f32 and nothing it decides leaves
//! it as a constant: every variant the device KEEPS is re-derived on the host —
//! `snap_graft::confirmed` for the literal matches, and the R² comparison in
//! f64 with the crate's evaluator over the same rows — and a variant the host
//! turns down comes back `F64Refused` with the original standing. The
//! disagreements are counted (`Guarded::refused_literal`, `refused_r2`).

use std::ops::Range;

use super::device::SLOT;
use super::flat::{op_of, Flat};
use super::pack::arity;
use crate::gpu_eval::Op;
use super::snap_graft::{Snapped, Status, Variant, VARIANTS};
use crate::chrom_score::Splits;

pub const SNAP_GUARD_WGSL: &str = include_str!("snap_guard.wgsl");

/// `_snap_op.py`'s `r2_drop_tol`.
pub const R2_DROP_TOL: f64 = 1e-4;
/// Words per expression in the verdict block: (slot or NONE, status, R² of the
/// original, R² of each variant, 0).
pub const VERDICT_STRIDE: usize = 8;
const _: () = assert!(VERDICT_STRIDE == 3 + VARIANTS + 1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Kept = 0,
    NoVariant = 1,
    R2Dropped = 2,
    NotFinite = 3,
    Refused = 4,
    NoBaseline = 5,
    /// The device kept a variant and the host's f64 re-derivation did not.
    F64Refused = 6,
}

impl Verdict {
    pub fn from_code(code: u32) -> Result<Verdict, String> {
        [Verdict::Kept, Verdict::NoVariant, Verdict::R2Dropped, Verdict::NotFinite, Verdict::Refused, Verdict::NoBaseline]
            .get(code as usize)
            .copied()
            .ok_or_else(|| format!("status {code} is not one the guard kernel writes"))
    }
}

/// What the guard decided for one expression. An R² is NaN where there is none:
/// the slot was not grafted, or its predictions were not finite.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// The kept variant's slot; `None`: the original stands.
    pub slot: Option<usize>,
    pub status: Verdict,
    pub r2_original: f64,
    pub r2_variants: [f64; VARIANTS],
}

/// The data the guard decides on: every row of the problem, and which of them
/// decide.
#[derive(Debug, Clone)]
pub struct GuardData {
    pub cols: Vec<String>,
    /// Row-major, `cols.len()` wide.
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// The guarded rows.
    pub rows: Range<usize>,
}

impl GuardData {
    /// Guard on the train rows — what the engine hands `snap_individual`.
    pub fn train(cols: Vec<String>, x: Vec<f64>, y: Vec<f64>, splits: Splits) -> Result<GuardData, String> {
        if cols.is_empty() || x.len() != y.len() * cols.len() || y.len() != splits.total() {
            return Err(format!("{} x values, {} columns, {} targets, {} rows in the splits", x.len(), cols.len(), y.len(), splits.total()));
        }
        Ok(GuardData { cols, x, y, rows: 0..splits.n_train })
    }

    /// 1 / sqrt(sum((y - mean y)^2)) over the guarded rows: what a residual is
    /// multiplied by before it is squared, so that R² = 1 - sum(d^2). An error
    /// when the guarded target has no spread (`_snap_op.py`: `var <= 0`), is
    /// not finite, or when the scale does not fit the device's f32.
    pub fn inv_scale(&self) -> Result<f64, String> {
        if self.rows.is_empty() || self.rows.end > self.y.len() {
            return Err(format!("guarded rows {:?} of {}", self.rows, self.y.len()));
        }
        let y = &self.y[self.rows.clone()];
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
        let inv = 1.0 / ss_tot.sqrt();
        if !(ss_tot > 0.0 && ss_tot.is_finite() && (inv as f32).is_normal() && y.iter().all(|v| (*v as f32).is_finite())) {
            return Err(format!("the guarded target has sum of squares {ss_tot}: no R² on the device"));
        }
        Ok(inv)
    }

    /// The guarded rows as the evaluator's environments, each value under a
    /// name of its own: a data column named `c` and the constant `c` are
    /// different things, and a lookup by name would give the constant the
    /// column's value. `names` are the table's constants; one the crate has no
    /// value for is an error.
    pub fn envs(&self, names: &[String]) -> Result<Envs, String> {
        let known = crate::snap_karva::constant_values();
        let constants: Vec<f64> = names.iter().map(|name| known.get(name).copied().ok_or_else(|| format!("{name}: no value for it"))).collect::<Result<_, _>>()?;
        let n = self.cols.len();
        let values = self.rows.clone().flat_map(|r| self.x[r * n..(r + 1) * n].iter().copied().chain(constants.iter().copied())).collect();
        Ok(Envs { values, n_cols: n, n_names: names.len() })
    }
}

/// `GuardData::envs`: built once, read by every f64 evaluation — the guarded
/// rows' columns, then the table's constants, as one value per `Var` index.
#[derive(Debug, Clone)]
pub struct Envs {
    /// Row-major, `n_cols + n_names` wide.
    values: Vec<f64>,
    n_cols: usize,
    n_names: usize,
}

/// A form's predictions on the guarded rows, in f64, by the crate evaluator's
/// own arithmetic (`eval::apply`, operator by operator; a node's children are
/// after it, so one backward pass per row). Its `vars` are the columns, or the
/// columns and then the table's names.
pub fn predict(form: &Flat, envs: &Envs) -> Result<Vec<f64>, String> {
    let (n, names) = (envs.n_cols, envs.n_names);
    if form.vars.len() != n && form.vars.len() != n + names {
        return Err(format!("a form over {} names, the data has {n} columns and the table {names} constants", form.vars.len()));
    }
    let width = n + names;
    let ops: Vec<(Op, usize)> = form.nodes.iter().map(|node| (op_of(node.op), arity(node.op))).collect();
    if let Some(node) = form.nodes.iter().find(|node| node.op == Op::Var as u32 && node.arg0 as usize >= width) {
        return Err(format!("a form reads name {}, the data has {width}", node.arg0));
    }
    let mut value = vec![0.0f64; form.nodes.len()];
    envs.values
        .chunks_exact(width)
        .map(|row| {
            for i in (0..form.nodes.len()).rev() {
                let node = &form.nodes[i];
                value[i] = match ops[i].1 {
                    0 if node.op == Op::Num as u32 => node.lit,
                    0 => row[node.arg0 as usize],
                    1 => crate::eval::apply_op(ops[i].0, &[value[node.arg0 as usize]]).ok_or_else(|| format!("{:?} is not a Math operator", ops[i].0))?,
                    _ => crate::eval::apply_op(ops[i].0, &[value[node.arg0 as usize], value[node.arg1 as usize]]).ok_or_else(|| format!("{:?} is not a Math operator", ops[i].0))?,
                };
            }
            Ok(value[0])
        })
        .collect()
}

/// R² over the guarded rows, or `None` when a prediction is not finite.
fn r_squared_f64(preds: &[f64], data: &GuardData, inv_scale: f64) -> Option<f64> {
    if preds.iter().any(|p| !p.is_finite()) {
        return None;
    }
    let ss: f64 = preds.iter().zip(&data.y[data.rows.clone()]).map(|(p, y)| ((y - p) * inv_scale).powi(2)).sum();
    Some(1.0 - ss)
}

/// Neumaier's compensated addition, as `snap_guard.wgsl` writes it.
fn add_f32(s: (f32, f32), v: f32) -> (f32, f32) {
    let t = s.0 + v;
    let c = if s.0.abs() >= v.abs() { s.1 + ((s.0 - t) + v) } else { s.1 + ((v - t) + s.0) };
    (t, c)
}

/// `snap_guard.wgsl`'s `guard_residuals` then `r_squared`: the device's
/// arithmetic on `preds` (one form, every row), `y` and the scale rounded to
/// f32. `None`: a squared residual is not finite on a guarded row.
fn r_squared_f32(preds: &[f32], y: &[f32], rows: Range<usize>, inv_scale: f32) -> Option<f32> {
    let mut ss = (0.0f32, 0.0f32);
    let mut ok = true;
    for row in rows {
        let d = (y[row] - preds[row]) * inv_scale;
        let sq = d * d;
        if sq.is_finite() {
            ss = add_f32(ss, sq);
        } else {
            ok = false;
        }
    }
    ok.then_some(1.0 - (ss.0 + ss.1))
}

/// The rule, on R² values already reduced. `original` / a variant's R² is
/// `None` when its predictions were not finite; `floor` is `R²_original -
/// r2_drop_tol` in the arithmetic of whoever asks. `nodes[v]` is `None` for a
/// slot that is not `Grafted`.
fn decide(refused: bool, original: Option<f64>, floor: f64, nodes: [Option<usize>; VARIANTS], r2: [Option<f64>; VARIANTS]) -> Decision {
    let mut d = Decision { slot: None, status: Verdict::Refused, r2_original: f64::NAN, r2_variants: [f64::NAN; VARIANTS] };
    if refused {
        return d;
    }
    let Some(original) = original.filter(|r| r.is_finite()) else {
        d.status = Verdict::NoBaseline;
        return d;
    };
    d.r2_original = original;
    let mut best: Option<(usize, usize, f64)> = None;
    let (mut grafted, mut dropped) = (0, 0);
    for v in 0..VARIANTS {
        let Some(n) = nodes[v] else {
            continue;
        };
        grafted += 1;
        let Some(r) = r2[v] else {
            continue;
        };
        d.r2_variants[v] = r;
        if !r.is_finite() || r < floor {
            dropped += 1;
            continue;
        }
        if best.is_none_or(|(_, bn, br)| n < bn || (n == bn && r > br)) {
            best = Some((v, n, r));
        }
    }
    d.slot = best.map(|b| b.0);
    d.status = match (best, grafted, dropped) {
        (Some(_), _, _) => Verdict::Kept,
        (None, 0, _) => Verdict::NoVariant,
        (None, _, 0) => Verdict::NotFinite,
        (None, _, _) => Verdict::R2Dropped,
    };
    d
}

fn refused(expr: &Flat) -> bool {
    expr.nodes.is_empty() || expr.nodes.len() > SLOT
}

fn grafted_nodes(variants: &[Variant]) -> [Option<usize>; VARIANTS] {
    std::array::from_fn(|v| match (&variants[v].form, variants[v].status) {
        (Some(form), Status::Grafted) => Some(form.nodes.len()),
        _ => None,
    })
}

/// The CPU reference, in f64: ONE expression (its `vars` the column list) and
/// its variants (`snap_graft::snap_graft`), whose named constants are `names`.
pub fn snap_guard(expr: &Flat, variants: &[Variant], names: &[String], data: &GuardData, r2_drop_tol: f64) -> Result<Decision, String> {
    assert_eq!(variants.len(), VARIANTS, "one variant per slot");
    let inv_scale = data.inv_scale()?;
    if refused(expr) {
        return Ok(decide(true, None, 0.0, [None; VARIANTS], [None; VARIANTS]));
    }
    let envs = data.envs(names)?;
    let original = r_squared_f64(&predict(expr, &envs)?, data, inv_scale);
    let nodes = grafted_nodes(variants);
    let mut r2 = [None; VARIANTS];
    for v in (0..VARIANTS).filter(|v| nodes[*v].is_some()) {
        let form = variants[v].form.as_ref().expect("a grafted variant has a form");
        r2[v] = r_squared_f64(&predict(form, &envs)?, data, inv_scale);
    }
    Ok(decide(false, original, original.unwrap_or(f64::NAN) - r2_drop_tol, nodes, r2))
}

/// The device's twin: the same decisions from the DEVICE's predictions, in the
/// device's f32 arithmetic. `preds_org` holds `n_rows` values per expression,
/// `preds_var` per variant (expression-major, then slot).
pub fn guard_predictions(
    exprs: &[Flat],
    batch: &[Snapped],
    preds: (&[f32], &[f32]),
    data: &GuardData,
    r2_drop_tol: f64,
) -> Result<Vec<Decision>, String> {
    let n_rows = data.y.len();
    let inv_scale = data.inv_scale()? as f32;
    let tol = r2_drop_tol as f32;
    let y: Vec<f32> = data.y.iter().map(|v| *v as f32).collect();
    let (preds_org, preds_var) = preds;
    assert_eq!((preds_org.len(), preds_var.len()), (exprs.len() * n_rows, exprs.len() * VARIANTS * n_rows));
    Ok(exprs
        .iter()
        .zip(batch)
        .enumerate()
        .map(|(e, (expr, snapped))| {
            if refused(expr) {
                return decide(true, None, 0.0, [None; VARIANTS], [None; VARIANTS]);
            }
            let original = r_squared_f32(&preds_org[e * n_rows..(e + 1) * n_rows], &y, data.rows.clone(), inv_scale);
            let nodes = grafted_nodes(&snapped.variants);
            let r2: [Option<f64>; VARIANTS] = std::array::from_fn(|v| {
                let at = (e * VARIANTS + v) * n_rows;
                nodes[v]
                    .and_then(|_| r_squared_f32(&preds_var[at..at + n_rows], &y, data.rows.clone(), inv_scale))
                    .map(f64::from)
            });
            let floor = f64::from(original.unwrap_or(f32::NAN) - tol);
            decide(false, original.map(f64::from), floor, nodes, r2)
        })
        .collect())
}

/// The host's f64 word on a variant the device kept: the literal matches
/// (`snap_graft::confirmed`), then the R² comparison. `Ok(None)`: confirmed.
pub fn refusal(
    expr: &Flat,
    hits: &[Option<super::snap_table::Hit>],
    slot: usize,
    table: &super::snap_table::SnapTable,
    data: (&GuardData, &Envs),
    tolerances: (f64, f64),
) -> Result<Option<Refusal>, String> {
    let (data, envs) = data;
    let (rel_tol, r2_drop_tol) = tolerances;
    if !super::snap_graft::confirmed(expr, hits, slot, table, rel_tol) {
        return Ok(Some(Refusal::Literal));
    }
    let variant = &super::snap_graft::snap_graft(expr, hits, table)[slot];
    let form = match (&variant.form, variant.status) {
        (Some(form), Status::Grafted) => form,
        _ => return Err(format!("the device kept slot {slot}, which the host's graft leaves {:?}", variant.status)),
    };
    let inv_scale = data.inv_scale()?;
    let original = r_squared_f64(&predict(expr, envs)?, data, inv_scale);
    let snapped = r_squared_f64(&predict(form, envs)?, data, inv_scale);
    Ok(match (original, snapped) {
        (Some(o), Some(s)) if o.is_finite() && s.is_finite() && s >= o - r2_drop_tol => None,
        _ => Some(Refusal::R2),
    })
}

/// Why the host turned a device-kept variant down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// A literal it replaced does not snap, in f64, to the entry the graft used.
    Literal,
    /// In f64 the predictions are not finite, or R² drops past the tolerance.
    R2,
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use crate::gpu_eval::GpuEvaluator;
    use crate::lint::device::{encode, KernelTables};
    use crate::lint::snap_graft::{slot_contexts, slot_literals, SnapGraft, VINFO_STRIDE};
    use crate::lint::snap_table::{found_of, Found, Hit, SnapKernel, NONE};
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    const MAX_GROUPS_PER_DIM: u32 = crate::gpu_eval::MAX_GROUPS_PER_DIM;

    /// The columns an expression may use are the guard's, in the guard's order.
    fn check_columns(exprs: &[Flat], cols: &[String]) -> Result<(), String> {
        match exprs.iter().find(|f| f.vars != cols) {
            Some(f) => Err(format!("an expression over {:?}; the guard's columns are {cols:?}", f.vars)),
            None => Ok(()),
        }
    }

    /// A batch through the guard.
    #[derive(Debug, Clone)]
    pub struct Guarded {
        /// One per expression, after the host's confirmation.
        pub decisions: Vec<Decision>,
        /// The device's own word, before it: what `guard_predictions` twins.
        pub device_decisions: Vec<Decision>,
        /// The device's hits per node of each expression (empty when refused).
        pub hits: Vec<Vec<Option<Hit>>>,
        /// Per variant, the graft's (length, status, sites, atoms) words.
        pub vinfo: Vec<u32>,
        /// Variants the device kept, before the host's word.
        pub device_kept: usize,
        /// Of those, turned down by `snap_graft::confirmed`.
        pub refused_literal: usize,
        /// Of those, turned down by the f64 R² comparison.
        pub refused_r2: usize,
        /// Literals whose band reached past `snap_table::BAND`: not matched.
        pub band_overflows: usize,
        /// Wall time of the device's command buffer with its read-back, and of
        /// the host's f64 word.
        pub device_time: std::time::Duration,
        pub confirm_time: std::time::Duration,
    }

    /// The blocks a dispatch leaves on the device for the write-back: the
    /// variants (`VARIANTS * SLOT` nodes per expression, the evaluator's
    /// layout, a named constant's index + 1 in `arg0`), their info words and
    /// the device's verdicts. The caller destroys them.
    pub struct GuardBlocks {
        pub nodes_out: wgpu::Buffer,
        pub vinfo_out: wgpu::Buffer,
        pub verdicts: wgpu::Buffer,
    }

    impl GuardBlocks {
        pub fn destroy(&self) {
            for b in [&self.nodes_out, &self.vinfo_out, &self.verdicts] {
                b.destroy();
            }
        }
    }

    /// Match, graft, evaluate and guard on the EVALUATOR's device.
    pub struct SnapGuard<'a> {
        evaluator: &'a GpuEvaluator,
        kernel: &'a SnapKernel<'a>,
        graft: &'a SnapGraft,
        lengths_pipeline: wgpu::ComputePipeline,
        lengths_layout: wgpu::BindGroupLayout,
        residuals_pipeline: wgpu::ComputePipeline,
        residuals_layout: wgpu::BindGroupLayout,
        guard_pipeline: wgpu::ComputePipeline,
        guard_layout: wgpu::BindGroupLayout,
        y_buf: wgpu::Buffer,
        data: GuardData,
        envs: Envs,
        inv_scale: f32,
        pub r2_drop_tol: f64,
    }

    fn layout_of(device: &wgpu::Device, label: &str, bindings: &[(u32, Option<bool>)]) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &bindings
                .iter()
                .map(|(binding, read_only)| wgpu::BindGroupLayoutEntry {
                    binding: *binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match read_only {
                            Some(read_only) => wgpu::BufferBindingType::Storage { read_only: *read_only },
                            None => wgpu::BufferBindingType::Uniform,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        })
    }

    impl<'a> SnapGuard<'a> {
        /// `kernel` (and so `graft`) must live on `evaluator`'s device
        /// (`SnapKernel::on_device`): buffers do not cross devices. `data`'s
        /// rows are the evaluator's, in f64.
        pub fn new(
            evaluator: &'a GpuEvaluator,
            kernel: &'a SnapKernel<'a>,
            graft: &'a SnapGraft,
            data: GuardData,
        ) -> Result<Self, String> {
            let device = evaluator.device();
            // The very same `Device`: a kernel that owns one answers with its own.
            if !std::ptr::eq(device, kernel.device()) {
                return Err("the snap kernel is on another device than the evaluator".to_string());
            }
            if data.y.len() != evaluator.n_rows() as usize || data.x.len() != data.y.len() * data.cols.len() {
                return Err(format!("{} targets and {} x values for the evaluator's {} rows", data.y.len(), data.x.len(), evaluator.n_rows()));
            }
            let inv_scale = data.inv_scale()? as f32;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fuller-snap-guard"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SNAP_GUARD_WGSL)),
            });
            let lengths_layout = layout_of(device, "fuller-snap-guard-lengths", &[(1, Some(true)), (2, Some(false)), (3, None)]);
            let residuals_layout = layout_of(device, "fuller-snap-guard-residuals", &[(3, None), (6, Some(true)), (8, Some(false))]);
            let guard_layout = layout_of(
                device,
                "fuller-snap-guard",
                &[(0, Some(true)), (1, Some(true)), (3, None), (4, Some(true)), (5, Some(true)), (6, Some(true)), (7, Some(false))],
            );
            let pipeline = |layout: &wgpu::BindGroupLayout, entry_point: &str| {
                let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[layout],
                    push_constant_ranges: &[],
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point,
                    compilation_options: Default::default(),
                })
            };
            let y32: Vec<f32> = data.y.iter().map(|v| *v as f32).collect();
            let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("guard-y-resident"),
                contents: bytemuck::cast_slice(&y32),
                usage: wgpu::BufferUsages::STORAGE,
            });
            Ok(Self {
                lengths_pipeline: pipeline(&lengths_layout, "guard_lengths"),
                residuals_pipeline: pipeline(&residuals_layout, "guard_residuals"),
                guard_pipeline: pipeline(&guard_layout, "guard_main"),
                lengths_layout,
                residuals_layout,
                guard_layout,
                evaluator,
                kernel,
                graft,
                y_buf,
                envs: data.envs(&kernel.table().names)?,
                data,
                inv_scale,
                r2_drop_tol: R2_DROP_TOL,
            })
        }

        pub fn data(&self) -> &GuardData {
            &self.data
        }

        /// Guard a batch; the device blocks are released.
        pub fn run(&self, exprs: &[Flat], tables: &KernelTables, rel_tol: f64) -> Result<Guarded, String> {
            let (guarded, blocks) = self.run_resident(exprs, tables, rel_tol, MAX_GROUPS_PER_DIM)?;
            if let Some(blocks) = blocks {
                blocks.destroy();
                self.evaluator.device().poll(wgpu::Maintain::Poll);
            }
            Ok(guarded)
        }

        /// ONE command buffer: match -> graft -> the variants' lengths ->
        /// evaluate the originals -> evaluate the variants -> both blocks of
        /// predictions to scaled squared residuals -> the R² reduction and the
        /// decision; then one read-back of the verdicts, the variant
        /// info words and the hits, and the host's f64 word on what was kept.
        /// The variant, info and verdict blocks STAY on the device for the
        /// write-back (`None` for an empty batch). `max_groups` is the most
        /// workgroups one dispatch row of the per-expression passes may hold;
        /// the per-row passes (the evaluator, the residuals) wrap at the
        /// device's limit.
        pub fn run_resident(
            &self,
            exprs: &[Flat],
            tables: &KernelTables,
            rel_tol: f64,
            max_groups: u32,
        ) -> Result<(Guarded, Option<GuardBlocks>), String> {
            self.run_resident_offered(exprs, None, tables, rel_tol, max_groups)
        }

        /// [`SnapGuard::run_resident`] with only SOME literals OFFERED to the
        /// match: `offered[e][i]` false withholds node `i` of expression `e`
        /// (the match kernel is handed `NOT_A_LITERAL` for it, so it has no hit
        /// and is no atom — on the device and, as every later step reads the
        /// device's hits, on the host). The evolution engine guards a whole
        /// MODEL and offers one gene's folded constants: the fitted scale and
        /// offset, the linker's divisor and the other genes are not snap's to
        /// touch. `None` offers every literal.
        pub fn run_resident_offered(
            &self,
            exprs: &[Flat],
            offered: Option<&[Vec<bool>]>,
            tables: &KernelTables,
            rel_tol: f64,
            max_groups: u32,
        ) -> Result<(Guarded, Option<GuardBlocks>), String> {
            check_columns(exprs, &self.data.cols)?;
            if let Some(offered) = offered {
                if offered.len() != exprs.len() || offered.iter().zip(exprs).any(|(o, f)| o.len() != f.nodes.len()) {
                    return Err("offered: one flag per node of every expression".to_string());
                }
            }
            if let Some(f) = exprs.iter().find(|f| f.nodes.iter().any(|n| n.op == Op::Num as u32 && n.arg0 != 0)) {
                return Err(format!("a literal's arg0 must be 0 (it names a constant on the device): {:?}", f.nodes));
            }
            if exprs.is_empty() {
                let none = Guarded {
                    decisions: Vec::new(),
                    device_decisions: Vec::new(),
                    hits: Vec::new(),
                    vinfo: Vec::new(),
                    device_kept: 0,
                    refused_literal: 0,
                    refused_r2: 0,
                    band_overflows: 0,
                    device_time: std::time::Duration::ZERO,
                    confirm_time: std::time::Duration::ZERO,
                };
                return Ok((none, None));
            }
            assert!((1..=MAX_GROUPS_PER_DIM).contains(&max_groups));
            let start = std::time::Instant::now();
            let enc = encode(exprs, tables);
            let n_expr = u32::try_from(exprs.len()).map_err(|_| "more than u32::MAX expressions".to_string())?;
            let n_rows = self.evaluator.n_rows();
            let n_var = n_expr.checked_mul(VARIANTS as u32).filter(|n| u64::from(*n) * u64::from(n_rows) <= u64::from(u32::MAX));
            let n_var = n_var.ok_or("the batch's variants x rows do not index in a u32")?;
            let n_lit = n_expr.checked_mul(SLOT as u32).filter(|n| n.checked_mul(8).is_some());
            let n_lit = n_lit.ok_or("the batch's node slots do not index in a u32")?;
            let device = self.evaluator.device();

            let storage = |label: &str, words: &[u32]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(words),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
            };
            let out_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let read_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
            let nodes_in = storage("guard-nodes-in", &enc.nodes);
            let len_in = storage("guard-len-in", &enc.lengths);
            let mut lits = slot_literals(&enc);
            for (e, flags) in offered.unwrap_or(&[]).iter().enumerate().filter(|(e, _)| enc.lengths[*e] > 0) {
                for (lit, _) in lits[e * SLOT..].iter_mut().zip(flags).filter(|(_, offered)| !**offered) {
                    *lit = crate::lint::snap_graft::NOT_A_LITERAL;
                }
            }
            let lits_buf = storage("guard-literals", &lits);
            let contexts_buf = storage("guard-contexts", &slot_contexts(&enc));
            let org_offsets = storage("guard-offsets-originals", &(0..n_expr).map(|e| e * SLOT as u32).collect::<Vec<_>>());
            let var_offsets = storage("guard-offsets-variants", &(0..n_var).map(|t| t * SLOT as u32).collect::<Vec<_>>());
            let hit_bytes = u64::from(n_lit) * 8;
            let vinfo_bytes = u64::from(n_var) * (VINFO_STRIDE * 4) as u64;
            let verdict_bytes = u64::from(n_expr) * (VERDICT_STRIDE * 4) as u64;
            let hits_buf = make("guard-hits", hit_bytes, out_usage);
            let nodes_out = make("guard-variants", u64::from(n_var) * (SLOT * 16) as u64, out_usage);
            let vinfo_out = make("guard-vinfo", vinfo_bytes, out_usage);
            let var_len = make("guard-variant-lengths", u64::from(n_var) * 4, wgpu::BufferUsages::STORAGE);
            let preds_org = make("guard-predictions-originals", u64::from(n_expr) * u64::from(n_rows) * 4, wgpu::BufferUsages::STORAGE);
            let preds_var = make("guard-predictions-variants", u64::from(n_var) * u64::from(n_rows) * 4, wgpu::BufferUsages::STORAGE);
            let verdicts = make("guard-verdicts", verdict_bytes, out_usage);
            let hits_read = make("guard-hits-read", hit_bytes, read_usage);
            let vinfo_read = make("guard-vinfo-read", vinfo_bytes, read_usage);
            let verdicts_read = make("guard-verdicts-read", verdict_bytes, read_usage);

            let groups = |threads: u32| {
                let groups = threads.div_ceil(64);
                let groups_x = groups.min(max_groups);
                (groups_x, groups.div_ceil(groups_x))
            };
            let cfg_of = |stride: u32, n_forms: u32| {
                let cfg = [
                    n_expr,
                    n_rows,
                    self.data.rows.start as u32,
                    self.data.rows.end as u32,
                    stride,
                    (self.r2_drop_tol as f32).to_bits(),
                    self.inv_scale.to_bits(),
                    n_forms,
                    // `GuardCfg::zero`: see `snap_guard.wgsl::add`.
                    0,
                ];
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("guard-cfg"),
                    contents: bytemuck::cast_slice(&cfg),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            };
            let (lengths_groups, guard_groups) = (groups(n_var), groups(n_expr));
            let (lengths_cfg, guard_cfg) = (cfg_of(lengths_groups.0 * 64, 0), cfg_of(guard_groups.0 * 64, 0));
            // A thread per (form, row), like the evaluator's.
            let row_groups = |forms: u32| {
                let groups = (forms * n_rows).div_ceil(64);
                let groups_x = groups.min(MAX_GROUPS_PER_DIM);
                (groups_x, groups.div_ceil(groups_x))
            };
            let (org_groups, var_groups) = (row_groups(n_expr), row_groups(n_var));
            let (org_cfg, var_cfg) = (cfg_of(org_groups.0 * 64, n_expr), cfg_of(var_groups.0 * 64, n_var));
            let bind = |layout: &wgpu::BindGroupLayout, bound: &[(u32, &wgpu::Buffer)]| {
                let entries: Vec<wgpu::BindGroupEntry> =
                    bound.iter().map(|(binding, b)| wgpu::BindGroupEntry { binding: *binding, resource: b.as_entire_binding() }).collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
            };
            let lengths_bind = bind(&self.lengths_layout, &[(1, &vinfo_out), (2, &var_len), (3, &lengths_cfg)]);
            let org_bind = bind(&self.residuals_layout, &[(3, &org_cfg), (6, &self.y_buf), (8, &preds_org)]);
            let var_bind = bind(&self.residuals_layout, &[(3, &var_cfg), (6, &self.y_buf), (8, &preds_var)]);
            let guard_bind = bind(
                &self.guard_layout,
                &[(0, &len_in), (1, &vinfo_out), (3, &guard_cfg), (4, &preds_org), (5, &preds_var), (6, &self.y_buf), (7, &verdicts)],
            );

            let mut cmd = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let match_cfg = self.kernel.match_pass(&mut cmd, [&lits_buf, &contexts_buf, &hits_buf], n_lit, rel_tol, max_groups);
            let graft_cfg =
                self.graft.graft_pass(self.kernel, &mut cmd, [&nodes_in, &len_in, &hits_buf, &nodes_out, &vinfo_out], n_expr, max_groups);
            let dispatch = |cmd: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, g: (u32, u32)| {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(g.0, g.1, 1);
            };
            dispatch(&mut cmd, &self.lengths_pipeline, &lengths_bind, lengths_groups);
            let org_meta = self.evaluator.eval_pass(&mut cmd, [&nodes_in, &org_offsets, &len_in, &preds_org], n_expr)?;
            let var_meta = self.evaluator.eval_pass(&mut cmd, [&nodes_out, &var_offsets, &var_len, &preds_var], n_var)?;
            dispatch(&mut cmd, &self.residuals_pipeline, &org_bind, org_groups);
            dispatch(&mut cmd, &self.residuals_pipeline, &var_bind, var_groups);
            dispatch(&mut cmd, &self.guard_pipeline, &guard_bind, guard_groups);
            cmd.copy_buffer_to_buffer(&hits_buf, 0, &hits_read, 0, hit_bytes);
            cmd.copy_buffer_to_buffer(&vinfo_out, 0, &vinfo_read, 0, vinfo_bytes);
            cmd.copy_buffer_to_buffer(&verdicts, 0, &verdicts_read, 0, verdict_bytes);
            self.evaluator.queue().submit(Some(cmd.finish()));

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
            let hit_words = read(&hits_read)?;
            let vinfo = read(&vinfo_read)?;
            let verdict_words = read(&verdicts_read)?;

            // Release per-dispatch buffers explicitly and drain wgpu's deferred
            // queue — see gpu_eval::GpuEvaluator::eval for what happens otherwise.
            for b in [
                &nodes_in, &len_in, &lits_buf, &contexts_buf, &org_offsets, &var_offsets, &hits_buf, &var_len, &preds_org,
                &preds_var, &hits_read, &vinfo_read, &verdicts_read, &lengths_cfg, &guard_cfg, &org_cfg, &var_cfg, &match_cfg,
                &graft_cfg, &org_meta, &var_meta,
            ] {
                b.destroy();
            }
            device.poll(wgpu::Maintain::Poll);

            let device_time = start.elapsed();
            let start = std::time::Instant::now();
            let mut guarded = self.confirm(exprs, &enc.lengths, &hit_words, vinfo, &verdict_words, rel_tol)?;
            (guarded.device_time, guarded.confirm_time) = (device_time, start.elapsed());
            Ok((guarded, Some(GuardBlocks { nodes_out, vinfo_out, verdicts })))
        }

        /// Decode the device's words and ask the host about every kept variant.
        fn confirm(
            &self,
            exprs: &[Flat],
            lengths: &[u32],
            hit_words: &[u32],
            vinfo: Vec<u32>,
            verdict_words: &[u32],
            rel_tol: f64,
        ) -> Result<Guarded, String> {
            let table = self.kernel.table();
            let mut out = Guarded {
                decisions: Vec::with_capacity(exprs.len()),
                device_decisions: Vec::with_capacity(exprs.len()),
                hits: Vec::with_capacity(exprs.len()),
                vinfo,
                device_kept: 0,
                refused_literal: 0,
                refused_r2: 0,
                band_overflows: 0,
                device_time: std::time::Duration::ZERO,
                confirm_time: std::time::Duration::ZERO,
            };
            for (e, expr) in exprs.iter().enumerate() {
                let found: Vec<Found> = (e * SLOT..e * SLOT + lengths[e] as usize).map(|i| found_of(&hit_words[i * 2..i * 2 + 2])).collect();
                out.band_overflows += found.iter().filter(|f| **f == Found::BandOverflow).count();
                let hits: Vec<Option<Hit>> = found.iter().map(|f| if let Found::Hit(hit) = f { Some(*hit) } else { None }).collect();
                let w = &verdict_words[e * VERDICT_STRIDE..(e + 1) * VERDICT_STRIDE];
                let r2 = |bits: u32| f64::from(f32::from_bits(bits));
                let mut decision = Decision {
                    slot: (w[0] != NONE).then_some(w[0] as usize),
                    status: Verdict::from_code(w[1])?,
                    r2_original: r2(w[2]),
                    r2_variants: std::array::from_fn(|v| r2(w[3 + v])),
                };
                if (decision.status == Verdict::Kept) != decision.slot.is_some() {
                    return Err(format!("expression {e}: status {:?} with slot {:?}", decision.status, decision.slot));
                }
                out.device_decisions.push(decision.clone());
                if let Some(slot) = decision.slot {
                    out.device_kept += 1;
                    match refusal(expr, &hits, slot, table, (&self.data, &self.envs), (rel_tol, self.r2_drop_tol))? {
                        None => {}
                        Some(why) => {
                            out.refused_literal += usize::from(why == Refusal::Literal);
                            out.refused_r2 += usize::from(why == Refusal::R2);
                            decision.slot = None;
                            decision.status = Verdict::F64Refused;
                        }
                    }
                }
                out.decisions.push(decision);
                out.hits.push(hits);
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::{GuardBlocks, Guarded, SnapGuard};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lint::test_rng::draw;
    use crate::lint::engine::LitMode;
    use crate::lint::node::Tree;
    use crate::lint::snap_graft::{match_literals, snap_graft};
    use crate::lint::snap_table::SnapTable;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-3;

    fn table() -> SnapTable {
        SnapTable::standard().expect("the lattice builds")
    }

    /// A deterministic number in [0, 1).
    fn unit(seed: u32, row: usize, col: usize) -> f64 {
        (draw(seed, 0, row as u32, col as u32, 0) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `n` rows of `cols`, each value `x_of(row, column)`, the target `y_of(row
    /// values)`; the first two thirds are train rows, and guarded.
    fn data(cols: &[&str], n: usize, x_of: impl Fn(usize, usize) -> f64, y_of: impl Fn(&[f64]) -> f64) -> GuardData {
        let x: Vec<f64> = (0..n).flat_map(|r| (0..cols.len()).map(move |c| (r, c))).map(|(r, c)| x_of(r, c)).collect();
        let y: Vec<f64> = x.chunks(cols.len()).map(&y_of).collect();
        let n_train = n * 2 / 3;
        let splits = Splits { n_train, n_val: (n - n_train) / 2, n_extrap: n - n_train - (n - n_train) / 2 };
        GuardData::train(cols.iter().map(|c| c.to_string()).collect(), x, y, splits).unwrap()
    }

    fn guarded(math: &str, data: &GuardData, t: &SnapTable) -> (Vec<Variant>, Decision) {
        let f = Flat::from_tree_in(&Tree::parse(math).unwrap(), &data.cols).unwrap();
        let hits = match_literals(&f, t, TOL, LitMode::F64);
        let variants = snap_graft(&f, &hits, t);
        let decision = snap_guard(&f, &variants, &t.names, data, R2_DROP_TOL).unwrap();
        (variants, decision)
    }

    fn math_of(v: &Variant) -> String {
        v.form.as_ref().expect("a form").to_tree().to_math()
    }

    #[test]
    fn a_snap_that_keeps_r2_is_kept() {
        let t = table();
        let d = data(&["x0"], 300, |r, _| 1.0 + 99.0 * unit(41, r, 0), |x| x[0] / (4.0 * PI));
        assert_eq!(d.rows, 0..200, "the train rows decide");
        let (variants, got) = guarded(r#"(Mul (Num 0.0795775) (Var "x0"))"#, &d, &t);
        assert_eq!(math_of(&variants[0]), r#"(Mul (Div (Num 1.0) (Mul (Num 4.0) (Var "pi"))) (Var "x0"))"#);
        assert_eq!((got.slot, got.status), (Some(0), Verdict::Kept));
        // The snapped form IS the law: its R² does not drop, it rises to 1.
        assert!(got.r2_variants[0] >= got.r2_original && got.r2_variants[0] > 1.0 - 1e-15, "{got:?}");
        assert!(got.r2_original > 1.0 - 1e-9 && got.r2_original < 1.0, "{got:?}");
        assert!(got.r2_variants[1..].iter().all(|r| r.is_nan()));
    }

    #[test]
    fn a_snap_that_moves_r2_is_dropped_and_the_original_stands() {
        let t = table();
        let model = r#"(Mul (Num 0.0795) (Var "x0"))"#;
        // y = 0.0795 x exactly, and 1/(4 pi) = 0.0795775 is 9.7e-4 away. With no
        // re-fit, 1 - R² of the snapped model is (rel err)^2 * sum(x^2) /
        // sum((x - mean)^2): a band of x far from zero makes that large.
        let narrow = data(&["x0"], 300, |r, _| 100.0 + 10.0 * unit(43, r, 0), |x| 0.0795 * x[0]);
        let (variants, got) = guarded(model, &narrow, &t);
        assert_eq!(math_of(&variants[0]), r#"(Mul (Div (Num 1.0) (Mul (Num 4.0) (Var "pi"))) (Var "x0"))"#);
        assert_eq!((got.slot, got.status), (None, Verdict::R2Dropped));
        assert_eq!(got.r2_original, 1.0);
        let x: Vec<f64> = narrow.x[..200].to_vec();
        let mean = x.iter().sum::<f64>() / 200.0;
        let rel = (1.0 / (4.0 * PI) - 0.0795) / 0.0795;
        let expected = rel * rel * x.iter().map(|v| v * v).sum::<f64>() / x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
        let drop = 1.0 - got.r2_variants[0];
        eprintln!("snap guard, 0.0795 -> 1/(4*pi) on x in 100..110: R² drops by {drop:.3e} (predicted {expected:.3e}), tolerance {R2_DROP_TOL}");
        assert!((drop - expected).abs() <= 1e-6 * expected, "{drop} against {expected}");
        assert!(drop > 10.0 * R2_DROP_TOL, "{drop}");
        // The same model on x spread over 1..100: the same snap costs 1e-6 of
        // R² and is kept. The guard decides on the data, not on the form.
        let wide = data(&["x0"], 300, |r, _| 1.0 + 99.0 * unit(43, r, 0), |x| 0.0795 * x[0]);
        let (_, got) = guarded(model, &wide, &t);
        assert_eq!((got.slot, got.status), (Some(0), Verdict::Kept));
        assert!((1e-7..1e-5).contains(&(1.0 - got.r2_variants[0])), "{got:?}");
    }

    #[test]
    fn a_variant_that_leaves_its_domain_is_not_finite() {
        let t = table();
        // Row 0 is pi itself: 1 / (x0 - 3.1415) is finite there, 1 / (x0 - pi) is not.
        // pi as a gene prints it (text: a number typed as 3.1415 is rounded on purpose).
        let rounded: f64 = "3.1415".parse().expect("a number");
        let d = data(&["x0"], 90, |r, _| if r == 0 { PI } else { 4.0 + 2.0 * unit(47, r, 0) }, |x| 1.0 / (x[0] - rounded));
        let (variants, got) = guarded(r#"(Inv (Sub (Var "x0") (Num 3.1415)))"#, &d, &t);
        assert_eq!(math_of(&variants[0]), r#"(Inv (Sub (Var "x0") (Var "pi")))"#);
        assert_eq!((got.slot, got.status), (None, Verdict::NotFinite));
        assert!(got.r2_original > 1.0 - 1e-12 && got.r2_variants.iter().all(|r| r.is_nan()), "{got:?}");
        // The ORIGINAL not finite on a guarded row: nothing to guard against.
        let (_, got) = guarded(r#"(Mul (Num 3.1416) (Log (Neg (Pow2 (Var "x0")))))"#, &d, &t);
        assert_eq!((got.slot, got.status), (None, Verdict::NoBaseline));
        assert!(got.r2_original.is_nan());
    }

    #[test]
    fn of_two_acceptable_variants_the_smaller_wins() {
        let t = table();
        let d = data(&["x0"], 300, |r, _| 1.0 + 9.0 * unit(53, r, 0), |x| PI * x[0] + 1.0 / (4.0 * PI));
        let (variants, got) = guarded(r#"(Add (Mul (Num 3.1416) (Var "x0")) (Num 0.0796))"#, &d, &t);
        let nodes: Vec<usize> = variants.iter().map(|v| v.form.as_ref().unwrap().nodes.len()).collect();
        let status: Vec<Status> = variants.iter().map(|v| v.status).collect();
        // Level order puts 0.0796 first: slot 0 is its 5-node form, slot 1 is pi.
        assert_eq!((nodes, status), (vec![9, 5, 5, 9], vec![Status::Grafted, Status::Grafted, Status::NoHit, Status::Grafted]));
        let floor = got.r2_original - R2_DROP_TOL;
        assert!([0, 1, 3].iter().all(|v| got.r2_variants[*v] >= floor), "all three are acceptable: {got:?}");
        assert!(got.r2_variants[3] > got.r2_variants[1], "the pair fits best, and is not the smallest");
        assert_eq!((got.slot, got.status), (Some(1), Verdict::Kept));
        assert!(got.r2_variants[2].is_nan());
    }

    #[test]
    fn nothing_to_snap_and_nothing_to_score() {
        let t = table();
        let d = data(&["x0"], 90, |r, _| 1.0 + unit(59, r, 0), |x| x[0].sin() * x[0]);
        let (_, got) = guarded(r#"(Mul (Sin (Var "x0")) (Var "x0"))"#, &d, &t);
        assert_eq!((got.slot, got.status), (None, Verdict::NoVariant));
        assert_eq!(got.r2_original, 1.0);
        // Longer than the slot: no form, no verdict on it.
        let mut long = r#"(Num 3.1416)"#.to_string();
        for _ in 0..40 {
            long = format!(r#"(Add (Var "x0") {long})"#);
        }
        let (variants, got) = guarded(&long, &d, &t);
        assert!(variants.iter().all(|v| v.status == Status::Refused));
        assert_eq!((got.slot, got.status), (None, Verdict::Refused));
        // A target with no spread on the guarded rows has no R².
        let flat = data(&["x0"], 90, |r, _| 1.0 + unit(59, r, 0), |_| 2.5);
        assert!(flat.inv_scale().is_err());
        // A target near 1e-34 has one: the residual is scaled before it is squared.
        let tiny = data(&["x0"], 90, |r, _| 1.0 + unit(59, r, 0), |x| 6.62607015e-34 * x[0]);
        let (_, got) = guarded(r#"(Mul (Num 6.62607015e-34) (Var "x0"))"#, &tiny, &t);
        assert!(got.r2_original > 1.0 - 1e-12, "{got:?}");
    }

    /// Feynman I.29.4 has a column named `c`, and the lattice a constant `c`.
    #[test]
    fn a_column_named_like_a_constant_stays_a_column() {
        let t = table();
        let light = crate::snap_karva::constant_values()["c"];
        let d = data(&["c"], 90, |r, _| 1.0 + unit(61, r, 0), |x| light * x[0]);
        let (variants, got) = guarded(r#"(Mul (Num 299792000.0) (Var "c"))"#, &d, &t);
        // By name this reads (Mul c c); by index it is the constant times the column.
        assert_eq!(math_of(&variants[0]), r#"(Mul (Var "c") (Var "c"))"#);
        assert_eq!((got.slot, got.status), (Some(0), Verdict::Kept));
        assert!(got.r2_variants[0] > 1.0 - 1e-15 && got.r2_variants[0] > got.r2_original, "{got:?}");
    }

    #[test]
    fn wgsl_constants_match_the_host() {
        use crate::lint::snap_graft::VINFO_STRIDE;
        for needle in [
            format!("const VARIANTS: u32 = {VARIANTS}u;"),
            format!("const VINFO_STRIDE: u32 = {VINFO_STRIDE}u;"),
            format!("const VERDICT_STRIDE: u32 = {VERDICT_STRIDE}u;"),
            format!("const GRAFTED: u32 = {}u;", Status::Grafted as u32),
            format!("const KEPT: u32 = {}u;", Verdict::Kept as u32),
            format!("const NO_VARIANT: u32 = {}u;", Verdict::NoVariant as u32),
            format!("const R2_DROPPED: u32 = {}u;", Verdict::R2Dropped as u32),
            format!("const NOT_FINITE: u32 = {}u;", Verdict::NotFinite as u32),
            format!("const REFUSED: u32 = {}u;", Verdict::Refused as u32),
            format!("const NO_BASELINE: u32 = {}u;", Verdict::NoBaseline as u32),
            "const NONE: u32 = 0xffffffffu;".to_string(),
        ] {
            assert!(SNAP_GUARD_WGSL.contains(&needle), "snap_guard.wgsl lacks `{needle}`");
        }
    }

    #[cfg(feature = "gpu")]
    mod device {
        use super::*;
        use crate::gpu_eval::{ExprBatch, GpuEvaluator, GpuNode};
        use crate::lint::device::KernelTables;
        use crate::lint::pack::pack;
        use crate::lint::snap_graft::tests::corpus;
        use crate::lint::snap_graft::{device_words, name_values, snap_batch, SnapGraft};
        use crate::lint::snap_table::SnapKernel;
        use crate::lint::tables::{Rule, Tables};

        const COLS: [&str; 3] = ["x", "y", "z"];

        fn kernel_tables() -> KernelTables {
            let tables = Tables::standard().unwrap();
            let rules: Vec<&Rule> = tables.rules.iter().collect();
            let p = pack(&rules);
            KernelTables::new(&p.rules, p.codes, &tables.guards).unwrap()
        }

        /// Three columns in 0.5 .. 2.5 and a law with pi in it.
        fn problem(n: usize) -> GuardData {
            data(&COLS, n, |r, c| 0.5 + 2.0 * unit(67, r, c), |x| PI * x[0] + (x[1] * 2.0 * PI).sin() * x[2])
        }

        fn flat(math: &str) -> Flat {
            let cols: Vec<String> = COLS.iter().map(|c| c.to_string()).collect();
            Flat::from_tree_in(&Tree::parse(math).unwrap(), &cols).unwrap()
        }

        fn evaluator_of(d: &GuardData) -> GpuEvaluator {
            let rows: Vec<f32> = d.x.iter().map(|v| *v as f32).collect();
            GpuEvaluator::new(&rows, d.cols.len()).expect("a GPU adapter")
        }

        /// An R² with NaN as "none", for comparing decisions.
        fn key(d: &Decision) -> (Option<usize>, Verdict, Option<u64>, Vec<Option<u64>>) {
            let bits = |r: f64| (!r.is_nan()).then_some(r.to_bits());
            (d.slot, d.status, bits(d.r2_original), d.r2_variants.iter().map(|r| bits(*r)).collect())
        }

        /// The twin's decisions, from the DEVICE's own predictions of the F32
        /// reference's variants.
        fn reference(evaluator: &GpuEvaluator, exprs: &[Flat], t: &SnapTable, d: &GuardData) -> Vec<Decision> {
            let batch = snap_batch(exprs, t, TOL, LitMode::F32);
            let name_vals = name_values(t).unwrap();
            let nodes_of = |words: Vec<u32>| -> Vec<GpuNode> {
                words.chunks_exact(4).map(|w| GpuNode { op: w[0], arg0: w[1], arg1: w[2], konst: f32::from_bits(w[3]) }).collect()
            };
            let (mut originals, mut variants) = (ExprBatch::new(), ExprBatch::new());
            for (f, s) in exprs.iter().zip(&batch) {
                let own: Vec<GpuNode> =
                    f.nodes.iter().map(|n| GpuNode { op: n.op, arg0: n.arg0, arg1: n.arg1, konst: n.lit as f32 }).collect();
                originals.push(&own);
                for v in &s.variants {
                    let words = if v.status == Status::Grafted { device_words(v, d.cols.len() as u32, &name_vals, &|_| 0) } else { Vec::new() };
                    variants.push(&nodes_of(words));
                }
            }
            let preds = (evaluator.eval(&originals).unwrap(), evaluator.eval(&variants).unwrap());
            guard_predictions(exprs, &batch, (&preds.0, &preds.1), d, R2_DROP_TOL).unwrap()
        }

        fn count(decisions: &[Decision]) -> [usize; 7] {
            let mut out = [0; 7];
            decisions.iter().for_each(|d| out[d.status as usize] += 1);
            out
        }

        fn assert_parity(guard: &SnapGuard, evaluator: &GpuEvaluator, kt: &KernelTables, exprs: &[Flat], t: &SnapTable, max_groups: u32, name: &str) {
            let start = std::time::Instant::now();
            let (got, blocks) = guard.run_resident(exprs, kt, TOL, max_groups).unwrap();
            let device = start.elapsed();
            blocks.expect("a batch leaves its blocks").destroy();
            let start = std::time::Instant::now();
            let want = reference(evaluator, exprs, t, guard.data());
            let twin = start.elapsed();
            eprintln!(
                "device snap guard, {name}: {} expressions x {} rows; device verdicts [kept, no_variant, r2_dropped, not_finite, refused, no_baseline, f64_refused] {:?}; \
                 after the host's f64 word {:?}: of {} kept, {} refused on a literal, {} on R²; {} band overflows; \
                 one command buffer (match + graft + evaluate + guard) + readback {:?}, the f64 word {:?}, all {device:?}; F32 twin (two device evaluations + CPU) {twin:?}",
                exprs.len(),
                guard.data().y.len(),
                count(&got.device_decisions),
                count(&got.decisions),
                got.device_kept,
                got.refused_literal,
                got.refused_r2,
                got.band_overflows,
                got.device_time,
                got.confirm_time
            );
            assert_eq!(got.device_decisions.len(), want.len(), "{name}");
            for (e, (g, w)) in got.device_decisions.iter().zip(&want).enumerate() {
                assert_eq!(key(g), key(w), "{name}: expression {e}: {}", exprs[e].to_tree().to_math());
            }
            assert_eq!(got.band_overflows, 0);
            assert_eq!(got.device_kept, got.decisions.iter().filter(|d| d.status == Verdict::Kept).count() + got.refused_literal + got.refused_r2);
        }

        fn hand_cases() -> Vec<Flat> {
            let mut long = r#"(Num 3.1416)"#.to_string();
            for _ in 0..40 {
                long = format!(r#"(Add (Var "x") {long})"#);
            }
            [
                // The law with its constants rounded: every snap is kept.
                r#"(Add (Mul (Num 3.1416) (Var "x")) (Mul (Sin (Mul (Var "y") (Num 6.2832))) (Var "z")))"#.to_string(),
                r#"(Add (Mul (Num 3.1416) (Var "x")) (Mul (Sin (Mul (Var "y") (Num 6.28))) (Var "z")))"#.to_string(),
                r#"(Mul (Sin (Var "x")) (Var "y"))"#.to_string(),
                r#"(Mul (Num 3.1416) (Log (Neg (Pow2 (Var "x")))))"#.to_string(),
                r#"(Inv (Sub (Var "x") (Num 3.1415)))"#.to_string(),
                r#"(Mul (Num 6.62607015e-34) (Var "x"))"#.to_string(),
                long,
            ]
            .iter()
            .map(|m| flat(m))
            .collect()
        }

        #[test]
        fn device_guard_agrees_with_the_f32_twin() {
            let d = problem(600);
            let evaluator = evaluator_of(&d);
            let kernel = SnapKernel::on_device(evaluator.device(), evaluator.queue(), table()).unwrap();
            let graft = SnapGraft::new(&kernel).unwrap();
            let guard = SnapGuard::new(&evaluator, &kernel, &graft, d.clone()).unwrap();
            let kt = kernel_tables();
            let t = kernel.table().clone();

            let none = guard.run(&[], &kt, TOL).unwrap();
            assert!(none.decisions.is_empty() && none.device_kept == 0);

            let one = [flat(r#"(Add (Mul (Num 3.1416) (Var "x")) (Mul (Sin (Mul (Var "y") (Num 6.2832))) (Var "z")))"#)];
            let got = guard.run(&one, &kt, TOL).unwrap();
            // Both constants snapped at once is the law itself, and the smallest
            // form R² allows is asked for: pi alone (11 nodes, as the original).
            assert_eq!(got.decisions[0].status, Verdict::Kept, "{got:?}");
            assert_eq!(got.decisions[0].slot, snap_guard_f64(&one[0], &t, &d).slot);
            assert_parity(&guard, &evaluator, &kt, &one, &t, crate::gpu_eval::MAX_GROUPS_PER_DIM, "a batch of 1");

            let mut exprs = corpus(&t, 200);
            exprs.extend(hand_cases());
            assert_parity(&guard, &evaluator, &kt, &exprs, &t, crate::gpu_eval::MAX_GROUPS_PER_DIM, "the 200 and the hand cases");
            let again = guard.run(&exprs, &kt, TOL).unwrap();
            let twice = guard.run(&exprs, &kt, TOL).unwrap();
            assert_eq!(again.decisions.iter().map(key).collect::<Vec<_>>(), twice.decisions.iter().map(key).collect::<Vec<_>>(), "two dispatches");

            // The device against the f64 reference of the WHOLE rule, both ways.
            let (mut same, mut device_only, mut host_only, mut other_slot) = (0, 0, 0, 0);
            for (f, dev) in exprs.iter().zip(&again.decisions) {
                let host = snap_guard_f64(f, &t, &d);
                match (dev.slot, host.slot) {
                    (a, b) if a == b => same += 1,
                    (Some(_), None) => device_only += 1,
                    (None, Some(_)) => host_only += 1,
                    _ => other_slot += 1,
                }
            }
            eprintln!(
                "device snap guard (after the f64 word) against the f64 reference of the whole rule, {} expressions: {same} the same slot, \
                 {device_only} kept on the device only, {host_only} kept by the f64 reference only, {other_slot} another slot",
                exprs.len()
            );
            assert_eq!(device_only, 0, "the f64 word lets nothing through that f64 refuses");
        }

        fn snap_guard_f64(f: &Flat, t: &SnapTable, d: &GuardData) -> Decision {
            let hits = match_literals(f, t, TOL, LitMode::F64);
            snap_guard(f, &snap_graft(f, &hits, t), &t.names, d, R2_DROP_TOL).unwrap()
        }

        /// 1,200 expressions over 1,000 rows: the variants' evaluation is 4.8M
        /// threads, past one dispatch row; the snap passes in rows of 3 groups.
        #[test]
        fn device_guard_in_two_dimensional_dispatches() {
            let d = problem(1000);
            let evaluator = evaluator_of(&d);
            let kernel = SnapKernel::on_device(evaluator.device(), evaluator.queue(), table()).unwrap();
            let graft = SnapGraft::new(&kernel).unwrap();
            let guard = SnapGuard::new(&evaluator, &kernel, &graft, d).unwrap();
            let kt = kernel_tables();
            let t = kernel.table().clone();
            let exprs = corpus(&t, 1200);
            assert!((exprs.len() * VARIANTS * 1000).div_ceil(64) > crate::gpu_eval::MAX_GROUPS_PER_DIM as usize);
            assert_parity(&guard, &evaluator, &kt, &exprs, &t, 3, "1,200 in rows of 3 groups");
            assert_parity(&guard, &evaluator, &kt, &exprs, &t, crate::gpu_eval::MAX_GROUPS_PER_DIM, "1,200 at the full width");
        }

        /// A kernel on its own device cannot share the evaluator's buffers.
        #[test]
        fn the_guard_wants_one_device() {
            let d = problem(60);
            let evaluator = evaluator_of(&d);
            let kernel = SnapKernel::new(table()).expect("a GPU adapter");
            let graft = SnapGraft::new(&kernel).unwrap();
            assert!(SnapGuard::new(&evaluator, &kernel, &graft, d.clone()).is_err());
            let shared = SnapKernel::on_device(evaluator.device(), evaluator.queue(), table()).unwrap();
            let graft = SnapGraft::new(&shared).unwrap();
            let guard = SnapGuard::new(&evaluator, &shared, &graft, d).unwrap();
            let other: Vec<String> = vec!["a".to_string()];
            let stranger = Flat::from_tree_in(&Tree::parse(r#"(Var "a")"#).unwrap(), &other).unwrap();
            assert!(guard.run(&[stranger], &kernel_tables(), TOL).is_err());
        }
    }
}
