//! The reduction half of GPU-speed pattern matching: turn per-column value
//! vectors (a gene's subtree partials, and a library of pattern columns) into a
//! per-column-pair affine 1 - R², cheaply, from SUMS.
//!
//! Given a subtree's value vector `S` and a pattern's value vector `P` over the
//! same rows, "is S an affine image of P" (i.e. does `S ≈ a·P + b` for some
//! constants) is answered by the residual variance of the least-squares fit, and
//! that residual is a function of five per-column sums only:
//! `Σs, Σp, Σs², Σp², Σsp`. So the expensive part — the per-row values — is
//! produced once by [`crate::gpu_eval::GpuEvaluator::eval_with_partials`], reduced
//! to a handful of sums per column, and every (subtree × pattern) score is O(1)
//! arithmetic on those sums. No per-row host traffic per pair.
//!
//! [`ColumnSums`]/[`affine_residual`] are the HOST reduction — the correctness
//! reference. [`PatternVerify`] is the ON-GPU reduction: the whole
//! (subtree × pattern) score is computed on the device from the resident eval
//! predictions (`pattern_verify.wgsl`), and only the `n_cand × n_pat` scalar
//! `1 - R²` values return. That is the order-of-magnitude path; the host
//! reduction stays as the reference the GPU is tested against.

/// The five sums of one value column needed for an affine fit against a target,
/// plus the target's own sums, over the JOINTLY-FINITE rows. `n` is that finite
/// count. Rows where either side is non-finite are dropped (a NaN/inf subtree
/// value or target row carries no information and must not poison the sum).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColumnSums {
    pub n: f64,
    pub sx: f64,
    pub sy: f64,
    pub sxx: f64,
    pub syy: f64,
    pub sxy: f64,
}

impl ColumnSums {
    /// Accumulate the sums of `x` (a candidate column) against `y` (the target),
    /// over rows where BOTH are finite.
    pub fn of(x: &[f32], y: &[f64]) -> ColumnSums {
        debug_assert_eq!(x.len(), y.len(), "column and target must be the same rows");
        let mut s = ColumnSums::default();
        for (&xi, &yi) in x.iter().zip(y) {
            let xi = f64::from(xi);
            if !xi.is_finite() || !yi.is_finite() {
                continue;
            }
            s.n += 1.0;
            s.sx += xi;
            s.sy += yi;
            s.sxx += xi * xi;
            s.syy += yi * yi;
            s.sxy += xi * yi;
        }
        s
    }
}

/// The `1 - R²` of the least-squares affine fit `y ≈ a·x + b`, from the sums.
///
/// `None` when there are too few finite rows, or `x` is (near-)constant so no
/// slope is determined, or the target has no variance. A returned `0.0` means an
/// exact affine match — `x` IS the target up to scale and offset, which is the
/// "this subtree is the pattern" signal. Small values are near-matches.
///
/// This is scale- and offset-free by construction (the outer model wrap supplies
/// `a` and `b`), which is exactly why a pattern need carry no free constants.
pub fn affine_residual(s: ColumnSums, min_rows: f64) -> Option<f64> {
    if s.n < min_rows {
        return None;
    }
    let n = s.n;
    let var_x = s.sxx - s.sx * s.sx / n;
    let var_y = s.syy - s.sy * s.sy / n;
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    let cov = s.sxy - s.sx * s.sy / n;
    // R² of a simple linear regression is cov² / (var_x · var_y); 1 - R² is the
    // fraction of the target's variance the fit leaves unexplained. Clamped to
    // [0, 1] against f64 rounding at a near-perfect fit.
    let r2 = (cov * cov / (var_x * var_y)).clamp(0.0, 1.0);
    Some(1.0 - r2)
}

/// The absolute Pearson correlation |r| = sqrt(R²), from the sums — the same
/// evidence as [`affine_residual`] in the units the incomplete-beta p-value
/// consumes (`theta = acos(|r|)`). `None` under the same degenerate conditions.
pub fn abs_corr(s: ColumnSums, min_rows: f64) -> Option<f64> {
    affine_residual(s, min_rows).map(|one_minus_r2| (1.0 - one_minus_r2).sqrt())
}

/// THE RESIDUAL TEST, host reference — the coefficient of variation of the ratio
/// `subtree / pattern` over the rows. This mirrors [`PatternVerify`] exactly and
/// is the reference the GPU kernel is tested against.
///
/// A CONSTANT ratio means the subtree IS the pattern up to a single scale factor
/// (the outer model supplies the scale) — the "y / template is simple" test the
/// whole-output residual path uses, per subtree. NO affine fit, NO intercept: a
/// pattern is a MULTIPLICATIVE factor, matched by a flat ratio. Returns
/// `std(ratio)/|mean(ratio)|` in `[0, ∞)`, `0` = perfectly constant = the shape.
/// `None` when too few rows have a usable (finite, non-zero-divisor) ratio, or
/// the mean ratio is ~0 (the subtree is ~0 everywhere — no scale to speak of).
///
/// Rows where the subtree or pattern is non-finite, or the pattern is ~0, are
/// dropped: the ratio is undefined there, which is not evidence against the shape.
pub fn ratio_cov(subtree: &[f32], pattern: &[f32], min_rows: usize) -> Option<f64> {
    debug_assert_eq!(subtree.len(), pattern.len(), "same rows");
    let (mut n, mut sr, mut srr) = (0.0f64, 0.0f64, 0.0f64);
    for (&s, &p) in subtree.iter().zip(pattern) {
        let (s, p) = (f64::from(s), f64::from(p));
        if !s.is_finite() || !p.is_finite() || p.abs() <= 1e-20 {
            continue;
        }
        let r = s / p;
        if !r.is_finite() {
            continue;
        }
        n += 1.0;
        sr += r;
        srr += r * r;
    }
    if (n as usize) < min_rows {
        return None;
    }
    let mean = sr / n;
    if mean.abs() < 1e-20 {
        return None;
    }
    let variance = (srr / n - mean * mean).max(0.0);
    Some(variance.sqrt() / mean.abs())
}

/// Score every subtree column of ONE expression's partials against a target.
///
/// `partials` is the [`crate::gpu_eval::GpuEvaluator::eval_with_partials`] output;
/// `expr` is the expression's index in the batch; `n_nodes` is its node count
/// (only `k < n_nodes` are real subtrees — higher slots are 0 padding);
/// `max_nodes` is [`crate::gpu_eval::MAX_NODES`]; `n_rows` the row count.
///
/// Returns, per subtree node `k`, its affine `1 - R²` against the target (or
/// `None` for a degenerate column). The caller picks the smallest — the subtree
/// whose values are most nearly an affine image of the target.
pub fn score_subtrees(
    partials: &[f32],
    expr: usize,
    n_nodes: usize,
    max_nodes: usize,
    n_rows: usize,
    target: &[f64],
    min_rows: f64,
) -> Vec<Option<f64>> {
    (0..n_nodes)
        .map(|k| {
            let base = (expr * max_nodes + k) * n_rows;
            let col = &partials[base..base + n_rows];
            affine_residual(ColumnSums::of(col, target), min_rows)
        })
        .collect()
}

#[cfg(feature = "gpu")]
pub use gpu::PatternVerify;

/// The on-GPU (subtree × pattern) reduction. A second compute kernel over the
/// eval predictions the evaluator leaves resident, so the ratio coefficient of
/// variation of every pair (the residual test, [`ratio_cov`]) is computed on the
/// device and only the pair scores return.
#[cfg(feature = "gpu")]
mod gpu {
    use crate::gpu_eval::GpuEvaluator;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    const PATTERN_VERIFY_WGSL: &str = include_str!("pattern_verify.wgsl");

    /// Owns the verify pipeline. Built from a [`GpuEvaluator`]'s device and driven
    /// with predictions that same evaluator produced.
    pub struct PatternVerify {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
    }

    impl PatternVerify {
        pub fn new(evaluator: &GpuEvaluator) -> PatternVerify {
            let device = evaluator.device();
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pattern-verify"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PATTERN_VERIFY_WGSL)),
            });
            // binding 0 preds (read), 1 out (read-write), 2 cfg (uniform).
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pattern-verify-layout"),
                entries: &(0..3)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: if i == 2 {
                            wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            }
                        } else {
                            wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: i == 0 },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            }
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
                label: Some("pattern-verify-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "verify_main",
                compilation_options: Default::default(),
            });
            PatternVerify { pipeline, layout }
        }

        /// Score every (candidate subtree, pattern) pair from predictions the
        /// evaluator left resident (`preds`, `n_expr = n_cand + n_pat` columns of
        /// `n_rows`, candidates first). Returns `n_cand * n_pat` values, indexed
        /// `out[cand * n_pat + pat]` = the ratio coefficient of variation (0 = the
        /// subtree is the pattern up to scale), or `2.0` for an unscorable pair
        /// (too few usable rows, or a ~0 mean ratio). The reduction runs on the
        /// GPU; only these scalars cross to the host.
        pub fn verify(
            &self,
            evaluator: &GpuEvaluator,
            preds: &wgpu::Buffer,
            n_cand: u32,
            n_pat: u32,
            n_rows: u32,
            min_rows: u32,
        ) -> Result<Vec<f32>, String> {
            let device = evaluator.device();
            let pairs = (n_cand as u64) * (n_pat as u64);
            if pairs == 0 {
                return Ok(Vec::new());
            }
            if pairs > u32::MAX as u64 {
                return Err(format!("{n_cand} candidates x {n_pat} patterns overflows the pair index"));
            }
            let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pattern-verify-out"),
                size: pairs * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pattern-verify-readback"),
                size: pairs * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let cfg = [n_cand, n_pat, n_rows, min_rows];
            let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pattern-verify-cfg"),
                contents: bytemuck::cast_slice(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: preds.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
                ],
            });
            let groups = (pairs as u32).div_ceil(64);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, pairs * 4);
            evaluator.queue().submit(Some(enc.finish()));

            let slice = read_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().map_err(|e| format!("map_async channel: {e}"))?.map_err(|e| format!("map_async: {e}"))?;
            let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
            read_buf.unmap();
            out_buf.destroy();
            read_buf.destroy();
            cfg_buf.destroy();
            device.poll(wgpu::Maintain::Poll);
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exact_affine_image_scores_zero() {
        // y = 3x + 5 exactly → 1 - R² = 0.
        let x: Vec<f32> = (0..200).map(|i| i as f32 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&v| 3.0 * f64::from(v) + 5.0).collect();
        let r = affine_residual(ColumnSums::of(&x, &y), 10.0).expect("a fit");
        assert!(r < 1e-9, "1 - R² = {r}");
        let c = abs_corr(ColumnSums::of(&x, &y), 10.0).expect("corr");
        assert!(c > 1.0 - 1e-6, "|r| = {c}");
    }

    #[test]
    fn an_unrelated_column_scores_near_one() {
        // y independent of x → R² ≈ 0 → 1 - R² ≈ 1.
        let x: Vec<f32> = (0..500).map(|i| ((i * 7 % 13) as f32) * 0.3).collect();
        let y: Vec<f64> = (0..500).map(|i| ((i * 5 % 11) as f64) * 0.2 + 1.0).collect();
        let r = affine_residual(ColumnSums::of(&x, &y), 10.0).expect("a fit");
        assert!(r > 0.5, "unrelated should be poorly explained: 1 - R² = {r}");
    }

    #[test]
    fn a_constant_column_is_none() {
        let x = vec![2.0f32; 100];
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(affine_residual(ColumnSums::of(&x, &y), 10.0).is_none(), "no slope from a constant column");
    }

    #[test]
    fn non_finite_rows_are_dropped_not_poisoned() {
        // y = 2x + 1, but a few rows have a NaN subtree value (a pole). The fit
        // over the finite rows is still exact.
        let mut x: Vec<f32> = (0..300).map(|i| i as f32 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * f64::from(v) + 1.0).collect();
        x[10] = f32::NAN;
        x[200] = f32::INFINITY;
        let s = ColumnSums::of(&x, &y);
        assert_eq!(s.n, 298.0, "two non-finite rows dropped");
        let r = affine_residual(s, 10.0).expect("a fit");
        assert!(r < 1e-9, "finite rows fit exactly: 1 - R² = {r}");
    }

    #[test]
    fn score_subtrees_indexes_the_partials_layout() {
        // Two subtrees in one expr's partials, max_nodes 4, 3 rows.
        // node0 = target itself (score 0), node1 = unrelated.
        let max_nodes = 4;
        let n_rows = 3;
        let target = vec![1.0, 2.0, 3.0];
        let mut partials = vec![0.0f32; max_nodes * n_rows];
        // expr 0, node 0: exactly the target → affine image, 1-R²=0.
        partials[0..n_rows].copy_from_slice(&[1.0, 2.0, 3.0]);
        // expr 0, node 1: constant → None.
        partials[n_rows..2 * n_rows].copy_from_slice(&[5.0, 5.0, 5.0]);
        let scores = score_subtrees(&partials, 0, 2, max_nodes, n_rows, &target, 2.0);
        assert_eq!(scores.len(), 2);
        assert!(scores[0].unwrap() < 1e-9, "node 0 is the target");
        assert!(scores[1].is_none(), "node 1 is constant");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_verify_matches_the_host_reduction() {
        use crate::gpu_eval::{ExprBatch, GpuEvaluator, GpuNode, Op};
        // 4 rows, 2 vars. Two CANDIDATE columns: c0 = a (ratio to pattern a is
        // constant 1 -> CoV 0), c1 = a*b (ratio to a is b, which varies -> CoV
        // large). One PATTERN column p0 = a.
        let rows: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n_rows = 4u32;
        let ev = match GpuEvaluator::new(&rows, 2) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let var_a = [GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 }];
        let mul_ab = [
            GpuNode { op: Op::Mul as u32, arg0: 1, arg1: 2, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 0, arg1: 0, konst: 0.0 },
            GpuNode { op: Op::Var as u32, arg0: 1, arg1: 0, konst: 0.0 },
        ];
        // Batch: candidate 0 (a), candidate 1 (a*b), pattern 0 (a). n_cand=2, n_pat=1.
        let mut batch = ExprBatch::new();
        batch.push(&var_a);
        batch.push(&mul_ab);
        batch.push(&var_a);
        let preds = ev.eval_resident(&batch).expect("eval_resident");
        // Host reference: eval the same batch, score each candidate's ratio to the
        // pattern with the ratio_cov reference the GPU kernel mirrors.
        let flat = ev.eval(&batch).expect("eval");
        let col = |e: usize| &flat[e * n_rows as usize..(e + 1) * n_rows as usize];
        let host0 = ratio_cov(col(0), col(2), 2).expect("host c0");
        let host1 = ratio_cov(col(1), col(2), 2).expect("host c1");

        let pv = PatternVerify::new(&ev);
        let got = pv.verify(&ev, &preds, 2, 1, n_rows, 2).expect("gpu verify");
        preds.destroy();
        assert_eq!(got.len(), 2, "n_cand * n_pat pair scores");
        // out[cand * n_pat + pat]; pat=0 so index == cand.
        assert!((f64::from(got[0]) - host0).abs() < 1e-4, "GPU c0 {} vs host {host0}", got[0]);
        assert!((f64::from(got[1]) - host1).abs() < 1e-4, "GPU c1 {} vs host {host1}", got[1]);
        assert!(f64::from(got[0]) < 1e-5, "c0 (ratio to pattern a is constant): {}", got[0]);
        assert!(f64::from(got[1]) > 1e-2, "c1 (ratio to a varies): {}", got[1]);
    }
}
