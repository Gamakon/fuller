//! Scoring on the device (`score.wgsl`): the evaluator's predictions are
//! linked, wrapped, scaled and reduced to metrics where they already are, and
//! only `candidates x 9` numbers come back — 260 KB a generation for a
//! population of 800 instead of 55 MB of predictions.
//!
//! The kernel is f32 and RANKS; `chrom_score` in f64 remains the definition,
//! and whatever is reported or stops a fit is re-scored with it on the host.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::chrom_score::Splits;
use crate::gpu_eval::{ExprBatch, GpuEvaluator};

pub const SCORE_WGSL: &str = include_str!("score.wgsl");
/// Candidates per chromosome: 3 linkers (avg, mul, add) x 3 wrappers
/// (identity, log_abs, sqrt_abs), in that order.
pub const CANDIDATES: usize = 9;
/// `[a, b, mse_train, mse_val, max_err_val, mse_extrap, mae_train, mae_val, mae_extrap,
/// redundancy]`. The first [`METRICS`] are chrom_score's; `redundancy` is the
/// kernel's leave-one-gene-out score in [0, 1] (0 = every varying gene matters).
pub const WIDTH: usize = 10;
pub const METRICS: usize = 9;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta {
    n_chromosomes: u32,
    genes_per: u32,
    n_rows: u32,
    n_train: u32,
    n_val: u32,
    n_extrap: u32,
    y_mean_train: f32,
    pad0: u32,
}

pub struct GpuScorer {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    y_buf: wgpu::Buffer,
    splits: Splits,
    y_mean_train: f32,
}

impl GpuScorer {
    /// `y`: the targets in row order (train, validation, edge), resident from here on.
    pub fn new(evaluator: &GpuEvaluator, y: &[f64], splits: Splits) -> Result<GpuScorer, String> {
        if y.len() != splits.total() || y.len() != evaluator.n_rows() as usize {
            return Err(format!("y has {} values, the splits {} and the evaluator {} rows", y.len(), splits.total(), evaluator.n_rows()));
        }
        if splits.n_train == 0 || splits.n_val == 0 {
            return Err("train and validation need rows".to_string());
        }
        let device = evaluator.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fuller-score"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCORE_WGSL)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fuller-score-layout"),
            entries: &(0..6)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match i {
                            0 => wgpu::BufferBindingType::Uniform,
                            _ => wgpu::BufferBindingType::Storage { read_only: i != 5 },
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
            label: Some("fuller-score-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "score_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let y32: Vec<f32> = y.iter().map(|v| *v as f32).collect();
        let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("y-resident"),
            contents: bytemuck::cast_slice(&y32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let y_mean_train = (y[..splits.n_train].iter().sum::<f64>() / splits.n_train as f64) as f32;
        Ok(GpuScorer { pipeline, layout, y_buf, splits, y_mean_train })
    }

    /// Evaluate `batch` and score every chromosome's nine candidates without the
    /// predictions leaving the device. `chromosomes` are gene indices into the
    /// batch. Returns `chromosomes.len() * CANDIDATES * WIDTH` values,
    /// chromosome-major, then linker, then wrapper; a rejected candidate is NaN.
    pub fn eval_and_score(
        &self,
        evaluator: &GpuEvaluator,
        batch: &ExprBatch,
        gene_ok: &[bool],
        chromosomes: &[Vec<usize>],
    ) -> Result<Vec<f32>, String> {
        if chromosomes.is_empty() {
            return Ok(Vec::new());
        }
        let genes_per = chromosomes[0].len();
        if genes_per == 0 || chromosomes.iter().any(|c| c.len() != genes_per) {
            return Err("every chromosome needs the same, non-zero, number of genes".to_string());
        }
        if gene_ok.len() != batch.len() || chromosomes.iter().flatten().any(|&g| g >= batch.len()) {
            return Err("gene_ok and the chromosomes must index the batch".to_string());
        }
        let (device, queue) = (evaluator.device(), evaluator.queue());
        let preds = evaluator.eval_resident(batch)?;
        let storage = |v: &[u32], label| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let ok: Vec<u32> = gene_ok.iter().map(|&b| u32::from(b)).collect();
        let flat: Vec<u32> = chromosomes.iter().flatten().map(|&g| g as u32).collect();
        let (ok_buf, chrom_buf) = (storage(&ok, "gene_ok"), storage(&flat, "chromosomes"));
        let meta = Meta {
            n_chromosomes: chromosomes.len() as u32,
            genes_per: genes_per as u32,
            n_rows: evaluator.n_rows(),
            n_train: self.splits.n_train as u32,
            n_val: self.splits.n_val as u32,
            n_extrap: self.splits.n_extrap as u32,
            y_mean_train: self.y_mean_train,
            pad0: 0,
        };
        let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("score-meta"),
            contents: bytemuck::bytes_of(&meta),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let size = (chromosomes.len() * CANDIDATES * WIDTH * 4) as u64;
        let scores_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let buffers = [&meta_buf, &preds, &ok_buf, &chrom_buf, &self.y_buf, &scores_buf];
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.layout, entries: &entries });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((chromosomes.len() as u32 * 3).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&scores_buf, 0, &staging, 0, size);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("map_async channel: {e}"))?
            .map_err(|e| format!("map_async: {e}"))?;
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        for b in [&preds, &ok_buf, &chrom_buf, &meta_buf, &scores_buf, &staging] {
            b.destroy();
        }
        device.poll(wgpu::Maintain::Poll);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chrom_score::{score_chromosomes, Linker, ScoreSpec, Wrapper, SCORE_WIDTH};
    use crate::gpu_eval::math_to_nodes;

    /// The device's f32 metrics against chrom_score's f64 ones, from the SAME
    /// device predictions: same candidates accepted and rejected, the scale and
    /// every error within f32's reach of the f64 value.
    #[test]
    fn device_scores_agree_with_the_f64_scorer() {
        let names = vec!["x".to_string(), "y".to_string()];
        let n = 600usize;
        let rows: Vec<f32> = (0..n).flat_map(|i| [0.5 + i as f32 * 0.01, 2.0 + ((i * 7) % 13) as f32 * 0.3]).collect();
        let truth: Vec<f64> = rows.chunks(2).map(|r| 3.0 * f64::from(r[0]) * f64::from(r[1]) + 1.5).collect();
        let exprs = [
            r#"(Mul (Var "x") (Var "y"))"#,
            r#"(Add (Var "x") (Var "y"))"#,
            r#"(Sin (Var "x"))"#,
            r#"(Num 2.0)"#,
            r#"(ProtectedLog (Sub (Var "x") (Var "x")))"#,
            r#"(Pow3 (Var "y"))"#,
            r#"(Sin (Mul (Num 50.0) (Var "x")))"#,
        ];
        let mut batch = ExprBatch::new();
        for e in exprs {
            batch.push(&math_to_nodes(e, &names).unwrap());
        }
        let gene_ok = vec![true, true, true, true, true, false, true];
        let chromosomes: Vec<Vec<usize>> = vec![vec![0, 3, 3], vec![0, 1, 2], vec![1, 1, 1], vec![3, 3, 3], vec![4, 0, 1], vec![5, 0, 1], vec![2, 0, 3], vec![6, 0, 3]];
        let splits = Splits { n_train: 400, n_val: 120, n_extrap: 80 };
        let evaluator = GpuEvaluator::new(&rows, 2).expect("evaluator");
        let scorer = GpuScorer::new(&evaluator, &truth, splits).expect("scorer");
        let on_device = scorer.eval_and_score(&evaluator, &batch, &gene_ok, &chromosomes).expect("score");

        let preds = evaluator.eval(&batch).expect("eval");
        let spec = ScoreSpec {
            linkers: vec![Linker::AVG, Linker::MUL, Linker::ADD],
            wrappers: vec![Wrapper::Identity, Wrapper::LogAbs, Wrapper::SqrtAbs],
            splits,
            linear_scaling: true,
        };
        let reference = score_chromosomes(&preds, &gene_ok, &chromosomes, &truth, &spec).expect("reference");
        let (mut compared, mut rejected) = (0, 0);
        for c in 0..chromosomes.len() * CANDIDATES {
            let want = &reference[c * SCORE_WIDTH..c * SCORE_WIDTH + METRICS];
            let got = &on_device[c * WIDTH..(c + 1) * WIDTH];
            assert_eq!(want[0].is_finite(), got[0].is_finite(), "candidate {c}: accepted on one side only ({want:?} vs {got:?})");
            if !want[0].is_finite() {
                rejected += 1;
                continue;
            }
            compared += 1;
            // f32 resolves a residual to about 1e-4 of the target's size: below
            // that floor both sides are "zero" and only the floor is compared.
            let y_size = truth.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for k in 0..METRICS {
                let (w, g) = (want[k], f64::from(got[k]));
                let floor = match k {
                    0 | 1 => 1e-3,
                    2 | 3 | 5 => (1e-4 * y_size).powi(2),
                    _ => 1e-4 * y_size,
                };
                assert!(
                    (w - g).abs() < 2e-3 * w.abs() + floor,
                    "candidate {c} value {k}: f64 {w} against device {g}"
                );
            }
        }
        assert!(compared >= 20 && rejected >= 9, "compared {compared}, rejected {rejected}");
        // REDUNDANCY. Chromosome 0 = (x*y, 2, 2) under mul: one varying gene and it
        // is the whole model -> 0. Chromosome 7 = (sin 50x, x*y, 2) under ADD: x*y
        // carries the fit and the fast sine nothing -> the median of (about 1,
        // about 0) puts it near one half.
        let redundancy = |chromosome: usize, candidate: usize| f64::from(on_device[(chromosome * CANDIDATES + candidate) * WIDTH + 9]);
        assert!(redundancy(0, 3) < 0.01, "{}", redundancy(0, 3));
        assert!((0.4..0.6).contains(&redundancy(7, 6)), "{}", redundancy(7, 6));
        // Chromosome 0 is (x*y, 2, 2): under the mul linker it is 4xy, which the
        // scale turns into the target exactly. Candidate 3 = linker mul, identity.
        let exact = &on_device[3 * WIDTH..4 * WIDTH];
        assert!(f64::from(exact[2]) < 1e-6 && f64::from(exact[3]) < 1e-6, "{exact:?}");
    }
}
