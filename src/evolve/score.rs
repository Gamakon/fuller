//! Scoring on the device (`score.wgsl`): the evaluator's predictions are
//! linked, wrapped, scaled and reduced to metrics where they already are, and
//! only `candidates x 9` numbers come back — 260 KB a generation for a
//! population of 800 instead of 55 MB of predictions.
//!
//! The kernel is f32 and RANKS; `chrom_score` in f64 remains the definition,
//! and whatever is reported or stops a fit is re-scored with it on the host.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::chrom_score::{gene_linker_combinations, GeneLinker, Splits};
use crate::gpu_eval::{ExprBatch, GpuEvaluator};

pub const SCORE_WGSL: &str = include_str!("score.wgsl");
/// Candidates per chromosome: 3 linkers (avg, mul, add) x 3 wrappers
/// (identity, log_abs, sqrt_abs), in that order. With the gene-subset choice it
/// is `combinations x 3` — see [`GpuScorer::eval_and_score_with`].
pub const CANDIDATES: usize = 9;
/// The kernel's linkers (avg, mul, add) and wrappers (identity, log_abs, sqrt_abs).
pub const LINKERS: usize = 3;
pub const WRAPPERS: usize = 3;
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
    /// 0, which the shader compiler cannot know: see `score.wgsl::keep`.
    zero: u32,
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
            entries: &(0..7)
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
        let Some(first) = chromosomes.first() else { return Ok(Vec::new()) };
        if first.is_empty() {
            return Err("every chromosome needs the same, non-zero, number of genes".to_string());
        }
        let combinations = gene_linker_combinations(first.len(), LINKERS, false)?;
        self.eval_and_score_with(evaluator, batch, gene_ok, chromosomes, &combinations)
    }

    /// [`GpuScorer::eval_and_score`] under the given GENE-LINKER COMBINATIONS
    /// (`chrom_score::gene_linker_combinations`): `combinations x 3` candidates a
    /// chromosome — chromosome-major, then combination, then wrapper. A gene that
    /// failed to evaluate rejects only the combinations that use it, and
    /// `redundancy` is over a combination's used genes.
    pub fn eval_and_score_with(
        &self,
        evaluator: &GpuEvaluator,
        batch: &ExprBatch,
        gene_ok: &[bool],
        chromosomes: &[Vec<usize>],
        combinations: &[GeneLinker],
    ) -> Result<Vec<f32>, String> {
        if chromosomes.is_empty() {
            return Ok(Vec::new());
        }
        let genes_per = chromosomes[0].len();
        if genes_per == 0 || chromosomes.iter().any(|c| c.len() != genes_per) {
            return Err("every chromosome needs the same, non-zero, number of genes".to_string());
        }
        if genes_per > 24 {
            return Err(format!("{genes_per} genes: the kernel packs a chromosome's used genes into 24 bits"));
        }
        if combinations.is_empty() || combinations.iter().any(|c| c.genes == 0 || c.genes >= 1 << genes_per || c.linker >= LINKERS) {
            return Err(format!("the combinations do not fit chromosomes of {genes_per} genes and {LINKERS} linkers"));
        }
        let threads = chromosomes.len() * combinations.len();
        if threads.div_ceil(64) > 65_535 {
            return Err(format!("{} chromosomes x {} combinations is more than one dispatch holds", chromosomes.len(), combinations.len()));
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
        let packed: Vec<u32> = combinations.iter().map(|c| c.genes | (c.linker as u32) << 24).collect();
        let (ok_buf, chrom_buf, comb_buf) = (storage(&ok, "gene_ok"), storage(&flat, "chromosomes"), storage(&packed, "combinations"));
        let meta = Meta {
            n_chromosomes: chromosomes.len() as u32,
            genes_per: genes_per as u32,
            n_rows: evaluator.n_rows(),
            n_train: self.splits.n_train as u32,
            n_val: self.splits.n_val as u32,
            n_extrap: self.splits.n_extrap as u32,
            y_mean_train: self.y_mean_train,
            zero: 0,
        };
        let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("score-meta"),
            contents: bytemuck::bytes_of(&meta),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let size = (threads * WRAPPERS * WIDTH * 4) as u64;
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
        let buffers = [&meta_buf, &preds, &ok_buf, &chrom_buf, &self.y_buf, &scores_buf, &comb_buf];
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
            pass.dispatch_workgroups((threads as u32).div_ceil(64), 1, 1);
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
        for b in [&preds, &ok_buf, &chrom_buf, &comb_buf, &meta_buf, &scores_buf, &staging] {
            b.destroy();
        }
        device.poll(wgpu::Maintain::Poll);
        Ok(out)
    }
}

/// The mean of the train targets as the kernel is given it.
fn y_mean_train_f32(y: &[f64], splits: Splits) -> f32 {
    (y[..splits.n_train].iter().sum::<f64>() / splits.n_train as f64) as f32
}

/// `score.wgsl`'s `add`. With `compensated` off it is the plain sum — what a
/// compiler that reassociates the (algebraically zero) compensation away leaves.
fn add_f32(s: (f32, f32), v: f32, compensated: bool) -> (f32, f32) {
    let t = s.0 + v;
    if !compensated {
        return (t, 0.0);
    }
    let c = if s.0.abs() >= v.abs() { s.1 + ((s.0 - t) + v) } else { s.1 + ((v - t) + s.0) };
    (t, c)
}

/// `score.wgsl`'s `div`: the quotient, corrected by its exactly formed residual
/// (Dekker's product by Veltkamp's split). On the host `x / y` is already
/// correctly rounded and the correction changes nothing; the steps are the
/// kernel's so that the two agree where they could differ (overflow, underflow).
fn div_f32(x: f32, y: f32) -> f32 {
    let q = x / y;
    let p = q * y;
    let (qh, yh) = (high_f32(q), high_f32(y));
    let (ql, yl) = (q - qh, y - yh);
    let e = ql * yl - (((p - qh * yh) - ql * yh) - qh * yl);
    let r = (x - p) - e;
    if !r.is_finite() || !q.is_finite() {
        return q;
    }
    q + r / y
}

/// Veltkamp's split: the high part of `v` for Dekker's exact product.
fn high_f32(v: f32) -> f32 {
    let c = 4097.0 * v;
    c - (c - v)
}

/// `score.wgsl`'s `root`: the square root corrected by its exactly formed residual.
fn root_f32(x: f32) -> f32 {
    let s = x.sqrt();
    if s == 0.0 || !s.is_finite() {
        return s;
    }
    let p = s * s;
    let sh = high_f32(s);
    let sl = s - sh;
    let e = sl * sl - (((p - sh * sh) - sl * sh) - sh * sl);
    let r = (x - p) - e;
    if !r.is_finite() {
        return s;
    }
    s + r / (2.0 * s)
}

/// `score.wgsl`'s `wrapped`, by the host's libm.
fn wrapped_f32(v: f32, w: usize) -> f32 {
    match w {
        1 => (v.abs() + 1e-12).ln(),
        2 => root_f32(v.abs()),
        _ => v,
    }
}

/// `score.wgsl`'s `linked` / `linked_without`: `genes` under `linker` (0 avg,
/// 1 mul, 2 add) on `row`, the gene at position `skip` held at `held`.
fn linked_f32(preds: &[f32], n_rows: usize, genes: &[usize], linker: usize, row: usize, skip: Option<(usize, f32)>) -> f32 {
    let mut acc: f32 = if linker == 1 { 1.0 } else { 0.0 };
    for (g, &gene) in genes.iter().enumerate() {
        let v = match skip {
            Some((at, held)) if at == g => held,
            _ => preds[gene * n_rows + row],
        };
        acc = if linker == 1 { acc * v } else { acc + v };
    }
    if linker == 0 {
        acc = div_f32(acc, genes.len() as f32);
    }
    acc
}

/// What one thread of `score.wgsl` computes: `genes` under `linker`, the three
/// wrappers, `WIDTH` values each.
fn score_thread_f32(preds: &[f32], genes: &[usize], linker: usize, y: &[f32], splits: Splits, y_mean: f32, compensated: bool) -> [f32; 3 * WIDTH] {
    const LOO_STRIDE: usize = 4;
    const CONSTANT_REL_TOL: f32 = 2e-6;
    let mut out = [f32::NAN; 3 * WIDTH];
    let n_rows = splits.total();
    let (nt, v1) = (splits.n_train, splits.n_train + splits.n_val);
    let link = |row: usize| linked_f32(preds, n_rows, genes, linker, row, None);
    let add = |s: (f32, f32), v: f32| add_f32(s, v, compensated);

    // Pass 1: the mean as offsets from the first row's value, and the range.
    let v0 = link(0);
    let x0: [f32; 3] = std::array::from_fn(|w| wrapped_f32(v0, w));
    let (mut sum, mut ok, mut lo, mut hi) = ([(0.0f32, 0.0f32); 3], [true; 3], x0, x0);
    for row in 0..n_rows {
        let v = link(row);
        for w in 0..3 {
            let x = wrapped_f32(v, w);
            if !v.is_finite() || !x.is_finite() {
                ok[w] = false;
            } else if row < nt {
                sum[w] = add(sum[w], x - x0[w]);
                lo[w] = lo[w].min(x);
                hi[w] = hi[w].max(x);
            }
        }
    }
    let mx: [f32; 3] = std::array::from_fn(|w| x0[w] + div_f32(sum[w].0 + sum[w].1, nt as f32));

    // Pass 2: centred sums for the least-squares scale.
    let (mut sxx, mut sxy) = ([(0.0f32, 0.0f32); 3], [(0.0f32, 0.0f32); 3]);
    for (row, target) in y.iter().enumerate().take(nt) {
        let v = link(row);
        let dy = target - y_mean;
        for w in 0..3 {
            if ok[w] {
                let dx = wrapped_f32(v, w) - mx[w];
                sxx[w] = add(sxx[w], dx * dx);
                sxy[w] = add(sxy[w], dx * dy);
            }
        }
    }
    let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
    for w in 0..3 {
        let xx = sxx[w].0 + sxx[w].1;
        if !ok[w] || (hi[w] - lo[w]) <= 2.0 * (1e-8 + CONSTANT_REL_TOL * mx[w].abs()) || xx <= 0.0 {
            ok[w] = false;
            continue;
        }
        a[w] = div_f32(sxy[w].0 + sxy[w].1, xx);
        b[w] = y_mean - a[w] * mx[w];
        if !a[w].is_finite() || !b[w].is_finite() {
            ok[w] = false;
        }
    }

    // Pass 3: squared and absolute residuals per split.
    let (mut sq, mut ab, mut worst) = ([(0.0f32, 0.0f32); 9], [(0.0f32, 0.0f32); 9], [0.0f32; 3]);
    for (row, target) in y.iter().enumerate().take(n_rows) {
        let v = link(row);
        let split = if row < nt { 0 } else if row < v1 { 1 } else { 2 };
        for w in 0..3 {
            if ok[w] {
                let d = target - (a[w] * wrapped_f32(v, w) + b[w]);
                sq[w * 3 + split] = add(sq[w * 3 + split], d * d);
                ab[w * 3 + split] = add(ab[w * 3 + split], d.abs());
                if split == 1 {
                    worst[w] = worst[w].max(d.abs());
                }
            }
        }
    }
    let n = [nt as f32, splits.n_val as f32, splits.n_extrap.max(1) as f32];
    for w in 0..3 {
        if !ok[w] {
            continue;
        }
        let total = |s: (f32, f32), k: usize| div_f32(s.0 + s.1, n[k]);
        let o = &mut out[w * WIDTH..(w + 1) * WIDTH];
        o.copy_from_slice(&[
            a[w], b[w], total(sq[w * 3], 0), total(sq[w * 3 + 1], 1), worst[w], total(sq[w * 3 + 2], 2),
            total(ab[w * 3], 0), total(ab[w * 3 + 1], 1), total(ab[w * 3 + 2], 2), 0.0,
        ]);
        if o.iter().any(|v| !v.is_finite()) {
            o.fill(f32::NAN);
            ok[w] = false;
        }
    }

    // Leave one gene out — see the kernel; a single gene is the whole model.
    if genes.len() == 1 {
        return out;
    }
    let mut syy = (0.0f32, 0.0f32);
    for row in (0..nt).step_by(LOO_STRIDE) {
        let dy = y[row] - y_mean;
        syy = add(syy, dy * dy);
    }
    let yy = syy.0 + syy.1;
    let mut loss: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (g, &gene) in genes.iter().enumerate().take(8) {
        let g0 = preds[gene * n_rows];
        let (mut gsum, mut gmin, mut ghi, mut n_read) = ((0.0f32, 0.0f32), g0, g0, 0.0f32);
        for row in (0..nt).step_by(LOO_STRIDE) {
            let v = preds[gene * n_rows + row];
            gsum = add(gsum, v - g0);
            gmin = gmin.min(v);
            ghi = ghi.max(v);
            n_read += 1.0;
        }
        let gmean = g0 + div_f32(gsum.0 + gsum.1, n_read);
        if (ghi - gmin) <= 2.0 * (1e-8 + CONSTANT_REL_TOL * gmean.abs()) {
            continue;
        }
        let without = |row: usize| linked_f32(preds, n_rows, genes, linker, row, Some((g, gmean)));
        let mut m1 = [(0.0f32, 0.0f32); 3];
        let first = without(0);
        let (mut lo1, mut hi1): ([f32; 3], [f32; 3]) = (std::array::from_fn(|w| wrapped_f32(first, w)), std::array::from_fn(|w| wrapped_f32(first, w)));
        for row in (0..nt).step_by(LOO_STRIDE) {
            let v = without(row);
            for (w, m) in m1.iter_mut().enumerate() {
                let x = wrapped_f32(v, w);
                *m = add(*m, x);
                lo1[w] = lo1[w].min(x);
                hi1[w] = hi1[w].max(x);
            }
        }
        let (mut xx, mut xy) = ([(0.0f32, 0.0f32); 3], [(0.0f32, 0.0f32); 3]);
        for row in (0..nt).step_by(LOO_STRIDE) {
            let v = without(row);
            let dy = y[row] - y_mean;
            for w in 0..3 {
                let dx = wrapped_f32(v, w) - div_f32(m1[w].0 + m1[w].1, n_read);
                xx[w] = add(xx[w], dx * dx);
                xy[w] = add(xy[w], dx * dy);
            }
        }
        for w in 0..3 {
            if !ok[w] {
                continue;
            }
            let full = 1.0 - div_f32(out[w * WIDTH + 2] * nt as f32, (yy * LOO_STRIDE as f32).max(1e-30));
            let (sxx_w, sxy_w) = (xx[w].0 + xx[w].1, xy[w].0 + xy[w].1);
            let mut r2_without = 0.0f32;
            let mean1 = div_f32(m1[w].0 + m1[w].1, n_read);
            let varying = (hi1[w] - lo1[w]) > 2.0 * (1e-8 + CONSTANT_REL_TOL * mean1.abs());
            if varying && sxx_w > 0.0 && yy > 0.0 && sxx_w.is_finite() && sxy_w.is_finite() {
                r2_without = div_f32(sxy_w * sxy_w, sxx_w * yy).clamp(0.0, 1.0);
            }
            loss[w].push(if full > 1e-6 { (1.0 - div_f32(r2_without, full)).clamp(0.0, 1.0) } else { 1.0 });
        }
    }
    for w in 0..3 {
        if !ok[w] || loss[w].is_empty() {
            continue;
        }
        loss[w].sort_by(f32::total_cmp);
        let n = loss[w].len();
        out[w * WIDTH + 9] = 1.0 - 0.5 * (loss[w][(n - 1) / 2] + loss[w][n / 2]);
    }
    out
}

/// THE CPU TWIN of `score.wgsl`: the same procedure in f32, pass for pass and sum
/// for sum, over predictions the device produced — what the kernel is checked
/// against bit for bit (where no transcendental of the device's is involved: its
/// `log` is not libm's). Same layout as [`GpuScorer::eval_and_score`].
/// `compensated` = false switches Neumaier's compensation off: the sum a compiler
/// leaves when it reassociates the compensation away, for the test that tells
/// which of the two the device computes.
pub fn score_f32(preds: &[f32], gene_ok: &[bool], chromosomes: &[Vec<usize>], y: &[f64], splits: Splits, compensated: bool) -> Vec<f32> {
    let all = chromosomes.first().map_or(1, |genes| (1u32 << genes.len()) - 1);
    let combinations: Vec<GeneLinker> = (0..LINKERS).map(|linker| GeneLinker { genes: all, linker }).collect();
    score_f32_with(preds, gene_ok, chromosomes, y, splits, &combinations, compensated)
}

/// [`score_f32`] under the given gene-linker combinations: the twin of
/// [`GpuScorer::eval_and_score_with`], same layout. A combination is scored as
/// the chromosome of its used genes alone, as `chrom_score::score_gene_subsets`
/// defines it.
pub fn score_f32_with(
    preds: &[f32],
    gene_ok: &[bool],
    chromosomes: &[Vec<usize>],
    y: &[f64],
    splits: Splits,
    combinations: &[GeneLinker],
    compensated: bool,
) -> Vec<f32> {
    let y32: Vec<f32> = y.iter().map(|v| *v as f32).collect();
    let y_mean = y_mean_train_f32(y, splits);
    let mut out = Vec::with_capacity(chromosomes.len() * combinations.len() * WRAPPERS * WIDTH);
    for genes in chromosomes {
        for combination in combinations {
            let used: Vec<usize> = combination.positions().map(|g| genes[g]).collect();
            if used.iter().any(|&g| !gene_ok[g]) {
                out.extend([f32::NAN; 3 * WIDTH]);
            } else {
                out.extend(score_thread_f32(preds, &used, combination.linker, &y32, splits, y_mean, compensated));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chrom_score::{score_chromosomes, Linker, ScoreSpec, Wrapper, SCORE_WIDTH};
    use crate::gpu_eval::math_to_nodes;


    /// A unit draw in [0, 1), for building test data.
    fn unit(i: usize, stream: u32) -> f64 {
        f64::from(crate::evolve::below(crate::evolve::draw(99, 0, i as u32, 0, stream), 1 << 24)) / f64::from(1u32 << 24)
    }

    /// IS THE COMPENSATED SUM ALIVE ON THE DEVICE? Neumaier's compensation is
    /// algebraically zero, and a shader compiler that treats floats as reals may
    /// remove it. The case: y = 2x exactly, x in [0.5, 1.5), 40,000 train rows, no
    /// offset — with compensated sums f32 recovers a = 2, b = 0 and a residual of
    /// exactly 0; with plain sums the mean and the scale drift (a = 1.9999992) and
    /// 1 - R² is 4.4e-12 instead of 0. The identity wrapper involves no
    /// transcendental, so the device must equal ONE of the two twins bit for bit.
    #[test]
    fn the_compensated_sum_is_alive_on_the_device() {
        let n = 50_000usize;
        let x: Vec<f32> = (0..n).map(|i| (0.5 + unit(i, 7)) as f32).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * f64::from(*v)).collect();
        let splits = Splits { n_train: 40_000, n_val: 10_000, n_extrap: 0 };
        let var = {
            let t = &y[..splits.n_train];
            let m = t.iter().sum::<f64>() / t.len() as f64;
            t.iter().map(|v| (v - m).powi(2)).sum::<f64>() / t.len() as f64
        };
        let names = vec!["x".to_string()];
        let mut batch = ExprBatch::new();
        batch.push(&math_to_nodes(r#"(Var "x")"#, &names).unwrap());
        let chromosomes = vec![vec![0usize]];
        let evaluator = GpuEvaluator::new(&x, 1).expect("evaluator");
        let scorer = GpuScorer::new(&evaluator, &y, splits).expect("scorer");
        let device = scorer.eval_and_score(&evaluator, &batch, &[true], &chromosomes).expect("score");
        let preds = evaluator.eval(&batch).expect("eval");
        assert_eq!(preds, x, "the device's predictions of x are x");
        let compensated = score_f32(&preds, &[true], &chromosomes, &y, splits, true);
        let plain = score_f32(&preds, &[true], &chromosomes, &y, splits, false);
        let omr2 = |s: &[f32]| f64::from(s[2]) / var;
        // The two sums differ by orders of magnitude on this case (or it proves nothing).
        assert!(omr2(&compensated) < 1e-15 && omr2(&plain) > 1e-12, "compensated {:e}, plain {:e}", omr2(&compensated), omr2(&plain));
        let bits = |s: &[f32]| s[..METRICS].iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        // Candidate 0: the avg linker of one gene (the gene itself), identity wrapper.
        let (d, c, p) = (&device[..WIDTH], &compensated[..WIDTH], &plain[..WIDTH]);
        assert_eq!(bits(d), bits(c), "the device is not the compensated twin: {d:?} against {c:?}");
        assert_ne!(bits(d), bits(p), "the device is the PLAIN sum: its compensation is dead");
    }

    /// A few hundred chromosomes of three genes drawn from a mixed bag (laws,
    /// junk, a constant, a gene that is not finite, one that failed to decode).
    struct Bag {
        rows: Vec<f32>,
        truth: Vec<f64>,
        splits: Splits,
        batch: ExprBatch,
        gene_ok: Vec<bool>,
        chromosomes: Vec<Vec<usize>>,
    }

    fn mixed_bag() -> Bag {
        let names = vec!["x".to_string(), "y".to_string()];
        let n = 3_000usize;
        let rows: Vec<f32> = (0..n).flat_map(|i| [(0.5 + 4.0 * unit(i, 1)) as f32, (1.0 + 2.0 * unit(i, 2)) as f32]).collect();
        let truth: Vec<f64> = rows.chunks(2).map(|r| 3.0 * f64::from(r[0]) * f64::from(r[1]) + f64::from(r[0]) + 1.5).collect();
        let exprs = [
            r#"(Mul (Var "x") (Var "y"))"#,
            r#"(Var "x")"#,
            r#"(Add (Var "x") (Var "y"))"#,
            r#"(Sin (Var "x"))"#,
            r#"(Num 2.0)"#,
            r#"(ProtectedLog (Sub (Var "x") (Var "x")))"#,
            r#"(Pow3 (Var "y"))"#,
            r#"(Sin (Mul (Num 50.0) (Var "x")))"#,
            r#"(Sub (Var "y") (Var "x"))"#,
            r#"(Div (Var "x") (Var "y"))"#,
            r#"(Exp (Var "y"))"#,
            r#"(Mul (Num -0.37) (Cos (Var "y")))"#,
        ];
        let mut batch = ExprBatch::new();
        for e in exprs {
            batch.push(&math_to_nodes(e, &names).unwrap());
        }
        let gene_ok: Vec<bool> = (0..exprs.len()).map(|g| g != 6).collect();
        let pick = |c: usize, g: u32| crate::evolve::below(crate::evolve::draw(5, 1, c as u32, g, 9), exprs.len() as u32) as usize;
        let chromosomes: Vec<Vec<usize>> = (0..300).map(|c| (0..3).map(|g| pick(c, g)).collect()).collect();
        Bag { rows, truth, splits: Splits { n_train: 2_000, n_val: 600, n_extrap: 400 }, batch, gene_ok, chromosomes }
    }

    /// THE KERNEL AGAINST ITS CPU TWIN, from the device's own predictions: the same
    /// candidates accepted and rejected, and under the identity and sqrt wrappers
    /// every value — scale, offset, the seven errors, redundancy — BIT FOR BIT.
    /// (Add, subtract and multiply are IEEE once `keep`-ed; the device's division
    /// and square root are not correctly rounded, and the kernel corrects them.)
    /// The log wrapper is the one place the device's own arithmetic stays in: its
    /// `log` is not libm's. Measured here, that moves the scale by 9e-7 of itself,
    /// an error by 1.4e-6, the offset (a difference of two large numbers) by
    /// 1.4e-5, and redundancy (a ratio of two R², then a median) by 4e-3; the
    /// bounds asserted are those with headroom, not 2e-3 of everything.
    #[test]
    fn device_scores_equal_the_f32_twin() {
        let Bag { rows, truth, splits, batch, gene_ok, chromosomes } = mixed_bag();
        let evaluator = GpuEvaluator::new(&rows, 2).expect("evaluator");
        let scorer = GpuScorer::new(&evaluator, &truth, splits).expect("scorer");
        let device = scorer.eval_and_score(&evaluator, &batch, &gene_ok, &chromosomes).expect("score");
        let preds = evaluator.eval(&batch).expect("eval");
        let twin = score_f32(&preds, &gene_ok, &chromosomes, &truth, splits, true);
        let (exact, through_log, rejected) = assert_device_is_twin(&device, &twin);
        assert!(exact >= 900 && through_log >= 450 && rejected >= 300, "bit for bit {exact}, through the log {through_log}, rejected {rejected}");
    }

    /// The parity rule of `device_scores_equal_the_f32_twin`, candidate for
    /// candidate (the wrapper is the candidate's index mod 3). Returns how many
    /// were compared bit for bit, how many through the log, how many both rejected.
    fn assert_device_is_twin(device: &[f32], twin: &[f32]) -> (usize, usize, usize) {
        assert_eq!(device.len(), twin.len());
        let (mut exact, mut through_log, mut rejected) = (0, 0, 0);
        for c in 0..device.len() / WIDTH {
            let (d, t) = (&device[c * WIDTH..(c + 1) * WIDTH], &twin[c * WIDTH..(c + 1) * WIDTH]);
            assert_eq!(d[0].is_finite(), t[0].is_finite(), "candidate {c}: accepted on one side only ({d:?} vs {t:?})");
            if !d[0].is_finite() {
                rejected += 1;
                continue;
            }
            if c % 3 != 1 {
                exact += 1;
                let bits = |s: &[f32]| s.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(d), bits(t), "candidate {c}: device {d:?} against twin {t:?}");
                continue;
            }
            through_log += 1;
            for k in 0..WIDTH {
                let (dv, tv) = (f64::from(d[k]), f64::from(t[k]));
                let bound = match k {
                    1 => 1e-4 * tv.abs() + 1e-5 * f64::from(t[0].abs()),
                    9 => 2e-2,
                    _ => 1e-5 * tv.abs(),
                };
                assert!((dv - tv).abs() <= bound, "candidate {c} value {k}: device {dv} against twin {tv}");
            }
        }
        (exact, through_log, rejected)
    }

    /// THE GENE-SUBSET CHOICE ON THE DEVICE, over the mixed bag's 300 chromosomes:
    ///  * the 45 candidates are the CPU twin's (the same parity rule as above) and
    ///    chrom_score's f64 ones (the same candidates accepted; errors within 2e-3);
    ///  * SWITCH OFF IS THE ENGINE AS IT WAS: the 9 candidates of all three genes
    ///    are, bit for bit and candidate for candidate, what `eval_and_score`
    ///    returns — which is what it returned before combinations existed;
    ///  * a gene that failed rejects only the subsets that hold it;
    ///  * redundancy is over the USED genes: a varying gene alone is the whole model.
    #[test]
    fn the_gene_subset_choice_on_the_device() {
        let Bag { rows, truth, splits, batch, gene_ok, chromosomes } = mixed_bag();
        let evaluator = GpuEvaluator::new(&rows, 2).expect("evaluator");
        let scorer = GpuScorer::new(&evaluator, &truth, splits).expect("scorer");
        let combinations = gene_linker_combinations(3, LINKERS, true).expect("combinations");
        let per = combinations.len() * WRAPPERS;
        assert_eq!(per, 45);
        let device = scorer.eval_and_score_with(&evaluator, &batch, &gene_ok, &chromosomes, &combinations).expect("score");
        assert_eq!(device.len(), chromosomes.len() * per * WIDTH);
        let preds = evaluator.eval(&batch).expect("eval");
        let twin = score_f32_with(&preds, &gene_ok, &chromosomes, &truth, splits, &combinations, true);
        let (exact, through_log, rejected) = assert_device_is_twin(&device, &twin);
        assert!(exact >= 5_000 && through_log >= 2_500 && rejected >= 1_000, "bit for bit {exact}, through the log {through_log}, rejected {rejected}");

        let spec = ScoreSpec {
            linkers: vec![Linker::AVG, Linker::MUL, Linker::ADD],
            wrappers: vec![Wrapper::Identity, Wrapper::LogAbs, Wrapper::SqrtAbs],
            splits,
            linear_scaling: true,
        };
        let reference = crate::chrom_score::score_gene_subsets(&preds, &gene_ok, &chromosomes, &truth, &spec, &combinations).expect("reference");
        let y_size = truth.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for c in 0..chromosomes.len() * per {
            let (want, got) = (&reference[c * SCORE_WIDTH..c * SCORE_WIDTH + METRICS], &device[c * WIDTH..(c + 1) * WIDTH]);
            assert_eq!(want[0].is_finite(), got[0].is_finite(), "candidate {c}: accepted on one side only");
            for k in (2..METRICS).filter(|_| want[0].is_finite()) {
                let floor = if matches!(k, 2 | 3 | 5) { (1e-4 * y_size).powi(2) } else { 1e-4 * y_size };
                assert!((want[k] - f64::from(got[k])).abs() < 2e-3 * want[k].abs() + floor, "candidate {c} value {k}: f64 {} against device {}", want[k], got[k]);
            }
        }

        let off = scorer.eval_and_score(&evaluator, &batch, &gene_ok, &chromosomes).expect("score");
        let whole = combinations.iter().position(|c| c.genes == 0b111).expect("all three genes");
        for (i, genes) in chromosomes.iter().enumerate() {
            let bits = |s: &[f32]| s.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
            let on = &device[(i * per + whole * WRAPPERS) * WIDTH..(i * per + whole * WRAPPERS + CANDIDATES) * WIDTH];
            assert_eq!(bits(on), bits(&off[i * CANDIDATES * WIDTH..(i + 1) * CANDIDATES * WIDTH]), "chromosome {i}");
            for (k, combination) in combinations.iter().enumerate() {
                let failed = combination.positions().any(|g| !gene_ok[genes[g]]);
                let scored = device[(i * per + k * WRAPPERS) * WIDTH..(i * per + (k + 1) * WRAPPERS) * WIDTH].iter().any(|v| v.is_finite());
                assert!(!(failed && scored), "chromosome {i} {combination:?}: scored with a gene that failed");
                // gene 0 of the bag (x*y) alone, identity: a varying gene that is the whole model
                if combination.genes.count_ones() == 1 && genes[combination.positions().next().unwrap_or(0)] == 0 {
                    assert_eq!(device[(i * per + k * WRAPPERS) * WIDTH + 9], 0.0, "chromosome {i} {combination:?}");
                }
            }
        }
        // A chromosome with a failed gene still has its other subsets scored.
        let poisoned = chromosomes.iter().position(|g| !gene_ok[g[0]] && gene_ok[g[1]] && g[1] != 4 && g[1] != 5).expect("such a chromosome");
        assert!(device[(poisoned * per + WRAPPERS) * WIDTH].is_finite(), "gene 1 alone of chromosome {poisoned}");
        assert!(gene_linker_combinations(4, LINKERS, true).is_err(), "four genes with subsets on is an error");
    }

    /// THE F32 FLOOR ON 1 - R²: what the device can resolve for a TRUE law on real
    /// data. Not run by default — it reads a PMLB file that is not in the repo:
    ///   SCORE_FLOOR_TSV=feynman_I_12_5.tsv cargo test --features gpu --lib f32_floor -- --ignored --nocapture
    /// The law is the product of the first two columns; 5,000 rows, 4,000 train.
    #[test]
    #[ignore = "reads the dataset named by SCORE_FLOOR_TSV"]
    fn f32_floor_of_a_true_law() {
        let path = std::env::var("SCORE_FLOOR_TSV").expect("SCORE_FLOOR_TSV names a PMLB .tsv");
        let text = std::fs::read_to_string(&path).expect("read the dataset");
        let table: Vec<Vec<f64>> = text.lines().skip(1).take(5_000).map(|l| l.split('\t').map(|v| v.trim().parse().expect("a number")).collect()).collect();
        let rows: Vec<f32> = table.iter().flat_map(|r| [r[0] as f32, r[1] as f32]).collect();
        let y: Vec<f64> = table.iter().map(|r| r[2]).collect();
        let splits = Splits { n_train: 4_000, n_val: 1_000, n_extrap: 0 };
        let names = vec!["a".to_string(), "b".to_string()];
        let mut batch = ExprBatch::new();
        batch.push(&math_to_nodes(r#"(Mul (Var "a") (Var "b"))"#, &names).unwrap());
        let chromosomes = vec![vec![0usize]];
        let evaluator = GpuEvaluator::new(&rows, 2).expect("evaluator");
        let scorer = GpuScorer::new(&evaluator, &y, splits).expect("scorer");
        let device = scorer.eval_and_score(&evaluator, &batch, &[true], &chromosomes).expect("score");
        let preds = evaluator.eval(&batch).expect("eval");
        let spec = ScoreSpec { linkers: vec![Linker::AVG], wrappers: vec![Wrapper::Identity], splits, linear_scaling: true };
        let f64_scores = score_chromosomes(&preds, &[true], &chromosomes, &y, &spec).expect("reference");
        let var = |lo: usize, hi: usize| {
            let m = y[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            y[lo..hi].iter().map(|v| (v - m).powi(2)).sum::<f64>() / (hi - lo) as f64
        };
        let (var_t, var_v) = (var(0, 4_000), var(4_000, 5_000));
        println!("device  a {} b {:e} | 1 - R² train {:e} validation {:e}", device[0], device[1], f64::from(device[2]) / var_t, f64::from(device[3]) / var_v);
        for (name, on) in [("twin, compensated", true), ("twin, plain sums", false)] {
            let t = score_f32(&preds, &[true], &chromosomes, &y, splits, on);
            println!("{name}  a {} b {:e} | 1 - R² train {:e} validation {:e}", t[0], t[1], f64::from(t[2]) / var_t, f64::from(t[3]) / var_v);
        }
        println!("f64 on the same f32 predictions  a {} b {:e} | 1 - R² train {:e} validation {:e}", f64_scores[0], f64_scores[1], f64_scores[2] / var_t, f64_scores[3] / var_v);
    }

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
