//! The challenge kernel's harness (`docs/PLAN_wgsl_gram_superopt.md` task 1):
//! the weighted Gram `H = ΦᵀΦ` of Blue World, built on the device through
//! the same banded driver shape as the production code
//! (`build_weighted_gram_item_banded_gpu` in qdrant's
//! `lib/blueworld/src/sphere/generator.rs`): item bands and output-row bands
//! sized from the per-binding limit, every item band's `phi` uploaded once,
//! one compute pass per item band inside one encoder per output band, the
//! accumulator resident between bands and read back once per output band.
//! Every pass carries GPU timestamp queries, so a build's time is the sum of
//! its passes on the device, uploads and read-backs apart.
//!
//! The reference is `H` in f64 on the host and the criterion is the plan's:
//! `|H_gpu − H_ref| ≤ τ ‖φ_·r‖ ‖φ_·c‖` with `τ = 4 ε_f32 √m`.

use std::time::Instant;

use wgpu::util::DeviceExt;

use crate::wgsl::oracle::device::Device;

/// The shipped kernel, verbatim from `H_GRAM_ACCUM_WGSL`.
pub const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");

/// The entry point every Gram kernel exposes; the bindings are fixed by
/// the driver: `phi` at 0, `h` at 1, `cfg` at 2.
pub const ENTRY: &str = "main";

/// Blue World's per-binding ceiling (`gpu_limits::effective_buffer_limit`):
/// the device's storage-binding limit capped by Metal's 256 MiB queue
/// write limit, halved.
pub fn effective_buffer_limit(device: &wgpu::Device) -> u64 {
    const QUEUE_WRITE_BUFFER_LIMIT: u64 = 256 * 1024 * 1024;
    u64::from(device.limits().max_storage_buffer_binding_size).min(QUEUE_WRITE_BUFFER_LIMIT) / 2
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic Φ, item-major `m × d`: each item row a standard
/// Gaussian vector (Box–Muller over splitmix64) normalised to unit length,
/// which is the sphere generator's convention for a projector row; the
/// production √Dₜ weight is a scalar per row and changes nothing the
/// kernel does. The same `(d, m, seed)` gives the same bits.
pub fn phi(d: usize, m: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut out = Vec::with_capacity(d * m);
    for _ in 0..m {
        let start = out.len();
        let mut i = 0;
        while i < d {
            let u1 = (splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
            let r = (-2.0 * (1.0 - u1).ln()).sqrt();
            let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
            out.push((r * c) as f32);
            if i + 1 < d {
                out.push((r * s) as f32);
            }
            i += 2;
        }
        let norm = out[start..].iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        for x in &mut out[start..] {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
    out
}

/// `H = ΦᵀΦ` in f64 on the host, `d × d` row-major, computed as the upper
/// triangle over threads and mirrored.
pub fn reference(phi: &[f32], d: usize, m: usize) -> Vec<f64> {
    assert_eq!(phi.len(), d * m, "phi is m × d");
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut h = vec![0.0f64; d * d];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                scope.spawn(move || {
                    // Rows interleaved across threads so the triangle balances.
                    let rows: Vec<usize> = (k..d).step_by(threads).collect();
                    let mut out: Vec<(usize, Vec<f64>)> = rows.iter().map(|&r| (r, vec![0.0f64; d - r])).collect();
                    for t in 0..m {
                        let row = &phi[t * d..(t + 1) * d];
                        for (r, acc) in &mut out {
                            let pr = f64::from(row[*r]);
                            for (a, &x) in acc.iter_mut().zip(&row[*r..]) {
                                *a += pr * f64::from(x);
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        for handle in handles {
            for (r, acc) in handle.join().expect("reference thread") {
                for (j, v) in acc.into_iter().enumerate() {
                    h[r * d + r + j] = v;
                    h[(r + j) * d + r] = v;
                }
            }
        }
    });
    h
}

/// The criterion's verdict over every entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// `τ = 4 ε_f32 √m`.
    pub tau: f64,
    /// The largest `|H_gpu − H_ref| / (τ ‖φ_·r‖ ‖φ_·c‖)`; passes at ≤ 1.
    pub max_ratio: f64,
    /// The mean of that ratio over the entries.
    pub mean_ratio: f64,
    /// Entries over 1 (the criterion failed there).
    pub failures: usize,
    /// Entries that are not finite on the device.
    pub non_finite: usize,
    /// Where the largest ratio sits.
    pub worst: (usize, usize),
    /// The largest absolute difference.
    pub max_abs: f64,
}

impl Verdict {
    pub fn passes(&self) -> bool {
        self.failures == 0 && self.non_finite == 0
    }
}

/// Check `h_gpu` against `h_ref` under the plan's criterion; the column
/// norms are the square roots of the reference diagonal.
pub fn check(h_gpu: &[f32], h_ref: &[f64], d: usize, m: usize) -> Verdict {
    assert_eq!(h_gpu.len(), d * d);
    assert_eq!(h_ref.len(), d * d);
    let tau = 4.0 * f64::from(f32::EPSILON) * (m as f64).sqrt();
    let norms: Vec<f64> = (0..d).map(|r| h_ref[r * d + r].sqrt()).collect();
    let mut v = Verdict { tau, max_ratio: 0.0, mean_ratio: 0.0, failures: 0, non_finite: 0, worst: (0, 0), max_abs: 0.0 };
    let mut sum = 0.0;
    for r in 0..d {
        for c in 0..d {
            let g = f64::from(h_gpu[r * d + c]);
            if !g.is_finite() {
                v.non_finite += 1;
                v.failures += 1;
                continue;
            }
            let diff = (g - h_ref[r * d + c]).abs();
            let ratio = diff / (tau * norms[r] * norms[c]);
            sum += ratio;
            if diff > v.max_abs {
                v.max_abs = diff;
            }
            if ratio > v.max_ratio {
                v.max_ratio = ratio;
                v.worst = (r, c);
            }
            if ratio > 1.0 {
                v.failures += 1;
            }
        }
    }
    v.mean_ratio = sum / (d * d) as f64;
    v
}

/// One build of `H` on the device: the result and its times.
#[derive(Debug, Clone, PartialEq)]
pub struct Build {
    pub h: Vec<f32>,
    /// GPU milliseconds, the sum of every compute pass's timestamps.
    pub gpu_ms: f64,
    /// Wall-clock milliseconds of the dispatches and read-backs (uploads excluded).
    pub wall_ms: f64,
    /// Wall-clock milliseconds of the `phi` uploads.
    pub upload_ms: f64,
    pub passes: usize,
    pub item_bands: usize,
    pub row_bands: usize,
}

/// A compiled Gram kernel on the oracle's device.
pub struct Gram<'a> {
    dev: &'a Device,
    pipeline: wgpu::ComputePipeline,
    workgroup: (u32, u32),
    period_ns: f32,
}

impl<'a> Gram<'a> {
    /// Compile `source` (naga-validated first, so an ill-formed variant is
    /// refused with its message, not a device error). `workgroup` is the
    /// kernel's `@workgroup_size` in x (columns) and y (rows); the driver
    /// dispatches `ceil(d / x) × ceil(r_rows / y)`.
    pub fn compile(dev: &'a Device, source: &str) -> Result<Gram<'a>, String> {
        let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| format!("validation: {}", e.emit_to_string(source)))?;
        let ep = module.entry_points.iter().find(|e| e.name == ENTRY).ok_or_else(|| format!("no entry point {ENTRY}"))?;
        let workgroup = (ep.workgroup_size[0], ep.workgroup_size[1]);
        if !dev.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err("this device has no timestamp queries".into());
        }
        dev.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gram"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)) });
        let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("gram"), layout: None, module: &shader, entry_point: ENTRY, compilation_options: Default::default() });
        if let Some(e) = pollster::block_on(dev.device.pop_error_scope()) {
            return Err(format!("pipeline: {e}"));
        }
        Ok(Gram { dev, pipeline, workgroup, period_ns: dev.queue.get_timestamp_period() })
    }

    /// Build `H` over `phi` (item-major `m × d`) as production does, under
    /// `limit` bytes per binding.
    pub fn build(&self, phi: &[f32], d: usize, m: usize, limit: u64) -> Result<Build, String> {
        if phi.len() != d * m {
            return Err(format!("phi has {} values, not m × d = {}", phi.len(), d * m));
        }
        let device = &self.dev.device;
        let queue = &self.dev.queue;
        let row_bytes = d as u64 * 4;
        if row_bytes > limit {
            return Err(format!("one d-length row of {row_bytes} bytes exceeds the limit {limit}"));
        }
        let band_items = ((limit / row_bytes).max(1) as usize).min(m);
        let row_band_rows = ((limit / row_bytes).max(1) as usize).min(d);
        let item_bands = m.div_ceil(band_items);
        let row_bands = d.div_ceil(row_band_rows);

        let t0 = Instant::now();
        let phi_bands: Vec<(wgpu::Buffer, u32)> = (0..item_bands)
            .map(|i| {
                let start = i * band_items;
                let end = (start + band_items).min(m);
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gram phi band"), contents: bytemuck::cast_slice(&phi[start * d..end * d]), usage: wgpu::BufferUsages::STORAGE });
                (buffer, (end - start) as u32)
            })
            .collect();
        queue.submit(None);
        device.poll(wgpu::Maintain::Wait);
        let upload_ms = t0.elapsed().as_secs_f64() * 1e3;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("gram timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * item_bands as u32 });
        let stamps_bytes = 16 * item_bands as u64;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram timestamps resolve"), size: stamps_bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let stamps_read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram timestamps read"), size: stamps_bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let layout = self.pipeline.get_bind_group_layout(0);

        let mut h = vec![0.0f32; d * d];
        let mut gpu_ns = 0.0f64;
        let t1 = Instant::now();
        let mut r_off = 0usize;
        while r_off < d {
            let r_rows = (r_off + row_band_rows).min(d) - r_off;
            let band_bytes = (r_rows * d * 4) as u64;
            let h_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram accumulator"), size: band_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let cfgs: Vec<wgpu::Buffer> = phi_bands
                .iter()
                .map(|(_, mb)| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gram cfg"), contents: bytemuck::cast_slice(&[d as u32, *mb, r_rows as u32, r_off as u32]), usage: wgpu::BufferUsages::UNIFORM }))
                .collect();
            let binds: Vec<wgpu::BindGroup> = phi_bands
                .iter()
                .zip(&cfgs)
                .map(|((pb, _), cfg)| device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gram bind"), layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: pb.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: h_buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: cfg.as_entire_binding() }] }))
                .collect();
            let wg_x = (d as u32).div_ceil(self.workgroup.0);
            let wg_y = (r_rows as u32).div_ceil(self.workgroup.1);
            let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram staging"), size: band_bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram output band") });
            enc.clear_buffer(&h_buffer, 0, None);
            for (i, bind) in binds.iter().enumerate() {
                let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &query_set, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gram item band"), timestamp_writes });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            enc.resolve_query_set(&query_set, 0..2 * item_bands as u32, &resolve_buf, 0);
            enc.copy_buffer_to_buffer(&resolve_buf, 0, &stamps_read, 0, stamps_bytes);
            enc.copy_buffer_to_buffer(&h_buffer, 0, &staging, 0, band_bytes);
            queue.submit(Some(enc.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, { let tx = tx.clone(); move |r| { let _ = tx.send(r); } });
            stamps_read.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device.poll(wgpu::Maintain::Wait);
            for _ in 0..2 {
                rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
            }
            {
                let mapped = staging.slice(..).get_mapped_range();
                let band: &[f32] = bytemuck::cast_slice(&mapped);
                h[r_off * d..(r_off + r_rows) * d].copy_from_slice(band);
            }
            staging.unmap();
            {
                let mapped = stamps_read.slice(..).get_mapped_range();
                let stamps: &[u64] = bytemuck::cast_slice(&mapped);
                for pair in stamps.chunks(2) {
                    gpu_ns += pair[1].wrapping_sub(pair[0]) as f64 * f64::from(self.period_ns);
                }
            }
            stamps_read.unmap();
            h_buffer.destroy();
            staging.destroy();
            r_off += r_rows;
        }
        let wall_ms = t1.elapsed().as_secs_f64() * 1e3;
        for (b, _) in &phi_bands {
            b.destroy();
        }
        device.poll(wgpu::Maintain::Poll);
        Ok(Build { h, gpu_ms: gpu_ns / 1e6, wall_ms, upload_ms, passes: item_bands * row_bands, item_bands, row_bands })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_is_deterministic_and_unit_rows() {
        let a = phi(5, 7, 3);
        let b = phi(5, 7, 3);
        assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert_ne!(a, phi(5, 7, 4));
        for row in a.chunks(5) {
            let n: f64 = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            assert!((n - 1.0).abs() < 1e-6, "{n}");
        }
    }

    #[test]
    fn reference_is_symmetric_and_exact_on_a_small_case() {
        let d = 3;
        let p: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let h = reference(&p, d, 2);
        // H[r][c] = Σ_t p[t][r] p[t][c]
        assert_eq!(h, vec![17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn check_counts_failures_and_non_finite() {
        let d = 2;
        let h_ref = vec![1.0, 0.5, 0.5, 1.0];
        let v = check(&[1.0, 0.5, 0.5, 1.0], &h_ref, d, 16);
        assert!(v.passes());
        assert_eq!(v.max_ratio, 0.0);
        let v = check(&[1.0, 0.5, f32::NAN, 1.0], &h_ref, d, 16);
        assert_eq!((v.failures, v.non_finite), (1, 1));
        let v = check(&[1.0, 0.6, 0.5, 1.0], &h_ref, d, 16);
        assert_eq!(v.failures, 1);
        assert_eq!(v.worst, (0, 1));
    }

    #[test]
    fn shipped_kernel_passes_the_criterion_on_the_device() {
        // Small enough to run in a test, large enough to band: d = 64 rows
        // of 256 B, the limit set to 4 KiB so item and row bands are both 16.
        let (d, m) = (64, 1000);
        let p = phi(d, m, 11);
        let h_ref = reference(&p, d, m);
        let dev = Device::new().expect("device");
        let gram = Gram::compile(&dev, SHIPPED).expect("compile");
        let build = gram.build(&p, d, m, 4096).expect("build");
        assert_eq!((build.item_bands, build.row_bands, build.passes), (63, 4, 252));
        let v = check(&build.h, &h_ref, d, m);
        assert!(v.passes(), "{v:?}");
        assert!(build.gpu_ms > 0.0);
        // The unbanded build gives the same entries within the criterion too.
        let one = gram.build(&p, d, m, effective_buffer_limit(&dev.device)).expect("build");
        assert_eq!((one.item_bands, one.row_bands), (1, 1));
        assert!(check(&one.h, &h_ref, d, m).passes());
    }
}
