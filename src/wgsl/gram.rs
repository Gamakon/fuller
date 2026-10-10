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
use crate::wgsl::render::{Grid, Rendered};

/// The seed tree must be the one the tests proved bit-identical: 82 nodes
/// with `kernel.entry` at the root.
pub fn gram_shipped_tree_check(t: &crate::wgsl::render::Tree) -> Result<(), String> {
    if t.name != "kernel.entry" || t.size() != 82 {
        return Err(format!("the shipped tree is {} with {} nodes, not kernel.entry with 82", t.name, t.size()));
    }
    Ok(())
}

/// The hand-written tiled kernel, the positive control of
/// `docs/PLAN_hgp_gram.md` step 0.
pub const TILED: &str = include_str!("../../kingdoms/wgsl/gram/tiled.wgsl");

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

/// The twelve kinds of Φ of the plan's §1b. Each is item-major `m × d`;
/// `seed` fixes the draw. `Gaussian` is [`phi`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Unit-normalised Gaussian rows, the sphere convention.
    Gaussian,
    /// Gaussian rows scaled by √Dₜ, the weights log-uniform over six decades.
    Weighted,
    /// Rows in a subspace of rank 4, 16 or 64 (by seed), plus noise at 1e-3.
    LowRank,
    /// Rows with 90 % or 99 % zeros (by seed).
    Sparse,
    /// Student t with two degrees of freedom: a few terms dominate.
    HeavyTailed,
    /// Each row paired with its negation. H = ΦᵀΦ is a sum of squares and
    /// cannot cancel; what the pair does is contribute the same term twice
    /// with the same sign (r rᵀ and (−r)(−r)ᵀ), so every off-diagonal cell
    /// is an exact doubling and the sign handling of the product is tested.
    /// (The review of 2026-10-09 found the earlier description, "sums
    /// cancel exactly", false.)
    Cancelling,
    /// Small integers in −3..=3: every product and sum is exact.
    Integer,
    /// Rows at 1e-20 (odd rows) and 1e15 (even rows): underflow and overflow.
    Extreme,
    /// Gaussian rows with NaN and ±∞ at known cells.
    NonFinite,
    /// Half the rows duplicates of the first row, a quarter constant.
    Duplicate,
    /// Gaussian rows; the shape itself is ragged (the caller chooses m, d).
    Ragged,
}

impl Kind {
    pub const ALL: [Kind; 11] = [Kind::Gaussian, Kind::Weighted, Kind::LowRank, Kind::Sparse, Kind::HeavyTailed, Kind::Cancelling, Kind::Integer, Kind::Extreme, Kind::NonFinite, Kind::Duplicate, Kind::Ragged];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Gaussian => "gaussian",
            Kind::Weighted => "weighted",
            Kind::LowRank => "lowrank",
            Kind::Sparse => "sparse",
            Kind::HeavyTailed => "heavytailed",
            Kind::Cancelling => "cancelling",
            Kind::Integer => "integer",
            Kind::Extreme => "extreme",
            Kind::NonFinite => "nonfinite",
            Kind::Duplicate => "duplicate",
            Kind::Ragged => "ragged",
        }
    }
}

fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn gaussian_pair(state: &mut u64) -> (f64, f64) {
    let u1 = unit(state);
    let u2 = unit(state);
    let r = (-2.0 * (1.0 - u1).ln()).sqrt();
    let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
    (r * c, r * s)
}

fn gaussian_row(d: usize, state: &mut u64) -> Vec<f64> {
    let mut row = Vec::with_capacity(d);
    while row.len() < d {
        let (a, b) = gaussian_pair(state);
        row.push(a);
        if row.len() < d {
            row.push(b);
        }
    }
    row
}

fn normalise(row: &mut [f64]) {
    let n = row.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 0.0 {
        for x in row.iter_mut() {
            *x /= n;
        }
    }
}

/// Φ of a kind. Deterministic in `(kind, d, m, seed)`.
pub fn phi_of(kind: Kind, d: usize, m: usize, seed: u64) -> Vec<f32> {
    if kind == Kind::Gaussian || kind == Kind::Ragged {
        return phi(d, m, seed);
    }
    let mut state = seed ^ 0xA076_1D64_78BD_642F ^ ((kind as u64) << 48);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(m);
    match kind {
        Kind::Gaussian | Kind::Ragged => unreachable!("handled above"),
        Kind::Weighted => {
            for _ in 0..m {
                let mut r = gaussian_row(d, &mut state);
                normalise(&mut r);
                let w = 10f64.powf(unit(&mut state) * 6.0 - 3.0).sqrt();
                for x in &mut r {
                    *x *= w;
                }
                rows.push(r);
            }
        }
        Kind::LowRank => {
            let rank = [4usize, 16, 64][(seed % 3) as usize].min(d);
            let basis: Vec<Vec<f64>> = (0..rank).map(|_| { let mut b = gaussian_row(d, &mut state); normalise(&mut b); b }).collect();
            for _ in 0..m {
                let coeffs = gaussian_row(rank, &mut state);
                let mut r = vec![0.0; d];
                for (c, b) in coeffs.iter().zip(&basis) {
                    for (x, y) in r.iter_mut().zip(b) {
                        *x += c * y;
                    }
                }
                let noise = gaussian_row(d, &mut state);
                for (x, n) in r.iter_mut().zip(noise) {
                    *x += 1e-3 * n;
                }
                normalise(&mut r);
                rows.push(r);
            }
        }
        Kind::Sparse => {
            let density = if seed.is_multiple_of(2) { 0.1 } else { 0.01 };
            for _ in 0..m {
                let mut r = gaussian_row(d, &mut state);
                for x in &mut r {
                    if unit(&mut state) >= density {
                        *x = 0.0;
                    }
                }
                normalise(&mut r);
                rows.push(r);
            }
        }
        Kind::HeavyTailed => {
            for _ in 0..m {
                // t with ν = 2: z / sqrt(chi²₂ / 2), chi²₂ = −2 ln u.
                let r: Vec<f64> = gaussian_row(d, &mut state).into_iter().map(|z| { let chi = -2.0 * (1.0 - unit(&mut state)).ln(); z / (chi / 2.0).sqrt() }).collect();
                rows.push(r);
            }
        }
        Kind::Cancelling => {
            for i in 0..m {
                if i % 2 == 0 {
                    let mut r = gaussian_row(d, &mut state);
                    normalise(&mut r);
                    rows.push(r);
                } else {
                    let prev = rows[i - 1].iter().map(|x| -x).collect();
                    rows.push(prev);
                }
            }
        }
        Kind::Integer => {
            for _ in 0..m {
                rows.push((0..d).map(|_| (unit(&mut state) * 7.0).floor() - 3.0).collect());
            }
        }
        Kind::Extreme => {
            for i in 0..m {
                let scale = if i % 2 == 0 { 1e15 } else { 1e-20 };
                let mut r = gaussian_row(d, &mut state);
                normalise(&mut r);
                for x in &mut r {
                    *x *= scale;
                }
                rows.push(r);
            }
        }
        Kind::NonFinite => {
            for i in 0..m {
                let mut r = gaussian_row(d, &mut state);
                normalise(&mut r);
                // Item 3: NaN at column 1; item 7: +∞ at column 2; item 11: −∞ at column 0.
                match i {
                    3 if d > 1 => r[1] = f64::NAN,
                    7 if d > 2 => r[2] = f64::INFINITY,
                    11 => r[0] = f64::NEG_INFINITY,
                    _ => {}
                }
                rows.push(r);
            }
        }
        Kind::Duplicate => {
            let mut first = gaussian_row(d, &mut state);
            normalise(&mut first);
            for i in 0..m {
                if i % 2 == 0 {
                    rows.push(first.clone());
                } else if i % 4 == 1 {
                    rows.push(vec![1.0 / (d as f64).sqrt(); d]);
                } else {
                    let mut r = gaussian_row(d, &mut state);
                    normalise(&mut r);
                    rows.push(r);
                }
            }
        }
    }
    rows.into_iter().flat_map(|r| r.into_iter().map(|x| x as f32)).collect()
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
            let want = h_ref[r * d + c];
            // A non-finite cell: the classes must agree (NaN with NaN, +∞ with
            // +∞, −∞ with −∞), the plan's ShapeDistance for such a cell.
            if !want.is_finite() {
                let same = (g.is_nan() && want.is_nan()) || g == want;
                if !same {
                    v.failures += 1;
                    v.non_finite += 1;
                    if v.max_ratio < 1.0 {
                        v.max_ratio = 1.0;
                        v.worst = (r, c);
                    }
                    sum += 1.0;
                }
                continue;
            }
            if !g.is_finite() {
                v.non_finite += 1;
                v.failures += 1;
                sum += 1.0;
                if v.max_ratio < 1.0 {
                    v.max_ratio = 1.0;
                    v.worst = (r, c);
                }
                continue;
            }
            let diff = (g - want).abs();
            // A zero column norm (an all-zero column of Φ) makes the scale zero;
            // the entry must then be exactly zero.
            let scale = tau * norms[r] * norms[c];
            let ratio = if scale > 0.0 { diff / scale } else if diff == 0.0 { 0.0 } else { f64::INFINITY };
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
    /// GPU milliseconds, the sum of every compute pass's timestamps,
    /// resolved in a submission AFTER the passes have completed: resolved in
    /// the same submission, short passes read 0 on Metal (measured, the
    /// test `every_build_in_a_sequence_has_a_compute_time`). The search's time.
    pub gpu_ms: f64,
    /// Wall-clock milliseconds of the compute submissions alone: submit to
    /// completion of the passes, before any read-back; about 1–2 ms above
    /// `gpu_ms` (the submission).
    pub compute_ms: f64,
    /// Wall-clock milliseconds of the dispatches and read-backs (uploads excluded).
    pub wall_ms: f64,
    /// Wall-clock milliseconds of the `phi` uploads.
    pub upload_ms: f64,
    pub passes: usize,
    pub item_bands: usize,
    pub row_bands: usize,
}

/// The `// grid x y bx by` header of a kernel text, if present on any line.
pub fn grid_header(source: &str) -> Option<(u32, u32, u32, u32)> {
    source.lines().map(str::trim).find_map(|l| {
        let rest = l.strip_prefix("// grid ")?;
        let v: Vec<u32> = rest.split_whitespace().map(|t| t.parse().ok()).collect::<Option<Vec<u32>>>()?;
        (v.len() == 4).then_some((v[0], v[1], v[2], v[3]))
    })
}

/// A compiled Gram kernel on the oracle's device.
pub struct Gram<'a> {
    dev: &'a Device,
    pipeline: wgpu::ComputePipeline,
    /// Workgroup size and the cells per thread; the driver dispatches
    /// `ceil(d / (x bx)) × ceil(r_rows / (y by))`.
    pub grid: Grid,
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
        let (x, y) = (ep.workgroup_size[0], ep.workgroup_size[1]);
        // The optional header `// grid x y bx by`: x and y must agree with
        // naga's workgroup size (the header is never trusted for them); bx
        // and by, the cells per thread, come only from the header.
        let (bx, by) = match grid_header(source) {
            Some((hx, hy, bx, by)) => {
                if (hx, hy) != (x, y) {
                    return Err(format!("the grid header says workgroup {hx} x {hy}, the shader says {x} x {y}"));
                }
                if !(1..=16).contains(&bx) || !(1..=16).contains(&by) {
                    return Err(format!("the grid header's block {bx} x {by} is outside 1..=16"));
                }
                (bx, by)
            }
            None => (1, 1),
        };
        Self::compile_with(dev, source, Grid { x, y, bx, by })
    }

    /// A rendered kernel: its text and the grid it asked for.
    pub fn compile_rendered(dev: &'a Device, r: &Rendered) -> Result<Gram<'a>, String> {
        naga::front::wgsl::parse_str(&r.text).map_err(|e| e.emit_to_string(&r.text)).and_then(|m| naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&m).map(|_| ()).map_err(|e| format!("validation: {}", e.emit_to_string(&r.text))))?;
        Self::compile_with(dev, &r.text, r.grid)
    }

    fn compile_with(dev: &'a Device, source: &str, grid: Grid) -> Result<Gram<'a>, String> {
        if !dev.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err("this device has no timestamp queries".into());
        }
        dev.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gram"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)) });
        // An explicit layout over the role's three bindings: a kernel that
        // reads none of them (a random gene may) still binds them all.
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gram"), entries: &[entry(0, wgpu::BufferBindingType::Storage { read_only: true }), entry(1, wgpu::BufferBindingType::Storage { read_only: false }), entry(2, wgpu::BufferBindingType::Uniform)] });
        let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("gram"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("gram"), layout: Some(&layout), module: &shader, entry_point: ENTRY, compilation_options: Default::default() });
        if let Some(e) = pollster::block_on(dev.device.pop_error_scope()) {
            return Err(format!("pipeline: {e}"));
        }
        Ok(Gram { dev, pipeline, grid, period_ns: dev.queue.get_timestamp_period() })
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
        let mut compute_ms = 0.0f64;
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
            let wg_x = (d as u32).div_ceil(self.grid.x * self.grid.bx);
            let wg_y = (r_rows as u32).div_ceil(self.grid.y * self.grid.by);
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
            // The compute alone: submit, wait, the wall clock around it.
            let t_compute = Instant::now();
            queue.submit(Some(enc.finish()));
            device.poll(wgpu::Maintain::Wait);
            compute_ms += t_compute.elapsed().as_secs_f64() * 1e3;
            // Then the timestamps and the band, read back.
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram read-back") });
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
        Ok(Build { h, gpu_ms: gpu_ns / 1e6, compute_ms, wall_ms, upload_ms, passes: item_bands * row_bands, item_bands, row_bands })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_is_deterministic_and_has_its_property() {
        let (d, m) = (16, 40);
        for kind in Kind::ALL {
            let a = phi_of(kind, d, m, 3);
            assert_eq!(a.len(), d * m);
            assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), phi_of(kind, d, m, 3).iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{kind:?}");
            let h = reference(&a, d, m);
            match kind {
                Kind::Integer => assert!(a.iter().all(|x| x.fract() == 0.0 && x.abs() <= 3.0)),
                Kind::Cancelling => {
                    assert!((0..m / 2).all(|i| a[2 * i * d..(2 * i + 1) * d].iter().zip(&a[(2 * i + 1) * d..(2 * i + 2) * d]).all(|(x, y)| *x == -*y)));
                    // The pair doubles: H equals twice the Gram of the even rows.
                    let even: Vec<f32> = (0..m / 2).flat_map(|i| a[2 * i * d..(2 * i + 1) * d].to_vec()).collect();
                    let h2 = reference(&even, d, m / 2);
                    assert!(h.iter().zip(&h2).all(|(x, y)| (x - 2.0 * y).abs() <= 1e-9 * (1.0 + y.abs())));
                }
                Kind::NonFinite => assert!(a[3 * d + 1].is_nan() && a[7 * d + 2] == f32::INFINITY && h[d + 1].is_nan()),
                Kind::Sparse => assert!(a.iter().filter(|x| **x == 0.0).count() > d * m / 2),
                Kind::Extreme => assert!(a.iter().any(|x| x.abs() > 1e13) && a.iter().any(|x| x.abs() < 1e-18 && *x != 0.0)),
                Kind::Duplicate => assert_eq!(a[..d], a[2 * d..3 * d]),
                _ => assert!(h.iter().all(|x| x.is_finite()), "{kind:?}"),
            }
        }
        // The criterion on the non-finite kind: a kernel that agrees in class passes, one that does not fails.
        let a = phi_of(Kind::NonFinite, d, m, 3);
        let h = reference(&a, d, m);
        let same: Vec<f32> = h.iter().map(|x| *x as f32).collect();
        assert!(check(&same, &h, d, m).passes());
        let mut wrong = same.clone();
        wrong[d + 1] = 0.0;
        assert_eq!(check(&wrong, &h, d, m).non_finite, 1);
    }

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

    /// The shipped kernel as a typed tree, rendered, on the device: the same
    /// H within the criterion as the shipped text.
    #[test]
    fn the_rendered_shipped_tree_passes_the_criterion_on_the_device() {
        use crate::wgsl::render::{gram_role, gram_shipped_tree, Renderer};
        use crate::wgsl::{loader::WgslKingdom, table::FunctionTable};
        let (k, t, r) = (WgslKingdom::load(), FunctionTable::shipped(), gram_role());
        let out = Renderer::new(&k, &t, &r).kernel(&gram_shipped_tree(&k).unwrap()).unwrap();
        let (d, m) = (64, 1000);
        let p = phi(d, m, 11);
        let h_ref = reference(&p, d, m);
        let dev = Device::new().expect("device");
        let gram = Gram::compile_rendered(&dev, &out).expect("compile");
        let build = gram.build(&p, d, m, 4096).expect("build");
        let v = check(&build.h, &h_ref, d, m);
        assert!(v.passes(), "{v:?}");
        // Bit-identical to the shipped text: the same arithmetic in the same order.
        let shipped = Gram::compile(&dev, SHIPPED).expect("compile").build(&p, d, m, 4096).expect("build");
        assert_eq!(build.h.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), shipped.h.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    /// The kingdom gate on the device: random typed kernels rendered,
    /// compiled by Metal and dispatched at a small size, every failure named.
    #[test]
    fn random_typed_kernels_compile_and_run_on_the_device() {
        use crate::wgsl::render::{dual_ty, gram_role, sample, Renderer};
        use crate::wgsl::{loader::WgslKingdom, table::FunctionTable};
        let (k, t, r) = (WgslKingdom::load(), FunctionTable::shipped(), gram_role());
        let rr = Renderer::new(&k, &t, &r);
        let root = dual_ty(&k, "kernel.kernel").unwrap();
        let (d, m) = (64, 100);
        let p = phi(d, m, 5);
        let dev = Device::new().expect("device");
        let limit = effective_buffer_limit(&dev.device);
        let n = 100u64;
        let (mut refused, mut compiled, mut ran, mut finite) = (0, 0, 0, 0);
        let mut failed: Vec<String> = Vec::new();
        for seed in 0..n {
            let tree = sample(&k, &r, root, 7, 1000 + seed).unwrap();
            let Ok(out) = rr.kernel(&tree) else { refused += 1; continue };
            match Gram::compile_rendered(&dev, &out) {
                Err(e) => failed.push(format!("seed {seed} compile: {}", e.lines().next().unwrap_or(""))),
                Ok(g) => {
                    compiled += 1;
                    match g.build(&p, d, m, limit) {
                        Err(e) => failed.push(format!("seed {seed} run: {e}")),
                        Ok(b) => {
                            ran += 1;
                            if b.h.iter().all(|x| x.is_finite()) {
                                finite += 1;
                            }
                        }
                    }
                }
            }
        }
        println!("{n} random kernels on the device: {refused} refused by construction, {compiled} compiled, {ran} ran, {finite} with every cell finite");
        assert!(failed.is_empty(), "{} failures:\n{}", failed.len(), failed.join("\n"));
        assert_eq!(compiled, ran);
    }

    /// Every build reports a compute time, build after build, shape after
    /// shape; the timestamp sum is printed beside it (it reads 0 for short
    /// passes on Apple silicon, which is why the search uses compute_ms).
    #[test]
    fn every_build_in_a_sequence_has_a_compute_time() {
        let dev = Device::new().expect("device");
        let gram = Gram::compile(&dev, SHIPPED).expect("compile");
        let limit = effective_buffer_limit(&dev.device);
        let shapes = [(64usize, 1000usize), (32, 500), (64, 1000), (128, 2000), (32, 500), (32, 500)];
        let mut times = Vec::new();
        for (d, m) in shapes {
            let p = phi(d, m, 1);
            let b = gram.build(&p, d, m, limit).expect("build");
            times.push((b.compute_ms, b.gpu_ms));
        }
        println!("compute ms and timestamp ms per build: {times:?}");
        assert!(times.iter().all(|t| t.0 > 0.0 && t.1 > 0.0 && t.1 < t.0), "{times:?}");
    }

    #[test]
    fn the_grid_header_is_read_and_checked_against_the_shader() {
        assert_eq!(grid_header(TILED), Some((16, 16, 4, 4)));
        assert_eq!(grid_header(SHIPPED), None);
        let dev = Device::new().expect("device");
        assert_eq!(Gram::compile(&dev, TILED).unwrap().grid, Grid { x: 16, y: 16, bx: 4, by: 4 });
        let wrong = TILED.replace("// grid 16 16 4 4", "// grid 8 8 4 4");
        let err = match Gram::compile(&dev, &wrong) { Ok(_) => panic!("a mismatched header compiled"), Err(e) => e };
        assert!(err.contains("the grid header says workgroup 8 x 8"), "{err}");
    }

    /// The hand-written tiled kernel (plan step 0) agrees with the reference
    /// under the criterion, banded and unbanded, at ragged shapes.
    #[test]
    fn the_tiled_kernel_passes_the_criterion_on_the_device() {
        let dev = Device::new().expect("device");
        let gram = Gram::compile(&dev, TILED).expect("compile");
        for (d, m, limit) in [(64usize, 1000usize, 4096u64), (130, 77, 8192), (200, 33, u64::MAX), (64, 1000, u64::MAX)] {
            let p = phi(d, m, 11);
            let h_ref = reference(&p, d, m);
            let b = gram.build(&p, d, m, limit).expect("build");
            let v = check(&b.h, &h_ref, d, m);
            assert!(v.passes(), "d {d} m {m} limit {limit}: {v:?}");
        }
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
