//! The challenge kernel's harness (the Gram superoptimisation problem):
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

//!
//! The device parts (the driver, the residents) compile under `gpu`; the
//! reference and the criterion need only `wgsl`. The kernels themselves are
//! the WGSL kingdom's (`crate::wgsl_rows`, variant `wgsl.gram`); the shipped,
//! tiled and blocked kernels are reference programs (`kingdoms/wgsl_true/gram/`).

#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
use crate::wgsl::oracle::device::{Device, Fuel};

/// The grid a Gram kernel asks the driver for: the workgroup size `x` x `y`
/// and the output cells a thread computes, `bx` columns by `by` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub bx: u32,
    pub by: u32,
    /// The layout knob: the output's rows on thread x, the fast index, and
    /// its columns on y; false is columns on x. `x`, `y` stay the
    /// `@workgroup_size`; `bx` is always the column block and `by` the row
    /// block.
    pub rows_x: bool,
}

impl Grid {
    /// The thread-id component that carries the output's rows and the
    /// workgroup's extent along it.
    pub fn row_axis(&self) -> (&'static str, u32) {
        if self.rows_x { ("x", self.x) } else { ("y", self.y) }
    }

    /// The thread-id component that carries the output's columns and the
    /// workgroup's extent along it.
    pub fn col_axis(&self) -> (&'static str, u32) {
        if self.rows_x { ("y", self.y) } else { ("x", self.x) }
    }

    /// The workgroups to dispatch for `rows` output rows of `d` columns:
    /// `(along x, along y)`.
    pub fn dispatch(&self, rows: u32, d: u32) -> (u32, u32) {
        let (r, c) = (rows.div_ceil(self.row_axis().1 * self.by), d.div_ceil(self.col_axis().1 * self.bx));
        if self.rows_x { (r, c) } else { (c, r) }
    }
}

/// The hand-written tiled kernel, the positive control of
/// `docs/PLAN_hgp_gram.md` step 0.
pub const TILED: &str = include_str!("../../kingdoms/wgsl_true/gram/tiled.wgsl");

/// The hand-written register-blocked kernel (no workgroup memory): 3.6x
/// the shipped kernel's speed at the bench size (280 ms against 1,012 ms,
/// 2026-10-10), the positive control that passed.
pub const BLOCKED: &str = include_str!("../../kingdoms/wgsl_true/gram/blocked.wgsl");

/// The shipped kernel, verbatim from `H_GRAM_ACCUM_WGSL`.
pub const SHIPPED: &str = include_str!("../../kingdoms/wgsl_true/gram/shipped.wgsl");

/// The longest one blocking wait of ours lasts before it names the work
/// still running. wgpu-core 0.21 waited at most 5 s (`CLEANUP_WAIT_MS`),
/// then marked the submission done whether or not the GPU had finished it
/// (run 6, 2026-10-13: a kernel still running after 10 s was read, the next
/// kernel queued behind it read as slow, and the shipped kernel's timestamps
/// after it read zero). wgpu 30 waits as long as it is told; every wait of
/// ours is told this, and a wait that times out is a kernel still on the GPU
/// (`gpu_io::wait_for`). Not gated on `gpu`: a problem's plan names it as
/// its default wait.
pub const WAIT_GAVE_UP_MS: f64 = 4_900.0;

/// The longest the host drains a runaway before naming it undrained.
#[cfg(feature = "gpu")]
pub const DRAIN_LIMIT_MS: f64 = 120_000.0;

/// Wait until the GPU has really finished everything submitted. A wait
/// that gave up (`WAIT_GAVE_UP_MS`) is a runaway: the host then submits an
/// empty submission and waits again, until a wait returns early (the fence
/// reached, so the GPU has passed the runaway), and returns the runaway as
/// an error, so nothing the GPU was still writing is read; a GPU that does
/// not drain within `DRAIN_LIMIT_MS` is named (the fit's watchdog exits
/// sooner, naming the kernel).
#[cfg(feature = "gpu")]
pub fn wait_gpu(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
    wait_gpu_within(device, queue, WAIT_GAVE_UP_MS)
}

/// [`wait_gpu`] for work that may run up to `limit_ms` (a problem's
/// `budget.gpu_wait_ms`): the GPU finishing within it is done, not a
/// runaway. kmeans_assign's 18,000 x 9,000 x 2,048 case is one dispatch of
/// over 5 s, which the fixed limit named a runaway (2026-10-15).
#[cfg(feature = "gpu")]
pub fn wait_gpu_within(device: &wgpu::Device, queue: &wgpu::Queue, limit_ms: f64) -> Result<(), String> {
    let ms_since = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    if crate::gpu_io::wait_for(device, WAIT_GAVE_UP_MS)? {
        return Ok(());
    }
    let first = ms_since(t);
    loop {
        queue.submit(None);
        if crate::gpu_io::wait_for(device, WAIT_GAVE_UP_MS)? {
            if ms_since(t) <= limit_ms {
                return Ok(());
            }
            return Err(format!("runaway: the GPU still ran after {:.1} s; drained after {:.1} s, nothing read", first / 1e3, ms_since(t) / 1e3));
        }
        if ms_since(t) > DRAIN_LIMIT_MS {
            return Err(format!("runaway: the GPU still ran after {:.1} s and had not drained after {:.1} s", first / 1e3, ms_since(t) / 1e3));
        }
    }
}

/// The entry point every Gram kernel exposes; the bindings are fixed by
/// the driver: `phi` at 0, `h` at 1, `cfg` at 2.
pub const ENTRY: &str = "main";

/// Blue World's per-binding ceiling (`gpu_limits::effective_buffer_limit`):
/// the device's storage-binding limit capped by Metal's 256 MiB queue
/// write limit, halved.
#[cfg(feature = "gpu")]
pub fn effective_buffer_limit(device: &wgpu::Device) -> u64 {
    const QUEUE_WRITE_BUFFER_LIMIT: u64 = 256 * 1024 * 1024;
    device.limits().max_storage_buffer_binding_size.min(QUEUE_WRITE_BUFFER_LIMIT) / 2
}

/// The device's own per-binding ceiling: its storage-binding limit and its
/// largest buffer, the adapter's own (`oracle::Device::new` requests the
/// adapter's limits); a problem that declares `binding_limit = "adapter"`
/// plans by it (`harness::DeviceExecutor::for_problem`).
#[cfg(feature = "gpu")]
pub fn adapter_buffer_limit(device: &wgpu::Device) -> u64 {
    let l = device.limits();
    l.max_storage_buffer_binding_size.min(l.max_buffer_size)
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

/// [`check`]'s `max_ratio` and `non_finite`, the two figures the fitness
/// reads, over row blocks on every core: the same per-cell arithmetic, and
/// a maximum and a count do not depend on the order, so the result is
/// bit-identical to [`check`]'s (tested).
pub fn check_max(h_gpu: &[f32], h_ref: &[f64], d: usize, m: usize) -> (f64, usize) {
    check_max_tau(h_gpu, h_ref, d, 4.0 * f64::from(f32::EPSILON) * (m as f64).sqrt())
}

/// [`check_max`] at a given `tau` (the generic harness's `cauchy_schwarz`
/// norm, its `tau` an expression of the case's shape).
pub fn check_max_tau(h_gpu: &[f32], h_ref: &[f64], d: usize, tau: f64) -> (f64, usize) {
    assert_eq!(h_gpu.len(), d * d);
    assert_eq!(h_ref.len(), d * d);
    let norms: Vec<f64> = (0..d).map(|r| h_ref[r * d + r].sqrt()).collect();
    let threads = std::thread::available_parallelism().map_or(1, usize::from).min(d.max(1));
    let rows_per = d.div_ceil(threads);
    let norms = &norms;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                scope.spawn(move || {
                    let (mut max_ratio, mut non_finite) = (0.0f64, 0usize);
                    for r in (k * rows_per)..((k + 1) * rows_per).min(d) {
                        for c in 0..d {
                            let g = f64::from(h_gpu[r * d + c]);
                            let want = h_ref[r * d + c];
                            if !want.is_finite() {
                                if !((g.is_nan() && want.is_nan()) || g == want) {
                                    non_finite += 1;
                                    max_ratio = max_ratio.max(1.0);
                                }
                                continue;
                            }
                            if !g.is_finite() {
                                non_finite += 1;
                                max_ratio = max_ratio.max(1.0);
                                continue;
                            }
                            let diff = (g - want).abs();
                            let scale = tau * norms[r] * norms[c];
                            let ratio = if scale > 0.0 { diff / scale } else if diff == 0.0 { 0.0 } else { f64::INFINITY };
                            if ratio > max_ratio {
                                max_ratio = ratio;
                            }
                        }
                    }
                    (max_ratio, non_finite)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("check thread")).fold((0.0f64, 0usize), |(a, n), (b, k)| (if b > a { b } else { a }, n + k))
    })
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

/// The `// grid x y bx by [rows-x]` header of a kernel text, if present on any line.
pub fn grid_header(source: &str) -> Option<(u32, u32, u32, u32)> {
    header_fields(source).map(|(g, _)| g)
}

/// Whether the header says `rows-x`: the output's rows on thread x.
pub fn grid_rows_x(source: &str) -> bool {
    header_fields(source).is_some_and(|(_, r)| r)
}

fn header_fields(source: &str) -> Option<((u32, u32, u32, u32), bool)> {
    source.lines().map(str::trim).find_map(|l| {
        let rest = l.strip_prefix("// grid ")?;
        let mut tokens: Vec<&str> = rest.split_whitespace().collect();
        let rows_x = tokens.last() == Some(&"rows-x");
        if rows_x {
            tokens.pop();
        }
        let v: Vec<u32> = tokens.iter().map(|t| t.parse().ok()).collect::<Option<Vec<u32>>>()?;
        match v.as_slice() {
            &[x, y, bx, by] => Some(((x, y, bx, by), rows_x)),
            _ => None,
        }
    })
}

/// The driver's bands for `d x d` output over `m` items under `limit`
/// bytes per binding: item bands of `band_items` items (the last shorter),
/// output-row bands of `band_rows` rows (the last shorter). The one place
/// the formulas live; [`Resident::upload`], [`Gram::run`] and
/// [`Gram::run_batch`] use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandLayout {
    pub band_items: usize,
    pub item_bands: usize,
    pub band_rows: usize,
    pub row_bands: usize,
}

pub fn band_layout(d: usize, m: usize, limit: u64) -> BandLayout {
    let row_bytes = d as u64 * 4;
    let band_items = ((limit / row_bytes).max(1) as usize).min(m);
    let band_rows = ((limit / row_bytes).max(1) as usize).min(d);
    BandLayout { band_items, item_bands: m.div_ceil(band_items), band_rows, row_bands: d.div_ceil(band_rows) }
}

/// One step of the driver's command sequence for one Phi, as
/// [`Resident::upload`] and [`Gram::run`] issue it.
/// `Upload` is the resident buffer of items `start..end`; per output-row
/// band, `Clear` of the accumulator of `rows` rows, one `Dispatch` per
/// item band with its `cfg` words and workgroups, then `Sync` (the compute
/// submission) and `Read` of the band into `H` at row `r_off`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverStep {
    Upload { band: usize, start: usize, end: usize },
    Clear { rows: usize },
    Dispatch { band: usize, cfg: [u32; 4], workgroups: (u32, u32) },
    Sync,
    Read { r_off: usize, rows: usize },
}

/// The driver's command sequence for `(d, m, limit)` under `grid`, the
/// uploads first, from the same formulas [`Gram::run`] uses.
pub fn driver_steps(d: usize, m: usize, limit: u64, grid: Grid) -> Vec<DriverStep> {
    let b = band_layout(d, m, limit);
    let mut out = Vec::new();
    for i in 0..b.item_bands {
        let start = i * b.band_items;
        out.push(DriverStep::Upload { band: i, start, end: (start + b.band_items).min(m) });
    }
    let mut r_off = 0usize;
    while r_off < d {
        let r_rows = (r_off + b.band_rows).min(d) - r_off;
        out.push(DriverStep::Clear { rows: r_rows });
        for i in 0..b.item_bands {
            let start = i * b.band_items;
            let mb = ((start + b.band_items).min(m) - start) as u32;
            out.push(DriverStep::Dispatch { band: i, cfg: [d as u32, mb, r_rows as u32, r_off as u32], workgroups: grid.dispatch(r_rows as u32, d as u32) });
        }
        out.push(DriverStep::Sync);
        out.push(DriverStep::Read { r_off, rows: r_rows });
        r_off += r_rows;
    }
    out
}

/// The host half of [`Gram::compile`]: naga parse and validation for the
/// current target, the entry point, and the grid from the workgroup size
/// and the optional `// grid x y bx by [rows-x]` header (x and y must agree
/// with naga's workgroup size, the header is never trusted for them; bx and
/// by, the cells per thread, come only from the header, in 1..=16).
pub fn grid_of(source: &str) -> Result<Grid, String> {
    let module = crate::wgsl::naga_shape::parse(source)?;
    super::target::Target::current().validate(&module, source).map_err(|e| format!("validation: {e}"))?;
    let ep = module.entry_points.iter().find(|e| e.name == ENTRY).ok_or_else(|| format!("no entry point {ENTRY}"))?;
    let (x, y) = (ep.workgroup_size[0], ep.workgroup_size[1]);
    let rows_x = grid_rows_x(source);
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
    Ok(Grid { x, y, bx, by, rows_x })
}

/// The invocations of one output-row band's submit: `passes` item-band passes, each of [`Grid::dispatch`]
/// workgroups of `grid.x * grid.y` invocations. The work guard's `invocations`.
pub fn submit_invocations(grid: Grid, d: usize, r_rows: usize, passes: usize) -> u64 {
    let (wg_x, wg_y) = grid.dispatch(r_rows as u32, d as u32);
    u64::from(wg_x).saturating_mul(u64::from(wg_y)).saturating_mul(u64::from(grid.x) * u64::from(grid.y)).saturating_mul(passes as u64)
}

/// A compiled Gram kernel on the oracle's device.
#[cfg(feature = "gpu")]
pub struct Gram<'a> {
    dev: &'a Device,
    pipeline: wgpu::ComputePipeline,
    /// Workgroup size, the cells per thread and the layout; the driver
    /// dispatches [`Grid::dispatch`] (`ceil(d / (x bx)) × ceil(r_rows / (y by))`,
    /// the two swapped with the rows on x).
    pub grid: Grid,
    period_ns: f32,
    /// The work guard: what is known of the kernel's work per invocation. `Fuel::Proven` (never a default: every constructor takes its fuel; the
    /// constructors) is for the shipped and hand kernels; a random gene, an evolved child or a seed gets
    /// [`Gram::with_fuel`] before any run, and every submit is checked against the cap first.
    fuel: Fuel,
}

#[cfg(feature = "gpu")]
impl<'a> Gram<'a> {
    /// Compile `source` (naga-validated first, so an ill-formed variant is
    /// refused with its message, not a device error). `workgroup` is the
    /// kernel's `@workgroup_size` in x (columns) and y (rows); the driver
    /// dispatches `ceil(d / x) × ceil(r_rows / y)`.
    pub fn compile(dev: &'a Device, source: &str, fuel: Fuel) -> Result<Gram<'a>, String> {
        Self::compile_with(dev, source, grid_of(source)?, fuel)
    }

    fn compile_with(dev: &'a Device, source: &str, grid: Grid, fuel: Fuel) -> Result<Gram<'a>, String> {
        fuel.refuse_unported()?;
        if !dev.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err("this device has no timestamp queries".into());
        }
        let scope = crate::wgsl::oracle::device::ErrorScopes::push(&dev.device);
        let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gram"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)) });
        // An explicit layout over the role's three bindings: a kernel that
        // reads none of them (a random gene may) still binds them all.
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gram"), entries: &[entry(0, wgpu::BufferBindingType::Storage { read_only: true }), entry(1, wgpu::BufferBindingType::Storage { read_only: false }), entry(2, wgpu::BufferBindingType::Uniform)] });
        let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("gram"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("gram"), layout: Some(&layout), module: &shader, entry_point: Some(ENTRY), compilation_options: Default::default(), cache: None });
        if let Some(e) = scope.pop() {
            return Err(format!("pipeline: {e}"));
        }
        Ok(Gram { dev, pipeline, grid, period_ns: dev.queue.get_timestamp_period(), fuel })
    }


    /// Build `H` over `phi` (item-major `m × d`) as production does, under
    /// `limit` bytes per binding: upload, run with the read-back, free.
    pub fn build(&self, phi: &[f32], d: usize, m: usize, limit: u64) -> Result<Build, String> {
        let r = Resident::upload(self.dev, phi, d, m, limit)?;
        let mut b = self.run(&r, true)?;
        b.upload_ms = r.upload_ms;
        r.free();
        Ok(b)
    }

    /// Run over Φ already on the device. With `read_h` false the result is
    /// not read back (`Build::h` is empty): a timing build, whose output
    /// nothing reads. The passes, their order and their timestamps are the
    /// same either way; `upload_ms` is 0 (the upload was earlier).
    pub fn run(&self, r: &Resident, read_h: bool) -> Result<Build, String> {
        let (d, limit) = (r.d, r.limit);
        let device = &self.dev.device;
        let queue = &self.dev.queue;
        let bands = band_layout(d, r.m, limit);
        let row_band_rows = bands.band_rows;
        let item_bands = r.bands.len();
        let row_bands = bands.row_bands;
        let phi_bands = &r.bands;
        // The work guard, before anything is encoded or submitted: the largest submit is one row band.
        self.fuel.check("a Gram run", submit_invocations(self.grid, d, row_band_rows.min(d), item_bands))?;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("gram timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * item_bands as u32 });
        let stamps_bytes = 16 * item_bands as u64;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram timestamps resolve"), size: stamps_bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let stamps_read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram timestamps read"), size: stamps_bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let layout = self.pipeline.get_bind_group_layout(0);

        let mut h = if read_h { vec![0.0f32; d * d] } else { Vec::new() };
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
            let (wg_x, wg_y) = self.grid.dispatch(r_rows as u32, d as u32);
            let staging = read_h.then(|| device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram staging"), size: band_bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
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
            wait_gpu(device, queue)?;
            compute_ms += t_compute.elapsed().as_secs_f64() * 1e3;
            // Then the timestamps and (when asked) the band, read back.
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram read-back") });
            enc.resolve_query_set(&query_set, 0..2 * item_bands as u32, &resolve_buf, 0);
            enc.copy_buffer_to_buffer(&resolve_buf, 0, &stamps_read, 0, stamps_bytes);
            if let Some(st) = &staging {
                enc.copy_buffer_to_buffer(&h_buffer, 0, st, 0, band_bytes);
            }
            queue.submit(Some(enc.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            if let Some(st) = &staging {
                st.slice(..).map_async(wgpu::MapMode::Read, { let tx = tx.clone(); move |r| { tx.send(r).expect("the receiver waits for the mapping"); } });
            }
            stamps_read.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).expect("the receiver waits for the mapping"); });
            wait_gpu(device, queue)?;
            for _ in 0..(1 + usize::from(staging.is_some())) {
                rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
            }
            if let Some(st) = &staging {
                {
                    let mapped = crate::gpu_io::mapped(&st.slice(..))?;
                    let band: &[f32] = bytemuck::cast_slice(&mapped);
                    h[r_off * d..(r_off + r_rows) * d].copy_from_slice(band);
                }
                st.unmap();
                st.destroy();
            }
            {
                let mapped = crate::gpu_io::mapped(&stamps_read.slice(..))?;
                let stamps: &[u64] = bytemuck::cast_slice(&mapped);
                for pair in stamps.chunks(2) {
                    gpu_ns += pair[1].wrapping_sub(pair[0]) as f64 * f64::from(self.period_ns);
                }
            }
            stamps_read.unmap();
            h_buffer.destroy();
            r_off += r_rows;
        }
        let wall_ms = t1.elapsed().as_secs_f64() * 1e3;
        crate::gpu_io::drain(device)?;
        Ok(Build { h, gpu_ms: gpu_ns / 1e6, compute_ms, wall_ms, upload_ms: 0.0, passes: item_bands * row_bands, item_bands, row_bands })
    }
}

/// The result of [`Gram::run_batch`]: each case's H and GPU milliseconds,
/// and the one submission's wall milliseconds, submit to completion.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub h: Vec<Vec<f32>>,
    pub gpu_ms: Vec<f64>,
    pub compute_ms: f64,
}

#[cfg(feature = "gpu")]
impl Gram<'_> {
    /// Run over several Φ already on the device in ONE compute submission
    /// and one read-back: every case's passes are those [`Gram::run`]
    /// encodes (each output-row band cleared, then one pass per item band,
    /// the same dispatch), each into its own accumulator, so each case's H
    /// is what `run` gives. Returns each case's H and its GPU milliseconds
    /// (its passes' timestamps), and the submission's wall milliseconds,
    /// submit to completion, for all of them together.
    pub fn run_batch(&self, rs: &[&Resident]) -> Result<Batch, String> {
        let device = &self.dev.device;
        let queue = &self.dev.queue;
        let layout = self.pipeline.get_bind_group_layout(0);
        // Every (case, row band): its accumulator, its offset in the staging buffer.
        struct Band { case: usize, r_off: usize, r_rows: usize, h: wgpu::Buffer, at: u64, binds: Vec<wgpu::BindGroup>, first_query: u32 }
        let mut bands: Vec<Band> = Vec::new();
        let mut keep_cfgs: Vec<wgpu::Buffer> = Vec::new();
        let (mut staging_bytes, mut queries) = (0u64, 0u32);
        for (case, r) in rs.iter().enumerate() {
            let d = r.d;
            let row_band_rows = band_layout(d, r.m, r.limit).band_rows;
            let mut r_off = 0usize;
            while r_off < d {
                let r_rows = (r_off + row_band_rows).min(d) - r_off;
                let band_bytes = (r_rows * d * 4) as u64;
                let h = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram batch accumulator"), size: band_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                let mut binds = Vec::with_capacity(r.bands.len());
                for (pb, mb) in &r.bands {
                    let cfg = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gram cfg"), contents: bytemuck::cast_slice(&[d as u32, *mb, r_rows as u32, r_off as u32]), usage: wgpu::BufferUsages::UNIFORM });
                    binds.push(device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gram bind"), layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: pb.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: h.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: cfg.as_entire_binding() }] }));
                    keep_cfgs.push(cfg);
                }
                let first_query = queries;
                queries += 2 * r.bands.len() as u32;
                bands.push(Band { case, r_off, r_rows, h, at: staging_bytes, binds, first_query });
                staging_bytes += band_bytes;
                r_off += r_rows;
            }
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("gram batch timestamps"), ty: wgpu::QueryType::Timestamp, count: queries.max(2) });
        let stamps_bytes = 8 * u64::from(queries.max(2));
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram batch resolve"), size: stamps_bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let stamps_read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram batch stamps read"), size: stamps_bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gram batch staging"), size: staging_bytes.max(4), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        // The work guard, before the batch is encoded: the whole batch is one submit.
        let batch_invocations = bands.iter().fold(0u64, |a, b| a.saturating_add(submit_invocations(self.grid, rs[b.case].d, b.r_rows, b.binds.len())));
        self.fuel.check("a Gram batch", batch_invocations)?;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram batch") });
        for b in &bands {
            let d = rs[b.case].d;
            enc.clear_buffer(&b.h, 0, None);
            let (wg_x, wg_y) = self.grid.dispatch(b.r_rows as u32, d as u32);
            for (i, bind) in b.binds.iter().enumerate() {
                let q = b.first_query + 2 * i as u32;
                let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &query_set, beginning_of_pass_write_index: Some(q), end_of_pass_write_index: Some(q + 1) });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gram batch item band"), timestamp_writes });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        let t_compute = Instant::now();
        queue.submit(Some(enc.finish()));
        wait_gpu(device, queue)?;
        let compute_ms = t_compute.elapsed().as_secs_f64() * 1e3;
        // The timestamps resolved after the passes completed (resolved in the
        // same submission, short passes read 0 on Metal), and every band read back.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram batch read-back") });
        enc.resolve_query_set(&query_set, 0..queries, &resolve_buf, 0);
        enc.copy_buffer_to_buffer(&resolve_buf, 0, &stamps_read, 0, 8 * u64::from(queries));
        for b in &bands {
            enc.copy_buffer_to_buffer(&b.h, 0, &staging, b.at, (b.r_rows * rs[b.case].d * 4) as u64);
        }
        queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, { let tx = tx.clone(); move |r| { tx.send(r).expect("the receiver waits for the mapping"); } });
        stamps_read.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).expect("the receiver waits for the mapping"); });
        wait_gpu(device, queue)?;
        for _ in 0..2 {
            rx.recv().map_err(|e| format!("map: {e}"))?.map_err(|e| format!("map: {e}"))?;
        }
        let mut hs: Vec<Vec<f32>> = rs.iter().map(|r| vec![0.0f32; r.d * r.d]).collect();
        let mut gpu_ms = vec![0.0f64; rs.len()];
        {
            let mapped = crate::gpu_io::mapped(&staging.slice(..))?;
            let all: &[f32] = bytemuck::cast_slice(&mapped);
            let stamps_map = crate::gpu_io::mapped(&stamps_read.slice(..))?;
            let stamps: &[u64] = bytemuck::cast_slice(&stamps_map);
            for b in &bands {
                let d = rs[b.case].d;
                let at = (b.at / 4) as usize;
                hs[b.case][b.r_off * d..(b.r_off + b.r_rows) * d].copy_from_slice(&all[at..at + b.r_rows * d]);
                for i in 0..b.binds.len() {
                    let q = (b.first_query + 2 * i as u32) as usize;
                    gpu_ms[b.case] += stamps[q + 1].wrapping_sub(stamps[q]) as f64 * f64::from(self.period_ns) / 1e6;
                }
            }
        }
        staging.unmap();
        stamps_read.unmap();
        staging.destroy();
        for b in &bands {
            b.h.destroy();
        }
        drop(keep_cfgs);
        crate::gpu_io::drain(device)?;
        Ok(Batch { h: hs, gpu_ms, compute_ms })
    }
}

/// Φ on the device in the driver's item bands, uploaded once and run over
/// by any number of kernels ([`Gram::run`]).
#[cfg(feature = "gpu")]
pub struct Resident {
    pub d: usize,
    pub m: usize,
    pub limit: u64,
    bands: Vec<(wgpu::Buffer, u32)>,
    pub upload_ms: f64,
}

#[cfg(feature = "gpu")]
impl Resident {
    /// Upload `phi` (item-major `m × d`) in item bands sized from `limit`.
    pub fn upload(dev: &Device, phi: &[f32], d: usize, m: usize, limit: u64) -> Result<Resident, String> {
        if phi.len() != d * m {
            return Err(format!("phi has {} values, not m × d = {}", phi.len(), d * m));
        }
        let row_bytes = d as u64 * 4;
        if row_bytes > limit {
            return Err(format!("one d-length row of {row_bytes} bytes exceeds the limit {limit}"));
        }
        let BandLayout { band_items, item_bands, .. } = band_layout(d, m, limit);
        let t0 = Instant::now();
        let bands = (0..item_bands)
            .map(|i| {
                let start = i * band_items;
                let end = (start + band_items).min(m);
                let buffer = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gram phi band"), contents: bytemuck::cast_slice(&phi[start * d..end * d]), usage: wgpu::BufferUsages::STORAGE });
                (buffer, (end - start) as u32)
            })
            .collect();
        dev.queue.submit(None);
        wait_gpu(&dev.device, &dev.queue)?;
        Ok(Resident { d, m, limit, bands, upload_ms: t0.elapsed().as_secs_f64() * 1e3 })
    }

    /// Release the device memory now rather than at drop.
    pub fn free(&self) {
        for (b, _) in &self.bands {
            b.destroy();
        }
    }
}

/// The two timing rungs (d, m) of the old Gram fitness: from the largest
/// correctness case (d 40 m 50) to the smallest accuracy pair (d 512
/// m 2,048) in steps of at most 32x item-cells.
pub const RUNGS: [(usize, usize); 2] = [(128, 64), (256, 512)];

/// The median of a sample (the mean of the middle two for an even count).
pub fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_is_the_middle_of_the_sorted_sample() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut [7.0]), 7.0);
        assert_eq!(median(&mut []), 0.0);
    }

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

    /// The parallel check returns the serial check's max ratio and
    /// non-finite count to the bit, on random cells with NaN, ±inf, a zero
    /// column and exact cells.
    #[test]
    fn check_max_is_bit_identical_to_check() {
        for (d, m, seed) in [(1usize, 4usize, 1u64), (7, 9, 2), (64, 100, 3), (129, 33, 4), (300, 1000, 5)] {
            let p = phi_of(Kind::NonFinite, d, m, seed);
            let mut h_ref = reference(&p, d, m);
            if d > 3 {
                for c in 0..d {
                    h_ref[3 * d + c] = 0.0;
                    h_ref[c * d + 3] = 0.0;
                }
            }
            let mut state = seed;
            let h: Vec<f32> = h_ref.iter().map(|x| { let u = unit(&mut state); match (u * 50.0) as u32 { 0 => f32::NAN, 1 => f32::INFINITY, 2 => *x as f32 + 1.0, _ => (*x * (1.0 + 1e-7 * (u - 0.5))) as f32 } }).collect();
            let v = check(&h, &h_ref, d, m);
            let (mx, nf) = check_max(&h, &h_ref, d, m);
            assert_eq!((mx.to_bits(), nf), (v.max_ratio.to_bits(), v.non_finite), "d {d} m {m}");
        }
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

    /// Every build reports a compute time, build after build, shape after
    /// shape; the timestamp sum is printed beside it (it reads 0 for short
    /// passes on Apple silicon, which is why the search uses compute_ms).
    #[cfg(feature = "gpu")]
    #[test]
    fn every_build_in_a_sequence_has_a_compute_time_on_the_device() {
        let dev = Device::new().expect("device");
        let gram = Gram::compile(&dev, SHIPPED, Fuel::Proven("the shipped Gram kernel")).expect("compile");
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

    /// A RUNAWAY IS NAMED AND NOTHING OF IT IS READ (run 6, 2026-10-13: a
    /// kernel still running after wgpu's 5 s wait was read, and the shipped
    /// kernel's timestamps after it read zero). A kernel that spins past the
    /// wait (the shipped kernel behind a dependent chain of 3e9 multiply-adds
    /// a thread, its result written so it is kept) is an error naming the
    /// runaway, returned only after the GPU drained; the shipped kernel timed
    /// right after it times above zero. On the device: about 30 s.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_runaway_is_named_and_the_next_timing_is_not_zero_on_the_device() {
        let dev = Device::new().expect("device");
        let limit = effective_buffer_limit(&dev.device);
        let (d, m) = (32usize, 64usize);
        let p = phi(d, m, 3);
        let spin = SHIPPED.replacen(
            "    let lr = gid.y;",
            "    var spin_acc: f32 = f32(gid.x) * 1e-9;\n    for (var i: u32 = 0u; i < 3000000000u; i = i + 1u) { spin_acc = spin_acc * 1.0000001 + 1e-9; }\n    if (spin_acc == 12345.0) { h[0] = spin_acc; }\n    let lr = gid.y;",
            1,
        );
        assert_ne!(spin, SHIPPED, "the spin was not inserted");
        let gram = Gram::compile(&dev, &spin, Fuel::Proven("a hand or test kernel of the oracle")).expect("the spinning kernel compiles");
        let t = Instant::now();
        let r = gram.build(&p, d, m, limit);
        let secs = t.elapsed().as_secs_f64();
        println!("the spinning kernel: {:.1} s, {}", secs, match &r { Ok(_) => "read".to_string(), Err(e) => e.clone() });
        assert!(secs * 1e3 >= WAIT_GAVE_UP_MS, "the spin ran {secs:.1} s, under the wait it is meant to outlast: lengthen it");
        let e = match r {
            Ok(_) => panic!("a kernel the wait gave up on was read"),
            Err(e) => e,
        };
        assert!(e.contains("runaway"), "{e}");
        let shipped = Gram::compile(&dev, SHIPPED, Fuel::Proven("the shipped Gram kernel")).expect("compile");
        let b = shipped.build(&phi(64, 1000, 1), 64, 1000, limit).expect("the shipped kernel after the runaway");
        println!("the shipped kernel after it: {:.3} ms on the GPU", b.gpu_ms);
        assert!(b.gpu_ms > 0.0, "the shipped kernel timed at zero after the runaway");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_grid_header_is_read_and_checked_against_the_shader_on_the_device() {
        assert_eq!(grid_header(TILED), Some((16, 16, 4, 4)));
        assert_eq!(grid_header(SHIPPED), None);
        let dev = Device::new().expect("device");
        assert_eq!(Gram::compile(&dev, TILED, Fuel::Proven("the shipped Gram kernel")).unwrap().grid, Grid { x: 16, y: 16, bx: 4, by: 4, rows_x: false });
        let wrong = TILED.replace("// grid 16 16 4 4", "// grid 8 8 4 4");
        let err = match Gram::compile(&dev, &wrong, Fuel::Proven("a hand or test kernel of the oracle")) { Ok(_) => panic!("a mismatched header compiled"), Err(e) => e };
        assert!(err.contains("the grid header says workgroup 8 x 8"), "{err}");
    }

    /// The hand-written kernels (plan step 0) agree with the reference
    /// under the criterion, banded and unbanded, at ragged shapes.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_tiled_and_blocked_kernels_pass_the_criterion_on_the_device() {
        let dev = Device::new().expect("device");
        for (name, text) in [("tiled", TILED), ("blocked", BLOCKED)] {
            let gram = Gram::compile(&dev, text, Fuel::Proven("a hand or test kernel of the oracle")).expect("compile");
            assert_eq!(gram.grid, Grid { x: 16, y: 16, bx: 4, by: 4, rows_x: false });
            for (d, m, limit) in [(64usize, 1000usize, 4096u64), (130, 77, 8192), (200, 33, u64::MAX), (64, 1000, u64::MAX), (33, 5, 1024)] {
                let p = phi(d, m, 11);
                let h_ref = reference(&p, d, m);
                let b = gram.build(&p, d, m, limit).expect("build");
                let v = check(&b.h, &h_ref, d, m);
                assert!(v.passes(), "{name} d {d} m {m} limit {limit}: {v:?}");
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn shipped_kernel_passes_the_criterion_on_the_device() {
        // Small enough to run in a test, large enough to band: d = 64 rows
        // of 256 B, the limit set to 4 KiB so item and row bands are both 16.
        let (d, m) = (64, 1000);
        let p = phi(d, m, 11);
        let h_ref = reference(&p, d, m);
        let dev = Device::new().expect("device");
        let gram = Gram::compile(&dev, SHIPPED, Fuel::Proven("the shipped Gram kernel")).expect("compile");
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
