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

//!
//! The device parts (the driver, the residents, the fitness context)
//! compile under `gpu`; the reference, the criterion, the cost model and
//! the fitness procedure ([`fitness_with`], over any [`Driver`]) need only
//! `wgsl`, so the generic harness (`harness.rs`) is compared with this
//! procedure on the CPU.

use std::time::Instant;

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
use crate::wgsl::oracle::device::Device;
use crate::wgsl::render::Grid;
#[cfg(feature = "gpu")]
use crate::wgsl::render::Rendered;

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

/// The hand-written register-blocked kernel (no workgroup memory): 3.6x
/// the shipped kernel's speed at the bench size (280 ms against 1,012 ms,
/// 2026-10-10), the positive control that passed.
pub const BLOCKED: &str = include_str!("../../kingdoms/wgsl/gram/blocked.wgsl");

/// The shipped kernel, verbatim from `H_GRAM_ACCUM_WGSL`.
pub const SHIPPED: &str = include_str!("../../kingdoms/wgsl/gram/shipped.wgsl");

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
    super::safe_reader::grid_header(source)
}

/// Whether the header says `rows-x`: the output's rows on thread x.
pub fn grid_rows_x(source: &str) -> bool {
    super::safe_reader::grid_rows_x(source)
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
/// [`Resident::upload`] and [`Gram::run`] issue it: what the generic
/// harness's plan of `problems/gram` must equal (`harness` tests).
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
}

#[cfg(feature = "gpu")]
impl<'a> Gram<'a> {
    /// Compile `source` (naga-validated first, so an ill-formed variant is
    /// refused with its message, not a device error). `workgroup` is the
    /// kernel's `@workgroup_size` in x (columns) and y (rows); the driver
    /// dispatches `ceil(d / x) × ceil(r_rows / y)`.
    pub fn compile(dev: &'a Device, source: &str) -> Result<Gram<'a>, String> {
        Self::compile_with(dev, source, grid_of(source)?)
    }

    /// A rendered kernel: its text and the grid it asked for.
    pub fn compile_rendered(dev: &'a Device, r: &Rendered) -> Result<Gram<'a>, String> {
        crate::wgsl::naga_shape::parse(&r.text).and_then(|m| super::target::Target::current().validate(&m, &r.text).map(|_| ()).map_err(|e| format!("validation: {e}")))?;
        Self::compile_with(dev, &r.text, r.grid)
    }

    fn compile_with(dev: &'a Device, source: &str, grid: Grid) -> Result<Gram<'a>, String> {
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
        Ok(Gram { dev, pipeline, grid, period_ns: dev.queue.get_timestamp_period() })
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

/// The evaluation's time budget before its timing rounds: compile, the
/// 128 correctness cases, the timing rungs and the six accuracy pairs.
/// phylu's `wgsl_fit` watchdog ends the process when an evaluation passes
/// `BASE_BUDGET_MS + ROUND_BUDGET_MS * timing_rounds` (a Metal dispatch
/// cannot be cancelled; on 2026-10-08 one froze the machine), and it reads
/// both figures from here so the gate and the watchdog cannot disagree.
pub const BASE_BUDGET_MS: f64 = 30_000.0;

/// The time budget of one timing round: the candidate and the shipped
/// kernel built once each at the three timing shapes.
pub const ROUND_BUDGET_MS: f64 = 20_000.0;

/// The two timing rungs (d, m) run between the correctness cases and the
/// accuracy pairs. The largest correctness case is d 40 m 50 (8e4
/// item-cells) and the smallest pair d 512 m 2,048 (5.4e8): a step of
/// 6,700x, too far to predict across. The rungs (1.0e6, then 3.4e7
/// item-cells) make every step from one measured build to the next at
/// most 32x. Their output is not judged: the pairs judge accuracy.
pub const RUNGS: [(usize, usize); 2] = [(128, 64), (256, 512)];

/// The seed of the rungs' Phi ([`phi`]).
pub const RUNG_SEED: u64 = 13;

/// The three timing shapes (d, m), 2.75e11 item-cells each, and the seed
/// of their Phi ([`phi`]).
pub const TIMING: [(usize, usize); 3] = [(2048, 65536), (4096, 16384), (8192, 4096)];
pub const TIMING_SEED: u64 = 7;

/// The correctness cases run this many to a submission.
pub const CASE_CHUNK: usize = 16;

/// The work of one build: d x d output cells, each a sum over m items.
pub fn work(d: usize, m: usize) -> f64 {
    d as f64 * d as f64 * m as f64
}

/// THE MEASURED COST MODEL. A kernel's wall time on the device (submit to
/// completion, `Build::compute_ms`) is modelled as a floor plus a rate per
/// item-cell: `ms = floor + rate * d * d * m`. The floor is the median
/// time of the correctness cases, whose work (at most 8e4 item-cells) is
/// below the submission's own cost; the rate is the least-squares slope of
/// every sample so far through that floor, `sum w (t - floor) / sum w^2`,
/// clamped at 0. The `w^2` weighting makes the largest measured build
/// dominate the slope, which is the build closest to the next one; the
/// noise of a correctness case (a millisecond over 8e4 item-cells) moves a
/// prediction one step ahead by milliseconds, not seconds.
///
/// The model sees only `d * d * m`. A kernel whose time scales another way
/// (a loop over d inside the item loop) is caught because every gated
/// build is at most 32x the work of a measured one; the static cost
/// estimate (`safe_karva::default_bound`) stays as the loose guard
/// against a kernel too big to run even at the correctness cases' sizes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Meter {
    /// (work in item-cells, wall milliseconds), in the order measured.
    pub samples: Vec<(f64, f64)>,
    /// The floor in ms, fixed by [`Meter::fix_floor`] after the correctness cases.
    pub floor_ms: f64,
}

impl Meter {
    pub fn push(&mut self, work: f64, ms: f64) {
        self.samples.push((work, ms));
    }

    /// The floor: the median of the samples so far (the correctness cases).
    pub fn fix_floor(&mut self) {
        let mut t: Vec<f64> = self.samples.iter().map(|s| s.1).collect();
        self.floor_ms = median(&mut t);
    }

    /// Milliseconds per item-cell, the least-squares slope through the floor.
    pub fn rate(&self) -> f64 {
        let (num, den) = self.samples.iter().fold((0.0, 0.0), |(n, d), &(w, t)| (n + w * (t - self.floor_ms), d + w * w));
        if den > 0.0 { (num / den).max(0.0) } else { 0.0 }
    }

    /// The predicted wall milliseconds of a build of `work` item-cells.
    pub fn predict(&self, work: f64) -> f64 {
        self.floor_ms + self.rate() * work
    }

    /// The rate in ms per 1e9 item-cells, for the reports.
    pub fn rate_per_g(&self) -> f64 {
        self.rate() * 1e9
    }
}

/// The time limit of one candidate build at a timing shape: a timing round
/// builds the candidate and the shipped kernel once at each of the three
/// shapes inside `round_budget_ms`, so after the shipped kernel's three
/// measured times the candidate has a third of what remains per shape
/// (`(20 s - shipped) / 3`, about 5.7 s with the shipped kernel at about
/// 1 s a shape). A kernel over it is at least five times slower than the
/// shipped kernel, so its speed columns are already pinned at 1 and
/// stopping it loses no ranking; it only keeps the watchdog from ending
/// the run.
pub fn slow_limit_ms(round_budget_ms: f64, shipped_ms: &[f64; 3]) -> f64 {
    (round_budget_ms - shipped_ms.iter().sum::<f64>()) / 3.0
}

/// The gate before a build: `Err` with the detail of a "slow" outcome when
/// the predicted time of the build exceeds `limit_ms`, or when the build
/// would carry the evaluation past `budget_ms` (from its start, `elapsed_ms`
/// already spent). `what` names the build (the shape and its role).
pub fn gate(meter: &Meter, what: &str, work: f64, elapsed_ms: f64, limit_ms: f64, budget_ms: f64) -> Result<f64, String> {
    let p = meter.predict(work);
    let tail = format!("from a measured rate of {:.1} ms per 1e9 item-cells over a floor of {:.2} ms ({} builds measured, the largest {:.2e} item-cells)", meter.rate_per_g(), meter.floor_ms, meter.samples.len(), meter.samples.iter().map(|s| s.0).fold(0.0, f64::max));
    if p > limit_ms {
        return Err(format!("slow: predicted {:.1} s at {what} ({work:.2e} item-cells), over the limit of {:.1} s a build; {tail}; not dispatched", p / 1e3, limit_ms / 1e3));
    }
    if elapsed_ms + p > budget_ms {
        return Err(format!("slow: predicted {:.1} s at {what} ({work:.2e} item-cells) after {:.1} s spent, over the evaluation's budget of {:.1} s; {tail}; not dispatched", p / 1e3, elapsed_ms / 1e3, budget_ms / 1e3));
    }
    Ok(p)
}

/// Where an evaluation's time went, in wall milliseconds per phase, and
/// inside the device builds of all phases: the Phi uploads, and the
/// read-backs with their buffer set-up (a build's wall time less its
/// compute time). [`Split::add`] sums evaluations for a generation line.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Split {
    pub compile: f64,
    pub cases: f64,
    pub rungs: f64,
    /// The pairs' device builds, and the host check against the f64 reference.
    pub pairs_build: f64,
    pub pairs_check: f64,
    /// The candidate's timing builds (the shipped kernel is timed once a
    /// generation, [`FitnessContext::time_shipped`]).
    pub timing_cand: f64,
    pub upload: f64,
    pub readback: f64,
    /// The whole evaluation.
    pub total: f64,
}

impl Split {
    pub fn add(&mut self, o: &Split) {
        self.compile += o.compile;
        self.cases += o.cases;
        self.rungs += o.rungs;
        self.pairs_build += o.pairs_build;
        self.pairs_check += o.pairs_check;
        self.timing_cand += o.timing_cand;
        self.upload += o.upload;
        self.readback += o.readback;
        self.total += o.total;
    }

    pub fn build(&mut self, b: &Build) {
        self.upload += b.upload_ms;
        self.readback += (b.wall_ms - b.compute_ms).max(0.0);
    }

    /// One line, seconds: "compile 0.2, cases 0.4, ... (uploads 0.3, read-backs 0.9)".
    pub fn line(&self) -> String {
        format!("compile {:.1}, cases {:.1}, rungs {:.1}, pairs {:.1} + check {:.1}, timing {:.1}, other {:.1} (inside the builds: uploads {:.1}, read-backs {:.1})", self.compile / 1e3, self.cases / 1e3, self.rungs / 1e3, self.pairs_build / 1e3, self.pairs_check / 1e3, self.timing_cand / 1e3, (self.total - self.compile - self.cases - self.rungs - self.pairs_build - self.pairs_check - self.timing_cand) / 1e3, self.upload / 1e3, self.readback / 1e3)
    }
}

/// The fast fitness evaluation phylu calls per individual
/// (`docs/PLAN_hgp_gram.md`, step 2 of the evolutionary loop): the
/// constraint on its 128 stratified test cases, then six columns scaled
/// into [0, 1] with 0 best — the time ratio against the shipped kernel on
/// three problem sizes (one paired round each), and the worst error
/// relative to the shipped kernel's on a few training, validation and
/// extrapolation pairs — and the TrueNorth angle θ = acos(1 − mean(x²))
/// (phylu's `hff.wgsl`, lines 212–213). About ten seconds for a feasible
/// individual, two for an infeasible one.
/// An accuracy pair for the fast fitness: name, d, m, Φ, the f64 reference, the shipped kernel's largest ratio on it.
pub type AccuracyPair = (String, usize, usize, Vec<f32>, Vec<f64>, f64);

#[cfg(feature = "gpu")]
pub struct FitnessContext {
    pub probes: Vec<crate::wgsl::probes::Probe>,
    pub admission: Vec<usize>,
    pub shipped_source: String,
    /// The three timing shapes (d, m) and their Φ.
    pub timing: Vec<(usize, usize, Vec<f32>)>,
    /// Accuracy pairs by split: (name, d, m, phi, h_ref, shipped's max ratio).
    pub train: Vec<AccuracyPair>,
    pub val: Vec<AccuracyPair>,
    pub extrap: Vec<AccuracyPair>,
    pub limit: u64,
    /// Paired timing rounds per shape; the speed-up is the median over them.
    /// One round scattered equal kernels by 15 % (the re-timing of
    /// 2026-10-11); five or more are needed to rank kernels.
    pub timing_rounds: usize,
    /// The timing rungs (d, m, Φ), [`RUNGS`].
    pub rungs: Vec<(usize, usize, Vec<f32>)>,
    /// The shipped kernel's wall milliseconds at the three timing shapes,
    /// measured once when the context is built.
    pub shipped_ms: [f64; 3],
    /// The budgets the gate holds a kernel to; [`BASE_BUDGET_MS`] and
    /// [`ROUND_BUDGET_MS`] unless the caller sets them.
    pub base_budget_ms: f64,
    pub round_budget_ms: f64,
    /// Φ of the rungs, the pairs (train, val, extrap, in their order) and
    /// the timing shapes, uploaded once at [`FitnessContext::new`] and kept
    /// on the device (about 0.9 GB, most of it the timing shapes).
    pub resident_rungs: Vec<Resident>,
    /// Φ of the 128 correctness cases, in `admission` order.
    pub resident_cases: Vec<Resident>,
    pub resident_pairs: [Vec<Resident>; 3],
    pub resident_timing: Vec<Resident>,
    /// The shipped kernel's GPU milliseconds at the three timing shapes as
    /// [`FitnessContext::time_shipped`] last measured them (once a
    /// generation in phylu's `wgsl_fit`).
    pub shipped_gpu_ms: std::cell::Cell<[f64; 3]>,
    /// The least of every measurement so far, per shape: the denominator of
    /// every speed-up. Contention only slows a kernel, so the least time is
    /// the shipped kernel's own; dividing by a contended time inflated
    /// every speed-up of a generation by up to 60 % (fit_B gens 12-16,
    /// 2026-10-11: shipped 1,013-1,787 ms, a 10.7x kernel scored as 17x).
    pub shipped_min_gpu_ms: std::cell::Cell<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fitness {
    pub feasible: bool,
    /// Test cases of the constraint passed, of 128.
    pub passed: usize,
    /// The error the evaluation stopped on, for a graded infeasible fitness
    /// (2026-10-12, Andrew: evolution climbs toward correctness), 0 a match:
    /// at the constraint, the mean `probes::distance` over the cases run; at
    /// a pair, its worst error over the tolerance (`max / max(2 shipped, 1)`,
    /// above 1 when it fails), infinite with a non-finite output; infinite
    /// where no output was compared (compile, the timing gate, a run error
    /// after the cases); 0 when feasible.
    pub distance: f64,
    /// HOW CLOSE an infeasible kernel's output is to the reference, column by
    /// column in [0, 1], 0 the reference (`closeness::COLUMNS`: the cases
    /// failed, correlation, scale, shift, transpose, sorted values, sums,
    /// cells written, cells within tolerance, input sensitivity), from the
    /// cases' outputs and a failing pair's; all 0 when feasible, all 1 with
    /// no output to compare.
    pub closeness: [f64; crate::wgsl::closeness::N],
    /// The HFF angle of `closeness` but its logged-only columns
    /// (`closeness::angle`; `scale` is logged, not in it), in [0, π/2]: the
    /// infeasible kernel's place within the infeasible band (phylu
    /// `wgsl_fit`); 0 when feasible.
    pub closeness_hff: f64,
    /// The six columns in [0, 1], 0 best, when feasible.
    pub columns: [f64; 6],
    /// Speed-up per timing shape (shipped / candidate), when feasible.
    pub speedups: [f64; 3],
    /// θ, when feasible; `f64::INFINITY` otherwise. With `inputs`, θ over
    /// them; else over `columns`.
    pub hff: f64,
    /// THE HFF INPUTS of a problem that measures a vector (the v24
    /// `pooled_recall`: per window x_M, x_P, x_B, then one per timing
    /// sample; harness `pooled_inputs`), each in [0, 1], 0 best, θ the hff
    /// library's TrueNorth over them (`score::truenorth_angle`); empty for
    /// the six-column problems (the Gram, nluca, kmeans), whose θ is over
    /// `columns`.
    pub inputs: Vec<f64>,
    pub detail: String,
    pub seconds: f64,
    /// The predicted wall milliseconds at the three timing shapes, from the
    /// measured rate (0 when the evaluation stopped before predicting).
    pub predicted_ms: [f64; 3],
    /// The measured rate in ms per 1e9 item-cells and the floor in ms.
    pub rate_per_g: f64,
    pub floor_ms: f64,
    /// The candidate's measured wall milliseconds at the three timing
    /// shapes, the median over the rounds (0 when it was not timed).
    pub measured_ms: [f64; 3],
    /// Where the evaluation's time went.
    pub split: Split,
}

/// θ over columns in [0, 1], 0 best: the hff library's TrueNorth angle
/// (normalisation off, the all-zero pole, `crate::score::truenorth_angle`),
/// each column clamped into [0, 1] first. The clamp is the caller's column
/// scaling; the angle and its edge cases (the pole test at `f64::EPSILON`,
/// π for a non-finite column) are the library's.
pub fn truenorth(x: &[f64]) -> f64 {
    let columns: Vec<f64> = x.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    crate::score::truenorth_angle(&columns)
}

#[cfg(feature = "gpu")]
impl FitnessContext {
    /// Build the context once: the test cases, three training pairs, two
    /// validation pairs and one extrapolation pair from the splits, the
    /// three timing shapes, and the shipped kernel's own error on each pair.
    pub fn new(dev: &Device, cache: &std::path::Path) -> Result<FitnessContext, String> {
        use crate::wgsl::pairs::splits;
        let probes = crate::wgsl::probes::probes(1);
        let admission = crate::wgsl::probes::admission(&probes, 1);
        let limit = effective_buffer_limit(&dev.device);
        let shipped = Gram::compile(dev, SHIPPED)?;
        let sp = splits(cache)?;
        let pick = |pairs: &[crate::wgsl::pairs::Pair], idx: &[usize]| -> Result<Vec<AccuracyPair>, String> {
            idx.iter()
                .map(|&i| {
                    let p = &pairs[i];
                    let b = shipped.build(&p.phi, p.d, p.m, limit)?;
                    let v = check(&b.h, &p.h_ref, p.d, p.m);
                    Ok((p.name(), p.d, p.m, p.phi.clone(), p.h_ref.clone(), v.max_ratio))
                })
                .collect()
        };
        // Three kinds on train (gaussian, weighted, ragged), two on validation, one production-width pair.
        // (The selection lives in `pairs::fitness_pick`, which the problem
        // directory's generator uses too.)
        let [train_idx, val_idx, extrap_idx] = crate::wgsl::pairs::fitness_pick(&crate::wgsl::pairs::split_shapes())?;
        let timing: Vec<(usize, usize, Vec<f32>)> = TIMING.iter().map(|&(d, m)| (d, m, phi(d, m, TIMING_SEED))).collect();
        // The shipped kernel's wall time at each timing shape, after one
        // warm-up build at the first, so the limit comes from this device.
        shipped.build(&timing[0].2, timing[0].0, timing[0].1, limit)?;
        let mut shipped_ms = [0.0f64; 3];
        for (k, (d, m, p)) in timing.iter().enumerate() {
            shipped_ms[k] = shipped.build(p, *d, *m, limit)?.compute_ms;
        }
        let worst_shipped = shipped_ms.iter().copied().fold(0.0, f64::max);
        if slow_limit_ms(ROUND_BUDGET_MS, &shipped_ms) < worst_shipped {
            return Err(format!("the shipped kernel takes {shipped_ms:?} ms at the timing shapes: a timing round of {ROUND_BUDGET_MS} ms cannot hold it and a kernel at its speed"));
        }
        let rungs: Vec<(usize, usize, Vec<f32>)> = RUNGS.iter().map(|&(d, m)| (d, m, phi(d, m, RUNG_SEED))).collect();
        let (train, val, extrap) = (pick(&sp.train, &train_idx)?, pick(&sp.val, &val_idx)?, pick(&sp.extrap, &extrap_idx)?);
        let up = |set: &[(usize, usize, &[f32])]| -> Result<Vec<Resident>, String> { set.iter().map(|(d, m, p)| Resident::upload(dev, p, *d, *m, limit)).collect() };
        let pairs_of = |set: &[AccuracyPair]| -> Result<Vec<Resident>, String> { up(&set.iter().map(|p| (p.1, p.2, p.3.as_slice())).collect::<Vec<_>>()) };
        let resident_cases = admission.iter().map(|&i| { let p = &probes[i]; Resident::upload(dev, &p.phi, p.d, p.m, p.limit) }).collect::<Result<Vec<_>, String>>()?;
        let resident_rungs = up(&rungs.iter().map(|(d, m, p)| (*d, *m, p.as_slice())).collect::<Vec<_>>())?;
        let resident_pairs = [pairs_of(&train)?, pairs_of(&val)?, pairs_of(&extrap)?];
        let resident_timing = up(&timing.iter().map(|(d, m, p)| (*d, *m, p.as_slice())).collect::<Vec<_>>())?;
        let ctx = FitnessContext { train, val, extrap, probes, admission, shipped_source: SHIPPED.to_string(), timing, limit, timing_rounds: 1, rungs, shipped_ms, base_budget_ms: BASE_BUDGET_MS, round_budget_ms: ROUND_BUDGET_MS, resident_rungs, resident_cases, resident_pairs, resident_timing, shipped_gpu_ms: std::cell::Cell::new([0.0; 3]), shipped_min_gpu_ms: std::cell::Cell::new([f64::INFINITY; 3]) };
        ctx.time_shipped(dev)?;
        Ok(ctx)
    }

    /// Time the shipped kernel at the three timing shapes (after one
    /// warm-up build at the first): the median of `timing_rounds` builds a
    /// shape, GPU timestamps as the candidate's are. The least time so far
    /// per shape ([`shipped_min`]) is the speed-ups' denominator. Returns
    /// this measurement.
    pub fn time_shipped(&self, dev: &Device) -> Result<[f64; 3], String> {
        let shipped = Gram::compile(dev, &self.shipped_source)?;
        shipped.run(&self.resident_timing[0], false)?;
        let mut out = [0.0f64; 3];
        for (k, res) in self.resident_timing.iter().enumerate() {
            let mut v = Vec::with_capacity(self.timing_rounds.max(1));
            for _ in 0..self.timing_rounds.max(1) {
                v.push(shipped.run(res, false)?.gpu_ms);
            }
            out[k] = median(&mut v);
            if out[k] <= 0.0 {
                return Err(format!("the shipped kernel timed at zero at shape {k}"));
            }
        }
        self.shipped_gpu_ms.set(out);
        self.shipped_min_gpu_ms.set(shipped_min(self.shipped_min_gpu_ms.get(), out));
        Ok(out)
    }

    /// The speed-ups' denominators: the least shipped time per shape so far.
    pub fn shipped_reference(&self) -> [f64; 3] {
        self.shipped_min_gpu_ms.get()
    }

    /// Re-time a kernel already scored feasible: only its timing builds
    /// rerun (`timing_rounds` a shape, the median), against the least
    /// shipped times; its accuracy columns (`columns[3..6]`) are kept. The
    /// new speed-ups, columns and HFF. A memoised elite must not keep a
    /// speed measured against a contended shipped time.
    pub fn retime(&self, dev: &Device, source: &str, columns: &[f64; 6]) -> Result<([f64; 3], [f64; 6], f64), String> {
        let gram = Gram::compile(dev, source)?;
        let reference = self.shipped_reference();
        let mut speedups = [0.0f64; 3];
        for (k, res) in self.resident_timing.iter().enumerate() {
            let mut v = Vec::with_capacity(self.timing_rounds.max(1));
            for _ in 0..self.timing_rounds.max(1) {
                v.push(speedup(reference[k], gram.run(res, false)?.gpu_ms)?);
            }
            speedups[k] = median(&mut v);
        }
        let c = fitness_columns(speedups, [columns[3], columns[4], columns[5]]);
        Ok((speedups, c, truenorth(&c)))
    }

    /// The limit of one candidate build at a timing shape, [`slow_limit_ms`].
    pub fn slow_limit_ms(&self) -> f64 {
        slow_limit_ms(self.round_budget_ms, &self.shipped_ms)
    }

    pub fn fitness(&self, dev: &Device, source: &str) -> Fitness {
        let spec = Spec {
            probes: &self.probes,
            admission: &self.admission,
            pairs: [&self.train, &self.val, &self.extrap],
            rungs: self.rungs.iter().map(|(d, m, _)| (*d, *m)).collect(),
            timing: self.timing.iter().map(|(d, m, _)| (*d, *m)).collect(),
            limit_ms: self.slow_limit_ms(),
            base_budget_ms: self.base_budget_ms,
            timing_rounds: self.timing_rounds,
            shipped_reference: self.shipped_reference(),
        };
        fitness_with(&spec, &DeviceDriver { dev, ctx: self }, source)
    }
}

/// The device [`Driver`]: [`Gram::compile`], [`Gram::run_batch`] over the
/// resident cases and [`Gram::run`] over the resident rungs, pairs and
/// timing shapes, exactly the calls the fitness made before it was
/// written over a driver.
#[cfg(feature = "gpu")]
struct DeviceDriver<'a> {
    dev: &'a Device,
    ctx: &'a FitnessContext,
}

#[cfg(feature = "gpu")]
impl<'a> Driver for DeviceDriver<'a> {
    type Kernel = Gram<'a>;

    fn compile(&self, source: &str) -> Result<Gram<'a>, String> {
        Gram::compile(self.dev, source)
    }

    fn batch(&self, k: &Gram<'a>, cases: std::ops::Range<usize>) -> Result<Batch, String> {
        let rs: Vec<&Resident> = self.ctx.resident_cases[cases].iter().collect();
        k.run_batch(&rs)
    }

    fn run(&self, k: &Gram<'a>, which: Which, read_h: bool) -> Result<Build, String> {
        let r = match which {
            Which::Rung(i) => &self.ctx.resident_rungs[i],
            Which::Pair(s, i) => &self.ctx.resident_pairs[s][i],
            Which::Timing(i) => &self.ctx.resident_timing[i],
        };
        k.run(r, read_h)
    }
}

/// Which resident Phi a single build runs over: a timing rung, a pair
/// (split 0 train, 1 validation, 2 extrapolation; its index), a timing shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Which {
    Rung(usize),
    Pair(usize, usize),
    Timing(usize),
}

/// What the fitness procedure runs a kernel with: the device driver in a
/// fit, a host model of a kernel in the CPU tests.
pub trait Driver {
    type Kernel;
    /// Compile; `Err` is the compile message.
    fn compile(&self, source: &str) -> Result<Self::Kernel, String>;
    /// The correctness cases at these positions of the admission order, in
    /// one submission ([`Gram::run_batch`]).
    fn batch(&self, k: &Self::Kernel, cases: std::ops::Range<usize>) -> Result<Batch, String>;
    /// One build ([`Gram::run`]); `read_h` false leaves `Build::h` empty.
    fn run(&self, k: &Self::Kernel, which: Which, read_h: bool) -> Result<Build, String>;
}

/// The data the fitness procedure reads, borrowed from a
/// [`FitnessContext`] (or built on the host for a test).
pub struct Spec<'a> {
    pub probes: &'a [crate::wgsl::probes::Probe],
    pub admission: &'a [usize],
    /// The accuracy pairs by split: train, validation, extrapolation.
    pub pairs: [&'a [AccuracyPair]; 3],
    /// The rungs' and the timing shapes' (d, m).
    pub rungs: Vec<(usize, usize)>,
    pub timing: Vec<(usize, usize)>,
    /// The limit of one build at a timing shape ([`slow_limit_ms`]).
    pub limit_ms: f64,
    pub base_budget_ms: f64,
    pub timing_rounds: usize,
    /// The speed-ups' denominators, the least shipped GPU ms per shape.
    pub shipped_reference: [f64; 3],
}

/// THE FAST FITNESS of one kernel over `drv` (`FitnessContext::fitness`
/// on the device): the constraint on the correctness cases, the rungs, the
/// accuracy pairs, the timing shapes, as documented at [`FitnessContext`].
pub fn fitness_with<D: Driver>(spec: &Spec, drv: &D, source: &str) -> Fitness {
    let t = Instant::now();
    let mut meter = Meter::default();
    let mut split = Split::default();
    let infeasible = |passed: usize, distance: f64, closeness: [f64; crate::wgsl::closeness::N], detail: String, t: Instant, meter: &Meter, split: &Split| Fitness { feasible: false, passed, distance, closeness, closeness_hff: crate::wgsl::closeness::angle(&closeness), columns: [1.0; 6], speedups: [0.0; 3], hff: f64::INFINITY, inputs: Vec::new(), detail, seconds: t.elapsed().as_secs_f64(), predicted_ms: [0.0; 3], rate_per_g: meter.rate_per_g(), floor_ms: meter.floor_ms, measured_ms: [0.0; 3], split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..*split } };
    let elapsed_ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let compiled = drv.compile(source);
    split.compile = elapsed_ms(t);
    let gram = match compiled {
        Ok(g) => g,
        Err(e) => return infeasible(0, f64::INFINITY, crate::wgsl::closeness::WORST, format!("compile: {}", e.lines().find(|l| l.contains("error") || l.contains("header")).unwrap_or(e.lines().next().unwrap_or(""))), t, &meter, &split),
    };
    // The correctness cases, run in chunks of `CASE_CHUNK`, one
    // submission a chunk (one submission a case cost 16 ms of
    // synchronisation; 2026-10-11). Each case's sample for the meter is
    // the chunk's wall time shared equally. They are the same size, so
    // the projection over them is linear in the count: a kernel whose
    // cases alone would spend a third of the base budget is stopped
    // between two chunks (one chunk cannot be cancelled, the rest can).
    let mut passed = 0;
    let mut first_fail = String::new();
    // The mean distance over the cases run (infinite before any ran).
    let (mut sum_distance, mut ran) = (0.0f64, 0usize);
    let mean_distance = |sum: f64, ran: usize| if ran == 0 { f64::INFINITY } else { sum / ran as f64 };
    // Every case's closeness columns and its output's and reference's
    // hashes, in the admission's order (None for a case never run).
    let cases = spec.admission.len();
    let mut per_case: Vec<Option<[f64; crate::wgsl::closeness::N]>> = vec![None; cases];
    let mut outs: Vec<Option<(u64, u64, usize)>> = vec![None; cases];
    // The projection is over the cases alone: the pipeline compile
    // (50-300 ms on Metal) before them would be multiplied by 128.
    let t_cases = Instant::now();
    let mut done = 0usize;
    for (c, idx) in spec.admission.chunks(CASE_CHUNK).enumerate() {
        let Batch { h: hs, compute_ms, .. } = match drv.batch(&gram, c * CASE_CHUNK..c * CASE_CHUNK + idx.len()) {
            Ok(x) => x,
            Err(e) => return infeasible(passed, mean_distance(sum_distance, ran), crate::wgsl::closeness::columns(&per_case, passed, &outs), format!("run: {e}"), t, &meter, &split),
        };
        for (&i, h) in idx.iter().zip(&hs) {
            let p = &spec.probes[i];
            meter.push(work(p.d, p.m), compute_ms / idx.len() as f64);
            let d = crate::wgsl::probes::distance(h, p);
            sum_distance += if d.is_nan() { f64::INFINITY } else { d };
            per_case[ran] = Some(crate::wgsl::closeness::case_columns(h, &p.h_ref, p.d));
            outs[ran] = Some((crate::wgsl::closeness::hash_bits(h, |v: f32| u64::from(v.to_bits())), crate::wgsl::closeness::hash_bits(&p.h_ref, f64::to_bits), p.d));
            ran += 1;
            if d <= crate::wgsl::probes::PASS {
                passed += 1;
            } else if first_fail.is_empty() {
                first_fail = format!("{} ({d:.2e})", p.name);
            }
        }
        done += idx.len();
        let projected = elapsed_ms(t_cases) * cases as f64 / done as f64;
        if done < cases && projected > spec.base_budget_ms / 3.0 {
            return infeasible(passed, mean_distance(sum_distance, ran), crate::wgsl::closeness::columns(&per_case, passed, &outs), format!("slow: the correctness cases are projected at {:.1} s ({:.1} s for the first {done}), over a third of the {:.1} s budget; the rest not dispatched", projected / 1e3, elapsed_ms(t_cases) / 1e3, spec.base_budget_ms / 1e3), t, &meter, &split);
        }
    }
    split.cases = elapsed_ms(t_cases);
    if passed < cases {
        return infeasible(passed, mean_distance(sum_distance, ran), crate::wgsl::closeness::columns(&per_case, passed, &outs), format!("constraint {passed}/{cases} cases: {first_fail}"), t, &meter, &split);
    }
    meter.fix_floor();
    let limit_ms = spec.limit_ms;
    // The rungs: timed, gated, not judged.
    let t_rungs = Instant::now();
    for (i, (d, m)) in spec.rungs.iter().enumerate() {
        if let Err(e) = gate(&meter, &format!("the timing rung d {d} m {m}"), work(*d, *m), elapsed_ms(t), limit_ms, spec.base_budget_ms) {
            return infeasible(passed, f64::INFINITY, crate::wgsl::closeness::columns(&per_case, passed, &outs), e, t, &meter, &split);
        }
        match drv.run(&gram, Which::Rung(i), false) {
            Ok(b) => {
                meter.push(work(*d, *m), b.compute_ms);
                split.build(&b);
            }
            Err(e) => return infeasible(passed, f64::INFINITY, crate::wgsl::closeness::columns(&per_case, passed, &outs), format!("run: {e}"), t, &meter, &split),
        }
    }
    split.rungs = elapsed_ms(t_rungs);
    // Accuracy columns: the worst error relative to shipped's, min(1, e / 2),
    // per set. The pairs run smallest work first across the three sets, so
    // every pair is predicted from a measured build at most 32x smaller.
    let mut order: Vec<(usize, usize, &AccuracyPair)> = spec.pairs.iter().enumerate().flat_map(|(k, set)| set.iter().enumerate().map(move |(i, p)| (k, i, p))).collect();
    order.sort_by(|a, b| work(a.2 .1, a.2 .2).total_cmp(&work(b.2 .1, b.2 .2)));
    let mut worst = [0.0f64; 3];
    for (k, i, (name, d, m, _, h_ref, smax)) in order {
        if let Err(e) = gate(&meter, &format!("pair {name} (d {d} m {m})"), work(*d, *m), elapsed_ms(t), limit_ms, spec.base_budget_ms) {
            return infeasible(passed, f64::INFINITY, crate::wgsl::closeness::columns(&per_case, passed, &outs), e, t, &meter, &split);
        }
        let t_pb = Instant::now();
        let b = match drv.run(&gram, Which::Pair(k, i), true) {
            Ok(b) => b,
            Err(e) => return infeasible(passed, f64::INFINITY, crate::wgsl::closeness::columns(&per_case, passed, &outs), format!("run: {e}"), t, &meter, &split),
        };
        split.pairs_build += elapsed_ms(t_pb);
        split.build(&b);
        meter.push(work(*d, *m), b.compute_ms);
        let t_pc = Instant::now();
        let (max_ratio, non_finite) = check_max(&b.h, h_ref, *d, *m);
        split.pairs_check += elapsed_ms(t_pc);
        if non_finite > 0 || max_ratio > (2.0 * smax).max(1.0) {
            let over = if non_finite > 0 { f64::INFINITY } else { max_ratio / (2.0 * smax).max(1.0) };
            // The cases' columns, each error column the mean of theirs and
            // the failing pair's own (a pair is wrong where the cases were not).
            let mut pair_closeness = crate::wgsl::closeness::columns(&per_case, passed, &outs);
            let pair = crate::wgsl::closeness::case_columns(&b.h, h_ref, *d);
            for (c, x) in pair_closeness.iter_mut().zip(pair).take(9).skip(1) {
                *c = (*c + x) / 2.0;
            }
            return infeasible(passed, over, pair_closeness, format!("pair {name}: max {max_ratio:.3} tau (shipped {smax:.3}), non-finite {non_finite}"), t, &meter, &split);
        }
        worst[k] = worst[k].max(if *smax > 0.0 { max_ratio / smax } else { max_ratio });
    }
    let [et, ev, ex] = worst;
    // The timing shapes predicted before any is dispatched; a kernel
    // predicted over the limit at any of them is stopped here.
    let mut predicted_ms = [0.0f64; 3];
    for (k, (d, m)) in spec.timing.iter().enumerate() {
        match gate(&meter, &format!("the timing shape d {d} m {m}"), work(*d, *m), 0.0, limit_ms, f64::INFINITY) {
            Ok(p) => predicted_ms[k] = p,
            Err(e) => return Fitness { predicted_ms, ..infeasible(passed, f64::INFINITY, crate::wgsl::closeness::columns(&per_case, passed, &outs), e, t, &meter, &split) },
        }
    }
    // Speed columns: `timing_rounds` candidate builds per shape, the
    // median of their ratios to the shipped kernel's times of this
    // generation ([`FitnessContext::time_shipped`]; until 2026-10-11 the
    // shipped kernel was built beside every candidate, 3.4 s of 6.4).
    let shipped_gpu = spec.shipped_reference;
    let mut speedups = [0.0f64; 3];
    let mut measured_ms = [0.0f64; 3];
    let rounds = spec.timing_rounds.max(1);
    for (k, (d, m)) in spec.timing.iter().enumerate() {
        let mut ratios = Vec::with_capacity(rounds);
        let mut walls = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let tb = Instant::now();
            let b = match drv.run(&gram, Which::Timing(k), false) {
                Ok(b) => b,
                Err(e) => return Fitness { predicted_ms, ..infeasible(passed, f64::INFINITY, crate::wgsl::closeness::columns(&per_case, passed, &outs), e, t, &meter, &split) },
            };
            split.timing_cand += elapsed_ms(tb);
            split.build(&b);
            match speedup(shipped_gpu[k], b.gpu_ms) {
                Ok(r) => ratios.push(r),
                Err(e) => return Fitness { predicted_ms, ..infeasible(passed, f64::INFINITY, crate::wgsl::closeness::columns(&per_case, passed, &outs), format!("{e} at d {d} m {m}"), t, &meter, &split) },
            }
            walls.push(b.compute_ms);
        }
        speedups[k] = median(&mut ratios);
        measured_ms[k] = median(&mut walls);
    }
    let columns = fitness_columns(speedups, [(et / 2.0).clamp(0.0, 1.0), (ev / 2.0).clamp(0.0, 1.0), (ex / 2.0).clamp(0.0, 1.0)]);
    Fitness { feasible: true, passed, distance: 0.0, closeness: [0.0; crate::wgsl::closeness::N], closeness_hff: 0.0, columns, speedups, hff: truenorth(&columns), inputs: Vec::new(), detail: String::new(), seconds: t.elapsed().as_secs_f64(), predicted_ms, rate_per_g: meter.rate_per_g(), floor_ms: meter.floor_ms, measured_ms, split: Split { total: t.elapsed().as_secs_f64() * 1e3, ..split } }
}

/// One pair to write into a problem directory: `(kind, d, m, seed)`.
pub type PairShape = (Kind, usize, usize, u64);

/// THE CASES OF `problems/gram` (the problem owner's reference, written as
/// data for the generic harness): the constraint's 128 admission probes in
/// admission order (their Phi, their per-binding limit, their f64 H), the
/// rungs and the timing shapes (Phi only, from [`RUNG_SEED`] and
/// [`TIMING_SEED`]), and the accuracy pairs of each split (Phi and the f64
/// reference, read from `cache` when given, as `pairs::Pair::load` does,
/// else computed). Returns the bytes written.
pub fn write_problem_cases(dir: &std::path::Path, rungs: &[(usize, usize)], pairs: &[Vec<PairShape>; 3], timing: &[(usize, usize)], cache: Option<&std::path::Path>) -> Result<u64, String> {
    use crate::wgsl::harness::{f32_bytes, f64_bytes, write_suite, CaseOut};
    let shape = |d: usize, m: usize| vec![("d".to_string(), d as f64), ("m".to_string(), m as f64)];
    let label = |d: usize, m: usize| Some(format!("d {d} m {m}"));
    let probes = crate::wgsl::probes::probes(1);
    let constraint: Vec<CaseOut> = crate::wgsl::probes::admission(&probes, 1)
        .into_iter()
        .map(|i| {
            let p = &probes[i];
            let mut params = shape(p.d, p.m);
            params.push(("limit".into(), p.limit as f64));
            CaseOut { name: p.name.clone(), label: None, params, inputs: vec![("phi".into(), f32_bytes(&p.phi))], expected: vec![("h".into(), vec![f64_bytes(&p.h_ref)])] }
        })
        .collect();
    let mut total = write_suite(dir, "constraint", &constraint)?;
    let phi_only = |set: &[(usize, usize)], seed: u64| -> Vec<CaseOut> { set.iter().map(|&(d, m)| CaseOut { name: format!("d {d} m {m}"), label: label(d, m), params: shape(d, m), inputs: vec![("phi".into(), f32_bytes(&phi(d, m, seed)))], expected: Vec::new() }).collect() };
    total += write_suite(dir, "rungs", &phi_only(rungs, RUNG_SEED))?;
    for (suite, set) in ["train", "val", "extrap"].iter().zip(pairs) {
        let mut cases = Vec::new();
        for &(kind, d, m, seed) in set {
            let (name, p, h) = match cache {
                Some(c) => {
                    let pair = crate::wgsl::pairs::Pair::load(kind, d, m, seed, c)?;
                    (pair.name(), pair.phi, pair.h_ref)
                }
                None => {
                    let p = phi_of(kind, d, m, seed);
                    let h = reference(&p, d, m);
                    (format!("{}_d{d}_m{m}_s{seed}", kind.name()), p, h)
                }
            };
            cases.push(CaseOut { name, label: label(d, m), params: shape(d, m), inputs: vec![("phi".into(), f32_bytes(&p))], expected: vec![("h".into(), vec![f64_bytes(&h)])] });
        }
        total += write_suite(dir, suite, &cases)?;
    }
    total += write_suite(dir, "timing", &phi_only(timing, TIMING_SEED))?;
    Ok(total)
}

/// The pairs the fast fitness scores ([`crate::wgsl::pairs::fitness_pick`]), by split.
pub fn fitness_pairs() -> Result<[Vec<PairShape>; 3], String> {
    let shapes = crate::wgsl::pairs::split_shapes();
    let pick = crate::wgsl::pairs::fitness_pick(&shapes)?;
    Ok(std::array::from_fn(|k| pick[k].iter().map(|&i| shapes[k][i]).collect()))
}

/// The six columns: 1 / speed-up per timing shape, clamped to [0, 1], then
/// the three accuracy columns as given.
pub fn fitness_columns(speedups: [f64; 3], accuracy: [f64; 3]) -> [f64; 6] {
    [(1.0 / speedups[0]).clamp(0.0, 1.0), (1.0 / speedups[1]).clamp(0.0, 1.0), (1.0 / speedups[2]).clamp(0.0, 1.0), accuracy[0], accuracy[1], accuracy[2]]
}

/// The running least shipped time per shape after a new measurement.
pub fn shipped_min(so_far: [f64; 3], new: [f64; 3]) -> [f64; 3] {
    [so_far[0].min(new[0]), so_far[1].min(new[1]), so_far[2].min(new[2])]
}

/// Whether a shipped measurement is contended: any shape more than 10 %
/// above the least so far.
pub fn shipped_contended(latest: [f64; 3], least: [f64; 3]) -> bool {
    latest.iter().zip(&least).any(|(l, m)| *l > 1.1 * m)
}

/// A speed-up: the shipped kernel's GPU time over the candidate's, both
/// positive (a zero is a timestamp that did not resolve, not a speed).
pub fn speedup(shipped_gpu_ms: f64, candidate_gpu_ms: f64) -> Result<f64, String> {
    if candidate_gpu_ms <= 0.0 || shipped_gpu_ms <= 0.0 {
        return Err(format!("zero time (shipped {shipped_gpu_ms} ms, candidate {candidate_gpu_ms} ms)"));
    }
    Ok(shipped_gpu_ms / candidate_gpu_ms)
}

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

    /// The rate fit: exact on a noiseless linear kernel, the floor the
    /// median of the correctness cases.
    #[test]
    fn the_meter_fits_a_linear_kernel_exactly() {
        let mut m = Meter::default();
        for (k, t) in [2.0, 1.0, 3.0, 2.0, 2.0].into_iter().enumerate() {
            m.push(1e4 + k as f64, t);
        }
        m.fix_floor();
        assert_eq!(m.floor_ms, 2.0);
        // 4 ms per 1e9 item-cells above the floor, at 1e6, 3.4e7 and 5.4e8.
        let mut lin = Meter { samples: Vec::new(), floor_ms: 2.0 };
        for w in [1e6, 3.4e7, 5.4e8] {
            lin.push(w, 2.0 + 4e-9 * w);
        }
        assert!((lin.rate_per_g() - 4.0).abs() < 1e-9, "{}", lin.rate_per_g());
        assert!((lin.predict(2.75e11) - (2.0 + 1100.0)).abs() < 1e-6);
    }

    /// The noise of the correctness cases does not predict seconds one rung
    /// ahead, and a slope below the floor clamps at zero.
    #[test]
    fn noise_at_the_floor_predicts_milliseconds_not_seconds() {
        let mut m = Meter::default();
        for k in 0..128 {
            // 1-3 ms of submission cost at up to 8e4 item-cells: all noise.
            m.push(8e4 * (k % 5 + 1) as f64 / 5.0, 1.0 + (k % 3) as f64);
        }
        m.fix_floor();
        assert!(m.predict(work(RUNGS[0].0, RUNGS[0].1)) < 50.0, "{}", m.predict(work(128, 64)));
        let mut neg = Meter { samples: vec![(1e6, 0.5), (3.4e7, 0.4)], floor_ms: 2.0 };
        assert_eq!(neg.rate(), 0.0);
        assert_eq!(neg.predict(2.75e11), 2.0);
        neg.push(5.4e8, 2.0 + 1e-8 * 5.4e8);
        assert!(neg.rate() > 0.0);
    }

    /// Every gated step is at most 32x the work of the build before it.
    #[test]
    fn the_rungs_bridge_the_cases_to_the_pairs_in_steps_of_at_most_32() {
        let largest_case = work(40, 50);
        let smallest_pair = work(512, 2048);
        let mut ladder = vec![largest_case];
        ladder.extend(RUNGS.iter().map(|&(d, m)| work(d, m)));
        ladder.push(smallest_pair);
        for w in ladder.windows(2) {
            assert!(w[1] / w[0] <= 32.0 && w[1] > w[0], "{ladder:?}");
        }
    }

    /// The limit and the gate: a third of what the shipped kernel leaves of a
    /// round; a prediction over it, or past the budget, is "slow" and names
    /// the build and the prediction.
    #[test]
    fn the_gate_stops_a_build_predicted_over_the_limit_or_the_budget() {
        let limit = slow_limit_ms(ROUND_BUDGET_MS, &[1000.0, 1100.0, 900.0]);
        assert!((limit - 17_000.0 / 3.0).abs() < 1e-9);
        // The shipped kernel's rate (about 1 s at 2.75e11 item-cells) passes.
        let shipped = Meter { samples: vec![(6.9e10, 2.0 + 250.0)], floor_ms: 2.0 };
        let p = gate(&shipped, "the timing shape d 2048 m 65536", work(2048, 65536), 0.0, limit, f64::INFINITY).expect("shipped passes");
        assert!((p - (2.0 + 250.0 * work(2048, 65536) / 6.9e10)).abs() < 1e-6);
        // Ten times slower is predicted at about 10 s: stopped, named.
        let slow = Meter { samples: vec![(6.9e10, 2.0 + 2500.0)], floor_ms: 2.0 };
        let e = gate(&slow, "the timing shape d 2048 m 65536", work(2048, 65536), 0.0, limit, f64::INFINITY).unwrap_err();
        assert!(e.starts_with("slow: predicted 10.0 s at the timing shape d 2048 m 65536") && e.contains("over the limit of 5.7 s") && e.contains("not dispatched"), "{e}");
        // Under the limit but past the evaluation's budget.
        let e = gate(&shipped, "pair x", 6.9e10, 29_900.0, limit, BASE_BUDGET_MS).unwrap_err();
        assert!(e.contains("over the evaluation's budget of 30.0 s"), "{e}");
    }

    /// GPU (run when the device is free): the shipped kernel's fitness
    /// predicts its own measured time at the three timing shapes within a
    /// factor of two. The three shapes have the same work (2.75e11
    /// item-cells), so the model predicts one time for all three; the
    /// factor of two covers the spread of the shipped kernel's own times
    /// across the shapes (its memory access differs by d) and the run-to-
    /// run scatter of one round (15 %, 2026-10-11).
    #[cfg(feature = "gpu")]
    #[test]
    fn the_shipped_kernel_predicts_its_own_time_at_the_timing_shapes_on_the_device() {
        let dev = Device::new().expect("device");
        let ctx = FitnessContext::new(&dev, &crate::wgsl::pairs::cache_dir()).expect("context");
        let f = ctx.fitness(&dev, SHIPPED);
        assert!(f.feasible, "{f:?}");
        println!("shipped measured {:?} ms, predicted {:?} ms, rate {:.2} ms per 1e9 item-cells, floor {:.2} ms, limit {:.0} ms", ctx.shipped_ms, f.predicted_ms, f.rate_per_g, f.floor_ms, ctx.slow_limit_ms());
        println!("shipped as the candidate, measured {:?} ms", f.measured_ms);
        for k in 0..3 {
            let ratio = f.predicted_ms[k] / ctx.shipped_ms[k];
            assert!((0.5..=2.0).contains(&ratio), "shape {k}: predicted {} ms, measured {} ms", f.predicted_ms[k], ctx.shipped_ms[k]);
            let own = f.predicted_ms[k] / f.measured_ms[k];
            assert!((0.5..=2.0).contains(&own), "shape {k}: predicted {} ms, measured as the candidate {} ms", f.predicted_ms[k], f.measured_ms[k]);
        }
    }

    /// The shipped kernel doing 64 times its work with the same H (see
    /// `a_deliberately_slow_kernel_is_stopped_before_the_timing_shapes`).
    fn slow_shipped() -> String {
        let item_loop = "    var acc: f32 = 0.0;\n    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {\n        let base = t * cfg.d;\n        acc = acc + phi[base + gr] * phi[base + c];\n    }\n";
        assert!(SHIPPED.contains(item_loop), "the item loop was not found in the shipped text");
        SHIPPED.replace(item_loop, "    var acc: f32 = 0.0;\n    for (var k: u32 = 0u; k < 64u; k = k + 1u) {\n        var s: f32 = 0.0;\n        let off = (k * cfg.d) / (64u * cfg.d);\n        for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {\n            let base = (t + off) * cfg.d;\n            s = s + phi[base + gr] * phi[base + c];\n        }\n        if (k == 0u) { acc = s; } else { acc = 0.5 * (acc + s); }\n    }\n")
    }

    /// The slow test kernel is valid WGSL (naga, no device).
    #[test]
    fn the_slow_test_kernel_is_valid_wgsl() {
        let slow = slow_shipped();
        let module = crate::wgsl::naga_shape::parse(&slow).unwrap_or_else(|e| panic!("{e}"));
        crate::wgsl::target::Target::current().validate(&module, &slow).unwrap_or_else(|e| panic!("{e}"));
        assert!(slow.contains("k < 64u") && slow.contains("(k * cfg.d) / (64u * cfg.d)"));
    }

    /// GPU (run when the device is free): a correct kernel doing 64 times
    /// the shipped kernel's work is stopped as "slow" before the timing
    /// shapes. Its item loop runs 64 times, each pass's sum `s` the shipped
    /// kernel's own sum in the same order; the passes are folded as
    /// `acc = 0.5 * (acc + s)`, exact for equal values (2s and the halving
    /// are exact, and inf and NaN carry through), so its H is bit-identical
    /// to the shipped kernel's and it passes the correctness cases and the
    /// pairs it reaches. Each pass reads at an offset `(k d) / (64 d)`, zero
    /// for every k < 64 but not provably so to the shader compiler, so no
    /// pass can be hoisted out of the k loop as invariant. A "feasible"
    /// result here means the compiler removed the repeated work anyway and
    /// the test kernel must be made slower another way, not that the gate
    /// failed. The detail names the shape and the prediction, and the
    /// evaluation takes well under the watchdog's base budget.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_deliberately_slow_kernel_is_stopped_before_the_timing_shapes_on_the_device() {
        let slow = slow_shipped();
        let dev = Device::new().expect("device");
        let ctx = FitnessContext::new(&dev, &crate::wgsl::pairs::cache_dir()).expect("context");
        let f = ctx.fitness(&dev, &slow);
        println!("{:.1} s: {}", f.seconds, f.detail);
        assert!(!f.feasible && f.passed == ctx.admission.len(), "{f:?}");
        assert!(f.detail.starts_with("slow: predicted") && f.detail.contains("not dispatched"), "{}", f.detail);
        assert!(f.detail.contains("the timing shape d") || f.detail.contains("pair "), "{}", f.detail);
        assert!(f.seconds * 1e3 < BASE_BUDGET_MS, "{f:?}");
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

    /// The speed-up is the generation's shipped time over the candidate's;
    /// a zero on either side is an error, never a speed.
    #[test]
    fn the_speedup_is_shipped_over_candidate_and_refuses_a_zero() {
        assert_eq!(speedup(1000.0, 125.0), Ok(8.0));
        assert_eq!(speedup(1000.0, 2000.0), Ok(0.5));
        assert!(speedup(1000.0, 0.0).unwrap_err().starts_with("zero time"));
        assert!(speedup(0.0, 10.0).is_err());
    }

    /// A contended (inflated) shipped measurement never raises a speed-up:
    /// the denominator is the least time so far, and the inflated
    /// generation is flagged.
    #[test]
    fn an_inflated_shipped_time_does_not_raise_a_speedup() {
        let first = [1010.0, 1005.0, 1012.0];
        let mut least = shipped_min([f64::INFINITY; 3], first);
        assert_eq!(least, first);
        let candidate = [100.0, 95.0, 101.0];
        let before: Vec<f64> = (0..3).map(|k| speedup(least[k], candidate[k]).unwrap()).collect();
        let inflated = [1787.0, 1500.0, 1013.0];
        assert!(shipped_contended(inflated, least));
        least = shipped_min(least, inflated);
        assert_eq!(least, first);
        let after: Vec<f64> = (0..3).map(|k| speedup(least[k], candidate[k]).unwrap()).collect();
        assert_eq!(before, after);
        // A faster, uncontended measurement lowers the reference and is not flagged.
        let quicker = [1000.0, 1005.0, 1012.0];
        assert!(!shipped_contended(quicker, least));
        assert_eq!(shipped_min(least, quicker), [1000.0, 1005.0, 1012.0]);
    }

    /// An elite's HFF follows a re-time: the accuracy columns kept, the
    /// speed columns from the new speed-ups.
    #[test]
    fn an_elites_hff_follows_a_retime() {
        let old = fitness_columns([17.0, 17.0, 17.0], [0.25, 0.3, 0.28]);
        let new = fitness_columns([10.7, 10.7, 10.7], [old[3], old[4], old[5]]);
        assert_eq!(&new[3..], &old[3..]);
        assert!(truenorth(&new) > truenorth(&old));
        assert_eq!(new[0], 1.0 / 10.7);
    }

    /// GPU (run when the device is free): with the shipped times taken once,
    /// the shipped kernel scored as a candidate has speed-ups of 1 within
    /// one-round noise (15 %, 2026-10-11), and a second timing of the
    /// shipped kernel moves its times by less than that.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_generations_shipped_times_score_shipped_at_one_on_the_device() {
        let dev = Device::new().expect("device");
        let ctx = FitnessContext::new(&dev, &crate::wgsl::pairs::cache_dir()).expect("context");
        let first = ctx.shipped_gpu_ms.get();
        let f = ctx.fitness(&dev, SHIPPED);
        assert!(f.feasible, "{f:?}");
        let again = ctx.time_shipped(&dev).expect("time shipped");
        println!("shipped {first:?} then {again:?} ms; shipped as a candidate: speed-ups {:?}", f.speedups);
        for k in 0..3 {
            assert!((0.85..=1.15).contains(&f.speedups[k]), "{:?}", f.speedups);
            assert!((again[k] / first[k] - 1.0).abs() < 0.15, "{first:?} {again:?}");
        }
    }

    /// GPU (run when the device is free): the 128 correctness cases run in
    /// chunks give each case the H a one-case run gives, bit for bit, for
    /// the shipped, tiled and blocked kernels.
    #[cfg(feature = "gpu")]
    #[test]
    fn batched_cases_give_the_one_case_h_bit_for_bit_on_the_device() {
        let dev = Device::new().expect("device");
        let probes = crate::wgsl::probes::probes(1);
        let admission = crate::wgsl::probes::admission(&probes, 1);
        let res: Vec<Resident> = admission.iter().map(|&i| { let p = &probes[i]; Resident::upload(&dev, &p.phi, p.d, p.m, p.limit).expect("upload") }).collect();
        for (name, text) in [("shipped", SHIPPED), ("tiled", TILED), ("blocked", BLOCKED)] {
            let g = Gram::compile(&dev, text).expect("compile");
            for chunk in res.chunks(CASE_CHUNK) {
                let rs: Vec<&Resident> = chunk.iter().collect();
                let Batch { h: hs, gpu_ms: gpu, compute_ms: ms } = g.run_batch(&rs).expect("batch");
                assert!(ms > 0.0 && gpu.iter().all(|x| *x > 0.0), "{name}: {gpu:?}");
                for (r, h) in rs.iter().zip(&hs) {
                    let one = g.run(r, true).expect("run").h;
                    assert_eq!(h.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), one.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{name} d {} m {}", r.d, r.m);
                }
            }
        }
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

    /// The shipped kernel as a typed tree, rendered, on the device: the same
    /// H within the criterion as the shipped text.
    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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

    /// THE SAFE KINGDOM ON THE DEVICE: the 1,000 seeded sampler trees of
    /// the reader's completeness test, rendered, passed by the checker and
    /// compiled by Metal, every failure named. naga can accept a module
    /// whose Metal text Metal refuses to compile (a unary minus on a matrix,
    /// fit_SF_random 2026-10-12), so validation alone is not the gate.
    #[cfg(feature = "gpu")]
    #[test]
    fn random_safe_kernels_compile_on_the_device() {
        use crate::wgsl::render::{dual_ty, gram_role, sample};
        use crate::wgsl::{check::Checker, loader::WgslKingdom, safe_karva::render_tree_with, target::Target};
        let (k, r) = (WgslKingdom::load_safe(), gram_role());
        let checker = Checker::new(&k, &r, &Target::recorded());
        let root = dual_ty(&k, "kernel.safe").unwrap();
        let dev = Device::new().expect("device");
        let n = 1000u64;
        let mut compiled = 0usize;
        let mut failed: Vec<String> = Vec::new();
        for seed in 0..n {
            let tree = sample(&k, &r, root, 7, seed).unwrap();
            let verdict = render_tree_with(&tree, &k, &crate::wgsl::render::gram_role()).and_then(|out| {
                checker.check(&tree).map_err(|e| format!("check: {} {}", e.layer.number(), e.reason))?;
                Gram::compile_rendered(&dev, &out).map(|_| ()).map_err(|e| format!("compile: {}", e.lines().next().unwrap_or("")))
            });
            match verdict {
                Ok(()) => compiled += 1,
                Err(e) => failed.push(format!("seed {seed} {e}")),
            }
        }
        println!("{n} random safe kernels: {compiled} compiled on the device");
        assert!(failed.is_empty(), "{} failures:\n{}", failed.len(), failed.join("\n"));
    }

    /// Every build reports a compute time, build after build, shape after
    /// shape; the timestamp sum is printed beside it (it reads 0 for short
    /// passes on Apple silicon, which is why the search uses compute_ms).
    #[cfg(feature = "gpu")]
    #[test]
    fn every_build_in_a_sequence_has_a_compute_time_on_the_device() {
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
        let gram = Gram::compile(&dev, &spin).expect("the spinning kernel compiles");
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
        let shipped = Gram::compile(&dev, SHIPPED).expect("compile");
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
        assert_eq!(Gram::compile(&dev, TILED).unwrap().grid, Grid { x: 16, y: 16, bx: 4, by: 4, rows_x: false });
        let wrong = TILED.replace("// grid 16 16 4 4", "// grid 8 8 4 4");
        let err = match Gram::compile(&dev, &wrong) { Ok(_) => panic!("a mismatched header compiled"), Err(e) => e };
        assert!(err.contains("the grid header says workgroup 8 x 8"), "{err}");
    }

    /// The hand-written kernels (plan step 0) agree with the reference
    /// under the criterion, banded and unbanded, at ragged shapes.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_tiled_and_blocked_kernels_pass_the_criterion_on_the_device() {
        let dev = Device::new().expect("device");
        for (name, text) in [("tiled", TILED), ("blocked", BLOCKED)] {
            let gram = Gram::compile(&dev, text).expect("compile");
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

    /// The fast fitness: the shipped kernel is feasible at θ near
    /// acos(1 − (3·1 + 3·0.25)/6) = 1.186; the blocked kernel is feasible
    /// with a smaller θ; a kernel with its product broken is infeasible
    /// with the failing case named.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_fast_fitness_ranks_blocked_above_shipped_and_refuses_a_broken_kernel_on_the_device() {
        let dev = Device::new().expect("device");
        let ctx = FitnessContext::new(&dev, &crate::wgsl::pairs::cache_dir()).expect("context");
        let s = ctx.fitness(&dev, SHIPPED);
        assert!(s.feasible, "{s:?}");
        assert!((s.hff - 1.186).abs() < 0.08, "{s:?}");
        let b = ctx.fitness(&dev, BLOCKED);
        assert!(b.feasible && b.hff < s.hff && b.speedups.iter().all(|x| *x > 2.0), "{b:?}");
        let broken = SHIPPED.replace("phi[base + gr] * phi[base + c]", "phi[base + gr]");
        let w = ctx.fitness(&dev, &broken);
        assert!(!w.feasible && w.detail.starts_with("constraint"), "{w:?}");
        println!("shipped {:.3} in {:.1} s; blocked {:.3} in {:.1} s; broken in {:.1} s", s.hff, s.seconds, b.hff, b.seconds, w.seconds);
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

/// THE CERTIFICATION OF THE LIBRARY'S TRUENORTH against the hand-written
/// angle it replaced (`acos(1 − mean(x²))` over columns clamped into [0, 1],
/// snapped to 0 for a cosine above 1 − `f32::EPSILON`). A fitness change is
/// certified by identical selection: the two angles must rank every in-range
/// vector the same, and every value that moved by more than 1e-12 must be one
/// of the cases [`explain`] names from the code.
#[cfg(test)]
pub(crate) mod truenorth_certification {
    /// splitmix64: a fixed stream of numbers for the random vectors.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(state: &mut u64) -> f64 {
        (next(state) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The vectors of width `m`: 2,000 uniform in [0, 1]^m, 400 near the pole
    /// (uniform times 10^-k, k = 1..=20), and the edge cases: all 0, all 1,
    /// one column 1, all at 1e-4, 1e-8, `f64::EPSILON` and 1e-200, one column
    /// NaN, +inf, -inf, 2 and -0.5.
    pub(crate) fn vectors(m: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut out: Vec<Vec<f64>> = (0..2000).map(|_| (0..m).map(|_| unit(&mut s)).collect()).collect();
        for k in 1..=20 {
            let scale = 10f64.powi(-k);
            for _ in 0..20 {
                out.push((0..m).map(|_| unit(&mut s) * scale).collect());
            }
        }
        out.push(vec![0.0; m]);
        out.push(vec![1.0; m]);
        for i in 0..m {
            let mut v = vec![0.0; m];
            v[i] = 1.0;
            out.push(v);
        }
        for x in [1e-4, 1e-8, f64::EPSILON, 1e-200] {
            out.push(vec![x; m]);
            let mut v = vec![0.0; m];
            v[0] = x;
            out.push(v);
        }
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0, -0.5] {
            let mut v = vec![0.25; m];
            v[0] = x;
            out.push(v);
        }
        out
    }

    /// Why the two angles of one vector differ, from the code, or `None` when
    /// nothing explains it.
    pub(crate) fn explain(columns: &[f64], hand: f64, library: f64) -> Option<&'static str> {
        if columns.iter().any(|x| x.is_nan()) && hand == std::f64::consts::FRAC_PI_2 && library == std::f64::consts::PI {
            return Some("a NaN column: the hand angle's clamp passed NaN through and NaN.min(1.0) is 1.0, so cos 0 and π/2; the library returns π for any non-finite column");
        }
        let mean = columns.iter().map(|x| x.clamp(0.0, 1.0).powi(2)).sum::<f64>() / columns.len().max(1) as f64;
        if hand == 0.0 && library > 0.0 && mean < 1.1920929e-7 {
            return Some("the pole band: the hand angle snapped to 0 for a cosine above 1 - f32::EPSILON (mean x² below 1.19e-7); the library snaps only at f64::EPSILON and returns acos(1 - mean x²)");
        }
        None
    }

    pub(crate) struct Report {
        pub largest_in_range: f64,
        pub in_range_ranking_identical: bool,
    }

    /// Compare `hand` and `library` on `vectors`; `angle_columns` picks the
    /// columns that enter the angle. Prints every vector whose values differ
    /// by more than 1e-12 with its explanation, panics on one nothing
    /// explains, and checks the ranking of the in-range vectors (every
    /// angle column finite and in [0, 1]): no pair the hand angle ordered
    /// strictly is reversed or tied by the library. Pairs the hand angle tied
    /// and the library orders are counted; each is a vector
    /// already explained as the pole band. Both are printed.
    pub(crate) fn certify(
        name: &str,
        vectors: &[Vec<f64>],
        hand: impl Fn(&[f64]) -> f64,
        library: impl Fn(&[f64]) -> f64,
        angle_columns: impl Fn(&[f64]) -> Vec<f64>,
    ) -> Report {
        let mut largest = 0.0f64;
        let mut largest_in_range = 0.0f64;
        let mut differing = 0usize;
        let mut in_range: Vec<(f64, f64)> = Vec::new();
        for v in vectors {
            let (h, l) = (hand(v), library(v));
            let cols = angle_columns(v);
            let ok = cols.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x));
            let d = (h - l).abs();
            assert!(!d.is_nan(), "{name}: {v:?} hand {h} library {l}");
            largest = largest.max(d);
            if ok {
                largest_in_range = largest_in_range.max(d);
                in_range.push((h, l));
            }
            if d > 1e-12 {
                differing += 1;
                let why = explain(&cols, h, l).unwrap_or_else(|| panic!("{name}: our software has a bug: {cols:?} hand {h} library {l}, a difference nothing in the code explains"));
                eprintln!("{name}: m {} {cols:?}: hand {h:e} library {l:e} (|d| {d:e}): {why}", cols.len());
            }
        }
        // Sorted by the hand angle, then the library's: along that order the
        // library must never fall, and must rise wherever the hand angle does.
        in_range.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut identical = true;
        let mut ties_broken = 0usize;
        for w in in_range.windows(2) {
            let ((h0, l0), (h1, l1)) = (w[0], w[1]);
            if h0 < h1 && l0 >= l1 {
                identical = false;
                eprintln!("{name}: ORDER CHANGED: hand {h0:e} < {h1:e} but library {l0:e} >= {l1:e}");
            }
            if h0 == h1 && l0 != l1 {
                ties_broken += 1;
            }
        }
        eprintln!(
            "{name}: {} vectors ({} in range): largest |hand - library| {largest:e} (in range {largest_in_range:e}), {differing} differ by more than 1e-12, in-range ranking identical: {identical}, adjacent hand ties the library orders: {ties_broken}",
            vectors.len(),
            in_range.len()
        );
        Report { largest_in_range, in_range_ranking_identical: identical }
    }

    /// `gram::truenorth` before it called the hff library.
    fn hand_truenorth(x: &[f64]) -> f64 {
        let energy = x.iter().map(|v| { let c = v.clamp(0.0, 1.0); c * c }).sum::<f64>() / x.len().max(1) as f64;
        let cos = (1.0 - energy.min(1.0)).clamp(-1.0, 1.0);
        if cos > 1.0 - 1.1920929e-7 { 0.0 } else { cos.acos() }
    }

    /// CERTIFIED BY IDENTICAL SELECTION, m = 1..=12: `gram::truenorth` (the
    /// hff library) ranks every in-range vector as the hand angle did; the
    /// values that moved are the explained NaN and pole-band cases.
    #[test]
    fn truenorth_from_the_hff_library_selects_as_the_hand_angle_did() {
        for m in 1..=12 {
            let report = certify(&format!("gram::truenorth m {m}"), &vectors(m, 0x7e57 + m as u64), hand_truenorth, super::truenorth, |v| v.to_vec());
            assert!(report.in_range_ranking_identical, "m {m}");
            assert!(report.largest_in_range < 1e-3, "m {m}: {}", report.largest_in_range);
        }
    }
}
