//! Probes: a thousand independent behavioural measures of distance to the
//! correct Gram kernel (`docs/PLAN_wgsl_gram_superopt.md` §1b, Andrew
//! 2026-10-09: "a HFF that has 1000 measures of edit distance to the
//! correct kernel where each measure is independent"). A probe is a tiny
//! Φ whose correct H is known and which exercises one combination of what
//! the kernel must do: reach an item, read a column, multiply, accumulate,
//! store at a cell, offset a band, carry a magnitude or a non-finite. A
//! kernel can pass any probe while failing any other, which is the
//! independence; the vector of a kernel's probe distances is what the HFF
//! angle is taken over.

use crate::wgsl::gram::reference;

#[derive(Debug, Clone)]
pub struct Probe {
    pub name: String,
    pub family: char,
    pub d: usize,
    pub m: usize,
    /// The per-binding limit handed to the driver: small values force
    /// item bands and output-row bands on a tiny Φ.
    pub limit: u64,
    pub phi: Vec<f32>,
    pub h_ref: Vec<f64>,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// The probe set for `seed`: the same families and counts for every seed,
/// different items, columns and values, so a second seed is a validation
/// set the search never selected on.
pub fn probes(seed: u64) -> Vec<Probe> {
    let mut state = seed ^ 0x7A3F_1D2E_5B4C_9E81;
    let mut out = Vec::with_capacity(1024);
    let (d, m) = (32usize, 16usize);
    let values: [f32; 5] = [1.0, -1.0, 2.5, 0.125, 3.0];
    let columns: Vec<usize> = (0..16).map(|k| (k * 2 + (splitmix64(&mut state) % 2) as usize).min(d - 1)).collect();
    let items = [0usize, 1, 2, m / 2, m - 1];
    let mut push = |family: char, name: String, d: usize, m: usize, limit: u64, cells: Vec<(usize, usize, f32)>| {
        let mut phi = vec![0.0f32; d * m];
        for (t, j, v) in cells {
            phi[t * d + j] = v;
        }
        let h_ref = reference(&phi, d, m);
        out.push(Probe { name, family, d, m, limit, phi, h_ref });
    };
    // A: one non-zero cell.
    for &t in &items {
        for &j in &columns {
            for &v in &values {
                push('A', format!("A t{t} c{j} v{v}"), d, m, u64::MAX, vec![(t, j, v)]);
            }
        }
    }
    // B: two non-zeros in one item.
    for n in 0..200 {
        let t = items[n % items.len()];
        let j = columns[(splitmix64(&mut state) % 16) as usize];
        let mut k = columns[(splitmix64(&mut state) % 16) as usize];
        if k == j {
            k = (j + 1) % d;
        }
        let (a, b) = (values[n % 5], values[(n / 5) % 5]);
        push('B', format!("B t{t} c{j},{k} v{a},{b}"), d, m, u64::MAX, vec![(t, j, a), (t, k, b)]);
    }
    // C: two items, one column.
    for n in 0..200 {
        let t1 = items[n % items.len()];
        let mut t2 = items[(n / 5) % items.len()];
        if t2 == t1 {
            t2 = (t1 + 3) % m;
        }
        let j = columns[(splitmix64(&mut state) % 16) as usize];
        let (a, b) = (values[(n / 25) % 5], values[(n / 7) % 5]);
        push('C', format!("C t{t1},{t2} c{j} v{a},{b}"), d, m, u64::MAX, vec![(t1, j, a), (t2, j, b)]);
    }
    // D: small random Φ at ragged shapes, bands forced.
    for n in 0..100 {
        let dd = [8usize, 16, 24, 33, 40][n % 5];
        let mm = [3usize, 7, 16, 31, 50][(n / 5) % 5];
        // The limit forces 1–4 item bands and 1–2 row bands.
        let rows_per_band = [mm, mm.div_ceil(2), mm.div_ceil(4), dd.div_ceil(2).max(1)][(n / 25) % 4];
        let limit = (rows_per_band.max(1) * dd * 4) as u64;
        let mut cells = Vec::with_capacity(dd * mm);
        for t in 0..mm {
            for j in 0..dd {
                cells.push((t, j, (unit(&mut state) * 2.0 - 1.0) as f32));
            }
        }
        push('D', format!("D d{dd} m{mm} bands{}", rows_per_band), dd, mm, limit, cells);
    }
    // E: extremes and non-finites, as A and C.
    let extremes: [(f32, &str); 6] = [(f32::NAN, "nan"), (f32::INFINITY, "inf"), (f32::NEG_INFINITY, "-inf"), (1e-20, "1e-20"), (1e15, "1e15"), (-7.0, "-7")];
    for n in 0..124 {
        let (v, label) = extremes[n % 6];
        let t = items[(n / 6) % 5];
        let j = columns[(n / 30) % 16];
        if n % 2 == 0 {
            push('E', format!("E t{t} c{j} v{label}"), d, m, u64::MAX, vec![(t, j, v)]);
        } else {
            let t2 = (t + 5) % m;
            push('E', format!("E t{t},{t2} c{j} v{label},1"), d, m, u64::MAX, vec![(t, j, v), (t2, j, 1.0)]);
        }
    }
    out
}

/// The probe's distance: over its d × d cells, the error of each cell
/// against the reference, scaled by the reference's largest finite
/// magnitude (or 1), with a non-finite cell counted 0 when the classes
/// agree and 1 when they do not; the mean over cells, so a probe is in
/// [0, ∞) and 0 exactly when the kernel wrote the right table.
pub fn distance(h: &[f32], p: &Probe) -> f64 {
    distance_to(h, &p.h_ref)
}

/// [`distance`] against any reference tensor (the generic harness's
/// `mean_abs_over_max` norm).
pub fn distance_to(h: &[f32], h_ref: &[f64]) -> f64 {
    let scale = h_ref.iter().filter(|v| v.is_finite()).fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
    let mut sum = 0.0;
    for (g, want) in h.iter().zip(h_ref) {
        let g = f64::from(*g);
        sum += if !want.is_finite() || !g.is_finite() {
            let same = (g.is_nan() && want.is_nan()) || g == *want;
            if same { 0.0 } else { 1.0 }
        } else {
            (g - want).abs() / scale
        };
    }
    sum / h_ref.len() as f64
}

/// The constraint's stratified sample: every band-forcing case (family D, 100) and
/// seven of each other family, chosen by `seed`; 128 probes, so a kernel
/// wrong on bands, items, columns, accumulation or non-finites fails here
/// in seconds before the full set is paid for.
pub fn admission(all: &[Probe], seed: u64) -> Vec<usize> {
    let mut state = seed ^ 0x1234_5678_9ABC_DEF0;
    let mut out: Vec<usize> = all.iter().enumerate().filter(|(_, p)| p.family == 'D').map(|(i, _)| i).collect();
    for family in ['A', 'B', 'C', 'E'] {
        let mut idx: Vec<usize> = all.iter().enumerate().filter(|(_, p)| p.family == family).map(|(i, _)| i).collect();
        for i in (1..idx.len()).rev() {
            idx.swap(i, (splitmix64(&mut state) % (i as u64 + 1)) as usize);
        }
        out.extend(idx.into_iter().take(7));
    }
    out.sort_unstable();
    out
}

/// A probe passes when its distance is within f32 rounding of a few terms.
pub const PASS: f64 = 1e-5;

/// Every probe's distance for one compiled kernel, in probe order; the
/// seconds it took.
#[cfg(feature = "gpu")]
pub fn run_probes(gram: &crate::wgsl::gram::Gram, probes: &[Probe]) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(probes.len());
    for p in probes {
        let b = gram.build(&p.phi, p.d, p.m, p.limit)?;
        out.push(distance(&b.h, p));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped kernel passes every probe. The shipped kernel with the
    /// product replaced by one operand fails every probe (measured
    /// 2026-10-09: 1,024 of 1,024): a global break fails every probe, so
    /// the probes' independence shows only for local breaks, which the
    /// second broken kernel is: the item loop stopped one item early
    /// fails exactly the probes that touch the last item.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_shipped_kernel_passes_every_probe_and_a_broken_one_fails_some_on_the_device() {
        use crate::wgsl::gram::{Gram, SHIPPED};
        use crate::wgsl::oracle::device::Device;
        let dev = Device::new().expect("device");
        let ps = probes(1);
        let t = std::time::Instant::now();
        let shipped = run_probes(&Gram::compile(&dev, SHIPPED, crate::wgsl::oracle::device::Fuel::Proven("the shipped Gram kernel")).unwrap(), &ps).unwrap();
        let secs = t.elapsed().as_secs_f64();
        let failing: Vec<&str> = ps.iter().zip(&shipped).filter(|(_, d)| **d > PASS).map(|(p, _)| p.name.as_str()).collect();
        println!("{} probes in {secs:.2} s; shipped fails {}: {:?}", ps.len(), failing.len(), &failing[..failing.len().min(10)]);
        assert!(failing.is_empty(), "{failing:?}");
        // The product replaced by its first operand: sums of phi instead of phi².
        let broken = SHIPPED.replace("phi[base + gr] * phi[base + c]", "phi[base + gr]");
        assert_ne!(broken, SHIPPED);
        let b = run_probes(&Gram::compile(&dev, &broken, crate::wgsl::oracle::device::Fuel::Proven("legacy kingdom candidate: no fuel budget exists for it, bounded only by the harness runaway guard")).unwrap(), &ps).unwrap();
        let fails = b.iter().filter(|d| **d > PASS).count();
        println!("product replaced by one operand: fails {fails} of {}", ps.len());
        assert_eq!(fails, ps.len());
        // The loop stopped one item early: only the probes that place a value at the last item fail.
        let short = SHIPPED.replace("t < cfg.m;", "t + 1u < cfg.m;");
        assert_ne!(short, SHIPPED);
        let b = run_probes(&Gram::compile(&dev, &short, crate::wgsl::oracle::device::Fuel::Proven("legacy kingdom candidate: no fuel budget exists for it, bounded only by the harness runaway guard")).unwrap(), &ps).unwrap();
        let failing: Vec<&Probe> = ps.iter().zip(&b).filter(|(_, d)| **d > PASS).map(|(p, _)| p).collect();
        let touches_last = ps.iter().filter(|p| p.phi[(p.m - 1) * p.d..].iter().any(|x| *x != 0.0)).count();
        println!("loop one item short: fails {} of {}; probes with a value at the last item: {touches_last}", failing.len(), ps.len());
        assert!(!failing.is_empty() && failing.len() < ps.len());
        assert!(failing.iter().all(|p| p.phi[(p.m - 1) * p.d..].iter().any(|x| *x != 0.0)), "a failing probe has no value at the last item");
    }

    #[test]
    fn a_thousand_probes_with_known_answers() {
        let ps = probes(1);
        assert_eq!(ps.len(), 1024);
        let count = |f: char| ps.iter().filter(|p| p.family == f).count();
        assert_eq!((count('A'), count('B'), count('C'), count('D'), count('E')), (400, 200, 200, 100, 124));
        // A: exactly one non-zero cell in H, the square.
        let a = ps.iter().find(|p| p.name == "A t1 c0 v2.5" || p.name.starts_with("A t1 c") && p.name.ends_with("v2.5")).unwrap();
        assert_eq!(a.h_ref.iter().filter(|v| **v != 0.0).count(), 1);
        assert!(a.h_ref.iter().any(|v| (*v - 6.25).abs() < 1e-12));
        // C: the squares add.
        let c = ps.iter().find(|p| p.family == 'C').unwrap();
        let nonzero: Vec<f64> = c.h_ref.iter().copied().filter(|v| *v != 0.0).collect();
        assert_eq!(nonzero.len(), 1);
        // The distance is zero for the reference and positive for a wrong table.
        let exact: Vec<f32> = c.h_ref.iter().map(|v| *v as f32).collect();
        assert_eq!(distance(&exact, c), 0.0);
        let mut wrong = exact.clone();
        wrong[0] += 1.0;
        assert!(distance(&wrong, c) > 0.0);
        let zero = vec![0.0f32; c.d * c.d];
        assert!(distance(&zero, c) > distance(&wrong, c) || nonzero[0].abs() < 1.0);
        // Validation probes differ from training probes.
        let qs = probes(2);
        assert!(ps.iter().zip(&qs).any(|(p, q)| p.phi != q.phi));
        // D forces bands: at least one probe has a limit under one row band of its Φ.
        assert!(ps.iter().any(|p| p.family == 'D' && p.limit < (p.m * p.d * 4) as u64));
        // The constraint's sample: 128, every D case, seven of each other family.
        let adm = admission(&ps, 1);
        assert_eq!(adm.len(), 128);
        assert_eq!(adm.iter().filter(|&&i| ps[i].family == 'D').count(), 100);
        assert_eq!(adm.iter().filter(|&&i| ps[i].family == 'A').count(), 7);
    }
}
