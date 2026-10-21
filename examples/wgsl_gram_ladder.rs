//! The ladder of `docs/PLAN_hgp_gram.md` §"The ladder" for ONE kernel text:
//!
//!   rung 1  naga parses it
//!   rung 2  naga validates it and Metal compiles it (the `// grid` header
//!           checked against the shader's workgroup size)
//!   stage 3  the constraint's 128 test cases, then all 1,024, then every
//!           training, validation and extrapolation pair under the
//!           criterion (correct = every probe and every pair)
//!   rung 4  paired timing against the shipped kernel on three shapes,
//!           `--rounds` rounds in random order, beside shipped-against-
//!           shipped rounds (the null), the geometric mean and the worst
//!           shape
//!
//! One kernel per process, so a wedged device or a panic costs one
//! candidate and the caller's loop goes on. A summary line is appended to
//! `--ledger` (TSV) and everything is printed with timestamps.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_gram_ladder -- <kernel.wgsl> [--rounds 5] [--probes-only] [--ledger logs/HGP/ladder.tsv] [--cache <dir, default $FULLER_GRAM_CACHE or ~/.cache/fuller/wgsl_gram_pairs>]

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use fuller::wgsl::gram::{check, effective_buffer_limit, grid_header, Gram, SHIPPED};
use fuller::wgsl::oracle::device::Device;
use fuller::wgsl::pairs::{splits, Pair};
use fuller::wgsl::probes::{admission, distance, probes, PASS};

fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The gate on one pair: every non-finite cell in class, the largest ratio
/// within τ or within twice the shipped kernel's own on that pair.
fn gate(v: &fuller::wgsl::gram::Verdict, shipped_max: f64) -> bool {
    v.non_finite == 0 && v.max_ratio <= (2.0 * shipped_max).max(1.0)
}

struct Summary {
    rung: u32,
    detail: String,
    probes_passed: usize,
    pairs_passed: usize,
    pairs_total: usize,
    worst_pair_ratio: f64,
    /// The worst error over the training, validation and extrapolation
    /// pairs, each in units of the shipped kernel's own worst error on that
    /// pair: the accuracy and generalisation columns of the HFF.
    worst_train: f64,
    worst_val: f64,
    worst_extrap: f64,
    speedups: Vec<(String, f64, f64, f64)>,
    null: Vec<(String, f64)>,
}

/// The HFF of a feasible individual: six columns in [0, 1], 0 best — the
/// time ratio against the shipped kernel on each of the three problem
/// sizes (min(1, 1 / speed-up)), and the worst error relative to the
/// shipped kernel's on the training, validation and extrapolation pairs
/// (min(1, ratio / 2): at the gate's limit, twice the shipped error, the
/// column is 1; at the shipped error 0.5; better than shipped below).
/// Fitness θ = acos(1 − mean(x²)) (phylu's hff.wgsl, lines 212–213); 0 is
/// the ideal, π/2 a kernel no faster and no more accurate than shipped at
/// the gate's limit. An infeasible individual has no angle.
fn hff(s: &Summary) -> Option<f64> {
    if s.rung < 4 || s.speedups.len() != 3 {
        return None;
    }
    let mut x: Vec<f64> = s.speedups.iter().map(|(_, m, _, _)| (1.0 / m).clamp(0.0, 1.0)).collect();
    for e in [s.worst_train, s.worst_val, s.worst_extrap] {
        x.push((e / 2.0).clamp(0.0, 1.0));
    }
    let energy = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let cos = (1.0 - energy.min(1.0)).clamp(-1.0, 1.0);
    Some(if cos > 1.0 - 1.1920929e-7 { 0.0 } else { cos.acos() })
}

fn main() -> Result<(), String> {
    let mut rounds = 5usize;
    let mut probes_only = false;
    let mut timing_only = false;
    let mut ledger = PathBuf::from("logs/HGP/ladder.tsv");
    let mut cache = fuller::wgsl::pairs::cache_dir();
    let mut kernel: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--rounds" => rounds = value("--rounds")?.parse().map_err(|e| format!("--rounds: {e}"))?,
            "--probes-only" => probes_only = true,
            "--timing-only" => timing_only = true,
            "--ledger" => ledger = PathBuf::from(value("--ledger")?),
            "--cache" => cache = PathBuf::from(value("--cache")?),
            path => kernel = Some(path.to_string()),
        }
    }
    let path = kernel.ok_or("usage: wgsl_gram_ladder <kernel.wgsl> [--rounds n] [--probes-only] [--ledger f] [--cache dir]")?;
    let source = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let t_all = Instant::now();
    println!("[{}] ladder: {path} ({} bytes, header {:?})", stamp(), source.len(), grid_header(&source));
    let mut s = Summary { rung: 0, detail: String::new(), probes_passed: 0, pairs_passed: 0, pairs_total: 0, worst_pair_ratio: 0.0, worst_train: 0.0, worst_val: 0.0, worst_extrap: 0.0, speedups: Vec::new(), null: Vec::new() };

    // Rung 1: parse.
    match fuller::wgsl::naga_shape::parse(&source) {
        Ok(_) => {
            s.rung = 1;
            println!("[{}] rung 1 parse: ok", stamp());
        }
        Err(e) => {
            s.detail = e.lines().find(|l| l.contains("error")).unwrap_or("parse error").to_string();
            println!("[{}] rung 1 parse: FAILED {}", stamp(), s.detail);
            return finish(&ledger, &path, &s, t_all);
        }
    }
    // Rung 2: validate and compile.
    let dev = Device::new()?;
    let limit = effective_buffer_limit(&dev.device);
    let gram = match Gram::compile(&dev, &source, fuller::wgsl::oracle::device::Fuel::Proven("legacy kingdom candidate: no fuel budget exists for it, bounded only by the harness runaway guard")) {
        Ok(g) => {
            s.rung = 2;
            println!("[{}] rung 2 validate+compile: ok, grid {:?}", stamp(), g.grid);
            g
        }
        Err(e) => {
            s.detail = e.lines().find(|l| l.contains("error") || l.contains("header") || l.contains("pipeline")).unwrap_or(e.lines().next().unwrap_or("")).to_string();
            println!("[{}] rung 2 validate+compile: FAILED {}", stamp(), s.detail);
            return finish(&ledger, &path, &s, t_all);
        }
    };
    let cache_dir = cache.clone();
    if timing_only {
        s.rung = 3;
        s.detail = "timing only (probes and pairs passed in an earlier run)".into();
        return rung4(&dev, &gram, limit, &cache_dir, rounds, &mut s, &ledger, &path, t_all);
    }
    // Stage 3: the constraint's test cases first.
    let all = probes(1);
    let adm = admission(&all, 1);
    let t = Instant::now();
    let mut passed = 0;
    let mut first_fail: Option<String> = None;
    for &i in &adm {
        let p = &all[i];
        let b = gram.build(&p.phi, p.d, p.m, p.limit).map_err(|e| format!("probe {}: {e}", p.name))?;
        let dist = distance(&b.h, p);
        if dist <= PASS {
            passed += 1;
        } else if first_fail.is_none() {
            first_fail = Some(format!("{} (distance {dist:.2e})", p.name));
        }
    }
    println!("[{}] stage 3 constraint cases: {passed}/{} in {:.1} s{}", stamp(), adm.len(), t.elapsed().as_secs_f64(), first_fail.as_ref().map(|f| format!("; first failure {f}")).unwrap_or_default());
    s.probes_passed = passed;
    if passed < adm.len() {
        s.detail = format!("constraint {passed}/{} cases: {}", adm.len(), first_fail.unwrap_or_default());
        return finish(&ledger, &path, &s, t_all);
    }
    let t = Instant::now();
    let mut passed = 0;
    let mut fails: Vec<String> = Vec::new();
    for p in &all {
        let b = gram.build(&p.phi, p.d, p.m, p.limit).map_err(|e| format!("probe {}: {e}", p.name))?;
        let dist = distance(&b.h, p);
        if dist <= PASS {
            passed += 1;
        } else if fails.len() < 5 {
            fails.push(format!("{} ({dist:.2e})", p.name));
        }
    }
    println!("[{}] rung 3 all probes: {passed}/{} in {:.1} s{}", stamp(), all.len(), t.elapsed().as_secs_f64(), if fails.is_empty() { String::new() } else { format!("; failures {fails:?}") });
    s.probes_passed = passed;
    if passed < all.len() {
        s.detail = format!("probes {passed}/{}: {}", all.len(), fails.join(", "));
        return finish(&ledger, &path, &s, t_all);
    }
    if probes_only {
        s.rung = 3;
        s.detail = "probes only".into();
        return finish(&ledger, &path, &s, t_all);
    }
    // Rung 3: the pairs.
    let sp = splits(&cache)?;
    println!("[{}] pairs: {} train, {} validation, {} extrapolation (references in {:.1} s)", stamp(), sp.train.len(), sp.val.len(), sp.extrap.len(), sp.seconds);
    let shipped = Gram::compile(&dev, SHIPPED, fuller::wgsl::oracle::device::Fuel::Proven("the shipped Gram kernel"))?;
    let mut pair_fail: Vec<String> = Vec::new();
    for (split, pairs) in [("train", &sp.train), ("validation", &sp.val), ("extrapolation", &sp.extrap)] {
        let t = Instant::now();
        let mut ok = 0;
        for p in pairs.iter() {
            let b = gram.build(&p.phi, p.d, p.m, limit)?;
            let v = check(&b.h, &p.h_ref, p.d, p.m);
            let sb = shipped.build(&p.phi, p.d, p.m, limit)?;
            let sv = check(&sb.h, &p.h_ref, p.d, p.m);
            s.pairs_total += 1;
            let ratio = if sv.max_ratio > 0.0 { v.max_ratio / sv.max_ratio } else { v.max_ratio };
            s.worst_pair_ratio = s.worst_pair_ratio.max(ratio);
            match split {
                "train" => s.worst_train = s.worst_train.max(ratio),
                "validation" => s.worst_val = s.worst_val.max(ratio),
                _ => s.worst_extrap = s.worst_extrap.max(ratio),
            }
            if gate(&v, sv.max_ratio) {
                ok += 1;
                s.pairs_passed += 1;
            } else {
                pair_fail.push(format!("{split} {}: max {:.3} tau (shipped {:.3}), non-finite {}", p.name(), v.max_ratio, sv.max_ratio, v.non_finite));
            }
        }
        println!("[{}] rung 3 {split} pairs: {ok}/{} in {:.0} s", stamp(), pairs.len(), t.elapsed().as_secs_f64());
    }
    if !pair_fail.is_empty() {
        for f in &pair_fail {
            println!("    {f}");
        }
        s.detail = format!("pairs {}/{}: {}", s.pairs_passed, s.pairs_total, pair_fail[0]);
        return finish(&ledger, &path, &s, t_all);
    }
    s.rung = 3;
    rung4(&dev, &gram, limit, &cache, rounds, &mut s, &ledger, &path, t_all)
}

#[allow(clippy::too_many_arguments)]
fn rung4(dev: &Device, gram: &Gram, limit: u64, cache: &std::path::Path, rounds: usize, s: &mut Summary, ledger: &PathBuf, path: &str, t_all: Instant) -> Result<(), String> {
    let shipped = Gram::compile(dev, SHIPPED, fuller::wgsl::oracle::device::Fuel::Proven("the shipped Gram kernel"))?;
    // Rung 4 only on an accepted GPU: the shipped kernel alone, at the
    // bench shape, must build within 12 % of its reference 1,015 ms (the
    // quiet-machine measurement of 2026-10-10 15:30). Contention hurts a
    // kernel with barriers far more than the shipped one, so the
    // shipped-against-shipped null alone does not detect it (the tiled
    // kernel read 0.4x under contention and 7.9x on a quiet chip).
    // Measured 2026-10-10 16:01-16:08: after the 87 pairs the shipped build
    // reads 1,200-1,313 ms with nothing else on the chip; a laptop GPU
    // lowers its clock under sustained load. A lower clock scales both
    // kernels alike, so the paired ratio stands; another process on the
    // chip does not, and the null shows it. So: a cool-down pause, the
    // absolute reading logged, and rung 4 refused only when the null
    // (shipped against shipped) is off 1.0 by more than 5 %.
    const REFERENCE_MS: f64 = 1015.0;
    std::thread::sleep(std::time::Duration::from_secs(15));
    {
        let p = Pair::load(fuller::wgsl::gram::Kind::Gaussian, 2048, 65536, 7, cache)?;
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            best = best.min(shipped.build(&p.phi, 2048, 65536, limit)?.gpu_ms);
        }
        println!("[{}] GPU state after a 15 s pause: shipped alone {best:.0} ms against the quiet-chip reference {REFERENCE_MS:.0} ms ({:+.0} %)", stamp(), 100.0 * (best / REFERENCE_MS - 1.0));
    }
    let shipped2 = Gram::compile(dev, SHIPPED, fuller::wgsl::oracle::device::Fuel::Proven("the shipped Gram kernel"))?;
    let mut state = 0x5EED_u64;
    let shapes: [(usize, usize); 3] = [(2048, 65536), (4096, 16384), (8192, 4096)];
    for (d, m) in shapes {
        let p = Pair::load(fuller::wgsl::gram::Kind::Gaussian, d, m, 7, cache)?;
        let t = Instant::now();
        let mut ratios = Vec::new();
        let mut nulls = Vec::new();
        for _ in 0..rounds {
            // Candidate and shipped back to back in a random order; then
            // the second shipped instance against the first (the null).
            let cand_first = next(&mut state).is_multiple_of(2);
            let (tc, ts) = if cand_first {
                let tc = gram.build(&p.phi, d, m, limit)?;
                (tc, shipped.build(&p.phi, d, m, limit)?)
            } else {
                let ts = shipped.build(&p.phi, d, m, limit)?;
                (gram.build(&p.phi, d, m, limit)?, ts)
            };
            let v = check(&tc.h, &p.h_ref, d, m);
            if !gate(&v, check(&ts.h, &p.h_ref, d, m).max_ratio) {
                s.detail = format!("timed build wrong at d {d} m {m}: max {:.3} tau", v.max_ratio);
                return finish(ledger, path, s, t_all);
            }
            ratios.push(ts.gpu_ms / tc.gpu_ms);
            let n2 = shipped2.build(&p.phi, d, m, limit)?;
            nulls.push(ts.gpu_ms / n2.gpu_ms);
        }
        ratios.sort_by(f64::total_cmp);
        nulls.sort_by(f64::total_cmp);
        let (med, lo, hi) = (ratios[ratios.len() / 2], ratios[0], ratios[ratios.len() - 1]);
        let null_med = nulls[nulls.len() / 2];
        if (null_med - 1.0).abs() > 0.05 {
            s.detail = format!("GPU shared: the null shipped/shipped reads {null_med:.3} at d {d} m {m}; rung 4 not accepted");
            println!("[{}] rung 4 d {d} m {m}: REFUSED, null {null_med:.3}", stamp());
            return finish(ledger, path, s, t_all);
        }
        println!("[{}] rung 4 d {d} m {m}: speed-up median {med:.3} (min {lo:.3} max {hi:.3}); null shipped/shipped median {null_med:.3} (min {:.3} max {:.3}); {:.0} s", stamp(), nulls[0], nulls[nulls.len() - 1], t.elapsed().as_secs_f64());
        s.speedups.push((format!("d{d}_m{m}"), med, lo, hi));
        s.null.push((format!("d{d}_m{m}"), null_med));
    }
    s.rung = 4;
    finish(ledger, path, s, t_all)
}

fn finish(ledger: &PathBuf, path: &str, s: &Summary, t_all: Instant) -> Result<(), String> {
    let geo = if s.speedups.is_empty() { 0.0 } else { (s.speedups.iter().map(|x| x.1.ln()).sum::<f64>() / s.speedups.len() as f64).exp() };
    let worst = s.speedups.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    let angle = hff(s);
    println!("[{}] result: rung {} probes {} pairs {}/{} worst error train {:.3} val {:.3} extrap {:.3} (x shipped's); speed-up geometric mean {geo:.3}, min over sizes {:.3}; HFF {}; {:.0} s; {}", stamp(), s.rung, s.probes_passed, s.pairs_passed, s.pairs_total, s.worst_train, s.worst_val, s.worst_extrap, if worst.is_finite() { worst } else { 0.0 }, angle.map_or("infeasible".to_string(), |a| format!("{a:.4}")), t_all.elapsed().as_secs_f64(), s.detail);
    if let Some(dir) = ledger.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let new = !ledger.exists();
    let mut f = std::fs::OpenOptions::new().append(true).create(true).open(ledger).map_err(|e| e.to_string())?;
    if new {
        writeln!(f, "time\tkernel\trung\tprobes\tpairs\tworst_pair_ratio\tspeedups\tnull\tgeo\tworst\tseconds\tdetail\tworst_train\tworst_val\tworst_extrap\thff").map_err(|e| e.to_string())?;
    }
    writeln!(f, "{}\t{path}\t{}\t{}\t{}/{}\t{:.4}\t{}\t{}\t{geo:.4}\t{:.4}\t{:.0}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}", stamp(), s.rung, s.probes_passed, s.pairs_passed, s.pairs_total, s.worst_pair_ratio, s.speedups.iter().map(|(n, m, lo, hi)| format!("{n}:{m:.3}[{lo:.3},{hi:.3}]")).collect::<Vec<_>>().join(","), s.null.iter().map(|(n, m)| format!("{n}:{m:.3}")).collect::<Vec<_>>().join(","), if worst.is_finite() { worst } else { 0.0 }, t_all.elapsed().as_secs_f64(), s.detail, s.worst_train, s.worst_val, s.worst_extrap, angle.map_or("-".to_string(), |a| format!("{a:.4}"))).map_err(|e| e.to_string())?;
    Ok(())
}
