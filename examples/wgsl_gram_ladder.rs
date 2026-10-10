//! The ladder of `docs/PLAN_hgp_gram.md` §"The ladder" for ONE kernel text:
//!
//!   rung 1  naga parses it
//!   rung 2  naga validates it and Metal compiles it (the `// grid` header
//!           checked against the shader's workgroup size)
//!   rung 3  the 128 admission probes, then all 1,024 probes, then every
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
//!   cargo run --release --features gpu,wgsl --example wgsl_gram_ladder -- <kernel.wgsl> [--rounds 5] [--probes-only] [--ledger logs/HGP/ladder.tsv] [--cache logs/WGSL_GRAM/pairs]

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
    speedups: Vec<(String, f64, f64, f64)>,
    null: Vec<(String, f64)>,
}

fn main() -> Result<(), String> {
    let mut rounds = 5usize;
    let mut probes_only = false;
    let mut ledger = PathBuf::from("logs/HGP/ladder.tsv");
    let mut cache = PathBuf::from("logs/WGSL_GRAM/pairs");
    let mut kernel: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--rounds" => rounds = value("--rounds")?.parse().map_err(|e| format!("--rounds: {e}"))?,
            "--probes-only" => probes_only = true,
            "--ledger" => ledger = PathBuf::from(value("--ledger")?),
            "--cache" => cache = PathBuf::from(value("--cache")?),
            path => kernel = Some(path.to_string()),
        }
    }
    let path = kernel.ok_or("usage: wgsl_gram_ladder <kernel.wgsl> [--rounds n] [--probes-only] [--ledger f] [--cache dir]")?;
    let source = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let t_all = Instant::now();
    println!("[{}] ladder: {path} ({} bytes, header {:?})", stamp(), source.len(), grid_header(&source));
    let mut s = Summary { rung: 0, detail: String::new(), probes_passed: 0, pairs_passed: 0, pairs_total: 0, worst_pair_ratio: 0.0, speedups: Vec::new(), null: Vec::new() };

    // Rung 1: parse.
    match naga::front::wgsl::parse_str(&source) {
        Ok(_) => {
            s.rung = 1;
            println!("[{}] rung 1 parse: ok", stamp());
        }
        Err(e) => {
            s.detail = e.emit_to_string(&source).lines().find(|l| l.contains("error")).unwrap_or("parse error").to_string();
            println!("[{}] rung 1 parse: FAILED {}", stamp(), s.detail);
            return finish(&ledger, &path, &s, t_all);
        }
    }
    // Rung 2: validate and compile.
    let dev = Device::new()?;
    let limit = effective_buffer_limit(&dev.device);
    let gram = match Gram::compile(&dev, &source) {
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
    // Rung 3: probes, admission first.
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
    println!("[{}] rung 3 admission probes: {passed}/{} in {:.1} s{}", stamp(), adm.len(), t.elapsed().as_secs_f64(), first_fail.as_ref().map(|f| format!("; first failure {f}")).unwrap_or_default());
    s.probes_passed = passed;
    if passed < adm.len() {
        s.detail = format!("admission {passed}/{}: {}", adm.len(), first_fail.unwrap_or_default());
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
    let shipped = Gram::compile(&dev, SHIPPED)?;
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
    // Rung 4: paired timing on three shapes with the null.
    let shipped2 = Gram::compile(&dev, SHIPPED)?;
    let mut state = 0x5EED_u64;
    let shapes: [(usize, usize); 3] = [(2048, 65536), (4096, 16384), (8192, 4096)];
    for (d, m) in shapes {
        let p = Pair::load(fuller::wgsl::gram::Kind::Gaussian, d, m, 7, &cache)?;
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
                return finish(&ledger, &path, &s, t_all);
            }
            ratios.push(ts.gpu_ms / tc.gpu_ms);
            let n2 = shipped2.build(&p.phi, d, m, limit)?;
            nulls.push(ts.gpu_ms / n2.gpu_ms);
        }
        ratios.sort_by(f64::total_cmp);
        nulls.sort_by(f64::total_cmp);
        let (med, lo, hi) = (ratios[ratios.len() / 2], ratios[0], ratios[ratios.len() - 1]);
        let null_med = nulls[nulls.len() / 2];
        println!("[{}] rung 4 d {d} m {m}: speed-up median {med:.3} (min {lo:.3} max {hi:.3}); null shipped/shipped median {null_med:.3} (min {:.3} max {:.3}); {:.0} s", stamp(), nulls[0], nulls[nulls.len() - 1], t.elapsed().as_secs_f64());
        s.speedups.push((format!("d{d}_m{m}"), med, lo, hi));
        s.null.push((format!("d{d}_m{m}"), null_med));
    }
    s.rung = 4;
    finish(&ledger, &path, &s, t_all)
}

fn finish(ledger: &PathBuf, path: &str, s: &Summary, t_all: Instant) -> Result<(), String> {
    let geo = if s.speedups.is_empty() { 0.0 } else { (s.speedups.iter().map(|x| x.1.ln()).sum::<f64>() / s.speedups.len() as f64).exp() };
    let worst = s.speedups.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    println!("[{}] result: rung {} probes {} pairs {}/{} worst pair ratio {:.3}x shipped's error; speed-up geometric mean {geo:.3}, worst shape {:.3}; {:.0} s; {}", stamp(), s.rung, s.probes_passed, s.pairs_passed, s.pairs_total, s.worst_pair_ratio, if worst.is_finite() { worst } else { 0.0 }, t_all.elapsed().as_secs_f64(), s.detail);
    if let Some(dir) = ledger.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let new = !ledger.exists();
    let mut f = std::fs::OpenOptions::new().append(true).create(true).open(ledger).map_err(|e| e.to_string())?;
    if new {
        writeln!(f, "time\tkernel\trung\tprobes\tpairs\tworst_pair_ratio\tspeedups\tnull\tgeo\tworst\tseconds\tdetail").map_err(|e| e.to_string())?;
    }
    writeln!(f, "{}\t{path}\t{}\t{}\t{}/{}\t{:.4}\t{}\t{}\t{geo:.4}\t{:.4}\t{:.0}\t{}", stamp(), s.rung, s.probes_passed, s.pairs_passed, s.pairs_total, s.worst_pair_ratio, s.speedups.iter().map(|(n, m, lo, hi)| format!("{n}:{m:.3}[{lo:.3},{hi:.3}]")).collect::<Vec<_>>().join(","), s.null.iter().map(|(n, m)| format!("{n}:{m:.3}")).collect::<Vec<_>>().join(","), if worst.is_finite() { worst } else { 0.0 }, t_all.elapsed().as_secs_f64(), s.detail).map_err(|e| e.to_string())?;
    Ok(())
}
