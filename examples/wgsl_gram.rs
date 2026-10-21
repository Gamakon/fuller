//! Task 1 of `docs/PLAN_wgsl_gram_superopt.md`: the baseline. Builds a
//! seeded Φ, the f64 reference (cached by `(d, m, seed)` under `--cache`),
//! then `--rounds` builds of each kernel text given (the shipped kernel when
//! none), each through the production-shaped banded driver with GPU
//! timestamps per pass; prints the criterion's verdict and the times.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_gram -- \
//!       [--d 2048] [--m 65536] [--seed 1] [--rounds 5] [--cache logs/WGSL_GRAM] [kernel.wgsl ...]

use std::path::PathBuf;
use std::time::Instant;

use fuller::wgsl::gram::{check, effective_buffer_limit, phi, reference, Gram, SHIPPED};
use fuller::wgsl::oracle::device::Device;

fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (h, mi, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{mi:02}:{s:02}Z")
}

fn main() -> Result<(), String> {
    let mut d = 2048usize;
    let mut m = 65536usize;
    let mut seed = 1u64;
    let mut rounds = 5usize;
    let mut warm = 0usize;
    let mut cache = PathBuf::from("logs/WGSL_GRAM");
    let mut kernels: Vec<(String, String)> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--d" => d = value("--d")?.parse().map_err(|e| format!("--d: {e}"))?,
            "--m" => m = value("--m")?.parse().map_err(|e| format!("--m: {e}"))?,
            "--seed" => seed = value("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--rounds" => rounds = value("--rounds")?.parse().map_err(|e| format!("--rounds: {e}"))?,
            "--warm" => warm = value("--warm")?.parse().map_err(|e| format!("--warm: {e}"))?,
            "--cache" => cache = PathBuf::from(value("--cache")?),
            path => kernels.push((path.to_string(), std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)),
        }
    }
    if kernels.is_empty() {
        kernels.push(("shipped".into(), SHIPPED.to_string()));
    }
    std::fs::create_dir_all(&cache).map_err(|e| format!("{}: {e}", cache.display()))?;

    println!("[{}] phi d={d} m={m} seed={seed} ({:.1} MB f32)", stamp(), (d * m * 4) as f64 / 1e6);
    let t = Instant::now();
    let p = phi(d, m, seed);
    println!("[{}] phi built in {:.2} s", stamp(), t.elapsed().as_secs_f64());

    let ref_path = cache.join(format!("ref_d{d}_m{m}_s{seed}.f64"));
    let h_ref: Vec<f64> = if ref_path.exists() {
        let bytes = std::fs::read(&ref_path).map_err(|e| format!("{}: {e}", ref_path.display()))?;
        if bytes.len() != d * d * 8 {
            return Err(format!("{}: {} bytes, expected {}", ref_path.display(), bytes.len(), d * d * 8));
        }
        println!("[{}] reference read from {}", stamp(), ref_path.display());
        bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
    } else {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        println!("[{}] reference: {:.3e} multiply-adds in f64 over {threads} threads", stamp(), d as f64 * (d + 1) as f64 / 2.0 * m as f64);
        let t = Instant::now();
        let h = reference(&p, d, m);
        println!("[{}] reference built in {:.1} s", stamp(), t.elapsed().as_secs_f64());
        let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&ref_path, bytes).map_err(|e| format!("{}: {e}", ref_path.display()))?;
        println!("[{}] reference cached at {}", stamp(), ref_path.display());
        h
    };

    let dev = Device::new()?;
    let limit = effective_buffer_limit(&dev.device);
    println!("[{}] device {} limit {} MiB per binding", stamp(), dev.name, limit >> 20);
    println!("{:<28} {:>5} {:>9} {:>10} {:>9} {:>9} {:>6} {:>9} {:>9} {:>8}", "kernel", "round", "gpu ms", "compute ms", "wall ms", "upload ms", "passes", "max ratio", "mean", "fails");
    for (name, source) in &kernels {
        let t = Instant::now();
        let gram = match Gram::compile(&dev, source, fuller::wgsl::oracle::device::Fuel::Proven("legacy kingdom candidate: no fuel budget exists for it, bounded only by the harness runaway guard")) {
            Ok(g) => g,
            Err(e) => {
                println!("{name:<28} refused: {e}");
                continue;
            }
        };
        let compile_ms = t.elapsed().as_secs_f64() * 1e3;
        // `--warm n`: n tiny builds (d = 32, m = 16) before the rounds, to
        // see whether a pipeline's speed changes after many dispatches.
        if warm > 0 {
            let (wd, wm) = (32usize, 16usize);
            let wp = phi(wd, wm, 3);
            let t = Instant::now();
            for _ in 0..warm {
                gram.build(&wp, wd, wm, limit)?;
            }
            println!("{name:<28}   warmed with {warm} tiny builds in {:.1} s", t.elapsed().as_secs_f64());
        }
        let mut gpu = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let b = gram.build(&p, d, m, limit)?;
            let v = check(&b.h, &h_ref, d, m);
            println!("{name:<28} {round:>5} {:>9.2} {:>10.2} {:>9.2} {:>9.2} {:>6} {:>9.4} {:>9.5} {:>8}{}", b.gpu_ms, b.compute_ms, b.wall_ms, b.upload_ms, b.passes, v.max_ratio, v.mean_ratio, v.failures, if v.passes() { "" } else { "  FAIL" });
            if round == 0 {
                println!("{name:<28}   compile {compile_ms:.1} ms; bands {} item × {} row; tau {:.3e}; worst entry ({}, {}) |diff| {:.3e}; non-finite {}", b.item_bands, b.row_bands, v.tau, v.worst.0, v.worst.1, v.max_abs, v.non_finite);
            }
            gpu.push(b.gpu_ms);
        }
        gpu.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        println!("{name:<28} gpu ms median {:.2}  min {:.2}  max {:.2}  ({} rounds)", gpu[gpu.len() / 2], gpu[0], gpu[gpu.len() - 1], gpu.len());
    }
    println!("[{}] done", stamp());
    Ok(())
}
