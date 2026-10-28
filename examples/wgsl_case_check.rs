//! Judge kernels of a harness problem case by case (`fuller::wgsl::case_set`), and time them against the problem's
//! shipped kernel on its timing cases in interleaved pairs.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_case_check -- --problem problems/kmeans_update \
//!       [--pairs 11 --runs 3 [--shape i]] <kernel.wgsl>...
//!
//! For every kernel, one line a constraint case: the split, the suite's norm against the expected output (the
//! shipped kernel's own output: the oracle), the suite's pass value, the cells not bit for bit the expected, the
//! cells, and the run's GPU ms. With `--pairs N`, per timing case (`--shape i`: only the `i`th), N pairs of the
//! kernel against the shipped kernel, the order alternating (shipped first in the even pairs), each side the least of
//! `--runs` timed runs; one line a pair (the ratio kernel / shipped, above 1 slower), then the pairs' least, median
//! and largest ratio. The shipped kernel given as a kernel is the A/A pair.
//!
//! Every dispatch runs with the shipped kernel's and hand kernels' fuel, `Fuel::Proven`: only hand-written kernels
//! are given to this check, never a random or evolved gene. One GPU job at a time, under the GPU lock.

use std::path::PathBuf;
use std::time::Instant;

use fuller::wgsl::case_set::{CaseSet, Split};
use fuller::wgsl::harness::DeviceExecutor;
use fuller::wgsl::oracle::device::{Device, Fuel};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn stamp() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", (s / 3600) % 24, (s / 60) % 60, s % 60)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    if s.is_empty() {
        return f64::NAN;
    }
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--problem").ok_or("--problem DIR")?);
    let pairs: usize = arg(&args, "--pairs").map_or(Ok(0), |v| v.parse().map_err(|e| format!("--pairs: {e}")))?;
    let runs: usize = arg(&args, "--runs").map_or(Ok(3), |v| v.parse().map_err(|e| format!("--runs: {e}")))?;
    let shape: Option<usize> = arg(&args, "--shape").map(|v| v.parse().map_err(|e| format!("--shape: {e}"))).transpose()?;
    let skip: Vec<&str> = ["--problem", "--pairs", "--runs", "--shape"].to_vec();
    let mut kernels = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if skip.contains(&args[i].as_str()) {
            i += 2;
            continue;
        }
        kernels.push(PathBuf::from(&args[i]));
        i += 1;
    }
    if kernels.is_empty() {
        return Err("no kernel given".into());
    }
    let dev = Device::new()?;
    let t0 = Instant::now();
    let problem = fuller::wgsl::harness::Problem::load(&dir)?;
    let exec = DeviceExecutor::for_problem(&dev, &problem, Fuel::Proven("wgsl_case_check: the shipped kernel and hand-written kernels only"));
    let set = CaseSet::new(&dir, exec)?;
    println!("# {} {} staged: {} constraint cases, {} timing cases ({:.1} s)", stamp(), set.problem.spec.name, set.cases.len(), set.timing.len(), t0.elapsed().as_secs_f64());
    let (shipped, sinfo) = set.compile(&set.problem.reference)?;
    println!("kernel\tcase\tsplit\tnorm\tpass\tdiffer\tcells\tgpu_ms");
    for path in &kernels {
        let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let (k, info) = match set.compile(&src) {
            Ok(x) => x,
            Err(e) => {
                println!("{name}\tCOMPILE\t{e}");
                continue;
            }
        };
        for c in set.judge(&k, &info)? {
            let split = if c.split == Split::Held { "held" } else { "train" };
            println!("{name}\t{}\t{split}\t{:.3e}\t{:.3e}\t{}\t{}\t{:.3}", c.name, c.norm, c.pass, c.differ, c.cells, c.gpu_ms);
        }
        if pairs == 0 {
            continue;
        }
        for (ti, t) in set.timing.iter().enumerate() {
            if shape.is_some_and(|s| s != ti) {
                continue;
            }
            let best = |kk: &_, ii: &_| -> Result<f64, String> {
                let mut m = f64::INFINITY;
                for _ in 0..runs {
                    m = m.min(set.time(kk, ii, ti)?);
                }
                Ok(m)
            };
            let mut ratios = Vec::new();
            for p in 0..pairs {
                let (a, b) = if p % 2 == 0 {
                    let a = best(&shipped, &sinfo)?;
                    (a, best(&k, &info)?)
                } else {
                    let b = best(&k, &info)?;
                    (best(&shipped, &sinfo)?, b)
                };
                ratios.push(b / a);
                println!("# {} pair\t{name}\t{}\t{p}\tshipped {a:.3} ms\tkernel {b:.3} ms\tratio {:.4}", stamp(), t.case.name, b / a);
            }
            let lo = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = ratios.iter().cloned().fold(0.0, f64::max);
            println!("pairs\t{name}\t{}\t{pairs} pairs\tleast {lo:.4}\tmedian {:.4}\tlargest {hi:.4}", t.case.name, median(&ratios));
        }
    }
    println!("# {} done ({:.1} s)", stamp(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
