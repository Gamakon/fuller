//! The measured cost gate on real kernels: for each WGSL file, the fast
//! fitness (`gram::FitnessContext::fitness`) with its predicted and measured
//! wall time at the three timing shapes, whether it passed, and where the
//! evaluation's time went. With `--verdicts <file>` it also writes one
//! line a kernel: feasible, cases passed, detail, the six columns as f64
//! bits, the speed-ups; two such files compare verdicts to the bit.
//!
//! `cargo run --release --features gpu,wgsl --example wgsl_gram_gate -- [--verdicts out.tsv] <kernel.wgsl>...`

use std::io::Write;

use fuller::wgsl::gram::{FitnessContext, Split};
use fuller::wgsl::oracle::device::Device;

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let verdicts = match args.iter().position(|a| a == "--verdicts") {
        Some(i) => {
            let path = args.get(i + 1).ok_or("--verdicts needs a file")?.clone();
            args.drain(i..i + 2);
            Some(std::fs::File::create(&path).map_err(|e| format!("{path}: {e}"))?)
        }
        None => None,
    };
    if args.is_empty() {
        return Err("usage: wgsl_gram_gate [--verdicts out.tsv] <kernel.wgsl>...".into());
    }
    let mut verdicts = verdicts;
    let dev = Device::new()?;
    let t = std::time::Instant::now();
    let ctx = FitnessContext::new(&dev, std::path::Path::new("logs/WGSL_GRAM/pairs"))?;
    println!("context {:.1} s\tshipped measured\t{:.0} / {:.0} / {:.0} ms\tslow limit {:.0} ms a build", t.elapsed().as_secs_f64(), ctx.shipped_ms[0], ctx.shipped_ms[1], ctx.shipped_ms[2], ctx.slow_limit_ms());
    println!("kernel\tfeasible\thff\tpredicted ms (3 shapes)\tmeasured ms (3 shapes)\trate ms/1e9\tfloor ms\tseconds\tdetail");
    let (mut feasible, mut infeasible) = ((Split::default(), 0usize), (Split::default(), 0usize));
    for f in &args {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        let r = ctx.fitness(&dev, &text);
        println!("{f}\t{}\t{:.4}\t{:.0} / {:.0} / {:.0}\t{:.0} / {:.0} / {:.0}\t{:.2}\t{:.2}\t{:.2}\t{}", r.feasible, r.hff, r.predicted_ms[0], r.predicted_ms[1], r.predicted_ms[2], r.measured_ms[0], r.measured_ms[1], r.measured_ms[2], r.rate_per_g, r.floor_ms, r.seconds, r.detail);
        println!("  split, s: {}", r.split.line());
        let side = if r.feasible { &mut feasible } else { &mut infeasible };
        side.0.add(&r.split);
        side.1 += 1;
        if let Some(v) = verdicts.as_mut() {
            let bits: Vec<String> = r.columns.iter().map(|c| format!("{:016x}", c.to_bits())).collect();
            writeln!(v, "{f}\t{}\t{}\t{}\t{}\t{:.4} {:.4} {:.4}", r.feasible, r.passed, r.detail, bits.join(" "), r.speedups[0], r.speedups[1], r.speedups[2]).map_err(|e| e.to_string())?;
        }
    }
    for (name, (s, n)) in [("feasible", feasible), ("infeasible", infeasible)] {
        if n > 0 {
            println!("{name}: {n} kernels, {:.2} s a kernel; summed split, s: {}", s.total / 1e3 / n as f64, s.line());
        }
    }
    Ok(())
}
