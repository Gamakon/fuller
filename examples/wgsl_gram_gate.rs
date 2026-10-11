//! The measured cost gate on real kernels: for each WGSL file, the fast
//! fitness (`gram::FitnessContext::fitness`) with its predicted and measured
//! wall time at the three timing shapes, and whether it passed.
//!
//! `cargo run --release --features gpu,wgsl --example wgsl_gram_gate -- <kernel.wgsl>...`

use fuller::wgsl::gram::FitnessContext;
use fuller::wgsl::oracle::device::Device;

fn main() -> Result<(), String> {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        return Err("usage: wgsl_gram_gate <kernel.wgsl>...".into());
    }
    let dev = Device::new()?;
    let ctx = FitnessContext::new(&dev, std::path::Path::new("logs/WGSL_GRAM/pairs"))?;
    println!("shipped measured\t{:.0} / {:.0} / {:.0} ms\tslow limit {:.0} ms a build", ctx.shipped_ms[0], ctx.shipped_ms[1], ctx.shipped_ms[2], ctx.slow_limit_ms());
    println!("kernel\tfeasible\thff\tpredicted ms (3 shapes)\tmeasured ms (3 shapes)\trate ms/1e9\tfloor ms\tseconds\tdetail");
    for f in &files {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        let r = ctx.fitness(&dev, &text);
        println!("  split, s: {}", r.split.line());
        println!("{f}\t{}\t{:.4}\t{:.0} / {:.0} / {:.0}\t{:.0} / {:.0} / {:.0}\t{:.2}\t{:.2}\t{:.1}\t{}", r.feasible, r.hff, r.predicted_ms[0], r.predicted_ms[1], r.predicted_ms[2], r.measured_ms[0], r.measured_ms[1], r.measured_ms[2], r.rate_per_g, r.floor_ms, r.seconds, r.detail);
    }
    Ok(())
}
