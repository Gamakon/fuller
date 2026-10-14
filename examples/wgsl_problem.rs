//! Score kernels against a problem directory with the generic harness
//! (`fuller::wgsl::harness`) on the device, one TSV line a kernel on
//! stdout: feasibility, cases passed, the distance, the closeness angle,
//! the HFF angle, the detail. With `--compare-gram` (the Gram problem only)
//! each kernel is scored by `gram::FitnessContext::fitness` too and the
//! line says whether the two agree on feasibility, cases passed and every
//! closeness column, bit for bit: the device half of the harness's
//! acceptance (the host half is `harness::tests`).
//!
//! A watchdog ends the process, naming the kernel, if one evaluation runs
//! past `--deadline` seconds (120 by default): no call can cancel GPU work.
//!
//! `cargo run --release --features gpu,wgsl --example wgsl_problem -- --problem problems/gram [--compare-gram] [--deadline 120] <kernel.wgsl or dir>...`

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use fuller::wgsl::gram::FitnessContext;
use fuller::wgsl::harness::{DeviceExecutor, Harness, Problem};
use fuller::wgsl::oracle::device::Device;

static DEADLINE: AtomicU64 = AtomicU64::new(0);

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Result<Option<String>, String> {
        match args.iter().position(|a| a == flag) {
            Some(i) => {
                let v = args.get(i + 1).cloned().ok_or(format!("{flag} needs a value"))?;
                args.drain(i..i + 2);
                Ok(Some(v))
            }
            None => Ok(None),
        }
    };
    let problem = take("--problem")?.ok_or("usage: wgsl_problem --problem <dir> [--compare-gram] [--deadline 120] <kernel.wgsl or dir>...")?;
    let deadline_s: u64 = take("--deadline")?.map_or(Ok(120), |v| v.parse().map_err(|e| format!("--deadline: {e}")))?;
    let compare = args.iter().position(|a| a == "--compare-gram").map(|i| args.remove(i)).is_some();
    let mut kernels: Vec<PathBuf> = Vec::new();
    for a in &args {
        let p = Path::new(a);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p).map_err(|e| format!("{a}: {e}"))?.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "wgsl")).collect();
            v.sort();
            kernels.extend(v);
        } else {
            kernels.push(p.to_path_buf());
        }
    }
    let start = Instant::now();
    let current = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    {
        let current = current.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let d = DEADLINE.load(Ordering::SeqCst);
            if d != 0 && start.elapsed().as_millis() as u64 > d {
                eprintln!("watchdog: {} ran past its deadline; exiting", current.lock().map(|c| c.clone()).unwrap_or_default());
                std::process::exit(3);
            }
        });
    }
    let dev = Device::new()?;
    let exec = DeviceExecutor::new(&dev);
    let harness = Harness::new(Problem::load(Path::new(&problem))?, &exec)?;
    let ctx = if compare { Some(FitnessContext::new(&dev, Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/logs/WGSL_GRAM/pairs")))?) } else { None };
    println!("kernel\tfeasible\tpassed\tdistance\tcloseness_hff\thff\tdetail{}", if compare { "\tgram_feasible\tgram_passed\tagree" } else { "" });
    let (mut agree, mut n) = (0usize, 0usize);
    for k in &kernels {
        let text = std::fs::read_to_string(k).map_err(|e| format!("{}: {e}", k.display()))?;
        *current.lock().map_err(|e| e.to_string())? = k.display().to_string();
        DEADLINE.store(start.elapsed().as_millis() as u64 + deadline_s * 1000, Ordering::SeqCst);
        let f = harness.evaluate(&text);
        let g = ctx.as_ref().map(|c| c.fitness(&dev, &text));
        DEADLINE.store(0, Ordering::SeqCst);
        let mut line = format!("{}\t{}\t{}\t{:.3e}\t{:.4}\t{:.4}\t{}", k.display(), f.feasible, f.passed, f.distance, f.closeness_hff, f.hff, f.detail.replace(['\t', '\n'], " "));
        if let Some(g) = g {
            let same = f.feasible == g.feasible && f.passed == g.passed && f.closeness.iter().zip(&g.closeness).all(|(a, b)| a.to_bits() == b.to_bits());
            agree += usize::from(same);
            line += &format!("\t{}\t{}\t{same}", g.feasible, g.passed);
        }
        n += 1;
        println!("{line}");
    }
    if compare {
        eprintln!("{agree} of {n} kernels: the harness agrees with gram.rs on feasibility, cases passed and every closeness column");
    }
    Ok(())
}
