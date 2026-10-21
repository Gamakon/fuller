//! HOW CLOSE THE SAVED RUNS' WRONG KERNELS WERE (2026-10-12): every kernel a
//! run's `errors.tsv` names, scored again by the fast fitness
//! (`gram::FitnessContext::fitness`), with its closeness columns
//! (`closeness::COLUMNS`) and their HFF angle, one TSV line a kernel on
//! stdout. Each kernel is read from the run's own `kernels/` (the paths in
//! `errors.tsv` name the directory the run had while it ran).
//!
//! A watchdog ends the process, naming the kernel, if one evaluation runs
//! past `--deadline` seconds (120 by default): no call can cancel GPU work.
//!
//! `cargo run --release --features gpu,wgsl --example wgsl_closeness -- [--deadline 120] <run dir>... > closeness.tsv`

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use fuller::wgsl::closeness;
use fuller::wgsl::gram::FitnessContext;
use fuller::wgsl::oracle::device::Device;

static DEADLINE: AtomicU64 = AtomicU64::new(0);

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut deadline_s = 120u64;
    if let Some(i) = args.iter().position(|a| a == "--deadline") {
        deadline_s = args.get(i + 1).ok_or("--deadline needs seconds")?.parse().map_err(|e| format!("--deadline: {e}"))?;
        args.drain(i..i + 2);
    }
    if args.is_empty() {
        return Err("usage: wgsl_closeness [--deadline 120] <run dir>...".into());
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
    let ctx = FitnessContext::new(&dev, &fuller::wgsl::pairs::cache_dir())?;
    println!("run\tkernel\tgen\trow\torigin\tstage\tfeasible\tpassed\tdistance\tcloseness_hff\t{}\tdetail", closeness::COLUMNS.join("\t"));
    for run in &args {
        let run = Path::new(run);
        let name = run.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let errors = std::fs::read_to_string(run.join("errors.tsv")).map_err(|e| format!("{}: {e}", run.display()))?;
        for line in errors.lines().skip(1) {
            let cols: Vec<&str> = line.split('\t').collect();
            let [g, row, origin, stage, _, path] = cols.as_slice() else { return Err(format!("{}: errors.tsv line of {} columns", run.display(), cols.len())) };
            let file = Path::new(path).file_name().ok_or_else(|| format!("{path}: no file name"))?;
            let kernel = run.join("kernels").join(file);
            let Ok(text) = std::fs::read_to_string(&kernel) else {
                eprintln!("{}: not saved (the failure was before rendering)", kernel.display());
                continue;
            };
            *current.lock().map_err(|e| e.to_string())? = kernel.display().to_string();
            DEADLINE.store(start.elapsed().as_millis() as u64 + deadline_s * 1000, Ordering::SeqCst);
            let r = ctx.fitness(&dev, &text);
            DEADLINE.store(0, Ordering::SeqCst);
            let c: Vec<String> = r.closeness.iter().map(|x| format!("{x:.4}")).collect();
            println!("{name}\t{}\t{g}\t{row}\t{origin}\t{stage}\t{}\t{}\t{:.3e}\t{:.4}\t{}\t{}", file.to_string_lossy(), r.feasible, r.passed, r.distance, r.closeness_hff, c.join("\t"), r.detail.replace(['\t', '\n'], " "));
        }
    }
    Ok(())
}
