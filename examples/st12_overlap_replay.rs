//! ST12 step 1: replay kernels a run scored as device failures because the two timed dispatches of a pair overlapped.
//!
//!   cargo run --release --features gpu,wgsl --example st12_overlap_replay -- --list FILE [--calls 5] [--pairs 9]
//!
//! `--list` is a TSV of `path<TAB>work bound an invocation` (the bound the run's ledger records for that kernel, so the
//! work guard holds the run's own fuel). Each kernel is compiled at the run's timing size (64 x 64, t 5) and timed against
//! the SHIPPED volumepassage under the kernel's submit plan with `image::timed_pairs` exactly as the fit calls it
//! (1 warm-up pass, `--pairs` pairs), `--calls` times. Every call that fails is printed with its error; the last line
//! counts the calls, the overlaps and every other failure.
use fuller::wgsl::cuneus_originals::case_bindings;
use fuller::wgsl::image::{median, timed_pairs, ImageKernel};
use fuller::wgsl::oracle::device::{Device, Fuel};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn utc() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (t / 3600) % 24, (t / 60) % 60, t % 60)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let number = |name: &str, default: usize| -> Result<usize, String> { arg(&args, name).map(|a| a.parse().map_err(|e| format!("{name}: {e}"))).transpose().map(|v| v.unwrap_or(default)) };
    let (calls, pairs) = (number("--calls", 5)?, number("--pairs", 9)?);
    let list_path = arg(&args, "--list").ok_or("--list FILE (path<TAB>bound per line)")?;
    let list = std::fs::read_to_string(&list_path).map_err(|e| format!("{list_path}: {e}"))?;
    let mut kernels = Vec::new();
    for line in list.lines().filter(|l| !l.trim().is_empty()) {
        let (path, bound) = line.split_once('\t').ok_or_else(|| format!("{list_path}: not path<TAB>bound: {line}"))?;
        let bound: u128 = bound.trim().parse().map_err(|e| format!("{list_path}: bound {bound}: {e}"))?;
        kernels.push((path.to_string(), std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?, bound));
    }
    let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/volumepassage.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| e.to_string())?;
    let dev = Device::new()?;
    let size = 64u32;
    let b = case_bindings("volumepassage", 5.0, size, size)?;
    let shipped = ImageKernel::compile(&dev, "volumepassage", &original, "main", &b, size, size, Fuel::Proven("cuneus original volumepassage"))?;
    println!("# {} {} kernels, {calls} calls of {pairs} pairs each, at {size}x{size} t 5 against the shipped volumepassage", utc(), kernels.len());
    let (mut total, mut overlaps, mut other) = (0usize, 0usize, 0usize);
    for (path, src, bound) in &kernels {
        let cand = ImageKernel::compile(&dev, "candidate", src, "main", &b, size, size, Fuel::Tree { per_invocation: *bound })?;
        let planned = if shipped.plan() == cand.plan() { None } else { Some(shipped.with_plan(&b, &cand.plan())?) };
        let reference = planned.as_ref().unwrap_or(&shipped);
        let (mut ok, mut ratios, mut c_ms, mut s_ms) = (0usize, Vec::new(), Vec::new(), Vec::new());
        for _ in 0..calls {
            total += 1;
            match timed_pairs(&cand, reference, 1, pairs) {
                Ok(pt) => {
                    ok += 1;
                    ratios.extend(pt.times.iter().map(|t| t.0 / t.1));
                    c_ms.extend(pt.times.iter().map(|t| t.0));
                    s_ms.extend(pt.times.iter().map(|t| t.1));
                }
                Err(e) => {
                    if e.contains("overlapped") {
                        overlaps += 1;
                    } else {
                        other += 1;
                    }
                    println!("FAIL\t{path}\t{e}");
                }
            }
        }
        let med = |v: &[f64]| if v.is_empty() { f64::NAN } else { median(v) };
        println!("{path}\tbound {bound}\t{} submit(s)\t{ok} of {calls} calls timed\tcandidate / shipped median {:.4}\tcandidate {:.4} ms\tshipped {:.4} ms", cand.submits(), med(&ratios), med(&c_ms), med(&s_ms));
    }
    println!("# {} calls {total}\toverlaps {overlaps}\tother failures {other}", utc());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
