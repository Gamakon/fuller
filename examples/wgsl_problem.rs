//! Score kernels against a problem directory with the generic harness
//! (`fuller::wgsl::harness`) on the device, one TSV line a kernel on
//! stdout: feasibility, cases passed, the distance, the closeness angle,
//! the HFF angle, the detail. With `--compare-gram` (the Gram problem only)
//! each kernel is scored by `gram::FitnessContext::fitness` too and the
//! line says whether the two agree on feasibility, cases passed and every
//! closeness column and the accuracy columns, bit for bit: the device half of the harness's
//! acceptance (the host half is `harness::tests`). Where they do not, the
//! line names the first differing field and both values. With
//! `--self-check` each implementation also scores every kernel a second
//! time, and the line says whether each agrees with itself: a kernel whose
//! output varies from run to run (a read-modify-write race between
//! invocations) disagrees with itself, and no comparison of two
//! implementations can agree on it bit for bit.
//!
//! A watchdog ends the process, naming the kernel, if one evaluation runs
//! past `--deadline` seconds (120 by default): no call can cancel GPU work.
//!
//! `cargo run --release --features gpu,wgsl --example wgsl_problem -- --problem problems/gram [--compare-gram [--self-check]] [--deadline 120] <kernel.wgsl or dir>...`

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use fuller::wgsl::closeness;
use fuller::wgsl::gram::{Fitness, FitnessContext};
use fuller::wgsl::harness::{DeviceExecutor, Harness, Problem};
use fuller::wgsl::oracle::device::Device;

static DEADLINE: AtomicU64 = AtomicU64::new(0);

/// The first field in which two fitnesses differ, with both values; `None`
/// when they agree bit for bit. Compared: feasibility, cases passed, the
/// distance, every closeness column and its angle, the three accuracy
/// columns. Left out: what a device measures in time (the speed columns,
/// the HFF angle over them, the detail of a timing gate, seconds, split).
fn first_difference(a: &Fitness, b: &Fitness) -> Option<String> {
    if a.feasible != b.feasible {
        return Some(format!("feasible {} / {}", a.feasible, b.feasible));
    }
    if a.passed != b.passed {
        return Some(format!("passed {} / {}", a.passed, b.passed));
    }
    let mut fields: Vec<(String, f64, f64)> = vec![("distance".into(), a.distance, b.distance)];
    fields.extend(closeness::COLUMNS.iter().enumerate().map(|(i, n)| (format!("closeness.{n}"), a.closeness[i], b.closeness[i])));
    fields.push(("closeness_hff".into(), a.closeness_hff, b.closeness_hff));
    fields.extend((3..6).map(|i| (format!("columns[{i}]"), a.columns[i], b.columns[i])));
    fields.into_iter().find(|(_, x, y)| x.to_bits() != y.to_bits()).map(|(n, x, y)| format!("{n} {x:.17e} / {y:.17e}"))
}

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
    let self_check = args.iter().position(|a| a == "--self-check").map(|i| args.remove(i)).is_some();
    if self_check && !compare {
        return Err("--self-check needs --compare-gram".into());
    }
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
    println!("kernel\tfeasible\tpassed\tdistance\tcloseness_hff\thff\tdetail{}{}", if compare { "\tgram_feasible\tgram_passed\tagree\tfirst_difference (harness / gram)" } else { "" }, if self_check { "\tgram_vs_gram\tharness_vs_harness" } else { "" });
    let (mut agree, mut n, mut gram_self, mut harness_self) = (0usize, 0usize, 0usize, 0usize);
    for k in &kernels {
        let text = std::fs::read_to_string(k).map_err(|e| format!("{}: {e}", k.display()))?;
        *current.lock().map_err(|e| e.to_string())? = k.display().to_string();
        DEADLINE.store(start.elapsed().as_millis() as u64 + deadline_s * 1000, Ordering::SeqCst);
        let f = harness.evaluate(&text);
        let g = ctx.as_ref().map(|c| c.fitness(&dev, &text));
        let again = if self_check { ctx.as_ref().map(|c| (harness.evaluate(&text), c.fitness(&dev, &text))) } else { None };
        DEADLINE.store(0, Ordering::SeqCst);
        let mut line = format!("{}\t{}\t{}\t{:.3e}\t{:.4}\t{:.4}\t{}", k.display(), f.feasible, f.passed, f.distance, f.closeness_hff, f.hff, f.detail.replace(['\t', '\n'], " "));
        if let Some(g) = &g {
            let diff = first_difference(&f, g);
            agree += usize::from(diff.is_none());
            line += &format!("\t{}\t{}\t{}\t{}", g.feasible, g.passed, diff.is_none(), diff.unwrap_or_default());
            if let Some((f2, g2)) = &again {
                let (gd, hd) = (first_difference(g, g2), first_difference(&f, f2));
                gram_self += usize::from(gd.is_none());
                harness_self += usize::from(hd.is_none());
                line += &format!("\t{}\t{}", gd.unwrap_or_else(|| "same".into()), hd.unwrap_or_else(|| "same".into()));
            }
        }
        n += 1;
        println!("{line}");
    }
    if compare {
        eprintln!("{agree} of {n} kernels: the harness's fitness equals gram.rs's bit for bit (verdict, distance, closeness, accuracy columns)");
    }
    if self_check {
        eprintln!("{gram_self} of {n} kernels: gram.rs's fitness equals its own second run; {harness_self} of {n}: the harness's equals its own");
    }
    Ok(())
}
