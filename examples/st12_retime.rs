//! ST12 step 3: re-time a run's champion and the encoded original against the SHIPPED volumepassage in one session.
//!
//!   cargo run --release --features gpu,wgsl --example st12_retime -- --candidate FILE --bound N [--rounds 5] [--pairs 41]
//!
//! `--candidate` is the champion's kernel text (a run's `hall_of_fame/g<gen>_rank1.wgsl`) and `--bound` its work bound an
//! invocation as the run's ledger records it (the work guard holds the run's own fuel). Pictures first: the champion and
//! the encoded original against the shipped original at the run's cases (256 x 256 at t 2, 5, 8; 200 x 136 at t 5) and
//! at 64 x 64, t 5. Then `--rounds` rounds, interleaved (even rounds the champion first, odd rounds the encoded original
//! first), each `--pairs` timed pairs (`image::timed_pairs`: one command buffer a pair, the order alternating by pair, every
//! pass fenced and its order proved) of that kernel against the shipped original under the kernel's submit plan, at
//! 64 x 64, t 5. Per kernel: the median of all its pair ratios (kernel / shipped) with a 95 % bootstrap interval of the
//! median (10,000 resamples, a fixed seed), the median of each round, and the median milliseconds.
use fuller::wgsl::cuneus_originals::{case_bindings, image_dir, volumepassage};
use fuller::wgsl::image::{compare, median, timed_pairs, ImageKernel};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::{Rng, TrueKingdom};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn utc() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (t / 3600) % 24, (t / 60) % 60, t % 60)
}

/// The 95 % bootstrap interval of the median of `v` (10,000 resamples).
fn bootstrap_median(v: &[f64], rng: &mut Rng) -> (f64, f64) {
    let mut meds: Vec<f64> = (0..10_000)
        .map(|_| {
            let s: Vec<f64> = (0..v.len()).map(|_| v[(rng.next_u64() % v.len() as u64) as usize]).collect();
            median(&s)
        })
        .collect();
    meds.sort_by(f64::total_cmp);
    (meds[249], meds[9_749])
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let number = |name: &str, default: usize| -> Result<usize, String> { arg(&args, name).map(|a| a.parse().map_err(|e| format!("{name}: {e}"))).transpose().map(|v| v.unwrap_or(default)) };
    let (rounds, pairs) = (number("--rounds", 5)?, number("--pairs", 41)?);
    let path = arg(&args, "--candidate").ok_or("--candidate FILE")?;
    let bound: u128 = arg(&args, "--bound").ok_or("--bound N (the run's ledger)")?.parse().map_err(|e| format!("--bound: {e}"))?;
    let champion = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/volumepassage.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| e.to_string())?;
    let k = TrueKingdom::load(&image_dir("volumepassage"))?;
    let tree = volumepassage(&k).map_err(|g| format!("table gaps: {g:?}"))?;
    // the control: the shipped text compiled a second time, timed against the shipped original like the others (its
    // ratio is the harness's own bias)
    let kernels = [("champion", champion, Fuel::Tree { per_invocation: bound }), ("encoded original", k.render(&tree)?, k.tree_fuel(&tree)?), ("shipped (control)", original.clone(), Fuel::Proven("cuneus original volumepassage"))];
    let dev = Device::new()?;
    println!("# {} ST12 re-time: {path} (bound {bound}) and the encoded original against the shipped volumepassage", utc());
    for (time, w, h) in [(2.0f32, 256u32, 256u32), (5.0, 256, 256), (8.0, 256, 256), (5.0, 200, 136), (5.0, 64, 64)] {
        let b = case_bindings("volumepassage", time, w, h)?;
        let shipped = ImageKernel::compile(&dev, "shipped", &original, "main", &b, w, h, Fuel::Proven("cuneus original volumepassage"))?.run(0, 1)?;
        for (name, src, fuel) in &kernels {
            let got = ImageKernel::compile(&dev, name, src, "main", &b, w, h, fuel.clone())?.run(0, 1)?;
            let d = compare(&got.picture, &shipped.picture)?;
            println!("picture\t{name}\tt {time} {w}x{h}\t{} differing channels of {}", d.differing_channels, d.pixels * 4);
        }
    }
    let size = 64u32;
    let b = case_bindings("volumepassage", 5.0, size, size)?;
    let shipped = ImageKernel::compile(&dev, "shipped", &original, "main", &b, size, size, Fuel::Proven("cuneus original volumepassage"))?;
    let mut compiled = Vec::new();
    for (name, src, fuel) in &kernels {
        let kk = ImageKernel::compile(&dev, name, src, "main", &b, size, size, fuel.clone())?;
        let planned = if shipped.plan() == kk.plan() { None } else { Some(shipped.with_plan(&b, &kk.plan())?) };
        compiled.push((*name, kk, planned));
    }
    // ratios[kernel] over all rounds, and each round's median
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); compiled.len()];
    let mut ms: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); compiled.len()];
    let mut round_medians: Vec<Vec<f64>> = vec![Vec::new(); compiled.len()];
    // ratios by order: (kernel first, shipped first)
    let mut by_order: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); compiled.len()];
    for r in 0..rounds {
        let order: Vec<usize> = if r % 2 == 0 { (0..compiled.len()).collect() } else { (0..compiled.len()).rev().collect() };
        for i in order {
            let (_, kk, planned) = &compiled[i];
            let pt = timed_pairs(kk, planned.as_ref().unwrap_or(&shipped), 3, pairs)?;
            let rr: Vec<f64> = pt.times.iter().map(|t| t.0 / t.1).collect();
            round_medians[i].push(median(&rr));
            for (x, first) in rr.iter().zip(&pt.candidate_first) {
                if *first { by_order[i].0.push(*x) } else { by_order[i].1.push(*x) }
            }
            ratios[i].extend(rr);
            ms[i].0.extend(pt.times.iter().map(|t| t.0));
            ms[i].1.extend(pt.times.iter().map(|t| t.1));
        }
    }
    let mut rng = Rng(12);
    println!("kernel\tsubmits\tpairs\tkernel / shipped median\t95 % interval\tround medians\tkernel first\tshipped first\tkernel ms\tshipped ms");
    for (i, (name, kk, _)) in compiled.iter().enumerate() {
        let (lo, hi) = bootstrap_median(&ratios[i], &mut rng);
        let rm: Vec<String> = round_medians[i].iter().map(|v| format!("{v:.4}")).collect();
        println!("{name}\t{}\t{}\t{:.4}\t[{lo:.4}, {hi:.4}]\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}", kk.submits(), ratios[i].len(), median(&ratios[i]), rm.join(" "), median(&by_order[i].0), median(&by_order[i].1), median(&ms[i].0), median(&ms[i].1));
    }
    println!("# {}", utc());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
