//! ST9 timing measurement: the spread of per-pair GPU-time ratios, encoded volumepassage over the shipped original, at
//! the timing size (64 x 64, t 5, one submit), timed two ways over the same number of pairs:
//!
//! * BEFORE: each pass its own submit (the reference's, then the candidate's), as `timed_pairs` did to 2026-10-22;
//! * AFTER: `image::timed_pairs`, one command buffer a pair, warm-up of the reference first, the order alternating.
//!
//!   cargo run --release --features gpu,wgsl --example st9_pair_spread -- [PAIRS]
use fuller::wgsl::cuneus_originals::{case_bindings, image_dir, volumepassage};
use fuller::wgsl::image::{timed_pairs, ImageKernel};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::TrueKingdom;

fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

fn report(what: &str, r: &[f64], c: &[f64], s: &[f64]) {
    let (p10, p50, p90) = (quantile(r, 0.1), quantile(r, 0.5), quantile(r, 0.9));
    println!(
        "{what}\tpairs {}\tratio p10 {p10:.4} p50 {p50:.4} p90 {p90:.4} (p90 - p10 {:.4})\tmin {:.4} max {:.4}\tencoded ms p10 {:.4} p90 {:.4}\tshipped ms p10 {:.4} p90 {:.4}",
        r.len(),
        p90 - p10,
        quantile(r, 0.0),
        quantile(r, 1.0),
        quantile(c, 0.1),
        quantile(c, 0.9),
        quantile(s, 0.1),
        quantile(s, 0.9)
    );
}

fn run() -> Result<(), String> {
    let pairs: usize = std::env::args().nth(1).map(|a| a.parse().map_err(|e| format!("PAIRS: {e}"))).transpose()?.unwrap_or(50);
    let k = TrueKingdom::load(&image_dir("volumepassage"))?;
    let tree = volumepassage(&k).map_err(|g| format!("table gaps: {g:?}"))?;
    let src = k.render(&tree)?;
    let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/volumepassage.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| e.to_string())?;
    let dev = Device::new()?;
    let size = 64u32;
    let bindings = case_bindings("volumepassage", 5.0, size, size)?;
    let shipped = ImageKernel::compile(&dev, "volumepassage", &original, "main", &bindings, size, size, Fuel::Proven("cuneus original volumepassage"))?;
    let encoded = ImageKernel::compile(&dev, "encoded", &src, "main", &bindings, size, size, k.tree_fuel(&tree)?)?;
    if shipped.plan() != encoded.plan() || encoded.submits() != 1 {
        return Err(format!("our software has a bug: the timing size is not one submit for both ({} and {})", encoded.submits(), shipped.submits()));
    }
    let utc = || std::process::Command::new("date").args(["-u", "+%H:%M:%SZ"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    println!("# {} {size}x{size} t 5, encoded / shipped, {pairs} pairs each way", utc());
    // BEFORE: the reference's pass, then the candidate's, each in a submit of its own
    shipped.passes(1, 0)?;
    encoded.passes(1, 0)?;
    let (mut r, mut c, mut s) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..pairs {
        let sm = shipped.passes(0, 1)?.1[0];
        let cm = encoded.passes(0, 1)?.1[0];
        r.push(cm / sm);
        c.push(cm);
        s.push(sm);
    }
    report("before (a submit a pass)", &r, &c, &s);
    println!("# {}", utc());
    // AFTER: one command buffer a pair
    let pt = timed_pairs(&encoded, &shipped, 1, pairs)?;
    let r: Vec<f64> = pt.times.iter().map(|t| t.0 / t.1).collect();
    let c: Vec<f64> = pt.times.iter().map(|t| t.0).collect();
    let s: Vec<f64> = pt.times.iter().map(|t| t.1).collect();
    report("after (one command buffer a pair)", &r, &c, &s);
    let (cf, rf): (Vec<f64>, Vec<f64>) = (r.iter().step_by(2).copied().collect(), r.iter().skip(1).step_by(2).copied().collect());
    println!("after, by order\tcandidate first median {:.4}\treference first median {:.4}", quantile(&cf, 0.5), quantile(&rf, 0.5));
    println!("# {}", utc());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
