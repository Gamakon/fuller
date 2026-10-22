//! ST11 planted-win test, step 1: does a kernel draw the shipped volumepassage's picture, and how much slower is it?
//!
//!   cargo run --release --features gpu,wgsl --example st11_planted_time -- --candidate FILE [--pairs 41]
//!   cargo run --release --features gpu,wgsl --example st11_planted_time -- --tree planted|original [--pairs 41]
//!
//! The candidate is a WGSL text (`--candidate`, entry `main`, the shipped bindings), or the encoded tree of the true
//! kingdom's image instance (`--tree`: `cuneus_originals::volumepassage` or `volumepassage_planted`, rendered as the run
//! renders it). Its picture is compared with the SHIPPED original's (`tests/corpus/cuneus_visual/volumepassage.wgsl`)
//! at the run's cases (256 x 256 at t 2, 5 and 8, 200 x 136 at t 5) and at the timing size; then `--pairs` timed pairs
//! (`image::timed_pairs`, one command buffer a pair) at 64 x 64, t 5, candidate against the shipped original under the candidate's submit plan.
use fuller::wgsl::cuneus_originals::{
    case_bindings, image_dir, volumepassage, volumepassage_planted,
};
use fuller::wgsl::image::{compare, median, timed_pairs, ImageKernel};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::TrueKingdom;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

fn utc() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (t / 3600) % 24, (t / 60) % 60, t % 60)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let pairs: usize = arg(&args, "--pairs")
        .map(|a| a.parse().map_err(|e| format!("--pairs: {e}")))
        .transpose()?
        .unwrap_or(41);
    let original = std::fs::read_to_string(format!(
        "{}/tests/corpus/cuneus_visual/volumepassage.wgsl",
        env!("CARGO_MANIFEST_DIR")
    ))
    .map_err(|e| e.to_string())?;
    let k = TrueKingdom::load(&image_dir("volumepassage"))?;
    let (what, src, fuel) = match (arg(&args, "--candidate"), arg(&args, "--tree")) {
        (Some(path), None) => (
            path.clone(),
            std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?,
            Fuel::Proven("a hand-written volumepassage variant: the original's loops"),
        ),
        (None, Some(which)) => {
            let tree = match which.as_str() {
                "original" => volumepassage(&k),
                "planted" => volumepassage_planted(&k),
                o => return Err(format!("--tree {o}: original or planted")),
            }
            .map_err(|g| format!("table gaps: {g:?}"))?;
            (
                format!("encoded {which}"),
                k.render(&tree)?,
                k.tree_fuel(&tree)?,
            )
        }
        _ => return Err("give one of --candidate FILE or --tree original|planted".into()),
    };
    let dev = Device::new()?;
    println!("# {} {what} against the shipped volumepassage", utc());
    for (time, w, h) in [
        (5.0f32, 256u32, 256u32),
        (8.0, 256, 256),
        (2.0, 256, 256),
        (5.0, 200, 136),
        (5.0, 64, 64),
    ] {
        let b = case_bindings("volumepassage", time, w, h)?;
        let shipped = ImageKernel::compile(
            &dev,
            "volumepassage",
            &original,
            "main",
            &b,
            w,
            h,
            Fuel::Proven("cuneus original volumepassage"),
        )?
        .run(0, 1)?;
        let cand = ImageKernel::compile(&dev, "candidate", &src, "main", &b, w, h, fuel.clone())?
            .run(0, 1)?;
        let d = compare(&cand.picture, &shipped.picture)?;
        println!(
            "picture t {time} {w}x{h}\t{} differing channels of {}",
            d.differing_channels,
            d.pixels * 4
        );
    }
    let size = 64u32;
    let b = case_bindings("volumepassage", 5.0, size, size)?;
    let shipped = ImageKernel::compile(
        &dev,
        "volumepassage",
        &original,
        "main",
        &b,
        size,
        size,
        Fuel::Proven("cuneus original volumepassage"),
    )?;
    let cand = ImageKernel::compile(&dev, "candidate", &src, "main", &b, size, size, fuel)?;
    // the shipped original takes the candidate's submit plan, as in the run's timed pairs
    let planned = if shipped.plan() == cand.plan() { None } else { Some(shipped.with_plan(&b, &cand.plan())?) };
    let reference = planned.as_ref().unwrap_or(&shipped);
    let pt = timed_pairs(&cand, reference, 3, pairs)?;
    let r: Vec<f64> = pt.times.iter().map(|t| t.0 / t.1).collect();
    let c: Vec<f64> = pt.times.iter().map(|t| t.0).collect();
    let s: Vec<f64> = pt.times.iter().map(|t| t.1).collect();
    println!(
        "timing {size}x{size} t 5\t{pairs} pairs, {} submit(s) each\tcandidate / shipped median {:.4} (p10 {:.4}, p90 {:.4})\tcandidate median {:.4} ms\tshipped median {:.4} ms",
        pt.submits,
        median(&r),
        quantile(&r, 0.1),
        quantile(&r, 0.9),
        median(&c),
        median(&s)
    );
    println!("# {}", utc());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
