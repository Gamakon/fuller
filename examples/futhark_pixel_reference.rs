//! The Futhark route to a cuneus shader, step 3: volumepassage written by hand in the pixel family's rows
//! (`fuller::futhark_pixel::VOLUMEPASSAGE_REFERENCE`, the reference only), compiled by the branch compiler's webgpu
//! backend, run on our wgpu/Metal path through the shim (`futhark_wgpu_host`), and its picture compared with the
//! WGSL original's, drawn headless by the image harness at the same uniforms. Futhark's C backend draws it too, so the
//! difference can be put on the device (fast math) or on the program.
//!
//!   cargo run --release --features gpu,wgsl --example futhark_pixel_reference -- --sizes 64,128 --times 2,5,8 \
//!       --out /private/tmp/claude-501/fpix/reference
//!
//! The device runs only under the caller's GPU lock (one process). The original runs first and its device is dropped
//! before the Futhark programs run in their own processes; every Futhark band is bounded by the per-submit cap.

use std::path::PathBuf;

use fuller::futhark_pixel::{build, compile_c, draw, inputs, interface, load, program, tiles, work_per_pixel, Tile, VOLUMEPASSAGE, VOLUMEPASSAGE_REFERENCE};
use fuller::futhark_wgpu_host::compile_program;
use fuller::wgsl::cuneus_originals::case_bindings;
use fuller::wgsl::image::{compare, side_by_side, write_png, BindingData, Difference, ImageKernel, Picture};
use fuller::wgsl::oracle::device::{Device, Fuel, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>, String> {
    s.split(',').map(|x| x.parse::<T>().map_err(|_| format!("{x} is not a number"))).collect()
}

fn blocks(b: &[BindingData]) -> Vec<(u32, u32, &[u8])> {
    b.iter()
        .filter_map(|d| match d {
            BindingData::Uniform { group, binding, bytes } => Some((*group, *binding, bytes.as_slice())),
            _ => None,
        })
        .collect()
}

fn stats(d: &Difference) -> String {
    format!(
        "differing channels {} of {}, max |d| {:.3e}, mean |d| {:.3e}, rmse {:.3e}, pixels over 1/255 {}, non-finite mismatches {}",
        d.differing_channels,
        d.pixels * 4,
        d.max_abs,
        d.mean_abs,
        d.rmse,
        d.pixels_over_one_step,
        d.non_finite_mismatch
    )
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let sizes: Vec<u32> = list(&arg(&args, "--sizes").unwrap_or_else(|| "64,128".into()))?;
    let times: Vec<f32> = list(&arg(&args, "--times").unwrap_or_else(|| "2,5,8".into()))?;
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "/private/tmp/claude-501/fpix/reference".into()));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    println!("# {} futhark_pixel_reference: sizes {sizes:?}, times {times:?}", utc());

    // 1. The WGSL original, drawn by the image harness; the device is dropped before any Futhark program runs.
    let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/volumepassage.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| format!("original: {e}"))?;
    let mut originals = Vec::new();
    {
        let dev = Device::new()?;
        println!("# device {}", dev.name);
        for &size in &sizes {
            for &t in &times {
                let b = case_bindings("volumepassage", t, size, size)?;
                let k = ImageKernel::compile(&dev, "volumepassage", &original, "main", &b, size, size, Fuel::Proven("cuneus original volumepassage"))?;
                let run = k.run(2, 40)?;
                println!("original {size}x{size} t={t}: gpu ms fastest of 40 {:.4}, median {:.4}", run.gpu_ms.iter().cloned().fold(f64::INFINITY, f64::min), run.median_gpu_ms());
                originals.push((size, t, run.picture));
            }
        }
    }

    // 2. The reference as a gene's tree, its program, and both compilers.
    let k = load(&VOLUMEPASSAGE)?;
    let iface = interface(&VOLUMEPASSAGE)?;
    let tree = build(&k, VOLUMEPASSAGE_REFERENCE)?;
    let source = program(&k, &iface, &tree)?;
    std::fs::write(out.join("reference.fut"), &source).map_err(|e| e.to_string())?;
    let work = work_per_pixel(&k, &tree)?;
    println!("reference: {} nodes, {work} node executions a pixel", tree.level_order().len());
    let t0 = std::time::Instant::now();
    let gpu_exe = compile_program(&source, &out.join("gpu"))?;
    println!("{} branch compiler (webgpu) + shim link: {:.1} s", utc(), t0.elapsed().as_secs_f64());
    let c_exe = compile_c(&source, &out.join("c"))?;

    let mut tsv = String::from("size\ttime\tpath\tdiffering_channels\tchannels\tmax_abs\tmean_abs\trmse\tpixels_over_1_255\tnon_finite\tgpu_ms\n");
    for (size, t, want) in &originals {
        let b = case_bindings("volumepassage", *t, *size, *size)?;
        let bl = blocks(&b);
        let ps_of = |t: Tile| inputs(&iface, &bl, *size, *size, t);
        let plan = tiles(work, *size, *size, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
        let (gpu_rgba, ns) = draw(&gpu_exe, true, &ps_of, &plan, *size, *size, 40)?;
        let (c_rgba, _) = draw(&c_exe, false, &ps_of, &plan, *size, *size, 1)?;
        let gpu_pic = Picture { width: *size, height: *size, rgba: gpu_rgba };
        let c_pic = Picture { width: *size, height: *size, rgba: c_rgba };
        for (path, pic) in [("futhark webgpu (shim, Metal) vs WGSL original", &gpu_pic), ("futhark c vs WGSL original", &c_pic), ("futhark webgpu vs futhark c", &gpu_pic)] {
            let d = if path.ends_with("futhark c") { compare(pic, &c_pic)? } else { compare(pic, want)? };
            println!("{size}x{size} t={t} {path}: {}", stats(&d));
            tsv.push_str(&format!(
                "{size}\t{t}\t{path}\t{}\t{}\t{:e}\t{:e}\t{:e}\t{}\t{}\t{:.4}\n",
                d.differing_channels,
                d.pixels * 4,
                d.max_abs,
                d.mean_abs,
                d.rmse,
                d.pixels_over_one_step,
                d.non_finite_mismatch,
                if path.starts_with("futhark webgpu (") { ns / 1e6 } else { 0.0 }
            ));
        }
        println!("{size}x{size} t={t}: {} tile(s), GPU {:.4} ms (the fastest of 40 runs, summed over tiles)", plan.len(), ns / 1e6);
        write_png(&out.join(format!("original_vs_futhark_{size}_t{t}.png")), &side_by_side(&[want.clone(), gpu_pic.clone(), c_pic.clone()])?)?;
    }
    std::fs::write(out.join("difference.tsv"), tsv).map_err(|e| e.to_string())?;
    println!("# {} done: {}", utc(), out.display());
    Ok(())
}

/// UTC wall time `HH:MM:SSZ` from the system clock.
fn utc() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) % 86_400;
    format!("{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
