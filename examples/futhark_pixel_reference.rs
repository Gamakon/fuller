//! The Futhark route to a cuneus shader, step 3: the shader (`--shader volumepassage|cuneus`) written by hand in the
//! pixel family's rows (`fuller::futhark_pixel::instance`, the reference only), compiled by the branch compiler's webgpu
//! backend, run on our wgpu/Metal path through the shim (`futhark_wgpu_host`), and its picture compared with the
//! WGSL original's, drawn headless by the image harness at the same uniforms. Futhark's C backend draws it too, so the
//! difference can be put on the device (fast math) or on the program.
//!
//!   cargo run --release --features gpu,wgsl --example futhark_pixel_reference -- --shader cuneus --sizes 96x48,512x256 \
//!       --times 2,5,8 //!       --out /private/tmp/claude-501/fpix/reference
//!
//! The device runs only under the caller's GPU lock (one process). The original runs first and its device is dropped
//! before the Futhark programs run in their own processes; every Futhark band is bounded by the per-submit cap.

use std::path::PathBuf;

use fuller::futhark_pixel::{build, compile_c, draw, inputs, instance, interface, load, program, tiles, work_per_pixel, Tile};
use fuller::futhark_wgpu_host::compile_program;
use fuller::wgsl::cuneus_originals::case_bindings;
use fuller::wgsl::image::{compare, ocr, ocr_word, side_by_side, upscaled, write_png, BindingData, Difference, ImageKernel, Picture};
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
    let shader = arg(&args, "--shader").unwrap_or_else(|| "volumepassage".into());
    let sizes: Vec<(u32, u32)> = arg(&args, "--sizes")
        .unwrap_or_else(|| "64x64,128x128".into())
        .split(',')
        .map(|s| {
            let (w, h) = s.split_once('x').ok_or_else(|| format!("{s} is not WxH"))?;
            Ok((w.parse::<u32>().map_err(|e| e.to_string())?, h.parse::<u32>().map_err(|e| e.to_string())?))
        })
        .collect::<Result<_, String>>()?;
    let times: Vec<f32> = list(&arg(&args, "--times").unwrap_or_else(|| "2,5,8".into()))?;
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "/private/tmp/claude-501/fpix/reference".into()));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    println!("# {} futhark_pixel_reference: {shader}, sizes {sizes:?}, times {times:?}", utc());

    // 1. The WGSL original, drawn by the image harness; the device is dropped before any Futhark program runs.
    let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/{shader}.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| format!("original: {e}"))?;
    let mut originals = Vec::new();
    {
        let dev = Device::new()?;
        println!("# device {}", dev.name);
        for &(width, height) in &sizes {
            for &t in &times {
                let b = case_bindings(&shader, t, width, height)?;
                let k = ImageKernel::compile(&dev, &shader, &original, "main", &b, width, height, Fuel::Proven("a cuneus original"))?;
                let run = k.run(2, 40)?;
                println!("original {width}x{height} t={t}: gpu ms fastest of 40 {:.4}, median {:.4}", run.gpu_ms.iter().cloned().fold(f64::INFINITY, f64::min), run.median_gpu_ms());
                originals.push((width, height, t, run.picture));
            }
        }
    }

    // 2. The reference as a gene's tree, its program, and both compilers.
    let (inst, reference) = instance(&shader)?;
    let k = load(&inst)?;
    let iface = interface(&inst)?;
    let tree = build(&k, reference)?;
    let source = program(&k, &iface, &tree)?;
    std::fs::write(out.join("reference.fut"), &source).map_err(|e| e.to_string())?;
    let work = work_per_pixel(&k, &tree)?;
    println!("reference: {} nodes, {work} node executions a pixel", tree.level_order().len());
    let t0 = std::time::Instant::now();
    let gpu_exe = compile_program(&source, &out.join("gpu"))?;
    println!("{} branch compiler (webgpu) + shim link: {:.1} s", utc(), t0.elapsed().as_secs_f64());
    let c_exe = compile_c(&source, &out.join("c"))?;

    let mut tsv = String::from("size\ttime\tpath\tdiffering_channels\tchannels\tmax_abs\tmean_abs\trmse\tpixels_over_1_255\tnon_finite\tgpu_ms\n");
    for (width, height, t, want) in &originals {
        let (width, height) = (*width, *height);
        let b = case_bindings(&shader, *t, width, height)?;
        let bl = blocks(&b);
        let ps_of = |t: Tile| inputs(&iface, &bl, width, height, t);
        let plan = tiles(work, width, height, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT)?;
        let (gpu_rgba, ns) = draw(&gpu_exe, true, &ps_of, &plan, width, height, 40)?;
        let (c_rgba, _) = draw(&c_exe, false, &ps_of, &plan, width, height, 1)?;
        let gpu_pic = Picture { width, height, rgba: gpu_rgba };
        let c_pic = Picture { width, height, rgba: c_rgba };
        for (path, pic) in [("futhark webgpu (shim, Metal) vs WGSL original", &gpu_pic), ("futhark c vs WGSL original", &c_pic), ("futhark webgpu vs futhark c", &gpu_pic)] {
            let d = if path.ends_with("futhark c") { compare(pic, &c_pic)? } else { compare(pic, want)? };
            println!("{width}x{height} t={t} {path}: {}", stats(&d));
            tsv.push_str(&format!(
                "{width}x{height}\t{t}\t{path}\t{}\t{}\t{:e}\t{:e}\t{:e}\t{}\t{}\t{:.4}\n",
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
        println!("{width}x{height} t={t}: {} tile(s), GPU {:.4} ms (the fastest of 40 runs, summed over tiles)", plan.len(), ns / 1e6);
        write_png(&out.join(format!("original_vs_futhark_{width}x{height}_t{t}.png")), &side_by_side(&[want.clone(), gpu_pic.clone(), c_pic.clone()])?)?;
        if shader == "cuneus" {
            for (name, pic) in [("original", want), ("reference_gpu", &gpu_pic)] {
                let path = out.join(format!("ocr_{name}_{width}x{height}_t{t}.png"));
                write_png(&path, &upscaled(pic, 512))?;
                let raw = ocr(&path, "CUNEUS", 7)?;
                let prepared = ocr_word(pic, &out.join(format!("ocr_prepared_{name}_{width}x{height}_t{t}.png")), "CUNEUS")?;
                for (how, o) in [("raw line", &raw), ("thresholded word", &prepared)] {
                    println!("{width}x{height} t={t} OCR {name} ({how}): {:?} (confidence {:.1}), equals CUNEUS: {}, edit distance {}", o.text, o.confidence, o.matches, o.distance);
                    tsv.push_str(&format!("{width}x{height}\t{t}\tOCR {name} {how}\t{:?}\t{:.1}\t{}\t{}\n", o.text, o.confidence, o.matches, o.distance));
                }
            }
        }
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
