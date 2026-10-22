//! ST9 phase 1: draw a cuneus visual shader headless, time it, and write it as a gene of the true kingdom.
//!
//!   cargo run --release --features gpu,wgsl --example cuneus_image -- --shader volumepassage --size 512 --time 5.0 \
//!       --repeats 21 --out /private/tmp/claude-501/st9
//!
//! 1. The original text runs at fixed uniforms and resolution (`Fuel::Proven("cuneus <name> original")`); the picture is
//!    read back, saved as `<out>/<name>_original.png`, and the dispatch is timed (median of `--repeats` after warmup).
//! 2. The original is written as a tree of the rows of `kingdoms/wgsl_true/img/<name>` (`cuneus_originals`), written into
//!    a gene with the section 5 write-back, expressed again (the tree must come back unchanged), rendered by substitution,
//!    validated by naga, compiled and run (`Fuel::Proven` for this hand-built tree: its loops are the original's, 50 x 7),
//!    and the picture is compared with the original's.
//!
//! The device runs only under the caller's lock and a load check; one process.

use std::path::PathBuf;

use fuller::typed_gep::{express, write_back, GeneLayout, Tree};
use fuller::wgsl::cuneus_originals::{case_bindings, image_dir, nebula, volumepassage};
use fuller::wgsl::image::{compare, write_png, ImageKernel};
use fuller::wgsl::oracle::device::{Device, Fuel};
use fuller::wgsl::true_kingdom::{random_gene, Rng, Share, TrueKingdom};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn functions(t: &Tree, k: &TrueKingdom) -> usize {
    usize::from(!k.rows[t.symbol as usize].args.is_empty()) + t.kids.iter().map(|c| functions(c, k)).sum::<usize>()
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let shader = arg(&args, "--shader").unwrap_or_else(|| "volumepassage".into());
    let size: u32 = arg(&args, "--size").map_or(Ok(512), |v| v.parse().map_err(|e| format!("--size: {e}")))?;
    let time: f32 = arg(&args, "--time").map_or(Ok(5.0), |v| v.parse().map_err(|e| format!("--time: {e}")))?;
    let repeats: usize = arg(&args, "--repeats").map_or(Ok(21), |v| v.parse().map_err(|e| format!("--repeats: {e}")))?;
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "/private/tmp/claude-501/st9".into()));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let original = std::fs::read_to_string(format!("{}/tests/corpus/cuneus_visual/{shader}.wgsl", env!("CARGO_MANIFEST_DIR"))).map_err(|e| format!("original: {e}"))?;
    let entry = if shader == "nebula" { "main_image" } else { "main" };
    let bindings = case_bindings(&shader, time, size, size)?;
    let dev = Device::new()?;
    println!("# device {}, {size}x{size}, time {time}, {repeats} timed passes", dev.name);

    let k = ImageKernel::compile(&dev, &shader, &original, entry, &bindings, size, size, Fuel::Proven("cuneus original (volumepassage or nebula)"))?;
    let (gx, gy, inv) = k.dispatch();
    let run = k.run(3, repeats)?;
    write_png(&out.join(format!("{shader}_original.png")), &run.picture)?;
    println!("original: dispatch {gx}x{gy} workgroups, {inv} invocations; median wall {:.3} ms, median gpu {:.3} ms (min gpu {:.3}, max gpu {:.3})", run.median_wall_ms(), run.median_gpu_ms(), run.gpu_ms.iter().cloned().fold(f64::INFINITY, f64::min), run.gpu_ms.iter().cloned().fold(0.0, f64::max));

    let dir = image_dir(&shader);
    let kingdom = TrueKingdom::load(&dir)?;
    let contract = kingdom.relations().check().map_err(|d| format!("the kingdom check failed: {:?}", &d[..d.len().min(3)]))?;
    let tree = match shader.as_str() {
        "volumepassage" => volumepassage(&kingdom).map_err(|g| format!("table gaps:\n  {}", g.join("\n  ")))?,
        "nebula" => nebula(&kingdom).map_err(|g| format!("table gaps:\n  {}", g.join("\n  ")))?,
        o => return Err(format!("no encoded original for {o}")),
    };
    let n_nodes = tree.level_order().len();
    let n_funcs = functions(&tree, &kingdom);
    println!("tree: {n_nodes} nodes, {n_funcs} functions");
    let layout = GeneLayout::from_program_size(&contract, kingdom.root, n_funcs).map_err(|e| e.to_string())?;
    let const_len = kingdom.const_lens();
    let previous = random_gene(&contract, &layout, &const_len, &mut Rng(0x0579), Share::Half, true);
    let gene = write_back(&contract, &layout, &previous, &tree).map_err(|e| format!("write-back: {e}"))?;
    let expressed = express(&contract, &layout, &gene, &const_len).map_err(|e| format!("express: {e}"))?;
    if expressed.tree() != tree {
        return Err("our software has a bug: the expressed tree is not the tree written".into());
    }
    println!("gene: program size N = {n_funcs}, {} words; write-back and express agree", gene.len());
    let kfuel = kingdom.fuel(&contract, n_funcs)?;
    println!(
        "work guard for an evolved candidate of this kingdom (N = {n_funcs}): H = {}, K = {}; bound at {size}x{size} ({inv} invocations) {}, at 64x64 {}, at 16x16 {}; cap {}",
        kingdom.head_total(&contract, n_funcs)?,
        kingdom.fuel_budget,
        kfuel.bound(inv).unwrap_or(0),
        kfuel.bound(64 * 64).unwrap_or(0),
        kfuel.bound(16 * 16).unwrap_or(0),
        fuller::wgsl::oracle::device::CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT
    );
    let src = kingdom.render(&tree)?;
    std::fs::write(out.join(format!("{shader}_encoded.wgsl")), &src).map_err(|e| e.to_string())?;
    let module = fuller::wgsl::naga_shape::parse_unscanned(&src).map_err(|e| format!("naga: {e}"))?;
    fuller::wgsl::target::Target::current().validate(&module, &src).map_err(|e| format!("naga validate: {e}"))?;
    let kk = ImageKernel::compile(&dev, &shader, &src, "main", &bindings, size, size, Fuel::Proven("cuneus original written by hand as a tree of rows: its loops are the original's 50 x 7"))?;
    let run2 = kk.run(3, repeats)?;
    write_png(&out.join(format!("{shader}_encoded.png")), &run2.picture)?;
    let d = compare(&run2.picture, &run.picture)?;
    println!("encoded: median wall {:.3} ms, median gpu {:.3} ms; against the original: {d:?}", run2.median_wall_ms(), run2.median_gpu_ms());
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
