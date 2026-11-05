//! The pixel family on the rows (plan 7.7 K4): every instance loads from the master tables and passes the kingdom
//! check; random genes render programs `futhark check` accepts, N of N.

use std::process::Command;
use std::time::Duration;

use super::*;

const INSTANCES: [&str; 3] = ["volumepassage", "cuneus", "image"];

fn have_futhark() -> bool {
    Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// Every instance loads: seven data types, the root `image`, a variable terminal and two constant pools.
#[test]
fn every_instance_loads_from_the_master_tables() {
    for name in INSTANCES {
        let k = PixelRows::load(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(k.tables.data_types.len(), 7, "{name}");
        assert_eq!(k.tables.data_type(k.root()).map(|d| d.name.as_str()), Some("image"), "{name}");
        assert!(k.symbol_named("var@f32").is_some() && k.symbol_named("?@f32").is_some() && k.symbol_named("?@count").is_some(), "{name}");
        let layout = k.layout(24).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(layout.width > 0, "{name}");
    }
}

/// THE SMOKE TEST: 20 random genes (7, 7 and 6 of the three instances, program size 24) decode on the rows and render
/// programs `futhark check` accepts, 20 of 20; some of them bind variables and loop.
#[test]
fn twenty_random_genes_render_programs_futhark_accepts() {
    if !have_futhark() {
        eprintln!("skipped: futhark is not installed");
        return;
    }
    let dir = std::env::temp_dir().join("futhark_pixel_rows_check").join(format!("t{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = crate::futhark_table::GeneRng::new(0x5EED_0704);
    let (mut checked, mut loops, mut lets) = (0usize, 0usize, 0usize);
    for (name, count) in INSTANCES.into_iter().zip([7usize, 7, 6]) {
        let k = PixelRows::load(name).unwrap();
        let (inst, _) = super::super::instance(name).unwrap();
        let iface = super::super::interface(&inst).unwrap();
        let layout = k.layout(24).unwrap();
        for i in 0..count {
            let gene = k.random_gene(&layout, HeadDraw::Half, &mut rng).unwrap();
            let tree = k.express(&layout, &gene).unwrap();
            let text = k.program(&iface, &tree).unwrap_or_else(|e| panic!("{name} gene {i}: {e}"));
            loops += text.matches(" loop (").count();
            lets += text.matches("let v").count();
            let f = dir.join(format!("{name}_{i}.fut"));
            std::fs::write(&f, &text).unwrap();
            let out = crate::futhark_exec::output_with_timeout(Command::new("futhark").arg("check").arg(&f), Duration::from_secs(60), None).unwrap();
            assert!(out.status.success(), "futhark rejects {name} gene {i}:\n{}\n{text}", String::from_utf8_lossy(&out.stderr));
            checked += 1;
        }
    }
    assert_eq!(checked, 20);
    eprintln!("pixel rows: {checked} random genes accepted by futhark check ({lets} lets, {loops} loops)");
}

/// A uniform block per (group, binding) of the interface, every word `value`.
fn blocks(iface: &[super::super::Input], value: f32) -> Vec<(u32, u32, Vec<u8>)> {
    let mut by: std::collections::BTreeMap<(u32, u32), usize> = std::collections::BTreeMap::new();
    for i in iface {
        let words = by.entry((i.group, i.binding)).or_insert(0);
        *words = (*words).max(i.word + 1);
    }
    by.into_iter().map(|((g, b), words)| (g, b, (0..words).flat_map(|_| value.to_le_bytes()).collect())).collect()
}

/// THE SHIPPED SHADERS READ AND RENDER THROUGH THE ROWS: each reference (volumepassage, cuneus) reads into a tree of
/// the rows, and its program, compiled by `futhark c`, draws the same picture BIT FOR BIT as the dual tables' program
/// of the same reference (12 x 8 pixels).
#[test]
fn the_references_read_on_the_rows_draw_the_dual_tables_picture() {
    if !have_futhark() || Command::new("cc").arg("--version").output().is_err() {
        eprintln!("skipped: futhark or cc is not installed");
        return;
    }
    let dir = std::env::temp_dir().join("futhark_pixel_rows_refs").join(format!("t{}", std::process::id()));
    for name in ["volumepassage", "cuneus"] {
        let (inst, reference) = super::super::instance(name).unwrap();
        let reference = reference.unwrap();
        let iface = super::super::interface(&inst).unwrap();
        let rows = PixelRows::load(name).unwrap();
        let tree = rows.read(reference).unwrap_or_else(|e| panic!("{name}: {e}"));
        let new = rows.program(&iface, &tree).unwrap();
        let old_k = super::super::load(&inst).unwrap();
        let old = super::super::program(&old_k, &iface, &super::super::build(&old_k, reference).unwrap()).unwrap();
        let b = blocks(&iface, 0.75);
        let b: Vec<(u32, u32, &[u8])> = b.iter().map(|(g, bb, v)| (*g, *bb, v.as_slice())).collect();
        let (w, h) = (12u32, 8u32);
        let tile = super::super::Tile { col0: 0, row0: 0, cols: w, rows: h };
        let ps = super::super::inputs(&iface, &b, w, h, tile).unwrap();
        let draw = |src: &str, sub: &str| super::super::run_exe(&super::super::compile_c(src, &dir.join(format!("{name}_{sub}"))).unwrap_or_else(|e| panic!("{name} {sub}: {e}\n{src}")), &ps).unwrap();
        let (a, z) = (draw(&new, "rows"), draw(&old, "dual"));
        assert_eq!(a.len(), z.len(), "{name}");
        assert!(a.iter().any(|v| *v != 0.0), "{name}: a black picture proves nothing");
        assert!(a.iter().zip(&z).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: the rows' picture differs from the dual tables'");
        eprintln!("{name}: the reference on the rows draws the dual tables' picture bit for bit ({} values)", a.len());
    }
}

/// THE INTERPRETER ON THE ROWS AGREES WITH FUTHARK'S C BACKEND: 12 random genes of the image instance (half head
/// draw, program size 24) and the two references, drawn both ways at 12 x 8, agree to 1e-4 relative (libm and
/// Futhark's `sin`/`pow` are not bit-identical), a NaN where the other is a NaN. The references' work a pixel is the
/// dual tables' count.
#[test]
fn the_interpreter_on_the_rows_agrees_with_futhark_c() {
    if !have_futhark() || Command::new("cc").arg("--version").output().is_err() {
        eprintln!("skipped: futhark or cc is not installed");
        return;
    }
    let dir = std::env::temp_dir().join("futhark_pixel_rows_interp").join(format!("t{}", std::process::id()));
    let mut rng = crate::futhark_table::GeneRng::new(0x1A7E);
    let mut checked = 0;
    for name in ["image", "volumepassage", "cuneus"] {
        let (inst, reference) = super::super::instance(name).unwrap();
        let iface = super::super::interface(&inst).unwrap();
        let k = PixelRows::load(name).unwrap();
        let trees: Vec<Tree> = match reference {
            Some(r) => {
                let tree = k.read(r).unwrap();
                let old_k = super::super::load(&inst).unwrap();
                let old = super::super::work_per_pixel(&old_k, &super::super::build(&old_k, r).unwrap()).unwrap();
                assert_eq!(k.work_per_pixel(&k.decode(&tree).unwrap()).unwrap(), old, "{name}: the work a pixel");
                vec![tree]
            }
            None => {
                let layout = k.layout(24).unwrap();
                (0..12).map(|_| k.express(&layout, &k.random_gene(&layout, HeadDraw::Half, &mut rng).unwrap()).unwrap()).collect()
            }
        };
        let b = blocks(&iface, 0.75);
        let b: Vec<(u32, u32, &[u8])> = b.iter().map(|(g, bb, v)| (*g, *bb, v.as_slice())).collect();
        for t in &trees {
            let (w, h) = (12u32, 8u32);
            let tile = super::super::Tile { col0: 0, row0: 0, cols: w, rows: h };
            let ps = super::super::inputs(&iface, &b, w, h, tile).unwrap();
            let src = k.program(&iface, t).unwrap();
            let got = super::super::run_exe(&super::super::compile_c(&src, &dir.join(format!("p{checked}"))).unwrap(), &ps).unwrap();
            let d = k.decode(t).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let want = k.eval_pixel(&d, &iface, &ps, x as f32, y as f32).unwrap();
                    for c in 0..3 {
                        let (g, e) = (got[((y * w + x) * 3 + c) as usize], want[c as usize]);
                        let ok = (g.is_nan() && e.is_nan()) || g == e || (g - e).abs() <= 1e-4 * g.abs().max(e.abs()).max(1e-3);
                        assert!(ok, "{name} pixel ({x}, {y}) channel {c}: futhark c {g}, the interpreter {e}\n{src}");
                    }
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 14);
}
