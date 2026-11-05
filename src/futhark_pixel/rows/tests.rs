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

/// The FNV-1a hash of each reference's 12 x 8 picture (every word 0.75), taken when it equalled the dual tables'.
const VOLUMEPASSAGE_BITS: u64 = 0xb1aa_4a69_3b71_8725;
const CUNEUS_BITS: u64 = 0x2ec0_f3e2_220c_ed42;

/// THE SHIPPED SHADERS READ AND RENDER THROUGH THE ROWS: each reference (volumepassage, cuneus) reads into a tree of
/// the rows, and its program, compiled by `futhark c`, draws a picture (12 x 8 pixels, not all black). Before
/// the dual tables were deleted this test required the picture BIT FOR BIT equal to theirs, and it was (fuller
/// e0b1d550, 288 of 288 values for each reference); the tables are gone, so the pictures' bits are pinned instead.
#[test]
fn the_references_read_on_the_rows_draw_their_pinned_picture() {
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
        let b = blocks(&iface, 0.75);
        let b: Vec<(u32, u32, &[u8])> = b.iter().map(|(g, bb, v)| (*g, *bb, v.as_slice())).collect();
        let (w, h) = (12u32, 8u32);
        let tile = super::super::Tile { col0: 0, row0: 0, cols: w, rows: h };
        let ps = super::super::inputs(&iface, &b, w, h, tile).unwrap();
        let draw = |src: &str, sub: &str| super::super::run_exe(&super::super::compile_c(src, &dir.join(format!("{name}_{sub}"))).unwrap_or_else(|e| panic!("{name} {sub}: {e}\n{src}")), &ps).unwrap();
        let a = draw(&new, "rows");
        assert_eq!(a.len(), (w * h * 3) as usize, "{name}");
        assert!(a.iter().any(|v| v.is_finite() && *v != 0.0), "{name}: not a picture");
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for v in &a {
            for b in v.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        eprintln!("{name}: the reference on the rows draws {} values, bits 0x{hash:016x}", a.len());
        assert_eq!(hash, if name == "volumepassage" { VOLUMEPASSAGE_BITS } else { CUNEUS_BITS }, "{name}: the picture's bits moved");
    }
}

/// Each reference's work a pixel, equal to the dual tables' count when they were compared (fuller 7d474c38).
const VOLUMEPASSAGE_WORK: u128 = 26_908;
const CUNEUS_WORK: u128 = 42_530;

/// THE INTERPRETER ON THE ROWS AGREES WITH FUTHARK'S C BACKEND: 12 random genes of the image instance (half head
/// draw, program size 24) and the two references, drawn both ways at 12 x 8, agree to 1e-4 relative (libm and
/// Futhark's `sin`/`pow` are not bit-identical), a NaN where the other is a NaN. The references' work a pixel is
/// pinned.
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
                let work = k.work_per_pixel(&k.decode(&tree).unwrap()).unwrap();
                eprintln!("{name}: the reference's work a pixel {work}");
                assert_eq!(work, if name == "volumepassage" { VOLUMEPASSAGE_WORK } else { CUNEUS_WORK }, "{name}: the work a pixel");
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

/// The number of function nodes of a tree (its program size).
fn functions(k: &PixelRows, t: &Tree) -> usize {
    usize::from(!k.contract.is_terminal(t.symbol)) + t.kids.iter().map(|c| functions(k, c)).sum::<usize>()
}

/// Each reference round-trips through a gene at its own program size: written back onto a random gene, expressed
/// again, the same tree. The instance's size is printed for the report.
#[test]
fn the_references_round_trip_through_a_gene() {
    for name in ["volumepassage", "cuneus"] {
        let (_, reference) = super::super::instance(name).unwrap();
        let k = PixelRows::load(name).unwrap();
        let tree = k.read(reference.unwrap()).unwrap();
        let n = functions(&k, &tree);
        let layout = k.layout(n).unwrap();
        let mut rng = crate::futhark_table::GeneRng::new(7);
        let carrier = k.random_gene(&layout, HeadDraw::Uniform, &mut rng).unwrap();
        let gene = crate::typed_gep::write_back(&k.contract, &layout, &carrier, &tree).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(k.express(&layout, &gene).unwrap(), tree, "{name}");
        let work = k.work_per_pixel(&k.decode(&tree).unwrap()).unwrap();
        eprintln!("{name}: {} data types, {} symbols; the reference has {} nodes, {n} functions, {work} node executions a pixel; gene {} words", k.tables.data_types.len(), k.contract.n_symbols(), tree.level_order().len(), layout.width);
    }
}

/// The head draw decides how big a random tree is: on 1,000 random cuneus genes at program size 60 each, the half
/// draw's median tree is smaller than the uniform draw's, and every gene expresses.
#[test]
fn the_half_head_draw_makes_smaller_random_trees() {
    let k = PixelRows::load("cuneus").unwrap();
    let layout = k.layout(60).unwrap();
    let mut medians = Vec::new();
    for draw in [HeadDraw::Uniform, HeadDraw::Half] {
        let mut rng = crate::futhark_table::GeneRng::new(0x51DE);
        let mut sizes: Vec<usize> = (0..1000).map(|_| k.express(&layout, &k.random_gene(&layout, draw, &mut rng).unwrap()).unwrap().level_order().len()).collect();
        sizes.sort_unstable();
        eprintln!("{draw:?}: 1000 random cuneus genes at N = 60 ({} words): nodes min {}, median {}, p90 {}, max {}", layout.width, sizes[0], sizes[500], sizes[900], sizes[999]);
        medians.push(sizes[500]);
    }
    assert!(medians[1] < medians[0], "the half draw's median tree {} is not below the uniform draw's {}", medians[1], medians[0]);
}

/// The e-graph over the rows loads, the volumepassage reference imports into it, and the variants it extracts are
/// trees of the rows whose programs Futhark's C backend draws BIT FOR BIT as the reference (the rules are IEEE
/// identities).
#[test]
fn the_egraph_variants_of_the_reference_draw_the_reference_bit_for_bit() {
    if !have_futhark() || Command::new("cc").arg("--version").output().is_err() {
        eprintln!("skipped: futhark or cc is not installed");
        return;
    }
    let (inst, reference) = super::super::instance("volumepassage").unwrap();
    let k = PixelRows::load("volumepassage").unwrap();
    let iface = super::super::interface(&inst).unwrap();
    let e = PixelEgraph::new(&k, &inst).unwrap();
    let tree = k.read(reference.unwrap()).unwrap();
    let vs = e.variants(&tree, 2, 8, 4).unwrap();
    assert!(!vs.is_empty(), "the e-graph found no variant");
    let b = blocks(&iface, 0.75);
    let b: Vec<(u32, u32, &[u8])> = b.iter().map(|(g, bb, v)| (*g, *bb, v.as_slice())).collect();
    let ps = super::super::inputs(&iface, &b, 12, 8, super::super::Tile { col0: 0, row0: 0, cols: 12, rows: 8 }).unwrap();
    let base = std::env::temp_dir().join("futhark_pixel_rows_egraph").join(format!("t{}", std::process::id()));
    let draw = |t: &Tree, sub: &str| super::super::run_exe(&super::super::compile_c(&k.program(&iface, t).unwrap(), &base.join(sub)).unwrap(), &ps).unwrap();
    let want = draw(&tree, "ref");
    assert_eq!(want.len(), 12 * 8 * 3);
    for (i, v) in vs.iter().enumerate() {
        let got = draw(v, &format!("v{i}"));
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "variant {i} draws another picture");
    }
    eprintln!("{} e-graph variants of the reference on the rows, each bit-identical under Futhark's C backend", vs.len());
}

/// THE GATE for an image target (bugberman, `tests/corpus/targets`): 1,000 random genes of the image instance (half
/// head draw, program size 60), drawn by the interpreter at 24 x 22, must draw something other than uniform black at
/// least half the time, and a constant colour near the target's mean (a pool expression) must score clearly better
/// than black on the quality inputs.
#[cfg(feature = "wgsl")]
#[test]
fn random_image_programs_draw_colour_and_the_mean_colour_beats_black() {
    use crate::wgsl::image::{largest_colour_share, quality_inputs, read_ppm, srgb_display, Picture};
    let target = read_ppm(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/targets/bugberman_24x22.ppm")).unwrap();
    let (w, h) = (target.width, target.height);
    let k = PixelRows::load("image").unwrap();
    let iface = super::super::interface(&super::super::IMAGE).unwrap();
    let layout = k.layout(60).unwrap();
    let ps = super::super::inputs(&iface, &[], w, h, super::super::Tile { col0: 0, row0: 0, cols: w, rows: h }).unwrap();
    let draw_tree = |t: &Tree| -> Picture {
        let d = k.decode(t).unwrap();
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let c = k.eval_pixel(&d, &iface, &ps, x as f32, y as f32).unwrap();
                rgba.extend_from_slice(&[c[0], c[1], c[2], 1.0]);
            }
        }
        Picture { width: w, height: h, rgba }
    };
    let theta = |p: &Picture| crate::score::truenorth_angle(&quality_inputs(p, &target).unwrap().0);
    let mut rng = crate::futhark_table::GeneRng::new(0xB06B);
    let (mut not_black, mut skipped) = (0usize, 0usize);
    for _ in 0..1000 {
        let t = k.express(&layout, &k.random_gene(&layout, HeadDraw::Half, &mut rng).unwrap()).unwrap();
        // The host draws only what the GPU guard would let through in one submit of this small picture.
        if k.work_per_pixel(&k.decode(&t).unwrap()).unwrap() * u128::from(w * h) > 170_000_000 {
            skipped += 1;
            continue;
        }
        let p = draw_tree(&t);
        if p.rgba.chunks_exact(4).flat_map(|c| c[..3].iter().map(|v| (srgb_display(*v).max(0.0) * 255.0 + 0.5) as u8).collect::<Vec<_>>()).any(|v| v > 1) {
            not_black += 1;
        }
    }
    let drawn = 1000 - skipped;
    eprintln!("1000 random image genes on the rows (half draw, N = 60): {drawn} drawn, {not_black} not uniform black; largest colour of the target {:.1} %", largest_colour_share(&target, 4).0 * 100.0);
    assert!(not_black * 2 >= drawn, "only {not_black} of {drawn} random programs draw something other than black");
    let mean: Vec<f64> = (0..3).map(|c| target.rgba.chunks_exact(4).map(|p| f64::from(p[c])).sum::<f64>() / f64::from(w * h)).collect();
    let pool: Vec<(String, f64)> = k.literals(k.symbol_named("?@f32").unwrap()).iter().map(|l| (l.trim_end_matches("f32").to_string(), l.trim_end_matches("f32").parse::<f64>().unwrap())).collect();
    let mut forms: Vec<(String, f64)> = pool.clone();
    for (a, x) in &pool {
        for (b, y) in &pool {
            forms.push((format!("(mul {a} {b})"), x * y));
            if *y != 0.0 {
                forms.push((format!("(div {a} {b})"), x / y));
            }
        }
    }
    let pick = |m: f64| forms.iter().min_by(|p, q| (p.1 - m).abs().total_cmp(&(q.1 - m).abs())).unwrap().clone();
    let chosen: Vec<(String, f64)> = mean.iter().map(|m| pick(*m)).collect();
    let text = format!("(image (rgb {} {} {}))", chosen[0].0, chosen[1].0, chosen[2].0);
    let mean_tree = k.read(&text).unwrap();
    let black_tree = Tree { symbol: k.symbol_named("image.black").unwrap(), const_index: None, kids: vec![] };
    let (tm, tb) = (theta(&draw_tree(&mean_tree)), theta(&draw_tree(&black_tree)));
    eprintln!("the target's mean colour {mean:.4?} spelled {text}; quality angle {tm:.4} against black's {tb:.4}");
    assert!(tm + 0.1 < tb, "the mean colour ({tm}) does not score clearly better than black ({tb})");
}
