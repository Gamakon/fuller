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
            let gene = k.random_gene(&layout, &mut rng);
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
