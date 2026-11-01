//! The view's regression: the Rust view of every committed specification writes the committed tables byte for byte
//! (the tables `kingdoms/futhark/gen_tables.py` writes from the same master with `view.py`), so the two
//! implementations of the view agree, and the engine's runtime kingdom is the table the witnesses checked.

use super::*;

const SIG: [(&str, &str); 4] = [
    ("histogram", include_str!("../../kingdoms/futhark/sig/histogram/view.tsv")),
    ("pathfinder", include_str!("../../kingdoms/futhark/sig/pathfinder/view.tsv")),
    ("reduce", include_str!("../../kingdoms/futhark/sig/reduce/view.tsv")),
    ("scan", include_str!("../../kingdoms/futhark/sig/scan/view.tsv")),
];

fn committed(rel: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn first_difference(a: &str, b: &str) -> String {
    match a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y) {
        Some((i, (x, y))) => format!("line {}: {x:?} vs {y:?}", i + 1),
        None => format!("lengths {} and {}", a.len(), b.len()),
    }
}

#[test]
fn every_signature_view_writes_the_committed_tables_byte_for_byte() {
    let master = Master::committed().unwrap();
    for (name, spec) in SIG {
        let v = View::load(&master, spec).unwrap();
        let arts = v.artefacts();
        assert_eq!(arts.len(), 6, "{name}: a signature view writes six artefacts");
        for (file, text) in arts {
            let want = committed(&format!("kingdoms/futhark/sig/{name}/{file}"));
            assert!(text == want, "{name}/{file}: the Rust view differs from the committed table at {}", first_difference(text, &want));
        }
    }
}

#[test]
fn the_a7_view_writes_the_committed_matrix_tables_byte_for_byte() {
    let v = View::load(&Master::committed().unwrap(), include_str!("../../kingdoms/futhark/view.tsv")).unwrap();
    for (file, text) in v.artefacts() {
        let want = committed(&format!("kingdoms/futhark/{file}"));
        assert!(text == want, "A7 {file}: the Rust view differs from the committed table at {}", first_difference(text, &want));
    }
}

#[test]
fn a_view_pinned_before_a_row_was_added_does_not_have_it() {
    // A master whose `mod` row is of version 2: a view pinned to version 1 loses exactly that row's instances.
    let rows = MASTER_ROWS.replace("\t1\tmod\t", "\t2\tmod\t");
    assert_ne!(rows, MASTER_ROWS, "the master has a mod row");
    let m = Master::parse("2", &rows, MASTER_FRAMES, MASTER_RULES, MASTER_POOLS, MASTER_TYPES).unwrap();
    let spec = SIG[0].1;
    let latest = View::load(&m, &spec.replace("master\t1\t", "master\t2\t")).unwrap();
    let pinned = View::load(&m, spec).unwrap();
    let has_mod = |v: &View| v.functions_tsv.lines().filter(|l| l.split('\t').nth(1).is_some_and(|n| n.starts_with("mod."))).count();
    assert!(has_mod(&latest) > 0, "version 2 has the mod rows");
    assert_eq!(has_mod(&pinned), 0, "version 1 does not");
}

#[test]
fn a_specification_with_an_unknown_line_or_family_row_variable_is_an_error_not_a_guess() {
    assert!(ViewSpec::parse("colour\tred\t\t\n").is_err());
    let rows = MASTER_ROWS.replacen("add on {E}", "add on {Q}", 1);
    let m = Master::parse(MASTER_VERSION, &rows, MASTER_FRAMES, MASTER_RULES, MASTER_POOLS, MASTER_TYPES).unwrap();
    let e = View::load(&m, SIG[2].1).unwrap_err();
    assert!(e.contains("{Q}"), "{e}");
}

#[test]
fn the_cache_gives_one_view_a_specification() {
    let a = cached(SIG[1].1).unwrap();
    let b = cached(SIG[1].1).unwrap();
    assert!(std::ptr::eq(a, b));
}
