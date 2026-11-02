//! The view's regression: every signature view's written text is pinned by its fingerprint, A7's equals its committed
//! tables byte for byte, and the writer writes what the witnesses (`kingdoms/futhark/gen_tables.py --views DIR`)
//! check, so the engine's runtime kingdom is the table the witnesses checked.

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

/// THE REGRESSION OF THE WRITTEN VIEWS (`docs/PLAN_futhark_kingdom.md` 3g): the per-problem tables are no longer
/// committed; each signature view's written text is pinned by its fingerprint. Until 3g the committed tables were
/// compared byte for byte with the Python view's and this one's; the fingerprints below are of the text those
/// committed tables had (commit 1fe8a742), so the retirement changed no byte.
#[test]
fn every_signature_view_writes_the_pinned_text() {
    let master = Master::committed().unwrap();
    let got: Vec<(&str, u64)> = SIG.iter().map(|(name, spec)| (*name, View::load(&master, spec).unwrap().fingerprint())).collect();
    assert_eq!(got, [("histogram", 0x2945_ea1e_1d1d_c2d8), ("pathfinder", 0xa262_fc80_5844_d42c), ("reduce", 0x0949_f66a_a873_425e), ("scan", 0x5dd6_33e7_ed01_2bb4)], "the views' written text changed");
    for (name, spec) in SIG {
        assert_eq!(View::load(&master, spec).unwrap().artefacts().len(), 6, "{name}: a signature view writes six artefacts");
    }
}

/// The writer writes every committed view, and what it writes reads back as the view's own text.
#[test]
fn write_committed_writes_every_view_and_its_text() {
    let dir = std::env::temp_dir().join(format!("futhark_view_write_{}", std::process::id()));
    let views = write_committed(&dir).unwrap();
    assert_eq!(views.iter().map(|(s, _)| *s).collect::<Vec<_>>(), ["sig/histogram", "sig/pathfinder", "sig/reduce", "sig/scan", "a7"]);
    for (sub, v) in views {
        for (file, text) in v.artefacts() {
            let got = std::fs::read_to_string(dir.join(sub).join(file)).unwrap();
            assert!(got == text, "{sub}/{file}: the written file differs at {}", first_difference(&got, text));
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
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

#[test]
fn a_signature_view_carries_every_family_its_types_reach() {
    // Reduce's source has no `scan`, `hist` or `%`; before 3e its view had none of those rows. Every family is now
    // admitted and the reachability keeps each row whose types it reaches both ways.
    let v = View::load(&Master::committed().unwrap(), SIG[2].1).unwrap();
    let ops: BTreeSet<&str> = v.functions_tsv.lines().skip(1).filter_map(|l| l.split('\t').nth(1)).map(|n| n.split('.').next().unwrap_or("")).collect();
    for op in ["scan", "hist", "mod"] {
        assert!(ops.contains(op), "reduce's view has no {op} row: {ops:?}");
    }
}

#[test]
fn a_signature_specification_with_a_families_line_is_an_error() {
    let spec = SIG[3].1.replace("depth\t2\t\t\n", "depth\t2\t\t\nfamilies\tscan\t\t\n");
    assert_ne!(spec, SIG[3].1);
    let e = ViewSpec::parse(&spec).unwrap_err();
    assert!(e.contains("families"), "{e}");
}

#[test]
fn the_kept_rows_are_reachable_both_ways() {
    let row = |name: &str, kind: &str, out: &str, args: &[&str]| ViewRow {
        reference: "M".into(),
        name: name.into(),
        kind: kind.into(),
        out: out.into(),
        args: args.iter().map(|a| a.to_string()).collect(),
        spelling: String::new(),
        note: String::new(),
    };
    // A leaf of `a`; `f: a -> b` and `g: b -> r` chain to the root, `b` has no leaf of its own and is kept; `h: z -> r`
    // is unusable (nothing builds `z`); `u: a -> q` is usable but the root never uses `q`.
    let rows = vec![
        row("x", "terminal", "a", &[]),
        row("f", "function", "b", &["a"]),
        row("g", "function", "r", &["b"]),
        row("h", "function", "r", &["z"]),
        row("u", "function", "q", &["a"]),
    ];
    let kept: Vec<String> = reachable(rows.clone(), "r").unwrap().into_iter().map(|r| r.name).collect();
    assert_eq!(kept, ["x", "f", "g"]);
    let e = reachable(rows, "z").unwrap_err();
    assert!(e.contains("cannot be built"), "{e}");
}

/// The measurement of `docs/PLAN_futhark_kingdom.md` 3e: per view, its types, function rows and leaf rows after the
/// bidirectional reachability (the types and leaf rows are unchanged; the function rows grew: before 3e pathfinder
/// 3395, reduce 351, scan 369, histogram 2610).
#[test]
fn the_view_sizes_are_the_measured_ones() {
    let master = Master::committed().unwrap();
    for ((name, spec), want) in SIG.iter().zip([(607, 2700, 1779), (832, 3975, 2062), (73, 408, 221), (73, 408, 221)]) {
        let v = View::load(&master, spec).unwrap();
        let types = v.types_tsv.lines().count() - 1;
        let funcs = v.functions_tsv.lines().skip(1).filter(|l| l.split('\t').nth(2) == Some("function")).count();
        let leaves = v.functions_tsv.lines().count() - 1 - funcs;
        assert_eq!((types, funcs, leaves), want, "{name}");
    }
}
