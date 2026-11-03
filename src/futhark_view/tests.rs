//! The view's regression: every signature view's written text is pinned by its fingerprint, A7's equals its committed
//! tables byte for byte, and the writer writes what the witnesses (`kingdoms/futhark/gen_tables.py --views DIR`)
//! check, so the engine's runtime kingdom is the table the witnesses checked.

use super::*;

const SIG: [(&str, &str); 9] = [
    ("argmin", include_str!("../../kingdoms/futhark/sig/argmin/view.tsv")),
    ("cell", include_str!("../../kingdoms/futhark/sig/cell/view.tsv")),
    ("histogram", include_str!("../../kingdoms/futhark/sig/histogram/view.tsv")),
    ("kahan", include_str!("../../kingdoms/futhark/sig/kahan/view.tsv")),
    ("kernels", include_str!("../../kingdoms/futhark/sig/kernels/view.tsv")),
    ("kmeans", include_str!("../../kingdoms/futhark/sig/kmeans/view.tsv")),
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
/// compared byte for byte with the Python view's and this one's; at the retirement (commit 0d819cf4) the fingerprints
/// were those of the removed files (histogram 2945ea1e1d1dc2d8, pathfinder a262fc805844d42c, reduce
/// 0949f66aa873425e, scan 5dd633e7ed012bb4), so it changed no byte. Since the count contexts (3g) they are these;
/// the four i32 views are PINNED to master version 2, so the f32 rows of version 3 (step 4) changed none of them, and
/// k-means (f32, version 3, depth 3) is pinned since 2026-11-03. The pair rows and the second element of version 4
/// (step 4c (b)) changed none of the five; the two test views of the pair constructs (`argmin`, `kahan`: f32 and i32,
/// version 4, depth 2) are pinned since 2026-11-03 too. The triple rows and the kernel boundary of version 5 (4c (c))
/// changed none of the seven; their test views (`cell`: the triple state; `kernels`: a `let` of an `[n]i32` at the
/// top) are pinned from the same day.
#[test]
fn every_signature_view_writes_the_pinned_text() {
    let master = Master::committed().unwrap();
    let got: Vec<(&str, u64)> = SIG.iter().map(|(name, spec)| (*name, View::load(&master, spec).unwrap().fingerprint())).collect();
    assert_eq!(
        got,
        [("argmin", 0x7d3d_cfe6_f1f6_b037), ("cell", 0x5590_65d7_74ea_1544), ("histogram", 0x2605_d528_715c_9478), ("kahan", 0x9550_5694_e22a_cb48), ("kernels", 0x2a0f_8b0d_9e70_b161), ("kmeans", 0xec8a_d19c_c33e_da6c), ("pathfinder", 0xfc02_c5a2_4062_3234), ("reduce", 0x6281_6de9_35b3_f197), ("scan", 0x0a38_fd72_a536_0859)],
        "the views' written text changed"
    );
    for (name, spec) in SIG {
        assert_eq!(View::load(&master, spec).unwrap().artefacts().len(), 6, "{name}: a signature view writes six artefacts");
    }
}

/// The writer writes every committed view, and what it writes reads back as the view's own text.
#[test]
fn write_committed_writes_every_view_and_its_text() {
    let dir = std::env::temp_dir().join(format!("futhark_view_write_{}", std::process::id()));
    let views = write_committed(&dir).unwrap();
    assert_eq!(views.iter().map(|(s, _)| *s).collect::<Vec<_>>(), ["sig/argmin", "sig/cell", "sig/histogram", "sig/kahan", "sig/kernels", "sig/kmeans", "sig/pathfinder", "sig/reduce", "sig/scan", "a7"]);
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
    // A master whose `mod` row is of version 3: a view pinned to version 2 (the committed specifications') loses
    // exactly that row's instances.
    let rows = MASTER_ROWS.replace("\t1\tmod\t", "\t3\tmod\t");
    assert_ne!(rows, MASTER_ROWS, "the master has a mod row");
    let m = Master::parse(MASTER_VERSION, &rows, MASTER_FRAMES, MASTER_RULES, MASTER_POOLS, MASTER_TYPES).unwrap();
    let spec = SIG[2].1;
    assert!(spec.contains("master\t2\t"), "the specification pins version 2");
    let latest = View::load(&m, &spec.replace("master\t2\t", "master\t3\t")).unwrap();
    let pinned = View::load(&m, spec).unwrap();
    let has_mod = |v: &View| v.functions_tsv.lines().filter(|l| l.split('\t').nth(1).is_some_and(|n| n.starts_with("mod."))).count();
    assert!(has_mod(&latest) > 0, "version 3 has the mod rows");
    assert_eq!(has_mod(&pinned), 0, "version 2 does not");
}

#[test]
fn a_specification_with_an_unknown_line_or_family_row_variable_is_an_error_not_a_guess() {
    assert!(ViewSpec::parse("colour\tred\t\t\n").is_err());
    let rows = MASTER_ROWS.replacen("add on {E}", "add on {Q}", 1);
    let m = Master::parse(MASTER_VERSION, &rows, MASTER_FRAMES, MASTER_RULES, MASTER_POOLS, MASTER_TYPES).unwrap();
    let e = View::load(&m, SIG[7].1).unwrap_err();
    assert!(e.contains("{Q}"), "{e}");
}

#[test]
fn the_cache_gives_one_view_a_specification() {
    let a = cached(SIG[6].1).unwrap();
    let b = cached(SIG[6].1).unwrap();
    assert!(std::ptr::eq(a, b));
}

#[test]
fn a_signature_view_carries_every_family_its_types_reach() {
    // Reduce's source has no `scan`, `hist` or `%`; before 3e its view had none of those rows. Every family is now
    // admitted and the reachability keeps each row whose types it reaches both ways.
    let v = View::load(&Master::committed().unwrap(), SIG[7].1).unwrap();
    let ops: BTreeSet<&str> = v.functions_tsv.lines().skip(1).filter_map(|l| l.split('\t').nth(1)).map(|n| n.split('.').next().unwrap_or("")).collect();
    for op in ["scan", "hist", "mod"] {
        assert!(ops.contains(op), "reduce's view has no {op} row: {ops:?}");
    }
}

#[test]
fn a_signature_specification_with_a_families_line_is_an_error() {
    let spec = SIG[8].1.replace("depth\t2\t\t\n", "depth\t2\t\t\nfamilies\tscan\t\t\n");
    assert_ne!(spec, SIG[8].1);
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

/// The measurement of `docs/PLAN_futhark_kingdom.md` 3e, 3g and step 4: per view, its types, function rows and leaf
/// rows. Stack contexts after 3e: histogram (607, 2700, 1779), pathfinder (832, 3975, 2062), reduce and scan (73,
/// 408, 221). Count contexts (3g): these; k-means at depth 3 over 9 classes (2026-11-03): 3804 types. The two test
/// views of the pair constructs (4c (b), two elements over one dimension at depth 2, the four pair classes in every
/// context): 328 types each. The triple rows (4c (c), the eight triple classes and their loop frames) take the same
/// specification to 1,041 types (`cell`); one kernel boundary (`kernels`, `lets a.i32.n`) doubles the contexts below
/// it: 1,892 types.
#[test]
fn the_view_sizes_are_the_measured_ones() {
    let master = Master::committed().unwrap();
    for ((name, spec), want) in SIG.iter().zip([(328, 2196, 620), (1041, 6749, 1721), (162, 1032, 455), (328, 2196, 658), (1892, 12371, 2962), (3804, 21620, 8421), (184, 1188, 441), (28, 202, 79), (28, 202, 79)]) {
        let v = View::load(&master, spec).unwrap();
        let types = v.types_tsv.lines().count() - 1;
        let funcs = v.functions_tsv.lines().skip(1).filter(|l| l.split('\t').nth(2) == Some("function")).count();
        let leaves = v.functions_tsv.lines().count() - 1 - funcs;
        assert_eq!((types, funcs, leaves), want, "{name}");
    }
}

/// THE CLASS GRAMMAR (review finding 1): the text of a class reads one way. A view of one element implies the
/// element of an array (`a.n`); a view of several writes it (`a.f32.n`); a tuple's components are inside its own
/// parentheses, so `(a.n.d,f32)` and `(a.n,a.d)` differ where a dotted spelling would not; a dimension named as an
/// element, or an array without its element in a view of several, does not parse.
#[test]
fn a_class_reads_one_way_and_writes_back_to_its_text() {
    let one = vec!["i32".to_string()];
    let two = vec!["f32".to_string(), "i32".to_string()];
    for (text, elements) in [("i32", &one), ("ix.n", &one), ("a.n", &one), ("a.rows.cols", &one), ("(i32,i32)", &one), ("f32", &two), ("a.f32.n", &two), ("a.i32.k.d", &two), ("(f32,i32)", &two), ("(a.f32.n.d,f32)", &two), ("(a.f32.n,a.f32.d)", &two)] {
        let c = Class::parse(text, elements, None).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(c.text(elements.len() > 1), text, "{text} does not write back");
    }
    assert_eq!(Class::parse("a.n", &one, None).unwrap(), Class::Array { elem: "i32".into(), dims: vec!["n".into()] }, "the one element is implied");
    assert_eq!(Class::parse("a.n", &two, Some("i32")).unwrap().text(true), "a.i32.n", "the row's element is taken");
    assert_ne!(Class::parse("(a.f32.n.d,f32)", &two, None).unwrap(), Class::parse("(a.f32.n,a.f32.d)", &two, None).unwrap());
    for text in ["a.n", "a.f64.n", "f64", "(f32)", "(f32,i32", "a.", "ix.f32", "a.f32.i32", "colour"] {
        assert!(Class::parse(text, &two, None).is_err(), "{text} parses in a view of two elements");
    }
    assert_eq!(Class::Tuple(vec![Class::Scalar("f32".into()), Class::Scalar("i32".into())]).futhark_type(), "(f32, i32)");
    assert_eq!(Class::parse("a.f32.n", &two, None).unwrap().tag(true), "f32_n");
    assert_eq!(Class::parse("a.n", &one, None).unwrap().tag(false), "n");
}

/// A specification of several elements: each element's pool on its own line, an element the pools miss is an error,
/// a dimension named as an element is an error, a scalar input that is not a size is an error.
#[test]
fn a_specification_of_two_elements_is_checked() {
    let spec = SIG[3].1;
    let s = ViewSpec::parse(spec).unwrap();
    assert_eq!(s.elements, vec!["f32".to_string(), "i32".to_string()]);
    assert!(s.multi());
    assert_eq!(s.literals["i32"], vec!["0", "1", "2", "-1", "10"]);
    assert_eq!(s.literals["f32"].len(), 6);
    let e = ViewSpec::parse(&spec.replace("literals\t0,1,2,-1,10\ti32\t\n", "")).unwrap_err();
    assert!(e.contains("no literals line"), "{e}");
    let e = ViewSpec::parse(&spec.replace("\tf32\tn\n", "\tf32\tf64\n")).unwrap_err();
    assert!(e.contains("f64"), "{e}");
    let e = ViewSpec::parse(&spec.replace("input\tms\ti32\tn\n", "input\tms\ti32\t\n")).unwrap_err();
    assert!(e.contains("scalar that is not a size"), "{e}");
    // An element the line drops: its literals line and its input are both out of place; the first found is the error.
    let e = ViewSpec::parse(&spec.replace("element\tf32,i32", "element\tf32")).unwrap_err();
    assert!(e.contains("i32"), "{e}");
    let e = ViewSpec::parse(&spec.replace("element\tf32,i32", "element\tf32").replace("literals\t0,1,2,-1,10\ti32\t\n", "")).unwrap_err();
    assert!(e.contains("does not list"), "{e}");
}

/// A view of two elements has every element's rows, the rows of one element only (`mod` and the i32 `clamp`, `div`
/// and the f32 `clamp`) named by their element where their output class has none, the pair rows of every pair of
/// scalars, the two conversions, and the zero arrays named by their class.
#[test]
fn a_view_of_two_elements_has_both_elements_rows() {
    let v = View::load(&Master::committed().unwrap(), SIG[3].1).unwrap();
    let names: BTreeSet<&str> = v.functions_tsv.lines().skip(1).filter_map(|l| l.split('\t').nth(1)).collect();
    for n in ["add@f32:c", "add@i32:c", "mod.n@i32:c", "div@f32:c", "clamp.i32@ix.n:c", "clamp.f32@ix.n:c", "pair@(f32,i32):c", "fst.f32.i32@f32:c", "snd.f32.i32@i32:c", "psel.f32@(f32,f32):c", "psel.i32@(f32,f32):c", "zero@(i32,f32):c", "loop.n@(f32,f32):c", "loop1.n@(f32,i32):c", "cvt.i32@f32:c", "cvt.f32@i32:c", "zeros@a.f32.n:c", "zeros@a.i32.n:c", "s_f32@f32:c/{f32*2,ix.n}@1", "s2_f32@f32:c/{f32*2,ix.n}@1", "s_i32@i32:c/{f32,i32,ix.n}@1"] {
        assert!(names.contains(n), "the kahan view has no row {n}");
    }
    assert!(!names.contains("mod.n@f32:c") && !names.contains("div@i32:c") && !names.contains("clamp@ix.n:c"));
    let spelling = |name: &str| v.functions_tsv.lines().find(|l| l.split('\t').nth(1) == Some(name)).and_then(|l| l.split('\t').nth(5)).unwrap_or("").to_string();
    assert_eq!(spelling("loop.n@(f32,f32):c"), "loop ((s_f32, s2_f32): (f32, f32)) = {0} for i_n < n do {1}");
    assert_eq!(spelling("loop.n@(f32,i32):c"), "loop ((s_f32, s_i32): (f32, i32)) = {0} for i_n < n do {1}");
    assert_eq!(spelling("zero@(f32,i32):c"), "(0f32, 0i32)");
    assert_eq!(spelling("zeros@a.i32.n:c"), "zeros_i32_n");
    assert!(v.rules.contains("(rewrite (Xfst_f32_i32__f32_c (Xpair___f32_i32__c a b)) a :ruleset futhark_sig)"), "the pair rule at (f32,i32):c");
}

/// THE TRIPLE ROWS AND THE KERNEL BOUNDARY (master version 5, step 4c (c)): a view of two elements has the eight
/// triple classes with their constructor, three projections, zero, select and destructuring loops (the frames `L3`,
/// `F3`); a view with a `lets` line has the `let` row for every class of the view at the top only, its body in the
/// context that binds the let's class at depth 0 (`c/{a.i32.n}@0`, a context of no trip), and the bound array as a
/// terminal there; a pinned view of version 4 has none of it; a `lets` line of a class the view lacks, or of a scalar,
/// is an error.
#[test]
fn a_view_of_version_five_has_the_triple_rows_and_the_kernel_boundary() {
    let master = Master::committed().unwrap();
    let cell = View::load(&master, SIG[1].1).unwrap();
    let names = |v: &View| v.functions_tsv.lines().skip(1).filter_map(|l| l.split('\t').nth(1).map(str::to_string)).collect::<BTreeSet<String>>();
    let spelling = |v: &View, name: &str| v.functions_tsv.lines().find(|l| l.split('\t').nth(1) == Some(name)).and_then(|l| l.split('\t').nth(5)).unwrap_or("").to_string();
    let cn = names(&cell);
    for n in ["triple@(f32,f32,i32):c", "fst.f32.f32.i32@f32:c", "snd.f32.f32.i32@f32:c", "thd.f32.f32.i32@i32:c", "zero@(f32,f32,i32):c", "psel.i32@(f32,f32,i32):c", "psel.f32@(f32,f32,i32):c", "loop.n@(f32,f32,i32):c", "loop1.n@(i32,i32,i32):c", "s_f32@f32:c/{f32*2,i32,ix.n}@1", "s2_f32@f32:c/{f32*2,i32,ix.n}@1", "s_i32@i32:c/{f32*2,i32,ix.n}@1", "s3_f32@f32:c/{f32*3,ix.n}@1"] {
        assert!(cn.contains(n), "the cell view has no row {n}");
    }
    assert_eq!(spelling(&cell, "loop.n@(f32,f32,i32):c"), "loop ((s_f32, s2_f32, s_i32): (f32, f32, i32)) = {0} for i_n < n do {1}");
    assert_eq!(spelling(&cell, "zero@(f32,f32,i32):c"), "(0f32, 0f32, 0i32)");
    assert_eq!(spelling(&cell, "thd.f32.f32.i32@i32:c"), "{0}.2");
    assert!(cell.rules.contains("(rewrite (Xthd_f32_f32_i32__i32_c (Xtriple___f32_f32_i32__c a b d)) d :ruleset futhark_sig)"), "the triple's third projection rule");
    assert!(!cn.iter().any(|n| n.starts_with("let@")), "the cell view has no lets line, so no let");
    let kernels = View::load(&master, SIG[4].1).unwrap();
    let kn = names(&kernels);
    for n in ["let@f32:c", "let@a.i32.n:c", "let@a.f32.n:c", "s_a_i32_n@a.i32.n:c/{a.i32.n}@0", "loop.n@f32:c/{a.i32.n}@0", "s_a_i32_n@a.i32.n:c/{a.i32.n,f32,ix.n}@1", "s2_a_i32_n@a.i32.n:c/{a.i32.n*2,ix.n}@1"] {
        assert!(kn.contains(n), "the kernels view has no row {n}");
    }
    assert!(!kn.iter().any(|n| n.starts_with("let@") && !n.ends_with(":c")), "a let stands at the top only: {:?}", kn.iter().filter(|n| n.starts_with("let@")).collect::<Vec<_>>());
    assert_eq!(spelling(&kernels, "let@f32:c"), "let s_a_i32_n = {0} in {1}");
    let row = kernels.functions_tsv.lines().find(|l| l.split('\t').nth(1) == Some("let@f32:c")).unwrap();
    assert_eq!(row.split('\t').nth(4), Some("a.i32.n:c;f32:c/{a.i32.n}@0"));
    // The pinned pair views have no triple and no let.
    let kahan = View::load(&master, SIG[3].1).unwrap();
    assert!(!names(&kahan).iter().any(|n| n.starts_with("triple@") || n.starts_with("let@")));
    // A lets line of a class the view lacks, or of a scalar.
    for bad in ["lets\ta.i32.d\t\t\n", "lets\tf32\t\t\n", "lets\ta.i32.n,a.i32.n\t\t\n"] {
        let spec = SIG[4].1.replace("lets\ta.i32.n\t\t\n", bad);
        assert_ne!(spec, SIG[4].1);
        assert!(ViewSpec::parse(&spec).is_err(), "{bad:?} parses");
    }
}
