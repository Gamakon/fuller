//! The reference programs as master 6 trees (plan section 7, task F2a): each expresses, writes back to itself, and
//! decodes (G3) with no failing requirement and no work-bound substitution; its expected text, its master 5 rendering,
//! is stored for task F2.

use super::*;
use crate::typed_environment::Choice;
use crate::typed_gep::{express, write_back};

/// The fixture as master 5 renders it now, in [`REFERENCES6`]'s order.
fn renderings() -> String {
    REFERENCES6.iter().map(|r| format!("{}{}", references6_separator(&r.key()), master5_rendering(r).unwrap_or_else(|e| panic!("{}: {e}", r.key())))).collect()
}

/// Whether a run asks to rewrite the fixture: `FULLER_WRITE_REFERENCES6` unset is no, exactly `1` is yes, and any
/// other value is an error (a mistyped request is never read as either).
fn write_requested(value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("FULLER_WRITE_REFERENCES6 is {other:?}: it must be exactly \"1\" (rewrite the fixture) or unset")),
    }
}

/// The fixture check: with `write`, `want` is written to `path` and the result is an error that says so (a rewrite is
/// never a pass: the binary under test holds the old text, compiled in); otherwise `stored` must equal `want`.
fn check_or_write(path: &std::path::Path, stored: &str, want: &str, write: bool) -> Result<(), String> {
    if write {
        std::fs::write(path, want).map_err(|e| format!("{}: {e}", path.display()))?;
        return Err(format!("wrote {} ({} bytes, {} references); run again without FULLER_WRITE_REFERENCES6 to check it", path.display(), want.len(), REFERENCES6.len()));
    }
    if stored != want {
        return Err(format!("{} is not master 5's rendering (FULLER_WRITE_REFERENCES6=1 rewrites it, and that run fails)", path.display()));
    }
    Ok(())
}

/// THE EXPECTED TEXTS ARE MASTER 5'S: the stored fixture is, byte for byte, what master 5's renderer writes for every
/// reference program today, and [`expected_text6`] gives each one back. `FULLER_WRITE_REFERENCES6=1` rewrites it, and
/// that run fails.
#[test]
fn the_stored_texts_are_master_5_renderings_byte_for_byte() {
    let want = renderings();
    let write = write_requested(std::env::var("FULLER_WRITE_REFERENCES6").ok().as_deref()).unwrap_or_else(|e| panic!("{e}"));
    let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/futhark_m6_references.txt"));
    check_or_write(path, REFERENCES6_MASTER5_TEXTS, &want, write).unwrap_or_else(|e| panic!("{e}"));
    for r in &REFERENCES6 {
        let text = expected_text6(&r.key()).unwrap_or_else(|| panic!("{}: no stored text", r.key()));
        assert_eq!(text, master5_rendering(r).unwrap(), "{}", r.key());
        assert!(text.starts_with("entry main ") || text.starts_with("def matmul "), "{}", r.key());
    }
    assert_eq!(expected_text6("nothing.reference"), None);
}

/// A rewrite of the fixture never passes, and the request must be exactly `1`.
#[test]
fn a_fixture_rewrite_fails_and_its_request_is_exactly_1() {
    assert_eq!(write_requested(None), Ok(false));
    assert_eq!(write_requested(Some("1")), Ok(true));
    for bad in ["", "0", "true", "yes", " 1", "1 "] {
        assert!(write_requested(Some(bad)).is_err(), "{bad:?}");
    }
    let dir = std::env::temp_dir().join(format!("fuller_references6_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("fixture.txt");
    let e = check_or_write(&path, "old", "new", true).unwrap_err();
    assert!(e.starts_with("wrote ") && e.contains("run again"), "{e}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    assert!(check_or_write(&path, "old", "new", false).is_err());
    assert_eq!(check_or_write(&path, "same", "same", false), Ok(()));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// EVERY REFERENCE IS A MASTER 6 TREE: it builds in its view, writes back into a gene of the problem's program size
/// and expresses to itself (the gene written back again is the gene), and decodes with no failing requirement, no
/// failing requirement of the entry and no work-bound substitution; every variable leaf reads as a variable (never the
/// constant of an empty scope). Pass 1 to 4 on the same tree twice gives the same annotations.
#[test]
fn every_reference_expresses_writes_back_and_decodes_with_no_failure() {
    let mut rng = GeneRng::new(1);
    for r in &REFERENCES6 {
        let key = r.key();
        let k = kingdom6(r.problem).unwrap_or_else(|e| panic!("{key}: {e}"));
        let contract = k.contract.as_ref().clone();
        let tree = build6(&k, &(r.master6)()).unwrap_or_else(|e| panic!("{key}: {e}"));
        let n = problem(r.problem).unwrap().program_size;
        let layout = GeneLayout::from_program_size(&contract, k.entry.output, n).unwrap_or_else(|e| panic!("{key}: {e:?}"));
        let consts = const_len6(&k);
        let carrier = random_gene(&contract, &layout, &consts, &mut rng).unwrap();
        let gene = write_back(&contract, &layout, &carrier, &tree).unwrap_or_else(|e| panic!("{key}: does not fit at N = {n}: {e:?}"));
        let expressed = express(&contract, &layout, &gene, &consts).unwrap_or_else(|e| panic!("{key}: {e:?}")).tree();
        assert_eq!(expressed, tree, "{key}: the gene does not express the tree");
        assert_eq!(write_back(&contract, &layout, &gene, &expressed).unwrap(), gene, "{key}: the tree does not write back to its gene");
        let decoding = k.decode(&tree).unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(decoding.failures(), Vec::new(), "{key}: a requirement fails or a node is substituted");
        assert!(decoding.root_outcomes.iter().all(|o| o.holds), "{key}");
        let mut stack = vec![&decoding.root];
        let (mut nodes, mut variables) = (0, 0);
        while let Some(node) = stack.pop() {
            nodes += 1;
            assert!(node.substituted.is_none() && node.outcomes.iter().all(|o| o.holds), "{key}: {}", k.name(node.symbol));
            if contract.is_variable(node.symbol) {
                variables += 1;
                assert!(matches!(node.choice, Some(Choice::Variable(_))), "{key}: a variable leaf reads as {:?}", node.choice);
            }
            stack.extend(node.kids.iter());
        }
        assert_eq!(k.decode(&tree).unwrap(), decoding, "{key}");
        eprintln!("{key}: {nodes} nodes, {variables} variables, {} functions, N = {n}", tree.level_order().iter().filter(|(t, _)| !t.kids.is_empty()).count());
    }
}

/// The trees name the variables the master 5 programs name: the histogram's is design section 6's reference (G3's
/// test: the tabulate's size `n`, the bins' `k`, `is[i]` with `i` the tabulate's index), pathfinder's inner `get`
/// reads the loop's state at the tabulate's index, and k-means's `assign` is the `let`'s variable.
#[test]
fn the_trees_name_the_variables_master_5_names() {
    let k = kingdom6("histogram").unwrap();
    let tree = build6(&k, HISTOGRAM_REFERENCE6).unwrap();
    let d = k.decode(&tree).unwrap();
    let name_at = |path: &[u32]| {
        let n = d.node(path).unwrap();
        let Some(Choice::Variable(b)) = n.choice else { panic!("{path:?} is not a variable") };
        let v = n.environment.variables.iter().find(|v| v.binder == b).unwrap();
        (v.name.clone(), b.depth)
    };
    assert_eq!(name_at(&[0]), (Some("k".into()), 0));
    assert_eq!(name_at(&[1, 0]), (Some("n".into()), 0));
    assert_eq!(name_at(&[1, 1, 0]), (Some("k".into()), 0));
    assert_eq!(name_at(&[1, 1, 1, 0]), (Some("is".into()), 0));
    assert_eq!(name_at(&[1, 1, 1, 1]), (Some("i_n".into()), 1), "the index is the tabulate's, named by its spelling");
    assert_eq!(name_at(&[2]), (Some("vs".into()), 0));

    let k = kingdom6("pathfinder").unwrap();
    let tree = build6(&k, PATHFINDER_REFERENCE6).unwrap();
    let d = k.decode(&tree).unwrap();
    // (loop1 rows init (tab cols (add (get (row walls i) j) (min (min (get s j) ...) ...)))): get s j at [2,1,1,0,0].
    let get = d.node(&[2, 1, 1, 0, 0]).unwrap();
    let choice = |p: &[u32]| match d.node(p).unwrap().choice {
        Some(Choice::Variable(b)) => b,
        other => panic!("{p:?} reads as {other:?}"),
    };
    assert_eq!(k.name(get.symbol), "get@i32");
    assert_eq!((choice(&[2, 1, 1, 0, 0, 0]).depth, choice(&[2, 1, 1, 0, 0, 0]).ordinal), (1, 0), "the loop's state");
    assert_eq!((choice(&[2, 1, 1, 0, 0, 1]).depth, choice(&[2, 1, 1, 0, 0, 1]).ordinal), (2, 0), "the tabulate's index");
    assert_eq!((choice(&[2, 1, 0, 0, 1]).depth, choice(&[2, 1, 0, 0, 1]).ordinal), (1, 1), "the loop's counter");

    let k = kingdom6("kmeans").unwrap();
    let tree = build6(&k, &kmeans_reference6()).unwrap();
    assert_eq!(k.name(tree.symbol), "let.a.f32@m.f32");
}

/// What master 5's program names each bound variable of a reference, as `<binder symbol> v<N>`: the binder row's
/// `{k.vN}` in its spelling (master 5's `s_f32` is a loop state's `v0`, `s2_f32` its `v1`, `i_k` the counter). A
/// name not listed is a root variable of that name.
fn bound_names(key: &str) -> &'static [(&'static str, &'static str)] {
    const KMEANS: &[(&str, &str)] = &[
        ("assign", "let.a.f32@m.f32 v0"),
        ("i", "tab@a.f32 v0"),
        ("best", "loop1@f32.f32 v0"),
        ("bi", "loop1@f32.f32 v1"),
        ("c", "loop1@f32.f32 v2"),
        ("acc", "loop@f32 v0"),
        ("j", "loop@f32 v1"),
        ("acc2", "loop@f32 v0"),
        ("j2", "loop@f32 v1"),
        ("ck", "tabrows@m.f32 v0"),
        ("cj", "tab@a.f32 v0"),
        ("sum", "loop@f32.f32.f32 v0"),
        ("comp", "loop@f32.f32.f32 v1"),
        ("count", "loop@f32.f32.f32 v2"),
        ("p", "loop@f32.f32.f32 v3"),
    ];
    const TEXTBOOK: &[(&str, &str)] = &[("i", "tab@a.f32 v0"), ("m", "loop@f32 v0"), ("c", "loop@f32 v1"), ("j", "tab@a.f32 v0")];
    match key {
        "histogram.reference" | "histogram.planted" => &[("i", "tab@a.i32 v0")],
        "pathfinder.reference" | "pathfinder.planted" => &[("s", "loop1@a.i32 v0"), ("i", "loop1@a.i32 v1"), ("j", "tab@a.i32 v0")],
        "kmeans.reference" | "kmeans.planted" => KMEANS,
        "kmeans.textbook" | "kmeans.textbook_planted" => TEXTBOOK,
        _ => &[],
    }
}

/// Each variable leaf of `text` (a reference of `problem`, its names those of [`bound_names`] for `key`) whose decoded
/// variable is not the one its name says: `<name> at <path>: <got>, not <want>`.
fn variable_mismatches(key: &str, problem: &str, text: &str) -> Vec<String> {
    let k = kingdom6(problem).unwrap();
    let (tree, leaves) = build6_named(&k, text).unwrap_or_else(|e| panic!("{key}: {e}"));
    let d = k.decode(&tree).unwrap();
    let expected = bound_names(key);
    let mut out = Vec::new();
    for (path, name) in &leaves {
        let leaf = d.node(path).unwrap();
        let Some(Choice::Variable(b)) = leaf.choice else { panic!("{key}: {name} at {path:?} reads as {:?}", leaf.choice) };
        let got = if b.depth == 0 {
            let v = leaf.environment.variables.iter().find(|v| v.binder == b).unwrap();
            format!("root {}", v.name.clone().unwrap_or_default())
        } else {
            let i = (0..path.len()).find(|i| d.node(&path[..=*i]).unwrap().environment.depth == b.depth).unwrap_or_else(|| panic!("{key}: no binder of {name}"));
            format!("{} v{}", k.name(d.node(&path[..i]).unwrap().symbol), b.ordinal)
        };
        let want = expected.iter().find(|(n, _)| n == name).map_or_else(|| format!("root {name}"), |(_, w)| w.to_string());
        if got != want {
            out.push(format!("{name} at {path:?}: {got}, not {want}"));
        }
    }
    assert!(expected.iter().all(|(n, _)| leaves.iter().any(|(_, l)| l == n)), "{key}: a listed name is used nowhere");
    out
}

/// EVERY VARIABLE OF EVERY REFERENCE IS THE ONE MASTER 5 NAMES: each variable leaf decodes to the binder and the
/// ordinal [`bound_names`] gives for its name, or to the root variable of its name, so two variables of one data
/// type with their names swapped in a binder fail here (shown on k-means's `[bi,best,c]`); and every binder row's
/// spelling names its bound variables `{k.v0}` .. `{k.v<n-1>}`, one for each of its bindings, so an ordinal is the
/// spelling's.
#[test]
fn every_variable_of_every_reference_is_the_one_master_5_names() {
    for r in &REFERENCES6 {
        let key = r.key();
        assert_eq!(variable_mismatches(&key, r.problem, &(r.master6)()), Vec::<String>::new(), "{key}");
        let k = kingdom6(r.problem).unwrap();
        let tree = build6(&k, &(r.master6)()).unwrap();
        let mut stack = vec![&tree];
        while let Some(t) = stack.pop() {
            if let Some((pos, count)) = k.binds(t.symbol).unwrap() {
                let spelling = k.symbol(t.symbol).and_then(|r| r.spelling("")).unwrap_or_default().to_string();
                let named: Vec<usize> = (0..count + 2).filter(|n| spelling.contains(&format!("{{{pos}.v{n}}}"))).collect();
                assert_eq!(named, (0..count).collect::<Vec<_>>(), "{key}: {}'s spelling {spelling:?} and its {count} variables", k.name(t.symbol));
            }
            stack.extend(t.kids.iter());
        }
    }
    let swapped = kmeans_reference6().replace("[best,bi,c]", "[bi,best,c]");
    assert_ne!(swapped, kmeans_reference6());
    let found = variable_mismatches("kmeans.reference", "kmeans", &swapped);
    assert!(found.iter().any(|m| m.starts_with("best at ")) && found.iter().any(|m| m.starts_with("bi at ")), "{found:?}");
}

/// THE MASTER 5 DRAWS ARE UNCHANGED by d499f633's `random_gene` change (a Dc or Dv word is no longer read as a symbol
/// to split by kind, which panicked on a Dv word at or above the number of symbols): eight genes of each master 5
/// problem but k-means, at its program size from one seed, hash (FNV-1a over the words) to what the code before
/// d499f633 drew, measured on a `git archive` copy of 594a4936.
#[test]
fn random_gene_draws_the_master_5_genes_it_drew_before_d499f633() {
    let mut out = Vec::new();
    for name in ["reduce", "scan", "histogram", "pathfinder", "a7"] {
        let p = problem(name).unwrap();
        let k = load(&p.instance).unwrap();
        let contract = k.relations().check().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], p.program_size).unwrap();
        let const_len = k.const_len();
        let mut rng = GeneRng::new(0x60_1DE6);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for _ in 0..8 {
            for w in random_gene(&contract, &layout, &const_len, &mut rng).unwrap() {
                h = (h ^ u64::from(w)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        out.push((name, layout.width, h));
    }
    assert_eq!(
        out,
        [("reduce", 10025, 1041675751412024729), ("scan", 10021, 11103297550919678028), ("histogram", 47999, 6993417665150567413), ("pathfinder", 57487, 224478049938280798), ("a7", 4846, 8447035347256795500)]
    );
}

/// A text that names a variable in scope nowhere, of another data type, a binder that binds nothing, or an unknown
/// symbol is not built.
#[test]
fn a_master_6_text_that_names_nothing_in_scope_is_not_built() {
    let k = kingdom6("histogram").unwrap();
    for bad in [
        "(hist.add@a.i32 k (tab@a.i32[i] n (mod@i32 k (get@i32 is j))) vs)",
        "(hist.add@a.i32 k (tab@a.i32[i] n (mod@i32 k (get@i32 is i))) i)",
        "(hist.add@a.i32 k (tab@a.i32[i] n (mod@i32 n (get@i32 is i))) is)x",
        "(hist.add@a.i32[x] k (tab@a.i32 n (mod@i32 k (get@i32 is zero@i64))) vs)",
        "(hist.add@a.i32 k (tab@a.i32[i,x] n (mod@i32 k (get@i32 is i))) vs)",
        "(hist.nothing@a.i32 k vs vs)",
        "(hist.add@a.i32 k (tab@a.i32[i] n (mod@i32 k (get@i32 is i))) vs",
        "(hist.add@a.i32 k vs)",
        "(hist.add@a.i32 is is vs)",
        "(hist.add@a.i32 k (tab@a.i32[] n (mod@i32 k (get@i32 is zero@i64))) vs)",
        "(hist.add@a.i32 k (tab@a.i32[i,] n (mod@i32 k (get@i32 is i))) vs)",
    ] {
        let e = build6(&k, bad).expect_err(bad);
        assert!(e.starts_with("our software has a bug: "), "{bad}: {e}");
    }
    // A name repeated in one binder: k-means's argmin `loop1` binds two `f32`s, so `[best,best,c]` with every `bi`
    // read as `best` is well typed, and only the repeated name is wrong.
    let k = kingdom6("kmeans").unwrap();
    let twice = kmeans_reference6().replace("[best,bi,c]", "[best,best,c]").replace("best bi)", "best best)");
    assert!(!twice.contains(" bi") && twice.contains("[best,best,c]"));
    let e = build6(&k, &twice).expect_err(&twice);
    assert!(e.starts_with("our software has a bug: ") && e.contains("names best twice"), "{e}");
}
