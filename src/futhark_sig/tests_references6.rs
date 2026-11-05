//! The reference programs as trees of their views (plan section 7, task F2a): each expresses, writes back to itself,
//! and decodes (G3) with no failing requirement and no work-bound substitution, and every variable is the one its
//! stored text (`tests/fixtures/futhark_m6_references.txt`) names.

use super::*;
use crate::typed_environment::Choice;
use crate::typed_gep::{express, write_back};

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

/// The trees name the variables the stored texts name: the histogram's is design section 6's reference (G3's
/// test: the tabulate's size `n`, the bins' `k`, `is[i]` with `i` the tabulate's index), pathfinder's inner `get`
/// reads the loop's state at the tabulate's index, and k-means's `assign` is the `let`'s variable.
#[test]
fn the_trees_name_the_variables_the_stored_texts_name() {
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

/// What a reference's stored text names each bound variable, as `<binder symbol> v<N>`: the binder row's `{k.vN}` in
/// its spelling (a loop state's first component is its `v0`, its second its `v1`, the counter after them). A name not
/// listed is a root variable of that name.
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

/// EVERY VARIABLE OF EVERY REFERENCE IS THE ONE ITS STORED TEXT NAMES: each variable leaf decodes to the binder and the
/// ordinal [`bound_names`] gives for its name, or to the root variable of its name, so two variables of one data
/// type with their names swapped in a binder fail here (shown on k-means's `[bi,best,c]`); and every binder row's
/// spelling names its bound variables `{k.v0}` .. `{k.v<n-1>}`, one for each of its bindings, so an ordinal is the
/// spelling's.
#[test]
fn every_variable_of_every_reference_is_the_one_its_stored_text_names() {
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
