//! The reference programs as master 6 trees (plan section 7, task F2a): each expresses, writes back to itself, and
//! decodes (G3) with no failing requirement and no work-bound substitution; its expected text, its master 5 rendering,
//! is stored for task F2.

use super::*;
use crate::typed_environment::{Choice, Decoder, Entry};
use crate::typed_gep::{express, write_back};

/// The fixture as master 5 renders it now, in [`REFERENCES6`]'s order.
fn renderings() -> String {
    REFERENCES6.iter().map(|r| format!("{}{}", references6_separator(&r.key()), master5_rendering(r).unwrap_or_else(|e| panic!("{}: {e}", r.key())))).collect()
}

/// THE EXPECTED TEXTS ARE MASTER 5'S: the stored fixture is, byte for byte, what master 5's renderer writes for every
/// reference program today, and [`expected_text6`] gives each one back. `FULLER_WRITE_REFERENCES6=1` rewrites it.
#[test]
fn the_stored_texts_are_master_5_renderings_byte_for_byte() {
    let want = renderings();
    if std::env::var("FULLER_WRITE_REFERENCES6").is_ok_and(|v| v == "1") {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/futhark_m6_references.txt"), &want).unwrap();
        return;
    }
    assert!(REFERENCES6_MASTER5_TEXTS == want, "tests/fixtures/futhark_m6_references.txt is not master 5's rendering (FULLER_WRITE_REFERENCES6=1 rewrites it)");
    for r in &REFERENCES6 {
        let text = expected_text6(&r.key()).unwrap_or_else(|| panic!("{}: no stored text", r.key()));
        assert_eq!(text, master5_rendering(r).unwrap(), "{}", r.key());
        assert!(text.starts_with("entry main ") || text.starts_with("def matmul "), "{}", r.key());
    }
    assert_eq!(expected_text6("nothing.reference"), None);
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
        let (tables, contract) = (k.tables().unwrap(), k.contract().unwrap());
        let tree = build6(&k, &(r.master6)()).unwrap_or_else(|e| panic!("{key}: {e}"));
        let n = problem(r.problem).unwrap().program_size;
        let layout = GeneLayout::from_program_size(&contract, k.root.data_type, n).unwrap_or_else(|e| panic!("{key}: {e:?}"));
        let consts = const_len6(&k);
        let carrier = random_gene(&contract, &layout, &consts, &mut rng).unwrap();
        let gene = write_back(&contract, &layout, &carrier, &tree).unwrap_or_else(|e| panic!("{key}: does not fit at N = {n}: {e:?}"));
        let expressed = express(&contract, &layout, &gene, &consts).unwrap_or_else(|e| panic!("{key}: {e:?}")).tree();
        assert_eq!(expressed, tree, "{key}: the gene does not express the tree");
        assert_eq!(write_back(&contract, &layout, &gene, &expressed).unwrap(), gene, "{key}: the tree does not write back to its gene");
        let decoder = Decoder::new(&tables, &contract).unwrap();
        let decoding = decoder.decode(&Entry::of_kingdom6(&k), &tree).unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(decoding.failures(), Vec::new(), "{key}: a requirement fails or a node is substituted");
        assert!(decoding.root_outcomes.iter().all(|o| o.holds), "{key}");
        let mut stack = vec![&decoding.root];
        let (mut nodes, mut variables) = (0, 0);
        while let Some(node) = stack.pop() {
            nodes += 1;
            assert!(node.substituted.is_none() && node.outcomes.iter().all(|o| o.holds), "{key}: {}", k.symbols[node.symbol as usize].name);
            if contract.is_variable(node.symbol) {
                variables += 1;
                assert!(matches!(node.choice, Some(Choice::Variable(_))), "{key}: a variable leaf reads as {:?}", node.choice);
            }
            stack.extend(node.kids.iter());
        }
        assert_eq!(decoder.decode(&Entry::of_kingdom6(&k), &tree).unwrap(), decoding, "{key}");
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
    let (tables, contract) = (k.tables().unwrap(), k.contract().unwrap());
    let decoder = Decoder::new(&tables, &contract).unwrap();
    let d = decoder.decode(&Entry::of_kingdom6(&k), &tree).unwrap();
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
    assert_eq!(name_at(&[1, 1, 1, 1]), (None, 1), "the index is the tabulate's");
    assert_eq!(name_at(&[2]), (Some("vs".into()), 0));

    let k = kingdom6("pathfinder").unwrap();
    let tree = build6(&k, PATHFINDER_REFERENCE6).unwrap();
    let (tables, contract) = (k.tables().unwrap(), k.contract().unwrap());
    let decoder = Decoder::new(&tables, &contract).unwrap();
    let d = decoder.decode(&Entry::of_kingdom6(&k), &tree).unwrap();
    // (loop1 rows init (tab cols (add (get (row walls i) j) (min (min (get s j) ...) ...)))): get s j at [2,1,1,0,0].
    let get = d.node(&[2, 1, 1, 0, 0]).unwrap();
    let choice = |p: &[u32]| match d.node(p).unwrap().choice {
        Some(Choice::Variable(b)) => b,
        other => panic!("{p:?} reads as {other:?}"),
    };
    assert_eq!(k.symbols[get.symbol as usize].name, "get@i32");
    assert_eq!((choice(&[2, 1, 1, 0, 0, 0]).depth, choice(&[2, 1, 1, 0, 0, 0]).ordinal), (1, 0), "the loop's state");
    assert_eq!((choice(&[2, 1, 1, 0, 0, 1]).depth, choice(&[2, 1, 1, 0, 0, 1]).ordinal), (2, 0), "the tabulate's index");
    assert_eq!((choice(&[2, 1, 0, 0, 1]).depth, choice(&[2, 1, 0, 0, 1]).ordinal), (1, 1), "the loop's counter");

    let k = kingdom6("kmeans").unwrap();
    let tree = build6(&k, &kmeans_reference6()).unwrap();
    assert_eq!(k.symbols[tree.symbol as usize].name, "let.a.f32@m.f32");
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
    ] {
        let e = build6(&k, bad).expect_err(bad);
        assert!(e.starts_with("our software has a bug: "), "{bad}: {e}");
    }
}
