//! THE VIEW REGRESSION (`docs/PLAN_futhark_kingdom.md`, design D6 and 3g): each problem's kingdom, computed at
//! runtime from the master table and its view specification, is the kingdom its written-out table (`View::write`,
//! read back from the files; the text is pinned by its fingerprint in `futhark_view::tests`) loads: the same duals and rows, so the same
//! contract, constant lengths and alphabet at every cell of a layout, so the same genes express the same trees and the
//! same Futhark programs; and its signature is the one the old `interface.tsv` declared.

use super::*;
use crate::typed_gep::express;

/// The problems whose view is written out and loaded back.
const WRITTEN: [&str; 4] = ["pathfinder", "reduce", "scan", "histogram"];

/// A view's tables as `View::write` writes them, read back from the files: the written-out table (the option the plan
/// keeps), which the generator's witnesses check (`gen_tables.py --views DIR`).
fn written_tables(name: &str, view: &crate::futhark_view::View) -> (String, String) {
    let dir = std::env::temp_dir().join(format!("futhark_sig_written_{name}_{}", std::process::id()));
    view.write(&dir).unwrap();
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    let out = (read("types.tsv"), read("functions.tsv"));
    std::fs::remove_dir_all(&dir).unwrap();
    out
}

fn p(name: &str, elem: &str, dims: &[&str]) -> Param {
    Param { name: name.into(), elem: elem.into(), dims: dims.iter().map(|d| d.to_string()).collect() }
}

/// The signatures the hand-written `interface.tsv` files declared (commit d26a018a), restated.
fn declared(name: &str) -> Signature {
    let strs = |v: &[&str]| v.iter().map(|d| d.to_string()).collect::<Vec<_>>();
    match name {
        "pathfinder" => Signature { inputs: vec![p("walls", "i32", &["rows", "cols"])], output: p("main", "i32", &["cols"]), depth: 2, dims: strs(&["rows", "cols"]), size_inputs: vec![] },
        "reduce" => Signature { inputs: vec![p("xs", "i32", &["n"])], output: p("main", "i32", &[]), depth: 2, dims: strs(&["n"]), size_inputs: vec![] },
        "scan" => Signature { inputs: vec![p("xs", "i32", &["n"])], output: p("main", "i32", &["n"]), depth: 2, dims: strs(&["n"]), size_inputs: vec![] },
        "histogram" => Signature {
            inputs: vec![p("k", "i32", &[]), p("is", "i32", &["n"]), p("vs", "i32", &["n"])],
            output: p("main", "i32", &["k"]),
            depth: 2,
            dims: strs(&["k", "n"]),
            size_inputs: strs(&["k"]),
        },
        other => panic!("no declared signature for {other}"),
    }
}

#[test]
fn every_view_is_the_written_kingdom_and_the_same_genes_express_the_same_programs() {
    for name in WRITTEN {
        let prob = problem(name).unwrap();
        let k = load(&prob.instance).unwrap();
        let (types, functions) = written_tables(name, prob.instance.view().unwrap());
        let written = FutharkKingdom::from_tables(crate::futhark_table::FutharkFamily::Shape, &k.kingdom, &types, &functions).unwrap();
        let sig = signature(&prob.instance).unwrap();
        assert_eq!(sig, declared(name), "{name}: the view's signature is not the declared one");
        assert_eq!(k.duals, written.duals, "{name}: the duals differ");
        assert_eq!(k.rows, written.rows, "{name}: the rows differ");
        assert_eq!(format!("{:?}", k.relations()), format!("{:?}", written.relations()), "{name}: the contract relations differ");
        let const_len = k.const_len();
        assert_eq!(const_len, written.const_len(), "{name}: the constant lengths differ");
        let (ca, cb) = (k.relations().check().unwrap(), written.relations().check().unwrap());
        let root = k.roots()[0];
        assert_eq!(vec![root], written.roots(), "{name}: the roots differ");
        let layout = GeneLayout::from_program_size(&ca, root, 13).unwrap();
        let layout_b = GeneLayout::from_program_size(&cb, root, 13).unwrap();
        assert_eq!(format!("{layout:?}"), format!("{layout_b:?}"), "{name}: the layouts differ");
        for pos in 0..layout.width {
            assert_eq!(layout.alphabet(&ca, pos, &const_len), layout_b.alphabet(&cb, pos, &const_len), "{name}: the alphabet of cell {pos} differs");
        }
        // Every row renders the same text: a leaf as itself, a constant at every pool index, a function over leaf
        // children of its argument types.
        let leaf_of = |kk: &FutharkKingdom, ty: Ty| kk.rows.iter().position(|r| r.output == ty && r.inputs.is_empty() && r.kind == FutharkKind::Terminal).map(|i| Tree { symbol: i as u32, const_index: None, kids: vec![] });
        for (i, r) in k.rows.iter().enumerate() {
            let trees: Vec<Tree> = match r.kind {
                FutharkKind::Const => (0..const_len[&r.output]).map(|c| Tree { symbol: i as u32, const_index: Some(c), kids: vec![] }).collect(),
                _ => vec![Tree { symbol: i as u32, const_index: None, kids: r.inputs.iter().map(|t| leaf_of(&k, *t).unwrap_or_else(|| panic!("{name}: no terminal of {t:?}"))).collect() }],
            };
            for t in trees {
                assert_eq!(k.render(&t).unwrap(), written.render(&t).unwrap(), "{name}: the row {} renders differently", r.name);
            }
        }
        // The reference and 200 random genes: the same trees, the same programs.
        let reference = build(&k, prob.reference).unwrap();
        assert_eq!(program(&k, &sig, &reference).unwrap(), program(&written, &sig, &build(&written, prob.reference).unwrap()).unwrap(), "{name}: the reference's program differs");
        let mut rng = GeneRng::new(0x71E3_0001);
        for g in 0..200 {
            let gene = random_gene(&ca, &layout, &const_len, &mut rng).unwrap();
            let ta = express(&ca, &layout, &gene, &const_len).unwrap().tree();
            let tb = express(&cb, &layout_b, &gene, &const_len).unwrap().tree();
            assert_eq!(ta, tb, "{name}: gene {g} expresses differently");
            assert_eq!(program(&k, &sig, &ta).unwrap(), program(&written, &sig, &tb).unwrap(), "{name}: gene {g}'s program differs");
        }
        // The CPU interpreter on the view's kingdom: the reference is exact on every stored case.
        let interp = Interp::new(&k, &sig).unwrap();
        for c in load_cases(&prob).unwrap() {
            assert_eq!(interp.eval(&reference, &c.inputs).unwrap(), c.output, "{name}: the reference on {}", c.name);
        }
    }
}

/// THE PROGRAMS BEFORE BIDIRECTIONAL REACHABILITY (`docs/PLAN_futhark_kingdom.md`, 3e): each problem's reference and
/// planted slowdown, and A7's reference (the product twice, at n = 2, m = 3, p = 4), written as Futhark programs by
/// the views of commit 97dbbb80, when a signature view still admitted only the families its source used. Since the
/// contexts became sets (3g) a bound variable is named by its class, so a signature program's text differs from the
/// record in its binder names ONLY (checked token by token, one new name an old name, one old name a new name), and
/// the program's values are exact on every stored case (host interpreter). A7's matrix view has no binders: its
/// program is the record byte for byte.
const RECORDED: [(&str, &str, &str); 4] = [
    ("pathfinder", include_str!("../../tests/fixtures/futhark_view_programs/pathfinder.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/pathfinder.planted.fut")),
    ("reduce", include_str!("../../tests/fixtures/futhark_view_programs/reduce.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/reduce.planted.fut")),
    ("scan", include_str!("../../tests/fixtures/futhark_view_programs/scan.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/scan.planted.fut")),
    ("histogram", include_str!("../../tests/fixtures/futhark_view_programs/histogram.reference.fut"), include_str!("../../tests/fixtures/futhark_view_programs/histogram.planted.fut")),
];

/// Whether two Futhark texts differ only in binder names: equal token for token, except where the old token is a
/// stack-context binder name (`j1`, `i2`, `s1`, `t1`) and the new one a count-context name (`i_D`, `i2_D`, `ti_D`, `s_C`), with
/// one consistent renaming both ways. Returns the renaming.
fn binder_renaming(old: &str, new: &str) -> Option<BTreeMap<String, String>> {
    let toks = |t: &str| {
        let mut v = Vec::new();
        let mut cur = String::new();
        for c in t.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                cur.push(c);
            } else {
                if !cur.is_empty() {
                    v.push(std::mem::take(&mut cur));
                }
                if !c.is_whitespace() {
                    v.push(c.to_string());
                }
            }
        }
        if !cur.is_empty() {
            v.push(cur);
        }
        v
    };
    let (a, b) = (toks(old), toks(new));
    if a.len() != b.len() {
        return None;
    }
    let is_old = |x: &str| x.strip_prefix(['i', 'j', 's', 't']).is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
    let is_new = |x: &str| x.strip_prefix('t').unwrap_or(x).strip_prefix(['i', 's']).is_some_and(|r| r.trim_start_matches(|c: char| c.is_ascii_digit()).starts_with('_'));
    let (mut fwd, mut back): (BTreeMap<String, String>, BTreeMap<String, String>) = (BTreeMap::new(), BTreeMap::new());
    for (x, y) in a.iter().zip(&b) {
        if x == y {
            continue;
        }
        if !(is_old(x) && is_new(y)) || fwd.insert(x.clone(), y.clone()).is_some_and(|p| p != *y) || back.insert(y.clone(), x.clone()).is_some_and(|p| p != *x) {
            return None;
        }
    }
    Some(fwd)
}

#[test]
fn the_reference_programs_differ_from_the_recorded_ones_in_binder_names_only_and_are_exact() {
    for (name, reference, planted) in RECORDED {
        let prob = problem(name).unwrap();
        let k = load(&prob.instance).unwrap();
        let sig = signature(&prob.instance).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let cases = load_cases(&prob).unwrap();
        for (what, text, recorded) in [("reference", prob.reference, reference), ("planted slowdown", prob.planted, planted)] {
            let t = build(&k, text).unwrap();
            let got = program(&k, &sig, &t).unwrap();
            let renaming = binder_renaming(recorded, &got).unwrap_or_else(|| panic!("{name}: the {what}'s program differs from the record in more than binder names:\n{recorded}\n{got}"));
            eprintln!("{name} {what}: renamed {renaming:?}");
            for c in &cases {
                assert_eq!(interp.eval(&t, &c.inputs).unwrap(), c.output, "{name}: the {what} on {}", c.name);
            }
        }
    }
    let k = FutharkKingdom::load().unwrap();
    let sizes: BTreeMap<String, usize> = [("n", 2), ("m", 3), ("p", 4)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    let got = crate::futhark::program(&k, k.roots()[0], &crate::futhark_search::product_twice(&k), &sizes).unwrap();
    assert_eq!(got, include_str!("../../tests/fixtures/futhark_view_programs/a7.reference.fut"), "A7: the reference's program changed");
}

#[test]
fn the_binder_renaming_check_finds_a_difference_that_is_not_a_name() {
    assert!(binder_renaming("tabulate n (\\j1 -> xs[j1])", "tabulate n (\\i_n -> xs[i_n])").is_some());
    assert!(binder_renaming("tabulate n (\\j1 -> xs[j1])", "tabulate n (\\i_n -> ys[i_n])").is_none(), "a changed input");
    assert!(binder_renaming("f j1 j2", "f i_n i_n").is_none(), "two old names to one new name");
    assert!(binder_renaming("f j1 j1", "f i_n i_k").is_none(), "one old name to two new names");
}

/// The gene layouts of `docs/PLAN_futhark_kingdom.md` 3e, at the runner's default program size (the reference's
/// function count): the head length (the sum of the per-type heads) and the gene width in words. Before 3e (commit
/// 97dbbb80): pathfinder 14065 and 82573, reduce 1 and 13, scan 1 and 9, histogram 2506 and 13945; after 3e (stack
/// contexts) pathfinder 14065 and 84015, histogram 2506 and 13945. Count contexts (3g): these.
#[test]
fn the_gene_layouts_are_the_measured_ones() {
    let got: Vec<(&str, usize, usize, usize)> = ["pathfinder", "reduce", "scan", "histogram"]
        .into_iter()
        .map(|name| {
            let prob = problem(name).unwrap();
            let k = load(&prob.instance).unwrap();
            let reference = build(&k, prob.reference).unwrap();
            let n = reference.level_order().iter().filter(|(t, _)| !k.rows[t.symbol as usize].inputs.is_empty()).count();
            let layout = GeneLayout::from_program_size(&k.relations().check().unwrap(), k.roots()[0], n).unwrap();
            (name, n, layout.head_lens().values().sum::<usize>(), layout.width)
        })
        .collect();
    assert_eq!(got, [("pathfinder", 13, 3289, 22261), ("reduce", 1, 1, 11), ("scan", 1, 1, 8), ("histogram", 4, 721, 4655)]);
}

/// THE PROGRAM SIZE IS AN EXPERIMENT PARAMETER (`docs/reviews/2026-11-01_futhark_kingdom_design_fable.md`, F3): the
/// runner's default `DEFAULT_PROGRAM_SIZE` is above every problem's reference and planted slowdown, and both fit its
/// layout, so a run that starts from either (regime P, the slope check) needs no `--program-size`.
#[test]
fn the_default_program_size_is_above_every_reference_and_both_programs_fit() {
    let mut got = Vec::new();
    for name in ["pathfinder", "reduce", "scan", "histogram"] {
        let prob = problem(name).unwrap();
        let k = load(&prob.instance).unwrap();
        let contract = k.relations().check().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], DEFAULT_PROGRAM_SIZE).unwrap();
        for text in [prob.reference, prob.planted] {
            let t = build(&k, text).unwrap();
            assert!(function_count(&k, &t) < DEFAULT_PROGRAM_SIZE, "{name}: {} functions, the default program size is {DEFAULT_PROGRAM_SIZE}", function_count(&k, &t));
            crate::typed_gep::fits(&contract, &layout, &t).unwrap_or_else(|e| panic!("{name}: {text} does not fit N = {DEFAULT_PROGRAM_SIZE}: {e}"));
        }
        assert!(layout.guaranteed_program_size(&contract) >= DEFAULT_PROGRAM_SIZE, "{name}: the layout guarantees fewer functions than asked");
        got.push((name, layout.head_lens().values().sum::<usize>(), layout.width));
    }
    assert_eq!(got, [("pathfinder", 8495, 57487), ("reduce", 1551, 10025), ("scan", 1551, 10021), ("histogram", 7441, 47999)], "the head lengths and gene widths at N = {DEFAULT_PROGRAM_SIZE}");
}
