//! THE VIEW REGRESSION (`docs/PLAN_futhark_kingdom.md`, design D6): each problem's kingdom, now computed at runtime
//! from the master table and its view specification, is the kingdom its written-out table (the generator's record,
//! byte for byte what the hand-declared signature gave before the view) loads: the same duals and rows, so the same
//! contract, constant lengths and alphabet at every cell of a layout, so the same genes express the same trees and the
//! same Futhark programs; and its signature is the one the old `interface.tsv` declared.

use super::*;
use crate::typed_gep::express;

const WRITTEN: [(&str, &str, &str); 4] = [
    ("pathfinder", include_str!("../../kingdoms/futhark/sig/pathfinder/types.tsv"), include_str!("../../kingdoms/futhark/sig/pathfinder/functions.tsv")),
    ("reduce", include_str!("../../kingdoms/futhark/sig/reduce/types.tsv"), include_str!("../../kingdoms/futhark/sig/reduce/functions.tsv")),
    ("scan", include_str!("../../kingdoms/futhark/sig/scan/types.tsv"), include_str!("../../kingdoms/futhark/sig/scan/functions.tsv")),
    ("histogram", include_str!("../../kingdoms/futhark/sig/histogram/types.tsv"), include_str!("../../kingdoms/futhark/sig/histogram/functions.tsv")),
];

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
    for (name, types, functions) in WRITTEN {
        let prob = problem(name).unwrap();
        let k = load(&prob.instance).unwrap();
        let written = FutharkKingdom::from_tables(crate::futhark_table::FutharkFamily::Shape, &k.kingdom, types, functions).unwrap();
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
