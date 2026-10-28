//! The reader's checks: every row's spelling parses as a pattern; the round trip, every program the kingdom renders
//! (the reference and planted genes of every problem, and random genes) reads back to its own tree; the published
//! yardsticks read to the reference gene, or are a gap that names the construct the kingdom lacks.

use super::*;
use crate::futhark_kingdom::GeneRng;
use crate::futhark_sig::{build, load, problem, program, random_gene, sexp, signature, Problem};
use crate::typed_gep::{express, GeneLayout};

const PROBLEMS: [&str; 4] = ["histogram", "pathfinder", "reduce", "scan"];

fn functions(k: &FutharkKingdom, t: &Tree) -> usize {
    usize::from(!k.rows[t.symbol as usize].inputs.is_empty()) + t.kids.iter().map(|c| functions(k, c)).sum::<usize>()
}

fn setup(p: &Problem) -> (FutharkKingdom, Signature) {
    (load(&p.instance).unwrap(), signature(&p.instance).unwrap())
}

#[test]
fn every_spelling_of_every_instance_parses_as_a_pattern() {
    for name in PROBLEMS {
        let p = problem(name).unwrap();
        let (k, sig) = setup(&p);
        Reader::new(&k, &sig).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn the_reference_and_planted_genes_read_back_to_their_own_trees() {
    for name in PROBLEMS {
        let p = problem(name).unwrap();
        let (k, sig) = setup(&p);
        let r = Reader::new(&k, &sig).unwrap();
        for text in [p.reference, p.planted] {
            let tree = build(&k, text).unwrap();
            let src = program(&k, &sig, &tree).unwrap();
            let got = r.read_program(&src).unwrap_or_else(|g| panic!("{name}: {g}\n{src}"));
            assert_eq!(sexp(&k, &got), sexp(&k, &tree), "{name}\n{src}");
            assert_eq!(got, tree, "{name}");
        }
    }
}

/// The two trees differ only where both hold `ival.D x` with another `D` and an `x` of the same text: the rows spelled
/// alike in every index class (`zero` is `0i64`, `prev x` is `i64.max 0 (x - 1)`), so the program is one program.
fn only_index_classes_differ(k: &FutharkKingdom, a: &Tree, b: &Tree) -> bool {
    let name = |t: &Tree| k.rows[t.symbol as usize].name.split('@').next().unwrap_or("").to_string();
    if a.symbol == b.symbol && a.const_index == b.const_index && a.kids.len() == b.kids.len() {
        return a.kids.iter().zip(&b.kids).all(|(x, y)| only_index_classes_differ(k, x, y));
    }
    let ival = |t: &Tree| name(t).starts_with("ival.") && t.kids.len() == 1;
    ival(a) && ival(b) && k.rows[a.symbol as usize].output == k.rows[b.symbol as usize].output && k.render(&a.kids[0]).ok() == k.render(&b.kids[0]).ok()
}

#[test]
fn every_rendered_random_gene_reads_back_to_its_own_tree() {
    for name in PROBLEMS {
        let p = problem(name).unwrap();
        let (k, sig) = setup(&p);
        let r = Reader::new(&k, &sig).unwrap();
        let contract = k.relations().check().unwrap();
        let reference = build(&k, p.reference).unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], functions(&k, &reference)).unwrap();
        let const_len = k.const_len();
        let mut rng = GeneRng::new(0x2EAD_0001);
        let (mut nodes, mut same_text) = (0, 0);
        for g in 0..400 {
            let gene = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
            let tree = express(&contract, &layout, &gene, &const_len).unwrap().tree();
            nodes += tree.level_order().len();
            let src = program(&k, &sig, &tree).unwrap();
            let got = r.read_program(&src).unwrap_or_else(|gap| panic!("{name} gene {g}: {gap}\n{src}"));
            // The program read renders to the same text, always.
            assert_eq!(program(&k, &sig, &got).unwrap(), src, "{name} gene {g}: the tree read renders to another program");
            if got != tree {
                // Rows spelled alike in every index class (`zero`, `prev`): a tree that differs only in the class of
                // such an index under `ival` is the same program. Anything else is a reader bug.
                assert!(only_index_classes_differ(&k, &got, &tree), "{name} gene {g}: {} against {}", sexp(&k, &got), sexp(&k, &tree));
                same_text += 1;
            }
        }
        eprintln!("{name}: 400 random genes ({nodes} nodes) read back to the same program; {same_text} of them to another tree of the same text (the class of an index under `ival`)");
    }
}

#[test]
fn the_published_yardsticks_read_to_the_reference_or_name_their_gap() {
    for name in PROBLEMS {
        let p = problem(name).unwrap();
        let (k, sig) = setup(&p);
        let r = Reader::new(&k, &sig).unwrap();
        let reference = build(&k, p.reference).unwrap();
        match r.read_program(p.yardstick) {
            Ok(t) => assert_eq!(sexp(&k, &t), sexp(&k, &reference), "{name}: the published program reads to another program"),
            Err(g) => {
                // Pathfinder's published kernel uses `map5`, `rotate` and a tuple-parameter `min`: constructs the
                // signature kingdom does not have (its reference gene spells the same algorithm with `tab`).
                assert_eq!(name, "pathfinder", "{name}: the published program is a gap: {g}");
                eprintln!("pathfinder's published program is a gap of the kingdom: {g}");
            }
        }
    }
}

#[test]
fn hand_written_forms_read_to_the_kingdoms_rows() {
    let p = problem("histogram").unwrap();
    let (k, sig) = setup(&p);
    let r = Reader::new(&k, &sig).unwrap();
    let cases = [
        ("entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : []i32 = hist (+) 0 (i64.i32 k) (map (\\i -> i64.i32 (i % k)) is) vs", p.reference),
        ("entry main [m] (bins: i32) (idx: [m]i32) (val: [m]i32) : []i32 =\n  let k64 = i64.i32 bins\n  let b = map (\\x -> x % bins) idx\n  in reduce_by_index (replicate k64 0) (+) 0 (map i64.i32 b) val", p.reference),
        ("entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : []i32 = hist (+) 0 (i64.i32 k) (map (\\i -> i64.i32 i % i64.i32 k) is) vs", p.reference),
    ];
    for (src, want) in cases {
        let t = r.read_program(src).unwrap_or_else(|g| panic!("{g}\n{src}"));
        assert_eq!(sexp(&k, &t), sexp(&k, &build(&k, want).unwrap()), "{src}");
    }
    let gaps = [
        ("entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : []i32 = hist (+) 0 (i64.i32 k) (map (\\i -> i64.i32 (i % k)) is) (map (\\v -> v / 2) vs)", "operator /"),
        ("entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : []i32 = map (\\x -> x + 1.0) vs", "float literal"),
        ("entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : []i32 = let (a, b) = (is, vs) in a", "tuple pattern"),
    ];
    for (src, construct) in gaps {
        let g = r.read_program(src).expect_err(src);
        assert_eq!(g.construct, construct, "{src}: {g}");
    }
}
