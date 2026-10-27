//! ADFs in the two Futhark kingdoms: the population fold by saturated
//! share-equivalence ([`crate::saturated_fold`]), the per-type write-back, the
//! unfold the interpreters and e-graphs evaluate, and the `def adf_*` rendering
//! checked against Futhark 0.28.1.
//!
//! Mirrors the WGSL kingdom's ADF tests (`wgsl/true_share.rs`) and the shared
//! core's 100,000-chromosome fold test (`typed_chromosome.rs`).

use std::collections::BTreeMap;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::futhark_egraph::FoldEgraph as PlainFold;
use crate::futhark_interp::Value;
use crate::futhark_kingdom::{FutharkKingdom, GeneRng};
use crate::futhark_shapes::program_folded;
use crate::futhark_shapes_egraph::{Bridge, FoldEgraph as ShapeFold};
use crate::futhark_shapes_evolve::suite;
use crate::futhark_shapes_interp::{format_matrix, identical, parse_matrix, MValue, Mat};
use crate::geneframe::Ty;
use crate::kingdom_contract::KingdomContract;
use crate::population_fold::AdfTable;
use crate::saturated_fold::{fold_saturated, Budget, EgraphKingdom, SaturatedFold};
use crate::typed_chromosome::{AdfSet, Epoch, RefKind};
use crate::typed_gep::{express, validate, GeneLayout, Tree};

fn random_gene(c: &KingdomContract, layout: &GeneLayout, cl: &BTreeMap<Ty, u32>, rng: &mut GeneRng) -> Vec<u32> {
    (0..layout.width)
        .map(|pos| {
            let a = layout.alphabet(c, pos, cl);
            a[rng.below(a.len())]
        })
        .collect()
}

/// An ADF body holds no genic terminal and references only strictly lower levels; `AdfSet::new` enforces it, and
/// this checks it again from the rows.
fn assert_bodies(epoch: &Epoch, adfs: &AdfSet) {
    for (id, level) in adfs.schedule() {
        let a = adfs.get(id).expect("a scheduled ADF has a body");
        assert_eq!(a.level, level);
        for n in &a.nodes {
            match epoch.reference(n.symbol) {
                Some(RefKind::Genic { gene_id }) => panic!("ADF {id} holds the genic terminal G{gene_id}"),
                Some(RefKind::Adf { adf_id }) => {
                    let l = adfs.get(adf_id).expect("a referenced ADF exists").level;
                    assert!(l < a.level, "ADF {id} at level {} references ADF {adf_id} at level {l}", a.level);
                }
                None => assert!((n.symbol as usize) < epoch.n_base(), "ADF {id} holds a symbol outside the alphabet"),
            }
        }
    }
}

/// What every fold must give back: legal genes over the epoch alphabet, bodies with no genic terminal and only
/// lower references, and folded trees that unfold to their extracted forms.
fn assert_fold(f: &SaturatedFold, layout: &GeneLayout, cl: &BTreeMap<Ty, u32>) {
    assert_bodies(&f.epoch, &f.adfs);
    let bodies = f.adfs.expanded(&f.epoch);
    for (i, (g, t)) in f.genes.iter().zip(&f.folded).enumerate() {
        validate(f.epoch.kingdom(), layout, g, cl).unwrap_or_else(|e| panic!("{i}: the folded gene is not legal: {e}"));
        assert_eq!(&express(f.epoch.kingdom(), layout, g, cl).expect("express").tree(), t, "{i}: the gene does not express its folded tree");
        assert_eq!(crate::typed_chromosome::unfold_with(&f.epoch, &bodies, t), f.forms[i], "{i}: unfold");
    }
}

fn same_value(a: &Value, b: &Value) -> bool {
    let bits = |x: &f32, y: &f32| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
    match (a, b) {
        (Value::Real(x), Value::Real(y)) => bits(x, y),
        (Value::Arr(x), Value::Arr(y)) => x.len() == y.len() && x.iter().zip(y).all(|(x, y)| bits(x, y)),
        _ => false,
    }
}

fn same_matrix(a: &Mat, b: &Mat) -> bool {
    a.rows == b.rows && a.cols == b.cols && a.data.iter().zip(&b.data).all(|(x, y)| identical(*x, *y))
}

/// The 1-D kingdom's three roots, each with its layout at program size 3.
/// The 1-D kingdom, its contract, its `C_T` lengths, and a layout per root.
type Plain = (FutharkKingdom, KingdomContract, BTreeMap<Ty, u32>, Vec<(Ty, GeneLayout)>);

fn plain() -> Plain {
    let k = FutharkKingdom::load().expect("the 1-D tables load");
    let c = k.relations().check().expect("the contract");
    let cl = k.const_len();
    let layouts = ["arr.f32", "scn.f32", "f32.real"].iter().map(|r| {
        let root = k.ty(r).expect("root");
        (root, GeneLayout::from_program_size(&c, root, 3).expect("layout"))
    });
    let layouts = layouts.collect();
    (k, c, cl, layouts)
}

fn shapes() -> (FutharkKingdom, Bridge, KingdomContract, BTreeMap<Ty, u32>, Ty, GeneLayout) {
    let k = FutharkKingdom::load_shapes().expect("the shape tables load");
    let c = k.relations().check().expect("the contract");
    let cl = k.const_len();
    let root = k.ty("mat.n.p").expect("root");
    let layout = GeneLayout::from_program_size(&c, root, 3).expect("layout");
    (k, Bridge::load(), c, cl, root, layout)
}

/// One population: random genes, their trees, the fold.
fn fold_population(table: &mut AdfTable, eg: &dyn EgraphKingdom, c: &KingdomContract, layout: &GeneLayout, cl: &BTreeMap<Ty, u32>, n: usize, rng: &mut GeneRng) -> (Vec<Tree>, SaturatedFold) {
    let genes: Vec<Vec<u32>> = (0..n).map(|_| random_gene(c, layout, cl, rng)).collect();
    let trees: Vec<Tree> = genes.iter().map(|g| express(c, layout, g, cl).expect("a random gene expresses").tree()).collect();
    let f = fold_saturated(table, eg, c, layout, cl, &genes, &trees, &Budget::default()).unwrap_or_else(|e| panic!("{e}"));
    assert_fold(&f, layout, cl);
    eprintln!(
        "  population of {n}: {} distinct, tuples {} -> {} in {} iterations{}, {} ADFs (structural {}), {} kept their spelling; structural {:.0} ms, saturate {:.0} ms, extract {:.0} ms, write-back {:.0} ms",
        f.distinct, f.tuples_asserted, f.tuples_saturated, f.iterations, if f.stopped_by_size { " (size guard)" } else { "" }, f.used.len(), f.structural.adfs, f.kept,
        f.timings[0].as_secs_f64() * 1e3, f.timings[1].as_secs_f64() * 1e3, f.timings[2].as_secs_f64() * 1e3, f.timings[3].as_secs_f64() * 1e3
    );
    (trees, f)
}

const GENES: usize = 100_000;
const POPULATION: usize = 1_000;

/// Fold, write-back, unfold on 100,000 random genes of the 1-D kingdom (100 populations of 1,000, a third per
/// root): every value of the folded population, unfolded and interpreted, is bit for bit the submitted gene's.
#[test]
fn plain_fold_write_back_unfold_keeps_every_value_on_100000_genes() {
    let started = Instant::now();
    let (k, c, cl, layouts) = plain();
    let eg = PlainFold { kingdom: &k };
    let input: Vec<f32> = (0..16).map(|i| (i as f32 - 5.5) * 0.375).collect();
    let mut rng = GeneRng::new(0xADF0_0001);
    let mut table = AdfTable::default();
    let (mut adfs, mut rewritten, mut reformed, mut refs, mut structural, mut compared) = (0, 0, 0, 0, 0, 0);
    for p in 0..GENES / POPULATION {
        let (_, layout) = &layouts[p % layouts.len()];
        let (trees, f) = fold_population(&mut table, &eg, &c, layout, &cl, POPULATION, &mut rng);
        for (i, (t, folded)) in trees.iter().zip(&f.folded).enumerate() {
            let want = k.eval(t, &input).unwrap_or_else(|e| panic!("population {p}, {i}: {e}"));
            let got = k.eval_folded(&f.epoch, &f.adfs, folded, &input).unwrap_or_else(|e| panic!("population {p}, {i}: {e}"));
            assert!(same_value(&want, &got), "population {p}, individual {i}: {want:?} folded gives {got:?}");
            compared += 1;
        }
        adfs += f.used.len();
        structural += f.structural.adfs;
        rewritten += f.rewritten;
        reformed += f.reformed;
        refs += f.references;
    }
    eprintln!("1-D: {compared} genes, {adfs} ADFs (structural hashing: {structural}), {rewritten} genes rewritten, {reformed} reformed by the e-graph, {refs} references, {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!(compared, GENES);
    assert!(adfs > 0 && refs > 0, "the fold made no ADF: the test exercised nothing");
}

/// The same on 100,000 random genes of the shape kingdom.
#[test]
fn shape_fold_write_back_unfold_keeps_every_value_on_100000_genes() {
    let started = Instant::now();
    let (k, bridge, c, cl, _, layout) = shapes();
    let eg = ShapeFold { kingdom: &k, bridge: &bridge };
    let sizes: BTreeMap<String, usize> = [("m", 3usize), ("n", 2), ("p", 4)].iter().map(|(r, s)| (r.to_string(), *s)).collect();
    let inputs = suite(&k, &sizes, 1).remove(0);
    let mut rng = GeneRng::new(0xADF0_0002);
    let mut table = AdfTable::default();
    let (mut adfs, mut rewritten, mut reformed, mut refs, mut structural, mut compared) = (0, 0, 0, 0, 0, 0);
    for p in 0..GENES / POPULATION {
        let (trees, f) = fold_population(&mut table, &eg, &c, &layout, &cl, POPULATION, &mut rng);
        for (i, (t, folded)) in trees.iter().zip(&f.folded).enumerate() {
            let MValue::Mat(want) = k.eval_matrices(t, &inputs).unwrap_or_else(|e| panic!("population {p}, {i}: {e}")) else { panic!("not a matrix") };
            let MValue::Mat(got) = k.eval_matrices_folded(&f.epoch, &f.adfs, folded, &inputs).unwrap_or_else(|e| panic!("population {p}, {i}: {e}")) else { panic!("not a matrix") };
            assert!(same_matrix(&want, &got), "population {p}, individual {i}: the folded value differs");
            compared += 1;
        }
        adfs += f.used.len();
        structural += f.structural.adfs;
        rewritten += f.rewritten;
        reformed += f.reformed;
        refs += f.references;
    }
    eprintln!("shapes: {compared} genes, {adfs} ADFs (structural hashing: {structural}), {rewritten} genes rewritten, {reformed} reformed by the e-graph, {refs} references, {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!(compared, GENES);
    assert!(adfs > 0 && refs > 0, "the fold made no ADF: the test exercised nothing");
}

/// A second fold of the same population reuses every id; a new population only appends.
#[test]
fn adf_ids_are_stable_across_folds() {
    let (k, bridge, c, cl, _, layout) = shapes();
    let eg = ShapeFold { kingdom: &k, bridge: &bridge };
    let mut rng = GeneRng::new(0xADF0_0003);
    let mut table = AdfTable::default();
    let genes: Vec<Vec<u32>> = (0..200).map(|_| random_gene(&c, &layout, &cl, &mut rng)).collect();
    let trees: Vec<Tree> = genes.iter().map(|g| express(&c, &layout, g, &cl).expect("express").tree()).collect();
    let a = fold_saturated(&mut table, &eg, &c, &layout, &cl, &genes, &trees, &Budget::default()).expect("fold");
    let b = fold_saturated(&mut table, &eg, &c, &layout, &cl, &genes, &trees, &Budget::default()).expect("fold again");
    assert!(!a.used.is_empty());
    assert_eq!(a.used.iter().map(|u| u.0).collect::<Vec<_>>(), b.used.iter().map(|u| u.0).collect::<Vec<_>>(), "a second fold reuses every id");
    assert_eq!(a.genes, b.genes, "the fold is deterministic");
    let before = table.next;
    let g2: Vec<Vec<u32>> = (0..200).map(|_| random_gene(&c, &layout, &cl, &mut rng)).collect();
    let t2: Vec<Tree> = g2.iter().map(|g| express(&c, &layout, g, &cl).expect("express").tree()).collect();
    fold_saturated(&mut table, &eg, &c, &layout, &cl, &g2, &t2, &Budget::default()).expect("new population");
    assert!(table.next >= before && table.index.values().all(|id| *id < table.next), "ids only append");
}

/// The e-graph proves `transpose (transpose A) = A`, so a gene spelling the double transpose and a gene spelling
/// `A` share one part that structural hashing cannot see.
#[test]
fn the_saturated_fold_shares_what_structural_hashing_cannot() {
    let (k, bridge, c, cl, root, _) = shapes();
    let layout = GeneLayout::from_program_size(&c, root, 8).expect("layout");
    let eg = ShapeFold { kingdom: &k, bridge: &bridge };
    let row = |name: &str, kids: Vec<Tree>| Tree { symbol: k.rows.iter().position(|r| r.name == name).unwrap_or_else(|| panic!("{name}")) as u32, const_index: None, kids };
    let a = || row("mat.n.m.input", vec![]);
    let b = || row("mat.m.p.input", vec![]);
    let product = |x: Tree| row("matmul@mat.n.mxmat.m.p", vec![x, b()]);
    let tt = row("transpose@mat.m.n", vec![row("transpose@mat.n.m", vec![a()])]);
    let add = |x: Tree, y: Tree| row("map2_rows@mat.n.p", vec![row("fn.arr2_f32_arr_f32_arr_f32.add", vec![]), x, y]);
    let trees = [add(product(a()), product(a())), add(product(tt.clone()), product(tt))];
    let mut rng = GeneRng::new(5);
    let genes: Vec<Vec<u32>> = trees.iter().map(|t| crate::typed_gep::write_back(&c, &layout, &random_gene(&c, &layout, &cl, &mut rng), t).expect("fits")).collect();
    let f = fold_saturated(&mut AdfTable::default(), &eg, &c, &layout, &cl, &genes, &trees, &Budget::default()).expect("fold");
    assert_fold(&f, &layout, &cl);
    eprintln!("structural {} ADFs, saturated {} ADFs, {} reformed", f.structural.adfs, f.used.len(), f.reformed);
    assert_eq!(f.structural.adfs, 0, "the two genes share no tree, and within a gene a repeat in one individual is not an ADF");
    assert_eq!(f.forms[0], f.forms[1], "the e-graph extracts one form for both genes");
    assert_eq!(f.reformed, 1, "the double transpose is rewritten away");
    assert!(!f.used.is_empty(), "the shared product is an ADF");
}

fn have(p: &str) -> bool {
    Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

const COMPILED: usize = 200;

/// The rendered program with ADFs type-checks and compiles with `futhark c`, and its outputs equal the
/// interpreter's on the submitted tree bit for bit, on 200 folded genes holding a reference, a third per root.
#[test]
fn plain_programs_with_adfs_compile_and_agree_bit_for_bit() {
    assert!(have("futhark") && have("cc"), "this test needs futhark 0.28.1 and a C compiler");
    let started = Instant::now();
    let (k, c, cl, layouts) = plain();
    let eg = PlainFold { kingdom: &k };
    let input: Vec<f32> = (0..16).map(|i| (i as f32 - 5.5) * 0.375).collect();
    let dir = std::env::temp_dir().join("futhark_adf_plain").join(format!("t{}", std::process::id()));
    let mut rng = GeneRng::new(0xADF0_0004);
    let mut table = AdfTable::default();
    let mut checked = 0usize;
    let mut shown = false;
    'roots: for (r, (root, layout)) in layouts.iter().enumerate() {
        let want_here = COMPILED * (r + 1) / layouts.len();
        loop {
            let (trees, f) = fold_population(&mut table, &eg, &c, layout, &cl, 300, &mut rng);
            for (t, folded) in trees.iter().zip(&f.folded) {
                if checked >= want_here {
                    continue 'roots;
                }
                if count_refs(&f.epoch, folded) == 0 {
                    continue;
                }
                let source = k.plain_program_folded(&f.epoch, &f.adfs, *root, folded).expect("renders");
                if !shown {
                    eprintln!("{source}");
                    shown = true;
                }
                let scalar = k.dual(*root).is_some_and(|d| d.slot == "f32");
                let got = crate::futhark_exec::compile_and_run(&source, scalar, &input, &dir).unwrap_or_else(|e| panic!("{e}\n{source}"));
                let want = match k.eval(t, &input).expect("the interpreter") {
                    Value::Arr(a) => a,
                    Value::Real(x) => vec![x],
                    Value::Fun(_) => panic!("a function root"),
                };
                assert_eq!(got.len(), want.len(), "length\n{source}");
                for (j, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "element {j}: futhark {a}, interpreter {b}\n{source}");
                }
                checked += 1;
            }
        }
    }
    eprintln!("1-D: {checked} folded programs compiled and agreed bit for bit, {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!(checked, COMPILED);
}

fn count_refs(epoch: &Epoch, t: &Tree) -> usize {
    usize::from(matches!(epoch.reference(t.symbol), Some(RefKind::Adf { .. }))) + t.kids.iter().map(|k| count_refs(epoch, k)).sum::<usize>()
}

/// The same for the shape kingdom, through Futhark's textual format.
#[test]
fn shape_programs_with_adfs_compile_and_agree_bit_for_bit() {
    assert!(have("futhark") && have("cc"), "this test needs futhark 0.28.1 and a C compiler");
    let started = Instant::now();
    let (k, bridge, c, cl, root, layout) = shapes();
    let eg = ShapeFold { kingdom: &k, bridge: &bridge };
    let sizes: BTreeMap<String, usize> = [("m", 3usize), ("n", 2), ("p", 4)].iter().map(|(r, s)| (r.to_string(), *s)).collect();
    let inputs = suite(&k, &sizes, 1).remove(0);
    let mut text = String::new();
    for d in k.duals.iter().filter(|d| d.slot == "mat") {
        text.push_str(&format_matrix(&inputs[&k.ty(&d.dual).expect("dual")]));
    }
    let dir = std::env::temp_dir().join("futhark_adf_shapes").join(format!("t{}", std::process::id()));
    let mut rng = GeneRng::new(0xADF0_0005);
    let mut table = AdfTable::default();
    let mut checked = 0usize;
    let mut shown = false;
    while checked < COMPILED {
        let (trees, f) = fold_population(&mut table, &eg, &c, &layout, &cl, 300, &mut rng);
        for (t, folded) in trees.iter().zip(&f.folded) {
            if checked >= COMPILED || count_refs(&f.epoch, folded) == 0 {
                continue;
            }
            let source = program_folded(&k, root, &f.epoch, &f.adfs, folded, &sizes).expect("renders");
            if !shown {
                eprintln!("{source}");
                shown = true;
            }
            let out = crate::futhark_exec::run_text_program(&source, &text, &dir, Duration::from_secs(120)).unwrap_or_else(|e| panic!("{e}\n{source}"));
            let got = parse_matrix(&out).unwrap_or_else(|e| panic!("{e}\n{out}"));
            let MValue::Mat(want) = k.eval_matrices(t, &inputs).expect("the interpreter") else { panic!("not a matrix") };
            assert!(same_matrix(&got, &want), "futhark and the interpreter differ\n{source}\nfuthark {got:?}\ninterpreter {want:?}");
            checked += 1;
        }
    }
    eprintln!("shapes: {checked} folded programs compiled and agreed bit for bit, {:.1} s", started.elapsed().as_secs_f64());
}
