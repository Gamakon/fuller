//! ADFs in the deprecated 1-D Futhark kingdom: the population fold by saturated
//! share-equivalence ([`crate::saturated_fold`]), the per-type write-back, the
//! unfold the interpreters and e-graphs evaluate, and the `def adf_*` rendering
//! checked against Futhark 0.28.1.
//!
//! Mirrors the WGSL kingdom's ADF tests (`wgsl/true_share.rs`) and the shared
//! core's 100,000-chromosome fold test (`typed_chromosome.rs`).

use std::collections::BTreeMap;
use std::process::Command;
use std::time::Instant;

use crate::futhark_1d_egraph::FoldEgraph as PlainFold;
use crate::futhark_1d_interp::Value;
use crate::futhark_table::{FutharkKingdom, GeneRng};
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

/// The 1-D Futhark kingdom's three roots, each with its layout at program size 3.
/// The 1-D Futhark kingdom, its contract, its `C_T` lengths, and a layout per root.
type Plain = (FutharkKingdom, KingdomContract, BTreeMap<Ty, u32>, Vec<(Ty, GeneLayout)>);

fn plain() -> Plain {
    let k = FutharkKingdom::load_1d().expect("the 1-D tables load");
    let c = k.relations().check().expect("the contract");
    let cl = k.const_len();
    let layouts = ["arr.f32", "scn.f32", "f32.real"].iter().map(|r| {
        let root = k.ty(r).expect("root");
        (root, GeneLayout::from_program_size(&c, root, 3).expect("layout"))
    });
    let layouts = layouts.collect();
    (k, c, cl, layouts)
}

/// One population: random genes, their trees, the fold.
fn fold_population(table: &mut AdfTable, eg: &dyn EgraphKingdom, c: &KingdomContract, layout: &GeneLayout, cl: &BTreeMap<Ty, u32>, n: usize, rng: &mut GeneRng) -> (Vec<Tree>, SaturatedFold) {
    let genes: Vec<Vec<u32>> = (0..n).map(|_| random_gene(c, layout, cl, rng)).collect();
    let trees: Vec<Tree> = genes.iter().map(|g| express(c, layout, g, cl).expect("a random gene expresses").tree()).collect();
    let f = fold_saturated(table, eg, c, layout, cl, &genes, &trees, &Budget::default()).unwrap_or_else(|e| panic!("{e}"));
    assert_fold(&f, layout, cl);
    eprintln!(
        "  population of {n}: {} distinct, tuples {} -> {} in {} iterations{}, {} ADFs (structural {}), {} refitted to their layout ({} subtrees took their joint form); structural {:.0} ms, saturate {:.0} ms, extract {:.0} ms, write-back {:.0} ms",
        f.distinct, f.tuples_asserted, f.tuples_saturated, f.iterations, if f.stopped_by_size { " (size guard)" } else { "" }, f.used.len(), f.structural.adfs, f.refitted, f.refit_adopted,
        f.timings[0].as_secs_f64() * 1e3, f.timings[1].as_secs_f64() * 1e3, f.timings[2].as_secs_f64() * 1e3, f.timings[3].as_secs_f64() * 1e3
    );
    (trees, f)
}

const GENES: usize = 100_000;
const POPULATION: usize = 1_000;

/// Fold, write-back, unfold on 100,000 random genes of the 1-D Futhark kingdom (100 populations of 1,000, a third per
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

/// A second fold of the same population reuses every id; a new population only appends.
#[test]
fn adf_ids_are_stable_across_folds() {
    let (k, c, cl, layouts) = plain();
    let layout = layouts[0].1.clone();
    let eg = PlainFold { kingdom: &k };
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

/// The comparison of the two folds on the same populations: the shared parts each finds, the DAG the population
/// needs (distinct nodes, and the functions among them: the cost both minimise), and what became of every part
/// structural hashing shares. Saturation only adds equivalences and the joint extraction minimises the DAG's
/// function count, so the saturated DAG is never more expensive than the submitted one; a population where it is
/// would be a bug.
fn compare(name: &str, eg: &dyn EgraphKingdom, c: &KingdomContract, layouts: &[GeneLayout], cl: &BTreeMap<Ty, u32>, seed: u64) {
    let mut rng = GeneRng::new(seed);
    let mut table = AdfTable::default();
    eprintln!("{name}: population | parts structural / saturated | DAG nodes structural / saturated | DAG functions structural / saturated | structural parts: matched merged collapsed unshared unreached | refitted, size guard");
    let mut sum = [0u64; 6];
    for p in 0..10 {
        let layout = &layouts[p % layouts.len()];
        let (_, f) = fold_population(&mut table, eg, c, layout, cl, POPULATION, &mut rng);
        let s = f.fate;
        eprintln!(
            "  {p:>2} | {:>4} / {:>4} | {:>5} / {:>5} | {:>5} / {:>5} | {:>4} {:>4} {:>4} {:>4} {:>4} | {}, {}",
            f.structural.adfs, f.used.len(), f.structural_dag.0, f.dag.0, f.structural_dag.1, f.dag.1, s.matched, s.merged, s.collapsed, s.unshared, s.unreached, f.refitted, f.stopped_by_size
        );
        assert_eq!(s.matched + s.merged + s.collapsed + s.unshared + s.unreached, f.structural.adfs);
        assert!(f.dag.1 <= f.structural_dag.1, "{name} population {p}: the saturated DAG costs {} functions, the submitted one {}", f.dag.1, f.structural_dag.1);
        for (i, v) in [f.structural.adfs, f.used.len(), f.structural_dag.0, f.dag.0, f.structural_dag.1 as usize, f.dag.1 as usize].iter().enumerate() {
            sum[i] += *v as u64;
        }
    }
    eprintln!("  all | {} / {} | {} / {} | {} / {}", sum[0], sum[1], sum[2], sum[3], sum[4], sum[5]);
}

#[test]
fn the_saturated_dag_is_never_larger_than_the_structural_one() {
    let (k, c, cl, layouts) = plain();
    let layouts: Vec<GeneLayout> = layouts.into_iter().map(|(_, l)| l).collect();
    compare("1-D", &PlainFold { kingdom: &k }, &c, &layouts, &cl, 0xADF0_0006);
}
