//! Saturated share-equivalence against structural hashing, on a population
//! where the same part is written in different ways.
//!
//! The population: random genes of the shape kingdom, each with a variant
//! that spells part of it differently by one of the e-graph's own rules
//! (`X . Y` as `(Y' . X')'`, `X` as `(X')'`, `X` as `map id X`), and the A7
//! kernel (the product computed directly and the transposed way) beside the
//! product computed once. Every gene is folded both ways; then two genes that
//! call one `def adf_*` neither spelled the same way are rendered, compiled
//! with `futhark c`, and checked against the interpreter bit for bit.
//!
//! ```text
//! cargo run --release --example futhark_adf_saturated
//! ```

use std::collections::BTreeMap;
use std::time::Duration;

use fuller::futhark_exec::run_text_program;
use fuller::futhark_kingdom::{FutharkKingdom, GeneRng};
use fuller::futhark_shapes::program_folded;
use fuller::futhark_shapes_egraph::{Bridge, FoldEgraph};
use fuller::futhark_shapes_evolve::{product_twice, suite};
use fuller::futhark_shapes_interp::{format_matrix, identical, parse_matrix, MValue};
use fuller::population_fold::AdfTable;
use fuller::saturated_fold::{fold_saturated, Budget};
use fuller::typed_chromosome::RefKind;
use fuller::typed_gep::{express, write_back, GeneLayout, Tree};

fn row(k: &FutharkKingdom, name: &str, kids: Vec<Tree>) -> Tree {
    let symbol = k.rows.iter().position(|r| r.name == name).unwrap_or_else(|| panic!("no row {name}")) as u32;
    Tree { symbol, const_index: None, kids }
}

/// The class `a.b` of a matrix row's output, or `None` for a row function.
fn class_of(k: &FutharkKingdom, t: &Tree) -> Option<String> {
    let d = k.dual(k.rows[t.symbol as usize].output)?;
    (d.slot == "mat").then(|| d.form.clone())
}

fn flip(class: &str) -> String {
    let (a, b) = class.split_once('.').expect("a matrix class has two roles");
    format!("{b}.{a}")
}

/// `t` with one of the e-graph's rules applied backwards at the `at`-th matrix node (pre-order): an equal tree,
/// spelled differently.
fn respell(k: &FutharkKingdom, t: &Tree, at: &mut usize, rule: usize) -> Tree {
    let kids: Vec<Tree> = t.kids.iter().map(|c| respell(k, c, at, rule)).collect();
    let here = Tree { symbol: t.symbol, const_index: t.const_index, kids };
    let Some(cls) = class_of(k, &here) else { return here };
    if *at != 0 {
        *at -= 1;
        return here;
    }
    *at = usize::MAX; // done: no second rewrite
    let name = k.rows[here.symbol as usize].name.clone();
    match rule {
        // X . Y  ->  (Y' . X')'
        0 if name.starts_with("matmul@") => {
            let (x, y) = (here.kids[0].clone(), here.kids[1].clone());
            let (xc, yc) = (class_of(k, &x).expect("a matrix"), class_of(k, &y).expect("a matrix"));
            let inner = row(k, &format!("matmul@mat.{}xmat.{}", flip(&yc), flip(&xc)), vec![row(k, &format!("transpose@mat.{yc}"), vec![y]), row(k, &format!("transpose@mat.{xc}"), vec![x])]);
            row(k, &format!("transpose@mat.{}", flip(&cls)), vec![inner])
        }
        // X  ->  (X')'
        1 => row(k, &format!("transpose@mat.{}", flip(&cls)), vec![row(k, &format!("transpose@mat.{cls}"), vec![here])]),
        // X  ->  map id X
        _ => row(k, &format!("map_rows@mat.{cls}"), vec![row(k, "fn.arr_f32_arr_f32.id", vec![]), here]),
    }
}

fn matrix_nodes(k: &FutharkKingdom, t: &Tree) -> usize {
    usize::from(class_of(k, t).is_some()) + t.kids.iter().map(|c| matrix_nodes(k, c)).sum::<usize>()
}

fn contains(t: &Tree, s: &Tree) -> bool {
    t == s || t.kids.iter().any(|c| contains(c, s))
}

fn refs(t: &Tree, sym: u32) -> bool {
    t.symbol == sym || t.kids.iter().any(|c| refs(c, sym))
}

fn main() {
    let k = FutharkKingdom::load_shapes().expect("the shape tables load");
    let c = k.relations().check().expect("the contract");
    let cl = k.const_len();
    let root = k.ty("mat.n.p").expect("root");
    let small = GeneLayout::from_program_size(&c, root, 3).expect("layout");
    let layout = GeneLayout::from_program_size(&c, root, 16).expect("layout");
    let mut rng = GeneRng::new(0xADF0_DE40);
    let random = |l: &GeneLayout, rng: &mut GeneRng| -> Vec<u32> {
        (0..l.width).map(|p| {
            let a = l.alphabet(&c, p, &cl);
            a[rng.below(a.len())]
        }).collect()
    };

    // The population's trees: random programs and their respellings, the A7 kernel and the product once.
    let mut trees: Vec<Tree> = Vec::new();
    for _ in 0..150 {
        let t = express(&c, &small, &random(&small, &mut rng), &cl).expect("express").tree();
        let rule = rng.below(3);
        let mut at = rng.below(matrix_nodes(&k, &t));
        let v = respell(&k, &t, &mut at, rule);
        trees.push(t);
        trees.push(v);
    }
    let a = row(&k, "mat.n.m.input", vec![]);
    let b = row(&k, "mat.m.p.input", vec![]);
    let ab = row(&k, "matmul@mat.n.mxmat.m.p", vec![a, b]);
    trees.push(product_twice(&k));
    trees.push(row(&k, "map2_rows@mat.n.p", vec![row(&k, "fn.arr2_f32_arr_f32_arr_f32.add", vec![]), ab.clone(), ab]));
    let genes: Vec<Vec<u32>> = trees.iter().map(|t| write_back(&c, &layout, &random(&layout, &mut rng), t).expect("a program of at most 16 functions fits")).collect();

    let bridge = Bridge::load();
    let eg = FoldEgraph { kingdom: &k, bridge: &bridge };
    let f = fold_saturated(&mut AdfTable::default(), &eg, &c, &layout, &cl, &genes, &trees, &Budget::default()).expect("the population folds");
    println!("== the population ==");
    println!("individuals     {} ({} distinct trees): 150 random programs, each with a respelled variant, the A7 kernel and the product once", trees.len(), f.distinct);
    println!("shared parts    structural hashing {}   saturated share-equivalence {}", f.structural.adfs, f.used.len());
    println!("DAG nodes       structural {}   saturated {}", f.structural_dag.0, f.dag.0);
    println!("DAG functions   structural {}   saturated {}", f.structural_dag.1, f.dag.1);
    let s = f.fate;
    println!("structural parts in the saturated fold: {} matched, {} merged with another spelling, {} collapsed to no function, {} no longer shared, {} unreached", s.matched, s.merged, s.collapsed, s.unshared, s.unreached);
    println!("genes           {} rewritten, {} hold a form the e-graph found, {} refitted to their layout", f.rewritten, f.reformed, f.refitted);

    // Two genes calling one ADF, one of which never spelled its body.
    let bodies = f.adfs.expanded(&f.epoch);
    let mut pick: Option<(usize, usize, u64)> = None;
    for (id, body) in bodies.iter().rev() {
        let sym = f.epoch.adf_terminal(*id).expect("a reference");
        let holders: Vec<usize> = (0..trees.len()).filter(|i| refs(&f.folded[*i], sym)).collect();
        let spelled = holders.iter().find(|i| contains(&trees[**i], body));
        let other = holders.iter().find(|i| !contains(&trees[**i], body));
        if let (Some(i), Some(j)) = (spelled, other) {
            if pick.is_none_or(|(_, _, p)| bodies[&p].kids.len() < body.kids.len()) {
                pick = Some((*i, *j, *id));
            }
        }
    }
    let Some((i, j, id)) = pick else {
        println!("no ADF is called by two genes that spell it differently");
        std::process::exit(1);
    };
    assert!(matches!(f.epoch.reference(f.epoch.adf_terminal(id).expect("ref")), Some(RefKind::Adf { .. })));
    println!("\n== adf_{id}: called by gene {i} and gene {j}, which spell it differently ==");
    println!("gene {i} as submitted:  {}", k.render(&trees[i]).expect("render"));
    println!("gene {j} as submitted:  {}", k.render(&trees[j]).expect("render"));
    println!("adf_{id}'s body:        {}", k.render(&bodies[&id]).expect("render"));

    let sizes: BTreeMap<String, usize> = [("m", 3usize), ("n", 2), ("p", 4)].iter().map(|(r, s)| (r.to_string(), *s)).collect();
    let inputs = suite(&k, &sizes, 1).remove(0);
    let mut text = String::new();
    for d in k.duals.iter().filter(|d| d.slot == "mat") {
        text.push_str(&format_matrix(&inputs[&k.ty(&d.dual).expect("dual")]));
    }
    let dir = std::env::temp_dir().join("futhark_adf_saturated");
    for g in [i, j] {
        let source = program_folded(&k, root, &f.epoch, &f.adfs, &f.folded[g], &sizes).expect("renders");
        println!("\n-- gene {g}, folded --\n{}", source.trim());
        let out = run_text_program(&source, &text, &dir, Duration::from_secs(120)).unwrap_or_else(|e| panic!("futhark c: {e}"));
        let got = parse_matrix(&out).expect("a matrix");
        let MValue::Mat(want) = k.eval_matrices(&trees[g], &inputs).expect("the interpreter") else { panic!("not a matrix") };
        let same = got.rows == want.rows && got.cols == want.cols && got.data.iter().zip(&want.data).all(|(x, y)| identical(*x, *y));
        println!("futhark c compiled and ran it; bit for bit the interpreter's value on the submitted gene: {same}");
        assert!(same);
    }
}
