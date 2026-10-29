//! The signature family's checks: the kingdom loads and passes its contract; the published program and the
//! yardstick agree with the published outputs; the reference written as a gene is exact on every stored case, by
//! the interpreter and compiled; random genes are legal (they express, compile with `futhark c`, and the compiled
//! program equals the interpreter); the e-graph's variants are equal programs; a tree writes back to a gene.

use super::*;
use crate::typed_gep::{express, write_back};
use std::process::Command;

fn have_futhark() -> bool {
    Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_sig_{name}_{}", std::process::id()))
}

fn functions(k: &FutharkKingdom, t: &Tree) -> usize {
    usize::from(!k.rows[t.symbol as usize].inputs.is_empty()) + t.kids.iter().map(|c| functions(k, c)).sum::<usize>()
}

#[test]
fn the_kingdom_loads_and_the_layout_is_reported() {
    let k = load(&PATHFINDER).unwrap();
    let sig = signature(&PATHFINDER).unwrap();
    assert_eq!(sig.dims, vec!["rows".to_string(), "cols".to_string()]);
    assert_eq!(sig.inputs, vec![Param { name: "walls".into(), elem: "i32".into(), dims: vec!["rows".into(), "cols".into()] }]);
    let contract = k.relations().check().unwrap_or_else(|d| panic!("the kingdom check: {:?}", &d[..d.len().min(3)]));
    let root = k.ty("a.cols:c").unwrap();
    assert_eq!(k.roots(), vec![root]);
    Interp::new(&k, &sig).unwrap();
    for n in [8usize, 13, 20] {
        let layout = GeneLayout::from_program_size(&contract, root, n).unwrap();
        eprintln!("pathfinder: program size {n}: {} words a gene, {} frames, at most {} nodes", layout.width, layout.frame_rows(), layout.max_nodes());
    }
}

#[test]
fn the_published_program_and_the_yardstick_give_the_published_outputs() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let dir = corpus_dir(&PATHFINDER_PROBLEM);
    let pub_dir = dir.join("published");
    // medium.out is in Futhark's text format: one `[cols]i32` array, then a `--` comment.
    let text = std::fs::read_to_string(pub_dir.join("medium.out")).unwrap();
    let text = text.split("--").next().unwrap();
    let data: Vec<i32> = text.trim().trim_start_matches('[').trim_end_matches(']').split(',').map(|x| x.trim().trim_end_matches("i32").parse().unwrap()).collect();
    let medium_out = Array { dims: vec![data.len()], data };
    assert_eq!(medium_out.dims, vec![100_000]);
    // The published program, unchanged, on its own data file.
    let b = scratch("published");
    std::fs::create_dir_all(&b).unwrap();
    let exe = b.join("pub");
    let st = Command::new("futhark").arg("c").arg(pub_dir.join("pathfinder.fut")).arg("-o").arg(&exe).status().unwrap();
    assert!(st.success());
    let got = run_exe(&exe, &std::fs::read(pub_dir.join("medium.in")).unwrap()).unwrap();
    assert_eq!(got, medium_out, "the published program does not give medium.out");
    // The yardstick (the published kernel on given walls) on the published walls.
    let sw = b.join("sw");
    let st = Command::new("futhark").arg("c").arg(dir.join("sobol_walls.fut")).arg("-o").arg(&sw).status().unwrap();
    assert!(st.success());
    let walls = crate::futhark_exec::output_with_timeout(Command::new(&sw).arg("-b"), std::time::Duration::from_secs(120), Some("100000i64 100i64")).unwrap().stdout;
    let ys = compile_c(PATHFINDER_PROBLEM.yardstick, &b.join("ys")).unwrap();
    assert_eq!(run_exe(&ys, &walls).unwrap(), medium_out, "the yardstick does not give medium.out on the published walls");
    // And the reference gene, by the interpreter, on the published walls.
    let k = load(&PATHFINDER).unwrap();
    let sig = signature(&PATHFINDER).unwrap();
    let tree = build(&k, PATHFINDER_REFERENCE).unwrap();
    let w = read_binary(&walls).unwrap();
    assert_eq!(Interp::new(&k, &sig).unwrap().eval(&tree, &w).unwrap(), medium_out, "the reference gene does not give medium.out");
    let _ = std::fs::remove_dir_all(&b);
}

#[test]
fn the_reference_gene_is_exact_on_every_case_interpreted_and_compiled() {
    let k = load(&PATHFINDER).unwrap();
    let sig = signature(&PATHFINDER).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, PATHFINDER_REFERENCE).unwrap();
    assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), PATHFINDER_REFERENCE.split_whitespace().collect::<String>());
    assert_eq!(functions(&k, &tree), 13);
    let cases = load_cases(&PATHFINDER_PROBLEM).unwrap();
    let count = |s: Split| cases.iter().filter(|c| c.split == s).count();
    assert_eq!((count(Split::Train), count(Split::Held), count(Split::Test)), (13, 6, 4), "training, held-out and test cases");
    for c in &cases {
        assert_eq!(interp.eval(&tree, &c.inputs).unwrap(), c.output, "the interpreter on {}", c.name);
        assert_eq!(compare(&c.output, &c.output), [0.0; 3]);
    }
    // Its write-back into a gene of the layout at its own program size expresses the same tree.
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], 13).unwrap();
    let const_len = k.const_len();
    let mut rng = GeneRng::new(3);
    let gene = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &tree).unwrap();
    assert_eq!(express(&contract, &layout, &gene, &const_len).unwrap().tree(), tree);
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let src = program(&k, &sig, &tree).unwrap();
    let exe = compile_c(&src, &scratch("reference")).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for c in &cases {
        assert_eq!(run_exe(&exe, &c.stdin).unwrap(), c.output, "the compiled reference on {}", c.name);
    }
    // The quality program: every case in one process, the same outputs.
    let batch = program_cases(&k, &sig, &tree, cases.len()).unwrap();
    let bexe = compile_c(&batch, &scratch("reference_batch")).unwrap_or_else(|e| panic!("{e}\n{batch}"));
    let stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
    let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin)).unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got = read_binary(&out.stdout).unwrap();
    assert_eq!(got, cases.iter().map(|c| c.output.clone()).collect::<Vec<_>>(), "the batched reference");
    // The work of the reference on 256 x 256: 13 functions and 7 terminals, the loop body run 255 times.
    let sizes: BTreeMap<String, usize> = [("rows".to_string(), 256), ("cols".to_string(), 256)].into_iter().collect();
    let w = interp.work(&tree, &sizes).unwrap();
    assert!(w > 255 * 256 * 10 && w < 255 * 256 * 20, "the reference's work on 256x256 is {w}");
}

/// Legality, measured: random genes express, Futhark compiles every one, and the compiled program equals the
/// interpreter on a small case (both edge cases of the signature included).
#[test]
fn random_genes_compile_and_the_compiled_program_equals_the_interpreter() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let k = load(&PATHFINDER).unwrap();
    let sig = signature(&PATHFINDER).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], 13).unwrap();
    let const_len = k.const_len();
    let mut rng = GeneRng::new(0x51C_0001);
    let mk = |rows: usize, cols: usize, rng: &mut GeneRng| Array { dims: vec![rows, cols], data: (0..rows * cols).map(|_| rng.below(10) as i32 - 2).collect() };
    let small = [mk(5, 7, &mut rng), mk(1, 4, &mut rng), mk(6, 1, &mut rng)];
    let base = scratch("random");
    let mut sizes_seen = Vec::new();
    for g in 0..24 {
        let gene = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
        let tree = express(&contract, &layout, &gene, &const_len).unwrap().tree();
        sizes_seen.push(tree.level_order().len());
        let src = program(&k, &sig, &tree).unwrap();
        let exe = compile_c(&src, &base.join(format!("g{g}"))).unwrap_or_else(|e| panic!("gene {g} does not compile: {e}\n{src}"));
        let mut wants = Vec::new();
        for w in &small {
            let want = interp.eval(&tree, std::slice::from_ref(w)).unwrap();
            let got = run_exe(&exe, &write_binary(&[w])).unwrap_or_else(|e| panic!("gene {g}: {e}\n{src}"));
            assert_eq!(got, want, "gene {g} on {:?}: compiled and interpreted differ\n{src}", w.dims);
            wants.push(want);
        }
        // The quality program of the same gene: the three cases in one process, the same outputs.
        let batch = program_cases(&k, &sig, &tree, small.len()).unwrap();
        let bexe = compile_c(&batch, &base.join(format!("b{g}"))).unwrap_or_else(|e| panic!("gene {g}'s quality program does not compile: {e}\n{batch}"));
        let refs: Vec<&Array> = small.iter().collect();
        let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&write_binary(&refs))).unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(read_binary(&out.stdout).unwrap(), wants, "gene {g}: the quality program differs\n{batch}");
    }
    eprintln!("random gene sizes (nodes): {sizes_seen:?}");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn the_egraph_variants_of_the_reference_are_equal_programs() {
    let k = load(&PATHFINDER).unwrap();
    let sig = signature(&PATHFINDER).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let bridge = Bridge::load(&PATHFINDER).unwrap();
    let tree = build(&k, PATHFINDER_REFERENCE).unwrap();
    let vs = variants(&PATHFINDER, &k, &bridge, &tree, 2, 16, 12).unwrap();
    assert!(vs.len() >= 4, "only {} variants", vs.len());
    let cases = load_cases(&PATHFINDER_PROBLEM).unwrap();
    for v in &vs {
        for c in &cases {
            assert_eq!(interp.eval(v, &c.inputs).unwrap(), c.output, "the variant {} on {}", sexp(&k, v), c.name);
        }
    }
}

#[test]
fn the_binary_format_round_trips() {
    let a = Array { dims: vec![2, 3], data: vec![1, -2, 3, i32::MIN, i32::MAX, 0] };
    let s = Array { dims: vec![], data: vec![7] };
    let raw = write_binary(&[&a, &s]);
    assert_eq!(read_binary(&raw).unwrap(), vec![a.clone(), s]);
    assert!(read_one(&raw).is_err());
    assert_eq!(compare(&a, &Array { dims: vec![3, 2], data: a.data.clone() }), [1.0; 3]);
    let mut b = a.clone();
    b.data[0] = 2;
    let q = compare(&b, &a);
    assert!(q[0] > 0.16 && q[0] < 0.17 && q[1] > 0.0 && q[2] > 0.0, "{q:?}");
}

/// The planted slowdown is an equal program, about twice the work, and the e-graph recovers the reference from it.
#[test]
fn the_planted_slowdown_is_equal_and_the_egraph_recovers_the_reference() {
    let k = load(&PATHFINDER).unwrap();
    let sig = signature(&PATHFINDER).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let planted = build(&k, PATHFINDER_PLANTED).unwrap();
    let reference = build(&k, PATHFINDER_REFERENCE).unwrap();
    let cases = load_cases(&PATHFINDER_PROBLEM).unwrap();
    for c in &cases {
        assert_eq!(interp.eval(&planted, &c.inputs).unwrap(), c.output, "the planted slowdown on {}", c.name);
    }
    let sizes: BTreeMap<String, usize> = [("rows".to_string(), 256), ("cols".to_string(), 256)].into_iter().collect();
    let (wp, wr) = (interp.work(&planted, &sizes).unwrap(), interp.work(&reference, &sizes).unwrap());
    assert!(wp > 2 * wr, "planted {wp}, reference {wr}");
    let bridge = Bridge::load(&PATHFINDER).unwrap();
    let vs = variants(&PATHFINDER, &k, &bridge, &planted, 3, 32, 64).unwrap();
    let small = vs.iter().map(|v| interp.work(v, &sizes).unwrap()).min().unwrap();
    assert!(small <= wr, "the smallest variant's work {small}, the reference's {wr}");
}

/// The quality inputs give partial correctness a slope: on every pathfinder case of the fitness (training and
/// held-out; a test case is never a fitness input), an output right on half the cells scores clearly better than one
/// right on none, an output off by one everywhere better still than all zero, and the answer is 0 on every input.
/// (Measured 2026-10-29 with the formula of `QUALITY_PARTS`: every fitness case clears the 0.2 margin by 0.40 or
/// more; the 3-cell test case test_300x3 by 0.168.)
#[test]
fn the_quality_inputs_give_partial_correctness_a_slope() {
    let angle = |q: &[f64]| crate::runner::staged_hff(crate::runner::FitnessStage::Quality, q, &[]).unwrap().quality;
    for c in load_cases(&PATHFINDER_PROBLEM).unwrap().into_iter().filter(|c| c.split.in_fitness()) {
        let w = &c.output;
        let with = |f: &dyn Fn(usize, i32) -> i32| Array { dims: w.dims.clone(), data: w.data.iter().enumerate().map(|(i, x)| f(i, *x)).collect() };
        assert_eq!(quality_inputs(w, w), vec![0.0; quality_input_names(w).len()], "{}: the answer is not 0", c.name);
        let zero = angle(&quality_inputs(&with(&|_, _| 0), w));
        if w.data.len() >= 2 {
            let half = angle(&quality_inputs(&with(&|i, x| if i % 2 == 0 { x } else { 0 }), w));
            assert!(half < zero - 0.2, "{}: half right {half:.4}, none right {zero:.4}", c.name);
        }
        let near = angle(&quality_inputs(&with(&|_, x| x + 1), w));
        // A margin of 0.1: on held_1x37 (walls 0..9, the answer is row 0) one off is up to 100 % of a cell.
        assert!(near < zero - 0.1, "{}: off by one {near:.4}, all zero {zero:.4}", c.name);
        assert_eq!(quality_inputs(&Array { dims: vec![w.data.len() + 1], data: vec![0; w.data.len() + 1] }, w), vec![1.0; quality_input_names(w).len()]);
    }
}

/// The tree edit distance: 0 to itself, 1 a relabel, 1 a leaf inserted, symmetric; the textbook example of Zhang and
/// Shasha (f(d(a c(b)) e) against f(c(d(a b)) e)) is 2; and on the histogram, one-symbol edits of the reference are
/// at distance 1 and the program putting every value in bin 0 is a relabel and four deletions away.
#[test]
fn the_tree_edit_distance_counts_relabels_insertions_and_deletions() {
    let l = |s: u32| Tree { symbol: s, const_index: None, kids: vec![] };
    let n = |s: u32, kids: Vec<Tree>| Tree { symbol: s, const_index: None, kids };
    let (a, b, c, d, e, f) = (0, 1, 2, 3, 4, 5);
    let t1 = n(f, vec![n(d, vec![l(a), n(c, vec![l(b)])]), l(e)]);
    let t2 = n(f, vec![n(c, vec![n(d, vec![l(a), l(b)])]), l(e)]);
    assert_eq!(tree_distance(&t1, &t1), 0);
    assert_eq!(tree_distance(&t1, &t2), 2);
    assert_eq!(tree_distance(&t2, &t1), 2);
    assert_eq!(tree_distance(&l(a), &l(b)), 1);
    assert_eq!(tree_distance(&n(a, vec![l(b)]), &n(a, vec![l(b), l(c)])), 1);
    assert_eq!(tree_distance(&Tree { symbol: a, const_index: Some(1), kids: vec![] }, &Tree { symbol: a, const_index: Some(2), kids: vec![] }), 1);
    let k = load(&HISTOGRAM).unwrap();
    let r = build(&k, HISTOGRAM_REFERENCE).unwrap();
    assert_eq!(tree_distance(&r, &build(&k, "(hist.add.n (tab (mod.k (get.n vs j1))) vs)").unwrap()), 1);
    assert_eq!(tree_distance(&r, &build(&k, "(hist.min.n (tab (mod.k (get.n is j1))) vs)").unwrap()), 1);
    assert_eq!(tree_distance(&r, &build(&k, "(hist.add.n (tab (get.n is j1)) vs)").unwrap()), 1);
    assert_eq!(tree_distance(&r, &build(&k, "(hist.add.n zeros vs)").unwrap()), 5);
}
