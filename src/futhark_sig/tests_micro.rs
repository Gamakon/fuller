//! The study's micro benchmarks in the signature family (reduce, scan, histogram): each kingdom loads and passes its
//! contract with the declared root; the published entry and the yardstick give every stored output; the published
//! program written as a gene is exact on every case, interpreted, compiled and batched, and writes back into a gene
//! of the study's program size; random genes compile and equal the interpreter (the size input and the optional
//! families included); the planted slowdown is equal and the e-graph returns to the reference's work.

use super::*;
use crate::typed_gep::{express, write_back};
use std::process::Command;

const MICRO: [Problem; 3] = [REDUCE_PROBLEM, SCAN_PROBLEM, HISTOGRAM_PROBLEM];

/// The program size the study's launch scripts give these problems (`--program-size`): above the references' own
/// (1, 1 and 4 functions), so the search has room.
const STUDY_PROGRAM_SIZE: usize = 8;

fn have_futhark() -> bool {
    Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_sig_micro_{name}_{}", std::process::id()))
}

fn stdin_of(cases: &[Case]) -> Vec<u8> {
    cases.iter().flat_map(|c| c.stdin.clone()).collect()
}

#[test]
fn the_micro_kingdoms_load_with_their_declared_roots() {
    for (p, root, dims, sizes) in [
        (REDUCE_PROBLEM, "i32:c", vec!["n"], vec![]),
        (SCAN_PROBLEM, "a.n:c", vec!["n"], vec![]),
        (HISTOGRAM_PROBLEM, "a.k:c", vec!["k", "n"], vec!["k"]),
    ] {
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        assert_eq!(sig.dims, dims, "{}", p.name);
        assert_eq!(sig.size_inputs, sizes, "{}", p.name);
        let contract = k.relations().check().unwrap_or_else(|d| panic!("{}: the kingdom check: {:?}", p.name, &d[..d.len().min(3)]));
        assert_eq!(k.roots(), vec![k.ty(root).unwrap()], "{}", p.name);
        Interp::new(&k, &sig).unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], STUDY_PROGRAM_SIZE).unwrap();
        eprintln!("{}: {} types, {} rows; program size {STUDY_PROGRAM_SIZE}: {} words a gene, at most {} nodes", p.name, k.duals.len(), k.rows.len(), layout.width, layout.max_nodes());
    }
}

#[test]
fn the_published_micro_programs_and_yardsticks_give_the_stored_outputs() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    for (p, file) in [(REDUCE_PROBLEM, "reduce.fut"), (SCAN_PROBLEM, "scan.fut"), (HISTOGRAM_PROBLEM, "reduce_by_index.fut")] {
        let cases = load_cases(&p).unwrap();
        assert_eq!(cases.len(), 5, "{}", p.name);
        assert_eq!(cases.iter().filter(|c| c.split == Split::Held).count(), 2, "{}", p.name);
        // The published file, unchanged, its entry sum_i32.
        let b = scratch(&format!("published_{}", p.name));
        std::fs::create_dir_all(&b).unwrap();
        let exe = b.join("pub");
        let st = Command::new("futhark").arg("c").arg(corpus_dir(&p).join("published").join(file)).arg("-o").arg(&exe).status().unwrap();
        assert!(st.success(), "{}: the published program does not compile", p.name);
        let ys = compile_c(p.yardstick, &b.join("ys")).unwrap();
        for c in &cases {
            let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&exe).arg("-e").arg("sum_i32").arg("-b"), std::time::Duration::from_secs(120), Some(&c.stdin)).unwrap();
            assert!(out.status.success(), "{}: {}", p.name, String::from_utf8_lossy(&out.stderr));
            assert_eq!(read_one(&out.stdout).unwrap(), c.output, "{}: the published entry on {}", p.name, c.name);
            assert_eq!(run_exe(&ys, &c.stdin).unwrap(), c.output, "{}: the yardstick on {}", p.name, c.name);
        }
        let _ = std::fs::remove_dir_all(&b);
    }
}

#[test]
fn the_micro_reference_genes_are_exact_on_every_case_interpreted_compiled_and_batched() {
    for p in MICRO {
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let tree = build(&k, p.reference).unwrap();
        assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), p.reference.split_whitespace().collect::<String>());
        let cases = load_cases(&p).unwrap();
        for c in &cases {
            assert_eq!(interp.eval(&tree, &c.inputs).unwrap(), c.output, "{}: the interpreter on {}", p.name, c.name);
        }
        let contract = k.relations().check().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], STUDY_PROGRAM_SIZE).unwrap();
        let const_len = k.const_len();
        let mut rng = GeneRng::new(3);
        let gene = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &tree).unwrap();
        assert_eq!(express(&contract, &layout, &gene, &const_len).unwrap().tree(), tree, "{}: the write-back", p.name);
        if !have_futhark() {
            panic!("futhark 0.28.1 is needed for this test");
        }
        let src = program(&k, &sig, &tree).unwrap();
        let exe = compile_c(&src, &scratch(&format!("reference_{}", p.name))).unwrap_or_else(|e| panic!("{e}\n{src}"));
        for c in &cases {
            assert_eq!(run_exe(&exe, &c.stdin).unwrap(), c.output, "{}: the compiled reference on {}", p.name, c.name);
        }
        let batch = program_cases(&k, &sig, &tree, cases.len()).unwrap();
        let bexe = compile_c(&batch, &scratch(&format!("reference_batch_{}", p.name))).unwrap_or_else(|e| panic!("{e}\n{batch}"));
        let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin_of(&cases))).unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(read_binary(&out.stdout).unwrap(), cases.iter().map(|c| c.output.clone()).collect::<Vec<_>>(), "{}: the batched reference", p.name);
    }
}

/// Legality, measured: random genes of the study's program size express, Futhark compiles every one, and the
/// compiled program equals the interpreter on small cases, the edges of each signature included.
#[test]
fn random_micro_genes_compile_and_equal_the_interpreter() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    for p in MICRO {
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let contract = k.relations().check().unwrap();
        let layout = GeneLayout::from_program_size(&contract, k.roots()[0], STUDY_PROGRAM_SIZE).unwrap();
        let const_len = k.const_len();
        let mut rng = GeneRng::new(0x51C_0002);
        let vec_of = |n: usize, rng: &mut GeneRng| Array { dims: vec![n], data: (0..n).map(|_| rng.below(41) as i32 - 20).collect() };
        // (k, n): the histogram's bins and length; reduce and scan read only n.
        let small: Vec<Vec<Array>> = [(3usize, 7usize), (1, 1), (5, 2)]
            .iter()
            .map(|&(kb, n)| if p.name == "histogram" { vec![Array { dims: vec![], data: vec![kb as i32] }, vec_of(n, &mut rng), vec_of(n, &mut rng)] } else { vec![vec_of(n, &mut rng)] })
            .collect();
        let base = scratch(&format!("random_{}", p.name));
        let mut used = BTreeMap::<String, usize>::new();
        for g in 0..12 {
            let gene = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
            let tree = express(&contract, &layout, &gene, &const_len).unwrap().tree();
            for (t, _) in tree.level_order() {
                let op = k.rows[t.symbol as usize].name.split('@').next().unwrap_or("").split('.').next().unwrap_or("").to_string();
                *used.entry(op).or_default() += 1;
            }
            let src = program(&k, &sig, &tree).unwrap();
            let batch = program_cases(&k, &sig, &tree, small.len()).unwrap();
            let bexe = compile_c(&batch, &base.join(format!("b{g}"))).unwrap_or_else(|e| panic!("{}: gene {g}'s quality program does not compile: {e}\n{batch}", p.name));
            let refs: Vec<&Array> = small.iter().flatten().collect();
            let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&write_binary(&refs))).unwrap();
            assert!(out.status.success(), "{}: {}", p.name, String::from_utf8_lossy(&out.stderr));
            let wants: Vec<Array> = small.iter().map(|w| interp.eval(&tree, w).unwrap()).collect();
            assert_eq!(read_binary(&out.stdout).unwrap(), wants, "{}: gene {g}: compiled and interpreted differ\n{src}", p.name);
        }
        eprintln!("{}: rows used by 12 random genes: {used:?}", p.name);
        let _ = std::fs::remove_dir_all(&base);
    }
}

/// The optional families, compiled against the interpreter on purpose (a random gene may not draw them): every scan
/// and histogram operator, and the modulo of negative values, on the edges.
#[test]
fn the_scan_hist_and_mod_rows_equal_the_interpreter() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let xs = Array { dims: vec![6], data: vec![-7, 3, i32::MAX, i32::MIN, 0, 5] };
    let scan = load(&SCAN).unwrap();
    let ssig = signature(&SCAN).unwrap();
    let si = Interp::new(&scan, &ssig).unwrap();
    // Every scan of a scan (Futhark 0.28.1's fusion breaks on one unless the scan's array is `opaque`), and a scan
    // of a map of one.
    let ops = ["add", "min", "max"];
    let mut texts: Vec<String> = ops.iter().map(|o| format!("(scan.{o} xs)")).collect();
    texts.extend(ops.iter().flat_map(|a| ops.iter().map(move |b| format!("(scan.{a} (scan.{b} xs))"))));
    texts.push("(scan.add (map2.add xs (scan.min xs)))".into());
    for text in &texts {
        let tree = build(&scan, text).unwrap();
        let exe = compile_c(&program(&scan, &ssig, &tree).unwrap(), &scratch("scan_rows")).unwrap();
        assert_eq!(run_exe(&exe, &write_binary(&[&xs])).unwrap(), si.eval(&tree, std::slice::from_ref(&xs)).unwrap(), "{text}");
    }
    let hist = load(&HISTOGRAM).unwrap();
    let hsig = signature(&HISTOGRAM).unwrap();
    let hi = Interp::new(&hist, &hsig).unwrap();
    let vs = Array { dims: vec![6], data: vec![1, -2, 3, 4, i32::MAX, 6] };
    for kb in [1, 4, 9] {
        let inputs = [Array { dims: vec![], data: vec![kb] }, xs.clone(), vs.clone()];
        for text in [
            "(hist.add.n (tab (mod.k (get.n is j1))) vs)",
            "(hist.min.n (tab (mod.k (get.n is j1))) vs)",
            "(hist.max.n is vs)",
            "(hist.add.n vs is)",
            "(tab (mod.k (get.k (hist.add.n is vs) j1)))",
        ] {
            let tree = build(&hist, text).unwrap();
            let exe = compile_c(&program(&hist, &hsig, &tree).unwrap(), &scratch("hist_rows")).unwrap();
            let refs: Vec<&Array> = inputs.iter().collect();
            assert_eq!(run_exe(&exe, &write_binary(&refs)).unwrap(), hi.eval(&tree, &inputs).unwrap(), "k {kb}: {text}");
        }
    }
}

#[test]
fn the_micro_planted_slowdowns_are_equal_and_the_egraph_recovers_the_reference() {
    for p in MICRO {
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let interp = Interp::new(&k, &sig).unwrap();
        let planted = build(&k, p.planted).unwrap();
        let reference = build(&k, p.reference).unwrap();
        let cases = load_cases(&p).unwrap();
        for c in &cases {
            assert_eq!(interp.eval(&planted, &c.inputs).unwrap(), c.output, "{}: the planted slowdown on {}", p.name, c.name);
        }
        let sizes = sig.sizes(&cases[2].inputs).unwrap();
        let (wp, wr) = (interp.work(&planted, &sizes).unwrap(), interp.work(&reference, &sizes).unwrap());
        assert!(wp > wr, "{}: planted {wp}, reference {wr}", p.name);
        let bridge = Bridge::load(&p.instance).unwrap();
        let vs = variants(&p.instance, &k, &bridge, &planted, 3, 32, 64).unwrap();
        for v in &vs {
            for c in &cases {
                assert_eq!(interp.eval(v, &c.inputs).unwrap(), c.output, "{}: the variant {} on {}", p.name, sexp(&k, v), c.name);
            }
        }
        let small = vs.iter().map(|v| interp.work(v, &sizes).unwrap()).min().unwrap_or(wp);
        assert!(small <= wr, "{}: the smallest variant's work {small}, the reference's {wr}", p.name);
    }
}
