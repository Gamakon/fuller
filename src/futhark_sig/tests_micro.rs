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
        // (training, held-out, test): the histogram has stepping-stone training cases and a test set (plan C, G).
        let want = if p.name == "histogram" { (11, 4, 4) } else { (3, 2, 0) };
        let count = |s: Split| cases.iter().filter(|c| c.split == s).count();
        assert_eq!((count(Split::Train), count(Split::Held), count(Split::Test)), want, "{}", p.name);
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
            assert_eq!(read_one(&out.stdout).unwrap(), *c.answer().unwrap(), "{}: the published entry on {}", p.name, c.name);
            assert_eq!(run_exe(&ys, &c.stdin).unwrap(), *c.answer().unwrap(), "{}: the yardstick on {}", p.name, c.name);
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
        let tree = build(&k, p.reference().unwrap()).unwrap();
        assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), p.reference().unwrap().split_whitespace().collect::<String>());
        let cases = load_cases(&p).unwrap();
        for c in &cases {
            assert_eq!(interp.eval(&tree, &c.inputs).unwrap(), *c.answer().unwrap(), "{}: the interpreter on {}", p.name, c.name);
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
            assert_eq!(run_exe(&exe, &c.stdin).unwrap(), *c.answer().unwrap(), "{}: the compiled reference on {}", p.name, c.name);
        }
        let batch = program_cases(&k, &sig, &tree, cases.len()).unwrap();
        let bexe = compile_c(&batch, &scratch(&format!("reference_batch_{}", p.name))).unwrap_or_else(|e| panic!("{e}\n{batch}"));
        let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin_of(&cases))).unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(read_binary(&out.stdout).unwrap(), cases.iter().map(|c| c.answer().unwrap().clone()).collect::<Vec<_>>(), "{}: the batched reference", p.name);
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
        let vec_of = |n: usize, rng: &mut GeneRng| Array::i32s(vec![n], (0..n).map(|_| rng.below(41) as i32 - 20).collect());
        // (k, n): the histogram's bins and length; reduce and scan read only n.
        let small: Vec<Vec<Array>> = [(3usize, 7usize), (1, 1), (5, 2)]
            .iter()
            .map(|&(kb, n)| if p.name == "histogram" { vec![Array::scalar_i32(kb as i32), vec_of(n, &mut rng), vec_of(n, &mut rng)] } else { vec![vec_of(n, &mut rng)] })
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
    let xs = Array::i32s(vec![6], vec![-7, 3, i32::MAX, i32::MIN, 0, 5]);
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
    let vs = Array::i32s(vec![6], vec![1, -2, 3, 4, i32::MAX, 6]);
    for kb in [1, 4, 9] {
        let inputs = [Array::scalar_i32(kb), xs.clone(), vs.clone()];
        for text in [
            "(hist.add.n (tab (mod.k (get.n is i_n))) vs)",
            "(hist.min.n (tab (mod.k (get.n is i_n))) vs)",
            "(hist.max.n is vs)",
            "(hist.add.n vs is)",
            "(tab (mod.k (get.k (hist.add.n is vs) i_k)))",
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
        let planted = build(&k, p.planted().unwrap()).unwrap();
        let reference = build(&k, p.reference().unwrap()).unwrap();
        let cases = load_cases(&p).unwrap();
        for c in &cases {
            assert_eq!(interp.eval(&planted, &c.inputs).unwrap(), *c.answer().unwrap(), "{}: the planted slowdown on {}", p.name, c.name);
        }
        let sizes = sig.sizes(&cases[2].inputs).unwrap();
        let (wp, wr) = (interp.work(&planted, &sizes).unwrap(), interp.work(&reference, &sizes).unwrap());
        assert!(wp > wr, "{}: planted {wp}, reference {wr}", p.name);
        let bridge = Bridge::load(&p.instance).unwrap();
        let vs = variants(&k, &bridge, &planted, 3, 32, 64).unwrap();
        for v in &vs {
            for c in &cases {
                assert_eq!(interp.eval(v, &c.inputs).unwrap(), *c.answer().unwrap(), "{}: the variant {} on {}", p.name, sexp(&k, v), c.name);
            }
        }
        let small = vs.iter().map(|v| interp.work(v, &sizes).unwrap()).min().unwrap_or(wp);
        assert!(small <= wr, "{}: the smallest variant's work {small}, the reference's {wr}", p.name);
    }
}

/// THE FIVE HISTOGRAM MUTANTS OF THE G3 GPU CHECK (2026-11-04, phylu 20b352ad, regime P, generation 0), as the run's
/// ledger holds them: each timed for minutes on the GPU, each counted under the timing bound (5.9e10 node executions)
/// before a loop's array state was charged a trip. The work at histogram's timing input (k = 100, n = 15 x 2^20).
const G3_SLOW_MUTANTS: [&str; 5] = [
    "(hist.add.n (tab (neg (get.n vs i_n))) (loop1.k zeros (loop.k (hist.add.n vs zeros) zeros)))",
    "(hist.add.n (tab (neg (get.n vs i_n))) (loop1.k (loop1.n (scan.add vs) s_a_n) (loop.k (hist.add.n vs zeros) zeros)))",
    "(hist.add.n (tab (neg (get.n vs i_n))) (loop1.k (loop1.n (scan.add vs) s_a_n) (loop.k (hist.add.n zeros zeros) vs)))",
    "(hist.add.n (scan.add (loop1.n vs s_a_n)) (loop1.k (scan.add (hist.max.n (map2.max vs vs) (hist.max.k (hist.min.k zeros (loop.k zeros (hist.min.n is (loop.n zeros vs)))) zeros))) (loop.k (hist.add.n zeros zeros) vs)))",
    "(hist.add.n (scan.add (loop1.n vs s_a_n)) (loop1.k (hist.add.k (hist.min.k zeros (loop.k zeros (hist.min.n (loop.n zeros vs) (loop.n (hist.add.k s_a_k s_a_k) zeros)))) zeros) (loop.k (hist.add.n zeros zeros) vs)))",
];

/// A LOOP'S ARRAY STATE COSTS ITS CELLS EVERY TRIP when the body copies it (the work estimate's rule of 2026-11-04,
/// refined by the F3a replay the same day): `(loop.k (scan.add zeros) zeros)` on k = 3 is the loop (1), its start (the
/// scan, 1 + 3, and its `zeros`, 1), its body's `zeros` a trip (3) and its `[k]` state copied a trip (3 x 3). No state
/// is charged for a loop Futhark removes (its body its own state, or its start: `(loop.k zeros zeros)`), nor for a body
/// that writes a fresh array charged its cells already (`(scan.add s_a_k)`), nor for a scalar state. The five slow
/// mutants of the G3 GPU check are each over 1e11 at histogram's timing input, where the published program is 7.9e7.
#[test]
fn a_loops_array_state_costs_its_cells_every_trip() {
    let hist = load(&HISTOGRAM).unwrap();
    let sig = signature(&HISTOGRAM).unwrap();
    let interp = Interp::new(&hist, &sig).unwrap();
    let small: BTreeMap<String, usize> = [("k".to_string(), 3), ("n".to_string(), 5)].into_iter().collect();
    let parts = |text: &str, sizes: &BTreeMap<String, usize>| interp.work_parts(&build(&hist, text).unwrap(), sizes).unwrap();
    let w = |text: &str, sizes: &BTreeMap<String, usize>| parts(text, sizes).total();
    assert_eq!(parts("(loop.k (scan.add zeros) zeros)", &small), Work { nodes: 1 + (1 + 3 + 1) + 3, state: 3 * 3 });
    assert_eq!(parts("(loop1.k (scan.add zeros) zeros)", &small), Work { nodes: 1 + (1 + 3 + 1) + 2, state: 2 * 3 }, "a loop from 1 makes k - 1 trips");
    // A loop Futhark's simplifier removes (its loop-invariant merge variable): the body is the loop's own state, or
    // the body is its start and reads none of the loop's variables.
    assert_eq!(parts("(loop.k zeros s_a_k)", &small), Work { nodes: 1 + 1 + 3, state: 0 }, "the body is the state");
    assert_eq!(parts("(loop.k zeros zeros)", &small), Work { nodes: 1 + 1 + 3, state: 0 }, "the body is the start");
    assert_eq!(parts("(loop.k (scan.add zeros) (scan.add zeros))", &small).state, 0, "the body is the start, an expression");
    // A body that writes a fresh array (a scan: 1 + its 3 cells, and its variable, a trip) writes the state in place.
    assert_eq!(parts("(loop.k zeros (scan.add s_a_k))", &small), Work { nodes: 1 + 1 + 3 * (1 + 3 + 1), state: 0 });
    // A scalar state: `(tab (loop.n 0 s_i32))` is the tabulate (1), and k times the loop (1), its start (1) and n
    // trips of its body (5): no state cells.
    assert_eq!(parts("(tab (loop.n 0 s_i32))", &small), Work { nodes: 1 + 3 * (1 + 1 + 5), state: 0 });
    let timing: BTreeMap<String, usize> = [("k".to_string(), 100), ("n".to_string(), 15 << 20)].into_iter().collect();
    let published = w("(hist.add.n (tab (mod.k (get.n is i_n))) vs)", &timing);
    assert_eq!(published, 78_643_303, "the published program's work at the timing input, as the G3 run printed it");
    for text in G3_SLOW_MUTANTS {
        let got = w(text, &timing);
        assert!(got > 100_000_000_000, "{got} for {text}");
        eprintln!("G3 slow mutant: {got:.3e} node executions ({:.0} x the published program's): {text}", got as f64 / published as f64);
    }
}

/// THE F3a REPLAY'S HISTOGRAM PROGRAM (generation 1 of the deprecated runner's GPU run of 2026-11-04 21:08Z, frozen at
/// fuller 2e1a3b72): the runner compiled it and scored it on every case (its runs 9.0 s in all). Its loops copy
/// `[n]` states n x n times on the larger cases (1.1e9 cells at n = 1000), which the per-submit cap must not read: its
/// node executions are under the cap on every case, so it is scored as before. And `(loop.n is is)`, scored worst at
/// 1.0e9 by the G3 smoke run of 21:28Z, is a loop Futhark removes: no state traffic.
const F3A_REPLAY_HISTOGRAM: &str = "(hist.max.n vs (loop.n (map2.min (map2.add (loop.n vs vs) (map2.min zeros (hist.max.n zeros is))) (hist.add.k (hist.max.n (hist.min.k zeros zeros) (map2.max (hist.add.k zeros (hist.max.n (loop1.k is is) (map2.max (hist.min.n (map2.max (scan.max (map2.max (map2.min (map2.sub (loop.n is (scan.max (hist.add.n s_a_n (map2.min (map2.sub (scan.max (hist.min.n s_a_n (hist.min.k zeros zeros))) zeros) zeros)))) (hist.min.n (loop.k zeros (scan.add vs)) (scan.min (loop.n (scan.add (loop.k vs (hist.min.n (loop1.n s_a_n zeros) s_a_n))) (loop1.k vs is))))) (hist.add.k zeros zeros)) (loop.n vs (replicate (minimum.n (map2.min (map2.sub (hist.max.n vs is) (loop.n (hist.min.k zeros (hist.min.n vs (hist.max.k zeros (hist.min.k (scan.add zeros) (map2.add zeros zeros))))) is)) is)))))) (hist.max.n vs vs)) (hist.min.n is (replicate n))) zeros))) zeros)) (loop.k zeros zeros))) (hist.add.k (scan.min (scan.add zeros)) zeros)))";

#[test]
fn a_program_the_runner_scored_is_under_the_per_submit_cap_on_every_case() {
    let hist = load(&HISTOGRAM).unwrap();
    let sig = signature(&HISTOGRAM).unwrap();
    let interp = Interp::new(&hist, &sig).unwrap();
    let tree = build(&hist, F3A_REPLAY_HISTOGRAM).unwrap();
    let removed = build(&hist, "(hist.add.n (tab (mod.k (get.n (loop.n is is) i_n))) vs)").unwrap();
    let mut most = Work::default();
    for c in load_cases(&HISTOGRAM_PROBLEM).unwrap() {
        let sz = sig.sizes(&c.inputs).unwrap();
        let w = interp.work_parts(&tree, &sz).unwrap();
        assert!(w.nodes <= CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT, "{}: {w:?}", c.name);
        most = Work { nodes: most.nodes.max(w.nodes), state: most.state.max(w.state) };
        let r = interp.work_parts(&removed, &sz).unwrap();
        assert_eq!(r.state, 0, "{}: the removed loop copies nothing", c.name);
        assert!(r.nodes <= CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT, "{}: {r:?}", c.name);
    }
    assert!(most.state > 1_000_000_000 && most.nodes < 25_000_000, "the most on a case: {most:?}");
}

/// THE F3b REPLAY'S K-MEANS PROGRAM (generation 0 of the deprecated runner's GPU run of 2026-11-04 21:13Z, frozen at
/// fuller 2e1a3b72, which compiled it and whose run of it failed, worst on every input): its inner loop over d copies
/// the `[d]` row `centres[last]` into its state every trip, k x d x d times. With that traffic it is over k-means's
/// work cap (10 x the seed's work) on every fitness case, so phylu's evaluator does not run it; its node executions
/// alone are under the seed's. The sizes are phylu's `KMEANS_CASES` (d = 1536, the sample's).
#[test]
fn the_f3b_kmeans_program_copies_a_row_k_d_d_times_over_ten_times_the_seed_s_work() {
    let km = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&km, &sig).unwrap();
    let tree = build(&km, "(tabrows (loop1.d zeros (loop1.d (scan.max (loop1.k (map2.sub (hist.max.k zeros zeros) (loop1.d zeros (row.k.d centres (next zero)))) (row.n.d points last))) (row.k.d centres last))))").unwrap();
    let seed = build(&km, KMEANS_PROBLEM.reference().unwrap()).unwrap();
    for (n, k) in [(1000, 16), (1000, 50), (500, 8), (200, 4), (1000, 32), (500, 16)] {
        let sz: BTreeMap<String, usize> = [("n".to_string(), n), ("k".to_string(), k), ("d".to_string(), 1536)].into_iter().collect();
        let (w, s) = (interp.work_parts(&tree, &sz).unwrap(), interp.work_parts(&seed, &sz).unwrap());
        assert_eq!(s.state, 0, "n {n} k {k}: the seed's loops carry scalars");
        let d = 1536u128;
        assert!(w.state >= k as u128 * (d - 1) * (d - 1) * d, "n {n} k {k}: {w:?}");
        assert!(w.total() > 10 * s.total() && w.nodes < s.nodes, "n {n} k {k}: {w:?}, the seed {s:?}");
    }
}
