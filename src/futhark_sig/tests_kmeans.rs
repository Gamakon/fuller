//! THE K-MEANS PROBLEM, the first f32 view (`docs/PLAN_futhark_kingdom.md`, step 4): the kingdom loads from the
//! master and the sniffer's specification; the textbook step (`KMEANS_TEXTBOOK`, a pool alternative; the seed is
//! UltraDim's kernels, 4c (d)) computes the same bits through the host interpreter and through `futhark c` on the
//! stored case small enough for the interpreter (the cases are cut from the dbpedia sample and store inputs only:
//! k-means is scored on its objective, `docs/KMEANS_FITNESS.md`; the interpreter does about 4e7 node executions a
//! second, so one step of 1,000 x 1,536 x 16 is a minute: the dbpedia-sized cases are the compiled program's); the
//! f32 rows' semantics equal `futhark c`'s on the edges (NaN, the signed zeros, the infinities, a denormal); random
//! genes express to programs `futhark check` accepts and `futhark c` computes as the interpreter does; the planted
//! slowdown is equal and the e-graph recovers the textbook step's work; the layout at the runner's default program
//! size holds both; the cases are the declared cuts with UltraDim's starts.

use super::*;
use crate::typed_gep::{express, fits, write_back};
use std::collections::BTreeSet;
use std::process::Command;

fn have_futhark() -> bool {
    Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_sig_kmeans_{name}_{}", std::process::id()))
}

/// `futhark check` on a program: Futhark's type checker accepts it.
fn futhark_check(src: &str, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fut = dir.join("prog.fut");
    std::fs::write(&fut, src).map_err(|e| format!("write {}: {e}", fut.display()))?;
    let out = crate::futhark_exec::output_with_timeout(Command::new("futhark").arg("check").arg(&fut), std::time::Duration::from_secs(120), None)?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("futhark check:\n{}", String::from_utf8_lossy(&out.stderr)))
    }
}

#[test]
fn the_kmeans_kingdom_loads_from_the_master_and_its_specification() {
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    assert_eq!(sig.dims, vec!["n".to_string(), "d".to_string(), "k".to_string()]);
    assert_eq!(sig.depth, 3);
    assert_eq!(sig.inputs, vec![Param { name: "points".into(), elem: "f32".into(), dims: vec!["n".into(), "d".into()] }, Param { name: "centres".into(), elem: "f32".into(), dims: vec!["k".into(), "d".into()] }]);
    assert_eq!(sig.output, Param { name: "main".into(), elem: "f32".into(), dims: vec!["k".into(), "d".into()] });
    let contract = k.relations().check().unwrap_or_else(|d| panic!("the kingdom check: {:?}", &d[..d.len().min(3)]));
    assert_eq!(k.roots(), vec![k.ty("a.k.d:c").unwrap()]);
    Interp::new(&k, &sig).unwrap();
    let view = KMEANS.view().unwrap();
    assert_eq!(view.spec.master, Some(5));
    assert_eq!(view.spec.lets.iter().map(|c| c.text(false)).collect::<Vec<_>>(), ["a.n"], "the kernel boundary names the assignments");
    let functions = k.rows.iter().filter(|r| r.kind == FutharkKind::Function).count();
    let ops: BTreeSet<&str> = k.rows.iter().map(|r| r.name.split('@').next().unwrap_or("").split('.').next().unwrap_or("")).collect();
    for op in ["div", "sqrt", "tabrows", "histrows", "rowdiv", "hist", "clamp", "map2", "sum", "loop", "let", "triple", "thd", "pair", "psel"] {
        assert!(ops.contains(op), "the k-means view has no {op} row: {ops:?}");
    }
    assert!(!ops.contains("mod"), "the mod row is i32 only");
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], KMEANS_PROGRAM_SIZE).unwrap();
    eprintln!("kmeans: {} types, {} rows ({functions} functions, {} leaves); N = {KMEANS_PROGRAM_SIZE}: {} words a gene, head {}, at most {} nodes; fingerprint {:016x}", k.duals.len(), k.rows.len(), k.rows.len() - functions, layout.width, layout.head_lens().values().sum::<usize>(), layout.max_nodes(), view.fingerprint());
    // Measured 2026-11-03 (count contexts at depth 3, master 5 with the pairs, the triples and one kernel boundary;
    // pinned, as the i32 views' sizes are in futhark_view::tests). Before the kernel boundary and the tuples (master 3)
    // the view was 3,804 types, 21,620 functions and 8,421 leaves.
    assert_eq!((k.duals.len(), functions, k.rows.len() - functions), KMEANS_SIZES, "the k-means view's sizes changed");
}

/// (types, function rows, leaf rows), measured 2026-11-03.
const KMEANS_SIZES: (usize, usize, usize) = (14359, 79660, 31211);

/// The textbook step and its planted slowdown build, round-trip through `sexp`, fit the layout at the default program
/// size, and compute the same bits through the host interpreter, through `futhark c` (the plain program and the quality
/// program of every case) on every stored case: the kingdom's f32 rows mean what the compiler computes. (The cases
/// store no answer: k-means is scored on its clustering objective, `docs/KMEANS_FITNESS.md`.)
#[test]
fn the_kmeans_textbook_step_and_planted_slowdown_compute_the_same_bits_interpreted_and_compiled() {
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], KMEANS_PROGRAM_SIZE).unwrap();
    let const_len = k.const_len();
    let all = load_cases(&KMEANS_PROBLEM).unwrap();
    assert!(all.iter().all(|c| c.output.is_none()), "k-means stores no answers");
    let cases = interpreter_cases(&all);
    let tree = build(&k, KMEANS_TEXTBOOK).unwrap();
    let mut rng = GeneRng::new(3);
    let gene = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &tree).unwrap();
    assert_eq!(express(&contract, &layout, &gene, &const_len).unwrap().tree(), tree, "the write-back");
    eprintln!("kmeans textbook step: {} functions", function_count(&k, &tree));
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let mut same = 0;
    for (what, text) in [("textbook step", KMEANS_TEXTBOOK), ("planted slowdown", KMEANS_TEXTBOOK_PLANTED)] {
        let t = build(&k, text).unwrap();
        assert_eq!(sexp(&k, &t).split_whitespace().collect::<String>(), text.split_whitespace().collect::<String>());
        assert!(function_count(&k, &t) < KMEANS_PROGRAM_SIZE, "the {what} has {} functions", function_count(&k, &t));
        fits(&contract, &layout, &t).unwrap_or_else(|e| panic!("the {what} does not fit N = {KMEANS_PROGRAM_SIZE}: {e}"));
        let src = program(&k, &sig, &t).unwrap();
        let exe = compile_c(&src, &scratch(&format!("textbook_{}", what.replace(' ', "_")))).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let mut wants = Vec::new();
        for c in &cases {
            let t0 = std::time::Instant::now();
            let want = interp.eval(&t, &c.inputs).unwrap();
            assert_eq!((want.elem, want.dims.clone()), (Elem::F32, vec![c.inputs[1].dims[0], c.inputs[1].dims[1]]));
            assert_eq!(run_exe(&exe, &c.stdin).unwrap(), want, "the compiled {what} on {}", c.name);
            eprintln!("kmeans {what} on {}: interpreted and compiled agree ({} cells, {:.3} s interpreted)", c.name, want.len(), t0.elapsed().as_secs_f64());
            same += 1;
            wants.push(want);
        }
        let batch = program_cases(&k, &sig, &t, cases.len()).unwrap();
        let bexe = compile_c(&batch, &scratch(&format!("textbook_batch_{}", what.replace(' ', "_")))).unwrap_or_else(|e| panic!("{e}\n{batch}"));
        let stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
        let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin)).unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(read_binary(&out.stdout).unwrap(), wants, "the batched {what}");
    }
    assert_eq!(same, 2 * cases.len());
}

/// The stored cases the host interpreter is run on: those of at most [`INTERPRETER_ROWS`] rows (`train_n200_k4`;
/// `kmeans_converged::KMEANS_CASES`). At least one, or the test says so.
fn interpreter_cases(cases: &[Case]) -> Vec<Case> {
    let small: Vec<Case> = cases.iter().filter(|c| c.inputs[0].dims[0] <= INTERPRETER_ROWS).cloned().collect();
    assert!(!small.is_empty(), "no stored case of at most {INTERPRETER_ROWS} rows for the host interpreter");
    small
}

/// The most rows of a case the host interpreter is run on in these tests.
const INTERPRETER_ROWS: usize = 200;

/// THE F32 SEMANTICS, compiled against the interpreter on purpose (a random gene may not draw them): every f32 row on
/// arrays holding a NaN, the two zeros, the two infinities, a denormal and ordinary values, through `futhark c`, bit
/// for bit under the comparison policy.
#[test]
fn the_f32_rows_equal_futhark_c_on_the_edges() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let edges = [0.0f32, -0.0, f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY, 3e38, -4.0, 1e-45, 2.5, -0.5, 16_777_217.0];
    // centres [k = 4][d = 3]: every edge; points [n = 2][d = 3]: a NaN, -0.0, 1.0 and an infinity, 2.5, -4.0.
    let centres = Array::f32s(vec![4, 3], edges.to_vec());
    let points = Array::f32s(vec![2, 3], vec![f32::NAN, -0.0, 1.0, f32::INFINITY, 2.5, -4.0]);
    let inputs = [points, centres];
    // Inside a tabulate of rows over k and a tabulate over d: the centre's row (every edge) and the first point's.
    let p = "(row.k.d centres i_k)";
    let c = "(row.n.d points zero)";
    let e = "(get.d (row.k.d centres i_k) i_d)";
    let f = "(get.d (row.n.d points zero) i_d)";
    let scalar = |body: &str| format!("(tabrows (tab {body}))");
    let texts: Vec<String> = [
        scalar(&format!("(add {e} {f})")),
        scalar(&format!("(sub {e} {f})")),
        scalar(&format!("(mul {e} {f})")),
        scalar(&format!("(div {e} {f})")),
        scalar(&format!("(div {f} {e})")),
        scalar(&format!("(min {e} {f})")),
        scalar(&format!("(min {f} {e})")),
        scalar(&format!("(max {e} {f})")),
        scalar(&format!("(max {f} {e})")),
        scalar(&format!("(neg {e})")),
        scalar(&format!("(abs {e})")),
        scalar(&format!("(sqrt {e})")),
        scalar(&format!("(sel {e} {f} 10)")),
        scalar(&format!("(ival.d (clamp {e}))")),
        scalar(&format!("(ival.k (clamp {e}))")),
        scalar(&format!("(ival.n (next (clamp {e})))")),
        scalar(&format!("(get.d (map2.add {p} {c}) i_d)")),
        scalar(&format!("(get.d (map2.sub {p} {c}) i_d)")),
        scalar(&format!("(get.d (map2.mul {p} {c}) i_d)")),
        scalar(&format!("(get.d (map2.min {p} {c}) i_d)")),
        scalar(&format!("(get.d (map2.max {p} {c}) i_d)")),
        scalar(&format!("(sum.d {p})")),
        scalar(&format!("(minimum.d {p})")),
        scalar(&format!("(maximum.d {p})")),
        scalar(&format!("(get.d (scan.add {p}) i_d)")),
        scalar(&format!("(get.d (scan.min {p}) i_d)")),
        scalar(&format!("(get.d (scan.max {p}) i_d)")),
        // Histograms at f32 indices: the edges as indices (a NaN, infinities, 2.5 truncated, negatives ignored).
        scalar(&format!("(get.d (hist.add.d {p} {c}) i_d)")),
        scalar(&format!("(get.d (hist.min.d {p} {c}) i_d)")),
        scalar(&format!("(get.d (hist.max.d {p} {c}) i_d)")),
        scalar(&format!("(get.d (replicate (sum.d {p})) i_d)")),
        "(histrows.add.n (tab (get.d (row.n.d points i_n) zero)) points)".to_string(),
        "(histrows.add.n (tab (ival.n i_n)) points)".to_string(),
        "(histrows.add.k (tab (ival.k i_k)) centres)".to_string(),
        "(rowdiv centres (tab (get.d (row.k.d centres i_k) last)))".to_string(),
        "(rowdiv (histrows.add.n (tab (ival.n i_n)) points) (tab (get.d (row.k.d centres i_k) zero)))".to_string(),
        "(tabrows (loop.d (row.k.d centres i_k) (map2.add s_a_d (row.k.d centres (clamp (ival.d i_d))))))".to_string(),
    ]
    .into_iter()
    .collect();
    let base = scratch("edges");
    for (i, text) in texts.iter().enumerate() {
        let tree = build(&k, text).unwrap_or_else(|e| panic!("{text}: {e}"));
        let src = program(&k, &sig, &tree).unwrap();
        let exe = compile_c(&src, &base.join(format!("e{i}"))).unwrap_or_else(|e| panic!("{text}: {e}\n{src}"));
        let want = interp.eval(&tree, &inputs).unwrap_or_else(|e| panic!("{text}: the interpreter: {e}"));
        let got = run_exe(&exe, &write_binary(&[&inputs[0], &inputs[1]])).unwrap_or_else(|e| panic!("{text}: {e}\n{src}"));
        assert_eq!(got, want, "{text}: compiled and interpreted differ\n{src}\ncompiled {:?}\ninterpreted {:?}", got.f32_values(), want.f32_values());
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// Legality, measured: 50 random genes of the runner's default program size express to programs `futhark check`
/// accepts, and the first 12 compile with `futhark c` and equal the interpreter on three small cases (the edges of
/// the signature, n = d = k = 1 included).
#[test]
fn random_kmeans_genes_are_accepted_by_futhark_check_and_equal_the_interpreter() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], KMEANS_PROGRAM_SIZE).unwrap();
    let const_len = k.const_len();
    let mut rng = GeneRng::new(0x51C_0003);
    let unit = |m: usize, rng: &mut GeneRng| (0..m).map(|_| (rng.below(1 << 16) as f32) / 65536.0 - 0.25).collect::<Vec<f32>>();
    let small: Vec<Vec<Array>> = [(5usize, 3usize, 2usize), (1, 1, 1), (3, 2, 4)].iter().map(|&(n, d, kk)| vec![Array::f32s(vec![n, d], unit(n * d, &mut rng)), Array::f32s(vec![kk, d], unit(kk * d, &mut rng))]).collect();
    let base = scratch("random");
    let mut used = BTreeMap::<String, usize>::new();
    let (mut checked, mut compiled) = (0, 0);
    for g in 0..50 {
        let gene = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
        let tree = express(&contract, &layout, &gene, &const_len).unwrap().tree();
        for (t, _) in tree.level_order() {
            let op = k.rows[t.symbol as usize].name.split('@').next().unwrap_or("").split('.').next().unwrap_or("").to_string();
            *used.entry(op).or_default() += 1;
        }
        let src = program(&k, &sig, &tree).unwrap();
        futhark_check(&src, &base.join(format!("c{g}"))).unwrap_or_else(|e| panic!("gene {g} is not accepted: {e}\n{src}"));
        checked += 1;
        if g < 12 {
            let batch = program_cases(&k, &sig, &tree, small.len()).unwrap();
            let bexe = compile_c(&batch, &base.join(format!("b{g}"))).unwrap_or_else(|e| panic!("gene {g}'s quality program does not compile: {e}\n{batch}"));
            let refs: Vec<&Array> = small.iter().flatten().collect();
            let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&write_binary(&refs))).unwrap();
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            let wants: Vec<Array> = small.iter().map(|w| interp.eval(&tree, w).unwrap()).collect();
            assert_eq!(read_binary(&out.stdout).unwrap(), wants, "gene {g}: compiled and interpreted differ\n{src}");
            compiled += 1;
        }
    }
    eprintln!("kmeans: {checked} random genes accepted by futhark check, {compiled} compiled and equal to the interpreter; rows used: {used:?}");
    assert_eq!((checked, compiled), (50, 12));
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn the_kmeans_planted_slowdown_is_equal_and_the_egraph_recovers_the_reference() {
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let planted = build(&k, KMEANS_TEXTBOOK_PLANTED).unwrap();
    let reference = build(&k, KMEANS_TEXTBOOK).unwrap();
    let cases = interpreter_cases(&load_cases(&KMEANS_PROBLEM).unwrap());
    let wants: Vec<Array> = cases.iter().map(|c| interp.eval(&reference, &c.inputs).unwrap()).collect();
    let sizes = sig.sizes(&cases[0].inputs).unwrap();
    let (wp, wr) = (interp.work(&planted, &sizes).unwrap(), interp.work(&reference, &sizes).unwrap());
    assert!(wp > wr, "planted {wp}, reference {wr}");
    // The work at the timing input's sizes, against the timing bound.
    let timing: BTreeMap<String, usize> = [("n", KMEANS_PROBLEM.timing_n), ("d", KMEANS_TIMING_D), ("k", KMEANS_TIMING_K)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    let wt = interp.work(&reference, &timing).unwrap();
    eprintln!("kmeans work: reference {wr}, planted {wp} on {sizes:?}; the reference at the timing input {wt} (the bound {TIMING_WORK_CAP})");
    let bridge = Bridge::load(&KMEANS).unwrap();
    let vs = variants(&k, &bridge, &planted, 3, 256, 64).unwrap();
    assert!(!vs.is_empty(), "no variants");
    for c in &cases {
        assert_eq!(interp.eval(&planted, &c.inputs).unwrap(), wants[cases.iter().position(|x| x.name == c.name).unwrap()], "the planted slowdown on {}", c.name);
    }
    for v in &vs {
        for (c, want) in cases.iter().zip(&wants) {
            assert_eq!(interp.eval(v, &c.inputs).unwrap(), *want, "the variant {} on {}", sexp(&k, v), c.name);
        }
    }
    for v in &vs {
        eprintln!("kmeans variant: work {} of {}: {}", interp.work(v, &sizes).unwrap(), wp, sexp(&k, v));
    }
    let small = vs.iter().map(|v| interp.work(v, &sizes).unwrap()).min().unwrap_or(wp);
    assert!(small <= wr, "the smallest variant's work {small}, the reference's {wr}");
}

/// The comparison policy, stated as tests: bit for bit, two NaNs agree, the signed zeros differ; an `i32` array and an
/// `f32` array of the same bits differ; the binary format round-trips both elements.
#[test]
fn the_comparison_policy_is_bit_for_bit_with_nans_equal() {
    let a = Array::f32s(vec![4], vec![1.0, -0.0, f32::NAN, f32::INFINITY]);
    assert_eq!(a, a.clone());
    let other_nan = Array::f32s(vec![4], vec![1.0, -0.0, f32::from_bits(0x7fc0_1234), f32::INFINITY]);
    assert_eq!(a, other_nan, "a NaN's payload is not a value");
    let plus_zero = Array::f32s(vec![4], vec![1.0, 0.0, f32::NAN, f32::INFINITY]);
    assert_ne!(a, plus_zero, "the signed zeros are different values");
    assert_ne!(a, Array { dims: vec![4], elem: Elem::I32, data: a.data.clone() }, "an i32 array is not an f32 array");
    let raw = write_binary(&[&a, &Array::scalar_i32(7)]);
    assert_eq!(&raw[3..7], b" f32");
    assert_eq!(read_binary(&raw).unwrap(), vec![a.clone(), Array::scalar_i32(7)]);
    assert_eq!(compare(&a, &a), [0.0; 3]);
    assert_eq!(compare(&a, &other_nan), [0.0; 3]);
    let q = compare(&a, &plus_zero);
    assert_eq!(q[0], 0.25, "one cell of four differs, by a zero's sign");
    assert_eq!(q[1], 0.0, "and by no distance");
    // A NaN against a number is an infinite error, squashed to 1.
    let number = Array::f32s(vec![4], vec![1.0, -0.0, 2.0, f32::INFINITY]);
    let q = compare(&a, &number);
    assert_eq!(q, [0.25, 1.0, 1.0]);
    for part in [QualityPart::Wrong, QualityPart::CellLog, QualityPart::CellLevel, QualityPart::Step, QualityPart::Shape, QualityPart::Order, QualityPart::Outside(1), QualityPart::Study] {
        let inputs = quality_inputs_of(&[part], &a, &number);
        assert_eq!(inputs.len(), quality_input_names_of(&[part], &number).len(), "{part:?}");
        assert!(inputs.iter().all(|x| (0.0..=1.0).contains(x)), "{part:?}: {inputs:?}");
        assert_eq!(quality_inputs_of(&[part], &number, &number), vec![0.0; inputs.len()], "{part:?}: the answer is 0");
    }
}

/// THE TIMING INPUT is the whole dbpedia sample (n = 5,000 rows of d = 1,536) with k = 256 stored starting centres:
/// the starts are UltraDim's k-means++ draws over the sample at the recorded seed (recomputed here, byte for byte),
/// every row a unit vector, fixed bytes; the published program (the yardstick, `futhark c`) runs on it and gives a
/// `[k][d]f32` of finite values. The host interpreter is not run on it: measured here at the rate it does the small
/// case, the reference's work there (printed with the timing bound) would take hours, so the timing input's output
/// is the yardstick's (as the runner takes it, `TimingInput::of`).
#[test]
fn the_kmeans_timing_input_is_the_dbpedia_sample_with_its_stored_starts_and_the_yardstick_runs_on_it() {
    use crate::futhark_sig::kmeans_converged::{cut_case, KMEANS_TIMING_CUT};
    let p = problem("kmeans").unwrap();
    let stdin = timing_input(&p).unwrap().expect("k-means has its own timing input");
    let inputs = read_binary(&stdin).unwrap();
    assert_eq!(inputs.len(), 2, "the points and the centres");
    assert_eq!((inputs[0].dims.clone(), inputs[0].elem), (vec![KMEANS_TIMING_N, KMEANS_TIMING_D], Elem::F32));
    assert_eq!((inputs[1].dims.clone(), inputs[1].elem), (vec![KMEANS_TIMING_K, KMEANS_TIMING_D], Elem::F32));
    assert_eq!((p.timing_n, KMEANS_TIMING_CUT.k, KMEANS_TIMING_CUT.end), (KMEANS_TIMING_N, KMEANS_TIMING_K, KMEANS_TIMING_N));
    for a in &inputs {
        let d = a.dims[1];
        let v = a.f32_values().unwrap();
        assert!(v.chunks_exact(d).all(|r| (r.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt() - 1.0).abs() < 1e-3), "every row is a unit vector");
    }
    let sample = read_one(&std::fs::read(corpus_dir(&p).join(KMEANS_SAMPLE)).unwrap()).unwrap();
    assert_eq!(sample, inputs[0], "the points are the sample as it is");
    let t = std::time::Instant::now();
    let (_, starts) = cut_case(&sample, &KMEANS_TIMING_CUT).unwrap();
    eprintln!("kmeans: the timing input's {} k-means++ starts recomputed in {:.1} s", KMEANS_TIMING_K, t.elapsed().as_secs_f64());
    assert_eq!(starts, inputs[1], "the stored starts are UltraDim's k-means++ draws at seed {}", KMEANS_TIMING_CUT.seed);
    assert_eq!(timing_input(&p).unwrap().unwrap(), stdin, "the input is fixed: the same bytes every time");
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let sizes = sig.sizes(&inputs).unwrap();
    assert_eq!((sizes["n"], sizes["d"], sizes["k"]), (KMEANS_TIMING_N, KMEANS_TIMING_D, KMEANS_TIMING_K));
    let interp = Interp::new(&k, &sig).unwrap();
    let reference = build(&k, kmeans_seed_static()).unwrap();
    // The interpreter's rate on the small case, and the work at the timing input.
    let cases = load_cases(&p).unwrap();
    let small = &interpreter_cases(&cases)[0];
    let t = std::time::Instant::now();
    let small_out = interp.eval(&reference, &small.inputs).unwrap();
    assert_eq!(small_out.dims, vec![small.inputs[1].dims[0], small.inputs[1].dims[1]]);
    let secs = t.elapsed().as_secs_f64();
    let w_small = interp.work(&reference, &sig.sizes(&small.inputs).unwrap()).unwrap();
    let w_timing = interp.work(&reference, &sizes).unwrap();
    eprintln!("kmeans: the reference on {} ({w_small} node executions) in {secs:.3} s, {:.2e} a second; at the timing input {w_timing} node executions, {:.0} s at that rate; the timing bound {}", small.name, w_small as f64 / secs, w_timing as f64 / (w_small as f64 / secs), timing_work_cap(w_timing));
    assert!(w_timing <= timing_work_cap(w_timing));
    if !have_futhark() {
        panic!("futhark is not installed: the published program cannot be checked on the timing input");
    }
    let exe = compile_c(p.yardstick, &scratch("timing")).unwrap();
    let t = std::time::Instant::now();
    let got = run_exe(&exe, &stdin).unwrap();
    eprintln!("kmeans: the yardstick (futhark c) on the timing input in {:.2} s", t.elapsed().as_secs_f64());
    assert_eq!((got.dims.clone(), got.elem), (vec![KMEANS_TIMING_K, KMEANS_TIMING_D], Elem::F32));
    assert!(got.values().all(|x| x.is_finite() && x.abs() <= 1.0), "every centre is a mean of unit vectors, or zero");
}

/// THE CASES (the advisor's review of 2026-11-03; `make_cases.sh`, `kmeans_converged::KMEANS_CASES`): every stored
/// case is its declared cut of the dbpedia sample byte for byte, with UltraDim's k-means++ starts at its seed
/// (recomputed here), `cases.tsv` records the table, the six fitness and test cuts are disjoint and use the 5,000 rows
/// once (the small training case lies inside the first training case), k is at most n / 20, every row is a unit
/// vector, the splits are 4 train, 2 held, 1 test; and the f64 reference of UltraDim's step (`reference64.fut`
/// through `futhark c`) equals the Rust f64 step bit for bit on every case (both sequential f64: the translation's
/// check, never a score).
#[test]
fn the_dbpedia_cases_are_the_declared_cuts_with_ultradims_starts_and_the_f64_reference_agrees_with_the_rust_step() {
    use crate::futhark_sig::kmeans_converged::{cut_case, ultradim_step_f64, KMEANS_CASES};
    let p = problem("kmeans").unwrap();
    let cases = load_cases(&p).unwrap();
    let count = |s: Split| cases.iter().filter(|c| c.split == s).count();
    assert_eq!((count(Split::Train), count(Split::Held), count(Split::Test)), (4, 2, 1));
    assert!(cases.iter().all(|c| c.output.is_none()));
    assert_eq!(cases.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), KMEANS_CASES.iter().map(|c| c.name).collect::<Vec<_>>(), "cases.tsv lists the table's cases in order");
    let index = std::fs::read_to_string(corpus_dir(&p).join("cases.tsv")).unwrap();
    let rows: Vec<Vec<&str>> = index.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()).map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), cases.len());
    let sample = read_one(&std::fs::read(corpus_dir(&p).join(KMEANS_SAMPLE)).unwrap()).unwrap();
    assert_eq!(sample.dims, vec![KMEANS_TIMING_N, KMEANS_TIMING_D]);
    let exe = have_futhark().then(|| compile_c(&std::fs::read_to_string(corpus_dir(&p).join("reference64.fut")).unwrap(), &scratch("reference64")).unwrap());
    assert!(exe.is_some(), "futhark 0.28.1 is needed for this test");
    let mut covered = vec![0u8; KMEANS_TIMING_N];
    for ((c, row), cut) in cases.iter().zip(&rows).zip(&KMEANS_CASES) {
        let (n, d, k): (usize, usize, usize) = (row[3].parse().unwrap(), row[4].parse().unwrap(), row[5].parse().unwrap());
        let seed: u64 = row[6].parse().unwrap();
        assert_eq!((row[0], row[1], row[2], n, d, k, seed), (cut.name, cut.split, format!("{}..{}", cut.start, cut.end).as_str(), cut.end - cut.start, KMEANS_TIMING_D, cut.k, cut.seed), "{}: cases.tsv against the table", c.name);
        assert_eq!(format!("{:?}", c.split).to_lowercase(), cut.split, "{}", c.name);
        assert_eq!((c.inputs[0].dims.clone(), c.inputs[1].dims.clone()), (vec![n, d], vec![k, d]), "{}", c.name);
        assert!(k * 20 <= n, "{}: k {k} over n / 20 ({n})", c.name);
        let (points, starts) = cut_case(&sample, cut).unwrap();
        assert_eq!(c.inputs[0], points, "{}: the points are not the sample's rows {}..{}", c.name, cut.start, cut.end);
        assert_eq!(c.inputs[1], starts, "{}: the starts are not UltraDim's k-means++ draws at seed {}", c.name, cut.seed);
        assert_eq!(c.stdin, write_binary(&[&points, &starts]), "{}: the stored bytes", c.name);
        let pts = c.inputs[0].f32_values().unwrap();
        assert!(pts.chunks_exact(d).all(|r| (r.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt() - 1.0).abs() < 1e-3), "{}: every row is a unit vector", c.name);
        if cut.name != "train_n200_k4" {
            for r in &mut covered[cut.start..cut.end] {
                *r += 1;
            }
        } else {
            assert!(KMEANS_CASES.iter().any(|o| o.split == "train" && o.name != cut.name && o.start <= cut.start && cut.end <= o.end), "the small case lies inside a training case");
        }
        // The f64 reference through futhark c against the Rust f64 step, bit for bit.
        let t = std::time::Instant::now();
        let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(exe.as_ref().unwrap()).arg("-b"), std::time::Duration::from_secs(600), Some(&c.stdin)).unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let (got_a, got_c) = read_i32_then_f64(&out.stdout, n, k * d);
        let (want_a, want_c) = ultradim_step_f64(&c.inputs[0], &c.inputs[1]).unwrap();
        assert_eq!(got_a, want_a.iter().map(|a| *a as i32).collect::<Vec<_>>(), "{}: the assignments", c.name);
        assert_eq!(got_c.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want_c.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{}: the f64 centres differ in bits", c.name);
        let counts = want_a.iter().fold(vec![0usize; k], |mut v, a| { v[*a] += 1; v });
        eprintln!("{}: rows {}..{} ({n} x {d}), k {k}, seed {seed}; the first step's cluster sizes {}..{}; the f64 reference agrees bit for bit ({:.1} s)", c.name, cut.start, cut.end, counts.iter().min().unwrap(), counts.iter().max().unwrap(), t.elapsed().as_secs_f64());
    }
    assert!(covered.iter().all(|r| *r == 1), "the six cuts use every one of the 5,000 rows exactly once");
}

/// The two values `reference64.fut` writes, from Futhark's binary format: `[n]i32`, then `[k][d]f64` (`read_binary`
/// reads the signature family's i32 and f32 only).
fn read_i32_then_f64(raw: &[u8], n: usize, cells: usize) -> (Vec<i32>, Vec<f64>) {
    assert_eq!(&raw[..7], b"b\x02\x01 i32", "the assignments' header");
    let mut at = 7 + 8;
    let a: Vec<i32> = raw[at..at + 4 * n].chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    at += 4 * n;
    // Futhark writes a newline between two values.
    while raw[at].is_ascii_whitespace() {
        at += 1;
    }
    assert_eq!(&raw[at..at + 7], b"b\x02\x02 f64", "the centres' header: {:?}", &raw[at..at + 7]);
    at += 7 + 16;
    let c: Vec<f64> = raw[at..at + 8 * cells].chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    (a, c)
}
