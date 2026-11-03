//! THE K-MEANS PROBLEM, the first f32 view (`docs/PLAN_futhark_kingdom.md`, step 4): the kingdom loads from the
//! master and the sniffer's specification; the published step written as a gene is exact (the comparison policy of
//! `Array`: bit for bit, two NaNs agree) on every stored case through the host interpreter, through `futhark c` and
//! batched; the f32 rows' semantics equal `futhark c`'s on the edges (NaN, the signed zeros, the infinities, a
//! denormal); random genes express to programs `futhark check` accepts and `futhark c` computes as the interpreter
//! does; the planted slowdown is equal and the e-graph recovers the reference's work; the layout at the runner's
//! default program size holds both.

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
    assert_eq!(view.spec.master, Some(3));
    let functions = k.rows.iter().filter(|r| r.kind == FutharkKind::Function).count();
    let ops: BTreeSet<&str> = k.rows.iter().map(|r| r.name.split('@').next().unwrap_or("").split('.').next().unwrap_or("")).collect();
    for op in ["div", "sqrt", "tabrows", "histrows", "rowdiv", "hist", "clamp", "map2", "sum", "loop"] {
        assert!(ops.contains(op), "the k-means view has no {op} row: {ops:?}");
    }
    assert!(!ops.contains("mod"), "the mod row is i32 only");
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], KMEANS_PROGRAM_SIZE).unwrap();
    eprintln!("kmeans: {} types, {} rows ({functions} functions, {} leaves); N = {KMEANS_PROGRAM_SIZE}: {} words a gene, head {}, at most {} nodes; fingerprint {:016x}", k.duals.len(), k.rows.len(), k.rows.len() - functions, layout.width, layout.head_lens().values().sum::<usize>(), layout.max_nodes(), view.fingerprint());
    // Measured 2026-11-03 (count contexts at depth 3; pinned, as the i32 views' sizes are in futhark_view::tests).
    assert_eq!((k.duals.len(), functions, k.rows.len() - functions), KMEANS_SIZES, "the k-means view's sizes changed");
}

/// (types, function rows, leaf rows), measured 2026-11-03.
const KMEANS_SIZES: (usize, usize, usize) = (3804, 21620, 8421);

#[test]
fn the_kmeans_reference_and_planted_slowdown_are_exact_on_every_stored_case_interpreted() {
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let cases = load_cases(&KMEANS_PROBLEM).unwrap();
    let count = |s: Split| cases.iter().filter(|c| c.split == s).count();
    assert_eq!((count(Split::Train), count(Split::Held), count(Split::Test)), (3, 2, 0));
    let mut exact = 0;
    for (what, text) in [("reference", KMEANS_REFERENCE), ("planted slowdown", KMEANS_PLANTED)] {
        let tree = build(&k, text).unwrap();
        assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), text.split_whitespace().collect::<String>());
        for c in &cases {
            let t = std::time::Instant::now();
            let got = interp.eval(&tree, &c.inputs).unwrap();
            assert_eq!(got.elem, Elem::F32);
            assert_eq!(got, c.output, "the {what} on {}", c.name);
            eprintln!("kmeans {what} on {}: exact ({} cells, {:.3} s)", c.name, got.len(), t.elapsed().as_secs_f64());
            exact += 1;
        }
    }
    assert_eq!(exact, 2 * cases.len());
    let reference = build(&k, KMEANS_REFERENCE).unwrap();
    eprintln!("kmeans reference: {} functions", function_count(&k, &reference));
}

#[test]
fn the_kmeans_reference_is_exact_compiled_and_batched_and_fits_the_default_layout() {
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], KMEANS_PROGRAM_SIZE).unwrap();
    let const_len = k.const_len();
    let tree = build(&k, KMEANS_REFERENCE).unwrap();
    for (what, text) in [("reference", KMEANS_REFERENCE), ("planted slowdown", KMEANS_PLANTED)] {
        let t = build(&k, text).unwrap();
        assert!(function_count(&k, &t) < KMEANS_PROGRAM_SIZE, "the {what} has {} functions", function_count(&k, &t));
        fits(&contract, &layout, &t).unwrap_or_else(|e| panic!("the {what} does not fit N = {KMEANS_PROGRAM_SIZE}: {e}"));
    }
    let mut rng = GeneRng::new(3);
    let gene = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &tree).unwrap();
    assert_eq!(express(&contract, &layout, &gene, &const_len).unwrap().tree(), tree, "the write-back");
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let cases = load_cases(&KMEANS_PROBLEM).unwrap();
    let src = program(&k, &sig, &tree).unwrap();
    let exe = compile_c(&src, &scratch("reference")).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for c in &cases {
        assert_eq!(run_exe(&exe, &c.stdin).unwrap(), c.output, "the compiled reference on {}", c.name);
    }
    let batch = program_cases(&k, &sig, &tree, cases.len()).unwrap();
    let bexe = compile_c(&batch, &scratch("reference_batch")).unwrap_or_else(|e| panic!("{e}\n{batch}"));
    let stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
    let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin)).unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(read_binary(&out.stdout).unwrap(), cases.iter().map(|c| c.output.clone()).collect::<Vec<_>>(), "the batched reference");
}

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
    let planted = build(&k, KMEANS_PLANTED).unwrap();
    let reference = build(&k, KMEANS_REFERENCE).unwrap();
    let cases = load_cases(&KMEANS_PROBLEM).unwrap();
    let sizes = sig.sizes(&cases[2].inputs).unwrap();
    let (wp, wr) = (interp.work(&planted, &sizes).unwrap(), interp.work(&reference, &sizes).unwrap());
    assert!(wp > wr, "planted {wp}, reference {wr}");
    // The work at the timing input's sizes, against the timing bound.
    let timing: BTreeMap<String, usize> = [("n", KMEANS_PROBLEM.timing_n), ("d", KMEANS_TIMING_D), ("k", KMEANS_TIMING_K)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    let wt = interp.work(&reference, &timing).unwrap();
    eprintln!("kmeans work: reference {wr}, planted {wp} on {sizes:?}; the reference at the timing input {wt} (the bound {TIMING_WORK_CAP})");
    let bridge = Bridge::load(&KMEANS).unwrap();
    let vs = variants(&k, &bridge, &planted, 3, 256, 64).unwrap();
    assert!(!vs.is_empty(), "no variants");
    for v in &vs {
        for c in &cases {
            assert_eq!(interp.eval(v, &c.inputs).unwrap(), c.output, "the variant {} on {}", sexp(&k, v), c.name);
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

/// THE DATA MAKES THE ORDER IRRELEVANT (`KMEANS_GRANULE_BITS`): every stored case's points and centres, and the timing
/// input's, are multiples of 2^-6 in [0, 1), and the two sums of the step fit f32 whole for the case's n and d
/// (`kmeans_dyadic`: n * 2^6 < 2^24 granules for a cluster sum, d * 2^12 < 2^24 for a squared distance), so every
/// partial sum of every order of additions is exact and the step's output is one set of bits on every path up to the
/// final divisions. The sufficient condition is what is proved; the cases' outputs are then checked exact on each CPU
/// path by the other tests. Also counted: the points whose two nearest centres tie (distances are exact, so the
/// published fold settles a tie by index, the same on every path); measured 0 of 1,678 on the stored cases.
#[test]
fn every_stored_case_and_the_timing_input_are_dyadic_so_every_order_of_additions_is_exact() {
    let p = problem("kmeans").unwrap();
    let cases = load_cases(&p).unwrap();
    assert_eq!(cases.len(), 5);
    let mut ties = 0usize;
    for c in &cases {
        let (sum_granules, dist_granules) = kmeans_dyadic(&c.inputs[0], &c.inputs[1]).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let (n, d, k) = (c.inputs[0].dims[0], c.inputs[0].dims[1], c.inputs[1].dims[0]);
        assert_eq!((sum_granules, dist_granules), ((n as u64) << 6, (d as u64) << 12));
        // The squared distances in f64 are exact (multiples of 2^-12 below d); a point whose two nearest centres tie.
        let (pts, cen) = (c.inputs[0].f32_values().unwrap(), c.inputs[1].f32_values().unwrap());
        for i in 0..n {
            let mut dist: Vec<f64> = (0..k).map(|j| (0..d).map(|t| (f64::from(pts[i * d + t]) - f64::from(cen[j * d + t])).powi(2)).sum()).collect();
            dist.sort_by(f64::total_cmp);
            ties += usize::from(k > 1 && dist[0] == dist[1]);
        }
        eprintln!("kmeans {}: dyadic, n = {n} (2^6 n = {sum_granules} < 2^24), d = {d} (2^12 d = {dist_granules} < 2^24)", c.name);
    }
    eprintln!("kmeans: {ties} points of the stored cases tie between their two nearest centres");
    let stdin = timing_input(&p).unwrap().expect("k-means has its own timing input");
    let inputs = read_binary(&stdin).unwrap();
    let (sum_granules, dist_granules) = kmeans_dyadic(&inputs[0], &inputs[1]).unwrap_or_else(|e| panic!("the timing input: {e}"));
    assert_eq!((sum_granules, dist_granules), (8192 << 6, 256 << 12), "the timing input's bounds: 2^19 and 2^20 granules");
    // The condition is a condition: 2^18 points or 2^12 coordinates would not fit f32 whole.
    let too_many = Array::f32s(vec![1 << 18, 1], vec![0.5; 1 << 18]);
    assert!(kmeans_dyadic(&too_many, &Array::f32s(vec![1, 1], vec![0.5])).is_err());
    let too_wide = Array::f32s(vec![1, 1 << 12], vec![0.5; 1 << 12]);
    assert!(kmeans_dyadic(&too_wide, &too_wide).is_err());
    assert!(kmeans_dyadic(&Array::f32s(vec![1, 1], vec![0.3]), &Array::f32s(vec![1, 1], vec![0.5])).is_err(), "0.3 is not a multiple of 2^-6");
    assert!(kmeans_dyadic(&Array::f32s(vec![1, 1], vec![1.0]), &Array::f32s(vec![1, 1], vec![0.5])).is_err(), "1 is out of [0, 1)");
}

/// THE TIMING INPUT is the champion's shape (`docs/kmeans_champion_s1/README.md`: n = 8,192 points, d = 256, k =
/// 1,024), fixed bytes, points and centres dyadic in [0, 1); the published program (the yardstick, `futhark c`) runs on it
/// and gives a `[k][d]f32`. The host interpreter is not run on it: measured here at the rate it does the stored
/// cases, the reference's work there (2.1e11 node executions, printed with the timing bound) would take over an
/// hour, so the timing input's output is the yardstick's (as the runner takes it, `TimingInput::of`).
#[test]
fn the_kmeans_timing_input_is_the_champions_shape_and_the_yardstick_runs_on_it() {
    let p = problem("kmeans").unwrap();
    let stdin = timing_input(&p).unwrap().expect("k-means has its own timing input");
    let inputs = read_binary(&stdin).unwrap();
    assert_eq!(inputs.len(), 2, "the points and the centres");
    assert_eq!((inputs[0].dims.clone(), inputs[0].elem), (vec![8192, KMEANS_TIMING_D], Elem::F32));
    assert_eq!((inputs[1].dims.clone(), inputs[1].elem), (vec![KMEANS_TIMING_K, KMEANS_TIMING_D], Elem::F32));
    assert!(inputs.iter().all(|a| a.values().all(|x| (0.0..1.0).contains(&x))), "points and centres in [0, 1)");
    assert_eq!(timing_input(&p).unwrap().unwrap(), stdin, "the input is fixed: the same bytes every time");
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let sizes = sig.sizes(&inputs).unwrap();
    assert_eq!((sizes["n"], sizes["d"], sizes["k"]), (8192, 256, 1024));
    let interp = Interp::new(&k, &sig).unwrap();
    let reference = build(&k, KMEANS_REFERENCE).unwrap();
    // The interpreter's rate on the largest stored case, and the work at the timing input.
    let cases = load_cases(&p).unwrap();
    let big = cases.iter().max_by_key(|c| c.inputs[0].len()).unwrap();
    let t = std::time::Instant::now();
    assert_eq!(interp.eval(&reference, &big.inputs).unwrap(), big.output);
    let secs = t.elapsed().as_secs_f64();
    let w_big = interp.work(&reference, &sig.sizes(&big.inputs).unwrap()).unwrap();
    let w_timing = interp.work(&reference, &sizes).unwrap();
    eprintln!("kmeans: the reference on {} ({w_big} node executions) in {secs:.3} s, {:.2e} a second; at the timing input {w_timing} node executions, {:.0} s at that rate; the timing bound {TIMING_WORK_CAP}", big.name, w_big as f64 / secs, w_timing as f64 / (w_big as f64 / secs));
    if !have_futhark() {
        panic!("futhark is not installed: the published program cannot be checked on the timing input");
    }
    let exe = compile_c(p.yardstick, &scratch("timing")).unwrap();
    let t = std::time::Instant::now();
    let got = run_exe(&exe, &stdin).unwrap();
    eprintln!("kmeans: the yardstick (futhark c) on the timing input in {:.2} s", t.elapsed().as_secs_f64());
    assert_eq!((got.dims.clone(), got.elem), (vec![KMEANS_TIMING_K, KMEANS_TIMING_D], Elem::F32));
    assert!(got.values().all(|x| x.is_finite() && (0.0..1.0).contains(&x)), "every centre is a mean of points in [0, 1), or zero");
}
