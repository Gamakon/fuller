//! A7 AS A PROBLEM OF THE ONE RUNNER (`docs/PLAN_futhark_kingdom.md`, step 4): the matrix kingdom loads through the
//! signature family's machinery; its signature is the interface's nine matrices; the reference, the winner and the
//! planted slowdown express from the view, read back from their s-expressions, and are exact on every stored case
//! through the host interpreter and through `futhark c` (the plain program and the quality program of every case);
//! the e-graph's variants of the reference hold the winner; random genes are accepted by `futhark check` and the
//! compiled ones equal the interpreter; the batch source holds the `matmul` helper once; the timing input is the
//! declared shape and the yardstick runs on it.

use super::*;
use crate::futhark_batch::{Backend as BatchBackend, Host, Inputs, Member, Program, Step};
use crate::typed_gep::{express, fits, write_back};
use std::process::Command;

fn have_futhark() -> bool {
    Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_sig_a7_{name}_{}", std::process::id()))
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

/// The three programs of the problem, by name.
fn programs() -> [(&'static str, &'static str); 3] {
    [("reference", A7_REFERENCE), ("winner", A7_WINNER), ("planted slowdown", A7_PLANTED)]
}

#[test]
fn the_a7_kingdom_loads_through_the_signature_family_and_its_signature_is_the_nine_matrices() {
    let p = problem("a7").unwrap();
    let k = load(&p.instance).unwrap();
    // The same view `FutharkKingdom::load` (the deprecated runner) loads: the same duals and rows.
    let old = FutharkKingdom::load().unwrap();
    assert_eq!(k.duals, old.duals);
    assert_eq!(k.rows, old.rows);
    assert_eq!((k.duals.len(), k.rows.len()), (11, 71));
    let sig = signature(&p.instance).unwrap();
    let names: Vec<&str> = sig.inputs.iter().map(|q| q.name.as_str()).collect();
    assert_eq!(names, ["m_n_n", "m_n_m", "m_n_p", "m_m_n", "m_m_m", "m_m_p", "m_p_n", "m_p_m", "m_p_p"]);
    assert_eq!(sig.inputs[1].dims, vec!["n".to_string(), "m".to_string()]);
    assert!(sig.inputs.iter().all(|q| q.elem == "f32" && q.dims.len() == 2));
    assert_eq!(sig.output, Param { name: "main".into(), elem: "f32".into(), dims: vec!["n".into(), "p".into()] });
    assert_eq!((sig.depth, sig.dims.clone(), sig.size_inputs.clone()), (0, vec!["n".to_string(), "m".into(), "p".into()], vec![]));
    // The interpreter covers every row of the view (it refuses a row with no meaning).
    Interp::new(&k, &sig).unwrap();
    // The prelude is the matmul helper, once; a signature view has none.
    assert_eq!(prelude(&k), format!("{}\n", crate::futhark::MATMUL_HELPER));
    assert_eq!(prelude(&load(&REDUCE).unwrap()), "");
}

#[test]
fn the_a7_programs_read_back_from_their_sexps_and_fit_the_default_layout() {
    let p = problem("a7").unwrap();
    let k = load(&p.instance).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], p.program_size).unwrap();
    let const_len = k.const_len();
    let mut rng = GeneRng::new(0xA7_0001);
    for (what, text) in programs() {
        let tree = build(&k, text).unwrap_or_else(|e| panic!("the {what} does not build: {e}"));
        assert_eq!(sexp(&k, &tree).split_whitespace().collect::<String>(), text.split_whitespace().collect::<String>(), "the {what}'s s-expression round trip");
        assert_eq!(build(&k, &sexp(&k, &tree)).unwrap(), tree);
        assert!(function_count(&k, &tree) < p.program_size, "the {what} has {} functions", function_count(&k, &tree));
        fits(&contract, &layout, &tree).unwrap_or_else(|e| panic!("the {what} does not fit N = {}: {e}", p.program_size));
        let gene = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &tree).unwrap();
        assert_eq!(express(&contract, &layout, &gene, &const_len).unwrap().tree(), tree, "the {what}'s write-back");
    }
    // The reference is the deprecated runner's own tree (`futhark_search::product_twice`), and its program renders
    // the recorded text's kernel.
    let reference = build(&k, A7_REFERENCE).unwrap();
    assert_eq!(reference, crate::futhark_search::product_twice(&k), "the reference gene is product_twice");
    let sizes: BTreeMap<String, usize> = [("n", 2), ("m", 3), ("p", 4)].into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    let old = crate::futhark::program(&k, k.roots()[0], &reference, &sizes).unwrap();
    let new = program(&k, &signature(&p.instance).unwrap(), &reference).unwrap();
    let body = |s: &str| s.lines().find(|l| l.contains("map2 (\\xs ys")).map(str::trim).unwrap_or("").to_string();
    assert_eq!(body(&old), body(&new), "the rendered kernel");
    assert!(!new.contains("\n  in "), "a body with no let has no `in`:\n{new}");
    assert!(new.starts_with(crate::futhark::MATMUL_HELPER), "the program opens with the helper:\n{new}");
    assert!(new.contains("entry main [n][m][p] (m_n_n: [n][n]f32) (m_n_m: [n][m]f32)"), "the declared signature:\n{new}");
    assert!(!new.contains("zeros_"), "the matrix family spells no zero array:\n{new}");
    // A row of another output class is refused by its full name, and a missing row is named.
    assert!(build(&k, "(transpose@mat.n.m mat.n.m.input)").unwrap_err().contains("is of mat.m.n, not mat.n.p"));
    assert!(build(&k, "(matmul mat.n.m.input mat.m.p.input)").unwrap_err().contains("has no row matmul@mat.n.p"));
}

#[test]
fn the_a7_reference_winner_and_planted_slowdown_are_exact_on_every_stored_case_interpreted() {
    let p = problem("a7").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let cases = load_cases(&p).unwrap();
    let count = |s: Split| cases.iter().filter(|c| c.split == s).count();
    assert_eq!((count(Split::Train), count(Split::Held), count(Split::Test)), (3, 2, 1));
    let mut exact = 0;
    for (what, text) in programs() {
        let tree = build(&k, text).unwrap();
        for c in &cases {
            let got = interp.eval(&tree, &c.inputs).unwrap();
            assert_eq!((got.elem, got.dims.clone()), (Elem::F32, c.answer().unwrap().dims.clone()));
            assert_eq!(got, *c.answer().unwrap(), "the {what} on {}", c.name);
            exact += 1;
        }
    }
    assert_eq!(exact, 3 * cases.len());
    eprintln!("a7: the reference, the winner and the planted slowdown are exact on {} of {} stored cases each (host interpreter)", cases.len(), cases.len());
    // The work model: the planted slowdown does more than the reference, the winner less; the product dominates.
    let sizes = sig.sizes(&cases[2].inputs).unwrap();
    let w = |t: &str| interp.work(&build(&k, t).unwrap(), &sizes).unwrap();
    let (wr, ww, wp) = (w(A7_REFERENCE), w(A7_WINNER), w(A7_PLANTED));
    assert!(ww < wr && wr < wp, "winner {ww}, reference {wr}, planted {wp}");
    assert!(wr >= 2 * 32 * 32 * 32, "two products of 32^3 at least: {wr}");
    // A one-cell matrix: the degenerate roles.
    let one = &cases.iter().find(|c| c.name.contains("n1_m1_p1")).unwrap();
    assert_eq!(interp.eval(&build(&k, A7_WINNER).unwrap(), &one.inputs).unwrap().dims, vec![1, 1]);
}

#[test]
fn the_a7_programs_are_exact_compiled_and_as_quality_programs_and_the_batch_source_has_one_helper() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let p = problem("a7").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let cases = load_cases(&p).unwrap();
    let stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
    let mut exact = 0;
    for (what, text) in programs() {
        let tree = build(&k, text).unwrap();
        let src = program(&k, &sig, &tree).unwrap();
        let exe = compile_c(&src, &scratch(&format!("{}_plain", what.replace(' ', "_")))).unwrap_or_else(|e| panic!("{e}\n{src}"));
        for c in &cases {
            assert_eq!(run_exe(&exe, &c.stdin).unwrap(), *c.answer().unwrap(), "the compiled {what} on {}", c.name);
            exact += 1;
        }
        let batch = program_cases(&k, &sig, &tree, cases.len()).unwrap();
        assert_eq!(batch.matches("def matmul").count(), 1, "the helper once:\n{batch}");
        let bexe = compile_c(&batch, &scratch(&format!("{}_cases", what.replace(' ', "_")))).unwrap_or_else(|e| panic!("{e}\n{batch}"));
        let out = crate::futhark_exec::output_with_timeout_bytes(Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin)).unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(read_binary(&out.stdout).unwrap(), cases.iter().map(|c| c.answer().unwrap().clone()).collect::<Vec<_>>(), "the {what}'s quality program");
    }
    eprintln!("a7: the three programs compiled with futhark c are exact on {exact} of {} case runs", 3 * cases.len());
    // THE BATCH SOURCE (`futhark_batch::Host::source`, the GPU and rust paths): two members with quality and timed
    // entries and the yardstick, one `def matmul`, accepted by `futhark check`. The host is built on the rust backend
    // (the C driver is i32 only); nothing is run.
    let dir = scratch("batch");
    let timing = timing_input(&p).unwrap().unwrap();
    let host = Host::new(&k, &sig, cases.len(), BatchBackend::Rust { rt: dir.join("rt") }, &dir, Inputs { cases_stdin: &stdin, timing_stdin: Some(&timing), yardstick: Some(p.yardstick) }).unwrap();
    let (reference, winner) = (build(&k, A7_REFERENCE).unwrap(), build(&k, A7_WINNER).unwrap());
    let members = [
        Member { program: Program::Tree(&reference), steps: vec![Step::Quality, Step::Pair { yard_first: true, itself: false }] },
        Member { program: Program::Tree(&winner), steps: vec![Step::Quality, Step::Pair { yard_first: false, itself: false }] },
        Member { program: Program::Text(p.yardstick), steps: vec![Step::Pair { yard_first: true, itself: false }] },
    ];
    let src = host.batch_source(&members).unwrap();
    // The prelude's helper once at the top level; the yardstick's own copy inside each module (the text member's and
    // the yard's), where it cannot clash.
    assert_eq!(src.matches("\ndef matmul").count(), 3, "the prelude once, one inside each module:\n{src}");
    assert_eq!(src.matches("module ").count(), 2, "the yardstick and the text member each in a module:\n{src}");
    assert_eq!(src.matches("\n}\nentry yard ").count() + src.matches("\n}\nentry t2 ").count(), 2, "each module's entry follows it:\n{src}");
    futhark_check(&src, &dir.join("check")).unwrap_or_else(|e| panic!("the batch source is not accepted: {e}\n{src}"));
    // A signature view's batch source is unchanged: no prelude, the yardstick bare.
    let rk = load(&REDUCE).unwrap();
    let rsig = signature(&REDUCE).unwrap();
    let rcases = load_cases(&REDUCE_PROBLEM).unwrap();
    let rstdin: Vec<u8> = rcases.iter().flat_map(|c| c.stdin.clone()).collect();
    let rhost = Host::new(&rk, &rsig, rcases.len(), BatchBackend::C, &dir.join("reduce"), Inputs { cases_stdin: &rstdin, timing_stdin: None, yardstick: Some(REDUCE_PROBLEM.yardstick) }).unwrap();
    let rref = build(&rk, REDUCE_REFERENCE).unwrap();
    let rsrc = rhost.batch_source(&[Member { program: Program::Tree(&rref), steps: vec![Step::Quality, Step::Pair { yard_first: true, itself: false }] }]).unwrap();
    assert!(rsrc.contains("entry yard [n]"), "the yardstick stays a bare entry without a prelude:\n{rsrc}");
    assert!(!rsrc.contains("module "), "{rsrc}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_a7_egraph_variants_of_the_reference_hold_the_winner_and_are_exact() {
    let p = problem("a7").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let cases = load_cases(&p).unwrap();
    let bridge = Bridge::load(&p.instance).unwrap();
    assert_eq!(bridge.ruleset, "futhark_shape");
    let reference = build(&k, A7_REFERENCE).unwrap();
    let winner = build(&k, A7_WINNER).unwrap();
    let vs = variants(&k, &bridge, &reference, 4, 64, 32).unwrap();
    assert!(vs.contains(&winner), "the winner is not among {} variants: {:?}", vs.len(), vs.iter().map(|v| sexp(&k, v)).collect::<Vec<_>>());
    let mut exact = 0;
    for v in &vs {
        for c in &cases {
            assert_eq!(interp.eval(v, &c.inputs).unwrap(), *c.answer().unwrap(), "the variant {} on {}", sexp(&k, v), c.name);
            exact += 1;
        }
    }
    eprintln!("a7: {} e-graph variants of the reference, the winner among them, exact on {exact} case runs", vs.len());
    // The planted slowdown's variants reach the reference and the winner (map id = id, then the transposed product).
    let planted = build(&k, A7_PLANTED).unwrap();
    let pv = variants(&k, &bridge, &planted, 4, 64, 32).unwrap();
    assert!(pv.contains(&reference) && pv.contains(&winner), "{:?}", pv.iter().map(|v| sexp(&k, v)).collect::<Vec<_>>());
    // The saturated fold's e-graph reads the same bridge.
    let fe = FoldEgraph { kingdom: &k, bridge: &bridge };
    assert_eq!(crate::saturated_fold::EgraphKingdom::rulesets(&fe), vec!["futhark_shape"]);
}

#[test]
fn random_a7_genes_are_accepted_by_futhark_check_and_equal_the_interpreter() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let p = problem("a7").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], p.program_size).unwrap();
    let const_len = k.const_len();
    let mut rng = GeneRng::new(0xA7_0003);
    let draw = |cells: usize, rng: &mut GeneRng| -> Vec<f32> {
        let mut v: Vec<f32> = (0..cells).map(|_| (rng.below(1 << 16) as f32) / 4096.0 - 8.0).collect();
        if let Some(x) = v.first_mut() {
            *x = -0.0;
        }
        v
    };
    let shapes = |n: usize, m: usize, pp: usize| vec![[n, n], [n, m], [n, pp], [m, n], [m, m], [m, pp], [pp, n], [pp, m], [pp, pp]];
    let small: Vec<Vec<Array>> = [(2usize, 3usize, 4usize), (1, 1, 1), (3, 2, 2)].iter().map(|&(n, m, pp)| shapes(n, m, pp).into_iter().map(|d| Array::f32s(d.to_vec(), draw(d[0] * d[1], &mut rng))).collect()).collect();
    let base = scratch("random");
    let (mut checked, mut compiled) = (0, 0);
    for g in 0..40 {
        let gene = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
        let tree = express(&contract, &layout, &gene, &const_len).unwrap().tree();
        let src = program(&k, &sig, &tree).unwrap();
        futhark_check(&src, &base.join(format!("c{g}"))).unwrap_or_else(|e| panic!("gene {g} is not accepted: {e}\n{src}"));
        checked += 1;
        if g < 10 {
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
    eprintln!("a7: {checked} random genes accepted by futhark check, {compiled} compiled and equal to the interpreter bit for bit");
    assert_eq!((checked, compiled), (40, 10));
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn the_a7_timing_input_is_the_declared_shape_and_the_yardstick_runs_on_it() {
    let p = problem("a7").unwrap();
    let stdin = timing_input(&p).unwrap().expect("A7 has its own timing input");
    let inputs = read_binary(&stdin).unwrap();
    assert_eq!(inputs.len(), 9);
    assert!(inputs.iter().all(|a| a.dims == vec![A7_TIMING_N, A7_TIMING_N] && a.elem == Elem::F32));
    assert!(inputs.iter().all(|a| a.values().all(|x| (-8.0..8.0).contains(&x))));
    assert!(inputs.iter().all(|a| a.data[0] == (-0.0f32).to_bits()), "the first of every matrix is -0.0");
    assert_eq!(timing_input(&p).unwrap().unwrap(), stdin, "the input is fixed");
    let sig = signature(&p.instance).unwrap();
    let sizes = sig.sizes(&inputs).unwrap();
    assert_eq!((sizes["n"], sizes["m"], sizes["p"]), (A7_TIMING_N, A7_TIMING_N, A7_TIMING_N));
    let k = load(&p.instance).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let reference = build(&k, A7_REFERENCE).unwrap();
    let w = interp.work(&reference, &sizes).unwrap();
    assert!(w <= TIMING_WORK_CAP, "the reference's work at the timing input {w} is over the bound {TIMING_WORK_CAP}");
    assert_eq!(timing_work_cap(w), TIMING_WORK_CAP);
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let exe = compile_c(p.yardstick, &scratch("timing")).unwrap();
    let got = run_exe(&exe, &stdin).unwrap();
    assert_eq!((got.dims.clone(), got.elem), (vec![A7_TIMING_N, A7_TIMING_N], Elem::F32));
    // The host interpreter agrees with the yardstick on the timing input too (2 x 256^3 multiply-adds, under a second).
    let t = std::time::Instant::now();
    assert_eq!(interp.eval(&reference, &inputs).unwrap(), got, "the interpreter against futhark c on the timing input");
    eprintln!("a7: the reference on the timing input, {w} node executions, interpreted in {:.2} s and equal to futhark c", t.elapsed().as_secs_f64());
}
