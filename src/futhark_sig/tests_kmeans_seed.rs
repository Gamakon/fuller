//! THE PROOF OF THE TRANSLATION (`docs/PLAN_futhark_kingdom.md` 4c (d), `docs/kmeans_seed.md`): the k-means seed,
//! UltraDim's three kernels as a gene of the kingdom, computes on every stored case what a straight Rust port of the
//! three shaders in f32 with the same loop order computes (`kmeans_converged::ultradim_step_f32`, Kahan included):
//! the assignments identical as integers, the centres bit for bit; compiled with `futhark c` on every dbpedia case
//! and through the host interpreter on the case small enough for it (`train_n200_k4`: the interpreter does about
//! 4e7 node executions a second, a minute a step at 1,000 x 1,536 x 16); and no farther from the f64 reference
//! (`reference64.fut`, `ultradim_step_f64`) than the port.
//! The planted slowdown is the same bits, and the seed, the planted slowdown and the textbook pool programs fit the
//! layout at the problem's program size. The faithful two-element view (i32 assignments, `lets a.i32.n`) is measured:
//! 131,825 types without the kernel boundary and 237,325 with it, over the 65,536 type codes, so it does not load as a
//! kingdom; the one-element view holds the integers exactly in f32.

use super::*;
use crate::futhark_sig::kmeans_converged::{assign, ultradim_step_f32, ultradim_step_f64};
use crate::typed_gep::{express, fits, write_back};

fn have_futhark() -> bool {
    std::process::Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_sig_kmeans_seed_{name}_{}", std::process::id()))
}

/// The assignments the seed's kernel boundary binds, as integers, through the interpreter: the `let`'s bound
/// expression built at its own type (`a.n:c`) and evaluated on the case.
fn seed_assignments(k: &FutharkKingdom, interp: &Interp, c: &Case) -> Vec<usize> {
    let t = crate::futhark_rings::build_at(k, &KMEANS_SEED_ASSIGN.replace("DIST", KMEANS_SEED_DIST), "a.n:c").unwrap();
    let a = interp.eval(&t, &c.inputs).unwrap();
    assert_eq!((a.elem, a.dims.clone()), (Elem::F32, vec![c.inputs[0].dims[0]]));
    a.f32_values().unwrap().into_iter().map(|x| {
        assert!(x >= 0.0 && x.fract() == 0.0 && x < 16_777_216.0, "an assignment that is not a small integer: {x}");
        x as usize
    }).collect()
}

/// The largest cell distance of an f32 output from the f64 reference's centres.
fn distance_from_f64(out: &Array, exact: &[f64]) -> f64 {
    out.f32_values().unwrap().iter().zip(exact).map(|(x, y)| (f64::from(*x) - y).abs()).fold(0f64, f64::max)
}

#[test]
fn the_seed_builds_reads_back_fits_the_layout_and_its_size_is_measured() {
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let contract = k.relations().check().unwrap();
    let t0 = std::time::Instant::now();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], KMEANS_PROGRAM_SIZE).unwrap();
    eprintln!("kmeans: {} types, {} rows; N = {KMEANS_PROGRAM_SIZE}: {} words a gene, head {}, max nodes {}; layout {:.1} s", k.duals.len(), k.rows.len(), layout.width, layout.head_lens().values().sum::<usize>(), layout.max_nodes(), t0.elapsed().as_secs_f64());
    let const_len = k.const_len();
    let mut rng = GeneRng::new(0x5EED_4C0D);
    let mut counts = Vec::new();
    for (what, text) in [("seed", kmeans_seed()), ("planted slowdown", kmeans_planted()), ("textbook", KMEANS_TEXTBOOK.to_string()), ("textbook planted", KMEANS_TEXTBOOK_PLANTED.to_string())] {
        let t = build(&k, &text).unwrap_or_else(|e| panic!("the {what} does not build: {e}"));
        assert_eq!(sexp(&k, &t).split_whitespace().collect::<String>(), text.split_whitespace().collect::<String>(), "the {what} reads back");
        let n = function_count(&k, &t);
        assert!(n < KMEANS_PROGRAM_SIZE, "the {what} has {n} functions, the program size is {KMEANS_PROGRAM_SIZE}");
        fits(&contract, &layout, &t).unwrap_or_else(|e| panic!("the {what} does not fit N = {KMEANS_PROGRAM_SIZE}: {e}"));
        let gene = write_back(&contract, &layout, &random_gene(&contract, &layout, &const_len, &mut rng).unwrap(), &t).unwrap();
        assert_eq!(express(&contract, &layout, &gene, &const_len).unwrap().tree(), t, "the {what}'s write-back");
        counts.push((what, n));
    }
    eprintln!("kmeans function counts: {counts:?}");
    assert_eq!(counts[0].1, SEED_FUNCTIONS, "the seed's function count");
    assert_eq!(counts[1].1, PLANTED_FUNCTIONS, "the planted slowdown's function count");
    // The seed's program text is the one in the pool (`kingdoms/futhark/sig/kmeans/pool/seed.fut`), the gene the
    // pool's line.
    let src = program(&k, &sig, &build(&k, &kmeans_seed()).unwrap()).unwrap();
    assert_eq!(src, include_str!("../../kingdoms/futhark/sig/kmeans/pool/seed.fut"), "the pool's seed.fut is not the rendered seed");
    let pool = crate::futhark_rings::load_pool(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kingdoms/futhark/sig/kmeans/pool/seed.tsv")).unwrap();
    assert_eq!(pool.len(), 1);
    assert_eq!(pool[0].sexp, sexp(&k, &build(&k, &kmeans_seed()).unwrap()), "the pool's seed.tsv is not the seed's gene");
    assert_eq!(pool[0].dual, "a.k.d:c");
}

/// The seed's functions (the cell's loop written three times, the centre's distance twice), and the planted
/// slowdown's (the assignments twice).
const SEED_FUNCTIONS: usize = 131;
const PLANTED_FUNCTIONS: usize = 183;

/// THE PROOF on every stored case: the assignments equal the f32 port's as integers (through the interpreter on the
/// small case; through the f64 reference's on every case); the centres are the f32 port's bit for bit (the same
/// order of additions, both sequential), compiled with `futhark c` on every case (the plain program and the quality
/// program) and interpreted on the small case; no farther from the f64 reference than the port (the same, since the
/// bits are); the planted slowdown the same bits (interpreted, the small case). The counts are printed.
#[test]
fn the_seed_equals_the_f32_port_bit_for_bit_and_the_f64_reference_within_the_ports_distance_on_every_case() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let cases = load_cases(&KMEANS_PROBLEM).unwrap();
    let seed = build(&k, &kmeans_seed()).unwrap();
    let planted = build(&k, &kmeans_planted()).unwrap();
    let src = program(&k, &sig, &seed).unwrap();
    let exe = compile_c(&src, &scratch("seed")).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let batch = program_cases(&k, &sig, &seed, cases.len()).unwrap();
    let bexe = compile_c(&batch, &scratch("seed_batch")).unwrap_or_else(|e| panic!("{e}\n{batch}"));
    let (mut n_cases, mut assignments_equal, mut centres_identical, mut within, mut interpreted) = (0, 0, 0, 0, 0);
    let mut wants = Vec::new();
    for c in &cases {
        let (points, centres) = (&c.inputs[0], &c.inputs[1]);
        let d = points.dims[1];
        let rows: Vec<Vec<f32>> = centres.f32_values().unwrap().chunks_exact(d).map(<[f32]>::to_vec).collect();
        let port_assign = assign(&points.f32_values().unwrap(), d, &rows);
        let port = ultradim_step_f32(points, centres, true).unwrap();
        let (f64_assign, exact) = ultradim_step_f64(points, centres).unwrap();
        let t0 = std::time::Instant::now();
        let got = run_exe(&exe, &c.stdin).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        assert_eq!(port_assign, f64_assign, "{}: the f32 port's assignments differ from the f64 reference's", c.name);
        assignments_equal += 1;
        assert_eq!(got.dims, centres.dims);
        assert!(got.data == port.data, "{}: the compiled seed's centres differ in bits from the f32 port's", c.name);
        centres_identical += 1;
        let (d_seed, d_port) = (distance_from_f64(&got, &exact), distance_from_f64(&port, &exact));
        assert!(d_seed <= d_port, "{}: the seed is farther from the f64 reference ({d_seed:.3e}) than the port ({d_port:.3e})", c.name);
        within += 1;
        let mut how = format!("compiled {secs:.3} s");
        if points.dims[0] <= 200 {
            let t0 = std::time::Instant::now();
            let interp_out = interp.eval(&seed, &c.inputs).unwrap();
            assert_eq!(seed_assignments(&k, &interp, c), port_assign, "{}: the interpreted assignments", c.name);
            assert!(interp_out.data == got.data, "{}: compiled and interpreted differ", c.name);
            assert!(interp.eval(&planted, &c.inputs).unwrap().data == got.data, "{}: the planted slowdown differs", c.name);
            how.push_str(&format!(", interpreted {:.3} s and the planted slowdown the same bits", t0.elapsed().as_secs_f64()));
            interpreted += 1;
        }
        eprintln!("{}: n {} d {d} k {}: assignments equal, centres bit-identical to the f32 port ({how}), {:.3e} from the f64 reference (the port {:.3e})", c.name, points.dims[0], centres.dims[0], d_seed, d_port);
        n_cases += 1;
        wants.push(got);
    }
    let stdin: Vec<u8> = cases.iter().flat_map(|c| c.stdin.clone()).collect();
    let out = crate::futhark_exec::output_with_timeout_bytes(std::process::Command::new(&bexe).arg("-b"), std::time::Duration::from_secs(120), Some(&stdin)).unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let batched = read_binary(&out.stdout).unwrap();
    assert_eq!(batched.len(), wants.len());
    for (b, w) in batched.iter().zip(&wants) {
        assert!(b.data == w.data, "the batched seed differs");
    }
    eprintln!("the seed: {n_cases} cases, assignments equal {assignments_equal}, centres bit-identical to the f32 port {centres_identical}, within the f64 reference's distance {within}, {interpreted} interpreted; compiled and batched alike");
    assert_eq!((n_cases, assignments_equal, centres_identical, within), (cases.len(), cases.len(), cases.len(), cases.len()));
    assert!(interpreted >= 1, "no case small enough for the host interpreter");
}

/// THE FAITHFUL TWO-ELEMENT VIEW, the one the plan's item 1 asked for (the assignments an `i32` array beside the f32
/// points, `lets a.i32.n`), measured 2026-11-03: at master version 4 (the pairs, no triple, no kernel boundary) it is
/// 53,773 types and 416,671 rows, 14 times the view of 4a, and its gene at N = 96 is 62,693,641 words, laid out in
/// 488 s; at version 5 the triples take it to 131,825 types and 1,014,061 rows WITHOUT the kernel boundary, and to
/// 237,325 types with it (`lets a.i32.n`, loaded once in 4.5 minutes): both over the 65,536 type codes a Futhark kingdom
/// has (`Ty::MAX_FUTHARK_DUALS`), so neither loads as a kingdom and the i32 translation cannot be built as a gene. That
/// is why the problem's view holds the assignments and the count as integers in f32 (`docs/kmeans_seed.md`, "The two
/// views"). This test pins the version-5 count without the let (96 s to load) and the limit.
#[test]
fn the_two_element_view_is_over_the_type_codes_with_or_without_the_kernel_boundary() {
    let two = KMEANS.spec.replace("element\tf32\t\t\n", "element\tf32,i32\t\t\n").replace("lets\ta.n\t\t\n", "").replace("literals\t0,1,2,-1,0.5,10\t\t\n", "literals\t0,1,2,-1,0.5,10\tf32\t\nliterals\t0,1,2,-1,10\ti32\t\n");
    assert_ne!(two, KMEANS.spec);
    let t0 = std::time::Instant::now();
    let view = crate::futhark_view::View::load(&crate::futhark_view::Master::committed().unwrap(), &two).unwrap();
    let types = view.types_tsv.lines().count() - 1;
    let rows = view.functions_tsv.lines().count() - 1;
    eprintln!("kmeans with i32 assignments, no kernel boundary: {types} types, {rows} rows, loaded in {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!((types, rows), (131_825, 1_014_061), "the two-element view's size changed");
    assert!(types > Ty::MAX_FUTHARK_DUALS, "the two-element view is over the type codes");
    assert!(view.kingdom("two elements").unwrap_err().contains("65536"));
    // The problem's one-element view, for the ratio.
    let one = load(&KMEANS).unwrap();
    assert_eq!(one.duals.len(), 14_359);
    eprintln!("the two-element view is {:.1}x the one-element view with the kernel boundary, {:.1}x the one-element view of 4a (3,804 types)", types as f64 / one.duals.len() as f64, types as f64 / 3804.0);
    // With the kernel boundary the view has 237,325 types (measured 2026-11-03, 4.5 minutes to load); the kingdom
    // loader names the limit: `Ty::check_dual_count`.
    let e = Ty::check_dual_count("Futhark Sig kmeans two elements", 237_325, Ty::MAX_FUTHARK_DUALS).unwrap_err();
    assert!(e.contains("237325") && e.contains("65536"), "{e}");
}
