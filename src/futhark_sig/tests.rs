//! The signature family's checks that hold for every problem: the published programs and the yardsticks give the
//! published and stored outputs; Futhark's binary format round trips; the comparison policy; the quality measures;
//! the tree edit distance. The views' own checks (every reference, random genes compiled and interpreted, the e-graph)
//! are in `tests_interp6`, `tests_references6` and the master 6 modules' tests.

use super::*;
use std::process::Command;

fn have_futhark() -> bool {
    Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_sig_{name}_{}", std::process::id()))
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
    let medium_out = Array::i32s(vec![data.len()], data);
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
    let k = Arc::new(kingdom6("pathfinder").unwrap());
    let sig = signature("pathfinder").unwrap();
    let tree = build6(&k, PATHFINDER_REFERENCE6).unwrap();
    let w = read_binary(&walls).unwrap();
    assert_eq!(Interp::new6(&k, &sig).unwrap().eval(&tree, &w).unwrap(), medium_out, "the reference gene does not give medium.out");
    let _ = std::fs::remove_dir_all(&b);
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
fn the_binary_format_round_trips() {
    let a = Array::i32s(vec![2, 3], vec![1, -2, 3, i32::MIN, i32::MAX, 0]);
    let s = Array::scalar_i32(7);
    let raw = write_binary(&[&a, &s]);
    assert_eq!(read_binary(&raw).unwrap(), vec![a.clone(), s]);
    assert!(read_one(&raw).is_err());
    assert_eq!(compare(&a, &Array { dims: vec![3, 2], elem: Elem::I32, data: a.data.clone() }), [1.0; 3]);
    let mut b = a.clone();
    b.data[0] = 2;
    let q = compare(&b, &a);
    assert!(q[0] > 0.16 && q[0] < 0.17 && q[1] > 0.0 && q[2] > 0.0, "{q:?}");
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

/// The quality inputs give partial correctness a slope: on every pathfinder case, an output right on half the cells
/// scores clearly better than one right on none, an output off by one everywhere better still than all zero, and
/// the answer is 0 on every input.
#[test]
fn the_quality_inputs_give_partial_correctness_a_slope() {
    let angle = |q: &[f64]| crate::runner::staged_hff(crate::runner::FitnessStage::Quality, q, &[]).unwrap().quality;
    for c in load_cases(&PATHFINDER_PROBLEM).unwrap() {
        let w = c.answer().unwrap();
        let with = |f: &dyn Fn(usize, i32) -> i32| Array::i32s(w.dims.clone(), w.i32_values().unwrap().iter().enumerate().map(|(i, x)| f(i, *x)).collect());
        assert_eq!(quality_inputs(w, w), vec![0.0; quality_input_names(w).len()], "{}: the answer is not 0", c.name);
        let zero = angle(&quality_inputs(&with(&|_, _| 0), w));
        if w.data.len() >= 2 {
            let half = angle(&quality_inputs(&with(&|i, x| if i % 2 == 0 { x } else { 0 }), w));
            assert!(half < zero - 0.2, "{}: half right {half:.4}, none right {zero:.4}", c.name);
        }
        let near = angle(&quality_inputs(&with(&|_, x| x + 1), w));
        // A margin of 0.1: on held_1x37 (walls 0..9, the answer is row 0) one off is up to 100 % of a cell.
        assert!(near < zero - 0.1, "{}: off by one {near:.4}, all zero {zero:.4}", c.name);
        assert_eq!(quality_inputs(&Array::i32s(vec![w.len() + 1], vec![0; w.len() + 1]), w), vec![1.0; quality_input_names(w).len()]);
    }
}

/// The tree edit distance: 0 to itself, 1 a relabel, 1 a leaf inserted, symmetric; the textbook example of Zhang and
/// Shasha (f(d(a c(b)) e) against f(c(d(a b)) e)) is 2; and on the histogram, a one-symbol edit of the reference is
/// at distance 1.
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
    let k = kingdom6("histogram").unwrap();
    let r = build6(&k, HISTOGRAM_REFERENCE6).unwrap();
    assert_eq!(tree_distance(&r, &r), 0);
    assert_eq!(tree_distance(&r, &build6(&k, "(hist.min@a.i32 k (tab@a.i32[i] n (mod@i32 k (get@i32 is i))) vs)").unwrap()), 1);
}

/// THE QUALITY FINGERPRINT IS PINNED: the gate of a discovery run compares it with the slope check's record, so a
/// change to the formula's values on i32 outputs would invalidate every recorded gate. Recorded at a535d464, before
/// the values became typed (step 4): the i32 formula must give the same bits after.
#[test]
fn the_quality_fingerprint_of_the_i32_formula_is_pinned() {
    let every = [QualityPart::Wrong, QualityPart::CellLog, QualityPart::CellLevel, QualityPart::Step, QualityPart::Shape, QualityPart::Order, QualityPart::Outside(10)];
    assert_eq!((quality_fingerprint(), quality_fingerprint_of(&[QualityPart::Study]), quality_fingerprint_of(&every)), ("80b71efa5c5adfb0".to_string(), "7405ef68d2104e9c".to_string(), "756d9a9553fe716b".to_string()));
}
