//! The k-means fitness, tested on hand-made runs (the inputs' arithmetic), on hand-made data (the measures), on
//! Rust steps through the loop (UltraDim's step, a step that does not finish), and on the textbook pool program
//! through the host interpreter and through `futhark c`.

use super::*;
use crate::futhark_sig::{build, compile_c, load, load_cases, program, signature, Interp, KMEANS, KMEANS_PROBLEM, KMEANS_TEXTBOOK};

fn run(inertia: f64, mean_log10_p: f64, seconds: f64, iterations: u32, inertia_per_iteration: f64) -> ConvergedRun {
    ConvergedRun { centroids: Vec::new(), assignments: Vec::new(), inertia, mean_log10_p, seconds, iterations, inertia_per_iteration, finished: true, failure: String::new() }
}

/// THE INPUTS' ARITHMETIC (`docs/KMEANS_FITNESS.md`): a run equal to the seed is 1/2 on every input; a worse one is
/// above 1/2 on the inputs where it is worse and nowhere else; a better one below; a run that did not finish is 1 on
/// all five; the percentile input is a ratio of p; the rate input is inverted; the fitness is TrueNorth over them.
#[test]
fn the_inputs_are_half_at_the_seed_above_when_worse_below_when_better_and_one_when_unfinished() {
    let seed = run(0.4, -2.0, 1.5, 12, 3.0);
    assert_eq!(kmeans_inputs(&seed, &seed), vec![0.5; 5]);
    // Twice the inertia, twice the p (log10 2 more), twice the seconds, twice the iterations, half the gain an
    // iteration: 2/(1 + 2) on every input.
    let worse = run(0.8, -2.0 + 2f64.log10(), 3.0, 24, 1.5);
    let w = kmeans_inputs(&worse, &seed);
    for x in &w {
        assert!((x - 2.0 / 3.0).abs() < 1e-12, "{w:?}");
    }
    let better = run(0.2, -2.0 - 2f64.log10(), 0.75, 6, 6.0);
    let b = kmeans_inputs(&better, &seed);
    for x in &b {
        assert!((x - 0.5 / 1.5).abs() < 1e-12, "{b:?}");
    }
    // The percentile input is a ratio of p: ten times the p is 10/11.
    assert!((kmeans_inputs(&run(0.4, -1.0, 1.5, 12, 3.0), &seed)[1] - 10.0 / 11.0).abs() < 1e-12);
    // One measure worse, the others equal: only its input moves.
    let slow = run(0.4, -2.0, 4.5, 12, 3.0);
    assert_eq!(kmeans_inputs(&slow, &seed), vec![0.5, 0.5, 0.75, 0.5, 0.5]);
    let unfinished = ConvergedRun { finished: false, failure: "a NaN".into(), ..better.clone() };
    assert_eq!(kmeans_inputs(&unfinished, &seed), vec![1.0; 5]);
    // A seed measure of 0 does not divide by zero: the ratio is against the smallest positive f64.
    let zero_seed = run(0.0, -2.0, 0.0, 1, 0.0);
    let z = kmeans_inputs(&run(0.1, -2.0, 0.1, 1, 0.0), &zero_seed);
    assert!(z.iter().all(|x| (0.0..=1.0).contains(x)), "{z:?}");
    assert_eq!(z[0], 1.0, "an inertia against a seed's 0 is the worst ratio");
    assert_eq!(z[4], 0.0, "no gain against no gain");
    // The fitness: TrueNorth over the inputs, 0 best; the better run scores below the seed, the worse above.
    let f = |r: &ConvergedRun| crate::fitness::converged(&kmeans_inputs(r, &seed)).unwrap().fitness;
    assert!(f(&better) < f(&seed) && f(&seed) < f(&worse) && f(&worse) < f(&unfinished), "{} {} {} {}", f(&better), f(&seed), f(&worse), f(&unfinished));
    assert!(crate::fitness::converged(&[]).is_err(), "no case, no fitness");
    assert!(crate::fitness::converged(&[0.5, 1.5]).is_err(), "an input outside [0, 1] is a bug");
}

/// THE MEASURES on hand-made data: the assignment is the nearest centre, the first on a tie; the Euclidean inertia is
/// UltraDim's `calculate_inertia`; the cosine inertia on unit vectors is UltraDim's formula; the percentile has
/// UltraDim's anchors (the equator is chance, the antipode the worst, the centroid the tightest, finite) and the beta
/// machinery matches closed forms.
#[test]
fn the_measures_are_ultradims_on_hand_made_data() {
    // d = 2: three points, two centres; the third point ties between them (distance 1 to each): the first wins.
    let points = [0.0f32, 0.0, 4.0, 0.0, 2.0, 0.0];
    let centroids = vec![vec![0.0f32, 0.0], vec![4.0, 0.0]];
    let a = assign(&points, 2, &centroids);
    assert_eq!(a, vec![0, 1, 0]);
    assert_eq!(euclidean_inertia(&points, &centroids, &a, 2), 4.0, "only the tied point is off its centre, by 2");
    // Unit vectors at 60 degrees: cosine inertia 1 - 1/2 each.
    let u = [1.0f32, 0.0, 0.5, 0.75f32.sqrt()];
    let c = vec![vec![1.0f32, 0.0]];
    let ci = cosine_inertia(&u, &c, &[0, 0], 2);
    assert!((ci - 0.25).abs() < 1e-6, "{ci}");
    // The spherical pair normalises: the same on the vectors scaled by 1000.
    let big: Vec<f32> = u.iter().map(|x| x * 1000.0).collect();
    let (s_unit, p_unit) = spherical(&u, &c, &[0, 0], 2);
    let (s_big, p_big) = spherical(&big, &c, &[0, 0], 2);
    assert!((s_unit - s_big).abs() < 1e-6 && (p_unit - p_big).abs() < 1e-9, "{s_unit} {s_big} {p_unit} {p_big}");
    // ln Gamma anchors: Gamma(1) = 1, Gamma(1/2) = sqrt(pi), Gamma(10) = 9!.
    assert!(ln_gamma(1.0).abs() < 1e-14, "{}", ln_gamma(1.0));
    assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
    assert!((ln_gamma(10.0) - 362_880f64.ln()).abs() < 1e-12);
    // I_x(1, 1) = x; I_x(1/2, 1/2) = (2/pi) asin(sqrt x).
    for x in [0.1, 0.3, 0.5, 0.9] {
        assert!((ln_betainc(1.0, 1.0, x).exp() - x).abs() < 1e-10, "{x}");
        assert!((ln_betainc(0.5, 0.5, x).exp() - 2.0 / std::f64::consts::PI * x.sqrt().asin()).abs() < 1e-10, "{x}");
    }
    // UltraDim's anchors (its quality.rs tests): F(equator) = 1/2, F(antipode) = 1, F(centroid) ~ 0 and finite.
    for d in [2usize, 3, 10, 100, 1000, 50_000] {
        let f = |c: f64| 10f64.powf(log10_angular_percentile(c, d));
        assert!((f(0.0) - 0.5).abs() < 1e-9, "d = {d}: F(equator) = {}", f(0.0));
        assert!((f(-1.0) - 1.0).abs() < 1e-9, "d = {d}: F(antipode) = {}", f(-1.0));
        assert!(f(1.0) < 1e-9 && log10_angular_percentile(1.0, d).is_finite(), "d = {d}: the centroid");
        assert!(log10_angular_percentile(1.0, d) < log10_angular_percentile(1.0 - 1e-9, d), "d = {d}: coincident is tightest");
    }
    // Monotone in the angle over the full range at d = 10.
    let mut prev = f64::NEG_INFINITY;
    let mut c = 1.0;
    while c >= -1.0 {
        let cur = log10_angular_percentile(c, 10);
        assert!(cur >= prev - 1e-9, "not monotone at cos {c}: {cur} < {prev}");
        prev = cur;
        c -= 0.05;
    }
    assert_eq!(exact_mean_log10_p(&[], 5), 0.0);
    let m = exact_mean_log10_p(&[0.0, 0.0], 7);
    assert!((m - 0.5f64.log10()).abs() < 1e-9, "{m}");
}

/// A small case: 60 points in three tight groups around (0,0), (10,0), (0,10), the starting centres off each group.
fn small_case() -> (Array, Array) {
    let mut rng = crate::futhark_table::GeneRng::new(0x5EED_0001);
    let mut pts = Vec::new();
    for g in 0..3 {
        let (cx, cy) = [(0.0f32, 0.0f32), (10.0, 0.0), (0.0, 10.0)][g];
        for _ in 0..20 {
            pts.push(cx + (rng.below(1000) as f32) / 1000.0);
            pts.push(cy + (rng.below(1000) as f32) / 1000.0);
        }
    }
    (Array::f32s(vec![60, 2], pts), Array::f32s(vec![3, 2], vec![1.0, 1.0, 7.0, 1.0, 1.0, 7.0]))
}

/// THE LOOP on Rust steps: UltraDim's step converges on the small case (finished, a few iterations, the inertia down
/// from the start, the measures finite); the compensated and the plain sum agree on it; a step that returns a NaN does
/// not finish and says so; a step that never settles reaches the cap and does not finish; a step of the wrong shape
/// does not finish; the interleaved run keeps each side's measures and takes the median of its seconds.
#[test]
fn the_loop_finishes_on_ultradims_step_and_not_on_a_nan_a_wrong_shape_or_a_step_that_never_settles() {
    let (points, start) = small_case();
    let seed = converge(&points, &start, |c| ultradim_step_f32(&points, c, true)).unwrap();
    assert!(seed.finished, "{}", seed.failure);
    assert!(seed.iterations >= 2 && seed.iterations < MAX_ITERATIONS, "{} iterations", seed.iterations);
    assert!(seed.inertia_per_iteration > 0.0 && seed.inertia.is_finite() && seed.mean_log10_p.is_finite() && seed.seconds > 0.0, "{seed:?}");
    assert_eq!(seed.centroids.len(), 3);
    assert_eq!(seed.assignments.iter().filter(|a| **a == 0).count(), 20, "each group its centre");
    eprintln!("ultradim step on the small case: {} iterations, cosine inertia {:.3e}, mean log10 p {:.3}, {:.1} us, gain an iteration {:.3}", seed.iterations, seed.inertia, seed.mean_log10_p, seed.seconds * 1e6, seed.inertia_per_iteration);
    let plain = converge(&points, &start, |c| ultradim_step_f32(&points, c, false)).unwrap();
    assert_eq!((plain.iterations, plain.finished), (seed.iterations, true));
    // The same clustering to f32 rounding (the plain sum rounds the centroids a little differently): 1/2 on the four
    // non-timing inputs to 1e-4, exactly 1/2 on the iterations.
    let inputs = kmeans_inputs(&plain, &seed);
    assert_eq!(inputs[3], 0.5, "{inputs:?}");
    assert!([0, 1, 4].iter().all(|i| (inputs[*i] - 0.5).abs() < 1e-4), "{inputs:?}");
    // A NaN on the second iteration.
    let mut calls = 0;
    let nan = converge(&points, &start, |c| {
        calls += 1;
        if calls == 2 { Ok(Array::f32s(vec![3, 2], vec![f32::NAN; 6])) } else { ultradim_step_f32(&points, c, true) }
    })
    .unwrap();
    assert!(!nan.finished && nan.failure.contains("iteration 2") && nan.failure.contains("NaN"), "{}", nan.failure);
    assert_eq!(nan.iterations, 2);
    assert_eq!(kmeans_inputs(&nan, &seed), vec![1.0; 5]);
    // Never settles: the centres cycle between two placements.
    let mut flip = false;
    let cycling = converge(&points, &start, |_| {
        flip = !flip;
        Ok(Array::f32s(vec![3, 2], if flip { vec![1.0, 1.0, 7.0, 1.0, 1.0, 7.0] } else { vec![2.0, 2.0, 8.0, 2.0, 2.0, 8.0] }))
    })
    .unwrap();
    assert!(!cycling.finished && cycling.iterations == MAX_ITERATIONS && cycling.failure.contains("cap"), "{}", cycling.failure);
    // The wrong shape.
    let shape = converge(&points, &start, |_| Ok(Array::f32s(vec![2, 2], vec![0.0; 4]))).unwrap();
    assert!(!shape.finished && shape.failure.contains("[3][2]f32"), "{}", shape.failure);
    // A failing step.
    let failing = converge(&points, &start, |_| Err::<Array, String>("no such row".into())).unwrap();
    assert!(!failing.finished && failing.failure.contains("no such row"), "{}", failing.failure);
    // Interleaved: 3 pairs, both sides UltraDim's step; the measures are a single run's, the seconds a median.
    let (c, s) = converge_interleaved(&points, &start, 3, |c| ultradim_step_f32(&points, c, true), |c| ultradim_step_f32(&points, c, true)).unwrap();
    assert_eq!((c.iterations, c.inertia, c.mean_log10_p), (s.iterations, s.inertia, s.mean_log10_p));
    assert!(c.seconds > 0.0 && s.seconds > 0.0);
    assert!(converge_interleaved(&points, &start, 0, |c| Ok(c.clone()), |c| Ok(c.clone())).is_err());
    // d = 1 cannot be measured (the spherical percentile needs d >= 2).
    assert!(converge(&Array::f32s(vec![2, 1], vec![0.0, 1.0]), &Array::f32s(vec![1, 1], vec![0.5]), |c| Ok(c.clone())).unwrap_err().contains("d >= 2"));
}

/// The f64 reference step agrees with the f32 step on the assignments and, to f32 rounding, on the centres of the
/// small case; an empty cluster keeps its centre in both.
#[test]
fn the_f64_reference_step_agrees_with_the_f32_step_and_an_empty_cluster_keeps_its_centre() {
    let (points, start) = small_case();
    let f32_out = ultradim_step_f32(&points, &start, true).unwrap().f32_values().unwrap();
    let (a64, c64) = ultradim_step_f64(&points, &start).unwrap();
    assert_eq!(a64, assign(&points.f32_values().unwrap(), 2, &[vec![1.0, 1.0], vec![7.0, 1.0], vec![1.0, 7.0]]));
    for (x, y) in f32_out.iter().zip(&c64) {
        assert!((f64::from(*x) - y).abs() <= 1e-5 * y.abs().max(1.0), "{x} against {y}");
    }
    // A fourth centre far away takes no point and stays where it was.
    let far = Array::f32s(vec![4, 2], vec![1.0, 1.0, 7.0, 1.0, 1.0, 7.0, 500.0, 500.0]);
    let out = ultradim_step_f32(&points, &far, true).unwrap().f32_values().unwrap();
    assert_eq!(&out[6..], &[500.0, 500.0]);
    let (_, out64) = ultradim_step_f64(&points, &far).unwrap();
    assert_eq!(&out64[6..], &[500.0, 500.0]);
}

/// THE TEXTBOOK POOL PROGRAM through the host interpreter, in the loop, on a stored case: it runs to the stopping point
/// (an empty cluster becomes the zero vector in the textbook step, so its clustering differs from UltraDim's step's on
/// the same case: the inputs against the UltraDim stand-in are five numbers in [0, 1], not all 1/2) and, with
/// `futhark c`, the compiled program's loop is the interpreter's bit for bit (iterations, centroids, measures).
#[test]
fn the_textbook_program_runs_to_the_stopping_point_interpreted_and_compiled_alike() {
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let interp = Interp::new(&k, &sig).unwrap();
    let tree = build(&k, KMEANS_TEXTBOOK).unwrap();
    let cases = load_cases(&KMEANS_PROBLEM).unwrap();
    let c = cases.iter().filter(|c| c.inputs[0].dims[1] >= 2).min_by_key(|c| c.inputs[0].len()).expect("a case with d >= 2");
    let (points, start) = (&c.inputs[0], &c.inputs[1]);
    let t0 = Instant::now();
    let textbook = converge(points, start, |cen| interp.eval(&tree, &[points.clone(), cen.clone()])).unwrap();
    eprintln!("textbook on {}: {} iterations in {:.2} s interpreted; finished {} {}", c.name, textbook.iterations, t0.elapsed().as_secs_f64(), textbook.finished, textbook.failure);
    assert!(textbook.finished, "{}", textbook.failure);
    let seed = converge(points, start, |cen| ultradim_step_f32(points, cen, true)).unwrap();
    assert!(seed.finished, "{}", seed.failure);
    let inputs = kmeans_inputs(&textbook, &seed);
    assert_eq!(inputs.len(), 5);
    assert!(inputs.iter().all(|x| (0.0..=1.0).contains(x)), "{inputs:?}");
    eprintln!("textbook against UltraDim's step on {}: inputs {inputs:?}; textbook {} iterations, seed {}", c.name, textbook.iterations, seed.iterations);
    let fit = crate::fitness::converged(&inputs).unwrap();
    assert!(fit.fitness.is_finite() && fit.speed == 0.0);
    if std::process::Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        let dir = std::env::temp_dir().join(format!("kmeans_converged_textbook_{}", std::process::id()));
        let exe = compile_c(&program(&k, &sig, &tree).unwrap(), &dir).unwrap();
        let compiled = converge(points, start, step_of_exe(&exe, points)).unwrap();
        assert_eq!((compiled.iterations, compiled.finished, &compiled.centroids, &compiled.assignments), (textbook.iterations, true, &textbook.centroids, &textbook.assignments));
        assert_eq!((compiled.inertia, compiled.mean_log10_p, compiled.inertia_per_iteration), (textbook.inertia, textbook.mean_log10_p, textbook.inertia_per_iteration));
        eprintln!("textbook compiled on {}: the same {} iterations, {:.3} s", c.name, compiled.iterations, compiled.seconds);
        let _ = std::fs::remove_dir_all(dir);
    } else {
        panic!("futhark 0.28.1 is needed for this test");
    }
}
