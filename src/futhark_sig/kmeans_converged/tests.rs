//! The k-means fitness, tested on hand-made runs (the inputs' arithmetic, total on [0, 1]), on hand-made data (the
//! measures), on Rust steps through the loop (UltraDim's step; a step that does not finish; the cap), on the dbpedia
//! sample (the identity, a worsening, a NaN, a rounded and a normalised step through the whole scheme against the
//! seed; the measures against UltraDim's vendored functions; the program's own time against the harness's), and on
//! the textbook pool program through the host interpreter and through `futhark c`.

use super::*;
use crate::futhark_sig::{build, compile_c, load, load_cases, program, signature, Interp, KMEANS, KMEANS_PROBLEM, KMEANS_TEXTBOOK};

fn run(inertia: f64, mean_log10_p: f64, seconds: f64, iterations: u32) -> ConvergedRun {
    ConvergedRun { centroids: Vec::new(), assignments: Vec::new(), inertia, mean_log10_p, mean_log10_p_all: mean_log10_p, coincident: 0, seconds, wall_seconds: seconds, iterations, inertia_per_iteration: 1.0, converged: true, finished: true, failure: String::new() }
}

/// THE INPUTS' ARITHMETIC (`docs/KMEANS_FITNESS.md`): a run equal to the seed is 1/2 on every input; a worse one is
/// above 1/2 on the inputs where it is worse and nowhere else; a better one below; a run that did not finish is 1 on
/// all four; and EVERY INPUT IS DEFINED ON [0, 1] FOR EVERY OUTCOME (the review's F1): a candidate with no tightness,
/// zero seconds, a negative inertia, a seed measure of 0, a NaN or an infinite measure.
#[test]
fn the_inputs_are_half_at_the_seed_above_when_worse_below_when_better_one_when_unfinished_and_always_in_range() {
    let seed = run(0.4, -2.0, 1.5, 12);
    let half = kmeans_inputs(&seed, &seed);
    assert_eq!((half.quality, half.speed), (vec![0.5; 2], vec![0.5; 2]));
    // Twice the inertia, half the tightness, twice the seconds, twice the iterations: 2/(1 + 2) on every input.
    let worse = run(0.8, -1.0, 3.0, 24);
    let w = kmeans_inputs(&worse, &seed);
    for x in w.quality.iter().chain(&w.speed) {
        assert!((x - 2.0 / 3.0).abs() < 1e-12, "{w:?}");
    }
    let better = run(0.2, -4.0, 0.75, 6);
    let b = kmeans_inputs(&better, &seed);
    for x in b.quality.iter().chain(&b.speed) {
        assert!((x - 0.5 / 1.5).abs() < 1e-12, "{b:?}");
    }
    // One measure worse, the others equal: only its input moves.
    let slow = run(0.4, -2.0, 4.5, 12);
    assert_eq!(kmeans_inputs(&slow, &seed), Inputs { quality: vec![0.5, 0.5], speed: vec![0.75, 0.5] });
    let unfinished = ConvergedRun { finished: false, failure: "a NaN".into(), ..better.clone() };
    assert_eq!(kmeans_inputs(&unfinished, &seed), Inputs::worst());
    // The cap (UltraDim's `converged` false) is a finished run scored on what it has: 100 iterations is slow, not worst.
    let capped = ConvergedRun { converged: false, iterations: MAX_ITERATIONS, ..seed.clone() };
    assert_eq!(kmeans_inputs(&capped, &seed), Inputs { quality: vec![0.5, 0.5], speed: vec![0.5, 100.0 / 112.0] });
    // TOTAL: the identity-like run (two iterations, no time, no tightness at all) and the degenerate seeds.
    let in_range = |i: &Inputs| i.quality.iter().chain(&i.speed).all(|x| (0.0..=1.0).contains(x) && x.is_finite());
    let identity = run(0.6, 0.0, 0.0, 2);
    let i = kmeans_inputs(&identity, &seed);
    assert!(in_range(&i), "{i:?}");
    assert_eq!(i.quality[1], 1.0, "no tightness at all is the worst percentile input");
    assert_eq!(i.speed[0], 0.0, "no time at all is the best seconds input");
    for (c, s) in [
        (run(f64::NAN, f64::NAN, f64::NAN, 0), seed.clone()),
        (run(f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, u32::MAX), seed.clone()),
        (run(-1e-9, -2.0, 1.5, 12), seed.clone()),
        (run(0.1, -2.0, 0.1, 1), run(0.0, 0.0, 0.0, 0)),
        (run(0.0, 0.0, 0.0, 0), run(0.0, 0.0, 0.0, 0)),
        (seed.clone(), run(f64::NAN, f64::NAN, f64::NAN, 0)),
    ] {
        let i = kmeans_inputs(&c, &s);
        assert!(in_range(&i), "{c:?} against {s:?}: {i:?}");
        assert!(crate::fitness::converged_gated(&i.quality, &i.speed).is_ok(), "{i:?}");
    }
    assert_eq!(kmeans_inputs(&run(-1e-9, -2.0, 1.5, 12), &seed).quality[0], 0.0, "a negative inertia is better than perfect");
    let z = kmeans_inputs(&run(0.1, -2.0, 0.1, 1), &run(0.0, 0.0, 0.0, 0));
    assert_eq!(z, Inputs { quality: vec![1.0, 0.0], speed: vec![1.0, 1.0] }, "a positive measure against a seed's 0 is the worst ratio; a tightness against a seed with none the best");
    // Two measures of 0 are the ratio 0 / MIN_POSITIVE = 0 (the seed never has a zero quality measure on real data).
    assert_eq!(kmeans_inputs(&run(0.1, 0.0, 0.1, 1), &run(0.1, 0.0, 0.1, 1)).quality[1], 0.0);
    assert_eq!(squashed_ratio(1.0, 1.0), 0.5);
    assert_eq!(squashed_ratio(f64::NAN, 1.0), 1.0);
    assert_eq!(squashed_ratio(1.0, f64::NAN), 1.0);
    assert_eq!(squashed_ratio(f64::INFINITY, 1.0), 1.0);
    assert_eq!(squashed_ratio(0.0, 0.0), 0.0);
    // The fitness, quality first: the better run scores below the seed, the worse above, the unfinished above that; a
    // run that is slower but no worse in quality scores above the seed and below a worse one.
    let f = |r: &ConvergedRun| {
        let i = kmeans_inputs(r, &seed);
        crate::fitness::converged_gated(&i.quality, &i.speed).unwrap().fitness
    };
    assert!(f(&better) < f(&seed) && f(&seed) < f(&slow) && f(&slow) < f(&worse) && f(&worse) < f(&unfinished), "{} {} {} {} {}", f(&better), f(&seed), f(&slow), f(&worse), f(&unfinished));
    assert!(crate::fitness::converged_gated(&[], &[]).is_err(), "no case, no fitness");
    assert!(crate::fitness::converged_gated(&[0.5, 1.5], &[0.5, 0.5]).is_err(), "an input outside [0, 1] is a bug");
    assert!(crate::fitness::converged_gated(&[0.5, 0.5], &[f64::NAN, 0.5]).is_err(), "a NaN input is a bug");
    // The retired scheme's inputs on the same runs: its NaN on no gain is why it is retired.
    let no_gain = ConvergedRun { inertia_per_iteration: 0.0, ..identity.clone() };
    assert!(kmeans_inputs_five(&no_gain, &seed)[4].is_nan(), "the retired rate input on no gain");
    assert_eq!(converged_work_cap(7), 70);
}

/// THE MEASURES on hand-made data: the assignment is the nearest centre, the first on a tie; the Euclidean inertia is
/// UltraDim's `calculate_inertia`; the cosine inertia on unit vectors is UltraDim's formula; the percentile has
/// UltraDim's anchors (the equator is chance, the antipode the worst, the centroid the tightest, finite) and the beta
/// machinery matches closed forms; a point at its centroid is counted and left out of the fitness's percentile.
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
    let s_unit = spherical(&u, &c, &[0, 0], 2);
    let s_big = spherical(&big, &c, &[0, 0], 2);
    assert!((s_unit.inertia - s_big.inertia).abs() < 1e-6 && (s_unit.mean_log10_p - s_big.mean_log10_p).abs() < 1e-9, "{s_unit:?} {s_big:?}");
    // The first point is AT the centroid: one coincident point, left out of the fitness's mean, in UltraDim's.
    assert_eq!(s_unit.coincident, 1);
    assert_eq!(s_unit.mean_log10_p, log10_angular_percentile(0.5, 2));
    assert_eq!(s_unit.mean_log10_p_all, (log10_angular_percentile(1.0, 2) + log10_angular_percentile(0.5, 2)) / 2.0);
    assert!(s_unit.mean_log10_p_all < s_unit.mean_log10_p, "the coincident point drags UltraDim's mean down");
    assert!(at_centroid(1.0) && at_centroid(1.0 - 1e-7) && !at_centroid(0.999) && !at_centroid(0.0) && !at_centroid(-1.0) && !at_centroid(-(1.0 - 1e-7)), "the antipode is not at the centroid");
    // ln B anchors: B(1, 1) = 1, B(1/2, 1/2) = pi.
    assert!(ln_beta(1.0, 1.0).abs() < 1e-14, "{}", ln_beta(1.0, 1.0));
    assert!((ln_beta(0.5, 0.5) - std::f64::consts::PI.ln()).abs() < 1e-14);
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
    // The sizes of the clamp at d = 1,536 (the review's F2): a point at its centroid against one a rounding off it.
    let (at, off, typical) = (log10_angular_percentile(1.0, 1536), log10_angular_percentile(1.0 - 1e-7, 1536), log10_angular_percentile(0.75, 1536));
    assert!(at < -100_000.0 && at > -1_000_000.0, "{at}");
    assert!(off < -1000.0 && off > -10_000.0, "{off}");
    assert!(typical < -100.0 && typical > -1000.0, "{typical}");
    eprintln!("log10 p at d = 1536: at the centroid {at:.0}, one f32 rounding off it {off:.0}, at cos 0.75 {typical:.1}");
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

/// UltraDim's step as a timed Rust step.
fn ultradim(points: &Array) -> impl FnMut(&Array) -> Result<Stepped, String> + '_ {
    timed(move |c| ultradim_step_f32(points, c, true))
}

/// THE LOOP on Rust steps: UltraDim's step converges on the small case (converged, a few iterations, the inertia down
/// from the start, the measures finite); the compensated and the plain sum agree on it; a step that returns a NaN does
/// not finish and says so; a step that never settles reaches the cap, which is UltraDim's `converged` false on a
/// finished run (measured, 100 iterations); a step of the wrong shape does not finish; the interleaved run keeps each
/// side's measures and takes the median of its seconds.
#[test]
fn the_loop_converges_on_ultradims_step_and_measures_the_cap_a_nan_a_wrong_shape_and_a_failing_step() {
    let (points, start) = small_case();
    let seed = converge(&points, &start, ultradim(&points)).unwrap();
    assert!(seed.converged && seed.finished, "{}", seed.failure);
    assert!(seed.iterations >= 2 && seed.iterations < MAX_ITERATIONS, "{} iterations", seed.iterations);
    assert!(seed.inertia_per_iteration > 0.0 && seed.inertia.is_finite() && seed.mean_log10_p.is_finite() && seed.seconds > 0.0 && seed.wall_seconds >= seed.seconds, "{seed:?}");
    assert_eq!(seed.centroids.len(), 3);
    assert_eq!(seed.assignments.iter().filter(|a| **a == 0).count(), 20, "each group its centre");
    // The coincident count is the count of points within 0.01 radians of their centroid's direction (in two
    // dimensions with coordinates on a 1/1000 grid, some are: the rule is sized for d = 1,536).
    let unit2 = |v: &[f32]| { let n = (v[0] * v[0] + v[1] * v[1]).sqrt(); [f64::from(v[0] / n), f64::from(v[1] / n)] };
    let pts = points.f32_values().unwrap();
    let want: usize = (0..60).filter(|i| { let p = unit2(&pts[i * 2..i * 2 + 2]); let c = unit2(&seed.centroids[seed.assignments[*i]]); at_centroid(p[0] * c[0] + p[1] * c[1]) }).count();
    assert_eq!(seed.coincident, want);
    eprintln!("ultradim step on the small case: {} iterations, cosine inertia {:.3e}, mean log10 p {:.3}, {:.1} us of {:.1} us wall", seed.iterations, seed.inertia, seed.mean_log10_p, seed.seconds * 1e6, seed.wall_seconds * 1e6);
    let plain = converge(&points, &start, timed(|c| ultradim_step_f32(&points, c, false))).unwrap();
    assert_eq!((plain.iterations, plain.converged), (seed.iterations, true));
    // The same clustering to f32 rounding (the plain sum rounds the centroids a little differently): 1/2 on the
    // quality inputs to 1e-4, exactly 1/2 on the iterations.
    let inputs = kmeans_inputs(&plain, &seed);
    assert_eq!(inputs.speed[1], 0.5, "{inputs:?}");
    assert!(inputs.quality.iter().all(|x| (x - 0.5).abs() < 1e-4), "{inputs:?}");
    // A NaN on the second iteration.
    let mut calls = 0;
    let nan = converge(&points, &start, timed(|c| {
        calls += 1;
        if calls == 2 { Ok(Array::f32s(vec![3, 2], vec![f32::NAN; 6])) } else { ultradim_step_f32(&points, c, true) }
    }))
    .unwrap();
    assert!(!nan.finished && !nan.converged && nan.failure.contains("iteration 2") && nan.failure.contains("NaN"), "{}", nan.failure);
    assert_eq!(nan.iterations, 2);
    assert_eq!(kmeans_inputs(&nan, &seed), Inputs::worst());
    // Never settles: the centres cycle between two placements. The cap: finished, measured, not converged, 100.
    let mut flip = false;
    let cycling = converge(&points, &start, timed(|_| {
        flip = !flip;
        Ok(Array::f32s(vec![3, 2], if flip { vec![1.0, 1.0, 7.0, 1.0, 1.0, 7.0] } else { vec![2.0, 2.0, 8.0, 2.0, 2.0, 8.0] }))
    }))
    .unwrap();
    assert!(cycling.finished && !cycling.converged && cycling.iterations == MAX_ITERATIONS && cycling.failure.is_empty(), "{cycling:?}");
    let c = kmeans_inputs(&cycling, &seed);
    assert!(c.quality.iter().all(|x| *x < 1.0) && c.speed[1] > 0.9 && c.speed[1] < 1.0, "the cap is slow, not worst: {c:?}");
    assert_eq!(kmeans_inputs_five(&cycling, &seed), vec![1.0; 5], "the retired scheme scored the cap worst");
    // The wrong shape.
    let shape = converge(&points, &start, timed(|_| Ok(Array::f32s(vec![2, 2], vec![0.0; 4])))).unwrap();
    assert!(!shape.finished && shape.failure.contains("[3][2]f32"), "{}", shape.failure);
    // A failing step.
    let failing = converge(&points, &start, timed(|_| Err::<Array, String>("no such row".into()))).unwrap();
    assert!(!failing.finished && failing.failure.contains("no such row"), "{}", failing.failure);
    // Interleaved: 3 pairs, both sides UltraDim's step; the measures are a single run's, the seconds a median.
    let (c, s) = converge_interleaved(&points, &start, 3, ultradim(&points), ultradim(&points)).unwrap();
    assert_eq!((c.iterations, c.inertia, c.mean_log10_p), (s.iterations, s.inertia, s.mean_log10_p));
    assert!(c.seconds > 0.0 && s.seconds > 0.0);
    assert!(converge_interleaved(&points, &start, 0, timed(|c| Ok(c.clone())), timed(|c| Ok(c.clone()))).is_err());
    // d = 1 cannot be measured (the spherical percentile needs d >= 2).
    assert!(converge(&Array::f32s(vec![2, 1], vec![0.0, 1.0]), &Array::f32s(vec![1, 1], vec![0.5]), timed(|c| Ok(c.clone()))).unwrap_err().contains("d >= 2"));
}

/// THE PROGRAM'S OWN TIME, NOT THE HARNESS'S (the review's F4): `seconds` is the sum of what the steps report, exactly,
/// whatever the harness takes around them. A step that sleeps 20 ms and reports 0 gives 0 seconds while the wall
/// clock holds the 20 ms; a step that reports 0.5 s and takes none gives iterations x 0.5 exactly. Then the compiled
/// path: the seed through `futhark c` on 200 dbpedia rows, its `-t` runtime positive and below the process's wall
/// clock on every iteration.
#[test]
fn the_programs_seconds_are_the_steps_own_time_and_the_harness_is_excluded() {
    let (points, start) = small_case();
    let sleeping = converge(&points, &start, |c| {
        std::thread::sleep(std::time::Duration::from_millis(20));
        Ok(Stepped { centres: ultradim_step_f32(&points, c, true)?, seconds: 0.0 })
    })
    .unwrap();
    assert!(sleeping.converged);
    assert_eq!(sleeping.seconds, 0.0, "{sleeping:?}");
    assert!(sleeping.wall_seconds >= 0.02 * f64::from(sleeping.iterations), "{} s wall for {} iterations", sleeping.wall_seconds, sleeping.iterations);
    let reporting = converge(&points, &start, |c| Ok(Stepped { centres: ultradim_step_f32(&points, c, true)?, seconds: 0.5 })).unwrap();
    assert_eq!(reporting.seconds, 0.5 * f64::from(reporting.iterations));
    assert!(reporting.wall_seconds < reporting.seconds);
    // The compiled seed: `-t` is the entry point's own time, under the process's.
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let (points, start) = dbpedia_case(0..200, 4, 1);
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let dir = std::env::temp_dir().join(format!("kmeans_converged_timer_{}", std::process::id()));
    let exe = compile_c(&program(&k, &sig, &build(&k, &crate::futhark_sig::kmeans_seed()).unwrap()).unwrap(), &dir).unwrap();
    let mut step = step_of_exe(&exe, &points, StepTimer::FutharkC);
    let mut walls = Vec::new();
    let r = converge(&points, &start, |c| {
        let t = std::time::Instant::now();
        let s = step(c)?;
        walls.push((s.seconds, t.elapsed().as_secs_f64()));
        Ok(s)
    })
    .unwrap();
    assert!(r.converged, "{r:?}");
    assert!(walls.iter().all(|(own, wall)| *own > 0.0 && own < wall), "{walls:?}");
    assert!(r.seconds > 0.0 && r.seconds < r.wall_seconds, "{} s own, {} s wall", r.seconds, r.wall_seconds);
    eprintln!("the compiled seed on 200 x 1536 x 4 (futhark c): {} iterations, the program's own {:.4} s of {:.4} s wall; a step {:?}", r.iterations, r.seconds, r.wall_seconds, walls[0]);
    let _ = std::fs::remove_dir_all(dir);
}

fn have_futhark() -> bool {
    std::process::Command::new("futhark").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// Rows of the dbpedia sample (`tests/corpus/futhark_problems/kmeans/dbpedia/sample_5000x1536.in`) with UltraDim's
/// k-means++ starts at `seed`, as a case's two inputs.
fn dbpedia_case(rows: std::ops::Range<usize>, k: usize, seed: u64) -> (Array, Array) {
    let sample = crate::futhark_sig::read_one(&std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_problems/kmeans/dbpedia/sample_5000x1536.in")).unwrap()).unwrap();
    let d = sample.dims[1];
    let flat = sample.f32_values().unwrap();
    let pts: Vec<f32> = flat[rows.start * d..rows.end * d].to_vec();
    let chosen = kmeans_plus_plus_rows(&pts, d, k, seed).unwrap();
    let starts: Vec<f32> = chosen.iter().flat_map(|i| pts[i * d..(i + 1) * d].to_vec()).collect();
    (Array::f32s(vec![rows.len(), d], pts), Array::f32s(vec![k, d], starts))
}

/// THE MEASURED TABLE OF THE REVIEW'S F1, THROUGH THE WHOLE SCHEME, on 500 dbpedia rows with k = 8 (k-means++ starts):
/// the seed (UltraDim's step), the identity step (the start centres returned), a worsening step (the start centres
/// pushed off the data), a NaN step, the seed's step rounded to 1e-3 and to 1e-2, and the seed's step with its
/// centroids made unit length (the spherical variant). The run continues on every one (no error); the identity, the
/// worsening, the NaN and both rounded steps rank BELOW the seed; the identity does not beat the seed on the
/// percentile input; the normalised step's rank is a finding, printed. Every input in [0, 1].
#[test]
fn the_identity_a_worsening_a_nan_and_a_rounded_step_rank_below_the_seed_through_the_scheme() {
    let (points, start) = dbpedia_case(2000..2500, 8, 3);
    let d = points.dims[1];
    let seed = converge(&points, &start, ultradim(&points)).unwrap();
    assert!(seed.converged, "{seed:?}");
    let f = |r: &ConvergedRun| {
        let i = kmeans_inputs(r, &seed);
        let t = crate::fitness::converged_gated(&i.quality, &i.speed).unwrap();
        assert!(i.quality.iter().chain(&i.speed).all(|x| (0.0..=1.0).contains(x)), "{i:?}");
        (t.fitness, i)
    };
    let (seed_f, seed_i) = f(&seed);
    assert_eq!((seed_i.quality.clone(), seed_i.speed.clone()), (vec![0.5; 2], vec![0.5; 2]));
    let identity = converge(&points, &start, timed(|c| Ok(c.clone()))).unwrap();
    assert!(identity.converged && identity.iterations == 2, "{identity:?}");
    assert!(identity.coincident >= 8, "the identity's centres are its 8 start rows, each at its own centroid: {} coincident", identity.coincident);
    let (identity_f, identity_i) = f(&identity);
    let pushed: Vec<f32> = start.f32_values().unwrap().chunks_exact(d).enumerate().flat_map(|(j, row)| row.iter().enumerate().map(|(t, x)| x + if (t + j) % 2 == 0 { 0.05 } else { -0.05 }).collect::<Vec<f32>>()).collect();
    let pushed = Array::f32s(vec![8, d], pushed);
    let worsening = converge(&points, &start, timed(|_| Ok(pushed.clone()))).unwrap();
    assert!(worsening.converged && worsening.inertia_per_iteration < 0.0, "a worsening step: {worsening:?}");
    let (worsening_f, worsening_i) = f(&worsening);
    let mut calls = 0;
    let nan = converge(&points, &start, timed(|c| {
        calls += 1;
        if calls == 3 { Ok(Array::f32s(vec![8, d], vec![f32::NAN; 8 * d])) } else { ultradim_step_f32(&points, c, true) }
    }))
    .unwrap();
    assert!(!nan.finished);
    let (nan_f, nan_i) = f(&nan);
    let rounded = |q: f32| {
        let pts = &points;
        converge(pts, &start, timed(move |c| {
            let s = ultradim_step_f32(pts, c, true)?;
            Ok(Array::f32s(s.dims.clone(), s.f32_values()?.iter().map(|x| (x / q).round() * q).collect()))
        }))
        .unwrap()
    };
    let (r3, r2) = (rounded(1e-3), rounded(1e-2));
    assert!(r3.converged && r2.converged);
    let (r3_f, r3_i) = f(&r3);
    let (r2_f, r2_i) = f(&r2);
    let normalised = converge(&points, &start, timed(|c| {
        let s = ultradim_step_f32(&points, c, true)?;
        let rows: Vec<f32> = s.f32_values()?.chunks_exact(d).flat_map(unit).collect();
        Ok(Array::f32s(s.dims.clone(), rows))
    }))
    .unwrap();
    let (norm_f, norm_i) = f(&normalised);
    eprintln!("the scheme on 500 dbpedia rows, k 8 (quality: cosine inertia ratio, percentile ratio; speed: seconds ratio, iterations ratio; fitness, quality first):");
    for (name, r, fit, i) in [("seed", &seed, seed_f, &seed_i), ("identity", &identity, identity_f, &identity_i), ("worsening", &worsening, worsening_f, &worsening_i), ("NaN at iteration 3", &nan, nan_f, &nan_i), ("rounded 1e-3", &r3, r3_f, &r3_i), ("rounded 1e-2", &r2, r2_f, &r2_i), ("normalised centroids", &normalised, norm_f, &norm_i)] {
        eprintln!("  {name:22} iterations {:3} converged {:5} finished {:5} cosine inertia {:.5} mean log10 p {:9.2} (all points {:9.2}, {} at their centroid) quality [{:.4}, {:.4}] speed [{:.4}, {:.4}] fitness {fit:.4}", r.iterations, r.converged, r.finished, r.inertia, r.mean_log10_p, r.mean_log10_p_all, r.coincident, i.quality[0], i.quality[1], i.speed[0], i.speed[1]);
    }
    assert!(identity_f > seed_f, "the identity beats the seed: {identity_f} against {seed_f}");
    assert!(identity_i.quality[1] >= 0.5, "the identity beats the seed on the percentile input: {identity_i:?}");
    assert!(identity_i.quality[0] > 0.5, "the identity beats the seed on the cosine inertia: {identity_i:?}");
    assert!(worsening_f > seed_f, "a worsening step beats the seed: {worsening_f} against {seed_f}");
    assert!(nan_f > worsening_f, "a NaN step is not the worst: {nan_f} against {worsening_f}");
    assert!(r3_f > seed_f, "the step rounded to 1e-3 beats the seed: {r3_f} against {seed_f} ({r3_i:?})");
    assert!(r2_f > seed_f, "the step rounded to 1e-2 beats the seed: {r2_f} against {seed_f} ({r2_i:?})");
    assert!(norm_f.is_finite());
}

/// ULTRADIM'S K-MEANS++ RULE on hand-made data: three tight groups far apart draw one start from each group (the D^2
/// sampling), the first start uniform; the same seed gives the same rows and another seed other rows; all points
/// identical duplicate the first; k over n is an error; and the rule is the sequential one (the parallel distances
/// change nothing: the same rows at one thread).
#[test]
fn the_starting_centres_are_ultradims_kmeans_plus_plus_draws() {
    let (points, _) = small_case();
    let flat = points.f32_values().unwrap();
    let rows = kmeans_plus_plus_rows(&flat, 2, 3, 7).unwrap();
    let groups: std::collections::BTreeSet<usize> = rows.iter().map(|r| r / 20).collect();
    assert_eq!(groups.len(), 3, "the three starts {rows:?} are not one a group");
    assert_eq!(rows, kmeans_plus_plus_rows(&flat, 2, 3, 7).unwrap(), "the same seed, the same draws");
    assert_ne!(rows, kmeans_plus_plus_rows(&flat, 2, 3, 8).unwrap(), "another seed, other draws");
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| kmeans_plus_plus_rows(&flat, 2, 3, 7).unwrap());
    assert_eq!(rows, one);
    let same = vec![0.5f32; 10 * 2];
    let dup = kmeans_plus_plus_rows(&same, 2, 4, 1).unwrap();
    assert_eq!(dup, vec![dup[0]; 4], "identical points duplicate the first start");
    assert!(kmeans_plus_plus_rows(&flat, 2, 61, 1).is_err() && kmeans_plus_plus_rows(&flat, 2, 0, 1).is_err());
    // On the dbpedia rows: k distinct rows at k <= n / 20 (no duplicate start), every one a row index.
    let (p, s) = dbpedia_case(0..200, 4, 1);
    let d = p.dims[1];
    let pts = p.f32_values().unwrap();
    let starts = s.f32_values().unwrap();
    let found: Vec<usize> = starts.chunks_exact(d).map(|row| (0..200).position(|i| pts[i * d..(i + 1) * d] == *row).expect("a start is a row")).collect();
    assert_eq!(found.iter().collect::<std::collections::BTreeSet<_>>().len(), 4, "{found:?}");
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

/// THE TEXTBOOK POOL PROGRAM through the host interpreter, in the loop, on the smallest stored case: it runs to the
/// stopping point (an empty cluster becomes the zero vector in the textbook step, so its clustering can differ from
/// UltraDim's step's on the same case: the inputs against the UltraDim stand-in are numbers in [0, 1]) and, with
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
    let textbook = converge(points, start, timed(|cen| interp.eval(&tree, &[points.clone(), cen.clone()]))).unwrap();
    eprintln!("textbook on {}: {} iterations in {:.2} s interpreted; converged {} {}", c.name, textbook.iterations, t0.elapsed().as_secs_f64(), textbook.converged, textbook.failure);
    assert!(textbook.finished && textbook.converged, "{}", textbook.failure);
    let seed = converge(points, start, ultradim(points)).unwrap();
    assert!(seed.converged, "{}", seed.failure);
    let inputs = kmeans_inputs(&textbook, &seed);
    assert!(inputs.quality.iter().chain(&inputs.speed).all(|x| (0.0..=1.0).contains(x)), "{inputs:?}");
    eprintln!("textbook against UltraDim's step on {}: inputs {inputs:?}; textbook {} iterations, seed {}", c.name, textbook.iterations, seed.iterations);
    let fit = crate::fitness::converged_gated(&inputs.quality, &inputs.speed).unwrap();
    assert!(fit.fitness.is_finite());
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let dir = std::env::temp_dir().join(format!("kmeans_converged_textbook_{}", std::process::id()));
    let exe = compile_c(&program(&k, &sig, &tree).unwrap(), &dir).unwrap();
    let compiled = converge(points, start, step_of_exe(&exe, points, StepTimer::FutharkC)).unwrap();
    assert_eq!((compiled.iterations, compiled.converged, &compiled.centroids, &compiled.assignments), (textbook.iterations, true, &textbook.centroids, &textbook.assignments));
    assert_eq!((compiled.inertia, compiled.mean_log10_p, compiled.coincident, compiled.inertia_per_iteration), (textbook.inertia, textbook.mean_log10_p, textbook.coincident, textbook.inertia_per_iteration));
    eprintln!("textbook compiled on {}: the same {} iterations, the program's own {:.3} s of {:.3} s wall", c.name, compiled.iterations, compiled.seconds, compiled.wall_seconds);
    let _ = std::fs::remove_dir_all(dir);
}
