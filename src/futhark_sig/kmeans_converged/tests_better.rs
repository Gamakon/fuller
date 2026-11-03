//! THE SCHEME `futhark-kmeans-better` (Andrew's specification of 2026-11-03, `docs/KMEANS_FITNESS.md`), tested on
//! hand-made runs (the three cliff inputs and the two speed inputs, total on [0, 1]), on a `k` record with the further
//! iterations, and on two dbpedia cases through the whole scheme: the table of angle, HFF percentile and d for the
//! reference program and nine individuals, with the first k-means run's best individual compiled by `futhark c`.

use super::tests::{dbpedia_case, have_futhark, reversed_step, ultradim};
use super::*;
use crate::fitness::{hff_percentile, quality_cliff, DEFAULT_QUALITY_CLIFF, KMEANS_BETTER_QUALITY_WEIGHT};
use crate::futhark_sig::{build, compile_c_server, load, program, signature, KMEANS};

fn run(inertia: f64, mean_log10_p: f64, seconds: f64, iterations: u32, further: Option<u32>) -> ConvergedRun {
    ConvergedRun { centroids: Vec::new(), assignments: Vec::new(), inertia, mean_log10_p, mean_log10_p_all: mean_log10_p, coincident: 0, seconds, wall_seconds: seconds, iterations, inertia_per_iteration: 1.0, converged: true, finished: true, failure: String::new(), further_iterations: further }
}

/// THE INPUTS' ARITHMETIC: equal to the reference is 1/2 on all five; a loss of ε or more on any quality measure is 1
/// on that input and nowhere else; half the loss is 3/4; a gain is below 1/2 (x/2); a candidate whose centroids need
/// two more of the reference's iterations is 1 on the convergence input; one at least as converged is at most 1/2;
/// an unfinished run is 1 on all five; a finished run without its further iterations, or a bad cliff, is our bug.
#[test]
fn the_better_inputs_are_the_cliff_of_each_ratio_and_total() {
    let eps = DEFAULT_QUALITY_CLIFF;
    let reference = run(0.4, -400.0, 1.5, 10, Some(2));
    let same = kmeans_inputs_better(&reference, &reference, eps).unwrap();
    assert_eq!((same.quality.clone(), same.speed.clone()), (vec![0.5; 3], vec![0.5; 2]));
    let lossy = kmeans_inputs_better(&run(0.4 * 1.16, -400.0, 1.5, 10, Some(2)), &reference, eps).unwrap();
    assert_eq!(lossy.quality, vec![1.0, 0.5, 0.5], "a 16 % worse inertia is the worst inertia input");
    let half = kmeans_inputs_better(&run(0.4, -400.0 / (1.0 + eps / 2.0), 1.5, 10, Some(2)), &reference, eps).unwrap();
    assert!((half.quality[1] - 0.75).abs() < 1e-9 && half.quality[0] == 0.5, "{half:?}");
    let gain = kmeans_inputs_better(&run(0.2, -800.0, 1.5, 10, Some(2)), &reference, eps).unwrap();
    assert_eq!(gain.quality, vec![0.25, 0.25, 0.5]);
    let early = kmeans_inputs_better(&run(0.4, -400.0, 0.3, 3, Some(4)), &reference, eps).unwrap();
    assert_eq!(early.quality[2], 1.0, "two more of the reference's iterations: x = 3");
    assert!((early.speed[0] - 0.2 / 1.2).abs() < 1e-12 && (early.speed[1] - 0.3 / 1.3).abs() < 1e-12, "{early:?}");
    assert_eq!(kmeans_inputs_better(&run(0.4, -400.0, 1.5, 10, Some(1)), &reference, eps).unwrap().quality[2], 0.0, "one fewer: x = 0");
    let unfinished = ConvergedRun { finished: false, failure: "a NaN".into(), ..reference.clone() };
    let w = kmeans_inputs_better(&unfinished, &reference, eps).unwrap();
    assert_eq!((w.quality, w.speed), (vec![1.0; 3], vec![1.0; 2]));
    assert_eq!(kmeans_inputs_better(&run(f64::NAN, -400.0, 1.5, 10, Some(2)), &reference, eps).unwrap().quality, vec![1.0; 3], "a non-finite measure is 1 on every input");
    assert_eq!(kmeans_inputs_better(&run(0.4, 0.0, 1.5, 10, Some(2)), &reference, eps).unwrap().quality[1], 1.0, "no tightness at all");
    assert!(kmeans_inputs_better(&run(0.4, -400.0, 1.5, 10, None), &reference, eps).unwrap_err().contains("bug"));
    assert!(kmeans_inputs_better(&reference, &run(0.4, -400.0, 1.5, 10, None), eps).unwrap_err().contains("k_ref"));
    assert!(kmeans_inputs_better(&reference, &reference, 0.0).is_err() && kmeans_inputs_better(&reference, &reference, f64::NAN).is_err());
    assert_eq!(better_ratios(&run(0.8, -200.0, 1.0, 1, Some(5)), &reference), [2.0, 2.0, 4.0]);
    // The flat fitness at m = 3 over one case: the reference acos(3/4); a 16 % loss alone is above it however fast.
    let f = |i: &Inputs| crate::fitness::qqs(&i.quality, &i.speed, KMEANS_BETTER_QUALITY_WEIGHT).unwrap().fitness;
    assert!((f(&same) - 0.75f64.acos()).abs() < 1e-12);
    assert!(f(&Inputs { quality: lossy.quality.clone(), speed: vec![0.0, 0.0] }) > f(&same));
}

/// A `k` RECORD WITH THE FURTHER ITERATIONS reads back into `further_iterations`; a record without them leaves it
/// `None`; a continuation that stopped short of both the rule and the cap is our bug; an unfinished run has none.
#[test]
fn a_k_record_with_the_further_iterations_reads_back() {
    let (points, start) = dbpedia_case(0..200, 4, 1);
    let r = converge(&points, &start, ultradim(&points)).unwrap();
    let centres = Array::f32s(vec![4, points.dims[1]], r.centroids.concat());
    let base = k_payload(r.iterations, true, true, 1.0, 10.0, 9.0, &k_tail(&centres, &r.assignments));
    assert_eq!(run_of_k_payload(&points, &start, &base, 1).unwrap().further_iterations, None);
    let with = k_payload_further(&base, (2, true)).unwrap();
    let back = run_of_k_payload(&points, &start, &with, 1).unwrap();
    assert_eq!((back.further_iterations, back.iterations, back.centroids.clone()), (Some(2), r.iterations, r.centroids.clone()));
    assert_eq!(run_of_k_payload(&points, &start, &k_payload_further(&base, (MAX_ITERATIONS, false)).unwrap(), 1).unwrap().further_iterations, Some(MAX_ITERATIONS), "the cap is a measurement");
    assert!(run_of_k_payload(&points, &start, &k_payload_further(&base, (7, false)).unwrap(), 1).unwrap_err().contains("bug"));
    let failed = k_payload_further(&k_payload(3, false, false, 0.0, 10.0, 10.0, b"iteration 3: a centroid is infinite or NaN"), (0, false)).unwrap();
    assert_eq!(run_of_k_payload(&points, &start, &failed, 1).unwrap().further_iterations, None);
    assert_eq!(k_command_further("kc", 2, 2000, 500, "kyard"), "k kc 2 2000 500 100 1e-4 kyard");
    // The further iterations of the reference program from its own final centroids: a fixed point, 2.
    assert_eq!(further_iterations(&points, &r.centroids, ultradim(&points)).unwrap(), 2);
}

/// One individual's runs on the cases and what the scheme reads of them.
struct Row {
    name: &'static str,
    /// A case: (the candidate's run, the reference's run it is judged against, the inputs).
    cases: Vec<(ConvergedRun, ConvergedRun, Inputs)>,
    angle: f64,
    percentile: f64,
    d: usize,
}

/// A Rust step that reports a stated cost (`cost` seconds an iteration) instead of its own clock (the existing qqs
/// test's cost model: a Rust step's own clock is noise on this machine).
fn costed<'a>(mut f: impl FnMut(&Array) -> Result<Array, String> + 'a, cost: f64) -> impl FnMut(&Array) -> Result<Stepped, String> + 'a {
    move |c: &Array| f(c).map(|centres| Stepped { centres, seconds: cost })
}

/// A run with its further iterations: the REFERENCE program's step (UltraDim's, bit for bit the compiled seed's,
/// `tests_kmeans_seed`) run on from its final centroids, when it finished.
fn with_further(points: &Array, mut r: ConvergedRun) -> ConvergedRun {
    if r.finished {
        r.further_iterations = Some(further_iterations(points, &r.centroids, ultradim(points)).unwrap());
    }
    r
}

/// THE ORDERING TABLE OF `futhark-kmeans-better` (Andrew's specification of 2026-11-03) on two dbpedia cases: the
/// 500 rows of `train_n500_k8` (k = 8) and the 1,000 rows of `held_n1000_k32` (k = 32), k-means++ starts, every
/// individual through UltraDim's loop, its further iterations measured by the reference program's step from its
/// final centroids, m = 3, ε = 0.005, the fitness the angle over the one flat vector of both cases (d = 22) and its
/// HFF percentile. The individuals: the reference program (UltraDim's step); returns the centroids unchanged; makes
/// them worse (the starts pushed off the data); a NaN at iteration 3; the reference rounded to 1e-3 and to 1e-2; the
/// reference with its additions reordered (a plain f32 sum; a reversed Kahan sum); the reference faster with the
/// same output (seconds / 2); one reference iteration, then the centroids unchanged; and the first k-means run's best
/// individual (`kmeans_qqs_s1` hall of fame rank 1, read into a gene by `build`, compiled with `futhark c` and run as
/// a Futhark server interleaved with the compiled seed, both programs' own times). Asserted: every individual worse
/// than the reference by more than ε on a final quality measure (cosine inertia or percentile) on any case, or that
/// did not finish, has a higher HFF than the reference; the faster-identical one a lower; the reorderings within the
/// reorder noise of the reference (the noise: the largest |x - 1| of their quality inputs, stated, below ε, and their
/// angle within the angle that noise moves the reference to); the first run's best individual ranks below the reference.
#[test]
fn the_better_scheme_ranks_every_quality_loss_below_the_reference_and_a_faster_identical_program_above() {
    if !have_futhark() {
        panic!("futhark 0.28.1 is needed for this test");
    }
    let eps = DEFAULT_QUALITY_CLIFF;
    let m = KMEANS_BETTER_QUALITY_WEIGHT;
    let cases = [("train_n500_k8", 2000..2500, 8usize, 3u64), ("held_n1000_k32", 2500..3500, 32, 5)];
    let k = load(&KMEANS).unwrap();
    let sig = signature(&KMEANS).unwrap();
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_problems/kmeans/first_run/kmeans_qqs_s1_rank1.txt")).unwrap();
    let sexp = text.lines().find(|l| l.starts_with('(')).expect("the hall of fame entry's program line");
    let best_tree = build(&k, sexp).unwrap();
    let seed_tree = build(&k, &crate::futhark_sig::kmeans_seed()).unwrap();
    let dir = std::env::temp_dir().join(format!("kmeans_better_{}", std::process::id()));
    let best_exe = compile_c_server(&program(&k, &sig, &best_tree).unwrap(), &dir.join("best")).unwrap();
    let seed_exe = compile_c_server(&program(&k, &sig, &seed_tree).unwrap(), &dir.join("seed")).unwrap();
    let names = ["the reference program (UltraDim's step)", "returns the centroids unchanged", "makes them worse (starts pushed off the data)", "NaN at iteration 3", "rounded to 1e-3", "rounded to 1e-2", "additions reordered: plain f32 sum", "additions reordered: reversed Kahan sum", "faster, identical output (seconds / 2)", "one iteration, then unchanged", "the first run's best individual (kmeans_qqs_s1 rank 1)"];
    let mut per_case: Vec<Vec<(ConvergedRun, ConvergedRun)>> = vec![Vec::new(); names.len()];
    for (case, rows, kk, seed) in cases.iter().cloned() {
        let t0 = std::time::Instant::now();
        let (points, start) = dbpedia_case(rows, kk, seed);
        let d = points.dims[1];
        let reference = with_further(&points, converge(&points, &start, ultradim(&points)).unwrap());
        assert!(reference.converged, "{case}: {reference:?}");
        let per = reference.seconds / f64::from(reference.iterations);
        let identity = converge(&points, &start, costed(|c: &Array| Ok(c.clone()), 0.0)).unwrap();
        let pushed: Vec<f32> = start.f32_values().unwrap().chunks_exact(d).enumerate().flat_map(|(j, row)| row.iter().enumerate().map(|(t, x)| x + if (t + j) % 2 == 0 { 0.05 } else { -0.05 }).collect::<Vec<f32>>()).collect();
        let pushed = Array::f32s(vec![kk, d], pushed);
        let worsening = converge(&points, &start, costed(|_: &Array| Ok(pushed.clone()), 0.0)).unwrap();
        let mut calls = 0;
        let nan = converge(&points, &start, costed(|c: &Array| {
            calls += 1;
            if calls == 3 { Ok(Array::f32s(vec![kk, d], vec![f32::NAN; kk * d])) } else { ultradim_step_f32(&points, c, true) }
        }, per))
        .unwrap();
        assert!(!nan.finished, "{case}: the NaN step finished");
        let rounded = |q: f32| {
            let pts = &points;
            converge(pts, &start, costed(move |c: &Array| {
                let s = ultradim_step_f32(pts, c, true)?;
                Ok(Array::f32s(s.dims.clone(), s.f32_values()?.iter().map(|x| (x / q).round() * q).collect()))
            }, per))
            .unwrap()
        };
        let (r3, r2) = (rounded(1e-3), rounded(1e-2));
        let plain = converge(&points, &start, costed(|c: &Array| ultradim_step_f32(&points, c, false), per)).unwrap();
        let reversed = converge(&points, &start, costed(|c: &Array| reversed_step(&points, c), per)).unwrap();
        let faster = converge(&points, &start, costed(|c: &Array| ultradim_step_f32(&points, c, true), per / 2.0)).unwrap();
        assert_eq!((faster.iterations, faster.inertia, faster.mean_log10_p), (reference.iterations, reference.inertia, reference.mean_log10_p), "{case}: the faster step's output is the reference's");
        let mut steps = 0;
        let early = converge(&points, &start, |c| {
            steps += 1;
            if steps == 1 { Ok(Stepped { centres: ultradim_step_f32(&points, c, true)?, seconds: per }) } else { Ok(Stepped { centres: c.clone(), seconds: 0.0 }) }
        })
        .unwrap();
        // The first run's best individual against the compiled seed, two interleaved pairs, both Futhark servers
        // (each program's own `runtime:`), the seed's run of the interleaving its yardstick. The compiled seed is the
        // reference program bit for bit (`tests_kmeans_seed`), so its further iterations are the Rust step's.
        let mut best_server = Server::start(&best_exe, &dir.join(format!("best_server_{case}")), &dir.join("points")).unwrap();
        let mut seed_server = Server::start(&seed_exe, &dir.join(format!("seed_server_{case}")), &dir.join("points")).unwrap();
        best_server.load_points(case, &points).unwrap();
        seed_server.load_points(case, &points).unwrap();
        let (best, compiled_seed) = converge_interleaved(&points, &start, 2, step_of_server(&mut best_server, case), step_of_server(&mut seed_server, case)).unwrap();
        assert_eq!((compiled_seed.iterations, compiled_seed.centroids.clone()), (reference.iterations, reference.centroids.clone()), "{case}: the compiled seed is the reference program bit for bit");
        let compiled_seed = with_further(&points, compiled_seed);
        let judged = |r: ConvergedRun| (with_further(&points, r), reference.clone());
        let runs = vec![(reference.clone(), reference.clone()), judged(identity), judged(worsening), judged(nan), judged(r3), judged(r2), judged(plain), judged(reversed), judged(faster), judged(early), (with_further(&points, best), compiled_seed)];
        for (slot, r) in runs.into_iter().enumerate() {
            per_case[slot].push(r);
        }
        eprintln!("{case}: n {} k {kk}, the reference {} iterations, k_ref {:?}, {:.1} s for the case", points.dims[0], reference.iterations, reference.further_iterations, t0.elapsed().as_secs_f64());
    }
    let _ = std::fs::remove_dir_all(&dir);
    let rows: Vec<Row> = names
        .iter()
        .zip(per_case)
        .map(|(name, runs)| {
            let cases: Vec<(ConvergedRun, ConvergedRun, Inputs)> = runs.into_iter().map(|(c, r)| { let i = kmeans_inputs_better(&c, &r, eps).unwrap(); (c, r, i) }).collect();
            let quality: Vec<f64> = cases.iter().flat_map(|c| c.2.quality.clone()).collect();
            let speed: Vec<f64> = cases.iter().flat_map(|c| c.2.speed.clone()).collect();
            assert!(quality.iter().chain(&speed).all(|x| (0.0..=1.0).contains(x)), "{name}: an input outside [0, 1]");
            let d = quality.len() * m + speed.len();
            let angle = crate::fitness::qqs(&quality, &speed, m).unwrap().fitness;
            Row { name, cases, angle, percentile: hff_percentile(angle, d).unwrap(), d }
        })
        .collect();
    let reference = &rows[0];
    assert_eq!(reference.d, 22);
    assert!((reference.angle - 0.75f64.acos()).abs() < 1e-12, "the reference against itself is 1/2 on every input");
    eprintln!("futhark-kmeans-better at m = {m}, eps = {eps}, over train_n500_k8 and held_n1000_k32 (d = {}; the angle in radians over the one flat vector; F = the HFF percentile I_(sin^2 theta)((d-1)/2, 1/2)):", reference.d);
    eprintln!("| individual | iterations (500/k8, 1000/k32) | further iterations k (k_ref {}, {}) | x inertia | x percentile | x convergence | speed inputs [s, it] | HFF angle (rad) | HFF percentile F | d |", rows[0].cases[0].1.further_iterations.unwrap(), rows[0].cases[1].1.further_iterations.unwrap());
    eprintln!("|---|---|---|---|---|---|---|---:|---:|---:|");
    let pair = |r: &Row, f: &dyn Fn(&(ConvergedRun, ConvergedRun, Inputs)) -> String| r.cases.iter().map(f).collect::<Vec<_>>().join(", ");
    for r in &rows {
        let x = |i: usize| pair(r, &|c| if c.0.finished { format!("{:.5}", better_ratios(&c.0, &c.1)[i]) } else { "-".into() });
        eprintln!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.4} | {:.4e} | {} |",
            r.name,
            pair(r, &|c| if c.0.finished { c.0.iterations.to_string() } else { format!("{} (unfinished)", c.0.iterations) }),
            pair(r, &|c| c.0.further_iterations.map_or("-".into(), |k| k.to_string())),
            x(0),
            x(1),
            x(2),
            pair(r, &|c| format!("[{:.3}, {:.3}]", c.2.speed[0], c.2.speed[1])),
            r.angle,
            r.percentile,
            r.d
        );
    }
    let of = |name: &str| rows.iter().find(|r| r.name.starts_with(name)).unwrap_or_else(|| panic!("no row {name}"));
    // 1. Every individual worse than the reference by more than eps on a final quality measure on any case (or that
    //    did not finish) has a higher HFF.
    let worse_than_eps = |r: &Row| r.cases.iter().any(|(c, rf, _)| !c.finished || better_ratios(c, rf)[..2].iter().any(|x| *x > 1.0 + eps));
    let losers: Vec<&Row> = rows.iter().filter(|r| worse_than_eps(r)).collect();
    eprintln!("worse than the reference by more than eps on a final quality measure on some case: {}", losers.iter().map(|r| r.name).collect::<Vec<_>>().join("; "));
    for name in ["returns the centroids unchanged", "makes them worse", "NaN at iteration 3", "rounded to 1e-2", "one iteration, then unchanged", "the first run's best individual"] {
        assert!(losers.iter().any(|r| r.name.starts_with(name)), "{name} is expected to lose more than eps on some case");
    }
    for r in &losers {
        assert!(r.angle > reference.angle, "{}: worse by more than eps but its HFF {} is not above the reference's {}", r.name, r.angle, reference.angle);
    }
    // 2. The faster-identical individual has a lower HFF.
    let faster = of("faster, identical output");
    assert!(faster.angle < reference.angle && faster.percentile < reference.percentile, "faster-identical {} against {}", faster.angle, reference.angle);
    // 3. The reorderings within the reorder noise: the noise is the largest |x - 1| of their quality inputs (the
    //    convergence x exactly 1: the same iterations and further iterations), below eps; their speed inputs are the
    //    reference's; their angle is within the angle the reference reaches with every quality input at the noise.
    let reorderings = [of("additions reordered: plain"), of("additions reordered: reversed")];
    let noise = reorderings.iter().flat_map(|r| r.cases.iter().flat_map(|(c, rf, _)| better_ratios(c, rf).into_iter().map(|x| (x - 1.0).abs()))).fold(0.0, f64::max);
    let at_noise = {
        let n_quality = reference.cases.len() * QUALITY_NAMES_BETTER.len();
        crate::fitness::qqs(&vec![quality_cliff(1.0 + noise, eps); n_quality], &vec![0.5; reference.cases.len() * SPEED_NAMES.len()], m).unwrap().fitness
    };
    eprintln!("the reorder noise: the largest |x - 1| of the reorderings' quality inputs is {noise:.3e} (eps {eps}); the reference's angle with every quality input at that noise is {at_noise:.6} (the reference {:.6}, the reorderings {:.6} and {:.6})", reference.angle, reorderings[0].angle, reorderings[1].angle);
    assert!(noise < eps, "the reorder noise {noise} is not below the cliff's width {eps}");
    for r in &reorderings {
        for (c, rf, i) in &r.cases {
            assert_eq!((c.iterations, c.further_iterations), (rf.iterations, rf.further_iterations), "{}: a reordering changed the iterations", r.name);
            // The stated cost summed over the same iterations: 1/2 to the rounding of that sum.
            assert!((i.speed[0] - 0.5).abs() < 1e-9 && i.speed[1] == 0.5, "{}: {:?}", r.name, i.speed);
        }
        assert!((r.angle - reference.angle).abs() <= at_noise - reference.angle + 1e-12, "{}: its angle {} is outside the reorder noise of the reference ({} to {at_noise})", r.name, r.angle, reference.angle);
        for l in &losers {
            assert!(r.angle < l.angle, "{} ranks below {}, which lost more than eps", r.name, l.name);
        }
    }
    // 4. The first run's best individual ranks below the reference.
    let best = of("the first run's best individual");
    assert!(best.angle > reference.angle, "the first run's best individual {} ranks above the reference {}", best.angle, reference.angle);
    // Every percentile on [0, 1], increasing with the angle.
    let mut by_angle: Vec<&Row> = rows.iter().collect();
    by_angle.sort_by(|a, b| a.angle.total_cmp(&b.angle));
    assert!(by_angle.windows(2).all(|w| w[0].percentile <= w[1].percentile) && rows.iter().all(|r| (0.0..=1.0).contains(&r.percentile)));
}
