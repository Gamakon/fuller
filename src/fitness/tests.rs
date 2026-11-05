//! The registry's own tests: names unique, defaults of their own family, the old schemes marked, the staged rule.

use super::*;

#[test]
fn the_registry_names_every_scheme_once_and_marks_the_old_ones() {
    let mut names: Vec<&str> = SCHEMES.iter().map(|s| s.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), SCHEMES.len(), "a scheme name is used twice");
    for fam in [Family::FutharkSig, Family::Staged, Family::Shapes, Family::Regex] {
        let d = default_for(fam);
        assert_eq!(d.family, fam);
        assert!(d.fault.is_none(), "{}: a default is never an old scheme", d.name);
        assert_eq!(scheme_for(fam, None).unwrap(), d);
    }
    assert!(scheme_for(Family::Regex, Some("futhark-study7")).unwrap_err().contains("regex-raw"));
    assert!(scheme_for(Family::FutharkSig, Some("no-such")).is_err());
    for s in SCHEMES {
        let fam_ok = matches!(
            (s.family, s.definition),
            (Family::FutharkSig, Definition::FutharkSig { .. } | Definition::FutharkConverged | Definition::FutharkConvergedGated | Definition::FutharkConvergedQqs | Definition::FutharkKmeansBetter) | (Family::Staged, Definition::Staged { .. }) | (Family::Shapes, Definition::ShapesTwoBand { .. } | Definition::ShapesHff { .. }) | (Family::Regex, Definition::Regex { .. })
        );
        assert!(fam_ok, "{}: its definition is not of its family", s.name);
        assert_eq!(s.card_line().contains(OLD_SCHEME_WARNING), s.fault.is_some(), "{}", s.name);
        // NO DEFAULT CARRIES THE OFFSET (Andrew, 2026-11-03): every gated scheme is old, with the offset named as its
        // fault, and every scheme that is not old is flat or never had a stage 2 to gate.
        let gated = matches!(s.definition, Definition::FutharkConvergedGated | Definition::FutharkSig { stage_two: StageTwo::Gated, .. } | Definition::Staged { stage_two: StageTwo::Gated } | Definition::ShapesHff { stage_two: StageTwo::Gated, .. });
        if gated {
            assert!(s.fault.is_some_and(|f| f.contains("pi/2 + theta offset")), "{}: a gated scheme without the offset fault", s.name);
        }
        assert_eq!(s.start().quality_weight.is_some(), s.is_flat(), "{}: the run_start records a weight exactly for a flat scheme", s.name);
        assert_eq!(s.start_weighted(5).quality_weight, s.is_flat().then_some(5), "{}", s.name);
        assert_eq!(s.start_with(5, 0.01).quality_cliff, s.uses_cliff().then_some(0.01), "{}: the run_start records a cliff exactly for a scheme with one", s.name);
    }
    for old in ["futhark-slope4", "futhark-study7", "futhark-kmeans-converged", "staged", "shapes-hff", "shapes-two-band", "regex-colmax", "futhark-slope4-gated", "futhark-kmeans-quality-first", "staged-gated", "shapes-hff-gated", "futhark-kmeans-qqs"] {
        assert!(scheme(old).unwrap().fault.is_some(), "{old} is an old scheme");
    }
    assert!(scheme("futhark-kmeans-qqs").unwrap().fault.is_some_and(|f| f.contains("rewards early stopping")), "the first k-means run's fault");
    for (flat, m) in [("futhark-kmeans-better", KMEANS_BETTER_QUALITY_WEIGHT), ("futhark-slope4-qqs", ANSWER_QUALITY_WEIGHT), ("staged-qqs", DEFAULT_QUALITY_WEIGHT), ("shapes-hff-qqs", DEFAULT_QUALITY_WEIGHT)] {
        let s = scheme(flat).unwrap();
        assert!(s.is_flat() && s.fault.is_none(), "{flat}");
        assert_eq!(s.default_quality_weight(), m, "{flat}");
    }
    assert!(scheme("futhark-kmeans-qqs").unwrap().is_converged() && scheme("futhark-kmeans-better").unwrap().is_converged());
    assert!(scheme("futhark-kmeans-better").unwrap().uses_cliff() && SCHEMES.iter().filter(|s| s.uses_cliff()).count() == 1);
    assert_eq!(crate::futhark_sig::KMEANS_PROBLEM.fitness, "futhark-kmeans-better", "k-means's default is its newest scheme");
    assert_eq!(scheme("regex-raw").unwrap().default_quality_weight(), 1);
}

/// THE FLAT FITNESS (Andrew, 2026-11-03): the angle of the one vector with the quality inputs entered m times, nothing
/// after it. The seed (1/2 everywhere) scores acos(3/4) at every m; m is the exchange rate between a quality loss and
/// a speed gain, exactly as the mean square says; the group angles are reported beside it; the inputs are checked.
#[test]
fn the_flat_fitness_is_truenorth_over_the_repeated_quality_inputs_and_nothing_else() {
    let seed_angle = (1.0f64 - 0.25).acos();
    for m in 1..=4 {
        let t = qqs(&[0.5, 0.5], &[0.5, 0.5], m).unwrap();
        assert!((t.fitness - seed_angle).abs() < 1e-12, "m = {m}: {}", t.fitness);
        assert!((t.quality - seed_angle).abs() < 1e-12 && (t.speed - seed_angle).abs() < 1e-12);
        assert_eq!(flat_vector(&[0.1, 0.2], &[0.3], m).len(), 2 * m + 1);
    }
    assert_eq!(flat_vector(&[0.1, 0.2], &[0.3], 2), vec![0.1, 0.2, 0.1, 0.2, 0.3]);
    // A candidate 10 % worse in quality (0.55) and twice as fast (1/3): at m = 1 it beats the seed, at m = 3 it does
    // not: m * (sum q^2 - Q/4) against S/4 - sum s^2 with Q = S = 2.
    let worse_faster = |m| qqs(&[0.55, 0.55], &[1.0 / 3.0, 1.0 / 3.0], m).unwrap().fitness;
    let gain = 0.5 - 2.0 / 9.0;
    let loss = 2.0 * 0.55f64 * 0.55 - 0.5;
    assert!(gain > loss && gain < 3.0 * loss, "the hand-made numbers do not straddle m = 1 and m = 3: {gain} {loss}");
    assert!(worse_faster(1) < seed_angle && worse_faster(3) > seed_angle, "{} {}", worse_faster(1), worse_faster(3));
    // The angle is exactly the fitness: no offset, so a worse program sits below pi/2 + anything the gate added.
    let t = qqs(&[0.9, 0.9], &[0.5, 0.5], 2).unwrap();
    assert!(t.fitness < std::f64::consts::FRAC_PI_2 && t.fitness > seed_angle);
    assert_eq!(t.fitness, crate::score::truenorth_angle(&[0.9, 0.9, 0.9, 0.9, 0.5, 0.5]));
    // The checks: an empty group, a weight of 0, an input outside [0, 1], a NaN.
    assert!(qqs(&[], &[0.5], 2).is_err() && qqs(&[0.5], &[], 2).is_err() && qqs(&[0.5], &[0.5], 0).is_err());
    assert!(qqs(&[1.5], &[0.5], 2).is_err() && qqs(&[0.5], &[f64::NAN], 2).is_err());
    // The scheme's methods: converged_fitness with the weight for the k-means scheme; staged at the default weight and
    // weighted for the answer-scored scheme; stage_two's flat rule is the default weight.
    let km = scheme("futhark-kmeans-qqs").unwrap();
    assert_eq!(km.converged_fitness(&[0.55, 0.55], &[1.0 / 3.0, 1.0 / 3.0], 3).unwrap().fitness, worse_faster(3));
    let s4 = scheme("futhark-slope4-qqs").unwrap();
    assert_eq!(s4.staged(FitnessStage::QualityAndSpeed, &[0.2, 0.0], &[0.4]).unwrap(), s4.staged_weighted(FitnessStage::QualityAndSpeed, &[0.2, 0.0], &[0.4], ANSWER_QUALITY_WEIGHT).unwrap());
    assert_eq!(s4.staged_weighted(FitnessStage::QualityAndSpeed, &[0.2, 0.0], &[0.4], 3).unwrap(), qqs(&[0.2, 0.0], &[0.4], 3).unwrap());
    assert_eq!(s4.staged(FitnessStage::Quality, &[0.2, 0.0], &[]).unwrap().fitness, crate::score::truenorth_angle(&[0.2, 0.0]), "stage 1 is quality alone");
    assert_eq!(stage_two(StageTwo::Flat, &[0.2, 0.0], &[0.4]).fitness, qqs(&[0.2, 0.0], &[0.4], DEFAULT_QUALITY_WEIGHT).unwrap().fitness);
    assert_eq!(stage_two(StageTwo::Flat, &[], &[]).fitness, 0.0);
    assert!(km.staged(FitnessStage::Quality, &[0.5], &[]).is_err(), "the k-means scheme has no stage 2 rule");
    assert!(scheme("futhark-slope4-gated").unwrap().converged_fitness(&[0.5], &[0.5], 2).is_err());
}

#[test]
fn the_stage_two_rules_differ_only_on_an_inexact_program() {
    let (exact, inexact, speed) = ([0.0, 0.0], [0.2, 0.0], [0.4, 0.1]);
    let g = |q: &[f64]| staged(StageTwo::Gated, FitnessStage::QualityAndSpeed, q, &speed).unwrap();
    let u = |q: &[f64]| staged(StageTwo::Ungated, FitnessStage::QualityAndSpeed, q, &speed).unwrap();
    assert_eq!(g(&exact), u(&exact));
    assert_eq!(u(&inexact), two_level_hff(&inexact, &speed));
    assert_eq!(g(&inexact).fitness, std::f64::consts::FRAC_PI_2 + u(&inexact).quality);
    assert_eq!(crate::runner::staged_hff(FitnessStage::QualityAndSpeed, &inexact, &speed).unwrap(), g(&inexact), "staged_hff is staged-gated");
    assert!(staged(StageTwo::Gated, FitnessStage::Quality, &inexact, &[0.1]).is_err(), "stage 1 takes no speed inputs");
}

/// THE CLIFF AND THE PERCENTILE of `futhark-kmeans-better` (Andrew's specification of 2026-11-03): `q = x/2` up to
/// `x = 1` (equal to the reference program, 1/2), then a straight rise to 1 at `x = 1 + ε` and 1 beyond; a NaN is 1;
/// a negative `x` is 0. The percentile is `I_{sin^2 theta}((d - 1)/2, 1/2)`: 0 at 0, 1 at pi/2, 1/2 at 1/2 for d = 2
/// (`I_x(1/2, 1/2) = (2/pi) asin(sqrt x)`, so F(theta) = 2 theta / pi), increasing in the angle, smaller at a larger d
/// for the same angle, and an error outside [0, pi/2] or below 2 inputs.
#[test]
fn the_quality_cliff_and_the_hff_percentile_are_as_specified() {
    let eps = DEFAULT_QUALITY_CLIFF;
    assert_eq!(eps, 0.005);
    assert_eq!(quality_cliff(1.0, eps), 0.5);
    assert_eq!(quality_cliff(0.5, eps), 0.25);
    assert_eq!(quality_cliff(0.0, eps), 0.0);
    assert_eq!(quality_cliff(-3.0, eps), 0.0);
    assert!((quality_cliff(1.0025, eps) - 0.75).abs() < 1e-12, "{}", quality_cliff(1.0025, eps));
    // x = 1 + eps in f64 is (x - 1)/eps = 0.99999999999998: the rise ends there, and 1 is reached past it.
    assert!((quality_cliff(1.005, eps) - 1.0).abs() < 1e-12, "{}", quality_cliff(1.005, eps));
    assert_eq!(quality_cliff(1.0051, eps), 1.0);
    assert_eq!(quality_cliff(1.16, eps), 1.0, "the first run's best individual's 16 % loss is the worst input");
    assert_eq!(quality_cliff(f64::INFINITY, eps), 1.0);
    assert_eq!(quality_cliff(f64::NAN, eps), 1.0);
    // Monotone and continuous at 1.
    let mut prev = -1.0;
    for i in 0..=400 {
        let x = 0.9 + 0.0005 * f64::from(i);
        let q = quality_cliff(x, eps);
        assert!(q >= prev && (0.0..=1.0).contains(&q), "x {x}: {q} after {prev}");
        prev = q;
    }
    assert!((quality_cliff(1.0 + 1e-12, eps) - 0.5).abs() < 1e-9);
    let half_pi = std::f64::consts::FRAC_PI_2;
    for theta in [0.1, 0.4, 0.7227, 1.2] {
        assert!((hff_percentile(theta, 2).unwrap() - theta / half_pi).abs() < 1e-9, "d = 2 at {theta}");
    }
    assert_eq!(hff_percentile(0.0, 22).unwrap(), 0.0);
    assert!((hff_percentile(half_pi, 22).unwrap() - 1.0).abs() < 1e-12);
    let mut prev = 0.0;
    for i in 1..=50 {
        let p = hff_percentile(half_pi * f64::from(i) / 50.0, 66).unwrap();
        assert!(p >= prev, "the percentile falls at step {i}");
        prev = p;
    }
    assert!(hff_percentile(0.7227, 66).unwrap() < hff_percentile(0.7227, 22).unwrap(), "the same angle is rarer in more dimensions");
    assert!(hff_percentile(0.5, 1).is_err() && hff_percentile(-0.1, 22).is_err() && hff_percentile(1.6, 22).is_err() && hff_percentile(f64::NAN, 22).is_err());
    let better = scheme("futhark-kmeans-better").unwrap();
    assert_eq!(better.default_quality_weight(), 3);
    assert_eq!(better.converged_fitness(&[0.5, 0.5, 0.5], &[0.5, 0.5], 3).unwrap().fitness, crate::score::truenorth_angle(&[0.5; 11]));
}

/// THE A7 F32 BOUND (`futhark-a7-f32bound-qqs`, F6 2026-11-04): A7's default on the GPU and not on futhark c; for A7
/// only. On every fitness case the GPU's published output, whose largest cell error the F6 log records (the third raw
/// measure, `e / (1 + e)` of the largest absolute error over the answer's largest magnitude), is within the bound, so
/// an output off by that error at every cell is the answer under the bound and not under the bits; a wrong program
/// (one product where A7 adds two) is wrong under the bound, and so is one cell off by more than the bound.
#[test]
fn the_a7_f32_bound_accepts_the_gpus_published_output_and_rejects_a_wrong_program() {
    use crate::futhark_sig::{build6, compare, kingdom6, load_cases, problem, signature, Interp};
    let a7 = problem("a7").unwrap();
    let s = scheme_for_problem_on(&a7, None, true).unwrap();
    assert_eq!((s.name, s.fault, s.is_flat(), s.default_quality_weight()), (A7_F32_BOUND, None, true, ANSWER_QUALITY_WEIGHT));
    assert_eq!(s.exactness(), Exactness::F32SumBound { products: 2 });
    assert!(s.description.contains("A7's DEFAULT ON THE GPU"));
    assert_eq!(scheme_for_problem_on(&a7, None, false).unwrap().name, "futhark-slope4-qqs", "futhark c keeps A7's own default");
    assert_eq!(scheme_for_problem_on(&a7, Some("futhark-slope4-qqs"), true).unwrap().name, "futhark-slope4-qqs", "a named scheme is kept on the GPU");
    let histogram = problem("histogram").unwrap();
    assert_eq!(scheme_for_problem_on(&histogram, None, true).unwrap().name, histogram.fitness);
    assert!(scheme_for_problem(&histogram, Some(A7_F32_BOUND)).unwrap_err().contains("is not A7"));
    assert!(SCHEMES.iter().filter(|x| x.name != A7_F32_BOUND).all(|x| x.exactness() == Exactness::Bits), "every other scheme computes what it did");
    // The bound itself: 2 gamma_n n B^2.
    let u = F32_UNIT_ROUNDOFF;
    assert_eq!(f32_sum_bound(64, 64.0), 2.0 * (64.0 * u / (1.0 - 64.0 * u)) * 64.0 * 64.0);
    assert_eq!((f32_sum_bound(0, 64.0), f32_sum_bound(1 << 24, 1.0)), (0.0, 0.0));
    // The F6 log's raw measures of the compiled published program on the GPU, in fitness-case order.
    const F6_GPU_RAW: [[f64; 3]; 5] = [
        [0.515625, 4.202031436407259e-8, 8.201833339402689e-8],
        [0.640625, 6.661820104110497e-8, 2.1048699534164214e-7],
        [0.744140625, 1.0334259968714284e-7, 2.1409319862303817e-7],
        [0.5714285714285714, 4.3589009701055455e-8, 7.427139942839245e-8],
        [0.0, 0.0, 0.0],
    ];
    let ex = s.exactness();
    let cases = load_cases(&a7).unwrap();
    let fit: Vec<_> = cases.iter().filter(|c| c.split.in_fitness()).collect();
    assert_eq!(fit.len(), F6_GPU_RAW.len());
    for (c, raw) in fit.iter().zip(F6_GPU_RAW) {
        let want = c.answer().unwrap();
        assert_eq!(raw[0] * want.len() as f64, (raw[0] * want.len() as f64).round(), "{}: the F6 share is of this case's cells", c.name);
        let max_mag = want.values().map(f64::abs).fold(1.0, f64::max);
        let largest = raw[2] / (1.0 - raw[2]) * max_mag;
        let tau = ex.tolerance(&c.inputs);
        assert!(largest <= tau, "{}: the GPU's largest error {largest:e} over the bound {tau:e}", c.name);
        eprintln!("{}: the bound {tau:.3e}, the GPU's largest cell error {largest:.3e} ({:.0}x below)", c.name, if largest > 0.0 { tau / largest } else { f64::INFINITY });
        let mut got = want.clone();
        for i in 0..got.len() {
            let d = if i % 2 == 0 { largest } else { -largest };
            got.data[i] = ((want.value(i) + d) as f32).to_bits();
        }
        assert_eq!(compare(&ex.as_judged(&got, want, &c.inputs), want), [0.0; 3], "{}: exact under the bound", c.name);
        assert!(ex.agrees(&got, want, &c.inputs));
        assert_eq!(Exactness::Bits.agrees(&got, want, &c.inputs), largest == 0.0, "{}: under the bits", c.name);
        // One cell off by twice the bound is wrong in that cell alone.
        let mut one = want.clone();
        one.data[0] = ((want.value(0) + 2.0 * tau.max(1e-3)) as f32).to_bits();
        let r = compare(&ex.as_judged(&one, want, &c.inputs), want);
        assert_eq!(r[0], 1.0 / want.len() as f64, "{}: one cell over the bound", c.name);
    }
    // A wrong program: the one product, where A7 adds two.
    let k = std::sync::Arc::new(kingdom6(a7.name).unwrap());
    let interp = Interp::new6(&k, &signature(a7.name).unwrap()).unwrap();
    let wrong = build6(&k, "(matmul@m.f32 m_n_m m_m_p)").unwrap();
    for c in &fit {
        let want = c.answer().unwrap();
        let got = interp.eval(&wrong, &c.inputs).unwrap();
        let r = compare(&ex.as_judged(&got, want, &c.inputs), want);
        assert!(!ex.agrees(&got, want, &c.inputs) && r[0] > 0.9, "{}: the one product is wrong under the bound: {r:?}", c.name);
    }
}
