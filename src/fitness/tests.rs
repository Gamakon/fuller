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
            (Family::FutharkSig, Definition::FutharkSig { .. } | Definition::FutharkConverged | Definition::FutharkConvergedGated | Definition::FutharkConvergedQqs) | (Family::Staged, Definition::Staged { .. }) | (Family::Shapes, Definition::ShapesTwoBand { .. } | Definition::ShapesHff { .. }) | (Family::Regex, Definition::Regex { .. })
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
    }
    for old in ["futhark-slope4", "futhark-study7", "futhark-kmeans-converged", "staged", "shapes-hff", "shapes-two-band", "regex-colmax", "futhark-slope4-gated", "futhark-kmeans-quality-first", "staged-gated", "shapes-hff-gated"] {
        assert!(scheme(old).unwrap().fault.is_some(), "{old} is an old scheme");
    }
    for (flat, m) in [("futhark-kmeans-qqs", DEFAULT_QUALITY_WEIGHT), ("futhark-slope4-qqs", ANSWER_QUALITY_WEIGHT), ("staged-qqs", DEFAULT_QUALITY_WEIGHT), ("shapes-hff-qqs", DEFAULT_QUALITY_WEIGHT)] {
        let s = scheme(flat).unwrap();
        assert!(s.is_flat() && s.fault.is_none(), "{flat}");
        assert_eq!(s.default_quality_weight(), m, "{flat}");
    }
    assert!(scheme("futhark-kmeans-qqs").unwrap().is_converged());
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
