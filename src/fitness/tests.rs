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
            (Family::FutharkSig, Definition::FutharkSig { .. } | Definition::FutharkConverged) | (Family::Staged, Definition::Staged { .. }) | (Family::Shapes, Definition::ShapesTwoBand { .. } | Definition::ShapesHff { .. }) | (Family::Regex, Definition::Regex { .. })
        );
        assert!(fam_ok, "{}: its definition is not of its family", s.name);
        assert_eq!(s.card_line().contains(OLD_SCHEME_WARNING), s.fault.is_some(), "{}", s.name);
    }
    for old in ["futhark-slope4", "futhark-study7", "staged", "shapes-hff", "shapes-two-band", "regex-colmax"] {
        assert!(scheme(old).unwrap().fault.is_some(), "{old} is an old scheme");
    }
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
