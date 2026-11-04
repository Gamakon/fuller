//! The evaluator's tests (host only: the c backend, `futhark` 0.28.1 and `cc`; no GPU). A configuration the judge
//! cannot score is an error that names it; histogram's evaluator passes its controls on the c backend, names one
//! column a fitness input, scores the reference program exact (HFF 0) and charges a program scored before nothing;
//! the genetics express the reference program from its gene and marry two programs into legal offspring.

use super::*;
use std::process::Command;

fn have(p: &str) -> bool {
    Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_evaluator_{name}_{}", std::process::id()))
}

fn config(problem: &str, out: PathBuf) -> EvaluatorConfig {
    EvaluatorConfig {
        problem: problem.into(),
        fitness: None,
        quality_weight: None,
        quality_cliff: None,
        backend: Backend::C,
        eval: EvalMode::Single,
        pairs: 11,
        runs: 20,
        program_size: None,
        out,
        kmeans_step_process: false,
        gate: None,
        rings: None,
        ring_walk: false,
    }
}

fn error_of(cfg: &EvaluatorConfig) -> String {
    match FutharkEvaluator::new(cfg) {
        Ok(_) => panic!("the configuration was accepted"),
        Err(e) => e,
    }
}

#[test]
fn a_configuration_the_judge_cannot_score_is_an_error_that_names_it() {
    let out = scratch("config");
    let mut c = config("histogram", out.clone());
    c.pairs = 0;
    assert!(error_of(&c).contains("--pairs 0"), "no timing slot");
    let mut c = config("histogram", out.clone());
    c.eval = EvalMode::Rust;
    assert!(error_of(&c).contains("--eval rust runs on the GPU"), "the rust path on the c backend");
    let mut c = config("histogram", out.clone());
    c.quality_cliff = Some(0.01);
    assert!(error_of(&c).contains("has no quality cliff"), "a cliff for an answer-scored scheme");
    let mut c = config("histogram", out.clone());
    c.quality_weight = Some(0);
    assert!(error_of(&c).contains("--quality-weight 0"), "a flat scheme with no quality input");
    let mut c = config("histogram", out.clone());
    c.ring_walk = true;
    assert!(error_of(&c).contains("ring-walk"), "a ring walk with no rings");
    let mut c = config("histogram", out.clone());
    c.kmeans_step_process = true;
    assert!(error_of(&c).contains("--kmeans-step process"), "the k-means step for an answer-scored scheme");
    let mut c = config("kmeans", out.clone());
    c.eval = EvalMode::Batch;
    assert!(error_of(&c).contains("runs UltraDim's loop"), "k-means on the batch driver");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn histogram_on_the_c_backend_passes_its_controls_and_scores_the_reference_program_exact() {
    assert!(have("futhark") && have("cc"), "futhark 0.28.1 and cc are needed for this test");
    let out = scratch("histogram");
    let mut ev = FutharkEvaluator::new(&config("histogram", out.clone())).unwrap();
    assert_eq!(ev.pairs(), 11);
    assert_eq!(ev.quality_weight(), ev.scheme().default_quality_weight(), "the scheme's own quality weight");
    let c = ev.controls(&ControlRequest { regime: Some(Regime::R), aa_rounds: 0 }).unwrap();
    // The c backend times nothing: no A/A set, no note.
    assert!(c.noise.is_empty() && c.aa.is_none() && c.inrun_aa.is_empty() && c.notes.is_empty(), "the c backend has no A/A sets");
    // The library prints nothing: with no sink set, its messages are held for the caller, in order, each stamped.
    let held = ev.take_messages();
    for want in ["timing bound:", "control: the reference gene is exact on all", "control: the compiled reference gene is exact"] {
        assert!(held.iter().any(|l| l.contains(want)), "{want}: {held:?}");
    }
    assert!(held.iter().position(|l| l.contains("timing bound:")) < held.iter().position(|l| l.contains("control: the reference gene is exact on all")), "in order: {held:?}");
    assert!(ev.take_messages().is_empty(), "taken once");
    let (reference, source) = (ev.reference().clone(), ev.reference_source().to_string());
    ev.judge.prefetch(&[(&reference, source.as_str())], FitnessStage::Quality).unwrap();
    let e = ev.judge.evaluate(&reference, &source, FitnessStage::Quality).unwrap();
    assert_eq!(e.fitness, 0.0, "the reference program is exact: HFF 0");
    assert!(e.quality.perfect());
    let names = ev.input_names();
    assert_eq!(names.len(), e.quality.inputs.len() + ev.pairs(), "one name a quality input, then one a timing slot");
    assert!(names[..e.quality.inputs.len()].iter().all(|n| n.starts_with("quality ")) && names[e.quality.inputs.len()..].iter().all(|n| n.starts_with("speed pair ")));
    assert_eq!((ev.judge.evaluations, ev.judge.hits), (1, 0), "one budgeted evaluation");
    let again = ev.judge.evaluate(&reference, &source, FitnessStage::Quality).unwrap();
    assert_eq!(again.fitness, 0.0);
    assert_eq!((ev.judge.evaluations, ev.judge.hits, ev.judge.scored), (1, 1, 2), "a program scored before is a cache hit and costs nothing");
    let _ = std::fs::remove_dir_all(&out);
}

/// A sink set after messages were held receives the held lines first, in order, then every later one; nothing is
/// held once it is set.
#[test]
fn a_sink_receives_the_held_messages_first_then_every_later_one() {
    let mut m = Messages::default();
    m.say("one");
    m.say("two");
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::<String>::new()));
    let s2 = std::rc::Rc::clone(&seen);
    m.set_sink(Box::new(move |l| s2.borrow_mut().push(l.to_string())));
    m.say("three");
    assert!(m.take().is_empty(), "nothing is held once a sink is set");
    let got: Vec<String> = seen.borrow().iter().map(|l| l.split_once(' ').expect("the UTC time, then the message").1.to_string()).collect();
    assert_eq!(got, ["one", "two", "three"]);
    assert!(seen.borrow().iter().all(|l| l.len() > 9 && l.as_bytes()[8] == b'Z'), "each line starts with HH:MM:SSZ: {:?}", seen.borrow());
}


#[test]
fn the_genetics_express_the_reference_program_and_marry_two_programs_into_legal_offspring() {
    assert!(have("futhark") && have("cc"), "futhark 0.28.1 and cc are needed for this test (the timing input's output)");
    let out = scratch("genetics");
    let ev = FutharkEvaluator::new(&config("pathfinder", out.clone())).unwrap();
    let g = &ev.gen;
    let mut rng = GeneRng::new(7);
    let ref_gene = write_back(&g.contract, &g.layout, &g.random_gene(&mut rng).unwrap(), ev.reference()).unwrap();
    assert_eq!(g.tree(&ref_gene).unwrap(), *ev.reference(), "the reference gene expresses the reference program");
    let planted_gene = write_back(&g.contract, &g.layout, &g.random_gene(&mut rng).unwrap(), ev.planted()).unwrap();
    let (a, b) = (g.tree(&ref_gene).unwrap(), g.tree(&planted_gene).unwrap());
    let (ga, gb) = g.marry((&a, &ref_gene), (&b, &planted_gene), &mut rng).unwrap().expect("the reference and the planted slowdown share an output type below their roots");
    for child in [&ga, &gb] {
        assert_eq!(child.len(), ref_gene.len());
        g.tree(child).unwrap();
    }
    assert!(g.neighbours(&ref_gene).unwrap().iter().all(|(_, t)| t != ev.reference()), "a neighbour is never the gene's own tree");
    let _ = std::fs::remove_dir_all(&out);
}
