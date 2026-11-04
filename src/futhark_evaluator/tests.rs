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
    // The controls run once a run: a second call is an error that says so, and changes nothing.
    let again = ev.controls(&ControlRequest { regime: Some(Regime::R), aa_rounds: 0 }).expect_err("a second call of the controls");
    assert!(again.contains("run once a run"), "{again}");
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

// ---- THE SEAM (task F3a): `evaluate_generation` against the deprecated runner's ledger ----

/// A row of the deprecated runner's ledger (`histogram_r_seed1_ledger.tsv`): its generation, operator, fitness, text
/// and the ledger columns eval, batch_size, quality_inputs, speed_inputs, behaviour, all_zero.
struct LedgerRow {
    generation: u32,
    operator: String,
    fitness: f64,
    text: String,
    cols: Vec<String>,
}

fn runner_ledger() -> Vec<LedgerRow> {
    let mut lines = include_str!("histogram_r_seed1_ledger.tsv").lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("a header").split('\t').collect();
    assert_eq!(header, ["generation", "id", "parents", "operator", "fitness", "text", "eval", "batch_size", "quality_inputs", "speed_inputs", "behaviour", "all_zero"]);
    lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            LedgerRow { generation: f[0].parse().unwrap(), operator: f[3].into(), fitness: f[4].parse().unwrap(), text: f[5].into(), cols: f[6..].iter().map(|s| s.to_string()).collect() }
        })
        .collect()
}

/// The ledger's columns of a row the seam scored (eval, batch_size, quality_inputs, speed_inputs, behaviour, all_zero).
fn seam_cols(r: &RowScore) -> Vec<String> {
    assert_eq!(r.ledger.len(), LEDGER_STAGE_COLUMNS.len());
    [0, 1, 13, 14, 15, 16].iter().map(|i| r.ledger[*i].clone()).collect()
}

/// The runner's generation 0 of regime R at random seed 1: 24 draws of `Genetics::random_gene` from
/// `GeneRng::new(1)` (`futhark_evolve`, the stream of regime R is the seed itself). The same genes, checked against
/// the ledger's texts.
fn generation_zero(ev: &FutharkEvaluator, ledger: &[LedgerRow]) -> Vec<Vec<u32>> {
    let mut rng = GeneRng::new(1);
    let genes: Vec<Vec<u32>> = (0..24).map(|_| ev.gen.random_gene(&mut rng).unwrap()).collect();
    for (g, row) in genes.iter().zip(ledger.iter().filter(|r| r.generation == 0)) {
        assert_eq!(sexp(&ev.gen.k, &ev.gen.tree(g).unwrap()), row.text, "the runner's generation-0 gene");
    }
    genes
}

/// A gene expressing a ledger text: the text built into a tree and written onto a random gene.
fn gene_of(ev: &FutharkEvaluator, text: &str, rng: &mut GeneRng) -> Vec<u32> {
    let t = build(&ev.gen.k, text).unwrap();
    let g = write_back(&ev.gen.contract, &ev.gen.layout, &ev.gen.random_gene(rng).unwrap(), &t).unwrap();
    assert_eq!(sexp(&ev.gen.k, &ev.gen.tree(&g).unwrap()), text);
    g
}

fn evaluator_with_run_file(name: &str, eval: EvalMode) -> (FutharkEvaluator, PathBuf) {
    assert!(have("futhark") && have("cc"), "futhark 0.28.1 and cc are needed for this test");
    let out = scratch(name);
    let mut c = config("histogram", out.clone());
    c.eval = eval;
    let ev = FutharkEvaluator::new(&c).unwrap();
    crate::run_file::RunFile::create(&out.join(crate::run_file::RUN_FILE)).unwrap();
    (ev, out)
}

fn refs(genes: &[Vec<u32>]) -> Vec<&[u32]> {
    genes.iter().map(Vec::as_slice).collect()
}

/// A stage-1 row's vector is its quality group, and its angle is TrueNorth over that vector with its percentile.
fn assert_stage_one_vector(r: &RowScore) {
    assert_eq!(r.inputs, r.quality);
    assert_eq!(r.fitness.to_bits(), crate::score::truenorth_angle(&r.inputs).to_bits(), "the angle is TrueNorth over the row's vector");
    assert_eq!((r.d, r.percentile), (r.inputs.len(), Some(crate::fitness::hff_percentile(r.fitness, r.inputs.len()).unwrap())));
}

/// THE PLUMBING THROUGH THE SEAM (`--eval single`, the ledger's own path): the runner's generation 0 (the same 24
/// genes) and two repeats in one call give the runner's fitness, quality inputs, behaviour and ledger columns bit for
/// bit; the repeats are free; then the runner's generation 1 (its 8 programs, written onto genes) gives its 8 rows
/// and the runner's evaluation count (24, then 30).
#[test]
fn the_seam_gives_the_runner_s_ledger_bit_for_bit_and_charges_a_repeat_nothing() {
    let ledger = runner_ledger();
    let (mut ev, out) = evaluator_with_run_file("seam_single", EvalMode::Single);
    ev.controls(&ControlRequest { regime: Some(Regime::R), aa_rounds: 0 }).unwrap();
    let mut genes = generation_zero(&ev, &ledger);
    genes.push(genes[0].clone());
    genes.push(genes[5].clone());
    let s = ev.evaluate_generation(&refs(&genes), 0).unwrap();
    assert!(!s.stopped && !s.stage_changed && s.stage == FitnessStage::Quality);
    assert_eq!(s.rows.len(), 26);
    let gen0: Vec<&LedgerRow> = ledger.iter().filter(|r| r.generation == 0).collect();
    for (r, want) in s.rows.iter().zip(&gen0) {
        assert_eq!((r.text.as_str(), r.fitness.to_bits()), (want.text.as_str(), want.fitness.to_bits()), "the runner's angle");
        assert_eq!(seam_cols(r), want.cols, "the runner's ledger columns: {}", r.text);
        assert!(r.new_evaluation);
        assert_stage_one_vector(r);
    }
    for (rep, of) in [(24, 0), (25, 5)] {
        let (r, o) = (&s.rows[rep], &s.rows[of]);
        assert_eq!((r.fitness.to_bits(), &r.inputs, &r.behaviour), (o.fitness.to_bits(), &o.inputs, &o.behaviour), "a repeat scores as its first row");
        assert!(!r.new_evaluation && r.ledger[0] == "hit" && r.ledger[1] == "0", "a repeat is a cache hit");
    }
    assert_eq!((s.evaluations_new, s.evaluations_total), (24, 24), "24 new programs; the 2 repeats are free (the runner's series: 24)");
    assert_eq!((s.cost.modules, s.cost.failed_compiles, s.cost.halvings), (24, 0, 0), "one module a new program on --eval single");

    let mut rng = GeneRng::new(99);
    let gen1: Vec<&LedgerRow> = ledger.iter().filter(|r| r.generation == 1).collect();
    assert_eq!(gen1.len(), 8);
    let genes1: Vec<Vec<u32>> = gen1.iter().map(|r| gene_of(&ev, &r.text, &mut rng)).collect();
    let s1 = ev.evaluate_generation(&refs(&genes1), 1).unwrap();
    for (r, want) in s1.rows.iter().zip(&gen1) {
        assert_eq!((r.text.as_str(), r.fitness.to_bits()), (want.text.as_str(), want.fitness.to_bits()), "the runner's angle ({})", want.operator);
        assert_eq!(seam_cols(r), want.cols, "the runner's ledger columns: {}", r.text);
        assert_eq!(r.new_evaluation, want.cols[0] != "hit", "new exactly where the runner compiled");
    }
    assert_eq!((s1.evaluations_new, s1.evaluations_total), (6, 30), "the runner's series: 30 evaluations after generation 1");
    let _ = std::fs::remove_dir_all(&out);
}

/// ONE MODULE A CALL (`--eval batch`, the c backend's batch driver): the runner's generation 0 in one compiled
/// module, the runner's angles, quality inputs and behaviours bit for bit; the same call again compiles nothing and
/// charges nothing, with the same angles.
#[test]
fn a_generation_is_one_module_and_a_cached_generation_is_free() {
    let ledger = runner_ledger();
    let (mut ev, out) = evaluator_with_run_file("seam_batch", EvalMode::Batch);
    let genes = generation_zero(&ev, &ledger);
    let s = ev.evaluate_generation(&refs(&genes), 0).unwrap();
    assert_eq!((s.cost.modules, s.cost.failed_compiles, s.cost.halvings), (1, 0, 0), "one compiled module for the call");
    assert_eq!((s.evaluations_new, s.evaluations_total), (24, 24));
    for (r, want) in s.rows.iter().zip(ledger.iter().filter(|r| r.generation == 0)) {
        assert_eq!(r.fitness.to_bits(), want.fitness.to_bits(), "the runner's angle: {}", r.text);
        let c = seam_cols(r);
        assert_eq!((c[0].as_str(), c[1].as_str()), ("batch", "24"), "scored from the call's batch of 24");
        assert_eq!(c[2..], want.cols[2..], "the runner's inputs and behaviour: {}", r.text);
    }
    let again = ev.evaluate_generation(&refs(&genes), 1).unwrap();
    assert_eq!((again.cost.modules, again.evaluations_new, again.evaluations_total), (0, 0, 24), "a cached generation compiles nothing and costs nothing");
    assert!(again.rows.iter().zip(&s.rows).all(|(a, b)| a.fitness.to_bits() == b.fitness.to_bits() && !a.new_evaluation));
    let _ = std::fs::remove_dir_all(&out);
}

/// THE STAGE CHANGE INSIDE THE CALL: the reference program (exact) among five of the runner's random genes, and
/// once more. The c backend has no GPU timing, so every program's timing slots are filled before the call (the state
/// the in-run A/A and a restore leave), and stage 2 then times nothing. The best of stage 1 is exact, so every row is
/// scored again in stage 2 within the call, once: its vector the quality group 8 times and the 11 speed inputs, its
/// angle `futhark-slope4-qqs`'s of them; six programs are charged, once each. The next call stays in stage 2.
#[test]
fn the_stage_change_scores_every_row_again_in_stage_two_within_the_call() {
    let ledger = runner_ledger();
    let (mut ev, out) = evaluator_with_run_file("seam_stage", EvalMode::Batch);
    let random = generation_zero(&ev, &ledger);
    let mut rng = GeneRng::new(5);
    let reference = write_back(&ev.gen.contract, &ev.gen.layout, &ev.gen.random_gene(&mut rng).unwrap(), ev.reference()).unwrap();
    let mut genes = vec![random[0].clone(), reference.clone()];
    genes.extend(random[1..5].iter().cloned());
    genes.push(reference);
    let mut ratios: HashMap<String, Vec<f64>> = HashMap::new();
    for (j, g) in genes.iter().enumerate() {
        let src = program(&ev.gen.k, &ev.judge.sig, &ev.gen.tree(g).unwrap()).unwrap();
        if let std::collections::hash_map::Entry::Vacant(v) = ratios.entry(src) {
            let r: Vec<f64> = (0..ev.pairs()).map(|p| 0.9 + 0.01 * j as f64 + 0.001 * p as f64).collect();
            ev.judge.slots.add(v.key(), &r).unwrap();
            v.insert(r);
        }
    }
    assert_eq!(ratios.len(), 6, "six distinct programs");
    let scored = ev.judge.scored;
    let s = ev.evaluate_generation(&refs(&genes), 0).unwrap();
    assert!(s.stage_changed && s.stage == FitnessStage::QualityAndSpeed && ev.stage() == FitnessStage::QualityAndSpeed && !s.stopped);
    assert_eq!(ev.judge.scored - scored, 2 * genes.len() as u64, "every row scored once in each stage");
    assert_eq!((s.evaluations_new, s.evaluations_total), (6, 6), "six programs charged once each; the stage-2 re-score is free");
    let qw = ev.quality_weight();
    assert_eq!(qw, 8);
    let gen0: Vec<&LedgerRow> = ledger.iter().filter(|r| r.generation == 0).collect();
    for (i, r) in s.rows.iter().enumerate() {
        let speed: Vec<f64> = ratios[&r.source].iter().map(|x| x / (1.0 + x)).collect();
        assert_eq!(r.speed, speed, "the speed inputs are the timing slots");
        assert_eq!(r.inputs, crate::fitness::flat_vector(&r.quality, &speed, qw));
        assert_eq!(r.d, r.quality.len() * qw + ev.pairs());
        let want = ev.scheme().staged_weighted(FitnessStage::QualityAndSpeed, &r.quality, &speed, qw).unwrap();
        assert_eq!(r.fitness.to_bits(), want.fitness.to_bits(), "the scheme's stage-2 angle");
        assert_eq!(r.fitness.to_bits(), crate::score::truenorth_angle(&r.inputs).to_bits(), "TrueNorth over the row's vector");
        assert_eq!(r.percentile, Some(crate::fitness::hff_percentile(r.fitness, r.d).unwrap()));
        let random_row = match i {
            0 => Some(0),
            2..=5 => Some(i - 1),
            _ => None,
        };
        match random_row {
            Some(j) => assert_eq!(seam_cols(r)[2], gen0[j].cols[2], "a random row's quality inputs are the runner's"),
            None => assert!(r.perfect && r.quality.iter().all(|x| *x == 0.0), "the reference program is exact"),
        }
    }
    let again = ev.evaluate_generation(&refs(&genes), 1).unwrap();
    assert!(!again.stage_changed && again.stage == FitnessStage::QualityAndSpeed);
    assert_eq!(again.evaluations_new, 0);
    assert!(again.rows.iter().zip(&s.rows).all(|(a, b)| a.fitness.to_bits() == b.fitness.to_bits()), "the same stage-2 angles");
    let _ = std::fs::remove_dir_all(&out);
}

/// THE RUN FILE BETWEEN BATCHES (`--eval single`: a batch a new program): the run file removed after the first batch
/// returns the first batch's row and a repeat of it (a cache hit needs no batch), and `stopped`; a call whose first
/// row needs a batch with the run file gone returns no row; a call of cache hits alone scores them.
#[test]
fn a_run_file_removed_between_two_batches_returns_the_first_batch_s_rows_and_stopped() {
    let ledger = runner_ledger();
    let (mut ev, out) = evaluator_with_run_file("seam_stop", EvalMode::Single);
    let g = generation_zero(&ev, &ledger);
    let run_file = out.join(crate::run_file::RUN_FILE);
    let genes = vec![g[0].clone(), g[0].clone(), g[6].clone(), g[7].clone()];
    let mut seen = Vec::new();
    let s = ev
        .evaluate_generation_with(&refs(&genes), 0, &mut |b| {
            seen.push(*b);
            std::fs::remove_file(&run_file).unwrap();
        })
        .unwrap();
    assert_eq!(seen, [BatchDone { batch: 1, stage: FitnessStage::Quality, rows_scored: 1 }], "one batch ran");
    assert!(s.stopped && !s.stage_changed);
    assert_eq!(s.rows.len(), 2, "the first batch's row and its repeat");
    assert_eq!(s.rows[0].fitness.to_bits(), ledger[0].fitness.to_bits());
    assert_eq!((s.evaluations_new, s.cost.modules), (1, 1));
    let none = ev.evaluate_generation(&refs(&genes[2..]), 1).unwrap();
    assert!(none.stopped && none.rows.is_empty() && none.evaluations_new == 0, "no batch starts with the run file gone");
    let hits = ev.evaluate_generation(&refs(&genes[..2]), 2).unwrap();
    assert!(!hits.stopped && hits.rows.len() == 2 && hits.evaluations_new == 0, "cache hits need no batch");
    let _ = std::fs::remove_dir_all(&out);
}

/// A gene drawn uniformly cell by cell (each cell a random member of its own alphabet), as phylu's engine draws a row:
/// its expressed tree is far larger than `Genetics::random_gene`'s.
fn uniform_gene(ev: &FutharkEvaluator, rng: &mut GeneRng) -> Vec<u32> {
    let g = &ev.gen;
    let mut gene = g.random_gene(rng).unwrap();
    for pos in 0..gene.len() {
        let alphabet = g.layout.alphabet(&g.contract, pos, &g.const_len);
        if !alphabet.is_empty() {
            let v = alphabet[rng.below(alphabet.len())];
            g.layout.mutate(&g.contract, &mut gene, pos, v, &g.const_len).unwrap();
        }
    }
    gene
}

/// THE SIZE GUARD (the P1 code review): histogram's bound is 64 x the reference program's 7 nodes; a gene drawn cell
/// by cell expresses far over it, is scored worst on every input (angle pi/2 in stage 1) without being rendered or
/// compiled, is charged once (a repeat is free) and is counted in the call's `oversized`; the ordinary row beside it
/// is the runner's, compiled alone in the call's one module.
#[test]
fn a_program_over_the_size_bound_is_scored_worst_and_never_compiled() {
    let ledger = runner_ledger();
    let (mut ev, out) = evaluator_with_run_file("seam_size", EvalMode::Batch);
    assert_eq!((ev.reference().level_order().len(), ev.size_bound()), (7, 7 * SIZE_BOUND_FACTOR));
    let mut rng = GeneRng::new(3);
    let big = (0..20).map(|_| uniform_gene(&ev, &mut rng)).find(|g| ev.gen.tree(g).unwrap().level_order().len() > ev.size_bound()).expect("a cell-by-cell gene over the bound in 20 draws");
    let nodes = ev.gen.tree(&big).unwrap().level_order().len();
    let genes = vec![big.clone(), generation_zero(&ev, &ledger)[0].clone(), big];
    let s = ev.evaluate_generation(&refs(&genes), 0).unwrap();
    assert_eq!(s.oversized, 2, "two rows over the bound");
    assert_eq!((s.cost.modules, s.evaluations_new), (1, 2), "one module for the ordinary row; the oversized program charged once");
    for r in [&s.rows[0], &s.rows[2]] {
        assert_eq!(r.ledger[0], "oversized");
        assert!(r.quality.iter().all(|x| *x == 1.0) && r.exact == [0, 0, 0] && !r.perfect, "worst on every input");
        assert_eq!(r.fitness, std::f64::consts::FRAC_PI_2, "the worst stage-1 angle");
        assert!(r.text.contains(&format!("{nodes} nodes")) && r.source.starts_with("oversized "), "never rendered to Futhark: {}", r.text);
    }
    assert!(s.rows[0].new_evaluation && !s.rows[2].new_evaluation, "a repeat is free");
    assert_eq!(s.rows[1].fitness.to_bits(), ledger[0].fitness.to_bits(), "the ordinary row is the runner's");
    assert!(ev.take_messages().iter().any(|l| l.contains("2 of 3 rows over the size bound")));
    let _ = std::fs::remove_dir_all(&out);
}
