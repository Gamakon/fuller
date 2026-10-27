//! THE RUNNER STANDARD, ENFORCED (`docs/STANDARD_runners.md`). Fails when an evolution runner in fuller lacks a
//! standard piece.
//!
//! 1. `every_example_with_a_generation_loop_is_a_registered_runner`: every example whose source has a generation
//!    loop is in [`RUNNERS`], and every registered runner reaches the standard through `RunTelemetry` (directly, or
//!    through a library loop that creates one).
//! 2. `the_shape_loop_meets_the_standard`: `futhark_shapes_evolve::evolve` runs 2 generations and its directory
//!    holds the stream (run_start, a snapshot a generation with the generic view's numbers, a hall of fame, mutation
//!    events, run_end), the ledger with lineage and the checkpoint; a resume continues it to generation 3 on the
//!    same stream; deleting `killme.trigger` stops a long run with `stopped_by: killed`.

use std::path::{Path, PathBuf};

use fuller::futhark_kingdom::FutharkKingdom;
use fuller::futhark_shapes_evolve::{evolve, product_twice, suite, Config};
use fuller::telemetry::{parse_stream, EventKind, Record, STREAM_FILE};

/// Every runner in fuller's examples, and how it reaches the standard: `RunTelemetry::start` in its own source, or a
/// library loop that creates one.
const RUNNERS: &[(&str, &str)] = &[
    ("futhark_shapes_evolve", "use fuller::futhark_shapes_evolve::{evolve"),
    ("wgsl_gram_evolve", "RunTelemetry::start"),
    ("futhark_pixel_evolve", "RunTelemetry::start"),
];

/// A generation loop, as the examples write one.
fn has_generation_loop(src: &str) -> bool {
    let words = ["for gen in 0..", "for generation in 0..", "generation += 1", "gen += 1", "while generation <"];
    words.iter().any(|w| src.contains(w))
}

#[test]
fn every_example_with_a_generation_loop_is_a_registered_runner() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut problems = Vec::new();
    for entry in std::fs::read_dir(root.join("examples")).expect("the examples") {
        let path = entry.expect("an entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path.file_stem().expect("a stem").to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).expect("the source");
        if has_generation_loop(&src) && !RUNNERS.iter().any(|(n, _)| *n == name) {
            problems.push(format!("examples/{name}.rs has a generation loop and is not a registered runner"));
        }
    }
    for (name, via) in RUNNERS {
        let src = std::fs::read_to_string(root.join(format!("examples/{name}.rs"))).unwrap_or_default();
        if !src.contains(via) {
            problems.push(format!("examples/{name}.rs does not write the standard stream: it does not use {via}"));
        }
    }
    assert!(problems.is_empty(), "runners below the standard (docs/STANDARD_runners.md):\n  {}", problems.join("\n  "));
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fuller_runner_standard_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn small(generations: usize) -> Config {
    Config { population: 8, generations, measure_gpu: false, ..Config::default() }
}

fn records(dir: &Path) -> Vec<Record> {
    let (records, bad) = parse_stream(&std::fs::read_to_string(dir.join(STREAM_FILE)).expect("the stream"));
    assert_eq!(bad, 0, "every line of the stream parses");
    records
}

#[test]
fn the_shape_loop_meets_the_standard() {
    let k = FutharkKingdom::load_shapes().expect("the shape tables");
    let reference = product_twice(&k);
    let cfg = small(2);
    let input = suite(&k, &cfg.sizes, cfg.samples);
    let dir = scratch("shapes");
    evolve(&reference, &input, &cfg, &dir).expect("two generations");

    let rs = records(&dir);
    let Record::RunStart(start) = &rs[0] else { panic!("the stream opens with run_start") };
    assert_eq!(start.kingdom.as_deref(), Some("futhark_shapes"));
    assert_eq!(start.max_generations, 2);
    assert!(!start.hff_objectives.is_empty(), "the HFF inputs are named");
    let snaps: Vec<_> = rs.iter().filter_map(|r| if let Record::Snapshot(s) = r { Some(s) } else { None }).collect();
    assert_eq!(snaps.iter().map(|s| s.header.generation).collect::<Vec<_>>(), [0, 1], "a snapshot a generation");
    for s in &snaps {
        let g = &s.global;
        assert!(g.best_hff.is_some() && g.median_hff.is_some() && g.distinct_genes.is_some(), "the generic view's numbers");
        assert!(g.angle_groups.contains_key("quality") && g.angle_groups.contains_key("speed"), "the two-level HFF's groups");
        assert!(g.extras.contains_key("correct"), "the runner's extras");
    }
    let fame = rs.iter().rev().find_map(|r| if let Record::HallOfFame(h) = r { Some(h) } else { None }).expect("a hall of fame");
    assert!(!fame.rows.is_empty() && dir.join("hall_of_fame/rank1.txt").exists(), "the hall of fame, as text");
    assert!(
        rs.iter().any(|r| matches!(r, Record::Event(e) if e.kind == EventKind::EgraphExact && e.text_before.is_some() && e.text_after.is_some() && !e.parents.is_empty())),
        "the e-graph polish is a mutation event with its texts and lineage"
    );
    let Some(Record::RunEnd(end)) = rs.last() else { panic!("the stream ends with run_end") };
    assert_eq!((end.stopped_by.as_str(), end.generations), ("n_gen", 2));
    let ledger = std::fs::read_to_string(dir.join("ledger.tsv")).expect("the ledger");
    assert!(ledger.lines().any(|l| l.starts_with("1\t") && l.split('\t').nth(2).is_some_and(|p| p != "-")), "generation 1's children name their parents");
    assert!(dir.join("checkpoint.json").exists(), "the checkpoint");
    assert!(!dir.join("killme.trigger").exists(), "the kill trigger is removed at the end");

    // RESUME: the same directory, one generation more, the same stream.
    let more = Config { generations: 3, resume: true, ..small(3) };
    evolve(&reference, &input, &more, &dir).expect("the resume");
    let rs = records(&dir);
    assert_eq!(rs.iter().filter(|r| matches!(r, Record::RunStart(_))).count(), 2, "the resume appends its own run_start");
    let gens: Vec<u32> = rs.iter().filter_map(|r| if let Record::Snapshot(s) = r { Some(s.header.generation) } else { None }).collect();
    assert_eq!(gens, [0, 1, 2], "the resume continues at generation 2, not from the start");

    // THE KILL TRIGGER: a long run, stopped by deleting the file once it has written a generation.
    let dir = scratch("killed");
    let long = small(400);
    let (k_dir, k_ref, k_input) = (dir.clone(), reference.clone(), input.clone());
    let handle = std::thread::spawn(move || evolve(&k_ref, &k_input, &long, &k_dir));
    let trigger = dir.join("killme.trigger");
    let started = std::time::Instant::now();
    loop {
        let text = std::fs::read_to_string(dir.join(STREAM_FILE)).unwrap_or_default();
        if text.contains("\"type\":\"snapshot\"") && trigger.exists() {
            std::fs::remove_file(&trigger).expect("the operator deletes the trigger");
            break;
        }
        assert!(started.elapsed().as_secs() < 300, "the run wrote no generation in 300 s");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    handle.join().expect("the run thread").expect("the killed run returns");
    let Some(Record::RunEnd(end)) = records(&dir).pop() else { panic!("run_end") };
    assert_eq!(end.stopped_by, "killed");
    assert!(end.generations < 400, "the run stopped early: {} generations", end.generations);
}
