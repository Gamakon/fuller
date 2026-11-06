//! THE RUNNER STANDARD, ENFORCED (`docs/STANDARD_runners.md`). Fails when an evolution runner in fuller lacks a
//! standard piece.
//!
//! 1. `every_example_with_a_generation_loop_is_a_registered_runner`: every example whose source has a generation
//!    loop is in [`RUNNERS`]; every registered runner reaches the standard through `RunTelemetry` (directly, or
//!    through a library loop that creates one) and carries the two-level HFF. (The legacy WGSL kingdom's excluded
//!    runner `wgsl_gram_evolve` was deleted with that kingdom, plan 7.7 K5.)
//!
//! fuller runs no evolution (`CLAUDE.md`): the Futhark runners (`futhark_evolve`, `futhark_a7_evolve`,
//! `futhark_pixel_evolve`, the `futhark_search` loop) were frozen as binaries at the tag `pre-5b` and deleted
//! (`docs/PLAN_futhark_kingdom.md` 5b), so no runner is registered here; phylu's `tests/runner_standard.rs` registers
//! the Futhark runner on phylu's engine.

use std::path::Path;

/// Every runner in fuller's examples, how it reaches the standard (`RunTelemetry::start` in its own source, or a
/// library loop that creates one), and the source marker of its two-level HFF (θ_quality, θ_speed; every runner here
/// has speed inputs).
const RUNNERS: &[(&str, &str, &str)] = &[];

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
        if has_generation_loop(&src) && !RUNNERS.iter().any(|(n, _, _)| *n == name) {
            problems.push(format!("examples/{name}.rs has a generation loop and is not a registered runner"));
        }
    }
    let source = |name: &str| std::fs::read_to_string(root.join(format!("examples/{name}.rs"))).unwrap_or_default();
    for (name, via, hff) in RUNNERS {
        let src = source(name);
        if !src.contains(via) {
            problems.push(format!("examples/{name}.rs does not write the standard stream: it does not use {via}"));
        }
        if !src.contains(hff) {
            problems.push(format!("examples/{name}.rs has speed inputs and its source lacks the two-level HFF's {hff:?}"));
        }
    }
    assert!(problems.is_empty(), "runners below the standard (docs/STANDARD_runners.md):\n  {}", problems.join("\n  "));
}
