//! Write the cases of the Gram problem directory (`problems/gram`) for the
//! generic harness (`fuller::wgsl::harness`): the constraint's 128 probes,
//! the two rungs, the five accuracy pairs the fast fitness scores and the
//! three timing shapes, as `gram::FitnessContext::new` builds them. CPU
//! only. The pairs' f64 references come from the pairs' cache (built there
//! on first use). About 1.6 GB, most of it the timing shapes' Phi and the
//! extrapolation pair's H; the manifests and the constraint suite's files
//! are committed, the rest is written here.
//!
//! `cargo run --release --features wgsl --example wgsl_problem_gram_cases -- <problem dir> [--cache <dir>]`

use std::path::Path;
use std::time::Instant;

use fuller::wgsl::gram::{fitness_pairs, write_problem_cases, RUNGS, TIMING};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().ok_or("usage: wgsl_problem_gram_cases <problem dir> [--cache <dir>]")?;
    let cache = args.iter().position(|a| a == "--cache").map(|i| args.get(i + 1).cloned().ok_or("--cache needs a directory")).transpose()?;
    let t = Instant::now();
    let bytes = write_problem_cases(Path::new(dir), &RUNGS, &fitness_pairs()?, &TIMING, cache.as_deref().map(Path::new))?;
    println!("{dir}: {:.1} MB written in {:.1} s", bytes as f64 / 1e6, t.elapsed().as_secs_f64());
    Ok(())
}
