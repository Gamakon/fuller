//! THE K-MEANS CASES, cut from the dbpedia sample (`fuller::futhark_sig::kmeans_converged::KMEANS_CASES`; the advisor's
//! review of 2026-11-03): reads `tests/corpus/futhark_problems/kmeans/dbpedia/sample_5000x1536.in` (written once by
//! `dbpedia/sample.py`, Andrew's sampling rule) and writes, in the same corpus directory, each case's `<name>.in`
//! (its rows, then UltraDim's k-means++ starts over them at the case's seed, in Futhark's binary format), the timing
//! input's starts (`timing_n5000_k256_starts.in`, over the whole sample) and `cases.tsv`. Deterministic: the same
//! bytes every run. Run by `make_cases.sh`:
//!
//!   cargo run --release --example kmeans_cases

use std::path::Path;

use fuller::futhark_sig::kmeans_converged::{cut_case, KMEANS_CASES, KMEANS_TIMING_CUT};
use fuller::futhark_sig::{read_one, write_binary, KMEANS_SAMPLE, KMEANS_TIMING_STARTS};

fn real_main() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_problems/kmeans");
    let sample_path = dir.join(KMEANS_SAMPLE);
    let sample = read_one(&std::fs::read(&sample_path).map_err(|e| format!("{}: {e}", sample_path.display()))?)?;
    let write = |name: &str, bytes: &[u8]| std::fs::write(dir.join(name), bytes).map_err(|e| format!("{name}: {e}"));
    let mut tsv = String::from("# name\tsplit\trows (of dbpedia/sample_5000x1536.in, start..end)\tn\td\tk\tinit seed (UltraDim k-means++, ChaCha8; the starts are stored after the points in <name>.in; inputs only, no output: k-means is scored on its clustering objective run to UltraDim's stopping rule, docs/KMEANS_FITNESS.md; written by examples/kmeans_cases.rs through make_cases.sh)\n");
    for cut in &KMEANS_CASES {
        let t = std::time::Instant::now();
        let (points, starts) = cut_case(&sample, cut)?;
        write(&format!("{}.in", cut.name), &write_binary(&[&points, &starts]))?;
        tsv.push_str(&format!("{}\t{}\t{}..{}\t{}\t{}\t{}\t{}\n", cut.name, cut.split, cut.start, cut.end, cut.end - cut.start, points.dims[1], cut.k, cut.seed));
        println!("{}: rows {}..{}, k {}, seed {}, {:.1} s", cut.name, cut.start, cut.end, cut.k, cut.seed, t.elapsed().as_secs_f64());
    }
    let t = std::time::Instant::now();
    let (_, starts) = cut_case(&sample, &KMEANS_TIMING_CUT)?;
    write(KMEANS_TIMING_STARTS, &write_binary(&[&starts]))?;
    tsv.push_str(&format!("# the timing input: {} = the whole sample with the starts in {} (rows {}..{}, k {}, seed {})\n", KMEANS_TIMING_CUT.name, KMEANS_TIMING_STARTS, KMEANS_TIMING_CUT.start, KMEANS_TIMING_CUT.end, KMEANS_TIMING_CUT.k, KMEANS_TIMING_CUT.seed));
    println!("{}: the starts, k {}, seed {}, {:.1} s", KMEANS_TIMING_CUT.name, KMEANS_TIMING_CUT.k, KMEANS_TIMING_CUT.seed, t.elapsed().as_secs_f64());
    write("cases.tsv", tsv.as_bytes())?;
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
