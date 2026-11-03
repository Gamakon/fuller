#!/bin/bash
# THE K-MEANS CASES (docs/KMEANS_FITNESS.md; the advisor's review of 2026-11-03): cut from the dbpedia sample
# `dbpedia/sample_5000x1536.in` (1,000 random start rows of UltraDim's dbpedia-openai-1M-angular embeddings, 5
# consecutive rows each, 5,000 unit vectors of 1,536 f32; written ONCE by `dbpedia/sample.py`, Andrew's sampling
# rule, never by this script). Six disjoint row sets, 5,000 rows used once, k at most n / 20; the held cases hold out
# rows; the test case is never a fitness input; one small training case (the first 200 rows of the first training
# case) for the host interpreter's tests. Each case's starting centres are UltraDim's k-means++ draws over its rows at
# its seed (`fuller::futhark_sig::kmeans_converged::kmeans_plus_plus_rows`: the same crates at the same versions as
# UltraDim, so the draws are its), stored after the points in `<name>.in`. The timing input is the whole sample with
# the starts in `timing_n5000_k256_starts.in`. The table is `KMEANS_CASES` in `src/futhark_sig/kmeans_converged.rs`,
# written out by `examples/kmeans_cases.rs`; `cases.tsv` records it. INPUTS ONLY: k-means is scored on its clustering
# objective, run to UltraDim's stopping rule, so there is no answer to store. `reference64.fut` (UltraDim's step in
# f64) is a check of the seed's translation, not a stored output: a Rust test compiles and runs it on the cases.
#   bash tests/corpus/futhark_problems/kmeans/make_cases.sh
set -eu
cd "$(dirname "$0")"
test -f dbpedia/sample_5000x1536.in || { echo "dbpedia/sample_5000x1536.in is missing: run dbpedia/sample.py first (it needs the parquet file)" >&2; exit 1; }
cargo run --release --manifest-path ../../../../Cargo.toml --example kmeans_cases
