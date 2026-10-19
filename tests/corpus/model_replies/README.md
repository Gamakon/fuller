# The model-reply corpus (1,798 kernels)

The distinct kernels in every model reply under logs/HGP/001/*/calls (Gram, nluca, k-means problems,
2026-10-11 to 10-16), one file each (`kNNNN.wgsl`); `sources.tsv` maps each to the reply it came from.
Collected by the kingdom agent on 2026-10-16/17; committed 2026-10-19 so the measure is reproducible.

Measure: `cargo run --release --features wgsl --example wgsl_safe_convert -- tests/corpus/model_replies --out <dir>`;
its last line counts the kernels the safe reader converts. 2026-10-17 12:58 (fuller at that day's head): 1,756 of 1,798.
