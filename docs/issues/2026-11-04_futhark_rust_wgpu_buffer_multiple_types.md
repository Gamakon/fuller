# rust-wgpu fork: "Buffer ... used at multiple types" on k-means modules (2026-11-04)

**What happens.** `futhark rust-wgpu` (our fork, `~/.local/share/fuller/futhark-rust/bin/futhark`) stops with an
internal compiler error on modules of k-means programs:

    Internal compiler error (unhandled IO exception).
    Buffer lifted_const_reg_rep_tr_subhistos_mem_36002 used at multiple types: [IntType Int32,FloatType Float32]

**Where.** The F3b GPU check (`/private/tmp/claude-501/fbench/plan5a/f3b_kmeans_generation.sh`, 2026-11-04 10:21Z):
the deprecated runner's generation 0 on k-means, regime R, `--eval rust`; 77 halvings logged in its `bugs.log`. The
evaluator through the seam hits the same error on the same programs (22 failed compiles, 15 halvings, generation 0).
Every program still gets scored, because the batch is halved down to programs that compile.

**Reproduce.** `2026-11-04_futhark_rust_wgpu_buffer_multiple_types.fut` beside this note is one failing module (a
batch of 10 k-means programs): `futhark rust-wgpu 2026-11-04_futhark_rust_wgpu_buffer_multiple_types.fut`.

**Reading.** A lifted constant buffer of a histogram's sub-histograms is shared by an `i32` and an `f32` histogram
in one module. The error names a buffer the backend's memory code gives two element types; it is in the rust-wgpu
backend (ours, patched), not in the upstream pipeline, until shown otherwise. Not reduced yet.
