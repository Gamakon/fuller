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
backend (ours, patched), not in the upstream pipeline, until shown otherwise. (Superseded below: the check is
upstream's.)

## Cause (2026-11-05)

**Upstream's, in the webgpu code generator both our backends share.** `futhark webgpu` stops on the reproducer with
the same error as `futhark rust-wgpu`, and the check (`findSingleMemoryType`, `src/Futhark/CodeGen/ImpGen/WebGPU.hs`)
is in the webgpu branch's base `14e9ccb` as it is ("Buffer used at multiple types"; our patches only added the name).
WGSL needs a type on every storage binding, so the generator gives each buffer one element type. A histogram whose
operator has no WGSL atomic, `f32.max` or `f32.min`, updates its buckets by compare-and-exchange of their bits (an
`i32` atomic on the sub-histogram buffer) and reads them as `f32`, in one kernel: two types. Any such histogram fails,
not only a lifted constant's: the reduced program is one line,

    entry main [n] (is: [n]i64) (xs: [n]f32) : []f32 = hist f32.max f32.lowest 10 is xs

(`tests/corpus/futhark_compile_failures/minimal/buffer_multiple_types.fut`). Behind it were two more faults the
reproducer needs fixed: the CAS loop's `f32` to bits conversion was emitted as `not_implemented(fptobits_f32_i32)`,
invalid WGSL, and the rust-wgpu host code had no `f32.isnan` (ImpCode spells a builtin called in host code as a `Call`
of `isnan32`, which the backend looked up among the program's own functions: "found no function isnan32";
`minimal/host_builtin_call.fut`, one line too).

## Fix

Patch 0013 (`futhark-rust-backend/patches/0013-fix-webgpu-a-buffer-accessed-at-i32-and-f32-is-one-a.patch`): a buffer
accessed at `i32` and `f32` is ONE binding of type `array<atomic<i32>>`, and every access to it goes through the
existing atomic helpers, which reinterpret the word (`atomic_read_f32_global` is `bitcast<f32>(atomicLoad(p))`); a
shared allocation accessed atomically at both is one `array<atomic<i32>>` likewise; `FPToBits`/`BitsToFP` on `f32` are
`bitcast<i32>`/`bitcast<f32>`. Patch 0014: a `Call` of a builtin in host code compiles to the scalar function it names
(`rt::sc::fun_isnan32`), as GenericC does.

**The design choice.** One word buffer reinterpreted at each access, not one binding a type. Two bindings of one
writable storage buffer in a dispatch are forbidden by WebGPU (writable aliasing), and two typed views would give the
CAS loop's `i32` writes and the `f32` reads no defined order. Reinterpreting is what Futhark's OpenCL and CUDA kernels
do with their untyped buffers (`atomic_cmpxchg_i32_global(...fptobits_f32_i32(...))`, then a `float` read), so the
generated code keeps their semantics. The cost is relaxed atomic loads and stores in place of plain ones on that one
binding of that one kernel; programs that compiled before are unchanged (their Rust is byte-identical, below).

**Checked (CPU only).** With the rebuilt fork (`rust-wgpu @ 5f6c334`): the reproducer, every entry of it, the reduced
programs and `tests/webgpu/hist_f32_minmax.fut` / `host_builtin_call.fut` compile, and their WGSL (ported as the
runtime ports it) parses and validates in naga 30.0.1; on 393 programs (the corpus, the minimal reproducers, the
fork's `tests/webgpu`, the k-means yardstick) the Rust is byte-identical to the cc09b31 build's but for the version
line. Not run on a GPU.

**In our test suite.** The reproducer is in the compile-failure corpus as
`tests/corpus/futhark_compile_failures/buffer_multiple_types__7d7b23e6af9f.fut` (index row: kind
`buffer_multiple_types`), with the two reduced programs in `minimal/`. It is not in `known_faults.tsv` (it was never
listed): the corpus tests require it to compile, through the rust path and the webgpu path.
