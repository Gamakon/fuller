# futhark-rust-backend: our Futhark fork, as a patch series

The `rust-wgpu` backend lives in a local fork of Futhark (`docs/PLAN_futhark_rust_wgpu.md`). This directory is what
makes that fork reproducible: the patches and the script that applies them.

| | |
|---|---|
| base | Futhark's `webgpu` branch (diku-dk/futhark PR "Add WebGPU backend") at `14e9ccbf3a47c06ed585d15e6827d186dd74d8a0`, the commit the shim's pinned compiler was built from. The webgpu backend is not in any release (not in 0.28.1), so the series cannot apply to one |
| `patches/0001` | the backend: `src/Futhark/CodeGen/Backends/RustWGPU.hs`, the `rust-wgpu` command, `builtinKernels` exported, and SetMem in a GPU kernel resolved by renaming outside the kernel's loops and conditionals |
| `patches/0002` | SetMem of register memory (`ScalarSpace`) in a kernel as a WGSL value copy, anywhere (A7's loop-rotated register tiles) |
| `patches/0003`, `0005` | names keyed by their full spelling (Futhark's `VName` compares by tag, and a constant and a local can share one); a helper renamed |
| `patches/0004` | a register array sized by a kernel constant (A7's `[Rx][Rx]f32` tile): a module-scope `const` with the marker `/*futhark-specialise:OVERRIDE*/ 0`, which the runtime fills in from the program's macro values before it makes the shader module (Futhark's OpenCL/CUDA backends pass such constants to the kernel compiler as macros) |
| `patches/0006` | an upstream defect: a boolean kernel constant was declared an `i64`, so an `if` on it was invalid WGSL (both A7 programs) |
| `patches/0007` | the rust-wgpu backend passes a unit scalar to a kernel (`rt::KArg::Unit`, the C backend's one-byte zero), and `bare` no longer strips the unit value `()` to an empty argument (192 corpus programs) |
| `patches/0008` | SetMem of device and shared memory in a kernel, in any loop or conditional: a memory variable that can name one binding is renamed to it; one that can name several holds a selector (the number of the binding it names now), and every access through it dispatches on the selector. Replaces 0001's rename-outside-control-flow (143 corpus programs) |
| `patches/0009` | an upstream defect, also in Futhark 0.28.1: the flatten pass paired every SegHist operator's destinations with the first operator's result shapes, an ill-typed SegHist ("Type error after pass 'simplify'") when two histograms in one map have different bin counts. Reproducer and bug report: `UPSTREAM_flatten_seghist.txt`; Futhark test `tests/flattening/hist/different-bins.fut` (38 corpus programs) |
| `patches/0010` | i64 reductions (a non-segmented i64/u64 reduction's block results are stored by an atomic write, which WGSL has no 64-bit type for: a plain write), shared memory merged across element types (one workgroup array a type), f32<->i64 conversions (FPToSI/FPToUI were unimplemented, u64->f32 scaled the high word by 2e32), u64 entry arguments (faxl-core's virtual-block generator, 2026-11-01) |
| `patches/0011` | a 1-D grid over WebGPU's 65,535 blocks a dimension (a map over 2^24 elements at 256 a block, an lmad copy of over 16,777,216 elements failed validation): every generated kernel's scalars end with its own block count `futhark_grid_x`, written by the runtime at each launch (`grid_offset` in the kernel table); the block id is `workgroup_id.x + workgroup_id.y * num_workgroups.x` and a block at or past the count returns; both runtimes dispatch such a grid as a rectangle (`futhark_rust_rt::program::split_grid`); lmad_copy reads its flat index the same way. Test: `tests/webgpu/large_grid.fut` |
| `patches/0012` | `futhark rust-wgpu --fuse=N`: the SOAC fusion pass bounded to N fusions a function (`futhark dev --fuse=N`'s gas), through `standardPipelineWith`/`gpuPipelineWith`/`gpumemPipelineWith` and `compilerMainWith` (a backend's pipeline chosen by its options); without `--fuse` the backend is byte for byte what it was. The batch host retries a single program the unbounded compile could not finish with `--fuse=10` (`futhark_batch::Host::compile`): it rescues the program whose fusion never settles (`docs/RESCUE_futhark.md`, 0.17 s against a kill at 53 s). Measured 2026-11-03 over the 378 programs of the corpus, the minimal reproducers and the k-means seed, `prog.rs` against the unbounded compiler's: `--fuse=1000` byte-identical on all 377 that compile, `--fuse=100` other code on 39 and one not finished, `--fuse=10` byte-identical on 34, other code on 248, 96 not compiled ("Known compiler limitation": an unfused form the webgpu code generator cannot copy) — so the bound is a retry, never the default |
| `patches/0013` | an upstream defect (the same check is in the webgpu branch's base): a histogram whose operator has no WGSL atomic (`f32.max`, `f32.min`) updates its buckets by compare-and-exchange of their bits (an `i32` atomic) and reads them as `f32` in one kernel, and the code generator gave each binding one element type ("Buffer ... used at multiple types: [IntType Int32,FloatType Float32]", every such k-means program). A global buffer accessed at `i32` and `f32` is now one `array<atomic<i32>>` whose every access reinterprets the word through the atomic helpers (as OpenCL/CUDA cast one untyped buffer; two typed bindings of one writable buffer are forbidden in WebGPU), a shared allocation accessed atomically at both likewise, and `FPToBits`/`BitsToFP` on `f32` are `bitcast<i32>`/`bitcast<f32>` (they were `not_implemented(...)`, invalid WGSL). Test: `tests/webgpu/hist_f32_minmax.fut`; issue `docs/issues/2026-11-04_futhark_rust_wgpu_buffer_multiple_types.md` |
| `patches/0014` | a builtin function called in host code (`f32.isnan` on an entry's scalar or a reduction's result) is an ImpCode `Call` of its name (`isnan32`), which the rust-wgpu backend looked up among the program's own functions ("found no function isnan32"); it is compiled to the scalar function it names, as GenericC does. Test: `tests/webgpu/host_builtin_call.fut` |
| `patches/0015` | an upstream defect, also in Futhark 0.28.1 and on master: a scan kernel turned into a stream to fuse with a stream producer names its accumulator results after its own lambda's parameters, and the kernel's lambda (unlike the producer's) was not renamed, so each name was bound twice ("Type error after pass 'Fuse SOACs' ... Name inpacc_... bound twice", 11 of 15 failing histogram and pathfinder rows of 225 to 1,100 nodes). The kernel's stream now has its lambda renamed. Test: `tests/fusion/scan-stream-horizontal.fut`; issue `docs/issues/2026-11-05_futhark_fusion_inpacc_bound_twice.md` |
| `patches/0016` | an upstream defect, also in Futhark 0.28.1 and on master (any GPU backend; `futhark c` does not run the pass): reduce-device-syncs searched every operand of a loop body with one reachable-bindings cache, whose results for a vertex on a cycle (a loop parameter and its update) hold only for the search that made them; an operand reaching such a vertex got no edge to the loop's result ("Migration graph sink could be reached from source after it had been attempted routed"). Each operand is now searched with a cache of its own. Test: `tests/migration/loop-operand-cycle.fut`; issue `docs/issues/2026-11-05_futhark_migration_graph_sink.md` |
| `build.sh [dir]` | clone the base (or bring an existing clone up to the series: its commits must be the series' first patches by `git patch-id`, the rest are applied), build with cabal (GHC 9.14.1, cabal 3.18, from Homebrew), install `<dir>/bin/futhark` and copy the series to `<dir>/bin/patches/`. Default `<dir>`: `$HOME/.local/share/fuller/futhark-rust` |

## The default compiler

fuller's default compiler, for the rust-wgpu path and the webgpu path alike (`fuller::futhark_rust::default_compiler`,
used when `FUTHARK_RUST_WGPU` / `FUTHARK_BRANCH` are unset), is `$HOME/.local/share/fuller/futhark-rust/bin/futhark`.
The test `futhark_rust::tests::the_default_compiler_is_built_from_the_committed_patch_series` fails unless
`bin/patches/` beside it is this series, name for name and byte for byte.

Installed 2026-10-31 as a COPY of the proven build `/private/tmp/claude-501/fixc/bins/fix4` (since 2026-11-01 kept at `/private/tmp/claude-501/keep/fix4`; md5
`b9fee28e3cf4f960e55d0ce69d9a33f7`, `--version`: `rust-wgpu @ e41f68c`), not a fresh `build.sh` run (one build at a
time on this machine; the rebuild is to be run under the cargo lock when no other build runs, and must compile the
corpus identically). fix4 was built 2026-10-29 08:12Z in `/private/tmp/claude-501/futhark-rust-fix` at `e41f68c`, a
clean tree (no uncommitted changes, no source newer than the binary) whose 9 commits on `14e9ccb` are patches
0001–0009, checked one by one with `git patch-id --stable`; its build log is `/private/tmp/claude-501/fixc/build_final2.log`.
The previous default, `/private/tmp/claude-501/futhark-rust/bin/futhark`, had patches 0001–0006 only.

Rebuilt 2026-10-31 13:44Z by `build.sh` (a fresh clone of the pinned base, the 9 patches applied by `git am`, under the
cargo lock; 2 min): `rust-wgpu @ 0323c7c`, md5 `c0b7e43e19a5f68f835c4d22e1b5b3ac`. On all 376 programs of
`tests/corpus/futhark_compile_failures` its `webgpu` C and its `rust-wgpu` Rust are byte-identical to fix4's, compiled
from the same path, but for the `// git:` version line.

Rebuilt 2026-11-03 17:06Z by `build.sh` (the clone brought up to the series: patch 0012 applied, cabal's incremental
build under the cargo lock; 1 min): `rust-wgpu @ cc09b31`, md5 `e6ddc2bf733f343f25b01568808154c2`. The install is an
atomic rename (`bin/futhark.new` moved over `bin/futhark`), so a compile running on the old binary keeps it.

Rebuilt 2026-11-05 02:09Z by `build.sh` (patches 0013 and 0014 applied, cabal's incremental build under the cargo lock; the
`Futhark.Version` module recompiled so `--version` names the commit): `rust-wgpu @ 5f6c334`, md5
`e326025decd6f4d70e7e535279bc1269`. Against the cc09b31 build, compiled from the same path, the `rust-wgpu` Rust of the 378
corpus programs, the minimal reproducers, `tests/webgpu/*.fut` and the k-means yardstick is byte-identical but for the
version line on every program both compile (393), and the `webgpu` C of `tests/webgpu/*.fut` likewise; the two new tests
compile only with the new build; the eight `tests/webgpu` programs with `f16` host values fail on both ("does not support
f16 on the host", unchanged).

Rebuilt 2026-11-05 03:15Z by `build.sh` (patches 0015 and 0016, cabal's incremental build under the cargo lock, the
`Futhark.Version` module recompiled): `rust-wgpu @ fb53cf1`, md5 `a66123a7e492999c11897f1373d82e2d`. Against the
5f6c334 build, compiled from the same path, on 596 programs (the 378 corpus programs, the 7 minimal reproducers, the
fork's `tests/webgpu`, `tests/fusion` and `tests/migration`): the `rust-wgpu` Rust and the `webgpu` C are
byte-identical but for the version line on all 582 both builds compile; 6 compile only with the new build (the two
new reproducers, their reduced programs and the two new Futhark tests); 8 fail on both (6 `tests/migration` programs
that need features the backends lack, and the two known faults). A first version of 0015 renamed every kernel turned
into a stream, a reduction's too, which shifted the name numbering of one program (`tests/fusion/map-scan3.fut`,
identical once names are normalised); it renames only a scan's stream now.

The fork is never pushed anywhere; regenerate the series from it with
`git format-patch 14e9ccb..HEAD -o futhark-rust-backend/patches`.
