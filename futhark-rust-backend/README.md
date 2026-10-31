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
| `build.sh [dir]` | clone the base (or bring an existing clone up to the series: its commits must be the series' first patches by `git patch-id`, the rest are applied), build with cabal (GHC 9.14.1, cabal 3.18, from Homebrew), install `<dir>/bin/futhark` and copy the series to `<dir>/bin/patches/`. Default `<dir>`: `$HOME/.local/share/fuller/futhark-rust` |

## The default compiler

fuller's default compiler, for the rust-wgpu path and the webgpu path alike (`fuller::futhark_rust::default_compiler`,
used when `FUTHARK_RUST_WGPU` / `FUTHARK_BRANCH` are unset), is `$HOME/.local/share/fuller/futhark-rust/bin/futhark`.
The test `futhark_rust::tests::the_default_compiler_is_built_from_the_committed_patch_series` fails unless
`bin/patches/` beside it is this series, name for name and byte for byte.

Installed 2026-10-31 as a COPY of the proven build `/private/tmp/claude-501/fixc/bins/fix4` (md5
`b9fee28e3cf4f960e55d0ce69d9a33f7`, `--version`: `rust-wgpu @ e41f68c`), not a fresh `build.sh` run (one build at a
time on this machine; the rebuild is to be run under the cargo lock when no other build runs, and must compile the
corpus identically). fix4 was built 2026-10-29 08:12Z in `/private/tmp/claude-501/futhark-rust-fix` at `e41f68c`, a
clean tree (no uncommitted changes, no source newer than the binary) whose 9 commits on `14e9ccb` are patches
0001–0009, checked one by one with `git patch-id --stable`; its build log is `/private/tmp/claude-501/fixc/build_final2.log`.
The previous default, `/private/tmp/claude-501/futhark-rust/bin/futhark`, had patches 0001–0006 only.

The fork is never pushed anywhere; regenerate the series from it with
`git format-patch 14e9ccb..HEAD -o futhark-rust-backend/patches`.
