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
| `build.sh [dir]` | clone the base, apply the series, build with cabal (GHC 9.14.1, cabal 3.18, from Homebrew), install `<dir>/bin/futhark` |

The fork is never pushed anywhere; regenerate the series from it with
`git format-patch 14e9ccb..HEAD -o futhark-rust-backend/patches`.
