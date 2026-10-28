# PLAN: a Futhark `rust-wgpu` backend, and evolution evaluating through it

## Context

**Objective.** Futhark compiles a program to Rust: WGSL kernels plus a Rust host module over wgpu-rs. The generated
module runs in-process on a `wgpu::Device` and `wgpu::Queue` its caller gives. Evolution scores candidates through
THE SAME PATH: each generation's programs are compiled to Rust+WGSL and run in this process on our own device, so
what evolution scores is exactly what ships. There is no C, no shim, and no process per candidate.

**Approach.**
- A backend in a local fork of Futhark's webgpu branch (`/private/tmp/claude-501/futhark-rust`, never pushed). It
  reads the same ImpCode the webgpu backend reads and writes one Rust module.
- A runtime crate in fuller, `futhark-rust-rt/`, that does what Futhark's C runtime does for webgpu: memory and the
  free list, the pipeline cache, launches, sizes and tuning, copies.
- In-process loading: each batch's module is built by `rustc` into a dylib that links the runtime as a dylib, and
  the runtime's host loads it with `dlopen`.
- `futhark_evolve --eval rust` uses the batch host (`src/futhark_batch.rs`) with a new backend, so the commands,
  records, halving and outcomes are the batch path's.

**Unique feature.** The scored artefact is the shipped artefact: the generated Rust module, on a device the host
gives it. The batch path scores a C program that runs through a WebGPU C shim in a child process.

**Status (2026-10-28).**
- Steps 1–4 built.
- The gates, with their measurements, are in §7.
- The UltraDim demonstration is out of scope (Andrew, 2026-10-28): this is fuller/phylu tooling for the Futhark
  kingdom.

**Why.** The shim path (`docs/PLAN_futhark_wgpu_host.md`) proved that Futhark's WGSL runs exactly on our device. But
every generation paid for C compiles, a link and a process start, and the code it scored was not code anyone could
call from Rust.

## 1. Which layer to start from

The input is `Futhark.CodeGen.ImpGen.WebGPU.compileProg`'s `Program`. That is the webgpu backend's own input:
- `webgpuProgram` and `webgpuPrelude`: the WGSL;
- `webgpuKernels`: each kernel's `KernelInterface`, which gives the scalar offsets, binding slots, override names,
  dynamic block dimensions and shared-memory overrides;
- `webgpuMacroDefs`: override constants, as `KernelConstExp`s;
- `hostDefinitions`: ImpCode `Code HostOp`.

Starting here keeps every Futhark optimisation and the WGSL generator unchanged. Both backends get identical WGSL,
the same kernel interfaces and the same launch sequence. Only the host language differs, and that is what makes the
two paths comparable bit for bit.

Futhark's Python backends (`GenericPython`, `PyOpenCL`) show a non-C host can be generated from this same ImpCode.
The Rust generator follows their shape: a statement compiler, an expression compiler, entry-point wrappers.

## 2. The generated Rust

One module per program (`futhark rust-wgpu prog.fut -o prog` writes `prog.rs`). Its contents:

| item | what |
|---|---|
| `WGSL_PROGRAM`, `WGSL_PRELUDE`, `TEMPLATE_*` | the program's WGSL module; the prelude and the six built-in templates (copies, transpositions) the runtime instantiates, as `CWebGPU.builtinKernels` does. The generator checks its instantiation equals the webgpu backend's |
| `KERNELS: &[rt::KernelInfo]` | every kernel's interface, the program's own and the 24 built-in ones |
| `TUNING`, `MACRO_NAMES`, `macro_values` | the tuning parameters (name, class, default) and the override constants, evaluated once per context |
| `State`, `state(ctx)` | the program in one context: `rt::Program` (tuning, overrides, kernels) and the program's constants, initialised on first use |
| `pub fn f_<name>(ctx, st, ...)` | every ImpCode function; scalars are `rt::Var<T>` (a `Cell`), memory `rt::DevMem` / `rt::HostMem` |
| `pub fn <entry>(ctx: &rt::FutharkCtx, a0: &rt::Array<i32, 1>, ...) -> Result<_, rt::Error>` | every entry point, typed; an opaque result (a tuple) is `rt::Opaque` |
| `pub fn dyn_<entry>`, `ENTRIES: &[rt::EntryDesc]` | the entry points over dynamic values (`rt::Value`), with their types, for a host that loads a module it was not compiled with |

`rt::Array<T, R>` is a handle: memory and a shape. `Array::from_buffer(wgpu::Buffer, shape)` wraps a caller's
buffer without copying it; `from_slice` copies host data; `buffer()` and `to_vec(ctx)` read a result.

`FutharkCtx::new(device, queue)` takes the caller's device. It must have `SHADER_F16`, because Futhark's prelude says
`enable f16;`. GPU timing also needs `TIMESTAMP_QUERY`.

**Generated code is warning-free by construction.** `rustc -D warnings` compiles every generation.
- No `mut`: every variable has interior mutability.
- A declaration is emitted only for a name the body mentions; unused parameters are `_`.
- Every name is snake case (a VName's base and its unique tag; a function name gets a hash suffix when spelling
  changes it).
- Parentheses go only around operator operands, never around a whole argument, condition or value.
- Every item is `pub`.

**C's implicit conversions, made explicit.** Futhark's ImpCode relies on them: an `i64` variable set from an `i32`
expression, `sitofp_i32_f64` applied to an `i64`, a variable read at a different type than it was declared with. The
generator converts every operand to the type its operator takes (`binOpType`, `cmpOpType`, `convOpType`,
`unOpType`, `primFuns`), every assignment to its target's type, and every leaf to the type it is read at. It does
this with `as`, which is C's modular conversion between integer types.

## 3. The runtime (`futhark-rust-rt/`)

| piece | file | the C it reimplements |
|---|---|---|
| scalars: every operator by Futhark's C name (`sdiv32`, `smod_safe64`, `sext_i64_i32`, `fptosi_f32_i32`, `umul_hi32`, ...) | `sc.rs` | `rts/c/scalar.h`, host branch: floor `sdiv`/`smod`, `-1` divisors, `_safe` forms, unsigned operations on signed storage, wrapping, `fptosi` 0 out of range |
| memory: reference-counted blocks; `Allocate`, `SetMem`, `Free`; Futhark's free list (smallest acceptable block; < 4096 bytes, same tag, or < 2x wasted); a caller's buffer never enters the free list | `mem.rs` | `gpu_alloc`, `gpu_free`, `free_list.h`, `memblock_*` |
| the context: the caller's device and queue, shader modules (once per text), bind-group and pipeline layouts, the pipeline cache (module, entry, overrides), module states, GPU pass timing, a bound on every wait | `ctx.rs` | `backend_context_setup`, `gpu_create_kernel` |
| programs and launches: tuning defaults by class (block 256, grid 16 or 32 for `.seghist_`, tile 32, reg tile 2, threshold 32768, shared memory 16384, cache 2 MiB); macro overrides filtered by each kernel's `used_overrides`; scalars packed at the kernel's offsets; dynamic block dimensions and shared-memory sizes as per-launch overrides; one pass and one submission per launch; a launch with any zero dimension skipped | `program.rs` | `wgpu_size_setup`, `gpu_launch_kernel`, the generated `gpu_kernel_*` wrappers |
| copies: LMAD classification (`lmad_map_tr`, `lmad_memcpyable`); transpositions on the built-in kernels with the C runtime's variant choice; flat copies; general copies on `lmad_copy_*`; flat host copies; scalar reads and writes | `copy.rs` | `gpu.h` (`gpu_map_transpose`, `lmad_copy_gpu2gpu`, `gpu_memcpy`, `memcpy_*`), `copy.h` |
| values: `Array`, `Value`, `Opaque`, `EntryDesc`, Futhark's binary format | `values.rs` | the C entry-point wrappers and `values.h` |
| the WGSL port for naga (storage-pointer helpers specialised per call; f16 bitcasts deleted) | `port.rs` | a line-for-line port of `futhark-wgpu-shim/tools/port_futhark_c.py`, so both paths give naga the same text |
| the in-process host: the batch driver's commands and records, a device per session | `host.rs` | the batch driver (`src/futhark_batch.rs`, `DRIVER`) |

**Where the runtime knowingly differs from the C runtime.** No difference changes a value:
- pipelines are made at a kernel's first launch, not at context creation, and every pipeline is cached (the C runtime
  remakes a dynamic-block kernel's pipeline at every launch; the shim caches them);
- a scalar read or write at an offset that is not 4-aligned uses the aligned word around it (WebGPU rejects the
  unaligned copy the C runtime issues);
- host padding bytes are zeros, not the bytes after the source.

## 4. SetMem

The webgpu backend stops on `SetMem` inside a GPU kernel ("Cannot handle SetMem statement ... in GPU kernel").

A kernel's memory blocks are module-scope WGSL bindings (or workgroup arrays), never runtime values. So `SetMem x y`
is resolved at code generation, as `docs/PROPOSAL_memory_views_setmem.md` proposes: every later use of `x` is spelled
with the binding `y` names at that point. The WGSL generator's own rename map (`ksNameReplacements`) already exists
for this.

The rename is exact when the SetMem is outside every loop and conditional of the kernel. Then the statements
translated after it are exactly the statements executed after it. The fork tracks control depth (`ksControlDepth`).
Inside a loop or conditional the binding could differ between iterations or branches, so that remains a compiler
limitation, with a message that says so.

The rename is in the shared WGSL generator (`ImpGen/WebGPU.hs`), so the fork's webgpu backend gets it too. The shim
path uses the pinned upstream binary, which does not have it.

On the host, `SetMem` is a reference copy (`DevMem::set_from`), as `memblock_set` is.

## 5. How evolution loads candidates in-process

**Option one: a dylib per batch.**
- The batch's module is compiled by `rustc` (`--crate-type dylib`, `-C prefer-dynamic`, `-C opt-level=1`,
  `-D warnings`) against the runtime, which is itself a dylib built with `-C prefer-dynamic`.
- The process then holds one runtime, one wgpu and one libstd.
- The host `dlopen`s the dylib and calls `futhark_rust_module()`, by the Rust ABI (same compiler, same runtime image).
- fuller reaches the runtime only through its C entry points (`futhark_rust_rt_session_new`, `_run`,
  `_free_bytes`, ...), as it reaches the shim. fuller's own wgpu and the runtime's never meet.

**Option two: an interpreter of the launch plan, with naga-validated WGSL.**
- No per-generation compile.
- But it needs a second emitter (a serialised launch plan) and a second implementation of the host semantics, an
  interpreter beside the generated code.
- What evolution scored would then not be what ships, which is the direction's first requirement.

**Measured (2026-10-28, M3 Max), the reduce yardstick's module (141 KB of Rust):**

| step | seconds |
|---|---|
| `rustc` at opt-level 0 | 0.17 |
| `rustc` at opt-level 1 | 0.21 |
| `rustc` at opt-level 2 | 0.21 |
| for comparison, the batch path's `cc` of one reduce generation | 0.4 (`PLAN_futhark_wgpu_host.md` §13) |

**Option one is chosen.**
- Its compile costs less than the C it replaces.
- It scores the shipped code.
- Option two's only advantage, no compile, would be worth less than 0.2 s a generation.

How the build works:
- `futhark-rust-rt/prebuild_rt.sh <dir>` builds the runtime once per run and copies it, with the dependency rlibs
  and proc-macro dylibs `rustc` needs, into the run's own directory. It sets an `@rpath` install name and an rpath to
  the toolchain's libstd, and re-signs the dylib. A later rebuild never changes a running fit's runtime.
- Each batch gets a crate name unique in the process (`futhark_gen_p<pid>_b<n>`).
- A loaded module is never unloaded: its module state's code lives in it.

**Failures in-process** are handled as the driver's failures:
- a command that fails writes its failure record, and the next command runs on a new context (as the driver's next
  process does);
- a wait past its bound, or a lost device, also drops the device;
- a panic in a generated module is caught and becomes that command's failure;
- the program's WGSL module is made and validated before the first command, as the C runtime makes it at context
  creation. A module that does not validate fails the batch's set-up, and the batch is halved as on the batch path.

## 6. Coverage: every ImpCode construct of the five benchmarks

The generator's match over `Code HostOp` is total: GHC's `-Wall` would name a missing constructor. Every construct
has a translation, except the stated limitations below.

| construct | in the five benchmarks | translation |
|---|---|---|
| `DeclareScalar`, `SetScalar` | all | `let v = rt::Var::<T>::new(0)`, `v.set(e)` |
| `DeclareMem`, `Allocate`, `Free` (device) | all | `rt::DevMem::new()`, `.alloc(ctx, bytes, tag)?`, `.free()` |
| `SetMem` (host) | memory-returning functions | `.set_from(&y)` |
| `Op (LaunchKernel ...)` | all (21 launches in the five) | `rt::launch(ctx, &st.prog, k, grid, block, &[scalars], &[mems])?` |
| `Op (GetSize ...)`, `Op (GetSizeMax ...)`, `Op (CmpSizeLe ...)` | all; histogram and kmeans read maxima | `st.prog.tuning(i)`, `ctx.size_max(class)`, a comparison |
| `If`, `For` | histogram, kmeans (`if`); pathfinder's row loop, histogram, kmeans (`for`) | `if ... {} else {}`, `for v_i in 0..n` |
| `While` | (none of the five) | `while c {}` |
| `Call` | histogram, kmeans, every program's `builtin#replicate` | `let r = f_name(ctx, st, args)?;` then each destination set |
| `Copy` device↔device (1-D, 2-D) | histogram, pathfinder, kmeans | `rt::lmad_copy_dev2dev` (transposition, flat copy, or the general kernel) |
| `Copy` host↔device, host↔host | (none of the five) | `rt::lmad_copy_host2dev` / `_dev2host` / `_host2host` |
| `Read` / `Write` (device scalar) | reduce reads its result | `rt::read_scalar` / `rt::write_scalar` |
| `Assert` | histogram, pathfinder | `if !c { return Err(...) }` with the message |
| `Meta`, `DebugPrint`, `TracePrint`, `GetUserParam`, `DeclareArray` | comments, debug | comment / nothing / `ctx.trace` / the tuning or default / a host constant array |
| constants and their initialisation | every program (`counters_mem`) | `State` fields, `init_constants` on first use |
| entry points: transparent arrays and scalars in; arrays, scalars or an opaque tuple out | all; the batch's quality entries return an opaque tuple | typed `pub fn` plus a dynamic adapter |

**Stated limitations** (each a `compilerLimitation` naming itself):
- `f16` values on the host (f16 in kernels is supported);
- opaque entry-point arguments;
- memory spaces other than device and host;
- `SetMem` inside a kernel loop or conditional.

## 7. The gates

| step | gate | result |
|---|---|---|
| 1 design | covers every ImpCode GPU construct the five benchmarks use | §6 |
| 2 backend | reduce, scan, histogram, pathfinder and the k-means step compile to Rust+WGSL | the fork writes all five modules (130–254 KB); all five compile against the runtime with no warning |
| 3 runtime | every benchmark's output bit-identical to `futhark c` on its cases, on Metal; the time a call no worse than the shim path's | `futhark_rust::tests::on_the_gpu_the_benchmarks_equal_futhark_c_and_the_shim` (table below) |
| 4 evolution | `--eval rust`: quality inputs bit-identical to `--eval batch` over 3 generations; evaluations per second against batch | `futhark_batch::tests::on_the_gpu_the_rust_backend_equals_the_batch_backend` and the paired runs (below) |

### Measured (2026-10-28, Apple M3 Max, wgpu 30.0.1 on Metal)

**Step 2: compiles.** The fork writes all five benchmark modules (130–254 KB of Rust each). With `-D warnings` they
compile against the runtime on wgpu 30.0.1 and on 27.0.1.

| set | rust-wgpu compiles | webgpu (pinned branch binary) compiles |
|---|---|---|
| the five benchmarks' published programs | 5 of 5 | 5 of 5 |
| 24 random kingdom programs and the reference, for each of the four problems (`every_problems_random_programs_compile_to_rust`) | 100 of 100, in 4 of 4 batches | 4 of 4 batches |
| the A7 shape programs (`futhark-wgpu-shim/tests/s5/a7_*.fut`) | 2 of 2, both naga-valid | 0 of 2 ("Cannot handle SetMem statement in GPU kernel") |

A7 needed three fork changes:
- SetMem of register memory as a value copy (patch 0002);
- a register tile sized by a specialisation constant (0004);
- an upstream defect: a boolean kernel constant was declared `i64`, so `if` on it was invalid WGSL (0006).

**Compile time, a batch of 25 programs (seconds):**

| problem | fork compiler | rustc | webgpu + port + cc + link |
|---|---|---|---|
| reduce | 0.27 | 2.49 | 1.66 |
| scan | 0.46 | 2.37 | 2.86 |
| histogram | 0.91 | 5.76 | 5.76 |
| pathfinder | 0.33 | 2.30 | 1.88 |

`rustc` at opt-level 0 instead of 1 saved 0.03 s of 0.21 s on a reduce generation, so 1 is kept.

**Step 4: backend equivalence (GPU).** `futhark_batch::tests::on_the_gpu_the_rust_backend_equals_the_batch_backend`
runs, for each problem, 8 random programs, the reference and the published program as members. Every outcome is
identical between `Backend::Rust` and `Backend::Gpu`: every quality output, the published program's output, and timed
pairs on both.

| problem | batch compile s | batch run s | rust compile s (fork + rustc) | rust run s (of it pipelines) |
|---|---|---|---|---|
| reduce | 1.15 | 3.25 | 1.53 (0.20 + 1.33) | 3.19 (0.22) |
| scan | 1.20 | 3.55 | 0.86 (0.21 + 0.66) | 3.47 (0.64) |
| histogram | 2.96 | 25.38 | 2.83 (0.45 + 2.38) | 25.49 (2.66) |
| pathfinder | 1.30 | 1.57 | 1.06 (0.23 + 0.83) | 1.53 (0.11) |

**Step 3: the benchmarks against `futhark c`, and the GPU time a call.** On reduce: 5 of 5 cases bit-identical to C;
the fastest GPU time a call is 8.3 µs on the rust path against 12.6 µs on the shim. The full table (five benchmarks,
the shim comparison with trailing whitespace stripped) and the A7 values against C come from GPU session 5.

**What the first sessions found and fixed:**
- the runtime's rpath must be the toolchain's libstd directory (`<sysroot>/lib/rustlib/<host>/lib`), now covered by a
  CPU test;
- the in-process host must bound a whole command, as the driver's `alarm` does, not each wait. A slow reduce program
  that the batch driver's 120 s alarm stopped finished on the rust path, the only difference in 72 ledger rows;
- a session's binaries must be its own copies: an installed fork compiler rebuilt mid-session invalidated a run.
