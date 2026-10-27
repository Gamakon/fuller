# PLAN: the Futhark WGSL host on wgpu — a C-ABI shim over `wgpu-rs`

**Status:** S1–S5 built and measured 2026-10-27 (§12); S6 in progress. The §10 questions are decided. **Parent:** `docs/ASSESSMENT_futhark_kingdom.md` §3 (question A) and §6 build-plan item 6, where the host is left as *"H2: C-ABI shim, 35 functions, or H1: launch-plan interpreter"* — **this plan closes that choice in favour of H2.**

---

## 1. The one-paragraph version

Futhark's experimental WebGPU backend emits WGSL we can already validate and build a Metal compute pipeline from (measured — `docs/ASSESSMENT_futhark_kingdom.md` §10.7). The blocker is not the GPU and not the WGSL: it is that Futhark's generated host runtime is written against the **WebGPU C API** (`#include <webgpu/webgpu.h>`, or `<dawn/webgpu.h>` under `USE_DAWN`), while `wgpu-native` exposes a **Rust API only**. So the plan is: **own a small C ABI.** Write the WebGPU C entry points Futhark needs as Rust over `wgpu-rs`, generate the matching `webgpu/webgpu.h` with `cbindgen` (so the header cannot drift from the implementation), and let Futhark's generated C compile against *our* header with plain `cc` and link against our static library. Futhark's host keeps all of its own logic — allocation, aliasing, the event loop, the `futhark_context_*` API — and the GPU work lands on exactly the same `wgpu`/Metal path everything else in this repo uses.

---

## 2. The measured position this rests on

| fact | evidence |
|---|---|
| Futhark's `webgpu` backend emits real WGSL | 28 modules, 25 real WGSL programs, extracted from `out.c` by `tools/extract_futhark_wgsl.py` |
| the WGSL is **standard** and naga's front end can be made to accept it | `tools/port_futhark_wgsl.py` removes 13 f16-bitcast functions and 80 pointer-taking helpers; the scalar↔vector `bitcast` is a *naga* limitation, not invalid WGSL |
| **one** module reaches the driver today | `gpu_program.wgsl` passes naga's front end and validator, and **Metal builds a compute pipeline from it** (57.4 ms) once the workgroup-size `override` is set — the override is resolved by `wgpu` through naga's `ProgrammableStageDescriptor.constants` |
| the ABI is the wall | `rts/c/backends/webgpu.h` (1,087 lines) is written against the WebGPU C API; `wgpu-native` has no C API. Not "implemented inconsistently" — a hard ABI mismatch |

**The generator picks the include, so we do not need to patch Haskell.** `src/Futhark/CodeGen/Backends/CWebGPU.hs:374-381`:

```c
#ifdef USE_DAWN
  #include <dawn/webgpu.h>
#else
  #include <webgpu/webgpu.h>
  #include <emscripten.h>
#endif
```

Building **without** `USE_DAWN` is the path we want: our own `webgpu/webgpu.h` on the include path. The cost is a second, tiny header: `emscripten.h`, for the RTS's four `emscripten_sleep(0)` calls (the spin-yield inside the callback loops, lines 61/118/178/225). In a native build `emscripten_sleep(0)` is a no-op or `std::this_thread::yield()` — the real waiting happens in `wgpuInstanceWaitAny`.

---

## 3. Why this and not the alternatives

**H1 — reimplement the host in Rust.** Rejected. The launch plan (which kernels run, in what order, with what grid and buffers) is **per-program and not in the shader metadata**: Futhark may split a program into several kernels, reorder them, insert copies and transposes, and compile grid expressions into the generated C (`num_tblocks = max 1 (min (ceil (w / tblock_size)) max_num_tblocks)`). A Rust host must therefore either reimplement the host side of Futhark's code generator or reverse-engineer it from the generated C — no fixed format, no test oracle.

**H3 — build Dawn.** Rejected. It needs a heavy C++ dependency and abandons `wgpu`, which is the one thing this repo is built on.

**Full Rust (own Futhark→WGSL backend).** Rejected for this plan: Futhark's compiler — size-polymorphic parallel decomposition, fusion, tiling, the whole-program optimisation pipeline — is the value, and it is maintained by someone else. Owning the emission side is a different project.

**H2 keeps all of Futhark's host semantics** (buffer allocation, array aliasing, the event loop, `futhark_context_*`) — exactly where the subtle bugs live — and the shim's surface is **fixed and enumerable**, written once.

---

## 4. The surface: what we must implement

### 4.1 The functions

Verified from the branch clone (`rts/c/backends/webgpu.h`), **34 WebGPU API calls** (the assessment's "35" counts one extra; the authoritative list must be taken from the *generated* artefact in step S1). The RTS also has 21 `wgpu_*`-named helpers of its own — those are Futhark's, not ours.

| # | C entry point | the `wgpu-rs` operation behind it | difficulty |
|---|---|---|---|
| 1 | `wgpuCreateInstance` | `wgpu::Instance::new` (or `default`, by wgpu version) | mechanical |
| 2 | `wgpuInstanceRequestAdapter` | `Instance::request_adapter` | mechanical |
| 3 | `wgpuInstanceRequestAdapterF` | the same, as a *future* | **hard** (§4.3) |
| 4 | `wgpuInstanceWaitAny` | blocking wait over our futures | **hard** (§4.3) |
| 5 | `wgpuAdapterGetLimits` | `Adapter::limits` | mechanical |
| 6 | `wgpuAdapterRequestDevice` | `Adapter::request_device` | mechanical |
| 7 | `wgpuAdapterRequestDeviceF` | the same, as a future | **hard** |
| 8 | `wgpuDeviceCreateBuffer` | `Device::create_buffer` | mechanical |
| 9 | `wgpuDeviceCreateShaderModule` | `Device::create_shader_module` (Wgsl source) | mechanical |
| 10 | `wgpuDeviceCreateBindGroupLayout` | `Device::create_bind_group_layout` | mechanical |
| 11 | `wgpuDeviceCreatePipelineLayout` | `Device::create_pipeline_layout` | mechanical |
| 12 | `wgpuDeviceCreateComputePipeline` | `Device::create_compute_pipeline`; the `override` constants go in `ProgrammableStageDescriptor::constants` | mechanical |
| 13 | `wgpuDeviceCreateBindGroup` | `Device::create_bind_group` | mechanical |
| 14 | `wgpuDeviceCreateCommandEncoder` | `Device::create_command_encoder` | mechanical |
| 15 | `wgpuDeviceGetQueue` | `Device::queue` | mechanical |
| 16 | `wgpuDeviceDestroy` | drop the handle | mechanical |
| 17 | `wgpuShaderModuleGetCompilationInfo` | `ShaderModule::get_compilation_info` | mechanical |
| 18 | `wgpuQueueWriteBuffer` | `Queue::write_buffer` | mechanical |
| 19 | `wgpuQueueSubmit` | `Queue::submit` | mechanical |
| 20 | `wgpuQueueOnSubmittedWorkDone` | `Device::poll(Maintain::Wait)` | mechanical |
| 21 | `wgpuQueueOnSubmittedWorkDoneF` | the same, as a future | **hard** |
| 22 | `wgpuCommandEncoderBeginComputePass` | `Encoder::begin_compute_pass` | mechanical |
| 23 | `wgpuCommandEncoderCopyBufferToBuffer` | `Encoder::copy_buffer_to_buffer` | mechanical |
| 24 | `wgpuCommandEncoderFinish` | `Encoder::finish` | mechanical |
| 25 | `wgpuComputePassEncoderSetPipeline` | `ComputePass::set_pipeline` | mechanical |
| 26 | `wgpuComputePassEncoderSetBindGroup` | `ComputePass::set_bind_group` | mechanical |
| 27 | `wgpuComputePassEncoderDispatchWorkgroups` | `ComputePass::dispatch_workgroups` | mechanical |
| 28 | `wgpuComputePassEncoderEnd` | `ComputePass::end` | mechanical |
| 29 | `wgpuBufferMapAsync` | `BufferSlice::map_async` | mechanical |
| 30 | `wgpuBufferMapAsyncF` | the same, as a future | **hard** |
| 31 | `wgpuBufferGetConstMappedRange` | `BufferSlice::get_mapped_range` | mechanical |
| 32 | `wgpuBufferUnmap` | `BufferSlice::unmap` | mechanical |
| 33 | `wgpuBufferGetSize` | `Buffer::size` | mechanical |
| 34 | `wgpuBufferDestroy` | drop the handle | mechanical |

**28 mechanical, 6 tied to the future/callback API.** The mechanical 28 are *not new GPU code*: `src/gpu_eval.rs` already performs the `wgpu-rs` equivalent of essentially every one of them (shader modules, compute pipelines, bind groups, bind-group layouts, pipeline layouts, buffers, command encoders, writes, submits, dispatch, `map_async`/`get_mapped_range`/`unmap`, device polling). The shim is a **marshalling layer over code we already have**.

### 4.2 The types

The RTS names **~60 `WGPU*` types and enum constants** (`WGPUBuffer`, `WGPUBufferDescriptor`, `WGPUBindGroupEntry`, `WGPUStringView`, `WGPUMapMode`, `WGPUBufferUsage_*`, `WGPUStatus_Success`, …). They are plain C structs and enums, and `cbindgen` writes them from the Rust definitions — so they cannot drift from the implementation.

### 4.3 The hard subset: the future / callback API

Futhark's RTS uses the **Dawn-style callback API**, not the older callback form:

```
WGPUFuture, WGPUFutureWaitInfo, WGPUCallbackMode_WaitAnyOnly,
WGPUBufferMapCallbackInfo, WGPUQueueWorkDoneCallbackInfo,
WGPURequestAdapterCallbackInfo, WGPURequestDeviceCallbackInfo,
WGPUCompilationInfoCallbackInfo, WGPUUncapturedErrorCallbackInfo
```

with the `*F` entry points (`wgpuBufferMapAsyncF`, `wgpuQueueOnSubmittedWorkDoneF`, `wgpuInstanceRequestAdapterF`, `wgpuAdapterRequestDeviceF`) and the blocking `wgpuInstanceWaitAny`.

`wgpu-rs` has no future type. So the shim needs one: a small `WGPUFuture` — a shared slot holding a status and a payload, completed by the underlying `wgpu-rs` callback, plus a `wgpuInstanceWaitAny` that **blocks until a future completes** (driving `Device::poll`).

The semantics that matter: `WGPUCallbackMode_WaitAnyOnly` means "complete this future only by waiting", and `WGPUCallbackMode_AllowSpontaneous` means "the callback may fire on its own". The RTS's spin loops (`emscripten_sleep(0)` around `wgpuInstanceWaitAny`) are the consumer of that contract. **This is the one piece of the shim that needs a design, not a translation** — budget it as such, and write it first: if the future model does not work, nothing else matters.

### 4.4 The two headers we own

- **`webgpu/webgpu.h`** — generated by `cbindgen` from the shim's Rust definitions; installed into the include path used by the Futhark C compile.
- **`emscripten.h`** — a stub providing `emscripten_sleep(0)` (and whatever else the generated C references — to be enumerated in S1). Not a real Emscripten header; a native stand-in.

---

### 4.5 The future/callback model (S4 design, written before the code)

**What S1 found changes the shape of the problem.** The generated C has *two*
wait paths, chosen by the preprocessor, and each path needs a different half
of the surface (`futhark-wgpu-shim/surface/`):

| path | how the runtime waits | the functions only it calls |
|---|---|---|
| default (no `USE_DAWN`) | starts the request with `WGPUCallbackMode_AllowSpontaneous`, then spins `while (!released) emscripten_sleep(0);` | `wgpuBufferMapAsync`, `wgpuQueueOnSubmittedWorkDone`, `wgpuInstanceRequestAdapter`, `wgpuAdapterRequestDevice` |
| `USE_DAWN` | starts it with `WGPUCallbackMode_WaitAnyOnly`, then spins `while (!released) wgpuInstanceWaitAny(instance, 1, &f, 0);` | the four `*F` twins and `wgpuInstanceWaitAny` |

29 functions on the default path, 30 on the Dawn path, **34 in the union** —
the plan's count. The shim implements both, and the S5 test runs both.

**The model.**

1. **A future is a slot.** `WGPUFuture` is `{ uint64_t id }`. The instance owns
   a registry: id → (the slot, the C callback and its two userdata pointers,
   the device to poll). The slot is an `Arc<Mutex<Option<Completion>>>`; the
   `wgpu-rs` callback (the `map_async` closure, the
   `on_submitted_work_done` closure) does one thing — it fills the slot. It
   never calls C: `wgpu` fires those closures inside `Device::poll`, and a C
   callback must not run inside `wgpu`'s own call stack.
2. **Requests that `wgpu-rs` answers with a Rust future** (adapter, device,
   compilation info) are driven with `pollster::block_on` when they are
   started, so their slot is filled at once. That is legal under both modes:
   *completing* is not *calling back*.
3. **`WaitAnyOnly`: the callback runs only inside `wgpuInstanceWaitAny`.**
   `WaitAny(instance, n, infos, timeoutNS)` polls every device with pending
   work (`PollType::Poll` when `timeoutNS == 0` — the runtime's case — and a
   bounded `PollType::Wait` otherwise), then, for every listed future whose
   slot is full, removes it from the registry, runs its callback **once**, and
   sets `completed = 1`. It returns `WGPUWaitStatus_Success` if any listed
   future completed, `WGPUWaitStatus_TimedOut` otherwise, and
   `WGPUWaitStatus_Error` for an id it never issued (or one already
   delivered). A timeout of 0 never blocks, which is what the spec asks and
   what the runtime's spin expects.
4. **`AllowSpontaneous`: the callback runs before the starting call returns.**
   The callback "may fire at any time", including inside the call that
   started it. The shim uses that latitude: it registers the future exactly
   as in (3), then drives it to completion at once (`PollType::Wait` on the
   device until the slot fills) and delivers it, so the runtime's
   `released` flag is already set when its `while` loop is reached.
   **The `emscripten_sleep(0)` spin is therefore never entered — it is not
   load-bearing.** The shim counts every `emscripten_sleep` call and every
   `wgpuInstanceWaitAny` call, so a test can show it: the default path's count
   of `emscripten_sleep` calls is 0.
5. **`AllowProcessEvents`** (callbacks only inside
   `wgpuInstanceProcessEvents`) is not on the surface: the runtime never
   asks for it and the shim has no `ProcessEvents`. A request in that mode is
   a loud stop, never a silent re-interpretation.
6. **Errors.** A `wgpu` validation, out-of-memory or internal error goes to the
   device's uncaptured-error callback from `WGPUDeviceDescriptor` (Futhark's
   panics). Something the shim does not implement (an unknown `sType`, an
   unknown feature, an unknown usage bit) stops the process with a message
   naming it: a missing piece of the surface, never routed round.

**Why this is faithful, not just working.** `WaitAnyOnly`'s one promise — no
callback outside `WaitAny` — holds by construction, because only
`WaitAny` and the `AllowSpontaneous` start calls ever run a callback, and
they run only their own. `AllowSpontaneous` promises nothing about *when*,
so firing before return is inside the contract. The risk the plan named (the
model "cannot express `WaitAnyOnly` faithfully") does not arise.

## 5. The artefacts

| # | artefact | where | notes |
|---|---|---|---|
| 1 | the shim crate | `futhark-wgpu-shim/` (a **separate crate**, not a module of `fuller`) | `crate-type = ["staticlib"]`; a separate crate keeps `fuller`'s crate types (the maturin/PyO3 build) untouched, and the C side links the static library |
| 2 | the Rust implementation | `futhark-wgpu-shim/src/lib.rs` + modules per object kind | over `wgpu-rs`; handles are opaque pointers (`Box`/`Arc` + a tag) |
| 3 | the generated header | `futhark-wgpu-shim/build.rs` → `cbindgen` → `OUT_DIR/webgpu/webgpu.h` | `cbindgen` as a **build-dependency**, so the header is regenerated on every build and cannot drift |
| 4 | the stub `emscripten.h` | `futhark-wgpu-shim/include/emscripten.h` | hand-written, tiny |
| 5 | the link step | `futhark-wgpu-shim/build_futhark.sh` (or a `build.rs` helper) | compiles Futhark's generated C + the vendored `rts/c/` with plain `cc`, links our staticlib, runs it |
| 6 | the vendored Futhark C runtime | `futhark-wgpu-shim/futhark/rts/c/` | the `rts/c` tree from the branch, vendored so the build does not need a Futhark checkout (see open question 2) |

The Rust side of the repo gets one small module, `src/futhark_wgpu_host.rs`, that drives the above: given a Futhark source file, produce the WGSL via the branch compiler, build and link the executable, run it, and read the output arrays back — the sibling of the existing `src/futhark_exec.rs`, but on the GPU path.

---

## 6. The steps, each with its gate

| step | what | the gate |
|---|---|---|
| **S1** | enumerate the exact surface **from the generated artefact**, not from the RTS | the list of `wgpu*` symbols the generated `wgpuout.c` references is extracted, and the set of `emscripten.h` symbols it needs is listed. Output: the authoritative function list and the header's type list. *Gate: the counts are written down and the artefact is committed.* |
| **S2** | the header + a **stub** shim whose every function returns `WGPUStatus_Error` | Futhark's generated C **compiles and links** against our header with `cc`. *Gate: `cc` succeeds and the executable runs and reports "no device".* |
| **S3** | the 28 mechanical functions | the program creates a device, allocates a buffer, builds the compute pipeline, dispatches, and reads the buffer back. *Gate: a Metal capture shows the dispatch; the output matches the C backend bit-for-bit.* |
| **S4** | the future/callback subset and `wgpuInstanceWaitAny` | the RTS's own wait paths work without the `emscripten_sleep(0)` spin being load-bearing. *Gate: the full program runs to completion.* |
| **S5** | the **decisive** test | the same program compiled by (a) Futhark's C backend and (b) the branch's webgpu backend + our shim produce **bit-identical** output arrays on the same inputs, and both match `src/futhark_interp.rs`. *Gate: bit-identical, zero differing bits.* |
| **S6** | wire it into Fuller | `src/futhark_wgpu_host.rs` runs a **gene**: express → render → branch compiler → link → run → read back, so a gene's fitness can come from the GPU. *Gate: a gene's values equal the interpreter's.* |

**S1–S3 are the cheap part and they de-risk the whole route.** S4 is the design. S5 is the proof. S6 is the product.

---

## 7. Acceptance criteria

1. **The decisive one:** a Futhark program compiled through the branch's webgpu backend and hosted by our shim produces **bit-identical** output arrays to the same program compiled by Futhark's C backend, on the same inputs — and both match `src/futhark_interp.rs`.
2. A Metal capture (or `MTLCommandBuffer` counters) shows the kernel **actually dispatched**, not merely a pipeline built.
3. The shim's exported symbol set is **exactly** the enumerated list — a test asserts the generated C's undefined `wgpu*` symbols are a subset of the shim's exports, so the surface cannot quietly grow.
4. The header is **generated**, never hand-edited, and a build from a clean checkout reproduces it.
5. The link step needs **plain `cc`** — not Emscripten, not Dawn.

---

## 8. Risks and stop conditions

| risk | what it means | the response |
|---|---|---|
| **the upstream WGSL defects** | the pointer-argument helpers (`ptr<storage, array<T>, read_write>` is invalid WGSL) break 24 of 25 generated modules | not fatal for a compact program (the one module that worked is exactly that), but the compact program must be checked against it; `tools/port_futhark_wgsl.py` is the recorded workaround and its deletions are auditable |
| **the future/callback model** | `wgpu-rs` has no futures; we must design one | S4 is designed first; if the model cannot express `WGPUCallbackMode_WaitAnyOnly` faithfully, the route stops here and the answer becomes a Rust host (H1) |
| **the C toolchain** | the build now needs `cc` and a staticlib | a decision, not a risk — see open question 1 |
| **API drift** | `wgpu`'s Rust API moves; our C header is a contract with Futhark, not with wgpu | the header is generated from the Rust, so only the *Rust* side tracks wgpu; the C side is stable against Futhark's RTS, which changes rarely |
| **the size of the header** | ~60 types | mechanical with `cbindgen`; the types we must hand-design are the future/callback ones only |
| **compile throughput** | a Futhark compile is seconds; naga's is milliseconds | not a risk for this plan (it is a *host* plan, not an evolution plan), but it caps the evolution loop and belongs to the population-DAG work |

**Stop condition:** if S3 cannot produce a dispatch — i.e. the mechanical 28 do not get a kernel onto the GPU — the route is dead, and the correct answer is to own the emission side (a Rust backend) rather than bridge to a runtime we do not control. That is a cheap way to find out.

---

## 9. Out of scope

H1, H3, the upstream WGSL defects (those are Futhark's to fix, and the assessment records them for an upstream report), the population DAG, the device arena, the shape-aware IR, and the Futhark e-graph work in `docs/EGRAPH_futhark.md`.

---

## 10. The open questions, decided (2026-10-27, by the lead)

1. **A C toolchain and a `staticlib`: yes.** `cc` (Xcode CLT) and one extra
   crate, `futhark-wgpu-shim/`, with its own workspace root so fuller's crate
   types are untouched.
2. **`rts/c/` is vendored** into `futhark-wgpu-shim/futhark/rts/c/`, from the
   branch head `14e9ccb`; the source commit is recorded in the shim's
   `README.md`. (The generated C embeds the runtime, so the build does not
   read the vendored copy; it is the reviewable source, and its
   `backends/webgpu.h` was checked byte-identical to the embedded one.)
3. **The CI artefact is pinned:** run `39795721627`, the binary's sha256 and
   the merge commit it reports are in the shim's `README.md`, and
   `futhark-wgpu-shim/fetch_futhark_branch.sh` re-fetches and checks it.
4. **The C ABI is a bridge, not a permanent surface:** it stays exactly the
   34 functions of S1, behind `src/futhark_wgpu_host.rs`.

---

## 11. What already exists (do not rebuild it)

| artefact | where |
|---|---|
| the branch compiler (CI artefact) | `/tmp/branch_artifact/futhark-nightly-macos-arm64`; fetched with `gh run download 39795721627 -R diku-dk/futhark -n futhark-nightly-macos-arm64` |
| the branch's source tree (for `rts/c/`) | `/tmp/futhark_webgpu` (a depth-1 clone) |
| the WGSL extractor | `tools/extract_futhark_wgsl.py` |
| the naga error bisector | `tools/bisect_wgsl.py` |
| the port (f16 bitcast + pointer helpers) | `tools/port_futhark_wgsl.py` — it prints exactly what it removes, so the deletion is auditable |
| the probe (naga parse / validate / device pipeline) | `examples/futhark_wgsl_probe.rs` — three verdicts per module |
| the existing GPU machinery the shim marshals over | `src/gpu_eval.rs` (shader modules, pipelines, bind groups, buffers, dispatch, `map_async`, device polling) |
| the existing Futhark host harness (the C-backend sibling) | `src/futhark_exec.rs` |

---

## 12. What was built and measured (2026-10-27)

Everything below is in `futhark-wgpu-shim/`; the machine is the Apple M3 Max,
`wgpu` 30.0.1 on Metal, the compiler the pinned artefact.

| step | gate | result |
|---|---|---|
| **S1** | the surface from the generated artefact | **passed.** `surface/extract_surface.sh` resolves the runtime's two wait paths with `unifdef` and reads the identifiers left in the code: **29** `wgpu*` functions on the default path, **30** on the `USE_DAWN` path, **34** in the union (`surface/wgpu_functions.txt`); **65** `WGPU*` type and constant names; from `emscripten.h`, only `emscripten_sleep`. |
| **S2** | the generated C compiles and links with plain `cc` | **passed.** `build.rs` writes the header with `cbindgen` (34 functions, the 65 names); `cc -O2` compiles Futhark's `probe.c` with no diagnostics and links it against the static library. The plan's interim "stubs that report no device" were skipped: the functions were written for real at once. |
| **S3** | a real dispatch; output equal to the C backend's | **passed.** The probe program (`map2 (\x y -> x*2+y)`, 100 000 elements) runs on Metal, **bit-identical** to `futhark c`, and the dispatch is shown by the GPU's own pass timestamps (one pass, 15–16 µs, 6 of 6 runs). |
| **S4** | the future model; the full program runs | **passed.** Designed in §4.5 before the code. Both wait paths run to completion; the default path calls `emscripten_sleep` **0** times in every run, the `USE_DAWN` path completes through `wgpuInstanceWaitAny` alone. A unit test drives the model through the C entry points (no `WaitAnyOnly` callback outside `WaitAny`, exactly once; `AllowSpontaneous` done before return; a mapped buffer through the runtime's own `WaitAny(..., 0)` loop). |
| **S5** | the same program, C backend vs webgpu + shim, bit-identical | **passed on the exact programs, and the differences elsewhere are explained** — table below. |

**S5, on both wait paths** (`tests/s5/run.sh`; the inputs are seeded, 100 000
elements; the two paths agree with each other on every program):

| program | values bit-identical to `futhark c` | why |
|---|---|---|
| `p1` `map2 (\x y -> x*2 + y)` | **100 000 of 100 000** | |
| `p4` `reduce (+) 0i32 (map (*3))` | **1 of 1** | integer addition is associative |
| `p5` `map (abs . neg) (reverse xs)` | **100 000 of 100 000** | |
| `p2` `map2 (\x y -> x*y + 1)` | 99 728 of 100 000 | **the GPU fuses the multiply-add**: all 100 000 GPU values equal the once-rounded `fma(x, y, 1)`, all 100 000 C values the twice-rounded product-then-sum. Metal compiles with fast math, and neither `wgpu` nor naga lets the shim turn the contraction off. |
| `p3` `scan (+) 0f32` | 641 of 100 000 | the parallel scan adds in another order than C's loop (f32 addition is not associative) |
| `p6` a kingdom-shaped gene (scan, map2, reverse, map) | 836 of 100 000 | the same scan |
| A7 reference | — | **the branch compiler does not compile it**: "Known compiler limitation encountered … WebGPU backend Cannot handle SetMem statement … in GPU kernel" |
| A7 winner | — | the same compiler limitation |

So the shim is exact: where the arithmetic is exact, every bit agrees, on both
wait paths. Where it is not, the differences are the device's (contraction)
and the algorithm's (summation order), and both are named. The A7 kernels stop
in Futhark's webgpu code generator, before any C or WGSL exists — upstream
work, not the shim's.

**The WGSL port, extended.** `tools/port_futhark_wgsl.py` deleted the pointer
helpers, which moved the failure: the built-in kernels call them, and the
runtime builds all 24 built-in modules at start-up. `futhark-wgpu-shim/tools/
port_futhark_c.py` instead **specialises** each call whose pointer argument is
`&name` or `&name[i]` into a copy of the helper that names the variable — the
same meaning, in WGSL naga accepts — and rewrites the WGSL inside the
generated C. On the probe: 48 specialisations, 0 calls left unported, all 25
modules accepted by naga. (The pointer arguments are not invalid WGSL, as §8
says: they are WGSL under the `unrestricted_pointer_parameters` language
extension, which naga 30 lists as unimplemented.)

**Acceptance (§7).** 1: S5 above, and S6 adds the interpreter. 2: the GPU's
pass timestamps, read back by the shim (`FUTHARK_WGPU_SHIM_TIMESTAMPS=1`) —
resolved in the same command buffer as the pass, Metal returned zero for both
timestamps in 3 of 4 runs, so the shim resolves them in a submission of its
own, after which 6 of 6 runs gave real timestamps. 3: `build_futhark.sh`
checks on every link that each `wgpu*`/`emscripten*` symbol the objects leave
undefined is exported by the library **and** listed in the surface; a Rust
test checks the header declares exactly the 34. 4: a Rust test checks the
committed header is byte-identical to the one the build generated. 5: plain
`cc`.

