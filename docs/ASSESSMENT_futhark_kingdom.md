# Assessment: a Futhark-inspired, evolution-native GPU big-data kingdom

Date: 2026-10-26. Status: research findings, then a built and measured kingdom.

Scope: can Futhark's experimental WGSL backend become the execution backend for a new
compact Karva kingdom (Karva genes -> Futhark-style typed AST -> Futhark -> WGSL ->
our Rust/wgpu runtime on Apple Metal -> HFF fitness -> Phylu)?

**Where this stands.** Sections 1–9 are the assessment, written from the sources.
Section 10 is what was then *measured*, on this machine, with the real tools:
the kingdom is built (4 types, 38 rows), every random gene is legal, every
rendered program compiles with the real Futhark compiler, and every one of them
agrees with an independent interpreter of the kingdom, element for element. The
WGSL path is not there yet, and §10.6 says exactly why, in numbers.

---

## 1. Verdict

**Feasible, but not as a drop-in.** The three moving parts each exist; the gaps are
integration, not invention. But the architecture as drawn has one structural mismatch
and one scaling problem that the current `wgsl_true` numbers make concrete:

1. **Futhark's grid dimensions and kernel launch order are not in the shader
   metadata.** They are computed on the host, at run time, by the generated C program.
   A Rust/wgpu host cannot launch a Futhark kernel from the WGSL alone.
2. **A Futhark compile is a whole-program optimisation pipeline (seconds); a naga
   validation of a WGSL text is milliseconds.** At the scale `wgsl_true` actually
   runs at, per-genome Futhark compilation dominates everything else.

Neither is fatal. Both are bounded and have known fixes, and both are *cheaper* in the
compact language than they are in `wgsl_true`. The recommendation in section 8 is
therefore to build the kingdom first (it is needed under every option) and to de-risk
the WGSL path in parallel with a cheap, toolchain-free probe.

**The third gap, found by building it.** A well-typed program is not necessarily a
well-*defined* program. Futhark's `reduce` and `scan` require the neutral element to
be a true neutral element for the operator, and a program that violates that is
undefined — the compiler cannot detect it, and the wrong answer is silent. A
kingdom that claims "every gene is legal and means exactly one thing" therefore has
to carry the target language's *preconditions* in its types, not just its types.
§10.3 has the measurements; §10.5 has the second instance of the same lesson
(`f32.round` rounds a half to even, not away from zero).

---

## 2. What already exists

### 2.1 On our side

| Component | State | Where |
|---|---|---|
| Typed multi-typed GEP core | built, tested | `src/typed_gep.rs` (`express`, `validate`, `write_back`, `promotions`) |
| Kingdom contract / loader | built | `src/kingdom_contract.rs` |
| `wgsl_true` kingdom | built, GPU-validated | `kingdoms/wgsl_true/` (305 types, 8,841 rows) |
| WGSL -> Metal runtime | built | `src/wgsl/harness.rs`, `src/gpu_eval.rs` (3,377 lines) |
| naga front end | in-tree | `naga` 30.0.1 with `wgsl-out` + `msl-out` features (`Cargo.toml:53`) |
| Declarative kernel contract for fitness | built | `problems/*/problem.toml` (bindings, entry, `grid_header`, inputs/outputs, cases) |
| GPU-resident fitness | built | `phylu/src/evolve/hff_gpu.rs`, `gpu_fold.rs` |

Measured behaviour of the existing GPU path (`docs/reports/2026-10-22_ST9_volumepassage_champion/README.md:91`):

> **Fitness evaluations:** 85,437, with 0 device failures. ... The pipeline cache hit
> rate was 10 %.

So one ST9 run compiled roughly **7,700-8,500 distinct kernels** through naga +
Metal. Best HFF moved 0.29242 -> 0.26826, almost entirely inside the first 250
generations. Tiling overhead measured at 3x. GPU clock is bimodal (a real source of
fitness noise). This is the number that matters for question D.

### 2.2 On Futhark's side

Everything below was read from the actual PR diff (703 files) and cross-checked
against raw.githubusercontent at the PR head sha.

* **PR:** [diku-dk/futhark#2140 "Add WebGPU backend"](https://github.com/diku-dk/futhark/pull/2140),
  author `spaarmann`, state **open and draft**, branch `webgpu` **in `diku-dk/futhark`
  itself**, head sha `14e9ccbf3a47c06ed585d15e6827d186dd74d8a0`, 286 commits / 703
  files, last activity 2026-10-22. Participants include `spaarmann`, `athas`
  (Futhark lead), `candrdk`, `TudorPal`, `Jesssullivan`, `teddyjfpender`.
* **No WGSL/WebGPU support in upstream `master`.** `futhark.cabal` on master has no
  such module. The backend is PR-only. It also had **bit-rotted** and needed 286
  rebase commits to come back to life (Andreasen's BSc thesis, section 2).
* **New CLI command** `futhark webgpu` (`src/Futhark/CLI/WebGPU.hs`, 17 lines) —
  a thin wrapper over `compileWebGPUAction`. Accepts the `futhark-c` option set.
  There is **no "emit WGSL only" mode**, but two harvest paths exist:
  * `futhark dev --compile-imp-webgpu prog.fut` writes the WGSL to stdout
    (caveat: the `Pretty Program` instance is marked `-- TODO: print everything`, so
    it prints prelude + WGSL + host definitions but **not** the kernel metadata map).
  * `futhark webgpu prog.fut --library` writes `out.c`, `out.wrapper.js` and
    `out.json` **before** the emcc step runs, so the artefacts survive an emcc failure.
* **Structured metadata exists inside the compiler.** `Futhark.CodeGen.ImpCode.WebGPU`
  defines
  `KernelInterface { safety, scalarsOffsets, scalarsSize, scalarsBindSlot,
  memBindSlots, overrideNames, dynamicBlockDims, sharedMemoryOverrides, gpuProgram }`,
  and `Program { webgpuProgram, webgpuPrelude, webgpuMacroDefs,
  webgpuKernels :: Map KernelName KernelInterface, ... }`.
  It is serialised into the generated C as a `struct wgpu_kernel_info
  wgpu_kernel_infos[]` table (name, num_scalars, scalars_binding, scalars_size,
  scalar_offsets, num_bindings, binding_indices, num_overrides, used_overrides,
  num_dynamic_block_dims, dynamic_block_dim_indices/names, shared-mem overrides,
  gpu_program). **Mechanically extractable, but not JSON.**
* **The WGSL is never written to a `.wgsl` file** — it is embedded in the generated C
  as `static const char *gpu_program[] = {...}`.

---

## 3. Question A: can we run Futhark-generated WGSL on Apple Metal?

**Shortest path, and what is missing.**

The generated C runtime (`rts/c/backends/webgpu.h`, 1,088 lines) targets the **WebGPU
C API**: `#include <webgpu/webgpu.h>` (Emscripten) or `<dawn/webgpu.h>` under
`USE_DAWN`. `wgpu-native` exposes a Rust API only — there is no C API. So **the
shipped host runtime cannot be linked against `wgpu`**. This confirms the earlier
suspicion, and it is the concrete form it takes: not an "API implemented
inconsistently" problem, but a hard ABI mismatch.

The C surface actually used is small and enumerable — **35 distinct `wgpu*` calls**:

```
wgpuCreateInstance, wgpuInstanceRequestAdapter[F], wgpuInstanceWaitAny,
wgpuAdapterGetLimits, wgpuAdapterRequestDevice[F],
wgpuDeviceCreateBuffer, wgpuDeviceCreateShaderModule, wgpuDeviceCreateBindGroupLayout,
wgpuDeviceCreatePipelineLayout, wgpuDeviceCreateComputePipeline,
wgpuDeviceCreateBindGroup, wgpuDeviceCreateCommandEncoder, wgpuDeviceGetQueue,
wgpuDeviceDestroy, wgpuShaderModuleGetCompilationInfo,
wgpuQueueWriteBuffer, wgpuQueueSubmit, wgpuQueueOnSubmittedWorkDone[F],
wgpuCommandEncoderBeginComputePass, wgpuCommandEncoderCopyBufferToBuffer,
wgpuCommandEncoderFinish,
wgpuComputePassEncoderSetPipeline, wgpuComputePassEncoderSetBindGroup,
wgpuComputePassEncoderDispatchWorkgroups, wgpuComputePassEncoderEnd,
wgpuBufferMapAsync[F], wgpuBufferGetConstMappedRange, wgpuBufferUnmap,
wgpuBufferGetSize, wgpuBufferDestroy
```

Three ways to close the gap, in increasing order of work:

* **H1 - reimplement the host in Rust.** Requires the *launch plan*: which kernels
  run, in what order, with what grid and what buffers. **This is not in the shader
  metadata** (see below). Cheapest to implement, most work to extract.
* **H2 - C-ABI shim over `wgpu`.** Implement those 35 functions in Rust over
  `wgpu-rs`, expose them with `cbindgen`, provide our own `webgpu/webgpu.h`, then
  compile Futhark's generated C (plus the Futhark RTS C) with plain `cc` and link.
  ~700-1,000 lines of Rust. **Reuses 100 % of Futhark's host logic** (allocation,
  aliasing, event loop, `futhark_context_*` API). Needs the C toolchain, not Emscripten.
* **H3 - Dawn's C API.** Build Dawn, compile the generated C with `-DUSE_DAWN`.
  No shim to write, but a heavy C++ dependency and it abandons `wgpu`.

**Recommendation: H2**, because it keeps our existing `wgpu`/Metal path and because
Futhark's host program is the part we should *not* reimplement.

**The grid problem (the hard part of A).** The dispatch dimensions are computed on the
host from the array shapes, at run time:

* `Futhark.CodeGen.ImpGen.GPU`: `num_tblocks = max 1 (min (ceil (w / tblock_size)) max_num_tblocks)`
  (`max_num_tblocks` is a per-entry-point size-class tuning parameter).
* `Futhark.CodeGen.Backends.GPU:102`: `(grid_x, grid_y, grid_z) <- mkDims <$> mapM GC.compileExp num_tblocks`
  — compiled into the generated C as expressions.
* `rts/c/backends/webgpu.h:1076`: `wgpuComputePassEncoderDispatchWorkgroups(pass_encoder, grid[0], grid[1], grid[2])`.

The workgroup size, by contrast, **is** in the metadata — but it arrives as WGSL
**pipeline-overridable constants** (`@workgroup_size(block_size_x, block_size_y,
block_size_z)` where those names are `override` declarations), set per launch from
the host. `wgpu` resolves those through naga (`ProgrammableStageDescriptor.constants`),
so each distinct block-size set costs one more pipeline creation. Futhark mirrors this
with a `static_pipeline` flag for the static case.

**Concrete portability risks to Metal, and how bad they are:**

| Risk | Detail | Severity |
|---|---|---|
| `enable f16;` | Unconditional in the WGSL prelude (`RTS/WGSL.hs:59`) | Requires `Features::SHADER_F16`; Apple Silicon (Metal 3) has it, but it becomes a hard device requirement |
| `alias i8 = i32; alias i16 = i32; alias i64 = vec2<i32>;` | `i8/i16/i64` are WGSL **reserved words**; Firefox/Tint rejects them (Andreasen §4.4). `i64` is also in naga's `BUILTIN_IDENTIFIER` list | **Must be tested**; a one-line prelude patch fixes it |
| Storage-buffer count | Dawn caps `maxStorageBuffersPerShaderStage` at 10 (Chromium-specific); Firefox allows 64 | Caps the arity of any single Futhark kernel |
| Atomics | Only `atomic<i32>`; 8/16-bit emulated by bit-shifting; 64-bit impossible under WebGPU | Limits the primitive set |
| Workgroup memory | Only in the transpose built-ins, statically sized | Not a problem |
| Bind group layout | Explicit: one bind group, binding 0 uniform (scalars), the rest storage | Well-defined; a Rust host must rebuild it exactly |
| Subgroups / textures / push constants / timestamp queries | None used | No problem |
| Backend restrictions | No memory parameters in GPU functions; no GPU functions that use arrays; no functions that may fail | Constrains which Futhark constructs are expressible |

**Can we demonstrate one Futhark vector operation on the Mac's GPU without the
browser runtime?** Yes, and it is the recommended first experiment — but it needs the
PR branch built (Haskell toolchain), because no released Futhark emits WGSL.

**However, the WGSL-compatibility risks can be killed today, without Futhark.** The
PR's actual WGSL runtime files (`rts/wgsl/*.wgsl`: `scalar{,8,16,32,64}.wgsl`,
`atomics.wgsl`, `lmad_copy.wgsl`, `map_transpose*.wgsl`) are in the diff. We can
assemble the exact prelude the backend emits, run it through our existing naga
front end, and execute it through our existing wgpu/Metal harness. That tests
`enable f16`, the `i8/i16/i64` aliases, `atomic<i32>`, static `var<workgroup>`,
`workgroupBarrier`, `override` constants and the explicit bind group layout — i.e.
the entire portability question — with zero new dependencies.

---

## 4. Question B: can we create a Futhark-inspired kingdom?

**Futhark is a usable library.** `futhark.cabal` has a `library` stanza
(`hs-source-dirs: src`, ~400 exposed modules) exposing, among others:

* `Language.Futhark`, `Language.Futhark.Core`, `Language.Futhark.Parser` — the
  **source AST and its parser**;
* `Language.Futhark.Interpreter` — a CPU interpreter of Futhark programs;
* `Futhark.Construct` — a documented **AST-building DSL** (`MonadBuilder`/`BuilderT`)
  over the core IR;
* `Futhark.IR.Prop`, `Futhark.Internalise`, `Futhark.Passes`, `Futhark.Actions`,
  `Futhark.Compiler`, `Futhark.Compiler.CLI`, `Futhark.CodeGen.Backends.GenericC`.

The PR additionally exposes `Futhark.CodeGen.Backends.CWebGPU`,
`Futhark.CodeGen.ImpGen.WebGPU`, `Futhark.CodeGen.ImpCode.WebGPU`,
`Futhark.CodeGen.RTS.{WebGPU,WGSL}`.

Note: there is **no** `language-futhark` package (404 on Hackage and in the repo) —
the AST lives in the main `futhark` package. There is **no JSON/S-expression AST
dump**, but there **is** a textual IR format with parsers: `Futhark.IR.Parse` exposes
`parseSOACS`, `parseGPU`, `parseGPUMem`, ... keyed on extensions `.fut_soacs`,
`.fut_gpu_mem`, etc., which `futhark dev` dispatches on.

**Trade-off of the three routes:**

| Route | Genetic representation stays native Karva? | Cost | Robustness |
|---|---|---|---|
| (i) Emit Futhark **source text** from the expressed AST, call the `futhark` binary | yes — genes are Karva; text is a rendering | low | **High.** Source text is the only stable cross-language contract. The genetic representation is untouched; the emitter is a pure function of the AST. |
| (ii) Build the **Haskell AST** (`Language.Futhark.Core` or `Futhark.Construct`) and call `CWebGPU.compileProg` from a Haskell shim | yes | medium (Haskell shim, ~300-600 lines, plus a Haskell toolchain) | Medium: tied to the PR's internal API, which is a draft PR and has already bit-rotted once. |
| (iii) Generate the **core IR text** (`.fut_gpu_mem`) and feed `futhark dev` | yes | low-medium | Low for now: the PR's `futhark dev` backend list does **not** include `webgpu`, so it will not produce a runnable program. Useful as a research probe only. |

**Recommendation: (i) for the kingdom, with (ii) as an optimisation later.** The
kingdom's AST is ours; rendering it to Futhark source is a ~200-line pure function.
Text is a *rendering* of the gene, not the gene. This also gives a free
correctness oracle: the same AST rendered to Futhark source can be compiled and run
with **released** Futhark's C backend on the CPU today, with no PR branch needed.

The important boundary to hold: **the genetic representation is Karva, and the
Futhark text is regenerated from the expressed AST on every evaluation.** Nothing
downstream may read the text back into the genome.

---

## 5. Question C: the minimum viable symbol table

**The `wgsl_true` explosion has a specific, measurable cause.** From
`kingdoms/wgsl_true/types.tsv` and `functions.tsv`:

* 305 types, 8,841 function rows.
* **112 of 305 types (36.7 %) are pure scaffolding** in the families `stmt.*` (46)
  and `block.*` (66). They encode *statement nesting*: rung (`C/L/O/W/B`) x depth
  (0-5) x fuel stratum (`f0/f1`).
* **5,295 of 8,841 rows (59.9 %) have a scaffolding type as their output.**

So roughly **three fifths of the genetic material of `wgsl_true` buys ways to spell a
statement nest, not ways to compute anything.** That is the space evolution is
searching, and it is why evolution finds syntax and not speed.

**The lesson generalises: do not encode syntactic nesting in the type system.** A
Futhark-style language gets this for free, because `map`/`reduce`/`scan`/`filter`/
`gather`/`scatter` are *value-level* combinators — nesting them is not a legality
question at all. The type system collapses.

**Proposed minimal type set (12 types, ~4 families):**

| Type | Meaning |
|---|---|
| `f32`, `i32`, `u32`, `f64`, `bool` | element types (f64 optional — WGSL has no f64) |
| `f32v`, `i32v`, `u32v` | vectors of 2/4 lanes (optional, later) |
| `acc<T>` | an accumulator type distinct from the element type (needed so `reduce` is not trivially `map`-shaped) |
| `arr<T>` | a 1-D array of `T` (shape carried by the value, not the type) |
| `arr2<T>` | a 2-D array of `T` |
| `idx` | an index (`i32`), distinct from a general `i32` so index arithmetic is legal by construction |

Two dimensions are enough to start. **Shape is a value, not a type** — this is the
single decision that keeps the type count flat. Futhark's own type system makes the
same move with size types, but we can do better: in the Karva kingdom, the array
shape is carried in the *program* (a per-gene shape annotation), so `arr<T>` never
needs to be specialised per length.

**Proposed symbol set (~45-60 rows), all in one or two families:**

*Combinators* (the bulk of the power, each one row per arity/type combination):
`map`, `reduce`, `scan`, `filter`, `gather`, `scatter`, `zip`, `split`, `flatten`,
`length`, `iota`, `replicate`, `singleton`.

*Scalar ops* (small, shared with the existing `expr.rs` vocabulary where possible):
`+ - * /`, `min`, `max`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `pow`, `select`,
`compare`, and the boolean combinators `& | !`.

*Shape/utility:* `row`, `col`, `transpose`, `matmul` (maybe later).

Legality rules that keep it legal by construction (no repair, no refusal):

* `reduce` needs an `acc<T>` accumulator and a combining op `(acc, T) -> acc`.
  A separate accumulator type means the combining op can only be built from symbols
  that accept it — **illegal combinations are unrepresentable, not rejected.**
* `scan` is `map` over a prefix — same shape discipline.
* `filter` needs a predicate over the element type only.
* `gather`/`scatter` need an index array; `gather` of `arr<T>` by `arr<idx>` gives
  `arr<T>`; `scatter` is the dual.
* Element type of `map` is determined by the function's output type, so `map` needs
  one row per `(in_elem, out_elem)` pair — this is where the row count goes, and it
  is *semantic* row count, not scaffolding.

**Estimate: ~12 types, ~50-80 rows.** That is a **~100x reduction** in symbol rows
versus `wgsl_true`'s 8,841, with vastly more semantic density per row. This is the
quantitative case for the hypothesis, and it is the thing the kingdom build will
actually demonstrate.

**Built and measured (`kingdoms/futhark_true`).** The v0 that followed this
estimate came out at **4 types and 38 rows** — and it is *smaller* than the
estimate, because two of the proposed types turned out to be unnecessary and one
turned out to be necessary:

* `arr.f32` and `scn.f32` are the two array flavours. `scn.f32` ("a scan
  result") is the one type that earns its place: it makes **scan-over-scan
  unrepresentable**, and Futhark 0.28.1 has an internal compiler error on
  `scan (+) 0.0 (scan (+) 1.0 xs)` that ~5 % of random genes otherwise hit.
  `map` and `map2` preserve their input's flavour, so a scanned array can
  still be mapped, zipped and reduced; `reverse` does not — it drops the
  accumulator — so `rev_s` returns an ordinary array (§10.4).
* `fn.f32_bool` was **dropped**: it existed only for `filter`, and `filter`
  cannot be in v0 (below).
* `i32.int`, `u32.idx` and `bool.flag` were **dropped**: one element type is
  enough for v0.

The rows are fewer than the first build's 42 because the numerical validation
found that Futhark's `reduce` and `scan` need a *true* neutral element, which
rules out a free neutral and rules out `reduce f32.max`/`reduce f32.min`
altogether (§10.3).

**Why `filter` is not in v0.** Futhark's `filter` returns an array of *unknown*
size, and Futhark cannot zip an array of unknown size with one of known size.
A kingdom offering both `filter` and `map2` can therefore express programs
Futhark rejects — a hole in "legal by construction", which the design forbids.
The fix is a second shape dimension (`uns.f32` alongside `arr.f32`) with
`map2` restricted to the sized flavour: one more type, and the hole closed.

**Measured on the built v0** (60 random genes per root and program size, each
emitted program compiled by the real Futhark 0.28.1):

| root | program size | compiled by `futhark c` |
|---|---:|---|
| `arr.f32` | 2 | 60 / 60 |
| `arr.f32` | 4, 8 | see §10 |
| `scn.f32`, `f32.real` | 2, 4, 8 | see §10 |

---

## 6. Question D: can compilation and fitness scale?

**This is the risk that decides the project.** Separating the two costs:

* **GPU execution cost**: unchanged in character from `wgsl_true` — dispatch a
  kernel over N elements. Already solved (`src/gpu_eval.rs`), and already fast enough
  that it is not the bottleneck (ST9: 85,437 evaluations in a run whose stated
  limitation is *learning* throughput, not GPU throughput).
* **Compilation cost**: **completely different in character.** In `wgsl_true`,
  "compiling" a genome means naga-validating a ~600-line WGSL text — milliseconds.
  In the Futhark route it means running Futhark's full pipeline: parse, type check,
  size/dimensionality analysis, the whole optimisation stack, then WGSL codegen.
  That is **seconds**, and it is a full Haskell-compiled compiler doing real work.

At ST9 scale that is ~7,700-8,500 Futhark compiles per run, i.e. **2-9 hours of pure
compilation per run** before a single fitness evaluation happens.

**Mitigations, in order of preference:**

1. **Cache on the expressed AST, not on the text.** A content-addressed cache keyed by
   the canonicalised expressed AST (the same key Fuller already computes for
   share-equivalence) rather than by rendered source. The compact language should
   produce a *much* higher hit rate than 10 %: there are far fewer distinct programs,
   and evolution converges.
2. **Two-tier evaluation.** Screen candidates with a cheap evaluator and compile only
   the survivors. `Language.Futhark.Interpreter` exists in the Futhark library (CPU),
   and Phylu's island/HFF selection already produces exactly this ranking. Screen
   on CPU, compile the top-k, evaluate those on GPU.
3. **Kernel-library architecture (H2').** Pre-compile a *fixed* library of Futhark
   kernels (one per primitive) once, and let the genome be a **composition plan**
   over that library, dispatched by a small Rust driver. Per-genome cost collapses to
   a dispatch. The cost is that Futhark's optimiser sees each primitive in isolation
   rather than the whole program — so this is Arch-1 with the fusion benefit removed.
4. **Kernel specialisation.** Futhark specialises on size classes; our kingdom
   should fix a small number of size classes (e.g. `n < 1024`, `n < 1M`, `n >= 1M`)
   so the number of distinct compilations per distinct program is small and stable.

**Grouping structurally related programs** is where Fuller earns its place (see E):
programs that share a subtree should share a compiled component. Futhark's own
fusion pass does this at the WGSL level; doing it at the AST level lets us reuse
*compiled artefacts* instead of re-running the compiler.

**GPU-resident fitness evaluation is already solved** (`phylu/src/evolve/hff_gpu.rs`:
at 200,000 population / 9,000,000 angles, 238 s of host arithmetic versus 1.49 s on
the GPU). That is not the constraint.

---

## 7. Question E: how can Fuller contribute?

Fuller's share-equivalence and DAG machinery has four natural insertion points, in
descending order of value:

1. **On the Futhark-style AST (best).** The expressed AST is the level at which
   *the genetic representation lives*. Finding that two subtrees are semantically
   equivalent lets us (a) **cache compilations per shared subtree**, which is the
   direct answer to question D; and (b) emit a **DAG-shaped Futhark program** —
   Futhark itself has no let-binding of computed values across the whole program in
   the way we would want, but `let` bindings in Futhark source are exactly that, and
   `Futhark.Passes` includes CSE. A shared definition becomes a `let` in the emitted
   source, and the genetic representation stays a DAG.
2. **On the Karva-expressed AST.** Already supported: `src/geneframe.rs` (typed
   ADFs, homeotic rows) and `src/population_dag.rs` (hash-consed nodes, memos keyed
   by node + data version + type + input binding, level schedule) are built for
   exactly this. The new kingdom should reuse them rather than invent a parallel
   mechanism.
3. **On the generated WGSL.** Possible (the `wgsl_true` kingdom already round-trips
   WGSL through naga), but the transformations would have to be *heritable* — they
   would need a corresponding Karva-level operator, otherwise they are not part of
   the evolutionary system, just a post-processing filter.
4. **On the compiler IR.** Futhark's own `Futhark.IR.Prop.Aliases` and the CSE /
   `Futhark.IR.Prop` passes already do this. We would be reimplementing the
   compiler. **Not worth it.**

The rule to hold: **every Fuller transformation must have a Karva-level counterpart
that is itself an operator on the genome.** Otherwise it is not evolution, it is
post-processing, and the result is not heritable.

---

## 8. Assessment summary

**Is the architecture feasible?** Yes. The pieces exist:

* Futhark's WGSL backend is real, is in the upstream repo on a branch, and does the
  hard optimisation work (fusion, vectorisation, tiling, size/dimensionality
  polymorphism) that a GEP cannot discover.
* The kernel metadata we need (entry points, bind slots, scalars layout, workgroup
  size, WGSL source) exists **as typed data inside the compiler**, not just as
  rendered C.
* Futhark is a **library**, so the AST can be handed to the compiler without a
  source-text round trip (though source text is the better near-term contract).
* Our side already has the naga front end, the Metal runtime, the declarative
  kernel contract, and the GPU-resident fitness loop.

**What we'd actually need to build:**

| # | Component | Size | Where |
|---|---|---|---|
| 1 | The kingdom: compact type set, symbol table, `gen_tables.py`, `interface.py` | ~12 types, ~60 rows | `kingdoms/futhark_true/` |
| 2 | The AST + gene -> AST -> Futhark source renderer (Rust) | ~500-800 lines | `src/futhark_true/` |
| 3 | A Futhark toolchain, including the PR branch | hours of build time | Docker + GHC/cabal |
| 4 | A JSON manifest emitter patched into `CWebGPU.hs` (per-kernel metadata + WGSL files) | ~150 lines Haskell | PR branch |
| 5 | The launch plan (kernel order, grid expressions, buffer allocation) | ~300-500 lines Haskell | PR branch, `Backends/GPU.hs` |
| 6 | The Rust/wgpu host (H2: C-ABI shim, 35 functions, or H1: launch-plan interpreter) | ~700-1,200 lines | `src/futhark_true/` |
| 7 | Fuller integration: AST-level share-equivalence feeding the compile cache | ~300 lines | `src/` |

**The most significant technical risk: per-genome compilation throughput (question D).**
It is the only risk that scales with population size, it is not fixable by a local
patch, and it is the difference between "one program running" and "evolution running".
The WGSL-on-Metal risks (question A) are bounded and testable; the Haskell-toolchain
cost is an engineering expense, not a risk to the idea. Mitigation is the
two-tier evaluation and the AST-keyed compile cache — and note that both only pay
off *because* the language is compact, which is the same property the hypothesis is
about.

**The smallest end-to-end proof of concept:**

1. Assemble the **exact WGSL prelude** the backend emits (`rts/wgsl/*.wgsl` from the
   PR) and validate it with our in-tree naga front end. *Kills the `enable f16`,
   `i8/i16/i64` alias and `override`-constant risks in minutes, with no toolchain.*
2. Hand-write a **single Futhark-shaped WGSL kernel** in the backend's style (one
   `@compute @workgroup_size(...)` entry point, one uniform scalars struct, one
   storage buffer) and execute it through the existing `src/wgsl/harness.rs` on
   Metal. *Proves wgpu-native runs Futhark-style WGSL at all.*
3. Emit **one** program from the new kingdom to Futhark source, compile it with
   **released** Futhark's C backend, and check it runs and matches a reference.
   *Proves the kingdom's semantics and the source emitter, with no PR branch needed.*
4. Only then: build the PR branch, extract WGSL + manifest for the same program,
   and check the Metal result matches the CPU result. *Closes the loop.*

Steps 1-3 are all reachable with **no Haskell toolchain** and they de-risk 90 % of
the project. Step 4 is the one that needs the long build, so it should be started in
the background immediately and in parallel.

**Recommended first implementation step:**

> **Build the kingdom v0 — the compact typed symbol table plus the gene -> AST ->
> Futhark source renderer — and validate it against released Futhark on the CPU.**
>
> It is the component needed under every option in this document; it needs only a
> ~20 MB toolchain install rather than a 10 GB Haskell build; it produces the AST
> that Fuller will operate on; and it directly tests the scientific hypothesis at
> the language-design level — *can a compact, legal-by-construction Karva symbol
> table express a useful family of parallel programs that Futhark accepts and that
> compute the right answers?*

In parallel, in the background: install released Futhark 0.28.1 (Homebrew) and start
the Docker build of the PR branch, and run the naga/Metal portability probe (step 1
above).

---

## 9. Sources

**Primary**

* PR: <https://github.com/diku-dk/futhark/pull/2140> — head sha `14e9ccbf3a47c06ed585d15e6827d186dd74d8a0`
* Upstream `futhark.cabal` (library stanza): <https://raw.githubusercontent.com/diku-dk/futhark/master/futhark.cabal>
* Upstream `Futhark.CodeGen.Backends.GPU` (grid computation): <https://raw.githubusercontent.com/diku-dk/futhark/master/src/Futhark/CodeGen/Backends/GPU.hs>

**Theses**

* S. Paarmann, *A WebGPU backend for Futhark*, MSc thesis, DIKU, 2024:
  <https://futhark-lang.org/student-projects/sebastian-msc-thesis.pdf>
  - "we have focused on the web use case, leaving potential native WebGPU targets to
    future work."
  - "We initially considered adding functionality for compiling to a native
    executable with the WebGPU backend using one of the native implementations of
    the WebGPU API -- Dawn and wgpu -- instead of compiling using Emscripten...
    However we ultimately decided that since the main target for this backend is the
    web, the testing should ideally reflect that environment."
  - Limitations: no f64; i64 division missing; no in-kernel error handling; NaN/inf
    semantics diverge; atomics only 32-bit; no memory fences; shared memory
    uniform-load only.
* C. Andreasen, *Pushing the Boulder: Addressing key limitations of the Futhark
  WebGPU backend*, BSc thesis, DIKU, 2025:
  <https://futhark-lang.org/student-projects/caleb-bsc-thesis.pdf>
  - The branch had bit-rotted: "the backend had fallen out of date with the rest of
    the Futhark compiler, rendering it unable to generate any functional programs."
  - "on Dawn ... the highest limit supported is 10 storage buffers per shader stage.
    This is a somewhat arbitrary limit unique to Chromium."
  - Firefox is broken: "their WGSL compiler has reserved the i8, i16 and i64 keywords
    that the Futhark WGSL runtime uses as type aliases."
  - Performance: WebGPU ~4x slower than CUDA overall, 2-6x faster on large `fluid`
    and `LocVolCalib`.

**Local**

* `kingdoms/wgsl_true/{types,functions}.tsv` — 305 types, 8,841 rows; 112 scaffolding
  types; 5,295 rows with a scaffolding output type.
* `docs/reports/2026-10-22_ST9_volumepassage_champion/README.md:91` — 85,437
  fitness evaluations, 0 device failures, 10 % pipeline cache hit rate.
* `phylu/src/evolve/hff_gpu.rs` — GPU-resident HFF (238 s host vs 1.49 s GPU at
  200k population / 9M angles).
* `src/typed_gep.rs`, `src/kingdom_contract.rs`, `src/geneframe.rs`,
  `src/population_dag.rs` — the typed GEP and DAG machinery.
* `problems/*/problem.toml` — the declarative kernel contract used by the harness.

**Not verified** (stated honestly)

* No Futhark build was performed; no WGSL was validated by naga; no Futhark-generated
  kernel was executed on Metal. Every Futhark claim above is read from source, not
  from a run.
* The exact contents of `wgpu_kernel_infos[]` initialisers (whether every array
  binding is covered without gaps) require a built artefact.
* The exact `futhark_<entry>_<arg>` symbol names per entry point were not enumerated
  (the `futhark_` prefix and the manifest's `c_fun` field were confirmed).
* Whether `wgpu` on Metal accepts Futhark's `override`-based workgroup sizes was not
  tested; wgpu resolves them through naga, which suggests it will, but that is
  inference, not measurement.

---

## 10. What was measured after the assessment

The assessment above was written from the sources. This section is what was
then *measured*, on this machine (Apple M3 Max, Metal), with the real tools.

### 10.1 The portability probe

`examples/futhark_wgsl_probe.rs` (new) takes WGSL files and reports three
verdicts: naga's front end, naga's validator plus this device's limits, and
the driver (shader module + compute pipeline, with and without pipeline
constants). `tools/port_futhark_wgsl.py` (new) rewrites Futhark's WGSL runtime
for naga and prints exactly what it removed.

The PR's WGSL runtime (`rts/wgsl/*.wgsl`, taken from the PR diff) was
assembled into the exact prelude the backend emits
(`"enable f16;\n" ++ scalar ++ scalar32 ++ scalar8 ++ scalar16 ++ scalar64 ++
atomics`) and put through the probe. The device reported:

```
device: Apple M3 Max (Metal)
features: SHADER_F16=true
limits: max_storage_buffers_per_stage=29  max_compute_invocations_per_workgroup=1024
```

**The Metal side is good news:** 29 storage buffers per stage (against Dawn's
Chromium-specific cap of 10), and `SHADER_F16` is present. Neither of the
hazards from the theses bites here.

**The naga side is where the work is.** The prelude as emitted does not
validate. Two distinct defects, both real, both in Futhark's emitted WGSL:

1. **`bitcast` between a scalar and a vector.** naga lowers `bitcast<T>(e)` to
   `Expression::As { convert: None }`, which needs the source and destination
   to have the same *number of components*; the WGSL spec only requires the
   same total bit width. So `bitcast<vec2<f16>>(a: i32)` — legal WGSL, used by
   Futhark's f16 bit-twiddling helpers — is rejected. naga's `pack2x16float`
   is `VEC2 of F32 -> U32`, so there is no drop-in f16 replacement either.
2. **Pointer arguments outside the `function`/`private` address spaces.**
   Futhark's runtime helpers take `ptr<storage, ...>` and
   `ptr<workgroup, ...>` arguments — the shape OpenCL C allows. naga rejects
   them (`naga/src/valid/function.rs:1843` allows only `private` and
   `function`), and so does the WGSL spec: *"For user-defined functions, a
   parameter of pointer type must be in one of the following address spaces:
   function, private"* (gpuweb/gpuweb#5032). So this is **Futhark emitting
   invalid WGSL**, not a naga limitation — which means the PR's own browser
   path cannot be exercising those helpers.

`tools/port_futhark_wgsl.py` removes the 13 functions that use the f16
bitcast and the 80 that take a storage or workgroup pointer. The remaining
runtime then **passes naga's front end and validator**: 238 functions, no
entry points, no errors.

The cost is real and worth stating: the removed functions are Futhark's
**atomic helpers** (every `atomic_*_global`/`atomic_*_shared` for i8/i16/i32)
and the f16 bit-level helpers. A kingdom that needs atomics or f16 bit tricks
needs those helpers rewritten, not deleted.

### 10.2 The kingdom

Built in this session, and it works:

* `kingdoms/futhark_true/` — `gen_tables.py` (the design as code), the
  generated `types.tsv` and `functions.tsv`, `README.md` (the typing rules).
* `src/futhark_kingdom.rs` — the loader, the kingdom's relations, and the
  renderer from an expressed tree to Futhark source.
* `src/futhark_interp.rs` — the kingdom's **own interpreter**: what a gene
  *means*, independent of Futhark.
* `src/futhark_exec.rs` — compile the rendered source with the real Futhark
  compiler, link it with `cc`, run it, and read the values back.
* `Ty::Futhark(u16)` — one new variant in `src/geneframe.rs`, the same
  table-defined-family pattern as `Ty::Wgsl` and `Ty::Tsr`.
* `examples/futhark_true_emit.rs` — random genes in, Futhark source out, the
  real Futhark compiler's verdict, and the interpreter's value against the
  compiled program's, element for element.

Measured:

* **4 types, 38 rows** (34 functions, 4 terminals), against `wgsl_true`'s
  305 types and 8,841 rows. **0 rows are syntactic scaffolding.**
* The kingdom passes its own contract check (one output type per symbol,
  ordered contiguous signatures, a terminal for every demanded type).
* Every random gene validates and expresses: **no repair, no refusal, no
  substitution.**
* Every emitted program compiles with the real Futhark 0.28.1.
* **Every emitted program agrees with the interpreter, element for element.**
  The sweep: 12 random genes for each of 3 roots × 4 program sizes (2…5) —
  144 programs, each compiled with the real Futhark, linked with `cc`, run,
  and compared with the interpreter on a fixed 16-element input.

  | | |
  |---|---:|
  | programs compiled, run and **in agreement with the interpreter** | **143 / 144** |
  | legal programs the compiler does not converge on | 1 / 144 (0.7 %) |

  The one that does not converge is the pathology of §10.6. It is legal and
  well-defined; Futhark's optimiser simply does not finish on it, so the
  compile is bounded and reported as a failed compile.

  The one thing that had to be *designed out* rather than coded out: Futhark's
  `filter` cannot coexist with `map2` in a legal-by-construction kingdom, and
  Futhark 0.28.1's scan-over-scan internal compiler error had to be made
  unrepresentable. Both are documented in the kingdom's README.

### 10.3 Futhark's neutral-element precondition, and what it cost

The numerical validation found a class of bug that the *compilation* check
could never have found. Futhark's manual says:

> In an expression `reduce f ne xs`, the function `f` must be associative and
> have `ne` as neutral element. […] If we pass `reduce` a function that is not
> associative, or does not have the provided neutral element, **we will get
> wrong results at run-time**. What's worse, the compiler will not be able to
> detect that we messed up.

The kingdom's first version let the gene choose the neutral, which is exactly
the undefined case. Measured on Futhark 0.28.1, over `[1,2,3,4]`:

| expression | right answer | Futhark 0.28.1 |
|---|---:|---:|
| `scan (+) 0.5 xs` | `[1.5, 3.5, 6.5, 10.5]` | `[1.5, 3.5, 6.5, 10.5]` ✔ |
| `reduce (+) 0.0 (scan (+) 0.5 xs)` | 22.0 | **27.0** ✘ |
| `reduce (+) 0.0 (scan (+) 0.0 xs)` | 20.0 | 20.0 ✔ |
| `reduce (+) 5.0 (scan (+) 0.0 xs)` | 25.0 | **45.0** ✘ |
| `reduce (+) 5.0 xs` | 15.0 | 15.0 ✔ |
| `foldl (+) 5.0 (scan (+) 0.0 xs)` | 25.0 | 25.0 ✔ |

Three things to read out of that table:

* the excess is the neutral counted **once per element** instead of once —
  `ne·n(n+1)/2` for a scan's neutral, `ne·n` for a reduce's;
* it only bites over an **`acc`** array (a scan result). Over a plain array
  the reduce happens to be right, because Futhark's parallel reduce only uses
  the neutral for an empty chunk;
* it **poisons its siblings**: in a program holding both
  `last (scan (+) 0.5 xs)` and a reduce over that scan, `last` returns 12.5
  where 10.5 is right. The fusion of the two is what goes wrong.

The fix is the identity neutrals, and it is not a workaround — it is
Futhark's precondition. `reduce (+) 0.0`, `reduce (*) 1.0`, `scan (+) 0.0`,
`scan (*) 1.0`: exact at every size from 1 to 9. The cost is one degree of
freedom per reduce and per scan, which was worth little anyway (the neutral
of a prefix sum is 0 and of a prefix product 1), and the loss of the four
`reduce f32.max`/`reduce f32.min` rows: max's identity is −∞, Futhark has no
infinity literal (`f32.infinity` is an unknown name), so a max reduction has
no neutral element that is both exact and defined. The elementwise max and
min survive as `map2`, which takes no neutral at all.

The general lesson, and it is the one that matters for a kingdom: **a row's
type is not enough — the row's *preconditions* have to be in the type too.**

### 10.4 Futhark's `acc` type, and the hole `reverse` makes

The two array flavours are not a stylistic choice: Futhark distinguishes a
scan result (an `acc` array) from an ordinary array, and the distinction is
exactly where the neutral bug lives. Measured, by looking for the reduce's
excess:

| expression | result is an `acc` array? |
|---|---|
| `scan f xs` | yes |
| `map f (scan f xs)` | yes |
| `map2 f (scan f xs) (scan f xs)` | yes |
| `map2 f (scan f xs) xs` | yes |
| `reverse (scan f xs)` | **no** |
| `map f xs` | no |

`reverse` is the one combinator that *drops* the accumulator. So `rev_s` is
typed `scn.f32 → arr.f32`, not `scn.f32 → scn.f32`: the kingdom's types track
Futhark's `acc` exactly, so a scan result can be mapped, zipped and reduced,
but reversing it gives back an ordinary array. The first version of the table
said `scn.f32 → scn.f32` and was wrong.

### 10.5 One primitive: `f32.round`

The other thing the numerical validation found. Futhark's `f32.round` rounds
a half to the nearest **even** — it is `rintf`, and WGSL's `round` is
ties-to-even too. Rust's `f32::round` rounds a half **away from zero**. On
`round(2.5)` that is a whole unit, and it propagates:

```
map2 (*) (map f32.round (scan (+) 0.0 xs)) (scan (+) 0.0 xs)   then   f32.tan
```

gives −8.451288223266602 under ties-to-even and 6.7650346755981445 under
away-from-zero — and Futhark says −8.451288223266602, so Futhark rounds
halves to even. The interpreter now uses `round_ties_even`, and the whole
sweep agrees again.

The lesson is the same as §10.3's: *every* primitive has to be checked
against the real compiler, not just the ones that look exotic.

### 10.6 The compiler does not always converge

The validation sweep found something worse than a wrong answer: a legal program
that `futhark c` never finishes. The sweep hung on it twice, at
`f32.real`/size 4, and the reproducer is stable:

```futhark
entry main (xs: []f32) : f32 =
  reduce (*) 1.0
    (map2 f32.max
      (map2 f32.min (reverse (xs))
        (map (\x -> x - (reduce (+) 0.0 (map (\x -> x) (scan (+) 0.0 xs)))) (xs)))
      (reverse (map (\x -> f32.round x)
        (map2 (*) (map (\x -> x) (scan (+) 0.0 xs)) (scan (*) 1.0 (xs))))))
```

Killed at 300 s with no output. Narrowing it down, the trigger needs all of:

| change | result |
|---|---|
| the program as above | **never finishes** |
| the inner `reduce` over a *plain* array instead of a scan | 0.1 s |
| the inner `reduce` removed | 0.1 s |
| the second `map2` operand not reversed | 0.1 s |
| the second operand a plain reversed array | 0.1 s |
| the outer operator `+` or `f32.max` instead of `(*)` | never finishes |
| the middle operator `f32.max` instead of `f32.min` | never finishes |

So it is an interaction: **a reduce over a scan result, inside a `map`, inside
a `map2`, combined with a reversed scan-derived array, under an outer
reduce.** All of it legal, all of it well-defined (§10.3's identity neutrals
are in force), and the compiler does not converge on it.

Two consequences.

* **"Legal by construction" is a statement about the language, not about the
  compiler.** The kingdom guarantees that every gene denotes a well-defined
  program. It cannot guarantee that Futhark's optimiser terminates on every
  such program — that is a property of Futhark, not of the typing. So the
  fitness harness needs a **wall-clock bound on compilation**, and a program
  that exceeds it is a *failed compile*, not a failed gene. `src/futhark_exec.rs`
  now enforces one (120 s), and `examples/futhark_true_emit.rs` reports a
  non-converging program as its own outcome rather than as an error.
* **This sharpens the compile-throughput risk.** §6 said a Futhark compile
  costs seconds against a naga validation's milliseconds, and estimated hours
  of compilation per ST9-scale run. The measured reality is worse than
  "seconds": *some* programs do not finish at all, so the per-compile cost
  distribution has an unbounded tail. The mitigations of §6 (an AST-keyed
  cache, a two-tier screen, a fixed kernel library) are therefore not
  optimisations — they are **what makes the route viable at all**. A program
  that never compiles cannot be cached away either, which is an argument for
  the fixed-kernel-library architecture.

This is also a good candidate for an upstream bug report: the reproducer is
small, stable, and shows a legal Futhark program whose compilation does not
terminate.

**It is not a stale-compiler problem.** Three compilers were tried on the
same program, each with a wall-clock bound and a control program that
compiles in 0.1 s:

| compiler | the control | the program |
|---|---|---|
| Futhark 0.28.1 (the installed one) | 0.1 s | **hangs** (killed at 120 s and at 300 s) |
| Futhark 0.29.0-prerelease, upstream master, 2026-10-11 (the `nightly` release) | 0.1 s | **hangs** (120 s) |
| Futhark 0.29.0-prerelease, the `webgpu` branch at `14e9ccbf`, 2026-10-22 (its own CI's build) | 0.1 s | **hangs** (120 s) |

The three reduced programs of the table above (the inner reduce over a plain
array, the inner reduce removed, the second operand not reversed) compile in
0.1 s on the branch's compiler too, so the *same* conjunction is the trigger
in the newest compiler. And the reason the branch does not fix it is visible
in the diff: between the nightly's tree and the branch head, the only
changes under `src/Futhark/Optimise/` are a mechanical
`TraverseOpStms` → `TraverseOpBodies` plumbing refactor, one *narrowing*
guard added to an `Update` rule the program does not use, and changes to
`InliningDeadFun.hs` and `ArrayCoalescing.hs` that mention nothing about
`acc`, scan, reduce or reverse. The optimiser's rules are unchanged.

So the defect is live in upstream master and in the experimental branch: a
report upstream now has three independent compilers reproducing it. Fuller
cannot fix it by rewriting the program — no equality-preserving rewrite of the
program changes the outcome, measured in `docs/RESCUE_futhark.md` §6.1 — but
it can *work around* it: the hang is in the SOAC fusion pass, whose fixpoint
loop is unbounded, and the same pipeline with that loop capped compiles the
program in 0.15 s (`docs/RESCUE_futhark.md` §7).

### 10.7 The real WGSL, through the probe

The PR branch (Futhark 0.29.0 prerelease, head `14e9ccb`) was built for this
probe in a podman VM (`haskell:9.12-slim-bookworm`, GHC 9.12.4). That VM is
not on the machine any more — `podman machine list` reports
`podman-machine-default` with `LastUp: 0001-01-01` and no disk — but the
compiler does not need rebuilding: **the branch's own CI publishes a macOS
arm64 build as an artifact** on every run
(`futhark-nightly-macos-arm64.tar.xz`, 33.6 MB), and one command fetches it:

```bash
gh run download 39795721627 -R diku-dk/futhark -n futhark-nightly-macos-arm64
```

(run `39795721627` is the branch head `14e9ccb`; the binary reports
`Futhark 0.29.0 (prerelease)` and its command list contains `webgpu`, which
master does not have). `futhark webgpu w.fut
--library -o /tmp/wgpuout` produces real artefacts — `wgpuout.c` (1.7 MB),
`wgpuout.json` (720 B), `wgpuout.wrapper.js` (12 KB) — before the emcc step
fails, so they survive.

The WGSL is not a file of its own: it is C string literals inside `out.c`.
`tools/extract_futhark_wgsl.py` (new) reads them back out — 28 modules, of
which 25 are real WGSL programs (the runtime prelude plus one generated
kernel each), and it prints what it removed. The subtlety is that the WGSL
itself contains `};`, so the array's terminator has to be found by skipping
the string literals; a naive "find the next `};`" truncates the file, and the
truncation shows up as a naga *parse* error rather than a validation one.

Put through the probe, **all 25 modules fail naga's front end**, with the same
error. `tools/bisect_wgsl.py` (new) narrows it to one line:

```wgsl
fn futrts_to_bits16(a: f16) -> i16 { return bitcast<i32>(vec2<f16>(a, 0.0)); }
```

the scalar↔vector `bitcast` of §10.1. This is a **naga limitation, not
invalid WGSL**: naga lowers `bitcast<T>(e)` to `Expression::As { convert:
None }`, which needs the source and destination to have the same number of
components, where the spec only requires the same bit width.

After the port (the 13 f16-bitcast functions and the 80 pointer-taking
helpers removed), the verdict splits:

* **1 of 25** — the kernel for the program we compiled, `gpu_program.wgsl` —
  passes naga's front end and validator, and **Metal builds a compute
  pipeline from it** (57.4 ms) once the workgroup-size `override` is set.
  Futhark emits `override x: i32 = 0;` and lets the host set it per dispatch,
  so the zero default is expected; with `--override
  mainzisegmap_7646_block_size_x=64` the pipeline builds.
* **24 of 25** now fail to *parse*, because the generated kernels **call** the
  helpers the port deleted: `read_i8`, `write_i8`, `read_i16`, `write_i16`,
  `read_i32`, `write_i32`, `read_i64`, `write_i64`. The port is not a
  workaround; it moves the failure.

So the WGSL route's cost is now precise, and it is not a one-line patch:

* the **f16 bitcast** is a naga strictness we could live with by rewriting
  those helpers to use `pack2x16float`/`unpack2x16float` on f32 — a change to
  Futhark's runtime, in the PR;
* the **pointer arguments** are invalid WGSL, and the generated kernels depend
  on them, so they must be **rewritten**: a helper that takes
  `ptr<storage, array<T>, read_write>` has to become one that takes the array
  (or the module-level binding) itself. That is a change to Futhark's WGSL
  backend, and it is the single biggest piece of work the WGSL route needs
  before it can run on wgpu.

Neither is fatal for a compact kingdom that uses plain array operations — the
one kernel we could test end to end is exactly such a kernel, and it works.
But it is the work, and it is in Futhark, not in Fuller.

### 10.8 What the assessment got wrong, and what it got right

**Right:** the architecture is feasible; the metadata is inside the compiler;
the host runtime is the hard part; the C API surface is small (35 functions);
per-genome compilation throughput is the dominant risk; the compact symbol
table is the point.

**Wrong / refined:**

* *The `bitcast` and pointer-argument problems were not in the assessment.*
  They are the real cost of the Futhark WGSL route, and the second one means
  the PR emits invalid WGSL. Neither is fatal for a compact kingdom that uses
  plain array operations — the one kernel we could test end to end is exactly
  such a kernel, and it builds a Metal pipeline — but both must be fixed in
  Futhark before Futhark's atomics, f16 or the copy/transpose kernels are in
  scope.
* *A legal-by-construction kingdom needs the target language's
  **preconditions**, not just its types.* The assessment assumed that a
  well-typed program was a well-defined program. It is not: Futhark's
  `reduce`/`scan` need a true neutral element, and a typed program that
  violates that is undefined (§10.3). The same applies to `f32.round`'s
  tie-breaking (§10.5). This is the single most useful thing the numerical
  validation taught, and it generalises to every kingdom: the type system has
  to carry the semantic preconditions, or the guarantee "every gene is legal
  and means exactly one thing" is false.
* *`futhark webgpu` has no "WGSL only" mode*, but `futhark dev
  --compile-imp-webgpu prog.fut` writes the WGSL to stdout, and `futhark
  webgpu --library` writes `out.c`, `out.wrapper.js` and `out.json` *before*
  the emcc step — so the artefacts are harvestable even when emcc fails. The
  `Pretty Program` instance does not print the kernel metadata map, though, so
  the metadata still has to come from the `wgpu_kernel_infos[]` table in the
  generated C, or from a patch that emits it as JSON. The WGSL itself has to
  come out of the C as string literals (§10.6).
* *The launch plan is the real gap, and it is worse than "a JSON manifest".*
  The grid dimensions are computed on the host at run time
  (`num_tblocks = max 1 (min (ceil (w / tblock_size)) max_num_tblocks)`), and
  the buffer allocation and launch order live in the generated C host program.
  Reimplementing that host in Rust means reimplementing Futhark's allocator
  and event loop. The 35-function C-ABI shim (H2 in §3) is the cheaper path.
* *The compile-throughput risk is real and quantifiable — and its tail is
  unbounded.* A Futhark compile is a whole-program optimisation pipeline:
  seconds, against the milliseconds of a naga validation. Worse, some legal
  programs never compile at all (§10.6), so the per-compile cost distribution
  has no finite bound. At ST9's scale (~8,500 distinct kernels per run) that
  is hours of pure compilation per run at best. The mitigations in §6 stand,
  and the compact kingdom makes them more effective: the number of *distinct*
  programs is far smaller, so the AST-keyed cache hits far more often. But
  they have moved from "optimisation" to "requirement".
* *A `Ty::Futhark(u16)` variant is all a new kingdom needs in the type
  system* — the same table-defined-family pattern `Ty::Wgsl` and `Ty::Tsr`
  already use. The expensive part is not the type system; it is the fitness
  harness and the GPU execution path.

### 10.9 Still open

* **The WGSL helpers have to be rewritten, in Futhark.** The pointer-taking
  runtime helpers are invalid WGSL *and* the generated kernels call them, so
  deleting them is not a fix. Until that is done, the WGSL route can only run
  kernels that do not touch atomics or the copy/transpose paths.
* **The GPU path for the kingdom does not exist.** The renderer produces
  Futhark source; Futhark's released compilers run it on the CPU, and the
  PR-branch WebGPU backend runs on a WebGPU C API (Dawn or Emscripten), not on
  wgpu. The 35-function C-ABI shim over `wgpu-rs` (§3, H2) is the next piece
  of work, and it is the piece that decides whether this kingdom ever runs on
  Metal.
* **The compile-throughput mitigation is untested.** The AST-keyed cache and
  the two-tier screen are designed, not built; a small experiment measuring
  the distinct-program count of a real generation would settle whether the
  risk is as large as §6 says. The compile timeout is in place, but the right
  budget is not: 120 s is a guess, and the *distribution* of compile times
  over a real population is not measured.
* **The non-convergence is worth reporting upstream.** The reproducer in
  §10.6 is small and stable; whether Futhark's optimiser has a bounded
  compile time for legal programs is a question about Futhark that this
  project should not have to answer for itself.
* `filter`, `gather`, `scatter`, `f16` and the integer element types are
  v1, with the shape dimension (`uns.f32`) that `filter` needs.
* The other element types bring their own preconditions with them: an integer
  `reduce` needs an identity in the integer domain, and an f16 `reduce` on a
  GPU needs `SHADER_F16`, which this device has but others may not.
