# PLAN: UltraDim as a data source, Futhark as its procedure language

## Context

**Objective.** Make UltraDim a data source for fuller and phylu:

- High-dimensional data lives in UltraDim, on its GPU.
- Futhark is UltraDim's execution language over that data, as PL/SQL or T-SQL is a database's: a procedure is a
  Futhark program that runs where the data lives.
- Evolution writes those procedures and makes them faster.

**Approach.**

- A collection's schema becomes a Futhark signature, and the signature becomes a kingdom: the signature family of
  `src/futhark_sig.rs`, generated from a schema rather than from a hand-declared interface.
- Observations are drawn from the stored rows, with held-out cases drawn apart from them.
- A procedure runs on UltraDim's own `wgpu::Device` and buffers through the rust-wgpu backend (`futhark-rust-rt`,
  `Array::from_buffer`), with no copy.
- Fitness is staged: exact against UltraDim's existing kernel first, then speed, under the runner standard.

**What is unique.**

- The procedures are typed programs that evolution writes and improves, not hand-written stored procedures.
- Every candidate is legal by its types, the kingdom is made from the data's own schema, and the judge is the
  kernel the database ships today, run on the same buffers.

**Status (2026-10-28).**

- A plan, nothing built for it.
- Its first procedure, the k-means centroid update, is already a benchmark in the WGSL kingdom
  (`docs/kmeans_update_dropin.md`, fuller 4f54a507 and 1e840384).
- Two facts measured there carry over:
  - Metal's compiler removes A7's Kahan compensation: a plain sum gives A7's bits on 8 of 8 cases;
  - a counting-sort scatter with A7's semantics gives A7's bits on 8 of 8 cases.
- The rust-wgpu backend is being built by another agent. Its plan, `docs/PLAN_futhark_rust_wgpu.md`, does not exist
  in the tree yet. Its runtime crate, `futhark-rust-rt/`, is untracked work in progress; this plan reads its source.

**Why.** UltraDim's GPU work today is fixed kernels (76 `@compute` entry points, `GPU_FUNCTIONS_INVENTORY.md`), and
some are known to be slow: A7's gather is O(k·n·d). A procedure language over resident data lets the system write
the next kernel instead of a person, judged against the one it replaces. The data never leaves the device to be
judged.

## Sources

Read in full:

- **qdrant:**
  - `ultradimdb/src/core.rs`;
  - `ultradimdb/src/store/dense_sidecar.rs`;
  - `ultradimdb/src/store/mod.rs`;
  - `ultradimdb/src/gpu_cluster_ops.rs`;
  - A7's kernel and driver (`gpu_kmeans.rs` 654-715, 1457-1528);
  - `GPU_FUNCTIONS_INVENTORY.md`.
- **fuller:**
  - `src/futhark_sig.rs`;
  - the `futhark-rust-rt` sources `lib.rs`, `ctx.rs`, `values.rs`, `mem.rs`, `program.rs` and `host.rs`.

Read in part (named where used):

- **qdrant:**
  - `gpu_kmeans.rs` 196-470 and 860-966 (the device set-up, the cluster loop, the persistent buffer);
  - `projection-voting-kmeans/src/lib.rs` 2060-2400 (A6's counting sort and segmented sum) and `cpu_mean_update`;
  - `holdout_split_ops.rs` 1-300;
  - the heads of `key_sidecar.rs` and `payload_sidecar.rs`.
- **fuller:**
  - `examples/futhark_evolve.rs` (its header, the method, 1-200);
  - `docs/PLAN_futhark_wgpu_host.md` §1-4.5.

Not read, so nothing below rests on them:

- `lib/trellis` (the trellis structures);
- `kmeans_store.rs`, `kmeans_artifact.rs` (stored centroids);
- `csr_sidecar.rs` and `facet_index.rs` beyond their use above.

## 1. A kingdom made from a schema

**What a collection is, as stored.** A UDV23 family (`core.rs`: `CoreService`) holds rows `0..N` (contiguous ids,
contiguity-gated in every store):

| part | where | form |
|---|---|---|
| dense rows | `udv23_dense/{family}/rows.trldense` | n × d_raw f32, unit norm (v23), tombstone bitmap |
| sparse rows | `udv23_csr` (CSR sidecar) | per-row (index, value) pairs |
| per-seed keys | `udv23_keys/{family}/s{seed}.trlkeys` | n × D_proj f32 unit keys, one file a seed |
| payloads | `udv23_payloads/{family}/rows.trlpayl` | one flat JSON map of facets a row |

The canonical seed's keys are what `GpuKmeansCluster` clusters today, with A7's `GpuKMeans` (`gpu_cluster_ops.rs`).

**The mapping: a schema entry becomes a typed input of a signature.** The signature family already does this for
hand-declared interfaces:

- an input is `(name, element, dims)`;
- a dimension is a role and a class is a type (`a.n.d` is `[n][d]`);
- an index is in range by its type;
- the generator writes the tables offline, with witness tests and `--check`.

The new step is a schema reader that writes the instance's declaration from a family's metadata instead of from
`SIG_INSTANCES`:

| schema part | signature input | dims |
|---|---|---|
| dense rows | `rows: [n][d]f32` | n = committed rows, d = d_raw |
| a seed's keys | `keys_s: [n][p]f32` | p = D_proj |
| an integer facet of the payloads (a label, a time bucket) | `facet: [n]i32` | n |
| a procedure's own state (centroids, assignments) | `centroids: [k][d]f32`, `labels: [n]i32` | k a size input |

The output is the procedure's declared result type.

**What the family must gain first.** Each item is a general extension, opt-in per instance, with the existing
instances byte for byte as before:

1. **f32 elements.** The family holds one element type, `i32` (`sig_signature` refuses others). `Array`, the binary
   format, the interpreter and `quality_inputs` are all i32.
2. **Mixed element types in one instance** (f32 rows, i32 labels), with conversions as rows.
3. **Rank-2 outputs** (`sig_signature`: "a rank-2 output is not yet a family row").
4. **Float rows:**
   - division (total in IEEE);
   - `hist` and `reduce_by_index` on f32;
   - a vectorised `hist` over rows (`hist (map2 (+))`), the scatter form;
   - an equality select on labels.
5. **Context budget.** The depth is global (`SIG_DEPTH = 2`), and every loop binder over every state multiplies the
   contexts. The centroid update's gather needs depth 3 (`tab k` / `tab d` / `sum (tab n)`). The family therefore
   needs a per-instance depth and a per-instance binder set; a tabulate-only set at depth 3 is 40 contexts at three
   dimensions.
6. **A float tolerance as the exactness rule** (§5), in place of `compare`'s integer equality.

**Not yet covered.** Trellis structures and stored centroids are not mapped here: their code has not been read
(Sources).

## 2. Observations and held-out cases from the stored data

**A case.** A case is a slice of the family's rows: an id range or a seeded sample. The case's inputs are the
procedure's inputs on that slice, and its expected output is UltraDim's existing kernel run on the same slice
(§5). Nothing is synthesised: the inputs are the stored rows.

- **Training cases:** seeded samples at two or more sizes.
- **Held-out cases:** drawn with a seed and a rule the training draws never see. The rule follows UltraDim's own
  `holdout_split_ops.rs`:
  - deterministic: a stable hash of `(split_seed, row_id)` orders the rows;
  - content-addressed: the split id is a hash of the family's generation fingerprint and the parameters;
  - persisted, and refused when the family has changed under it (a stored fingerprint that no longer matches).

  The held-out ids are a fixed partition of the family, so a procedure is never judged on a row it was trained on.
- **Edge cases from the data's own shape:** an empty cluster, n = 1, a width d not a multiple of the workgroup.
  Each must be a real slice (a sample that leaves a cluster empty), not generated values.
- **The timing input:** a large slice, never a quality case, sized under the work guard.
  - The k-means benchmark found the limit: one workgroup of A7's form at n = 8,192 is about 8e7 node executions,
    under the 1.7e8 cap a submit.

Today fuller draws cases from files (`tests/corpus/futhark_problems/<name>/cases.tsv`). The data source adds a case
source that reads a family by name and writes the same `Case` records, so the runner is unchanged.

## 3. Invoking a procedure on UltraDim's own buffers

**The path.** It is the rust-wgpu backend's, as `futhark-rust-rt` implements it today:

- A procedure is one generated Rust module (WGSL kernels, a kernel table, host code over the runtime).
- It runs on a `wgpu::Device` and `wgpu::Queue` the caller gives (`FutharkCtx::new`).
- Its arguments may be the caller's own buffers (`Array::from_buffer`: "no copy; the program reads it in place").
- A caller's buffer is never put in the runtime's free list (`mem.rs`: "the caller's buffer is only ever the
  caller's").
- Entry points are typed (`EntryDesc`: inputs, outputs, call), and the in-process host loads a generation's module
  as a dylib (`host.rs`).

So UltraDim calls a procedure by handing its resident buffers and its device to the module's entry point. The
result is a device array UltraDim keeps or reads.

**The gaps, measured or read, each to be closed before a procedure can run on UltraDim's device:**

1. **`SHADER_F16`.**
   - `FutharkCtx::new` refuses a device without `SHADER_F16`, because Futhark's WGSL prelude enables f16.
   - UltraDim's k-means device manager requests no features: `required_features: wgpu::Features::empty()`, in
     `device_manager.rs` line 212 (read by grep).
   - The same problem appeared in the WGSL kingdom and was closed by generating no f16.
   - For Futhark either UltraDim requests the feature, which is an UltraDim change and needs Andrew's say-so, or the
     backend emits no f16 when the program uses none.
2. **Buffer usages.**
   - `DevMem::from_buffer` requires `STORAGE | COPY_SRC | COPY_DST`.
   - A7's persistent vectors buffer is created `STORAGE` only (`upload_vectors_persistent`).
   - Either UltraDim creates its resident buffers with the copy usages (an UltraDim change), or the runtime accepts a
     storage-only buffer for an input the program only reads.
3. **The data is not all resident.**
   - The dense, key and payload stores are files (`read_exact_at`).
   - Residency today is per engine: A6's Lloyd loop keeps raw bands, keys, assignments and centroids on the device
     (`GPU_FUNCTIONS_INVENTORY.md` Part C), and A7 keeps a persistent vectors buffer.
   - A data source needs one resident copy per family that every procedure binds. That is UltraDim's to provide; the
     plan asks for it and does not build it.
4. **The work guard.**
   - `program.rs` `launch` submits each kernel with no bound on its work.
   - Every evolved procedure must run under fuller's work guard (`wgsl::work_guard`), as every WGSL candidate does,
     before it touches UltraDim's device. A runaway kernel on the display GPU has crashed this machine.
5. **Per-binding limits.**
   - A7 refuses a centroid buffer over 128 MiB (`MAX_STORAGE_BUFFER_BINDING_SIZE`).
   - The M3 Max adapter reports 4 GiB.
   - A procedure's arrays must respect the device limits UltraDim sets, not the adapter's.

## 4. The procedure catalogue

| order | procedure | UltraDim's kernel (the judge) | why first |
|---|---|---|---|
| 1 | centroid update: mean of each cluster's rows, an empty cluster kept | A7 `update_centroids` (live in `GpuKmeansCluster`); A6 `mean_update_segmented` + counting sort | built as a WGSL benchmark; A7 is O(k·n·d) and the scatter gives A7's bits |
| 2 | assignment: nearest centroid by Euclidean distance, or by cosine in key space | A7 `compute_distances` + `assign_points`; A6 `build_cosine_assign_pipeline` | the other half of the Lloyd step; `problems/kmeans_assign` exists for the WGSL kingdom |
| 3 | per-row L2 normalise (empty row left zero) | A6 `build_l2_normalise_pipeline` (B6) | rank-2 in, rank-2 out, exact rule simple |
| 4 | inertia: Σ(1 − xᵀμ) by fixed-order blocked sum | A6 `inertia_contrib` + `block_reduce` (B2) | a reduction with a fixed order the judge states |
| 5 | weighted Gram H = ΦᵀΦ | A4 `H_GRAM_ACCUM_WGSL` | long-running today (130 s at 419,734 × 8,192); the Gram problem exists |
| 6 | top-k of a distance matrix | A7 `select_top_k`, A8 `select_top_k` | an argmax output with a tie rule (`problems/kmeans_assign` has one) |

Later, once their code is read: the trellis kernels and the UMAP epoch (A11). The HNSW shaders (A5) are retired and
are never a judge.

## 5. Staged fitness against UltraDim's kernel

This is the method of `examples/futhark_evolve.rs` and `phylu/examples/wgsl_true_case_fit.rs`, unchanged.

**The judge.** Each case's expected output is UltraDim's kernel run on the case's slice, on the same device: the
shipped kernel is the oracle, and outputs are cached by a key over the kernel's text, the device, the case and the
input bytes (`harness::oracle_fill`).

**Exact means within a measured tolerance, per procedure.**

- Integer outputs (assignments, top-k ids) must be exact, with a stated tie rule.
- A float sum is exact within the bound on any order of summing its terms, `(n + 2)·ε` of the output's RMS
  (measured for the centroid update: reorderings at most 1.37e-6 of the RMS).
- The tolerance is set by probes run on the device before any run, never chosen after one.

**Stage 1: quality only, no timing.**

- Every case gives its own linear inputs: the share outside tolerance, the shares off by 1 %, 10 % and 50 %, and
  the mean and largest error.
- Training and held-out cases are both in the quality group.
- R and C stop at the first exact procedure.

**Stage 2 (P): the two-level HFF.**

- θ_quality over the quality inputs, θ_speed over 11 timing slots, each an interleaved pair against UltraDim's
  kernel on the timing input, `r / (1 + r)`, never averaged.
- The gate before the run: two A/A sets of 11 pairs within ±5 %.
- A speed claim counts only when the champion's re-time median is below the least A/A ratio.
- UltraDim's own kernel text is timed against the yardstick, so the final comparison is against UltraDim itself.

## 6. The runner standard

Every runner of this programme meets `docs/STANDARD_runners.md`:

- the stream (`OUT/stream.jsonl` through `fuller::telemetry`);
- the staged two-level HFF;
- checkpoints with `--resume`;
- the kill trigger `OUT/killme.trigger`;
- the guards: an evaluation budget, the wall clock as a guard only, and the work guard on every dispatch;
- the hall of fame with program text;
- the ledger with lineage;
- the mutation events.

It is registered in the standard's tests (fuller and phylu `tests/runner_standard.rs`). The Futhark runner
(`examples/futhark_evolve.rs`) already meets it; the data source changes where its cases come from (§2) and where a
candidate runs (§3), not the runner.

## Order of work

Each step ends in a test or a measurement, before the next starts:

1. **The schema reader** writes a signature instance from a family's metadata, starting with the dense rows.
   Test: generator `--check`, witness tests.
2. **The signature family's f32 extension** (§1, items 1-6). Test: the centroid update's gather and scatter as genes,
   both exact against A7 on every case, interpreted and compiled.
3. **The case source** draws training and held-out slices from a family (§2). Test: a split is reproducible and is
   refused after the family changes.
4. **The no-copy invocation** on a device and buffers made as UltraDim makes them (§3, gaps 1-5). Test: a procedure
   reads a caller's buffer in place, bit for bit against the copied path, under the work guard.
   - Gaps 1 and 2 need either an UltraDim change (Andrew's say-so) or a backend change. The plan recommends the
     backend change: no f16 when unused, and storage-only buffers for read-only inputs.
5. **The centroid update** in the Futhark kingdom, R/C/P runs prepared, not launched. Then the catalogue in order.

**GPU only.** The CPU appears only as a test reference: the host interpreter of `futhark_sig.rs` checks the compiled
programs; it is never a path a procedure runs on.
