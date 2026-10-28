# PLAN: Futhark as UltraDim's procedure language, run server-side

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
- A procedure runs SERVER-SIDE, as a stored procedure does in PL/SQL or T-SQL:
  - inside the UltraDim server process, on the server's GPU;
  - over data the server loads from its own sidecar stores and index;
  - in buffers the server owns, within the server's configured limits.
- **The procedure engine.** A procedure engine embedded in the server holds a catalogue of registered Futhark
  procedures (Rust+WGSL modules from the rust-wgpu backend, loaded in-process).
- **gRPC is only the call.** A client invokes a procedure by name (the CALL statement) and receives its result.
- **Evolution is a server-side job.** Candidate procedures are compiled, run and scored inside the server, against
  UltraDim's own kernel, on the server's data.
- Fitness is staged: exact against UltraDim's existing kernel first, then speed, under the runner standard.

**What is unique.**

- The procedures are typed programs that evolution writes and improves, not hand-written stored procedures.
- Every candidate is legal by its types, the kingdom is made from the data's own schema, and the judge is the
  kernel the database ships today, run beside the candidate on the server's own data.

**Status (2026-10-28).**

- A plan, nothing built for it. Every UltraDim change it needs is listed file by file in §9, each marked "needs
  Andrew's approval". None is built.
- Its first procedure, the k-means centroid update, is already a benchmark in the WGSL kingdom
  (`docs/kmeans_update_dropin.md`, fuller 4f54a507 and 1e840384).
- Two facts measured there carry over:
  - on Metal (Apple M-series), A7's Kahan compensation changes no bit: a plain sum gives A7's bits on 8 of 8 cases.
    This is written up for UltraDim in `docs/reports/2026-10-28_metal_removes_a7_kahan.md`. Vulkan is not yet
    measured, and nothing is claimed for it;
  - a counting-sort scatter with A7's semantics gives A7's bits on 8 of 8 cases.
- The rust-wgpu backend is being built by another agent. Its plan, `docs/PLAN_futhark_rust_wgpu.md`, does not exist
  in the tree yet. Its runtime crate, `futhark-rust-rt/`, is untracked work in progress; this plan reads its source.

**Why.** UltraDim's GPU work today is fixed kernels (76 `@compute` entry points, `GPU_FUNCTIONS_INVENTORY.md`), and
some are known to be slow: A7's gather is O(k·n·d). A procedure language over UltraDim's data lets the system write
the next kernel instead of a person, judged against the one it replaces.

## Sources

Read in full:

- **qdrant:**
  - `ultradimdb/src/core.rs`;
  - `ultradimdb/src/store/dense_sidecar.rs`;
  - `ultradimdb/src/store/mod.rs`;
  - `ultradimdb/src/gpu_cluster_ops.rs`;
  - `lib/projection-voting-kmeans/src/gpu_limits.rs` 1-380 (every function but the tests);
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
  - `key_sidecar.rs` 1-215 (append, `read_seed_keys_rows`, `read_seed_keys`);
  - `payload_sidecar.rs` 1-560 (every reader, `row_passes`);
  - `facet_index.rs` 1-120 and `rows_matching`;
  - `family_ops.rs` `get_ultradim_v23_keys_impl`;
  - `template_ops.rs` 1680-1710 (the facet RPC);
  - `ultradim.proto`: `GetUltradimV23Keys`, `GetUltradimV23RowIdsByFacet`, `GetUltradimV23RowByContentKey`,
    `GetCollectionVectors`.
  - `service.rs` 1-40 (how an RPC routes to its handler);
  - `device/device_manager.rs` 200-225 (the device request: no features).
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
6. **A float tolerance as the exactness rule** (§7), in place of `compare`'s integer equality.
7. **No f16.** UltraDim computes in f32, with Kahan correction for accuracy.
   - The kingdom for UltraDim procedures has no f16 type and no f16 row. It is an option of the generation, not an
     edit of a shared table: the WGSL kingdom's interfaces already do this (`has_f16=False`, `lib.drop_f16`), and
     every shared table is byte for byte unchanged.
   - The signature family has no f16 family today.
   - The rust-wgpu runtime must not need f16 for a program that uses none. `FutharkCtx::new` refuses a device
     without `SHADER_F16`, because Futhark's prelude enables it, so the runtime has to change.

**Not yet covered.** Trellis structures and stored centroids are not mapped here: their code has not been read
(Sources).

## 2. Observations and held-out cases, read on the server

**A case.** A case is a slice of a family's rows, read by the server from its own stores:

- the row ids come from the index: a facet query (`facet_index::rows_matching`, liveness from the CSR's
  `is_live`, as `template_ops.rs` applies it), a content key, or a seeded sample of ids below the committed count;
- the data comes from the sidecars by those ids:
  - keys: `key_sidecar::read_seed_keys_rows`, ascending unique ids;
  - dense rows: `DenseSidecar::get_row` and `get_row_range`;
  - facets: `payload_sidecar::read_payloads_at_rows`.

The case's expected output is UltraDim's existing kernel run on the same slice (§7). Nothing is synthesised: the
inputs are the stored rows.

- **Training cases:** seeded samples at two or more sizes.
- **Held-out cases:** drawn with a seed and a rule the training draws never see. The rule follows UltraDim's own
  `holdout_split_ops.rs`:
  - deterministic: a stable hash of `(split_seed, row_id)` orders the rows;
  - content-addressed: the split id is a hash of the family's generation fingerprint and the parameters;
  - persisted, and refused when the family has changed under it.
- **Edge cases from the data's own shape:** an empty cluster, n = 1, a width d not a multiple of the workgroup. Each
  must be a real slice, not generated values.
- **The timing input:** a large slice, never a quality case, sized under the work guard and under the server's
  limits.

## 3. The procedure engine, in the server

**The catalogue.** A registry in `CoreService`, as the server keeps its other resident engines (`core.rs`: one map
a kind, a cache over an on-disk store, never the source of truth). It holds the registered procedures:

- each a module the rust-wgpu backend generated from a Futhark program: WGSL kernels, a kernel table, host code over
  `futhark-rust-rt`, typed entry points (`EntryDesc`: inputs, outputs, the call);
- its name, its signature (the schema-made kingdom's, §1), the family it is over, and the UltraDim kernel it
  replaces or extends;
- its provenance: hand-written or evolved, the run that found it, its measured exactness and speed against that
  kernel.

A procedure is persisted under the storage root with its source, generated module and metadata. The catalogue
survives a restart, as the other registries do.

**Loading.** Registered modules are compiled into the server or loaded as dylibs. `futhark-rust-rt/src/host.rs`
already loads a generation's module as a dylib, with one copy of the runtime and of wgpu in the process.

**The device.**

- A procedure runs on a server `wgpu::Device` made by UltraDim's own device manager (`device_manager.rs`), with the
  server's limits (`gpu_limits.rs`):
  - `apply_auto_limits` and `ULTRADIM_GPU_BUFFER_LIMIT_GB`;
  - `effective_buffer_limit`, 128 MiB a buffer on the M3 Max, "the single authoritative source";
  - `GPU_STORAGE_BUFFERS_PER_STAGE`.
- The adapter's maxima are never used.
- The engine checks every buffer against `effective_buffer_limit` before creating it, and bands what is larger, as
  UltraDim's kernels do.

**Buffers.** The engine loads a call's inputs from the stores (§2) into buffers it creates on the server's device.
They are uploaded in writes under the 256 MB Metal `write_buffer` limit (`QUEUE_WRITE_BUFFER_LIMIT`). It hands them
to the procedure as `futhark-rust-rt` arrays (`Array::from_buffer`) and reads the result back, or keeps it resident
for the next call.

**No f16** (§1 item 7). The runtime must run a program that uses no f16 on a device without `SHADER_F16`, which
UltraDim's device manager does not request.

**The work guard** (fuller's, ours to add). Every launch of an EVOLVED candidate is bounded by the work guard in the
backend's launch (`futhark-rust-rt` `program.rs` `launch` has none today). A registered procedure has passed it.

## 4. CALL: invoking a procedure by name

One gRPC call, the CALL statement:

```proto
rpc CallProcedure (CallProcedureRequest) returns (CallProcedureResponse) {}

message CallProcedureRequest {
  string procedure = 1;            // the catalogue name
  string family = 2;               // the family it runs over
  UdV23Filter rows = 3;            // the rows, by facet (the WHERE clause), or
  repeated uint64 row_ids = 4;     //   by id; one of the two
  repeated ProcedureArg args = 5;  // scalar and small array arguments (k, a seed, initial centroids)
  bool keep_result = 6;            // keep the result resident under a handle for a later CALL
}
message ProcedureArg { string name = 1; oneof v { int64 i = 2; double f = 3; FloatArray fa = 4; IntArray ia = 5; string handle = 6; } }
message CallProcedureResponse {
  bool success = 1;
  repeated ProcedureResult results = 2;   // each output: its shape and values, or a handle
  double load_ms = 3; double gpu_ms = 4;  // the server's own timings
  optional string error_message = 5;
}
```

The arguments are typed by the procedure's signature: a family-bound input is named by the call and loaded by the
server, never sent. A result is returned, or kept as a handle, so a pipeline of procedures (assign, then update)
runs without the data leaving the server.

Beside it:

- `RegisterProcedure`: a module and its metadata into the catalogue;
- `ListProcedures`;
- `DropProcedure`.

## 5. Evolution as a server-side job

**The job.** An evolution run is a job the server runs, started by `StartProcedureEvolution`:

- the family;
- the procedure's signature;
- the UltraDim kernel to beat;
- the regime (R, C or P), the seed and the budget.

The job runs `examples/futhark_evolve.rs`'s loop, the same machinery, inside the server:

- every candidate is compiled by the rust-wgpu backend, loaded in-process (`host.rs`), and run on the server's
  device;
- the candidates are scored on the server's cases (§2) against UltraDim's kernel (§7), under the work guard.

Only the champion is registered, and only on a request.

**The runner standard holds** (§8):

- the stream `OUT/stream.jsonl`, under the server's storage root;
- `OUT/killme.trigger`, also exposed as `StopProcedureEvolution`;
- the guards: the budget, the wall clock, the work guard;
- checkpoints, the hall of fame and the ledger.

`GetProcedureEvolutionStatus` returns the job's state and the stream's path, which `evowatch` reads.

**One GPU job at a time.** An evolution job and the server's live work share one GPU. The job yields to live calls,
or runs only when the server is idle, as Andrew sets.

## 6. The procedure catalogue

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

## 7. Staged fitness against UltraDim's kernel, measured on the server

This is the method of `examples/futhark_evolve.rs` and `phylu/examples/wgsl_true_case_fit.rs`, unchanged.

**The judge.** Each case's expected output is UltraDim's kernel run on the case's slice, on the same device: the
shipped kernel is the oracle, and outputs are cached by a key over the kernel's text, the device, the case and the
input bytes (`harness::oracle_fill`).

**Exact means within a measured tolerance, per procedure.**

- Integer outputs (assignments, top-k ids) must be exact, with a stated tie rule.
- A float sum is exact within the bound on any order of summing its terms, `(n + 2)·ε` of the output's RMS
  (measured for the centroid update: reorderings at most 1.37e-6 of the RMS).
- The tolerance is set by probes run on the device before any run, never chosen after one.
- **Per backend.** The bit-exactness test against UltraDim's kernel, and the tolerance, are measured and stated
  separately for each backend UltraDim runs on (Metal, Vulkan), because the compilers differ: Metal removes A7's
  Kahan compensation, and Vulkan has not been measured. The Vulkan check is part of the NVIDIA/RunPod stage. No
  backend's tolerance is carried to another.

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

## 8. The runner standard

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
(`examples/futhark_evolve.rs`) already meets it; the server-side job changes where its cases come from (§2) and where a
candidate runs (§3), not the runner.

## 9. The UltraDim changes this needs (each needs Andrew's approval; none is built)

| file (qdrant) | change |
|---|---|
| `ultradimdb/Cargo.toml` | depend on `futhark-rust-rt`, and on fuller's runner pieces for the evolution job |
| `Cargo.lock` / `lib/projection-voting-kmeans/Cargo.toml` | ONE wgpu: UltraDim is on wgpu 27.0.1, `futhark-rust-rt` pins `=30.0.1`. Either UltraDim moves to 30, or the runtime builds against 27 (our change, not UltraDim's) |
| `ultradimdb/proto/ultradim.proto` | `CallProcedure`, `RegisterProcedure`, `ListProcedures`, `DropProcedure`, `StartProcedureEvolution`, `StopProcedureEvolution`, `GetProcedureEvolutionStatus` and their messages (§4, §5) |
| `ultradimdb/src/service.rs` | one arm a new RPC, routing to its handler, as every arm does |
| `ultradimdb/src/core.rs` | `CoreService` gains the procedure catalogue (`procedures`), the resident result handles, and the evolution jobs, each a map like the others |
| `ultradimdb/src/procedure_ops.rs` (new) | the handlers: load a call's rows from the stores (§2), run the module on the server's device, return or keep the result; register, list, drop |
| `ultradimdb/src/procedure_store.rs` (new) | the catalogue's on-disk store under the storage root: source, module, metadata; rehydrated at boot |
| `ultradimdb/src/procedure_evolution.rs` (new) | the evolution job: the runner on the server's data, its stream and kill trigger under the storage root, one job at a time |
| `ultradimdb/src/lib.rs` | the three new modules |
| `lib/projection-voting-kmeans/src/device/device_manager.rs` | for the evolution job only: request `TIMESTAMP_QUERY` when the adapter has it (the GPU timing of the speed inputs). The live device's features are unchanged. No `SHADER_F16`: the runtime runs without it (§3) |

No change is needed to the stores, the facet index or the limits: the engine calls them as they are.

## Order of work

Each step ends in a test or a measurement, before the next starts:

1. **The schema reader** writes a signature instance from a family's metadata, starting with the dense rows.
   Test: generator `--check`, witness tests.
2. **The signature family's f32 extension** (§1, items 1-6). Test: the centroid update's gather and scatter as genes,
   both exact against A7 on every case, interpreted and compiled.
3. **The case source** reads training and held-out slices through the stores' own functions (§2). Test: a split is
   reproducible and is refused after the family changes.
4. **The engine, outside the server first.** The engine runs as a library that opens a storage root read-only,
   against a copy of a family.
   - This needs no UltraDim change: catalogue, load from the stores, run under the server's limits, no f16, the work
     guard.
   - Test: a procedure's output equals UltraDim's kernel's on the same rows, per backend.
   - Moving it into the server (§9) waits for Andrew's approval.
5. **The centroid update** in the Futhark kingdom, R/C/P runs prepared, not launched; the per-backend check against
   A7 (Metal here, Vulkan on the NVIDIA/RunPod stage). Then the catalogue in order.

**GPU only.** The CPU appears only as a test reference: the host interpreter of `futhark_sig.rs` checks the compiled
programs; it is never a path a procedure runs on.
