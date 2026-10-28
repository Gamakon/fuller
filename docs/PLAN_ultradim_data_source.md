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
- A procedure's data is FETCHED from UltraDim through its gRPC API, by row id or by facet query, into our own GPU
  buffers, and the procedure runs there through the rust-wgpu backend (`futhark-rust-rt`). There is no zero-copy and
  no access to UltraDim's buffers (Andrew's ruling, 2026-10-28).
- Fitness is staged: exact against UltraDim's existing kernel first, then speed, under the runner standard.

**What is unique.**

- The procedures are typed programs that evolution writes and improves, not hand-written stored procedures.
- Every candidate is legal by its types, the kingdom is made from the data's own schema, and the judge is the
  kernel the database ships today, run on the same fetched data.

**Status (2026-10-28).**

- A plan, nothing built for it.
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
7. **No f16.** UltraDim computes in f32, with Kahan correction for accuracy.
   - The kingdom for UltraDim procedures has no f16 type and no f16 row. It is an option of the generation, not an
     edit of a shared table: the WGSL kingdom's interfaces already do this (`has_f16=False`, `lib.drop_f16`), and
     every shared table is byte for byte unchanged.
   - The signature family has no f16 family today.
   - The rust-wgpu runtime must not need f16 for a program that uses none. `FutharkCtx::new` refuses a device
     without `SHADER_F16`, because Futhark's prelude enables it, so the runtime has to change.

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

## 3. Fetching the data, then running the procedure on our own buffers

**No zero-copy, no access to UltraDim's buffers (Andrew's ruling).** The procedure's inputs are fetched from
UltraDim's sidecar stores through its gRPC API (`ultradim.proto`, the only interface: Rules.md), loaded into
buffers on our own `wgpu::Device`, and the procedure runs there. UltraDim's data, devices and kernels are never
touched.

**What UltraDim serves today (read in its source):**

| need | RPC | what it returns |
|---|---|---|
| rows by facet query (a window, a label, a keyword) | `GetUltradimV23RowIdsByFacet` (`template_ops.rs`) | every LIVE row id whose payload satisfies the filter, ascending, and the committed payload count |
| one row by its content key | `GetUltradimV23RowByContentKey` | the row id of the exact content-key facet value |
| a row's projected keys | `GetUltradimV23Keys` (`family_ops.rs`) | for the given row ids and seeds, each row's unit `D_proj` keys, row-major, with the row ids |
| a family's metadata | `GetUltradimV23CollectionInfo` | dims, substrate, seeds, shard readiness |

The row ids are UltraDim's index keys:

- contiguous from 0;
- every store contiguity-gated (`dense_sidecar.rs`, `key_sidecar.rs`, `payload_sidecar.rs`);
- the facet index answers from sorted columns (`facet_index.rs` `rows_matching`), with liveness applied by the
  caller from the CSR.

**The flow of a case.**

1. Select rows: a facet query, or a seeded sample of ids from the committed count.
2. Fetch their keys with `GetUltradimV23Keys`. These are the canonical-seed keys `GpuKmeansCluster` clusters with
   A7 today (`gpu_cluster_ops.rs`), so the first procedure's data is fully served.
3. Upload them into our own buffers, each under UltraDim's per-binding limit (below), banded when larger.
4. Run the procedure and UltraDim's kernel (the judge, §5) on the same uploaded data.

**The one gap.** No RPC serves a family's raw dense rows by id. The dense sidecar holds them (`TRLDNSE1`, read by
`DenseSidecar::get_row` and `get_row_range` in-process only), and `GetCollectionVectors` serves Blue World worlds,
not UDV23 families.

- Procedures over keys need nothing new.
- Procedures over raw rows need an RPC that returns them by id, which is an UltraDim change and Andrew's to decide.
- Until then, the catalogue starts on keys.

**UltraDim's configured limits, never the adapter's.** A procedure's buffers respect the limits UltraDim sets
(`gpu_limits.rs`), read from its own code and configuration:

- `ULTRADIM_GPU_BUFFER_LIMIT_GB`: the buffer ceiling, else two-thirds of system RAM, clamped to the adapter.
- `effective_buffer_limit`: half of the lesser of the device's binding size and the 256 MB Metal
  `write_buffer` limit, so 128 MiB on the M3 Max. UltraDim's code calls it "the single authoritative source".
- `GPU_STORAGE_BUFFERS_PER_STAGE`: 16 by default.
- A7's own 128 MiB check on its buffers.

Our device is requested with those limits, and every procedure's buffer is checked against
`effective_buffer_limit` as UltraDim checks it. The adapter's maxima (4 GiB on the M3 Max) are never used.

- The k-means benchmark harness plans by fuller's own `effective_buffer_limit`, not UltraDim's. Its largest buffer
  (the wide-d timing input's points, 8,192 × 4,096 f32) is exactly 128 MiB: at UltraDim's limit, not over it. The
  harness must read UltraDim's limit before any run that claims UltraDim's conditions.

**The work guard is ours.** fuller's work guard runs on every evolved candidate. It is added in the Futhark-to-Rust
backend's launch (`futhark-rust-rt` `program.rs` `launch` submits with no bound today) for evolution's random
candidates. It is not an UltraDim matter.

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
4. **The fetch-and-run path** (§3).
   - The gRPC client: facet query, keys by row id.
   - Upload under UltraDim's configured limits.
   - The no-f16 runtime.
   - The work guard in the backend's launch.
   - Test: the fetched keys equal the sidecar's, bit for bit; a procedure and UltraDim's kernel agree on them.
5. **The centroid update** in the Futhark kingdom, R/C/P runs prepared, not launched; the per-backend check against
   A7 (Metal here, Vulkan on the NVIDIA/RunPod stage). Then the catalogue in order.

**GPU only.** The CPU appears only as a test reference: the host interpreter of `futhark_sig.rs` checks the compiled
programs; it is never a path a procedure runs on.
