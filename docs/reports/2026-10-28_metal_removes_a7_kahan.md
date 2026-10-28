# Finding for UltraDim: on Metal, A7's Kahan compensation does nothing

**On Metal, UltraDim's centroid-update kernel (inventory A7, `update_centroids`) gives exactly the same bits with
its Kahan compensation as without it.** Measured on Metal (Apple M-series, an M3 Max), on 8 of 8 cases. Vulkan has
not been measured, and nothing here is claimed for it.

**Conditions:**

- wgpu 30.0.1 and naga 30.0.1, fuller's own pins (`Cargo.lock`). The wording requested was "wgpu 0.20"; this is the
  version that was actually measured.
- fuller 4f54a507, 2026-10-28 10:08Z, the problem `problems/kmeans_update`, run by `examples/wgsl_case_check`.

## What was measured

**The kernels.** A7 verbatim (`problems/kmeans_update/shipped.wgsl`, from qdrant
`lib/projection-voting-kmeans/src/gpu_kmeans.rs`, lines 658-693) against three probes in
`problems/kmeans_update/variants/`. Each probe is A7 with one change:

| probe | the change | what it tests |
|---|---|---|
| `plain.wgsl` | the Kahan lines replaced by `sum = sum + term` (same order) | whether the compensation changes any bit |
| `reverse.wgsl` | plain sum, points in reverse order | the size of an ordinary reordering error |
| `blocked.wgsl` | plain sum in blocks of 64, then the blocks | a tree-like order |

**The cases.** 8 cases: normal(0, 1) points; labels uniform in `[0, k − empty)`; old centroids normal. The case
sizes are in `problems/kmeans_update/cases/constraint.toml`:

- n from 1 to 4,000;
- d from 1 to 2,048;
- k from 1 to 300;
- empty clusters both at the end and at random.

**The comparison.** The expected output is A7's own output on the same device. "Differ" counts cells not bit for bit
equal. The norm is the largest cell error over the RMS of A7's output.

| probe | cases bit-exact with A7 | cells differing (all cases) | largest error / RMS |
|---|---|---|---|
| plain (no Kahan) | **8 of 8** | **0** | 0 |
| reverse order | 2 of 8 | 27,832 | 1.20e-6 |
| blocks of 64 | 4 of 8 | 23,026 | 1.37e-6 |

The reorderings change up to 71 % of the cells (17,031 of 24,576 on the n = 3,000, d = 2,048 case), so the cases are
large enough that a working compensation would have changed bits too. The plain sum changed none.

## What it means

**A7's Kahan correction is very likely removed by the compiler on Metal.**

- Kahan's `c = (t − sum) − y` is zero in real arithmetic. A compiler allowed to reassociate floating point (fast
  math) folds it away, and that is the likely cause. We did not inspect the Metal shader the driver built, so that
  is an inference, not a measurement.
- So the accuracy A7 is written to give, Kahan-compensated sums, is very likely not what it gives on Metal today.
  It gives a plain sequential sum.

**On the size.** The difference from any summing order on these cases is at most about 1.4e-6 of the RMS. Kahan's
benefit would matter for long sums of values of mixed magnitude, which these cases are not.

## Not measured

- Vulkan (NVIDIA): the same probes on the RunPod NVIDIA stage, before any claim for that backend.
- Whether a WGSL form exists that Metal does not fold. A data-dependent barrier, or the `fma`-based TwoSum, are
  candidates, not tested.
- Other UltraDim kernels that use Kahan: A8 `calculate_cosine_similarities` is marked Kahan-compensated in the
  inventory. It has not been probed.

## Reproduce

```bash
cargo run --release --features gpu,wgsl --example wgsl_case_check -- --problem problems/kmeans_update \
    problems/kmeans_update/shipped.wgsl problems/kmeans_update/variants/plain.wgsl \
    problems/kmeans_update/variants/reverse.wgsl problems/kmeans_update/variants/blocked.wgsl
```

The raw output of the measured run is `/private/tmp/claude-501/kmu/checks.log`, section 1.
