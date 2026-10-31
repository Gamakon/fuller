# A faster k-means centroid update for UltraDim: evidence and a change request

*fuller / phylu, 2026-10-31. A draft for discussion with the UltraDim developers.*

## Context

**What we propose.** Replace UltraDim's k-means centroid update (inventory A7, `update_centroids`) with a five-kernel form: a counting sort of the points by cluster, then a per-cluster mean. Use the evolved variant of that form described below.

**What we measured.** On an Apple M3 Max, under Metal:
- the candidate's output is bit-identical to A7's on every test case;
- it runs at **0.0273 of A7's time** (about 37 times faster), at n = 8,192 points, d = 256 dimensions and k = 1,024 clusters.

**Where the speed comes from.**
- **Almost all of it is the algorithm.** A7 gathers: each output cell loops over every point, which costs O(k·d·n). The candidate sorts the points by cluster once, then each cluster sums only its own points, which costs O(n·d). The hand-written form of this algorithm runs at 0.0277 of A7's time.
- **Evolution added 1.6 %** on top of the hand-written form (0.9840 of its time, 11 of 11 pairs faster). Most of that gain comes from removing the Kahan compensation, which Metal's compiler already removes from A7 (see Caveats).

**What we ask of the UltraDim team.**
- Review the kernel.
- Validate it in UltraDim, on a substantial dataset, on Metal and on Vulkan, including rolling k-means, the time-sensitive path.
- Decide whether to adopt it.

We have changed no UltraDim code.

> **Status, 2026-10-31 14:00Z: the ratios below are under re-measurement.** On this Mac, the same WGSL text compiled into separate pipelines runs each entry at one of two speeds. Measured: up to about 1.7× apart per entry; whole-module totals of 0.638 to 0.900 ms for one and the same text. The speed sticks to the pipeline. The A/A gate timed only one pipeline against itself, so it could not see this. Any ratio between two separately compiled kernels is therefore unconfirmed. The candidate's direction, about 0.027 of A7, lies far outside this range. Its exact values, and the 1.6 % over the hand-written form, are being re-measured with a fix: A/A across separate pipelines, and each side's fastest over several pipeline instances.

## 1. The kernel

- Five entry points share UltraDim's buffers, with one scratch binding added:
  - a histogram of the assignments;
  - offsets;
  - scatter;
  - the per-cluster sum;
  - the mean.
- **The candidate:** `docs/kmeans_champion_s1/` (WGSL, dispatch file, and the diff against A7 and against the hand-written form).
- **What evolution changed against the hand-written form:**
  1. **The mean kernel: Kahan compensation removed.** The compensation term now reads a register that is never written, so the sum is a plain sum.
  2. **The histogram kernel's workgroup width:** 128, not 256.
  3. **Rewrites with no effect:** a loop's final counter used in place of the literal 256, constants that are never used, and one dead store.

## 2. How it was measured

| Item | Method |
|---|---|
| Dispatch | One unsplit dispatch per kernel, exactly as UltraDim issues it |
| Pairs | The candidate and the yardstick interleaved run by run (A, B, A, B, …), each side the fastest of about 118 runs, at least 1 s a side, with one discarded warm-up pair |
| Noise floor (A/A) | The yardstick against itself, 2 × 11 pairs; every ratio must lie within ±5 %. Measured: 0.9966 to 1.0039 |
| "Faster" | The median ratio below the least A/A ratio, AND at least 9 of 11 pairs faster |
| Correctness | Bit for bit, cell by cell, on all 8 cases (4 training, 4 held out) and on the timing case's own output |

**Results:**

| Comparison (11 interleaved pairs) | Median ratio | Pairs faster |
|---|---|---|
| candidate against A7 | 0.0273 | 11 of 11 |
| candidate against the hand-written scatter form | 0.9840 | 11 of 11 |
| hand-written scatter form against A7 | 0.0277 | 11 of 11 |

## 3. Faults we found in our own method, all fixed before these numbers

| Fault | What it did | Fix |
|---|---|---|
| Timing split into many submits | It favoured kernels with small work bounds, and didn't match UltraDim's dispatch | Timing is unsplit, one dispatch |
| The display shares the GPU | Self-timing ranged 0.92 to 1.10 when each side was timed in its own block | Run-by-run interleaving |
| Our reader re-wrote A7 1.72× slower | The kernel's `let`s were inlined into its loop | `let`s are kept and loop-invariant terms hoisted; A7 as expressed now runs at 0.9995 of its own text |
| A u32 overflow | A guard `gid.x >= (k·d)²` wrapped to 0 at the timing size, so a kernel did no work there and claimed 450×. Two evolved kernels used it. | Every timed kernel's output on the timing case is checked against the yardstick's. Both are withdrawn. |
| The dispatch limit | A mutated workgroup width needed 262,144 workgroups (the limit is 65,535) | The types bound every workgroup width |

## 4. Caveats

- **Kahan compensation.** On Metal, the shader compiler already removes A7's compensation: a plain sum gives A7's bits on 8 of 8 cases (measured 2026-10-28). So the candidate is bit-identical on Metal. **On Vulkan this has not been measured.** If Vulkan keeps the compensation, A7 is slightly more accurate there. Earlier measurements put other summing orders within 1.37e-6 of the output's RMS.
- **One timing shape:** n 8,192, d 256, k 1,024. Other shapes, especially much larger n, or the small-batch shapes rolling k-means uses, are not measured.
- **One machine:** an Apple M3 Max under Metal, through wgpu 30.
- **The scratch binding:** the five-kernel form needs one more storage buffer than A7, sized for the points' order. UltraDim's per-binding limits apply (128 MiB on this machine).

## 5. Proposed validation in UltraDim

1. Edit UltraDim's centroid update to dispatch the five kernels, and allocate the scratch buffer.
2. Build, and run UltraDim's own tests.
3. Load a new, substantial dataset.
4. Run k-means on Metal and on Vulkan. Compare the centroids against today's kernel, bit for bit and by RMS, and compare the final cluster assignments.
5. Time it inside UltraDim, end to end, at production shapes.
6. Test rolling k-means in particular, the time-sensitive path, at its own batch sizes.

## 6. Reproduction

- **Runner:** `phylu/examples/wgsl_true_case_fit.rs`, binary `tcf_724370d3_ac6f4d98` (phylu 724370d3, fuller ac6f4d98).
- **Problems:** `fuller/problems/kmeans_update_segmented`, `fuller/problems/kmeans_update`.
- **Evolution run:** `kmeans_P_s1`, 97 generations, 1,415 evaluations. Its per-generation table is in the run directory.
