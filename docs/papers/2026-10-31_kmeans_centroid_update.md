# A faster k-means centroid update for UltraDim: evidence and a change request

*fuller / phylu, 2026-10-31. A draft for discussion with the UltraDim developers.*

## Context

**What we propose.** Replace UltraDim's k-means centroid update (inventory A7, `update_centroids`) with a five-kernel form: a counting sort of the points by cluster, then a per-cluster mean. Use the evolved variant of that form described below.

**What we measured.** On an Apple M3 Max, under Metal:
- the candidate's output is bit-identical to A7's on every test case;
- it runs at **0.0274 of A7's time** (about 36 times faster), at n = 8,192 points, d = 256 dimensions and k = 1,024 clusters.

**Where the speed comes from.**
- **Almost all of it is the algorithm.** A7 gathers: each output cell loops over every point, which costs O(k·d·n). The candidate sorts the points by cluster once, then each cluster sums only its own points, which costs O(n·d). The hand-written form of this algorithm runs at 0.0278 of A7's time.
- **Evolution added 1.6 %** on top of the hand-written form (0.9839 of its time, 11 of 11 pairs faster). Most of that gain comes from removing the Kahan compensation, which Metal's compiler already removes from A7 (see Caveats).

**What we ask of the UltraDim team.**
- Review the kernel.
- Validate it in UltraDim, on a substantial dataset, on Metal and on Vulkan, including rolling k-means, the time-sensitive path.
- Decide whether to adopt it.

We have changed no UltraDim code.

> **Status, 2026-10-31 14:45Z: re-measured.** Earlier on 14:00Z we found that one WGSL text, compiled into separate pipelines, ran each entry at one of two speeds, up to about 1.7× apart. The cause was not the pipeline. It was the harness: it allocated fresh buffers for every run, and the speed followed where those buffers landed. The harness now makes each case's buffers once and rewrites them before every run (fuller 40ae60fc). With that fix, the same text compiled five times gives whole-module totals of 0.2322 to 0.2327 ms, a spread of 0.2 %. Every ratio below was measured again with this fix, with A/A pairs across separately compiled pipelines. The memory the process gets still sets the absolute speed: one process measured the scatter form at 0.878 ms rather than 0.234 ms. Both sides of a comparison run on the same buffers, so the ratios hold.

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
| Pairs | The candidate and the yardstick interleaved run by run (A, B, A, B, …), each side the fastest of its runs, at least 1 s a side, with one discarded warm-up pair; each side's runs rotate over 4 separately compiled pipelines |
| Noise floor (A/A) | The yardstick's 4 pipelines against 4 more compiled from the same text, 2 × 11 pairs; every ratio must lie within ±5 %. Measured per comparison in the table below |
| "Faster" | The median ratio below the least A/A ratio, AND at least 9 of 11 pairs faster |
| Correctness | Bit for bit, cell by cell, on all 8 cases (4 training, 4 held out) and on the timing case's own output |

**Results:**

| Comparison (11 interleaved pairs, 2026-10-31 14:18-14:42Z) | Median ratio | Pairs faster | A/A range |
|---|---|---|---|
| candidate against A7 | 0.0274 | 11 of 11 | 0.9998 to 1.0002 |
| candidate against the hand-written scatter form | 0.9839 | 11 of 11 | 0.9961 to 1.0055 |
| hand-written scatter form against A7 | 0.0278 | 11 of 11 | 0.9662 to 1.0086 |

A7 is `problems/kmeans_update_segmented/gather.wgsl`, A7's kernel unchanged in the five-entry interface. Every kernel was bit for bit A7's output on all 8 cases and on the timing case.

**What our kingdom's spelling costs.** Our evolution system rewrites each kernel into its own spelling before it times it. For the hand-written scatter form, that rewritten text runs at **1.0255 and 1.0238 of the original** (two runs, 11 of 11 pairs slower each time). For the candidate it runs at 0.9856 of the scatter form, where the original runs at 0.9839. Per-entry timings found the cause: the index clamps `a[min(i, arrayLength(&a) - 1u)]` that the kingdom puts on every read and write. With the clamps, the loop form, the break flag and the store guard all removed, the rewritten text matches the original to within 0.7 % on four entries. The fifth entry, scatter, stays 1.4 % slower (0.0295 against 0.0291 ms).

## 3. Faults we found in our own method, all fixed before these numbers

| Fault | What it did | Fix |
|---|---|---|
| Timing split into many submits | It favoured kernels with small work bounds, and didn't match UltraDim's dispatch | Timing is unsplit, one dispatch |
| The display shares the GPU | Self-timing ranged 0.92 to 1.10 when each side was timed in its own block | Run-by-run interleaving |
| Our reader re-wrote A7 1.72× slower | The kernel's `let`s were inlined into its loop | `let`s are kept and loop-invariant terms hoisted; A7 as expressed now runs at 0.9995 of its own text |
| A u32 overflow | A guard `gid.x >= (k·d)²` wrapped to 0 at the timing size, so a kernel did no work there and claimed 450×. Two evolved kernels used it. | Every timed kernel's output on the timing case is checked against the yardstick's. Both are withdrawn. |
| The dispatch limit | A mutated workgroup width needed 262,144 workgroups (the limit is 65,535) | The types bound every workgroup width |
| Fresh buffers every run | The speed followed where the buffers were allocated. One entry moved by up to 25 %, and one text compiled separately ran at 0.64 to 0.90 ms | Each case's buffers are made once and rewritten before every run (fuller 40ae60fc). The same text compiled five times now agrees within 0.2 % |

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

- **Runner:** `phylu/examples/wgsl_true_case_fit.rs`, binary `tcf_c705ef22_40ae60fc` (phylu c705ef22, fuller 40ae60fc), each comparison a P run with `--start FILE --gate`. Logs: `/private/tmp/claude-501/fbench/rt_champ_vs_scatter.log`, `rt_champ_vs_a7.log`, `rt2_scatter_vs_a7.log`, `rt2_reply2_vs_scatter.log`, `rt_per_entry.log`, `pe5.log`, `sus2.log`, `sus3.log`.
- **Problems:** `fuller/problems/kmeans_update_segmented`, `fuller/problems/kmeans_update`.
- **Evolution run:** `kmeans_P_s1`, 97 generations, 1,415 evaluations. Its per-generation table is in the run directory.
