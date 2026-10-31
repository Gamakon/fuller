# The k-means champion of seed 1: what evolution changed

The fastest exact kernel evolved for UltraDim's k-means centroid update: run `kmeans_P_s1` (regime P, 1,415
evaluations, 97 generations, 2026-10-31), `champion.wgsl` and its dispatch `champion.dispatch.toml` beside this file.

## Context

- **Objective.** A faster drop-in for UltraDim's A7 `update_centroids` (`docs/kmeans_update_dropin.md`): the same
  bindings, the same values, a dispatch UltraDim can issue.
- **Approach.** Regime P on the scatter interface (`problems/kmeans_update_segmented`, five entry points dispatched in
  order). Generation 0 was A7 (`gather.wgsl`), the hand-written scatter form (`segmented.wgsl`) and A7's point mutants.
  Every kernel is spelled through the share pass; speed was timed unsplit, as UltraDim dispatches, in interleaved pairs.
- **Status.** Re-timed 2026-10-31 11:02-11:08Z, M3 Max, Metal, timing case n = 8,192, d = 256, k = 1,024; 11
  interleaved pairs each, A/A gates passed:

| measurement | ratio (champion / yardstick) | pairs faster | A/A range |
|---|---|---|---|
| against A7 (`gather.wgsl`, UltraDim's text) | 0.0273 | 11 of 11 | 0.9992-1.0003 |
| against the hand-written scatter form (`segmented.wgsl`) | 0.9840 | 11 of 11 | 0.9966-1.0039 |

- Its outputs are A7's bit for bit on all 8 constraint cases (0 cells differ). On the timing case they are within the
  tolerance of the yardstick's: 0 % of cells outside.

## Against A7

A7 is a gather: one invocation per (cluster, dimension) cell scans all n points and sums those assigned to its cluster,
O(k n d). The champion is the scatter form evolution started from beside A7 (a stable counting sort of the points by
cluster, then one invocation per cell summing only its cluster's points, O(n d)). Evolution did not invent the
scatter. It made the scatter form 1.6 % faster.

## Against the hand-written scatter form

1. **Kahan's compensation is gone in `mean`, the main cost.** The source subtracts the running compensation
   (`y = term - sum_c`). In the champion the subtraction reads a register that is never written (`rf21`, always 0),
   so the sum is a plain sum. The compensation is still computed, but nothing reads it. This is the likely source of
   the gain.
   - The values stay A7's bits on Metal, whose compiler already removes Kahan's compensation:
     `docs/reports/2026-10-28_metal_removes_a7_kahan.md` measured A7's plain sum equal to A7 on 8 of 8 cases.
   - Vulkan and other backends are not measured. A drop-in there needs that check first.
2. **`hist` runs at workgroup width 128**, not 256.
3. **Rewrites that change nothing:**
   - `scatter`'s block-prefix index reads the loop's final counter (256 after the loop) where the source has the
     literal 256;
   - knob constants changed that no line reads (`offsets` KTILE 256, `scatter` BCOL 8);
   - one dead store at the end of `mean`.

## Dropping it into UltraDim

As `docs/kmeans_update_dropin.md`, five entries over one explicit layout of five bindings:
- the scratch buffer: `n + 2k + 2 ceil(n/256) k` u32 words, zeroed per update;
- the dispatches in order (`champion.dispatch.toml`), with `hist` at width 128.
