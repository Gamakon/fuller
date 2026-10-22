# ST9 addendum: LEAF mutation at real speed (2026-10-22)

**With LEAF mutation, a child keeps 0.998 of its parent's tree, against about 0.5 under LOW in the morning run.** Best HFF moved from 0.29375 to 0.26305 in 500 generations (last step at generation 123), at 64 x 64 where every kernel is timed as one submit. The population also became near-identical: median pair overlap 0.998, against 0.21 in the morning run.

## Why

The morning's ST9 note (`README.md`, this directory) measured three problems.

- **The champion is not faster.** Its HFF gain came from the timed pairs, which were measured under 16 tiled submits at 256 x 256. Its one plausible speed change (workgroup 8 x 8 became 2 x 16) helped only there: 0.978x tiled, 1.0016x as one submit.
- **A literal became a parameter.** `sin(rf2 * 1.0 + t)` became `sin(rf2 * params.speed + t)`. It passes because no fitness case varies the parameters, and `speed` is 1.0 at the host's defaults.
- **Mutation was destructive.** A LOW child's tree overlapped its first parent's only about 0.5. A Karva edit that changes a node's argument signature shifts every later read of every type frame, so one edit rewrites much of the expressed tree. The median pair overlap in the population was 0.21.

## What changed

### LEAF mutation (phylu 231da1f4, `src/evolve/typed_vary.rs`, `typed_vary.wgsl` kernel `typed_leaf_main`, `src/evolve/true_wgsl_fit.rs`)

- **Positions.** Only the parent's expressed nodes are edited, read by fuller's typed walk.
- **Weight.** Each position has weight `(1/s)^alpha`, where `s` is its Horton-Strahler order (a leaf is order 1). `--leaf-weight depth` gives `(depth+1)^alpha` instead. With alpha = 2, a leaf is 4 times as likely as an order-2 node. Edits are drawn without replacement (`--leaf-edits`, default 1).
- **Replacement.** A symbol of the same output type with the identical ordered argument-type signature. An expressed `?_T` takes a new `C_T` index. A symbol alone in its class is absent from the draw.
- **Legality.** The replacement occupies the same cell, has the same output type and reads the same argument types in the same order. No later read of any type frame moves, so the child's tree is the parent's tree with exactly the edited nodes relabelled. Every symbol drawn is in the cell's typed domain, the same domain LOW draws from, with a different distribution. So every child is a legal kernel by construction, with no filter and no repair. The host test checks this on 10,000 genes (exact edit: the child differs in exactly the edited nodes).
- **Schedule.** LEAF is a fourth strength of the ST8 schedule (`--mutation-leaf P` is its share of children). A LEAF child takes no other mutation; crossover still pairs it. The diversity dial lowers the exponent to `alpha * (1 - dial)` when diversity falls, which spreads edits towards the root.

### Real-speed timing at 64 x 64

At 64 x 64 the whole pass fits under the per-submit work cap, so every kernel runs as one submit, as the shipped shader runs. The log: `encoded original ... 1 submits a pass` at every case; shipped original 0.174 ms, encoded original 0.182 ms GPU median (t = 5). The timed pairs in HFF are therefore measured in the setting where a speed-up would be real.

### Population dump (`write_dump`, format also in `docs/PLAN_true_gep_wgsl.md` ST8b)

Every `--dump-every N` generations, `OUT/dump/gNNNNNN.bin`, little-endian u32 words:

- **Header:** magic `ST8BDMP1` (2 words), version 1, generation, rows, framed types `n_t`, buckets `B` (1024), fitness inputs `I` (29), record words, cell words `C`; then per framed type its type code, its width `W_T` (`h_T + t_T` cut to `--dump-type-width`, 256) and `h_T + t_T`.
- **Record per individual:** generation, row, HFF (f32), picture exact (1, 0 or unknown), GPU ms (f32), born generation, hall of fame flag, cells cut, expressed nodes; then three vectors:
  1. **expressed cells:** per type, the symbols of its expressed nodes in frame order (`C` words, padded);
  2. **subtree-hash bag:** every expressed node's subtree hash counted in bucket `hash % 1024`;
  3. **the 29 HFF inputs** (f32).

UltraDim reads these to map the population: HDBSCAN and k-means for clusters (species), UMAP for a picture of the population over generations, and novelty detection for individuals far from every cluster. The dump is the input for measuring diversity in genotype (vectors 1 and 2) and in behaviour (vector 3) without choosing a distance in advance.

## How it was run

`/private/tmp/claude-501/st9p3/run.sh` took the GPU lock and ran, from phylu, under an 18-minute guard:

    wgsl_true_image_fit_leaf --shader volumepassage --size 64 --pop 10 --generations 500 --seed 3 \
        --mutation-leaf 1.0 --leaf-alpha 2 --pump-every 50 --checkpoint-every 100 --chart-every 100 \
        --dump-every 1 --out /private/tmp/claude-501/st9p3/run

Started 08:37:20Z on fuller e36faaee, phylu dedab1ad plus the ST8b working tree (committed as phylu 231da1f4 at 08:37:35Z). Binary `/private/tmp/claude-501/st9p3/wgsl_true_image_fit_leaf`, log `/private/tmp/claude-501/st9p3/run.log`, output `/private/tmp/claude-501/st9p3/run/` (`series.tsv`, `ledger.tsv`, `chart.png`, `hall_of_fame/`, `dump/`). Every child was a LEAF child (share 1.0), one edit each. Fitness is HFF as in the morning run (29 inputs: 2 fixed cases, 5 timed pairs, 2 held-out cases), at 64 x 64. Elites, tournament, crossover and the pump were as in the morning run except population 10.

## Results

Run ended 08:51:03Z, exit 0, 500 generations (0 to 500), about 1.6 s a generation. Pipeline cache: 367 hits, 3,630 misses (9.2 %).

HFF (0 best, about 1.57 worst), from `series.tsv`:

| generation | best HFF | mean HFF | distinct trees of 10 | median pair overlap |
|---|---|---|---|---|
| 0 | 0.29375 | 0.47823 | 10 | 0.724 |
| 100 | 0.27036 | 0.84108 | 7 | 0.998 |
| 200 | 0.26305 | 0.74413 | 6 | 0.998 |
| 300 | 0.26305 | 0.83405 | 8 | 0.996 |
| 400 | 0.26305 | 0.74149 | 6 | 0.998 |
| 500 | 0.26305 | 0.73550 | 8 | 0.998 |

- **Best line:** four values, 0.29375 (g0), 0.29105 (g4), 0.27036 (g11), 0.26305 (g123), then flat to 500.
- **Parent overlap:** the median child's tree overlap with its first parent is 0.998 in every one of the 500 generations (morning run, LOW: about 0.5). Cells changed per child: median 1.
- **Population diversity:** median pair overlap 0.998 over the run, median 7 distinct trees of 10 (morning run: 0.21, 28 of 32). The diversity dial did not move; alpha stayed 2.00.
- **Picture-exact share:** 1,480 of 5,010 population rows in the dumps (29.5 %) draw the original's picture exactly. This counts rows of the scored population each generation, not children separately.
- **Mean HFF** stays between about 0.74 and 1.04 after generation 0, so most rows still draw a different picture even with one leaf edit.
- The champion at generation 500 was not re-timed against the shipped original outside the run (no GPU use for this note).

Chart: `addendum_chart.png` (best and mean HFF against generation). Hall of fame at generation 500: `addendum_hof.png`. Dumps: `/private/tmp/claude-501/st9p3/run/dump/` (501 files).

## Limits

- **Population 10, one seed (3), 500 generations.** The morning run had population 32 at 256 x 256 for 3,468 generations, so the two runs are not a paired comparison.
- **No A/B.** LEAF was not run against LOW at the same settings (size 64, population 10, seed 3). Any difference from the morning run mixes the operator with the size, the timing mode and the population.
- **Crossover was still on.** A child's change from its first parent includes crossover; the parent-overlap figure covers both.
- **Parameters still never vary across fitness cases**, so a literal replaced by an equal parameter still passes.
- **One GPU** (Apple M3 Max, Metal). The 64 x 64 pass takes about 0.18 ms, so the timed pairs are short and launch overhead is a larger share of each.

## Next steps

1. **A/B of operators, several seeds.** LEAF against LOW against the classic GEP operators already in the typed core: one-point, two-point and gene recombination, IS and RIS transposition, each alone and in Ferreira's mix. Same kingdom, size, population and generations; at least five seeds each; best and mean HFF against generation, parent overlap, picture-exact share of children, and distinct trees.
2. **Vary the parameters across fitness cases.** Add held-out cases with other values of `speed`, the colour fields and a negative time, so a literal replaced by a parameter no longer draws the same picture.
3. **NVIDIA/Vulkan timing target.** Repeat the timing on an NVIDIA GPU under Vulkan (the Runpod 4090 build), so a speed-up is measured on a second compiler and device.
4. **Map the dumps.** Run UltraDim over `run/dump/` (clusters, UMAP over generations, novelty) and report how many species the population holds and when they appear.
5. **Larger size in one submit.** Time at 256 and 512 with all tiles in one command buffer, so larger pictures are timed without the 3x wait cost.

## Speed of the champion

**The generation-500 champion is not meaningfully faster than the original: 0.33 % faster at 64 x 64 (interval reaches 1.000, sign p 0.087), 0.81 % faster at 128 x 128 (sign p 0.0099), and its three edits do not remove any work.**

How it was run: `timing_leaf/go.sh` waited for the run's `## 08:51:03 exit 0`, took the GPU lock at 08:51:04Z, ran fuller `st9_champion_time` (the 31da0d71 binary, unchanged) and released the lock at 08:51:07Z. Input: the top 5 of the final hall of fame (`timing_leaf/top5.py` keeps the 5 lowest HFF, 0.2631 to 0.2892), 61 rounds, every kernel once a round in a random order (seed 3), Apple M3 Max, Metal, t = 5. Outputs are in `timing_leaf/` (`times.tsv`, `summary.tsv`, `ratios.tsv`, `pictures.tsv`, each kernel's WGSL, `time.log`).

Submits: every hall-of-fame tree has a work bound of 36,659 per invocation against the original's 36,658, so the tool will not time any of them as one submit (`Fuel::Proven`). At 64 x 64 the tiled mode is a single submit, the same as for the original. At 128 x 128 tiled means 4 submits, so the 128 ratios compare 4-submit candidates with the 4-submit encoded original, not with the shipped shader's single submit.

| size | kernel / reference | submits | GPU ms median (cand / ref) | ratio median | 95 % bootstrap | faster / slower of 61 | sign p |
|---|---|---|---|---|---|---|---|
| 64 | rank1 / encoded | 1 / 1 | 0.0503 / 0.0506 | 0.9967 | [0.9950, 1.0000] | 36 / 22 | 0.087 |
| 64 | rank2 / encoded | 1 / 1 | 0.0504 / 0.0506 | 0.9972 | [0.9951, 1.0000] | 37 / 23 | 0.092 |
| 64 | rank3 / encoded | 1 / 1 | 0.0506 / 0.0506 | 0.9971 | [0.9942, 1.0024] | 35 / 25 | 0.245 |
| 64 | rank4 / encoded | 1 / 1 | 0.0501 / 0.0506 | 0.9945 | [0.9913, 0.9967] | 44 / 17 | 7.3e-4 |
| 64 | rank5 / encoded | 1 / 1 | 0.0505 / 0.0506 | 1.0025 | [1.0000, 1.0041] | 19 / 37 | 0.022 |
| 64 | encoded / shipped | 1 / 1 | 0.0506 / 0.0480 | 1.0518 | [1.0498, 1.0550] | 7 / 54 | 4.3e-10 |
| 128 | rank1 / encoded | 4 / 4 | 0.1994 / 0.2010 | 0.9919 | [0.9894, 0.9935] | 41 / 20 | 0.0099 |
| 128 | rank2 / encoded | 4 / 4 | 0.2007 / 0.2010 | 0.9986 | [0.9957, 1.0017] | 36 / 25 | 0.200 |
| 128 | rank3 / encoded | 4 / 4 | 0.2004 / 0.2010 | 0.9944 | [0.9905, 0.9983] | 43 / 18 | 0.0019 |
| 128 | rank4 / encoded | 4 / 4 | 0.1999 / 0.2010 | 0.9946 | [0.9932, 0.9975] | 42 / 19 | 0.0044 |
| 128 | rank5 / encoded | 4 / 4 | 0.1993 / 0.2010 | 0.9893 | [0.9862, 0.9928] | 42 / 19 | 0.0044 |
| 128 | encoded (4) / shipped (1) | 4 / 1 | 0.2010 / 0.0868 | 2.3099 | [2.2900, 2.3162] | 0 / 61 | 8.7e-19 |

The encoded original itself is 5.2 % slower than the shipped shader at 64 x 64, both as one submit. Every champion ratio above is measured against the encoded original, not the shipped shader.

Pictures: the differing channels against the shipped original are **0 at t = 2, 5 and 8** for all 5 ranks, at 64 x 64 (16,384 channels) and 128 x 128 (65,536), and 0 at the held-out 200 x 136, t = 5. The maximum absolute error is 0 everywhere.

WGSL diff, champion (rank 1) against the encoded original. One line of the body differs, at three places:

    - rf3 = (rf3 + lit_f32(1.0));
    + rf3 = (rf22 + lit_f32(1.0));                         // rf22 is never written (0.0), so rf3 = 1.0; rf3 is never read
    - cos(((w3[1] * lit_f32(1.1)) + cos((w3[0] * lit_f32(2.3)))))
    + cos(((w3[1] * lit_f32(1.1)) + cos((lit_f32(2.3) * w3[0]))))   // operands swapped
    - C7 = ((C5 * ((lit_f32(0.983729) * C5) + ...
    + C7 = ((C5 * ((C5 * vec3<f32>(lit_f32(0.983729))) + ...       // operands swapped, scalar splatted

None of the three edits removes an operation from the 50 x 7 inner loop. The +1 in the work bound comes from the splat. Ranks 2 to 5 differ from the encoded original by edits of the same kind (`rf3` reading another unwritten register, such as `rf62` in rank 5, and the same swaps). The 0.3 to 1.1 % differences are inside the spread one kernel shows between rounds (64 x 64: p10 0.050 ms, p90 0.186 ms, so the GPU clock is bimodal) and they change sign between ranks (rank 5 is 0.25 % slower at 64 x 64). The kernels draw identical pictures, and no rank is faster at both sizes with an interval that excludes 1.
