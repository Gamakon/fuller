# nluca generated_step tuned per width (2026-10-14 17:18Z to 2026-10-15 01:41Z)

**Result: an evolved kernel is faster than nluca's shipped kernel at four of the five widths: 0.83x its time at 10,000 rows, 0.87x at 1,048,576, 0.89x at 500,000 and at 1,000 (paired, 10 rounds, every 95 % upper bound under 0.92). At 100,000 rows no faster kernel was found.**

| width (rows) | shipped, ms | tuned, ms | ratio [95 %], three datasets | the tuned kernel |
|---|---|---|---|---|
| 1,048,576 | 25.27 / 25.33 / 25.34 | 21.91 / 21.94 / 21.94 | 0.867 [0.864, 0.870], 0.864 [0.862, 0.869], 0.866 [0.864, 0.869] | f1's design |
| 500,000 | 11.43 / 11.42 / 11.43 | 10.21 / 10.19 / 10.21 | 0.893 [0.892, 0.895], 0.892 [0.891, 0.895], 0.893 [0.891, 0.895] | f1's design |
| 100,000 | 3.89 / 3.61 / 3.34 | 4.17 / 3.62 / 3.35 | 1.031 [0.998, 1.092], 1.030 [0.998, 1.063], 1.001 [0.962, 1.041] | the shipped kernel, 5 lines changed: keep shipped |
| 10,000 | 1.540 / 1.539 / 1.541 | 1.295 / 1.279 / 1.284 | 0.839 [0.831, 0.844], 0.833 [0.826, 0.841], 0.833 [0.829, 0.838] | an unrolled design (a Sonnet "beat both" child at gen 2, a rule at gen 3) |
| 1,000 | 1.429 / 1.443 / 1.453 | 1.248 / 1.281 / 1.289 | 0.883 [0.862, 0.918], 0.886 [0.872, 0.892], 0.890 [0.879, 0.908] | the unrolled design again (20 lines from the 10,000-row one, found in its own run) |

ms: one 64-step run of the kernel, the median of 10 paired rounds (ABBA), each width timed on its own three datasets (seeds k1, k2, k3). Every tuned kernel passed nluca's oracle on all 17 cases of its problem: the 14 cases at widths 4,096 and 65,536 (noise on, loc, scale, leak and u varied) and the width's own three.

## The two designs

- **f1's design** (large widths): the codebook copied into workgroup memory once a workgroup, the ten neighbour indices loaded before the loop, the row hash computed once outside it, the early return replaced by a guarded store.
- **The unrolled design** (small widths): the ten-neighbour loop unrolled, no workgroup copy of the whole codebook. At small widths a workgroup's copy of 1,024 levels costs more than the reads it saves.

Both came the same way: a model rewrite made the leap (f1: Sonnet's "better memory locality" rewrite of an incorrect parent; 10,000 rows: Sonnet's "beat both parents" child of two correct ones), the rules polished it.

## What had to be fixed first (each fixed, committed, tested)

| fault | effect | fix |
|---|---|---|
| The per-width problems' correctness cases all had the noise off | the first 1M winner deleted the noise hash and still passed | every width's constraint now includes the 14 cases at 4,096 and 65,536 rows |
| The fitness timed a kernel against the shipped kernel's least time so far | the GPU's speed drifts up to 2x between minutes; the first 1,000-row run returned a kernel 1.10x slower than shipped | paired timing: the shipped kernel timed beside each candidate, ABBA (fuller e834674) |
| A 100 ms sample at small widths is hundreds of runs, each rebuilding every buffer | 50 s an evaluation at 10,000 rows | 20 ms samples at widths up to 10,000; then the repeats in one GPU submission (fuller 6c6179a: 17.1 s to 9.3 s for the same scoring at 10,000 rows, HFF and GPU ms unchanged) |
| The safety checker gave a stepped loop counter any value | a strided codebook preload reported as a workgroup race: 10 of the 26 starting kernels, 11 of 21 dropped model rewrites | the counter is its start plus its step times the pass, when the loop's exit test bounds it (fuller 8f9105a, 6cfdaf0) |
| The laptop slept from 18:10Z to 21:56Z | the 1,000-row run took 3 h 48 min of wall for 9 min of work | every run under caffeinate; a 6-hour system-sleep block for the night |

## Caveats

- One seed a width. The 1,048,576 and 500,000-row runs scored with the least-time fitness and the old checker; their winners' speed is confirmed by the paired re-timing above. The 100,000-row run used the old checker, which dropped the strided Sonnet kernels the unrolled design comes from; it is being rerun with the fix.
- Not yet measured: each winner at every other width (the size table, queued), so the choice of kernel by width is not yet made.

## The open options, measured

- **f16** (emulated by pack2x16float rounding; fuller's wgpu 0.20 front end has no f16): every variant fails nluca's oracle at the first case (tolerance 1.0e-4): the codebook stored in f16 2.24e-3, the sums in f16 8.13e-4, all arithmetic in f16 3.24e-2. The f16 codebook (real f16 storage) is 0.990 [0.988, 0.993] of f1's time at 1M rows: 1 %. f16 does not pay for nluca.
- **Stable section ids (homologous section crossover, fuller `sections`)**: on nluca's 26 kernels, 1,000 children an operator, seeds 1-3, children new, clean and legal: section 994 / 994 / 982, Karva one-point 6 / 7 / 6, two-point 27 / 28 / 28, gene 15 / 13 / 7. Karva children are copies of a parent 61-82 % of the time.
- **E3 in a run: section crossover against typed crossover** (nluca, 1,048,576 rows, 10 generations, seeds 1-3, everything else identical; 01:41Z to 03:03Z). **Section crossover made correct children 2.3x as often and the faster best kernel at every seed.**

| seed | typed: crossover children correct | typed: best, paired ratio (3 datasets) | section: crossover children correct | section: best, paired ratio |
|---|---|---|---|---|
| 1 | 8 of 108 | 0.858 / 0.866 / 0.865 | 13 of 93 | 0.819 / 0.819 / 0.827 |
| 2 | 3 of 92 | 0.859 / 0.864 / 0.843 | 14 of 87 | 0.855 / 0.839 / 0.834 |
| 3 | 8 of 84 | 0.826 / 0.838 / 0.832 | 15 of 96 | 0.830 / 0.825 / 0.805 |
| all | 19 of 284 (6.7 %, Wilson 95 % [4.3, 10.2]) | mean 0.850 | 42 of 276 (15.2 %, [11.5, 19.9]) | mean 0.828 |

Crossover children: the crossover pairs, champion mating and the duplicate re-cross, all three made by the arm's crossover. Section seed 1's best (0.82) is itself a section child (champion mating, gen 3). The best kernel of the night, section seed 3, is 0.805-0.830 of shipped's time at 1M rows, against f1's 0.866. Section crossover is now `wgsl_tune`'s default (phylu 5a469806).
