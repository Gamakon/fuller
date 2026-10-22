# ST9: volumepassage, champion to faster champion (2026-10-22)

**The evolved champion does not draw volumepassage faster than the original.** Run the way the shipped shader runs (one submit), the generation-3,000 champion takes 1.0016x the original's GPU time at 256 x 256 and 1.0000x at 512 x 512. It is 2.2 % faster only under the tiled submits the run used, only at 256 x 256, and only when compared with the tiled encoded original. The tiling itself makes every kernel 3.0x to 3.2x slower. Every hall-of-fame kernel draws the original's picture bit for bit at every size and time tested.

The run continued to generation 3,468 before Andrew stopped it at convergence. All results here are taken at generation 3,000 (checkpoint, hall of fame and chart).

## What we did and why

The aim of ST9 (`docs/PLAN_true_gep_wgsl.md`) is regime 2, champion to better champion, in the true multi-typed GEP kingdom for WGSL. In that kingdom the types carry every legality rule, so every gene expresses a legal kernel, with no filter and no repair. We read the cuneus shader volumepassage exactly into Karva and seeded the population with equivalent programs from fuller's e-graph. Evolution was then asked for a kernel that draws the same picture in less GPU time.

## How

- **Kingdom.** The volumepassage image instance of the true WGSL kingdom (`kingdoms/wgsl_true/img/volumepassage`, spec batch 7: storage-texture output, uniform-block field loads, fuel budget K = 2). It frames 232 types.
- **Encoding.** The original is a tree of 492 nodes (294 functions). It is written into one gene of 2,875,505 words (program size N = 294) by the section 5 write-back. Rendered from that gene, it draws 0 differing channels against the shipped shader.
- **E-graph seeds.** egglog, 35 row-to-row rules (commute, fma, mix, sub, square, scale, per lane count), 3 iterations: 347 classes, 55 with a choice, 63 programs, all written into genes. The original plus 31 make the first population (half through one homologous crossover); the pump injects the other 32.
- **Fitness.** HFF (TrueNorth, library, normalisation off, 0 best, about 1.57 worst), 29 inputs: six picture statistics for each of two fixed cases (t = 5 and 8, 256 x 256), five interleaved timed pairs against the tiled encoded original (each ratio as r / (1 + r)), and the six statistics for two held-out cases (t = 2; 200 x 136 at t = 5).
- **Variation and selection.** Population 32, elites 2, tournament of 3, LOW change on every child (2 expressed cells a child expected), crossovers on. The diversity dial had threshold 0.25 and step 0.05; it stayed at 0 for the whole run. The pump replaced the 2 worst individuals every 50 generations with hall-of-fame and e-graph genes. The hall of fame kept 5 distinct trees.
- **Work bound and tiling.** `TrueKingdom::tree_work_bound` gives 36,658 node executions per invocation for the original and for every hall-of-fame tree. Under the per-submit cap, a pass is split into submits of whole workgroups (16 at 256 x 256, 64 at 512 x 512), and the harness waits for each submit to finish before the next.

## Where the code is

- fuller: `src/wgsl/image.rs` (harness, tiling, timing), `src/wgsl/cuneus_originals.rs`, `src/wgsl/true_kingdom.rs` and its `egraph` step, `kingdoms/wgsl_true/image.py`, `examples/cuneus_image.rs`. The timing for this note is `examples/st9_champion_time.rs`.
- phylu: `src/evolve/true_wgsl_fit.rs`, `src/evolve/image_fitness.rs`, `examples/wgsl_true_image_fit.rs`, `scripts/st8_chart.py`.
- Commits: fuller 737441fc, ff0260ec, 110e9ef1, 47394d7b, faab70a1, 660b64ff, 37cc49cb (the run's binary); phylu 818b0340, dedab1ad.

## How it was run, and how to restart

`run10k.sh` (copied here) took the GPU lock and ran this command under a 30-hour guard:

    wgsl_true_image_fit_run10k --shader volumepassage --size 256 --pop 32 --generations 10000 --seed 1 \
        --pump-every 50 --checkpoint-every 500 --chart-every 500 --out /private/tmp/claude-501/st9p2/run10k

Started 01:54:38Z; generation 3,000 at about 07:31Z (5.6 h, median 6.6 s a generation); stopped at generation 3,468, 08:23:55Z. Restart: the same binary and arguments plus `--resume` (reads `checkpoint.bin` and `hof.bin` in `--out`).

Timing, `time.sh` (61 rounds, seed 1, 08:30:51Z) and `time_rep.sh` (151 rounds, seed 2, 08:31:22Z), on fuller 37cc49cb plus the example:

    st9_champion_time --hof /private/tmp/claude-501/st9p2/run10k/hof.bin --rounds 151 --sizes 256,512 --seed 2 --out DIR

## Where every artefact is

- Run: `/private/tmp/claude-501/st9p2/run10k/` (`checkpoint.bin` and `hof.bin` at generation 3,000; `ledger.tsv` and `series.tsv` to 3,468; `chart.png`; `hall_of_fame/`). Log `/private/tmp/claude-501/st9p2/run10k.log`, binary `/private/tmp/claude-501/st9p2/wgsl_true_image_fit_run10k`, 50-generation trial `/private/tmp/claude-501/st9p2/run50*`.
- Timing: `/private/tmp/claude-501/st9time/` (outputs, logs, binary).
- Here: the chart, the original and encoded pictures (512 x 512), the generation-3,000 hall-of-fame strip and animation, `frame_t{2,5,8}_original_vs_champion.png` (#0 original, #1 champion), `series_to_g3000.tsv`, `hall_of_fame.tsv`, `timing_61/`, `timing_151/`, `kernels/` (WGSL of the encoded original and ranks 1 to 5, the champion's diff).

## Findings

### Speed (151 interleaved rounds, random order per round, GPU timestamps, t = 5)

GPU ms per pass:

| size | kernel | submits | median | p10 | p90 |
|---|---|---|---|---|---|
| 256 | original, shipped | 1 | 0.2610 | 0.2607 | 0.2615 |
| 256 | encoded original | 1 | 0.2610 | 0.2607 | 0.2615 |
| 256 | champion (rank 1) | 1 | 0.2615 | 0.2612 | 0.2620 |
| 256 | encoded original, tiled | 16 | 0.7929 | 0.7441 | 0.8116 |
| 256 | champion, tiled | 16 | 0.7736 | 0.7277 | 0.7920 |
| 512 | original, shipped | 1 | 0.9728 | 0.9723 | 0.9777 |
| 512 | encoded original | 1 | 0.9698 | 0.9693 | 0.9751 |
| 512 | champion (rank 1) | 1 | 0.9728 | 0.9723 | 0.9781 |
| 512 | encoded original, tiled | 64 | 3.1618 | 3.0205 | 3.2209 |
| 512 | champion, tiled | 64 | 3.1434 | 3.0048 | 3.2041 |

Paired ratio (candidate over reference, per round). The interval is a 95 % bootstrap over rounds; the sign test is two-sided.

| size | candidate / reference | median | 95 % | faster / slower rounds | sign p |
|---|---|---|---|---|---|
| 256 | champion / shipped, one submit | 1.0016 | 1.0014-1.0019 | 16 / 132 | 7e-24 |
| 512 | champion / shipped, one submit | 1.0000 | 0.9998-1.0001 | 79 / 72 | 0.63 |
| 256 | champion / encoded, tiled | 0.9777 | 0.9723-0.9806 | 124 / 27 | 5e-16 |
| 512 | champion / encoded, tiled | 0.9994 | 0.9919-1.0028 | 77 / 74 | 0.87 |
| 256 | encoded tiled / shipped (tiling cost) | 3.030 | 3.014-3.043 | 0 / 151 | 7e-46 |
| 512 | encoded tiled / shipped (tiling cost) | 3.240 | 3.217-3.250 | 0 / 151 | 7e-46 |

Ranks 2 to 5, one submit against the shipped original, fall between 0.9977 and 1.0027 at both sizes. Under tiling they fall between 0.9775 and 1.0013. The 61-round run with seed 1 gives the same picture: the champion is 0.978 under tiling at 256 (p 0.002) and 1.0019 as one submit (`timing_61/ratios.tsv`).

All 91 picture checks show 0 differing channels: 13 kernel-and-mode combinations, at 256 and 512 for t = 2, 5 and 8, and at 200 x 136 for t = 5.

### What changed in the program (`kernels/champion_vs_encoded.diff`)

- **Workgroup 8 x 8 became 2 x 16.** This is the only change that can plausibly affect speed. Its one measured effect is the 2.2 % under 16 tiled submits at 256 x 256.
- **Dead writes.** The unused step counter `rf3 += 1` became a write to an unused register, and the reset `rf0 = 0` (already 0 at declaration) became a write to another unused register.
- **Literal replaced by a parameter.** `sin(rf2 * 1.0 + t)` became `sin(rf2 * params.speed + t)`. Ranks 2 to 5 do the same with `speed` or `color3_b`. Both are 1.0 at the host's defaults, so these kernels draw the original's picture only at the default parameters.
- **A changed `select`.** The wrap of `C1.z % 4` takes the other register in a branch never reached when `C1.z >= 0`, which holds for non-negative time.
- **One e-graph rewrite:** `0.983729 * C5 + 0.432951` became `fma(...)`, most likely the same code after Metal's compiler.

### Learning and diversity

- **Best HFF** went from 0.29242 at generation 0 to 0.26826 at generation 2,266, in 13 steps, nearly all before generation 250 (0.26962). The mean stayed near 1.0, because LOW children often draw a different picture.
- **Diversity:** a median of 28 of 32 distinct trees (16 to 32), and a median pair overlap of 0.21. The dial never moved.
- **Fitness evaluations:** 85,437, with 0 device failures. 4,943 of them drew the identical picture at t = 5. The pipeline cache hit rate was 10 %.

### What HFF rewarded

Once the picture inputs were all 0, only the five timed-pair ratios separated the individuals. Those ratios were measured under tiling at 256 x 256, so selection kept the one kernel that is faster in that setting (rank 1, 2.2 %, reproducible). It also kept four that are not reliably faster.

### Open points

- **Bimodal GPU clock.** One pass swings between about 0.8 and 2.9 ms (256, tiled); five interleaved pairs an individual is a noisy input.
- **Tiling overhead.** A wait after each submit costs 3x, about 100 times any effect evolution found.
- **Karva shift.** A LOW child shifts the reading frames, so its tree overlaps its first parent's only about 0.5.
- **Held-out cases never vary the parameters**, so a literal replaced by a parameter equal to it passes.
- **E-graph rewrites are probably cosmetic** to Metal's compiler: the seeds differ in text, likely not in machine code.

## Next steps

ST8b, leaf-skewed mutation (edits weighted towards the leaves of the expressed tree, so a child changes little), is being built. For the next speed run: time as one submit (or all tiles in one command buffer), add held-out cases with other parameters and a negative time, and take more timed pairs an individual.
