# STUDY summary: can evolution deliver the benchmarked GPU challenges?

## Context

**Objective.** Find out whether our typed GEP system, given only examples of a published GPU benchmark's inputs and outputs, evolves a program that computes it exactly (H1), does so better than random search at the same budget (H2), and, given the published program, evolves a faster equivalent (H3). The plan, with hypotheses and expectations written before any run, is `docs/STUDY_evolving_benchmarks.md`.

**Approach.**
- Four benchmarks from `diku-dk/futhark-benchmarks`: reduce, scan, histogram, pathfinder.
- Candidates are expressed by the Futhark signature kingdom and run on our wgpu/Metal path.
- Three regimes:
  - R: evolution from random, 3 seeds;
  - C: random search, the same budget of 1,400 evaluations, 3 seeds;
  - P: from the published program toward faster, 1 seed, 60 min.
- Every P run is preceded by an A/A timing gate and a planted-slowdown control.

**Status.**
- Done, 2026-10-28 16:50Z: 30 counted runs, 24 champion rescores, 4 gates.
- One benchmark has no speed result: reduce's timing gate failed.
- Per-benchmark reports:
  - `2026-10-27_STUDY_reduce.md`
  - `2026-10-27_STUDY_scan.md`
  - `2026-10-27_STUDY_histogram.md`
  - `2026-10-27_STUDY_pathfinder.md`

## The answer, in plain English

**Can evolution deliver the benchmarked challenges?**
- **Only the two smallest, and random search delivers them as well.** Reduce and scan are 2-node programs. Both regimes found them exactly from examples in 3 of 3 seeds, random search slightly sooner. Histogram (7 nodes) and pathfinder (24 nodes) were found exactly by neither regime in 1,400 evaluations.

**Does it get there?**
- **No, not on any benchmark that needs more than a lookup.**
- On histogram, random search got as close as evolution or closer.
- On pathfinder, evolution's two best seeds got clearly closer than random search, and its third did not move at all.

**Does it work, i.e. did we beat the published program?**
- **No benchmark was beaten by evolution.** Scan and histogram: matched, the champion equal to the published program within the timing noise. Reduce: undecided.
- **Pathfinder is beaten, but not by the search.** The kingdom's expression of the published algorithm (clamped indices, `tabulate`, no edge branches) runs 5.5 % faster than the published port, 22 of 22 timed pairs faster. Evolution, starting from it, returned a program 3 % faster than the published port and 2.5 % slower than its own start.

## The relative score, every run

How to read this table:
- **Correctness:** the champion's quality inputs, as a mean over the stored cases. The published program's are all 0: `wrong_share` is the share of cells not exact, `outside_50pct` the share off by more than half the range, `mean_abs_over_scale` the mean error over the range (e/(1+e)), `order_wrong` the share of neighbouring pairs in the wrong order.
- **Speed:** the champion's GPU time over the published program's, the median of 11 interleaved pairs, measured only when the champion is exact.
- **Verdicts:**
  - "beat" needs the median below the least A/A ratio AND at least 9 of 11 pairs faster;
  - "matched" is exact and within the noise;
  - "not reached" is not exact.
- **Which champion:** for R the run's best; for C its best-ever, from the hall of fame; for P the run's best.
- **Rescoring:** the R and C champions were all rescored under the final protocol, fuller 3b470301, so their speed ratios are comparable with P's.

| benchmark | regime | seed | evaluations | exact | first exact at | wrong_share | outside_50pct | mean_abs_over_scale | order_wrong | champion / published (median of 11) | pairs faster | least A/A ratio | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reduce | R | 1 | 72 | yes | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.9850 | 8 of 11 | 0.9936 | matched; speed undecided (gate failed) |
| reduce | R | 2 | 118 | yes | 118 | 0.000 | 0.000 | 0.000 | 0.000 | 4.5602 | 0 of 11 | 0.9951 | exact but slower; speed undecided (gate failed) |
| reduce | R | 3 | 119 | yes | 119 | 0.000 | 0.000 | 0.000 | 0.000 | 2.3915 | 0 of 11 | 0.9951 | exact but slower; speed undecided (gate failed) |
| reduce | C | 1 | 19 | yes | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0005 | 4 of 11 | 0.9960 | matched; speed undecided (gate failed) |
| reduce | C | 2 | 127 | yes | 127 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0051 | 5 of 11 | 0.9973 | matched; speed undecided (gate failed) |
| reduce | C | 3 | 60 | yes | 60 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0006 | 4 of 11 | 0.9932 | matched; speed undecided (gate failed) |
| scan | R | 1 | 57 | yes | 57 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0006 | 2 of 11 | 0.9993 | matched |
| scan | R | 2 | 108 | yes | 108 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0000 | 3 of 11 | 0.9980 | matched |
| scan | R | 3 | 41 | yes | 41 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0002 | 5 of 11 | 0.9977 | matched |
| scan | C | 1 | 42 | yes | 42 | 0.000 | 0.000 | 0.000 | 0.000 | 0.9998 | 6 of 11 | 0.9984 | matched |
| scan | C | 2 | 46 | yes | 46 | 0.000 | 0.000 | 0.000 | 0.000 | 0.9998 | 7 of 11 | 0.9978 | matched |
| scan | C | 3 | 28 | yes | 28 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0554 | 0 of 11 | 0.9980 | exact but slower |
| scan | P-planted | 1 | 37 | yes | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0002 | 5 of 11 | 0.9988 | matched |
| scan | P | 1 | 119 | yes | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0000 | 5 of 11 | 0.9990 | matched |
| histogram | R | 1 | 1400 | no | - | 1.000 | 0.679 | 0.283 | 0.772 | - | - | - | not reached |
| histogram | R | 2 | 1400 | no | - | 1.000 | 0.654 | 0.284 | 0.720 | - | - | - | not reached |
| histogram | R | 3 | 1400 | no | - | 1.000 | 0.679 | 0.277 | 0.772 | - | - | - | not reached |
| histogram | C | 1 | 1400 | no | - | 0.800 | 0.800 | 0.176 | 0.756 | - | - | - | not reached |
| histogram | C | 2 | 1400 | no | - | 1.000 | 0.634 | 0.193 | 0.800 | - | - | - | not reached |
| histogram | C | 3 | 1400 | no | - | 0.800 | 0.800 | 0.176 | 0.756 | - | - | - | not reached |
| histogram | P-planted | 1 | 48 | yes | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0002 | 4 of 11 | 0.9829 | matched |
| histogram | P | 1 | 149 | yes | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0046 | 3 of 11 | 0.9880 | matched |
| pathfinder | R | 1 | 1400 | no | - | 0.875 | 0.875 | 0.633 | 0.120 | - | - | - | not reached |
| pathfinder | R | 2 | 1400 | no | - | 0.985 | 0.063 | 0.385 | 0.549 | - | - | - | not reached |
| pathfinder | R | 3 | 1400 | no | - | 0.871 | 0.128 | 0.452 | 0.120 | - | - | - | not reached |
| pathfinder | C | 1 | 1400 | no | - | 0.989 | 0.230 | 0.483 | 0.231 | - | - | - | not reached |
| pathfinder | C | 2 | 1400 | no | - | 0.996 | 0.149 | 0.458 | 0.120 | - | - | - | not reached |
| pathfinder | C | 3 | 1400 | no | - | 0.979 | 0.236 | 0.494 | 0.120 | - | - | - | not reached |
| pathfinder | P-planted | 1 | 20 | yes | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.9663 | 11 of 11 | 0.9632 | matched |
| pathfinder | P | 1 | 62 | yes | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.9705 | 11 of 11 | 0.9888 | beat |

Pathfinder's "beat" decomposed:

| program | median against published | pairs faster |
|---|---|---|
| the reference gene (the published algorithm as the kingdom writes it, never evolved) | 0.9473, 0.9434 (two re-times) | 11 of 11, 11 of 11 |
| P's champion (an e-graph variant of it) | 0.9705, 0.9699 | 11 of 11, 11 of 11 |

## Hypotheses against the expectations

| benchmark | H1 expected | H1 measured (R, exact seeds; first exact at) | H2 measured (C, exact seeds; first exact at) | H3 expected | H3 measured |
|---|---|---|---|---|---|
| reduce | yes, quickly | **yes**: 3/3; 19, 118, 119 | **not supported**: C 3/3; 19, 127, 60 | unlikely | **undecided**: timing gate failed |
| scan | yes | **yes**: 3/3; 57, 108, 41 | **not supported**: C 3/3; 42, 46, 28 | unlikely | **no**: matched, 1.0000 |
| histogram | likely | **no**: 0/3 | **not supported**: neither exact; C as close or closer | possible | **no**: matched, 1.0046 |
| pathfinder | possible | **no**: 0/3 | **undecided**: neither exact; R closer in 2 seeds, worse in 1 | possible | **beat the published port, not by evolution**: the start is 5.5 % faster, the champion 3 % |

H1 required at least 3 of 5 seeds; we ran 3 seeds. It holds where the target is 2 nodes and fails where the target is 7 or more.

## Why evolution did not help

- **Tiny targets need no search.** A random gene of program size 8 expresses `(sum.n xs)` or `(scan.add xs)` once in 20–130 draws. Selection adds nothing, and evolution's operators made bloated exact programs: reduce R seeds 2 and 3, 14 and 10 nodes, 4.6× and 2.4× slower.
- **Larger targets need a specific ingredient the search never assembles.**
  - Histogram's bin index is `tab (mod.k (get.n is j1))`. No R or C program contained it.
  - The R populations collapsed onto plateaus of many distinct programs with the same outputs. There the median equals the best: 23 or 24 distinct programs, with variation operators improving on the parent in 1–3 % of children.
- **P has little to work with, and little time.**
  - The e-graph finds 0 variants of the published reduce, scan and histogram programs, and 23 of pathfinder's.
  - Timing a new program in 11 pairs costs 30–60 s, so a P run scores only 60–150 programs in its hour.
  - On pathfinder, selection on the 11 in-run slots still preferred a slower variant: in-run 0.940 against 0.955, while the re-times give 0.970 against 0.945.

## Methods findings (machinery faults found and fixed during the study)

Each was found in a measurement, fixed, and recorded in the study plan's §3; every affected result was rerun or voided.

| # | finding | evidence | fix |
|---|---|---|---|
| 1 | timing a tiny kernel measures the GPU's clock state | reduce on its 1,000-element training case: A/A ratios in three clusters, 0.30, 1.00, 3.3. Stage 2 then selected on noise: the planted control's champion, in-run 0.428, re-timed 1.348 | a dedicated timing input, never a quality case (n = 15 x 2^20), `--runs 60`, an A/A gate before every P |
| 2 | a once-a-process GPU pass broke the timing parser | 22 passes for 21 runs (reduce), 85 for 21 (histogram) | the extra passes belong to the warm-up (79e81284) |
| 3 | the first process of a freshly compiled executable is slow | in-run A/A first pair 3.16 | that process is discarded (`time_fresh`) |
| 4 | n = 2^24 exceeds wgpu's dispatch limit for any map over n | "[65536, 1, 1] must be less or equal to 65535" | n = 15 x 2^20 (3b470301) |
| 5 | three pairs a candidate cannot rank speed | in-run 1.287, 11-pair re-time 1.000 for the same program | 11 timing slots per program text, each its own HFF input (f1c8bd17) |
| 6 | the batched evaluation's timing fails every gate | 13, 20, 19, 19 of 22 pairs within 5 %; the published scan at 46 ms in the batch process against 11.9 ms alone | P runs use `--eval single`; batched quality is bit-identical (measured: the same series and ledger) |
| 7 | the "faster" rule alone gave a false beat | reduce R seed 1's champion is the published program itself: median 0.985 below a quiet floor of 0.9936, 8 of 11 pairs faster | "faster" also needs 9 of 11 pairs (eaa1e4a3) |
| 8 | C's reported champion was its last random batch | pathfinder C seed 2: last batch 1.207, best-ever 1.076 | champions from the hall of fame; the runner reports `best_ever_*` (f8409a2f) |
| 9 | histogram ran at 0.06 evaluations a second | Futhark's webgpu runtime made a Metal pipeline at every kernel launch: 6,210 creations, 14.5 s, in one process | the shim caches pipelines (the batching agent, 50339252) |
| 10 | a compiler limitation was a silent link failure | the branch compiler's "SetMem in a GPU kernel" left an empty `prog.c` | named, with the compiler's message (b89f02a8), and counted as its own category |

Compile failures of legal programs, which the runs log and score worst:
- the branch compiler's SetMem limitation: 30 on histogram, 5 on pathfinder, 0 on reduce and scan;
- Futhark internal compiler errors: 4 on histogram and 1 in the pathfinder pilot.
The SetMem gap is the one `docs/PROPOSAL_memory_views_setmem.md` addresses.

## What it would take

Plain statements, each tied to a measurement above:
- **A fair test of H2 needs targets that random search cannot hit.** On 2-node targets random search wins by construction.
- **Discovery on histogram or pathfinder needs the search to assemble specific index expressions.** It never did in 1,400 evaluations, and its populations stalled on plateaus of equal outputs.
- **H3 needs a richer neighbourhood around the published program.** The e-graph found no variants for three of the four.
- **H3 also needs selection that does not reward timing noise.** The in-run ranking on pathfinder preferred a program that re-times slower.
- **Reduce needs multi-dispatch timing** (planned for the rust-wgpu runtime) before its P can run.
