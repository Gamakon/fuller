# Futhark reruns on the fixed fitness, 2026-10-29

**Result: on the fixed fitness, evolution with point mutation beat random search in all 4 pairs (histogram and pathfinder, seeds 1 and 2), at equal numbers of programs scored. No run found an exact program. Ring walk did worse than point mutation.**

Setup:
- fuller 72c37606: the quality formula of b8163635, with both slope checks passed;
- the new cases (19 for histogram, 23 for pathfinder, 4 of each never in the fitness);
- generation 0 made of ring-canvas genes;
- population 24, budget 1,400 new programs, a 30-min guard per run, `--eval rust` on Metal;
- R = evolution, C = random search with its own seed stream.

The per-generation tables are in `/private/tmp/claude-501/frerun2/<set>/<benchmark>_<R|C>_s<seed>/per_generation.tsv`.

## Best HFF so far (lower is better; 0 = exact), at the same number of programs scored

| set | benchmark | seed | programs (the shorter run's) | evolution | random search | evolution's median at end | random's median at end |
|---|---|---|---|---|---|---|---|
| point | histogram | 1 | 689 (R hit the guard) | **1.0584** | 1.1346 | 1.107 | 1.293 |
| point | histogram | 2 | 1,218 (R hit the guard) | **0.9513** | 1.0608 | 0.953 | 1.286 |
| point | pathfinder | 1 | 1,400 | **0.9834** | 1.0000 | 0.984 | 1.243 |
| point | pathfinder | 2 | 1,400 | **0.9536** | 1.0000 | 0.964 | 1.261 |
| ring walk | histogram | 1 | 858 (R hit the guard) | 1.1346 | **1.0928** (at 1,400) | 1.135 | 1.293 |
| ring walk | pathfinder | 1 | 1,400 | 1.0267 | **1.0000** | 1.131 | 1.243 |

- On histogram, evolution's best after a half-length run (1.058 at 689 programs, 0.951 at 1,218) beats random search's best after all 1,400 programs (1.093, 1.061).
- Random search's best on pathfinder stopped at exactly 1.0000 in both seeds.
- Evolution's median falls through each run, while random search's stays at about 1.25 to 1.29.
- The histogram runs hit the 30-minute guard before 1,400 programs: compile failures (74 to 133 a run) made batches halve. This is fixed since (fork patches 0007 to 0009, 7d0b2118) but was not in these runs.

## Ring walk
- Histogram R seed 1 ran 16,149 generations: 858 new programs and 386,742 repeats. Ring walk almost never changed the expressed program, and the best stayed at its generation-0 value.
- On pathfinder it learned (1.140 to 1.027) but less than point mutation (0.983).
- As built, ring walk does not replace point mutation.

## Fable set (generation 0 = one Fable program plus mutants)
All 8 runs stopped at evaluation 1, because Fable's program was exact.

| benchmark | Fable's program, speed against the published one (median, 11 pairs) |
|---|---|
| histogram | 70–72 times slower (per output bin: a `tabulate` over the whole input) |
| pathfinder | 0.94–0.96 (about 5 % faster; the same algorithm, written with `map2` over whole rows) |

The decision on running this set as an improvement (speed) search was not made, so no improvement run was made.

## What went wrong in the night
- The GPU was idle from 05:01Z to the end of the session: all queues finished, and the lead did not notice or start the next work.
- The Fable set ran as discovery before the decision on it had been made.

## Improvement run from Fable's histogram program (09:03Z to 09:25Z, stopped)
Generation 0 was Fable's exact program plus 23 point mutants. Stage 2 began at generation 0, as designed. Then a bug showed: under the two-level HFF, wrong but fast mutants outrank the exact but slow seed. For example, a mutant with quality angle 1.29 (wrong) and speed ratio 0.002 scored 0.848, ahead of the exact seed, which is 71 times slower than the published program. Over 7 generations the best fell 0.848 → 0.802, but the best was inexact in every generation (`exact 0+0` for the best), so selection was favouring wrong programs over the exact seed. A speed ratio here is the candidate's time over the published program's: 0.002 means a wrong program that does almost no work. The run was stopped by its run file.

**Bug:** in stage 2, an inexact program must never rank above an exact one (Andrew's rule: quality first, then performance). The fix is that once a run is in stage 2, selection is lexicographic: exact before inexact, and the two-level HFF only among the exact. It has not been built.

## Seed-less run: Fable's mutants without Fable's program (09:49Z to 10:27Z, stopped at shutdown)
Histogram, R, seed 1, `--gen0 fable-mutants-only --mutation point --eval single`, fuller 514602a3 (stage 2 gated by correctness, the Futhark timing work bound). **It learned: the best fell 1.2644 → 1.1024 (−13 %) and the median 1.2926 → 1.1358 over 23 generations (445 programs).** No exact program on this path (the same command on `--eval rust` reached an exact program at evaluation 34; see the open bugs). Full table: `docs/reports/2026-10-29_histogram_noseed_per_generation.tsv`.

| generation | programs | best | median |
|---|---|---|---|
| 0 | 24 | 1.2644 | 1.2926 |
| 4 | 98 | 1.2104 | 1.2594 |
| 7 | 160 | 1.1789 | 1.2300 |
| 10 | 213 | 1.1199 | 1.2068 |
| 16 | 325 | 1.1042 | 1.2367 |
| 19 | 382 | 1.1024 | 1.1330 |
| 23 | 445 | 1.1024 | 1.1358 |

## Open bugs at shutdown (2026-10-29 10:30Z)
1. `--eval rust` records no timed pass for Fable's histogram program, so it is scored as infinitely slow (`src/futhark_batch.rs:109`).
2. `--eval batch` timing slows the published program 1.3–3.6× and fails every A/A gate.
3. Generation 0 differs between `--eval rust` and `--eval single` for the same command and seed (rust: best 1.166, exact at evaluation 34; single: best 1.264, not exact).
4. The fork patches 0007–0009 (7d0b2118) compile all 373 failing programs, but their GPU value check has not been run.

The fix for bugs 1 to 3 was in progress when the session ended. Its edits (`src/futhark_wgpu_host.rs`, `src/futhark_batch/tests.rs` (a cross-path test), `src/futhark_sig.rs`, `examples/futhark_evolve.rs`) are committed as work in progress: clippy is clean with `gpu,wgsl`, and the tests have NOT been run.
