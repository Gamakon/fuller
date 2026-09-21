# HFF-SR on SRBench: the experiment record

## Context

**Objective.** Beat the SRBench ground-truth track (133 problems: 119 Feynman,
14 Strogatz) at noise 0 with HFF-SR. Published symbolic solution rates, 10 seeds,
up to 8 hours per fit: AIFeynman 54.1%, GP-GOMEA 27.1%, AFP_FE 26.2%, ITEA 20.8%,
AFP 20.5%, DSR 19.7%, Operon 16.0%, gplearn 15.6%.

**How every number here was made.** SRBench's own 75/25 split and SRBench's own
scorer (`assess_symbolic_model`). One development seed per race (7001–7020, never
one of SRBench's ten). One fit per problem, one search per fit. A percentage is
solved problems of 133 unless the row says otherwise. Logs and per-fit JSON are in
`hff/notebooks/sr_logs/<name>/` and `<name>.log`.

**What these numbers are not.** They are single-seed development scores, not the
10-seed official figure. Same-seed arms differ by about ±3 problems from the draw
alone, so a change is kept only if it holds on two seeds.

## Python engine (geppy + deap, GPU join for scoring)

| Race | Seed | Time per problem | Solved | % | What was tried |
|---|---|---|---|---|---|
| srbench_30_snaponly | 23654 (SRBench's — before the rule) | 120 s cap | 8 of 30 | 26.7 | Constants by snap only, RNC −100..100, nsimplify removed. 30-problem draw. |
| srbench_30_10min_e | 7001 | 600 s cap | 12 of 30 | 40.0 | Early stop raised to R² ≥ 1−1e-10, populations doubled (600+200), tournament 7%, pump every 4, cross every 5. |
| srbench_race_130 | 7002 | 30 s pass + shared budget, mean 58 s | 47 of 133 | **35.3** | First full race: 30-second pass over everything, leftover time to the unsolved. |
| srbench_race_edge | 7004 | mean 56 s | 46 of 133 | 34.6 | Edge validation (isolated rows by kNN + range extremes) as a separate HFF objective. |
| srbench_race_fixed2 | 7006 | mean 57 s | 46 of 133 | 34.6 | All reported-model faults fixed (protected div/sqrt, `?` cancellation, prune domain, tidy). 0 REPORT FAULTs. |
| srbench_race_seed2 | 7007 | mean 144 s | 4 of the 87 unsolved | — | Second seed on fixed2's unsolved only. Union over two seeds 50 of 133 (37.6%) — not a single-seed score. |
| srbench_race_tensor | 7008 | mean 43 s | 44 of 133 | 33.1 | numpy tensor variation phase (throwaway prototype). 3 REPORT FAULTs. |

Reading: the Python engine sat at 44–47 of 133 whatever was changed. One
generation of 800 cost about 1.2 s, 98% of it Python, so a fit saw 10–30
generations.

## Rust + wgpu engine (`fuller/src/evolve/`), 6 seconds per problem

Population 600+200, 3 genes, head 48, RNC −100..100, pump every 4, OLS for a and b.
About 35 ms per generation, so 6 s is about 170 generations.

| Race | Seed | Solved | % | REPORT FAULTs | What was tried |
|---|---|---|---|---|---|
| ab1_base | 7012 | 46 | 34.6 | 1 | First full Rust race: final form in Rust. Equals the Python engine's score in a tenth of the time. |
| ab1_cleanse | 7012 | 46 | 34.6 | 1 | Cleansing mutation at rate 0.1 (promote a child / collapse a node to `?`). 3 problems swapped each way. **Not kept.** |
| ab2_base_s11 | 7012 | 46 | 34.6 | 0 | Baseline v2: split-radical rule + protected operators printed faithfully. |
| ab2_base_s12 | 7013 | 44 | 33.1 | 1 | Same, second seed. |
| ab2_harvest_s11 | 7012 | 47 | **35.3** | 0 | Harvest-and-regrow, 4 harvests, parking lot, parsimony picks the final model. |
| ab2_harvest_s12 | 7013 | 45 | **33.8** | 1 | Same, second seed. +1 on both seeds, nothing lost. **KEPT — the current default.** |
| ab3_rnc5_s11 | 7012 | 48 | 36.1 | 2 | RNC range −5..5 instead of −100..100 (harvests off). One seed only; seed 7013 stopped at 19 fits. **Unconfirmed.** |
| ab3_restart3_s11 | 7012 | 54 | 40.6 | 0 | Time split into 3 independent searches. **Rejected by Andrew as cheating-like (that is what the 10 seeds are for). Not a valid score.** |
| big_pop_redundancy_g35_s11 | 7012 | 42 | 31.6 | 0 | Population 20,000, 35-generation cap (mean 4.2 s), leave-one-gene-out redundancy as an HFF objective. Gained 4 problems harvest lacks, lost more. **Not kept.** |

Partial runs, stopped early, not comparable: rust_race_7012 (13 of 30, 30 s),
rust_race_7012_b (9 of 20, 30 s, population 2,000), rust_race_7012_6s (26 of 90),
big_pop_redundancy_s11 (4 s, 10 fits), ab3_rnc5_s12 (19 fits).

Not yet run: AB4 (baseline v3 vs edge validation in Rust, seeds 7012 and 7013).

## Measurements that are not races

| Measurement | Result |
|---|---|
| Generation time, Python → Rust, population 800 | 1,200 ms → 35 ms |
| Rust generation time by population | 2,000: 46 ms · 20,000: 176 ms · 40,000: 296 ms (138,000 individuals/s) |
| Generation at which a solve lands (population 800) | median 4, 90th percentile 131–170, latest 387 |
| Generation at which a solve lands (population 20,000) | median 5, 90th percentile 25 |
| Leave-one-part-out, solved laws | median loss 0.49, no dead parts |
| Leave-one-part-out, fakes with R² ≥ 0.999 | median loss 0.003, 70% have dead parts |

## Summary

1. **Best valid single-seed score: 47 of 133 = 35.3%** (Rust engine, harvest-and-
   regrow, 6 s per problem, seed 7012; 45 on seed 7013). That is above every
   published method except AIFeynman (54.1%) — with the caveat that theirs is a
   10-seed figure and ours is one development seed.
2. **The move to Rust and wgpu bought speed, not solves.** 35× faster per
   generation; the score stayed 44–47. The same problems are solved in 6 s that
   took 30–60 s.
3. **Only one change has earned its place on two seeds**: harvest-and-regrow (+1,
   +1). Cleanse, a 20,000 population and the redundancy objective did not.
4. **Half of all solves are in by generation 5.** The search finds a law at once or
   mostly not at all; more generations and more individuals add little. The gap to
   AIFeynman is about what the search can reach, not how long it runs.
5. **14 unsolved problems have test R² ≥ 0.9999** and are approximations, not
   reporting losses. Two more (II_6_15b, test_5) are correct laws the SRBench
   scorer cannot recognise.
6. **Reporting is sound**: the current engine's races show 0–1 REPORT FAULTs, each
   traced.

Open: confirm RNC −5..5 on a second seed; AB4 edge validation in Rust; the cross
step between islands (in the notebook, not yet in the Rust engine); part-level
leave-one-out in the linter; the noisy tracks; the 10-seed official evaluation
(only Andrew triggers it).
