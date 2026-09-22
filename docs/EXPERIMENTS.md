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

## 2026-09-21/22: the engine's capabilities, and the two-pass race

Everything below is the Rust engine at 6 seconds per problem unless stated, on
development seeds, scored by SRBench's own scorer, with both submissions (fuller's
own string and the sympy-tidied one) recorded.

### The races

| Race | Seed | Setup | Solved of 133 |
|---|---|---|---|
| balanced-pole tournaments | 7014 | balanced pole for selection | 26 |
| TrueNorth, 3 genes x head 34 | 7013 | the baseline setup | **60** |
| TrueNorth, 1 gene x head 48 | 7013 | one long gene | **60** |
| ... either of the two | 7013 | | **68 (51.1%)** |
| first pass | 7014 | 6 s, everything default | 52 |
| second pass | 7014 | the 80 that did not meet the stop bar: compounds, 3 island pairs + the cross step, 60 s | +15 |
| **both passes** | 7014 | | **67 (50.4%)** |
| precision-fixed | 7013 | identical to the 60, on the fixed scorer | 56 |

**Distinct laws solved at least once by the Rust engine: 77 of 133 (57.9%).**
A union over configurations and seeds, not a single-configuration figure — but it
establishes what the engine can reach.

### What was built, and what it is worth

| Change | Evidence |
|---|---|
| **The balanced pole is wrong for selection** | 26 vs 53 on the same seed, everything else equal. TrueNorth won 30 laws the balanced pole missed. Confirmed Andrew's reading. |
| **The stop bar gained a p-value half** (log10 p <= -19 as well as 1-R2 <= 1e-10) | Every real law that met the old bar sat at -19.4 or below; the one fake that slipped under it sat at -17.55. On seed 7014, 53 fits met the new bar and 52 were exact. |
| **The two-pass race** (Andrew's design: pick off the easy laws, then spend the resources on the rest) | The stop bar — never SRBench's verdict — decides what is set aside. 52 of 53 set aside were exact; the second pass found 15 more in the remaining 80, five of them first-ever solves. |
| **The compensated sums were dead on this device** | The Metal compiler removed the Neumaier `add`; the device matched neither the compensated nor the plain-sum CPU twin. Fixed, plus a 1-ulp correction to the device's `/` and `sqrt`. The f32 floor on a true law fell from 1.2e-10 to 1.1e-14 — against a stop bar of 1e-10. |
| **Compound functions** (sqrt/1/sqrt/1/(a+-b) as single symbols, expanded at decode) | For the shapes the search never builds: a sum under a root was 1 of 18 solved, the 1/sqrt(1-v^2/c^2) family 0 of 9. Used in the second pass. |
| **Island pairs and the cross step** (Andrew's `_migrate_pump_cross`, ported) | Used in the second pass. One pair with no cross step is bit-identical to before. |
| **The dynamic gene-subset choice** | Every non-empty subset of the genes is a scoring candidate, so a chromosome decides whether it is a 1-, 2- or 3-gene model. On I_26_2: off, 328 generations and unsolved; on, early stop at generation 14 using one gene. Costs 2.7x per generation. |
| **Snap on the GPU, all four stages** | The lattice as a resident table; the choice of form is an HFF TrueNorth decision over nearness, size and fit-to-context; the R2 guard judges the whole model; the kept form is written back into the Karva gene. Measured: at RNC -100..100 about 5% of genes hold a non-whole constant and a third of those match; at -5..5 the matches are dominated by rationals that change nothing. |
| **Data guided rewrites** (Andrew's name) | Rewrites the final form may make because the DATA says they hold on every row, each checked against the model's own predictions. They recovered laws the engine had FOUND and the scorer had rejected: II.11.27, III.12.43, I.18.12, I.14.4, I.30.5, III.17.37, glider2, barmag2, II.10.9, I.44.4. |
| **A literal a product apart from the fitted scale** | strogatz lv2 was found as 0.037037 * ((27 * (2-y-x)) * y) — the law, scored wrong because SRBench rounds 0.037*27 to 0.999. Four orderings added to the rational ruleset. |

### What the evidence says

1. **A law is found early or not at all.** 54% of solves are in the initial
   population, 69% by generation 5. Nine hard laws given 1,000-10,000 generations:
   the error fell 0.3-3.4 decades and not one verdict changed — the imitation is
   refined, not replaced.
2. **Diversity, not effort, is what flips a law.** Every law that went from unsolved
   to solved did so under a different draw: another seed, another pole, another
   population, restarts. Two seeds at identical settings differ by ~8 laws.
3. **Effort still pays when it is spent on the right problems.** The second pass
   turned 60 s each on 80 hard problems into 15 solves, five of them firsts.
4. **Reporting has been worth as much as search.** Ten laws the engine had already
   found were being scored wrong because of the form they were written in.
5. **The two setups reach different laws.** 3 genes and 1 long gene each solve 60,
   together 68: one long gene gets the single nested structures (arcsin, the
   Gaussian), three genes get the sums of separate terms.

### Open

- The precision fix changed the search (different arithmetic, different tournament
  winners), so 60 vs 56 on one seed cannot separate a real effect from the draw.
  A second seed is running.
- Snap needs RNC -100..100 to bite; every recent race used -5..5.
- 9 laws of the 1/sqrt(1-v^2/c^2) family have never been solved in any run.

## Age and lineage: what the genealogy log measured (2026-09-22)

Every individual of a fit now carries an IDENTITY, an AGE (generations since its
genotype entered the population — Hornby's ALPS rule: an offspring is its
parent's age plus one, a fresh random individual is 0) and a LINEAGE back to the
founder its line began at. `EVOLVE_GENEALOGY_FILE` writes the study file; with it
unset the engine is bit-identical to before and costs nothing. This is the
measurement only — nothing about selection changed, and no age layers were added.

Six fits, seed 7013, population 1500 + 1500 = 3000, head 34, 30 s, the race's
settings (`EVOLVE_TOWER=1 EVOLVE_HFF_NO_VAL=1 EVOLVE_RNC_LO=-5 EVOLVE_RNC_HI=5
EVOLVE_PUMP_EVERY=5`). One seed, on a GPU shared with running races, so the
timings are under contention.

| law | gens | winner age | founder gen / origin | ages, all rows (min/med/max) | best 10 | founders best 50 | founders, all |
|---|---|---|---|---|---|---|---|
| I_12_5 (solved gen 1) | 1 | 1 | 0 init | 1 / 1 / 1 | 1 / 1 / 1 | 25 | 60 |
| I_14_3 (solved gen 1) | 1 | 1 | 0 init | 1 / 1 / 1 | 1 / 1 / 1 | 20 | 40 |
| II_38_14 (solved by search) | 204 | 204 | 0 init | 204 / 204 / 204 | 204 | **1** | **1** |
| I_39_11 (not solved) | 335 | 335 | 0 init | 0 / 335 / 335 | 335 | **1** | 1201 |
| I_48_2 (not solved) | 360 | 360 | 0 init | 0 / 360 / 360 | 360 | **1** | 1201 |
| III_4_32 (not solved) | 332 | 332 | 0 init | 332 / 332 / 332 | 332 | **1** | 1 |

**Every winner's founder was drawn in the initial population.** Not one fit was
won by a line the pump introduced later — consistent with `STUDY_near_misses.md`'s
finding that 54% of solves are already in the initial population.

**The population converges onto ONE line and the pump never breaks it.** Across
the four long fits the pump drew **294,000 fresh individuals** (40–72 refills of
~1,200 rows each). Of the 73,990 rows that were later kept as the intake's best
fifth or promoted into a champion island, the founder origin was `init` for
**every single one** — zero `pump_refill`. A fresh draw's descendants never once
reached the top fifth of the island they were drawn into. That is the number that
bears on ALPS: the pump is already supplying young material, and selection
discards all of it before it can compete, because it meets converged old
material in the same tournament.

**A fresh line is gone within two generations.** `founders, all` is 1201 on
exactly the two fits that ended ON a pump beat (335 = 67x5, 360 = 72x5): the
1,200 rows just drawn, plus ONE line for the other 1,800. On the two that ended
off the beat — II_38_14 at generation 204 (4 past the beat at 200) and III_4_32
at 332 (2 past 330) — it is 1. Every one of 1,200 fresh lines is extinct across
all 3,000 rows within two generations of arriving.

The mechanism is the tournament fraction against the keeper fraction, and it is
arithmetic, not luck. A refilled intake holds 300 keepers (its best fifth, by
construction fitter than anything just drawn) among 1,500 rows, and the
tournament size is 7% of 1,500 = 105. The chance that one 105-draw tournament
contains no keeper at all is (1200/1500)^105 = 6.7e-11, and `select` is
two-stage, so it is tighter still. Young material is not outcompeted; it is
never sampled without a converged elder in the same draw.

The `I_39_11` age line reads `0 / 335 / 335` with mean 201.00 — exactly
1800 x 335 / 3000. Every one of the 300 keepers and all 1,500 champion rows is
age 335 and descends from the same generation-0 founder; the remaining 1,200 are
the refill at age 0.

Winners are also not long-lived elites: minting age equals row age on three of
four long fits (the winning variation happened in the final generations), and the
one exception sat unchanged for 27 generations.

**Cost.** 0.86–1.15 s of a 30 s fit (2.9–3.8%) in `timing.genealogy`, plus the
pump hooks, which land in `timing.pump`. Per fit: 20,500–22,500 lines,
1.25–1.40 MB. 96% of the lines are `pump_keep` — the intake's best fifth
re-logged at every beat — so a batch line for keepers would cut the file
twenty-fold if the volume ever matters. Generations per second with and against
the log differ by less than the contention on a shared GPU (I_48_2: 83.6 ms/gen
on, 71.4 off; III_4_32: 90.6 on, 94.1 off — one slower, one faster).

**What the design can and cannot answer.** Each row carries its founder forward,
so "how old is the winner and where did its line come from" is always answerable.
The full edge table is kept for the fit (one 40-byte record per minted id: 0.66–1.17
million ids, 26–47 MB on these fits), so the winner's whole chain can be walked
— it is 202–317 steps here. What it cannot do is reconstruct an arbitrary row's
chain after the fit ends from the log alone: only the winner's chain is written
out, and only arrivals and the per-generation best are recorded, not every edge.

## The beam subset to edge-case wraps (2026-09-22)

Andrew's standing rule: "this is exactly the same function as wrapping the
symbolic solution in a linear regression ... the final mutation is functional",
and "subset the beam to edge cases like this [x/(exp(x)-1)]". So a mutation is a
FUNCTIONAL WRAP applied to what the search already has, its free parameters
solved by least squares rather than searched; and the beam is subset to the
shapes the gene provably never builds.

`BEAM_WRAPS` goes from seven to four. `Exp`, `Square` and `Recip` are dropped:
`ProtectedExp`, `Pow2` and `ProtectedInv` are all SAMPLED functions of the gene's
own table, so ordinary variation reaches those shapes by writing one symbol. What
is left is aimed at named laws — `1/(x-1)` at III.4.32, `x/(exp(x)-1)` at
III.4.33, `1/sqrt(1-x)` at the Lorentz family (I.10.7, II.13.23, I.48.2, I.15.10,
II.13.34, I.34.14, **0 of 9 solved in every race on record**), `1/(1-x)` at
I.34.1's `omega_0/(1 - v/c)`. The beam's TREE half is now off by default
(`Config::beam_tree`): a default beat is the wraps alone.

**Two blockers, both real, both measured.**

*The backreference.* `wrap_nodes` had `Unary`, `BinaryUnitFirst` and
`BinarySelfFirst` — every node takes the value below it ONCE — so a wrap was a
CHAIN and `x/(exp(x)-1)`, whose argument appears twice, had no spelling at all.
It was scored on every beat and could never land. Andrew: "in sed we have
`s/\(blah\)/andrewsays\1\1\1/g` so can we not do something at all?"
`WrapNode::HostFirst(op)` is `op(host, below)` — the left child is the gene's own
root, so the template may name the wrapped value as often as the shape needs.
Karva cannot share a subtree, so the host is written out again and the gene grows
by the host subtree's size. **That cost never bit: over 28 fits and ~25 million
genes evaluated, 0 exceeded the 64-node limit, and of 10 graft attempts 9
succeeded and 1 was refused.** (The largest grafted gene was not measured; those
two counts are what the logs carry.)

*Score is not graft.* A wrap was SCORED on `W(L(g0,g1,g2))` and GRAFTED as
`W(g0)` with the other genes set to 1 — different functions whenever the model
uses more than one gene. Run against commit e55b94c on a three-gene model,
`1/(x-1)` scored 1-R² **0.0759** and its graft computed **1.0100**: a wrap that
looked like it explained 92% of the variance landed as worse than the mean.

The first fix — score the wrap on the HOST GENE ALONE — makes score and graft
agree, and **cannot express four of the five aimed laws**. I.10.7 is
`m_0/sqrt(1-(v/c)^2)`, I.48.2 `m*c^2/sqrt(...)`, I.34.1 `omega_0/(1-v/c)`,
III.4.33 `kb*T * u/(exp(u)-1)` — all `prefactor(variables) * W(u)`. With only a
SCALAR outside the wrap the prefactor has nowhere to live. Measured, 24 fits, two
seeds: **8,300 wrap candidates, 0 better, 0 grafted, 0 kept.** A clean negative,
and not a search failure — a shape the scoring could not represent.

The wrap now goes INSIDE the linker, on one gene: scored as
`a * L(W(g_host), g_other, ...) + b` and grafted the same way, the other genes
untouched. Score and graft still agree by construction.

**The measurement.** Six laws x beam off/on x seeds 7013 and 7014, 30 s each,
race settings (`EVOLVE_TOWER=1 EVOLVE_HFF_NO_VAL=1 EVOLVE_RNC_LO=-5
EVOLVE_RNC_HI=5 EVOLVE_PUMP_EVERY=5`, population 1500+1500, head 34). Shared GPU,
so timings are under contention.

| law | seed | off log10 p | on log10 p | grafted/kept | on stopped by |
|---|---|---|---|---|---|
| III.4.32 | 7013 | -9.57 | **-11.17** | 1 / 1 | time |
| III.4.32 | 7014 | -10.36 | -10.53 | 2 / 0 | time |
| III.4.33 | 7013 | -5.91 | -6.55 | 0 / 0 | time |
| III.4.33 | 7014 | -7.26 | -7.37 | 0 / 0 | time |
| I.10.7 | 7013 | -9.11 | **-21.08** | 1 / 1 | **early_stop, gen 7** |
| I.10.7 | 7014 | -7.75 | -7.47 | 0 / 0 | time |
| I.48.2 | 7013 | -8.30 | -11.00 | 0 / 0 | time |
| I.48.2 | 7014 | -10.45 | -9.94 | 5 / 0 | time |
| II.11.3 | 7013 | -3.30 | -3.14 | 0 / 0 | time |
| II.11.3 | 7014 | -5.87 | -4.67 | 0 / 0 | time |
| I.12.5 (control) | 7013 | -22.42 | -22.42 | 0 / 0 | early_stop |
| I.12.5 (control) | 7014 | -22.35 | -22.35 | 0 / 0 | early_stop |

**feynman_I_10_7 is FOUND at generation 7, log10 p -21.08, test R² 1.000000**, by
one `1/sqrt(1-x)` wrap grafted and kept. The model is
`x_0/(sqrt(1 - v/c) * sqrt(1 + v/c))` — algebraically exactly
`m_0/sqrt(1-(v/c)^2)`. FOUND, not SOLVED, in the near-miss study's own sense: it
met the engine's 1e-10 stop bar and the form is the law by eye, but SRBench's
sympy scorer was NOT run on it, and `sqrt(1-v/c)*sqrt(1+v/c)` is exactly the kind
of form sympy may decline to collapse without domain assumptions. `_refit_rules.py`
can settle it without a race. That law is in the family the study records as 0 of
9 solved in every run ever. The control does not regress (identical both arms).

**The honest caveat, and it is large.** I.10.7 solved on seed 7013 and NOT on
7014 (-7.47, no graft). Run-to-run spread within an arm reaches 2.57 decades on
II.11.3 and 2.15 on I.48.2, so a one-seed 1.6-decade move on III.4.32 is inside
the noise. **One solve at one seed is a single event, not a rate.** What is
outside the noise is structural, not statistical: the wrap now GRAFTS (9 grafts
against 0 under the previous design), and the shape it grafts is one the search
does not otherwise build.

**9 GRAFTED, 2 KEPT — and the gap is the next fix.** Score equals graft on R², by
construction and under test. It does NOT hold on HFF FITNESS, and the measurement
shows where: `confirm_over` computes the candidate's tower from the UNWRAPPED
host gene's `t_depth`, while the graft carries a real `ProtectedExp` or
`ProtectedSqrt` node, so under `EVOLVE_TOWER=1` the re-score pays a tower penalty
the wrap's own score never did; and the wrap is compared against an f64 baseline
while the graft must beat the device's f32. I.48.2 seed 7014 is the evidence:
`x_over_expm1=6/0, grafted 5, kept 0` — six winning wraps, five grafted, none
kept, every one adding an `Exp` under the tower. That is the leading HYPOTHESIS,
not a diagnosis — it was not isolated. The fix is to score a wrap with the
GRAFTED gene's depth rather than the host's. Not done here.

Two of the aimed laws are probably out of reach regardless: III.4.32 and III.4.33
both have `h/(2*pi)` INSIDE the exponential, and with the wide table, no snap and
whole-number constants -5..5 there is no way to write pi in a gene. The near-miss
study's own rule puts "a non-whole-number constant inside the structure" at
verdict "no". Those two rows measure whether the beam hurts, not whether it
helps.

Logs: `logs/beam_host_gene_alone.log` (the negative), and
`logs/beam_inside_linker_measurement.log` (this table).

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
