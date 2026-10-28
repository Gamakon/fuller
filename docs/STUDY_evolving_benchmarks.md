# STUDY: can evolution deliver the benchmarked GPU challenges?

## Context

**Question.** Given only examples of a published GPU benchmark's inputs and outputs, does our typed GEP system evolve a program that computes it exactly? Given the published program, does it evolve a faster equivalent?

**Why it matters.** Our evolution has shown legality everywhere and real results in symbolic regression and regex. On GPU programs it has not yet beaten a hand-written original. In the WGSL and image studies (October 2026), every apparent gain was a measurement flaw, or a cheat the fitness allowed. A study on standard, published benchmarks, with controls and stated expectations, settles whether the machinery works on GPU code.

**Approach.**
- Rodinia and Parboil-style benchmarks, as ported in `diku-dk/futhark-benchmarks`.
- Each benchmark is a pure function: typed values in, typed values out. Its observations are random inputs paired with the published program's exact outputs.
- The Futhark kingdom expresses candidates.
- The shim runs them on our wgpu/Metal path.
- Fitness is staged: quality only until exact, then the two-level HFF (quality angle, speed angle; the hff library's TrueNorth; every measurement its own input).

**Status.** Pilot (pathfinder) in preparation, 2026-10-27.

## 1. Hypotheses, stated before any run

- **H1, discovery:** from random, evolution finds a program exactly correct on held-out inputs for the small benchmarks, within the budget, in at least 3 of 5 seeds.
- **H2, against chance:** at the same number of evaluations, random search (a fresh random gene per evaluation, same kingdom) finds fewer exact programs than evolution.
- **H3, improvement:** from the published program, evolution finds an exactly-correct variant faster than the published one on the GPU, by more than the measured timing noise.

**Expectations, stated now:**

| Benchmark | H1 expected | H3 expected |
|---|---|---|
| reduce | yes, quickly | unlikely: Futhark's reduction is already optimal |
| scan | yes | unlikely |
| histogram | likely | possible |
| pathfinder | possible | possible: fusion of the row loop |
| hotspot (stencil) | possible | possible |
| kmeans (one step) | unlikely in budget | possible |

## 2. Benchmarks, in order

| # | Benchmark | Input → output | Exact? |
|---|---|---|---|
| 1 | pathfinder (Rodinia) | `[rows][cols]i32` → `[cols]i32` | integers, exact |
| 2 | reduce / scan (micro) | `[n]i32` → `i32` / `[n]i32` | exact |
| 3 | histogram (Parboil-style) | `[n]i32`, bins → `[bins]i32` | exact |
| 4 | hotspot (Rodinia stencil) | temperature and power grids → temperature grid after k steps | f32, tolerance stated |
| 5 | kmeans step (Rodinia) | points, centres → new centres | f32, tolerance stated |

## 3. Method

- **Observations per benchmark:** training inputs at two small sizes, held-out inputs at a third size, and outputs from the published `.fut` program. The held-out cases are in the quality group, the validation column.
- **Regimes:**
  - **(R) from random:** 5 seeds.
  - **(C) random-search control:** the same evaluation budget, 5 seeds.
  - **(P) from the published program toward faster:** 1–3 seeds. It starts in stage 2.
- **Budget:** a fixed number of fitness evaluations per run, the same across R and C, with wall clock as a guard only. The value is set from the pilot's measured evaluation rate.
- **Change, 2026-10-27 20:40Z (before any study run but reduce R seed 1):**
  - R and C answer H1 and H2, which are about discovery. They stop at the first program exact on every case, training and held-out (`--stop-at-exact`, fuller b1fe8606), and they record the evaluation at which it was found.
  - A run with no exact program goes on to its budget of 1,400 evaluations, with the 60-minute guard.
  - Speed (stage 2) is measured only in P, which answers H3. P keeps stage 2, capped by its budget and the guard.
  - The reason: in stage 2 every candidate is timed (about 10 s each on reduce), so an R or C run that had already found an exact program spent its hour answering a question that only P asks.
  - Reduce R seed 1 ran before the change. It counts as exact at evaluation 19.
- **Changes, 2026-10-27 21:00Z to 2026-10-28 05:00Z (all before any P run that counts):**
  - **Timing (H3).**
    - A program's timed pairs are kept by its text in 11 slots (`runner::TimingSlots`). Each pair is its own speed input; they are never averaged.
    - The micro benchmarks are timed on their own input of n = 2^24 (`Problem::timing_n`), never a quality case. On the 1,000-element training case, reduce's A/A ratios fell in three clusters (0.30, 1.00, 3.3), the GPU's clock states.
    - The first process of a freshly compiled executable is discarded.
    - Micro P runs use `--runs 60`.
    - Every P run is gated first: both A/A sets (22 pairs) must be within ±5 %.
    - A speed claim counts only when the champion's 11-pair re-time median is below the least A/A ratio.
  - **Budget.**
    - R and C get a guard large enough for 1,400 evaluations (180 min), so H2 compares equal budgets.
    - Histogram R seed 1 was rerun at the full budget. Histogram R seed 2 (1,176 evaluations, stopped) is a short run and is replaced.
    - Pathfinder R seed 1 (the pilot: 1,000 evaluations) was rerun at 1,400.
  - **Batched evaluation (fuller 50339252; from the first run after pathfinder R seed 1, about 05:15Z 2026-10-28).**
    - A generation is one Futhark program in one process (`--eval batch`).
    - It gives bit-identical quality inputs and the same selection as the per-candidate path, tested on the GPU. Earlier runs used the per-candidate path, so the two sets of results are comparable.
    - The shim now caches Metal pipelines. Before that, histogram candidates spent most of their time creating a pipeline at every kernel launch (6,210 creations, 14.5 s, in one process).
- **Machinery, the same for all:**
  - the general Futhark signature kingdom;
  - marriage with distinct parents;
  - the e-graph and saturated share-equivalence fold in the loop;
  - the full runner standard: telemetry for evowatch, checkpoints, `killme.trigger`, hall of fame, lineage, mutation events.
- **Measurements:**
  - exact-correct (held-out) and the evaluation at which it was first reached;
  - HFF against generation;
  - the speed ratio against the published program (GPU, interleaved pairs, with a noise floor);
  - operator effectiveness;
  - diversity.
- **Controls before any run:**
  - the published program expressed as a gene gives exact outputs, so the kingdom can express it;
  - an A/A timing of the published program against itself;
  - a planted slowdown in the published program is removed in (P).

## 4. Report

One write-up per benchmark plus a summary: `docs/reports/2026-10-xx_STUDY_evolving_benchmarks.md`. It holds the hypotheses against the results, the tables, the charts, the champion programs and their diff against the published ones, and what failed and why.
