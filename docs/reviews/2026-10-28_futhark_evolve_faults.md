# Review: why evolution did no better than random search on the Futhark benchmarks

## Context

**Objective.** Catalogue every fault in fuller's Futhark evolution machinery that could make evolution no better than random search, or make the study's measurements wrong. The study is `docs/STUDY_evolving_benchmarks.md`, its reports `docs/reports/2026-10-27_STUDY_*.md`. The owner's premise: random search beats evolution only when the fitness is broken.

**Approach.** An independent read of `examples/futhark_evolve.rs`, `src/futhark_sig.rs`, `src/runner.rs`, `src/score.rs` (`truenorth_angle`) and the hff library's TrueNorth (`../hff/src/core_functions.rs`). Measurements on the study's own ledgers in `/private/tmp/claude-501/fbench/`. A Python re-implementation of the quality inputs and TrueNorth, checked against the ledgers: it gives exactly the ledger's fitness for the constant zero output (histogram 1.285553, pathfinder 1.379527), for histogram C's champion `(hist.add.n zeros vs)` (1.136701) and for pathfinder C seed 2's champion, the last wall row plus `rows` (1.075944). No GPU, no evolution run, no code changed. Written 2026-10-28, 18:30Z.

**Answer in one sentence.** The owner's premise holds: the fitness has no slope toward the target, so selection cannot beat random sampling. On both hard benchmarks, programs further from the published program score better, not worse.

**Status.** 22 faults: 6 break evolution, 11 distort the measurements, 5 are minor. Six checks found no fault (last section).

## Summary

| # | fault | severity | status |
|---|---|---|---|
| 1 | Fitness does not reward getting closer to the target: one-edit neighbours of the target score no better than unrelated programs, and fitness gets better with distance from the reference | breaks evolution | CONFIRMED |
| 2 | Four of the seven quality inputs are step functions, three of them near-copies, all 1 for a program that is a little wrong everywhere | breaks evolution | CONFIRMED |
| 3 | Error is scaled by the output's range, so pathfinder's fitness rewards matching the output's level, not its structure | breaks evolution | CONFIRMED |
| 4 | Histogram's cases cannot give partial credit: values over the whole `i32` range wrap, and no case is a stepping stone | breaks evolution | CONFIRMED |
| 5 | Degenerate cases (k = 1, rows = 1) reward trivial wrong programs and become the population's goal | breaks evolution | CONFIRMED |
| 6 | Behaviour diversity collapses while the run reports text diversity as high | breaks evolution | CONFIRMED |
| 7 | Elites are scored again every generation and counted as evaluations | distorts measurement | CONFIRMED |
| 8 | Repeated programs (cache hits) count as evaluations; this costs random search more than evolution | distorts measurement | CONFIRMED |
| 9 | The pump and the fold, in stage 1, can only make children of equal fitness, and they repeat themselves | distorts measurement | CONFIRMED |
| 10 | R and C with the same seed share generation 0 | distorts measurement | CONFIRMED |
| 11 | The held-out cases are in the fitness, so "exact on held-out" is not a test of generalisation | distorts measurement | CONFIRMED |
| 12 | `--rescore` still writes verdicts by the old rule (no 9-of-11 test) | distorts measurement | CONFIRMED |
| 13 | Timing: the batched path fails every A/A gate; selection on 11 in-run slots chose noise; the "least A/A ratio" floor is unstable | distorts measurement | CONFIRMED |
| 14 | Legal programs the compiler cannot build (SetMem, internal errors) are scored worst | distorts measurement | CONFIRMED |
| 15 | After the best-ever fix, the report's `best_*` fields and the champion re-time still come from the last batch | distorts measurement | CONFIRMED (latent) |
| 16 | The hall of fame mixes stage-1 and stage-2 fitness; after a stage change its rank 1 is the first exact program, not the fastest | distorts measurement | CONFIRMED (latent) |
| 17 | Seed 1 of R ran on a different binary and evaluation path from every other seed | distorts measurement | SUSPECTED |
| 18 | Programs bloat under neutral drift (8–17 nodes to 35–124), because N limits functions per type, not in total | minor | CONFIRMED |
| 19 | Marriage stops after the first 256 fitting pairs in level order, so it favours swaps near the first parent's root | minor | SUSPECTED |
| 20 | Crossover throws away its second child, and every crossover child is also mutated | minor | CONFIRMED |
| 21 | Partners need only a different text; on a plateau most partners give the same output | minor | CONFIRMED |
| 22 | 24 individuals for 58 generations is a small search for an evolutionary method to show an advantage | minor | SUSPECTED |

## Faults that break evolution

### 1. The fitness does not reward getting closer to the target (CONFIRMED)

**What.** On histogram and pathfinder, a program one edit away from the published program scores in the same band as unrelated programs, or worse. Over every distinct program the study evaluated, fitness IMPROVES as the tree edit distance to the reference grows.

**Evidence: fitness against distance.** Zhang–Shasha tree edit distance from each distinct program in the 12 R and C ledgers to the reference gene:

| benchmark | distinct programs | Spearman(fitness, distance) | expected for a good fitness | best fitness at distance 3–8 | best fitness at distance ≥ 60 |
|---|---|---|---|---|---|
| histogram | 6,588 | **−0.68** | positive | 1.124 | 1.124 |
| pathfinder | 6,027 | **−0.52** | positive | 1.207 (18–20) | 1.015 |

Lower fitness is better, so a negative correlation means the further a program is from the reference, the better it scores. The tree distance is to one written form of the reference, and big programs are both far away and the product of selection, so the number mixes cause and effect. The neighbour tables below do not have that problem.

**Evidence: the target's neighbours (histogram).** These are one- or two-symbol edits of `(hist.add.n (tab (mod.k (get.n is j1))) vs)`, scored on the stored cases. The plateau bands are measured on the ledgers: R's best 1.124–1.147, C's best-ever 1.120–1.137, the constant zero output 1.286.

| program | edit | fitness |
|---|---|---|
| the published program | — | 0.000 |
| bins `vs % k` instead of `is % k` | one terminal | 1.032 |
| bins `(is + 1) % k` | two nodes | 1.052 |
| `min` instead of `+` | one symbol | 1.108 |
| values `is` instead of `vs` | one terminal | 1.166 (worse than every R and C champion) |
| `max` instead of `+` | one symbol | 1.179 |
| values replaced by a count of 1 | one subtree | 1.210 |
| bins `is` with no `% k` | one node removed | 1.286 (equal to the zero output) |
| values `-vs` | one node | 1.323 (worse than the zero output) |

**Evidence: the target's neighbours (pathfinder).** Against R seed 3's champion at 1.015, a 46-node program that is wrong in 87 % of cells:

| program | edit | fitness |
|---|---|---|
| the published algorithm (the reference gene) | — | 0.000 |
| loop from 0 instead of 1 (one extra step) | one symbol | 0.408 |
| min over `j` and `j+1` only (drop `j-1`) | one subtree | 1.045 |
| min over `j` and `j-1` only (drop `j+1`) | one subtree | 1.059 |
| last wall row + `rows` (C seed 2's champion form) | unrelated | 1.076 |
| `max` instead of the outer `min` | one symbol | 1.176 |
| no `min` (column sums) | one subtree | 1.184 |
| last wall row only (where R seed 1 sat for 59 generations) | unrelated | 1.207 |

Dropping one neighbour from the published `min` gives a program that is right in structure and close in value, yet it ranks below R seed 3's wrong champion. Selection therefore moves the population away from the target. Faults 2 to 5 are the causes.

**Fix.** Make it a gate before any run, the way the A/A gate is for timing. Score every one-edit neighbour of the reference gene, plus a few thousand random genes, on the host interpreter (`Interp::eval`, CPU, seconds). A benchmark may run only if the neighbours rank above the random genes and the fitness-distance correlation is positive. Then fix faults 2 to 5 until the gate passes.

### 2. Four of the seven inputs are step functions, three of them near-copies (CONFIRMED)

**What.** `wrong_share`, `outside_1pct`, `outside_10pct` and `outside_50pct` (`src/futhark_sig.rs:1146–1150`) count cells past a threshold. A program that is a little wrong in every cell scores 1 on all four. For integer outputs, the first three are almost the same number, so the share of wrong cells counts three times over.

**Evidence.** The means over the cases of R seed 3's pathfinder champion: wrong 0.871, over 1 % 0.871, over 10 % 0.865, over 50 % 0.128. The drop-one-neighbour program: 0.750, 0.750, 0.742, 0.474. Histogram: every R champion, and every one-edit neighbour in the table above, is 1.00 wrong on every non-degenerate case.

**Fix.** Keep one exactness input (the share of wrong cells), drop the 1 % and 10 % copies, and add graded per-cell inputs with no threshold. One example is the mean over cells of `log(1+|d|)/log(1+M)`, with `M` the largest magnitude. Check the result with the gate of fault 1.

### 3. Error is scaled by the output's range, which rewards matching the level (CONFIRMED)

**What.** `mean_abs_over_scale` and `max_abs_over_scale` divide by the correct output's range, its largest value minus its smallest (`futhark_sig.rs:1138–1142`). Pathfinder's outputs are minimum path sums: large values in a narrow band. A structural miss is therefore larger than the range, and both inputs saturate. A program that outputs the right level plus noise, such as the last wall row plus a constant, gets most cells within 50 %. It also copies the shape of the last row, which partly satisfies `order_wrong`.

**Evidence.**
- R seed 3's champion, `(map2.add (row.rows.cols walls last) (replicate …))`, is the last wall row plus a computed constant. It is within 50 % in 87 % of cells, with `order_wrong` 0.12.
- The drop-one-neighbour program is within 50 % in only 53 % of cells, and its `order_wrong` is 0.21.
- Scaling by the mean magnitude instead (what `compare()` already does, line 1112) moves the neighbours from 1.045/1.059 to 0.962/0.978. It does not fix the ranking on its own, so it must be checked by the gate.

**Fix.** Scale by the magnitude of the correct values, not their range. Pass the gate of fault 1.

### 4. Histogram's cases cannot give partial credit (CONFIRMED)

**What.**
- The values are uniform over the whole `i32` range (`tests/corpus/futhark_problems/histogram/cases.tsv`), so every bin sum wraps. A wrong bin is off by a random amount, and the size of an error says nothing about how close the program is.
- More importantly, every one-edit neighbour of the target puts the elements in different bins, so its output is unrelated to the right one on every case.
- No case lets a partial program be exact.

**Evidence.**
- With values of 0..9 instead (same `k`, `n` and indices), the one-symbol neighbours still score 1.12–1.45 (only the two-node shift `(is + 1) % k` drops, to 0.89). Small values alone are not the fix.
- Adding two cases whose indices are already in `[0, k)` moves "bins `is`, no `% k`" from 1.286 (equal to the zero output) to 1.063. That beats every R and C champion, and it is the step evolution needs toward the `mod.k` it never built.

**Fix.** Stepping-stone cases that a partial program can solve exactly:
- indices already in range;
- `k` = 1, 2 and 3;
- values all 1, a count.

Each must be checked by the gate of fault 1. Small values remove the wrap, but measured alone they do not help.

### 5. Degenerate cases reward trivial wrong programs (CONFIRMED)

**What.** The held cases `held_1x37` (pathfinder, one row) and `held_k1_n13` (histogram, one bin) are solved exactly by trivial programs: the last wall row is the answer when there is one row, and all values in bin 0 is the answer when there is one bin. Each of these cases carries 1/8 or 1/5 of the fitness, so a program that is exact there jumps ahead of everything that is wrong everywhere.

**Evidence.**
- Pathfinder R seed 3's champion is exact on `held_1x37` and on no other case.
- 1,090 of R seed 3's 1,400 evaluations are exact on at least one case, against 271–278 for C.
- R seed 1 found `(row.rows.cols walls last)` (1.207) in generation 0 and stayed within 0.0003 of it for 59 generations: about 870 of its 1,400 evaluations sit at 1.2070–1.2074.
- Histogram C seeds 1 and 3 are exact only on `held_k1_n13`.

**Fix.** Keep edge sizes as checks of exactness, but do not let them outweigh the main cases. One option is to group a degenerate case's inputs with a normal case. Another is to add the stepping stones of fault 4, so trivial programs are not the only ones that are partly exact.

### 6. Behaviour diversity collapses while the reported diversity stays high (CONFIRMED)

**What.** The `distinct` column counts distinct program TEXTS (`futhark_evolve.rs:1415–1420`). The population keeps 23 of 24 distinct texts while the number of distinct outputs falls to about 5. Over a whole run, evolution also found fewer distinct behaviours than random search did.

**Evidence** (distinct quality-input vectors in the ledger; R seed 1 ran on a binary that did not record them, so its distinct fitness values are used instead):

| run | distinct texts a generation, last 20 | distinct behaviours a generation, last 20 | distinct behaviours, whole run |
|---|---|---|---|
| histogram R s1 / s2 / s3 | 23.1 / 23.1 / 23.0 | 5.1 / 7.5 / 4.5 | – / 205 / 78 |
| histogram C s1 / s2 / s3 | – | 11.3 (s2) | 205 / 189 / 190 |
| pathfinder R s1 / s2 / s3 | 23.1 / 23.2 / 23.0 | 10.3 / 18.6 / 5.7 | – / 469 / 120 |
| pathfinder C s1 / s2 / s3 | – | 10.7 (s2) | 201 / 208 / 219 |

Children equal in fitness to their best parent: histogram R 59 %, 49 % and 74 %; pathfinder R 44 %, 30 % and 65 %. From generation 18 on, pathfinder R seed 3's best equals its median in every generation but one.

**Fix.** Report distinct behaviours (outputs) beside distinct texts. Make selection aware of behaviour: at most one elite per output, and tournament pools and partners drawn from distinct outputs. This is a fold into the existing selection, not a new mechanism. It only helps once faults 1 to 5 give the plateau a slope.

## Faults that distort the measurements

### 7. Elites are scored again and counted (CONFIRMED)

**What.** The 2 elites are cloned into the next generation's candidates (`futhark_evolve.rs:1527–1538`). Every candidate is scored (`1364–1372`), and `Judge::evaluate` adds one evaluation before its cache lookup (`697–698`).

**Evidence.**
- 116 of every R run's 1,400 evaluations (8.3 %) are elites scored again: 116 `elite` rows in each histogram and pathfinder R ledger.
- They are also counted in "first exact at evaluation" (for example, 8 of reduce R seed 2's 118).

**Fix.** Carry an elite's evaluation over. Count only individuals not scored before.

### 8. Repeated programs count as evaluations, and this costs random search more (CONFIRMED)

**What.** A cache hit is an evaluation (`futhark_evolve.rs:44`, `380`). Random search draws small programs, which repeat, so it loses more of its budget to repeats than evolution does.

**Evidence:**

| benchmark | regime | repeated texts (seeds 1/2/3) | distinct programs (seeds 1/2/3) |
|---|---|---|---|
| histogram | R | 170 / 136 / 160 (116 of each are elites) | 1,230 / 1,264 / 1,240 |
| histogram | C | 382 / 361 / 334 | 1,018 / 1,039 / 1,066 |
| pathfinder | R | 238 / 146 / 206 | 1,162 / 1,254 / 1,194 |
| pathfinder | C | 496 / 472 / 481 | 904 / 928 / 919 |

Evolution scored 15–37 % MORE distinct programs than random search did, and still did no better. This fault does not explain the H2 result. Fixing it would give C more programs, not R.

**Fix.** Make the budget unit a program not seen before (or a behaviour not seen before), the same for both regimes. Report both counts.

### 9. The pump and the fold cannot change stage-1 fitness, and they repeat themselves (CONFIRMED)

**What.**
- Every e-graph rule is an identity (`futhark_sig.rs:1226`), so a pump or fold child has its parent's outputs, and in stage 1 its parent's fitness exactly.
- The pump takes variants of the best (`futhark_evolve.rs:1540–1560`). On a plateau the best does not change, so it sends in the same variants every 5 generations.
- Pump and fold children are placed before marriage and mutation children, so they take those children's places.

**Evidence:**
- **Equal fitness, every time.** 314 pump and fold children on histogram and 351 on pathfinder, all equal in fitness to their parent.
- **Budget spent.** Pump and fold take 61, 119 and 134 evaluations of the histogram R runs, and 116, 123 and 112 of the pathfinder R runs.
- **Repeats.** Pump children that repeat an earlier program: histogram 82 %, 29 % and 73 %; pathfinder 78 %, 0 % and 78 %.

**Fix.** In stage 1, do not run the pump or the fold, or do not count their children. In any stage, run the pump only when the best has changed.

### 10. R and C with the same seed share generation 0 (CONFIRMED)

**What.** Both regimes draw generation 0 from `GeneRng::new(seed)` (`futhark_evolve.rs:1255`, `1328–1332`).

**Evidence.**
- Generation 0 is identical text for text in all 12 R/C pairs.
- Reduce seed 1's "R and C both exact at evaluation 19" is one program, found once, in that shared generation, so it gives no information on H2.

**Fix.** Give C its own seed stream (for example, the seed combined with a regime tag), or report generation 0 as shared and compare from generation 1.

### 11. The held-out cases are in the fitness (CONFIRMED)

**What.** `Judge::quality` turns every case into fitness inputs, held-out ones included (`futhark_evolve.rs:682–692`). The split is used only to count exact cases. This follows the owner's rule (the held-out cases are the quality group's validation column). But H1 says "exactly correct on held-out inputs", and selection has seen those inputs.

**Fix.** Keep the validation column. Add a third set of cases that selection never sees, and make H1's claim on that set.

### 12. `--rescore` writes verdicts by the old rule (CONFIRMED)

**What.** The run's own report applies "median below the floor AND at least 9 of 11 pairs faster" (`futhark_evolve.rs:1724`). The `--rescore` path still uses the median alone (`1217–1223`).

**Evidence.** `fbench/rescore_reduce_R_s1/rescore.tsv` says "beat" with 8 of 11 pairs faster. The published summary corrected this by hand.

**Fix.** Use one function for the verdict on both paths.

### 13. The timing measurements (CONFIRMED)

**What and evidence.**
- **The batched path fails every A/A gate:** 13, 20, 19 and 19 of 22 pairs within 5 % (`fbench/superseded_batch_gate/*/aa_gate.tsv`). P runs therefore use `--eval single`, and timings from the batched path are not usable.
- **Selection in P chose on noise.** On pathfinder, the in-run medians were 0.940 for the champion and 0.955 for the start. The re-times are 0.970 and 0.945.
- **The verdict's floor is the least of 22 A/A ratios**, an extreme value that moves between re-times of the same two programs: 0.9525, 0.9845, 0.9888 and 0.9671. That spread is larger than the 2.5 % difference the pathfinder report draws between the champion and its start. Comparing a median of 11 against the minimum of 22 compares two different statistics.
- **Pathfinder is timed on a quality case** (256 × 256, about 1 ms per run). The micro benchmarks were moved off their quality cases after their 1,000-element case gave clock-state clusters.

**Fix.**
- Make the batched path pass the A/A gate before any batched timing is used.
- Compare the champion's median with a spread of A/A medians, for example by re-timing the A/A set several times.
- Give pathfinder its own large timing input.

### 14. Legal programs the compiler cannot build are scored worst (CONFIRMED)

**What.** A program the branch compiler fails on gets every quality input set to 1 (`Quality::worst`, `futhark_evolve.rs:318`). Under `docs/Rules.md` that is a kingdom bug: a legal gene must run.

**Evidence** (failures of legal genes in `bugs.log`):
- histogram: R 2, 3 and 0; C 8, 8 and 13; P 1; the planted control 1;
- pathfinder: R seed 1 1, R seed 3 2; C seed 1 2, C seed 3 1.

Most are "SetMem in a GPU kernel". There are also internal compiler errors in histogram C and the pathfinder pilot. Random search hits these about 6 times as often as evolution on histogram (29 against 5). The effect is under 1 % of any run's evaluations (13 of 1,400 at most).

**Fix.** `docs/PROPOSAL_memory_views_setmem.md` for SetMem. Report each internal compiler error upstream and record it as a kingdom bug.

### 15. The report still takes most of its champion fields from the last batch (CONFIRMED, latent)

**What.** f8409a2f added `best_ever_*` from the hall of fame (`futhark_evolve.rs:1730–1732`), and the published C champions were rescored from hall-of-fame rank 1. The ledger minimum equals hall-of-fame rank 1 in all 12 histogram and pathfinder runs, so the fix is verified. But `best_hff`, `best_program`, `best_nodes`, `best.fut` and the champion's 11-pair re-time still come from `population[0]` (`1669–1718`). For C that is the last random batch.

**Fix.** Take every champion field, and the re-time, from the hall of fame's rank 1.

### 16. The hall of fame mixes stage-1 and stage-2 fitness (CONFIRMED, latent)

**What.**
- In stage 1, an exact program scores 0.
- In stage 2 the fitness includes speed, so every program scores above 0.
- `HallOfFame::offer` keeps the lower fitness for a key (`runner.rs:290`), and the population re-score at the stage change (`futhark_evolve.rs:1406–1411`) is not offered.
- So in any run that reaches stage 2, rank 1 and `best_ever_*` stay with the first exact program, never the fastest one.

This did not affect the counted R and C runs, which stop at the first exact program, nor P, which starts in stage 2.

**Fix.** Clear the hall of fame at the stage change, or keep one hall of fame per stage.

### 17. Different binaries for seed 1 (SUSPECTED)

**What.** Histogram R seed 1 and pathfinder R seed 1 ran on e64bd1f3 with the per-candidate path. Every other R and C seed ran on 50339252 with the batched path. The two paths were shown equivalent on the series and the first 400 ledger rows of one run (histogram R seed 2). Pathfinder R seed 1, the one seed that never moved, is on the older path.

**Fix.** Run all seeds of a comparison on one binary, or show equivalence on a whole run.

## Minor faults

### 18. Bloat under neutral drift (CONFIRMED)

- **Growth.** Mean program size grows from 8–17 nodes in generation 0 to 35–124 nodes by generation 55. 49–57 % of mutation children are larger than their parent.
- **Why N does not stop it.** The program size N (8 for histogram, 13 for pathfinder) bounds the functions of each TYPE, and there are 607 and 832 types, so it does not bound the total.
- **Cost.** Big programs compile slowly: histogram R took 58–154 min against C's 15–18 min for the same 1,400 evaluations.
- **Cause.** This is a symptom of faults 1, 2 and 6, not a cause of them. The fix is the fitness, not a length penalty.

### 19. Marriage favours swaps near the first parent's root (SUSPECTED)

**What.** `Genetics::marry` enumerates pairs in level order of parent A, then parent B, and keeps the first 256 that fit (`futhark_evolve.rs:852`). For large parents (100 × 100 nodes), the drawn pair is always one of A's first few nodes.

**Fix.** Draw uniformly from the whole domain, for example with reservoir sampling.

### 20. Crossover's second child is discarded, and crossover is never alone (CONFIRMED)

**What.** `Genetics::crossover` writes only into `a` and drops `other` (`futhark_evolve.rs:827–828`). Every crossover child then also gets 1–4 mutations (`1650–1651`), so the operator-effectiveness tables cannot show what crossover itself does.

**Fix.** Keep both children. Make mutation after crossover a separate, recorded choice.

### 21. Partners need only a different text (CONFIRMED)

**What.** Marriage and crossover partners must differ in text (`futhark_evolve.rs:1602`, `1639`). On a plateau most such partners give the same output, so the exchange cannot combine two different behaviours.

**Fix.** Require different outputs, as fault 6 proposes.

### 22. A small search (SUSPECTED)

**What.** 24 individuals, 2 elites, tournament of 3, generational replacement, 58 generations. An evolutionary method's advantage over random sampling usually needs more generations than this.

**Fix.** Only after faults 1 to 9. A larger budget on the present fitness would buy more plateau.

## Checked and found sound

| check | finding | evidence |
|---|---|---|
| TrueNorth is monotone | yes: in stage 1, the angle is `arccos(1 − mean(x²))`, strictly increasing in every input | `../hff/src/core_functions.rs:73–100`; reproduced by the Python check |
| lower is better everywhere | yes | the sort (`futhark_evolve.rs:1390`), the tournament (`1591`), the partner tournament (`1610`), the elites, the hall of fame (`runner.rs:290`) |
| tournament size and pressure | size 3, with replacement; ties keep the first drawn, which is random among equals | `futhark_evolve.rs:1587–1596` |
| the published programs fit the head | yes: histogram's reference has 4 functions (N = 8), pathfinder's 13 (N = 13); each run's control expresses and compiles them exactly. Alphabets are at most 28 symbols a type, so a point mutation reaches any one symbol with probability at least 1/27 | run logs; `kingdoms/futhark_shapes/sig/*/functions.tsv` |
| neutral variation at the text level | rare: 0–1 % of children repeat a parent's text (marriage up to 5 %). The neutrality is in the outputs (fault 6) | ledgers |
| the hall of fame and the report agree | yes, for every R and C run: ledger minimum = hall-of-fame rank 1; R's `best_hff` equals both | ledgers, `hall_of_fame/rank1.txt`, `report.tsv` |
| R and C get equal budgets | yes in counted evaluations (1,400 each). R saw more distinct programs (fault 8) | ledgers |
| known fault 1 (plateaus) | verified: pathfinder R seed 3, 705 evaluations (582 distinct programs) at 1.015466; histogram R seed 2, 254 distinct programs at 1.220922 | ledgers, `series.tsv` |

## Order of the fixes

1. The neighbour gate of fault 1, on the host interpreter, before any run.
2. Faults 2 to 5 (the inputs, the scale, stepping-stone cases, degenerate cases), until the gate passes for histogram and pathfinder.
3. Faults 7 to 10 (what counts as an evaluation, pump and fold in stage 1, separate seeds), so R and C compare like with like.
4. Fault 6 (selection on distinct behaviours), then rerun R and C.
5. Faults 11 to 17 before any further speed claim.

Separately: `python3 docs/build_index.py --check` already fails on this branch (the index is stale before this file was added), so the docs-index test in `cargo test` fails until the index is regenerated.
