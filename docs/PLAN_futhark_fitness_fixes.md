# PLAN: fix the Futhark fitness so evolution can beat random search

## Context

**Objective.** Make the fitness of the general Futhark runner (`examples/futhark_evolve.rs`, `src/futhark_sig.rs`) reward getting closer to the target, so that evolution beats random search on histogram and pathfinder.

**Why.** The benchmark study (`docs/reports/2026-10-27_STUDY_evolving_benchmarks_summary.md`) found evolution no better than random search on all four benchmarks. Random search only beats evolution when the fitness is broken. The advisor's review (`docs/reviews/2026-10-28_futhark_evolve_faults.md`) confirmed it:
- fitness gets *worse* as programs get closer to the published one (Spearman −0.68 histogram, −0.52 pathfinder);
- one-edit neighbours of the target score inside the plateau band;
- the review lists 22 faults: 6 break evolution, 11 distort measurement, 5 minor.

**Approach.** A check that the fitness has a slope toward the target comes first, before any run. Then the fitness is fixed until it passes, then the budget, diversity and machinery, then the reruns. HFF (the hff library's TrueNorth) stays the only judge, with every measurement its own input and nothing averaged before it. The staged fitness stays: quality only until exact, then the two-level HFF with speed.

**Status (2026-10-28).** Proposed, not built.

## Fixes, by the faults they repair (fault numbers from the review)

### A. A check before any run (fault 1)
- For each benchmark, score every one-edit neighbour of the reference gene and a few thousand random genes on the host interpreter (`Interp::eval`, CPU, seconds).
- A run may start only if the neighbours rank above the random genes and the fitness-distance correlation is positive.
- This is a gate, like the A/A timing gate. A fitness that fails it cannot guide a search.

### B. The quality inputs (faults 1, 2, 3)
1. **Graded per-cell inputs:** each output cell's error is its own input, scaled by the magnitude of the correct value (not the output's range), with no thresholds.
2. **Drop the duplicate thresholds:** keep one share of wrong cells; drop `outside_1pct` and `outside_10pct`, which copy it.
3. **Shape credit:** per case, the rank agreement and the correlation of the output with the correct output, so a structurally right program beats a level-matching one (pathfinder's drop-one-neighbour program against "the last row plus a constant").

### C. The cases (faults 4, 5)
- Stepping-stone cases a partly right program can solve exactly. For histogram: indices already in range, k = 1, 2 and 3, values all 1.
- More normal cases, so the degenerate cases (one row, one bin) carry less weight. Every case stays its own input; nothing is averaged.

### D. Diversity on behaviour (faults 6, 21)
- Parents and crossover partners must differ in OUTPUT, not only in text: the parent-pairs rule extended to behaviour.
- At most one elite per distinct output.
- Report distinct outputs beside distinct texts in the stream and `series.tsv`.

### E. A fair budget (faults 7, 8, 10)
- An evaluation is a program not scored before. Elites and repeats are free. Report both counts.
- C gets its own seed stream (the seed combined with a regime tag), so R and C do not share generation 0.

### F. The machinery (faults 9, 18, 19, 20)
- Keep the pump and the fold. The pump fires only when the best has changed; children that repeat a program already scored are not counted.
- Crossover keeps both children; mutation after crossover is a separate, recorded choice.
- Marriage draws its swap point uniformly over the whole domain.

### G. The claims (faults 11 to 17)
- Keep the held-out cases in the fitness as the validation column. Add a third test set that selection never sees, and make the exactness claim on it.
- Every champion field and the re-time come from the hall of fame's rank 1. One hall of fame per stage.
- All seeds of a comparison run on one binary. One verdict function on both the run and `--rescore` paths.
- SetMem: the rust-wgpu backend's patches (`futhark-rust-backend/`) and `docs/PROPOSAL_memory_views_setmem.md`. Each internal compiler error is recorded as a kingdom bug.

## Order and cost
1. A and B together, until histogram and pathfinder pass the check (about a day).
2. C, then E, then D, then F.
3. Rerun R and C for histogram and pathfinder, 3 seeds each (about 6 h of GPU).
4. G before any further speed claim.

## Expectations, stated before any run
| measurement | expected |
|---|---|
| fitness-distance correlation after B and C | positive on histogram and pathfinder |
| one-edit neighbours against random genes | every neighbour ranks above the median random gene |
| R against C, histogram and pathfinder, after all fixes | R's best better than C's best in at least 2 of 3 seeds each |
| exact programs from random, histogram and pathfinder | possible, not expected |
