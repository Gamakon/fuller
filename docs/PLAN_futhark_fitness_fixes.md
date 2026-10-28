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

### H. Sample where real programs are: a pool of LLM-written Futhark (Andrew, 2026-10-28)
**The failing.** Random programs are dense at one dead point: in random search, 44 % (histogram) and 49 % (pathfinder) of programs output all zeros. LLM-written seeds would fix generation 0 only. When a gene is written, its unread cells are padded with random symbols. A mutation that grows the tree (a terminal turned function, or a function with more arguments) then exposes those random cells, so random junk returns within a few generations.

**The fix.**
1. **Generate:** language models write many real Futhark programs for the benchmark's signature.
2. **Read:** a reader parses Futhark source into the kingdom's AST. It does not exist yet for the general signature kingdom and is the first piece to build. A program the reader cannot express is a kingdom gap, recorded, never routed around.
3. **The rings (Andrew's method, agreed 2026-10-28):** each read program goes to Karva, and its coding symbols, unpadded, are split by type and appended to that type's ring. The programs are added one by one, until the rings are full. Each type has three rings: one of head symbols (functions and terminals), one of terminals (for tail cells, which take terminals only), and one of constants (for the constant cells).
4. **The canvas:** a new gene starts as a window of each type's rings, each from its own random start point. It is a blank open reading frame made of real sequences.
5. **Overwrite:** the next LLM program goes code → AST → Karva and is written onto the canvas (`write_back`), replacing only the cells the walk reads. Every unread cell stays a real sequence, so whatever a later mutation exposes is real symbols in real proportions.
6. **Two checks:**
   - Karva lays a tree out level by level, so exposed ring cells decode to a mix of real fragments, not one whole snippet. The measurements below say whether that is good enough.
   - Types LLM programs never reach (our 607 types are basic shapes inside scopes): a type's ring is built from the same operations recorded at its basic shape in other scopes, each written as a legal row of its own type. A type with no recorded operation at all is a kingdom gap to fix, never filled at random.

**Not done:** next-symbol probabilities. They would need a model trained (LoRA) to predict the next Karva symbol, which we will not do. The pool of real sequences replaces them.

**Measured before any run:** the share of random-from-pool programs giving the all-zeros output (now 44–49 %), the number of distinct outputs per 1,400 draws (now 189–208), and the check of section A.

## Order agreed with Andrew (2026-10-28)
1. Fix all the fitness: A to G. The check of section A passes on histogram and pathfinder.
2. The canvas: section H.
3. Rerun histogram and pathfinder, R and C. The goal is to prove the search learns: evolution beats random search.
4. Then the k-means centroid update (UltraDim's A7 against the A6 segmented form), not the trellis: it fits the kingdom better.

All measurement is on the GPU. The CPU interpreter is a test reference only.

## Order and cost (first draft)
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
