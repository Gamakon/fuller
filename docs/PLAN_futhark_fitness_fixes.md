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

**Built (2026-10-28).** `futhark_evolve --check --problem P --program-size N --eval rust --out DIR` (the GPU evaluation path; `--eval batch` also works, quality is bit-identical). The verdict is `fuller::runner::slope_check`, tested in `runner::tests::the_slope_check_passes_a_slope_and_fails_a_plateau_or_a_reversed_slope`; the distance is `futhark_sig::tree_distance` (Zhang and Shasha). A discovery run (R, C) now needs `--gate DIR` of a passing check of the same problem, program size, cases and quality-input formula.
- **The programs.** The reference gene (the reference written back into a random gene of the check's own seed stream); every one-edit neighbour of it (each expressed cell set to each other member of its alphabet, each tree once); 1,400 random genes (`--check-random`), every draw, repeats included.
- **The gate passes only if both:**
  1. the neighbours rank above the random genes: the median inexact neighbour is better than at least 99 % of the random genes (ties count as not beaten). Random search's best of its draws sets the plateau a population sits on, so a step from the target must look better than nearly all random programs;
  2. Spearman(fitness, tree distance to the reference) over every program of the check is positive.
- Neighbours exact on every case (equal programs) score 0 whatever the fitness; they are counted apart and are not in the ranking.
- Also reported: the share of neighbours better than the median random gene (the expectation below) and than the best one, the probability a neighbour beats a random gene, Spearman over the random genes alone, the all-zeros share, distinct outputs among the random genes.
- `OUT/check_outputs.bin` keeps every program's GPU outputs; `--check-replay FILE` re-scores them, so a change to the quality inputs alone is re-checked on the GPU's own outputs without running the programs again.

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

**Built (2026-10-28, commit 50e91778 and after).**
- The reader, `src/futhark_reader.rs`: Futhark text to a tree of the kingdom, typed, each row's spelling used as a pattern. Round trip: 1,600 rendered random genes (400 in each of the four kingdoms) read back to the same program text. 5 of them read to a different tree with the same text, because the kingdom spells `zero` and `prev` the same in every index class. Pathfinder's published kernel does not read: it uses a tuple, `map5` and `rotate`.
- The pool, `examples/futhark_pool.rs`, committed under `kingdoms/futhark_shapes/sig/<benchmark>/pool/`. `manifest.tsv` holds the counts and the gaps by construct; `bench_functions.tsv` and `llm_programs.tsv` hold every function and program, read or gap.
  - **Benchmarks:** only 10 of the 1,575 functions in the 221 files read, mostly `i32` constants. The kingdom has only `i32` arrays and no tuples, so f32 code (158 float literals), tuples (324), type parameters (76) and records are outside it. These are gaps in the kingdom. Fixing the reader would not close them. The leave-one-out rule takes the benchmark's own file out of the discovery rings: `micro/reduce_by_index.fut`, `rodinia/pathfinder/pathfinder.fut`.
  - **Topic programs (Andrew's correction, 2026-10-28):** Sonnet was given one topic per call from 60 unrelated topics (blur, hash, Kadane, Horner, game of life, ...), the kingdom's operations, and the entry signature, with no description of the benchmark's task. Histogram: 48 of 60 read. Pathfinder: 45 of 60 read. 120 calls, $1.74. Many histogram topic programs work on `hist.add.n is vs` (no `% k`), because the contract lists `hist`.
  - An earlier set of programs written for the benchmark's task (histogram 124 calls, pathfinder 34, $4.77) was stopped and is not used: the rings must not look like the task.
- The rings, the canvas, ring walk and growth, `src/futhark_rings.rs`. Empty rings are filled from the same class in other scopes, or from the slot family when a whole class is a gap (`OUT/rings.tsv` lists every fill).
- Wiring, `examples/futhark_evolve.rs`: `--start llm-ring --pool DIR --ring-sources bench|llm|both` (generation 0 = the pool's topic programs on canvases, every fresh gene a canvas); `--measure grow|point|walk --count N` (one operator application per child, on the GPU path).

**Andrew's clarification (2026-10-29): the canvas and the population are different things.**
- **The canvas is general and permanent.** It is one large store of real programs on unrelated topics, kept for ever and grown over time, whatever the problem. Its job is only to fill the open reading frames, so an unread cell may hold a fragment of any real program. Today the topic programs are written per benchmark kingdom, because a kingdom's types include its own inputs. The store keeps the programs themselves, and each new problem's rings are made from them by reading them into that problem's kingdom.
- **The population is the problem's.** For a problem with a reference program: the reference goes code → AST → Karva and is written onto the canvas. The other 15 to 20 individuals are Sonnet's high-perplexity alternatives FOR THE SAME TASK, each written onto the canvas the same way. (The 124 task programs for histogram belong here, not in the rings.) Without a reference (discovery), all individuals are Sonnet's alternatives.
- **No near copies.** An alternative within a few edits of another individual (one symbol apart, say), or with the same outputs, is not an individual. It is thrown away and replaced by a new, genuinely different alternative from Sonnet. The bar (a minimum tree edit distance, and distinct outputs) is stated and recorded before the population is built.
- So generation 0 is the reference plus distinct alternatives on canvases, never the topic programs themselves.

**Expectations, stated before the measurement (GPU, `--eval rust`).**

| measurement | random padding (today) | expected, ring canvas |
|---|---|---|
| all-zero share of 1,400 fresh genes (`--check`'s random genes) | 44 % histogram, 49 % pathfinder (study ledgers) | under 20 % |
| distinct outputs of 1,400 fresh genes | 189–208 | more than 400 |
| growth mutation of a topic program: all-zero share of children | about the fresh-gene share | under half of random padding's |
| growth mutation: children with an unseen output | | more than random padding's |

### I. Reading-frame operators, once the open reading frames are real (Andrew, 2026-10-28)
With random padding, an operator that moves unread cells into the reading frame brings in junk, which is why only point mutation on coding cells is used today (`futhark_evolve.rs:798`). With ring padding, the unread cells are real sequences, so such operators become worth having. Each works inside one type's frame, so every symbol stays a legal row of its own type, and a tail cell keeps a terminal.
1. **Ring walk:** advance one type's head or tail window by k positions along its ring. The reading frame takes in the next real symbols, and the rest shifts along.
2. **Transposition** (the classic GEP IS and RIS operators): copy a short run of a type's head, or one starting with a function (RIS), to another place in the same head; the head keeps its length.
3. **Inversion:** reverse a run inside one type's head.
4. **Sorting:** reorder the cells within a head; what reordering is to be settled with Andrew before it is built.

Each operator is recorded as its own mutation event and measured on its own:
- the share of children with an output not seen before;
- the share of children better than their parent;
- the share giving the all-zeros output.

They are built after section H, once its rings exist, and measured on the GPU before they enter a run.

**One at a time (Andrew, 2026-10-28).** Each step changes one thing, and is tested before the next:
1. **Baseline:** the fixed fitness (A to G), the ring canvas (H) and today's point mutation. R and C on histogram and pathfinder, 3 seeds each.
2. **Ring walk replaces point mutation:** the same runs, the same seeds and the same budget, with ring walk as the only mutation. Crossover and marriage are unchanged.
3. **Then transposition, then inversion, then sorting,** each added alone, each with the same runs, and kept only if it measures better than the step before.

Each step reports its per-operator measurements and R against C, as a table in this plan, before the next step is built.

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

## Measurements of the check

Expectations, stated before each measurement; results after.

| # | measurement (GPU, `--eval rust`, 1,400 random genes) | fitness, cases | expected | measured |
|---|---|---|---|---|
| A1 | histogram (N = 8), the fitness of the study | 7 inputs a case, the 5 stored cases | fail: the median neighbour beats under 99 % of random genes (the review puts neighbours at 1.03 to 1.32, inside random search's band); Spearman positive or near 0 (unlike the ledgers, random genes do not bloat) | **fail**: median neighbour beats 98.86 % of random genes; Spearman −0.080 (random genes alone −0.215); 66.7 % of neighbours better than the median random gene, 15.9 % better than the best; all-zeros share 44.2 %; 209 distinct outputs in 1,400 draws (1,039 distinct programs) |
| A1′ | A1's GPU outputs re-scored with graded per-cell inputs (offline, 20 variants of scale, shape and exactness inputs) | old cases | Spearman positive; neighbours still tied with random genes, because whole-range values wrap | Spearman +0.18 to +0.31 in every variant; median neighbour beats 40 to 52 % of random genes: **fail**. The old cases give no partial credit (fault 4); section C is needed |
| A2 | pathfinder (N = 13), the fitness of the study | 7 inputs a case, the 8 stored cases | fail, as A1 | |
| A3 | histogram, the fitness of the study | the new cases (11 training with stepping stones, 4 held-out, 4 test never in the fitness) | the stepping stones let partial programs be exact on some cases: the median neighbour beats more random genes than in A1; Spearman still not clearly positive (the step inputs) | |
| A4 | pathfinder, the fitness of the study | the new cases (10 training with 2- and 3-row stepping stones, 5 held-out, 4 test) | as A2, slightly better | |

## Expectations, stated before any run
| measurement | expected |
|---|---|
| fitness-distance correlation after B and C | positive on histogram and pathfinder |
| one-edit neighbours against random genes | every neighbour ranks above the median random gene |
| R against C, histogram and pathfinder, after all fixes | R's best better than C's best in at least 2 of 3 seeds each |
| exact programs from random, histogram and pathfinder | possible, not expected |
