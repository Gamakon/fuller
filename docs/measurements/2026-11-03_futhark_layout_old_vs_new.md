# The context layout, old against new: Fable's measurements 2 and 3 (2026-11-03)

**What was measured.** The review of 2026-11-01 (`docs/reviews/2026-11-01_futhark_kingdom_design_fable.md`) predicted
that the smaller gene layout (contexts as counts of bound variables, section 3g; pathfinder 832 -> 184 types, the gene
217,015 -> 57,487 words at N = 32) would halve the share of neutral crossover children (measurement 2) and reach an
exact program in fewer evaluations in at least 2 of 3 seeds (measurement 3). Both were measured on the GPU with the
same parents, seeds and budgets on both layouts: the old layout is the binary of b195fc6f with the pool as it was then
(`/private/tmp/claude-501/fable_m2/oldpool`), the new the build of main 78744bba.

## Measurement 2: the operators, 700 children of the same 23 (histogram) and 25 (pathfinder) evolved parents

Children better than their parent (per cent) / children with the parent's output (per cent). Results in
`/private/tmp/claude-501/fable_m2/` and `/private/tmp/claude-501/fpool/rows_*`.

| operator | histogram old | histogram new | pathfinder old | pathfinder new |
|---|---|---|---|---|
| inversion | 3.7 / 32 | 3.7 / 30 | 18.1 / 25 | 18.9 / 25 |
| transposition | 3.7 / 35 | 4.0 / 35 | 14.4 / 28 | 14.6 / 28 |
| crossover | 2.7 / 90 | 2.3 / 90 | 11.1 / 68 | 10.7 / 68 |
| inversion, rows | 1.3 / 69 | 1.4 / 67 | | |
| transposition, rows | 2.7 / 10 | 2.9 / 9 | | |
| crossover, rows | 1.3 / 96 | 0.6 / 96 | | |
| inversion, fractions | 1.4 / 71 | 1.4 / 71 | | |
| transposition, fractions | 2.0 / 1 | 1.7 / 2 | | |
| crossover, fractions | 3.6 / 91 | 3.3 / 90 | | |

Every rate is within 0.7 points of the other layout. The neutral share of crossover did not fall: the operators
already drew a coding cell before the change (section 3g, fix 2), so the padding they no longer carry was never where
they acted.

## Measurement 3: evolution (regime R), 1,400 evaluations or 30 minutes, population 24, point mutation, seeds 1-3

Best HFF (0 is exact, lower is better) and the generation of the first exact program. Results in
`/private/tmp/claude-501/fable_m3/m3/`. The run new pathfinder seed 3 did not run (the queue was stopped to free the
GPU for k-means); the controls (regime C) of seeds 1 and 2 ran before the reordering and none reached an exact program
(best HFF 0.99-1.12).

| run | generations | evaluations | best HFF | first exact |
|---|---|---|---|---|
| histogram, new, seed 1 | 49 | 823 | 1.1345 | - |
| histogram, new, seed 2 | 41 | 741 | 1.0442 | - |
| histogram, new, seed 3 | 62 | 1,184 | 1.0872 | - |
| histogram, old, seed 1 | 36 | 641 | 1.0390 | - |
| histogram, old, seed 2 | 6 | 118 | 0.0000 | generation 6 |
| histogram, old, seed 3 | 37 | 715 | 1.0902 | - |
| pathfinder, new, seed 1 | 70 | 1,353 | 0.9668 | - |
| pathfinder, new, seed 2 | 56 | 1,075 | 0.9772 | - |
| pathfinder, old, seed 1 | 69 | 1,259 | 0.8974 | - |
| pathfinder, old, seed 2 | 68 | 1,299 | 0.9850 | - |
| pathfinder, old, seed 3 | 64 | 1,265 | 0.9516 | - |

The one exact program (the published histogram, rediscovered: the same work count) came from the old layout. The new
layout reached no exact program in 5 runs; the review's prediction did not hold. Best HFF at the time bound is alike
on both layouts. No run on either layout found a faster correct program than the published one.

**What the smaller layout buys** is expression and variation time, not operator effect: that is the measurement
still to take (measurement 4, the time a generation on each layout, from these runs' streams).
