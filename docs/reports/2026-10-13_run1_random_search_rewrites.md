# Run 1: random search resumed with the full operator pool (fit_SF_rw2)

**Result: no correct kernel in 50 generations; the best closeness improved by 0.003 (1.2257 → 1.2225) and was flat from generation 20.**

## Configuration
- Start: generation 58 of fit_SF_pump (random genes, 50+ generations of selection), 80 individuals, re-encoded at fuller c12206a (`logs/HGP/001/popSF_g58c`).
- Binary: fuller c12206a + phylu f94bf3fb (`/tmp/claude-501/rel`), built from committed code only.
- Operators, all on: the rewrite pool on half the offspring (16 structural rules + 12 generic typed rewrites, drawn among those whose pattern matches, any individual); engine point mutation 33 per offspring on the rest; typed crossover 8 pairs; dedup re-variation; pump every 20 generations refilling with fresh random kernels.
- Off: models, swap gate, split by feasibility. Empty library.
- Fitness: infeasible individuals graded by the closeness angle (9 columns), always above every feasible one.
- Log: `logs/HGP/001/fit_SF_rw2/` (generations.tsv, ledger.tsv, fitness.log).

## Best closeness angle (0 = correct), generation by generation
| gen | evaluated | correct | best closeness |
|---|---|---|---|
| 0 | 80 | 0 | 1.2257 |
| 10 | 76 | 0 | 1.2232 |
| 20 | 75 | 0 | 1.2225 |
| 30 | 68 | 0 | 1.2225 |
| 40 | 74 | 0 | 1.2225 |
| 50 | 75 | 0 | 1.2225 |

## Offspring by operator (all 50 generations)
| operator | children | not scored worst | write nothing | unscorable |
|---|---|---|---|---|
| rewrite pool | 1,801 | 1,220 (68 %) | 578 | 3 |
| dedup | 1,186 | 901 (76 %) | 284 | 1 |
| point mutation (33 a child) | 739 | 336 (45 %) | 399 | 4 |
| pump (fresh random) | 102 | 2 (2 %) | 90 | 10 |

Rewrite pool, by operator (children, not scored worst): delete-statement 275/89, replace-leaf 212/145, swap-op 192/155, duplicate-statement 188/148, swap-operands 187/145, index-literal 179/153, swap-adjacent 175/147, wrap-unary 159/135, unwrap-unary 145/30, swap-leaves 81/73. Structural rules fired 8 times (drop-dead-store 4, wg-shape 3, loop-end 1): their patterns are tiled kernels, which random kernels lack.

## What it shows
- Single typed rewrites keep a kernel's structure far better than many point changes (68 % against 45 % not broken); delete-statement and unwrap-unary break most often.
- Fresh random kernels almost never compete: 98 % write nothing or cannot be scored.
- Neither gives a path to correctness here: the population is one family whose output is uncorrelated with ΦᵀΦ (correlation column ≈ 0.5, i.e. ρ ≈ 0).

## Next
- A sampler whose random kernels write values derived from Φ (fresh draws are 98 % wasted today).
- A sketch start: the outline the problem fixes (cell ownership, the item loop, the store) with random holes.
- The engine's Karva crossover, re-measured (kingdom-fixer-2, pending).
