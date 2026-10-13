# Run 2: the rule beam on 24 correct kernels (fit_beam_R24)

**Result: the best fitness fell 17 % in 5 generations (0.1115 → 0.0927) and was flat for the last 14; every structural-rule child was correct. Paired re-timing is queued after run 3; until then the speed figures below are single timings and drift.**

## Configuration
- Start: popR24 — 24 correct kernels written by Sonnet earlier (the 10 best one-step and 14 two-step members of popR_rules), re-encoded with typed tokens (`logs/HGP/001/popR24_t`). Empty library.
- Binary: fuller 430c299 (the emitter fix for unrolled item loops) + phylu f94bf3fb, `/tmp/claude-501/rel2`.
- Operators, all on: --rule-all at every pump (every one of the 16 structural rules at every site of every distinct correct kernel); the rewrite pool on half the offspring (16 rules + 12 generic rewrites); point mutation 2 a row; typed crossover 4 pairs; dedup; random refill of any rows left.
- Population 24 (intake 16, champion 8), 20 generations, pump every generation, no models, no swap gate.
- Log: `logs/HGP/001/fit_beam_R24/`.

## Best fitness (lower is better)
| gen | best | correct of evaluated |
|---|---|---|
| 0 | 0.1115 (r6, 11.5 / 11.0 / 9.2× shipped, single timing) | 23 of 24 |
| 3 | 0.0988 | 9 of 20 |
| 4 | 0.0959 | 5 of 20 |
| 5–19 | 0.0927 (r18, 13.7 / 13.3 / 13.0× shipped, single timing) | 5–10 of 20 |

The best came from a generic rewrite: swap-operands of an index sum, applied to a 0.0988 parent, which was itself a child of the rule climb in generation 3.

## Offspring by operator (correct / children; beat generation 0's best)
| operator | children | correct | beat 0.1115 |
|---|---|---|---|
| structural rules (all kinds) | 247 | 247 (100 %) | 17 |
| — unroll-k | 14 | 14 | 6 |
| — pad-shared | 58 | 58 | 3 |
| — shrink-rows | 31 | 31 | 3 |
| — grow-rows | 19 | 19 | 2 |
| — drop-dead-store, un-unroll-k, unroll-k4 | 70 | 70 | 3 |
| — drop-zero-fill, double-buffer, roll-k, split-accumulator | 55 | 55 | 0 |
| generic rewrites | 107 | 44 (41 %) | 5 |
| dedup | 104 | 15 | 3 |
| point mutation (2 a row) | 45 | 4 | 0 |
| pump random refill | 11 | 0 | 0 |

## What it shows
- The structural rules are value-preserving as designed: 247 of 247 children correct. They are the beam's safe moves; 17 of them beat the starting best.
- Generic rewrites break correct kernels more often (41 % correct) but found the final best.
- The climb stops after about 5 generations: the reachable rule sequences are used up. More rules (symmetric-half needs a kingdom change) or mined rewrites are the lever.

## Pending
- Paired re-timing of the best against generation 0's best and shipped (6 interleaved rounds), queued after run 3: `logs/HGP/001/fit_beam_R24/retime_paired/`.

## Paired re-timing (done 03:09Z)
6 interleaved rounds of the run's best (g005), generation 0's best and shipped, at the fit's shapes (`logs/HGP/001/fit_beam_R24/retime_paired/table.md`):

| kernel | correct | median ms (3 shapes) | median speed-up over shipped (paired) |
|---|---|---|---|
| run's best | 6 of 6 | 119.0 / 173.5 / 208.5 | 9.16 |
| generation 0's best | 6 of 6 | 152.0 / 242.0 / 232.0 | 7.36 |
| shipped | 6 of 6 | 1577 / 1461 / 1468 | 1.00 |

**The run's best is 1.25× faster than generation 0's best (median; range 1.22–1.47), faster in 6 of 6 rounds.** The machine was loaded during this timing (shipped at about 1,500 ms against its usual 1,005), so the absolute times are high; the paired ratios are the measure.

Lineage (`lineage.txt`): seed r8 → drop-zero-fill → grow-rows 4→8 → drop-dead-store → unroll-k → swap-operands. Single-timing fitness swings along it (0.1128 → 0.1694 across drop-zero-fill, which changes nothing computed) show the per-kernel fitness is noisy; only paired timings are trusted.
