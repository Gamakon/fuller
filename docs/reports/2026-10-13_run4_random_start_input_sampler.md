# Run 4: fresh random start with the input-reaching sampler (fit_SF_r4)

**Result: the best closeness fell from 1.2061 to 1.1392 in 23 generations — the steepest climb of any random run — but no kernel became correct; the run stopped at generation 24 on a bug in our software (the population fold rewrote a row into another kernel), caught by the new hard check.**

## Configuration
- Binary: fuller 557317f (stores read an input; conditions read a variable) + phylu b3608fae, `/tmp/claude-501/mb/target`. Note: that worktree was moved to ae3a484 during the run (the running process kept its own image; the kingdom tables are identical between the two), so the start line's 557317f is the build commit.
- Start: 80 random kernels, seed 4 (`--seed-from none`), every one storing a value read from Φ.
- Operators, all on: the rewrite pool (15 rules + 12 generic rewrites) on half the offspring; point mutation 33 a row; typed crossover 8 pairs; the engine's Karva crossover (one-point 0.3, two-point 0.2, gene 0.1 a pair); dedup; pump every 20 generations with fresh random kernels. No models, no swap gate, empty library.
- Log: `logs/HGP/001/fit_SF_r4_stopped_foldbug_g24/`.

## Best closeness angle (0 = correct)
| gen | evaluated | correct | best closeness |
|---|---|---|---|
| 0 | 48 | 0 | 1.2061 |
| 5 | 65 | 0 | 1.1964 |
| 10 | 76 | 0 | 1.1920 |
| 15 | 75 | 0 | 1.1902 |
| 20 | 74 | 0 | 1.1899 |
| 23 | 74 | 0 | 1.1392 |

For comparison: runs 1 (fit_SF_rw2) and fit_SF_pump, with the old sampler, ended at 1.2225 and 1.2247 after 50+ generations.

## The best kernel's columns (0 = matches the reference)
| column | generation 0 best | generation 24 best (r62, replace-leaf) |
|---|---|---|
| cases failed | 1.000 | 1.000 |
| correlation | 0.588 | 0.490 |
| shift | 0.975 | 0.982 |
| transpose | 0.976 | 0.984 |
| sorted values | 0.865 | 0.624 |
| row/column sums | 0.960 | 0.974 |
| cells written | 0.423 | 0.278 |
| cells within tolerance | 0.783 | 0.791 |
| input sensitivity | 0.285 | 0.000 |

The climb is real on three columns: the best now writes 72 % of the cells it should, every output changes with Φ, and its sorted values are much closer to the reference's. Correlation is still near zero (0.49 ↔ ρ ≈ 0.02): the values are in the right range but not in the right cells.

## Generation 0 of the random kernels
37 scored, 33 not dispatched (over the static cost bound), 11 writing nothing (about 5 under counter-only conditions, a sampler bug since fixed in ae3a484; the rest under conditions fixed in value).

## The stop
`our software has a bug: gen 24: the fold rewrote row 24 into another kernel` — the hard check added with the slot alignment (phylu 6113739e). Being fixed before run 5.
