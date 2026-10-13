# Runs 5 and 6: random start with every sampler fix (fit_SF_r5, fit_SF_r6)

**Result: no correct kernel in 180 generations. The best closeness fell from 1.2391 to 1.1303 — most of it in the first 5 generations — while the correlation with ΦᵀΦ never left zero. Run 5 found a bug in our software (integer constants printed as denormal floats), fixed before run 6; run 6 stopped at generation 181 on a timing error under investigation.**

## Configuration (both)
- Fuller 1b42c63: every random kernel stores a value read from Φ, every control condition reads a per-thread or data value, no self-comparison, none over the cost bound; random genes encoded straight from the AST (no parse).
- Seed 5, 80 individuals, the same operator set as run 4 (rewrite pool 0.5, point mutation 33, typed crossover 8, Karva crossover, dedup, pump every 20 with fresh random kernels). No models, no swap gate, empty library.
- Run 5: phylu b6835955 (with the fold-failure evidence dump). Run 6: phylu 5e5bc4e1 (the constant fix).

## Generation 0
68 of 80 random kernels scored by closeness (37 in run 4); 12 write nothing; 0 over the cost bound (33 in run 4).

## Run 5's stop: a bug in our software
At generation 22 the fold turned `subgroupAll(lit_bool(true))` into `subgroupAll(lit_bool(false))` (`logs/HGP/001/fit_SF_r5/foldbug_g022_r63/`). Cause: the gene text printed a bare integer constant (a flag, an index, a count) as the float its bits make — true as 1.4e-45, index 5 as 7e-45; the kernel read the tiny number as non-zero, the fold re-encoded it as 0. Every earlier run with point mutation evaluated such constants with wrong values; the "kernel text differs" refusals of 2026-10-11/12 were very likely this. Fixed in phylu 5e5bc4e1 (the constant read under its class); CPU stress 4 flips of 2,143 before, 0 after.

## Run 6: best closeness (0 = correct)
| gen | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 |
|---|---|---|---|---|---|---|---|---|---|---|
| best | 1.2391 | 1.1527 | 1.1460 | 1.1431 | 1.1424 | 1.1347 | 1.1343 | 1.1329 | 1.1316 | 1.1303 |

Generations 0–22 were identical to run 5 (same seed; the fix changed no kernel selection needed until then).

The best kernel's columns at generation 77 (0 = the reference): cases 1.000, correlation 0.550 (ρ ≈ −0.1), shift 0.975, transpose 0.976, sorted 0.660, sums 0.970, cells written 0.219, within tolerance 0.784, input sensitivity 0.130. The search learns to write most cells with values read from Φ; it never finds the arithmetic — correlation with ΦᵀΦ stays at zero across all random runs.

## Run 6's stop
`Error: "the shipped kernel timed at zero at shape 0"` at generation 181, 07:28Z, after a generation whose fitness split showed 20 s of "other" time. Under investigation (mutation-builder) before any further GPU run.

## What it shows
- The sampler fixes doubled the usable random kernels and gave the steepest early climb, but the climb is in the easy columns (cells written, input sensitivity, value range).
- Random genes plus single-edit operators do not reach the structure of the computation in 180 generations. The next experiment is a sketch start: the outline the problem fixes (cell ownership, the item loop, the store) with random holes, or a fresh Sonnet start.
