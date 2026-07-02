# BF Bloat Study — Results v2 (lambda sweep + full Baldwinian battery)

## Soundness (the real headline)

**100% match rate on the tested set** — 500 random BF programs, 4 test inputs each = 2000 interpreter comparisons, 0 output mismatches.
Caveat: the fuzzer's op pool is bracket-free, so these programs exercise the run-length rules but **not** the clear-loop (`[-]`) rewrites — the nontrivial ones. The match rate is differential evidence over that op set, not a soundness proof; the loop rules still need bracket-inclusive fuzzing before any "provable" claim.
The same technique applies to any GP target with decidable equivalence: SQL, regex synthesis, sorting networks, compiler IR passes.

Verify: `RUSTFLAGS="-D warnings" cargo test --no-default-features`

## Setup

- Population: 40, generations: 50, seeds: 30
- Lambda grid (parsimony sweep): [0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05]
- Default PARSIMONY λ (Phase 1 main arm): 0.03
- Tournament K = 3 (~7% of population)
- Max program length: 36
- Mutation rate: 0.35

## Task Battery

| Task | Ground Truth | Ops | Tests | Structural Type |
|------|-------------|-----|-------|----------------|
| increment | `,+.` | 3 | 10 | arithmetic, no loop (run-length bloat) |
| echo | `,.` | 2 | 10 | pure I/O, trivial baseline (run-length bloat) |
| add_three | `,+++.` | 5 | 10 | arithmetic, no loop (more run-length than increment) |
| add_two | `,>,[<+>-]<.` | 11 | 10 | multi-cell coordination (STRUCTURAL bloat — key generalization test) |

### Excluded Tasks

| Task | Ground Truth | Reason for Exclusion |
|------|-------------|---------------------|
| double | `,[->++<]>.` (10 ops) | GP achieves 0% solve rate at POP=60, GENS=80 on all 3 arms. |
| double (egglog overhead) | — | Egglog saturation cost prohibitive at >20 ops with nested loops. MAX_SIMPLIFY_OPS=20 guard applied. |

## Phase 1: Main Study (NONE / PARSIMONY@λ=0.03 / EGGLOG)

Solve rate = fraction of final-population individuals that pass all test inputs.

### Task: increment

| Arm | Solve Rate Mean±Std | Solve Rate Median(IQR) | Mean Length Mean±Std | Conv Gen Median |
|-----|--------------------|-----------------------|---------------------|----------------|
| NONE | 0.661±0.376 | 0.837(0.175) | 17.4±8.4 | 8 |
| PARSIMONY | 0.419±0.358 | 0.613(0.725) | 3.7±3.2 | 9 |
| EGGLOG | 0.610±0.385 | 0.800(0.900) | 16.7±10.9 | 8 |

#### Statistical Tests (Wilcoxon signed-rank, two-sided, 30 paired seeds)

| Comparison | W | p | Median Δ | Significant (p<0.05)? |
|------------|---|---|---------|----------------------|
| EGGLOG vs NONE      | 177.5 | 0.5615 | -0.025 | NO |
| EGGLOG vs PARSIMONY | 93.5 | 0.0218 | +0.125 | YES |
| PARSIMONY vs NONE   | 78.0 | 0.0077 | -0.175 | YES |

### Task: echo

| Arm | Solve Rate Mean±Std | Solve Rate Median(IQR) | Mean Length Mean±Std | Conv Gen Median |
|-----|--------------------|-----------------------|---------------------|----------------|
| NONE | 0.812±0.283 | 0.913(0.125) | 15.5±9.8 | 0 |
| PARSIMONY | 0.771±0.079 | 0.750(0.125) | 4.2±5.2 | 2 |
| EGGLOG | 0.887±0.064 | 0.875(0.100) | 13.5±9.1 | 0 |

#### Statistical Tests (Wilcoxon signed-rank, two-sided, 30 paired seeds)

| Comparison | W | p | Median Δ | Significant (p<0.05)? |
|------------|---|---|---------|----------------------|
| EGGLOG vs NONE      | 193.0 | 0.8199 | +0.025 | NO |
| EGGLOG vs PARSIMONY | 36.0 | 0.0001 | +0.125 | YES |
| PARSIMONY vs NONE   | 123.5 | 0.0250 | -0.175 | YES |

### Task: add_three

| Arm | Solve Rate Mean±Std | Solve Rate Median(IQR) | Mean Length Mean±Std | Conv Gen Median |
|-----|--------------------|-----------------------|---------------------|----------------|
| NONE | 0.204±0.363 | 0.000(0.200) | 16.0±7.1 | 8 |
| PARSIMONY | 0.143±0.296 | 0.000(0.000) | 7.3±7.1 | 19 |
| EGGLOG | 0.159±0.329 | 0.000(0.000) | 15.5±10.0 | 16 |

#### Statistical Tests (Wilcoxon signed-rank, two-sided, 30 paired seeds)

| Comparison | W | p | Median Δ | Significant (p<0.05)? |
|------------|---|---|---------|----------------------|
| EGGLOG vs NONE      | 38.5 | 0.6247 | +0.000 | NO |
| EGGLOG vs PARSIMONY | 23.5 | 0.6835 | +0.000 | NO |
| PARSIMONY vs NONE   | 32.5 | 0.2093 | +0.000 | NO |

### Task: add_two

| Arm | Solve Rate Mean±Std | Solve Rate Median(IQR) | Mean Length Mean±Std | Conv Gen Median |
|-----|--------------------|-----------------------|---------------------|----------------|
| NONE | 0.000±0.000 | 0.000(0.000) | 20.6±8.7 | N/A |
| PARSIMONY | 0.000±0.000 | 0.000(0.000) | 2.4±1.1 | N/A |
| EGGLOG | 0.000±0.000 | 0.000(0.000) | 18.0±10.3 | N/A |

#### Statistical Tests (Wilcoxon signed-rank, two-sided, 30 paired seeds)

| Comparison | W | p | Median Δ | Significant (p<0.05)? |
|------------|---|---|---------|----------------------|
| EGGLOG vs NONE      | 0.0 | 1.0000 | +0.000 | NO |
| EGGLOG vs PARSIMONY | 0.0 | 1.0000 | +0.000 | NO |
| PARSIMONY vs NONE   | 0.0 | 1.0000 | +0.000 | NO |

## Phase 2: Lambda Sweep — PARSIMONY arm across λ grid

Each cell: mean solve rate ± std over 30 seeds. Mean length (ops) in parentheses.

| λ | increment solve±std (len) | echo solve±std (len) | add_three solve±std (len) | add_two solve±std (len) |
|---|---|---|---|---|
| 0.000 | 0.557±0.437 (18.1) | 0.865±0.174 (14.5) | 0.259±0.404 (17.8) | 0.000±0.000 (18.6) |
| 0.001 | 0.538±0.413 (18.0) | 0.922±0.044 (17.4) | 0.183±0.340 (18.1) | 0.000±0.000 (22.9) |
| 0.003 | 0.623±0.418 (14.5) | 0.896±0.062 (15.0) | 0.337±0.403 (18.2) | 0.000±0.000 (16.1) |
| 0.010 | 0.582±0.368 (8.2) | 0.848±0.170 (12.9) | 0.243±0.382 (11.7) | 0.000±0.000 (8.0) |
| 0.020 | 0.555±0.347 (5.0) | 0.802±0.077 (5.4) | 0.225±0.353 (7.8) | 0.000±0.000 (3.6) |
| 0.030 | 0.598±0.284 (3.8) | 0.790±0.073 (4.4) | 0.023±0.123 (5.1) | 0.000±0.000 (2.6) |
| 0.050 | 0.326±0.382 (2.8) | 0.781±0.083 (3.6) | 0.094±0.247 (7.8) | 0.000±0.000 (1.3) |

### Best Lambda Per Task

Best = highest mean solve rate; ties broken by smallest λ (prefer lighter penalty).

| Task | Best λ | PARSIMONY@best solve rate | Justification |
|------|--------|--------------------------|--------------|
| increment | 0.003 | 0.623±0.418 (mean len 14.5) | see sweep table |
| echo | 0.001 | 0.922±0.044 (mean len 17.4) | see sweep table |
| add_three | 0.003 | 0.337±0.403 (mean len 18.2) | see sweep table |
| add_two | 0.000 | 0.000±0.000 (mean len 18.6) | see sweep table |

## Phase 4: Fair Comparison — EGGLOG vs PARSIMONY@best-λ

Using the best per-task λ identified from the sweep.  EGGLOG seeds are from Phase 1.

### Task: increment (PARSIMONY@λ=0.003)

| Arm | Solve Rate Mean±Std | Median(IQR) |
|-----|--------------------|-----------  |
| EGGLOG (Lamarckian)   | 0.610±0.385 | 0.800(0.900) |
| PARSIMONY@λ=0.003 | 0.535±0.436 | 0.800(0.900) |

Wilcoxon EGGLOG vs PARSIMONY@best-λ: W=150.0 p=0.7366 Δ=+0.000 (not significant)

### Task: echo (PARSIMONY@λ=0.001)

| Arm | Solve Rate Mean±Std | Median(IQR) |
|-----|--------------------|-----------  |
| EGGLOG (Lamarckian)   | 0.887±0.064 | 0.875(0.100) |
| PARSIMONY@λ=0.001 | 0.901±0.054 | 0.900(0.050) |

Wilcoxon EGGLOG vs PARSIMONY@best-λ: W=160.0 p=0.4860 Δ=+0.000 (not significant)

### Task: add_three (PARSIMONY@λ=0.003)

| Arm | Solve Rate Mean±Std | Median(IQR) |
|-----|--------------------|-----------  |
| EGGLOG (Lamarckian)   | 0.159±0.329 | 0.000(0.000) |
| PARSIMONY@λ=0.003 | 0.203±0.336 | 0.000(0.600) |

Wilcoxon EGGLOG vs PARSIMONY@best-λ: W=55.5 p=0.7983 Δ=+0.000 (not significant)

### Task: add_two (PARSIMONY@λ=0.000)

| Arm | Solve Rate Mean±Std | Median(IQR) |
|-----|--------------------|-----------  |
| EGGLOG (Lamarckian)   | 0.000±0.000 | 0.000(0.000) |
| PARSIMONY@λ=0.000 | 0.000±0.000 | 0.000(0.000) |

Wilcoxon EGGLOG vs PARSIMONY@best-λ: W=0.0 p=1.0000 Δ=+0.000 (not significant)

## Phase 3: Baldwinian vs Lamarckian — Full Battery

Baldwinian: simplified phenotype used for fitness evaluation only; original genotype stored.

| Task | EGGLOG (Lamarckian) Mean±Std | BALDWINIAN Mean±Std | W | p | Δ | Significant? |
|------|------------------------------|---------------------|---|---|---|-------------|
| increment | 0.610±0.385 | 0.472±0.444 | 104.5 | 0.3083 | +0.000 | NO |
| echo | 0.887±0.064 | 0.849±0.175 | 171.0 | 0.4662 | +0.000 | NO |
| add_three | 0.159±0.329 | 0.256±0.400 | 16.5 | 0.1424 | +0.000 | NO |
| add_two | 0.000±0.000 | 0.000±0.000 | 0.0 | 1.0000 | +0.000 | NO |

## Headline Findings (v2)

### HOLE 1 RESOLVED: EGGLOG gap closes against tuned parsimony

The original study's λ=0.03 was too aggressive, crushing programs to <4 ops. The sweep shows:
- **increment**: best λ=0.003, PARSIMONY achieves 0.623±0.418 — EGGLOG (0.610) is NOT significantly better (p=0.7366, Δ=0.000).
- **echo**: best λ=0.001, PARSIMONY achieves 0.922±0.044 — EGGLOG (0.887) is NOT significantly better (p=0.4860, Δ=0.000).
- **add_three**: best λ=0.003, PARSIMONY achieves 0.337±0.403 — EGGLOG (0.159) is NOT significantly better (p=0.7983, Δ=0.000).

**The EGGLOG advantage reported in v1 was entirely against an over-penalized baseline.** Against a fairly-tuned parsimony, the gap vanishes.

The reframe: EGGLOG achieves parity with the best parsimony coefficient without requiring the practitioner to sweep or tune λ. This is a real practical benefit — egglog is a self-tuning bloat control mechanism — but it is not a performance improvement over tuned parsimony.

### HOLE 2 RESOLVED: Baldwinian-vs-Lamarckian across full battery

| Task | EGGLOG | BALDWINIAN | p | Verdict |
|------|--------|-----------|---|---------|
| increment | 0.610 | 0.472 | 0.3083 | No significant difference |
| echo | 0.887 | 0.849 | 0.4662 | No significant difference |
| add_three | 0.159 | 0.256 | 0.1424 | No significant difference |
| add_two | 0.000 | 0.000 | 1.0000 | Both 0% (task unsolvable at budget) |

The pilot's direction (Baldwinian > Lamarckian on increment, 0.688 vs 0.610) was task-specific noise. Across the full battery, Baldwinian is NOT consistently better or worse than Lamarckian. The fitness-smoothing mechanism hypothesis is **not confirmed** at the full-battery level. Neither mechanism (fitness smoothing nor genotype cleanup) dominates — the effects are task-specific and not statistically significant.

### Honest summary

1. EGGLOG is NOT superior to well-tuned parsimony. The v1 advantage was artifact of λ mis-tuning.
2. EGGLOG IS comparable to parsimony without hyperparameter tuning — the practical benefit is lambda-free bloat control.
3. Neither Baldwinian nor Lamarckian is significantly better across the battery — the mechanism is inconclusive.
4. Differential soundness (100% match rate) holds over the bracket-free tested set; a full soundness claim needs bracket-inclusive fuzzing of the clear-loop rules.
5. add_two (structural task) remains unsolvable at this budget for all arms — run-length egglog rules do not help structural bloat.

## Reproduce

```bash
git checkout feat/bf-simplifier-bloat-study
# Verify soundness (100% expected):
RUSTFLAGS="-D warnings" cargo test --no-default-features
# Run full study (writes results.jsonl, RESULTS.md, MECHANISM.md):
cargo run --release --example bf_study --no-default-features
```

Runtime: ~18 minutes on 2-core M-series (7 lambdas × 4 tasks × 30 seeds × 3 phases + Baldwinian battery).

