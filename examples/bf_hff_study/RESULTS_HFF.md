# BF HFF Validation Study — Results

Validation-in-fitness via HFF (TrueNorth) for Brainfuck GP.

## Setup

- Population: 40, generations: 50, seeds: 30
- Tournament K = 3 (~7% of pop)
- Max program length: 36
- Mutation rate: 0.35
- TRAIN=10, VAL=10, EXTRAP=10 inputs per task

Arms:
- **NONE**: err_train (lower=better), single-objective baseline
- **HFF_TRAIN**: k=1 TrueNorth([err_train])
- **HFF_VAL**: k=2 TrueNorth([err_train, err_val])
- **HFF_EXTRAP**: k=3 TrueNorth([err_train, err_val, err_extrap])

## Input Split Scheme

Deterministic per-task seeds. All splits disjoint.

| Task | Train N | Val N | Extrap N | Extrap Regime |
|------|---------|-------|----------|--------------|
| increment | 10 | 10 | 10 | Near-wrap regime [200..255]. All 256 inputs are semantically uniform for increment; extrap is a held-out edge-value chunk, not truly OOD. |
| echo | 10 | 10 | 10 | Held-out high-end bytes [200..255]. Echo is semantically uniform; extrap is a held-out split, not OOD. |
| add_three | 10 | 10 | 10 | Near-wrap regime [200..255]; values where add-3 crosses the u8 wrap boundary. |
| add_two | 10 | 10 | 10 | Large-magnitude pairs [128..255]×[128..255]; sums cross the u8 wrap boundary. Genuinely OOD relative to train domain [0..127]². |

## Memoriser Attack Test (increment task)

A hand-constructed BF program (`build_memoriser_increment`) that outputs the
correct answer for every TRAIN input and produces no output (wrong) for any other input.
This is the gold-standard test for whether HFF_VAL catches input-set overfit.

```
Memoriser: train_acc=1.000  val_acc=0.000  HFF_VAL_score=1.047198
Ground truth `,+.`: HFF_VAL_score=0.000000
NONE:    NONE is fooled: memoriser err_train=0.0 wins single-objective selection
HFF_VAL: HFF_VAL DEFEATS memoriser: ground truth has strictly lower HFF angle (PASS)

```

**Overall verdict: PASS — HFF_VAL defeats the memoriser**

- Memoriser: train_acc=1.000, val_acc=0.000, HFF_VAL_score=1.047198
- Ground truth: HFF_VAL_score=0.000000

## Phase 1: Main Study Results

Metrics per arm, mean±std over 30 seeds.

> **Caveat:** these are NOT paired. The seed encodes the arm index, so arms
> share no common randomness — the Wilcoxon tests are run unpaired-in-effect,
> and the EGGLOG arm receives extra fitness evaluations per generation without
> budget equalization. Treat the significance results as indicative only until
> re-run with shared per-seed randomness and equalized eval budget.

- **train_solve_rate**: fraction of final population with train_acc=1.0
- **oracle_solve_rate**: fraction with train AND val AND extrap accuracy = 1.0
- **mean_val_acc**: mean validation accuracy in final population
- **mean_drift**: mean (train_acc - val_acc); higher = more overfit

### Task: increment

| Arm | TrainSolve Mean±Std | OracleSolve Mean±Std | ValAcc Mean±Std | Drift Mean±Std |
|-----|--------------------|-----------------------|----------------|---------------|
| NONE | 0.517±0.414 | 0.517±0.414 | 0.518±0.414 | -0.000±0.001 |
| HFF_TRAIN | 0.592±0.427 | 0.592±0.427 | 0.593±0.427 | -0.000±0.001 |
| HFF_VAL | 0.730±0.338 | 0.730±0.338 | 0.730±0.338 | -0.000±0.001 |
| HFF_EXTRAP | 0.628±0.384 | 0.628±0.384 | 0.638±0.369 | -0.010±0.026 |

#### Wilcoxon signed-rank tests (30 seeds per arm, two-sided) — see caveat

| Comparison | W | p | Median Δ | p<0.05? |
|------------|---|---|---------|--------|
| HFF_VAL vs HFF_TRAIN (train_solve)  | 117.0 | 0.2209 | +0.025 | NO |
| HFF_VAL vs NONE (train_solve)       | 61.0 | 0.0192 | +0.050 | YES |
| HFF_EXTRAP vs HFF_VAL (train_solve) | 143.0 | 0.1718 | -0.050 | NO |
| HFF_VAL vs NONE (val_acc)           | 61.0 | 0.0192 | +0.050 | YES |
| HFF_EXTRAP vs HFF_VAL (val_acc)     | 143.5 | 0.1754 | -0.050 | NO |

### Task: echo

| Arm | TrainSolve Mean±Std | OracleSolve Mean±Std | ValAcc Mean±Std | Drift Mean±Std |
|-----|--------------------|-----------------------|----------------|---------------|
| NONE | 0.858±0.177 | 0.858±0.177 | 0.858±0.177 | 0.000±0.000 |
| HFF_TRAIN | 0.896±0.061 | 0.896±0.061 | 0.896±0.061 | 0.000±0.000 |
| HFF_VAL | 0.872±0.098 | 0.872±0.098 | 0.872±0.098 | 0.000±0.000 |
| HFF_EXTRAP | 0.846±0.171 | 0.846±0.171 | 0.846±0.171 | 0.000±0.000 |

#### Wilcoxon signed-rank tests (30 seeds per arm, two-sided) — see caveat

| Comparison | W | p | Median Δ | p<0.05? |
|------------|---|---|---------|--------|
| HFF_VAL vs HFF_TRAIN (train_solve)  | 162.0 | 0.5165 | +0.000 | NO |
| HFF_VAL vs NONE (train_solve)       | 168.5 | 0.8589 | +0.000 | NO |
| HFF_EXTRAP vs HFF_VAL (train_solve) | 159.0 | 0.4711 | -0.025 | NO |
| HFF_VAL vs NONE (val_acc)           | 168.5 | 0.8589 | +0.000 | NO |
| HFF_EXTRAP vs HFF_VAL (val_acc)     | 159.0 | 0.4711 | -0.025 | NO |

### Task: add_three

| Arm | TrainSolve Mean±Std | OracleSolve Mean±Std | ValAcc Mean±Std | Drift Mean±Std |
|-----|--------------------|-----------------------|----------------|---------------|
| NONE | 0.033±0.160 | 0.033±0.160 | 0.033±0.160 | 0.000±0.000 |
| HFF_TRAIN | 0.051±0.195 | 0.051±0.195 | 0.051±0.195 | 0.000±0.000 |
| HFF_VAL | 0.111±0.289 | 0.111±0.289 | 0.111±0.289 | 0.000±0.000 |
| HFF_EXTRAP | 0.109±0.288 | 0.109±0.288 | 0.109±0.288 | 0.000±0.000 |

#### Wilcoxon signed-rank tests (30 seeds per arm, two-sided) — see caveat

| Comparison | W | p | Median Δ | p<0.05? |
|------------|---|---|---------|--------|
| HFF_VAL vs HFF_TRAIN (train_solve)  | 5.5 | 0.2945 | +0.000 | NO |
| HFF_VAL vs NONE (train_solve)       | 6.0 | 0.3454 | +0.000 | NO |
| HFF_EXTRAP vs HFF_VAL (train_solve) | 10.0 | 0.9165 | +0.000 | NO |
| HFF_VAL vs NONE (val_acc)           | 6.0 | 0.3454 | +0.000 | NO |
| HFF_EXTRAP vs HFF_VAL (val_acc)     | 10.0 | 0.9165 | +0.000 | NO |

### Task: add_two

| Arm | TrainSolve Mean±Std | OracleSolve Mean±Std | ValAcc Mean±Std | Drift Mean±Std |
|-----|--------------------|-----------------------|----------------|---------------|
| NONE | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.090±0.007 |
| HFF_TRAIN | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.084±0.024 |
| HFF_VAL | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.085±0.024 |
| HFF_EXTRAP | 0.000±0.000 | 0.000±0.000 | 0.005±0.028 | 0.073±0.049 |

#### Wilcoxon signed-rank tests (30 seeds per arm, two-sided) — see caveat

| Comparison | W | p | Median Δ | p<0.05? |
|------------|---|---|---------|--------|
| HFF_VAL vs HFF_TRAIN (train_solve)  | 0.0 | 1.0000 | +0.000 | NO |
| HFF_VAL vs NONE (train_solve)       | 0.0 | 1.0000 | +0.000 | NO |
| HFF_EXTRAP vs HFF_VAL (train_solve) | 0.0 | 1.0000 | +0.000 | NO |
| HFF_VAL vs NONE (val_acc)           | 0.0 | 1.0000 | +0.000 | NO |
| HFF_EXTRAP vs HFF_VAL (val_acc)     | -0.0 | 0.3173 | +0.000 | NO |

## Extrap Accuracy Summary

Mean extrap_acc in final population (higher = better generalisation).

| Task | NONE | HFF_TRAIN | HFF_VAL | HFF_EXTRAP |
|------|------|-----------|---------|------------|
| increment | 0.518±0.414 | 0.593±0.427 | 0.730±0.338 | 0.631±0.379 |
| echo | 0.858±0.177 | 0.896±0.061 | 0.872±0.098 | 0.846±0.171 |
| add_three | 0.033±0.160 | 0.051±0.196 | 0.111±0.289 | 0.160±0.270 |
| add_two | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.002±0.013 |

## Reproduce

```bash
git checkout feat/bf-hff-validation
RUSTFLAGS="-D warnings" cargo test --no-default-features
cargo run --release --example bf_hff_study --no-default-features
```

