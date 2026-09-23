# tsr-4k — TSR at 4,000 + 4,000 for 40,000 generations

## What was asked

> "With the exact same setup you had that got that answer, I want you now to
> expand 800 to being 4,000. And I want 400 to also be 4,000. And I want
> generations to be 40,000."

The previous TSR run was 800 + 400 for 14,167 generations — "basically a toy".
This is the same configuration at 5x the intake, 10x the champion island, and
2.8x the generations.

## The exact command

```bash
EVOLVE_TYPED_DEPTH=2 \
EVOLVE_POP_INTAKE=4000 EVOLVE_POP_CHAMPION=4000 \
EVOLVE_MAX_GENERATIONS=40000 EVOLVE_SECONDS=86400 \
EVOLVE_PUMP_EVERY=100 EVOLVE_COHORT_MERGE=10000 \
EVOLVE_GENE_SUBSETS=1 EVOLVE_SEED=7014 \
EVOLVE_TELEMETRY_FILE=$PWD/logs/tsr_4k/stream.jsonl \
EVOLVE_TELEMETRY_RUN_ID=tsr_4k \
EVOLVE_CHECKPOINT_DIR=$PWD/logs/tsr_4k/checkpoint EVOLVE_CHECKPOINT_EVERY=60 \
EVOLVE_PROGRESS_EVERY=500 \
nohup nice -n 5 ./target/release/examples/evolve_fit <bacres1.tsv> \
  > logs/tsr_4k/run.log 2>&1 &
```

Dataset: `strogatz_bacres1`. True law: `-x*y/(0.5*x**2 + 1) - x + 20`.

**6 HFF objectives** (no SMOGD/SMOTE), confirmed by the card at launch.

## The run it is scaling up from

`logs/typed_on_7014`, TSR at 800 + 400, 14,167 generations in 420 s:

| | |
|---|---|
| train 1-R² | 4.441e-6 |
| val 1-R² | 1.982e-7 |
| t_depth of the winner | 2 — at the TSR ceiling |
| law recovered | no |

Its equation, 180 characters:

```
-2.0082 * ( x_1/(x_0 + sin(9.43/x_1)) + sqrt(1/x_0)
          + x_1/(sqrt(93/(45*x_0)) + x_0) + x_0 ) / 2 + 20.2837
```

against `-x*y/(0.5x² + 1) - x + 20`. The `+20.28`, the -1.004x leading
coefficient, the quotient shape and the bare `+x_0` are all present. No ladder:
every function sits at the top of its own term rather than nested inside
another, which is what the typing is for. Untyped runs on this dataset produced
213-412 character `tanh(exp(cos(log(...))))` chains.

That is a near-miss on STRUCTURE, which is a different failure mode from the
untyped runs, and the reason for scaling it up.

## Status

Running. Results and the card's notes land here.
