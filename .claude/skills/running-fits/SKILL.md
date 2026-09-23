---
name: running-fits
description: How to configure, launch, watch and judge a symbolic-regression fit with the fuller engine. Read this BEFORE running evolve_fit or changing any search parameter — it holds the settings that have actually recovered laws and the traps that have cost whole afternoons.
---

# Running a fit

The objective is to recover SRBench's ground-truth equations exactly — 133 laws,
119 Feynman and 14 Strogatz. A fit that scores R² 1.000000 and does not recover
the equation is a **miss**, not a near-win.

## Read this first: the two halves of the stop bar

A fit stops early only when **all** of these hold on the confirmed f64 re-score:

```
train 1-R²  <= stop_one_minus_r2   (default 1e-10)
val   1-R²  <= stop_one_minus_r2
edge        <= stop_one_minus_r2   (skipped when smogd is on, or no third block)
log10 p     <= stop_log10_p        (default -19)
```

**When a fit does not stop, find out WHICH half failed before theorising.** Both
have failed in real runs for opposite reasons, and they look identical from the
outside:

| run | objectives | log10 p | train 1-R² | the failing half |
|---|---|---|---|---|
| 20k+20k, 2,000 gens | 9 | -13.65 | 2.6e-14 ✓ | **p** — too strict at 9 objectives |
| 800+400, 65,000 gens | 6 | -inf ✓ | 2.6e-6 ✗ | **error** — the law was not found |

A whole afternoon was lost to reporting `min_hff 0.000000, log10 p -inf` as
though it meant success. It does not. It means the f32 angle **saturated**. Read
`mse_train` and `r2_train` on the same line before saying anything.

## THE P-VALUE IS CALIBRATED FOR 4 OBJECTIVES, AND YOU ARE NOT RUNNING 4

`stop_log10_p = -19` was measured on development seed 7013 **with 4 objectives**
(train + t_depth). Its job was to separate one fake at -17.55 from real laws at
-19.4 and below. `Config::stop_log10_p`'s own docstring says "p depends on how
many objectives HFF has", and that sentence has been read and ignored twice.

`Engine::hff_dimensions()` is the count. It grows with:

- the validation block (+3 columns when present)
- the third block, SMOGD/SMOTE/edge (+3 columns)
- `redundancy` (+1), `tower` (+1)

Measured behaviour of p against objective count, same dataset and seed:

| objectives | third block | log10 p reached |
|---|---|---|
| 6 | none | **-inf** — saturated, discriminates nothing |
| 9 | SMOGD+SMOTE | **-13.6** — cannot reach -19 however good the fit |

So at 9 the bar is unreachable, and at 6 it is uninformative. **In both cases
1-R² is doing all the work.** The run that scored 75 of 133 recorded exactly
this in `docs/EXPERIMENTS.md`: *"The p-value is inert at this operating point.
Every log10 p sits between -30 and -38 against a -19 bar, and the solved and
unsolved distributions overlap completely."*

If you change the objective count, the p bar is no longer calibrated. Either
recalibrate it or switch that half off with `EVOLVE_STOP_LOG10_P=inf` and let
1-R² decide.

## SMOGD and SMOTE

The synthetic third block. Its own docstring states the rule: *"They are noisy
on purpose — they rank individuals in the tournaments; they never decide that a
fit is exact."*

The engine honours that for the 1-R² half (`c.smogd ||` short-circuits the edge
check) and **breaks it for the p half**, because `hff_dimensions()` counts the
third block's three columns anyway. That inconsistency is why a 9-objective run
cannot reach -19.

SMOGD/SMOTE are still worth having for what they were built for — ranking. Do
not switch them off to "fix" p; fix the dimension count, or the bar.

## The settings that have actually recovered laws

**Do not invent configurations.** Two are on record.

### 75 of 133 (best score to date, seed 7014, 360 s a fit)

```
pop_intake 800, pop_champion 400
pump_every 100
cohort_merge 10000
gene_subsets on
max_generations 50000   (so TIME binds, not generations)
no SMOGD/SMOTE          -> 6 objectives
```

### The bacres1 recovery (mass4k)

```
pop 2000 + 2000, pump_every 20, cohort_merge 100
no third block          -> 6 objectives
recovered at generation 19,060, log10 p -29.24
```

### What is NOT on record

Population above ~4,000 has never recovered anything. Measured:

| population | generations | train 1-R² | recovered |
|---|---|---|---|
| 800 + 400 | 65,264 | 2.6e-6 | no |
| 2,000 + 2,000 | 60,000 | 6.8e-9 | no |
| 10,000 + 10,000 | 20,000 | 6.2e-6 | no |
| 20,000 + 20,000 | 2,000 | 2.6e-14 | no (p failed) |
| 100,000 + 100,000 | 1,740 in 55 min | — | killed |

Big populations buy accuracy per generation and cost generations. 800+400 runs
**27 ms/generation**; 100k+100k runs **1,600 ms/generation** — 60× fewer beats
for the same wall clock. The champion island also converges to a single cohort
faster at scale (97% one cohort for 19,000 generations was measured at 2k).

## Writing and reading a run card

Every run writes a card holding the **whole** `Config` plus what `Config` does
not know: the dataset, SMOGD/SMOTE state, the git commit, and the derived
`hff_objectives` count.

```bash
EVOLVE_CARD=logs/cards/<tag>.json ./target/release/examples/evolve_fit
```

Precedence: **env > positional > card > default.** So a card is a starting
point you vary from, and the card that run writes records what actually ran.

`diff` two cards directly — they serialise in declaration order:

```
< "pop_intake": 800,      > "pop_intake": 2000,
< "hff_objectives": 6,    > "hff_objectives": 9,
```

That diff is the fact that cost an afternoon, and it was invisible in every
other output.

**Write your findings into the card's `notes` array when a run ends** — what you
were testing, what happened, and whether it recovered the law. A card with no
note is a configuration nobody can learn from.

## Watching a run

```bash
./target/release/hff-watch --follow logs/latest.jsonl
```

`logs/latest.jsonl` is a symlink repointed at each new run, so one path always
works. The launcher prints this command **before** the fit starts.

Banner: `SEARCHING` (blue) / `LAW FOUND` (green) / `LAW UNFOUND` (red). The
p-value is red above the bar, green at or below. `--dump` prints one frame as
text, which is how to report state without a terminal.

## Traps that have cost real time

1. **Never edit a script while it is running.** zsh reads scripts
   incrementally: an edit mid-run made the shell re-read at its old byte offset,
   land mid-word, and **run the fit a second time**. The second run's
   `run_start` truncated the telemetry the first had written — 678 records lost
   on a run that scored test R² 1.000000. `logs/bacres1_test.sh` is now one
   function so the file cannot do this, but the rule stands for every script.

2. **A model that fits is not a law.** `bacres1` has been reported as
   `-4096*y*tan(tan(0.000244*(x+y-2)))` at R² 1.0 — which IS `-y(x+y-2)`, until
   SRBench rounds 0.000244 to zero and the model becomes `0`. And as a 400-char
   blob whose 26-node subtree is the number 1 in a costume. Check the reported
   FORM, not only the score.

3. **A `run_start` truncates the stream.** Two fits writing the same
   `EVOLVE_TELEMETRY_FILE` means the second destroys the first's telemetry.
   Cards deliberately exclude output paths for this reason.

4. **Announce before launching.** Say what, why, how long, and the watch path —
   BEFORE the run starts, not after. A path given afterwards gets read while
   still watching the previous run.
