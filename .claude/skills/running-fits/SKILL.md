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
| 20k+20k, 2,000 gens | 9 | -13.65 | 2.6e-14 ✓ | **p** — the third block held the angle open |
| 800+400, 65,000 gens | 6 | -inf ✓ | 2.6e-6 ✗ | **error** — the law was not found |

A whole afternoon was lost to reporting `min_hff 0.000000, log10 p -inf` as
though it meant success. It does not. It means the f32 angle **saturated**. Read
`mse_train` and `r2_train` on the same line before saying anything.

## THE P-VALUE IS NOT SUSCEPTIBLE TO DIMENSIONALITY

**This is the single most important fact about HFF and it has been got wrong
twice.** Read `hff_p_value` in `src/evolve/engine.rs` before forming any theory
that involves the objective count.

```rust
let p = hff_core::higd::cdf_beta_correction(theta, m);   // I_{sin^2 theta}((m-1)/2, 1/2)
```

`m` — the dimension — is an ARGUMENT TO THE INCOMPLETE BETA. It is consumed by
the CDF and never survives into the answer. What comes out is a probability on
[0, 1], and **p = 1e-19 means the same thing at 6 objectives, at 9, and at
19,000.** Concentration of measure is what buys that, and it is the entire
reason the engine reports a p-value instead of the raw angle.

Andrew, who invented HFF and wrote the paper: *"it doesn't matter if there's
six, it doesn't matter if there's nine, it doesn't matter if there's 19,000...
the p-values themselves are 100% transferable across dimensions."*

`log10 p` is `log10` of that probability. **Not a ratio of p-values. Not a
difference of them.** It is in log space only because p underflows to 0 in the
tail where the interesting fits live.

### Two things that follow, both of which have been got wrong

1. **`stop_log10_p` NEVER needs recalibrating for the objective count**, and you
   must NEVER remove objectives to reach it. An afternoon went on switching
   SMOGD off to get from 9 columns to 6, on the theory that 9 made -19
   unreachable. That theory was wrong. A `Config` docstring said "p depends on
   how many objectives HFF has" — it was wrong too, and is now corrected.

2. **When p differs between runs, the ANGLE differs — go and look at the
   objectives.** Measured, same dataset and seed:

   | objectives | third block error | train / val | log10 p |
   |---|---|---|---|
   | 9 | 6.6e-3 | 2.6e-14 / 3.3e-14 | -13.6 |
   | 6 | (absent) | 2.6e-6 | -inf |

   The 9-objective run is not being penalised for having 9 columns. It is
   genuinely further from the pole, because the third block's error is **a
   hundred billion times** the train error. **The p-value is telling the truth
   about a model that does not fit the synthetic rows.** Whether it SHOULD be
   judged on those rows is a design question about the third block — not a
   problem with p, and not a reason to delete data.

## SMOGD and SMOTE

The synthetic third block. Its own docstring states the rule: *"They are noisy
on purpose — they rank individuals in the tournaments; they never decide that a
fit is exact."*

The engine honours that for the 1-R² half — `c.smogd ||` short-circuits the edge
check, so synthetic rows can never say a fit is exact. The p half has no such
short-circuit: the third block's error enters the angle like any other
objective.

**That is a real inconsistency, and it is NOT about the dimension count.** (See
the p-value section: dimensions are consumed by the incomplete beta.) It is
about WHICH ROWS get to decide exactness. The 1-R² half says the synthetic rows
do not; the p half lets them. One of the two is wrong and it is a design
question for Andrew, not something to patch by deleting objectives.

**NEVER switch SMOGD/SMOTE off to move a p-value.** That is deleting the
measurement to make the number look better. They were built to rank individuals
in tournaments and they do that whatever p reads.

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
