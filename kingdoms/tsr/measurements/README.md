# TSR measurements

What was measured in the `tsr` kingdom, with the run cards that produced it.
Every fit's own findings are in its card's `notes`; this file is the table
across them.

**There is no control arm here, on purpose.** Andrew:

> "We don't need an A test. We've been running A all week, and it's failed
> every time. What I need is a B test. Just do TSR."

Untyped numbers quoted below are **prior results from the record**
(`docs/EXPERIMENTS.md`, `logs/cards/`), cited for context and never re-run.

## What was live when these ran

Everything the spec asks for:

| part | state |
|---|---|
| `Ty::T1` / `T2` and the typed symbol table | live (`geneframe::typed_depth_table`) |
| typed sampler in `init` | live, GPU and CPU reference, bit-identical |
| typed sampler in point mutation | live, GPU and CPU reference, bit-identical |
| span-moving operators (inversion, IS, RIS, crossover) | **untyped on purpose** — left to the decode, which is the spec's own preference |
| decode-time refusal, counted apart from `oversized` | live (`FitResult::typed_refused`) |
| `write_back` refusing a graft that breaks the ceiling | live (`WriteBack::TypesDoNotClose`) |
| compounds | **off in these runs** — the compound depth cost is covered by unit test only |

Settings are the ones that recovered 75 of 133
(`.claude/skills/running-fits/SKILL.md`): 800 intake + 400 champion, pump every
100, `cohort_merge` 10,000, gene subsets on, no SMOGD/SMOTE (6 HFF objectives).
Development seeds only.

## Reading the refusal number

A refusal count has **two honest denominators and they answer different
questions**. Both are reported because quoting either alone misleads:

* **% of gene-slots per generation** = `refused / (generations × pop × n_genes)`
  — what the ceiling costs the search each beat. This is the number to compare
  with the spec's predicted 23%.
* **% of unique genes** — inflated by the "rides along" mechanism below, because
  a refused gene in an elite row is re-counted every generation it survives.

**A refused gene is not a dead row.** At `n_genes = 3` with `gene_subsets` on, a
row with one refused gene scores on its other two, so the illegal gene survives
unscored and keeps mutating. That is why a TSR population's depth histogram
still shows entries past the ceiling: **those entries are the refused genes.**

## Results

### feynman_I_15_10 — relativistic momentum, `m₀v/sqrt(1 − v²/c²)`, depth 1

**LAW RECOVERED IN ITS TRUE FORM**, seed 7014, at generation 1,001:

```
((x_0*x_1)/sqrt((1.0 - ((x_1/x_2)**2))))
```

which is exactly `m₀·v / sqrt(1 − (v/c)²)`. Not a model that scores well with
the wrong shape — the right shape.

| | |
|---|---|
| stopped by | `early_stop` (met the stop bar) |
| generations | 1,001 |
| ms/generation | 111.5 (75,000 rows) |
| train 1-R² | 1.158e-13 |
| validation 1-R² | 1.056e-13 |
| log10 p | −33.54 |
| test R² | 1.000000 |
| refused on type | 105,697 = **2.93% of gene-slots/generation** |
| refused as `oversized` | 0 |
| depth histogram | 0:2221 1:1107 **2:239** 3:19 4:10 5:3 6:1 |

**The T2 ceiling binds and is not vacuous**: 239 genes sit exactly at it. Only
0.9% of genes are past it, and those are refused genes riding along unscored.

### strogatz_bacres1, depth 0 — 420 s, seed 7014

Law not recovered, which is what 800+400 has always done on this problem
(prior result, `docs/EXPERIMENTS.md`: 65,264 generations at 800+400 reached
train 1-R² 2.6e-6 and did not recover it).

| | TSR (ceiling 2) | prior untyped, same seed and settings |
|---|---|---|
| generations in 420 s | 14,167 | 15,604 |
| ms/generation | 29.6 | 26.9 |
| train 1-R² | 4.441e-6 | 3.645e-6 |
| validation 1-R² | 1.982e-7 | 4.326e-7 |
| refused on type | 3.2% of gene-slots | — |
| genes past depth 2 | 78 (2.2%) | 178 (4.9%) |
| reported model's shape | **depth-2 legal**, `sin`/`sqrt` unnested | `cos(sqrt(acos(1/x_1)) + 59)`, a **depth-3 tower** |

The untyped column is from a run made before the instruction to drop the
control; it is reported here because it exists, not because it was needed.

## What the ceiling costs

On these runs the typing costs roughly **10% of the pace** (29.6 vs 26.9
ms/generation on bacres1) and **~3% of gene-slots per generation** to refusals.
Against the spec's estimate of "one `u32` per symbol, one comparison per draw",
the per-draw cost is as predicted; the refusal cost is the part the spec could
not predict, because it is a property of the search and not of the true-model
distribution.

## Limitations, stated

1. **Time binds, not generations.** Each fit gets a fixed wall clock, so the
   pace tax converts into fewer generations. For "does typing help the search"
   a same-generations comparison would be cleaner.
2. **Single development seed per problem** in the first pass. The record's own
   rule is that a change is kept only if it holds on two seeds.
3. **Compounds off**, so the compound depth cost is covered by unit test only.
4. **Meeting the stop bar is not the same as recovering the law.** Every claim
   of recovery above was checked by reading the FORM against the true law, as
   the skill file requires.
