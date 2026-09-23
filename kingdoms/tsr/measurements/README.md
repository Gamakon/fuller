# TSR measurements

What was measured in the `tsr` kingdom, with the run cards that produced it.
Every fit's own findings are in its card's `notes`; this file is the table
across them.

**There is no control arm here, on purpose.** Andrew:

> "We don't need an A test. We've been running A all week, and it's failed
> every time. What I need is a B test. Just do TSR."

Untyped numbers quoted below are **prior results from the record**, cited for
context and never re-run: the race ledgers under
`hff/notebooks/sr_logs/*/race_ledger.json`, `docs/EXPERIMENTS.md`,
`.claude/skills/running-fits/SKILL.md` and `logs/cards/`.

The one exception is the bacres1 untyped row, which comes from a run made
earlier in this session before the instruction to drop the control arm. It is
reported because it exists, not because it was needed.

## The headline

**On strogatz_bacres1 — the problem the untyped engine has failed on all week —
TSR fails too.** Same settings, same seed, no recovery, and it costs about 10%
of the pace. That is a null on the outcome, and it is the first thing to say.

What DID change there is the population's shape: genes past depth 2 fell from
178 to 78, and the reported model is depth-2 legal where the prior untyped
run's was `cos(sqrt(acos(1/x_1)) + 59)`, a depth-3 tower.

**On two Feynman problems TSR recovered the law in its true form** —
`I.15.10` and `I.26.2`, both of which are unsolved across every untyped race on
the record. The budgets are not matched, so see the prior-results table before
reading that as a win.

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

## What the untyped engine did on these problems — PRIOR RESULTS, cited

From the race ledgers under `hff/notebooks/sr_logs/*/race_ledger.json`
(`logs/tsr/prior.py` reads them). **Not re-run for this work.**

| problem | prior untyped races | solved |
|---|---|---|
| feynman_I_15_10 | 7 | **0** |
| feynman_I_26_2 | 7 | **0** |
| feynman_I_48_2 | 7 | 0 |
| feynman_I_15_3t | 7 | 0 |
| feynman_II_11_27 | 7 | 0 |
| feynman_II_24_17 | 6 | 0 |
| feynman_III_4_32 | 7 | 0 |
| strogatz_bacres1 | 6 | 0 |

**Read this carefully, because it is easy to overclaim.** The two laws TSR
recovered had never been solved in any untyped race on the record. But those
races are not a matched control:

| | prior untyped | TSR here |
|---|---|---|
| I.15.10 | 22–81 generations, 31–129 s, best test R² 0.9996 | **1,001 generations**, recovered |
| I.26.2 | 28–115 generations, 30–124 s, best test R² 0.9987 | **169 generations**, recovered |

Seconds are comparable; **generations are not** — the prior races ran older,
slower configurations, so they saw a tenth to a fortieth of the beats. The
honest statement is therefore: **TSR recovered two laws that are unsolved
across the whole untyped record, at generation counts those races never
reached.** Whether the untyped engine at 1,000 generations would also find
them is not measured here, and the instruction for this work was not to run it.

## Results

### feynman_I_15_10 — relativistic momentum, `m₀v/sqrt(1 − v²/c²)`, depth 1

**LAW RECOVERED IN ITS TRUE FORM**, seed 7014, at generation 1,001:

```
((x_0*x_1)/sqrt((1.0 - ((x_1/x_2)**2))))
```

**Verified against SRBench's own ground truth**, not merely against the score.
`pmlb_repo/datasets/feynman_I_15_10/metadata.yaml` gives

```
p = m_0*v/sqrt(1-v**2/c**2)
```

and the dataset's columns are `m_0, v, c`, so `x_0 = m₀`, `x_1 = v`,
`x_2 = c` and the recovered model is `m₀·v / sqrt(1 − (v/c)²)` — the law,
symbol for symbol. Not a model that scores well with the wrong shape.

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

### feynman_I_26_2 — Snell's law, `arcsin(n·sin θ₂)`, **DEPTH 2**

**The case the T2 rung exists for.** This law is a transcendental applied to
something containing another transcendental. At a ceiling of T1 it would have
no legal signature — though that is a property of the type system, true before
any run, and not something this fit measures.

**LAW RECOVERED IN ITS TRUE FORM**, seed 7014, at generation **169**, in 18
seconds:

```
asin((x_0*sin(x_1)))
```

Ground truth from `pmlb_repo/datasets/feynman_I_26_2/metadata.yaml` is
`theta1 = arcsin(n*sin(theta2))`, and the columns are `n, theta2` — so this is
`arcsin(n·sin θ₂)`, symbol for symbol.

| | |
|---|---|
| stopped by | `early_stop` |
| generations | **169** (18.1 s) |
| ms/generation | 107.3 |
| train 1-R² | 1.598e-14 |
| validation 1-R² | 1.716e-14 |
| log10 p | −35.68 |
| test R² | 1.000000 |
| refused on type | 32,262 = **5.30% of gene-slots/generation** |
| depth histogram | 0:185 1:2352 **2:810** 3:173 4:47 5:23 6:6 7:4 |

810 genes sit at the ceiling and the recovered law is itself depth 2.

**What this measures, stated precisely:** T2 is *sufficient* for a depth-2 law,
and the population *used* depth 2 rather than leaving the rung unreached. It
does **not** measure that a ceiling of 1 would fail — that follows from the
type system without a run. It does not measure that 2 beats 3. And it does not
measure TSR against untyped on this law; see the prior-results table above for
what is and is not comparable.

### feynman_I_15_3t — `x/sqrt(1 − v²/c²)`-family, 180 s, seed 7014

Law not recovered in the budget: 1,159 generations at 155.3 ms/generation,
train 1-R² 3.753e-4, log10 p −10.53. 4.21% of gene-slots refused on type.

Depth histogram 0:1234 1:723 **2:1519** 3:95 4:10 5:15 6:4 — **the population
lives AT the ceiling here**, with more genes at depth 2 than at any other
depth. This is the clearest case so far that T2 is not a formality: the search
spends its budget right up to the limit and would go further if allowed.

### strogatz_bacres1, depth 0 — 420 s, seed 7014

Law not recovered, which is what 800+400 has always done on this problem
(prior result, `.claude/skills/running-fits/SKILL.md`: 65,264 generations at 800+400 reached
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

## The cost the spec did not anticipate: dead weight, not dead rows

The spec expects a refused gene to kill its row — "the row scores `PI` and dies
in the next tournament". **At `n_genes = 3` with `gene_subsets` on it kills
nothing.** The row scores on its surviving genes, so there is no selection
pressure against carrying an illegal one.

The magnitude, from these runs: on feynman_I_26_2, 7.0% of the final
population's genes are past the ceiling — about 253 genes, spread over up to
~250 of the 1,200 rows. **Roughly a fifth of the population is carrying an
unscored illegal gene that nothing removes.** That is a design consequence of
combining the spec's "let it land" preference with multi-gene chromosomes, and
it is the clearest thing here that the spec did not foresee.

If it is worth fixing, the cheap version is to have the refusal count against
the row's fitness rather than be silently skipped — but that changes selection
and would need its own measurement.

## What the ceiling costs

The pace figure comes from **bacres1 only** — 29.6 vs 26.9 ms/generation, about
**10%** — because that is the one problem here with an untyped run at matched
settings to compare against. The Feynman fits have **no untyped pace
comparison** and none should be inferred; their ms/generation is dominated by
75,000 rows against bacres1's 300.

Refusals run **2.5–5.3% of gene-slots per generation** across the fits.
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
