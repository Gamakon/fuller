# Does the 75-of-133 result depend on cohorts never merging?

An A/B on `cohort_merge`, everything else held at the settings that scored 75.

## What was asked

`docs/audit/PAPER_CODE_ALIGNMENT.md` finding 6: the run that scored **75 of
133** set `cohort_merge = 10,000` against fits of a few thousand generations, so
the elder band was never reached and the cohorts stayed permanently separate for
the whole run. The paper's §4.3 describes the elders co-mingling freely —
Hornby's unbounded top layer — which is **a merge that never fired in the run
being reported**.

Three possibilities, all genuinely open going in: merging is irrelevant at these
budgets; never merging is load-bearing; or merging would have helped.

## Settings

The 75 configuration verbatim (`docs/EXPERIMENTS.md:513-516`, the running-fits
skill), with `cohort_merge` as the only variable:

```
pop_intake 800, pop_champion 400
pump_every 100
gene_subsets on
max_generations 50000      (so TIME binds)
360 s a fit
no SMOGD/SMOTE             -> 6 objectives
```

**One knob drives both islands.** `champion_cohort_merge` defaults to `None` and
falls through to `config.cohort_merge` (`engine.rs:2884`), and
`champion_open_fight` is `false` at HEAD (`engine.rs:853`, commit 5919f31), so
VIRTUAL ALPS runs on the intake *and* the champion island under the single
`EVOLVE_COHORT_MERGE`. Without that the A/B would have varied one island only.

| arm | bands by design | what it is |
|---|---|---|
| `cohort_merge = 10000` | 100 | never merges, as the 75 run |
| `cohort_merge = 500` | 5 | merges at 5 pump beats — the engine's own `LIVE_COHORTS` |
| `cohort_merge = 100` | 1 + elders | a merge inside one pump beat. The engine's test calls this *"ALPS switched off wearing labels"* (`engine.rs:7499`); kept at the value asked for as the **near-off control**, not silently moved to 200 |

**18 fits**: 3 arms × 3 dev seeds (7013, 7014, 7015 — SRBench's official seeds
are the test set and were not used) × 2 datasets (`strogatz_bacres1`,
`strogatz_barmag2`). The three arms of each seed-dataset pair ran
**concurrently**, so all three saw the same machine contention — what matters
when TIME binds the budget and generations are the thing compared. Binary built
from `fix/resume-clock` @ 5919f31 with `--features gpu`.

Ground truth: bacres1 `20 - x - x*y/(1+0.5x²)`; barmag2 `0.5*sin(y-x) - sin(y)`.

## Three numbers that are not the same number, and the measurement turns on it

The first cut of this measurement was wrong and would have reported the opposite
conclusion. Recorded here because the trap is easy:

- **labels** — distinct cohort ids with rows. What the telemetry table shows.
- **bands** — what ALPS actually keeps apart: each non-elder label is its own
  band and **all elders are ONE band**. At `merge = pump_every` nearly every
  label is an elder, so *21 labels can be 2 bands*. Counting labels reads fast
  merging as *more* diversity when it is the opposite ALPS structure.
- **co-residency in the elder band** — and this is the one §4.3 is about. The
  question is not "is any cohort old" but "do **two or more distinct labels sit
  in the elder band at once**", because that co-residency *is* the co-mingling.
  Cohort 0 (born gen 0, `vec![0u32; pop]`) ageing past the line by itself merges
  with nobody.

One more trap: the **`cohort_merge` event is not a firing.** `engine.rs:5218` writes
exactly one at generation 0 declaring the aggregation rule, whatever the value.
Its count is always 1 and is evidence of nothing. Only `cohort_born` /
`cohort_extinct` track the population.

HEAD bands on **age** (`generation - label`). The 75 run's commit `05ef7e9`
banded on the **label**. This A/B measures HEAD's mechanism; under the label
rule a different population merges (cohorts born *after* gen 10,000, rather than
cohorts born before ~2,000 surviving *to* 10,000).

## Results

| dataset | seed | merge | gens | wall s | stopped | gen/s | 1-R2 train | 1-R2 val | log10 p | labels alive (mean/max) | bands alive (mean/max) | max labels co-resident in elders | found gen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bacres1 | 7013 | 10000 | 12012 | 360.0 | time | 33.4 | 1.753e-06 | 1.616e-06 | -15.83 | 6.9/10 | 6.9/10 | **1** | 2557 |
| bacres1 | 7013 | 500 | 12111 | 360.0 | time | 33.6 | 4.350e-07 | 3.940e-06 | -inf (sat) | 6.39/11 | 5.05/6 | **6** | 8794 |
| bacres1 | 7013 | 100 | 13681 | 360.0 | time | 38.0 | 4.705e-08 | 1.764e-07 | -inf (sat) | 13.8/21 | 2/2 | **20** | 2828 |
| bacres1 | 7014 | 10000 | 13623 | 360.0 | time | 37.8 | 2.230e-07 | 1.194e-08 | -inf (sat) | 14.91/23 | 14.91/23 | **1** | 87 |
| bacres1 | 7014 | 500 | 13549 | 360.0 | time | 37.6 | 2.230e-07 | 1.194e-08 | -inf (sat) | 8.03/12 | 5.51/6 | **7** | 87 |
| bacres1 | 7014 | 100 | 13768 | 360.0 | time | 38.2 | 2.230e-07 | 1.194e-08 | -inf (sat) | 10.43/16 | 2/2 | **15** | 87 |
| bacres1 | 7015 | 10000 | 10840 | 360.0 | time | 30.1 | 6.886e-07 | 2.652e-07 | -inf (sat) | 6.24/10 | 6.24/10 | **1** | 8460 |
| bacres1 | 7015 | 500 | 10028 | 360.0 | time | 27.9 | 1.819e-06 | 3.217e-07 | -17.03 | 6.55/12 | 4.98/6 | **7** | 5370 |
| bacres1 | 7015 | 100 | 9296 | 360.0 | time | 25.8 | 3.790e-06 | 2.691e-07 | -inf (sat) | 7.77/12 | 2/2 | **11** | 6445 |
| barmag2 | 7013 | 10000 | 43 | 1.1 | early_stop | 40.2 | 1.189e-13 | 1.124e-13 | -inf (sat) | 1/1 | 1/1 | **0** | 43 |
| barmag2 | 7013 | 500 | 43 | 1.1 | early_stop | 40.3 | 1.189e-13 | 1.124e-13 | -inf (sat) | 1/1 | 1/1 | **0** | 43 |
| barmag2 | 7013 | 100 | 43 | 1.1 | early_stop | 40.4 | 1.189e-13 | 1.124e-13 | -inf (sat) | 1/1 | 1/1 | **0** | 43 |
| barmag2 | 7014 | 10000 | 205 | 5.9 | early_stop | 34.6 | 9.825e-14 | 1.495e-13 | -inf (sat) | 3/3 | 3/3 | **0** | 205 |
| barmag2 | 7014 | 500 | 205 | 5.8 | early_stop | 35.2 | 9.825e-14 | 1.495e-13 | -inf (sat) | 3/3 | 3/3 | **0** | 205 |
| barmag2 | 7014 | 100 | 205 | 5.9 | early_stop | 34.8 | 9.825e-14 | 1.495e-13 | -inf (sat) | 3/3 | 2/2 | **2** | 205 |
| barmag2 | 7015 | 10000 | 13 | 0.3 | early_stop | 41.1 | 9.248e-14 | 7.228e-14 | -inf (sat) | 1/1 | 1/1 | **0** | 13 |
| barmag2 | 7015 | 500 | 13 | 0.3 | early_stop | 39.2 | 9.248e-14 | 7.228e-14 | -inf (sat) | 1/1 | 1/1 | **0** | 13 |
| barmag2 | 7015 | 100 | 13 | 0.3 | early_stop | 39.3 | 9.248e-14 | 7.228e-14 | -inf (sat) | 1/1 | 1/1 | **0** | 13 |

Wall seconds are invariant across arms too: 360.0 for every bacres1 fit (the
budget), and per barmag2 seed 1.1 / 5.9 / 0.3 s in all three arms.

`-inf (sat)` is the f32 angle SATURATING, not a missing measurement, and not
success. `1 - R²` is the working number throughout.

### Per arm

| merge | n | median gens | median 1-R2 train | median 1-R2 val | mean labels | mean bands | runs that co-mingled (>=2 in elders) | laws recovered |
|---|---|---|---|---|---|---|---|---|
| 10000 | 6 | 5522 | 1.115e-07 | 5.972e-09 | 5.51 | 5.51 | **0/6** | 3 early stops (barmag2 × 3 seeds; 2 clean by form, 1 with a near-zero constant subtree) |
| 500 | 6 | 5116 | 1.115e-07 | 5.972e-09 | 4.33 | 3.42 | 3/6 | the same 3, same generations, byte-identical models |
| 100 | 6 | 4750 | 2.352e-08 | 5.972e-09 | 6.17 | 1.67 | 4/6 | the same 3, same generations, byte-identical models |

### Per dataset (median 1-R2 train)

| dataset | cm10000 | cm500 | cm100 |
|---|---|---|---|
| bacres1 | 6.886e-07 | 4.350e-07 | 2.230e-07 |
| barmag2 | 9.825e-14 | 9.825e-14 | 9.825e-14 |

**That bacres1 row trends monotone, and it must not be read as an effect.** The
median hides two seeds pulling in opposite directions:

| seed | cm10000 | cm500 | cm100 | direction |
|---|---|---|---|---|
| 7013 | 1.753e-06 | 4.350e-07 | 4.705e-08 | **37× better** with fast merge |
| 7014 | 2.230e-07 | 2.230e-07 | 2.230e-07 | identical |
| 7015 | 6.886e-07 | 1.819e-06 | 3.790e-06 | **5.5× worse** with fast merge |

Two seeds in opposite directions at comparable magnitude, one flat. That is
what makes this null — seed noise, not a trend. And **none of the nine bacres1
fits is anywhere near the 1e-10 stop bar**: every one is a miss, in every arm.

## What it shows

**1. The audit's premise is confirmed, and sharper than it was stated.** At
`cohort_merge = 10000` the maximum number of distinct labels co-resident in the
elder band is **at most 1, in every one of the six runs**: 1 in the three
bacres1 runs, which do pass generation 10,000 (33-38 gen/s × 360 s ≈ 12,000),
and **0** in the three barmag2 runs, which stop at 13-205 generations and never
go near the line. So a cohort *does* cross the age line on bacres1 — and
crosses alone. Bands equal labels for the whole run. The elder band at the 75
settings is a **relabelling of one lone survivor**; no two lines ever co-mingled
in it. §4.3's "the elders co-mingle freely… Hornby's unbounded top layer" did
not happen, and "the threshold was never reached" understates it: on Strogatz it
*was* reached, and still united nothing.

**2. Merging changes cohort structure exactly as designed — and changes nothing
else.** Cohorts alive over time, from the per-snapshot telemetry
(`L` = labels, `B` = bands, `E` = distinct labels in the elder band):

```
cm10000 s7013   g200 3L/3B/0E   g5000 8L/8B/0E    g9800 8L/8B/0E    g12012 7L/7B/1E
cm10000 s7014   g200 3L/3B/0E   g5400 15L/15B/0E  g10600 21L/21B/1E g13200 20L/20B/1E
cm500   s7013   g200 3L/3B/0E   g5000 7L/5B/3E    g9800 6L/6B/1E    g12111 9L/6B/4E
cm500   s7014   g200 3L/3B/0E   g5400 11L/6B/6E   g10600 6L/6B/1E   g13200 10L/6B/5E
cm100   s7013   g200 3L/2B/2E   g5400 17L/2B/16E  g10600 21L/2B/20E g13200 12L/2B/11E
cm100   s7014   g200 3L/2B/2E   g5400 11L/2B/10E  g10600 14L/2B/13E g13200 15L/2B/14E
```

The arms are unambiguously different mechanisms: cm10000 is elder-free for the
first ~10,000 generations and then gets exactly one; cm500 holds ~5-6 bands (the
engine's `LIVE_COHORTS`) with 1-7 labels pooled in the elder band throughout;
cm100 is pinned at **2 bands from generation 200** with up to 20 labels pooled —
"ALPS switched off wearing labels", as the engine's own test says. The
manipulation worked. On the outcome it made no difference: **3 early stops in
each of the three arms, the same 3** (see point 3 for what they are by form).

**3. The recoveries are bit-identical across arms.** All nine barmag2 fits
early-stopped, and per seed the three arms agree on generation (43 / 205 / 13),
on 1-R² to the last digit, and on the reported model **byte for byte**. On form:

- s7015 and s7014 are the law. s7015:
  `-0.5·log(exp(sin y)·exp(sin y)·exp(sin(x−y)))` = `-0.5(2 sin y + sin(x−y))`
  = `0.5·sin(y−x) − sin(y)`, test R² 1.000000.
- **s7013 is the skill's trap 2** — the law with a constant in costume. Its
  `sin(sin(log|−95|)·81)` evaluates to 0.99206, and the whole tail
  `-1.5/3 × 0.99206 + 0.49603` is **−1.4e-07**, i.e. zero wearing four
  operators. The law is there; a symbolic checker may or may not accept it.

So: **3 early stops per arm, 2 of them clean by form and 1 carrying a near-zero
constant subtree.** SRBench's own symbolic check was not run here.

bacres1 seed 7014 shows the same invariance without recovering: identical 1-R²
2.230e-07 and `found_generation 87` in all three arms. Three of the four
answers were fixed *before* the variable could act — generations 13, 43 and 87
are all below cm100's first two-label elder band. The fourth is better than
that: barmag2 s7014 found its model at generation **205**, and at cm100 that run
had **2 labels co-resident in the elder band** from generation 200. The merge
had fired, and the model is still byte-identical to the never-merging arm.

**So: merging is irrelevant at these budgets. A null result.** Not "never
merging is load-bearing", and not "merging would have helped". On these two
problems the cohort-merge threshold is not what decided anything.

**4. What the paper should say.** The §4.3 co-mingling claim is not supported
and the "unbounded top layer" framing should go or be marked untested — but the
reason to drop it is *not* that merging hurt. It is that at the reported
budgets the mechanism never engaged, in either direction. The honest sentence is
that the 75 run's cohorts functioned as permanently-separate bands and nothing
measured here says that was better or worse than merging.

## Where this evidence is weak, and what would be better

This experiment is **badly posed for the question it was meant to answer**, and
that is the most useful thing it produced.

- **The wrong problems.** barmag2 solves in 13-205 generations. Cohort merging
  cannot possibly matter to a fit that finishes two orders of magnitude before
  the first merge. bacres1 never recovers in any arm, so it cannot show a
  recovery difference either. **The datasets where merging could bite are the
  ones that run long and solve late** — the 11 that hit the generation cap
  (`EXPERIMENTS.md:411`, "7 of them Strogatz"). Those should be the corpus.
- **n = 2 datasets, 3 seeds.** A 75-of-133 sweep turns on a handful of problems
  flipping. Two problems cannot see an effect worth 1-2 laws across 133.
- **Only HEAD's age rule was measured.** The 75 run banded on the label
  (`05ef7e9`) and merges a *different* population. An arm built at that commit
  would be a separate measurement.
- **Concurrency confound**, named rather than hidden: gen/s within a round
  spread 25.8-38.2. It correlates with neither arm nor outcome, but it is there.
- **The direct test was never run.** `cohort_merge = 0` — cohorts genuinely off
  — was not an arm. The paper's cohorts-on/off claim (66→75) rests on a
  different comparison and this did not touch it.

**What would close it.** Run the 11 cap-hitting problems, all seeds, at
`cohort_merge` ∈ {0, 500, 10000}, at the full 360 s. That is the population in
which the merge threshold has room to act, and `0` makes it a cohorts-on/off
test at the same time.

## Artefacts

- **Run cards, all 18, committed beside this file**:
  `cohort-merge-ab-cards/cm{10000,500,100}_{bacres1,barmag2}_s{7013,7014,7015}.json`.
  Each carries a `notes` entry recording what was tested and what happened,
  including its own co-residency number. They are also live at
  `logs/cards/<tag>.json` (symlinks into `logs/cohort_ab/<tag>/card.json`), so
  `diff logs/cards/cm10000_bacres1_s7013.json logs/cards/cm100_bacres1_s7013.json`
  shows the one line that differs.
- Telemetry: `logs/cohort_ab/<tag>/stream.jsonl`.
- Launcher: `logs/cohort_merge_ab.sh` (one arm) and `logs/cohort_ab_sweep.sh`
  (the 18).
- Stdout: `logs/cohort_ab/<dataset>_s<seed>_cm<merge>.log`.
