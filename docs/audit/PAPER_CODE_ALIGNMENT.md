# Paper ↔ code alignment audit

What `docs/paper/hff_sr.tex` claims, against what the code does. A standing
document to work down, not a rewrite of the paper. Nothing here is fixed by
this file; each row says what *would* fix it and leaves the decision.

**Audited at** `8b43455` (HEAD of `fix/resume-clock`, 23 Sep 2026).

**The run the paper describes** is the 75-of-133 sweep, seed 7014. Its boundary
is `05ef7e9` *"feat(evolve): VIRTUAL ALPS — couples from the same century"*
(23 Sep 02:11) — the commit that introduced the mechanism the run is about. The
result was written up at `5d4e0e4` (08:53). Nine further commits touched
`src/evolve/` later the same day; those are **after** the run.

Code claims are checked against **both** `05ef7e9` and HEAD, because a reviewer
greps HEAD but fairness asks what was true when the run happened. Where the two
differ the verdict is **STALE**, and both states are given.

## Verdicts

| | meaning |
|---|---|
| **FALSE** | not true of the run, and not true now |
| **OVERCLAIMED** | true in a weaker sense than the sentence states |
| **STALE** | true of the run; the code has since moved |
| **UNMEASURED** | stated as fact, never measured either way |
| **UNVERIFIABLE-HERE** | may be true; cannot be checked from this repo |
| **TRUE-BUT-FRAGILE** | true, and one default away from not being |
| **GAP** | code does something material the paper does not mention |

Severity answers one question: **would a reviewer who greps the repo find it?**

---

## Summary table

| # | § | claim | verdict | severity |
|---|---|---|---|---|
| 1 | §10 | "The p-value depends on how many objectives HFF carries" | **FALSE** | **critical** |
| 2 | §3.2 | "The engine computes nine objectives" | **FALSE** | **critical** |
| 3 | §3.4 | "algebraic simplification is … a variation operator inside the search" | **FALSE** | **critical** |
| 4 | §3.5 | "A fit stops when validation 1−R² ≤ 1e-10" | **FALSE** | **high** |
| 5 | §4.3 | the `better_mate` listing | **STALE** | **high** |
| 6 | §4.3 | "`cohort_merge` is the label past which cohorts stop being separate" | **STALE** | **high** |
| 7 | §4.1 | Table 1's six fits, configuration undisclosed | **OVERCLAIMED** | **high** |
| 8 | §4.5 | the interaction of cohorts and editing | **UNMEASURED** | **high** (already stated in-paper) |
| 9 | §3.1 | the model is `a·L(g₀,g₁,g₂)+b` | **OVERCLAIMED** | medium |
| 10 | §8.2 | "a beam mutant beat its original 0.05% of the time" | **UNVERIFIABLE-HERE** | medium |
| 11 | §8.4 | "9.06 s of a 15.6 s fit … 58%" | **UNVERIFIABLE-HERE** | medium |
| 12 | §8.3 | "30,180 candidates a beat" | **UNVERIFIABLE-HERE** | medium |
| 13 | §3.4 | the parity figures (33.6% / 84.9% / 47% / 15% / 13%) | **UNVERIFIABLE-HERE** | medium |
| 14 | §4.3 | "Setting it to 0 turns cohorts off entirely" | **TRUE-BUT-FRAGILE** | medium |
| 15 | §3.3 | the pump: best two over worst two, keep the best fifth | **TRUE** | — |
| 16 | §4.2 | the arithmetic, (1200/1500)^105 = 6.7e-11 | **TRUE** | — |
| G1 | — | the champion island's cohort rule is now a knob, defaulting **off** | **GAP** | **high** |
| G2 | — | `fold_winners` — an edit that lands beside the original | **GAP** | medium |
| G3 | — | wrappers (`Identity`, `LogAbs`, `SqrtAbs`) | **GAP** | medium |
| G4 | — | `arrival_children = 4`, `promote_fraction`, `elites` | **GAP** | medium |
| G5 | — | the balanced-pole result (26 vs 53 on one seed) | **GAP** | medium |
| G6 | — | SMOGD/SMOTE inconsistency between the two halves of the bar | **GAP** | medium |
| G7 | — | `Config::srbench` defaults now describe bacres1, not the 75 run | **GAP** | low |
| G8 | — | code-internal doc bugs (`snap_every`, the fold's two comments, CLAUDE.md) | **GAP** | low |

---

## 1. "The p-value depends on how many objectives HFF carries" — FALSE — critical

**Claim** (§10, lines 820-825):

> The bar was calibrated on a different configuration: four objectives, a
> different seed, and before the precision fix … The p-value depends on how many
> objectives HFF carries, so both of those shift the distribution. … we should
> recalibrate it per configuration or drop it.

**Code.** `src/evolve/engine.rs:1232-1277`, `hff_p_value`, whose docstring is
now an extended refutation of exactly this sentence:

> **THE P-VALUE IS NOT SUSCEPTIBLE TO DIMENSIONALITY. THAT IS ITS WHOLE POINT.**
> `m` is an argument to the incomplete beta, so the dimension is CONSUMED by the
> CDF and never survives into the answer. … a p of 1e-19 means the same thing at
> 6 objectives, at 9, and at 19,000.

And `stop_log10_p`'s own docstring, `engine.rs:669-675`:

> THIS BAR DOES NOT NEED RECALIBRATING WHEN THE OBJECTIVE COUNT CHANGES. An
> earlier version of this comment ended "Measured with 4 objectives …; p depends
> on how many objectives HFF has", which is wrong and cost an afternoon.

The implementation is one line: `cdf_beta_correction(theta, m)` =
$I_{\sin^2\theta}((m-1)/2, 1/2)$ — which the paper itself writes correctly at
§3.2 line 156. **The paper contradicts itself**: §3.2 gives the dimension-free
formula, §10 reasons as though the dimension survives.

**Important:** the *measurements* in §10 stand — every $\log_{10} p$ between −30
and −38, medians −36.3 and −35.1 overlapping, the bar never binding. Those match
`docs/EXPERIMENTS.md:528-534`. It is only the **explanation** that is wrong, and
the remedy that follows from it ("recalibrate per configuration").

**What would make it true.** Delete the dimensionality clause. The paper already
names the defensible explanation in the same paragraph — the precision fix moved
the f32 floor from 1.2e-10 to 1.1e-14, so the *angle* moved. The corrected
reasoning is: p is dimension-free; when p differs between runs the angle
differs; so look at the objectives, not the count.

**Reviewer exposure.** Critical, and it is the kind a referee enjoys: the
refutation is in the docstring of the function the paper cites, and the paper's
own §3.2 formula disproves its §10 prose.

**Note.** `CLAUDE.md` still carries the superseded claim ("`stop_log10_p = -19`
is calibrated for FOUR HFF objectives … At 9 it is unreachable"). The skill and
the code are corrected; CLAUDE.md is not. See G8.

---

## 2. "The engine computes nine objectives" — FALSE — critical

**Claim** (§3.2, line 130):

> The engine computes nine objectives: mean squared error, $1 - R^2$ and mean
> absolute error, each on three blocks of rows (train, validation, and a third
> block that may be synthetic-noisy or edge rows).

**Code.** `src/evolve/engine.rs:1330-1341`, `hff_columns`:

```rust
let absent = (block == 2 && n_extrap == 0) || (block == 1 && without_validation);
```

The third block is **dropped entirely** when `n_extrap == 0`. `hff_dimensions`
(`engine.rs:5078-5082`) is `hff_columns(...).len() + redundancy + tower`.
Defaults at `engine.rs:857-861`: `redundancy: false`, `tower: false`,
`smogd: false`.

`.claude/skills/running-fits/SKILL.md:108-117` records the 75 run as
`no SMOGD/SMOTE -> 6 objectives`. `docs/EXPERIMENTS.md:513` lists that run's
settings and names neither a third block nor the tower.

**Evidence caveat.** No run card exists for the 75 run — `logs/cards/` holds only
`bacres1_fixed.json`. So the objective count rests on the skill's reconstruction
and on the defaults, not on an artifact the run emitted. The card format records
a derived `hff_objectives` field, which is exactly what would settle this; it
postdates the run.

**Why it matters beyond a count.** §3.2 spends a whole paragraph on the *tower*
objective — the transcendental-depth column, calibrated on 1,230 labelled fits.
`tower: false` is the default and the run's setting list does not turn it on. If
the tower was off, the paper describes as a component of the engine a mechanism
that contributed nothing to 75, and the "nine objectives" framing is what makes
it sound live.

**What would make it true.** Either state the run's actual objective set — six
columns, no third block, tower off — and present the tower and the third block
as available mechanisms that this run did not use; or produce the run card /
log showing `hff_objectives: 9`. The card format (`logs/cards/*.json`) records
the derived `hff_objectives` count, which settles it in one line.

**Reviewer exposure.** High. `hff_columns` is 11 lines and the conditional is
the second one.

---

## 3. "Algebraic simplification is a variation operator inside the search" — FALSE — critical

**Claim** (§3.4, lines 194-199):

> A chromosome decodes to an expression, the expression is saturated and a
> variant extracted, and the simplified form converts *back into a Karva gene*.
> Algebraic simplification is therefore a variation operator inside the search,
> not a cleanup pass at report time.

**Code.** `denoise` — the saturate → `extract_variants` → gene round trip — has
**no call site in `src/evolve/`**. Every occurrence outside `src/extract.rs` and
`src/python.rs` is a doc comment (`src/snap_karva.rs:13`, `src/snap.rs:13`,
`src/lib.rs:6`, `src/physics.rs:1`, `src/ruleset/wide.rs:3`, `src/karva.rs:412`).
`src/lib.rs:7-8` states its consumer: "The `python` feature exposes `denoise` to
the Python SR engine via PyO3."

The in-search editor is **snap**, which is a constant-lattice match, not
algebraic saturation: `src/evolve/write_back.rs:1-40` describes match → graft →
R² guard → write back in place, and imports `lint::snap_table`, not egglog.

**What would make it true.** Either a `denoise` call site in the Rust generation
loop, or reword §3.4 to distinguish the *capability* (the crate converts both
ways — `src/karva.rs`) from what ran (snap, in place, guard-gated). §4.5 of the
paper now does exactly this; §3.4 has not been brought into line with it, so the
paper currently says both.

**Reviewer exposure.** High. `grep -rn denoise src/evolve/` returns nothing.

---

## 4. "A fit stops when validation 1−R² ≤ 1e-10" — FALSE — high

**Claim** (§3.5, line 221):

> A fit stops when validation $1 - R^2 \le 10^{-10}$ and $\log_{10} p \le -19$.

**Code.** `src/evolve/engine.rs:5418`:

```rust
if s.one_minus_r2[1] <= c.stop_one_minus_r2 && train_ok && edge_ok && log10_p <= c.stop_log10_p
```

Three error conditions, not one: validation **and train** (`engine.rs:5417`)
**and edge** (`engine.rs:5405`, skipped when `smogd` is on or there is no third
block). The train condition was added by `3673062`, *"fix(evolve): the stop bar
counts the TRAIN side too"*, **22 Sep 22:24 — before the run**. So this is
FALSE for the run, not stale.

The commit's own justification is a result the paper would want: bacres2 stopped
with val 1−R² 9.46e-11 while train was 1.03e-9 — validation ten times better
than the rows the model was fitted on. Over the cascade's 117 early stops, all
113 real wins have train 1−R² ≤ 1e-10 (median 9.8e-15); bacres2 is the only one
that does not.

**Note** this interacts with §3.5's own calibration paragraph, which cites
bacres2 as "the one fake that slipped under the $R^2$ bar" and credits the
p-value half with catching it. The train-side fix catches it too, and on the
error half. The paper attributes to p a job the code now does twice.

**What would make it true.** State all three conditions. Optionally note that
the train side is what actually excludes bacres2 — which strengthens §10's
argument that the p half is not earning its place.

**Reviewer exposure.** High. One `grep stop_one_minus_r2` finds the conjunction.

---

## 5. The `better_mate` listing — STALE — high

**Claim** (§4.3, lines 320-327) — presented as the device implementation:

```
fn better_mate(me: u32, c: u32, w: u32) -> bool {
    let mc = same_cohort(me, c);
    let mw = same_cohort(me, w);
    if (mc != mw) { return mc; }
    return key(c) < key(w);
}
```

**Code.** `src/evolve/vary.wgsl:238-247`:

```wgsl
fn better_mate(me: u32, c: u32, w: u32, open_fight: bool, merge: u32) -> bool {
    if (!open_fight) {
        let mc = same_cohort(me, c, merge);
        let mw = same_cohort(me, w, merge);
        if (mc != mw) {
            return mc;
        }
    }
    return key(c) < key(w);
}
```

Two extra parameters, and the whole cohort test sits inside `if (!open_fight)`,
a per-island flag that **disables the cohort rule entirely**.

**At the run's commit the paper is exactly right.** `git show
05ef7e9:src/evolve/vary.wgsl:176-183` is the printed listing character for
character, three arguments and no `open_fight`. `open_fight` entered the WGSL at
`b6ef460` (23 Sep 12:44) and its default was set at `5befc0d` (18:25), both
hours after the run.

So this is **STALE, not FALSE** — the paper documented the code accurately and
the code moved the same day. It stays at high severity only because a verbatim
listing is the single most diff-able thing in a paper, and at HEAD the diff is
two extra parameters and a branch that turns the mechanism off.

**What would make it true.** Label the listing "as it stood for this run, commit
`05ef7e9`". One clause, and it is then permanently correct.

---

## 6. "`cohort_merge` is the label past which cohorts stop being separate" — STALE — high

**Claim** (§4.3, lines 334-337):

> `cohort_merge` is the label past which cohorts stop being separate: every
> cohort at or beyond it is clamped to one band.

**Code.** `src/evolve/vary.wgsl:177-189`, `band_of`:

```wgsl
var age = 0u;
if (gp.generation > label) { age = gp.generation - label; }
if (age >= merge) { return ELDERS; }
return label;
```

It bands on **age** (`generation - label`), not on the label. The comment above
it, `vary.wgsl:173-176`, records this as a fixed bug:

> The age is `generation - label`, not the label: the label says WHEN a line
> arrived and is fixed for ever, so banding on the label alone kept the OLDEST
> cohorts apart permanently and merged the youngest — the opposite of the
> design, and it meant the most refined lines never mingled at all.

**At the run's commit the paper is right again** — and that is the finding.
`git show 05ef7e9:src/evolve/vary.wgsl:156-163`:

```wgsl
fn same_cohort(a: u32, b: u32) -> bool {
    if (gp.cohort_merge == 0u) { return true; }
    let ca = min(cohort_now[a], gp.cohort_merge);
    let cb = min(cohort_now[b], gp.cohort_merge);
    return ca == cb;
}
```

That is a clamp on the **label**, exactly as §4.3 describes. So the run used the
version the current code comment calls backwards, and the paper documents it
faithfully. **STALE**, and the more interesting fact is what it implies about
the run rather than about the paper.

**The mechanism was close to inert in the run either way.** The 75 run set
`cohort_merge = 10,000` (§5, `docs/EXPERIMENTS.md:513`). Under the label clamp,
two lines merge only once both labels exceed 10,000 — i.e. only after generation
10,000. §5 reports Strogatz at ~33 generations/s and Feynman at ~12, so in 360 s
a Feynman fit reaches ~4,300 generations and never merges at all; only the
longest Strogatz fits cross the threshold late. Under the age rule at HEAD the
same threshold is just as far away.

So §4.3's *"every cohort at or beyond it is clamped to one band, so the elders
co-mingle freely"* — offered as the paper's realisation of Hornby's unbounded
top layer — describes a merge that **almost never fired** in the run being
reported. The cohorts stayed separate throughout, which may well be why the run
worked; but it is not what the section says happens.

**What would make it true.** Two clauses: state that the run banded on the label
(`05ef7e9`) and that HEAD bands on age, and note that at
`cohort_merge = 10,000` against fits of a few thousand generations the elder
band was effectively never reached. The second point is worth having on its own
— it means "the unbounded top layer" is untested, not merely differently
implemented.

**Reviewer exposure.** High. `band_of`'s comment at `vary.wgsl:173-176`
explicitly records label-banding as a fixed bug, so a reviewer at HEAD reads the
paper as describing the bug.

---

## 7. Table 1's six fits: configuration undisclosed — OVERCLAIMED — high

**Claim** (§4.1, lines 243-244):

> Six fits on seed 7013, population 1,500 intake plus 1,500 champion, head 34,
> 30 seconds each.

**Source.** `docs/EXPERIMENTS.md:142-144` gives the same six fits with the rest
of the configuration:

> the race's settings (`EVOLVE_TOWER=1 EVOLVE_HFF_NO_VAL=1 EVOLVE_RNC_LO=-5
> EVOLVE_RNC_HI=5 EVOLVE_PUMP_EVERY=5`). One seed, on a GPU shared with running
> races, so the timings are under contention.

Three omissions that bear on the result:

- `EVOLVE_HFF_NO_VAL=1` — **no validation block**. By `hff_columns`
  (`engine.rs:1334`) that removes three columns. The measurement that motivates
  the whole paper ran on a different objective set from the run it motivates.
- `EVOLVE_TOWER=1` — the tower *was* on here, and (per finding 2) apparently not
  in the 75 run. The paper has it the other way round by implication.
- `EVOLVE_RNC_LO/HI=-5..5` — against the 75 run's −100..100
  (`engine.rs:843-844`). `docs/EXPERIMENTS.md:130` notes "Snap needs RNC
  −100..100 to bite", so snap was effectively inert in these six fits.

**Why it is OVERCLAIMED rather than FALSE.** Everything the paper states is
true. The diversity-collapse finding is not in question — 294,000 fresh
individuals, zero surviving descendants, and the founder arithmetic all check
out against EXPERIMENTS.md:161-190. What is missing is that this configuration
is not the configuration of the run the paper reports.

**What would make it true.** One clause naming the settings, and a sentence
saying the collapse was measured under a different objective set and RNC range
and is assumed to carry. It does plausibly carry — the arithmetic in §4.2 is
about tournament size against keeper fraction and does not depend on objectives.
Say that.

**Reviewer exposure.** Medium-high: only if they read EXPERIMENTS.md. But the
paper cites the same six fits, so a reader who finds the source finds the gap.

---

## 8. The cohorts × editing interaction — UNMEASURED — high (already disclosed)

**Claim** (§4.5): cohorts and gene editing are one mechanism.

**Status.** The paper states this honestly and at length: the 2×2 has one row,
the editing-off row has never been run in either cohort condition, and §8.2's
beam null measures a different question. No action needed — logged so the audit
covers the section, and because it is the one place where the paper's disclosure
is already exemplary.

**What would close it.** A cohorts-on/editing-off arm on the 75 configuration.
Cheap: `EVOLVE_SNAP_EVERY=0` against the run's settings.

---

## 9. The model is `a·L(g₀,g₁,g₂)+b` — OVERCLAIMED — medium

**Claim** (§3.1, lines 106-108):

> the model as scored is $a \cdot L(g_0, g_1, g_2) + b$ with $a$ and $b$ fitted
> by ordinary least squares.

**Code.** `src/evolve/engine.rs:5091` and `write_back.rs:19`:

> `a * WRAPPER(LINKER(genes)) + b`

`WRAPPERS = [Identity, LogAbs, SqrtAbs]` (`engine.rs:1229`), and a row's
`wrapper_id` is carried through the pump, the fold and snap
(`engine.rs:3183`, `4020`). The chromosome keeps the best wrapper as it keeps
the best linker.

Since `Identity` is one of the three, the paper's formula is the wrapper=Identity
special case — not wrong, but it omits a third of the model's structure. §3.1
says "Two decisions here matter later" and lists the gene-subset choice and f32;
the wrapper is a third and is never mentioned.

**What would make it true.** Write `a · W(L(genes)) + b` and name the three
wrappers. One sentence.

---

## The numeric sweep

Every number in the paper, checked against `docs/EXPERIMENTS.md` or the code.
Recorded in full so a reader can tell what was checked from what was skipped.

**Checked and matching:**

| § | figure | source |
|---|---|---|
| abstract, §5 | 75 of 133 = 56.4%; cascade 66; 6 h 04 m | `EXPERIMENTS.md:493-505` |
| abstract, §5 | AIFeynman 54.1% (72/133), GP-GOMEA 27.1%, 0.57% of budget | `EXPERIMENTS.md:507-511` |
| §5 Table 2 | per-challenge 33/25/10/0/7 against 30/24/9/0/3 | `EXPERIMENTS.md:499-505` |
| §5 | settings: 800+400, pump 100, `cohort_merge` 10,000, cap 50,000 | `EXPERIMENTS.md:513-516` |
| §5 | Strogatz 33 gen/s against Feynman 12 | `EXPERIMENTS.md:514-516` |
| §5 | double submission: 75 tidied against 71 fuller's own | `EXPERIMENTS.md:535-537` |
| §4.1 | the six fits, ages, founders, 1201/1 | `EXPERIMENTS.md:146-153` |
| §4.1 | 294,000 fresh; 73,990 rows; zero `pump_refill`; 54%/69% | `EXPERIMENTS.md:161-190` |
| §4.2 | `(1200/1500)^105 = 6.7e-11`, two-stage | `EXPERIMENTS.md:183-190`, `vary.wgsl:159-161` |
| §4.4 | barmag2: 6.9e-3 → early stop gen 614 at 6.66e-14 | `EXPERIMENTS.md:518-521` |
| §4.4 | Strogatz 3 of 14 → 7 of 14 | `EXPERIMENTS.md:504` |
| §6 Table 4 | classes A/B/C/D = 3/16/11/53 | `EXPERIMENTS.md:408-412` |
| §6.3 | bloat: median 323 characters against 122, over 342 fits | `EXPERIMENTS.md:417-421` |
| §7 | the ten recovered laws, by name | `EXPERIMENTS.md:105` |
| §8.1 | 8,300 wrap candidates, 0 better, 0 grafted, 0 kept | `EXPERIMENTS.md:254` |
| §8.4 | population 20,000 at 42 against 46; redundancy objective | `EXPERIMENTS.md:51` |
| §9 | the 3-way restart split scored 54, rejected | `EXPERIMENTS.md:50` |
| §10 | log10 p −30 to −38, medians −36.3 / −35.1 | `EXPERIMENTS.md:528-534` |
| §10 | union 77 of 133; ceiling 82 | `EXPERIMENTS.md:89`, `:364` |
| §3.1 | head 34, 3 genes, 15 gene-linker combinations, 7% tournament | `engine.rs:822,837-840` |
| §3.2 | log scale `1 + log10(x)/12` | `engine.rs:1360,1377` (`HFF_LOG_FLOOR = 1e-12`) |
| §3.2 | tower: 1,230 labelled fits, every true law ≤ 2, ¾ of fakes ≥ 3 | `engine.rs:1279-1288` |
| §3.5 | bacres2 at −17.55; real laws −19.4 or below | `engine.rs:662-666` |

**Not found in this repo** — findings 10-12 below, plus:

- §4.4's III.8.54 arm (surviving lines 1 → 483, pump-founder share 0% → 100%,
  1−R² 6.70e-1 → 1.12e-1). No match in `EXPERIMENTS.md`. This is the paper's
  **only same-seed cohorts-on/off comparison** and the load-bearing evidence for
  §4.5's "editing-on row has both cells", so it deserves a cited log.
- §8.1's swim lanes (uniform arm 4, differentiated 3; I.15.10 solved at
  generation 186 against 958 unsolved). Not in `EXPERIMENTS.md`. The mechanism
  is real (`lanes` at `engine.rs:856`); the measurement is unsourced here.
- §6.2's non-finite scan (94/38, "one law timed out"). The corrected count of 38
  is at `EXPERIMENTS.md:c4661f5`; the 94/66 split is not.

**Superseded elsewhere, correct in the paper:** `EXPERIMENTS.md:45fe8d9` and
`d675797` (23 Sep 14:23-14:26) fixed the summary's "47 of 133" and re-baselined
the cascade from 47 to 64→66. The paper's Table 2 uses 66 and is unaffected.

---

## 10-12. §8 figures not traceable in this repo — UNVERIFIABLE-HERE — medium

Three numbers in §8 could not be located in `docs/EXPERIMENTS.md` or any log
under `logs/`:

| § | figure | searched |
|---|---|---|
| 8.2 | "a beam mutant beat its original 0.05% of the time", over six near misses | `0.05%` — no match in `docs/` or `logs/*.log` |
| 8.2 | "changing the population size alone moved $\log_{10} p$ by up to 1.8 decades" | not found |
| 8.3 | "30,180 candidates a beat where the wrap-only design made 180" | matches only in unrelated `checkpoint/*.json` and `.dat` |
| 8.4 | "9.06 s of a 15.6 s fit — 58%" | not found |

The related figure **is** traceable: §8.2's "8,300 wrap candidates, 0 better, 0
grafted, 0 kept" matches `docs/EXPERIMENTS.md:254` exactly. So the section is
sourced in part.

**Not an accusation.** These are very likely real measurements. Both beam logs
named at `docs/EXPERIMENTS.md:318` — `logs/beam_host_gene_alone.log` and
`logs/beam_inside_linker_measurement.log` — **are** present in this worktree, and
`grep` finds none of the four figures in either. So the numbers come from
somewhere else again: a third log, a notebook, or a calculation over the beam
table. Recorded so each gets a citation.

**§7 checked and TRUE, by contrast**, which is what a well-sourced claim looks
like: the small-angle bound of $|u| < 10^{-5}$ with the cubic term under
$3.4\times10^{-16}$ is `b1eb9df` (22 Sep 22:15, before the run), the bound
appearing in both the commit message and the guard itself. lv2's argument
reaching $1.95\times10^{-3}$, and the decision not to loosen, are recorded with
it.

**What would make it verifiable.** Add the source log path beside each figure in
EXPERIMENTS.md, as the beam table already does.

---

## 13. The parity figures — UNVERIFIABLE-HERE — medium

**Claim** (§3.4, lines 213-218): 33.6% overall, 84.9% powsimp, 47% trigsimp,
~15% radsimp/ratsimp, 13% simplify.

These match `CLAUDE.md` exactly, so they are the project's own recorded numbers.
Confirming them means running
`cargo run --release --bin parity -- parity/corpus/*.jsonl`, which is a
saturation run and per CLAUDE.md must be kill-guarded. **Not run here** — the
audit brief says not to invent measurements, and a parity run is a measurement.

**What would make it verifiable.** Re-run parity at a known commit and record
the figures with that commit in EXPERIMENTS.md. The paper's numbers would then
have a dated source rather than inheriting CLAUDE.md's.

**Related claim, checked and TRUE:** §3.4's non-confluence statement
(distribution co-saturated with trigonometry or rational arithmetic explodes the
e-graph; the in-search subset is bounded algebra-plus-powers; distribution is
excluded) matches CLAUDE.md's "Non-obvious things" and `src/ruleset/`. Note the
claim describes `denoise`'s subset — and per finding 3, `denoise` did not run in
the search, so "the in-search simplifier therefore uses only a bounded
algebra-plus-powers subset" describes a simplifier that was not in the search.

---

## 14. "Setting it to 0 turns cohorts off entirely" — TRUE-BUT-FRAGILE — medium

**Claim** (§4.3, lines 337-339): setting `cohort_merge` to 0 turns cohorts off,
"and the CPU/GPU bit-for-bit parity tests then prove the engine is what it always
was."

**Code.** True by two independent paths: `vary.wgsl:196-199` (`merge == 0u` →
`same_cohort` always true) and the host's `self.cohorts.is_empty()` guards
(`engine.rs:3185`, `3206`, `4026`).

**Fragile** because `engine.rs:7479` asserts the *opposite* configuration is what
the measurement means:

> `assert!(c.cohort_merge > c.pump_every, "a merge inside one beat is ALPS
> switched off wearing labels")`

So there are two ways to switch ALPS off — 0, and any value below `pump_every` —
and the paper documents only the first. A reader setting `cohort_merge = 10` with
`pump_every = 100` would believe cohorts are on.

**What would make it robust.** Note that `cohort_merge` must exceed `pump_every`
to mean anything, and that `LIVE_COHORTS = 5` (`engine.rs:807`) is the intended
ratio.

---

## 15-16. Checked and TRUE

**§3.3, the pump.** "the intake's best two individuals are copied over the
champion island's worst two; the intake then keeps its best fifth — one of each
distinct row — and the rest is refilled with fresh random individuals."
Matches `engine.rs:3176-3210`: keepers then arrivals then fresh, keepers keep
their cohort (`3186`), fresh rows take the current generation as their label
(`3207`). ✓

**§4.2, the arithmetic.** `(1200/1500)^105 = 6.7e-11`; two-stage selection makes
it tighter. Matches `docs/EXPERIMENTS.md:183-190` and `vary.wgsl:159-161`. ✓

**§3.2, the HFF formula and the log scale.** `hff_scaled`
(`engine.rs:1352-1365`) caps at the column max, scales to [0,1], and applies
`1 + log10(x)/12` via `HFF_LOG_FLOOR`. ✓ (Though `log_scale: [false; 3]` is the
default — see G6.)

**§3.1, head 34 and the 15 combinations.** `head: 34` with its justification at
`engine.rs:838-840`; `n_genes: 3`; 3 singles + 3 pairs × 3 linkers + 3 whole = 15. ✓

**§3.5, the calibration.** bacres2 at −17.55, real laws at −19.4 or below.
Matches `engine.rs:662-666`. ✓

---

# GAPS — what the code does that the paper does not say

## G1. The champion island's cohort rule is a knob, and at HEAD it defaults OFF — high

`engine.rs:833`: `champion_open_fight: true`. Per `vary.wgsl:238-245`,
`open_fight` **skips the cohort test entirely**. So at HEAD, Virtual ALPS runs on
the intake island only.

**In the run it ran on both.** At `05ef7e9` there is no `open_fight` anywhere in
`vary.wgsl` (`grep -c` returns 0), and `better_mate` is called unconditionally at
both tournament sites — `vary.wgsl:208` and `:249`. So the 75-of-133 result was
produced with cohort restriction on the intake *and* the champion island, and
HEAD's default reverses that.

`vary.wgsl:216-234` records the measurement that argued the other way:

> Measured on strogatz_bacres1 at 2,000 + 2,000: with the rule lifted, cohort 160
> held 96.9% of the champion island from generation 160 to 33,840, pinned within
> three rows the whole way — while being only SIX PERCENT better than its nearest
> challenger. … the open knockout becomes a closed one — the same failure the
> lift was meant to cure, in a worse form.

And concludes *"VIRTUAL ALPS RUNS ON BOTH ISLANDS"* — **while the default
immediately below it says otherwise**. That is a code bug independent of the
paper: `engine.rs:833` sets the flag that the comment argues against, citing a
measurement that says lifting the rule recreates the failure it was meant to
cure.

**Why it matters for the paper.** §4 never says which islands cohorts apply to.
A reader who builds HEAD and reruns gets the mechanism on one island instead of
two and will not reproduce the run. Worth one sentence in §4 and, separately,
worth someone deciding whether `champion_open_fight: true` is intended.

## G2. `fold_winners` — medium

`engine.rs:3784`, on the pump's beat: a subtree that flattens its input to a
constant is collapsed, and **the clean gene lands in its island's worst row,
original untouched, unevaluated, for the tournament to judge**
(`engine.rs:4015-4030`). This is the only mechanism in the engine matching §4.5's
"proposal beside the original".

Three things the paper would need if it mentioned it, and it currently does not:

- It **postdates the run** — not a component of 75.
- The A/B is **1 win / 2 losses** at 400 generations across seeds 7014/7015/7016
  (`engine.rs:890-900`).
- The edit **inherits its parent's cohort** (`engine.rs:4021`): *"so the
  candidate carries its parent's cohort and fights in its parent's ALPS band
  rather than wearing the label of the row it displaced."* So the built proposal
  pattern deliberately does **not** give an edit a young protected band — which
  is the mechanism §4.5 argues for. Worth stating as a design tension.

Default is contested: `engine.rs:907` reads `fold_every: pump_every` (**on**),
while the comment block immediately above says "THE FOLD OFF BY DEFAULT" and
CLAUDE.md says "OFF by default (`fold_every: 0`)". See G8.

## G3. Wrappers — medium

See finding 9. `Identity`, `LogAbs`, `SqrtAbs`, selected per chromosome. Nowhere
in the paper.

## G4. Selection parameters the paper does not state — medium

All at `engine.rs:820-856`, all bearing on §4's argument about who meets whom:

- `arrival_children: 4` — with a measurement: *"One was what the band shipped
  with and it could not hold ground: 60 arrivals in 120 rows against 1,878
  tournament rows were re-swamped within a beat, and the champion island sat at
  97% one cohort for 19,000 generations."* That is a diversity-collapse result of
  the same kind as §4.1's and is unreported.
- `elites: 2`, `promote_fraction: 0.03`, `champion_tournament_fraction: None`.
- `LIVE_COHORTS = 5` — the band count, `cohort_merge / pump_every`. The 75 run's
  `cohort_merge = 10,000` with `pump_every = 100` gives 100 bands, not 5.

## G5. The balanced pole — medium

`docs/EXPERIMENTS.md:96`: *"The balanced pole is wrong for selection — 26 vs 53
on the same seed, everything else equal. TrueNorth won 30 laws the balanced pole
missed."* Code: `hff_balanced` (`engine.rs:1370`), `balanced_tournaments`
(`engine.rs:867`, default false).

**A 27-law swing on one seed is larger than any effect the paper reports**,
including cohorts (66→75). It is a load-bearing design decision — the paper's
§3.2 asserts the TrueNorth pole without saying it was chosen by measurement or
what the alternative cost. Cheap and strong to add.

## G6. SMOGD/SMOTE: the two halves of the bar disagree — medium

The synthetic third block's docstring: *"They are noisy on purpose — they rank
individuals in the tournaments; they never decide that a fit is exact."*

The 1−R² half honours it — `c.smogd ||` short-circuits the edge check
(`engine.rs:5405`). The p half does not: the third block's error enters the angle
like any other objective. The skill documents this as *"a real inconsistency …
One of the two is wrong and it is a design question for Andrew."*

The paper's §3.2 mentions "a third block that may be synthetic-noisy or edge
rows" and says nothing about either the inconsistency or the fact that this block
is absent by default. Bears directly on §10's p-value discussion.

Also unmentioned: `log_scale: [false; 3]` is the default, while §3.2 argues at
length that the log scale "matters at the accuracy we need".

## G7. `Config::srbench` no longer describes the 75 run — low

Defaults have moved to the bacres1 recovery: `pump_every` → 20 and
`cohort_merge` → 100 (`engine.rs:855`, five bands), against the run's 100 and
10,000. `max_generations: 1500` / `max_seconds: 30.0` against the run's 50,000
and 360. The paper states its run's settings explicitly (§5), so this is not a
misalignment — but anyone reproducing from defaults gets a different engine.

## G8. Code-internal documentation bugs — low

Not paper misalignments; recorded because they will mislead the next reader.

1. **`snap_every` docstring vs. its default.** `engine.rs:699` ends *"0 = off,
   the default"*, but `engine.rs:881` sets `snap_every: pump_every` and
   `write_back.rs:952` asserts `snap is on by default`.
2. **The fold's two contradictory comments.** `engine.rs:888-906` contains both
   *"THE FOLD OFF BY DEFAULT, because the measurement does not yet say it pays"*
   and, immediately after, *"ON, on the pump's beat — Andrew, asked three times"*.
   The code does `fold_every: pump_every` (on).
3. **CLAUDE.md carries the superseded p-value claim** — *"`stop_log10_p = -19`
   is calibrated for FOUR HFF objectives. Runs have 6 or 9. At 9 it is
   unreachable"* — which `engine.rs:1238-1270` and the running-fits skill both
   now refute. CLAUDE.md also still says the fold is *"OFF by default
   (`fold_every: 0`)"*.

---

## The three worst

1. **Finding 1** — §10 reasons from a dimensionality dependence that the p-value
   does not have, and the paper's own §3.2 formula disproves it. Self-contradiction
   inside one paper, refuted by the docstring of the cited function.
2. **Finding 2** — "nine objectives" appears to describe a configuration the run
   did not use, and carries the tower objective's whole paragraph with it.
3. **Finding 3** — §3.4's "variation operator inside the search" for a function
   with no call site in the engine. §4.5 already says the honest version; §3.4
   has not been brought into line, so the paper says both.

All three are found by grepping, and the first two are in the sections a referee
reads most carefully.

## A note on the STALE ones

Findings 5 and 6 are the paper's §4.3 describing the cohort mechanism, and at
the run's own commit `05ef7e9` **both are exactly right** — the three-argument
`better_mate` character for character, and the label clamp. The code moved the
same day the paper was written. That is not a defect in the paper so much as a
reason to pin it: a version-dated sentence ("as it stood at `05ef7e9`") turns
two high-severity findings into permanent correctness for the cost of one
clause each.

The genuinely new fact those two turned up is not about the prose at all. The
75 run held `cohort_merge = 10,000` against fits of a few thousand generations,
so **the elder band was effectively never reached** — under either the label
version or the age version. Virtual ALPS ran in the run with its cohorts
permanently separate and Hornby's unbounded top layer inactive. Whether the
result depends on that is untested, and it is a cheaper experiment than the one
§4.5 asks for.
