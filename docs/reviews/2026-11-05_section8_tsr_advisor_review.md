# Advisor review: plan section 8, Typed Symbolic Regression (TSR) on the new data model

Reviewed: `docs/PLAN_futhark_kingdom.md` section 8 at fuller e1a017d1, against fuller `kingdoms/tsr/gen_tables.py`,
`src/geneframe.rs`, `src/typed_environment.rs`, `src/kingdom_contract.rs` and phylu `src/evolve/typed_decode.wgsl`,
`order.rs`, `tsr_typed.rs`, `scripts/law_runner.py`, `scripts/sr_old_new.py`, `logs/SR_OLD_NEW` (run1, run2, run3),
`logs/TSR_HARD1500`. 2026-11-05.

## Verdict

**The design is sound and the stream order is copied faithfully from the code, but the deciding run is not ready as
written.** Five things must change before the build starts:

1. **The run cannot show the fuller features working.** At 1,000,000 evaluations the law protocol lasts 11
   generations (1,270 of 1,270 seed-runs in `logs/SR_OLD_NEW`, the maximum is 11). With `fold_every` 20 and
   `snap_every` 20, fold, constant snap, share, beam and the finisher fired in **0 of 1,270** seed-runs; typed snap
   (every 10) fired in 131. Andrew's requirement (every feature working in the run) is not met by tests alone.
2. **The new arm changes two things at once.** Against old TSR it adds the order limit **and** changes a depth failure
   from "gene not expressed" to "node substituted". A loss or a win cannot be put on the order rule.
3. **The decision rests on about 15 passes an arm.** "At least as many" on the hardest ten is within noise; the rule
   needs the paired counts.
4. **The 1e-10 bar needs a code change and a record.** `law_runner.py` pins 1e-6 and refuses any other value, and a
   count at 1e-6 cannot be read from a run that stops only at 1e-10.
5. **The gene width figure is the generator's, not the engine's.** The engine's single gene is 104 words, not 101.

The gene-size drop is **not** a confound in the deciding run: old and new both run the single gene. The order
definition, the worked examples, the 20-law selection of the hardest solved ten, and the 15-minute estimate (on the
engine of 2026-10-20) check out.

## What was checked, question by question

### 1. Stream order and the legal programs

| check | result |
|---|---|
| 8.1's definition against `gen_tables.py` (`order`, `unary`, `binary`) | same: leaf (none, 1, 1); a unary gives (none, order, 1); a sum or product takes a same-family child's (b, ties), any other child's (order, 1); order = b + 1 when ties ≥ 2 |
| against the device (`typed_decode.wgsl` step 3, `N_BEST`/`N_TIES`/`N_ORDER`, lines 720-753) | same; the device's ties count is not capped at 2, the generator's is; only "2 or more" is read, so equal |
| the two worked examples | both correct (order 2, order 4) |
| depth rules (raising: sin cos tan tanh psqrt plog pexp abs pasin pacos; keeping: neg pow2 pow3 pinv) | same as `gen_tables.py` `UNARY` and `DEPTH_RAISING_OPS` |
| windows (sin cos tan tanh pow2 pow3 pinv, argument order ≤ 2) | same as the `windowed` flags |
| legal programs of `order_depth` and `order_depth_windows` | the same set: the typed tables are upper bounds with an exact typing for every legal tree, so "has a typing" equals "exact profile within the limits"; test 1 checks it |
| variants `single`, `depth`, `order` | **not defined** in 8.2; only the full rule set and the windows are |
| against today's run baseline (arm A) | a deliberate change: the order ceiling removes order-4 programs that arm A accepts. Correct to test, but the plan should say this is the change under test. The old hard rule's pair table and exp minimum are both absent, as in the typed tables |

Gaps in the data model as written: the input columns' properties are not stated (a `var` takes its variable's
properties, so the entry must give each column depth 0, order 1, fam 0, arm 1, ties 1); and a `PropertyRule` is one
expression for one property (`geneframe.rs:988`), so `(fam, arm, ties, order) = strahler.sum(in0, in1)` must be
written as one expression per property.

### 2. Substitution at decode instead of types

The set of legal programs is unchanged against the typed arm C, but how genes map to programs changes:

- **Many genes, one program.** A dropped arm stays in the gene but is not expressed. Mutations there change nothing
  (silent material inside the expressed region). Children whose program equals a parent's may still cost counted
  evaluations.
- **A one-sided bias.** The first input is always kept: `a - b` becomes `a`, `a / b` becomes `a`; a right arm is never
  kept. At initialisation the expressed programs shift towards smaller, left-heavy trees. In an 11-generation run the
  initial population carries much of the result.
- **Bloat.** The Karva gene has a fixed length, so programs cannot bloat; silent gene material costs nothing and is
  bounded by the head.
- **Against old TSR (arm A).** Today a depth-failing gene expresses nothing and rides along unscored; under
  substitution it is scored. That is the second change of verdict item 2.

The gene sizes: 101 words is right for the generator's sizing (head 33, tail 34, Dc 34 at N = 17), but the engine's
single gene is head 34, tail 35, Dc 35 = **104** words (`tsr_typed.rs:101`). 20,980 is arm C, which the run does not
include. Old and new both use the single gene, so size is not a confound. The comparison that isolates
"substitution against types" would be new against arm C; the plan leaves C out for time, which is acceptable if the
depth-only arm below is added.

### 3. The benchmark

| item | finding |
|---|---|
| source, seeds, budget | right: `logs/SR_OLD_NEW` run2, seeds 7001-7010, 1,000,000 evaluations; run1 380 of 400 and run2 101 of 600 (new arm) verified |
| the ten hardest solved | right: exactly the ten laws at 1 and 2 of 10 in run2's new arm, 15 of 100 |
| the ten unsolved | 0 of 10 in both arms, but **no selection rule** is given (33 laws qualify in run2) |
| "Four of these are the Lorentz and Gaussian laws the order rule lost" | **wrong**: three of the ten are among the six lost laws (I_15_10, II_13_34, I_12_11); two are Lorentz; I_12_11 is neither; none is Gaussian. The Gaussian laws I_6_2 (run2) and I_6_2b (run3) are unsolved in both arms and not in the list |
| run3 "no table" | its `result.json` files exist for 28 laws in both arms; `sr_old_new.py --aggregate` rebuilds the table. They do not change the hardest ten (feynman_test_5 is 4 of 10) |
| the bar | running at 1e-10 is right (Andrew's law bar). `law_runner.py` pins `STOP_ONE_MINUS_R2 = "1e-6"` (line 62), refuses any other value (`FIXED_BAR`, lines 217-220, 353) and its self-test assumes it; P3 must change it, for every later law run. The hardest ten were chosen at 1e-6, so at 1e-10 some may be 0 in both arms |
| time | 20-25 s a law and arm for 10 seeds is what the 2026-10-20 sweep logs show (about 1.1 s a seed). Since then 54 phylu commits touched `engine.rs`, `typed_decode.wgsl` or `law_runner.py`, and making fold and snap fire adds time. Credible, but unmeasured on today's engine |
| fairness | same binary, seeds, budget and settings: fair. Not fair in effect: features that never fire are equal but absent |
| evowatch | `law_runner.py` writes one stream per seed folder unless `--stream`/`PHYLU_STREAM` is set, and `sr_old_new.py` sets neither, so `evowatch OUT` finds no `OUT/stream.jsonl` |

### 4. Tasks

- P1 is the risk: renumbering the expanded tree after a substitution inside an 832-line shader, a host twin, and
  keeping the old reject path unchanged for the old arm. 2 days is tight; 3 is honest.
- P2 checks that each rewrite "decodes with no substitution", which needs P1's host decoder (`typed_decode.rs`). Either
  T3 fixes that function's signature or P2 follows P1's host half.
- T3 includes "phylu arms build", which is phylu work, and "contract width 101", which the engine's 104 will
  contradict.
- T1 and T2 edit `geneframe.rs` and `typed_environment.rs`, which the Futhark tasks F4 and F5 also use. Nothing waits on
  section 7, but master 6's tests must stay green and the edits must not overlap uncommitted Futhark work.
- "Both repos green after every task" is only in L1. P1 and P2 in parallel means two cold phylu builds at once, which
  the house rules forbid.
- Missing: a test that the old arm's decode is unchanged byte for byte; a calibration before R1; the result write-up.

### 5. Waste and side investigations

None of size. Three small corrections: test 3 compares the GPU node array and hash with fuller's decoder, which does
not produce that array (compare the GPU with `typed_decode.rs` word for word, and `typed_decode.rs` with fuller on the
tree and the substitution positions only, without building a converter); share-equivalence never fires in TSR law
runs (keep its test, since Andrew named it, and nothing more); 8.2 reason (2) cites `TSR_HARD1500`, which was arm C
(types, no substitution), and 7 of its 26 final lines, not 16, contain `(0*0)` (16 are `plain_fallback`).

## Recommendations (each with the plan text to change)

**R1. Make every feature fire in the run, in both arms, and prove it.** In 8.7 requirement (3), replace "Every fuller
feature of 8.6 item 5 on the new model, by task P2, before launch." with: "Every fuller feature of 8.6 item 5 on the new
model, by task P2, before launch, **and firing in the run**: at 1,000,000 evaluations a seed runs 11 generations
(`logs/SR_OLD_NEW`, 1,270 of 1,270), where fold, constant snap, share, beam and the finisher fired in 0 seed-runs. Both
arms set fold, constant snap and typed snap every 3 generations and keep the tower and the finisher on (P3 adds any
switch the engine lacks); the table adds, per arm, the evaluations by source from `result.json` (fold, snap,
typed_snap, share, beam, post_search), and a source at 0 in either arm is our software has a bug."

**R2. Add a depth-only arm to separate the two changes.** In 8.7 requirement (5), replace "two arms: old TSR
(`EVOLVE_TYPED_DEPTH=2`, single) and new TSR (`EVOLVE_SR_KINGDOM=props`, order and depth)" with "three arms: old TSR
(`EVOLVE_TYPED_DEPTH=2`, single: a depth failure is not expressed); new TSR, depth only (`props` with the order
requirement off: a depth failure is substituted); new TSR (`props`, order and depth). Old against depth-only measures
substitution; depth-only against full measures the order limit." Update the time to "20 × 3 × 22 s ≈ 22 minutes".

**R3. Define all five variants as switches.** In 8.2's Requirement row, add: "Each requirement group is a switch
(depth, order ceiling, windows), so `props` writes all five of today's variants: single (none), depth, order,
order_depth, order_depth_windows." In 8.6 item 1, replace "exactly when it has a typing in `order_depth`, and in
`order_depth_windows` with the windows on" with "exactly when it has a typing in the matching typed variant, for all
five variants (single: never substituted)".

**R4. Complete the property rows.** In 8.2, add to the DataType row: "each input column, an entry parameter of `F`,
has depth 0, order 1, fam 0, arm 1, ties 1". In the PropertyRule row, replace "`(fam, arm, ties, order) =
strahler.sum(in0, in1)`" with "`fam = 1` (sums) or `2` (products), and one expression each for `arm`, `ties` and
`order`, a new `PropertyExpr::Strahler { family, part }` over both inputs' four properties (`geneframe.rs:988`: one
expression per property)".

**R5. Measure what substitution does to the search.** In 8.7's table, replace the column "genes with a substitution,
new" with "genes with a substitution; nodes dropped per substituted gene; children whose program equals a parent's
(engine `gene_hash`) and how many of them cost a counted evaluation; per arm". In 8.2 after the three reasons, add: "The
first input is always kept, so `a − b` and `a / b` become `a`; the expressed programs at initialisation shift towards
smaller, left-heavy trees. The columns of 8.7 measure it."

**R6. Correct the gene width and say what it is compared with.** In 8.4, replace the new TSR row's "101 | 303 | 2.4
MB" with the engine's single layout, "104 (head 34, tail 35, Dc 35, `tsr_typed.rs:101`) | 312 | 2.5 MB", and add under
the table: "Old TSR (arm A) uses the same single gene, so width is not a difference in the deciding run; the saving is
against arm C, which is not run." In 8.5 and T3, replace "`tsr_relations` gives width 101" with "the run card's gene
width is equal in both arms".

**R7. State the selection of the unsolved ten, and fix the Lorentz/Gaussian claim.** In 8.7, replace "Four of these
are the Lorentz and Gaussian laws the order rule lost." with: "Chosen in this order from the laws at 0 of 10 in both
arms of run2 and run3 (run3's table rebuilt with `sr_old_new.py --aggregate`): the lost laws I_15_10, II_13_34
(Lorentz) and I_12_11; the Gaussians I_6_2 and I_6_2b; then by position in `order133.txt`." Replace two of the
remaining seven with I_6_2 and I_6_2b. Add an eleventh law outside the totals: "feynman_I_6_2a (6 of 10 in run1), the
sixth lost law, which tests whether the order limit still loses the Gaussian."

**R8. The bar: change the runner and record the second count.** In 8.7, replace "Both arms run at the law bar of 1e-10
(Andrew, 2026-10-08), and the counts at 1e-6 are reported beside them" with: "Both arms run at the law bar of 1e-10
(Andrew, 2026-10-08): P3 changes `STOP_ONE_MINUS_R2` in `law_runner.py` (line 62, its gate note and its self-test) for
every later law run. Each seed records the first counted evaluation at which log10 p ≤ −19 and 1 − R² ≤ 1e-6 held, so
the count at 1e-6 is reported beside the decision count. The hardest ten were chosen at 1e-6; a law at 0 in both arms at
1e-10 is reported, not replaced."

**R9. A decision rule that the noise cannot decide.** In 8.7, replace "new TSR is kept if it finds at least as many laws
(seed-runs at the bar) over the 20 as old TSR, paired by law and seed" with: "Paired by law and seed: n10 = seed-runs
only new passes, n01 = only old passes, with the exact two-sided McNemar p. New TSR is kept if n10 ≥ n01 on the 20 and
no law of the hardest ten loses more than 2 of 10; any law of the unsolved ten is reported first."

**R10. Calibrate, then run, and make it watchable.** In 8.7 add before the table: "R1 starts with one law
(feynman_I_10_7), all arms, 10 seeds, on today's binary; the projected total must be under 45 minutes or R1 stops and
says why. `tsr_decide.sh` sets `PHYLU_STREAM=OUT/stream.jsonl` (law_runner otherwise writes one stream per seed folder
and `evowatch OUT` finds none)." In R1's row, replace "about 15 minutes" with "about 22 minutes, calibrated first".

**R11. Tasks: honest sizes, the right order, green after each.** In 8.8: P1 size 2 → 3, and add "the old arm's decode
unchanged byte for byte on 20,000 random genes, GPU and host"; P2 "after: T3, P1 (host decoder half)"; move "phylu
arms build" from T3 to P3; add to every task's line "both repos' full suites and clippy green before the commit; phylu
builds one at a time (P1 and P2 never build together)"; add to T1 and T2 "master 6's tests unchanged; no edit to a file
with uncommitted Futhark work". Recompute the critical path (T1, T2, T3, P1, P3, L1, R1: about 7.1 days with P1 at 3; P2 runs beside P1's device
half).

**R12. Small text fixes.** In 8.6 item 3, replace "against `typed_decode.rs` and fuller's decoder" with "against
`typed_decode.rs` word for word; `typed_decode.rs` against fuller's decoder on the tree and the substitution
positions". In 8.2 reason (2), replace "TSR_HARD1500 already ends 16 of 26 laws in collapsed forms such as `(0*0)`"
with "TSR_HARD1500 (arm C, types, no substitution) ends 16 of 26 laws on a plain fallback, 7 of them with `(0*0)`".
