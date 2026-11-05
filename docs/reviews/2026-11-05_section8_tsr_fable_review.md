# Fable review: plan section 8, Typed Symbolic Regression (TSR) on the new data model

Reviewed: `docs/PLAN_futhark_kingdom.md` section 8 at fuller 5c28c5e2 (the advisor's review already applied), against
fuller `kingdoms/tsr/gen_tables.py` and `README.md`, `src/geneframe.rs` (1640-1670, the requirement checks), the design
`docs/DESIGN_futhark_types_and_properties.md`, and phylu `src/evolve/order.rs` (`stream_order`), `typed_decode.wgsl`
(step 3 and the order-ceiling block, lines 720-800), `tsr_typed.rs`, `scripts/law_runner.py`, and the 2,572 seed
streams of `logs/SR_OLD_NEW` (1,290 old, 1,282 new). My view was formed before reading the advisor's review; the
agreements and differences are at the end. No adversarial review exists at the time of writing. 2026-11-05.

## Verdict

**Build it, run it, and read the result as a test of substitution first and the order limit second.** The design is
sound typed GEP: the set of programs the search can express is exactly the typed set (test 1 makes that a contract),
the gene stays the 104-word single gene, and the rule lives in a generated, checked table instead of a 208× wider
type table. What changes is the word "unrepresentable". Under properties an illegal shape is **representable in the
gene and inexpressible in the program**: the gene can hold `sin(sin(sin x))` or an order-4 product, and the decoder
renders a legal program from it. That is the same relationship GEP already has between a gene's tail and its expressed
tree, so it is not a weakening of typed GEP; it is a weakening of the README's slogan, and the README should say
"inexpressible" after this lands.

Three things I would change or add before the run, none of which blocks the build:

1. **The instrument for requirement (3) is unproven.** The plan says a source at 0 in `result.json`'s evaluations by
   source is a bug. In `logs/SR_OLD_NEW` the `fold` counter is 0 in all 2,572 streams, yet 632 "fold:" notes appear
   (505 at generation 11). Those notes are the end-of-run tidy of the reported model (`engine.rs:8290`,
   `report_tidy`), not the fold operator, so the counter is right today; but it means a reader of the stream cannot
   tell the operator from the tidy by the word "fold". P3 should check, on the calibration law, that the counters move
   when `fold_every` is 3, before the rule "0 is a bug" is trusted.
2. **A tie should not adopt the order limit.** The rule "new is kept if n10 ≥ n01" gives a tie to the new arm. The
   order limit's prior is a loss (84.69 % against 85.61 % as a decode rule). Depth-only against full isolates the
   order limit; if that pair is a tie, the order requirement adds a rule and a bias for no laws found, and should stay
   a switch that is off. Substitution (old against depth-only) can be adopted on a tie, because it costs nothing and
   scores genes that today ride along unscored.
3. **Say what the first-input bias does to `sub` and `pdiv`.** A failing `a − b` becomes `a`, and `a / b` becomes
   `a`: the operator is not weakened, it is removed. For a sum or product that fails on ties (two arms at order 3),
   dropping either arm restores legality; the first is the cheapest rule, not the only one. The plan measures the
   bias (its last column), which is the right call for the deciding run; if the depth-only arm wins and the full arm
   loses, the first thing to try is a substitution that drops the *later* tying arm and keeps the node.

## (1) Is moving order and depth into properties sound typed GEP?

Yes. The properties are derived bottom-up and are local (the profile `(family, b, ties)` is all a parent needs), which
is what makes them a type in disguise: `gen_tables.py`'s `binary` and `stream_order` in `order.rs` are the same
function, and the device already computes `N_BEST`/`N_TIES`/`N_ORDER` in step 3. Monotonicity holds: dropping an arm
never raises a node's order and never raises its depth, so the induction in 8.2 (every input has passed its own
requirements, so a substituted node is legal) is right, provided the substituted node reports its **input's** family
and profile to its parent (the plan says so; it is the detail that bites, because a sum whose child was a product and
is now a leaf flattens differently).

Does it keep TSR's central claim? The claim as written in the README is "illegal shapes stop being penalised and become
unrepresentable". Under properties they are still not penalised and still never scored: no illegal program is ever
evaluated. What becomes representable is the *gene*, not the *program*. In arm C (types), initialisation and mutation
draw only legal rows, so every gene cell is live. Under properties a cell can be silent: a dropped arm sits in the
expressed region of the gene and mutations there change nothing. That is neutral genetic material, of three kinds:

- **Neutral in the GP sense**: several genes, one program. Neutral networks are, on the evidence of the GP literature
  and GEP's own tail, mildly helpful or harmless; they let the search move without changing fitness.
- **Not bloat**: the Karva gene has fixed length, so the program cannot grow past the head. Silent material is bounded.
- **A biased map**: the one real cost. The expressed distribution at initialisation shifts to smaller, left-heavy
  trees, and `sub`/`pdiv` lose their right operand when they fail. In an 11-generation run (see below) the initial
  distribution is much of the result, so this bias is not a footnote.

The design doc already allowed for this ("its adapter is not a wrap but a substitution, to be decided", line 208). It is
decided well: a constant substitute would erase subtrees (the plan's TSR_HARD1500 evidence), and rejection (today's arm
A) wastes the gene. Substitution is the right adapter; the choice of *which* input is the open question, and the plan
measures it rather than argues it, which is the right posture.

## (2) Is the deciding run fair and decisive?

**Fair, yes.** Same binary, same seeds (7001-7010, the development seeds, never SRBench's), same budget (1,000,000
counted evaluations, Andrew's stop), same features firing at the same cadence in all three arms, the new arms run
first on each law. The 20 laws come from the recorded tables, with the selection order stated. The three arms isolate
the two changes. The 1e-10 bar is the law bar, and recording the first evaluation at 1e-6 beside it keeps the old
tables readable.

**Decisive, only for a large effect.** The hardest ten gave 15 passes in 100 at 1e-6 in the recorded run; at 1e-10
some of those laws will go to 0 in every arm, and the unsolved ten are expected near 0. McNemar on 200 pairs with
perhaps 10 to 20 discordant pairs reaches p < 0.05 only at something like 7 to 1 or 9 to 2. So the run can show "the
order limit loses several laws" or "substitution finds laws old TSR did not"; it cannot show a one-or-two-law
difference, and the write-up must say so rather than call a 9 to 7 a win. Two further cautions: the hardest ten were
*chosen* by their pass counts on these same seeds, so re-running them regresses toward the mean in every arm (the pool
of passes will likely be below 15); and 1,000,000 evaluations is **11 generations** (I counted: final generation
median 11, maximum 11, median 1,078,020 evaluations). The test is a test of the protocol Andrew fixed, which is right,
but the result says nothing about what the order limit does over 100 generations.

**Is 22 minutes enough?** The clock is not the limit; the pairs are. Twenty-two minutes (measured on the engine of
2026-10-20, with features that did not fire) is a floor, and the calibration law with its 45-minute stop is the right
guard. If the run finishes and the discordant pairs are under about ten, the honest next step is the same 20 laws on
seeds 7011-7020, not a different design.

## (3) Fold, snap, share, beam and the finisher fired in 0 of 1,270

Confirmed from the streams, and sharper: in all 2,572 seed-runs of `logs/SR_OLD_NEW` (both arms) the evaluations by
source are `main` only, plus `typed_snap` in 241 (the runs that reached generation 10). Nothing else moved. The 632
"fold:" notes in the streams are the end-of-run tidy of the reported model, not the fold operator.

What it means for the past: every TSR law rate recorded (SR_OLD_NEW, and the production runs on the same cadence) is
**the typed table plus plain GEP variation, selection, the pump and the islands**, with typed snap once in the runs
that lasted. That is a clean measurement of the representation, and it is still the baseline, so nothing in the past
tables is wrong; but it means the recorded "TSR as run today" did not use fuller at all beyond typed snap, and the
earlier order-rule loss (84.69 % against 85.61 %) was measured the same way.

What it means for this test: the deciding run is also the **first** TSR law run with fold, snap, share, beam and the
finisher firing. Two consequences. The recorded 380 of 400 and 101 of 600 are no longer the comparison; the old arm is
re-run, and the plan should quote them only as history. And the three arms differ in exactly one thing each only if
P2's promise holds: every rewrite is offered only when it decodes with no substitution. If a fold or snap of a
substituted gene is accepted and then re-substituted, the arms diverge in a way the table cannot see. Test 5 covers
this; it is the test I would read first in L1.

## (4) The single biggest risk and its guard

**P1: the device decoder after a substitution.** Today a failing gene is a flag (`gene_failed`, lines 762-800 of the
shader) and the node array is never written. Substitution means a *second* level numbering of step 2 with the
substituted node and its other arms removed, a `fam` word per node, and a parent whose flattening changes because its
child's family changed, all in an 832-line shader with a host twin, while the old arm's path stays byte for byte. A
silent disagreement between the GPU and `typed_decode.rs` would not crash; it would score one program and report
another, and the deciding run would still produce a table.

Guard, already in the plan and to be kept as a gate, not a task: test 3 (20,000 random genes a variant, node array,
length, depth and hash word for word, GPU against host, and host against fuller on the tree and the substitution
positions) and test 4 (a gene that passes arm A's check gives today's node array) must be green before L1 lets R1
start. I would add one thing: run test 3's GPU half on genes *forced* to substitute (draw until 20,000 have at least
one), because random genes at head 34 may substitute rarely and the test would then mostly re-check the old path.

## (5) Against the advisor's review

The advisor's twelve recommendations are applied in the plan text I reviewed, and I agree with all of them. Where I
add or differ:

- **Agree, and verified**: 0 of 1,270 (I count 0 of 2,572 across both arms, `typed_snap` in 241); 11 generations;
  the gene width of 104 (`tsr_typed.rs`); the three-arm design; McNemar on pairs; the 1e-10 bar with the 1e-6 count
  recorded; `PHYLU_STREAM` for evowatch; the first-input bias measured in the table.
- **Add**: the "fold:" notes in the streams are the final tidy, not the operator, so P3 should prove the counters move
  on the calibration law before "0 is a bug" is a rule. The advisor's rule would have flagged nothing wrongly today,
  but a reader of the stream would be misled.
- **Differ on the tie**: the advisor's R9 keeps new TSR at n10 ≥ n01. I would adopt substitution on a tie and adopt the
  order limit only on a win in the depth-only-against-full pair, because the order limit's prior is a measured loss
  and it adds a bias; a tie means it bought nothing.
- **Add**: the decision's power. The advisor says 15 passes an arm is within noise and asks for pairs; I would also
  state in the write-up the discordant count the run reached and name seeds 7011-7020 as the next step if it is under
  about ten. A plan whose test cannot decide should say so before the run, not after.
- **Add**: test 3 on genes forced to substitute, and the README's "unrepresentable" becoming "inexpressible".
- **Agree on P1 at 3 days**, and on P2 following P1's host half. I would not shorten either.

No adversarial review was present to compare with.
