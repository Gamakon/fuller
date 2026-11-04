# Code review: phylu 24af1068, plan task F5 (`FutharkEvaluator::fresh`)

**Verdict: accept, with fixes before P4.** The library code does what the F5 row asks: the refill order per regime is
right, the refill is capped at the request's rows, no individual repeats a text or a tree or offers a text scored
before, the fold goes through the engine's own fold beat and landing and is counted as the engine's fold, and no
genetic operation or fitness moved into fuller. There is no blocking bug in what runs today. But histogram's regime P
generation 0 is one program, not 24, and the plan's R0 anchor and its landing table still assume 24 (finding 1); the
histogram test checks one gene (finding 2); and the engine's new per-pair fold code was not run by any test in this
commit (finding 3).

Reviewed: the whole diff (`src/evolve/engine.rs`, `src/evolve/external.rs`, `src/evolve/futhark_typed.rs`,
`src/futhark_eval/evaluator.rs`, `src/futhark_eval/evaluator/start.rs`, `.../start/tests.rs`), every line; the old
runner at fuller `pre-5b:examples/futhark_evolve.rs` lines 229-340, 746-751, 1172-1265, 1541-1622, 1671; the plan's
F5 row, 5a.2, 5a.4 and the F5 status paragraph (fuller 9b6f158e). Not built or run (read only).

## What was checked and holds

| check | result |
|---|---|
| regime P start reproduces the runner's draw (pad from `GeneRng::new(seed)`, `variants(..,3,32,2*rows)`, fold of all, `truncate(rows)`) | yes, `start.rs:236-287` against old 1209-1247 |
| R and C starts on their own streams (`seed ^ REGIME_C_TAG` for C) | yes, `start.rs:215`, old 1174 |
| alternatives and mutants starts | the old functions line for line (`start.rs:293-397` against old 229-340) |
| refill order: variants (at most 4, only when the best changed), then P mutants of the champion island's best (`1 + below(2)` cells, the old `m_of` for P), R variants only, C none | yes, `start.rs:401-470`; the best is the champion island's (`engine.rs` `pair_best_row`) |
| refill capped at the request's rows | yes, `start.rs:418, 450`, and `fresh` errors above the cap (`start.rs:182`) |
| no repeat, no text scored before | yes, `Offered` (text and tree) seeded with the best or the winners, plus `quality_cache` |
| fold forms land on their winner and count as the engine's fold | yes: `forms_by_winner` checks each form's first parent is a winner and no winner twice; `fold_external` lands through `fold_landing`, sets `EvalSource::Fold`, adds to `fold.counts` |
| fold forms scored before the refill of the same generation | yes, the fold beat evaluates its rows before the pump (`engine.rs` fold beat, then pump at ~9871) |
| zero warnings rules | no `#[allow]`, no `_` names in the diff |

## Findings

1. **Histogram's P generation 0 is the reference program alone, and the plan still assumes 24 (high, for P4 and R0).**
   `start.rs:283` truncates to the members found; for histogram that is 1 (the commit and the status say so). The plan
   was not updated where it depends on 24: the landing table (`PLAN_futhark_kingdom.md:632`, "the reference program
   and its first 15 variants land in the intake and the next 8 in the champion island") and R0 in 5a.4 (line 1107,
   "both runners scoring the same 24 programs"). Failure: in P4 the 1 individual goes in through `External.seed` and
   the other rows of the population are the device's random draw, in regime P — the 47,999-word random histogram genes
   section H calls a wasted evaluation. The old runner did not do that: its generation 1 was 23 children of the one
   program. And R0 on histogram compares one program, so it proves little. Fix: in P, when the members are fewer than
   the rows, fill the rest with `pump_mutants` of the start program, named as a departure from the runner, in the
   start message and on the card. Run R0 on pathfinder (24 programs), or on both problems. Correct line 632 and 5a.4.

2. **The histogram test checks one gene, and its doc says 24 (medium).** `tests.rs:100-102` says "the same 24
   programs in the same order"; for histogram the comparison at `tests.rs:121-133` is one gene, the reference written
   onto the first pad. Pathfinder (`tests.rs:110-113`) is the real comparison. Also, the comparison is against a
   transcription inside the test that calls the same functions (`variants`, `write_back`, `fold_saturated`,
   `random_gene` of phylu's moved `Genetics`), not against the frozen pre-5b binary's output. And the new start drops
   repeats (`offered.take`, `start.rs:252, 274`), where the runner kept them (old 1217-1244). This follows the house
   rule, but nowhere is it stated as a departure, and the test only covers problems where no repeat occurs. Failure: on a
   problem whose variants coincide after write-back (not histogram, not pathfinder), phylu's generation 0 differs from
   the runner's, and no test or note says so. Fix: correct the doc comment; assert `start.len() == 1` for histogram
   explicitly; state the repeat filter in the module header and the F5 status; let R0 (frozen binary, ledger bit for
   bit) be the real cross-check.

3. **The engine's per-pair fold request is not tested by anything that ran (medium).** `fold_external`
   (`engine.rs`, the hunk at 5694-5724) is covered only by the GPU-gated
   `the_external_hook_runs_ten_generations_on_two_pairs`, and the status says the GPU tests were not run. The CPU fold
   test (`tests.rs:218-236`) asserts `forms.len() <= winners`, so it passes with zero forms, and it never goes through
   `fold_landing`. Failure: a grouping or landing bug in the changed engine code would reach P4 unseen. Fix: assert at
   least one form for pathfinder's 6 winners (the test prints the count, so the number is known). Run `p2a_mock.sh`
   before P4 starts.

4. **Failed write-backs are dropped without a word (medium; Rules.md: no hiding bugs).** `start.rs:272` (P start fold
   forms), `start.rs:421` (refill variants) and `start.rs:503` (fold forms) do `let Ok(g) = write_back(..) else
   { continue }`. A form or variant the e-graph made from a gene that fits, and that will not write back onto that gene,
   is a bug in our software (an infeasible gene is a kingdom bug). The old runner hid it the same way; moving the code
   was the time to stop. Failure: the fold or the pump quietly offers fewer individuals, and no count shows why. Fix:
   count the failures and name them in the message and the mutation event; or make them an error, as an expression
   failure already is.

5. **The pump and the cross step draw the same refill mutants (low).** `start.rs:445` keys the stream on the seed, the
   generation and the pair only. When the pump and the cross step share a beat, both ask `PumpRefill` for the same pair
   in the same generation. The pump's rows are scored before the cross step, so the cross step's refill draws the
   pump's mutants again, throws them away as scored, then goes on. That wastes tries, and its mutants are correlated
   with the pump's. Fix: the request says which beat it is (or the engine passes its `CROSS_KEY`), and the tag
   includes it.

6. **The alternatives claim is not what the test asserts, and the consequence is not drawn (low).** The status says
   "histogram's 577 alternatives all give the published output". `tests.rs:295` asserts only `1 <= kept < 24`; it
   checks neither that exactly one is kept nor that its output is the reference's. The consequence: histogram has no
   alternatives start. At 24 rows it is an error (as in the runner). At its own size, its one member computes the
   published output (leave-one-out is by tree distance, not by output), so it is not a discovery start. On the way, it
   scores every alternative that passes the distance bar (up to 577 evaluations), charged only in
   `evaluations_total`. Fix: assert `kept == 1` and that its behaviour equals the reference's, if that is the claim.
   Say in the plan that histogram's R and C runs use `Mutants` or `Drawn`.

7. **The fresh state is not part of a checkpoint (low, for F7).** `FreshState::last_refilled` (`start.rs:107`) is lost
   on a resume, so the first refill after it offers the best's variants again. Only the quality cache, if it is
   restored, keeps them from being scored twice. Fix: carry it in the evaluator's state at F7.

8. **The refill mutants reimplement the `mutants` loop (note, no fix).** 5a.2 says the refill mutants are "built by
   `futhark_rings::mutants`". `pump_mutants` (`start.rs:444-470`) is its own copy of that draw, because `mutants` has
   no "scored before" filter. The copy is faithful (uniform member draw, text-and-tree filter, `200 * rows` tries, an
   error rather than a random fill). Record it in the plan, or add the filter to `mutants` and call it.

## Left for P4 (beyond the status paragraph's two items)

- The engine never asks `FreshPurpose::Start` (the GPU mock panics on it), and no runner calls `set_fresh` or builds an
  `ExternalFresh` over `FutharkEvaluator::fresh`: everything in `start.rs` is library code until P4 wires it.
- Through `External.seed`, generation 0's origins, parent rows and mutation events are lost, and `fresh_cost` (the P
  start's e-graph and fold seconds) is reported nowhere.
- Finding 1's fill of the P start, and R0 moved to pathfinder.
