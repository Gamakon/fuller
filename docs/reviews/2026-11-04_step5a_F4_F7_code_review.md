# Step 5a F4 and F7 code review: the report of a generation's best, the evaluator's state and restore

Reviewed 2026-11-04: phylu ffc4bd98 (F4, `src/futhark_eval/evaluator/report.rs`) and 4ecc9d89 (F7,
`src/futhark_eval/evaluator/state.rs`), against fuller `docs/PLAN_futhark_kingdom.md` (rows F4 and F7, 5a.2, section
5b, the status paragraphs of fuller 87521cf8), both CLAUDE.md files, `qdrant/docs/Rules.md` and Andrew's rulings (the
HFF angle is the fitness, untouched; the percentile F(θ) = I_{sin²θ}((d−1)/2, ½) is reported with its d; the test set
never enters the fitness inputs). Read only, no build. Line numbers are at 4ecc9d89.

## Verdict

**Accept with fixes.** No fitness value changes. The report's percentile is the hff library's function at the right d
for every scheme. The test set stays out of `inputs` on every problem. The report scores and counts nothing. The state
round trip is exact for every full timing slot, the champion and the refill's memory, and a failed restore changes
nothing. Fix findings 1 and 2 before P4 builds a resume on F7. The other findings are smaller.

## Checked and correct

| Item | Result |
|---|---|
| The percentile is the hff library's | `AngleReport` and `report` call `fuller::fitness::hff_percentile`, which is `hff_core::higd::cdf_beta_correction(θ, d)` with d at least 2 and θ in [0, π/2] (fuller `src/fitness.rs:105`). |
| d for every scheme | `report.rs:233` takes d from `row_score`, the length of the vector the scheme builds (`evaluator.rs`, `row_score`). Stage 1: the quality group, d = Q. A flat stage 2 and the flat k-means schemes (`futhark-kmeans-better`, the retired `futhark-kmeans-qqs`): `flat_vector(q, s, m)`, d = Q·m + S. `qqs` takes TrueNorth over that same vector (fuller `src/fitness.rs:654`), so the percentile is taken at the scheme's own angle. The retired two-level schemes (gated, ungated, the gated k-means scheme): the vector is `flat_vector(q, s, 1)`, its angle is TrueNorth over that vector and d = Q + S, reported beside the scheme's number and never instead of it. The five-input k-means scheme: the quality group alone. `vector_names` gives one name to each column in every branch, the m copies of a flat scheme included. |
| The group angles | `AngleReport::over` (`report.rs:44`) takes each group alone at its own d, and gives None for a group of fewer than 2 inputs. |
| The report scores nothing | `render_gene` is the old inline code moved out unchanged. `cached_eval` and `oversized_worst` read the caches and the scheme's function only. `assert_report` checks that `evaluations`, `scored`, `hits` and `modules` are unchanged on every call it makes. A best that was never scored is an error naming it. |
| `staged_eval`, `converged_eval`, `cached_eval` | They do the same arithmetic as the code they replace: the same `r/(1+r)` with an untimed slot as 1, the same median, the same `staged_weighted` and `converged_fitness` calls. The k-means cache-hit path keeps its error for a missing speed group. |
| The test set out of `inputs` | `Quality::of` puts a case that is not in fitness into `test_raw` and `test_inputs` only, and `inputs` and `raw` are built exactly as before. `exact_test` is counted as before, and the test cases were already run in the quality program for it, so no new run is added. k-means's `evaluate_converged` iterates over the fitness indices only. The test replaces the reference program's test-case outputs with zeros and gets `inputs` and `raw` bit for bit equal, on every answer-scored problem. It also checks that at least two problems have a test set. |
| The all-zero flag (`Quality::zero`) | It is computed from the same outputs, at construction (`zero_outputs`, `evaluator.rs:308`; in the k-means literal it is evaluated before `outputs` is moved). `outputs` is assigned once (`evaluator.rs:345`) and never cleared. The flag is read only by the ledger column `all_zero` and `RowScore::all_zero`, so it is not a fitness input. The worst and oversized literals give false, as the empty outputs did before. No fitness changes, and the golden ledger tests pin the column. |
| Exactness of the state | Every `f64` is stored as 16 hex digits of its bits, so infinity survives (the test puts an infinite slot in). `u128` values are stored as decimal strings. The sorted vectors make the JSON deterministic. Two tests check `state()` after `restore` against the saved state. |
| The k-means champion and the refill | `champion` is rebuilt from its text with `build` and `program`. `last_tested` and `champion_tested` are restored, and the k-means test shows that no test case runs again after a restore. `refilled` is restored. |
| Restore is all or nothing | `restore` parses and checks everything, including the champion's tree, before line 250, and assigns only after that. The test checks a wrong format, wrong pairs and a wrong scheme, and a restore before the controls, and the stage is left unchanged. |
| Restore order (new, controls, restore) | Restore overwrites everything the second `controls` writes: the timing slots (in regime P, the in-run A/A), the quality cache (the k-means A/A row) and the counts (the k-means A/A's budgeted evaluation). The controls rebuild what is not saved: the k-means seed runs and work caps. The work caps come from the interpreter's work count, so they are deterministic. Every k-means input is measured against the seed's run interleaved with the candidate's (the `ConvergedJudge` doc), so the re-run seed timing does not change the scale of the cached inputs. The order is sound for fitness. |
| House rules | No `_`-prefixed names and no `#[allow]` in either commit. |

## Findings

1. **The configuration check misses the backend, the eval mode, the regime, the cases and the quality inputs.**
   `state.rs:212-216` compares problem, scheme, quality weight, cliff, pairs and program size only. Failure: P4
   resumes a checkpoint into a run whose cases differ. Examples: the k-means sample is re-cut, `gen_tables`
   re-emitted the view, or the quality parts changed. Every cached program is then a hit, carrying the inputs of the
   other cases, and nothing reports an error. The same happens for a c-backend state restored into a GPU run: the
   timing slots and cached qualities come from another device. Fix: add `backend`, `mode`, the regime, the cases'
   fingerprint and `quality_fingerprint_of(parts)` to `Saved` and to `want`. These are the same fingerprints
   `gate_passed` checks at `evaluator.rs:1536-1540`. Add the kingdom table's version too, since the champion's text
   is rebuilt against it. The regime is not held by the evaluator today, so keep it when `controls` runs. Test each
   field the way pairs and scheme are tested.

2. **Two silent fallbacks in the state, against "no failovers, no hiding bugs".**
   - (a) `saved_slots` (`state.rs:274-289`) reads the slots back from `TimingSlots`'s own serde form, through its
     private field name `by_text`. If that form changes, `let Some(..) else { return Vec::new() }` saves no slots.
     After a resume every stage-2 program would then be timed again, giving new ratios and a different fitness, and
     no error would be raised.
   - (b) A partly filled program's ratios go through JSON decimals, with `as_f64().unwrap_or(INFINITY)`. Any `null`,
     a NaN included, becomes infinity. The median ratio in the report then changes from NaN to infinity.
   - (c) `state()` ends with `unwrap_or(serde_json::Value::Null)` (`state.rs:197`), so a failed serialisation writes
     a null state into the checkpoint.

   Fix: add a `TimingSlots::entries()` accessor to fuller (a fuller commit, tested there). Hex every ratio, full or
   partial. Make `state()` return `Result<serde_json::Value, String>`.

3. **The partial-slot path is untested.** Both round-trip tests fill every slot. The path a stopped run checkpoints
   through is a program with some of its slots filled, and that is the path behind finding 2(b). Fix: add a program
   with 3 of its pairs timed, one of them infinite, to the histogram test, and check after the restore that
   `missing` is the same and the ratios are equal bit for bit.

4. **`FreshState::refill_draws` and `FreshState::cost` are not saved.** The status paragraph says the refill's draw is
   keyed by seed, generation and pair, "so nothing else needs saving". That holds only when the checkpoint is written
   after every fresh request of its generation. `refill_draws` is a counter of requests per (generation, pair)
   (`start.rs:554-556`). Failure: the engine writes the checkpoint between two refills of one (generation, pair). The
   resumed counter starts at 0, so the resumed refill draws again what the first refill drew. The P start's cost
   (`FreshCost`) is lost after a resume, so F6's final report would leave it out. Fix: save both (they are small),
   or make P4's resume test assert where the checkpoint falls relative to the pump.

5. **The resumed `controls` are a second, real measurement.** This does not affect fitness: restore overwrites
   everything they write. But on the GPU they repeat the A/A sets, and in k-means regime P they repeat a compile and
   the reference program's runs, minutes of GPU time. The `Controls` they return describe the resumed process, while
   the slots hold the original A/A. Fix (P4): the run card and the stream record a resume, and show the A/A of the
   original run's card. The resumed controls' A/A is labelled as a re-check, never written as the run's yardstick.

6. **The retired k-means schemes' report branches are not exercised.** `report.rs:241-246` and `vector_names` have
   separate branches for `futhark-kmeans-qqs`, the gated k-means scheme and the five-input scheme. The tests run only
   the default `futhark-kmeans-better` (in the k-means seam test) and the answer-scored schemes. Reading the code, the
   branches are correct: the gated scheme gives quality and speed once with d = Q + S, the five-input scheme gives
   quality only. But nothing pins them. Fix: a host-only test that inserts a synthetic `Quality` and speed group into
   `quality_cache` and `converged_speed` for each k-means scheme, then calls `report`. No k-means run is needed.

7. **The extras drop every non-finite number without saying so** (`report.rs:151`). A best with an untimed slot has
   an infinite median ratio, and it disappears from the stream; a NaN angle would disappear too. Fix: an angle or a
   percentile that is not finite is an error (our software has a bug). An infinite ratio is written as its own key
   (`ratio_untimed = 1`), so the stream shows the best was not fully timed.

8. **`r/(1+r)` is written twice.** `report.rs:240` repeats the speed mapping of `staged_eval` (`evaluator.rs:1272`)
   for the reference program. Fix: one function `speed_inputs(ratios)` used by both, so the two cannot drift.

## Note on d under repetition

For the weighted vector (quality, quality, speed) the report takes d = Q·m + S, the vector's length. That follows
fuller's `hff_percentile` contract ("d is the length of the vector the angle was taken over") and Andrew's ruling. A
repeated vector lies in a (Q + S)-dimensional subspace, so F at d = Q·m + S is not the probability for a direction
drawn uniformly over that subspace. The report states m and d, so either reading can be recomputed from the stream.
