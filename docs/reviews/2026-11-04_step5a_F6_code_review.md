# Step 5a F6 code review: the controls' A/A gate and the final report

Reviewed 2026-11-04: phylu 24a08cd3 (`src/futhark_eval/evaluator/final_report.rs`, its `tests.rs`, and the controls
in `src/futhark_eval/evaluator.rs`), against fuller `docs/PLAN_futhark_kingdom.md` (the F6 row, 5a.2, section 5b, the
F6 status of fuller 21f69f2f), both CLAUDE.md files, `qdrant/docs/Rules.md`, the deprecated runner
(`examples/futhark_evolve.rs` at the tag `pre-5b`, lines 1059-1170 and 1832-1946) and Andrew's rulings (the HFF is
the final fitness; the percentile is reported with its d; tables name columns with units; never a parallel mechanism).
Read only, no build. Line numbers are at 24a08cd3.

## Verdict

**Accept with fixes.** The report's keys and their order are the old runner's, and I checked the test's lists against
the old runner's format strings key by key myself. The re-time and the verdict are the old runner's logic, through the
same functions. The A/A gate file is written from the same numbers as before. No fitness value changes. Fix findings 1
and 2 before the engine calls `final_report` (F8 and the P tasks). The others are smaller.

## Checked and correct

| Item | Result |
|---|---|
| report.tsv keys, every run | The 38 base keys in `tests.rs:15` equal the old format string at `pre-5b` 1837 (36 keys) plus 1867 (`reference_static_cost`, `best_static_cost`), in order. `report_lines` (final_report.rs:182-220) writes them in that order. |
| The GPU block | `GPU_KEYS` and `RETIME_KEYS` (tests.rs:22-23) equal old 1870-1889 in order, `inrun_aa_*` only when the in-run A/A ran, `champion_exact_on_timing_input` only when the problem stores the timing output. |
| The k-means block and the tail | `KMEANS_KEYS` and `TAIL_KEYS` (tests.rs:26-29) equal old 1896-1932, including the old runner's repeated `best_hff` and the `fitness` card line last. |
| Is the test really against the old list? | Partly. The lists are typed in by hand from the old source, not read from it, so the test pins the new code to a transcription. The transcription is correct (checked above), so this is not a fault today. |
| The re-time | final_report.rs:144-148: on the GPU, for an exact champion, `champion_batch(&tree, 11)`, the timing output check against `read_one(want)`, then `aa_medians(&reference, aa_retimes)`. Same calls and order as old 1876-1886. |
| The verdict | `fuller::runner::speed_verdict` (fuller `src/runner.rs:233`), the same function the old runner called: Beat when the median is below the least A/A median and at least 9 of 11 pairs are faster; Slower above the greatest; else Matched. `noise_floor_least_aa_median` and `speed_claim_beats_noise_floor` are computed as at old 1887-1889. |
| `--aa-retimes 0` | Refused with a message, as old 906 (final_report.rs:137). |
| The A/A gate | `controls` (evaluator.rs:2236-2240) calls `fuller::runner::write_aa_gate` with the rounds in µs, the noise pairs and the in-run A/A, exactly as old 1113-1116. It runs only when the A/A sets were timed (GPU, a regime, not k-means). It now also runs for the measure and check commands, which the old runner returned before; harmless. |
| The k-means A/A noise floor | final_report.rs:280-292 equals old 1914-1920: the floor is the widest distance of the A/A's speed inputs from 1/2; the claim needs every seconds input (the first of each case's speed group) below 1/2 minus the floor. |
| `staged_eval` uses `report::speed_inputs` | Same mapping, `r/(1+r)`, non-finite to 1 (report.rs:150). One mapping, not two. |
| The chart | `CHART` is the old script byte for byte (diffed). |
| No parallel mechanism | The report reads the judge's counters and the controls' kept numbers; it scores nothing new except the GPU re-time, which the old runner also did. |

## The reduce expectation (best_hff 0.205 to best_theta_quality == 0)

**The right reading, not a hidden fault, but the test now checks less than it should.** In regime P the controls put
the evaluator in stage 2, and the test fills the reference program's 11 timing slots with ratio 1, so every speed input
is 1/(1+1) = 1/2. The flat scheme's stage 2 is the TrueNorth angle over the quality inputs repeated m times plus the
speed inputs (fuller `src/fitness.rs:654-666`). An exact program at speed parity therefore has a positive HFF by
design: the zeros pull the angle down, the halves keep it above 0, and fewer quality inputs (reduce) give a larger
angle. `reduce` is the first entry of `PROBLEM_NAMES`, so an assertion `best_hff == 0` would fail there first and stop
the loop; it would have failed on every answer-scored problem. Checking the quality angle at 0 is the correct claim.
See finding 4 for what to add.

## The GPU script's 3 re-times

**Fine for this check, with one condition.** `f6_controls.sh` checks the path: the test asserts 11 pairs and one median
for each re-time (tests.rs:208) and does not assert the verdict, because the champion is the published program and any
"beat" is noise (up to 1 in 4 at 3 re-times, against 1 in 20 at 19). It is a test of the plumbing, not a measurement.
The condition: its verdicts must never be quoted as a speed result, and the default of 19 (`runner::AA_RETIMES`)
should run once, on one problem, before the first real run (Rules.md: no simplifying long-running tests), because no
test runs that default today.

## Findings

1. **The k-means `test_case` lines can describe a different program from the champion.** final_report.rs:295 writes
   `self.champion_tested`, which is the evaluator's own champion (evaluator.rs:2313-2327: the lowest fitness the
   evaluator has seen), while the report's other `best_*` lines describe `champion`, the genome the engine hands over
   (its hall of fame's rank 1). In the old runner both came from `run.champion`, so they always matched. Failure:
   the engine's hall of fame is reset at the stage change, or a resumed run's hall of fame differs, or the engine
   hands over another island's best; then `test_case` reports another program's test cases under this champion's
   name, silently. Fix: in `final_report`, require `champion_tested`'s source to equal the rendered champion's
   source, and if it does not, run `champion_test_case` for the handed-over champion (it is one module) or return
   "our software has a bug: the test cases are of another program".

2. **The hall of fame across the stage change is not settled, and the report will expose it.** The old runner started
   a new hall of fame at the stage change (`pre-5b` 1357-1369: stage 1's kept in `hall_of_fame_stage1/`, fault 16),
   so its rank 1 was always scored in stage 2. `final_report` takes the champion's score from `cached_eval(.., self.stage)`
   (final_report.rs:143). If the engine's hall of fame keeps stage-1 entries, an inexact stage-1 program at, say,
   0.1 ranks above an exact stage-2 program at 0.2 (finding 4 shows exact programs at parity are above 0), it is
   handed over, it has no full timing slots, and the report fails with "never scored in stage 2". Loud, so not a
   silent fault, but the run ends without a report. Fix: state in the `final_report` doc and in the F8/P task rows
   that the engine's hall of fame restarts at the stage change (as the old runner did), and add a test: a stage-1
   champion handed over after the stage change is an error that names the stage-change rule.

3. **A champion wrong on the timing input can still carry a "beat" verdict.** final_report.rs:234-248 writes
   `champion_exact_on_timing_input NO` and then the verdict from the ratios regardless. The old runner did the same
   (old 1878-1889), so the port is faithful, but it is a bug under the house rule "fix bugs, never just document": a
   program that is exact on the fitness cases and wrong on the timing input is not a speed result. Fix: when
   `exact_on_timing_input == Some(false)`, write `verdict not reached (wrong on the timing input)` and
   `speed_claim_beats_noise_floor no`; a test with `Retime { exact_on_timing_input: Some(false), ratios: [0.9; 11], .. }`.

4. **The stage-2 HFF of the reference program is no longer pinned.** tests.rs:99-101 now checks only
   `best_theta_quality == 0` for the answer-scored problems and nothing for k-means. Failure: a change to the flat
   scheme's vector, the weight m, or the speed mapping changes every stage-2 HFF and this test still passes. Fix:
   also assert `best_hff` equals `fuller::fitness::qqs(&zeros, &[0.5; pairs], quality_weight)?.fitness` formatted to six
   places, and `best_theta_speed` equals the TrueNorth angle of the halves; for k-means, the A/A's quality inputs at
   exactly 1/2 are checked already, so check `best_exact_*` instead.

5. **The percentile with its d is reported only for the cliff schemes.** final_report.rs:254-262 writes
   `hff_dimensions_d` and `best_hff_percentile` only when `scheme.uses_cliff()` (k-means). For reduce, scan, histogram,
   pathfinder and a7, `report.tsv` gives `best_hff` and `best_ever_hff` as raw angles, which Andrew's ruling says are
   not comparable across dimensions. F4 reports the percentile with d for every scheme in the generation report; the
   final report does not. Fix: after the old runner's keys (so the old order stays a prefix), add `hff_dimensions_d`,
   `best_hff_percentile` and `best_ever_hff_percentile` for every scheme, and extend the key test.

6. **The chart is optional.** final_report.rs:156 draws the chart only when `run.series` is given; the old runner
   always drew it (old 1935-1946), and the standing rule is a chart of HFF against generation for every evolution
   report. No phylu code writes `series.tsv` yet. Failure: the engine integration passes `series: None` and every run
   ends with no chart, silently. Fix: make `series` a required `PathBuf` (a measure command that has no series does
   not call `final_report`), or return an error when it is absent.

7. **Repeated keys, inherited.** On the k-means scheme `best_hff` appears twice; on the GPU the k-means block repeats
   `best_speed_inputs` and, with a re-time, `speed_claim_beats_noise_floor`, which can then hold two different values
   (the GPU re-time's and the k-means floor's). A reader that loads the file as a map keeps the last. The old runner
   did the same, and the brief asks for its keys, so this is not a port fault. Fix later, with the consumers: rename
   the k-means one `seed_aa_speed_claim_beats_noise_floor`, after checking nothing reads the old name.

8. **Units are missing from most key names, inherited.** `minutes` names its unit; `best_work` and `reference_work`
   (node executions), `timing_cap`, `evaluations_per_second` and the ratio keys (unitless) do not say so. Andrew's
   ruling is that tables name columns with units. Low: add the unit to the doc comment of `report_lines` now, and to
   the key names only when the consumers are updated with them.

9. **A failed control leaves a misleading message.** `controls` sets `controls_ran = true` (evaluator.rs:2166) before
   it can fail; `self.controls` stays `None`; a later `final_report` then says "the final report before the controls",
   although they ran and failed. Low: set `controls_ran` only on success, or keep the failure and name it.
