# Code review: step 5a, P2a fixes and P2b (phylu `b6045261`, `8973a6b7`, `4f648bce`), 2026-11-04

Reviewer: code-review agent, 03:00-03:10Z. Read: `qdrant/docs/Rules.md`, fuller and phylu CLAUDE.md, fuller
`docs/PLAN_futhark_kingdom.md` 5a.2 and the P2a/P2b rows of 5a.3, the P2a review
(`docs/reviews/2026-11-04_step5a_P2a_code_review.md`), every changed line of the three commits, and the engine code
they touch (`evaluate_external`, `remember`, the fit loop's checkpoint, the resume, the end of the fit, `report`). No
build was run by the reviewer. The lead's `p2a_mock.sh` logs were read (`/private/tmp/claude-501/fbench/plan5a/`).

## Verdict

**Pass with corrections.** The P2a findings are fixed. P2b does what its row asks. All five GPU tests pass at phylu
`4f648bce` / fuller `4e9c17a7` (`p2a_mock/p2a_mock.log`, 03:00:45-03:01:26Z, 5 passed in 8.8 s): ten generations on
two pairs, stopped in generation 3, stopped in generation 0, the stage change, and the killed run resumed. Finding 1
is a bug in the file that the hall of fame writes, and it must be fixed before P4. Findings 2 and 3 must be settled
in P4's card. The rest are minor.

## 1. The P2a review's findings

| P2a finding | state |
|---|---|
| defect 1, `progress_every = 0` in `mock_engine` | fixed (`84a80116`). The P2a GPU tests pass at `8973a6b7` (`p2a_mock_run3`, 3 of 3) |
| defect 2, a stop in generation 0 returns an error | fixed. The result is row 0 with NaN, `run_end` says killed, and `math` says nothing was scored (engine.rs 9711-9719). New GPU test `the_external_stop_in_generation_0_ends_the_run_as_killed` passes |
| defect 3, fold forms not in the genealogy or the fold counts | fixed. `record_born` makes an arrival with the winner as first parent. `FoldCounts` and `fold_landings` are set |
| defect 4, the refill drops `parents`/`origin` | fixed. `Born`, `Genealogy::from_hook`, `Origin::of_hook`. The five origin words match fuller's `FreshIndividual::origin` (futhark_evaluator.rs 1529) |
| a second fold beside the engine's (`external_fold_every`) | fixed. One `fold_every` beat. The winners come from `fold_winner_rows`, now shared by the SR fold, the regex simplify and this. Each form lands through `fold_landing` |
| `external::worst_rows` copies `make_room` | fixed (removed) |
| `_parents` in the mock | fixed. The parent array is checked, and the test asserts it has 48 rows from generation 1 on |
| `p2a_mock.sh` builds with `-D warnings` and runs without it | half fixed. Both steps now set `RUSTFLAGS`, but the run step still recompiles phylu and fuller *outside* the cargo lock: the built and the tested binaries differ (`phylu-cd38a9c0…` built, `phylu-8922b002…` run). The script also still runs after it prints "another phylu/fuller test process is running" (fuller pid 93783). See finding 6 |

## 2. Exact resume: what the checkpoint does not carry

I traced every piece of engine state that the external path changes between two checkpoints.

- `hof_restart` is not saved. That is safe: every `evaluate` call that can carry a stage change is followed, in the
  same generation, by `remember`, or by `restart_staged_hall_of_fame` after the pump and the cross step (engine.rs
  9649, 9670, 9704). The flag is therefore always clear at the top of the loop, which is where the checkpoint is
  written.
- `telemetry.best_seen` is reset by a stage change. It affects only the stream, and a resume opens a new writer.
- `row_origin` is `Main` for every row at the top of the loop, which is the resume's default.
- `scored` is restored from `cp.scored`. Generation 0 and the population are not scored again (`check_scored`).
- The evaluator's state and its fresh generator are one object in fuller (`FutharkEvaluator::state`/`restore`/
  `fresh`), so the single `ExternalState` slot is enough.
- The genealogy and the hall of fame's `mark` are not carried. This predates P2b (SR behaves the same way), but in
  the external case it breaks a file: see finding 1.

The killed run writing no last checkpoint is correct. The loop breaks in the middle of a generation, and the slot
written at the top of that generation resumes it exactly. The resume test shows this: B's generation-10 checkpoint
equals C's bit for bit, including the hook state.

## 3. Findings

1. **The hall-of-fame file's columns shift after a resume when the genealogy is on.** phylu src/evolve/external.rs
   117-127 (`hof_line`), engine.rs 9206 (`mark: None` on the restored winner), engine.rs 8661.
   - The header has `age founder_gen founder_origin` *before* `program` whenever `genealogy_path` is set (external.rs
     92).
   - The restored winner has no mark, so `hof_line` leaves those three columns out.
   - The program text then lands under `age`, and every reader of the file misparses those lines until a new winner
     replaces the restored one. In the resume test, fit B writes such lines from generation 8 on.
   - SR has the same missing mark, but there the lineage columns are the trailing ones, so they are merely absent and
     nothing shifts.
   - Nothing catches it: `hof_line` checks only the inputs against the names, and no test reads the resumed run's
     `hof.tsv`.

   Fix: `HofRow` takes the header's lineage flag. A missing mark writes `-\t-\t-`, and a mark given to a header
   without lineage is a bug that is named. Extend the resume GPU test so that every line of B's `hof.tsv` has as many
   columns as its header.
2. **One hook request per fold winner, with SR's default `fold_top_k = 50`.** engine.rs 5492-5512.
   - `fold_external` asks the fresh hook once for each winner. That is each island's best `fold_top_k`, which is 50
     in `Config::srbench`, and every row when it is 0.
   - `futhark_config_errors` does not look at `fold_top_k`. On k-means, where the plan measures 193 s for one
     saturated fold, 4 islands × 50 winners is about 200 requests a beat, roughly 10 hours.
   - The mock passes only because it sets `fold_top_k: 2`.
   - The plan (5a.2, "on a fold beat, the saturated fold's forms of the best") still describes one best. This was
     changed on the P2a review's advice and is not recorded there.

   Fix: either bound `fold_top_k` in `futhark_config_errors` (and name it on P4's card), or ask the hook once a beat
   with every winner so that F5 can share the e-graph work. Record the change in 5a.2.
3. **"The newest slot is the one written before it" holds only when a slot is written every generation.** engine.rs
   9821 and the P2b commit message. The checkpoint is on a timer (`checkpoint_every_seconds`, engine.rs 9258). If
   that interval is longer than a generation, the newest slot is older and the generations since it are redone. The
   resume is still exact, but the message is false. Fix: print the newest slot's actual generation. P4's card should
   state an interval shorter than one k-means generation (minutes), or the engine should checkpoint every generation
   for an external kingdom.
4. **The progress table's header is still SR's** (`mse_train r2_train r2_val_bl2 … log10_p`, engine.rs 2490). It is
   printed for the Futhark kingdom in every mock run. P2b changed only the hall-of-fame file. Assign it to P4, which
   owns "the evaluator's column names".
5. **One stage-change path is untested.** The stage-change test fires in generation 5's variation call, so it covers
   `remember`. A stage change inside the pump's or the cross step's call (`restart_staged_hall_of_fame`) has no test.
   Add a `MockPlan` that changes the stage in the pump's call and checks that the hall of fame line at that generation
   is a stage-2 winner.
6. **`p2a_mock.sh` (not in a repository) rebuilds outside the lock.** This is the "one build at a time" rule. Run the
   tests from the binary that the locked `--no-run` step printed, or keep the lock over both steps. Also stop the
   script, instead of warning, when another test process is running.

## 4. Rules.md and CLAUDE.md

- No `#[allow]`, and no new `_`-prefixed names in the added lines. The one `let _ = state.writer.event(...)`
  (engine.rs 9803) predates these commits.
- No new "witness", "seed" (for a reference program) or "step" in a comment. "The cross step" is the engine's existing
  name for a mechanism. `External.seed`, the reference program's rows, is older and goes to P4/5a.4 as before.
- Nothing evolves in fuller. Every mechanism added here extends the engine's own:
  - the fold beat, the landing and the winners
  - the hall of fame and its file
  - the checkpoint slots
  - the `stage_change` event, which is fuller's existing `EventKind::StageChange`

  `hof_header`/`hof_line` is a second writer for the hall-of-fame file. That is justified, because its columns are
  the evaluator's, and the SR writer is untouched.
- `Slots::newest` now fails on a whole slot of another format for every kingdom, not only external ones. That is
  intended (5a.3: never read a format-3 file as 4). A resume of any older SR run needs the build that wrote it.
- The commit messages say the GPU tests were "built, not run". They have since run and passed (03:01:26Z). This
  review records that result.

## 5. The tests

- `the_external_stage_change_restarts_the_hall_of_fame` is a real test. Every stage-2 fitness is worse than every
  stage-1 fitness, so a hall of fame that did not restart would fail it.
- `the_external_resume_continues_with_the_same_hook_state` compares the whole generation-10 checkpoint bit for bit
  and the hook's state as restored. It does not read the resumed run's hall-of-fame file or stream, which is how
  finding 1 got past it.
- The host tests cover the round trip of the hook state and the hall of fame's inputs (bit for bit). They also cover
  a format-3 slot, both a short one and one that would parse as format 4, the named unscored rows, and the
  hall-of-fame header and its escaped line.
