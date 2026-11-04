# Step 5a P4 code review: the Futhark runner on the engine (`futhark_fit`, `evolve::futhark_run`)

Reviewed 2026-11-04: phylu `ec141644` (fuller `docs/PLAN_futhark_kingdom.md` 5a, row P4; the P4 status in fuller
`61f80707`). Read only, no build, no run (a GPU job was running). Line numbers are at `ec141644`. Read in full:
`src/evolve/futhark_run.rs`, `src/evolve/futhark_run/tests.rs`, `examples/futhark_fit.rs`, the diffs of
`engine.rs`, `external.rs`, `futhark_eval/evaluator.rs`, `futhark_typed.rs`, `program_eval.rs`,
`tests/runner_standard.rs`, `experiments/README.md`, and `/private/tmp/claude-501/fbench/plan5a/g2_futhark_fit.sh`.
Compared with the old runner at fuller tag `pre-5b` (`examples/futhark_evolve.rs`).

## Verdict

**Accept with fixes. G2 cannot run as written, and four findings must be fixed before G2 runs.**

- The GPU script stops itself at its first check (finding 1).
- The ledger's ids, and so its parents column, are wrong after every pump (finding 2).
- A resumed run truncates the stream, the genealogy and `hof.tsv` (finding 3).
- A new run into an existing OUT appends to the old run's series (finding 4).

Findings 5 to 8 must be fixed before R1, because they change what R1 compares. Findings 9 to 12 are smaller.

The runner adds no generation loop and no parallel mechanism. Selection, the pump, the islands, the fold beat and
ALPS are the engine's. The runner supplies flags, hooks and files.

## Checked and correct

| Item | Result |
|---|---|
| Island shape (5a.6) | One pair, intake 16 + champion 8, the pump every 5, `cohort_merge` = 5 × 5 = 25 (`futhark_run.rs:234-239`, `:919`). The test pins all of these (`tests.rs:242`). |
| Defaults that match the old runner | All of these match `pre-5b` `futhark_evolve.rs:858-905` and `:1623-1631`: marriage 0.3, fold every 10, tournament 3 (the old one is 1 draw + 2 more), elites 2, 6,000 evaluations, 60 minutes, 11 timing pairs (k-means 2), 20 runs, `expressed_edits` 2 in P and 4 in R (the old `m_of`). Marriage offspring are not mutated (`MarriageSpec { mutation: 0.0 }`, `:907`), as in the old runner (old `:1650-1656`). The tournament fractions 3/16 and 3/8 round to 3 on both islands (`engine.rs:4561-4564`). |
| Checkpoint every generation | `checkpoint_every_seconds = 1e-9` (`:945`). The engine checkpoints at the top of each loop iteration, after generation g (`engine.rs:9520`), and again at the end of the fit. The plan asked for this in the P2b review (item 3). The F4/F7 review's item 4 (`refill_draws` not saved) is closed, because no checkpoint falls between two refills of one generation. |
| Resume order | `new`, then `controls` (`:976`), then `set_fresh` (`:977`), then the engine's restore inside `fit` (`engine.rs:9423-9427`). `restore` puts the evaluator's state back first and then the original controls (`replace_controls`, evaluator.rs `+2267`). It rewrites `aa_gate.tsv` from the saved original (`:815-817`), and the stream notes the re-check (`:827-835`, `:1027`). This settles the F4/F7 review's item 5. |
| Cut-back on resume | Checkpoint g holds the state after generation g, and a resume runs g + 1. `cut_after(<= g)` (`:504-515`) is therefore the right cut for the ledger, the series, `per_generation.tsv` and `test_case.tsv`. All four have the generation in their first column. |
| The stage-change fix (evaluator.rs `+2329-2339`) | Correct. `score_rows` gets the same `distinct` and `row_of` in both passes, so the zip pairs row r with row r. The stage-2 pass reads its quality from the cache, so OR-ing in the stage-1 flags names exactly the rows that `evaluations_new` counted. It is tested: `tests.rs:196-197` asserts `(1, [0])` across the stage change, and without the fix `score` returns the "charged 1, named 0" error (`:673-675`). |
| Fitness scheme recorded | The scheme is recorded in the header line (`:966`, `futhark_fit.rs:34`), on the card (`--fitness`, `fitness`, `fitness_commit`, `:885`, `:894`), in the stream's `run_start` (`ExternalStream.fitness` = `start_with(weight, cliff)`, engine `+9333-9337`) and in `report.tsv` (`final_report`). |
| Percentile with d | `per_generation.tsv` writes the percentile and `hff_dimensions_d` beside the angle (`:766-774`). The progress table has `hff_percentile`. |
| Runner standard | `standard_outputs`, `killme.trigger` (`RUN_FILE_ENV`, `:964`), the stream's kingdom `futhark_sig` and the evaluator's column names as `hff_objectives`. It is registered in `tests/runner_standard.rs`. No `#[allow]` and no new `_` names. The `.ok()` calls are test clean-up only. |
| Checkpoint save errors | `ExternalState::save` now returns a `Result`, so a state that cannot be saved ends the fit instead of being saved as null (`external.rs +141`, `engine.rs +8411`). |

## Findings

1. **The GPU script always stops at its own guard (high, blocks G2).** `g2_futhark_fit.sh` checks for other runs with
   `pgrep -fl "deps/fuller-|deps/phylu-|futhark_evolve|futhark_fit"` inside `{ … }`, which runs in the script's own
   shell. That shell's command line is `/bin/zsh …/g2_futhark_fit.sh`, which contains `futhark_fit`. `pgrep` leaves
   out only itself, so it matches the script's own shell every time. The script prints "STOP: another … process is
   running" and exits 3 before the build.
   Failure: the lead runs G2 and nothing runs.
   Fix: match the binaries, not the name: `pgrep -fl "deps/fuller-|deps/phylu-|examples/futhark_evolve|examples/futhark_fit|/futhark_fit "`.
   Or exclude `$$` and its children.
   Also: the 60-minute guard uses `kill -9`. Delete each run's `killme.trigger` first and wait (memory: killing a
   process does not stop its GPU work).

2. **The ledger's ids are wrong after every pump (high).** The engine calls the fitness hook again after a beat with
   `upload = true`, and then it passes no parent map (`engine.rs:5020`). For a scored row, `score` takes the id of
   whoever sat in that row before (`futhark_run.rs:615-622`: `None => prev[r]`). The pump moves scored rows:
   - a promotion copies an intake row, with its fitness, over a champion row (`engine.rs:6034-6040`);
   - `make_room` swaps champion rows;
   - `refill` moves the keepers.
   Each moved individual takes the id of the individual it displaced. Every child bred from it later names the wrong
   parent in `ledger.tsv`.
   Failure: from generation 5 on, the ledger's lineage disagrees with `genealogy.tsv`. Any lineage read of the ledger
   (where a winner came from) is wrong.
   Nothing tests this. Every runner test and every GPU run is at most 4 generations with `--pump-every 5` and
   `--fold-every 0` (`tests.rs:254-266`; the script runs 3 generations).
   Fix: give a scored row the id of its genome, keyed through `genome_key`, from the previous call's rows. A moved row
   keeps its genome, so it keeps its id. Alternatively the engine passes the beat's row moves to the hook. Then add a
   runner test with a pump and a fold beat (`--pump-every 2 --fold-every 3`, 6 generations) that checks every ledger
   parent is an id the ledger already holds, and that a promoted row keeps its id.

3. **A resumed run truncates `stream.jsonl`, `genealogy.tsv` and `hof.tsv` (high).**
   - The engine opens the stream with `telemetry::Writer::create` (`engine.rs:9284`), which truncates unless
     `HFF_TELEMETRY_APPEND=1`. The runner does not set it; `law_runner.py:299` does.
   - `GenealogyLog::create` (`engine.rs:9266`, `genealogy.rs:563`, `File::create`) starts again with an `init` batch
     at generation 0.
   - The hall-of-fame file's header is written over the file on every fit (`engine.rs:9484`).
   The module header says "a resumed run's files are an uninterrupted run's" (`futhark_run.rs:30-31`), but that holds
   only for the runner's own four files. The GPU resume test compares only the checkpoint, the ledger and the series
   (`tests.rs:312-338`).
   Failure: after any `--resume`, evowatch shows only the second leg, and the first leg's genealogy and hall-of-fame
   lines are lost.
   Fix (engine, for every kingdom): on `config.resume`, open the stream with `Writer::append`, append to the
   genealogy and to `hof.tsv` after cutting them back to the checkpoint's generation, and checkpoint the lineage
   tracker. Extend the resume test to compare `stream.jsonl` (two `run_start`s, every generation's snapshot once),
   `genealogy.tsv` and `hof.tsv`.

4. **A new run into an existing OUT appends to the old run's files (medium).** `append` (`:517-525`) writes a header
   only when the file is missing. `FutharkRun::new` cuts the files only on a resume (`:535-539`). A second run
   without `--resume` into the same OUT therefore adds its lines under the first run's in `series.tsv`,
   `per_generation.tsv` and `test_case.tsv`. Only the ledger is truncated (`Ledger::open_with`). Stale copies of
   `hall_of_fame_stage1/` and `aa_gate_recheck.tsv` also stay.
   Failure: the chart and the learning curve of the new run mix two runs.
   Fix: on a run that is not a resume, remove these files, or move them aside as the engine moves old checkpoint
   slots to `superseded/`.

5. **The start's evaluations are outside the budget (medium, R1's comparison).** The old runner stops on
   `judge.evaluations >= budget` (old `:1289`, `:1309`), which counts every program the start scored.
   Here the budget is the engine's `eval_counts`, charged by `evaluations_new` alone. The programs `start()` scored
   (`fresh_cost.evaluations`, the alternatives and any output checks) are printed (`:995-998`) and never charged.
   When generation 0 scores them, they are cache hits and free. F5 handed this item to P4 ("the alternatives'
   evaluations are charged in `evaluations_total` only"), and it is not done.
   Failure: a run with a start that spends evaluations gets them free, against the old runner's budget.
   Fix: charge `fresh_cost().evaluations` to the engine's count before the fit (an `External` field, or a
   `Main`-source charge). State it on the card.

6. **Generation 0's mutation events are lost, and the genealogy names every start individual `init` (medium).**
   `run.start` keeps each `FreshIndividual`'s origin for the ledger but drops its `event` (`:578-584`). The engine
   copies `External.seed` as bare genomes (`engine.rs:9238-9245`), and the genealogy logs the whole population as
   `Origin::Init` (`:9268`). The old runner emitted `egraph_exact`, `saturated_fold` and a note for each generation-0
   mutant (old `:1220`, `:1233`, `:323`). The standard requires a mutation event for every operator outside plain GEP
   variation.
   Consequence: in regime P the stream has no event for the e-graph variants, the fold forms or the mutants that make
   up generation 0. The genealogy's `founder_origin` cannot tell the reference program from its variants, so
   evowatch's founder statistics are flat. Only `ledger.tsv` keeps the origin.
   Fix: in the engine, let `External.seed` carry `FreshIndividual`s and land them through the same `Born` path as a
   refill (origin, parent rows, the event with its row). This is the engine's mechanism, not a runner note.

7. **Measures that the 5a.4 table needs are not written (medium).** The old runner wrote `stages.tsv` every
   generation (the stage seconds, `gen_seconds`, `evaluations_per_second`; old `:1462-1472`). It also put
   `stage_*_s` and `evaluations_per_second` in the snapshot's extras (old `:1404-1409`). The new runner writes
   neither. 5a.4's table needs "evaluations a second" and "modules a generation", and G2 needs the modules a
   generation. The judge counts the modules (`Judge::modules`), and `GenerationCost` carries them, but no output
   carries either. `judge.gen_stages` is never reset each generation, so it now accumulates.
   Fix: write `stages.tsv` from each call's `GenerationCost` (summed over the calls of a generation), and add
   `evaluations_per_second` and `modules` to the series and the extras.

8. **Four defaults differ from the old runner, and the plan does not say so (medium, R1 like for like).**
   - (a) `--fold-top-k` 1. The fold examines each island's best, so 2 programs a beat. The old fold jointly extracted
     the forms of the whole population of 24 (old `:1597-1600`).
   - (b) Elites 2 *an island*, which is 4 in the population. The old runner had 2 in the population, at most one an
     identity (old `:1555-1565`).
   - (c) Regime R's default start `random` is the engine's head-rule draw. The old runner's default R start was
     `Genetics::random_gene`, which is `--start drawn` here (old `:1258-1262`). F5's status says "R and C … start
     from `Mutants` or `Drawn`", and the GPU script runs histogram R on the engine's draw.
   - (d) `--stop-at-exact` (old `:900`) is gone. R2b can still read the evaluations to the first exact program from
     `first_exact`, but the old R runs stopped there.
   Fix: decide each one in the plan (5a.6 or the P4 row) and name it on the card. For R1/R2b like for like, the
   defaults should be the old runner's: R from `drawn`, and the fold over the whole population (`fold_top_k` up to
   the cap). Keep the others only if the plan says why.

9. **The re-check file's name disagrees with its documentation (low).** The module header says
   `aa_gate_recheck_g<N>.tsv` (`:29`). The code writes `aa_gate_recheck.tsv` (`:813`), so a second resume overwrites
   the first re-check. Also, on a resume `controls` (`:976`) rewrites `aa_gate.tsv` before `restore` puts the
   original back, so a crash between the two leaves the re-check as the yardstick.
   Fix: write `aa_gate_recheck_g<N>.tsv`, and have the resumed `controls` write somewhere other than `aa_gate.tsv`.

10. **Two start individuals with the same genome lose an origin (low).** `origins` is keyed by the genome hash, and
    `remove` takes the first match (`:581`, `:649`). The second copy is then labelled `random` at generation 0
    (`:655`). Refill individuals that the engine never lands stay in `origins` for the rest of the run.
    Fix: key the start by row (the seed's row is its index), and clear unlanded entries after each call.

11. **Unchecked arguments (low).** `--seed` is cast `as u32` (`:910`), so a seed above 2^32 silently becomes another
    seed. `--tournament` larger than the champion island is accepted (`:257`).
    Fix: reject both in `check()`.

12. **The ledger's operator names are imprecise (low).** A row that the engine's draw refilled at a pump is labelled
    `beat` with no parents (`:656`), while the genealogy calls it `pump_refill`. On the stage change every row is
    written again as `rescored` with its own id as its parent (`:652`), so one id has two ledger lines.
    Fix: name the beat (`pump_refill`, `fold`) from the call, and write `rescored` lines with no parent, or in a file
    of their own.

## The GPU script tests less than it claims

With finding 1 fixed, the script runs the three engine tests and two 3-generation runs, all with the pump every 5 and
the fold every 10. None of them reaches a pump or a fold beat. So the refill hook, the fold beat's forms, the ledger
across a pump (finding 2), a stage change inside a beat's call, and a resume after a pump are all unexercised on the
GPU. Its expected text "stage 2 from the start (regime P)" should say that the stage changes inside generation 0's
call (`hall_of_fame_stage1/` and a `stage_change` event at generation 0).

Fix: add one run with `--pump-every 2 --fold-every 3 --max-generations 6`, and a resume from generation 3 of it.
Check the ledger's parents, the refill and fold events in the stream, and that the stream and the genealogy are whole
across the resume (finding 3).

## Rules.md and CLAUDE.md

- No generation loop, population or selection in fuller. Everything added is phylu library code over the engine.
- Every new error message says "our software has a bug" where it is one. No `refused` or `declined`.
- Findings 2 to 4 are cases of "no hiding bugs": the resume test passes while three of the run's outputs are
  truncated. The test should fail loudly on them.
