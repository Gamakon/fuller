# Code review: step 5a, P2a (phylu `54e7d4f9`, `1ce1b8d0`, `db3fc88d`), 2026-11-04

Reviewer: code-review agent, 02:10-02:25Z. Read: `qdrant/docs/Rules.md`, fuller and phylu CLAUDE.md, fuller
`docs/PLAN_futhark_kingdom.md` 5a.1 and the P2a/P2b rows of 5a.3, the advisor's finding 1
(`docs/reviews/2026-11-03_step5a_phylu_futhark_advisor.md`), and every changed line of the three commits, with the
engine code they touch (`make_room`, `fold_winners`, `fold_landing`, `track_fresh`, the fit loop, the resume and the
last checkpoint). No build was run by the reviewer; the lead's `p2a_mock.sh` log was read.

## Verdict

**Not done yet: the GPU half of P2a fails.** The host half is done as planned and is sound. The lead's `p2a_mock.sh`
run (02:15:52Z, phylu `db3fc88d`, fuller `f18d50c6`) built cleanly and then failed both GPU tests in 5.7 s, before
generation 0:

| test | result | cause |
|---|---|---|
| `the_external_hook_runs_ten_generations_on_two_pairs` | FAILED | `the engine: telemetry_path: the stream rides on the progress report, so progress_every must be > 0` |
| `the_external_stop_ends_the_run_as_killed` | FAILED | the same |

The commit says the GPU tests were "built, not run". So the external path has still never run. P2a's first run is
still to do.

## 1. Done as planned?

| P2a item / advisor finding 1 | state |
|---|---|
| keyed on a name (`safeWGSL`) | done before P2a (P1's `KingdomKind`, engine.rs 3861-3891); P2a adds a check before the fit starts that an external kingdom has its hook (8847) |
| refill in the first pair's intake only | fixed: `refill` asks the hook once a pair (`external::refill_pair`, engine.rs 5346+). The hook's rows come first, the device draws the rest, and every freed row gets the refill's cohort label. Host tests cover one and two pairs |
| genomes only in `fresh` | fixed: `ExternalFresh = FnMut(&FreshRequest) -> Result<Vec<FreshIndividual>>`, fuller's types, with the length and row width checked (`external::ask`). Each individual's mutation event goes on the stream with its row |
| cache hits charged | fixed: `external::charge` charges `evaluations_new`, each new row to its source. It rejects an inconsistent answer and adds nothing. The host test covers it |
| `stopped` ends as killed before the NaN check | done for a stop from generation 1 on. **Not done for generation 0**, see defect 2 |
| a fold beat asks for forms | done (`fold_external`, `Config::external_fold_every`). Section 2 has the design issues |
| no checkpoint slot | correctly left to P2b |
| the k-means layout GPU test after G1 | not written yet (G1 has not run). It belongs to P2a's GPU row |

**Left for P2b** (its row, plus what P2a leaves behind): the `serde_json::Value` hook-state slot in `Checkpoint`,
format version 4, with a version-3 file named as an error. The last checkpoint of a killed external run
(engine.rs ~9669) now saves rows with NaN fitness, while the resume code says "every restored fitness is non-NaN"
(~9031). P2b's round-trip test must cover a killed run. P2b also owns the stage change (restart the hall of fame,
write a `stage_change` event), the `extras` hook, and the hall-of-fame columns. `External.seed` still fills rows 0..
across both islands (8863), which goes to P4 and 5a.4. `FreshIndividual.parents` and `.origin` do not reach the
genealogy yet: `track_fresh` records every hook row as `PumpRefill`. 5a.1 calls these individuals "with their
history", and no task owns that yet. Assign it to P2b or P4.

## 2. Duplicated mechanisms

- **`fold_external` is a second fold beside the engine's fold.** A separate source of forms is justified, because the
  SR fold's work (collapsing a flat subtree to a constant, on the device) cannot act on a Futhark gene. But the beat,
  the setting and the landing are all new, built beside the engine's own:
  - The new setting `external_fold_every` sits beside `fold_every`, which `futhark_config_errors` forbids for this
    kingdom.
  - It is a second `if` block in the loop (engine.rs 9392).
  - The forms land in the intake's worst rows (`external::worst_rows`). The engine's fold lands a candidate in its
    source's own island (`fold_landing`, 7424), and it takes winners from every island, not only the champion's.

  Extend instead. The existing `fold_every` beat should send an External kingdom to the hook (and
  `futhark_config_errors` should then allow `fold_every`). The forms should land through `fold_landing`, or the
  intake should be named as a deliberate change with its reason.
- **`external::worst_rows` (external.rs 171-181) copies `make_room`'s worst-first ordering** (engine.rs ~5782-5793):
  the same comparator, written twice. It should be one function that both call.
- The refill is extended properly, not duplicated: the host-side `Fresh.host_genome` path is gone, and the hook now
  sits inside the engine's `refill`.

## 3. Rules.md and CLAUDE.md

- No `#[allow]`, no stubs, no loosened test. The deletion of the stale dev-dependency and the lock (`54e7d4f9`) is
  right. The four population tests (`1ce1b8d0`) are real tests.
- **"Tests must complete" / no presenting results as passing:** db3fc88d's message reports "263 tests pass" and
  calls the GPU tests "built", but those tests fail on their first run (defect 1).
- `_parents` in the mock fitness closure (futhark_typed.rs, `mock_external`) silences an unused argument with a `_`
  prefix, which CLAUDE.md forbids. Use the parent array: check that it is `Some` with `pop` entries after
  generation 0.
- `p2a_mock.sh` builds with `RUSTFLAGS="-D warnings"` but runs without it, so the run step rebuilds (02:16:34Z, about
  40 s) and the tested binary is not the one checked for warnings. It also printed "another phylu/fuller test
  process is running" (fuller pid 93783) and ran anyway.

## 4. Defects

1. **phylu src/evolve/futhark_typed.rs:434 (`mock_engine`)**: the config sets `telemetry_path` but takes
   `progress_every = 0` from `engine_config()`, so `Engine::new_with_table` returns an error. Set
   `progress_every: 1`, then run `p2a_mock.sh` again. Neither assertion of either test has run yet.
2. **phylu src/evolve/engine.rs:9568**: if the hook returns `stopped` during generation 0's scoring before any row is
   scored (likely on k-means, where a generation takes minutes), `best()` is None and `fit` returns `Err("the run was
   stopped ...")`. The plan says the run ends as `killed`. Return a killed result, or write `run_end` killed first.
   There is no test for this.
3. **phylu src/evolve/engine.rs:5438-5470 (`fold_external`)**: the genealogy is not told about the landed forms.
   `fold_winners` records an `arrival` for each landing (`lin.tracker.snap`, ~7306), so with `genealogy_path` set,
   these rows keep the identity of the individuals they overwrote. `FoldCounts` and the fold report are not updated
   either.
4. **phylu src/evolve/engine.rs:5346+ (`refill`)**: the hook's `parents` and `origin` are dropped (section 1). This is
   not a P2a acceptance item, but no task owns it yet.
