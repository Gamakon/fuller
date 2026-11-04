# Code review: plan step 5a, task F2 (`1e5a0f4c`, `e32b39f1`, `c2e994a6`)

Reviewer: fuller code reviewer, 2026-11-04. Read against `~/dev/minkymorgan/qdrant/docs/Rules.md`, fuller's
`CLAUDE.md` and `docs/PLAN_futhark_kingdom.md` §5a.2 and the F2 row of §5a.3. Every changed line of the three commits
was read. For the moved code, the lines removed from `examples/futhark_evolve.rs` were also compared mechanically with
`src/futhark_evaluator.rs`, ignoring only indentation, `pub` and `fuller::` changed to `crate::`.

**Verdict: pass with corrections.** The move is faithful and behaviour is unchanged. No evolution logic is in the
library. One correction should land before P1: the work cap's feature gate (section 6). The other points are notes.

## 1. The task as the plan states it

| plan requirement (5a.2 table, F2 row) | done |
|---|---|
| `Judge` and its types (`Quality`, `Eval`, `Stages`, `TimingInput`, `AaSets`, `ConvergedJudge`, `TestCaseRun`, `pair_steps`, `pair_ratios`, `LEDGER_STAGE_COLUMNS`, `hff_dimensions`) moved, unchanged | yes. The mechanical comparison shows `Judge`'s whole `impl` (lines 466-1127) and every type unchanged. The only differences are `pub`, the imports, and `#[derive(Debug, Clone, Default)]` on `AaSets` (line 209). |
| `Genetics` moved | yes (1131-1258). The two signature changes are as the commit states: `marry` takes `(tree, gene)` pairs (1231) and `walk_event` takes the child's id and tree (1167). Neither changes what is computed. The runner's four `walk_event` calls and its one `marry` call are updated to match. |
| `FutharkEvaluator::new` | yes (1498-1651). The flag checks moved from the runner without change. The construction of the hosts, the shim, the yardstick and the timing input matches old runner lines 2271-2425. |
| the controls behind the owned object | yes, `controls(&ControlRequest)` (1777-1844) and `control_converged` (1850-1898), in the runner's order. |
| `FreshRequest`, `FreshIndividual` defined | yes (1449-1472), with the fields of 5a.2 and a `FreshPurpose` enum. `fresh()` itself is left to F5. |
| `futhark_evolve.rs` calls the library, behaviour unchanged | yes. Apart from the runner's flag parsing and the controls call, every remaining change is a rename from `run.gen`/`run.judge` to `run.ev.gen`/`run.ev.judge` (checked mechanically). |

The implementation departs from the 5a.2 sketch in two places. The plan's status paragraph records both:

- `controls` takes `&ControlRequest { regime: Option<Regime>, aa_rounds }` and returns `Controls`, not `controls(regime)`.
  The `None` regime is needed by the measurements (`--check`, `--measure`), which time nothing.
- `EvaluatorConfig` gains four fields: `kmeans_step_process`, `gate`, `rings` and `ring_walk`. The runner needs them.

The F2 row says "the controls and the start". The P start (`alternatives_population`, `fable_mutants_population`)
stays in the runner. The 5a.2 table gives it to F5 (`src/futhark_start.rs`), so leaving it is right. The row's wording
could say "the controls at the start".

The row asks for the k-means plumbing check with "the same quality and speed inputs bit for bit". The speed half
cannot hold (section 5). The plan's status says so plainly instead of claiming it.

## 2. Evolution logic in the library

None. `src/futhark_evaluator.rs` contains no loop over generations, no selection, no tournament, no elites, no
population type and no survival. `Genetics` holds the variation operators of one individual: mutation, ring walk,
crossover, marriage and neighbours. The plan's 5a.2 table puts it in the library. The library's only population-shaped
types are the engine's request and reply types `FreshRequest` and `FreshIndividual`, as the plan specifies. `Regime::C`
is only a tag. The per-generation caches in `Judge` (`batched_q`, `batched_t`, `batched_k`, `gen_stages`) are scoring
caches. They were moved unchanged.

## 3. Rules.md and CLAUDE.md

- No stubs. `fresh`, `evaluate_generation`, `report`, `state` and `final_report` are not declared; they are not faked.
  The module header names the task that builds each one.
- No test loosened. No existing test changed. The three new tests in `src/futhark_evaluator/tests.rs` are real:
  - every configuration error is checked by its message;
  - histogram on the C backend passes its controls, the reference scores HFF 0, and a repeat is a free cache hit
    (`evaluations` 1, `hits` 1, `scored` 2);
  - the reference gene expresses the reference, marriage gives offspring that express, and no neighbour equals its
    own tree.
- No `#[allow]`, `todo!`, `unimplemented!`, `dbg!` or `_`-renamed binding was added. The tests use
  `let _ = std::fs::remove_dir_all(..)` for cleanup, as 55 other places in `src/` do.
- Own paths only. The commits touch `src/futhark_evaluator.rs`, `src/futhark_evaluator/tests.rs`, `src/lib.rs`,
  `examples/futhark_evolve.rs` and the plan. The untracked `src/futhark_reader/kinds.rs` was not swept in.
- Zero warnings: see section 7.

## 4. Defects and notes

No defect changes a result. The notes:

1. `src/futhark_evaluator.rs` 1485-1489: `judge` and `gen` are `pub` "for the deprecated runner only". Nothing enforces
   this, and phylu could reach in. Make them private when the runner goes (X1), or when F3a gives the seam.
2. `controls()` (1777) is not idempotent. A second call adds the in-run A/A to the timing slots again (1819) and
   re-runs the k-means A/A as an evaluation. Either guard it or document "once a run" before phylu calls it (F6).
3. The library prints to stdout with `println!` in `new` (1567, 1573), `ring_canvas` (1331), `control_interpreted`
   and `controls`. This was moved unchanged. Under phylu's engine those lines go to the engine's console, not to the
   stream. F6 should return them as notes, as `Controls::notes` already does for three of them.
4. `Controls::noise` duplicates `Controls::aa.noise` (1789). Harmless.
5. The k-means control path (`control_converged`, the k-means A/A) has no library unit test. Only the before/after run
   exercises it. F3b's C-backend test can cover it.
6. The order of the runner's console and stream changed, and the commit states it. The quality-weight line now prints
   after the timing bound. A bad flag combination such as `--eval rust --backend c` now fails before `run_start` is
   written, where it used to fail after. The shim build also comes before `run_start` now. None of this changes a
   ledger, an input or an event.

## 5. The agent's two claims (logs in the session scratchpad `f2/`)

Re-checked here from `hist_*/ledger.tsv`, `km_*/ledger.tsv` and `*/stream.jsonl`. Both claims hold.

| claim | measured here |
|---|---|
| histogram R `--gate`, 30 evaluations: `ledger.tsv` identical on 33 x 12, the 11 `*_ms` excluded | identical: 33 lines (the header and 32 individuals) x 12 columns, before against after, and before against before. `run_start` differs only in `timestamp_utc`. The 8 stream events are the same kinds in the same order. |
| k-means P, 3 evaluations, pop 4: quality inputs identical, seconds speed inputs differ, also between two runs of the old binary | for each of the 3 individuals, only `fitness` and `speed_inputs` differ (text, quality inputs, behaviour and `all_zero` are identical). Within `speed_inputs`, every iterations input (0.5) is identical and every seconds input differs, before against after and before against before alike (e.g. 0.50052 / 0.49408 / 0.50125 for the first input). `run_start` differs only in `timestamp_utc`; the 15 events match. |

The seconds inputs are wall-clock ratios, so no two runs can agree on them bit for bit. Excluding them is justified.
The quality group, which selection reads first, is bit-identical.

## 6. The work cap and the `wgsl` feature

The dependency is wrong for a Futhark library, and should be fixed before P1. The module is built only under `wgsl`
(`src/lib.rs` 101-103), and the only reason is one `u128` constant,
`CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT = 170_000_000` (`src/wgsl/work_guard.rs` 24). `work_guard.rs` has no imports:
it is pure arithmetic about the M3's per-submit budget, not about WGSL. Its effects:

- `cargo test` with default features, the command in CLAUDE.md, never compiles the evaluator or runs its three tests.
  The "778 tests pass" in the plan's status therefore does not include them.
- phylu's Futhark engine (P1) would have to turn on `fuller/wgsl`, which pulls in naga and toml. phylu's `Cargo.toml`
  documents `wgsl` as "off by default: SR/TSR/REGEX never need naga".

Recommendation: move the cap, or the whole of `work_guard`, to an ungated module (for example `src/work_guard.rs`).
Keep `pub use` re-exports at `wgsl::work_guard` and `wgsl::oracle::device` so no caller changes. Then drop the
`#[cfg(feature = "wgsl")]` on `futhark_evaluator`. The evaluator's other imports (`futhark_batch`, `futhark_wgpu_host`,
`futhark_rust`, `futhark_rings`) are already ungated, since it builds under `wgsl` without `gpu`.

## 7. Re-run here

`cargo test --features wgsl --lib futhark_evaluator`, 01:07:36Z to 01:08:31Z (55 s, fuller included in the compile):
3 passed, 0 failed, 4.63 s; 0 warnings in the build. Clippy and the default-feature suite were not re-run here, to
avoid a second long build. The agent reports clippy clean with default, `wgsl` and `gpu,wgsl`.
