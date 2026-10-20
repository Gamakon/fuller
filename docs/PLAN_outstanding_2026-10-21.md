# PLAN: finish the outstanding work, 2026-10-21

## Context

**Objective.** Close every item left open by the multi-typed GEP rebuild, the TSR kingdom and the day of speed work, so that fuller and phylu are complete, measured against the original fast engine, and ready for the WGSL kingdom to evolve real GPU programs (the route to evolving our own tools).

**Approach.** One list (below), each item a small specified task. Haiku takes mechanical tasks with an exact spec; Sonnet takes tasks that need reading and design. The lead (fuller session) reviews every result, bundles commits, and sends each bundle to Codex for review. GPU work is queued, one process at a time, by the lead only.

**Status.** Written 2026-10-21 from the list given to Andrew (47 items). The Status column is updated as each task closes.

**Why.** The TSR plan's experiments never ran, several design items were left open, and speed and search problems were found today. Andrew: "Finish the plan."

**House rules for every task.** Read `~/dev/minkymorgan/qdrant/docs/Rules.md` first. Zero warnings, clippy clean, no `#[allow]`. Never loosen a test; a failing test is a finding. No CPU implementation beside a GPU one; CPU code is a test reference only. No sympy. No stash, no worktrees; commit only your own paths (`git commit -- <paths>`), push. No GPU process unless the lead hands you the GPU. Measure, never guess; "found the law" means SRBench's official verdict only.

## Waves

- **Wave 1 (no GPU, independent files):** T12, T14, T13, T24, T26, T46, T19-script, T45-report.
- **Wave 2 (GPU, one at a time, queued by the lead):** T19-runs (benchmark), T32 (official check resume), T14-gpu, T1/T3 (TSR experiment arms), T28 follow-ups.
- **Wave 3 (engine work, GPU-gated):** T16, T20, T22, T23, T25, T17/T18, T29/T30.
- **Wave 4 (WGSL kingdom):** T36–T44.
- **Wave 5 (design items):** T7–T11, T21, T6 removals after T4.

## Tasks

| id | item | model | files | spec and gate | status |
|---|---|---|---|---|---|
| T1 | TSR §7 experiment, arms A, B, C, E, F, B8, C8 | lead (GPU) | `phylu/scripts/sr_old_new.py`, `experiments/tsr_arms.json` | first 40 laws of `order133.txt` × dev seeds 7001–7010, 400 seed-runs per arm, fixed generations (not clock), official verdict; table against §7.3 expectations | open |
| T2 | arm D-raw table, arm G | Sonnet | `phylu/src/evolve/order.rs`, `experiments/tsr_arms.json` | harvest raw decoded spellings into the pair table; G needs scaled-variable terminals; both runnable or stated why not | open |
| T3 | B8, C8 width cost | lead (GPU) | as T1 | part of T1's arm list | open |
| T4 | §7.6 arm selection | lead | `PLAN_tsr_typed_gep.md` | apply the rule to T1's table | open |
| T5 | golden lines per TSR arm | Haiku | `phylu/tests/fixtures/`, `typed_device.rs` | verify whether pinned; if not, pin with `PHYLU_PIN_GOLDEN=1` (lead runs the GPU) | open |
| T6 | removals of `typed_depth` / `typed_order` | Sonnet | list in `PLAN_tsr_typed_gep.md` §5 | only after T4 selects types | blocked on T4 |
| T7 | ADF rename across the population DAG | Sonnet | `docs/PLAN_population_dag.md`, `src/population_dag.rs`, `src/homeotic.rs` | rename homeotic chromosome/row/tail to ADF where the design says; tests unchanged and passing | open |
| T8 | `PLAN_wgsl_terminals` rows | Sonnet | `kingdoms/wgsl`, plan doc | rows added so the kingdom check passes on the live kingdom | open |
| T9 | population DAG variant B | Sonnet | phylu engine, device arena | fold in the generation loop, device arena, checkpoints with homeotic rows; then the banded compiler and the A/B/C experiment | open |
| T10 | ADF fold on by default | lead | engine config | after T9's A/B/C measures it | blocked on T9 |
| T11 | ticket 0001, the `?` count gate | Haiku | `phylu/docs/tickets/0001_qmark_count_gate.md` | verify open or closed against the typed decode; report | closed: decode.wgsl and its gate deleted in phylu a2ecc962 (Haiku, verified) |
| T12 | stale status lines | Haiku | `phylu/docs/tickets/CR_multityped_gep.md`, `phylu/docs/PLAN_wgsl_on_typed_gep.md` | replace "not started" with the built state, citing commits found by `git log` | done phylu e492bf8c |
| T13 | infix → `Math` parser in Rust | Sonnet | new `fuller/src/infix.rs` | reads SRBench/Feynman infix (`+ - * / **`, unary minus, `sqrt exp log ln sin cos tan tanh arcsin arccos arctan`, `pi`, numbers, names); returns fuller's `Math` s-expression; test: all Feynman CSV formulas parse and `eval_expr_rows` matches the dataset target to 1e-9 relative on 100 rows (host test reference); no sympy | done fuller 99b12a06: 100/100 Feynman formulas parse, 98/98 with data match to 1e-9 (2 have no dataset); arctan has no Math constructor (Err) |
| T14 | laws → genes test, committed | Sonnet | `phylu/src/evolve/tsr_typed.rs` | finish the uncommitted test: e-graph step (`extract::denoise`) per law, print Karva per type column, clippy clean, commit; GPU half run by the lead | done phylu 6aea7aba: all 14 laws encode in single and TSR order_depth and express back identically (host test); GPU half queued |
| T15 | e-graph step for the laws | Sonnet | with T14 | as T14 | open |
| T16 | typed snap and fold on the GPU | Sonnet | phylu `engine.rs`, `gpu_fold.rs`, snap | the host enumeration and rewrite move to device kernels; trajectories identical | open |
| T17 | identity rewrites failing under exact retyping (72 %) | Sonnet | `typed_write_back.rs` `Retype` | measure, find why, fix so a rewrite that preserves the expression is always typed | open |
| T18 | rewrites beyond type limits dropped | Sonnet | with T17 | count per beat; report | open |
| T19 | benchmark against the original fast engine | Sonnet (script), lead (GPU) | `phylu/experiments/bench/` | fixed laws and settings, fixed generations; rows: 30 Sep build `ceadcfe5`, current single, current TSR C, current regex; ms/gen per stage; results appended per version in `bench/results.tsv` | open |
| T20 | TSR decode walk in parallel | Sonnet | `typed_decode.wgsl`, `walk.wgsl` | no single-thread-per-gene loops; device == host | open |
| T21 | head room per type = whole program | lead + Codex | design | measure with B8/C8 (T3); propose a bound change only with numbers | blocked on T3 |
| T22 | validate written cells only | Sonnet | decode bindings | merge bindings to free a slot under Metal's 29 | open |
| T23 | same-expression carry | Sonnet | `resident.rs`, `dedup.wgsl` | a child whose genes' decode hashes equal its parent's takes the parent's score; trajectories identical | open |
| T24 | regex: split the GPU time; fix the profile line | Haiku (line), Sonnet (split) | `engine/regex_profile.rs` | the "genome frame read-back" line removed or replaced by the bytes actually read; a stage split of the matcher and scorer | open |
| T25 | per-type mutation rate semantics | lead | `typed_vary.rs`, `engine.rs` | report edits per row per kingdom; regex's printed rate made true | open |
| T26 | two failing engine tests | Sonnet | `phylu/src/evolve/engine.rs` tests | cross-step test and wrap-graft test: find the cause of each; fix the cause, never the assertion | open |
| T27 | fuller GPU timing tests under load | Haiku | report only | document that they need an idle GPU; no change | done: both need an idle GPU, no change |
| T28 | TSR 8 vs single 19 on hard laws | lead | after T1 | read with T1's result | open |
| T29 | diversity collapse, tournament 70 | Sonnet | engine selection | measure distinct individuals vs tournament size; report | open |
| T30 | neutral mutations 35–56 % | Sonnet | with T29 | report by operator | open |
| T31 | our 1e-10 bar passes non-laws | lead | report | the gate stays a trigger; count gate passes vs official verdicts | open |
| T32 | official check on the LG runs | lead (GPU) | `scratchpad/official_lg.sh` | resume at 9 of 26 | open |
| T33 | SR laws 101–133, fair 133-law number | lead (GPU) | `srbench_tf_run.py` | after T1; official seeds only with Andrew's go | open |
| T34 | regex cleanse and crossover speed, bisect | Sonnet | regex vary | after T24's split | open |
| T35 | regex simplify fold-back via typed write-back | Sonnet | `regex_simplify.rs` | simplify on, typed | open |
| T36 | `check.rs` wrap patch | lead | fuller `src/wgsl/check.rs` | verify applied; else Andrew applies | open |
| T37 | WGSL fallback-node removal | Sonnet | fuller `src/wgsl` | from `fallback_removal_wip.patch` | open |
| T38 | WGSL aggregate gaps | Sonnet | fuller `src/wgsl` | struct/array reads and rebuilds | open |
| T39 | `load.cfg_*` render rows | Sonnet | fuller `src/wgsl/render.rs` | emitter fix, test with `pcfg.d` | open |
| T40 | WGSL children unlike seeds | Sonnet | `wgsl_fit` ledger | parent/operator columns; run 6f (lead GPU) | open |
| T41 | interface 3 legality 12.9 % | Sonnet | with T40 | cause and fix | open |
| T42 | legacy WGSL-only variation code | Sonnet | `wgsl/mutate.rs`, `sections.rs`, `llm.rs` | inventory callers; fold useful operators into the shared core or retire | open |
| T43 | any-WGSL completeness gate | Sonnet | fuller tests | N of N corpus kernels read, rebuild, compile | open |
| T44 | a second real WGSL example beyond Gram | lead | problem dir | chosen with T43 | open |
| T45 | codegraph daemons | Andrew | — | start command needed from Andrew; never `codegraph index` | open |
| T46 | uncommitted docs | Haiku | phylu, fuller | inventory only (list, no deletes); lead commits the keepers | done: keepers committed (phylu 109ff4d3, fuller a9035c7e); LaTeX junk unstaged; logs/, target-*/ and problems/nluca_* are run output |
| T47 | old worktrees, `halo` folder | Andrew | — | not ours to remove | open |
| T48 | `cargo clippy --all-targets` fails without the `wgsl` feature (`examples/wgsl_harvest_read`) | Haiku | fuller `Cargo.toml` | give the example `required-features = ["wgsl"]`; clippy clean with and without the feature | done fuller 3f4cdf40: clippy --all-targets clean with and without wgsl |
| T49 | laws read by the infix parser, not hand-written | Sonnet | `phylu/src/evolve/tsr_typed.rs` | `FEYNMAN_LAWS` becomes the CSV infix through `fuller::infix::infix_to_math` (T13); extend to all 100 formulas with datasets | open |
