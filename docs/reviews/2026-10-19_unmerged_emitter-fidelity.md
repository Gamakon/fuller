# Review 2026-10-19: the nine commits of `emitter-fidelity` not on `paired-fitness`

## Context

Branch `emitter-fidelity` holds nine commits that `git log paired-fitness..emitter-fidelity` lists (all dated 2026-10-14, branch tip 2367af2, merge base 0f4d5c6). This review accounts for each. Method: every line of every diff read with `git show`; each claim checked against the code at the `paired-fitness` tip (7e94cba); for the seven commits and the one perf commit that have a twin on `paired-fitness`, `git patch-id --stable` compared the two diffs. Nothing was built or run (another agent is upgrading naga/wgpu), so no test result below is mine: test counts are quoted from commit messages.

Result: eight of the nine were cherry-picked onto `paired-fitness` under new hashes with byte-identical diffs (equal patch-ids) and are SUPERSEDED. One, the `wip` commit, is WIP: its example-binary half is on `paired-fitness` (0d74f08), its shared-load half was solved there by a different mechanism (0ade835), and its early-exit half is on neither branch. A later `wip` of the same work, f5ab515, sits on branch `emitter-fix` and is not on `paired-fitness` either.

| # | commit | verdict | twin on paired-fitness |
|---|---|---|---|
| 1 | b2c25cc | SUPERSEDED | 78b2a41 |
| 2 | e178ab9 | SUPERSEDED | 6feca92 |
| 3 | 243eeec | SUPERSEDED | 322dc78 |
| 4 | 61f3254 | SUPERSEDED | 9df73ca |
| 5 | cf9dd81 | SUPERSEDED | 23a5312 |
| 6 | 5d664cd | SUPERSEDED | 3a0d79c |
| 7 | 7834e99 | SUPERSEDED (behaviour since changed by 24a209e) | 74a8060 |
| 8 | 0324cc5 | SUPERSEDED | c908733 |
| 9 | 2367af2 | WIP | 0d74f08 (part), 0ade835 (other mechanism); early exit not merged |

Tip status from commit messages: the tip commit is `wip`, "not gated", stopped 2026-10-14 07:39Z, so the branch tip is not known to pass anything. Commit 8 states every fuller test (650) passes; commit 7 states 631 kernels round-trip with 0 differing; commit 4 states `the_harness_reproduces_gram_rss_fitness_on_the_host` passes; commit 5 and commit 9 carry tests but state no run. Commits 1 to 8 are individually gated by their messages; the tip is not.

## 1. b2c25cc llm::contract takes the problem: SUPERSEDED

Adds `llm::Contract { system, output_rule, strategies }`, `contract(Option<&Problem>)`, `GRID_OUTPUT_RULE`, `GENERIC_STRATEGIES`, `gram_contract()`, `problems/nluca_step/contract.md`, and `wgsl_safe_convert --problem <dir>`; tests `the_gram_contract_and_task_are_as_before_problems` and `a_problems_contract_comes_from_its_directory`.
Evidence: 78b2a41 has the same patch-id (622a7e01370b). At the tip: `src/wgsl/llm.rs:34` (`struct Contract`), `:43` (`GRID_OUTPUT_RULE`), `:47` (`GENERIC_STRATEGIES`), `:76` (`contract`), `:116` (`gram_contract`), tests at `:444` to `:481`; `examples/wgsl_safe_convert.rs:32` (`problem_kernels`), `:45`; `examples/wgsl_gram_hgp.rs:40` calls `contract(None)?.system`; `problems/nluca_step/contract.md` present. Behaviour differences: `contract` at the tip has a changed body (line 86 area) because later commits added to it (the `LOOPS` paragraph of commit 7, v24 callers such as `examples/v24_seed.rs:121`); none of that is in this commit.

## 2. e178ab9 checks under a problem's interface: SUPERSEDED

Adds `knob::Under`, `check_legal_under`, `check_text_under`, `rewrites::mutate_under`/`mutate_in`, and `problems/nluca_step/safety.contract`.
Evidence: 6feca92, same patch-id (81e9a83680ef). Tip: `src/wgsl/knob.rs:292` (`Under`), `:306`, `:331`; `src/wgsl/rewrites.rs:260` (`mutate_under`); `problems/nluca_step/safety.contract` present. No difference in this commit's behaviour. No test accompanies it (its message names none).

## 3. 243eeec nluca_step contract.md asks for exactly the nine bindings: SUPERSEDED

One-line CONSTRAINTS change in `problems/nluca_step/contract.md`. 322dc78, same patch-id (61ddc91e4c0c); the sentence "Declare exactly the nine bindings above ..." is at `problems/nluca_step/contract.md:30` at the tip.

## 4. 61f3254 staged constraint and measured deadline: SUPERSEDED

Adds `SuiteSpec::staged`, `Budget::deadline_factor`/`deadline_floor_ms` with validation, `Harness::{staged, reference_eval_ms, deadline_ms, gated_ms}`, cheapest-first sort, per-chunk gate, stop at first failing chunk, slow timing build scored slow; `problems/nluca_step/problem.toml` gets chunk 1, staged, deadline factor 10, floor 60 s.
Evidence: 9df73ca, same patch-id (15925bdcd78a). Tip `src/wgsl/harness.rs`: `staged` at `:984` and `:3266`, `deadline_floor_ms` at `:1011`, `deadline_ms` at `:3420`, the gate at `:3588` to `:3590` (message "the N costlier cases not dispatched"), `:3639` (`!self.staged` projection), `:3707` (`slow = if self.staged`); `problems/nluca_step/problem.toml:226,227,240,241`. Line numbers moved because harness.rs grew (the multi-entry work); behaviour as in the commit.

## 5. cf9dd81 a speed-up below 1 ranks: SUPERSEDED

Adds `SpeedColumn { Reciprocal, Ranked }`, `ProblemSpec::speed_column`, `Harness::columns`, test `a_ranked_speed_column_orders_slower_kernels`; `problems/nluca_step` sets `ranked`.
Evidence: 23a5312, same patch-id (f6e9f59c984d). Tip `harness.rs:1071` (`SpeedColumn`), `:3427` (`columns`), test at `:5055`; `problems/nluca_step/problem.toml:25`.

## 6. 5d664cd problems/gram/contract.md: SUPERSEDED

New file, the Gram's model contract as a problem's. 3a0d79c, same patch-id (eaff7eb6b54a); `problems/gram/contract.md` exists at the tip.

## 7. 7834e99 the reader never reads a loop from 0 that enters elsewhere: SUPERSEDED, with a later behaviour change

Adds the refusal in `safe_reader.rs` for a loop counter entering with a value the reader cannot spell, the `LOOPS` paragraph in `llm::contract`, and `wgsl::roundtrip` (`same_values`, `Trial`, `RoundTrip`) with a corpus test over 631 kernels (592 same, 38 refused, 1 not runnable, 0 differ).
Evidence: 74a8060, same patch-id (be1ce663e068). Tip: `roundtrip.rs:116` (`same_values`), `mod.rs:45` (`pub mod roundtrip`).
Differences since: 24a209e (2026-10-16, "the grid-stride loop") made the reader read `for (var i = e; i < n; i += s)` as a guarded loop from 0 instead of refusing it (`safe_reader.rs:5395` to `:5416`; the refusal now stays only for entry values that are not a thread expression of ids, uniforms and literals). bdf7aa1 rewrote this commit's test into `a_loop_entered_at_a_threads_index_reads_with_its_own_values`, and 20af13a extended the corpus list. One stale item to fix, not a reason to merge: the `LOOPS.` paragraph at `src/wgsl/llm.rs:108` still tells models that `for (var i = lid; ...)` "is not read into the population", which is no longer true since 24a209e.

## 8. 0324cc5 the emitter declares only what a kernel names: SUPERSEDED

Adds `render::names_word`, `register_declarations_for`, and makes the default workgroup arrays conditional.
Evidence: c908733, same patch-id (f9849832581463e16e8d9156fb4e3eaa627d8f90), same author timestamp 07:26:39. Tip `render.rs:1100` (workgroup arrays filtered by `names_word`), `:1135` (`register_declarations_for(&body)`), `:3095`, `:3116`, `:3137`. Commit message: 650 tests pass.

## 9. 2367af2 wip emitter speed work: WIP (partly superseded, partly unmerged)

Three parts, 498 insertions in `examples/wgsl_problem.rs`, `src/wgsl/render.rs`, `src/wgsl/rules.rs`.

a. `examples/wgsl_problem.rs` (`--pairs N`, `--original`, `--shape`, ABBA rounds, bootstrap interval, `within 3 %`, per-shape GPU ms columns): SUPERSEDED by 0d74f08 "wgsl_problem prints GPU ms per timing shape and times kernels in pairs". The files differ by one line: the tip has `.is_multiple_of(2)` where the WIP has `% 2 == 0` (clippy fix, `examples/wgsl_problem.rs:102`).

b. Shared loads (`Scope::{shared, lets}`, `SharedLoad`, `Writes`, `run_items`, `copy_lets`, `shareable_load`, `run_t`, `reads_writes`, `shared_loads`, `share_key`, and the `value()` lookup): written once as `let ld<n>`, for nluca's `generated_step` (1.52x the original at 65,536 rows, 1.15x at 1M). Not on `paired-fitness` (no `shared_loads` in `render.rs`). The same slowdown was fixed there by a different mechanism: 0ade835 binds a memory-reading `let` used more than once to a register in the parser (`safe_reader.rs` `lets_used_in_deeper_loops`), measured 1.007 [0.998, 1.014], 1.000, 1.002 against the original. So this part is OBSOLETE for nluca; it would still catch repeated loads that the source itself writes twice (it does not distinguish), and its test `a_shared_load_ends_where_its_index_register_is_written` plus the changed assertion in `rules.rs` (`let ld1 = wg_va[...]; ... ld1.w`) would conflict with the register-binding fix: the two mechanisms would both fire and the `rules.rs` assertion (`render.rs` text around `rules.rs:2064`) at the tip still expects the unshared `.w` read.

c. Early exit (`Renderer::early_exit`: a `if (!(guard(row0, col0))) { return; }` first line when every store of the thread is a guarded `store.h_safe`/`store.h_mirror` of one origin, no effects): NEEDED, not on `paired-fitness` (`grep early_exit src` is empty). Fixes the Gram's emitted shipped kernel being 1.04x and the blocked one 1.035x the original (every thread of a partial band ran the item loop); 0ade835's message confirms "Gram's emitted shipped kernel is still 1.04x the original (separate cause)" and nothing later on `paired-fitness` addresses it. Would touch `render.rs` `Renderer::kernel` (the `kernel.entry` arm now near `render.rs:1100` to `:1135`) and `Role::guard_text` (`render.rs:255`, used at `:1447`, `:1462`); conflicts: `render.rs` grew by roughly 2,000 lines since the merge base (module AST work, groups 1 to 21, `lit_*` folding), so it needs a hand port, not a cherry-pick. Its test `renderings_share_repeated_loads_and_exit_early_and_read_back` bundles (b) and (c) and asserts `kingdoms/wgsl/gram/{shipped,blocked}.wgsl` exits before the item loop and `same_values` equality; it would need splitting. It states in its own message that it is "not gated": the blocked kernel's read-back is not a fixpoint (test comment `fixpoint=false`), and the exit's soundness argument (origin plus nonnegative offset, guard bounding row and column) has no test beyond the two Gram kernels.

Also unmerged: f5ab515 on branch `emitter-fix` ("emitter-fix agent's uncommitted render/rules edits (stopped 16:47Z), unverified") is a later wip of the same render.rs work (1,114 insertions across 8 files against 2367af2, includes `early_exit` and `shared_loads`). Whatever is decided for (c) should be decided together with that commit, which also touches `safe_reader.rs` and `safety.rs`.

Recommendation: nothing from commits 1 to 8 needs merging; the branch can be retired once (c) is either ported on top of the current renderer, with a Gram paired-timing measurement (`wgsl_problem --pairs`) before and after, or explicitly dropped.
