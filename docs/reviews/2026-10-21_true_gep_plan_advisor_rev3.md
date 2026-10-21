# Advisor review: PLAN_true_gep_wgsl.md revision 3

FAIL

2026-10-21. Read in full: GOAL.md, MultiTypedGEP.md rev 4, Rules.md, the plan (rev 3, `220f0323`), the three earlier reviews, the A1 and A3 reports, `check.rs`, `render.rs`, `gen_tables.py`. The reasons of `check.rs` and the rule names of `safety.rs` were re-enumerated from source and match A1 §7 exactly.

FAIL on one surviving post-expression judgement (finding 1) and one order fault (finding 3). Every finding of the three earlier reviews is resolved in the text; the fix is a few lines, not a redesign.

## 1. The earlier findings

All resolved: C4 restated to the contract; `K<target, interface, contract, alias partition>` validated before any gene; the ledger enumerates the layers; renderer rewrites forbidden and the ten A1 families assigned to B2 sub-steps; escape rows deleted in A0; build order B1 → B2 → B3 → C1 → C2 with C1 asserting the typed initialiser, every operator and no rewrite; S1 before E1 and R1 a prerequisite of E1; time-based refusal banned by C2; exact reading; the instrumented C1 boundary; layer 5, the work budget, T66 before A2, the cost estimator and runaway drain in A1, the C2 gate by module, the infeasible band for measured wrongness only.

Two are resolved only in part:

- A0 deletes the two escape rows and the fallback columns but not `wg.get` / `wg.set` (raw `u32.index` into a workgroup array, masked to 2,047 whatever the length; `gen_tables.py:1365` keeps them "for the reader") nor the unsafe rows unreachable from `kernel.safe`. A2 then measures a kingdom B2 will change. Low.
- Decision 2 says the range-check patch is "superseded"; it does not say it is never applied to the evolution path meanwhile.

## 2. Findings, ranked

**1. High. The harness's own refusals survive outside the ledger and outside C2's gate.** A1 H1 (`harness.rs:1686-1760`, `kernel_info`: the `// mode` must be one the problem lists, block factors in 1..=16, the `// grid` header equal to the shader's workgroup, every binding declared by the schema) and H2 (pipeline creation on the target) end an evaluation at `compile:` infeasible without running the kernel. In run 6i they fired seven times, more than layers 2, 4 and 6 together; H1's six were kernels writing mirrors (`// mode full`) on problems that do not list that mode. The ledger has no row: `mode`, the mirror form (`u32.mirror`, A3 7.3) and the problem's listed modes are contract facts of `K<...>`, so a kernel whose mode the problem does not list cannot be spelled. C2's gate does not name `harness.rs` `compile:`; C1's oracle is `check.rs` plus naga, and H2 is a Metal pipeline fault naga does not see (the `as_type` cast). Required: a ledger row for H1's four facts; C1's dispatch stage creates the pipeline on the target, a creation failure being a kingdom bug per target; C2's gate adds "no `compile:` verdict in `harness.rs` on an expressed kernel".

**2. Medium. Ledger rows missing against `check.rs` and `safety.rs` reasons.** Each needs its construction in D1:
- `memory_barrier_without_control` (safety.rs:2463): the ledger's barrier row produces `WgRead` only from barrier rows but does not say a storage or workgroup memory barrier exists only inside a control-barrier row. A1 S10 lists the storage barrier as a renderer erasure today.
- `knob_reference_unresolved`, `grid_not_in_grammar`, `workgroup_array_not_in_grammar` (A3 2.7): a knob id naming no knob, `decls` declaring one array twice, a grid of 1,024 × 2 cut by the renderer. The ledger's `WgSize`/`WgMem` row covers the sizes; the knob reference and the duplicate declaration have no row.
- `nonfinite_literal`, `literal_not_a_row`, the `blk` {1, 2, 4, 8} and `code4` ranges (A3 1.2): the `C_T` domain must be the `literal` range column, append-only per rev 4 §4. This is what S1 and S3 (grid and literal snapping) repair today; the ledger assigns the family to B2 but names no construction.
- `arena_index_not_in_bounds` and the `--layer7 strict | arena | arena+rows` policy (A1 P6): a contract choice today selected by a flag. It belongs in the contract instantiation, named.
- `too_large_to_prove` (`STEP_CAP`): in C1 a false oracle failure on a large gene. C1 must call a capacity verdict an oracle defect, or raise the cap.
- `module_entries_not_the_interfaces` (A3 2.1): one root dual per interface; name it.

**3. Medium. E1 waits on a decision the house rule says to measure.** "Mutation rate: per type or one rate per gene" is parked on Andrew. Both arms are cheap to build and C5 is the measurement. Run both; the numbers decide. As written E1 cannot start, and the plan ends at a question.

**4. Medium. The execution budget is iterations; the deadline is time.** Codex 2.2 asked for a budget that makes every expressed kernel complete within the device deadline. `Fuel<n>` bounds the trip product; wall time is trip product × dispatch size × the per-iteration cost of the expressed body (bounded by `h_T`). D1 must derive the deadline from the budget by a measured worst per-node cost on the target, so the deadline is a consequence of the kingdom and never a verdict. Otherwise P3 (exit code 3) and H10 (runaway drain) remain reachable in principle and C2's "infrastructure fault" sentence is a label on a refusal.

**5. Low. Renderer totality.** R2, R3 and R4 are renderer `Err` paths on an expressed tree (nesting rules repeated from layer 2, workgroup over 1,024, arrays over 32 KB, f16 without `enable f16`, a dual with no value fallback, rows the interface withholds). C1 implies zero `Err` through "render to naga-valid WGSL" but does not say so. State it: after B2 the renderer is total on expressed trees; every `Err` branch is deleted or proved unreachable, and C1 counts an `Err` as a failure.

**6. Low. C1's size.** 1,000,000 genes plus 1,000,000 children per operator, per interface of every contract, through render, naga and seven layers: at eleven interfaces and eight operators about 99 million oracle runs, layer 7 alone milliseconds a kernel. State the measured oracle cost per kernel and the run time before launch. At 58,090 words a gene (N = 8) a million genes is 232 GB; the paged geneframe must stream it.

**7. Low. C3's list.** A1 §6 names fourteen OFF items; C3 names three files. O1 to O9 (sampler redraws, the test mask) and O10 to O13 (knob mutation, rewrites, rule sweeps, `legality.rs` fold) are judgements a later caller could reach. T66 retires O8; list the rest as test-only or delete them.

## 3. Does executing revision 3 deliver the goal?

With findings 1, 2 and 4 written into the ledger and the gates, yes. The construction for the hard rule (ownership of a generic output binding) is the one both reviews asked for, on the critical path in B2 after B1's contexts. Nothing in revision 3 weakens the goal. The one place a rule is still judged after expression is the harness's `compile:` stage (finding 1); the one place a time refusal is still reachable is a deadline not derived from the budget (finding 4).

## 4. Shortcuts, order, missing tasks

- A0 should also take `wg.get` / `wg.set` and the unreachable unsafe rows.
- The H1 facts belong in B1 (contract roots).
- P9 raises `N` only up to `n_cap`; S1 should say `N` is the smallest at which every exactly-read seed encodes, with no cap that drops a seed.
- Decision 1 becomes a measured pair of E1 arms (finding 3).
