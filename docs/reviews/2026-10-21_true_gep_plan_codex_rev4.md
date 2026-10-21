# Codex review of the true multi-typed GEP plan, revision 4, 2026-10-21

FAIL

1. **High — the time-refusal fix is still not a proof, so a post-expression refusal can survive.** C2 relies on a “measured worst per-node cost” to guarantee that every legal kernel finishes before the device deadline. A measurement is not an upper bound on GPU execution time: memory contention, cache behavior, thermal/clock state, driver scheduling, and compilation/dispatch overhead are not bounded by a sampled per-node cost. Thus a legal expressed kernel can still reach the deadline/runaway path and be reported rather than scored, contrary to the goal. D1 must define a conservative, target-enforced execution bound covering every row’s dynamic work, memory operations, dispatch geometry, and evaluator overhead—or remove the candidate deadline entirely and isolate infrastructure recovery outside scoring.  
[PLAN_true_gep_wgsl.md:24](</Users/andrewmorgan/Dev/gamakon/fuller/docs/PLAN_true_gep_wgsl.md:24>) · [PLAN_true_gep_wgsl.md:53](</Users/andrewmorgan/Dev/gamakon/fuller/docs/PLAN_true_gep_wgsl.md:53>) · [harness.rs:3613](</Users/andrewmorgan/Dev/gamakon/fuller/src/wgsl/harness.rs:3613>) · [harness.rs:1002](</Users/andrewmorgan/Dev/gamakon/fuller/src/wgsl/harness.rs:1002>)

2. **Medium — atomic determinism remains a placeholder, not a typed construction.** A3 finds atomics race-free but nondeterministic when their returned value feeds a store. The ledger says only “determinism declared per reduction” and “as today”; it does not require the type/capability construction that prevents consumption of an order-dependent atomic result, nor specify permitted reductions and their semantics. That leaves D1 latitude to preserve a rule outside the table or retain nondeterministic kernels without an explicit kingdom definition. Require effect/result types: discard-only atomics by default, with separately typed, contract-declared deterministic reductions.  
[PLAN_true_gep_wgsl.md:44](</Users/andrewmorgan/Dev/gamakon/fuller/docs/PLAN_true_gep_wgsl.md:44>) · [2026-10-21_A3_rules_in_types.md:50](</Users/andrewmorgan/Dev/gamakon/fuller/docs/reports/2026-10-21_A3_rules_in_types.md:50>) · [gen_tables.py:1083](</Users/andrewmorgan/Dev/gamakon/fuller/kingdoms/safewgsl/gen_tables.py:1083>)

All explicit findings from the four earlier reviews are otherwise addressed: contract-indexed kingdoms, exact seed reading and E1 ordering, renderer purity/totality, harness compile gates, ledger omissions, shared-core closure, and the two mutation-rate arms.

The remaining judgement/refusal path is deadline/runaway handling. Renderer rewrites and legality checks are explicitly assigned for removal and independently tested, subject to the two findings above.

