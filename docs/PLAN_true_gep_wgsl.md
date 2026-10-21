# PLAN: true multi-typed GEP for WGSL

## Context

**Objective.** The WGSL kingdom becomes true Gene Expression Programming. Every gene, drawn by the typed initialiser or produced by any variation operator, expresses a legal WGSL compute kernel, because the symbol table's types describe exactly the legal kernels. Fitness runs every expressed kernel. The project is true GEP, incapable of malformed ASTs.

**The specification.** `nucleotable/docs/MultiTypedGEP.md` rev 4 (rendered as `docs/MultiTypedGEP.html`) and `docs/GOAL.md`. This plan adds a WGSL section to that specification (rev 5, task D1); it never replaces it. Every task is built and reviewed against the specification. Anything not to the specification is a failure and is redone.

**Approach.** Measure first: which legality rules does the kingdom already carry in its types, and which are still judged after expression? Then move each judged rule into the types, one rule a commit, until a property test shows that a million random genes and a million children per operator express kernels that every legality layer passes. At that point the layers are tests, not filters, and evolution runs on fitness alone.

**Unique feature.** WGSL legality (in-range indexing, a thread writing only the cell it owns, barriers in uniform control flow, bounded loops, defined arithmetic) is a property of the symbol table. A safe GPU program is reachable from any gene by ordinary GEP operators.

**Status.** Written 2026-10-21 after Andrew stopped all work: the WGSL kingdom had been built as typed genes plus a post-expression legality checker (`docs/PLAN_general_safety_layer.md`, `src/wgsl/check.rs`). That checker becomes the oracle for the property test (task C2), then a test only.

**Why.** Evolution can only build on what it can spell. A filter after expression throws away children, so search sees a cliff where GEP should see a landscape (run 6i: 20 % of children judged illegal; children of good seeds shared 0.6 % of their parent's expression).

## Prerequisite for every task

**READ the design doc** (`docs/MultiTypedGEP.html`, source `nucleotable/docs/MultiTypedGEP.md` rev 4, and `docs/GOAL.md`). **READ the rules** (`~/dev/minkymorgan/qdrant/docs/Rules.md`). State in the task report how the work delivers the true multi-typed design. The lead checks every result against the specification; Codex reviews every bundle against the specification (the review prompt quotes GOAL.md and asks "does this deliver true multi-typed GEP?").

## Conformance (what "done" means)

- **C1 Closure.** For every interface of the WGSL kingdom: 1,000,000 genes from the typed initialiser and 1,000,000 children per variation operator express trees that render to naga-valid WGSL and pass every legality layer of `check.rs`. Zero failures. (`MultiTypedGEP.md` §7 item 3, extended to legality.)
- **C2 No filter.** The WGSL evolution path (`phylu/src/evolve/wgsl_typed_fit.rs`, `examples/wgsl_fit.rs`) calls no legality layer. Every child is scored by fitness. The layers run only in tests.
- **C3 Shared core.** WGSL evolves on the shared typed core only; no WGSL-specific variation operator is on the evolution path.
- **C4 Coverage.** Every WGSL compute kernel naga accepts reads into the kingdom (the human authoring path of `nucleotable/docs/KingdomMatrix.md`).
- **C5 Evolution.** On a real problem with real seeds, children resemble their parents and fitness improves over generations.

## Tasks

| id | prerequisite | task | model | spec and gate |
|---|---|---|---|---|
| A1 | READ design doc + rules | Inventory every post-expression judgement in the WGSL evolution path | Sonnet | List every call from phylu `wgsl_typed_fit.rs` / `examples/wgsl_fit.rs` and fuller `src/wgsl` into `check.rs` layers, the harness's legality stages, and any refusal, infeasible score, mask, substitute or fallback-like draw (`render::terminal_leaf`, `own_or_bug`, `safe_karva` mask). File:line, what it judges, when. No code change. |
| A2 | READ design doc + rules | Measure which layer fails, and why, on random genes and children | Sonnet (GPU by lead's lock) | For each interface of `problems/v24_ingest5b`: 100,000 typed-init genes and 100,000 children per operator, every layer of `check.rs` run on each expressed tree; table of failures by layer and reason, with three example trees per reason. This table is the list of rules still outside the types. |
| A3 | READ design doc + rules | Map what the kingdom's types already carry | Sonnet | From `kingdoms/safewgsl` (types.tsv, functions.tsv, gen_tables.py) and `WGSL_COVERAGE.md`: which legality rules are already type structure (masked workgroup/private indices, `own.*` cells, barrier rungs, counted loops) and which are not. Table: rule, carried by (dual/rows), or not carried. |
| D1 | READ design doc + rules; A1-A3 done | Write the WGSL section of the specification (MultiTypedGEP.md rev 5, §9 "WGSL legality as types") | lead | For each rule A2/A3 show outside the types: the type construction that carries it, stated positively (the duals, the rows that produce and consume them, why every expressed tree obeys the rule). Covers at least: in-range index, owned store cell (write-set disjointness as a type of the store index), uniform context for barriers, bounded loops, defined arithmetic (no wrap where it matters). Codex reviews D1 against GOAL.md before any build. |
| B1..Bn | READ design doc + rules; D1 reviewed | One rule a commit: generate the rows (`gen_tables.py`, never hand-edited TSVs), the reader maps source text onto them, the renderer spells them | Sonnet each, sequential | Gate per rule: the kingdom check passes; C1 for that rule's layer reaches zero failures; round trips of the tracked corpora unchanged or better (`wgsl_reader_census`, `REFUSED.md` exact set). |
| C1 | READ design doc + rules; B done | The closure property test | Sonnet | A test that draws the 1,000,000 genes and children per operator of C1 and asserts zero layer failures, run under the GPU lock. |
| C2 | READ design doc + rules; C1 green | Remove every legality judgement from the evolution path | Sonnet | A1's inventory, each item removed or moved to a test; the run scores every child by fitness; Codex review against GOAL.md. |
| C3 | READ design doc + rules | Retire the WGSL-only operators from the evolution path | Sonnet | `wgsl/mutate.rs`, `sections.rs`, `llm.rs`: callers listed; none on the evolution path; kept code is labelled as tools outside GEP or deleted on the lead's word. |
| T66 | READ design doc + rules | Sampler terminal draws and the test generator (Codex bundle 3, #2 #3) | Sonnet | A draw ending at a terminal draws uniformly from the type's terminals; the random-gene test generator draws by the typed initialiser, never by masking. |
| R1 | READ design doc + rules | Coverage C4: the reader reads every naga-valid compute kernel | Sonnet, sequential | The open reader gaps of `tests/fixtures/wgsl_genomes/REFUSED.md` (E barrier under a uniform condition, K uniform-step and fifth tile rung, J struct construction, F3 branch-held row) each as kingdom rows by D1's rules. |
| E1 | READ design doc + rules; C2 done; Andrew's mutation-rate decision | Evolution on a real problem | lead (GPU) | `wgsl_fit` on V24 ingest with its seeds: ledger shows children resemble parents (overlap), fitness improves; report against C5. |

## Decisions waiting on Andrew

1. **Mutation rate.** Per type (each type domain takes a whole gene's expected edits, as now: ~5,800 cells per WGSL child) or one rate per gene spread over its cells (about two expressed edits a child). Measured in run 6i; E1 waits on it.
2. **The range-check patch** (`scratchpad/check_wrap_final_v2.patch`), until D1 moves wrapping arithmetic into the types.

## Parked (not this project's critical path)

The SR, TSR, regex, benchmark and finisher items of `docs/PLAN_outstanding_2026-10-21.md`. They resume when Andrew says.
