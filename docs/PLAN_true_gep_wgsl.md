# PLAN: true multi-typed GEP for WGSL

## NOW: the symbol table (the only active task)

Andrew, 2026-10-21: everything except the design of the WGSL kingdom's symbol table is PARKED. GEP itself (typed Karva genes, heads and tails per type, expression, operators, the GPU engine) is standard and already built; the only thing WGSL changes is the kingdom's symbol table: its types, its symbols, and the types of each symbol's arguments and output.

| id | prerequisite | task | owner | gate |
|---|---|---|---|---|
| ST1 | READ the design doc (`docs/MultiTypedGEP.html`, `docs/GOAL.md`) and READ the rules (`Rules.md`) | Design the WGSL symbol table: every type, every symbol (output type, ordered argument types, WGSL spelling), every terminal, the root per interface, and for each legality rule the type that carries it. File: `nucleotable/docs/MultiTypedGEP_s9_WGSL_draft.md`. | advisor | Reviewed for one question: can any combination of rows express an illegal or malformed WGSL kernel? Advisor, then Codex, fixed, Codex again. |

| ST2 | READ design doc + rules; ST1 passes | Generate the table (`gen_tables.py` writes `types.tsv`, `functions.tsv` from the ST1 design); the kingdom check of rev 4 §1 passes | Sonnet | kingdom check green; bound table printed |
| ST3 | READ design doc + rules; ST2 | The first test (Andrew, 2026-10-21): 100 random kernels from the typed initialiser on the GPU; every one renders, compiles (naga and the Metal pipeline), and runs on the right inputs through the problem's unit test without fault | Sonnet | 100 of 100 compile and run; the HFF fitness scores them, evolution finds the answer |
| ST4 | READ design doc + rules; ST3 | Seeding with a weak LLM writing Karva (Andrew, 2026-10-21): Haiku writes Karva symbol sequences, not programs; each cell takes the model's choice as an index into that cell's type alphabet (modulo its size), so any sequence, mistakes included, decodes to a valid gene with no repair; 100 such genes are the initial population, near-remembered and not quite right, for evolution to improve | Sonnet | 100 of 100 valid by construction; the HFF fitness scores them and evolution takes it from there |

Everything below this line is PARKED until ST1 passes its review.

---

## PARKED


## Context

**Objective.** The WGSL kingdom becomes true Gene Expression Programming. Every gene, drawn by the typed initialiser or produced by any variation operator, expresses a legal WGSL compute kernel, because the symbol table's types describe exactly the legal kernels. The renderer spells the expressed tree and nothing else. Fitness runs every expressed kernel. The project is true GEP, incapable of malformed or illegal ASTs.

**The specification.** `nucleotable/docs/MultiTypedGEP.md` rev 4 (rendered as `docs/MultiTypedGEP.html`) and `docs/GOAL.md`. This plan adds the WGSL section to that specification (rev 5, task D1). Every task is built and reviewed against the specification; anything not to it is a failure and is redone.

**Approach.** The WGSL kingdom is instantiated per contract: `K<target, interface, dispatch contract, alias partition>`. The contract (adapter limits, bindings and their ranges and aliasing, workgroup size, dispatch geometry) is validated once on the host before any gene exists; that is input validation, not a gene filter. Within one instantiated kingdom, every legality rule is a type: each rule of `src/wgsl/check.rs` is written into the specification as a type construction with its closure argument (D1), built in dependency order (B), and proved by a property test that also asserts no checker or renderer rewrite is ever invoked (C1). Then the evolution path loses every legality call (C2).

**Unique feature.** WGSL legality (in-range indexing, a thread writing only the cells it owns, race-free workgroup memory, barriers and subgroup operations in uniform control, bounded work, defined arithmetic) is a property of the symbol table of a contract-instantiated kingdom. A legal GPU program is all any gene can express.

**Status.** Revision 5, 2026-10-21: Codex's review of revision 4 (`docs/reviews/2026-10-21_true_gep_plan_codex_rev4.md`, FAIL, two findings) resolved here; the specification draft is `nucleotable/docs/MultiTypedGEP_s9_WGSL_draft.md`. Revision 4: the advisor's review of revision 3 (`docs/reviews/2026-10-21_true_gep_plan_advisor_rev3.md`, FAIL, seven findings) resolved here. Revision 3: Codex's second review (`docs/reviews/2026-10-21_true_gep_plan_codex_rev2.md`, FAIL, five findings) resolved here. Revision 2, after the advisor review (`docs/reviews/2026-10-21_true_gep_plan_advisor.md`) and Codex's (`docs/reviews/2026-10-21_true_gep_plan_codex.md`), and the measurements A1 (`docs/reports/2026-10-21_A1_post_expression_inventory.md`: on the evolution path 1 filter, 10 renderer substitutions, 27 refusals; run 6i 21 % of children judged illegal, of which 292 of 558 were render or parse failures of the `cfg` field rows) and A3 (`docs/reports/2026-10-21_A3_rules_in_types.md`: which rules the types carry today).

**Why.** Evolution can only build on what it can spell. A judgement after expression throws children away, so search sees a cliff where GEP should see a landscape (run 6i: 21 % of children judged illegal; children of good seeds shared 0.6 % of their parent's expression; A2 smoke: 88-98 % of structural-operator children of seeds failed a layer).

## Prerequisite for every task

**READ the design doc** (`docs/MultiTypedGEP.html`, source `nucleotable/docs/MultiTypedGEP.md` rev 4, and `docs/GOAL.md`). **READ the rules** (`~/dev/minkymorgan/qdrant/docs/Rules.md`). State in the task report how the work delivers the true multi-typed design. The lead checks every result against the specification; Codex reviews every bundle against GOAL.md and the specification; if Codex fails, the advisor reviews.

## Conformance (what "done" means)

- **C1 Closure.** For every interface of every contract in use: 1,000,000 genes from the typed initialiser and 1,000,000 children per shared-core variation operator, drawn on the GPU, express trees that render to naga-valid WGSL and pass every layer of `check.rs` run as an oracle; zero failures. The stages are instrumented separately (generation, expression, rendering, dispatch): a renderer `masked` event, a fallback branch, or any rewrite beyond a row's own declared semantics fails the test, and so does any legality call in those stages. `check.rs` runs only afterwards, as the test's oracle on the rendered kernels; a capacity verdict of the oracle (`too_large_to_prove`, `STEP_CAP`) is an oracle defect to fix, never a pass. The renderer is total on expressed trees: every renderer `Err` branch (A1 R2-R4) is deleted or proved unreachable, and an `Err` counts as a failure. The dispatch stage creates the pipeline on the target: a creation failure (A1 H2) is a kingdom bug for that target. Before launch, the measured oracle cost per kernel and the run time are stated; the paged geneframe streams the genes (a million genes at N = 8 is 232 GB).
- **C2 No judgement on the evolution path.** The evolution binary (`phylu/examples/wgsl_fit.rs`, `src/evolve/wgsl_typed_fit.rs`) has no call into `check.rs`, `safety.rs`, `check_legal_under`, the cost estimator or any legality stage. No `compile:` verdict in `harness.rs` on an expressed kernel (A1 H1, H2). No time-based candidate refusal either: the harness deadline, predicted-time rejection and the runaway drain are not verdicts on a kernel. Termination is a type property: the fuel strata (D1 §9) bound every expressed kernel's work to a finite count, so every kernel completes. There is no candidate deadline: elapsed time is a fitness measurement (speed), never a verdict. A device watchdog or driver fault is an infrastructure fault, handled outside scoring (the evaluation is retried; a fault that repeats is reported as a kingdom or driver bug), never a kernel's score. Every child is scored by fitness; an infeasible score measures only how wrong the kernel's output is.
- **C3 Shared core.** WGSL evolves on the shared typed core only; no WGSL-specific variation operator is on the evolution path.
- **C4 Coverage.** Every kernel that is legal under its declared contract reads into that contract's kingdom (the human authoring path of `nucleotable/docs/KingdomMatrix.md`). The reader and encoder read an exact typed tree or report a named outside-kingdom reason; they never rewrite (no `lower_returns_in_loops`, no override defaults, no `conform` conversion or zero substitution). A kernel that is illegal under its contract (a race, an unbounded loop, a non-injective store) is outside the kingdom by design and is listed by reason, the way `tests/fixtures/wgsl_genomes/REFUSED.md` lists invalid text.
- **C5 Evolution.** On a real problem whose seeds read into the kingdom, children resemble their parents and fitness improves over generations.

## The rule ledger (D1 writes the type for each; none is optional)

Every reason of every `check.rs` layer, plus the renderer rewrites A1 found. Starting constructions from the Codex review (§5) and the advisor review; D1 settles each with its closure argument and its cost in expressiveness.

| rule | construction to settle in D1 | expressiveness cost |
|---|---|---|
| contract, target limits, features (layer 5) | `K<target, interface, contract>`: feature rows present only where the target has them; `WgSize<target>`, `WgMem<target,bytes>` declarations compose only within limits | one kingdom per target and contract |
| array range: storage, workgroup, private (layer 3) | `Array<B,E>` and `Idx<B>` produced only by range-preserving rows (literal below the length, masks and clamps as explicit rows, affine forms proved in range); every access consumes `Idx<B>`; the renderer's mask disappears | raw indexing disappears; one index family per binding or length class |
| write ownership, write-set disjointness per output binding (layer 7) | `Own<out,region>` produced only from injective invocation maps generated from the dispatch contract; stores consume it; statement sequences compose only disjoint regions; scatter writes go through atomics | arbitrary and non-affine stores excluded; finite region templates |
| output reads, aliasing, read-after-write of outputs (layer 7) | output loads take `Own` only; `Binding<range, alias class>` from the contract; only declared-disjoint bindings coexist | overlapping bindings excluded |
| workgroup races and ordering (layer 7) | `WgOwn<A,phase>` for writes, `WgRead<A,phase+1>` produced only by barrier rows; no raw workgroup setter | statement and block rows per phase |
| barrier and subgroup uniformity (layer 4) | context-indexed values `Val<T,WgUniform>`, `Val<T,SubgroupUniform>`, blocks `Block<ctx>`; barriers and subgroup operations require their context; delete `uflag.of_flag` and `uniform.of_checked` | value and operation rows per provenance |
| bounded loops and work budget (layer 6, Metal hang) | `Bound<k>` with immutable bound snapshots (a bound held in a `let`); loop bodies carry `Fuel<n>`, nesting consumes a fixed total budget | finite bound and budget strata |
| wrap (layer 3 `index_wraps`, store cell, loop bound) | per §9 draft: wrapping arithmetic is WGSL-defined and its result never reaches an array index, a store cell or a loop bound except through rows that shrink it (mask, clamp, min) | modulo results reach indices only through shrinking rows |
| division, modulo, shifts | per §9 draft: two rows each, with their own spelling as declared meaning (`index.div` vs `div_nz`; `shl_lit` over a 0..=31 literal dual vs `shl` masked); renderer guards disappear | none beyond the declared spellings |
| atomics | the atomic's old value has its own result dual (`u32.atomicold`), consumed only by discard in a strict kingdom; deterministic reductions (sum, min, max, and, or, xor whose final value is order-independent) are separately typed rows, declared by the contract; any other use of an order-dependent result does not exist in the table | order-dependent atomic results not spellable outside a declared relaxed contract |
| typed tree, grammar (layers 1, 2) | already the typed core; the remaining fallback columns of the tables removed | none |
| render and parse (layer 7 `kernel_does_not_render/parse`) | a render defect, not legality: the `load.cfg_*` rows render (A0) | none |
| harness kernel facts (A1 H1: the `// mode` among the problem's listed modes, block factors 1..16, the `// grid` header equal to the workgroup, every binding in the schema) | contract facts of `K<...>`: mode, mirror form (`u32.mirror`), block factors and grid are duals whose rows exist only for the contract's values | one kingdom per listed mode |
| `memory_barrier_without_control` | a storage or workgroup memory barrier exists only inside a control-barrier row | none |
| `knob_reference_unresolved`, duplicate `decls`, `grid_not_in_grammar`, `workgroup_array_not_in_grammar` | a knob reference is a terminal of the knobs that exist; declarations are a typed sequence of distinct arrays; grid and arrays within `WgSize`/`WgMem` | none |
| literals: `nonfinite_literal`, `literal_not_a_row`, `blk` {1,2,4,8}, `code4` (what grid and literal snapping repair today) | each literal dual's `C_T` domain is its `literal` range column, append-only (rev 4 §4); a Dc index names a legal value only | literal values per dual |
| `arena_index_not_in_bounds`, the `--layer7 strict / arena / arena+rows` policy | the arena policy is a contract choice of `K<...>`, named, not a flag | one kingdom per policy |
| `module_entries_not_the_interfaces` | one root dual per interface | none |
| execution budget, device deadline (harness deadline, predicted time, runaway drain) | fuel strata (§9 draft): every expressed kernel terminates after a bounded work count; no candidate deadline, time is fitness | budget strata per target |
| renderer rewrites (A1 §4, ten families: grid, workgroup and literal snapping; fallback leaves; out-of-scope counters; loop caps; index and offset clamps; shift masking; conditional statement erasure; division guards) | each family becomes rows with declared semantics or is removed; the renderer spells the tree only | as the rows above |

## Tasks

| id | prerequisite | task | model | spec and gate |
|---|---|---|---|---|
| A1 | READ design doc + rules | Inventory of post-expression judgements | Sonnet | DONE fuller c8add64b. Add the cost estimator and the runaway drain (advisor). |
| A3 | READ design doc + rules | Rules the types carry today | Sonnet | DONE fuller 0c5848e6. |
| A0 | READ design doc + rules | Remove escape rows and fix render defects before measuring | Sonnet | Delete `uflag.of_flag`, `uniform.of_checked`, the raw `wg.get` / `wg.set` rows (a raw `u32.index` into a workgroup array, `gen_tables.py:1365`) and every unsafe row unreachable from `kernel.safe` (through `gen_tables.py`); fix the `load.cfg_*` render and parse failures (`pcfg.d`); remove the tables' remaining fallback columns. Gate: kingdom check passes; corpora round trips unchanged or better. |
| T66 | READ design doc + rules | Sampler terminal draws and the test generator | Sonnet | A draw ending at a terminal draws uniformly from the type's terminals; the random-gene test generator uses the typed initialiser, never a mask. |
| A2 | — | DROPPED (Andrew, 2026-10-21: stop measuring failure, focus on true GEP). The build is held to D1; C1 is the only measurement. | — | — |
| D1 | READ design doc + rules | WGSL section of the specification (MultiTypedGEP.md rev 5, §9 "WGSL legality as types") | lead, advisor | The contract's execution budget per target. For every row of the rule ledger and every reason A2 reports: the duals, the rows that produce and consume them, the closure argument (why every expressed tree obeys the rule), the expressiveness cost, the rows removed. The contract-instantiated kingdom `K<target, interface, contract, alias partition>`. The renderer as pure spelling. Advisor review, Codex review, both against GOAL.md, before any build. |
| B1 | READ design doc + rules; D1 reviewed | Contract-instantiated kingdom roots and context types | Sonnet | `K<...>` built from a validated contract; target feature rows; the harness kernel facts (mode, mirror form, block factors, grid, binding schema) and the arena policy as contract facts; the uniformity, phase, ownership and fuel contexts as duals. |
| B2 | READ design doc + rules; B1 | Dependent rows: arrays and indices, stores and loads, barriers and subgroups, loops and bounds, arithmetic ladders | Sonnet, sequential sub-steps as D1 orders them | Rows through `gen_tables.py` only; reader maps source text onto them exactly; renderer spells them without rewriting. Each of A1's ten renderer rewrite families is converted or removed in the sub-step that owns its rule, and its renderer branch deleted, listed in the commit. Gate per sub-step: kingdom check; C1 for the layers it closes reaches zero failures and zero renderer events. |
| B3 | READ design doc + rules; B2 | Typed frames and the shared operators on the new kingdom | Sonnet | Layouts, bounds and the device init and vary on the instantiated kingdom; MultiTypedGEP §7 tests 1-6 pass on it. |
| C1 | READ design doc + rules; B3 | The closure property test | Sonnet | As C1 above, on the GPU, under the lock. |
| C2 | READ design doc + rules; C1 green | Remove every legality judgement from the evolution path | Sonnet | Every A1 item on the path removed or moved to a test; the gate is C2 above; Codex review against GOAL.md. |
| C3 | READ design doc + rules | Retire the WGSL-only operators from the evolution path | Sonnet | `wgsl/mutate.rs`, `sections.rs`, `llm.rs` and A1 §6's fourteen OFF items (O1-O13: sampler redraws, the test mask, knob mutation, rewrites, rule sweeps, `legality.rs` fold): each test-only or deleted; none on the evolution path. |
| R1 | READ design doc + rules; B3 | Coverage C4 | Sonnet, sequential | Every kernel legal under its contract reads exactly (the reader's rewrites `lower_returns_in_loops`, override defaults and `conform` conversions removed); the open reader gaps (E, K, J, F3) as rows of the D1 types; kernels illegal under their contract listed by reason. |
| S1 | READ design doc + rules; R1 | Seed conformance | Sonnet | Every V24 seed (`seeds5b` and the run 1 seeds) reads exactly into its contract kingdom with no reader rewrite; every seed that does not is named with its outside-kingdom reason. E1 uses only exactly-read seeds and reports how many. Program size `N` is the smallest at which every exactly-read seed encodes, with no cap that drops a seed. |
| E1 | READ design doc + rules; C2, R1 (C4) and S1 done | Evolution on a real problem, both mutation-rate arms | lead (GPU) | `wgsl_fit` on V24 ingest whose seeds read into the kingdom, run with each rate (per type; one rate per gene over its cells), same seeds and budget; children resemble parents (overlap); fitness improves; both arms reported against C5 for Andrew. |

## Decisions waiting on Andrew

1. **Mutation rate.** Per type (each type domain takes a whole gene's expected edits: about 5,800 cells per WGSL child, run 6i) or one rate per gene spread over its cells (about two expressed edits a child). E1 measures both arms; Andrew decides on the numbers.
2. **The range-check patch** (`scratchpad/check_wrap_final_v2.patch`) changes only the checker, which after C2 is a test oracle; it never goes on the evolution path; §9 carries wrap in the types.

## Parked

The SR, TSR, regex, benchmark and finisher items of `docs/PLAN_outstanding_2026-10-21.md` resume when Andrew says.
