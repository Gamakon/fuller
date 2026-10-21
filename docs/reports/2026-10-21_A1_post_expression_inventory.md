# A1: every post-expression judgement in the WGSL evolution path (2026-10-21)

**Context.** Objective: true multi-typed Gene Expression Programming for WGSL (`docs/GOAL.md`, `nucleotable/docs/MultiTypedGEP.md` rev 4, `docs/PLAN_true_gep_wgsl.md`): the symbol table's types describe exactly the legal kernels, so every gene and every variation operator expresses a legal kernel and fitness runs every expressed kernel. Task A1 is an inventory only (no code change, no build, no GPU): every place where a WGSL program is judged, refused, filtered, masked, scored infeasible, substituted or repaired after or during expression, on the evolution path (phylu `src/evolve/wgsl_typed_fit.rs`, `examples/wgsl_fit.rs`) and on the paths that feed it. Task C2 removes each EVO and SEED item (sections 1 to 5) from the path or moves it to a test; task C3 covers section 6; A2 measures which layer fails and why. Read at fuller `890d6c03` and phylu `8c43cd45`. Rules read first: `docs/Rules.md` (no failovers, no hiding bugs, no graceful diversions).

**How to read it.** Kind: **F** filter (a program, seed or candidate is dropped, not admitted, or removed from a candidate set), **R** redraw (a draw is repeated until a condition holds), **S** substitute (a node, value or statement is replaced, clamped or masked by another), **X** refusal (an error, or a score in the infeasible band, instead of fitness). Scope: **EVO** = runs on the evolution path for every scored tree or at initialisation; **SEED** = feeds the path (reading and encoding seeds); **OFF** = WGSL code outside the path (tests, tools, WGSL-only operators), listed because C2/C3 must not leave it callable from the path. "Fires" quotes `logs/V24/run6i_t40/typed_ledger.tsv`: 2,668 scorings (a gene already scored is not scored again), 11 interfaces, generations 0 to 20; stages: feasible 2,103 (78.8 %), legality 558 (20.9 %), compile 7 (0.3 %). No other stage appears. Where the ledger cannot see an item, the cell says so.

## 1. phylu: the evolution path

`wgsl_typed_fit.rs` makes no legality judgement of its own on a tree (it passes the tree to the caller's scorer). Its judgements concern seeds and consistency.

| id | file:line | what it judges | when | what happens | kind | scope | fires |
|---|---|---|---|---|---|---|---|
| P1 | phylu `examples/wgsl_fit.rs:285-286` | the whole post-expression chain `knob::check_legal_under` (section 2): legality layers 1 to 7, render, naga validate | scoring, every expressed tree not already in the memo | the tree is not run; `Scored{fitness: infeasible_fitness(INF) = 2.5708, feasible: false, stage: "legality"}` (the top of the infeasible band) | X | EVO | 558 of 2,668 (20.9 %): by operator seed 22/29, random 22/119 (18.5 %), mutation 121/610 (19.8 %), mutation+crossover 393/1,910 (20.6 %); by interface 12.5 % to 25.0 % |
| P2 | phylu `examples/wgsl_fit.rs:290-297`, `102-122` | the harness verdict (`Fitness.feasible`); an infeasible kernel is graded by `closeness_hff` into [1.5718, 2.5708) | scoring, after P1 passes | scored infeasible in a band above every feasible score; stage named by `stage_of` (`125-149`, naming only) | X | EVO | compile 7 (0.3 %); constraint, slow, pairs, run: 0 |
| P3 | phylu `examples/wgsl_fit.rs:212-221`, `289-291` | an evaluation outliving its deadline (`scorer.deadline_ms`) | during a harness run | the process exits with code 3 naming `current.wgsl` (the GPU cannot be cancelled) | X | EVO | not in the ledger (no exit recorded in this run) |
| P4 | phylu `examples/wgsl_fit.rs:230-263` (`245`, `259-261`) | whether the safe reader reads a saved seed | initialisation | the seed is not read, listed as `# typed seed unread`, never enters a population | F | SEED | not in the ledger (stdout only) |
| P5 | phylu `examples/wgsl_fit.rs:76-98` | the reference kernel reads into the kingdom and binds exactly the problem's bindings; `safety.contract` parses | launch | the launch stops with "our software has a bug" | X | EVO | launch only |
| P6 | phylu `examples/wgsl_fit.rs:407-421` | `--layer7` (strict, arena, arena+rows): which contract layer 7 proves under | launch | switches how strict the layer-7 filter is | policy of a filter (not counted) | EVO | per run |
| P7 | phylu `src/evolve/wgsl_typed_fit.rs:231-249` (`239`) | whether the alphabet can name a seed (`encode_into`): a Gap other than "write-back" | `plan`, initialisation | the seed is "gapped" (not admitted), reported as `# typed seed gap` | F | SEED | not in the ledger (stdout only) |
| P8 | phylu `src/evolve/wgsl_typed_fit.rs:243-248` | all admitted seeds of one interface share one root dual | `plan` | an error, the run does not start | X | SEED | launch only |
| P9 | phylu `src/evolve/wgsl_typed_fit.rs:262-291` | a program size `N` exists at or below `n_cap` at which every admitted seed encodes (write-back) | `plan` | an error naming the gap | X | SEED | launch only |
| P10 | phylu `src/evolve/wgsl_typed_fit.rs:497`, `508`, `516` | the seeds are in the device rows; each seed expresses | island start | an error ("our software has a bug"), the run does not start | X | SEED | launch only |
| P11 | phylu `src/evolve/wgsl_typed_fit.rs:572` | every row of every generation expresses on the host (`express`) | every generation, every row | an error aborts the whole run (no row is skipped or replaced) | X (assertion) | EVO | 0 |
| P12 | phylu `src/evolve/wgsl_typed_fit.rs:573-576` | the device walk and the host express agree on the node count | generation 0 and every 10th generation | an error aborts the run | X (assertion) | EVO | 0 |
| P13 | phylu `src/evolve/wgsl_typed_fit.rs:605-625` | a gene already scored (memo) or an elite with a fitness | every generation | not scored again; the earlier verdict is reused (a score cache, no judgement) | not a judgement | EVO | children equal to a parent are not rescored |
| P14 | phylu `src/evolve/wgsl_typed_fit.rs:611-617` | the run file `killme.trigger` | before every scoring | the generation is left unscored and the run ends | not a judgement | EVO | stop only |
| P15 | phylu `examples/wgsl_fit.rs:451-477`, `526-530` (`--typed-seed-check`, `vary_probe`) | `check_legal_under` on seeds and on children of each operator, counted "legal" | diagnostic runs, no scoring | counts only | X (diagnostic) | OFF | tool |

The phylu typed core that makes the generations (`typed_init.rs`, `typed_vary.rs`, `typed_device.rs`, `typed_walk.rs`, `wgsl_typed_stage.rs`, `constants.rs`) was searched for reject, retry, redraw, substitute, fallback, repair, skip: none found for WGSL. Its walk defects are "our software has a bug" errors (`typed_walk.rs:41-42, 508, 724`), never repaired. `typed_decode.rs:120` (`DecodeOutcome::NotExpressed`, typed-depth ladder and stream order) is a refusal of the SR kernel only; `wgsl_typed_fit` passes `typed_depth: None`, so it is not reached by WGSL.

## 2. fuller: what `check_legal_under` runs (one call per scored tree)

`knob.rs:308-319` is the single entry from phylu. Order: `params::to_literals`, `Checker::check` (layers 1 to 7), `render_tree_with`, `validate_text`. The first failure wins and is the ledger's `detail`.

| id | file:line | what it judges | when | what happens | kind | scope | fires |
|---|---|---|---|---|---|---|---|
| K1 | fuller `src/wgsl/knob.rs:310`, `params.rs:226-233` | knob references written as literals ("no literal row") | scoring | `Stage::Legal` failure, tree not run | X | EVO | 0 |
| K2 | fuller `src/wgsl/knob.rs:315`, `check.rs:405-466` | `Checker::check`: layers 1 to 7 (section 4 lists every layer and reason). A module of several entry points is checked entry by entry (`check.rs:432-455`) | scoring | `Stage::Legal` failure, tree not run | X | EVO | 558 (the whole of P1's legality count) |
| K3 | fuller `src/wgsl/knob.rs:316`, `safe_karva.rs:1024-1036`, `render.rs` | `render_tree_with`: the renderer's own refusals (R1 to R4, section 3) | scoring | `Stage::Render` failure, tree not run | X | EVO | 0 as `render`; the same refusals fire inside layer 7 (see layer 7 below) |
| K4 | fuller `src/wgsl/knob.rs:317`, `render.rs:4629-4632`, `target.rs:93-125` | `validate_text`: naga parse and validation for the device (`Target::current`): workgroup size per axis and invocations, workgroup memory per entry point | scoring | `Stage::Validate` failure, tree not run | X | EVO | 0 as `validate` |
| K5 | fuller `src/wgsl/check.rs:469-488` | layer 7 renders the tree and parses the text before proving it (`kernel_does_not_render`, `kernel_does_not_parse`) | inside K2 | `Layer::Safety` failure | X | EVO | render 120, parse 172 (52.3 % of all legality failures) |

## 3. fuller: the renderer (`render.rs`) on every scored tree

The renderer runs inside K3 and inside layer 7 (K5), so every item here acts on every scored tree. The renderer does not report substitutions to the scorer: `Rendered.masked` (`render.rs:1032`) is returned but `knob::check_legal_under` drops it, so none of the S rows has a count in any ledger.

| id | file:line | what it judges | when | what happens | kind | scope | fires |
|---|---|---|---|---|---|---|---|
| R1 | `render.rs:2963` | a row name is a row of the function table | render | `Err("... is not a row of the function table")`; the 120 ledger cases are `load.cfg_m`, `load.cfg_d`, `load.cfg_r_rows` | X | EVO | 120 (21.5 % of legality failures) |
| R2 | `render.rs:1641`, `1776`, `1797`, `1704`, `1715`, `1744`, `1759`, `2046`, `2056`, `2083` | a fifth nested tile loop, an item loop inside an item loop, a tile loop inside a tile loop | render | `Err("... the safe grammar does not produce this")` (the renderer repeats layer 2's nesting rules) | X | EVO | 0 (layer 2 stops them first) |
| R3 | `render.rs:3153`, `3181`, `3175`, `3201`, `3220`, `3227`, `3237`, `3241`, `1475`, `2175`, `3138` | non-finite literal, a literal with no text, f16 fallback without `enable f16`, a dual with no value fallback, workgroup over 1,024 threads, workgroup arrays over 32 KB, a composite with no WGSL body | render | `Err(...)` | X | EVO | 0 |
| R4 | `render.rs:1810`, `2828-2829`, `2888`, `2948-2951`, `3805`, `3813` | rows the interface withholds (`store.h_safe` with no output, `load.phi_safe` with no input), a workgroup array used as a value, helpers without a scalar | render | `Err("our software has a bug: ...")` | X | EVO | 0 |
| S1 | `render.rs:1109-1112`, `1065-1090` (`grid_of`, `snap`) | the grid's block factors | render | block factors snapped to {1, 2, 4, 8}; a non-finite value is 0 | S | EVO | no log |
| S2 | `render.rs:1055-1064` (`workgroup_dim`, `workgroup_shape`) | workgroup extents | render | clamped to 1..=1,024; y cut to what fits beside x (64 x 32 becomes 64 x 16) | S | EVO | no log |
| S3 | `render.rs:1524-1530` (`lit`) | a bounded literal (`literal.lit64`, `lit4096`): count, unroll, stride | render | rounded and clamped into 1..=max; non-finite becomes 1 | S | EVO | no log |
| S4 | `render.rs:2533-2567`, `3568-3583` (`own_fallback`), `3209-3243` (`fallback_value`), `render.rs:29-65` (`own_leaf`, `terminal_leaf`) | an `own.*` terminal where a value is demanded | render | written as the thread's own row, column, cell or block origin, or the dual's neutral constant or zero (a fallback leaf is a real row, but it stands in for a draw that had nothing) | S | EVO | no log |
| S5 | `render.rs:1274`, `2725`, `2734`, `2882-2883` | a loop counter read outside its loop | render | written as the constant `lit_u32(0u)` | S | EVO | no log |
| S6 | `render.rs:2715-2718` | a computed loop bound (`bound.of_index`, `bound.of_uniform`) | render | wrapped in `min(bound, MAX_LOOP_TRIP = 2^24)` | S | EVO | no log |
| S7 | `render.rs:1259-1262`, `2109`, `2631`, `3533-3560` | an array index (workgroup, private) | render | masked with `&` (power of two) or `min` into the array's length | S | EVO | no log |
| S8 | `render.rs:2746-2800`, `2839` | offsets into a block, a group, a matrix column | render | clamped (`min`) into the block factor or the column count | S | EVO | no log |
| S9 | `render.rs:2992-2996` | a shift count | render | masked `& 31u` | S | EVO | no log |
| S10 | `render.rs:1862-1864`, `1963-1964`, `1994-1995`, `2019-2020`, `2027-2028`, `2038-2039` | a barrier, a workgroup put, a reduction, a subgroup or storage barrier in non-uniform control flow (`in_cond > 0` or after a conditional return) | render | the statement is replaced by a comment or by the unbarriered store; `scope.masked` records it and the scorer discards the record. Layers 2 and 4 normally stop the tree first | S | EVO | no log |

## 4. fuller: the harness (`harness.rs`, called as `scorer.fitness`)

Correctness against the reference and speed are the fitness (`GOAL.md` item 6). Rows H1 to H3 are legality gates ahead of that measurement; H4 to H9 are inside it and are listed because each ends an evaluation with the infeasible band instead of a measurement.

| id | file:line | what it judges | when | what happens | kind | scope | fires |
|---|---|---|---|---|---|---|---|
| H1 | `harness.rs:1686-1760` (`kernel_info`) | naga parse and `Target::validate`; the entry points exist; `// grid` header equals the shader's workgroup; block factors in 1..=16; `// mode` is listed by the problem (`1714`); every `@group(0)` binding is declared by the schema, of the declared kind and element | `Harness::evaluate` start (via `compile`) | `compile:` infeasible, graded `closeness::WORST` | X | EVO | 6 (mode "full" not listed by the problem) |
| H2 | `harness.rs:3110-3146` | device pipeline creation errors (`ErrorScopes`), timestamp queries present | `compile` | `pipeline:` error becomes `compile:` infeasible | X | EVO | 1 (`as_type` cast, Metal) |
| H3 | `harness.rs:3410-3417` | the reference kernel is feasible on its own problem | `Harness::new` | launch error | X | EVO | launch only |
| H4 | `harness.rs:3583` | the problem has a constraint output | evaluate | "our software has a bug" infeasible | X | EVO | 0 |
| H5 | `harness.rs:3598-3601`, `3655` | correctness cases (constraint): any failing chunk ends the cases | cases | `constraint n/N cases: ...` infeasible, graded by `closeness` columns | X (fitness) | EVO | 0 |
| H6 | `harness.rs:3615`, `3650`, `3662`, `3678`, `3712`, `3719` | predicted time over the budget (cases, rungs, pairs, timing shapes) | before each chunk, rung, pair, timing shape | `slow: predicted ...` infeasible, the dispatch is not made | X | EVO | 0 |
| H7 | `harness.rs:3620-3633`, `3669-3692` | a run error (executor); a judge/compare bug | each dispatch | `run:` or "our software has a bug" infeasible | X | EVO | 0 |
| H8 | `harness.rs:3703-3706` | accuracy pairs: max error against the shipped kernel's, non-finite count | pairs | `pair ...` infeasible | X (fitness) | EVO | 0 |
| H9 | `harness.rs:3756-3830` (`evaluate_pooled`) | the same compile, run and error stages for pooled-recall problems | evaluate | infeasible with `compile:` or `run:` | X | EVO | the V24 ledgers use this or the stages above; only compile shows |
| H10 | `gram.rs:65-90` (`wait_gpu`) | a kernel still running when a wait gave up | after a dispatch | named `runaway:`, the GPU is drained, infeasible | X | EVO | 0 |

## 5. fuller: seed reading, alphabet and encoding (feed the path)

| id | file:line | what it judges | when | what happens | kind | scope | fires |
|---|---|---|---|---|---|---|---|
| A1 | `typed_gene.rs:46-58` (`own_size`) | the alphabet keeps only the module row whose entry-point count equals the interface's | alphabet build, once per interface | other `kernel.module_N` rows are not symbols of this interface (a table restriction at construction, not after expression) | F | EVO | per interface |
| A2 | `typed_gene.rs:77` | the kingdom check of the interface's rows (`relations_from_rows(...).check()`) | alphabet build | `Gap "kingdom check"`, run does not start | X | EVO | launch only |
| A3 | `typed_gene.rs:232-256` (`typed_of`), `141-160` (`const_bits`) | a read node is a row of the alphabet at its exact signature; constants fit their dual | seed encoding | `Gap "no row"` or constant gap: the seed is not encodable (feeds P7) | X | SEED | not in the ledger |
| A4 | `typed_gene.rs:259-280` (`encode_into`), `283-308` (`encode`) | a read tree fits the layout (`j_T <= h_T`, tails, Dc) | seed encoding | `Gap "write-back"`; `plan` then raises `N` until it fits (P9) | X | SEED | not in the ledger |
| A5 | `typed_gene.rs:310-325` (`express`) | the host expression of a gene | every scored row | `Gap "express"` becomes P11's abort | X (assertion) | EVO | 0 |
| A6 | `safe_reader.rs:6790-6800` (`read_safe`), `6903-6944` (`read_safe_literal_of`), `reader::read_with`, `read_entry` (ownership proof) | whether a text reads into the safe grammar | seed reading | refusal with a reason string; `tests/fixtures/wgsl_genomes/REFUSED.md` lists the exact set: 48 not valid WGSL (A parse error 33, B undeclared identifier 6, C reserved word 5, D naga 1, F1 header mismatch 3) and 9 open reader gaps (E barrier under a condition, F3 branch-held row, K loop step or depth) | X | SEED | seeds only; counts in REFUSED.md |
| A7 | `safe_reader.rs:6882-6895` (`read_safe_literal`) | a text that does not read and has a return inside a loop | seed reading | read again with returns lowered (`lower::lower_returns_in_loops`): the program is rewritten before reading | S | SEED | not logged |
| A8 | `safe_reader.rs:6828-6872` (`with_overrides`) | an `override` with no default | seed reading | declared with a seed value (64 for sizes, 1 otherwise) | S | SEED | not logged |
| A9 | `safe_reader.rs:7250`, `safe_karva.rs:727-960` (`conform`) | every node under the dual its parent's row demands | end of `read_entry` | inserts conversions and substitutes nodes to make the read tree row-consistent; `Err` if none exists | S, X | SEED | not logged |
| A10 | `canonical.rs:549`, `671` | `canonical` uses `conform` and a zero substitute | canonical forms (tools) | node replaced by a conformed or zero node | S | OFF | tool |

## 6. fuller: WGSL-only code outside the evolution path (C2/C3 must keep it out)

Searched callers: phylu uses `fuller::wgsl` only through `wgsl_fit.rs` and `wgsl_typed_fit.rs` (grep of `src` and `examples`). The items below are therefore not on the path today; each is a judgement, filter or redraw that C3 labels as a tool outside GEP or deletes.

| id | file:line | what it judges | when | what happens | kind | scope | callers |
|---|---|---|---|---|---|---|---|
| O1 | `render.rs:3917-3960`, `4581-4630` (`sample_within`, `sample_sketch`) | the sampler's cost bound: while `cost(tree)` is over the bound, the costliest top-level statement or hole is drawn again | random kernel draw | up to 256 rounds; a site drawn again up to 16 times, then replaced by `terminal_leaf` (`4616-4617`); the epilogue store is drawn at the depth cap | R, S | OFF | tests, `sketch.rs`, `examples/wgsl_cost_census.rs` |
| O2 | `render.rs:4398-4406`, `4454-4463` | a comparison of a value with itself | sampler | second operand drawn again up to 8 times, then the whole node drawn again | R | OFF | same as O1 |
| O3 | `render.rs:4411-4418`, `4466-4473` | a `wg.put` whose value reads its own array | sampler | value drawn again up to 8 times, then the node drawn again | R | OFF | same as O1 |
| O4 | `render.rs:4284-4300` | an epilogue store's cell differs from the kernel's cell | sampler | the node drawn again at the depth cap | R | OFF | same as O1 |
| O5 | `render.rs:4347-4350` (`functions_kept`, `terminals_kept`; `nests` `4119`, `allowed` `4217`, `no_cont`) | rows allowed at this position (no nested long loop, no `continue` in a conditional of a loop body, no subgroup row where not allowed, literal-only, no read of `h` outside a store) | sampler | rows removed from the candidate set before the draw (a mask) | F | OFF | same as O1 |
| O6 | `render.rs:4326-4331` (`own_or_bug`), `4338`, `4429` | a demand with no row | sampler | the dual's `terminal_leaf` is returned and the bug recorded (`self.bug`), the tree meanwhile discarded | S, X | OFF | same as O1 |
| O7 | `render.rs:4706` | a leaf for a demand | test helper | `terminal_leaf`, else the first terminal row with literal 0.0 | S | OFF | tests |
| O8 | `safe_karva.rs:1463-1490` (`mask`, inside `#[cfg(test)] mod tests` at line 1087) | an href whose slot dual is not the demand; a token with no signature; a number | the test generator of random chromosomes | replaced by `terminal_leaf(demand)` | S | OFF | test only |
| O9 | `safe_karva.rs:1073-1085` (`rebuild_bounded_with`), `976-1015` (`project`), `1065-1068` (`default_bound`) | static cost over `640 x shipped` (per item) or 20,000 (base): "not dispatched"; a token with no row at its position | Karva rebuild (earlier path) | `Err("cost: ...")` or a named bug | X | OFF | `sketch.rs` tests, `render.rs:4906` test, `safe_reader.rs` tests, `examples/wgsl_cost_census.rs`; not called by `wgsl_fit` |
| O10 | `knob.rs:163-196` (`joins`), `204-222` (`legal_values`), `235-245` (`mutate`) | a knob mutation's values: workgroup within the adapter, block registers, exact references, then `check_legal_under` on each | WGSL-only knob mutation | values that fail are removed from the candidate set; "no knob of this kernel has a legal new value" is an error | F, X | OFF | `rules_more.rs:475, 828, 1541`, tests |
| O11 | `rewrites.rs:271`, `300` | a rewritten child through `check_text_under` (check, render, validate, read back) | WGSL-only rewrite | child kept only if it passes | F | OFF | `rewrites.rs` |
| O12 | `rules.rs:2004`, `rules_more.rs:1727`, `1810` | rule application sites filtered by `check_legal_under` | rule sweeps (tests) | sites whose tree fails are skipped | F | OFF | tests |
| O13 | `legality.rs` (`decide`), `chromosome.rs:185`, `206` | whether occurrences of a repeated expression may share one definition (fold) | WGSL fold | `Refuse { reason }` per site | F | OFF | `chromosome.rs` |
| O14 | `mutate.rs` (header), `sections.rs`, `llm.rs` | oracle-gated algebraic rewrites; shared sections; model replies | WGSL-only operators (C3) | gated by the numerical oracle or by the reader | F, X | OFF | listed by C3 |

## 7. The check.rs legality layers and reasons in full

Order of execution is `check.rs:405-466`: layers 1, 2, 3, 4, 5, 6, 7 (layer 4 computes the non-uniform set layer 6 reads). Counts are the ledger's 558 legality failures; "-" means the reason did not occur.

| layer | name (fn, line) | reason | meaning | ledger |
|---|---|---|---|---|
| 1 | typed tree (`typed`, `check.rs:500`) | `unknown_dual` | a node's type is not a dual of the kingdom | - |
| 1 | | `literal_with_children` | a literal row has children | - |
| 1 | | `nonfinite_literal` | a literal constant is NaN or infinite | - |
| 1 | | `literal_not_a_row` | a literal name is not a row at its dual | - |
| 1 | | `unknown_row` | a node's row is not in the kingdom or the role (e.g. `store.index_f..`, a store row the interface's alphabet lacks) | 48 |
| 1 | | `signature_mismatch` | a row's children do not have the row's input duals | - |
| 1 | | `no_own_terminal` (module root, `check.rs:447`) | a module's dual has no `own.*` terminal | - |
| 2 | structural grammar (`grammar`, `check.rs:533`) | `root_not_safe_kernel` | the root is not `kernel.entry_safe` (or a module of entry points) | - |
| 2 | | `module_entries_not_the_interfaces` | a module of N entry points under an interface of M | - |
| 2 | | `knob_reference_unresolved` | a knob reference cannot be resolved against a grid | - |
| 2 | | `grid_not_in_grammar` | the grid row is not a grid the renderer writes | - |
| 2 | | `workgroup_array_not_in_grammar` | the workgroup array plan cannot be built | - |
| 2 | | `break_outside_loop`, `continue_outside_loop` | a break or continue not inside a loop | - |
| 2 | | `block_not_in_safe_grammar`, `statement_not_in_safe_grammar` | a block or statement row the safe grammar does not spell | - |
| 2 | | `barrier_under_condition` | a barrier inside an `if` | - |
| 2 | | `loop_not_bounded_constructor` | a loop row of the unsafe kingdom | - |
| 2 | | `nested_item_loop` | an item loop inside an item loop | - |
| 2 | | `tile_depth_exceeded` | more than four tile loops deep | - |
| 3 | range (`range`, `check.rs:546`) | `index_out_of_range` | a workgroup or private array index not in its array's range (by mask or by its value at the worst shape d 8,192, m 65,536, 8,192 rows) | - |
| 3 | | `store_cell_wraps` | `u32` wrap in a store's cell index | - |
| 3 | | `index_wraps` | `u32` wrap in index arithmetic elsewhere | 19 |
| 4 | barrier uniformity (`uniformity`, `check.rs:554`) | `barrier_in_nonuniform_control_flow` | a barrier where a condition, loop trip count, break/continue guard or earlier conditional return is not workgroup-uniform | 4 |
| 5 | adapter (`adapter`, `check.rs:577`) | `workgroup_size_over_limit` | workgroup extents or invocations above the target's | - |
| 5 | | `workgroup_memory_over_limit` | workgroup arrays above the target's storage | - |
| 5 | | `feature_missing` | a row needs a feature the target lacks (e.g. `literal.real of f16.real: naga 0.20 has no f16`) | 25 |
| 6 | bounded loops (`bounded`, `check.rs:635`) | `loop_bound_wraps` | the loop's bound arithmetic wraps `u32` or the counter steps past `u32::MAX` | - |
| 6 | | `loop_bound_not_uniform` | the bound is not a literal, a binding, a value known at the worst shape, or uniform | - |
| 6 | | `loop_bound_changes_in_body` | the bound reads a register or array the body writes | - |
| 7 | write sets and ordering (`safety`, `check.rs:469`) | `kernel_does_not_render` | the tree does not render (R1 to R4) | 120 |
| 7 | | `kernel_does_not_parse` | the rendered text does not parse in naga (e.g. `invalid field accessor 'd'`) | 172 |
| 7 | (rules of `safety::check_module`, `safety.rs:3615`) | `write_sets_overlap` | two invocations may write one output cell | 8 |
| 7 | | `output_read_of_another_invocations_cell` | an invocation may read a cell another invocation writes | - |
| 7 | | `workgroup_memory_race` | two invocations of a workgroup may touch one element with no barrier between | 120 |
| 7 | | `barrier_in_nonuniform_control` | a barrier or `workgroupUniformLoad` where invocations may disagree on reaching it (`safety.rs:2469, 2495`) | - |
| 7 | | `subgroup_operation_in_nonuniform_control` | a subgroup operation in non-uniform control flow (`safety.rs:2511`) | 8 |
| 7 | | `memory_barrier_without_control` | a memory barrier with no control barrier (`safety.rs:2463`) | - |
| 7 | | `store_index_not_affine` | a store index is not an affine form of thread coordinates, loop counters and clamped digits (`safety.rs:2172`) | 16 |
| 7 | | `store_index_may_wrap` | store index arithmetic may wrap at the contract's ranges (`safety.rs:2171`) | - |
| 7 | | `store_stride_may_be_zero` | a thread coordinate's stride is not proved at least 1 (`safety.rs:3726`) | - |
| 7 | | `index_not_in_bounds` | an index is not proved below `arrayLength` (`safety.rs:3743`) | - |
| 7 | | `arena_index_not_in_bounds` | an arena index is not proved in bounds, strict rules kept (`safety.rs:3711`) | 18 |
| 7 | | `statement_not_in_kingdom` | a ray query, image atomic or cooperative store (`safety.rs:2548, 2554`) | - |
| 7 | | `too_large_to_prove` | more than `STEP_CAP` statements walked (`safety.rs:3682`) | - |
| 7 | | `interface`, `parse` | no compute entry point; the text does not parse (`safety.rs:3621, 3637, 3910, 3980`) | - |

Ledger total by layer: 1: 48, 2: 0, 3: 19, 4: 4, 5: 25, 6: 0, 7: 462 (render 120 + parse 172 + race 120 + arena 18 + affine 16 + overlap 8 + subgroup 8). Sum 558. Layers 2 and 6 never fire in this run: the types already carry those rules, or the layer 7 render/parse failure comes first. Three causes make up 412 of the 558 (74 %): the render refusal of R1 (the `cfg` field rows), the naga parse failure on the same field names, and the workgroup race.

## 8. Counts by kind

Counted from the kind and scope columns of sections 1 to 6 (a row with two kinds counts once in each; P6, P13 and P14 are not judgements and are excluded). Layer reasons of section 7 are not counted again: they are the inside of K2.

| kind | EVO (every scored tree or initialisation) | SEED (feeds the path) | OFF (outside the path) | EVO + SEED |
|---|---|---|---|---|
| F filter | 1 (A1) | 2 (P4, P7) | 6 (O5, O10, O11, O12, O13, O14) | 3 |
| R redraw | 0 | 0 | 4 (O1 to O4) | 0 |
| S substitute | 10 (S1 to S10) | 3 (A7, A8, A9) | 5 (O1, O6, O7, O8, A10) | 13 |
| X refusal | 27 | 7 | 5 (O6, O9, O10, O14, P15) | 34 |

Of the 34 EVO + SEED refusals, 21 act on a scored tree (P1, P2, P3, K1 to K5, R1 to R4, H1, H2, H4 to H10) and 13 are launch or assertion checks that stop a run without scoring a tree (P5, P8, P9, P10, P11, P12, H3, A2, A3, A4, A5, A6, A9).

What the ledger shows (run 6i, 2,668 scorings): 20.9 % of scored trees end at P1 before the harness runs them; 74 % of those are R1/K5 (render and parse of the same `cfg` field names) plus the workgroup race; layers 2 and 6 never fire. The S rows act silently on every tree and are not counted by any log, so how often each one changes a program's meaning is unmeasured (A2 measures layers; a count of S rows that fire needs `Rendered.masked` and the renderer's rewrites surfaced, which is a code change outside A1).
