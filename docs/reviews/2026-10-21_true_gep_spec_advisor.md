# Advisor review: plan revision 5 + §9 draft with the lead's decisions, as the specification to build to

FAIL

2026-10-21. Read in full: GOAL.md, MultiTypedGEP.md rev 4, Rules.md, the plan rev 5, the §9 draft with its decisions, the five reviews; checked against `gen_tables.py`, `harness.rs`, `phylu/src/evolve/mod.rs`.

FAIL on two closure arguments wrong as written (findings 1, 2) and one decision unsound as worded (finding 3). All three are text fixes to §9, not a redesign; the specification is buildable once they are in.

## Q2. Codex rev 4

**Finding 1 (deadline): resolved.** Termination is a theorem (`let`-snapshotted caps, §9.2.7; fuel strata, §9.2.8; state the induction over the subtree's function count, which the block case silently uses). No candidate deadline; time is a measurement. `E_target` is a measurement with a margin (decision 8), explicit, outside scoring, stopping the run rather than scoring a kernel. Acceptable against GOAL.md.

**Finding 2 (atomics): half resolved.** The RMW old value is typed (`atomicold`). But `atomic.load_*` still outputs plain `u32.index`/`i32.int` (`gen_tables.py:995`) and §9.2.5 allows atomic rows in every kind. A strict kernel can `atomicLoad` a workgroup atomic mid-phase while others `atomicAdd` it, or an arena atomic with no barrier across workgroups: race-free, order-dependent, and the value reaches a store. The strict closure sentence is false. Fix: in strict kingdoms a workgroup `atomicLoad` is an L-rung statement bracketed by barriers into a register (the `wg.uload` form); an in-kernel arena `atomicLoad` outputs `atomicold` (the harness reads the final value after the dispatch); plain loads exist only under `determinism relaxed`.

**Reopened:** decision 2 reopens A3 3.2 and layer 5's memory bound (finding 3). Nothing else.

## Findings, ranked

**1. High. §9.2.5's race closure is wrong for the clamped own index.** `safe.wg_own_<j>` spells `min(lidx + k * THREADS, 2^j − 1)`. With THREADS = 256 and `j = 8`, every lane at `k = 1` clamps to 255: two invocations write one element in one W phase, and lane 0's "own read" of 255 races lane 255's write. Today's `wg.put` is safe because its text guards `if (i < len)` (`gen_tables.py:1276`). Fix: `wg.set_*_W` and `wg.get_own_*` spell the guard, not the clamp, so the write set is injective by the row's text; §9.6.2 item 7 asserts injectivity over `lidx` for every `k`.

**2. High. Strict atomics (above).** Codex finding 2 stands for loads.

**3. Medium. Decision 2 (buddy-tree of declarations) is unsound as worded.** A gene-carried declaration subtree cannot bind the uses elsewhere in the gene: a use of `wgmem<f32,2048>` is legal wherever its dual is demanded, so a tree can use an array its declarations did not declare (a render failure) or declare a set over the limit. This is why the draft chose the map. The sound form of "evolvable lengths" is still a fixed map: one array per power-of-two stratum per element type, total within `max_workgroup_storage`, declared from the map in the prologue, uses choosing a stratum. State it so; otherwise `workgroup_array_not_in_grammar` and `workgroup_memory_over_limit` have a tree to fire on.

**4. Medium. The store row cannot spell its guard.** `store.<b>_<rung>: (u32.own_<b>, elem)` spells `if (r < R_b && c < C_b) { b[i] = v; }`, but its input is the flattened `r * S_b + c`; `r`, `c` are another node's. That violates §9.4 rule 1 (a row's text depends on no other node) and the reader cannot invert it. Fix: the store takes `(u32.threadrow, u32.threadcol, elem)` and spells cell and guard itself; `u32.own_<b>` remains for `load.<b>_own` only, or becomes the pair. Same for the mirror store. This is the critical-path rule; settle it before B2.

**5. Medium. Decision 12 leaves a row family unspecified and contradicts §9.2.4.** §9.2.4 still specifies `u32.ownrow_<r>` with intra-workgroup collisions "declared". Decision 12 makes ownership per invocation but gives no rows. The sound reading: `policy rows` disappears and becomes a fourth value of `partition`, `lane`: origin `wid.x * W + lidx`, block 1, store guard `lidx < W` in the row's text; §9.2.4 keeps arenas only. Also, OPEN 12 said the seeds decide; decision 12 fixes it before S1 reads the V24 `cands` writers. State the row and let S1 report which seeds fall outside.

**6. Low-Medium. Decision 3 depends on the target, not the row.** `load.<b>` spells `b[i]` and takes its meaning from naga's `Restrict`, which wgpu installs on Metal (`evolve/mod.rs:206`, `harness.rs:2317`). The Runpod Vulkan build may install another policy (zero on read). Either the clamp is the row's text (`b[min(i, arrayLength(&b) − 1u)]`, target-independent, and the reader reads bare `b[i]` as outside the kingdom) or the bounds policy is a `target` line, probed in §9.1.2 item 8. Decision 3 says "spelled in the row's text" while §9.2.2 spells `b[i]`; pick one.

**7. Low. Decision 7 versus §9.4.** `for (var k<d> = 0u; ...)` declares a new `k<d>`, shadowing the kernel-scope counter decision 7 reads after the loop (always 0). Spell `k<d> = 0u; for (; k<d> < n<d>; k<d> += s)`; an unrolled loop never assigns it, say the read after one is 0.

**8. Low. Knob references break renderer purity.** `literal.p_<knob>` computes its value from the grid node at render time (a row's text depending on another node; `lit_u32(37u)` is not readable back). Spell the WGSL expression over a kernel-scope `const` emitted from the grid in the prologue.

**9. Low. Decision 5's second form needs a transitive test.** A restricted W statement list over a shared value family is sound only if no value dual a W statement can reach has a producer reading a foreign element; §9.6.2 item 7 checks direct spellings only.

**10. Low. Plan text lags the spec.** Ledger rows 42 (`U32<lo,hi>` ladder: §9.2.9 has none), 43 (`NonZeroU32`: §9.2.9 uses `div`/`div_nz`), 53 ("within the deadline"), plan decision 2 ("D1's width ladder"), a duplicated sentence in C2.

## Q3 summary

§9.8 assigns every ledger rule a type and a closure argument. Wrong or incomplete: §9.2.5 (1), §9.2.12 (2), §9.2.3's store guard (4). Decisions 2 and 3 make a rule depend on something outside the row; none reintroduces a judgement after expression. Unspecified families: lane-partition rows (5), the stratum map (3), strict atomic loads (2), the W value family (decision 5, measured in B1).

## Q4. B1, in order

B1 first widens `safety::Contract::parse` with the §9.1.1 lines (`writes`, `primary`, `partition` including `lane`, `determinism`, `wgmap` as the stratum map, `budget`, `policy`) and writes `validate(C)` with its eight checks and the feature probes; then makes `gen_tables.py --contract` a function of the contract, emitting the per-entry tables under `kingdoms/safewgsl/<target>/<interface>/<entry>/`; then the roots and contexts in dependency order: the shape table and `grid.of` with `ktile`, the fuel strata over every statement-level dual with the lift and sequence rows and `kernel.entry_safe` at the contract's `K`, the W/WB block kinds (both value-family forms, widths measured), the origin and ownership duals per written binding with the store row taking row and column (finding 4), the stratum map's array terminals and index duals (finding 3), the `atomicold` duals with strict load rows (finding 2); then the fixed module linker per interface; then runs the kingdom check, §9.6.2 items 1, 2, 5, 6, 7 (with the injectivity check of finding 1), and prints the bound table at `N = 8` and `N = 32` for C1's size (decision 11).
