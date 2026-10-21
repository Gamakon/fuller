FAIL

# Advisor review: WGSL kingdom symbol table (§9 draft, commit b5ee1d96)

Read in full: GOAL.md, MultiTypedGEP.md rev 4, Rules.md, the §9 table; WGSL rules checked against the W3C text (section numbers are the spec's). The root is reachable (`kernel.entry_safe` is the only producer; every argument type has a producer). Ten holes, ranked: each gives the tree, the WGSL it spells, and the type change.

## 1. `continue` under a divergent `if` inside a uniform loop: barrier in non-uniform control flow; also skips the epilogue store

`block.cont_C<d>` exists for every `d >= 1`, and `stmt.if_L<d>` puts a `C<d>` block inside an `L<d>` body.

Tree: `loop.tile_L0_f0(lit64 4, block.seq2_L1_f0(stmt.if_L1(flag, block.cont_C1(flag2, own.block_C1)), stmt.barrier_L1))`.
Spells: `for (; k1 < n; k1 = k1 + 1u) { if (flag) { if (flag2) { continue; } {} } workgroupBarrier(); }`.
An invocation whose `flag && flag2` holds skips the barrier on that trip. WGSL §15.2.6: an `if` whose behavior is not `{Next}` yields `CFend -> {CF1, CF2}`, non-uniform when `e` is; the barrier after it fails the analysis (naga 0.20 does not run it; Metal deadlocks).

Same row in the epilogue: `loop.tile_O0_f0(lit64 1, block.seq2_O1(stmt.block_C1(block.cont_C1(lit_bool(true), own.block_C1)), own.stmt_O1))` spells `for (...) { { if (lit_bool(true)) { continue; } {} } <store>; }`: the demanded store never executes.

Fix (type): delete `block.cont_C<d>`. `continue` lives only in the divergent, barrier-free loop bodies: `block.cont_B<d>: (bool.flag, block.C<d>) -> block.B<d>` and `block.cont_WB<d>`. `B` is never an `O` body, and a `tile_brk` loop binds `kx` already. (A `break` is fine where it is: §15.2.6 gives a loop with `Next` in its body result `CF`, so control reconverges after it.)

## 2. Foreign reads leak into W through non-twin argument types: workgroup race

The twin family covers value types, but several W rows take a plain `u32.index`, `bool.flag`, or a helper type built from `u32.index`, and `wg.get_u_10 -> u32.index` is a foreign read.

Tree: `wg.phase_L0(block.one_W0(loop.tilex_6_W0(bound.of_index_6(wg.get_u_10(wg.u_10, safe.wg_of_10(literal.index 7))), block.one_W1(wg.set_u_10_W1(wg.u_10, safe.wg_own_t1_10(k1), literal.index_w 0)))), own.block_L0)`.
Spells: `workgroupBarrier(); let n = min(wg_u_10[7u & 1023u], 64u); kx1 = 0u; for (...) { if (lidx + k1 * KNOB_THREADS < 1024u) { wg_u_10[lidx + k1 * KNOB_THREADS] = lit_u32(0); } } workgroupBarrier();`
With `k1 = 0` (before its loop) invocation 7 writes `wg_u_10[7]` while every invocation reads it for the bound, between the two barriers. Other leaks: `atomic.<fun>_u_8`'s value (`u32.index`) under `stmt.discard_u_W`; `atomic.store_u_8_W`'s value; `safe.pidx_6(u32.index)` in `parr.set_f_6_W`; `block.brk_WB<d>`'s `bool.flag`; `reg.set_v_lane_W`'s lane.

Fix (type): every argument of a W-rung row is a `w` type. Give the helper types W twins produced only from `w` inputs (`safe.pidx_6_w: u32.indexw -> u32.pidx_6w`, `bound.of_index_<c>_w`, `safe.wg_of_<j>_w`), take `bool.flagw` in `brk_WB`/`cont_WB`, `u32.indexw` in the atomic value slots. The kingdom check should assert: no row with a W output has a non-`w` value-bearing input.

## 3. A uniform-typed counter bound under divergent control

`loop.tile_W<d>` binds `k<d+1>` (type `tcounter`, "uniform"), yet it may sit under `stmt.if_W<d>` (divergent `bool.flagw`) or inside the W body of `loop.tilex_W`/`tile_brk_W`.

Tree: `block.seq2_L0(wg.phase_L0(block.one_W0(stmt.if_W0(reg.get_b_w(b0), block.one_W0(loop.tile_W0(lit64 5, own.block_W1)))), own.block_L0), loop.tileu_6_L0(bound.of_uniform_6(uniform.of_tcounter1(k1)), block.one_L1(stmt.barrier_L1)))`.
Spells: `... if (b0) { let n = 5u; k1 = 0u; for (; k1 < n; k1 = k1 + 1u) {} } ... let n = min(k1, 64u); k1 = 0u; for (; k1 < n; ...) { workgroupBarrier(); }`.
`k1` is 5 in some invocations, 0 in others: the second loop runs 5 barriers or none. The same divergent `k1` reaches `wg.uload_L`'s pointer (§17.11.4 requires a uniform pointer).

Fix (type): every loop row at a W or O rung binds `kx<d+1>`; `tcounter<d>` is bound only by `loop.tile_L`/`tileu_L`/`tile_strided_L`. (`safe.wg_own_x_<j>` over `u32.index` stays injective: `lidx + k*T` with `lidx < T` cannot collide across invocations for any `k`, so losing `wg_own_t` in W costs nothing.)

## 4. Two sequential loops redeclare `let n` in one scope

Tree: `block.seq2_L0_f1(loop.tile_L0_f0(lit64 2, own.block_L1), loop.tile_L0_f0(lit64 3, own.block_L1))`.
Spells: `let n = lit_u32(2); k1 = 0u; for (...) {} let n = lit_u32(3); k1 = 0u; for (...) {}`.
WGSL §5: two declarations must not introduce the same name with the same end-of-scope; naga reports a redefinition. Fix (spelling): no `let`; a kernel-scope `n<d+1>` beside `k<d+1>`, `n<d+1> = {0}; k<d+1> = 0u; for (; k<d+1> < n<d+1>; ...)`. No row names `n<d+1>`, so the body still cannot change the trip count.

## 5. Demanded types with no terminal: the kingdom check of §1 fails

Every one of these is demanded by a function and has only function producers, so `tail_T` (length `t_T > 0` by §2.2) has an empty alphabet: `grid.grid` (demanded by the root and `grid.with_passes`), `u32.rowid`, `u32.colid`, `bounds<f32>.ordered`, `bounds<i32>.ordered`, `u32.pidx_6`, `u32.pidx_4`, `u32.boundx<c>`, `u32.boundu<c>`, `bool.uflag`, `u32.wgidx_<j>`, `u32.wgown_<j>`, `u32.lane`, `u32.atomicold`, `i32.atomicold`, `u32.aidx_<a>`, `u32.mirror_<b>`, `block.B<d>`, `block.WB<d>`, and `u32.uniform` on an interface with no u32 field. Fix: one `own.*` terminal each, in the table's own pattern: `own.grid` (the interface's default grid), `own.rowid = gid.y`, `own.colid = gid.x`, `own.bounds = vec2<f32>(0.0, 0.0)`, `own.pidx = 0u`, `own.boundx/boundu = lit_u32(1)`, `own.uflag = lit_bool(true)`, `own.wgidx = 0u`, `own.wgown_<j> = lidx`, `own.lane = 0u`, `own.atomicold = lit_u32(0)` (discarded), `own.aidx = 0u`, `own.block_B/WB = {}`, `own.uniform = lit_u32(1)`. Also §1 demands a terminal for the root type; the table has none for `kernel.safe` (correctly: `t_R = 0`), so the check's wording must exempt a root with `t_R = 0`.

## 6. `atomicStore` at an own element and a read-modify-write in one phase: order-dependent value typed deterministic

Tree: `wg.phase_L0(block.seq2_W0(stmt.discard_u_W0(atomic.add_u_8(wg.au_8, safe.wg_of_8(literal.index 0), literal.index_w 1)), atomic.store_u_8_W0(wg.au_8, safe.wg_own_t1_8(k1), literal.index_w 0)), block.one_L0(atomic.uload_u_8_L0(i0, wg.au_8, safe.wg_of_8(literal.index 0))))`.
Every invocation adds to element 0; invocation 0 (`k1 = 0`) stores 0 to it in the same phase. `i0` is the count or 0 by order, then flows to the output store. Fix (type): atomic RMW rows (`stmt.discard_*`) do not exist at W rungs; a phase holds stores only, and the phase's barriers separate the two.

## 7. `subgroupBroadcast` needs a const-expression lane

§17.12.6: `id` must be a const-expression in `[0, 128)`. `literal.lanelit` spells `lit_u32(n)`, a function call: naga rejects `subgroupBroadcast(f0, lit_u32(3))`. Fix (spelling): `lanelit` spells the bare literal `3u` (a lane has no overflow concern).

## 8. Unbounded kernel text

Rows that spell an argument more than once nest exponentially: `index.sub_floor` (`{0}` twice), `bounds.sorted`, `bounds.of_lo_width(_int)`, `convert.*_nan0` (three times), `load.<i>4` (four times). Tree: `index.sub_floor` nested 90 deep in `head_index` (`h = 93` at `N = 32`) spells `2^90` characters. `loop.unroll` compounds it: four nested unrolls of 64 copy the innermost body `64^4 = 16.7M` times. The Metal compiler crash of 2026-10-20 was a text-size failure of this kind. Fix: every duplicating row becomes a prologue helper (`sub_floor(a, b)`, `sorted(a, b)`, `nan0_u32(x)`, `load4_<i>(k)`), each argument spelled once; the store rows bind `let r = {0}; let c = {1};`. `loop.unroll` takes a `u32.lit8` and raises the stratum by its copy count, or goes (a literal-bounded `for` unrolls in the compiler).

## 9. Grouped partition: const-expression underflow when the block is smaller than `G`

`safe.row_plus_g_<b>` spells `KNOB_BROW / G - 1u`; `G` is the interface's, `KNOB_BROW` the gene's from `{1, 2, 4, 8}`. With `G = 4`, `grid.of(shape, blk 1, blk 1, ...)` spells `const KNOB_BROW: u32 = 1u` and the clamp `min({2}, 1u / 4u - 1u)`: a const-expression that overflows is a shader-creation error (§8.8), and if it compiled the clamp would be `min(x, 0xffffffff)`, leaving the block. Fix (type): on a grouped output `u32.blk_<b>` has domain `{G, 2G, 4G, 8G} ∩ limits`, or the spelling uses `max(KNOB_BROW / G, 1u)` and the base `lid.y * max(KNOB_BROW / G, 1u)`.

## 10. Mode `full`: the mirror store races with the lower owner's own store

`store.<b>_mirror` writes `(c, r)` for an owned `(r, c)`, `r < c`. The invocation owning `(c, r)` still has its unconditional own store and `load.<b>_own` of that cell. Tree: in the body `store.<b>_mirror_L0(own.u32_threadrow, own.u32_threadcol, v)` and in the epilogue `own.block_O0`. Invocation `gid = (1, 0)` writes `b[1 * C + 0]`; invocation `gid = (0, 1)` writes and reads `b[1 * C + 0]` as its own cell, possibly from another workgroup. Fix (type): in a `full` kingdom the own store and own load rows spell the triangle guard `{0} <= {1}` too, so a cell below the diagonal is written by its mirror only.

## Determinism and spelling notes (not counted above)

- §15.7.3: "implementations may assume that overflow, infinities, and NaNs are not present"; a NaN or infinity result is an indeterminate value. The table's "`x / 0` is a value fitness scores" is weaker than it reads: same kernel, same device gives the same bits in practice, but the spec promises nothing, and `u32(NaN)` (§15.7.6) likewise. In a `determinism strict` kingdom only the `nan0` conversions should exist.
- `wg.reduce`: "the result in `a[0]` is the exact reduction" holds only for `n <= 2 * KNOB_THREADS`; above it elements `KNOB_THREADS..s` are never paired. Deterministic, but the note is wrong; make `litpow2` for a reduction at most `2 * KNOB_THREADS`.
- `loop.unroll` rewrites the body's `safe.tile_counter<d+1>` terminals to literals: a spelling that rewrites what the tree says. Emit `k<d+1> = 0u; body; k<d+1> = 1u; body;` instead.
- Referenced but undefined: `load.<b>_mirror`, `own.u32_lit4096`, the `<X>` set of `loop.tile_brk`, which counter `loop.tile_O` binds, `own.block_O0` for a 1-D output.
- The other cited meanings (`clamp`, integer `/` `%`, shifts, `extractBits`, `u32(f)`, i32 wrap) match the spec.
