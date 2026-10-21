# Codex review of the WGSL symbol table (5853f666), 2026-10-21

FAIL

1. **Advisor Hole 2 remains open.** `atomic.store_u_8_W` still takes plain `u32.index` (draft line 286), contradicting the claimed all-`w` W-input rule. Tree:

`wg.phase_L0(block.seq2_W0(wg.set_u_10_W(wg.u_10, safe.wg_own_x_10(literal.index_w(0)), literal.index_w(0)), atomic.store_u_8_W(wg.au_8, safe.wg_own_x_8(literal.index_w(0)), wg.get_u_10(wg.u_10, safe.wg_of_10(literal.index(7))))), own.block_L0)`

It spells a W phase where invocation 7 can write `wg_u_10[7]` while others read it as the atomic-store value, without an intervening barrier: a workgroup data race. Change that final argument to `u32.indexw`; apply the analogous `i32.intw` correction to i32 atomic W stores.

2. **Advisor Hole 9 reopens in `grid.of_rows_x`.** Its signature remains `(grid.shape, u32.blk_col, u32.blk_row, …)`, but its spelling swaps them into `KNOB_BROW = bx`, `KNOB_BCOL = by`. For a grouped row axis with `G = 4` and an ungrouped column axis, this tree is legal by the table:

`grid.of_rows_x(literal.shape, literal.blk_col(1), literal.blk_row(4), own.u32_lit4096)`

Together with `safe.row_plus_g_<b>`, it spells `KNOB_BROW / 4u - 1u` with `KNOB_BROW = 1u`: a constant-expression underflow/shader-creation failure. Make `grid.of_rows_x` take `(grid.shape, u32.blk_row, u32.blk_col, …)`, or define orientation-specific block-factor types.

3. **New hole: subgroup shuffle offsets lack a uniform type.** `lane.of_index(builtin.subgroup_invocation_id)` produces `u32.lane`, and `subgroup.shuffle_{down,up,xor}` accept that type. Tree:

`subgroup.shuffle_xor_f_L0(reg.f0, literal.real(1), lane.of_index(builtin.subgroup_invocation_id))`

Spells:

```wgsl
f0 = subgroupShuffleXor(lit_f32(1.0), sg_lane & (sg_size - 1u));
```

The mask differs per lane. WGSL triggers `subgroup_uniformity` (default error) and specifies an indeterminate result when the mask is not subgroup-uniform. [WGSL §17.12.12](https://www.w3.org/TR/WGSL/#subgroupShuffleXor) Change the signatures of `shuffleDown`, `shuffleUp`, and `shuffleXor` to accept a bounded `u32.subgroup_uniform_offset`, produced only from literals/uniform values; retain `u32.lane` for dynamic lane IDs.

4. **New hole: “determinism strict” is not encoded for floating-point operations.** Removing only plain float-to-int conversions does not prevent indeterminacy. Tree:

`store.h_O0(own.u32_threadrow_h, own.u32_threadcol_h, arith.smoothstep(literal.real(1), literal.real(1), literal.real(0)))`

spells a dynamic `smoothstep(lit_f32(1.0), lit_f32(1.0), lit_f32(0.0))`; equal edges yield an indeterminate value. Ordinary `arith.div(1, 0)`, `sqrt(-1)`, overflow, `inverse` of singular matrices, and f16 overflow have the same issue. [WGSL §17.5.57](https://www.w3.org/TR/WGSL/#smoothstep), [§15.7.2](https://www.w3.org/TR/WGSL/#floating-point-evaluation). Strict mode needs range/domain-refined float types and only total, range-preserving rows—or must exclude these rows entirely.

Holes 1, 3–8, and 10 are closed in the draft as written. The documented `own.*` additions provide terminals for the prior missing demanded types; `kernel.entry_safe` is reachable; helper-call spellings and deletion of `loop.unroll` remove tree rewriting/exponential text growth. Those positives do not outweigh the four counterexamples above.

