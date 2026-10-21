# 9. The WGSL kingdom's symbol table (built one row at a time)

Batch 1, 2026-10-21: the kernel skeleton (entry point, root, statement sequences, the epilogue's store) and the ownership rows the store needs, for the Gram interface (`phi` in, `h` out, uniform `cfg { d, m, r_rows, r_off }`, partition contiguous, `length h = cfg.r_rows * cfg.d`). Every type demanded by a row in this batch has a terminal in this batch, so the batch is a kingdom that passes the §1 check on its own. `KNOB_BROW`, `KNOB_BCOL` are `const`s the entry point's prologue writes from its grid node. Fuel strata, loops, the W phase, uniform tests and the other interfaces come in later batches.

## Types (growing)

| type | what it holds |
|---|---|
| `kernel.safe` | the entry point; the gene's root |
| `grid.shape` | one of the target's legal workgroup shapes `(x, y, 1)`, `x * y <= max_invocations`: a literal from the target's table |
| `u32.blk` | a block factor, a literal from `{1, 2, 4, 8}`: cells per invocation along one axis |
| `grid.grid` | the entry's workgroup shape and block |
| `block.L0` | a statement sequence at the top level: every invocation runs all of it (uniform control) |
| `stmt.L0` | one top-level statement |
| `block.C0` | a sequence under a thread-dependent condition at the top level: no barrier, no return in it |
| `stmt.C0` | one such statement |
| `block.O0` | the epilogue: a sequence whose last statement stores this invocation's own cell of `h` |
| `stmt.O0` | an epilogue statement: the store itself, or a conditional whose every branch is an epilogue |
| `vec3<u32>.gid` | `global_invocation_id`, its own type (never confused with another id) |
| `u32.rowbase` | the origin of this invocation's block on the row axis |
| `u32.colbase` | the origin of this invocation's block on the column axis |
| `u32.threadrow` | a row this invocation owns: its origin plus an offset clamped into its block |
| `u32.threadcol` | a column this invocation owns, likewise |
| `u32.lit64` | a literal `1..=64` |
| `u32.index` | any u32 used as a position (no ownership) |
| `f32.real` | a number |
| `bool.flag` | a predicate that may differ between invocations |

## Rows

| symbol | output | arguments (ordered) | WGSL spelling | note |
|---|---|---|---|---|
| `kernel.entry_safe` | `kernel.safe` | `grid.grid`, `block.L0`, `block.O0` | `@compute @workgroup_size(X, Y, 1) fn main(@builtin(global_invocation_id) gid: vec3<u32>, ...) { <declarations> {1} {2} }` | the only producer of the root; a kernel is a grid, a body and an epilogue, so every kernel ends in the store of its own cell (`block.O0`) |
| `grid.of` | `grid.grid` | `grid.shape`, `u32.blk`, `u32.blk` | `// grid X Y bx by` and the `@workgroup_size` from `{0}`; `const KNOB_BCOL: u32 = bx; const KNOB_BROW: u32 = by; const KNOB_WCOL = X; const KNOB_WROW = Y;` | the shape is in the target's limits by the table; the `// grid` header and the attribute come from one node, so they agree (harness H1); the block factors are kernel-scope constants, so every ownership row of the kernel uses the same block |
| `literal.shape` | `grid.shape` | — | the table entry's `X Y` | `?_shape`; its `Dc` domain is the table, so no shape outside the target exists |
| `literal.blk` | `u32.blk` | — | `1u` / `2u` / `4u` / `8u` | `?_blk`; domain `{1, 2, 4, 8}` |
| `block.one_L0` | `block.L0` | `stmt.L0` | `{0}` | |
| `block.seq2_L0` | `block.L0` | `stmt.L0`, `stmt.L0` | `{0} {1}` | |
| `block.seq3_L0` | `block.L0` | `stmt.L0` × 3 | `{0} {1} {2}` | |
| `block.seq4_L0` | `block.L0` | `stmt.L0` × 4 | `{0} {1} {2} {3}` | arity 4 is the node's maximum (`K_MAX`); longer sequences nest through `stmt.block_L0` |
| `stmt.block_L0` | `stmt.L0` | `block.L0` | `{ {0} }` | |
| `stmt.if_L0` | `stmt.L0` | `bool.flag`, `block.C0` | `if ({0}) { {1} }` | the body is `block.C0`, whose rows hold no barrier and no return: a barrier under a divergent condition is unspellable (`barrier_under_condition`, layer 4) |
| `block.one_C0` | `block.C0` | `stmt.C0` | `{0}` | |
| `block.seq2_C0` | `block.C0` | `stmt.C0`, `stmt.C0` | `{0} {1}` | `seq3_C0`, `seq4_C0` follow the same pattern (next batch) |
| `stmt.if_C0` | `stmt.C0` | `bool.flag`, `block.C0` | `if ({0}) { {1} }` | nesting stays conditional |
| `block.one_O0` | `block.O0` | `stmt.O0` | `{0}` | |
| `block.seq2_O0` | `block.O0` | `stmt.C0`, `stmt.O0` | `{0} {1}` | leading statements are conditional statements (no barrier after the body's barriers, no return); the last is an epilogue statement, so the block ends in a store |
| `stmt.if_else_O0` | `stmt.O0` | `bool.flag`, `block.O0`, `block.O0` | `if ({0}) { {1} } else { {2} }` | both branches are epilogues, so a store happens whichever is taken; a one-armed `if` over `block.O0` is deliberately absent: with the condition false no store would happen and the epilogue's guarantee would be lost |
| `store.h_O0` | `stmt.O0` | `u32.threadrow`, `u32.threadcol`, `f32.real` | `if ({0} < cfg.r_rows && {1} < cfg.d) { h[{0} * cfg.d + {1}] = {2}; }` | the store takes a row and a column of this invocation's block (the only producers of those types are the origin and offset rows below), so no two invocations write one cell (`write_sets_overlap`); the guard is the row's own text over its own inputs and, with `length h = cfg.r_rows * cfg.d`, keeps the index below the length (`index_not_in_bounds`); `r * d + c` is below `r_rows * d` by the guard, so it cannot wrap (`store_cell_wraps`) |
| `store.h_L0` | `stmt.L0` | `u32.threadrow`, `u32.threadcol`, `f32.real` | as `store.h_O0` | a store may also stand in the body; it is the same invocation's cell, so an earlier store and the epilogue's never race |
| `store.h_C0` | `stmt.C0` | `u32.threadrow`, `u32.threadcol`, `f32.real` | as `store.h_O0` | likewise under a condition |
| `own.block_L0` | `block.L0` | — | `{}` | terminal of `block.L0` (an empty sequence) |
| `own.stmt_L0` | `stmt.L0` | — | `{}` | terminal of `stmt.L0` |
| `own.block_C0` | `block.C0` | — | `{}` | terminal |
| `own.stmt_C0` | `stmt.C0` | — | `{}` | terminal |
| `own.block_O0` | `block.O0` | — | `if (R0 < cfg.r_rows && C0 < cfg.d) { h[R0 * cfg.d + C0] = h[R0 * cfg.d + C0] + 0.0; }` with `R0 = gid.y * KNOB_BROW`, `C0 = gid.x * KNOB_BCOL` | terminal of the epilogue: a store of the invocation's first cell that leaves its value; the tail of `block.O0` always holds a store |
| `own.stmt_O0` | `stmt.O0` | — | as `own.block_O0` | terminal |
| `safe.thread_row` | `u32.rowbase` | `vec3<u32>.gid` | `({0}.y * KNOB_BROW)` | the origin of the invocation's block on the row axis (contiguous partition); the block factor is the kernel's constant, not an argument, so one block factor per kernel |
| `safe.thread_col` | `u32.colbase` | `vec3<u32>.gid` | `({0}.x * KNOB_BCOL)` | likewise on the column axis |
| `safe.row_plus` | `u32.threadrow` | `u32.rowbase`, `u32.lit64` | `({0} + min({1}, KNOB_BROW - 1u))` | the first argument is an origin, never a `threadrow`: an offset of an offset would leave the block (another invocation's cell); the offset is clamped by the row's own text |
| `safe.col_plus` | `u32.threadcol` | `u32.colbase`, `u32.lit64` | `({0} + min({1}, KNOB_BCOL - 1u))` | likewise |
| `safe.rowbase_as_row` | `u32.threadrow` | `u32.rowbase` | `{0}` | the origin is the block's first row (offset zero) |
| `safe.colbase_as_col` | `u32.threadcol` | `u32.colbase` | `{0}` | |
| `safe.row_as_index` | `u32.index` | `u32.threadrow` | `{0}` | an owned row used as a plain position (for loads, arithmetic); nothing goes the other way: no row makes a `threadrow` from a `u32.index` |
| `safe.col_as_index` | `u32.index` | `u32.threadcol` | `{0}` | |
| `literal.lit64` | `u32.lit64` | — | `lit_u32(n)` with `n` in `1..=64` | `?_lit64`; `lit_u32` is the identity helper that keeps every literal out of constant evaluation |
| `own.u32_rowbase` | `u32.rowbase` | — | `(gid.y * KNOB_BROW)` | terminal: the invocation's own origin |
| `own.u32_colbase` | `u32.colbase` | — | `(gid.x * KNOB_BCOL)` | terminal |
| `own.u32_threadrow` | `u32.threadrow` | — | `(gid.y * KNOB_BROW)` | terminal: the block's first row |
| `own.u32_threadcol` | `u32.threadcol` | — | `(gid.x * KNOB_BCOL)` | terminal |
| `builtin.global_invocation_id` | `vec3<u32>.gid` | — | `gid` | terminal; the only producer of `vec3<u32>.gid` |
| `literal.flag` | `bool.flag` | — | `lit_bool(true)` / `lit_bool(false)` | `?_flag`; terminal of `bool.flag` until the comparisons arrive (batch 2) |
| `literal.real` | `f32.real` | — | `lit_f32(v)`, `v` a finite f32 | `?_real`; domain the finite f32 range, so no NaN or infinity literal exists |

41 rows, 19 types. Next batch: the remaining builtins (`lid`, `wid`, `lidx`, `num_workgroups`) and their axis rows, `u32` and `f32` arithmetic, comparisons, the loads of `phi` and of `cfg`'s fields.

Rows I am unsure of:

- `own.block_O0` / `own.stmt_O0`: a terminal of the epilogue has to be a store (so the tail keeps the guarantee); spelled as `h[c] = h[c] + 0.0`, which writes the cell without changing it. The alternative is `h[c] = 0.0`, which changes the output; the read-add-write form was chosen so a terminal is value-neutral.
- `stmt.if_else_O0` with no one-armed `stmt.if_O0`: today's table has `stmt.if_O` over one `block.O`, which lets a false condition skip the store. Removing it keeps the guarantee; it costs the spelling `if (c) { store } ` with no else, which the reader would read as `if (c) { store } else { own store }` only by rewriting, so such a seed reads as outside the kingdom.
- `store.h_L0` and `store.h_C0` beside the epilogue's store: legal (same invocation, same cell set), kept because seeds store in the body.
