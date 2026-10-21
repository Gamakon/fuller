# 9. The WGSL kingdom's symbol table (built one row at a time)

The kingdom holds any WGSL compute kernel over a declared **interface**: its storage bindings (each an input, read-only, or an output, read-write, with an element type `<e>` and a shape over the uniform's fields), its uniform block's fields, and the target's limits. Rows that mention a binding or a field are **families**, generated once per declared binding or field; the Gram interface (`phi: array<f32>` in, `h: array<f32>` out with shape `cfg.r_rows x cfg.d`, `cfg { d, m, r_rows, r_off }`) is one worked instance, never the default. Notation: `<b>` an output binding, `<i>` an input binding, `<f>` a uniform field, `<e>` the binding's element type (`f32`, `u32`, `i32`, `vec4<f32>`, ...). A 2-D output declares `shape <b> <R_b> <C_b>` (rows and columns, each an expression over fields) and is laid out row-major with stride `C_b`; a 1-D output declares `shape <b> <L_b>`. Every binding with load or store rows declares a length lower bound of at least one. `KNOB_BROW`, `KNOB_BCOL`, `KNOB_WROW`, `KNOB_WCOL` are `const`s the entry point writes from its grid node. `lit_u32`, `lit_f32`, `lit_bool`, `lit_i32` are identity helpers that keep every literal out of constant evaluation.

Batch 1 (2026-10-21): the kernel skeleton, the epilogue's store, the ownership rows. Batch 2: the remaining builtins and axes, u32 and f32 arithmetic, comparisons and logic, input loads and uniform fields. Batch 3: registers and private arrays, fuel strata, loops and counters, uniform provenance (fields, workgroup ids, uniform counters, `workgroupUniformLoad`), the barrier. Batch 4: the W phase and workgroup-array writes, the stratum map's arrays and reads, reductions, atomics, subgroups. Batch 5: vectors and their arrays, i32 and the remaining scalar families, the strided, grouped and lane partitions, arenas and the relaxed conversions, the storage barrier.

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
| `block.O0` | the epilogue: a sequence whose last statement stores this invocation's own cell of the primary output |
| `stmt.O0` | an epilogue statement: the store itself, or a conditional whose every branch is an epilogue |
| `vec3<u32>.gid` | `global_invocation_id`, its own type (never confused with another id) |
| `vec3<u32>.lid` | `local_invocation_id` |
| `vec3<u32>.wid` | `workgroup_id` |
| `u32.lidx` | `local_invocation_index` |
| `vec3<u32>.nwg` | `num_workgroups` |
| `u32.rowid` | an id's component on the output's row axis (`.y`) |
| `u32.colid` | an id's component on the output's column axis (`.x`) |
| `u32.rowbase_<b>` | the origin of this invocation's block of output `<b>` on the row axis |
| `u32.colbase_<b>` | the origin on the column axis |
| `u32.threadrow_<b>` | a row of `<b>` this invocation owns: its origin plus an offset clamped into its block |
| `u32.threadcol_<b>` | a column of `<b>` this invocation owns, likewise |
| `u32.lit64` | a literal `1..=64` |
| `u32.index` | any u32 used as a position (no ownership; wraps as WGSL defines) |
| `u32.count` | any u32 used as a size or a field's value (no ownership) |
| `f32.real` | a number |
| `bool.flag` | a predicate that may differ between invocations |

## Batch 1: skeleton, epilogue, ownership

| symbol | output | arguments (ordered) | WGSL spelling | note |
|---|---|---|---|---|
| `kernel.entry_safe` | `kernel.safe` | `grid.grid`, `block.L0`, `block.O0` | `@compute @workgroup_size(X, Y, 1) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) { <declarations> {1} {2} }` | the only producer of the root; a kernel is a grid, a body and an epilogue, so every kernel ends in the store of its own cell (`block.O0`); the prologue (bindings, uniform struct, helpers) is the interface's, written once, so no gene declares a binding |
| `grid.of` | `grid.grid` | `grid.shape`, `u32.blk`, `u32.blk` | `// grid X Y bx by` and the `@workgroup_size` from `{0}`; `const KNOB_WCOL: u32 = X; const KNOB_WROW: u32 = Y; const KNOB_BCOL: u32 = bx; const KNOB_BROW: u32 = by;` | the shape is in the target's limits by the table; the `// grid` header and the attribute come from one node, so they agree (harness H1); the block factors are kernel-scope constants, so every ownership row of the kernel uses the same block |
| `literal.shape` | `grid.shape` | — | the table entry's `X Y` | `?_shape`; its `Dc` domain is the table, so no shape outside the target exists |
| `literal.blk` | `u32.blk` | — | `1u` / `2u` / `4u` / `8u` | `?_blk`; domain `{1, 2, 4, 8}` |
| `block.one_L0` | `block.L0` | `stmt.L0` | `{0}` | |
| `block.seq2_L0` | `block.L0` | `stmt.L0`, `stmt.L0` | `{0} {1}` | |
| `block.seq3_L0` | `block.L0` | `stmt.L0` × 3 | `{0} {1} {2}` | |
| `block.seq4_L0` | `block.L0` | `stmt.L0` × 4 | `{0} {1} {2} {3}` | arity 4 is the node's maximum (`K_MAX`); longer sequences nest through `stmt.block_L0` |
| `stmt.block_L0` | `stmt.L0` | `block.L0` | `{ {0} }` | |
| `stmt.if_L0` | `stmt.L0` | `bool.flag`, `block.C0` | `if ({0}) { {1} }` | the body is `block.C0`, whose rows hold no barrier and no return: a barrier under a divergent condition is unspellable (`barrier_under_condition`, layer 4) |
| `block.one_C0` | `block.C0` | `stmt.C0` | `{0}` | |
| `block.seq2_C0` | `block.C0` | `stmt.C0`, `stmt.C0` | `{0} {1}` | |
| `block.seq3_C0` | `block.C0` | `stmt.C0` × 3 | `{0} {1} {2}` | |
| `block.seq4_C0` | `block.C0` | `stmt.C0` × 4 | `{0} {1} {2} {3}` | |
| `stmt.block_C0` | `stmt.C0` | `block.C0` | `{ {0} }` | |
| `stmt.if_C0` | `stmt.C0` | `bool.flag`, `block.C0` | `if ({0}) { {1} }` | nesting stays conditional |
| `block.one_O0` | `block.O0` | `stmt.O0` | `{0}` | |
| `block.seq2_O0` | `block.O0` | `stmt.C0`, `stmt.O0` | `{0} {1}` | leading statements are conditional statements (no barrier, no return); the last is an epilogue statement, so the block ends in a store |
| `block.seq3_O0` | `block.O0` | `stmt.C0`, `stmt.C0`, `stmt.O0` | `{0} {1} {2}` | |
| `block.seq4_O0` | `block.O0` | `stmt.C0` × 3, `stmt.O0` | `{0} {1} {2} {3}` | |
| `stmt.block_O0` | `stmt.O0` | `block.O0` | `{ {0} }` | |
| `stmt.if_else_O0` | `stmt.O0` | `bool.flag`, `block.O0`, `block.O0` | `if ({0}) { {1} } else { {2} }` | both branches are epilogues, so a store happens whichever is taken; a one-armed `if` over `block.O0` is absent: with the condition false no store would happen (ruled, batch 1 review) |
| `store.<b>_O0` | `stmt.O0` | `u32.threadrow_<b>`, `u32.threadcol_<b>`, `<e>` | 2-D: `if ({0} < R_b && {1} < C_b) { b[{0} * C_b + {1}] = {2}; }`; 1-D: `if ({1} < L_b) { b[{1}] = {2}; }` (the row argument is absent for a 1-D output) | one family per output; the store takes a row and a column of this invocation's block of `<b>` (the only producers of those types are `<b>`'s origin and offset rows), so no two invocations write one cell (`write_sets_overlap`); the guard is the row's own text over its own inputs and, with the declared length bound `R_b * C_b` (or `L_b`), keeps the index below the length (`index_not_in_bounds`); `r * C_b + c < R_b * C_b` by the guard, so it cannot wrap (`store_cell_wraps`). Gram: `store.h_O0` over `u32.threadrow_h`, `u32.threadcol_h`, `f32.real`: `if ({0} < cfg.r_rows && {1} < cfg.d) { h[{0} * cfg.d + {1}] = {2}; }`. The epilogue's store is the **primary** output's (the interface names it); other outputs are stored from the body |
| `store.<b>_L0` | `stmt.L0` | `u32.threadrow_<b>`, `u32.threadcol_<b>`, `<e>` | as `store.<b>_O0` | a store may also stand in the body, for every output; same invocation, same cell set, so body stores and the epilogue's never race (ruled) |
| `store.<b>_C0` | `stmt.C0` | `u32.threadrow_<b>`, `u32.threadcol_<b>`, `<e>` | as `store.<b>_O0` | likewise under a condition |
| `own.block_L0` | `block.L0` | — | `{}` | terminal of `block.L0` (an empty sequence) |
| `own.stmt_L0` | `stmt.L0` | — | `{}` | terminal |
| `own.block_C0` | `block.C0` | — | `{}` | terminal |
| `own.stmt_C0` | `stmt.C0` | — | `{}` | terminal |
| `own.block_O0` | `block.O0` | — | the primary output `p`'s store at the invocation's first cell with its own value: 2-D `if (R0 < R_p && C0 < C_p) { p[R0 * C_p + C0] = p[R0 * C_p + C0]; }`, `R0 = gid.y * KNOB_BROW`, `C0 = gid.x * KNOB_BCOL` | terminal of the epilogue: a value-neutral store (ruled); the tail of `block.O0` always holds a store. Gram: `if (R0 < cfg.r_rows && C0 < cfg.d) { h[R0 * cfg.d + C0] = h[R0 * cfg.d + C0]; }` |
| `own.stmt_O0` | `stmt.O0` | — | as `own.block_O0` | terminal |
| `safe.thread_row_<b>` | `u32.rowbase_<b>` | `vec3<u32>.gid` | `({0}.y * KNOB_BROW)` | per output: the origin of the invocation's block of `<b>` on the row axis (contiguous partition; the other partitions come later, one per output as the interface declares); the block factor is the kernel's constant, not an argument, so one block per kernel |
| `safe.thread_col_<b>` | `u32.colbase_<b>` | `vec3<u32>.gid` | `({0}.x * KNOB_BCOL)` | likewise on the column axis |
| `safe.row_plus_<b>` | `u32.threadrow_<b>` | `u32.rowbase_<b>`, `u32.lit64` | `({0} + min({1}, KNOB_BROW - 1u))` | the first argument is an origin, never a `threadrow`: an offset of an offset would leave the block (another invocation's cell); the offset is clamped by the row's own text |
| `safe.col_plus_<b>` | `u32.threadcol_<b>` | `u32.colbase_<b>`, `u32.lit64` | `({0} + min({1}, KNOB_BCOL - 1u))` | likewise |
| `safe.rowbase_as_row_<b>` | `u32.threadrow_<b>` | `u32.rowbase_<b>` | `{0}` | the origin is the block's first row (offset zero) |
| `safe.colbase_as_col_<b>` | `u32.threadcol_<b>` | `u32.colbase_<b>` | `{0}` | |
| `safe.row_as_index_<b>` | `u32.index` | `u32.threadrow_<b>` | `{0}` | an owned row used as a plain position; nothing goes the other way: no row makes a `threadrow_<b>` from a `u32.index`, and no row makes a `threadrow_<b>` from a `threadrow_<b'>` of another output (their shapes differ) |
| `safe.col_as_index_<b>` | `u32.index` | `u32.threadcol_<b>` | `{0}` | |
| `literal.lit64` | `u32.lit64` | — | `lit_u32(n)`, `n` in `1..=64` | `?_lit64` |
| `own.u32_rowbase_<b>` | `u32.rowbase_<b>` | — | `(gid.y * KNOB_BROW)` | terminal: the invocation's own origin |
| `own.u32_colbase_<b>` | `u32.colbase_<b>` | — | `(gid.x * KNOB_BCOL)` | terminal |
| `own.u32_threadrow_<b>` | `u32.threadrow_<b>` | — | `(gid.y * KNOB_BROW)` | terminal: the block's first row |
| `own.u32_threadcol_<b>` | `u32.threadcol_<b>` | — | `(gid.x * KNOB_BCOL)` | terminal |
| `builtin.global_invocation_id` | `vec3<u32>.gid` | — | `gid` | terminal; the only producer of `vec3<u32>.gid` |
| `literal.flag` | `bool.flag` | — | `lit_bool(true)` / `lit_bool(false)` | `?_flag` |
| `literal.real` | `f32.real` | — | `lit_f32(v)`, `v` a finite f32 | `?_real`; domain the finite f32 range, so no NaN or infinity literal exists |

## Batch 2: builtins and axes, u32 and f32 arithmetic, comparisons, input loads, uniform fields

| symbol | output | arguments (ordered) | WGSL spelling | note |
|---|---|---|---|---|
| `builtin.local_invocation_id` | `vec3<u32>.lid` | — | `lid` | terminal; its own type, so a mutation cannot swap `lid` for `gid` inside an ownership row (`safe.thread_row_<b>` takes `gid` only) |
| `builtin.workgroup_id` | `vec3<u32>.wid` | — | `wid` | terminal |
| `builtin.local_invocation_index` | `u32.lidx` | — | `lidx` | terminal; its own type: the W-phase ownership rows will take it (later batch) |
| `builtin.num_workgroups` | `vec3<u32>.nwg` | — | `nwg` | terminal |
| `axis.x_<id>` | `u32.colid` | `vec3<u32>.<id>` | `{0}.x` | `<id>` in `{gid, lid, wid}`: 3 rows; the column axis |
| `axis.y_<id>` | `u32.rowid` | `vec3<u32>.<id>` | `{0}.y` | 3 rows; the row axis |
| `axis.z_<id>` | `u32.index` | `vec3<u32>.<id>` | `{0}.z` | 3 rows; `z` is a plain index |
| `count.nwg_<c>` | `u32.count` | `vec3<u32>.nwg` | `{0}.<c>` | `<c>` in `{x, y, z}`: 3 rows; a workgroup count is a size |
| `index.of_row` | `u32.index` | `u32.rowid` | `{0}` | an axis component as a plain position; one way only: no row makes a `rowid` from an index, so `axis` rows never carry arithmetic back into an axis |
| `index.of_col` | `u32.index` | `u32.colid` | `{0}` | |
| `index.of_lidx` | `u32.index` | `u32.lidx` | `{0}` | |
| `load.<f>` | `u32.count` | — | `cfg.<f>` | one per u32 field of the uniform block (`<f>` the field name; the uniform variable is the interface's): a field is a size. Gram: `load.d`, `load.m`, `load.r_rows`, `load.r_off`. An i32 or f32 field is `i32.int` / `f32.real` (types arrive with those rows) |
| `load.<i>` | `<e>` | `u32.index` | `i[min({0}, arrayLength(&i) - 1u)]` | one per input binding `<i>`: a read at any position; the clamp to the last element is the row's own text, so a read is in range whatever the device's bounds policy; `arrayLength >= 1` by the binding's declared length bound, so `- 1u` cannot wrap. Gram: `load.phi -> f32.real`: `phi[min({0}, arrayLength(&phi) - 1u)]` |
| `load.<i>_len` | `u32.count` | — | `arrayLength(&i)` | one per input binding |
| `load.<b>_own` | `<e>` | `u32.threadrow_<b>`, `u32.threadcol_<b>` | 2-D `b[min({0} * C_b + {1}, arrayLength(&b) - 1u)]`; 1-D `b[min({1}, arrayLength(&b) - 1u)]` | one per output: a read of the invocation's own cell only, so no invocation reads a cell another writes (`output_read_of_another_invocations_cell`); the clamp is the row's text (the identity on an owned cell in bounds). Gram: `load.h_own` |
| `load.<b>_len` | `u32.count` | — | `arrayLength(&b)` | one per output |
| `literal.index` | `u32.index` | — | `lit_u32(n)`, any u32 | `?_index` |
| `literal.count` | `u32.count` | — | `lit_u32(n)`, any u32 | `?_count` |
| `index.of_count` | `u32.index` | `u32.count` | `{0}` | a size used as a position |
| `index.add` | `u32.index` | `u32.index`, `u32.index` | `({0} + {1})` | wraps modulo `2^32` as WGSL defines; a wrapped value is a wrong position, never an unsafe one: no store takes a `u32.index`, and every load clamps |
| `index.mul` | `u32.index` | `u32.index`, `u32.index` | `({0} * {1})` | likewise |
| `index.sub` | `u32.index` | `u32.index`, `u32.index` | `({0} - {1})` | wraps modulo `2^32` as WGSL defines (§6.2.3), like `add` and `mul`; reaches no store or bound except through the capped bound rows; a seed's `a - b` reads as this row exactly (ruled, batch 2 review) |
| `index.sub_floor` | `u32.index` | `u32.index`, `u32.index` | `({0} - min({0}, {1}))` | the floored difference, a separate row with its own text |
| `index.div` | `u32.index` | `u32.index`, `u32.index` | `({0} / {1})` | WGSL §8.8: for a runtime divisor of zero the result is `e1`; defined |
| `index.mod` | `u32.index` | `u32.index`, `u32.index` | `({0} % {1})` | WGSL §8.8: a runtime divisor of zero gives `0`; defined |
| `index.min` | `u32.index` | `u32.index`, `u32.index` | `min({0}, {1})` | |
| `index.max` | `u32.index` | `u32.index`, `u32.index` | `max({0}, {1})` | |
| `index.clamp_to` | `u32.index` | `u32.index`, `u32.count` | `min({0}, max({1}, 1u) - 1u)` | `i < n`, and `0` for `n = 0` |
| `index.shl` | `u32.index` | `u32.index`, `u32.index` | `({0} << {1})` | WGSL §8.10: for runtime operands "the number of bits to shift is the value of e2, modulo the bit width of e1"; a shift count at or past the width is an error only for a const-expression, and every literal of this kingdom passes through `lit_u32`, so no const-expression shift exists; the row's meaning is WGSL's |
| `index.shr` | `u32.index` | `u32.index`, `u32.index` | `({0} >> {1})` | likewise (§8.10) |
| `index.shl_m` | `u32.index` | `u32.index`, `u32.index` | `({0} << (({1}) & 31u))` | the explicitly masked spelling, a separate row so a seed written with the mask reads exactly; `index.shr_m` likewise |
| `index.and` | `u32.index` | `u32.index`, `u32.index` | `({0} & {1})` | |
| `index.or` | `u32.index` | `u32.index`, `u32.index` | `({0} \| {1})` | |
| `count.add` | `u32.count` | `u32.count`, `u32.count` | `({0} + {1})` | sizes add (wrap defined); `count.sub_floor`, `count.min`, `count.max` as for `index` |
| `arith.<op>` | `f32.real` | `f32.real`, `f32.real` | `({0} + {1})`, `({0} - {1})`, `({0} * {1})`, `({0} / {1})`, `min(...)`, `max(...)`, `pow(...)`, `atan2(...)`, `step(...)` | `<op>` in `{add, sub, mul, div, min, max, pow, atan2, step}`: 9 rows; IEEE arithmetic, `x / 0` is an infinity or NaN, a value fitness scores, never a fault |
| `arith.<un>` | `f32.real` | `f32.real` | `neg: (-{0})`, else `<un>({0})` | `<un>` in `{neg, abs, sign, floor, ceil, round, fract, trunc, saturate, sqrt, inverse_sqrt, exp, exp2, log, log2, sin, cos, tan, tanh, asin, acos, atan}`: 22 rows; `sqrt(-1)` and `log(0)` give NaN or an infinity, scored |
| `arith.<tri>` | `f32.real` | `f32.real` × 3 | `clamp({0}, {1}, {2})`, `mix(...)`, `fma(...)`, `smoothstep(...)` | 4 rows |
| `convert.index_to_f32` | `f32.real` | `u32.index` | `f32({0})` | |
| `convert.count_to_f32` | `f32.real` | `u32.count` | `f32({0})` | |
| `convert.f32_to_index` | `u32.index` | `f32.real` | `u32({0})` | WGSL §15.7.6: a non-NaN value is clamped into the target's range and rounded toward zero (`-1f` is `0u`, `1e20f` is `4294967040u`): defined, so the row is plain `u32(x)` and a seed's `u32(x)` reads as it exactly (ruled); a NaN gives "an indeterminate value in T": a u32, scored by fitness, never a fault |
| `convert.f32_to_index_nan0` | `u32.index` | `f32.real` | `u32(select({0}, 0.0, {0} != {0}))` | the deterministic variant for the one indeterminate case of §15.7.6 (NaN gives `0u`), a separate row with its own text |
| `convert.f32_to_count` | `u32.count` | `f32.real` | `u32({0})` | as `convert.f32_to_index` |
| `compare.<cmp>_index` | `bool.flag` | `u32.index`, `u32.index` | `({0} < {1})` ... | `<cmp>` in `{lt, le, gt, ge, eq, ne}`: 6 rows |
| `compare.<cmp>_count` | `bool.flag` | `u32.count`, `u32.count` | likewise | 6 rows; a comparison of a position with a size goes through `index.of_count` |
| `compare.<cmp>_real` | `bool.flag` | `f32.real`, `f32.real` | likewise | 6 rows; a comparison with NaN is false, defined |
| `logic.and` | `bool.flag` | `bool.flag`, `bool.flag` | `({0} && {1})` | |
| `logic.or` | `bool.flag` | `bool.flag`, `bool.flag` | `({0} \|\| {1})` | |
| `logic.not` | `bool.flag` | `bool.flag` | `(!{0})` | |
| `select.real` | `f32.real` | `f32.real`, `f32.real`, `bool.flag` | `select({0}, {1}, {2})` | `select.index`, `select.count` likewise |

Batch 2: 49 rows (families counted once; 3-row axis and 6-row comparison patterns as written). Types added in batch 2: `vec3<u32>.lid`, `vec3<u32>.wid`, `u32.lidx`, `vec3<u32>.nwg`, `u32.rowid`, `u32.colid`, `u32.count`. Running total: 26 types; 48 + 49 = 97 rows (families).

## Batch 3: registers, private arrays, fuel strata, loops and counters, uniform provenance, the barrier

**Fuel strata, applied to every statement type.** From this batch every statement and block type carries a fuel stratum `_f<k>`, `k` in `0..=4`: `stmt.L0_f2`, `block.C1_f0`, `block.O0_f4`, ... The meaning of stratum `k`: one execution of the statement performs at most `H * 2^(6k)` node executions, `H` the layout's total head length (the walk reads each head row at most once, so a tree has at most `H` functions). Every row of batches 1 and 2 that takes or gives a statement or block type now exists once per stratum with **all** its statement inputs and its output at one stratum (`block.seq2_L0_f2: (stmt.L0_f2, stmt.L0_f2) -> block.L0_f2`; `stmt.if_L0_f1: (bool.flag, block.C0_f1) -> stmt.L0_f1`; `store.<b>_O0_f3 -> stmt.O0_f3`; `own.stmt_L0_f0`, ... at every stratum), and the root takes the interface's budget `K`: `kernel.entry_safe: (grid.grid, block.L0_f<K>, block.O0_f<K>)`. Only loop rows raise the stratum (by their trip class); the lift rows below raise it without text. So a kernel of budget `K` executes at most `H * 2^(6K)` nodes an invocation and terminates: the hang (A1 P3, H10) has no tree.

**Rung ladder.** A loop at rung `<X><d>` has a body at `<X><d+1>` and binds the counter of depth `d+1`; depths run `0..=5`, so five loops nest and no row takes a `block.*6`. Types added: `stmt.L<d>`, `block.L<d>`, `stmt.C<d>`, `block.C<d>` for `d` in `1..=5`; `stmt.O<d>`, `block.O<d>` for `d` in `1..=4`; `block.B<d>` for `d` in `1..=5` (a loop body that breaks: conditional statements, the guard, conditional statements; no barrier, so an invocation that leaves early skips no barrier).

| type | what it holds |
|---|---|
| `reg<f32>.real` | a named f32 register `f0..f63`, declared at kernel scope, initialised `0.0` |
| `reg<u32>.index` | a named u32 register `i0..i63`, initialised `0u` |
| `reg<bool>.flag` | a named bool register `b0..b15`, initialised `false` |
| `ureg<u32>.uniform` | a named u32 register `U0..U7` that holds the same value in every invocation: written only at L rungs from uniform values |
| `parr<f32,64>.real` | a private array of 64 f32, `pf0..pf7` |
| `parr<u32,16>.index` | a private array of 16 u32, `pi0..pi4` |
| `u32.pidx_6` | an index below 64, by the mask its row writes |
| `u32.pidx_4` | an index below 16, likewise |
| `u32.lit4096` | a literal `1..=4096` |
| `u32.boundx<c>` | an expression loop bound capped at `2^c`, `c` in `{6, 12, 18, 24}` (trip class `c/6`); any invocation's value |
| `u32.boundu<c>` | a loop bound capped at `2^c` that is the same in every invocation of the workgroup |
| `u32.tcounter<d>` | the counter `k<d>` of a loop whose bound is a literal or uniform, `d` in `1..=5`: uniform |
| `u32.xcounter<d>` | the counter `kx<d>` of a loop whose bound is any expression, or that breaks: divergent |
| `u32.uniform` | a u32 the same in every invocation of the workgroup: a field, a literal, the workgroup id, the grid, a uniform counter, a uniform register, arithmetic of these |
| `bool.uflag` | a predicate the same in every invocation: comparisons of uniform values and their logic |
| `wgmem<u32,1024>.index` | a workgroup array of 1,024 u32, `wg_u_10` (the stratum map's; the full map and its writes come in batch 4) |

| symbol | output | arguments (ordered) | WGSL spelling | note |
|---|---|---|---|---|
| `reg.f<n>` | `reg<f32>.real` | — | `f<n>` | `n` in `0..=63`: 64 terminals; a register is a name, never a value: only `reg.get_f` reads it |
| `reg.i<n>` | `reg<u32>.index` | — | `i<n>` | 64 terminals |
| `reg.b<n>` | `reg<bool>.flag` | — | `b<n>` | 16 terminals |
| `reg.get_f` | `f32.real` | `reg<f32>.real` | `{0}` | |
| `reg.get_i` | `u32.index` | `reg<u32>.index` | `{0}` | a register holds any invocation's value: no row makes a `u32.uniform` or an ownership type from it |
| `reg.get_b` | `bool.flag` | `reg<bool>.flag` | `{0}` | |
| `reg.set_f_<r>` | `stmt.<r>` | `reg<f32>.real`, `f32.real` | `{0} = {1};` | `<r>` every `L<d>_f<k>` and `C<d>_f<k>` rung (not `O`: the epilogue's last statement is a store, and its leading statements are `C`) |
| `reg.set_i_<r>` | `stmt.<r>` | `reg<u32>.index`, `u32.index` | `{0} = {1};` | likewise |
| `reg.set_b_<r>` | `stmt.<r>` | `reg<bool>.flag`, `bool.flag` | `{0} = {1};` | likewise |
| `parr.f<n>` | `parr<f32,64>.real` | — | `pf<n>` | `n` in `0..=7`; private arrays per element type and length as the kingdom's register file lists them (`vec4<f32>` × 16 and × 64, `u32` × 64 and × 128, `bool` × 16, `vec2<f32>` × 16 follow the same three-row pattern when their scalar families arrive) |
| `parr.i<n>` | `parr<u32,16>.index` | — | `pi<n>` | `n` in `0..=4` |
| `safe.pidx_6` | `u32.pidx_6` | `u32.index` | `(({0}) & 63u)` | the mask is the row's text; one index type per array length, so a 64-mask can never reach a 16-element array (`index_out_of_range`) |
| `safe.pidx_4` | `u32.pidx_4` | `u32.index` | `(({0}) & 15u)` | |
| `parr.get_f_6` | `f32.real` | `parr<f32,64>.real`, `u32.pidx_6` | `{0}[{1}]` | in range by the index type |
| `parr.get_i_4` | `u32.index` | `parr<u32,16>.index`, `u32.pidx_4` | `{0}[{1}]` | |
| `parr.set_f_6_<r>` | `stmt.<r>` | `parr<f32,64>.real`, `u32.pidx_6`, `f32.real` | `{0}[{1}] = {2};` | at the `L` and `C` rungs of every stratum |
| `parr.set_i_4_<r>` | `stmt.<r>` | `parr<u32,16>.index`, `u32.pidx_4`, `u32.index` | `{0}[{1}] = {2};` | |
| `literal.lit4096` | `u32.lit4096` | — | `lit_u32(n)`, `n` in `1..=4096` | `?_lit4096` |
| `bound.of_index_<c>` | `u32.boundx<c>` | `u32.index` | `min({0}, lit_u32(2^c))` | `c` in `{6, 12, 18, 24}`: 4 rows; the cap is the row's text, so every expression-bounded loop has a known trip bound (`loop_bound_wraps`: the counter steps by at most 4,096 from below `2^24`, so it never passes `u32::MAX`) |
| `bound.of_uniform_<c>` | `u32.boundu<c>` | `u32.uniform` | `min({0}, lit_u32(2^c))` | 4 rows; a uniform bound: the loop's trip count is the same in every invocation, so its body may hold a barrier |
| `stmt.lift_<X><d>_f<k>` | `stmt.<X><d>_f<k>` | `stmt.<X><d>_f<k-1>` | `{0}` | `k` in `1..=4`, every rung: a statement of a lower stratum stands in a higher one (needed so a sequence can mix a loop with plain statements) |
| `loop.tile_L<d>_f<k>` | `stmt.L<d>_f<k+1>` | `u32.lit64`, `block.L<d+1>_f<k>` | `let n = {0}; k<d+1> = 0u; for (; k<d+1> < n; k<d+1> = k<d+1> + 1u) { {1} }` | `d` in `0..=4`, `k` in `0..=3`: a literal bound (class 1) raises the stratum by one; the body keeps the L rung, so it may hold a barrier (every invocation runs the same trips); the counter is the kernel-scope `k<d+1>`, reset and never redeclared; the bound is in a `let`, so the body cannot change the trip count (`loop_bound_changes_in_body`) |
| `loop.tile_C<d>_f<k>` | `stmt.C<d>_f<k+1>` | `u32.lit64`, `block.C<d+1>_f<k>` | as above with `kx<d+1>` | under a condition: the body stays conditional; `loop.tile_O<d>_f<k>: (u32.lit64, block.O<d+1>_f<k>) -> stmt.O<d>_f<k+1>` for `d` in `0..=3` (the epilogue's loop: its body ends in the store, so the loop runs the store at least once because `n >= 1`) |
| `loop.tileu_<c>_L<d>_f<k>` | `stmt.L<d>_f<k+c/6>` | `u32.boundu<c>`, `block.L<d+1>_f<k>` | `let n = {0}; k<d+1> = 0u; for (; k<d+1> < n; k<d+1> = k<d+1> + 1u) { {1} }` | a uniform bound of class `c/6` at an L rung: the body keeps the L rung; exists only where `k + c/6 <= 4` |
| `loop.tilex_<c>_<X><d>_f<k>` | `stmt.<X><d>_f<k+c/6>` | `u32.boundx<c>`, `block.C<d+1>_f<k>` | as above with `kx<d+1>` | `<X>` in `{L, C}`: an expression bound differs between invocations, so the body is a `C` block whatever the rung (no barrier in it: `barrier_in_nonuniform_control_flow`) and the counter is `kx`; no `tilex` or `tileu` row exists at an O rung: a capped bound can be `0` and skip the epilogue's store, so an epilogue loop takes `u32.lit64` only (ruled, batch 3 review) |
| `loop.tile_strided_<X><d>_f<k>` | `stmt.<X><d>_f<k+1>` | `u32.lit64`, `u32.lit4096`, `block.<X'><d+1>_f<k>` | `... k = k + {1} ...` | the stride form of each loop row above (`tileu_strided`, `tilex_strided` with `u32.lit4096` as the stride); a stride only lowers the trip count, so the class is the bound's |
| `loop.unroll_<X><d>_f<k>` | `stmt.<X><d>_f<k+1>` | `u32.lit64`, `block.<X'><d+1>_f<k>` | the body written `n` times, `k<d+1>` a literal `0u, 1u, ...` in each copy | straight-line; the counter variable is not assigned |
| `block.brk_B<d>` | `block.B<d>_f<k>` | `block.C<d>_f<k>`, `bool.flag`, `block.C<d>_f<k>` | `{0} if ({1}) { break; } {2}` | `d` in `1..=5`; both parts are `C` blocks: no barrier before or after the break |
| `loop.tile_brk_<X><d>_f<k>` | `stmt.<X><d>_f<k+1>` | `u32.lit64`, `block.B<d+1>_f<k>` | as `loop.tile` with `kx<d+1>` | a loop that breaks binds `kx`: after it the counter differs between invocations; `loop.tilex_brk_<c>` likewise with `u32.boundx<c>` |
| `block.cont_C<d>` | `block.C<d>_f<k>` | `bool.flag`, `block.C<d>_f<k>` | `if ({0}) { continue; } {1}` | `d` in `1..=5` only: a `C<d>` block with `d >= 1` is inside a loop, so `continue_outside_loop` has no tree; the rest of the block is conditional |
| `safe.tile_counter<d>` | `u32.tcounter<d>` | — | `k<d>` | `d` in `1..=5`: terminal; `0u` before its loop, the trip inside, the final value after |
| `safe.xtile_counter<d>` | `u32.xcounter<d>` | — | `kx<d>` | terminal |
| `index.of_tcounter<d>` | `u32.index` | `u32.tcounter<d>` | `{0}` | 5 rows; `index.of_xcounter<d>` likewise |
| `uniform.<f>` | `u32.uniform` | — | `cfg.<f>` | one per u32 uniform field: the same in every invocation. Gram: `uniform.d`, `uniform.m`, `uniform.r_rows`, `uniform.r_off` |
| `uniform.lit` | `u32.uniform` | `u32.lit4096` | `{0}` | |
| `uniform.wid_<c>` | `u32.uniform` | `vec3<u32>.wid` | `{0}.<c>` | `<c>` in `{x, y, z}`: the workgroup id is uniform; `uniform.nwg_<c>` over `vec3<u32>.nwg` likewise; no row makes a `uniform` from `gid`, `lid` or `lidx` |
| `uniform.of_tcounter<d>` | `u32.uniform` | `u32.tcounter<d>` | `{0}` | the counter of a literal- or uniform-bounded loop is the same in every invocation; no row takes an `xcounter` |
| `uniform.<op>` | `u32.uniform` | `u32.uniform`, `u32.uniform` | as `index.<op>` | `<op>` in `{add, sub, mul, div, mod, min, max, shl, shr, and, or}`: 11 rows; arithmetic of uniforms is uniform (a wrap gives a uniform wrong value) |
| `index.of_uniform` | `u32.index` | `u32.uniform` | `{0}` | one way only |
| `uflag.<cmp>` | `bool.uflag` | `u32.uniform`, `u32.uniform` | `({0} < {1})` ... | `<cmp>` in `{lt, le, gt, ge, eq, ne}`: 6 rows |
| `uflag.and` / `uflag.or` / `uflag.not` | `bool.uflag` | `bool.uflag` (× 2 / × 2 / × 1) | `&&`, `\|\|`, `!` | 3 rows |
| `flag.of_uflag` | `bool.flag` | `bool.uflag` | `{0}` | one way only: a uniform predicate is a predicate; no row makes a `uflag` from a `flag` |
| `stmt.if_uflag_L<d>_f<k>` | `stmt.L<d>_f<k>` | `bool.uflag`, `block.L<d>_f<k>` | `if ({0}) { {1} }` | the body keeps the L rung: every invocation takes the same branch, so a barrier inside is reached by all (WGSL's uniformity rule) |
| `stmt.if_else_uflag_L<d>_f<k>` | `stmt.L<d>_f<k>` | `bool.uflag`, `block.L<d>_f<k>`, `block.L<d>_f<k>` | `if ({0}) { {1} } else { {2} }` | |
| `ureg.U<n>` | `ureg<u32>.uniform` | — | `U<n>` | `n` in `0..=7`: 8 terminals |
| `uniform.get_U` | `u32.uniform` | `ureg<u32>.uniform` | `{0}` | a uniform register reads as a uniform value: its only writers are the two rows below, both at L rungs from uniform sources |
| `ureg.set_L<d>_f<k>` | `stmt.L<d>_f<k>` | `ureg<u32>.uniform`, `u32.uniform` | `{0} = {1};` | at L rungs only (uniform control), from a uniform value only |
| `wg.u_10` | `wgmem<u32,1024>.index` | — | `wg_u_10` | terminal; the array is declared from the stratum map in the prologue |
| `wg.uload_L<d>_f<k>` | `stmt.L<d>_f<k>` | `ureg<u32>.uniform`, `wgmem<u32,1024>.index`, `u32.uniform` | `{0} = workgroupUniformLoad(&{1}[({2}) & 1023u]);` | WGSL §17.11.4: the pointer must be uniform (the index is a `u32.uniform`, masked by the row's text) and the call is a control barrier, so it stands at L rungs only; its result is uniform, so it lands in a `ureg` |
| `stmt.barrier_L<d>_f<k>` | `stmt.L<d>_f<k>` | — | `workgroupBarrier();` | §17.11.3; at L rungs only: every enclosing condition and bound is uniform by the types above, so every invocation reaches it (`barrier_in_nonuniform_control_flow`, `barrier_under_condition`) |

Batch 3: 52 family rows; 16 types added (plus the fuel suffix and the depth ladder on the statement types: `stmt`/`block` × `L0..5`, `C0..5`, `O0..4`, `B1..5` × 5 strata = 165 statement-level types in all, counted as patterns). Running total: 42 pattern types (165 statement-level instances); 97 + 52 = 149 family rows.

## Batch 4: the W phase and workgroup memory, reductions, atomics, subgroups

**The stratum map.** The interface's workgroup-memory map lists, per element type, the power-of-two strata it has; each listed `(e, 2^j)` is one array `wg_<e>_<j>`, declared in the prologue; their total is within the target. The worked map here: `f32` at `2^11`, `u32` at `2^10` (batch 3), `atomic<u32>` at `2^8`. `KNOB_THREADS = X * Y` is a prologue constant.

**The W value family (form A, ruled batch 4).** Outside W a foreign workgroup read is an ordinary value (`wg.get_f_11 -> f32.real`), so a seed's `x = wg[i] + 1.0` reads exactly. Inside W, statements take values of a twin family, `f32.realw`, `u32.indexw`, `u32.countw`, `bool.flagw` (and the vector twins of batch 5): every row of batches 2, 3 and 5 whose output is a value type and whose inputs are value types, builtins, registers, arrays, counters, uniforms or literals has a twin with the same spelling over the `w` family (`arith.add_w: (f32.realw, f32.realw) -> f32.realw`, `reg.get_f_w: reg<f32>.real -> f32.realw`, `literal.real_w`, `index.of_uniform_w`, `index.of_tcounter<d>_w`, `parr.get_f_6_w`, `load.<i>_w`, `load.<f>_w`, ...), **except** the foreign reads `wg.get_<e>_<j>`, which have no `w` twin. So no value a W statement takes can contain a foreign read, by the types. Four scalar twin types now (the vector twins come with the vectors); about 110 twin rows, generated from the value rows by suffix.

| type | what it holds |
|---|---|
| `block.W<d>` / `stmt.W<d>` | a write phase at depth `d` in `0..=5`: statements between a leading and a middle barrier; writes to workgroup arrays at this invocation's own elements only; no barrier, no foreign read, no return |
| `block.WB<d>` | a write-phase loop body that breaks, `d` in `1..=5` |
| `wgmem<f32,2048>.real` | the workgroup array `wg_f_11` |
| `wgmem<atomic<u32>,256>.index` | the workgroup array `wg_au_8` of atomics |
| `u32.wgidx_<j>` | an index below `2^j`, by the mask its row writes (`j` in the map: 8, 10, 11) |
| `u32.wgown_<j>` | the candidate `lidx + k * KNOB_THREADS` of this invocation's own element of a `2^j` array; the access rows guard it |
| `f32.realw` / `u32.indexw` / `u32.countw` / `bool.flagw` | the W twin of a value type: the values a W statement may take; produced by the twins of every value row except the foreign reads |
| `u32.litpow2_<j>` | a literal power of two from `1` to `2^j`: a reduction's element count |
| `u32.lane` | a subgroup lane below the subgroup size, by the mask its row writes |
| `u32.lanelit` | a literal lane `0..=min_subgroup_size - 1` of the target |
| `u32.atomicold` | the value an atomic read-modify-write returned: order-dependent across invocations; consumed only by `discard` |

| symbol | output | arguments (ordered) | WGSL spelling | note |
|---|---|---|---|---|
| `wg.f_11` | `wgmem<f32,2048>.real` | — | `wg_f_11` | terminal; the array is a name, read and written only by the access rows |
| `wg.au_8` | `wgmem<atomic<u32>,256>.index` | — | `wg_au_8` | terminal |
| `safe.wg_of_<j>` | `u32.wgidx_<j>` | `u32.index` | `(({0}) & (2^j - 1)u)` | `j` in `{8, 10, 11}`: 3 rows; the mask is the row's text, one index type per stratum, so an index can never reach an array shorter than its mask (`index_out_of_range`) |
| `wg.get_f_11` | `f32.real` | `wgmem<f32,2048>.real`, `u32.wgidx_11` | `{0}[{1}]` | a foreign read: an ordinary value outside W, in range by its index type; it has no `w` twin, so it can stand in no W statement (a foreign read inside a write phase would race with another invocation's write) |
| `wg.get_u_10` | `u32.index` | `wgmem<u32,1024>.index`, `u32.wgidx_10` | `{0}[{1}]` | likewise |
| `safe.wg_own_<j>` | `u32.wgown_<j>` | `u32.lit64` | `(lidx + {0} * KNOB_THREADS)` | 3 rows; unclamped: for one `k` the map `lidx -> lidx + k * KNOB_THREADS` is injective over the workgroup, so no two invocations name one element; a clamp would collapse lanes onto the last element (advisor finding 1) |
| `safe.wg_own_t<d>_<j>` | `u32.wgown_<j>` | `u32.tcounter<d>` | `(lidx + {0} * KNOB_THREADS)` | 5 × 3 rows; a uniform counter is the same `k` in every invocation at each trip, so the map stays injective; no row takes an `xcounter` (two invocations at different trips could name one element) |
| `wg.set_f_11_W<d>_f<k>` | `stmt.W<d>_f<k>` | `wgmem<f32,2048>.real`, `u32.wgown_<11>`, `f32.realw` | `if ({1} < 2048u) { {0}[{1}] = {2}; }` | the only write of `wg_f_11`; outputs a `W` statement only, so a write stands only inside a phase; guarded by the row's text, so it is in range and no two invocations write one element (`workgroup_memory_race`); `wg.set_u_10_W<d>_f<k>` likewise over `u32.wgown_10`, `u32.indexw` |
| `wg.get_own_f_11` | `f32.realw` | `wgmem<f32,2048>.real`, `u32.wgown_11` | `select(0.0, {0}[min({1}, 2047u)], {1} < 2048u)` | an own read: a `w` value, so it may stand inside W (it reads only the element this invocation writes); its plain twin `wg.get_own_f_11 -> f32.real` exists too; `wg.get_own_u_10` likewise with `0u` |
| `block.one_W<d>_f<k>` | `block.W<d>_f<k>` | `stmt.W<d>_f<k>` | `{0}` | `d` in `0..=5`; `seq2`, `seq3`, `seq4`, `stmt.block_W`, `own.block_W`, `own.stmt_W` (`{}`) as for the other kinds |
| `stmt.if_W<d>_f<k>` | `stmt.W<d>_f<k>` | `bool.flagw`, `block.W<d>_f<k>` | `if ({0}) { {1} }` | the body stays `W` (no barrier either way); `stmt.if_else_W` likewise |
| `reg.set_f_W<d>_f<k>` | `stmt.W<d>_f<k>` | `reg<f32>.real`, `f32.realw` | `{0} = {1};` | registers and private arrays are per invocation: the `set` rows of batch 3 exist at the W rungs too, over the `w` family (`reg.set_i_W`, `reg.set_b_W`, `parr.set_f_6_W`, `parr.set_i_4_W`) |
| `loop.tile_W<d>_f<k>` | `stmt.W<d>_f<k+1>` | `u32.lit64`, `block.W<d+1>_f<k>` | as `loop.tile` with `k<d+1>` | `d` in `0..=4`; the counter is uniform (literal bound), so `safe.wg_own_t<d+1>_<j>` over it stays injective; `loop.tilex_<c>_W<d>` with `block.W<d+1>` and `kx<d+1>`; `loop.tile_brk_W<d>` over `block.WB<d+1>`; `block.brk_WB<d>: (block.W<d>, bool.flag, block.W<d>) -> block.WB<d>`; `block.cont_W<d>` for `d >= 1` |
| `wg.phase_L<d>_f<k>` | `stmt.L<d>_f<k>` | `block.W<d>_f<k>`, `block.L<d>_f<k>` | `workgroupBarrier(); {0} workgroupBarrier(); {1}` | the phase: the leading barrier orders the writes after every earlier read, the middle barrier orders them before every later read; a `W` block holds no foreign read and writes own elements only, so no element is read by one invocation and written by another between two barriers; at L rungs only (uniform control) |
| `wg.put_f_11_L<d>_f<k>` | `stmt.L<d>_f<k>` | `wgmem<f32,2048>.real`, `u32.wgown_11`, `f32.real` | `workgroupBarrier(); if ({1} < 2048u) { {0}[{1}] = {2}; } workgroupBarrier();` | a one-write phase; `wg.put_u_10_L` likewise |
| `literal.litpow2_11` | `u32.litpow2_11` | — | `lit_u32(n)`, `n` in `{1, 2, 4, ..., 2048}` | `?_litpow2_11`: a power of two at most the array's stratum |
| `wg.reduce_<op>_f_11_L<d>_f<k>` | `stmt.L<d>_f<k+1>` | `wgmem<f32,2048>.real`, `u32.litpow2_11` | `{ var s = {1} >> 1u; loop { if (s == 0u) { break; } workgroupBarrier(); if (lidx < s) { {0}[lidx] = <op>({0}[lidx], {0}[lidx + s]); } s = s >> 1u; } workgroupBarrier(); }` | `<op>` in `{add (as +), max, min}`: 3 rows; the count is a power of two by its type, so every element of `a[0..n)` is paired at every step and the result in `a[0]` is the exact reduction (ruled); at each step invocation `lidx < s` writes `a[lidx]` and reads `a[lidx + s] >= s`: write and read sets are disjoint across invocations, and a barrier separates steps; at most 11 steps, class 1, so the stratum rises by one; `s` is the row's own local; in uniform control (L rung); elements `n..2048` are untouched |
| `atomic.<fun>_u_8` | `u32.atomicold` | `wgmem<atomic<u32>,256>.index`, `u32.wgidx_8`, `u32.index` | `atomic<Fun>(&{0}[{1}], {2})` | `<fun>` in `{add, sub, min, max, and, or, xor}`: 7 rows; race-free at any index and in any rung (an atomic is one operation); the returned old value is order-dependent, so it is `u32.atomicold`, whose only consumer is `discard`: no store, index or bound ever holds it (Codex rev4 finding 2) |
| `stmt.discard_u_<r>` | `stmt.<r>` | `u32.atomicold` | `_ = {0};` | every rung and stratum, W included: the effect without the value |
| `atomic.uload_u_8_L<d>_f<k>` | `stmt.L<d>_f<k>` | `reg<u32>.index`, `wgmem<atomic<u32>,256>.index`, `u32.wgidx_8` | `workgroupBarrier(); {0} = atomicLoad(&{1}[{2}]); workgroupBarrier();` | the strict load: the barriers order it after every read-modify-write before it and before any after it, so the value is the reduction's final value, the same under every order; L rungs only |
| `atomic.store_u_8_W<d>_f<k>` | `stmt.W<d>_f<k>` | `wgmem<atomic<u32>,256>.index`, `u32.wgown_8`, `u32.index` | `if ({1} < 256u) { atomicStore(&{0}[{1}], {2}); }` | a store at the invocation's own element inside a phase: deterministic; an `atomicStore` at an arbitrary index would leave one of several values, so it has no row |
| `builtin.subgroup_invocation_id` | `u32.index` | — | `sg_lane` | present only when the target has subgroups (`feature_missing`); the entry point declares `@builtin(subgroup_invocation_id) sg_lane: u32` then |
| `uniform.subgroup_size` | `u32.uniform` | — | `sg_size` | the subgroup size is the same in every invocation of the workgroup; `uniform.num_subgroups` likewise; `builtin.subgroup_id -> u32.index` |
| `subgroup.<op>_f_L<d>_f<k>` | `stmt.L<d>_f<k>` | `reg<f32>.real`, `f32.real` | `{0} = subgroup<Op>({1});` | `<op>` in `{add, mul, min, max, inclusive_add, inclusive_mul, exclusive_add, exclusive_mul}`: 8 rows; a collective is a statement at L rungs only, so it stands in uniform control (`subgroup_operation_in_nonuniform_control`); its result passes through a register; `subgroup.<op>_i_L` over `reg<u32>.index`, `u32.index` likewise (8 rows) |
| `subgroup.broadcast_first_f_L<d>_f<k>` | `stmt.L<d>_f<k>` | `reg<f32>.real`, `f32.real` | `{0} = subgroupBroadcastFirst({1});` | and `_i`; `subgroup.broadcast_f_L: (reg, f32.real, u32.lanelit)` spelled `subgroupBroadcast({1}, {2})`: WGSL requires a constant lane, and a `lanelit` is below the target's smallest subgroup size, so it selects an active invocation |
| `lane.of_index` | `u32.lane` | `u32.index` | `(({0}) & (sg_size - 1u))` | the only producer of a lane from a value: below the subgroup size by the row's text (the subgroup size is a power of two on every target the kingdom lists; the target line states it) |
| `lane.lit` | `u32.lane` | `u32.lanelit` | `{0}` | a literal lane below the target's smallest subgroup size |
| `literal.lanelit` | `u32.lanelit` | — | `lit_u32(n)`, `n` in `0..=min_sg - 1` | `?_lanelit` |
| `subgroup.shuffle_f_L<d>_f<k>` | `stmt.L<d>_f<k>` | `reg<f32>.real`, `f32.real`, `u32.lane` | `{0} = subgroupShuffle({1}, {2});` | the lane is below the subgroup size by its type, and at an L rung every invocation of the subgroup is active, so WGSL §17.12.12's indeterminate case (a lane that selects no active invocation) has no tree (ruled); `_i` likewise |
| `subgroup.shuffle_xor_f_L<d>_f<k>` | `stmt.L<d>_f<k>` | `reg<f32>.real`, `f32.real`, `u32.lane` | `{0} = subgroupShuffleXor({1}, {2});` | `lane ^ mask < sg_size` when both are below a power-of-two size |
| `subgroup.shuffle_down_f_L<d>_f<k>` | `stmt.L<d>_f<k>` | `reg<f32>.real`, `f32.real`, `u32.lane` | `{0} = subgroupShuffleDown({1}, {2});` | WGSL: `lane + delta` past the subgroup is indeterminate for the top lanes, so this row spells its own fill: `select(subgroupShuffleDown({1}, {2}), {1}, sg_lane + {2} >= sg_size)` — the value of the invocation itself where the source lane does not exist; `shuffle_up` likewise with `sg_lane < {2}` |
| `subgroup.all_L<d>_f<k>` | `stmt.L<d>_f<k>` | `reg<bool>.flag`, `bool.flag` | `{0} = subgroupAll({1});` | and `subgroup.any` |
| `stmt.subgroup_barrier_L<d>_f<k>` | `stmt.L<d>_f<k>` | — | `subgroupBarrier();` | present only with the subgroup-barrier feature |

Batch 4: 38 family rows plus the `w` twin family (about 110 twin rows generated by suffix; the W-rung twins of batch 3's sequence, if, set and loop rows are listed inside their rows); 16 types added (W and WB kinds as patterns; the four scalar `w` twins; `litpow2_11`, `lane`, `lanelit`). Running total: 58 pattern types; 149 + 38 = 187 family rows, plus the twin family.

## Batch 5: vectors and their arrays, i32 and the remaining scalars, the partitions, arenas, the storage barrier

**Lane counts.** `<L>` in `{2, 3, 4}`; `vec<L><S>.<form>` is the vector of `L` lanes of the scalar type `S.<form>`. Every scalar row of batches 2–4 whose operands are one scalar type exists per lane count with lanes equal across operands and result (`arith.add: (vec4<f32>.real, vec4<f32>.real) -> vec4<f32>.real`), and each vector type has its `w` twin. Rows listed below are the ones that change lane count or that only vectors have.

| type | what it holds |
|---|---|
| `vec<L><f32>.real`, `vec<L><u32>.index`, `vec<L><u32>.count`, `vec<L><i32>.int`, `vec<L><bool>.flag` | a vector of `L` lanes |
| `reg<vec4<f32>>.real` | a named vec4 register `v0..v31`; `reg<vec2<f32>>.real` `w0..w15`, `reg<vec3<f32>>.real` `C0..C15`, and one register kind per vector type likewise |
| `wgmem<vec4<f32>,512>.real` | the workgroup array `wg_v4_9` (the map's `vec4<f32>` stratum) |
| `parr<vec4<f32>,16>.real` | a private array of 16 vec4, `pv0..pv5`; `u32.pidx_4` indexes it |
| `i32.int` | a signed integer; `reg<i32>.int` `n0..n15` |
| `u32.bits` | a bit field: masks, packed lanes; no position, no size |
| `u32.hash32` | a 32-bit hash state |
| `u32.code<N>` | a codebook entry below `N`, `N` in `{4, 8, 16, 1024}`: literal domain `0..=N-1` |
| `i32.q15_16` | fixed point, 16 fractional bits; saturating arithmetic only |
| `bool.sign` | `+1`/`-1` as a bit |
| `u32.aidx_<a>` | an index into the arena binding `<a>`, below its declared length by the mask or modulus its row writes |
| `u32.atomicold` (relaxed) | as batch 4; in a `determinism relaxed` kingdom it also converts to a value |

| symbol | output | arguments (ordered) | WGSL spelling | note |
|---|---|---|---|---|
| `shape.splat<L>` | `vec<L><S>.<form>` | `S.<form>` | `vec<L><S>({0})` | per scalar family |
| `shape.compose2` | `vec2<S>.<form>` | `S.<form>`, `S.<form>` | `vec2<S>({0}, {1})` | `compose3: (vec2, S) -> vec3`, `compose4: (vec3, S) -> vec4` |
| `shape.access_<i>` | `S.<form>` | `vec<L><S>.<form>` | `{0}[<i>]` | `<i>` in `0..L-1`: a row exists only for `i < L`, so a constant lane is never past the vector |
| `shape.at` | `S.<form>` | `vec<L><S>.<form>`, `u32.index` | `at<L>({0}, ({1}) & (L-1)u)` for `L` in `{2, 4}` (a helper that switches on the lane); `at3({0}, ({1}) % 3u)` | a run-time lane is masked by the row's text; naga needs a constant index on a vector value, so the helper switches |
| `shape.swizzle_<pat>` | `vec<K><S>.<form>` | `vec<L><S>.<form>` | `{0}.<pat>` | `<pat>` the swizzles the interface's kernels use (`xy`, `xyz`, `zyx`, `xxxx`, ...), each a row only over lanes the source has |
| `arith.mul_scalar_vec` | `vec<L><f32>.real` | `f32.real`, `vec<L><f32>.real` | `({0} * {1})` | |
| `geom.dot` | `f32.real` | `vec<L><f32>.real`, `vec<L><f32>.real` | `dot({0}, {1})` | `geom.dot` over `vec<L><u32>.index` and `vec<L><i32>.int` wraps, defined |
| `geom.<op>` | per op | `vec<L><f32>.real` (× 1 or 2) | `length`, `distance`, `normalize`, `cross` (vec3 only), `reflect`, `refract(.., f32)`, `face_forward` | 7 rows; `normalize(0)` gives NaN lanes, scored |
| `compare.<cmp>_vec` | `vec<L><bool>.flag` | `vec<L><S>.<c>`, `vec<L><S>.<c>` | `({0} < {1})` ... | per lane; `logic.all`, `logic.any: vec<L><bool>.flag -> bool.flag`; `select.lane: (vec<L>, vec<L>, vec<L><bool>.flag) -> vec<L>` |
| `reg.v<n>` | `reg<vec4<f32>>.real` | — | `v<n>` | `n` in `0..=31`; `reg.get_v`, `reg.set_v_<r>` at every rung (W over the `w` twin); the other vector register kinds likewise |
| `reg.get_v_lane` | `f32.real` | `reg<vec4<f32>>.real`, `u32.index` | `{0}[({1}) & 3u]` | a register's lane at a run-time index, masked by the row's text; `reg.set_v_lane_<r>: (reg, u32.index, f32.real)` spelled `{0}[({1}) & 3u] = {2};` |
| `wg.v4_9` | `wgmem<vec4<f32>,512>.real` | — | `wg_v4_9` | terminal; `safe.wg_of_9`, `wg.get_v4_9 -> vec4<f32>.real` (foreign, no `w` twin), `wg.get_own_v4_9 -> vec4<f32>.realw`, `wg.set_v4_9_W<d>`, `wg.put_v4_9_L<d>`, `wg.reduce_<op>_v4_9_L<d>` as the f32 rows; lane forms `wg.get_v4_9_lane: (array, wgidx_9, u32.index) -> f32.real` spelled `{0}[{1}][({2}) & 3u]` and `wg.set_v4_9_lane_W<d>: (array, wgown_9, u32.indexw, f32.realw)` spelled `if ({1} < 512u) { {0}[{1}][({2}) & 3u] = {3}; }` |
| `parr.v<n>` | `parr<vec4<f32>,16>.real` | — | `pv<n>` | `n` in `0..=5`; `parr.get_v_4: (array, u32.pidx_4) -> vec4<f32>.real`, `parr.set_v_4_<r>`, and the lane forms `parr.get_v_4_lane`, `parr.set_v_4_lane_<r>` with `[({2}) & 3u]`; the vec2 arrays `pq0..pq3` and the 64-element vec4 arrays `pvq0..pvq3` (`u32.pidx_6`) likewise |
| `load.<i>4` | `vec4<f32>.real` | `u32.index` | `vec4<f32>(i[min({0}, L1)], i[min({0} + 1u, L1)], i[min({0} + 2u, L1)], i[min({0} + 3u, L1)])` with `L1 = arrayLength(&i) - 1u` | per `f32` input binding: four consecutive elements, each read clamped by the row's text (a `+ 3u` past `u32::MAX` wraps to a small index, still clamped, a wrong value) |
| `literal.int` | `i32.int` | — | `lit_i32(n)`, `n` in `i32` (`i32::MIN` spelled `(-2147483647i - 1i)`) | `?_int` |
| `int.<op>` | `i32.int` | `i32.int`, `i32.int` | `({0} + {1})`, `-`, `*`, `/`, `%`, `min`, `max`, `&`, `\|`, `^` | 10 rows; `+ - *` wrap (§6.2.3); `/` and `%` by a runtime zero give `e1` and `0`, `MIN / -1` gives `MIN` (§8.8): all defined |
| `int.<un>` | `i32.int` | `i32.int` | `(-{0})`, `abs`, `sign`, `~` | 4 rows; `abs(MIN) = MIN`, defined |
| `int.clamp` | `i32.int` | `i32.int` × 3 | `clamp({0}, {1}, {2})` | WGSL: indeterminate when `low > high`; the row spells `clamp({0}, min({1}, {2}), max({1}, {2}))` so every input is defined |
| `int.shl` / `int.shr` | `i32.int` | `i32.int`, `u32.index` | `({0} << {1})` / `({0} >> {1})` | §8.10: a runtime count is taken modulo 32; arithmetic right shift |
| `convert.index_to_int` | `i32.int` | `u32.index` | `i32({0})` | WGSL §17.1.2.5: a reinterpretation of bits, defined; `convert.int_to_index`: `u32({0})` likewise (§17.1.2.16); `index.of_int_floor`: `u32(max({0}, 0i))` as a separate row |
| `convert.f32_to_int` | `i32.int` | `f32.real` | `i32({0})` | §15.7.6: clamped and truncated for every non-NaN; NaN indeterminate (a value); `convert.f32_to_int_nan0` as for u32; `convert.int_to_f32`: `f32({0})` |
| `compare.<cmp>_int` | `bool.flag` | `i32.int`, `i32.int` | `({0} < {1})` ... | 6 rows; `select.int`; `reg.n<n>` `n` in `0..=15`, `reg.get_n`, `reg.set_n_<r>`; `uniform` has no i32 form (a loop bound is unsigned) |
| `literal.bits` | `u32.bits` | — | `lit_u32(n)` | `?_bits`; `bits.<op>: (u32.bits, u32.bits)` for `and, or, xor` and `bits.not`; `bits.shl`/`shr: (u32.bits, u32.index)` (§8.10 modulo 32); `bits.count_ones`, `leading_zeros`, `trailing_zeros`, `find_lsb`, `find_msb: u32.bits -> u32.count`; `bits.reverse` |
| `bits.extract` | `u32.bits` | `u32.bits`, `u32.index`, `u32.count` | `extractBits({0}, {1}, {2})` | WGSL §17.3: `offset + count > 32` is a shader-creation error only for constants; for runtime operands the spec clamps (`o = min(offset, 32)`, `c = min(count, 32 - o)`): defined; `bits.insert: (u32.bits, u32.bits, u32.index, u32.count)` likewise |
| `convert.index_to_bits` / `bits_to_index` / `count_to_bits` | `u32.bits` / `u32.index` / `u32.bits` | the other | `{0}` | the only crossings between a position, a size and a bit field |
| `convert.bitcast_f32_bits` / `bitcast_bits_f32` | `u32.bits` / `f32.real` | the other | `bitcast<u32>({0})` / `bitcast<f32>({0})` | a bitcast to f32 can make a NaN: a value |
| `bits.pack4x8snorm` ... | `u32.bits` | `vec4<f32>.real` or `vec2<f32>.real` | `pack4x8snorm({0})` ... | the five pack and five unpack built-ins, 10 rows |
| `hash.fmix32` | `u32.hash32` | `u32.hash32` | `fmix32({0})` (helper) | `hash.combine: (u32.hash32, u32.bits) -> u32.hash32`; `convert.hash32_to_bits`, `hash32_to_unit -> f32.real` (`f32({0} >> 8u) * 2^-24`); `hash32_to_index: (u32.hash32, u32.count) -> u32.index` spelled `({0} % max({1}, 1u))`; `literal.hash32` |
| `quantise.real_to_code<N>` | `u32.code<N>` | `f32.real` | `min(u32(clamp({0}, 0.0, N - 1.0) + 0.5), N - 1u)` | the row's text keeps the code below `N` for every input, NaN included (`clamp(NaN)` is NaN, `u32(NaN)` indeterminate, then `min(.., N - 1u)` bounds it); `quantise.code<N>_to_real: u32.code<N> -> f32.real`; `literal.code<N>` with domain `0..=N-1` |
| `fixed.<op>` | `i32.q15_16` | `i32.q15_16`, `i32.q15_16` | `add_sat`, `sub_sat`, `mul` (helpers: saturating) | 3 rows; `fixed.shl/shr: (q15_16, u32.count)`; `convert.f32_to_q15_16` (saturating), `q15_16_to_f32`; `literal.q15_16` |
| `convert.sign_to_real` | `f32.real` | `bool.sign` | `select(-1.0, 1.0, {0})` | `literal.sign`; `select.negate_if: (f32.real, bool.sign) -> f32.real`; `bits.pack_sign`, `unpack_sign` |
| `safe.thread_row_<b>` (partition forms) | `u32.rowbase_<b>` | per partition | contiguous `gid.y * KNOB_BROW` (batch 1); **strided** `(wid.y * (KNOB_WROW * KNOB_BROW) + lid.y)` over `vec3<u32>.wid`, `vec3<u32>.lid`; **grouped** `G` `(wid.y * (KNOB_WROW * KNOB_BROW) + lid.y * (KNOB_BROW / G))` over `wid`, `lid`; **lane** `W` `wid.x` over `vec3<u32>.wid` | one row per output, with the spelling its declared partition gives; the column row likewise (lane: `lidx` over `u32.lidx`). The interface declares the partition per output, so a kernel has one partition per output and no row mixes two |
| `safe.row_plus_<b>` (partition forms) | `u32.threadrow_<b>` | `u32.rowbase_<b>`, `u32.lit64` | contiguous `({0} + min({1}, KNOB_BROW - 1u))`; strided `({0} + min({1}, KNOB_BROW - 1u) * KNOB_WROW)`; grouped: see `row_plus_g`; lane: absent (block 1) | the offset stays inside the block under every partition; `safe.row_plus_t<d>_<b>` over `u32.tcounter<d>` and `safe.row_plus_x_<b>` over `u32.index` likewise, with the same clamp |
| `safe.row_plus_g_<b>` | `u32.threadrow_<b>` | `u32.rowbase_<b>`, `u32.lit64`, `u32.lit64` | `({0} + min({1}, G - 1u) * (KNOB_WROW * KNOB_BROW / G) + min({2}, KNOB_BROW / G - 1u))` | grouped partition only: group `g` of `G`, element `e` of `BROW / G`; both clamped, so the cell stays in the invocation's groups |
| `store.<b>_<r>` (lane form) | `stmt.<r>` | `u32.threadrow_<b>`, `u32.threadcol_<b>`, `<e>` | `if ({1} < Wu) { b[{0} * Wu + {1}] = {2}; }` | the lane partition's store: row `wid.x`, column `lidx`, width `W` from the interface; an invocation past the row's width writes nothing and the cell keeps the buffer's initial value; two invocations of one workgroup never share a cell, two workgroups never share a row |
| `arena.idx_<a>` | `u32.aidx_<a>` | `u32.index` | `(({0}) & (L_a - 1)u)` when `L_a` is a power of two, else `(({0}) % lit_u32(L_a))` | one per arena binding `<a>` (`policy arena`): the interface declares the arena's fixed length `L_a >= 1`; the mask or modulus is the row's text (`arena_index_not_in_bounds`) |
| `load.<a>` | `<e>` | `u32.aidx_<a>` | `a[{0}]` | a plain arena: in range by its index type; a value another workgroup may be writing (the design's own state; the interface declares it) |
| `store.<a>_<r>` | `stmt.<r>` | `u32.aidx_<a>`, `<e>` | `a[{0}] = {1};` | every rung and stratum (W included: an arena is not workgroup memory); two invocations may store one cell: WGSL gives one of the values, the arena policy accepts it |
| `atomic.<a>_<fun>` | `u32.atomicold` | `u32.aidx_<a>`, `u32.index` | `atomic<Fun>(&a[{0}], {1})` | an atomic arena (`array<atomic<u32>>`): 7 rows as for workgroup atomics; the old value is `atomicold`, consumed by `discard` in a strict kingdom |
| `atomic.<a>_load` | `u32.atomicold` | `u32.aidx_<a>` | `atomicLoad(&a[{0}])` | an in-kernel load of an arena atomic: no barrier spans workgroups, so its value is order-dependent and typed so; the harness reads the arena's final value after the dispatch |
| `index.of_atomicold` | `u32.index` | `u32.atomicold` | `{0}` | **relaxed kingdoms only** (`determinism relaxed`): the design consumes an order-dependent value (V24's slot allocation by `atomicAdd`); with it `atomic.exchange_u_8`, `atomic.<a>_exchange` (`atomicExchange`), `atomic.store_u_8_<r>` at any `u32.wgidx_8`, and `load.<a>` of an atomic arena as `atomicLoad(&a[{0}]) -> u32.index` exist; absent from a strict kingdom |
| `stmt.storage_barrier_L<d>_f<k>` | `stmt.L<d>_f<k>` | — | `storageBarrier();` | WGSL §17.11.2: a control barrier affecting the storage address space (arena stores before it are visible to the workgroup after it); at L rungs only, so in uniform control; `wg.phase_s_L<d>_f<k>: (block.W<d>, block.L<d>) -> stmt.L<d>` spelled `storageBarrier(); workgroupBarrier(); {0} storageBarrier(); workgroupBarrier(); {1}` is the phase whose separators also order arena stores |

Batch 5: 41 family rows (vector lane-count variants, `w` twins and per-rung sets generated by suffix); 14 pattern types added. Running total: 72 pattern types; 187 + 41 = 228 family rows, plus the lane-count variants and the `w` twin family.

Rows I am unsure of:

- `lane.of_index` masks by `sg_size - 1u`, which is a lane below the subgroup size only when the size is a power of two (true of every target the kingdom lists; the `target` line must state its subgroup sizes). A workgroup whose invocation count is not a multiple of the subgroup size has a partial last subgroup whose top lanes are inactive, and WGSL's shuffle to an inactive lane is indeterminate; the row above does not exclude it. The fix I propose: a target with subgroups admits only shapes whose `X * Y` is a multiple of its largest subgroup size (a shape-table fact), so every subgroup is full at an L rung.
- `shuffle_down` / `shuffle_up` spell their own fill for lanes past the edge (the invocation's own value); the alternative is a `u32.delta` type below `sg_size - sg_lane`, which is per invocation and cannot be a type.
- `shape.at` on a `vec3` uses `% 3u`, a different form from the `& (L-1)` masks; a `vec3` lane has no power-of-two mask.
- `int.clamp` reorders its bounds in the row's text because WGSL leaves `clamp` with `low > high` indeterminate; `arith.clamp` on f32 (batch 2) needs the same treatment and I have not changed it yet.
- `load.<i>4`: four clamps and a `+ 3u` that may wrap at the top of `u32`; defined, but the vec4 read of a seed written `vec4(phi[i], phi[i+1], ...)` without clamps reads as outside the kingdom.
- Arenas: the plain `store.<a>` admits a write race inside a workgroup and across workgroups by the arena policy; this is the design's state as today's arena mode has it, and it is the one place the kingdom spells a race on purpose.
