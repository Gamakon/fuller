# 9. The WGSL kingdom's symbol table (built one row at a time)

The kingdom holds any WGSL compute kernel over a declared **interface**: its storage bindings (each an input, read-only, or an output, read-write, with an element type `<e>` and a shape over the uniform's fields), its uniform block's fields, and the target's limits. Rows that mention a binding or a field are **families**, generated once per declared binding or field; the Gram interface (`phi: array<f32>` in, `h: array<f32>` out with shape `cfg.r_rows x cfg.d`, `cfg { d, m, r_rows, r_off }`) is one worked instance, never the default. Notation: `<b>` an output binding, `<i>` an input binding, `<f>` a uniform field, `<e>` the binding's element type (`f32`, `u32`, `i32`, `vec4<f32>`, ...). A 2-D output declares `shape <b> <R_b> <C_b>` (rows and columns, each an expression over fields) and is laid out row-major with stride `C_b`; a 1-D output declares `shape <b> <L_b>`. Every binding with load or store rows declares a length lower bound of at least one. `KNOB_BROW`, `KNOB_BCOL`, `KNOB_WROW`, `KNOB_WCOL` are `const`s the entry point writes from its grid node. `lit_u32`, `lit_f32`, `lit_bool`, `lit_i32` are identity helpers that keep every literal out of constant evaluation.

Batch 1 (2026-10-21): the kernel skeleton, the epilogue's store, the ownership rows. Batch 2: the remaining builtins and axes, u32 and f32 arithmetic, comparisons and logic, input loads and uniform fields. Later: fuel strata, loops, counters, the W phase, uniform provenance and barriers, registers and arrays, atomics, subgroups, the other partitions, arenas.

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
| `index.sub_floor` | `u32.index` | `u32.index`, `u32.index` | `({0} - min({0}, {1}))` | the kingdom's difference: floored at zero, so it never wraps below zero; the reader reads `a - b` as this row (same value where `a >= b`); a raw difference has no row |
| `index.div` | `u32.index` | `u32.index`, `u32.index` | `({0} / {1})` | WGSL's `x / 0 = x`: defined |
| `index.mod` | `u32.index` | `u32.index`, `u32.index` | `({0} % {1})` | WGSL's `x % 0 = 0`: defined |
| `index.min` | `u32.index` | `u32.index`, `u32.index` | `min({0}, {1})` | |
| `index.max` | `u32.index` | `u32.index`, `u32.index` | `max({0}, {1})` | |
| `index.clamp_to` | `u32.index` | `u32.index`, `u32.count` | `min({0}, max({1}, 1u) - 1u)` | `i < n`, and `0` for `n = 0` |
| `index.shl` | `u32.index` | `u32.index`, `u32.index` | `({0} << (({1}) & 31u))` | the count is masked by the row's text, so no shift by 32 or more exists (naga rejects it at constant evaluation; WGSL's runtime result would be the same masked one) |
| `index.shr` | `u32.index` | `u32.index`, `u32.index` | `({0} >> (({1}) & 31u))` | |
| `index.and` | `u32.index` | `u32.index`, `u32.index` | `({0} & {1})` | |
| `index.or` | `u32.index` | `u32.index`, `u32.index` | `({0} \| {1})` | |
| `count.add` | `u32.count` | `u32.count`, `u32.count` | `({0} + {1})` | sizes add (wrap defined); `count.sub_floor`, `count.min`, `count.max` as for `index` |
| `arith.<op>` | `f32.real` | `f32.real`, `f32.real` | `({0} + {1})`, `({0} - {1})`, `({0} * {1})`, `({0} / {1})`, `min(...)`, `max(...)`, `pow(...)`, `atan2(...)`, `step(...)` | `<op>` in `{add, sub, mul, div, min, max, pow, atan2, step}`: 9 rows; IEEE arithmetic, `x / 0` is an infinity or NaN, a value fitness scores, never a fault |
| `arith.<un>` | `f32.real` | `f32.real` | `neg: (-{0})`, else `<un>({0})` | `<un>` in `{neg, abs, sign, floor, ceil, round, fract, trunc, saturate, sqrt, inverse_sqrt, exp, exp2, log, log2, sin, cos, tan, tanh, asin, acos, atan}`: 22 rows; `sqrt(-1)` and `log(0)` give NaN or an infinity, scored |
| `arith.<tri>` | `f32.real` | `f32.real` × 3 | `clamp({0}, {1}, {2})`, `mix(...)`, `fma(...)`, `smoothstep(...)` | 4 rows |
| `convert.index_to_f32` | `f32.real` | `u32.index` | `f32({0})` | |
| `convert.count_to_f32` | `f32.real` | `u32.count` | `f32({0})` | |
| `convert.f32_to_index` | `u32.index` | `f32.real` | `u32(clamp({0}, 0.0, 4294967040.0))` | the clamp is the row's text: `u32()` of a negative, a NaN or a value past `u32::MAX` is indeterminate in WGSL, so the row fixes it (`4294967040` is the largest f32 below `2^32`) |
| `convert.f32_to_count` | `u32.count` | `f32.real` | as `convert.f32_to_index` | |
| `compare.<cmp>_index` | `bool.flag` | `u32.index`, `u32.index` | `({0} < {1})` ... | `<cmp>` in `{lt, le, gt, ge, eq, ne}`: 6 rows |
| `compare.<cmp>_count` | `bool.flag` | `u32.count`, `u32.count` | likewise | 6 rows; a comparison of a position with a size goes through `index.of_count` |
| `compare.<cmp>_real` | `bool.flag` | `f32.real`, `f32.real` | likewise | 6 rows; a comparison with NaN is false, defined |
| `logic.and` | `bool.flag` | `bool.flag`, `bool.flag` | `({0} && {1})` | |
| `logic.or` | `bool.flag` | `bool.flag`, `bool.flag` | `({0} \|\| {1})` | |
| `logic.not` | `bool.flag` | `bool.flag` | `(!{0})` | |
| `select.real` | `f32.real` | `f32.real`, `f32.real`, `bool.flag` | `select({0}, {1}, {2})` | `select.index`, `select.count` likewise |

Batch 2: 46 rows (families counted once; 3-row axis and 6-row comparison patterns as written). Types added in batch 2: `vec3<u32>.lid`, `vec3<u32>.wid`, `u32.lidx`, `vec3<u32>.nwg`, `u32.rowid`, `u32.colid`, `u32.count`. Running total: 26 types; 48 + 46 = 94 rows (families).

Rows I am unsure of:

- `convert.f32_to_index` clamps before the cast because WGSL leaves `u32(NaN)` and `u32(-1.0)` indeterminate (naga's backends differ); the clamp makes the row's value defined on every target. A seed written `u32(x)` reads as this row only when its `x` is provably in range, else as outside the kingdom.
- `index.sub_floor` as the only difference: a seed's `a - b` reads as it; where `a < b` the seed wrapped and the row floors, a different value. That is C4's named reason "a wrapping difference", not a rewrite.
- `load.<f>` outputs `u32.count` for every u32 field. `cfg.r_off` is used as a position (`gid.y + r_off`); it reaches `u32.index` through `index.of_count`, one extra node per use.
