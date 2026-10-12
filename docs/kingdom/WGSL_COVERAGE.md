# WGSL coverage of the safeWGSL kingdom

## Context

**Objective.** The kingdom has to be a fully evolvable WGSL AST (Andrew, 2026-10-12: "we are NOT BUILDING A POC"). Every WGSL compute construct must be a row of the kingdom, and every kernel naga accepts must read. A kernel our reader cannot read is a bug in our software.

**Approach.** This page lists the WGSL compute language from the W3C specification, area by area. Each construct gets one status, measured by a probe kernel rather than judged from the code. `kingdoms/safewgsl/coverage_probes.py` writes one Gram-role kernel per construct, and `wgsl_safe_convert` reads, renders and validates it. The rows that cover each construct are named.

**Status, 2026-10-12 (fuller 0203bb7).** 92 of 184 probes read, render and validate. The other 92 fail; their root causes are listed under "Gaps, in the order they are filled". Each fill is its own commit with a corpus row, and this page is updated as gaps close.

**Why.** Evolution can only find what the kingdom can spell. A construct the reader cannot read is a kernel, from a model or from a person, that the population cannot take in.

### Status keys

| status | meaning |
|---|---|
| covered | the probe reads, renders, validates; rows named |
| partial | some overloads or forms covered, the rest listed |
| missing | our software has a bug: naga accepts it, our reader does not read it |
| naga 0.20 | naga 0.20 (the wgpu 0.20 front end) cannot parse or lower it; it waits for the naga upgrade |
| binding | needs a binding the Gram role does not have (h, phi, cfg only); the generic interface is `docs/PLAN_constrained_mutation.md` |
| not compute | legal only in vertex or fragment shaders |

## Directives and module scope

| construct | status | rows / reason |
|---|---|---|
| `enable f16;` | naga 0.20 | naga 0.20 cannot parse an `enable` directive; the f16 duals exist (`f16.real`, `convert.f32_to_f16`) |
| `enable subgroups;` | naga 0.20 | same; naga 0.20 reads subgroup built-ins without it, the device asks for the feature |
| `requires`, `diagnostic` | naga 0.20 | not parsed |
| `const` at module scope, typed | missing | `const SCALE: f32 = 0.5;` read as leaf `const.SCALE`, no spelling |
| `const` at module scope, abstract | covered | `const N = 4;` folds into its uses |
| `const` in a function | naga 0.20 | "expected assignment or increment/decrement" |
| `const_assert` | naga 0.20 | not parsed |
| `override` | missing | leaf `override.SCALE`, no spelling (also needs pipeline constants at compile) |
| `override` in `@workgroup_size` | naga 0.20 | "Unexpected override-expression" |
| `alias` | covered | resolved by naga |
| `struct` (locals of struct type) | covered | members to registers |
| `@align`, `@size` on members | covered | resolved by naga |
| `var<private>` scalar | missing | `store.private.pg` has no spelling |
| `var<private>` array | missing | `store.private.pa.#` has no spelling |
| `var<workgroup>` arrays | covered | `wg.a`..`wg.d`, `wg.va`..`wg.vd`, `wg.v2a`..`wg.v2d`, `wg.decl_*` |
| `var<workgroup>` scalar | missing | `workgroupUniformLoad(&wv)` probe: "store to a workgroup array without an index" |
| `var<workgroup>` atomics | covered | `wg.au_*`, `wg.ai_*`, `atomic.*` |
| `var<storage>` / `var<uniform>` | binding | the Gram role's phi, h, cfg are rows (`load.phi_safe`, `load.h_safe`, `store.h_*`, `load.cfg_*`); other bindings need the generic interface |
| storage atomics | binding | `atomic<u32>` in a storage buffer |
| textures, samplers, `textureLoad` / `textureStore` / `textureDimensions` / `textureSampleLevel` / `textureGather` | binding | compute-legal, no texture binding in the Gram role |
| `textureSample`, `textureSampleBias`, derivatives | not compute | implicit derivatives |
| helper functions, one return | covered | inlined at the call (`stmt_call_value_helper`, `stmt_helper_with_loop`) |
| helper with an early return | missing | "helper clip returns twice (not inlined)" |
| helper with a pointer argument | missing | `bump(&g)`: "untyped leaf local.g@1" |
| `@must_use` | naga 0.20 | "unknown attribute" |

## Statements

| construct | status | rows / reason |
|---|---|---|
| `var` / `let` (typed and inferred) | covered | registers `reg.f*`, `reg.v*`, `reg.i*`, `reg.b*`; lets inline or register |
| assignment, `_ = e` | covered | `reg.set_*`, `stmt.discard_*` (atomics) |
| compound assignment on f32 | covered | lowered by naga |
| compound assignment on u32 (`^=` and the bit ops) | covered | `convert.index_to_bits`, `convert.bits_to_index` (the u32 form crossings) |
| `++` / `--` | covered | lowered by naga to `+ 1` / `- 1` |
| `if` / `else if` / `else` | covered | `stmt.if_*`, `stmt.if_else_*` |
| `switch` / `case` / `default` | covered as its if-else chain | WGSL has no fallthrough and a selector is a value; a `break` that ends a case is dropped, one earlier in a case is named. No switch row of its own |
| `for` | covered | `loop.items*`, `loop.tile*`, `loop.tilex*`, `loop.unroll_*`; a loop that starts at a literal a reads its counter as a + k (it was read as starting at 0, a bug that changed the meaning) |
| `while` | covered (bounded) | a while whose counter steps by a literal reads as its counted loop; an unbounded `while (c)` is outside the safe kingdom (a kernel must terminate) |
| `loop` + `continuing` | covered | as a for |
| `break if` | covered (bounded) | a do-while entered below its literal bound reads as its counted loop |
| `break` | covered | `loop.tile_brk_*`, `loop.tilex_brk_*` with body `block.brk_B<d>` (statements, `if (c) { break; }`, statements, all conditional: a loop that breaks holds no barrier, since a thread that breaks would skip the barriers the others reach). One break per loop body, at its top level |
| `continue` | covered | `block.cont_<rung>` at every loop rung: the guard, then the rest of the block at the conditional rung (no barrier after a continue). A barrier after a continue is refused by name |
| `return` | covered | `stmt.return_*` |
| function call statement | covered for pure helpers | see helpers |
| block `{ }` | covered | `stmt.block_*` |
| `workgroupBarrier()` | covered | `stmt.barrier_*` |
| `storageBarrier()` | covered | `stmt.storage_barrier_L*` (F1158-F1163); read as `workgroupBarrier()` before |
| `subgroupBarrier()` | covered | `stmt.subgroup_barrier_*` |
| `workgroupUniformLoad` | missing | needs a workgroup scalar |
| `textureBarrier()` | naga 0.20 | not parsed |
| `discard` | not compute | |

## Types

| construct | status | rows / reason |
|---|---|---|
| `bool`, `u32`, `f32` scalars | covered | duals `bool.flag`, `u32.index/count/bits`, `f32.real` |
| `i32` scalar values | covered | `reg.n0`..`n15`, `int.*`, literals written `-49i` |
| `f16` | naga 0.20 | needs `enable f16;` |
| abstract int / float literals | covered | `ty_abstract_literals`, hex and suffixed literals |
| `vec2<f32>`, `vec4<f32>` | covered | `shape.compose2/4`, `reg.v*` (vec4) |
| `vec3<f32>` | partial | `vec3` from `cross` and 3-lane swizzles: "geom.cross (vec3) where vec2 is demanded" |
| `vec2<u32>` | covered | |
| `vec3<u32>` as a value | missing | `gid + vec3<u32>(1u)`: the thread id cannot be an operand |
| `vec4<u32>` | partial | as a value (`ty_vec4_u32`) covered; no `vec4<u32>` register (blocks `subgroupBallot`) |
| `vecN<i32>` | covered as values | `int.*` per lane; no vector i32 register |
| `vecN<bool>` | partial | from comparisons with `any` / `all` covered; `vec2<bool>` built by a constructor in `select` missing |
| matrices `mat2x2<f32>` | missing | constructed from four scalars: no `shape.compose_mat` from scalars |
| matrices from columns (`mat3x3`, `mat4x4`, `mat3x4`) | missing | "naga.Compose.type6 has no row" |
| matrix locals, column access `m[i]`, `m[i][j]` | missing | no matrix register; `AccessIndex` on a matrix |
| `array<T, N>` locals of f32, u32, bool, vec4 | covered | private arrays `parr.*` |
| `array<vec2<f32>, N>` local | missing | "has no private array" |
| `array<S, N>` of structs | covered | |
| array constructors | covered | |
| runtime-sized arrays, `arrayLength` | missing | "naga.ArrayLength has no row" (phi is runtime-sized; its length is a binding fact) |
| `atomic<u32>`, `atomic<i32>` in workgroup memory | covered | |
| `ptr<function, T>` lets, `*p` | covered | `decl_ptr_local`, `decl_ptr_array_elem` |
| `ptr` as a helper argument | missing | see helpers |
| structs as values (`modf`, `frexp` results) | missing | "naga.AccessIndex.0 has no row" |

## Expressions and operators

| construct | status | rows / reason |
|---|---|---|
| `+ - * /` on f32 and vectors | covered | `arith.*` |
| `%` on f32 | covered | `arith.mod` (F1157) |
| `%` on u32 | covered | `index.mod` |
| unary `-` on vectors | covered | |
| comparisons, `&&`, `\|\|`, `!` | covered | `compare.*`, `logic.*` |
| `& \| ^ << >>` on u32 | covered | `bits.*`, `index.*`, with the u32 form crossings |
| `~` | covered | `bits.not` |
| swizzles, 1, 2 and 4 lanes | covered | `shape.access_*`, `shape.swizzle` |
| swizzles, 3 lanes | missing | `shape.swizzle.zyx` where vec2 is demanded |
| swizzle assignment | covered | `reg.set_v_lane_*` |
| `select` scalar | covered | `select.scalar_cond` |
| `select` with a `vec2<bool>` built by a constructor | missing | |
| `bitcast<f32>(u32)`, `bitcast<u32>(f32)` | covered | `convert.bitcast_*` |
| `bitcast<i32>(u32)`, `bitcast<u32>(i32)`, `bitcast<i32>(f32)`, `bitcast<f32>(i32)` | covered | `convert.bitcast_*` |
| `bitcast` on vectors | missing | |
| conversions `f32(u32)`, `u32(f32)`, `f32(bool)`, `u32(bool)` | covered | `convert.*` |
| `i32(u32)`, `u32(i32)`, `i32(f32)` | covered | `convert.index_to_int`, `int_to_index`, `f32_to_int` |
| a literal written as `2.0` beside an f32 in `max`, `clamp`, `mix`, `pow`, `step`, `smoothstep`, `fma`, `ldexp`, `atan2`, `dot`, `distance`, `reflect`, `faceForward`, `length`, `normalize` | covered | naga 0.20 lowers the abstract literal through `As { convert: Some(4) }` f32 to f32; the reader reads a conversion to the operand's own kind and width as the operand (`draft_float_literals`) |

## Built-in values (compute)

| value | status | rows |
|---|---|---|
| `global_invocation_id`, `local_invocation_id`, `workgroup_id`, `local_invocation_index` | covered | `builtin.*`, `axis.*` |
| `num_workgroups` | covered | `builtin.num_workgroups` (naga names it `num_work_groups`) |
| `subgroup_invocation_id`, `subgroup_size`, `num_subgroups`, `subgroup_id` | covered | `builtin.subgroup_*` (bdca29a) |

## Built-in functions

| group | status | rows / reason |
|---|---|---|
| float, one argument, f32 and vec4 (abs, acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, degrees, exp, exp2, floor, fract, inverseSqrt, log, log2, radians, round, saturate, sign, sin, sinh, sqrt, tan, tanh, trunc) | covered | `arith.*`, `trig.*` (60 probes) |
| `length`, `normalize`, `distance`, `dot`, `reflect`, `faceForward` | covered | `geom.*` |
| `cross` | missing | the vec3 gap |
| `refract` | covered | `geom.refract` |
| two and three arguments (atan2, clamp, fma, ldexp, max, min, mix, pow, smoothstep, step) | covered | `arith.*` |
| `mix` with a scalar weight | covered | `arith.mix_scalar_weight` |
| `modf`, `frexp` | missing | struct results |
| `quantizeToF16` | naga 0.20 | not parsed |
| integer abs, min, max, clamp, sign on i32 | covered | `int.*` |
| bit functions and shifts on i32 | covered | `int.and/or/xor/not/shl/shr`, `int.count_ones` ... `int.insert` |
| min, max, clamp on u32 | covered | `index.min/max/clamp` |
| `dot` on integer vectors | covered | `geom.dot` (F1155, F1156) |
| countOneBits, countLeadingZeros, countTrailingZeros, firstLeadingBit, firstTrailingBit, reverseBits on u32 | covered | `bits.*` |
| the same on vecN<u32> | covered | `bits.*` per lane (F1130-F1153) |
| `extractBits` (u32, vecN<u32>) | covered | `bits.extract` |
| `insertBits` | covered | `bits.insert` (F1154, arity 4) |
| pack4x8snorm/unorm | covered | `bits.pack*` |
| pack2x16snorm/unorm/float | partial | rows covered; the probes' `v.xy` argument fails (the swizzle gap) |
| unpack4x8snorm/unorm, unpack2x16snorm/unorm/float | covered | `bits.unpack*` (their naga column was `UnPack…`; naga writes `Unpack…`) |
| pack4xI8, pack4xU8, unpack4xI8, unpack4xU8, dot4I8Packed, dot4U8Packed | naga 0.20 | not parsed |
| `select`, `all`, `any` | covered | |
| `arrayLength` | missing | |
| atomics (workgroup): load, store, add, sub, max, min, and, or, xor, exchange | covered | `atomic.*` |
| `atomicCompareExchangeWeak` | naga 0.20 | Metal backend: "FeatureNotImplemented(atomic CompareExchange)" |
| subgroup reductions and scans on f32, u32 and vec4 | covered | `subgroup.*` |
| subgroup operations on i32 | covered | `subgroup.*` on `i32.int` |
| `subgroupBallot` | missing | its `vec4<u32>` result has no register |
| `subgroupElect`, `quadBroadcast`, `quadSwap*` | naga 0.20 | not parsed |
| `workgroupUniformLoad` | missing | workgroup scalar |
| derivatives (`dpdx` ...), `textureSample` | not compute | |

## Gaps, in the order they are filled

Each fill is its own commit with a corpus row and the full checklist. The order is by how many probes a fix reads.

1. Done: an f32-to-f32 `As` (naga's lowering of an abstract float literal) read as its operand: 16 probes (108 of 184 read).
2. Done: conversions between the u32 forms (`index`, `bits`, `count`), and vector overloads of the bit functions.
3. Done: the `unpack*` naga names, `insertBits`, `dot` on integer vectors, `%` on f32; a naga node the WGSL table has no row for is resolved in the safe table. Gaps 2 and 3 read 26 probes more (134 of 184).
4. Done: `storageBarrier()` as its own statement row (it was misread as `workgroupBarrier()`).
5. Done: `num_workgroups`, naga's name `num_work_groups` mapped to the row. 136 of 184 read. Also read now: a kernel that never writes h (`draft_no_store`; a mutant's top-level return).
6. Done: i32 registers, `i32(u32)`, `u32(i32)`, `i32(f32)`, `bitcast<i32>`, and with them `++` / `--`, i32 `abs/min/max/clamp/sign`, i32 subgroup operations and vecN<i32>; an atomic<i32>'s old value read later. 147 of 184 read.
7. Done: `switch`, `break`, `break if`, `while` with its step in the body, `continue` in any loop; loops entered at a non-zero literal. 153 of 184 read.
8. vec3 values (`cross`, 3-lane swizzles, the thread id as an operand), `vec4<u32>` registers (then `subgroupBallot`), and vec2 private arrays.
9. Matrices: construction from scalars and columns, matrix locals, column access.
10. `modf` / `frexp` results, `arrayLength`.
11. Module `const` and `override`, `var<private>`, workgroup scalars and `workgroupUniformLoad`.
12. Helpers with an early return or a pointer argument.

Not built here: the Gram role's bindings (h, phi, cfg) are the largest gap. Other storage buffers, uniforms, storage atomics, textures and samplers all wait for the generic interface in `docs/PLAN_constrained_mutation.md`, which is awaiting Andrew's review.
