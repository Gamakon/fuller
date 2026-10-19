# PLAN: naga and wgpu 30.0.1 together, in hff, fuller and phylu

## Context

**Objective.** Upgrade naga 0.20.0 and wgpu 0.20 to 30.0.1 in hff, fuller and
phylu together, so one naga (30.0.1) serves the reader, the host checks and the
device compiler. The WGSL kingdom then reads current WGSL: function-local
`const`, for one, and the 38 cuneus kernels, written for naga 30.

**Approach.** The upgrade has one rule: the kingdom does not move. Its rows and
duals stay the same, and so do the saved genes and the trees the reader
builds, with their roots and hole numbers. Where naga 30 lowers the same text
into other IR, the readers view it as the rows it was written as, without changing the IR
(`src/wgsl/naga_shape.rs`). The short-circuit `&&`/`||` is viewed as
logic.and and logic.or, and a phony `_ =` as a discard. A feature scan
refuses, by name, every variant the kingdom has no row for, right after the
one parse function. Where naga 30 is right and 0.20 was wrong (a `var`
declared in a loop is zeroed on every pass), the reader reads naga 30's
meaning. The wgpu changes are API changes, made the same way in all three
crates. One question is open: whether naga 30's new shader runtime checks run
on our kernels. It is measured, not assumed (section 4).

**Unique feature.** The host checks and the device compile with the same naga
again. Today's comment in fuller Cargo.toml ("the same naga wgpu already
pulls in") stays true, and `msl-out` is declared instead of arriving through
wgpu's feature unification. Upgrading naga alone was ruled out (the lead,
2026-10-19). With naga 30 on the host and naga 0.20 inside wgpu, the two
disagree in both directions: naga 30 accepts what wgpu 0.20 refuses, and it
refuses our `subgroupBroadcast` lanes.

**Status (2026-10-19).** This plan, from the code read in full (section 1);
not built. Worktrees on branch `naga-30` in all three repos, under
`scratchpad/n30/{hff,fuller,phylu}`, branched from hff 77a36b4,
fuller bf9ad3f (rebased onto fuller's head before the build) and phylu
9c531d41. Trial `cargo check` with both crates at
30.0.1: hff 11 errors, fuller 176. phylu's own errors stay unseen until
fuller compiles. `cargo tree -i naga` on a scratch crate with wgpu 30 and
naga 30 shows one naga, v30.0.1. Codex review: done 2026-10-19, 8 findings, all taken (section 6).

**Version.** 30.0.1, the latest on crates.io (the lead, 2026-10-19). naga
30.0.1's src is byte-identical to 30.0.0's (`diff -r`); wgpu 30.0.1 differs
from 30.0.0 only in its browser (webgpu) backend; the facts below were read
in 30.0.0 and hold for both.

**Why.** naga 0.20 cannot parse current WGSL. The external corpora (cuneus
on naga 30) and model-written kernels use it, and every one we cannot read
is a kingdom gap we cannot even see.

**Bars (the lead, 2026-10-19).** Zero warnings; the CPU tests pass. Floors:

| Measure | Floor |
|---|---|
| Random trees, AST | 128 |
| Random trees, fixpoint | 959 |
| Modules, AST | 11 |
| Modules, fixpoint | 970 |
| Modules, legal | 134 at least |

Saved genomes must read back 481 of 481. The model-reply corpus
(tests/corpus/model_replies, measured by `wgsl_safe_convert`) must convert
1,756 or more of 1,798; that is the 2026-10-17 floor, re-measured 1,756 at
fuller 7e94cba on 2026-10-19 09:56Z. The safewgsl manifest test must hold
its 101 of 112 exactly. Then the cuneus kernels without images, read and counted. GPU tests
are built, not run: the lead runs them, phylu's SR and regex goldens among
them.

## 1. What is there (read in full)

Three readers covered the code. Reader one read safe_reader, reader,
interface, safety and naga_names. Reader two read cpu, interp, scaffold,
harness, target, render, versions, oracle, roundtrip, gram, check, llm,
mod.rs and the naga examples. Reader three diffed naga 0.20.0 against 30.0.1
source; naga's CHANGELOG stops at 0.14. All three reports are pasted below
in substance.

- fuller: 481 `naga::` sites in 21 files. 31 `parse_str` calls in 17 files,
  each its own parse; no common entry. 560 `wgpu::` sites in 13 files:
  gpu_eval 178, wgsl/harness 74, lint/snap_table 57, wgsl/gram 49,
  lint/snap_guard 48, lint/device 41, lint/snap_graft 36, wgsl/oracle 32,
  pattern_match 30, three examples, target 2.
- phylu: 514 `wgpu::` sites in 10 files: regex_matcher 133, device 129,
  hff_gpu 59, score 47, fold_to_dag 44, regex_resident 42, resident 32,
  typer 26, engine 1, one example. No `naga::` use: phylu reads WGSL only
  through fuller.
- hff: 44 `wgpu::` sites in src/gpu.rs.
- 24 `create_shader_module` calls; none use `_unchecked` or `_trusted`.
  16 error-scope calls.
- No naga value ever crosses into wgpu: no `ShaderSource::Naga`, text only.

### 1.1 naga: changes the compiler reports

All file:line references are fuller's. Each has a fixed replacement.

| naga 0.20 | naga 30 | Sites |
|---|---|---|
| `Statement::Barrier(Barrier)` | `ControlBarrier(Barrier)` and `MemoryBarrier(Barrier)` | reader 733; safety 2243, 2457; safe_reader 4501-4513, 4920, 5916, 5947; cpu 300, 1428; scaffold 536; versions 70 |
| `Atomic.result: Handle` | `Option<Handle>` | reader 631-659; safety 2259-2266; safe_reader 3833, 3854, 4229, 5021-5026; cpu 323, 1454 |
| `MathFunction::FindLsb`, `FindMsb` | `FirstTrailingBit`, `FirstLeadingBit`, same meaning; 11 new functions | naga_names 169-170, 246-247, 421-422; cpu 633-647; interp |
| `BuiltIn::ClipDistance` | `ClipDistances`; 24 new builtins | naga_names 271, 304, 440 |
| `AddressSpace::PushConstant` | `Immediate`; new TaskPayload, RayPayload, IncomingRayPayload | reader 811-818; versions 146-153 |
| `Literal` | gains `F16`, `U16`, `I16` | reader 958-970 |
| `ArraySize` | gains `Pending(Handle<Override>)`, an override-sized workgroup array | interface 109 |
| `GatherMode` | gains `QuadBroadcast`, `QuadSwap` | reader 682-689; cpu 1483-1490 |
| `Function` | gains `diagnostic_filter_leaf` | scaffold 436 |
| `valid::ExpressionError::IndexMustBeConstant` | gone: a run-time index into a value array or matrix now validates | reader 221, 269 |
| `Capabilities` | u64: PUSH_CONSTANT is IMMEDIATES, CLIP_DISTANCE is CLIP_DISTANCES; the three non-uniform-indexing flags become four `*_BINDING_ARRAY_NON_UNIFORM_INDEXING` | target 59-84 |
| `Binding::Location.second_blend_source` | `blend_src: Option<u32>`, plus `per_primitive` | wherever a Location is built |
| `Statement` | gains `ImageAtomic`, `RayPipelineFunction`, `CooperativeStore` | the exhaustive matches: safety::stmt 2089-2318, safe_reader::stmt_name 5936-5955, cpu 1354-1499, versions 66-109 |

Each exhaustive match names the new variants explicitly, as refusals by
name: "our software has a bug: <variant> in a compute kernel has no row".
Wildcards hide them today (interface 192, reader 727, safety pure_block 2853,
scaffold 632). MemoryBarrier must be named everywhere ControlBarrier is, so
it cannot fall into a wildcard.

### 1.2 naga: the same text, other IR (no compiler error)

- **N1. Short-circuit `&&` and `||`** (naga 30 front/wgsl/lower/mod.rs
  2563-2691). In a runtime context, `a && b` becomes an unnamed bool local
  `v`, then `If { condition: a, accept: [<b's Emits and Calls>, Store(v, b)],
  reject: [Store(v, false)] }`, then `Load(v)`. `a || b` is the same over
  `!a`, with `true` in the reject. This change is most of the 176 rendered
  texts reader three found differing out of 235 fuller kernels. Affected:
  cpu 363-393, 1100; interp 717-727, 1050; safety 1240; render 2486; every
  reader root list.
- **N2. `_ = e` is a named expression "phony"** (lower 2294). Today reader
  455-483 makes any named expression a Let root (`is_bake` 932 filters only
  `_eN`), and safe_reader 2753-2783 would put a discarded subgroup operation
  into a register.
- **N3. `var x: T;` with no initializer, inside a loop,** gets
  `Store(x, ZeroValue)` on every pass (lower 1928-1937). naga 0.20 left the
  store out, which was wrong by the WGSL spec. ZeroValue reads as scalar 0
  even for a vector or array (reader 972).
- **N4. More constant folding in function bodies.** select, all/any, atan2,
  dot, cross, length, distance, normalize, firstLeading/TrailingBit, dot4*,
  unpack4xU8 and matrix arithmetic over constants now fold. Integer overflow
  in constants wraps, `sign(0.0)` is 0, and math is folded in f64 then
  rounded. Bitcast still does not fold. The emitter already writes every
  literal-only operation through `lit_*` calls (render 2044-2066,
  `const_depth`), so a drawn tree folds no further. Model kernels may.
- **N5. Module consts with an untyped initialiser stay abstract,** and their
  run-time uses become Literals: the leaf `const.N` vanishes (reader 973;
  safe_reader 1845-1878, step_of and module_constant).
- **N6. Function-local `const`** parses now; its uses are inlined as
  literals, with no name left.
- **N7. `ensure_block_returns` appends a Return after a trailing Emit,** so a
  path gains a statement.
- **N8. Debug spellings change.** "Barrier" is now "ControlBarrier"
  (reader 728-730, variant_name 864, safety pure_parts 1281). `Handle`'s
  Debug is 0-based, where 0.20 printed index+1. Math names go through
  `Math::FindLsb` in kingdoms/{wgsl,safewgsl}/functions.tsv's naga column
  (reader 1063), which no longer matches.
- **N9. Handle indices shift.** N1's hidden locals and N3's stores add arena
  entries. Every name built from a handle index (`let.x@N`, `call.f@N`,
  `atomic@N`, `load.local.i@N@vK`, `naga.Compose.typeN`) moves for a kernel
  that holds one. Genes name items by position in naga's module (scaffold
  780, 898, 902, 1040).
- **Verified unchanged:** the for/while lowering (an If with an empty accept
  and a Break, then the body block, the step in continuing, break_if); `let`
  as a named expression; var init; atomicLoad and atomicStore as Load and
  Store; increments; the emitter and bake prefix `_e`; subgroup lowering;
  bitcast; arrayLength; parse_str, emit_to_string, get_entry_point,
  Validator::new, msl::write_string.

### 1.3 naga: validator and back ends

- **New rejections of what 0.20 accepted:**
  - an Emit that covers a result expression, or a result not used exactly
    once
  - a constant shift that is too large, or a constant zero divisor
  - a `subgroupBroadcast` lane that is not a const-expression; the emitter
    writes the lane as `lit_u32(..)` (render 2062), a call, so render tests
    4495 and 4603 and the check.rs random genes fail
  - pack2x16float without SHADER_FLOAT16_IN_FLOAT32
  - unary minus on a matrix (the emitter already writes `* -1.0`)
- Structs compare by handle. `enable subgroups;` is a parse error, so we
  never write it. `validate_no_overrides` is now `validate_resolved_overrides`.
- **MSL:** `force_loop_bounding` and `emit_int_div_checks` default on. naga 30
  writes `atomicCompareExchangeWeak` to Metal, so safe_reader 8150 (expects an
  error) and the refusal reasons at safe_reader 2927 and 2956 become false.
- **WGSL writer:**
  - names that clash with builtins get `_` (`min_`)
  - overrides and `enable` lines are written
  - two bugs: `@workgroup_size(<override>)` is written `(1, 1, 1)`, and
    `array<T, <override>>` loses T
  - affected: scaffold tests 1139, 1146, 1198, 1226, 1252, 1282;
    roundtrip 122-123, 177, 197
- **API:**
  - Range is `index_range` / `from_index_range`
  - `compact` takes `KeepUnused`
  - FastHashMap is hashbrown
  - `Validator::new` fills subgroup support from the capabilities;
    target 91-95 still overrides it

### 1.4 wgpu 0.20 to 30: the trial check's errors

Counts are fuller's; hff shows the same kinds.

| Error | Count | Change |
|---|---|---|
| `Maintain` | 29 | `device.poll(PollType::Wait…)`, which returns `Result<PollStatus, PollError>`: a poll error is an error, never ignored |
| `PipelineLayoutDescriptor.push_constant_ranges` | 11 | the immediates field |
| compute pipeline `cache` | 12 | `cache: None` |
| `entry_point` | 10 | now `Option<&str>`: `Some("main")` |
| `layout` | 9 | `Option<&BindGroupLayout>` |
| `get_mapped_range` | 16 | returns `Result<BufferView, MapRangeError>`: `?` with the error named |
| `pop_error_scope` | 8 | `push_error_scope` returns an `ErrorScopeGuard` whose `pop()` gives the error |
| `request_adapter` | 4 | returns Result, not Option (`ok_or` becomes `map_err`) |
| `DeviceDescriptor` | 4 | `experimental_features`, `memory_hints`, `trace` |
| `RequestAdapterOptions.apply_limit_buckets` | 1 | |
| `InstanceDescriptor::default` | 1 | gone (hff) |
| Features | 3 | PUSH_CONSTANTS, SHADER_PRIMITIVE_INDEX and UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING renamed |

wgpu 30's `create_shader_module` compiles with
`ShaderRuntimeChecks::checked()`: bounds checks, `force_loop_bounding` and
`int_div_checks`, all on. wgpu 0.20 applied bounds checks only.

## 2. The change

### 2.1 One parse, one feature scan, a short-circuit view (fuller)

The IR is never rewritten. Codex's review (2026-10-19, section 6) refuted
the first draft, which folded naga 30's short-circuit back into an eager
Binary. That changes what a kernel does: an `a && f()` whose value is never
loaded, or a right side that loops, or a nested `&&`. The device runs naga
30's IR, and so do the CPU executor and the safety interpreter, unchanged.
Only the readers that build kingdom trees view the short-circuit as the
logic row it was written as.

- **`src/wgsl/naga_shape.rs`, the parse:**
  - `pub fn parse(text) -> Result<naga::Module, String>` is naga's
    `parse_str` with the error emitted, then `supported(&module)`.
  - All 31 parse sites move to it (17 files). A test greps the source and
    fails on any other `parse_str`.
- **`supported`, a feature scan** (Codex 4). It walks every type, global,
  function, expression and statement once, and refuses by name everything
  the kingdom has no row for:
  - `TypeInner::CooperativeMatrix`, `AccelerationStructure`, `RayQuery`,
    `Literal::F16`/`U16`/`I16`, `ArraySize::Pending`
  - the address spaces Immediate, TaskPayload, RayPayload and
    IncomingRayPayload
  - the expressions CooperativeLoad, CooperativeMultiplyAdd and the ray ones
  - the statements MemoryBarrier, ImageAtomic, RayPipelineFunction and
    CooperativeStore
  - any builtin outside compute's
  - the error is "our software has a bug: <variant> has no row in the
    kingdom"

  It runs before any reader, the safety checker or the CPU executor sees the
  module, so no wildcard downstream can silently swallow a new variant. The
  exhaustive matches then name the new variants as unreachable with that
  message, never `_`.
- **`ShortCircuit`, a view, not a rewrite** (decision a; Codex 1-3).
  `short_circuits(function) -> HashMap<Handle<LocalVariable>, ShortCircuit
  { op, lhs, if_path, rhs, rhs_block }>`, computed once per function by a
  post-order walk of every block: Block, If accept and reject, Switch cases,
  Loop body and continuing. The match is exact:
  - an `If { condition, accept, reject }`, where the local v is unnamed,
    bool and has no init
  - reject is exactly `[Store(LocalVariable(v), Literal(Bool(false)))]` for
    `&&`, whose condition is the lhs; or `Literal(Bool(true))` for `||`,
    whose condition is `Unary(LogicalNot, lhs)`
  - accept is Emits, Calls, nested ShortCircuit Ifs (recursively matched,
    `a && (b && c)`), and last `Store(LocalVariable(v), rhs)`
  - v is stored only there

  The tree readers (reader.rs, and through it safe_reader; scaffold and
  interp for their holes) use the view:
  - the If at `if_path` makes no root, no hole and no statement
  - each `Load(LocalVariable(v))` expands to `logic.and(lhs, rhs)` or
    `logic.or(lhs, rhs)`
  - the accept's Calls are read where the reader reads any call, as the
    roots they are today, in the same order

  That holds only when every callee is one the reader already reads as a
  pure helper (the `lit_*` pass-throughs, the kingdom's helpers: no store
  through a pointer argument or to a global, no atomic, no barrier, no
  loop). For any other accept the view is not taken: the If is read as the
  if it is, its rung's rules apply, and a refusal names it. A v that is
  never loaded makes the view empty: the If is read as an if, so its calls
  are not dropped.
- **The bar for the view** (decision a): roots and hole numbers exactly
  today's. Measured by the reading snapshot of section 3: every tree of
  every committed kernel, sexpr for sexpr, against naga 0.20's reading.
  Fixtures, each read and checked against its expected tree:
  - an unused `a && f()`
  - `if (a && b)`
  - `continuing { break if a && b; }`
  - a loop's `a < n && b`
  - nested `a && (b || c)`
  - two Emit ranges around a call
- **N2.** A named expression "phony" is the discard it is (`_ = e`). The
  readers skip it as they skip `_eN`; nothing in the module is changed.
- **N9, handle indices.** The view leaves naga 30's arena as it is, so the
  hidden local shifts later locals' indices. Names built from handle indices
  are made order-preserving: each local's ordinal among the function's
  *named* locals, in place of its raw handle index. A hidden local has no
  name, so a reading's names equal 0.20's. The reading snapshot proves it.
  If it does not hold for some name, that name is listed and decision (c)'s
  re-encode applies, with the lead told first.

### 2.2 fuller: reader, safety, interpreters

- Section 1.1, line by line: the new names. The exhaustive matches name every
  new variant. `Literal::F16` / `U16` / `I16`, `ArraySize::Pending` and the
  new address spaces, builtins and gather modes each refuse by name.
- **IndexMustBeConstant (decision b).** reader.rs makes the check itself: a
  run-time index into a value (not pointer) array or matrix is refused by
  name, at the site the validator error was caught (reader 221, 269). The
  lenient path that relied on the error is replaced by this check.
- **N3.** A `Store(local, ZeroValue)` inside a loop reads as the register
  set to its fallback, which is the zero, at that statement (naga 30's
  meaning). For a vector or array ZeroValue reads as that dual's fallback,
  not scalar 0 (reader 972 fixed).
- **N5, N6.** A literal where 0.20 gave `const.N` reads as the literal. In
  safe_reader `step_of` and `module_constant`, a module const's literal
  value comes through the same path.
- **N7.** The trailing Return is skipped where the reader already skips
  naga's own trailing return.
- **N8.** reader 728-730, 864 and safety 1281 match the new Debug names.
  The naga column of kingdoms/{wgsl,safewgsl}/functions.tsv (Math::FindLsb
  and Math::FindMsb) becomes Math::FirstTrailingBit and
  Math::FirstLeadingBit through gen_tables.py. That is the naga column only:
  no row, no dual, no ref changes, and a loader test asserts the row and
  dual counts are unchanged. naga_names' tests are updated to the new counts
  (70 and 25 become naga 30's), each named in the commit.
- **MSL.** safe_reader 8150 and the refusals at 2927 and 2956 are changed to
  naga 30's truth (decision e): naga 30 writes compare-exchange to Metal.
  Whether the row is then drawable is a kingdom question, left for later
  and named in the commit. The test asserts what naga 30 writes.
- **subgroupBroadcast lane.** render writes a broadcast lane that is a
  literal as a plain const literal (`5u`), not `lit_u32(5u)`. The reader
  reads both. The random-tree floors prove the round trip.
- **target.rs.** Capabilities are mapped from wgpu 30's Features to naga 30's
  Capabilities, the flags as renamed. The test that host validation agrees
  with the device's (target's own) stays.
- **Oracle cache key (decision d).** harness 4258-4271's key gains the naga
  version, a constant `NAGA_VERSION = "30.0.1"` beside the Cargo pin, with a
  test that it equals the version in Cargo.lock.

### 2.3 fuller: scaffold, versions, roundtrip

- scaffold builds a `naga::Function` with `diagnostic_filter_leaf: None`;
  ControlBarrier and MemoryBarrier are both named (scaffold 536).
- versions 66-109 and 146-153 take the new variants.
- The scaffold and roundtrip tests that compare naga-written WGSL are updated
  to naga 30's text (builtin-clash `_` names and the rest), each named in the
  commit. Where naga 30's WGSL writer has a bug (override sizes), any kernel
  of ours that would hit it is refused before writing, by name. We declare
  no override, and a test asserts no rendered kernel holds one.

### 2.4 wgpu in hff, fuller, phylu

Section 1.4, mechanically, in the order hff, then fuller, then phylu (each
depends on the one before). Each Result the new API returns is handled as an
error with its reason: `poll`, `get_mapped_range`, `request_adapter`, the
error scope. Never `unwrap_or` and never ignored.

### 2.5 Shader runtime checks

- **A, the default for every shader:** wgpu 30's checked
  `create_shader_module`, with bounds, loop bounding and integer-division
  checks on. Any WGSL not written by our renderer (model replies, cuneus,
  fixtures) always compiles this way (Codex 5).
- **B, measured, never assumed, and only for kernels our renderer wrote:**
  `create_shader_module_trusted` with all six `ShaderRuntimeChecks` fields
  written out:
  - `bounds_checks: true`
  - `force_loop_bounding: false`
  - `int_div_checks: false`
  - the ray-query, task-dispatch and mesh fields false (they do not apply
    to compute)

  B rests on three invariants of the safe kingdom:
  - every loop is bounded (a literal, or `min(e, 16777216u)`)
  - every divisor is a literal of at least 1 or `max(b, 1u)` for u32
  - no i32 division can see `INT_MIN / -1`

  A CPU test proves all three on 1,000 random trees and every rendered
  corpus kernel, from the rendered text. A kernel that fails any of them is
  compiled A. B is behind one function per crate (`fn shader(device, desc,
  origin)`), with `origin` either Rendered or External.
- **The lead's measurement, both options, recorded:**
  - adapter, OS and backend
  - the golden hashes: phylu a1ad2da1's SR and regex goldens, which must be
    identical
  - the v24 fit's ms per generation, as median and p95 over 5 runs after
    one warm-up, against wgpu 0.20's recorded on the main binary
  - A is kept unless B is faster by more than the 0.20 baseline's p95
    spread; the numbers decide

### 2.6 phylu

phylu has no naga. Its port is wgpu only (section 1.4 across 10 files) plus
the fuller API it calls. phylu's code belongs to v24-b5b-phylu. I send it
this plan, the fuller commit and the error list, and it ports phylu on its
own naga-30 branch. I do not edit phylu.

### 2.7 As built (2026-10-19), where it departs from 2.1 to 2.6

- **The short circuit.** A reading copy, not a per-reader view
  (`naga_shape::fold_short_circuits`). `reader::read` folds the module it
  holds (`Kernel.module`), after validation; the safe reader reads that
  module. `safety::check_text` and `check::Checker::safety` fold their own
  parse, so the checker judges naga 0.20's eager Binary: both sides are
  judged, which is stricter than the device's short circuit and never
  weaker. The CPU executor's module, the validator's and the device's are
  never folded.
  - A fold is taken when its right side holds only Emits, nested folds, and
    Calls of pure functions: no store through a global or a pointer
    argument, no atomic, no barrier, no image store. A loop is allowed: the
    tree is a value, and the device runs the short circuit itself.
  - A folded local is left in the arena (`naga_shape::is_dead_local`); the
    safe reader zeroes no register for it.
  - Fixtures: nested `&&`/`||`, a call in the right side, a loop's test,
    `break if`; a right side that stores stays an if.
- **N2.** `reader::is_bake` takes "phony".
- **N3.** A loop-local `var` zeroed at each pass is read as naga 30 means
  it: a scalar is a register set to its zero, and an array is zeroed
  element by element (`zero_array`, the code that already zeroes a
  helper's local arrays at each call). No new row.
- **wgpu waits.** wgpu-core 0.21 gave up a wait after 5 s (`CLEANUP_WAIT_MS`).
  wgpu 30 waits as long as it is told.
  - `src/gpu_io.rs` holds `wait`, `drain`, `wait_for` and `mapped`, each a
    Result with its error named.
  - `gram::wait_gpu_within` waits with an explicit 4.9 s timeout
    (`PollType::Wait { timeout }`), so a runaway is still named as before.
- **target.rs.** The capabilities come from `wgpu_naga_bridge::
  features_to_naga_capabilities`, wgpu 30's own mapping, and the validator
  is `Validator::new` with them. `Target::recorded` gains
  SHADER_FLOAT16_IN_FLOAT32: wgpu-hal 30's Metal adapter starts from
  `DownlevelFlags::all()`, so `pack2x16float` validates on the device.
- **subgroupBroadcast.** The lane is written as a plain literal, a
  const-expression for naga 30.
- **The reading snapshot** renumbers arena (`@N`) and type
  (`Compose.typeN`) indices by first appearance, so names that only moved
  with naga's arenas compare equal.
- **A CPU suite with no adapter.** `--features wgsl` (no `gpu`): every
  device test is compiled out. `gram::WAIT_GAVE_UP_MS` is no longer gated
  on `gpu`; it was the one item that kept the plain wgsl test build from
  compiling, at HEAD too. The CPU-executor tests are skipped by name (the
  lead's rule).
- **Cuneus.** tests/corpus/cuneus holds the four compute kernels with no
  image or sampler (MIT, provenance in its README). naga 30 parses and
  validates 4 of 4; 0 read, each with the kingdom gap named (two
  `arrayLength`s outside the Gram role, a uniform field with no safe
  spelling, a module of 5 entry points).

## 3. Measured, in order (CPU, then the lead's GPU)

1. Baseline at HEAD (naga 0.20), recorded before any change, as a committed
   fixture: **the reading snapshot** (Codex 6). For every kernel of
   tests/corpus/model_replies (1,798), tests/corpus/safewgsl (112), the 481
   saved genomes' texts and v24's reference:
   - the read tree's sexpr, its role and its refusal reason, if any
   - the reader's root list: kind, path, point
   - for scaffold, its holes

   One JSON line each, committed as tests/fixtures/reading_snapshot_naga020
   .jsonl from a naga 0.20 build. After the upgrade a CPU test reads them
   all again and requires equality line for line. Every line that differs
   is either fixed or listed, by kernel and first difference, as a change
   naga 30 makes right (N3's zeroed loop vars, for one), each named in the
   commit with the lead told before it is accepted. Then:
   - floors as in Bars
   - saved genomes 481 of 481
   - the model-reply corpus: 1,756 of 1,798 at 7e94cba (2026-10-19
     09:56Z), made a test in this branch (a CPU test that reads
     tests/corpus/model_replies and asserts a converted-count floor of
     1,756, its refusals tallied by reason)
   - the safewgsl manifest, 101 of 112
   - the 235 kernels' naga-written text
2. After 2.1-2.3 on fuller's branch:
   - zero warnings, clippy clean
   - every CPU test (`cargo test --release --features gpu,wgsl --lib`, the
     CPU executor tests skipped as today)
   - floors met
   - saved genomes 481 of 481: if fewer, the names that moved (N9) are
     listed, and the dumps are re-encoded by the dump writer at the branch's
     commit, in a clean worktree, with dated copies kept (decision c)
   - corpus 1,756 or more
3. The cuneus test: copy the cuneus kernels with no texture or sampler from
   scratchpad/ext/in (instances, meshlab_swarm, shader with its three `{…}`
   workgroup-size placeholders filled, skin; fallback and vertex have no
   compute entry point), as fixtures. The test reads each and counts how many
   read; each one that does not is listed with its reason. The count is
   reported, with no bar.
4. GPU tests built, not run; handed to the lead with the commands and the
   record of 2.5 to fill in. Expected:
   - goldens bit-exact under A and B
   - fuller's device tests and phylu's suite pass
   - `examples/wgsl_adapter` re-measures `Target::recorded` on wgpu 30
   No merge until the lead has run them and the record is in the plan
   (Codex 7).

## 4. Commits

Each commit passes the bars it can measure.

1. hff `naga-30`: wgpu 30.
2. fuller `naga-30`:
   - commit 1: Cargo (naga and wgpu 30, msl-out declared, the Cargo.toml
     comment kept true), naga_shape and the one parse function
   - commit 2: the reader, safety and interpreter changes
   - commit 3: the wgpu changes
   - commit 4: the test updates to naga 30's truth, each named
   - commit 5: the cuneus fixtures and test
   - commit 6 (if needed): re-encoded saved genomes
3. phylu `naga-30`: v24-b5b-phylu's.

No merge to main without the lead's go after the GPU runs.

## 5. Risks

- **N1's pattern.** A kernel whose `&&` right side calls a helper that
  stores. The pass leaves it as an If, and the reader reads that if; it may
  not read it today.
- **N9's names.** If the saved genomes break, they are re-encoded; any phylu
  checkpoint saved with handle-indexed per-kernel rows needs the same, and
  the lead hears of it before any run.
- **naga 30's const folding** of model-written kernels. The corpus count
  measures it; a drop is a reader gap to fix, not a floor to lower.
- **The two naga 30 WGSL writer bugs.** Our renderer writes WGSL itself and
  naga writes only scaffold modules, which hold no overrides.
- **wgpu 30's Metal backend may schedule differently.** Timing moves; the
  goldens catch any value change.

## 6. Codex review (2026-10-19) and what changed

Codex (`codex exec`, read-only, over the three worktrees and the registry
sources) found 2 blockers, 5 majors and 1 minor. All are taken:

1. Blocker: folding the short-circuit into an eager Binary changes
   semantics: an unused value, `if`, continuing and `break_if`. The IR is
   no longer rewritten; 2.1's view is reader-side only.
2. Blocker: "no global store and no atomic" does not prove eager
   evaluation equal; a call can loop or write through a pointer. The
   equivalence claim is gone. The view applies only to callees the reader
   reads as pure helpers, and otherwise the if is read.
3. Major: nested `&&` in a right side, and recursion through every block.
   The view is post-order over Block, If, Switch, Loop body and continuing,
   with fixtures.
4. Major: new naga 30 variants (CooperativeMatrix, CooperativeLoad and
   CooperativeMultiplyAdd, AccelerationStructure and RayQuery with
   vertex_return) would fall into the reader's `naga.<variant>` leaves
   (reader 1128-1133). `supported` refuses them by name first.
5. Major: B is an unsafe promise and has six fields, and `INT_MIN / -1`
   matters too. B is now only for our rendered kernels, all six fields
   written out, with a test of its three invariants; external WGSL is
   always A.
6. Major: 481 of 481 counts cannot show unchanged roots, holes or rows.
   The reading snapshot (section 3) gives tree-for-tree equality against
   naga 0.20's reading.
7. Major: the GPU sign-off cannot be measured as written. Section 2.5's
   record (adapter, goldens' hashes, median and p95 of 5 runs, the 0.20
   baseline), and no merge before it.
8. Minor: `Arena::get_mut` exists, but a rewritten module would need
   validating. Moot: the IR is not rewritten.
