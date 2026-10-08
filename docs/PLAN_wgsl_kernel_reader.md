# PLAN: the WGSL kernel reader — a kernel into Karva, and back (fuller)

## Context

**Objective.** Read an existing WGSL compute kernel into the WGSL kingdom's
chromosome shape and back out, exactly, so the representation is learned and
checked on real programs before anything is evolved. Deliverable: an example
`wgsl_read_kernel` that, for a kernel file, reports the chromosome it makes
(genes, head needed per gene, duals inferred, `opaque` share, tail partials)
and proves the round trip through naga.

**Approach.** naga parses the kernel. Every `Store` value and every `if`/
`loop` condition is a root; the statement tree is a fixed scaffold with one
hole per root. Each root's pure expression tree is typed with duals
(`slot.form`, `kingdoms/wgsl/types.tsv`), slot from naga's type, form inferred
from use, `opaque` where inference fails. The tree is rendered as an
s-expression whose operators are the kingdom's `class.instance` names
(`functions.tsv`), encoded with a generic Karva encoder, folded through the
homeotic fold for shared partials, decoded again, rebuilt into the scaffold,
validated by naga, written as WGSL, and compared structurally with the
original.

**Unique feature.** The kingdom's genome is the pure expression DAG between
loads and stores; control flow is the scaffold. Two-column typing makes a
content mistake unrepresentable. The reader is the first consumer of the
generated tables, so it is also their test.

**Status.** Specified: `kingdoms/wgsl/{README,TYPES,symbols}.md`,
`gen_tables.py`, `types.tsv` (71 duals), `functions.tsv` (197 templates).
Built: `examples/wgsl_dag_study.rs` (naga parse + DAG measurement),
`homeotic::{fold,encode,unfold}`, `karva::terms_to_karva_sized`. Not built:
everything below. fuller only; phylu's decoder, the evaluator and evolution
are later increments.

**Why this order.** Measured kernels are wide and shallow (about 7–10 nodes
per root, 100–200 roots). Head length, gene count and whether control flow
needs a `block.store` dual are facts the reader produces, not assumptions to
design on.

## Facts from the code (verified)

- **Karva walks any arity.** `karva_to_terms` loops `0..spec.arity`
  (`karva.rs:198-205`); tail length is `head × (pset_max_arity − 1) + 1`
  over the pset's max (`karva.rs:563-570`), so arity 3 gives `2h + 1`.
  What blocks arity 3 is the hard-coded `Math` semantic tables:
  `semantic_to_math` (`karva.rs:92-128`) and `math_ctor_to_semantic`
  (`karva.rs:260-289`) know only the 1- and 2-ary `Math` ops. `Token::Var`
  accepts any quoted name present in `pset.variables` (`karva.rs:45,
  137-142, 504-506`).
- **`Ty` is a closed `Copy + Ord` enum** (`geneframe.rs:30-85`); no
  string-keyed or custom variant exists. `Arity` is two `BTreeMap<Ty,u32>`
  (`geneframe.rs:114-133`); `regex_table()` (`geneframe.rs:383-444`) is the
  pattern: a `&[(semantic_id, alias, &[Ty], Ty)]` row list looped into
  `Symbol`s with `symbol = i+1`.
- **naga:** `front::wgsl::parse_str` parses only (`front/wgsl/mod.rs:58`);
  `valid::Validator::new(flags, caps).validate(&module) -> ModuleInfo`
  (`valid/mod.rs:373, 491`); the WGSL writer is
  `back::wgsl::write_string(&module, &info, WriterFlags)` behind feature
  `wgsl-out` (`back/mod.rs:16`, `Cargo.toml:165`), which wgpu does not enable
  on this platform. Expressions live in `Function.expressions: Arena`,
  statements in `Function.body: Block`, with `Statement::Emit(range)`
  marking which expressions a block evaluates.

## Approach — the pieces (all in fuller)

### 1. Cargo: naga as a real optional dependency, feature `wgsl`

`Cargo.toml`: move naga from dev-dependencies to
`naga = { version = "0.20", features = ["wgsl-in", "wgsl-out"], optional = true }`,
feature `wgsl = ["dep:naga"]`. `src/wgsl/` compiles only under it; `gpu`
stays independent. `examples/wgsl_dag_study.rs` and the new example require
`--features wgsl`.

### 2. `geneframe.rs`: one data-driven variant and the WGSL loader

- `Ty::Wgsl(u16)`: the index of a dual in `types.tsv` (ref `T001` → 0).
  One variant, appended last (the ordering rule at `geneframe.rs:78-84`).
  `Copy + Ord` keep every `BTreeMap` working. **Lands in step 3, not step
  1:** a payload variant stops every `ty as u32` in phylu compiling, so
  phylu's phase 2 moves its casts to `Ty::code()` first (delivered in step
  1 as the declaration index, equal to `as u32` today; `TABLE_CODE_BASE =
  256` reserved for the duals).
- `pub const WGSL: &str = "WGSL"`.
- `pub fn wgsl_table() -> SymbolTable` built from
  `include_str!("../kingdoms/wgsl/types.tsv")` and
  `include_str!("../kingdoms/wgsl/functions.tsv")`: for each function row,
  expand its template over the duals it names (the instantiation rule in
  `symbols.md`): `<S.real>` → every real dual, lanes equal; explicit duals
  as written; `<S.c>`/`<vecL.c>` over the forms the note allows. Each
  instance becomes a `Symbol` with `semantic_id = "class.instance"` (plus a
  dual suffix when a template instantiates to several rows,
  e.g. `arith.add@f32.real`), `alias = instance`, `symbol = i+1`, and a
  many-hot `Arity` whose keys are `Ty::Wgsl(idx)` (the k-hot encoding the
  tables do not yet carry). Terminals (`literal.*`, `builtin.*`) are rows
  with `symbol < 0`; `load.*`, `arg.*`, `store.*` are per-kernel and are
  added by the reader, not the loader.
- `pub struct WgslDual { ref, slot, form, lanes, fallback }` and
  `wgsl_duals() -> Vec<WgslDual>` for the reader and the tests.
- Add `Ty::code(&self) -> u32` (base variants in declaration order,
  `Wgsl(i)` = BASE + i) and generate the fallback table as a `Vec` keyed by
  code from `types.tsv`'s `fallback` column. phylu casts `ty as u32` at
  engine.rs:363-374 and 551-569 and device.rs:89 and sizes its fallback
  table by `Ty::Pattern as u32 + 1`; a payload variant does not cast, so
  this is a **fuller + phylu commit pair**, phylu's four sites moved to
  `code()` in the same change set.
- The loader test prints the instance count after template expansion, so
  phylu's per-fit device tables (`out_ty`, `in_ty`, fallback per id, sized by
  `n_ids`) are sized on a number, not a guess.
- Tests: 71 duals load; every function instance's inputs and output are
  duals; no instance outside `convert` has inputs and output of different
  forms except where `symbols.md` says (compare → flag, bits counts →
  count, quantise, hash → index); **max total arity 4** (`K_MAX`); the
  table has no row whose semantic id is not a naga node name or a
  `composite`.

### 2a. Arity: K per kingdom, K_MAX = 4, via phylu's richer-arity plan (decision, revised)

phylu's review first pushed this kingdom to lower `select`, `fma`, `clamp`
and `mix` to pairs of 2-ary rows. That bends the kingdom to the engine's
stride, which is the wrong way round; Andrew: "what I'd like is arbitrary,
what I will be is pragmatic." phylu's `docs/PLAN_arbitrary_arity.md`
(c45703f3) makes the node `(op, first_child, konst)` with children
contiguous in level order and arity from the op table, `K` per kingdom
(SR/TSR/REGEX stay at 2, byte-identical), `K_MAX = 4`. So: **no lowering
of 3-ary templates**; only templates above 4 children (`compose_mat` of
more than four vectors, `pack4` and friends) stay composed. The tail rule
becomes `head·(K−1)+1` with `K = 4` for this kingdom. The retired `arg1`
word becomes `ty_code`, the node's out dual (`Ty::code()`), which is how a
literal leaf carries its dual on the device; and the `op` word becomes the
**kingdom's function id** (an index into the uploaded symbol table, with
per-kingdom `op_arity`, `op_semantic`, `out_ty`, `in_ty × K` and fallback
tables), so the node is `(fn_id, first_child, ty_code, konst)`. SR's table
maps id to `Op` one to one, so its bytes are unchanged. Fixed ids, the same
in every kingdom's table, mark the engine-level leaves: `FN_ID_VAR` 0,
`FN_ID_NUM` 1, `FN_ID_GENE_REF` 47 (fixed values, not a contiguous prefix:
GeneRef was given 47 when it was added and moving it would change the bytes
of every fold fixture; a kingdom table must not reuse these ids). Tail genes are evaluated one dispatch per dependency level, head
last, so depth is unbounded and found by selection. Sequencing: fuller phase 1 of
that plan (node semantics, `Op::arity`, `Ty::code()`, generated fallback)
is built first, in this crate, and this reader builds on it.

### 2b. Literals carry their dual (decision)

`(Num v)` cannot rebuild `Literal::U32` against `F32` or `Bool`. A literal
leaf is `(literal.<form> v)` with the slot from naga's `Literal` variant, a
`class.instance` node whose one child is the number, so `parse_math`,
`homeotic::fold` and the generic encoder handle it as an ordinary 1-ary
node and the rebuild knows the type. The generic encoder treats
`literal.*` as the kingdom's RNC family.

### 3. `karva.rs`: a generic encoder and decoder keyed on the pset

Factor the BFS out of `terms_to_karva_sized` and the walk out of
`karva_to_terms` so a second pair,
`terms_to_karva_generic(term, pset, seed, head_len)` and
`karva_to_terms_generic(head, tail, pset)`, take the constructor name as the
token name directly (`pset.functions[name].arity` is the arity), with no
`Math` semantic table. `Num` and `Var` leaves unchanged. The `Math` pair stays
as is; the generic pair is what the WGSL kingdom and any future
`class.instance` kingdom use. Tests: dotted names tokenise, a `literal.index` leaf round-trips with its
dual, `(select.scalar_cond c a b)` round-trips at arity 3, the tail rule is
`3h+1` at `K = 4`, and a 5-ary token is refused.

### 4. `src/wgsl/` — the reader (new module, feature `wgsl`)

- `reader.rs`
  - `read(source) -> Result<Kernel, String>`: parse, validate (naga
    `Validator`, full flags), then for each function and entry point walk
    `body` recursively: a `Store { pointer, value }` yields a root
    `Root::Store { hole, target }`; an `If { condition, .. }` and a
    `Loop { break_if, .. }` (and `Switch { selector }`) yield
    `Root::Cond { hole }`. Each root's `value` handle is expanded into a tree
    `Node` by following operands through the arena, stopping at leaves:
    `Literal`, `Load`, `FunctionArgument`, `GlobalVariable`/`AccessIndex`
    chains that resolve a buffer or uniform field, builtins, and
    `CallResult`. A shared handle reached from several roots is expanded
    in each (the fold recovers it, under the rules below).
  - **Calls.** A `Statement::Call` stays in the scaffold; each argument
    expression is a root (`Root::Arg { hole, call, position }`) and its
    `CallResult` is a leaf of the callee's return dual (`opaque` unless the
    sidecar declares it). Atomics, barriers, images remain out.
  - **Loads and sharing (rule).** A `Load` may be shared between two roots
    only if no `Store` to the same pointer, or to one that may alias it
    (same buffer, index not provably equal), lies between them in
    statement order; otherwise each occurrence is unique and the fold
    cannot merge it. A dynamic `Access` load is therefore unique unless its
    index expression is itself shared under the same rule. The report
    counts how many of the DAG study's repeats survive this rule; phylu's
    review expects the number to be low on real kernels, and that is a
    result either way.
  - **Form conflicts (rule).** A shared partial has one dual. If a partial's
    occurrences are used under different forms across roots (an index in
    one, a count or bits in another), it is **not shared**: the fold is
    refused for that partial and the occurrences keep their own duals. No
    `convert` node is ever inserted that the kernel did not have. The report
    counts conflicting-use partials per kernel.
  - **Tail depth.** phylu's two-pass evaluation today computes definitions
    with `shared = None`, so a tail gene may not reference another.
    `homeotic::fold` gains `max_depth` (default 1 until phylu evaluates the
    tail last slot to first in one ordered pass, which allows any depth);
    then the cap is removed and depth is read off the population (tail
    slots read per individual, longest reference chain), decided by
    selection, not set.
  - Every `Node` is rendered as `(class.instance child …)` using
    `functions.tsv`'s naga column in reverse (naga node → `class.instance`,
    e.g. `Binary::Add` on `f32` → `arith.add`); leaves as `(Var "load.buf@i")`
    style names registered as per-kernel terminals in the pset, literals as
    `(Num v)` with their dual.
- `infer.rs` — duals
  - slot: from `module.types[expr.ty]` via naga's `TypeInner` (Scalar,
    Vector, Matrix, Array, Bool).
  - form, by use, in one pass over each tree with the function templates as
    the oracle: a `Binary::Add` over float slots → `real`; an index operand
    of `Access` → `index`; bitwise operators → `bits`; comparisons → `flag`;
    `Math::Pack*`/`Unpack*` → `bits`/`real`; an integer arithmetic result
    used only as an index → `index`, else `int`; a call to a function named
    `mix64`/`fmix32`/`hash` → `hash32/64`; a buffer whose name the kernel's
    optional sidecar `kernel.duals.tsv` declares → that form; otherwise
    `opaque`. Record which rule fired per node for the report.
- `scaffold.rs`
  - `Scaffold { module: naga::Module, holes: Vec<Hole> }` where each `Hole`
    is `(function index, statement path, which operand)`.
  - `rebuild(&self, genes: &[sexpr]) -> naga::Module`: rebuild each
    function's arena **from scratch** rather than appending: walk the
    scaffold's statements in order, and for each hole emit the gene's
    expressions in post-order into the hole's own block, immediately before
    the holed statement, as one `Statement::Emit(range)`; leaves are
    re-emitted locally (a `Load` is a fresh expression in that block, never
    a handle borrowed from another block, which naga rejects). Non-emitted
    kinds (`FunctionArgument`, `GlobalVariable`, `LocalVariable`,
    `Constant`) are shared by handle. No original expression is left
    orphaned; the arena is compact. Validate; `write_string` it; check
    what `back::wgsl` does with any unemitted expression and assert there
    are none.
  - `structural_eq(a: &Module, b: &Module) -> bool` using the canonical
    hash-cons from `wgsl_dag_study`, extended to compare **per block in
    statement order** (loads unique by pointer and block position), so an
    expression hoisted out of a loop body is not equal. This is the
    **gate**, necessary not sufficient; the proof is the device run of
    phase 2 (compile both, same inputs, bit-exact outputs).
- `mod.rs`: `pub fn chromosome(kernel) -> WgslChromosome { head: Vec<sexpr>,
  tail: Vec<sexpr>, pset, stats }` which runs `homeotic::fold` over the
  roots' trees (tail slots = a parameter, default 8; definition limit =
  head length), then `terms_to_karva_generic` per gene at the chosen head
  length, and returns per-gene head requirement from the encoder's
  `oversized` flag at a sweep of head lengths (8, 12, 16, 24, 32).

### 5. `examples/wgsl_read_kernel.rs`

`cargo run --release --features wgsl --example wgsl_read_kernel -- <kernel.wgsl> [--tail 8] [--head 16]`
prints: functions and roots (stores, conditions, call arguments); genes;
per-root node count, the head-length histogram and **roots over 64 nodes**
(phylu's `MAX_NODES`, unrepresentable); duals and templates actually used (of
68 and 197), the `opaque` share by node **and by slot**, sidecar coverage;
repeats surviving the load-sharing rule; conflicting-use partials; tail
slots filled at depth 1 and the top shared partials;
then runs the round trip (encode → decode → rebuild → validate → write) and
prints `ROUND_TRIP ok` with `structural_eq`, or the first difference. Run
over phylu's `decode.wgsl`, `score.wgsl`, `hff.wgsl`, `cand.wgsl`,
`mix64.wgsl + vary.wgsl` and fuller's `src/lint/kernel.wgsl`; put the table
in `kingdoms/wgsl/README.md` as the measured section.

### What is explicitly out

phylu's decoder and device evaluator for the kingdom; running the rebuilt
kernel on data (device parity, phase 2, the proof); `block.store` for
evolving control flow (recorded as open in the README after the numbers say
whether it is needed); widening phylu's node to three children; multi-pass
tail evaluation; atomics, barriers, images, pointers as values.

## Review findings folded in (phylu-regex, 2026-10-08)

Twelve findings, all accepted: arity lowered to 2 (§2a); `Ty::code()` and
the generated fallback table as a fuller + phylu commit pair (§2); form
conflicts refuse sharing, never insert converts (§4); literals carry their
dual (§2b); the load-sharing rule and its survival count (§4); structural
equality as the gate, device run as the proof, per-block comparison (§4
scaffold); operands re-emitted locally and the cross-block negative test
(§4, verification); calls as argument roots with `CallResult` leaves (§4);
tail depth capped at 1 (§4); roots over 64 nodes reported (§5); instance
count after expansion printed by the loader (§2); duals and templates used,
opaque by slot and sidecar coverage in the report (§5). Three of these (1, 2,
9 in the review) are phylu-shape decisions and are written here as such.

## Coordination with phylu (project management, this session owns it)

Two crates, one node layout, four phylu phases (`phylu/docs/
PLAN_arbitrary_arity.md`) interleaved with two fuller phases. Rules: each
crate unit-tests its own work and commits clippy-clean; every commit builds
against the other crate's HEAD; **fuller owns end-to-end testing** at every
join and reports results to Andrew. Each handoff is one message to
`phylu-regex` stating what is now true, the commit id, and what phylu may
start; each return is one message back with the commit id and their unit
test count. Nothing on the GPU without announcing to Andrew first.

| step | who | work | gate before the next step | fuller's end-to-end check |
|---|---|---|---|---|
| 1 | fuller | node semantics `(fn_id, first_child, ty_code, konst)`, `Op::arity` exhaustive + `op_arity_table()`, `Op::Select3` (48, the 3-ary op), fixed engine leaf ids, `K_MAX`, `Ty::code()`/`from_code`/`ALL`, `fallback_leaf`/`fallback_table`, n-ary `Flat::to_tree`, lint `arity`/`op_of` total | **DELIVERED 2026-10-08**, fuller `30f1639` + lint fix; 403 plain / 444 gpu tests, clippy clean; phylu's full gpu suite (380 + 22, golden included) passes against it | done: fuller's suites; phylu HEAD checked `-D warnings --features gpu` and its suite run here |
| 2 | phylu | decoder host+device with `fn_id` tables, `in_ty × K`, `out_ty`, fallback by code; kid arrays to 512 | **DELIVERED 2026-10-08**, phylu `3828717d` + `c701325c`: every `ty as u32` → `Ty::code()`, fallbacks from `fuller::geneframe::fallback_table()`, op tables sized `Op::LAST + 1`, host and device decoders walk contiguous children of any arity, `in_ty[sid·k_in + j]`; 382 + 22 tests, golden held, wide+Select3 kingdom gate | done here: their full gpu suite against fuller `bea8e74` (golden checksum, host↔device parity for TSR and REGEX; no fit-level golden for TSR/REGEX exists, none claimed). **Layout fact from phylu:** level order holds for DECODER output; host editors (gpu_fold, snap grafts) keep explicit child pointers until phase 4 reorders them, so `GpuNode::child(i) = arg0 + i` is a decoder-output rule and `engine::node_child` is the host migration rule. `MAX_HT` stays 256 until phase 3 (head ≤ 63 at K = 4). Phase 3 released at M2; phase 4's WGSL parts wait for M3 |
| 3 | fuller | WGSL loader (`wgsl_table()`), generic Karva pair, reader, scaffold, `wgsl_read_kernel` | **DELIVERED 2026-10-08**, fuller `35e28f9` (+ the dump): `Ty::Wgsl`, loader (1,061 rows), generic Karva pair, reader, chromosome (fold + Karva round trip, 59/59), scaffold rebuild + structural gate: `ROUND_TRIP ok` on the six kernels; 438 tests under `wgsl`; sample chromosomes `kingdoms/wgsl/samples/hff.chromosomes.json` | done: the six-kernel tables in the kingdom README (read, chromosome, round trip) |
| 4 | phylu | variation grafts `K_MAX`, fold/hash/export/order/tower loops, tail evaluated per dependency level, 3-ary test kingdom, retire `arg1` | their tests + the 3-ary kingdom end to end, commit id back | a WGSL chromosome from step 3 uploaded, decoded on the device, hash and export agree with the host; depth cap lifted in `homeotic::fold` |
| 5 | fuller | device parity of a rebuilt kernel (phase 2 proof): compile original and rebuilt, same inputs, bit-exact outputs | **DONE 2026-10-08 23:08**: all 10 entry points compile both ways on Metal; phylu's suite with each rebuilt kernel substituted (`PHYLU_WGSL_DIR`, their sweep script, one binary at a time, control first) holds the golden checksum and the hff-sample gate for decode, hff, cand, score, mix64+vary: five of five. The proof found a representation fault (inlined `let`s re-read after a store; the rebuilt decode refused 19 of 27 sample genes) fixed by making a source `let` a root (fuller 195dfe7+) | the proof the plan's gate pointed at, measured |

Messages to phylu, in order (sent only when the gate is met):

- **M1, after step 1 (sent 2026-10-08):** fuller HEAD id; what exists by name; the two deviations needing phylu's ack (fixed leaf ids 0/1/47, not a prefix; `Ty::Wgsl` in step 3, so the cast sites engine.rs:363-374, 551-569, device.rs:89 and the test at device.rs:1118 move to `code()` in phase 2); adopt `fallback_table()`; `Op::Select3` is the 3-ary op, add no other. Return: commit id, unit test count, parity and golden results.
- **M2, after step 2 returns:** "Golden runs on SR/TSR/REGEX reproduced here byte-identical (or: not, with the first diverging generation). Start nothing; I build the loader and reader against your HEAD."
- **M3, after step 3:** "fuller `<id>`: reader round-trips six kernels; a WGSL chromosome and its symbol table are at `<path>` (instance count N, head H, K=4). Start phases 3–4; the uploaded-table shape you need is in `kingdoms/wgsl/*.tsv` and `geneframe::wgsl_table()`."
- **M4, after step 4 returns:** results of the device decode/hash/export check on the WGSL chromosome; depth cap lifted; request the per-level tail evaluation is on by default for the kingdom.
- **M5, after step 5:** parity results to Andrew and phylu; what the kingdom still lacks before evolving (the evaluator that compiles and times).

Escalation: a gate not met within a step is reported to Andrew as a
measurement, not worked around; a phylu finding that changes the node
layout stops both crates until the layout is re-agreed in both plans.

## Verification

- `RUSTFLAGS="-D warnings" cargo test --features wgsl` and
  `cargo clippy --all-targets --features wgsl -- -D warnings`: zero warnings.
- Unit tests, each pinning one property: the 71 duals and 197 templates
  load and instantiate; arity-3 generic Karva round trip; a one-store
  kernel reads to one gene and rebuilds structurally equal; a kernel with
  an `if` yields a condition root; a kernel with the same subexpression in
  two stores folds it into one tail gene and still rebuilds equal; a node
  no rule types is `opaque` and round-trips untouched; the sidecar's
  declared form wins over inference; a partial used as `index` in one root
  and `count` in another is refused for sharing; a tail gene never
  references a tail gene (depth 1); naga rejects a rebuilt module whose Emit
  ranges are wrong, and one whose gene borrows an operand emitted in a
  sibling block (two negative tests for the scaffold); the rebuilt arena has
  no unemitted expression.
- End to end: the six kernels above read and round-trip `ROUND_TRIP ok`;
  the report table committed to the kingdom README with head length, gene
  count, opaque share and tail fills per kernel.
- Local commits only; conventional messages; no push.
