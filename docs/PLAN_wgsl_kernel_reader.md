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
`gen_tables.py`, `types.tsv` (68 duals), `functions.tsv` (197 templates).
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
  `Copy + Ord` keep every `BTreeMap` working.
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
- Tests: 68 duals load; every function instance's inputs and output are
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
literal leaf carries its dual on the device. Sequencing: fuller phase 1 of
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

## Verification

- `RUSTFLAGS="-D warnings" cargo test --features wgsl` and
  `cargo clippy --all-targets --features wgsl -- -D warnings`: zero warnings.
- Unit tests, each pinning one property: the 68 duals and 197 templates
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
