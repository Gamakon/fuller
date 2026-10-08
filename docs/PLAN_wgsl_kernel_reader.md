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
- Tests: 68 duals load; every function instance's inputs and output are
  duals; no instance outside `convert` has inputs and output of different
  forms except where `symbols.md` says (compare → flag, bits counts →
  count, quantise, hash → index); max arity 3; the table has no row whose
  semantic id is not a naga node name or a `composite`.

### 3. `karva.rs`: a generic encoder and decoder keyed on the pset

Factor the BFS out of `terms_to_karva_sized` and the walk out of
`karva_to_terms` so a second pair,
`terms_to_karva_generic(term, pset, seed, head_len)` and
`karva_to_terms_generic(head, tail, pset)`, take the constructor name as the
token name directly (`pset.functions[name].arity` is the arity), with no
`Math` semantic table. `Num` and `Var` leaves unchanged. The `Math` pair stays
as is; the generic pair is what the WGSL kingdom and any future
`class.instance` kingdom use. Tests: arity-3 round trip
(`(select.scalar_cond a b c)`), dotted names tokenise, tail rule `2h+1`.

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
    chains that resolve a buffer or uniform field, and builtins. A shared
    handle reached from several roots is expanded in each (the fold
    recovers it).
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
  - `rebuild(&self, genes: &[sexpr]) -> naga::Module`: clone the module;
    for each hole, append the gene's expressions to that function's arena
    (in post-order so operands precede users), insert a
    `Statement::Emit(range)` immediately before the holed statement, and
    re-point the `Store.value`/`If.condition` handle to the new root.
    Validate; `write_string` it.
  - `structural_eq(a: &Module, b: &Module) -> bool` using the canonical
    hash-cons from `wgsl_dag_study` (operands by canonical id, loads unique
    by pointer and position) — the round-trip proof that needs no device.
- `mod.rs`: `pub fn chromosome(kernel) -> WgslChromosome { head: Vec<sexpr>,
  tail: Vec<sexpr>, pset, stats }` which runs `homeotic::fold` over the
  roots' trees (tail slots = a parameter, default 8; definition limit =
  head length), then `terms_to_karva_generic` per gene at the chosen head
  length, and returns per-gene head requirement from the encoder's
  `oversized` flag at a sweep of head lengths (8, 12, 16, 24, 32).

### 5. `examples/wgsl_read_kernel.rs`

`cargo run --release --features wgsl --example wgsl_read_kernel -- <kernel.wgsl> [--tail 8] [--head 16]`
prints: functions and roots (stores, conditions); genes; per-root node
count; head length needed (max root size) and the histogram; duals used and
the `opaque` share by node; tail slots filled and the top shared partials;
then runs the round trip (encode → decode → rebuild → validate → write) and
prints `ROUND_TRIP ok` with `structural_eq`, or the first difference. Run
over phylu's `decode.wgsl`, `score.wgsl`, `hff.wgsl`, `cand.wgsl`,
`mix64.wgsl + vary.wgsl` and fuller's `src/lint/kernel.wgsl`; put the table
in `kingdoms/wgsl/README.md` as the measured section.

### What is explicitly out

phylu's decoder and device evaluator for the kingdom; running the rebuilt
kernel on data (device parity, phase 2); `block.store` for evolving control
flow (recorded as open in the README after the numbers say whether it is
needed); atomics, images, pointers as values.

## Verification

- `RUSTFLAGS="-D warnings" cargo test --features wgsl` and
  `cargo clippy --all-targets --features wgsl -- -D warnings`: zero warnings.
- Unit tests, each pinning one property: the 68 duals and 197 templates
  load and instantiate; arity-3 generic Karva round trip; a one-store
  kernel reads to one gene and rebuilds structurally equal; a kernel with
  an `if` yields a condition root; a kernel with the same subexpression in
  two stores folds it into one tail gene and still rebuilds equal; a node
  no rule types is `opaque` and round-trips untouched; the sidecar's
  declared form wins over inference; naga rejects a rebuilt module whose
  Emit ranges are wrong (negative test for the scaffold).
- End to end: the six kernels above read and round-trip `ROUND_TRIP ok`;
  the report table committed to the kingdom README with head length, gene
  count, opaque share and tail fills per kernel.
- Local commits only; conventional messages; no push.
