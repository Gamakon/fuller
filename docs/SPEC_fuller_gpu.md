# SPEC: fuller on the GPU — a K-expression rewriter driven by tables

Status tags: **BUILT** (running, tested) · **MEASURED** (number from a real
run) · **DESIGNED** (specified here, not implemented).

Scope: fuller only. nucleotable appears as the *shape of the inputs* (symbol
table, typed signatures), nothing more. Measurements that motivate this are in
`CR_fuller_wgpu_port.md`: at the 64 rows the join sends, fuller's data scoring
is not worth a kernel (MEASURED: 0.16 s per 847 expressions). egglog cannot be
moved. So a GPU fuller **is** the rewriter below; it is a new build, not a port.

---

## 1. What it is

A rewriter that takes a batch of K-expressions and returns, for each, a small
set of equivalent smaller K-expressions. It knows nothing about arithmetic.
It knows three tables and one layout.

## 2. Tables (DESIGNED)

```
symbols(semantic_id PK, kind, arity | signature, eval_op)
rules(rule_id PK, pattern, template, guard, ordering_class, family)
rule_symbols(rule_id FK, semantic_id FK)         -- every id on EITHER side
guard_seeds(semantic_id FK, fact)                -- Pow2 -> nonneg
guard_propagation(semantic_id FK, child_facts, fact)   -- Mul: nonneg,nonneg -> nonneg
```

- A rule is a theorem about specific symbols. `rule_symbols` lists every
  symbol the pattern **and the template** mention (`Div 1 x -> Inv x`
  depends on `Inv` too).
- **Usable rules are derived, not chosen.** For a kingdom `K` (a query over
  `symbols`): a rule is usable iff none of its `rule_symbols` rows points
  outside `K`. Relational division, written as an anti-join.
- **Identity is meaning.** Keys are `semantic_id`, never names (BUILT in
  `karva.rs`). `Div` and `ProtectedDiv` are different ids. **A changed
  definition is a new `semantic_id`** — moving ProtectedDiv's 1e-6 threshold
  under the same id would silently falsify every rule that references it.
- Guards are rows too: a seed is a fact about a symbol; propagation is a fact
  about a symbol given its children's facts.

What the kingdom join does **not** do: it does not make rule families
confluent. Non-confluence is two usable rules undoing each other
(distribute + trig), not a membership problem. Termination comes from §5.

## 3. Layout (BUILT for evaluation)

Level-order node arrays, as `gpu_eval::ExprBatch`: `nodes[]`, `offsets[]`,
`lengths[]`; children located by a prefix sum over arities. Every child sits
at a higher index than its parent, so one backward scan visits children before
parents.

Constraint: a rewritten form must decode into its gene's head length or it is
dropped and counted (`n_oversized`, BUILT in `denoise_karva_candidates_batch`).

## 4. Kernels

| # | Kernel | Work item | Status |
|---|---|---|---|
| K1 | guard facts | node (backward scan) → fact bitmask | DESIGNED |
| K2 | match | expression × position × rule → hit bit | DESIGNED |
| K3 | apply | hit → one new expression (splice + level-order re-layout) | DESIGNED |
| K4 | fold | closed subtree → literal | DESIGNED |
| K5 | evaluate | expression × row → value | **BUILT** (`gpu_eval`) |
| K6 | reduce | expression → 1−R² against its source's predictions | DESIGNED |
| K7 | signature | expression → 16-row behavioural signature | **BUILT** on host (`chrom_score`) |

- **K1** reuses K5's backward scan: a node's facts are a function of its symbol
  (seed rows) and its children's facts (propagation rows). Caller facts
  (`positive_vars`, `nonzero_vars`) are seed bits on variable nodes.
- **K2** patterns are bounded depth (≤ 3): symbol id at the position, symbol
  ids / wildcards / same-subtree constraints below, literal predicates
  (`= 1.0`, `|k| < 1e-6`), required fact bits from K1. Same-subtree equality
  (`Mul x x`) is a compare of two node ranges.
- **K3** emits one output per hit. Two hits on one expression produce two
  variants, never a merged rewrite. Re-layout is a BFS renumbering — the same
  arity prefix sum as §3.
- **K4 invariant: an input variable is never folded**, whatever its name
  (`c` is a column before it is the speed of light). Inputs are an explicit
  argument. Existing tests to carry over:
  `smallest_form_folds_constants_but_never_an_input`,
  `concretize_never_rewrites_an_input_variable`.
- **K5** semantics are the engine's (protected ops, poison flag, trig cut-off
  at 1e7) — the evaluator's contract, not the rewriter's.
- **K6** compares against the *input expression's own* predictions, never an
  extracted variant's (the existing `denoise` rule). f32 on device; the chosen
  form is re-checked in f64 on the host before it is returned.

## 5. Termination without an e-graph (DESIGNED)

"Shrink-only" is not enough — fuller's own `sign` ruleset (BUILT) has
size-neutral rules that move a `Neg` rootward so an absorber can delete it.
Each rule therefore carries an `ordering_class` under one well-founded measure:

```
measure(expr) = (node_count, sum of depths of Neg nodes, ...)   lexicographic
```

- class A: strictly reduces `node_count`.
- class B: keeps `node_count`, strictly reduces the secondary measure.
- anything else (expanding, bidirectional, commutativity) is **not admitted**.

Checked when the rule table is loaded, per rule, not at run time. Expanding
families — distribute, trig expansion — stay in egglog on the CPU.

## 6. Algorithm (DESIGNED)

```
frontier = input batch
repeat up to R rounds (R = 6, as SMALLEST_FORM_MAX_ROUNDS):
    K1 facts -> K2 match -> K3 apply -> K4 fold
    dedup by K7 signature (+ exact token equality)
    keep a beam of B per source expression, smallest measure first (B = 8, as ECLASS_K)
    stop when no expression produced a hit
return per source: the beam, each with node_count and (if rows given) K6 loss
```

Host picks: smallest within tolerance, else the input unchanged. Deterministic:
ties broken by token order, never by arrival order.

## 7. What is lost against egglog

- **Sharing.** An e-graph stores every equivalent form once; a beam stores B.
- **Grow-then-shrink.** A rewrite that must expand before it collapses
  (`(a+b)^2 - a^2 - 2ab`) is unreachable by construction.
- **Equality.** egglog proves two forms equal; this only ever produces forms.
  The parity scorer stays on egglog.

This is a real loss, not a footnote. The gate below measures it.

## 8. Multityped K-expressions (DESIGNED; depends on nucleotable)

- Arity is per *instance*: with many-hot signatures the prefix sum needs the
  arity this occurrence uses, resolved top-down from the root's required
  output type. One extra pass, parallel per expression.
- Patterns match `(semantic_id, signature)`. A template's output type must
  equal the replaced subtree's — checked at table load.
- The rewriter works on the expressed tree and never touches the tail. Writing
  a result back into a fixed-length typed gene is nucleotable's open
  typed-tail problem; this spec does not solve it. Class-A rules ease it (a
  smaller tree fits the head it came from); class-B rules do not change size.

## 9. Gate before any kernel is written

1. Build the rule table and the §6 loop **on the CPU**, from the rules already
   in `identities`, `powers`, `sign`, `rational` that pass §5.
2. Run it over the 847 hall-of-fame expressions with the existing harness
   (`examples/measure_smallest_form.rs`).
3. Report: **coverage** = share of egglog's reductions reproduced (weighted
   nodes; egglog today: 161,552 → ~151,000 MEASURED); **soundness** = K7
   signature agreement with the input on every output; **time** per expression.
4. Kernels K1–K4, K6 are built only if coverage justifies them. The CPU
   prototype stays as the reference the kernels are checked against.

## 10. Built today

| Piece | Where |
|---|---|
| Level-order layout, evaluator kernel, 2-D dispatch | `src/gpu_eval.rs` |
| Behavioural signature, train-only least squares | `src/chrom_score.rs` |
| `semantic_id` symbol table, many-hot arity model | `src/karva.rs`, `src/geneframe.rs` |
| Rules + guards as egglog text (the source for the rule table) | `src/ruleset/`, `src/expr.rs` |
| Hall-of-fame dataset + measurement harness | `hff/notebooks/_hof_dataset.py`, `examples/` |
