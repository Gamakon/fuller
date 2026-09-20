# SPEC: fuller on the GPU — a graph linter for K-expressions

Version 2 · 2026-09-20 · supersedes v1 (commit `85059b8`)

Status tags: **BUILT** (running, tested) · **MEASURED** (number from a real
run) · **DESIGNED** (specified here, not implemented).

Scope: fuller only. nucleotable appears as the *shape of the inputs* (symbol
table, typed signatures), nothing more. The measurements that motivate this are
in `CR_fuller_wgpu_port.md`: at the 64 rows the join sends, fuller's data
scoring is not worth a kernel (MEASURED: 0.16 s per 847 expressions), and
egglog cannot be moved. A GPU fuller **is** the rewriter below — a new build,
not a port.

Changes from v1: §1 says what the thing *is* (v1 only said how it works); the
rule classes in §5 now separate meaning-preserving fixes from data-justified
ones; §7 is corrected — the engine never used equality proofs, so that loss
costs it nothing; §9 adds the lint survey as the way rules are found.

---

## 1. What it is

**A graph rewriting tool: a linter, with autofix, for K-expressions.**

| Linter | fuller |
|---|---|
| a rule: "this shape, fix it like this" | `Sub a (Neg b) -> Add a b` |
| autofix must not change behaviour | every rule carries a soundness argument against the symbol's definition; `ProtectedDiv(x,x) -> 1` was refused because it breaks at x = 0 |
| rules are written against a language definition | rules reference symbols by `semantic_id` (§2) |
| rule sets are chosen per project | usable rules are derived from the kingdom (§2) |
| flow / type facts ("never null here") | guard facts (`is-nonneg`, `is-nonzero`) (§4, K1) |
| survey a codebase, write rules for what fires | the hall-of-fame census (§9) |

It works on **meaning**, where GEP's own operators work on **tokens**. It has
two modes and one extension:

1. **Fix mode** — one input, one output: the smallest equivalent writing.
   `smallest_form` (BUILT, egglog). Used for reporting, for exactness against
   the truth, and with the behavioural signature for "the same model".
2. **Generate mode** — one input, *k* outputs: different writings of the same
   function. The e-class expansion (BUILT, egglog). This is not mutation in the
   GEP sense — the phenotype does not change. It is a *constructed* neutral
   variation: the graft changes nothing about an individual's fitness and
   everything about what crossover, transposition and point mutation can do to
   it next. GEP gets this by luck through its neutral regions; here it is done
   by algebra.
3. **Data-justified fixes** — prune, additive strip, snap. These *do* change
   the function, by an amount the rows cannot see (`1.00000000451 -> 1`, a term
   whose removal moves nothing). They are not equivalences; without rows they
   are refused (BUILT: `prune_on_data` returns None on no data). A profile-
   guided optimiser, not a linter.

A linter returns one fix because its consumer is a person. This returns a
neighbourhood because its consumer is an evolutionary search.

## 2. Tables (DESIGNED)

```
symbols(semantic_id PK, kind, arity | signature, eval_op)
rules(rule_id PK, pattern, template, guard, rule_class, family)
rule_symbols(rule_id FK, semantic_id FK)         -- every id on EITHER side
guard_seeds(semantic_id FK, fact)                -- Pow2 -> nonneg
guard_propagation(semantic_id FK, child_facts, fact)   -- Mul: nonneg,nonneg -> nonneg
```

- **A rule is a theorem about specific symbols.** `x / 1 -> x` holds for
  `ProtectedDiv` only because that symbol is defined as "0 when |b| < 1e-6,
  else a/b". So a rule has a dependency on the symbol table, and
  `rule_symbols` records it: every symbol the pattern **and the template**
  mention (`Div 1 x -> Inv x` depends on `Inv` too).
- **That dependency is a join.** For a kingdom `K` (a query over `symbols`), a
  rule is usable iff none of its `rule_symbols` rows points outside `K` —
  relational division, written as an anti-join. The ruleset is *derived* from
  the kingdom the way the kingdom is derived from the master table.
- **Identity is meaning.** Keys are `semantic_id`, never names (BUILT in
  `karva.rs`). `Div` and `ProtectedDiv` are different ids, so a rule for one
  cannot fire on the other. **A changed definition is a new `semantic_id`** —
  moving ProtectedDiv's threshold under the same id would silently falsify
  every rule that references it.
- Guards are rows too: a seed is a fact about a symbol; propagation is a fact
  about a symbol given its children's facts.

What the kingdom join does **not** do: it does not make rules confluent.
Non-confluence is two usable rules undoing each other (distribute + trig), not
a membership problem. Termination comes from §5.

## 3. Layout (BUILT for evaluation)

Level-order node arrays, as `gpu_eval::ExprBatch`: `nodes[]`, `offsets[]`,
`lengths[]`; children located by a prefix sum over arities. Every child sits at
a higher index than its parent, so one backward scan visits children first.

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

A lint pass is exactly K1–K3: rule × node, bounded rounds, no global reasoning.

- **K1** reuses K5's backward scan: a node's facts are a function of its symbol
  (seed rows) and its children's facts (propagation rows). Caller facts
  (`positive_vars`, `nonzero_vars`) are seed bits on variable nodes.
- **K2** patterns are bounded depth (≤ 3): symbol id at the position; symbol
  ids, wildcards and same-subtree constraints below; literal predicates
  (`= 1.0`, `|k| < 1e-6`); required fact bits from K1. Same-subtree equality
  (`Mul x x`) is a compare of two node ranges.
- **K3** emits one output per hit. Two hits on one expression produce two
  variants, never a merged rewrite. Re-layout is a BFS renumbering — the same
  arity prefix sum as §3.
- **K4 invariant: an input variable is never folded**, whatever its name (`c`
  is a column before it is the speed of light). Inputs are an explicit
  argument. Tests to carry over:
  `smallest_form_folds_constants_but_never_an_input`,
  `concretize_never_rewrites_an_input_variable`.
- **K5** semantics are the engine's (protected ops, poison flag, trig cut-off
  at 1e7) — the evaluator's contract, not the rewriter's.
- **K6** compares against the *input expression's own* predictions, never an
  extracted variant's. f32 on device; a chosen form is re-checked in f64 on the
  host before it is returned. K6 is what licenses class-D rules (§5) and
  nothing else needs it.

## 5. Rule classes and termination (DESIGNED)

Every rule declares a `rule_class`, checked when the table is loaded.

Meaning-preserving (exact under the symbol definitions, NaN and ±inf included):

```
measure(expr) = (node_count, sum of depths of Neg nodes, ...)   lexicographic
```

- **A — shrinker**: strictly reduces `node_count`.
- **B — enabler**: keeps `node_count`, strictly reduces the secondary measure
  (the `sign` float-outs that move a `Neg` rootward so a class-A rule can
  delete it; BUILT in egglog form).
- anything expanding, bidirectional or commutative is **not admitted**. Those
  families (distribute, trig expansion) stay in egglog on the CPU.

"Shrink-only" would be wrong: fuller's own `sign` ruleset needs class B. The
measure is what gives termination without an e-graph.

Meaning-changing:

- **D — data-justified**: not an equivalence. Admitted only when rows are
  supplied, and only if K6's loss is within the caller's tolerance. Never fires
  in fix mode without data.

Fix mode uses A + B (+ D with rows). Generate mode uses A + B and keeps the
intermediate forms rather than only the smallest.

## 6. Algorithm (DESIGNED)

```
frontier = input batch
repeat up to R rounds (R = 6, as SMALLEST_FORM_MAX_ROUNDS):
    K1 facts -> K2 match -> K3 apply -> K4 fold
    dedup by exact tokens, then by K7 signature
    keep a beam of B per source expression (B = 8, as ECLASS_K)
        fix mode:      smallest measure first
        generate mode: smallest first, but structurally distinct forms preferred
    stop when no expression produced a hit
return per source: the beam, each with node_count and (if rows given) K6 loss
```

Host picks. Deterministic: ties broken by token order, never by arrival order.

## 6a. Where it sits in the engine (DESIGNED)

On the device the linter is not a side step; it is a stage of evaluation:

```
population -> K1 facts -> K2 match -> K3 apply -> K4 fold -> K5 evaluate -> scores
```

- **Every individual, every generation.** Today the expansion runs on the
  subset an ORF cache lets through, returns through Python, and is re-encoded
  into geppy tokens — where forms the primitive set cannot express are thrown
  away (MEASURED: 2,619 in one 450-generation run).
- **Meaning-preserving variants are not scored separately.** Class A and B
  rewrites have identical behaviour, so identical error terms. Evaluate once
  per individual, on its tidy form — the smallest, so also the cheapest
  (MEASURED in the same run: 46,849 nodes removed by grafts; on the device that
  is kernel work never done). Only class D variants (prune, snap) change the
  function and need their own fitness.
- **Tidy the phenotype always; the genotype by policy.** Evaluating the tidy
  form is free and safe. Writing it back over the gene is a separate decision:
  the untidy regions are the neutral material crossover and transposition draw
  on, and grafting on ties was MEASURED to collapse diversity about 5x (hence
  `GRAFT_MODE = improve`). The stage therefore emits two things per individual:
  the tidy form (evaluation, reporting, "same model") and, under the graft
  policy, a written-back gene. Baldwinian by default, Lamarckian by switch.

## 7. What is lost against egglog

- **Sharing.** An e-graph stores every equivalent form once; a beam stores B.
  This costs generate mode variety.
- **Grow-then-shrink.** A rewrite that must expand before it collapses
  (`(a+b)^2 - a^2 - 2ab`) is unreachable by construction. This costs fix mode
  reductions. The gate in §9 measures how many.
- **Equality proofs.** egglog can decide whether two *given* forms are equal;
  this only ever produces forms. **The engine never uses that.** When it needs
  "are these the same" it asks the behavioural signature (K7). Only the parity
  scorer proves equality, offline, and it stays on egglog. So this loss costs
  the engine nothing.

## 8. Multityped K-expressions (DESIGNED; depends on nucleotable)

- Arity is per *instance*: with many-hot signatures the prefix sum needs the
  arity this occurrence uses, resolved top-down from the root's required output
  type. One extra pass, parallel per expression.
- Patterns match `(semantic_id, signature)`. A template's output type must
  equal the replaced subtree's — checked at table load.
- The rewriter works on the expressed tree and never touches the tail. Writing
  a result back into a fixed-length typed gene is nucleotable's open typed-tail
  problem; this spec does not solve it. Class-A rules ease it (a smaller tree
  fits the head it came from); class-B rules do not change size.
- Nothing in §2–§6 is specific to floats. A different symbol table and rule
  table make it a linter for boolean, string or mixed-type K-expressions. What
  does not transfer is each domain's evaluator contract (K5) and guard facts.

## 9. How rules are found, and the gate before any kernel

**Lint survey (BUILT, run once).** Collect every hall-of-fame dump into one
dataset (204 dumps, 847 unique expressions), count subtree shapes weighted by
occurrence, and write rules for the shapes that survive the current linter.
MEASURED: four agents over that dataset produced the `sign`, `is-nonneg` and
reciprocal rules; weighted nodes after `smallest_form` went 154,554 → 150,926 (input
161,408; 846 expressions), 155 smaller, none larger. The survey also showed what *not* to
write: no Pythagorean or double-angle shape occurs in 847 expressions, and
every rejected rule has its breaking input recorded in `docs/rule_proposals/`.
Repeat the survey whenever a sweep adds material.

**Gate.**

1. Build the rule table and the §6 loop **on the CPU**, from the rules already
   in `identities`, `powers`, `sign`, `rational` that pass §5.
2. Run it over the 847 expressions with `examples/measure_smallest_form.rs`.
3. Report **coverage** (share of egglog's weighted-node reduction reproduced),
   **soundness** (K7 signature agreement with the input on every output),
   **variety** in generate mode (distinct forms per input against egglog's
   `extract_variants`), and **time** per expression.
4. Kernels K1–K4 and K6 are built only if coverage justifies them. The CPU
   prototype stays as the reference the kernels are checked against.

## 10. Built today

| Piece | Where |
|---|---|
| Level-order layout, evaluator kernel, 2-D dispatch | `src/gpu_eval.rs` |
| Behavioural signature, train-only least squares | `src/chrom_score.rs` |
| `semantic_id` symbol table, many-hot arity model | `src/karva.rs`, `src/geneframe.rs` |
| Rules + guards as egglog text (the source for the rule table) | `src/ruleset/`, `src/expr.rs` |
| Fix mode and generate mode on egglog | `src/extract.rs` (`smallest_form`, `denoise_candidates_assuming`) |
| Data-justified fixes | `src/extract.rs` (`prune_on_data`), `src/snap_karva.rs` |
| Lint survey: dataset, proposals, measurement harness | `hff/notebooks/_hof_dataset.py`, `docs/rule_proposals/`, `examples/` |
