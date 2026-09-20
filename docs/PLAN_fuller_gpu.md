# Plan: fuller on the GPU — the K-expression linter

Spec: `fuller/docs/SPEC_fuller_gpu.md` (v2 + §2a, §2b, §6a). This plan builds it.

## Context

### The objective

**Beat the SRBench benchmarks with our symbolic regression approach**: recover
the exact equation on every ground-truth problem (Feynman, Strogatz), match or
beat the best black-box scores (Operon: test R² 0.934 in ≈ 615 s), and do it
faster than anyone else.

### The approach

- **HFF-SR**: gene expression programming (K-expressions, multi-gene
  chromosomes, islands) selected by HFF, the hyperspherical fitness function,
  which finds solutions with almost no overfit and no parsimony pressure.
- **The join**: islands × population × genes × variants × linkers × wrappers
  scored in one GPU dispatch (BUILT). Train rows fit the regression; validation
  rows are only scored.
- **fuller replaces sympy.** Every SR system leans on a computer-algebra step
  to simplify what evolution produces. Ours was sympy: slow, unbounded (one
  `simplify` call hung a run for 2 h 20 m), complex-domain, and ignorant of the
  engine's protected operators. fuller is our egglog-based tool for discovering
  simplified forms of expressions: deterministic, real-domain, bounded, and
  exact about what `ProtectedDiv` and friends actually compute.

### The unique feature

**fuller's simplified forms go back into the evolution.** A simplified form is
converted back into a K-expression and so into a gene. Simplification is not a
report-time clean-up, as it is everywhere else in the field; it is a variation
operator that works on *meaning* where GEP's own operators work on *tokens*.
An individual can be replaced by an equivalent, smaller, differently-shaped
gene with identical fitness — which changes what crossover, transposition and
mutation can reach next — and the whole population is kept tidy as it evolves.
No other SRBench entrant feeds its algebra system back into the search.

### Where we are (MEASURED)

- Registry: 72 / 126 exact. The 54 misses are search misses (49 below R² 0.99),
  not simplification misses. Honest caveat: of 66 Feynman recoveries audited,
  12 were evolved, 31 came from the name-blind `power_law` rule and 23 from
  name-gated templates.
- Power plant (black-box): test MSE 18.96, R² 0.9338 at population 1000.
- fuller today: egglog on the CPU, ≈ 1 ms per expression, ≈ 4% of a join run;
  GPU scoring ≈ 1%; Python ≈ 94%. It reaches only the individuals an ORF cache
  lets through, goes out through Python, and loses forms the primitive set
  cannot express on the way back (2,619 in one 450-generation run).
- A rule-mining survey of 847 real hall-of-fame expressions added 60+ rules:
  weighted nodes after simplification 154,554 → 150,926, none larger.

### Why this plan

To make the unique feature count, it has to be applied to **every individual,
every generation**, at no cost — and the engine is heading onto the GPU, where
a CPU egglog step would become the stall. egglog cannot move to the GPU. A GPU
fuller is therefore a new, table-driven rewriter — a graph linter — working
directly on level-order K-expressions: rule × expression × position, bounded
rounds, no e-graph. The target is the spec's §6a: the linter as a **stage of
evaluation** on the device — each individual evaluated in its tidy form, with
write-back to the gene by policy. Because it rewrites K-expressions in place,
the round trip through Python and the re-encoding loss both disappear.

### What success looks like

1. The linter reproduces most of egglog's reductions on real expressions
   (the Phase 2 gate), soundly.
2. It runs inside the GPU evaluation pass at negligible cost.
3. An A/B — linter on vs off, evolved-only problems, several seeds — shows it
   helps the evolution: more exact recoveries, or fewer generations to exact.
   That last number is the one that matters for SRBench, and it has not yet
   been demonstrated for the egglog version either.

**Scope.** Symbolic Regression only is built and tested. The tables use
nucleotable's full `symbols` schema plus `semantic_id` and are keyed on
`semantic_id`, so a future kingdom (REGEX) is rows plus an evaluator rather
than a redesign. **No REGEX code, rules, fixtures, evaluator or tests here.**

**Going in (MEASURED).** egglog fuller ≈ 1 ms per expression and ≈ 4% of a join
run; Python ≈ 94%. So the plan is gated: a cheap CPU reference first, kernels
only if the gate number justifies them.

## Three findings from the design review that change the spec

1. **The spec's class-B measure is wrong.** "Sum of depths of Neg nodes" is
   *increased* by 7 of the 12 `sign` float-outs (`Sub (Neg a) b -> Neg (Add a b)`
   lifts the Neg but pushes all of `b` down a level). Replacement:
   **M_neg = Σ over Neg nodes of the number of non-Neg proper ancestors** — all
   12 decrease it by exactly 1; one forward scan computes it.
   Full SR measure, lexicographic: `(node_count, #Pow, #Div, #ProtectedInv, M_neg)`.
2. **Apply is a copy with index remapping — inside the evaluator only.** The
   eval kernel reads children through stored indices; its only invariant is
   child index > parent index, not strict BFS. (That layout is private to K5;
   see the codex section for what must be canonical BFS.) Rules are RHS-linear, so a rewrite is a three-block copy
   plus a dead-node compaction. No count pass, no prefix sum: classes A and B
   never grow, so every output has a fixed 64-node slot.
3. **The device never decides a literal match in f32 for anything that leaves
   the device.** A constant 1.0000001 is 1.0 in f32: `x*1 -> x` would fire on
   the device and not on the host. The host classifies literals in f64 and
   hands the kernel integer class ids. Any form used for reporting or
   write-back is re-derived by the CPU engine in f64.

## Changes after the codex review (log: `fuller/logs/codex_plan_review.log`)

Codex confirmed the termination measure (its finding 7) and the kernel's
child > parent claim, and found five things the plan had wrong or missing. All
five are adopted:

1. **Two layouts, stated explicitly.** Parent-before-child order is an
   *evaluator-private* layout: valid for K5 and nothing else. Karva decodes
   children from the next stream slots, so `[Add, Neg, Mul, c, b, a]` evaluates
   correctly but decodes as a different expression. Anything that leaves the
   evaluator — token key, dedup, reporting, write-back — is first re-laid-out
   in **canonical BFS** (`node.rs`), and every splice test asserts a Karva
   decode/encode semantic round trip, not merely child > parent.
2. **Some existing rules are not NaN/inf-exact.** `Mul x 0 -> 0`: `inf * 0` is
   NaN, the rewrite gives 0. These rules are in production in egglog today, but
   the spec's classes A/B claim exactness including NaN and ±inf, so they
   cannot sit there. New class **F (finite-exact)**: exact wherever every
   subterm is finite. The classifier assigns F by a per-rule differential test
   on non-finite probes, not by hand. F rules are admitted by default (parity
   with today's egglog behaviour) behind a named switch, and the gate reports
   A/B-only and A/B/F coverage separately so the cost of strictness is a
   number. Spec §5 is corrected to say so.
3. **Literals computed on the device.** Host-side f64 classification covers
   input literals only; a literal produced by `(Num (neg c))` or `(* a b)` in
   round 1 would need its class in f32 for round 2. v1 device subset therefore
   **excludes every rule with a computed literal** and every rule with an exact
   or threshold literal pattern that such a rule could feed. Constant folding
   already happens on the host at ingestion. The gate reports what that
   exclusion costs in coverage.
4. **The gate was measuring the wrong corpus alone.** 847 hall-of-fame
   expressions are end products; the linter runs on live populations. Add a
   **frozen live-population corpus** (genes captured from a real join run,
   every generation) and gate on it too — see Phase 2.
5. **Write-back needs a status contract.** Every result carries exactly one of
   `grafted`, `phenotype_only`, `head_oversize`, `node_oversize` (> 64 on the
   device), `f64_replay_mismatch` (CPU f64 re-derivation disagrees with the
   device's f32 form), `encode_error`. All counted and surfaced in the join
   summary. Nothing is dropped or substituted silently.

Also adopted: the reader gets a **complete accepted grammar** (relation
declarations, `rewrite`, `rule` with equality bindings, relation premises,
multiple conclusions, multiple numeric predicates, `union`, `:when`, nested
numeric terms) with one source-text fixture per form asserting either the exact
normalised rows or a deliberate refusal; and the termination claim is checked
**mechanically** — every admitted rule instantiated with nested Negs in every
metavariable position, lexicographic decrease asserted.

Step 0 of the work is to correct the spec (§5 measure, §4 K3, §6 dedup — a
within-source signature is a soundness assertion, not a dedup key, since A/B
variants have identical behaviour by construction).

## Phase 1 — CPU reference linter (new `src/lint/`)

The reference the kernels are checked against, and a faster CPU linter in its
own right. No GPU code.

| File | Contents |
|---|---|
| `sexp.rs` | tokenizer + `Sexp{Atom,List}`, strips `;` comments |
| `reader.rs` | derives rule rows from the **existing egglog text** (`ALGEBRA_`, `POWERS_`, `SIGN_`, `RATIONAL_RULESET`, `GUARD_RELATIONS`) — one source of truth. `rewrite` ± `:when`; `(rule … (union e T))` with `<`/`>`/`>=`; guard seeds / propagation / implication. An unrecognised form is a **load error**; a refusal carries a **reason**. |
| `tables.rs` | `Rule`, `GuardSeed`, `GuardProp`, `GuardImpl`, `Measure` rows, `rule_symbols`; kingdom anti-join against `geneframe::SymbolTable::kingdom()`; type-signature check (trivially `F→F` for SR, still run) |
| `classify.rs` | class from skeletons: `mult_rhs(v) ≤ mult_lhs(v)`; node delta → A / same-size / expanding; same-size admitted as B only if the lexicographic measure strictly drops |
| `node.rs` | `LNode{op,arg0,arg1,lit}` (`Lit{ty,bits:u64}` — no unconstructed string variant), `PNode` conversion, canonical BFS + token key |
| `engine.rs` | facts (backward scan) → exact hash-consing for same-subtree (`cid`) → match (rules indexed by root op) → three-block splice + liveness compaction → fold (mirrors `fold_constant_subtrees_excluding`: never an input, finite only) → dedup by token key → beam → ≤ 6 rounds. Fix and generate modes. `LitMode::{F64,F32}` so device parity can be exact. `Vec`-sized: **no 64-node cap on the CPU**. |
| `mod.rs` | `lint(...)` per spec §2a; `inputs` required |

CPU rule representation is arity-general: `Pat::{Op(SemId,Vec<Pat>), Mv(u8), Lit(Exact(f64) | Bound(mv, preds))}`, `Tmpl::{Op, Mv, LitConst, LitProg(postfix: Push/Neg/Mul/Add)}`.

**Expected classification** (asserted by a census snapshot test, so a new egglog rule can never silently vanish):
- Class A: all of `identities`; `sign` group A; the shrinking rules of `powers`/`rational`.
- Class B: the 12 `sign` float-outs (via M_neg); `Pow x -2/-3/4/6`, `Log(Pow x n)` (via #Pow); `Div -1 x`, `Div (Inv a) b` (via #Div); `ProtectedInv x -> Inv x` (via #ProtectedInv).
- Refused, with reason: `Pow x -4`, `Pow x 8`, `Log(Mul a b)`, both binomial squares (expanding); `leaf` / `sq-safe` (derived structural relations, not facts).

Existing-file edits (visibility only, **nothing renamed**): `pub mod lint` in `src/lib.rs`; `PNode`, `parse_pnode`, `to_math`, `cost_of`, `fold_constant_subtrees_excluding` → `pub(crate)` in `src/extract.rs`; a `pub(crate)` accessor for the constructor↔`semantic_id` map in `src/karva.rs`.

## Phase 2 — the gate (`examples/measure_lint.rs`)

Over `logs/hof_exprs.tsv` (847 real expressions), under `nohup`, log in `logs/`. egglog's column is already in `logs/smallest_form_all.tsv`.

| Reported | Against |
|---|---|
| coverage, weighted nodes — **beam 8 and greedy beam 1** | egglog 161,408 → 150,926 |
| soundness: `eval::eval_term` agreement on probes incl. 0, negatives, <1e-6, ±inf | input; agree wherever the input is finite, NaN divergences listed separately (existing `x*0 -> 0`, `x-x -> 0` are not NaN-exact) |
| variety, generate mode | `denoise_candidates_assuming`, k=8 |
| µs per expression; hits per round; count of expressions > 64 nodes | egglog ≈ 3.6 ms fix, ≈ 1.3 ms candidates |

Second corpus — **frozen live population** (genes captured every generation from one real join run; `HFF_DUMP_GENES`-style capture added to the notebook source, off by default):

| Reported | Why |
|---|---|
| device-eligible share (≤ 64 nodes) | the kernel cannot touch the rest |
| greedy `LitMode::F32` vs `F64` agreement | what the device would get wrong |
| canonical Karva round-trip rate | can the tidy form become a gene at all |
| write-back eligibility (fits the head) and status counts | the unique feature depends on this |
| per-rule hit distribution; hits per round | which rules earn their place in a fixed-width table |
| coverage with the device subset (no computed-literal rules) vs the full set | cost of finding 3 above |

**Decision point — I stop and report.** The go/no-go for Phase 3 is the live-population table plus greedy coverage; hall-of-fame coverage against egglog is diagnostic. If it falls short, Phase 1 still ships as the faster CPU linter with the kingdom filter. Known leak to watch: egglog matches modulo e-class within a pass; the linter needs extra rounds (`Sub (Mul x 1) x`).

## Phase 3 — the device kernel (`src/lint/device.rs`, `pack.rs`; `feature = "gpu"`)

One **fused greedy kernel**, one invocation per expression, no hit list and no readback: facts + 32-bit hash scan → first hit in (class A before B, position, rule id) order → three-block splice + liveness compaction → ping-pong buffers for R rounds → existing K5 evaluate. Beam and generate mode stay on the CPU.

- Rule row: fixed 544 B, all `u32`: header (`rule_id, root_op, m, k, n_mv, class, masks`), `pat[15]` and `tmpl[15]` heap-indexed (depth ≤ 3; a repeated metavar index is the same-subtree constraint), `prog[8]` packed postfix. Uploaded once per kingdom, resident.
- Per-node aux word from the host: literal-class id (exact literals seen in rules: −3,−2,−1,0,1,2,3,4,6) + predicate bits (`<0, >0, >=0, <1e-6, >-1e-6`). The kernel compares integers only.
- Same-subtree: hash is a **filter**; acceptance needs an exact lockstep walk (32-slot queue). A hash collision would turn `Sub x y` into 0 — a wrong answer, not a missed rewrite.
- No fold on the device (libm and Metal `sin`/`exp` differ bitwise); fold once on the host at ingestion.
- Unary nodes carry `arg1 = 0` (the root index): matching, hashing and liveness ignore `arg1` when arity < 2.
- Reuse `GpuEvaluator`'s device, 2-D dispatch, explicit buffer destroy + poll.
- Named SR seam: the binary heap assumes arity ≤ 2 (`HEAP_ARITY`); the CPU `Pat` does not.

## Phase 4 — engine integration

- PyO3: `GpuSession.lint(...)`.
- hff join: evaluate the tidy form always; write-back under `GRAFT_MODE`, through the CPU f64 path. Check with the existing `HFF_JOIN_CHECK` harness.
- Runs (nohup, `tail -f` given): 13-problem set must stay 13/13; then an A/B — linter on vs off, evolved-only problems (`HFF_RULES=none`), 3 seeds — exact count, generations-to-exact, wall time.

## Not in this plan

REGEX or any second kingdom; multityped arity resolution (spec §8); class D / K6 / signature on the device; engine stages 1–3 (population, reductions, HFF on the device) — separately the larger wall-time win.

## Verification

- Census snapshot: admitted / refused per file, with reasons.
- Classifier tests per verdict above, including one showing sum-of-depths rejects `Sub (Neg a) b` and M_neg accepts it.
- Per-rule differential test: instantiate each admitted rule with random subtrees, apply, compare `eval_term` on the probe rows.
- Splice invariant: child > parent, root at 0, no dead nodes — incl. `k = 0` (`Mul x 1 -> x`), hit at the root, dropped metavariables (`Mul x 0`).
- Carried invariants: `smallest_form_folds_constants_but_never_an_input`, `concretize_never_rewrites_an_input_variable`.
- Termination: the measure strictly decreases along every recorded chain on the 847. Determinism: two runs byte-identical (rule ids = source order; no HashMap iteration).
- GPU parity (`cfg(all(test, feature = "gpu"))`): canonicalised device output == CPU greedy `LitMode::F32`; hash `mix` value parity Rust ↔ WGSL; WGSL covers every rule-row opcode.
- Every commit: `RUSTFLAGS="-D warnings" cargo test`, `cargo clippy --all-targets`, and both with `--features gpu` from Phase 3. Local commits, no push.

## Estimate

Phase 1: 3–4 days (class F test, full reader grammar) · Phase 2: 1 day (two corpora) · **gate** · Phase 3: 5–6 days (lighter than first estimated — no variable-size output) · Phase 4: 2 days.
