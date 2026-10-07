# Saturated share-equivalence detection — fuller-side implementation plan

## Context

`fuller::extract::maximal_shared` already finds *exact* structural repeats
across a chromosome's genes (function-field equivalence), via pure
hash-consing with no rewrite rule firing. It cannot find equivalence that
only exists after rewriting — e.g. `exp(x+y)` and `exp(x)*exp(y)` are never
asserted as the same node, so hash-consing never merges them.

`docs/Saturated_Share_Equivalence_v1.md` is the governing design (arrived at
through several rounds of adversarial review, each catching a real flaw in
the previous round — recorded in its §4). Its corrected §5 sequence is:
assert all genes into one e-graph → saturate with a bounded rule family →
DAG-extract with free reuse → run the existing site/containment detection on
the extracted result → (cost-gated folding, out of scope here).

This plan implements **steps 1–3 only** (assert+saturate, DAG-extract,
detect) — fuller-side, nothing in phylu. Steps 4–9 (real per-use cost
gating, block-and-re-extract, chromosome-level fold/no-fold gate, GENEREF
write-back) are a later increment (`phylu/docs/PLAN_saturated_fold_cost_gating.md`),
once steps 1–3 are built and measured.

### Implementation status

Steps 1-4 (the `share` ruleset, `extract_dag.rs`'s extractor, `extract.rs`'s
refactor and new `maximal_shared_saturated` entry point, module
registration) are **built, tested, and committed** in fuller. §2's
algorithm description below has been corrected to match the real,
implemented per-root coverage rule (a flat-union version was tried first
and found wrong by running the real pipeline, not by inspection — see §2's
inline correction note). §6 (real-population rule mining) is **not yet
built** — it may belong in phylu instead of fuller (it needs a population
export and `fold_to_dag_measure.rs`'s harness, both phylu-side); this is
flagged as an open question, not decided unilaterally.

### Revision note (v1 → this version)

A prior version of this plan used plain greedy-DAG extraction (price each
e-class only by what's reachable *through it*) and asserted, in its test
(d), that this correctly picks the shared `exp(x)*exp(y)` form in the
motivating example. **That was wrong, and has been corrected.** Plain
greedy-DAG prices each root independently: for `gene0 = exp(x+y)` alone,
`exp(x+y)` costs 2 and `exp(x)*exp(y)` costs 3, so greedy-DAG picks
`exp(x+y)` — it has no way to know `exp(x)` and `exp(y)` are *already paid
for* by `gene1`/`gene2`. The algorithm below (§3.2) replaces greedy-DAG with
a **marginal-cost fixpoint across all roots jointly**, which is what
actually produces the motivating result. This was caught by concretely
probing the three-gene case before writing any implementation code — see
§3.3 for the probe results, which are real, not hypothetical.

A second round of review (after this document's first draft) caught five
more real problems, all fixed below: the three-gene test's expected result
was itself wrong (§5's test (d) now expects two 2-site matches, not one
3-site match — see the worked trace in §5); the marginal-cost fixpoint can
OSCILLATE, not just converge (§3.2 now caps rounds and keeps the
best-seen `total_cost`, explicitly framed as a heuristic, cross-checked
against brute force on small fixtures); `repeat N` bounds iteration COUNT
but not e-graph SIZE, and the reverse rule combined with
commutativity/associativity can blow up combinatorially on real genes
(§1 adds an explicit node-count guard, checked via egglog's own
`num_tuples()`, independent of the iteration cap); and the ruleset choice
(one hand-picked `Exp`/`Add` rule) proves the mechanism on a demo case but
says nothing about which rewrites matter on real evolved populations —
§6 (new) adds a mining step against real population exports before
deciding what else, if anything, to load.

A third thing was checked and fixed before writing this version: the
rewrite rule search needs is **one-directional** in fuller's existing
`powers.rs` (`Mul(Exp a, Exp b) → Exp(Add a b)`, never the reverse), and
egglog only fires rules on patterns that are literally present in the
e-graph — three bare roots (`exp(x+y)`, `exp(x)`, `exp(y)`) never construct
a `Mul(Exp a, Exp b)` node on their own, so without the reverse direction
they never unify. §2.1 below adds that reverse rule, in its own new
ruleset, confirmed working by direct probe (§3.3).

Facts this plan depends on, confirmed by two research passes (Explore +
Plan subagents) plus two direct probes against a live egglog instance
(not just read from source):

- `maximal_shared`'s exact mechanics (`src/extract.rs:1508-1602`): fresh
  egraph, load `MATH_DATATYPE`+`GUARD_RELATIONS` only (no saturation), assert
  each gene as `(let gene{i} {ast})`, `serialize()`, DFS-collect
  `(gene_idx, path)` sites keyed by `NodeId`, filter/sort/exclude by
  containment.
- **A previously-undocumented correctness hazard**: egglog's own
  `serialize()` resolves a child edge's `NodeId` by **round-robin rotation**
  over that child e-class's node pool
  (`vendor-egglog/src/serialize.rs::serialize_value`, ~line 314-342). On
  today's un-saturated input this is invisible (every e-class has exactly
  one member, so rotation over a 1-element pool is a no-op). The moment
  saturation runs, an e-class can have multiple members, and two different
  parent edges into the *same* e-class can resolve to *different* `NodeId`s.
  `maximal_shared`'s `NodeId`-keyed site map would silently split a
  genuinely-shared class's sites across multiple keys and undercount or miss
  it below the `>= 2` threshold. This means "saturate, then just call
  `maximal_shared` on the result" is unsound, independent of the extractor
  question — this is why extract-before-detect (§5 steps 2 then 3, in that
  order) is a correctness requirement, not an optimization.
- `extraction-gym` (the reference greedy-DAG implementation from the
  egg/egglog ecosystem) is **bin-only upstream** — no `lib.rs`, cannot be
  added as a Cargo dependency. Even if it could be, plain greedy-DAG is the
  wrong algorithm for this use case (see revision note above) — this plan's
  extractor is NOT a port of extraction-gym's algorithm, only informed by
  its data-structure shape (operating directly on `egraph_serialize`
  types).
- No DAG-cost extraction code exists anywhere in this dependency tree today
  (`egraph_serialize`'s own `algorithms.rs` only has visualization helpers).
  This is genuinely new code, not a wiring exercise.
- Saturation in this codebase is *always* `(repeat N (run <combined-ruleset>))`
  with a small fixed `N` (e.g. `denoise`'s `DENOISE_ITERS = 40`) — never
  unbounded `(saturate ...)`. "Kill-guard" here means "bounded repeat count,"
  not a runtime timeout wrapper. `denoise` uses only algebra+powers+sign
  (never distribute/trig/rational together — CLAUDE.md's non-confluence
  warning, verified: co-saturating those explodes the e-graph).
- `powers.rs:51` has `(rewrite (Mul (Exp a) (Exp b)) (Exp (Add a b))
  :ruleset powers)` — confirmed present, confirmed one-directional, by
  direct read.

## Approach

### 1. `src/ruleset/share.rs` (new): the reverse rule, in its own ruleset

```rust
pub const SHARE_RULESET: &str = r#"
(ruleset share)
(rewrite (Exp (Add a b)) (Mul (Exp a) (Exp b)) :ruleset share)
"#;
```

**Deliberately NOT added to `powers`** — this direction is an expander (it
can only grow an e-class, never shrink it), and `denoise`'s `extract_variants`
hangs on large classes built by expander rules (the exact reason
`distribute` is excluded from `denoise` today). `share` is loaded only by
the new `maximal_shared_saturated` entry point (§4 below), never by
`denoise`.

**Blowup guard, required, not optional**: the reverse rule combined with
`algebra`'s existing commutativity/associativity on `Add`/`Mul` can expand
`exp` of an n-term sum into every partition of those terms — confirmed by
direct probe, a 3-term nested sum already produced 31 serialized nodes
under `repeat 40` (§3.3's probe). Real evolved genes, with more terms and
deeper nesting, will be worse. `repeat N` bounds ITERATION COUNT, not
e-graph SIZE — these are not the same guard, and relying on the iteration
cap alone is not sufficient. `maximal_shared_saturated` (§4) must check
e-graph size independently, between (or instead of relying solely on)
repeat iterations:

```rust
/// Checked between schedule steps (via egglog's own, already-cheap
/// `EGraph::num_tuples()` — a running total tuple count, not a
/// serialize-time operation): if the e-graph's total size exceeds this
/// ceiling, saturation stops EARLY (whatever has been proven so far is
/// kept; this is not a kill-guard in the "machine got pegged" sense,
/// it is a planned, graceful early stop). Chosen generously above what a
/// real chromosome's genes should ever need; tightened only if real
/// population testing (§6) shows it is too generous in practice.
const SHARE_MAX_TUPLES: usize = 50_000; // provisional -- revisit after §6's real-population mining
```

Implementation note: egglog's `run-schedule` doesn't itself expose a
per-step size callback from the Rust API in the same call used elsewhere
in this codebase (`parse_and_run_program`) — the straightforward
implementation is to run `repeat` in SMALLER increments (e.g. `repeat 5`
at a time, in a Rust-side loop) and check `egraph.num_tuples()` between
increments, stopping the loop (not erroring) the first time the ceiling is
exceeded or `SHARE_ITERS` total iterations are reached, whichever comes
first. Both guards are real and independent; neither alone is sufficient
(confirmed: the existing codebase relies on iteration count alone today
BECAUSE its existing bounded rulesets, algebra+powers+sign, don't combine
with an expander like `share`'s reverse rule — this plan is the first
place in this codebase pairing a bounded family with a genuine expander in
the SAME schedule, so it needs a guard the existing code has never needed).

Confirmed by direct probe (not assumed): asserting three BARE roots
(`Exp(Add x y)`, `Exp(x)`, `Exp(y)`) with algebra+powers+sign alone leaves
every root in its own singleton e-class — no unification. Adding `share`
to the combined ruleset and re-running: `gene0`'s class gains a second
member (`Mul`), whose two children resolve to exactly `gene1`'s and
`gene2`'s classes. A second probe (`Exp(Add(Add(x,y),z))`, deeper nesting)
produced 31 serialized nodes under `repeat 40` — no blowup observed at this
depth. Both probes are reproducible by re-running the construction
described here against a live `egglog::EGraph`; they are not re-derived
from documentation.

### 2. `src/extract_dag.rs` (new): marginal-cost fixpoint extraction

Not plain greedy-DAG (see revision note). The real algorithm, concretely:

```rust
use egraph_serialize::{ClassId, EGraph as SerEGraph, Node, NodeId};
use std::collections::{BTreeSet, HashMap};

/// A concrete, acyclic extraction result: exactly one winning NodeId per
/// ClassId reachable from `roots`, plus the TOTAL joint cost (the union of
/// every root's reachable-set cost, each class counted once) — this total
/// IS the per-individual DAG cost a caller (phylu) will want to expose for
/// fitness/selection use, so it is a first-class field here, not
/// recomputed by a separate pass.
pub struct ExtractedDag {
    pub chosen: HashMap<ClassId, NodeId>,
    pub roots: Vec<ClassId>,
    pub total_cost: u64,
}

/// 0 for Num/Var leaves, 1 for every other op — matches
/// `extract::internal_op_count`'s existing convention. Skips nodes with
/// infinite cost (truncation placeholders) or `subsumed == true`.
fn op_cost(node: &Node) -> u64 { ... }

/// MARGINAL-COST FIXPOINT, jointly over ALL roots — NOT independent
/// per-root greedy-DAG, which is provably wrong here (revision note): a
/// candidate's cost must be priced relative to what the OTHER roots
/// already cover, or the extractor will pick the cheapest INDEPENDENT
/// form at every root even when a shared, jointly-cheaper form exists.
///
/// Algorithm (iterative, to a fixpoint — not a single pass):
///   1. Round 0: for each root independently, run a bottom-up worklist
///      (Bellman-Ford style, since e-graphs can have cycles among classes
///      via commutative/associative rules) picking the per-class minimum
///      COST node, same free-reuse rule as plain greedy-DAG: a node's
///      candidate cost = its own op_cost + the sum of its children's
///      CURRENT best candidate cost, each child's class counted once
///      even if reachable multiple ways (see `Candidate.reachable` below).
///      This round's result is a baseline, not the answer — it is exactly
///      where the old (wrong) version of this plan stopped.
///   2. **Correction found during implementation (the first version of
///      this plan had this wrong): track PER-ROOT reach, not one flat
///      union.** Build `root_reach: HashMap<ClassId /*root*/,
///      BTreeSet<ClassId>>` = each root's own round-0 chosen node's
///      reachable set, keyed by that root. A flat union loses which
///      root a class's coverage came from — and a root's OWN prior
///      choice making its own subtree "free" to itself is circular: it
///      produced a TIE (not a strict improvement) at the exact point
///      the motivating case needed a strict win, so the cheaper joint
///      form never displaced the independently-chosen one. Caught by
///      running the real pipeline end to end and getting the wrong
///      answer (`total_cost=4`/no matches on the three-gene case),
///      not by inspection.
///   3. Re-run the SAME bottom-up worklist. A class `k` is free to a
///      candidate being built for class `c` iff **some root `r`'s reach
///      (from step 2) contains `k` but does NOT contain `c`** — i.e. a
///      root whose own subtree doesn't even touch `c` already needs
///      `k` regardless of what `c` picks. (If `k`'s only coverage comes
///      from roots that also reach `c`, that coverage could be entirely
///      a consequence of `c`'s own current choice — not free.) Recompute
///      each root's chosen node under this per-target pricing.
///   4. Rebuild `root_reach` from the NEW per-root choices and compute
///      this round's `total_cost` (the union of every root's new reach,
///      each class counted once). **This is NOT guaranteed to
///      monotonically improve or settle** — pricing each root against the
///      OTHERS' current picks can oscillate (root A's pick depends on
///      root B's, whose pick depends on root A's, flipping back and
///      forth) rather than converging. Treat this as a bounded HEURISTIC,
///      not a provably-terminating fixpoint:
///        - Cap at `MAX_ROUNDS` (e.g. 8 — generous for 8 genes, cheap to
///          run further than needed since each round is one bottom-up
///          pass over a small e-graph).
///        - Keep the BEST `total_cost` seen across all rounds (and the
///          `chosen` map that produced it), not just the LAST round's —
///          an oscillating sequence can pass through its best answer
///          before leaving it again.
///        - If consecutive rounds repeat an already-seen `root_reach`
///          snapshot (cycle detected, not just "unchanged from last
///          round"), stop early — further rounds will only repeat the
///          same cycle.
///   5. `total_cost` = the best value kept per step 4, over the final
///      union-of-reach set that produced it (every class counted once).
///   6. Tie-break by lower NodeId (deterministic, not IndexMap order) —
///      required, not optional: ties are real (two forms costing the
///      same before one root's choice is known) and left to
///      HashMap/IndexMap iteration order otherwise, which is
///      non-deterministic across runs.
///   7. A class never reached from a leaf (fully cyclic, no leaf-bottomed
///      path) is an `Err`, not a panic — should not occur among real
///      root-reachable classes.
///
/// **This is a heuristic, not a proof of optimality** — cross-check against
/// brute-force (enumerate every e-node combination) or an ILP formulation
/// on SMALL hand-built fixtures (§5's tests) before trusting it on real
/// 8-gene chromosomes. If the heuristic and brute-force/ILP disagree on a
/// small fixture, that is a real bug to fix before this is used for
/// anything beyond the demo case, not a discrepancy to wave away as "close
/// enough."
///
/// Concretely verified END TO END (not just hand-traced) on the real
/// three-gene case, after the per-root fix (step 2): round 0 prices
/// gene0's `Exp(Add x y)` at 2, `Mul(Exp x, Exp y)` at 3 — `Exp(Add x y)`
/// wins round 0, exactly the baseline the old (wrong) version of this
/// plan's test (d) mistakenly asserted as final. Round 1, using gene1's
/// and gene2's round-0 reach (each just their own `Exp(x)`/`Exp(y)`
/// class): `Mul`'s candidate for gene0's class is free to use `Exp(x)`
/// (gene1's reach contains it, and gene1's reach does NOT contain
/// gene0's class) and `Exp(y)` similarly — marginal cost = own(1) + 0 +
/// 0 = 1. `Exp(Add x y)`'s marginal cost = own(1) + Add(1, NOT free to
/// anyone) = 2. `Mul` wins round 1, strictly (not a tie — the per-root
/// fix is exactly what turns this from a tie into a strict win; a flat
/// union version of this algorithm produced a tie here and never
/// switched). `total_cost` = 3 (Exp x + Exp y + Mul), versus the
/// round-0/independent 4 — confirmed by running the real code, including
/// the two real bugs this surfaced during implementation (`rank` costing
/// a multi-member class via an arbitrary member instead of the
/// candidate's own node; the flat-union circularity above) — both fixed,
/// both now covered by regression tests.
///
/// Oscillation (step 4's stated risk) has NOT been witnessed by a
/// concrete fixture as of this plan's implementation — an attempt to
/// construct one showed that a fixture where round 0's independent
/// choice never reaches the shareable class also prevents the fixpoint
/// from ever discovering that class is shareable (marginal re-pricing
/// only changes costs for classes some round already reached), so that
/// specific shape settles trivially rather than flipping. The round-cap
/// and best-seen safeguards remain in place as insurance since the
/// heuristic is not proven not to oscillate on some other shape, but
/// "oscillation can happen" is not yet demonstrated, only guarded against.
///
/// At 8 genes (the real chromosome width), a handful of multi-member
/// classes, this fixpoint is cheap — bounded by the number of distinct
/// reach-snapshot states, which in practice converges in 1-2 rounds
/// past the baseline (an ILP formulation is an alternative if empirical
/// convergence turns out to be slow in practice; not needed to start).
pub fn extract_shared_dag(ser: &SerEGraph, roots: &[ClassId]) -> Result<ExtractedDag, String> { ... }

/// Rebuild `extracted` as a fresh, concrete `egraph_serialize::EGraph`:
/// one node per class (the winner), every `children[i]` REWRITTEN to
/// point at that child's class's winning NodeId (resolved via the
/// ORIGINAL ser's `.eclass`, then `extracted.chosen`) — NOT copied
/// verbatim from `ser`, which would silently reintroduce the rotation
/// hazard (§Context). This is the object `shared_sites_in` walks; one
/// member per class makes `class.nodes.first()` unambiguous by
/// construction, so no further ClassId-rekeying is needed downstream.
pub fn materialize(ser: &SerEGraph, extracted: &ExtractedDag) -> SerEGraph { ... }
```

The load-bearing correctness point carried over from the prior version:
the rotation hazard is fully resolved by extract-before-detect **only if**
(a) the extractor always re-resolves children via `.eclass`, never via a
raw `NodeId`, and (b) `materialize` rewrites every child edge from
`extracted.chosen`, rather than copying `ser`'s original (rotated)
`children` field. Both must hold; this is unchanged by the algorithm swap.

### 3. `src/extract.rs` (modified)

1. **Extract `maximal_shared`'s site/containment block** (today inline,
   lines ~1540-1601) into a standalone `pub(crate) fn shared_sites_in(ser:
   &egraph_serialize::EGraph, min_ops: usize) -> Vec<Match>`. `maximal_shared`
   becomes: build+assert+serialize (unchanged) then `Ok(shared_sites_in(&ser,
   min_ops))`. No behavior change — the 4 existing `maximal_shared` tests
   must pass unmodified; this is a pure refactor enabling reuse.

2. **New public entry point**, now also returning the joint DAG cost
   (per the per-individual-cost feedback — this is a natural output of
   step 2's extractor, not a separate pass):
   ```rust
   const SHARE_ITERS: u32 = 40; // TOTAL iteration cap across all of §1's repeat-5 chunks, same convention as DENOISE_ITERS
   const SHARE_CHUNK: u32 = 5;  // iterations per chunk, between which §1's num_tuples() guard checks

   /// docs/Saturated_Share_Equivalence_v1.md §5 steps 1-3: assert all genes
   /// into one e-graph, saturate with the same bounded algebra+powers+sign
   /// family `denoise` trusts PLUS the new `share` ruleset (the reverse
   /// exp(a+b)->exp(a)*exp(b) direction, needed for forms to actually
   /// unify — see §1), DAG-extract jointly with the marginal-cost fixpoint
   /// (§2), run the same site/containment detection `maximal_shared` uses.
   /// Finds equivalence-after-rewriting (exp(x+y) vs exp(x)*exp(y)), not
   /// just exact repeats. Steps 4-9 (cost gating, fold) are NOT here.
   ///
   /// Returns the matches AND the chromosome's total joint DAG cost
   /// (`ExtractedDag::total_cost`) — phylu's fitness/selection layer can
   /// use the cost directly without re-deriving it.
   pub fn maximal_shared_saturated(genes: &[String], min_ops: usize) -> Result<(Vec<Match>, u64), String> { ... }
   ```
   Body, mirroring `denoise_assuming`'s proven order: load
   `MATH_DATATYPE`+`GUARD_RELATIONS`+`ALGEBRA_RULESET`+`POWERS_RULESET`+
   `SIGN_RULESET`+`SHARE_RULESET`, then one `parse_and_run_program`
   asserting every `(let gene{i} {ast})` plus
   `(unstable-combined-ruleset share_all guards algebra powers sign share)`
   — NOT followed by a single `(run-schedule (repeat {SHARE_ITERS} ...))`
   call. §1's blowup guard is load-bearing, not optional, so the actual
   saturation step here MUST be §1's chunked loop: run
   `(run-schedule (repeat 5 (run share_all)))` in a Rust-side loop,
   checking `egraph.num_tuples()` against `SHARE_MAX_TUPLES` after each
   chunk, stopping (not erroring) at whichever comes first — total
   iterations reaching `SHARE_ITERS`, or the tuple ceiling exceeded. (An
   earlier draft of this plan described §1's chunked guard but then wrote
   §3's body as if a single bare `repeat 40` call were still correct — it
   is not; this is the one real saturation call this plan makes, and it
   must be the guarded loop, not the unguarded shortcut.) After the loop,
   `eval_expr` each root, `serialize()` with the same `SerializeConfig`
   shape, `extract_dag::extract_shared_dag`, `extract_dag::materialize`,
   `shared_sites_in` on the result. Empty input
   → `Ok((Vec::new(), 0))`.

### 4. `src/lib.rs` (modified)

Add `pub mod extract_dag;` alongside the other flat `pub mod` declarations.
Add `pub mod ruleset::share;` registration alongside the other ruleset
files (`src/ruleset/mod.rs`).

### Dependencies

No `Cargo.toml` change — `egraph_serialize` is already a direct dependency;
no extraction-gym code is pulled in as a crate (confirmed bin-only, and the
wrong algorithm regardless — revision note).

## Test plan

**`extract_dag.rs`** (new `#[cfg(test)] mod tests`, hand-built
`egraph_serialize::EGraph` fixtures, no egglog involved — isolates the
extractor from saturation entirely):

- **Cycle/termination test**: hand-build the `(Neg (Neg x))` shape (two
  classes referencing each other). Assert `extract_shared_dag` terminates
  and picks the `Var("x")` leaf (cost 0) for its class.
- **Marginal-cost fixpoint test (replaces the old "free-reuse diamond"
  test, which only exercised round 0)**: hand-build the exact arithmetic
  from the motivating case's analysis above — three root classes where
  round 0's independent per-root minimum disagrees with the jointly
  cheaper form, confirming the fixpoint actually iterates past round 0 and
  converges to the cheaper joint answer, not the locally-cheapest one per
  root. This is the test that would pass on a naive port of plain
  greedy-DAG and must fail there — i.e., write it so a greedy-DAG
  implementation visibly fails it, as a regression guard against
  reintroducing the bug this revision fixes.
- **Brute-force/ILP cross-check, on small fixtures ONLY (not real
  chromosomes — the point is correctness verification, not performance)**:
  for 2-3 small hand-built e-graphs including the motivating 3-root case,
  compute the true minimum joint cost by exhaustively enumerating every
  combination of one e-node per reachable class (small fixtures only, so
  this is tractable), and assert the fixpoint's `total_cost` matches the
  brute-force optimum exactly. This is the test that catches the fixpoint
  heuristic settling on a WRONG (non-optimal, or oscillation-artifact)
  answer that a unit test asserting only "it finds SOME answer" would miss
  entirely — required before trusting the heuristic on real 8-gene
  chromosomes, per the Context section's revision note.
- **Oscillation test**: hand-build a small e-graph deliberately
  constructed so that two roots' marginal-cost choices flip back and
  forth across rounds (two classes, each cheaper only when the OTHER
  root's current pick already covers part of it, and vice versa) —
  confirm the round cap and best-seen tracking produce the correct
  answer despite the oscillation, not just "doesn't hang" (termination
  alone is not the claim; correctness despite non-monotonic rounds is).

**`extract.rs`** (extending the existing `#[cfg(test)] mod tests`, matching
its conventions: surgical imports, `r#"..."#` literals, a doc comment per
test naming the case it pins):

- **(a) Rule-exists sanity check, BOTH directions**: confirm `powers.rs`'s
  forward rule still fires (two roots `Exp(Add x y)` / `Mul(Exp x, Exp y)`
  unify under algebra+powers+sign alone, no `share`), AND confirm the new
  `share` ruleset's reverse direction is needed for the THREE-BARE-ROOT
  case (gene0/gene1/gene2 as three independent asserts) — assert this
  does NOT unify under algebra+powers+sign alone, and DOES unify once
  `share` joins the combined ruleset. Both halves of this test are
  required, not just the positive case — the negative half is what
  catches a future accidental removal of the `share` dependency.
- **(b) ClassId-rekeying regression**: two genes sharing a non-root
  `exp(x)*exp(y)`/`exp(x+y)` subexpression (referenced by exactly 2 parent
  edges — the exact shape that triggers `serialize()`'s rotation). Assert
  `maximal_shared_saturated` finds the match with the right sites and op
  count; separately assert directly on the raw saturated egraph that the
  shared class really does have ≥2 member nodes.
- **(c) Fixpoint correctness via a real saturated case**: a case with
  genuine commutative/associative rewriting in play, confirming
  `extract_shared_dag` converges and the chosen result is bit-consistent
  with (a)'s rule check.
- **(d) THE HEADLINE ACCEPTANCE TEST — the three-gene motivating case,
  corrected TWICE now (first the winning form, then the match shape)**:
  gene0 = `(Exp (Add (Var "x") (Var "y")))`, gene1 = `(Exp (Var "x"))`,
  gene2 = `(Exp (Var "y"))` — THREE bare roots, exactly as the motivating
  conversation posed it.

  **The match shape, worked through explicitly (this is NOT "one match
  spanning three genes" — an earlier draft of this test assumed that, and
  it is wrong)**: once extraction picks `Mul(Exp x, Exp y)` for gene0's
  class, gene0's materialized tree contains an `Exp(x)` sub-node (shared
  with gene1's WHOLE root) and an `Exp(y)` sub-node (shared with gene2's
  whole root) — but `Exp(x)` is NOT shared with gene2, and `Exp(y)` is NOT
  shared with gene1. So `maximal_shared_saturated` finds **two separate
  matches**, not one: `{sites: [(0, [0]), (1, [])], internal_op_count: 1}`
  (the `Exp(x)` match, gene0's path to it under the chosen `Mul` vs
  gene1's whole root) and `{sites: [(0, [1]), (2, [])],
  internal_op_count: 1}` (the `Exp(y)` match, symmetric). Each has
  `internal_op_count == 1`, so **`min_ops` must be passed as `1`, not the
  library's usual default assumption of `>= 2`** — at `min_ops >= 2` both
  matches are silently dropped by `shared_sites_in`'s own filter, and the
  test would wrongly read as "found nothing" rather than "found the right
  two small matches." This is exactly the mistake an earlier draft of this
  plan made in reasoning about the test (confirmed by walking the
  containment-exclusion logic by hand, not by running code that doesn't
  exist yet — flagged as a result to re-confirm once `shared_sites_in` and
  the extractor are actually implemented, since hand-tracing fuller's
  existing, real site/path convention against a NOT-YET-BUILT extractor's
  output is inference, not measurement).

  Assertions: `maximal_shared_saturated(&[gene0, gene1, gene2], 1)`
  returns `total_cost == 3` (not 4 — confirming the marginal-cost
  fixpoint, not independent per-root pricing, actually ran) and exactly
  TWO matches, each `internal_op_count == 1`, with the sites described
  above. Separately, confirm the chosen form for gene0's class is `Mul`
  by checking `extracted.chosen` directly, not inferred from `total_cost`
  or the match count alone.

## Verification

- `cargo test` (plain, no `gpu` feature needed — this is pure CPU/egglog
  code) — all new tests pass, all 4 existing `maximal_shared` tests pass
  unmodified (confirms the refactor is behavior-preserving).
- `RUSTFLAGS="-D warnings" cargo build` and `cargo clippy --all-targets --
  -D warnings` — zero warnings, per this repo's hard rule.
- Manual sanity: run `maximal_shared_saturated` against the three-gene
  motivating case directly (not just via the automated test) and print
  `(matches, total_cost, extracted.chosen)`, confirming by eye that the
  joint-cost reasoning in §2's worked example matches what the real code
  produces — a one-time confidence check before calling this done.

## 6. Which rules to load beyond the demo — mine the real population, don't guess

One hand-picked rule (`share`'s reverse `Exp`/`Add`) proves the mechanism
works. It says nothing about whether rewrite-discoverable equivalence
actually occurs often enough in real evolved chromosomes to be worth this
machinery's real dispatch cost — the SAME lesson `FoldToDag`'s own history
already taught once (exact-match folding was correct but measured
3.6%-slower-net at this population's actual repeat rate; a plausible
mechanism is not the same as a worthwhile one).

Before deciding what else (if anything) to load alongside `algebra` +
`powers` + `sign` + `share`, mine the existing 800-individual population
exports (`GeneExport` JSON, the same files `fold_to_dag_measure.rs` already
reads) for a concrete answer to: **which rewrite-reachable equivalences
actually occur across this population's real genes, and how often?**

**Correction, caught on review: mining each existing family ALONE finds
almost nothing, and would wrongly suggest this whole approach doesn't
pay off.** `algebra`, `powers`, `sign` (and `trig`, `rational`) are, as
written, overwhelmingly CANONICALIZING/COMPRESSING rules (many-forms →
one preferred form) — the same direction `denoise` already uses them in.
Compression collapses variety; it does not, by itself, expose NEW sharing
between two already-different-looking expressions. The `exp` motivating
case only unified once `share`'s REVERSE direction was added (confirmed
directly: three bare roots do not unify under algebra+powers+sign alone,
§1's probe) — the forward rule (`Mul(Exp a,Exp b) → Exp(Add a b)`,
already in `powers`) was present the whole time and was NOT sufficient on
its own. Mining `powers` alone (forward direction only) against real data
would have reported "basically no additional matches" and been
MISLEADING about what the reverse direction can find.

Concretely, corrected:

- For each existing bounded family with a plausible expander/reverse
  counterpart (algebra ↔ an expander for its own identities where one
  exists; powers ↔ `share`'s reverse `Exp`/`Add`; sign; trig — mine each
  family TOGETHER WITH its own expander direction, not the compressing
  direction alone), run the real `maximal_shared_saturated` pipeline
  (not a hypothetical "would this unify" probe) against a sample of real
  exported populations, one family-pair at a time (still never
  co-saturating multiple expander directions together, per CLAUDE.md's
  non-confluence warning — this is about which SINGLE family-pair to try
  next, not about combining them).
- **Rank by total ops saved (the `(sites.len()-1)*internal_op_count` sum,
  matching `fold_to_dag_measure.rs`'s existing report shape), NOT by raw
  match count.** A family-pair that finds many trivial 1-op matches on
  every individual can outrank, by count, a rarer family-pair that finds
  large multi-op shared subtrees — but the latter is very plausibly the
  actually-valuable one, matching this plan's own motivating case (a
  SMALL number of HIGH-value matches, not a large number of negligible
  ones). Report both numbers (match count AND ops-saved sum) per family
  -pair, but let ops-saved sum be the ranking key for deciding what to
  load, since that is the number that actually determines whether
  §2/§7's cost-gating has anything worth gating.
- Only add a rule family (with its expander direction) to the live
  `maximal_shared_saturated` schedule if this mining shows it finds a
  non-trivial ops-saved total on a non-trivial fraction of real
  individuals — a family-pair that never fires, or only ever finds
  negligible matches, on real evolved genes adds saturation cost (and
  blowup risk, per §1) for no real benefit.

This mining step is new work (a new example, modeled on
`examples/largest_common_subgraph.rs`'s existing population-scanning
shape), not yet built, and is a PREREQUISITE for any decision to add rule
families beyond what §1 defines for the demo — not a nice-to-have cleanup
after the fact.

**Open question, not decided here: which repo should this live in?**
Mining needs a real population export (`GeneExport` JSON) and
`fold_to_dag_measure.rs`'s existing timing/reporting harness — both are
phylu-side, not fuller-side. This step may be a more natural fit as a
phylu example that calls fuller's `maximal_shared_saturated` with
different ruleset configurations, rather than a fuller example that would
need its own population-loading/rendering code duplicating what phylu
already has. Left open rather than decided unilaterally; whoever picks
this up next should weigh in before building it in either place.

## Noted but out of scope here

- **Doc gap**: `docs/Saturated_Share_Equivalence_v1.md` §3.1/§3.2 doesn't
  yet mention the `serialize()` round-robin rotation hazard, or that plain
  greedy-DAG is insufficient and a marginal-cost fixpoint (or ILP) is
  needed instead — both discovered during this planning pass, not caught
  by the doc's existing four rounds of review. Not edited as part of this
  plan (implementation only).
- Steps 4-9 of the design doc's §5 — phylu-side
  (`phylu/docs/PLAN_saturated_fold_cost_gating.md`), consuming this plan's
  `maximal_shared_saturated(genes, min_ops) -> Result<(Vec<Match>, u64), String>`
  signature directly, including the re-encoder needed because a saturated
  match's chosen form can legitimately differ from a site's original
  as-written gene text (unlike an exact match) — see that plan for how
  `relevel`'s existing `Graft` mechanism already supports this without new
  low-level machinery.
