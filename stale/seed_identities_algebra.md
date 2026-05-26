# Seed identities — `algebra` module

Governance-supplied curated identity list for the rule-extraction agent
(`docs/BRIEF_rule_extraction.md`, module 1). **The engineer owns this list.**
The agent transcribes these into `src/ruleset/algebra.rs` as egglog `rewrite`
rules, generates a corpus, scores via HFF-TrueNorth, and surfaces accepted
rules for review. The agent may *propose* additional identities from cited
references; those go through the same soundness + governance gate.

All identities are over the `Math` datatype (`src/expr.rs`) and the **real
domain only**. Each entry gives: the identity, its direction, preconditions,
and notes. Direction `->` means a directional rewrite (LHS rewrites to RHS);
`<->` would mean bidirectional (avoid unless necessary — bidirectional rules
enlarge the e-graph).

## Hard reminders (carry into every rule)

- **No bare commutativity** (`Add a b -> Add b a`) or **associativity**. egglog
  merges proven-equal e-classes; these blow up saturation. Where a literal can
  sit on either side, write BOTH oriented forms (as done in `identities.rs`),
  never a swap rule.
- **RHS variables must all appear on the LHS.**
- **Conditional rewrites need explicit guards.** Where a precondition is listed
  below, the rule ships only if the guard is expressible in egglog 2.0;
  otherwise reject and log.
- Numeric literals are `(Num f64)`. `0` = `(Num 0.0)`, `1` = `(Num 1.0)`.

## Tier 0 — already shipped in `identities.rs` (do not duplicate; reuse)

These are the denoise five; the agent should import/extend, not re-derive:

| # | Identity | Dir | Pre |
|---|----------|-----|-----|
| 0.1 | `Mul x 1 -> x` and `Mul 1 x -> x` | -> | — |
| 0.2 | `Add x 0 -> x` and `Add 0 x -> x`; `Sub x 0 -> x` | -> | — |
| 0.3 | `Mul x 0 -> 0` and `Mul 0 x -> 0` | -> | — |
| 0.4 | `Neg (Neg x) -> x` | -> | — |
| 0.5 | `Abs (Abs x) -> Abs x`; `Sqrt (Pow2 x) -> Abs x` | -> | real |

## Tier 1 — atomic algebraic identities (transcribe first, simplest)

| # | Identity | Dir | Pre | Notes |
|---|----------|-----|-----|-------|
| 1.1 | `Sub x x -> 0` | -> | — | self-subtraction |
| 1.2 | `Div x x -> 1` | -> | `x != 0` | needs guard; reject if unguardable |
| 1.3 | `Add x x -> Mul (Num 2.0) x` | -> | — | doubling |
| 1.4 | `Mul x x -> Pow2 x` | -> | — | introduce square |
| 1.5 | `Mul (Pow2 x) x -> Pow3 x` and `Mul x (Pow2 x) -> Pow3 x` | -> | — | cube |
| 1.6 | `Neg (Num 0.0) -> 0` | -> | — | sign of zero |
| 1.7 | `Sub 0 x -> Neg x` | -> | — | negation via subtraction |
| 1.8 | `Div x 1 -> x` | -> | — | div identity |
| 1.9 | `Inv (Inv x) -> x` | -> | `x != 0` | double reciprocal; guard |
| 1.10 | `Mul x (Inv x) -> 1` and `Mul (Inv x) x -> 1` | -> | `x != 0` | guard |

## Tier 2 — composite / canonicalisation (transcribe after Tier 1 lands)

| # | Identity | Dir | Pre | Notes |
|---|----------|-----|-----|-------|
| 2.1 | `Div (Mul a b) b -> a` and `Div (Mul a b) a -> b` | -> | divisor `!= 0` | cancel common factor; guard |
| 2.2 | `Neg (Mul a b) -> Mul (Neg a) b` | -> | — | sign pushing — directional, choose one canonical form only |
| 2.3 | `Sub a (Neg b) -> Add a b` | -> | — | double-sign |
| 2.4 | `Add a (Neg b) -> Sub a b` | -> | — | normalise add-of-neg to sub |
| 2.5 | `Pow2 (Neg x) -> Pow2 x` | -> | — | even power kills sign |
| 2.6 | `Abs (Neg x) -> Abs x` | -> | — | abs kills sign |
| 2.7 | `Sqrt (Mul x x) -> Abs x` | -> | real | alt form of 0.5 (Mul instead of Pow2) |

## Tier 3 — `diff_sq` semantic op (from BRIEF semantic_id set)

`diff_sq(a,b) = (a-b)^2`. There is no `DiffSq` constructor; it is expressed by
expansion so the e-graph can rewrite through it:

| # | Identity | Dir | Pre | Notes |
|---|----------|-----|-----|-------|
| 3.1 | `Pow2 (Sub a b) <-> Add (Sub (Pow2 a) (Mul (Num 2.0) (Mul a b))) (Pow2 b)` | -> | — | (a-b)^2 expansion; ship the *contracting* direction (RHS->LHS) only, to shrink |

## Out of scope for the `algebra` module

- No trig identities (that's the `trig` module, later).
- No transcendental simplification (`log(exp x)`, `exp(log x)`) — those need
  domain guards (`x>0` for `exp(log x)`); defer to a guarded pass and only if
  the guard is expressible.
- No constant folding here (`cos(0) -> 1`): that is value-driven, handled by
  the evaluator/extraction loop, not a structural rewrite.

## Acceptance note for the agent

Transcribe in tier order (1 -> 2 -> 3), simplest first, one rule at a time, per
the brief's accept/reject gate. A rule that cannot be guarded soundly (1.2,
1.9, 1.10, 2.1) is **rejected and logged**, not shipped unguarded. Batch
accepted rules for engineer review before they land.
