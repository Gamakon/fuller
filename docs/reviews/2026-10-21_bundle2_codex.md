# Codex review, bundle 2 (fold statistics, scorer lanes, decode lanes, tie rule, fold test), 2026-10-21

Findings (ranked):

1. **High — landing-overwrite fix preserves only genome, not row metadata.**  
   [phylu/src/evolve/engine.rs:6746](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/engine.rs:6746) snapshots `to`’s genome when a prior winner lands on it, but later processing that winner reads its already-overwritten `wrapper_id` and cohort at [6751](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/engine.rs:6751) and [6757](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/engine.rs:6757).  
   Failing scenario: with `fold_top_k=0`, winner A lands in later winner B; B’s original genome is correctly folded from `taken`, then lands in C, but C inherits A’s wrapper/cohort rather than B’s. This can alter scoring/ALPS behavior. Snapshot wrapper/cohort alongside the genome.

2. **Medium — `23b5c743` is a legitimate response to “first differing row,” but weakens the test’s provenance proof.**  
   [phylu/src/evolve/engine.rs:10346](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/engine.rs:10346) now accepts *any changed row* whose first gene begins `Mul, x0, ?`. It does not prove that this row originated from the planted row-0 `Tanh` fold. Another independently folded/random gene can satisfy that three-token signature and contain a `1.0` constant, making the test pass without testing the planted candidate. Preserve/inspect an explicit landing record, or compare the full expected rewritten gene plus parent identity.

3. **Medium — tie tolerance intentionally trades correctness for stability at real small improvements.**  
   [phylu/src/evolve/engine.rs:2470](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/engine.rs:2470) rejects a candidate unless it improves by more than `1e-9 * |old|`. Thus a genuinely better subset is discarded whenever its relative improvement is, e.g., `5e-10`; this is not merely last-bit noise for f64 rescoring. It is deterministic, but can systematically prefer fewer genes over a measurable improvement. The stated f32-prediction error rationale is reasonable, but this is a policy change and should be tested at the decision boundary with realistic scores.

4. **Low — integer-power gene conversion changes floating-point/domain semantics.**  
   [fuller/src/gpu_eval.rs:779](/Users/andrewmorgan/Dev/gamakon/fuller/src/gpu_eval.rs:779) rewrites `Pow(x, integer)` into multiply chains for genes. For negative `x`, a general GPU `pow` may produce NaN while `x*x` is finite; even for positive values, overflow/rounding differs. `1c5b3a49` correctly confines this to gene encoding, but TSR tests now validate the expanded form rather than equivalence to source `Pow`. This is a numerical-order/domain hazard, not nondeterminism.

5. **No correctness collision bug in the new decoder hash.**  
   The additive 32-bit hash can collide, but dedup verifies length and every node word before merging at [dedup.wgsl:130](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/dedup.wgsl:130). Collisions cost dedup efficiency only. The lane reduction order is fixed, so it is deterministic.
