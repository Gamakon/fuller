# Adversarial review: TSR typed-GEP plan

## Verdict

The profile construction is sound for the *stated* 18-symbol `wide()` alphabet and rev-4 GEP layout.  It is not yet a complete kingdom specification for every engine/benchmark spelling: generic binary `pow` is explicitly left undefined, and the proposed experiment confounds the representation comparison with an old, known-bad stack table and uses only development seeds.

## 1. Profiles and limits

The 11 shapes for ceiling 3 are right.  They retain exactly the information a flattened add/sub or mul/pdiv parent needs: family, greatest arm order, and whether that greatest order is tied.  The merge table is symmetric in type arguments, as it must be even though `sub` and `pdiv` are not algebraically commutative.

Concrete probes all work:

* `add(P2,P2) -> A22` has order 3; `add(P3,P3)` has no row, so the equal-maximal-order case cannot make order 4.
* `psqrt(add(x,y)) -> P2.T1`; an enclosing add sees it as one order-2 arm.  A unary deliberately erases the child family, matching the stream-order definition.
* Reordering `add(M12,P1)` and `add(P1,M12)` gives `A21` in both cases.  The ordered signature rows still preserve evaluator argument order for subtraction/division.
* `pow2(x)`/`pow3(x)` are unary and correctly preserve `P1`; they are not a generic `pow(x,c)` rule.

With polymorphic leaves interpreted as *upper-bound* duals, induction gives actual profile <= declared profile, hence root order <= 3.  The same induction gives depth <= 2 because raising rows accept only levels 0/1.  This includes `Abs` as raising and `Neg` as level-keeping, exactly as the current depth oracle does.  Root terminals must be present as `ROOT` terminals as well as root-function twins; the stated 170 C terminals at four inputs implies this, but the generator/check should assert it explicitly.  Otherwise a bare variable/constant root would be lost and rev-4's demanded-root-terminal check would fail.

## 2. Signatures and spelling coverage

On paper the count is complete for the declared alphabet: 4 binaries (`add, sub, mul, pdiv`) and 14 unaries (`sin, cos, tan, tanh, psqrt, plog, pexp, abs, pasin, pacos, neg, pow2, pow3, pinv`).  Thus protected division/log/exp/sqrt/inverse, `Neg`, and `Abs` have rows.  `1/sqrt(z)` is not a primitive: it is `pinv(psqrt(z))`, so it is covered only if that is the actual raw evaluator spelling.

The gap is generic `pow`.  `pow(x, 2)` cannot silently use the binary merge rule: that would make its constant exponent a Strahler arm and turn two leaves into order 2, whereas the plan treats `pow2(x)` as order 1.  Conversely, if engine input can spell `Pow(x,c)`, omitting it means neither completeness nor the “every function” claim holds.  Fix: deterministically canonicalise supported literal exponents to the exact unary evaluator forms before evolution, or define a separate signature whose literal exponent is metadata, not an arm, and test it against `stream_order`; neither option rejects evolved genes.  Inventory actual `symbol_id` spellings (including aliases such as `ProtectedSqrt`/`ProtectedInv`) and require an exhaustive generated-row test, not merely a list in prose.

Every demanded dual needs both a variable terminal and `?_T` where constants are allowed.  In particular this includes `ROOT`; the root has `h=1`, `P=1`, no tail, and one Dc word.  Do not implement “every dual” as only the 33 non-root shape-depth pairs.

## 3. Rev-4 width arithmetic

The quoted widths are correct under the proposed rows and rev-4's per-producer bound, not the older total-head bound:

| arm | heads | leaf bound L | tails | Dc | width |
|---|---:|---:|---:|---:|---:|
| B, N=17 | 305 | 226 | 1,619 | 1,732 | **3,656** |
| C, N=17 | 913 | 674 | 9,625 | 10,442 | **20,980** |

For C, 24 duals have `h=32`, 9 have `h=16`, and `ROOT` has `h=1`: `24*32 + 9*16 + 1 = 913`.  Only the three plain shapes (`P1,P2,P3`) per depth level are unary-only producers; their head total is 240, so `L = 1 + 913 - 1 - 240 = 674` (the root's binary twin contributes the final one).  The plan should pin these intermediate values from generated TSVs, as proposed, rather than from a second hand-coded model.

## 4. Variation and Ferreira/GEP closure

Matched `(layout, gene, domain, output-type, offset)` crossover and same-domain mutation remain valid: a copied token has the same output dual, tails remain terminals, and rev-4's bound is independent of which legal token is selected.  Changing a head token can change which rows of other frames are expressed, but cannot make their tokens ill-typed; this is the typed analogue of Ferreira's head/tail closure, with no repair or rejection.

Two conditions are non-negotiable: crossover must include only homologous frames (including `Dc_T`), and root twins/ROOT terminals must use the same fixed root frame.  Do not call arbitrary physical-offset crossover “matched”: crossing a `P1.T0` byte with an `A22.T2` byte breaks the kingdom immediately.

## 5. Experiment fairness

The A/B/C/E square is a reasonable representation-location comparison: all enforce depth <= 2, with depth either decoded or typed.  D is useful as a historical reference, but not a fair estimate of “old hard rule”: it retains the tidied-spelling stack table already known to reject `pinv(psqrt(...))`, and it bundles the ceiling, windows, exp minimum, and pair table.  F removes some but not all of those confounds.

Fixes:

* Build any legacy stack table from raw decoded gene spellings/evaluator symbols, preserving both `pinv(psqrt(z))` and `psqrt(pinv(z))` when observed; report D-old and D-raw separately.  Do not derive it from tidied SymPy/model text.
* Keep 7001--7010 for implementation/calibration and arm selection only.  Pre-register the selected arm, then run previously untouched official/held-out seeds.  “Dev seeds only” makes the adoption decision a development result, not an SRBench comparison.
* Analyse paired seed/law outcomes (paired bootstrap or McNemar-style paired differences, clustered by law), not the independent-Bernoulli `sqrt(2np(1-p))` heuristic.  The same laws and seeds are run in every arm.
* State that C changes width, neutral-genotype multiplicity, operator sampling, and constraint location together.  It tests the kingdom, but does not isolate which of those mechanisms caused a gain/loss.

## 6. The exp lower bound

“`exp` never on a bare variable” is not expressible with these upper-bound types: `x` legitimately inhabits every demanded dual, so a `pexp` row can always receive it.  Removing leaf polymorphism would violate rev-4's requirement that every demanded tail type has a terminal; it is not a no-rejection solution.

The clean no-rejection choices are: drop the lower bound (the right primary TSR design), or introduce a genuine non-leaf terminal/atom such as a typed scaled-variable feature whose declared and evaluator semantics make it order >= 2, then give the restricted exp input type appropriate terminals.  The latter changes the hypothesis language and must be a separate, pre-registered arm; it must not be simulated by decoding `exp(x)` and rejecting it.
