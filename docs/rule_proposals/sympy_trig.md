# Rule proposals — mined from sympy's trig simplifiers

Source read: sympy 1.13.1, `sympy/simplify/fu.py` (2111 lines: TR0..TR22, TR111, TR2i,
TR10i, TR12i, TRmorrie, TRpower, `_osborne`) and `sympy/simplify/trigsimp.py`
(`_trigpats` tables at :813, `_match_div_rewrite` :945, `__trigsimp` :998, `exptrigsimp` :572).
Below, `fu:N` = `sympy/simplify/fu.py:N`, `ts:N` = `sympy/simplify/trigsimp.py:N`.
Nothing was compiled or run through cargo. Node counts are by hand (metavariables by multiplicity, `(Num k)` = 1).

## 0. Read this first

- **sympy has no rule for nested trig.** Neither file touches `Sin (Sin x)`, `Tan (Tan x)`, `Tan (Sin x)`.
  Every fu/trigsimp rule needs two trig calls on the SAME argument (or a sum argument). Checked directly:
  `trigsimp` and `simplify` return `cos(sin(Abs(x)))`, `Abs(sin(Abs(x)))`, `sin(Abs(x))**2`, `tanh(Abs(x))` unchanged for real `x`.
- **Measured on `genes.jsonl` (846 genes, `fuller_math`):** every binary node whose two children are both
  trig/Inv/Pow2-headed has DIFFERENT arguments (one `same` hit: `Sub (Tanh u) (ProtectedInv u)`, not a trig identity).
  `Sub 1 (Pow2 (Sin|Cos ..))`: 0. `(e^y - 1)/(e^y + 1)`: 0. pi-valued literals: 0.
  So the predicted fire count of every rule below on the current hall of fame is 0. They are correct shrinkers that
  wait for a same-argument pair to show up; they are not a fix for the nested-trig mass.
- Grammar limits that decided admissibility: leaf depth <= 3 (root = 0, `reader.rs:339`), so
  `(Sub (Inv (Pow2 (Cos x))) 1)` and the tan-addition quotient are refused; `:when` can only name a
  metavariable, never a subterm like `(Cos x)`; NUM has no `sin`, so special-angle folding cannot be written.
- `trig.rs` / `trig_fu.rs` are not among the linter's sources (`src/lint/mod.rs:30-33`) and `trig_fu` is all
  `birewrite`, which `reader.rs` rejects as a load error. Section B of the block is the ONE-WAY CONTRACTING half
  of identities those files already hold — flagged, separate, drop it if that counts as re-proposing.

## 1. Table — sympy rule -> ours

| sympy rule | where | ours |
|---|---|---|
| TR0 normal/factor/expand | fu:31 | not a rule (polynomial normaliser) |
| TR1 sec,csc -> 1/cos,1/sin | fu:40 | not expressible: no Sec/Csc; `Inv (Cos x)` already is the target form |
| TR2 tan -> sin/cos | fu:64 | not admissible: grows 2 -> 5 (in `trig_fu` as birewrite) |
| TR2i sin/cos -> tan | fu:180, ts:822, ts:950 | **A1** `Div (Sin x) (Cos x) -> Tan x` is in section B (trig_fu holds it bidirectionally); **A2** Mul/Inv spellings are new |
| TR2i cos/sin -> 1/tan | fu:190 | **A3** `Div (Cos x) (Sin x) -> Inv (Tan x)` 5 -> 3, + Mul/Inv spellings |
| TR2i half: sin/(1+cos) -> tan(x/2) | fu:185 | **A4** 7 -> 4, both Add orders |
| TR2i half: (1+cos)/sin -> 1/tan(x/2) | fu:197 | **A5** 7 -> 5, both Add orders |
| TR2i with powers `sin^c/cos^c` | fu:134, ts:822 | **A6** Pow2 only (`Pow2 (Div ..)` is reached by A1 anyway) |
| matchers_division[1] tan^c cos^c -> sin^c | ts:823, ts:953 | c=1 exists (`trig.rs`); **A7** the Pow2 form, both orders |
| matchers_division[2,4,5] cot forms | ts:824-827 | not expressible as cot; `Inv (Tan x)` spellings reduce to `x * Inv x`, which is `rational`'s job |
| matchers_division[3] tan/sin -> 1/cos | ts:825, ts:959 | **A8** guarded `is-nonzero x` (in f64 `sin x == 0` only at `x == 0`; lhs is NaN there, rhs is 1) |
| matchers_division[6,7] / TR14 (f+1)(f-1) -> -g^2 | ts:828-831, fu:1329 | **A9** four orientations of `(1-f)(1+f) -> g^2`; the `(f-1)(f+1) -> Neg` twins omitted (reach A9 through sign rules) |
| matchers_division[8..13] sinh/cosh/coth | ts:833-838 | not expressible: no Sinh/Cosh/Coth |
| matchers_division[14] tanh addition | ts:840 | not expressible: leaf at depth 4 |
| matchers_add[0,1] / TR10i sin(a+b), cos(a+b) | ts:845-846, fu:638 | section B (contracting half of the trig_fu birewrites) 11 -> 4 |
| matchers_add[2,3] sin(a-b), cos(a-b) | ts:847-848 | **A10** new, 11 -> 4 |
| matchers_add[4,5] sinh/cosh | ts:849-850 | not expressible |
| matchers_identity sin^2 -> 1-cos^2, tan^2 -> sec^2-1, angle expansion | ts:854-868 | not admissible: all grow (they are trigsimp's search moves, undone by `artifacts`) |
| artifacts[0] a - a cos^2 -> a sin^2 | ts:874 | **A11** a=1, Sub form, 5 -> 3 (and sin twin = TR6 read right-to-left). `trig.rs` has only the `Add (Num -1.0) ..` spelling. The general-coefficient form `Sub a (Mul a (Pow2 (Cos x)))` has its leaf at depth 4 — refused |
| artifacts[1] a - a/cos^2 -> -a tan^2 | ts:875 | **A12** via `Pow (Cos x) (Num -2.0)` (the form `trig.rs` emits); the `Inv (Pow2 (Cos x))` spelling is depth 4 — refused |
| artifacts[2] 1 - 1/sin^2 -> -cot^2 | ts:876 | not expressible at depth 3 with a shrinking target |
| artifacts cosh/sinh rows, noncommutative rows | ts:877-888 | not expressible / not applicable |
| TR3 induced formula, sign of argument | fu:210 | `sign.rs` has it (`Cos (Neg x)`, `Sin/Tan/Tanh (Neg x)`). pi-shift forms (`Sin (Add x (Num 3.14..))`) expressible but 0 pi literals in the census, and `Tan` near a pole fails any evaluation tolerance — not written |
| TR4 special angles | fu:259 | not expressible: NUM has no sin/cos primitive; `Sin/Cos/Tan (Num 0.0)` already in `algebra` |
| TR5 sin^2 -> 1-cos^2, TR6 cos^2 -> 1-sin^2 | fu:355, fu:376 | forward grows 3 -> 5; the reverse is A11 |
| TR7 cos^2 -> (1+cos 2x)/2 | fu:416 | not admissible: grows 3 -> 8 (in `trig.rs`) |
| TR8 product -> sum | fu:421 | not admissible: grows (in `trig_fu`) |
| TR9 sum -> product | fu:496 | not admissible: `Add (Sin a) (Sin b)` 5 -> 13; contracting reverse has a leaf at depth 4 |
| TR10 angle expansion | fu:590 | not admissible: grows (in `trig_fu`) |
| TR11 double angle expansion | fu:770 | grows; the contraction is section B; **A13** `sin^2 - cos^2 -> -cos 2x` is new |
| TR12 tan(a+b) expansion | fu:940 | not admissible: grows 4 -> 13 |
| TR12i tan sum quotient -> tan(a+b) | fu:946 | not expressible: `(Div (Add (Tan a) (Tan b)) (Sub (Num 1.0) (Mul (Tan a) (Tan b))))` has leaves at depth 4 |
| TR13 tan*tan -> 1 - ... | fu:1102 | not admissible: grows 5 -> 16 |
| TRmorrie cos x cos 2x -> sin 4x / (4 sin x) | fu:1205 | not admissible: 7 -> 9 |
| TR14 | fu:1235 | see A9 |
| TR15 sin^-2 -> 1+cot^2 | fu:1356 | not expressible: cot; grows |
| TR16 cos^-2 -> 1+tan^2 | fu:1389 | forward grows 4 -> 5; reverse `1 + tan^2` is in `trig.rs` for one Add order, **A12** adds the other |
| TR111 f^-i -> g^i | fu:1422 | not expressible: no Cot/Sec/Csc |
| TR22 tan^2 -> sec^2 - 1 | fu:1454 | forward grows; reverse is A12 |
| TRpower sin^n -> multiple-angle sum | fu:1483 | not admissible: grows |
| `_osborne` / `hyper_as_trig` | fu:1967, fu:2046 | not expressible: needs Sinh/Cosh and complex `I` |
| `sincos_to_sum` | fu:2089 | not admissible: grows |
| exptrigsimp `(1 - e^x)/(1 + e^x) -> -tanh(x/2)` | ts:639-645 | **A14** `(e^y - 1)/(e^y + 1) -> Tanh (0.5 y)` 9 -> 4. The `e^x, e^-x` spelling has a leaf at depth 4 |
| exptrigsimp sinh/cosh from exp | ts:600-636 | not expressible |

44 sympy rules / table rows examined. 32 candidate rules written (26 in section A, 6 in section B); block checked by script: every pattern/template leaf depth <= 3, no rule grows.

## 2. Candidate rules

Soundness notes that apply throughout:
- None is bit-exact (different libm calls); all are real identities and agree to rounding where both sides are finite.
- Ratio rules: `cos x == 0` never happens for an f64 `x`, and `sin x == 0` / `tan x == 0` happen only at `x == 0`.
  A1/A3 agree at `x == 0` (`0/1 = tan 0`; `1/0` and `Inv 0` are both NaN). A8 does not, hence its guard.
- A4/A5 lose relative accuracy within ~1e-7 of odd multiples of pi (`1 + cos x` cancels); the rhs is the accurate side.
- A11/A12/A13 likewise: the lhs cancels where the result is near 0; absolute error stays ~1e-16.
- A14: for `y > 709` `Exp y` is +inf and the lhs is NaN while `Tanh` gives 1.0 — the rhs is defined on more inputs.
  Written for raw `Exp` only; `ProtectedExp` has the same +inf tail, so a twin is equally (un)sound and is left out.
- No `ProtectedDiv` twins: `ProtectedDiv (Sin x) (Cos x)` is 0 where `|cos x| < 1e-6` while `Tan x` is ~1e6. Unsound.

```egglog
(ruleset sympy_trig)

; ===================== section A — not present in trig.rs / trig_fu.rs =====================

; A2  fu.py:180 TR2i / trigsimp.py:822 matchers_division[0], reciprocal spelling: sin * (1/cos) = tan   6 -> 2
(rewrite (Mul (Sin x) (Inv (Cos x))) (Tan x) :ruleset sympy_trig)
(rewrite (Mul (Inv (Cos x)) (Sin x)) (Tan x) :ruleset sympy_trig)

; A3  fu.py:190 TR2i: cos/sin = 1/tan   5 -> 3, and the reciprocal spelling 6 -> 3
(rewrite (Div (Cos x) (Sin x)) (Inv (Tan x)) :ruleset sympy_trig)
(rewrite (Mul (Cos x) (Inv (Sin x))) (Inv (Tan x)) :ruleset sympy_trig)
(rewrite (Mul (Inv (Sin x)) (Cos x)) (Inv (Tan x)) :ruleset sympy_trig)

; A4  fu.py:185 TR2i half=True: sin x / (1 + cos x) = tan(x/2)   7 -> 4
(rewrite (Div (Sin x) (Add (Num 1.0) (Cos x))) (Tan (Mul (Num 0.5) x)) :ruleset sympy_trig)
(rewrite (Div (Sin x) (Add (Cos x) (Num 1.0))) (Tan (Mul (Num 0.5) x)) :ruleset sympy_trig)

; A5  fu.py:197 TR2i half=True: (1 + cos x) / sin x = 1/tan(x/2)   7 -> 5
(rewrite (Div (Add (Num 1.0) (Cos x)) (Sin x)) (Inv (Tan (Mul (Num 0.5) x))) :ruleset sympy_trig)
(rewrite (Div (Add (Cos x) (Num 1.0)) (Sin x)) (Inv (Tan (Mul (Num 0.5) x))) :ruleset sympy_trig)

; A6  fu.py:134 TR2i integer powers / trigsimp.py:822 with c=2: sin^2/cos^2 = tan^2   7 -> 3
(rewrite (Div (Pow2 (Sin x)) (Pow2 (Cos x))) (Pow2 (Tan x)) :ruleset sympy_trig)

; A7  trigsimp.py:823 matchers_division[1] with c=2: tan^2 * cos^2 = sin^2   7 -> 3
(rewrite (Mul (Pow2 (Tan x)) (Pow2 (Cos x))) (Pow2 (Sin x)) :ruleset sympy_trig)
(rewrite (Mul (Pow2 (Cos x)) (Pow2 (Tan x))) (Pow2 (Sin x)) :ruleset sympy_trig)

; A8  trigsimp.py:825 matchers_division[3]: tan/sin = 1/cos  5 -> 3 ; and sin/tan = cos  5 -> 2
;     guard: at x == 0 the lhs is 0/0 = NaN, the rhs is 1
(rewrite (Div (Tan x) (Sin x)) (Inv (Cos x)) :when ((is-nonzero x)) :ruleset sympy_trig)
(rewrite (Div (Sin x) (Tan x)) (Cos x) :when ((is-nonzero x)) :ruleset sympy_trig)

; A9  fu.py:1329 TR14 / trigsimp.py:828-831 matchers_division[6,7]: (1 - f)(1 + f) = g^2   9 -> 3
(rewrite (Mul (Sub (Num 1.0) (Cos x)) (Add (Num 1.0) (Cos x))) (Pow2 (Sin x)) :ruleset sympy_trig)
(rewrite (Mul (Add (Num 1.0) (Cos x)) (Sub (Num 1.0) (Cos x))) (Pow2 (Sin x)) :ruleset sympy_trig)
(rewrite (Mul (Sub (Num 1.0) (Sin x)) (Add (Num 1.0) (Sin x))) (Pow2 (Cos x)) :ruleset sympy_trig)
(rewrite (Mul (Add (Num 1.0) (Sin x)) (Sub (Num 1.0) (Sin x))) (Pow2 (Cos x)) :ruleset sympy_trig)

; A10 trigsimp.py:847 matchers_add[2]: sin a cos b - cos a sin b = sin(a - b)   11 -> 4
(rewrite (Sub (Mul (Sin a) (Cos b)) (Mul (Cos a) (Sin b))) (Sin (Sub a b)) :ruleset sympy_trig)
; A10 trigsimp.py:848 matchers_add[3]: cos a cos b + sin a sin b = cos(a - b)   11 -> 4
(rewrite (Add (Mul (Cos a) (Cos b)) (Mul (Sin a) (Sin b))) (Cos (Sub a b)) :ruleset sympy_trig)

; A11 trigsimp.py:874 artifacts[0] with a=1 (= fu.py:355 TR5 read right-to-left): 1 - cos^2 = sin^2   5 -> 3
(rewrite (Sub (Num 1.0) (Pow2 (Cos x))) (Pow2 (Sin x)) :ruleset sympy_trig)
; A11 fu.py:376 TR6 read right-to-left: 1 - sin^2 = cos^2   5 -> 3
(rewrite (Sub (Num 1.0) (Pow2 (Sin x))) (Pow2 (Cos x)) :ruleset sympy_trig)

; A12 trigsimp.py:875 artifacts[1] / fu.py:1454 TR22 read right-to-left: cos^-2 - 1 = tan^2   6 -> 3
(rewrite (Sub (Pow (Cos x) (Num -2.0)) (Num 1.0)) (Pow2 (Tan x)) :ruleset sympy_trig)
; A12 fu.py:1389 TR16 read right-to-left, the Add order trig.rs lacks: tan^2 + 1 = 1/cos^2   5 -> 4
(rewrite (Add (Pow2 (Tan x)) (Num 1.0)) (Inv (Pow2 (Cos x))) :ruleset sympy_trig)

; A13 fu.py:770 TR11 read right-to-left, sign-flipped: sin^2 - cos^2 = -cos(2x)   7 -> 5
(rewrite (Sub (Pow2 (Sin x)) (Pow2 (Cos x))) (Neg (Cos (Mul (Num 2.0) x))) :ruleset sympy_trig)

; A14 trigsimp.py:639-645 exptrigsimp: (e^y - 1)/(e^y + 1) = tanh(y/2)   9 -> 4
(rewrite (Div (Sub (Exp y) (Num 1.0)) (Add (Exp y) (Num 1.0))) (Tanh (Mul (Num 0.5) y)) :ruleset sympy_trig)

; ===== section B — one-way CONTRACTING half of identities trig_fu.rs holds as birewrite =====
; (birewrite is a load error in reader.rs and trig_fu is not a linter source; drop this section if unwanted)

; B1  fu.py:180 TR2i / trigsimp.py:822: sin/cos = tan   5 -> 2
(rewrite (Div (Sin x) (Cos x)) (Tan x) :ruleset sympy_trig)
; B2  fu.py:638 TR10i / trigsimp.py:845: sin a cos b + cos a sin b = sin(a + b)   11 -> 4
(rewrite (Add (Mul (Sin a) (Cos b)) (Mul (Cos a) (Sin b))) (Sin (Add a b)) :ruleset sympy_trig)
; B3  fu.py:638 TR10i / trigsimp.py:846: cos a cos b - sin a sin b = cos(a + b)   11 -> 4
(rewrite (Sub (Mul (Cos a) (Cos b)) (Mul (Sin a) (Sin b))) (Cos (Add a b)) :ruleset sympy_trig)
; B4  fu.py:770 TR11 read right-to-left: 2 sin x cos x = sin(2x)   7 -> 4, both inner orders
(rewrite (Mul (Num 2.0) (Mul (Sin x) (Cos x))) (Sin (Mul (Num 2.0) x)) :ruleset sympy_trig)
(rewrite (Mul (Num 2.0) (Mul (Cos x) (Sin x))) (Sin (Mul (Num 2.0) x)) :ruleset sympy_trig)
; B5  fu.py:770 TR11 read right-to-left: cos^2 - sin^2 = cos(2x)   7 -> 4
(rewrite (Sub (Pow2 (Cos x)) (Pow2 (Sin x))) (Cos (Mul (Num 2.0) x)) :ruleset sympy_trig)
```

## 3. Not from sympy — the only idea here that touches nested trig

sympy does not do this (checked, section 0), so it is kept OUT of the block above. The census has
`O (Abs ..)` for odd `O` in {Sin, Tan, Tanh} at weight 509 over 23 genes, and one gene family
`Abs (Sin (Sin (Abs ..)))` (w 30). An odd function of `|x|` is `+-` the function of `x`, exactly (libm sin/tan/tanh are
odd-symmetric), so an EVEN op directly above absorbs the `Abs`. Depth 3 lets only ONE odd layer sit between them:

```text
(rewrite (Cos  (Sin  (Abs x))) (Cos  (Sin  x)) :ruleset sign)     ; 4 -> 3, same for Tan, Tanh
(rewrite (Abs  (Sin  (Abs x))) (Abs  (Sin  x)) :ruleset sign)     ; same for Tan
(rewrite (Pow2 (Sin  (Abs x))) (Pow2 (Sin  x)) :ruleset sign)     ; same for Tan, Tanh
(rewrite (Tanh (Abs x)) (Abs (Tanh x)) :ruleset sign)             ; size-neutral float-out: tanh|x| = |tanh x| exactly;
                                                                  ; shrinks once it meets Abs/Cos/Pow2/ProtectedSqrt/ProtectedLog above
```

Measured: the exact depth-2 shape `Even (Odd (Abs x))` has weight 0 today; the two-layer `Abs (Sin (Sin (Abs x)))`
needs a leaf at depth 4 and cannot be written. `Tanh (Abs ..)` occurs at weight ~240 but its parents in the census are
ProtectedExp / Sqrt / Pow3 / Mul / Add / top-level — none even — so the float-out would shrink nothing today either.
