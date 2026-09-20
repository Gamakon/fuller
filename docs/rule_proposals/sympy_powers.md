# Rules mined from sympy's source — powers, radicals, exp/log, Abs

Status: PROPOSAL. Nothing under `src/` was edited, no cargo was run, nothing is
tested. sympy 1.13.1, paths relative to `site-packages/sympy/`. Every rule was
checked by hand against `src/eval.rs`; the mechanical checker has the last word.

Weights ("w") are `count`-weighted occurrences of the exact left-hand shape in
`fuller_math` of `hff/notebooks/_ledgers/hof_dataset/genes.jsonl` (847 genes),
counted with an s-expression parser.

## What the data says first

- **Literal exponents do not occur in evolved genes** (`(Pow _ (Num _))` w 0 after
  `smallest_form`). Every `Pow` carries a variable exponent. So the literal-folding
  rules (group B, C) pay on the sympy parity corpora and on the output of other
  rules (`sympy_mined`'s `(Pow (Pow x a) b)` leaves `(Pow x (Mul 2.0 0.5))`), not on
  the hall of fame.
- The shapes that DO occur: `(Pow3 (Abs x))` w 150,
  `(Sqrt (Inv x))` w 75, `(Log (Inv x))` w 60, `(ProtectedInv (Abs x))` w 60,
  `(Div (Pow3 x) x)` w 30, `(Mul x (Sqrt x))` w 30, `(Inv (Abs x))` w 30,
  `(Mul (Sqrt a) (Sqrt b))` w 14. `(Div (Exp a) (Exp b))` is w 0 (the 120 for
  `(Div (Exp ..` is an Exp numerator over something else).
- `powers.rs` keeps `x^a * x^b` DISABLED "until exponent constant-folding lands".
  `(Num (+ a b))` in a template IS that fold. Group B is those rules, literal-only,
  each strictly shrinking, so they terminate.

Soundness classes used below:
**exact** = same f64 for every input incl. NaN/inf (sign of zero aside);
**ulp** = equal over the reals, may differ in the last place (the standard already
accepted for `c*(x/c) = x` in `rational`);
**overflow-class** = additionally disagrees only when an intermediate overflows or
underflows (`inf*0`, `inf/inf`), the hole the shipped `(Mul (Exp a) (Exp b))` has;
**-inf** = disagrees only at the input `-inf` (`powf(-inf, 0.5) = +inf`, `Sqrt(-inf) = NaN`).

## 1. sympy rule -> our rule

| # | sympy | file:line | our rule / why not |
|---|---|---|---|
| 1 | `x**0 -> 1`, `x**1 -> x`, `1**x -> 1` | core/power.py:160,162,185 | already in `powers` |
| 2 | `0**x -> 0` for positive x | core/numbers.py:2782 | **F1** `(Pow (Num 0.0) x) -> 0`, is-positive x |
| 3 | `(-c)**even -> c**even`, `(-c)**odd -> -(c**odd)` | core/power.py:177 | not expressible: needs integer parity of a literal. The Pow2/Pow3 cases are in `sign` |
| 4 | `b**(c*n/log(b)) -> E**(c*n)` | core/power.py:198 | not expressible: `(Pow x (Div y (Log x)))` is false at x = 1 (log = 0) and there is no `x != 1` guard |
| 5 | `(b**e)**other -> b**(e*other)`, other integer | core/power.py:256 | **C2 C3** `Pow2/Pow3` of `(Pow x (Num n))`. Real domain needs is-nonneg x: `(x^0.5)^2` is NaN for x < 0, `x^1` is x |
| 6 | same, `b.is_extended_nonnegative` | core/power.py:305 | **C1 C4 C5** `Sqrt` of / `Pow` of `Sqrt` / literal `(Pow (Pow x a) b)`, is-nonneg x. The non-literal form is already `sympy_mined` (size-neutral, leaves an unfolded `Mul`) |
| 7 | `(1/b)**other -> b**-other`, b not negative or other integer | core/power.py:289-297 | **C6 C7** literal exponent with premise `n > 0` (x = 0, n < 0: NaN vs 0); **E1** `(Sqrt (Inv x)) -> (Inv (Sqrt x))`. `Pow2/Pow3 (Inv x)` is the same branch — already claimed as an enabler in rational.md:274, not repeated |
| 8 | `(b**even)**e -> Abs(b)**(even*e)` | core/power.py:298 | not expressible usefully: `(Pow (Pow2 x) e)` is w 655 but e is a variable, and `(Pow (Abs x) (Mul 2 e))` GROWS 4 -> 6 |
| 9 | `sqrt(x) = Pow(x, 1/2)` | functions/elementary/miscellaneous.py:67 | **A1 A2** `(Pow x 0.5) -> (Sqrt x)`, `(Pow x -0.5) -> (Inv (Sqrt x))` |
| 10 | Mul.flatten gathers exponents of a common base (`x*x**2 -> x**3`, `x*sqrt(x) -> x**(3/2)`) | core/mul.py:464 | **B5-B14** on the dedicated constructors |
| 11 | powsimp `combine='exp'`: `x**a * x**b -> x**(a+b)` | simplify/powsimp.py:160 | **B1** (and B3 B4 B7 B8) literal exponents (the rules `powers.rs` keeps disabled), **B15 B16** general exponents, is-positive x |
| 12 | powsimp base / inverted-base pairs: `x**a * (1/x)**b -> x**(a-b)`, positive x | simplify/powsimp.py:182 | **B3 B4 B16** (Div spelling), **D1** `(Div (Exp a) (Exp b))` |
| 13 | powsimp base / negated-base pairs | simplify/powsimp.py:200 | not expressible: needs integer-exponent test and re-signs the base |
| 14 | powsimp `combine='base'`: `a**c * b**c -> (a*b)**c`, integer c or nonneg bases; "a single unk can join the rest" | simplify/powsimp.py:376,420-436 | **G1-G9**. sympy's "single unknown joins" is false at a = 0, b < 0 (`0*NaN` vs `sqrt(-0.0)`), so the one-sided guard here is is-POSITIVE |
| 15 | powsimp `2**(2*x) -> 4**x` | simplify/powsimp.py:388 | not expressible: needs `c^k` on literals (NUM has only neg, *, +) |
| 16 | powdenest `_denest_pow`: `(x**a)**b`, `exp(a)**b` | simplify/powsimp.py:613-620 | = rows 5-6. `(Pow (Exp x) y) -> (Exp (Mul x y))` is size-neutral and REJECTED in powers.md R7 |
| 17 | powdenest exp-with-logs `exp(c*log x) -> x**c` | simplify/powsimp.py:623-633 | already `sympy_mined` |
| 18 | `exp._eval_power`: `exp(x)**e -> exp(x*e)` | functions/elementary/exponential.py:123 | REJECTED powers.md R1/R7 (`Pow2 (Exp x)`, `Sqrt (Exp x)`: grows, overflow) |
| 19 | `exp(log x) -> x` | exponential.py:298 | already in `powers` |
| 20 | `exp(c*log x) -> x**c` (single log factor) | exponential.py:341-354 | c literal/any: `sympy_mined`. **D2** the c = -1 case spelled `Neg`: `(Exp (Neg (Log x))) -> (Inv x)` |
| 21 | `exp(log(x) + y) -> x*exp(y)` | exponential.py:356-377 | **D3-D6** incl. the `Sub` spellings, is-positive x |
| 22 | `exp(0) -> 1`, `log(1) -> 0` | exponential.py:289, 693 | already in `algebra` |
| 23 | `Pow(E, x)` is `exp(x)` | exponential.py:284, core/power.py:196 | **A3** `(Pow (Num 2.718281828459045) x) -> (Exp x)` |
| 24 | `log(exp x) -> x`, x real | exponential.py:706 | already in `powers`. Protected spellings REJECTED powers.md R2 |
| 25 | `log(1/q) -> -log(q)` | exponential.py:701 | **E2** `(Log (Inv x)) -> (Neg (Log x))`, no guard needed in the real domain |
| 26 | expand_log: `log(a*b)`, `log(x**n)`, `log(p/q)` | exponential.py:833-861 | already in `powers` / `sympy_mined` (guarded) |
| 27 | logcombine: `log a - log b -> log(a/b)` | simplify/simplify.py:973 | **D7**, is-positive both. The `Add -> Log (Mul ..)` twin is NOT proposed: it is the exact reverse of the shipped `powers` rule and the pair would cycle |
| 28 | `log(sqrt x)`, `log(x**2)` | exponential.py:853 | not expressible: `(Mul (Num 0.5) (Log x))` grows 3 -> 4 |
| 29 | `Abs(a*b) -> Abs(a)*Abs(b)`, `Abs(n/d)` | functions/elementary/complexes.py:537-556 | sympy's direction GROWS (it is in `sympy_mined`, unloaded). The shrinking direction is sympy's `collect_abs`, row 35 |
| 30 | `Abs(x**even) -> x**even` | complexes.py:568 | already (`Abs (Pow2 x)`, is-nonneg) — general parity not expressible |
| 31 | `Abs(x**n) -> Abs(x)**n`, integer n | complexes.py:572 | **E3-E5** in the OPPOSITE direction (Abs floats rootward, where `Pow2/Cos/ProtectedSqrt/ProtectedLog/Abs` absorb it). Size-neutral, exact |
| 32 | `Abs(base**e) -> base**e`, base nonneg | complexes.py:573 | already: `is-nonneg (Pow b p)` guard + `(Abs x) -> x` |
| 33 | `Abs(exp x) -> exp x`; Abs of nonneg arg | complexes.py:583, 596 | already (`sympy_mined`, is-nonneg) |
| 34 | `Abs(x)**even -> x**even`; `Abs(x)**n -> x**(n-1)*Abs(x)` | complexes.py:651-656 | even: already (`Pow x 2 -> Pow2`, `Pow2 (Abs x) -> Pow2 x`). odd: grows |
| 35 | `collect_abs`: `Abs(a)*Abs(b) -> Abs(a*b)` | simplify/radsimp.py:583 | **H1-H3** incl. `ProtectedDiv` (threshold tests `|b|`, unchanged by Abs). Exact |
| 36 | `collect_sqrt` | simplify/radsimp.py:504 | = general factoring `a*c + b*c -> (a+b)*c`, already in `wide`; nothing sqrt-specific survives |
| 37 | `radsimp` / `rad_rationalize`: multiply by the conjugate | simplify/radsimp.py:768, 1021 | not expressible: every step GROWS (`1/(a+sqrt b) -> (a-sqrt b)/(a^2-b)`). The only shrinking residue is **B11 B12** (`sqrt(x)/x`, `x/sqrt(x)`) |
| 38 | radsimp `1/d**i -> (1/d)**i` | simplify/radsimp.py:914 | reverse of row 7; not proposed (pushes Inv leafward) |
| 39 | `fraction`, `numer`, `denom` | simplify/radsimp.py:1043 | not a rewrite (structural accessor) |
| 40 | `sqrtdenest` numeric / biquadratic / `_denester` | simplify/sqrtdenest.py:443, 463, 537 | not expressible: needs `sqrt(a^2 - b^2 r)` to be a rational square — a test and a sqrt on literals |
| 41 | `_sqrt_symbolic_denest` | simplify/sqrtdenest.py:380 | not expressible: depth > 3 and a discriminant test. Its only in-grammar instance is the shipped `(Sqrt (Pow2 x)) -> (Abs x)` |
| 42 | `Mul._eval_power`: `(a*b)**n -> a**n*b**n` | core/mul.py:719 | grows; the reverse is **G6-G9** |

42 sympy rules examined; 17 already present, 11 not expressible or previously
rejected, 14 yield the 62 candidate rules below (operand orders counted separately).

## 2. Notes the checker cannot see

- **A1** `-inf` class. **A3** ulp (`powf(e, x)` vs `exp(x)`).
- **B1-B4, B15, B16** overflow-class (x = 1e200, a = 2, b = -2). is-nonneg is NOT
  enough: `0^1 * 0^-1` is NaN, `0^0` is 1.
- **B5** is exact (`Pow3` evaluates `(a*a)*a`, the same operations); B6-B10, B13, B14 ulp.
- **B9 B10 B13** unguarded: at x = 0 both sides are NaN. Overflow-class at |x| > 1e154.
- **B7** on its own is sound only with `n > 0` unguarded (x = 0, n = -1: NaN vs `0^0 = 1`),
  hence a `rule` with the premise; the is-nonzero `rewrite` covers n <= 0.
- **C*** with is-nonneg: at x = 0 every sign combination of the exponents agrees
  (`0^neg = NaN` propagates through the outer op; `NaN^0 = 1 = 0^0`).
- **G1** vs **G2/G3**: both-nonneg, or ONE operand strictly positive (then the other
  may be anything: b < 0 gives NaN on both sides). Overflow-class at 1e200 * 1e200.
- **G6-G9** overflow-class (`a^2` overflows where `(a*b)^2` does not).
- **E3-E5, H1-H3** exact. E4 is the reverse of `sympy_mined`'s `(Abs (Inv x)) -> (Inv (Abs x))`;
  `sympy_mined` is loaded nowhere, but the two must never be co-loaded. Same for
  G1-G3 vs `sympy_mined`'s `(Sqrt (Mul a b))` split, and H1/H2 vs its `Abs` splits.
- Deliberately absent: every `ProtectedExp/ProtectedLog` collapse (owner's decision,
  `identities.rs::protected_ops_are_inert`, powers.md R2); `ProtectedSqrt` joins
  (`ProtectedSqrt(1e200*1e200) = 0.0` vs 1e200, and w 0); `(ProtectedDiv (Pow2 a) (Pow2 b))`
  (threshold moves from `|b|` to `b^2`); `ProtectedLog (Inv x)` (x = inf: `+inf` vs `-inf`).

## 3. Candidates

```egglog
(ruleset sympy_powers)

; ===== A. Pow with a special literal -> dedicated constructor =====
; A1  x**(1/2) is sqrt(x)   functions/elementary/miscellaneous.py:67
(rewrite (Pow x (Num 0.5)) (Sqrt x) :ruleset sympy_powers)
; A2  x**(-1/2) = 1/sqrt(x)   functions/elementary/miscellaneous.py:67 + core/power.py:296
(rewrite (Pow x (Num -0.5)) (Inv (Sqrt x)) :ruleset sympy_powers)
; A3  Pow(E, x) is exp(x)   functions/elementary/exponential.py:284
(rewrite (Pow (Num 2.718281828459045) x) (Exp x) :ruleset sympy_powers)

; ===== B. common base: add the exponents =====
; B1  x**a * x**b = x**(a+b)   simplify/powsimp.py:160
(rewrite (Mul (Pow x (Num a)) (Pow x (Num b))) (Pow x (Num (+ a b)))
    :when ((is-positive x)) :ruleset sympy_powers)
; B3  x**a / x**b = x**(a-b)   simplify/powsimp.py:182
(rewrite (Div (Pow x (Num a)) (Pow x (Num b))) (Pow x (Num (+ a (neg b))))
    :when ((is-positive x)) :ruleset sympy_powers)
; B4  x**a / x = x**(a-1) ; x / x**a = x**(1-a)   simplify/powsimp.py:182
(rewrite (Div (Pow x (Num a)) x) (Pow x (Num (+ a -1.0)))
    :when ((is-nonzero x)) :ruleset sympy_powers)
(rewrite (Div x (Pow x (Num a))) (Pow x (Num (+ 1.0 (neg a))))
    :when ((is-nonzero x)) :ruleset sympy_powers)
; B7  x * x**n = x**(n+1), n > 0 needs no guard   core/mul.py:464
(rule ((= e (Mul x (Pow x (Num n)))) (> n 0.0))
      ((union e (Pow x (Num (+ n 1.0))))) :ruleset sympy_powers)
(rule ((= e (Mul (Pow x (Num n)) x)) (> n 0.0))
      ((union e (Pow x (Num (+ n 1.0))))) :ruleset sympy_powers)
; B8  x * x**n = x**(n+1), any n, x != 0   core/mul.py:464
(rewrite (Mul x (Pow x (Num n))) (Pow x (Num (+ n 1.0)))
    :when ((is-nonzero x)) :ruleset sympy_powers)
(rewrite (Mul (Pow x (Num n)) x) (Pow x (Num (+ n 1.0)))
    :when ((is-nonzero x)) :ruleset sympy_powers)
; B5  x**2 * x = x**3   core/mul.py:464
(rewrite (Mul (Pow2 x) x) (Pow3 x) :ruleset sympy_powers)
(rewrite (Mul x (Pow2 x)) (Pow3 x) :ruleset sympy_powers)
; B6  x**3 * x = x**4 ; x**2 * x**3 = x**5   core/mul.py:464
(rewrite (Mul (Pow3 x) x) (Pow2 (Pow2 x)) :ruleset sympy_powers)
(rewrite (Mul x (Pow3 x)) (Pow2 (Pow2 x)) :ruleset sympy_powers)
(rewrite (Mul (Pow2 x) (Pow3 x)) (Pow x (Num 5.0)) :ruleset sympy_powers)
(rewrite (Mul (Pow3 x) (Pow2 x)) (Pow x (Num 5.0)) :ruleset sympy_powers)
; B9  x / x**2 = 1/x ; x**2 / x**3 = 1/x   core/mul.py:464
(rewrite (Div x (Pow2 x)) (Inv x) :ruleset sympy_powers)
(rewrite (Div (Pow2 x) (Pow3 x)) (Inv x) :ruleset sympy_powers)
; B10  x / x**3 = 1/x**2   core/mul.py:464
(rewrite (Div x (Pow3 x)) (Inv (Pow2 x)) :ruleset sympy_powers)
; B14  x**3 / x = x**2 ; x**2 / x = x ; x**3 / x**2 = x   core/mul.py:464
(rewrite (Div (Pow3 x) x) (Pow2 x) :when ((is-nonzero x)) :ruleset sympy_powers)
(rewrite (Div (Pow2 x) x) x :when ((is-nonzero x)) :ruleset sympy_powers)
(rewrite (Div (Pow3 x) (Pow2 x)) x :when ((is-nonzero x)) :ruleset sympy_powers)
; B11  sqrt(x)/x = 1/sqrt(x)   core/mul.py:464 (simplify/radsimp.py:891 residue)
(rewrite (Div (Sqrt x) x) (Inv (Sqrt x)) :ruleset sympy_powers)
; B12  x/sqrt(x) = sqrt(x)   core/mul.py:464
(rewrite (Div x (Sqrt x)) (Sqrt x) :when ((is-nonzero x)) :ruleset sympy_powers)
; B13  x*sqrt(x) = x**(3/2)   core/mul.py:464
(rewrite (Mul x (Sqrt x)) (Pow x (Num 1.5)) :ruleset sympy_powers)
(rewrite (Mul (Sqrt x) x) (Pow x (Num 1.5)) :ruleset sympy_powers)
; B15  x**a * x**b = x**(a+b), any exponents   simplify/powsimp.py:160
(rewrite (Mul (Pow x a) (Pow x b)) (Pow x (Add a b))
    :when ((is-positive x)) :ruleset sympy_powers)
; B16  x**a / x**b = x**(a-b), any exponents   simplify/powsimp.py:182
(rewrite (Div (Pow x a) (Pow x b)) (Pow x (Sub a b))
    :when ((is-positive x)) :ruleset sympy_powers)

; ===== C. power of a power: multiply the exponents (literal fold) =====
; C1  sqrt(x**n) = x**(n/2), x >= 0   core/power.py:305
(rewrite (Sqrt (Pow x (Num n))) (Pow x (Num (* 0.5 n)))
    :when ((is-nonneg x)) :ruleset sympy_powers)
; C2  (x**n)**2 = x**(2n), x >= 0   core/power.py:256
(rewrite (Pow2 (Pow x (Num n))) (Pow x (Num (* 2.0 n)))
    :when ((is-nonneg x)) :ruleset sympy_powers)
; C3  (x**n)**3 = x**(3n), x >= 0   core/power.py:256
(rewrite (Pow3 (Pow x (Num n))) (Pow x (Num (* 3.0 n)))
    :when ((is-nonneg x)) :ruleset sympy_powers)
; C4  sqrt(x)**n = x**(n/2), x >= 0   core/power.py:305
(rewrite (Pow (Sqrt x) (Num n)) (Pow x (Num (* 0.5 n)))
    :when ((is-nonneg x)) :ruleset sympy_powers)
; C5  (x**a)**b = x**(a*b), x >= 0   core/power.py:305, simplify/powsimp.py:613
(rewrite (Pow (Pow x (Num a)) (Num b)) (Pow x (Num (* a b)))
    :when ((is-nonneg x)) :ruleset sympy_powers)
; C6  1/x**n = x**(-n), n > 0   core/power.py:296
(rule ((= e (Inv (Pow x (Num n)))) (> n 0.0))
      ((union e (Pow x (Num (neg n))))) :ruleset sympy_powers)
; C7  (1/x)**n = x**(-n), n > 0   core/power.py:289-297
(rule ((= e (Pow (Inv x) (Num n))) (> n 0.0))
      ((union e (Pow x (Num (neg n))))) :ruleset sympy_powers)
; C8  sqrt(x)**2 = x, widened from is-positive to x >= 0   core/power.py:305
(rewrite (Pow2 (Sqrt p)) p :when ((is-nonneg p)) :ruleset sympy_powers)

; ===== D. exp / log =====
; D1  exp(a)/exp(b) = exp(a-b)   simplify/powsimp.py:182
(rewrite (Div (Exp a) (Exp b)) (Exp (Sub a b)) :ruleset sympy_powers)
; D2  exp(-log x) = 1/x, x > 0   functions/elementary/exponential.py:341-354
(rewrite (Exp (Neg (Log x))) (Inv x) :when ((is-positive x)) :ruleset sympy_powers)
; D3 D4  exp(log(x) + y) = x*exp(y), x > 0   functions/elementary/exponential.py:356-377
(rewrite (Exp (Add (Log x) y)) (Mul x (Exp y)) :when ((is-positive x)) :ruleset sympy_powers)
(rewrite (Exp (Add y (Log x))) (Mul x (Exp y)) :when ((is-positive x)) :ruleset sympy_powers)
; D5  exp(y - log x) = exp(y)/x, x > 0   functions/elementary/exponential.py:356-377
(rewrite (Exp (Sub y (Log x))) (Div (Exp y) x) :when ((is-positive x)) :ruleset sympy_powers)
; D6  exp(log(x) - y) = x/exp(y), x > 0   functions/elementary/exponential.py:356-377
(rewrite (Exp (Sub (Log x) y)) (Div x (Exp y)) :when ((is-positive x)) :ruleset sympy_powers)
; D7  log(a) - log(b) = log(a/b), a > 0, b > 0   simplify/simplify.py:973 (logcombine)
(rewrite (Sub (Log a) (Log b)) (Log (Div a b))
    :when ((is-positive a) (is-positive b)) :ruleset sympy_powers)

; ===== E. size-neutral normalisers: move Inv / Neg / Abs one level ROOTWARD =====
; E1  sqrt(1/x) = 1/sqrt(x)   core/power.py:296
(rewrite (Sqrt (Inv x)) (Inv (Sqrt x)) :ruleset sympy_powers)
; E2  log(1/x) = -log(x)   functions/elementary/exponential.py:701
(rewrite (Log (Inv x)) (Neg (Log x)) :ruleset sympy_powers)
; E3  |x|**3 = |x**3|   functions/elementary/complexes.py:572 (reversed)
(rewrite (Pow3 (Abs x)) (Abs (Pow3 x)) :ruleset sympy_powers)
; E4  1/|x| = |1/x|   functions/elementary/complexes.py:543 (reversed)
(rewrite (Inv (Abs x)) (Abs (Inv x)) :ruleset sympy_powers)
; E5  protected_inv(|x|) = |protected_inv(x)| (both 1 at x = 0)   functions/elementary/complexes.py:543 (reversed)
(rewrite (ProtectedInv (Abs x)) (Abs (ProtectedInv x)) :ruleset sympy_powers)

; ===== F. zero base =====
; F1  0**x = 0, x > 0   core/numbers.py:2782
(rewrite (Pow (Num 0.0) x) (Num 0.0) :when ((is-positive x)) :ruleset sympy_powers)

; ===== G. common exponent: join the bases =====
; G1  sqrt(a)*sqrt(b) = sqrt(a*b), both >= 0   simplify/powsimp.py:432
(rewrite (Mul (Sqrt a) (Sqrt b)) (Sqrt (Mul a b))
    :when ((is-nonneg a) (is-nonneg b)) :ruleset sympy_powers)
; G2 G3  same, ONE operand > 0 and the other unknown   simplify/powsimp.py:436
(rewrite (Mul (Sqrt a) (Sqrt b)) (Sqrt (Mul a b)) :when ((is-positive a)) :ruleset sympy_powers)
(rewrite (Mul (Sqrt a) (Sqrt b)) (Sqrt (Mul a b)) :when ((is-positive b)) :ruleset sympy_powers)
; G4  sqrt(a)/sqrt(b) = sqrt(a/b), both >= 0   simplify/powsimp.py:432
(rewrite (Div (Sqrt a) (Sqrt b)) (Sqrt (Div a b))
    :when ((is-nonneg a) (is-nonneg b)) :ruleset sympy_powers)
; G5  a**c * b**c = (a*b)**c, both >= 0   simplify/powsimp.py:432
(rewrite (Mul (Pow a c) (Pow b c)) (Pow (Mul a b) c)
    :when ((is-nonneg a) (is-nonneg b)) :ruleset sympy_powers)
; G6 G7  a**2 * b**2 = (a*b)**2 ; a**2 / b**2 = (a/b)**2   simplify/powsimp.py:422 (integer exponent)
(rewrite (Mul (Pow2 a) (Pow2 b)) (Pow2 (Mul a b)) :ruleset sympy_powers)
(rewrite (Div (Pow2 a) (Pow2 b)) (Pow2 (Div a b)) :ruleset sympy_powers)
; G8 G9  a**3 * b**3 = (a*b)**3 ; a**3 / b**3 = (a/b)**3   simplify/powsimp.py:422 (integer exponent)
(rewrite (Mul (Pow3 a) (Pow3 b)) (Pow3 (Mul a b)) :ruleset sympy_powers)
(rewrite (Div (Pow3 a) (Pow3 b)) (Pow3 (Div a b)) :ruleset sympy_powers)

; ===== H. collect_abs =====
; H1  |a|*|b| = |a*b|   simplify/radsimp.py:583
(rewrite (Mul (Abs a) (Abs b)) (Abs (Mul a b)) :ruleset sympy_powers)
; H2  |a|/|b| = |a/b|   simplify/radsimp.py:583
(rewrite (Div (Abs a) (Abs b)) (Abs (Div a b)) :ruleset sympy_powers)
; H3  protected_div(|a|, |b|) = |protected_div(a, b)| (threshold reads |b|)   simplify/radsimp.py:583
(rewrite (ProtectedDiv (Abs a) (Abs b)) (Abs (ProtectedDiv a b)) :ruleset sympy_powers)
```
