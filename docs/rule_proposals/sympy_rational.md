# Rules mined from sympy's source — `simplify`, `cancel`/`together`/`factor_terms`, `Add.flatten`, `Mul.flatten`

sympy 1.13.1, `/opt/anaconda3/lib/python3.12/site-packages/sympy`. Paths below are relative to that directory.
Read from source, not from memory. Nothing here was compiled or run through cargo. The block of 85 rules was
checked by a throwaway Python script (not kept in the repo) for: every form being `ruleset`/`rewrite`/`rule` in the
`src/lint/reader.rs` grammar, NUM using only `neg * +`, arity, depth ≤ 3, template metavariables bound, strict
node shrink with metavariables at size 1, and numeric agreement under `eval.rs` semantics (Div-by-0 NaN,
`ProtectedDiv` band) at the grid {0, ±1e-7, 1, −2.5, 3}^n plus 200 random points per rule, honouring each rule's
guard. 85/85 passed. Not checked: egglog itself loading the text.

## 1. What `simplify` runs, in order (`simplify/simplify.py`)

| # | line | pass | note |
|---|---|---|---|
| 0 | 588 | `sympify(expr, rational=rational)` | default `rational=False`: Floats stay Floats |
| 1 | 596–597 | `expr.is_zero` early-out | |
| 2 | 599–601 | `_eval_simplify` dispatch | class-specific simplifiers return here |
| 3 | 603 | `collect_abs(signsimp(expr))` | `signsimp` = 349; `sub_pre`/`sub_post` = `simplify/cse_opts.py:10,41` |
| 4 | 608–609 | `inversecombine` | only when `inverse=True`; body 1139–1151 |
| 5 | 615–624 | recursive `simplify` of the args of every node that is not Add/Mul/Pow/exp | |
| 6 | 628–629 | `nc_simplify` | non-commutative only |
| 7 | 638–640 | `nsimplify(expr, rational=True)` | SKIPPED at the default `rational=False` |
| 8 | 642 | bottom-up `.normal()` | |
| 9 | 643 | `Mul(*powsimp(expr).as_content_primitive())` | |
| 10 | 644 | `_e = cancel(expr)` | `polys/polytools.py:7125` |
| 11 | 645 | `expr1 = shorter(_e, _mexpand(_e).cancel())` | |
| 12 | 646 | `expr2 = shorter(together(expr, deep=True), together(expr1, deep=True))` | `polys/rationaltools.py:11` |
| 13 | 647–650 | `expr = shorter(expr2, expr1, expr)` | |
| 14 | 655 | `factor_terms(expr, sign=False)` | `core/exprtools.py:1156`, via `gcd_terms` 980 |
| 15 | 658–659 | `rewrite(Abs)` if `sign` present | |
| 16 | 662–687 | Piecewise fold / simplify | |
| 17 | 690 | `hyperexpand` | |
| 18 | 692–696 | `kroneckersimp`, `besselsimp` | |
| 19 | 698–699 | `trigsimp(expr, deep=True)` | |
| 20 | 701–702 | `shorter(expand_log(expr, deep=True), logcombine(expr))` | `logcombine` = 973 |
| 21 | 704–722 | `combsimp`, `sum_simplify`, Integral `factor_terms`, `product_simplify`, `quantity_simplify` | |
| 22 | 725 | `short = shorter(powsimp(expr, combine='exp', deep=True), powsimp(expr), expr)` | |
| 23 | 726 | `short = shorter(short, cancel(short))` | |
| 24 | 727 | `short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))` | |
| 25 | 728–729 | `exptrigsimp` | |
| 26 | 732–740 | `hollow_mul`: `c*(a+b)` 2-arg Mul re-flattened, i.e. the number is distributed | |
| 27 | 742–746 | `radsimp(1/denom, symbolic=False, max_terms=1)` when the denominator is an Add | |
| 28 | 748–751 | `could_extract_minus_sign` → `signsimp(-n/(-d))` | |
| 29 | 753–754 | `measure(expr) > ratio*measure(original)` → revert to the original | ratio 1.7 |
| 30 | 757–758 | `nfloat` (only if floats were rationalised) | |
| 31 | 760 | `done`: `doit()` then `shorter(rv, collect_abs(rv))` | |

The point for us: **none of the literal collection is a `simplify` pass.** With Floats, pass 7 is skipped, and
every `3.0 + X + 4.0 -> X + 7.0`, `x*x -> x**2`, `x/x -> 1`, `2*x + 3*x -> 5*x` happens on CONSTRUCTION, in
`Add.flatten` (`core/add.py:182`) and `Mul.flatten` (`core/mul.py:196`), because sympy's Add and Mul are n-ary
and unordered. Our binary ordered tree has to spell that out one operand order at a time — that is what §3 is.
`ratsimp` (`simplify/ratsimp.py:9–29`) is `cancel` + polynomial division `reduced(f, [g])` + `cancel(r/g)`.

## 2. sympy behaviour → our rule

| sympy behaviour | source | ours |
|---|---|---|
| numeric terms of an Add accumulate into one `coeff` | `core/add.py:241–251` | L1–L14: every Add/Sub × literal-left/right × inner Add/Sub order. 2 of 16 exist (`rational.rs`), 14 new |
| nested Add is spliced into the parent (`seq.extend(o.args)`) | `core/add.py:274–277` | not expressible as such (associativity); L1–L14 are its directed literal-only shadow |
| numeric factors of a Mul accumulate into one `coeff` | `core/mul.py:358–363` | M1–M2 (the two missing Mul orders), M3–M6 (through a literal-numerator quotient, raw and protected), M7 (nested literal divisors), M8–M9 (literal over a literal divisor) |
| same, where it needs `a/b` on literals | `core/mul.py:358–363` | **not expressible**: `(Div (Mul (Num a) x) (Num b))`, `(Div (Num a) (Mul (Num b) x))`, `(Mul (Num a) (Div x (Num b)))`, `(Div (Div (Num a) x) (Num b))` — no literal division primitive. Only the `a == b` case exists (`rational.rs`) |
| `-x` is `Mul(-1, x)`, so `-1` joins `coeff` | `core/mul.py:358–363`, `simplify.py:405` (signsimp `-(-x)`) | N1–N10: a literal absorbs a `Neg` (−1 node); raw and protected divide (`\|−k\| = \|k\|`, same branch) |
| like terms: `terms[s] += c` | `core/add.py:280–281, 304–310, 335` | T1–T11: `a*x + b*x`, `a*x + x`, `a*x − x`, `x − a*x`, orders with the literal on the left of the Mul plus the right-literal `a*x ± b*x` forms |
| `x + x -> 2*x` | `core/add.py:304–305` | not proposed: 3 → 3 nodes, already declined in `rational.md` R8. `Add x (Add x x)` is R8 |
| `x − x -> 0`, `(a+b) − b -> a` | `core/add.py:318–319` (`c.is_zero` dropped) | exist in `identities.rs`; the four missing shapes are C1–C4 |
| `x*x -> x**2`, `x**2*x -> x**3` (`_gather` of common bases) | `core/mul.py:465–480, 508–534` | `Mul x x` exists; `Mul (Pow2 x) x` is the powers miner's (census A11 note) |
| `x**a / x**b -> x**(a−b)`; `x**1 -> x`; `x**0 -> 1` | `core/mul.py:465–480, 518` | P1–P3 guarded, P4–P6 UNGUARDED (both sides NaN at 0) |
| `cancel`: common factor of numerator and denominator | `polys/polytools.py:7125`, `F.cancel(G)` at 7212; also `_gather` when the factor is syntactically equal | K1–K10, all `:when ((is-nonzero ..))` on the cancelled factor |
| `cancel` of a polynomial gcd that is not a syntactic factor, e.g. `(x**2−1)/(x−1)` | `polys/polytools.py:7212` | not expressible: needs polynomial gcd, and the pattern is past depth 3 |
| any factor cancel through `ProtectedDiv` | — | **not proposed**: `rational.md` rejected the whole class (the `\|b\| < 1e-6` band: `ProtectedDiv (Pow2 x) x` is 0 there, `x` is not) |
| `0 * x -> 0`, so `0/x -> 0` | `core/mul.py:684–692` | Z1 guarded (raw `0/0` is NaN). Protected `0/x` rejected in census Tier C (NaN x) |
| `together`: terms over one denominator (`gcd_terms`) | `polys/rationaltools.py:66–67`, `core/exprtools.py:980` | G1–G4: SAME denominator only, 7 → 5 nodes, raw and protected. Different denominators (`1/a + 1/b -> (a+b)/(a*b)`) grow: not admissible |
| `factor_terms`: common factor out of a sum | `core/exprtools.py:1156`; `simplify.py:655, 727` | F1–F8, directed, shrinks by `1 + size(a)`. NOTE: the reverse of `distribute`; `wide.rs` holds the bidirectional pair. Use only where distribute is not loaded |
| `hollow_mul` / 2-arg Mul distributes a Number over an Add | `simplify.py:732–740`, `core/mul.py:280–296` | not admissible: grows the term (it is `distribute.rs`) |
| `radsimp` of an Add denominator | `simplify.py:742–746` | not admissible: grows |
| `ratsimp` polynomial division | `simplify/ratsimp.py:24–29` | not expressible beyond K/P rules: needs `reduced()` |
| `signsimp` / `sub_pre`: `y − x -> −(x − y)` by canonical ORDER | `simplify.py:349–418`, `cse_opts.py:10–38` | not expressible: the choice depends on sympy's sort key. The order-free part is already `sign.rs` |
| final `−n/−d` | `simplify.py:748–751` | exists (`sign.rs` `Div (Neg a) (Neg b)`) |
| `logcombine`: `log a + log b -> log(a*b)`, `log a − log b -> log(a/b)`, `n*log x -> log(x**n)` | `simplify.py:1019–1112` (same-coefficient 1090–1093, opposite sign 1096–1108, `gooda`/`goodlog` positivity 1023–1032) | LG1–LG3 under `is-positive`. NOTE: reverses the expanding `powers.rs` `Log (Mul a b)` rule; sympy keeps whichever is shorter (702). Use only where that expander is not loaded |
| `inversecombine`: `log(exp x) -> x`, `exp(log x) -> x` | `simplify.py:1139–1151` | exist (`powers.rs`) |
| `exp(a)/exp(b) -> exp(a−b)` (`_gather` on base E) | `core/mul.py:465–480` | E1, −1 node; companion of the existing `Mul (Exp a) (Exp b)` |
| `nsimplify`, `besselsimp`, `sum_simplify`, Piecewise, hyperexpand | — | out of scope |

Soundness notes common to the block:
- Literal folds are exact over the reals and move the last ulp in f64 (`(x + a) − b` vs `x + (a − b)`), the same
  licence `rational.rs` R5 already takes. They can differ at overflow of `a*b` / `a+b`; not at any finite sample.
- Like-term and factor rules differ only where an intermediate is ±inf (`a*x − b*x` is `inf − inf = NaN`,
  `(a−b)*x` is inf).
- M3–M6 are sound for `ProtectedDiv`: in the band the quotient is 0 and `a * 0 = 0` for a finite literal `a`.
  M7 is RAW only: `ProtectedDiv (ProtectedDiv x a) b` with `|a|,|b| >= 1e-6` but `|a*b| < 1e-6` breaks.
  M8–M9 need `b != 0` (`a/(x/0)` is NaN, `(a*0)/x` is 0), written as the two-sided literal test `rational.rs` uses.
- G1–G4 are sound for `ProtectedDiv`: both quotients share `c`, so both take the same branch (`0 + 0 = 0`).
- P4–P6 need no guard: `x/(x*x)` and `1/x` are both NaN at `x = 0`.
- Every guarded rule needs the fact on the CANCELLED factor. `rational.md` finding stands: a bare variable is
  never assumed nonzero, so on real models these fire only when the caller passes `nonzero_vars`.

## 3. Candidate rules

```egglog
(ruleset sympy_rational)

; ---- L: literals of a sum collect into one coefficient -------------------------------
; core/add.py:241-251 (coeff += o), 274-277 (nested Add spliced). a = outer literal, b = inner.
; L1  (b + q) + a
(rewrite (Add (Add (Num b) q) (Num a)) (Add (Num (+ a b)) q) :ruleset sympy_rational)
; L2  a + (q + b)
(rewrite (Add (Num a) (Add q (Num b))) (Add q (Num (+ a b))) :ruleset sympy_rational)
; L3  a + (b - q)
(rewrite (Add (Num a) (Sub (Num b) q)) (Sub (Num (+ a b)) q) :ruleset sympy_rational)
; L4  a + (q - b)
(rewrite (Add (Num a) (Sub q (Num b))) (Add q (Num (+ a (neg b)))) :ruleset sympy_rational)
; L5  (b - q) + a
(rewrite (Add (Sub (Num b) q) (Num a)) (Sub (Num (+ a b)) q) :ruleset sympy_rational)
; L6  (q - b) + a
(rewrite (Add (Sub q (Num b)) (Num a)) (Add q (Num (+ a (neg b)))) :ruleset sympy_rational)
; L7  a - (b + q)
(rewrite (Sub (Num a) (Add (Num b) q)) (Sub (Num (+ a (neg b))) q) :ruleset sympy_rational)
; L8  a - (q + b)
(rewrite (Sub (Num a) (Add q (Num b))) (Sub (Num (+ a (neg b))) q) :ruleset sympy_rational)
; L9  a - (b - q)
(rewrite (Sub (Num a) (Sub (Num b) q)) (Add (Num (+ a (neg b))) q) :ruleset sympy_rational)
; L10 a - (q - b)
(rewrite (Sub (Num a) (Sub q (Num b))) (Sub (Num (+ a b)) q) :ruleset sympy_rational)
; L11 (b + q) - a
(rewrite (Sub (Add (Num b) q) (Num a)) (Add (Num (+ b (neg a))) q) :ruleset sympy_rational)
; L12 (q + b) - a
(rewrite (Sub (Add q (Num b)) (Num a)) (Add q (Num (+ b (neg a)))) :ruleset sympy_rational)
; L13 (b - q) - a
(rewrite (Sub (Sub (Num b) q) (Num a)) (Sub (Num (+ b (neg a))) q) :ruleset sympy_rational)
; L14 (q - b) - a
(rewrite (Sub (Sub q (Num b)) (Num a)) (Sub q (Num (+ a b))) :ruleset sympy_rational)

; ---- M: literals of a product collect into one coefficient ---------------------------
; core/mul.py:358-363 (coeff *= o), 338-341 (nested Mul spliced).
; M1  a * (p * b)
(rewrite (Mul (Num a) (Mul p (Num b))) (Mul (Num (* a b)) p) :ruleset sympy_rational)
; M2  (b * p) * a
(rewrite (Mul (Mul (Num b) p) (Num a)) (Mul (Num (* a b)) p) :ruleset sympy_rational)
; M3-M6  a * (b / q) = (a*b) / q. Raw: NaN at q = 0 both sides. Protected: a * 0 = 0 in the band.
(rewrite (Mul (Num a) (Div (Num b) q)) (Div (Num (* a b)) q) :ruleset sympy_rational)
(rewrite (Mul (Div (Num b) q) (Num a)) (Div (Num (* a b)) q) :ruleset sympy_rational)
(rewrite (Mul (Num a) (ProtectedDiv (Num b) q)) (ProtectedDiv (Num (* a b)) q) :ruleset sympy_rational)
(rewrite (Mul (ProtectedDiv (Num b) q) (Num a)) (ProtectedDiv (Num (* a b)) q) :ruleset sympy_rational)
; M7  (p / a) / b = p / (a*b). RAW only (protected: |a*b| can fall under 1e-6 when neither does).
(rewrite (Div (Div p (Num a)) (Num b)) (Div p (Num (* a b))) :ruleset sympy_rational)
; M8-M9  a / (q / b) = (a*b) / q for a literal b != 0 (b = 0: NaN on the left, 0/q on the right).
(rule ((= e (Div (Num a) (Div q (Num b)))) (> b 0.0)) ((union e (Div (Num (* a b)) q))) :ruleset sympy_rational)
(rule ((= e (Div (Num a) (Div q (Num b)))) (< b 0.0)) ((union e (Div (Num (* a b)) q))) :ruleset sympy_rational)

; ---- N: -x is Mul(-1, x), so the -1 joins the coefficient ----------------------------
; core/mul.py:358-363; simplify/simplify.py:405 (signsimp rebuilds -(-x)). Each removes the Neg node.
(rewrite (Mul (Num a) (Neg x)) (Mul (Num (neg a)) x) :ruleset sympy_rational)
(rewrite (Mul (Neg x) (Num a)) (Mul (Num (neg a)) x) :ruleset sympy_rational)
(rewrite (Neg (Mul (Num a) x)) (Mul (Num (neg a)) x) :ruleset sympy_rational)
(rewrite (Neg (Mul x (Num a))) (Mul x (Num (neg a))) :ruleset sympy_rational)
; divide by a literal / literal numerator: |-k| = |k|, so the protected divide takes the same branch.
(rewrite (Div (Neg x) (Num a)) (Div x (Num (neg a))) :ruleset sympy_rational)
(rewrite (Neg (Div x (Num a))) (Div x (Num (neg a))) :ruleset sympy_rational)
(rewrite (Neg (Div (Num a) x)) (Div (Num (neg a)) x) :ruleset sympy_rational)
(rewrite (ProtectedDiv (Neg x) (Num a)) (ProtectedDiv x (Num (neg a))) :ruleset sympy_rational)
(rewrite (Neg (ProtectedDiv x (Num a))) (ProtectedDiv x (Num (neg a))) :ruleset sympy_rational)
(rewrite (Neg (ProtectedDiv (Num a) x)) (ProtectedDiv (Num (neg a)) x) :ruleset sympy_rational)

; ---- T: like terms -------------------------------------------------------------------
; core/add.py:280-281 (as_coeff_Mul), 304-310 (terms[s] += c), 335 (Mul(c, s)).
(rewrite (Add (Mul (Num a) x) (Mul (Num b) x)) (Mul (Num (+ a b)) x) :ruleset sympy_rational)
(rewrite (Add (Mul x (Num a)) (Mul x (Num b))) (Mul x (Num (+ a b))) :ruleset sympy_rational)
(rewrite (Add (Mul (Num a) x) (Mul x (Num b))) (Mul (Num (+ a b)) x) :ruleset sympy_rational)
(rewrite (Add (Mul x (Num a)) (Mul (Num b) x)) (Mul (Num (+ a b)) x) :ruleset sympy_rational)
(rewrite (Sub (Mul (Num a) x) (Mul (Num b) x)) (Mul (Num (+ a (neg b))) x) :ruleset sympy_rational)
(rewrite (Sub (Mul x (Num a)) (Mul x (Num b))) (Mul x (Num (+ a (neg b)))) :ruleset sympy_rational)
; a*x + x, x + a*x, a*x - x, x - a*x
(rewrite (Add (Mul (Num a) x) x) (Mul (Num (+ a 1.0)) x) :ruleset sympy_rational)
(rewrite (Add x (Mul (Num a) x)) (Mul (Num (+ a 1.0)) x) :ruleset sympy_rational)
(rewrite (Sub (Mul (Num a) x) x) (Mul (Num (+ a -1.0)) x) :ruleset sympy_rational)
(rewrite (Sub x (Mul (Num a) x)) (Mul (Num (+ 1.0 (neg a))) x) :ruleset sympy_rational)
; -x + a*x arrives as (Sub (Mul (Num a) x) x) through sign.rs; x - (-x) etc. likewise.
(rewrite (Add (Mul x (Num a)) x) (Mul x (Num (+ a 1.0))) :ruleset sympy_rational)

; ---- C: a term and its negative drop out of the sum ----------------------------------
; core/add.py:304-305, 318-319 (coefficient reaches zero, term dropped). identities.rs has
; (a+b)-b, (b+a)-b, (a-b)+b, b+(a-b); these are the remaining shapes.
(rewrite (Sub a (Add a b)) (Neg b) :ruleset sympy_rational)
(rewrite (Sub a (Add b a)) (Neg b) :ruleset sympy_rational)
(rewrite (Sub a (Sub a b)) b :ruleset sympy_rational)
(rewrite (Sub (Sub a b) a) (Neg b) :ruleset sympy_rational)

; ---- P: powers of one base collect (x**a / x**b = x**(a-b)) --------------------------
; core/mul.py:465-480 (_gather common bases), 518 (x**1 -> x).
; P1-P3 widen at x = 0 (NaN on the left), so they need the fact.
(rewrite (Div (Pow2 x) x) x :when ((is-nonzero x)) :ruleset sympy_rational)
(rewrite (Div (Pow3 x) x) (Pow2 x) :when ((is-nonzero x)) :ruleset sympy_rational)
(rewrite (Div (Pow3 x) (Pow2 x)) x :when ((is-nonzero x)) :ruleset sympy_rational)
; P4-P6 are exact with NO guard: both sides are NaN at x = 0.
(rewrite (Div x (Pow2 x)) (Inv x) :ruleset sympy_rational)
(rewrite (Div (Pow2 x) (Pow3 x)) (Inv x) :ruleset sympy_rational)
(rewrite (Div x (Pow3 x)) (Inv (Pow2 x)) :ruleset sympy_rational)
; P7-P8  x * x**2 = x**3 inside a quotient is the powers miner's; here only the mixed cancel:
; (x*y)/x**2 = y/x exact with no guard (NaN at x = 0 both sides).
(rewrite (Div (Mul x y) (Pow2 x)) (Div y x) :ruleset sympy_rational)
(rewrite (Div (Mul y x) (Pow2 x)) (Div y x) :ruleset sympy_rational)
; P9-P10 x**2/(x*y) = x/y widens at x = 0 (0/0 on the left, 0/y on the right).
(rewrite (Div (Pow2 x) (Mul x y)) (Div x y) :when ((is-nonzero x)) :ruleset sympy_rational)
(rewrite (Div (Pow2 x) (Mul y x)) (Div x y) :when ((is-nonzero x)) :ruleset sympy_rational)

; ---- K: cancel a common factor of numerator and denominator --------------------------
; polys/polytools.py:7125 (cancel), 7212 (F.cancel(G)); core/mul.py:465-480 when the factor
; is syntactically equal. RAW Div only. Guard is on the CANCELLED factor.
(rewrite (Div (Mul a b) (Mul a c)) (Div b c) :when ((is-nonzero a)) :ruleset sympy_rational)
(rewrite (Div (Mul b a) (Mul a c)) (Div b c) :when ((is-nonzero a)) :ruleset sympy_rational)
(rewrite (Div (Mul a b) (Mul c a)) (Div b c) :when ((is-nonzero a)) :ruleset sympy_rational)
(rewrite (Div (Mul b a) (Mul c a)) (Div b c) :when ((is-nonzero a)) :ruleset sympy_rational)
(rewrite (Div a (Mul a b)) (Inv b) :when ((is-nonzero a)) :ruleset sympy_rational)
(rewrite (Div a (Mul b a)) (Inv b) :when ((is-nonzero a)) :ruleset sympy_rational)
; (a/b)*b = a. identities.rs has (a*b)/b; rational.rs has this only for a literal b.
(rewrite (Mul (Div a b) b) a :when ((is-nonzero b)) :ruleset sympy_rational)
(rewrite (Mul b (Div a b)) a :when ((is-nonzero b)) :ruleset sympy_rational)
; (a/b)/a = 1/b
(rewrite (Div (Div a b) a) (Inv b) :when ((is-nonzero a)) :ruleset sympy_rational)
; a/(a/b) = b needs both: a = 0 gives 0/0, b = 0 gives a/NaN.
(rewrite (Div a (Div a b)) b :when ((is-nonzero a) (is-nonzero b)) :ruleset sympy_rational)

; ---- Z: 0 * x = 0 --------------------------------------------------------------------
; core/mul.py:684-692. Raw 0/0 is NaN, so guarded. (Protected 0/x was rejected: NaN x.)
(rewrite (Div (Num 0.0) x) (Num 0.0) :when ((is-nonzero x)) :ruleset sympy_rational)

; ---- G: together, same denominator ---------------------------------------------------
; polys/rationaltools.py:66-67 (_together on an Add -> gcd_terms), core/exprtools.py:980.
; Sound for the protected divide too: one c, one branch, 0 + 0 = 0.
(rewrite (Add (Div a c) (Div b c)) (Div (Add a b) c) :ruleset sympy_rational)
(rewrite (Sub (Div a c) (Div b c)) (Div (Sub a b) c) :ruleset sympy_rational)
(rewrite (Add (ProtectedDiv a c) (ProtectedDiv b c)) (ProtectedDiv (Add a b) c) :ruleset sympy_rational)
(rewrite (Sub (ProtectedDiv a c) (ProtectedDiv b c)) (ProtectedDiv (Sub a b) c) :ruleset sympy_rational)

; ---- F: factor_terms, a common factor out of a sum -----------------------------------
; core/exprtools.py:1156 (factor_terms) via gcd_terms:980; simplify/simplify.py:655, 727.
; Directed, shrinks by 1 + size(a). The reverse of distribute: do not co-load with it.
; The first Add rule and the first Sub rule are already shipped in wide.rs (bidirectional there).
(rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)) :ruleset sympy_rational)
(rewrite (Add (Mul b a) (Mul a c)) (Mul a (Add b c)) :ruleset sympy_rational)
(rewrite (Add (Mul a b) (Mul c a)) (Mul a (Add b c)) :ruleset sympy_rational)
(rewrite (Add (Mul b a) (Mul c a)) (Mul (Add b c) a) :ruleset sympy_rational)
(rewrite (Sub (Mul a b) (Mul a c)) (Mul a (Sub b c)) :ruleset sympy_rational)
(rewrite (Sub (Mul b a) (Mul a c)) (Mul a (Sub b c)) :ruleset sympy_rational)
(rewrite (Sub (Mul a b) (Mul c a)) (Mul a (Sub b c)) :ruleset sympy_rational)
(rewrite (Sub (Mul b a) (Mul c a)) (Mul (Sub b c) a) :ruleset sympy_rational)

; ---- E: exp(a)/exp(b) = exp(a-b) -----------------------------------------------------
; core/mul.py:465-480 (_gather on base E). Companion of powers.rs (Mul (Exp a) (Exp b)).
(rewrite (Div (Exp a) (Exp b)) (Exp (Sub a b)) :ruleset sympy_rational)

; ---- LG: logcombine ------------------------------------------------------------------
; simplify/simplify.py:1090-1093 (same coefficient multiply), 1096-1108 (opposite sign divide),
; 1023-1032 (gooda/goodlog: positive arguments only). Reverses the powers.rs log expander.
(rewrite (Add (Log a) (Log b)) (Log (Mul a b)) :when ((is-positive a) (is-positive b)) :ruleset sympy_rational)
(rewrite (Sub (Log a) (Log b)) (Log (Div a b)) :when ((is-positive a) (is-positive b)) :ruleset sympy_rational)
(rewrite (Mul (Num 2.0) (Log x)) (Log (Pow2 x)) :when ((is-positive x)) :ruleset sympy_rational)
```

## 4. The reported model, traced

`((3.0*(((3.0 + (q2*((Ef**2)/Ef))) + 4.0)/3.0)) + (-7.0))`, with `M = (Mul q2 (Div (Pow2 Ef) Ef))`:

1. `(Mul (Num 3.0) (Div X (Num 3.0)))` → `X` — EXISTS (`rational.rs`, literal `c != 0`).
2. `X = (Add (Add (Num 3.0) M) (Num 4.0))` → `(Add (Num 7.0) M)` — **L1**, missing today.
3. `(Add (Add (Num 7.0) M) (Num -7.0))` → `(Add (Num 0.0) M)` — **L1** again → `M` by the existing `Add 0 x`.
4. `(Div (Pow2 Ef) Ef)` → `Ef` — **P1**, needs `is-nonzero Ef` from the caller (`nonzero_vars`); without the
   fact it correctly does not fire (`Ef = 0` is NaN on the left). If the engine's divide here is `ProtectedDiv`,
   no sound rule exists (band `|Ef| < 1e-6` gives 0, not `Ef`).

Result `(Mul q2 Ef)`: 17 nodes → 3.
