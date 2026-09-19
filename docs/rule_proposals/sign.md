# Rule proposals — sign normalisation (Neg / Sub / Abs / negative literals)

Evidence: `hff/notebooks/_ledgers/hof_dataset/gaps.jsonl` (33 rows, all read) and
`genes.jsonl` (846 analysed genes, scanned structurally on `fuller_math`).
Row numbers below are 1-based line numbers in those files. Nothing was run
(no cargo): the tests are written, not executed.

> **Overlap with `powers.md`:** the powers agent owns the even-function
> absorption rules (A8 here: `Pow2/Abs/ProtectedSqrt/ProtectedLog` of `Neg x` /
> `Abs x`) and the Abs-shedding of nonneg ranges (A9 here; they propose an
> `is-nonneg` guard relation instead of per-op rewrites). Where both files state
> the same rule, land it ONCE, from `powers.md`. A8/A9 are kept below only
> because the group-B float-outs depend on an even absorber existing
> (`Pow2 (Neg x)`, `Cos (Neg x)`) and the evidence/soundness notes are already
> written. `Cos (Neg x)` / `Cos (Abs x)` are not in their claim — they stay here.

> **Overlap with `rational.md`:** the rational agent claims A4
> (`Sub (Num 0.0) x -> Neg x`) and the two negative-literal-numerator rules in A6
> (`ProtectedDiv (Num c) (Neg x)` and the raw `Div` twin). Land those from
> `rational.md`; A7's row-17 second step depends on A4 existing. NB their
> message numbers rows 0-BASED (their "gap row 16" = line 17 here; "gene 271" =
> line 272); this file is 1-based throughout. They independently confirm the
> `ProtectedInv (Neg x)` rejection and the section-0 metric finding, and note
> one more sound shrinker in the A7 family, added here:
> `(rewrite (Neg (ProtectedDiv c (Sub a b))) (ProtectedDiv c (Sub b a)) :ruleset sign)`
> (and the raw `Div` twin) — `b-a = -(a-b)` exactly, so `|a-b| = |b-a|`, same
> threshold branch, live branch exact. No ledger instance found in my scan; it
> only fires after a B2 float-out, so it is optional.

## 0. Read this first — two different size metrics

- `smallest_form` accepts a candidate only when `cost_of` (Math node count) is
  **strictly smaller** (`src/extract.rs:743`).
- `gap_nodes` is `fuller_tree − sympy_tree`: sympy's tree size of the *rendered*
  string. sympy has no Sub/Neg: `1.0 - x` is `Add(1.0, Mul(-1, x))` = 5 nodes,
  `x - 1.0` is `Add(x, -1.0)` = 3 nodes.

Consequence: **13 of the 33 gap rows are pure operand-orientation artifacts with
identical Math node count on both sides** — no rewrite rule can make
`smallest_form` prefer one over the other:

| shape | gap rows | genes.jsonl rows |
|---|---|---|
| `Pow2 (Sub (Num c) x)`, c>0 — sympy shows `(x - c)**2` | 9, 10, 13, 14, 15, 28, 30, 31 (+16 partly) | 22 rows (171, 172, 237, 249, 416, 420, 447, 479, 515, 528, 547, 549, 566, 572, 595, 692, 752, 756, 765, 774, 825, 826) |
| `Pow2 (Sub (Num c) x)`, c<0 — sympy shows `(x + |c|)**2` | 8, 11 (+12, 29 partly) | 9 rows (106, 139, 288, 343, 349, 473, 601, 630, 769) |
| `Pow2 (Sub a (Add b c))` — sympy flips to fewer minus signs | 6, 33 | — |
| sympy pulls a sign OUT of an odd fn (`sigma*sin(-u)` -> `-sigma*sin(u)`) | 20 | — |

I propose **no rule** for these. `Pow2 (Sub a b) = Pow2 (Sub b a)` is sound and
self-inverse (saturates in one step), but it is size-neutral, so the extractor
never selects it. If the lead wants those rows, it is a *cost/tie-break*
decision (e.g. secondary key = count of `Sub` whose RHS is non-`Num` + negative
literals + `Neg`), not a ruleset one. Also not rules: rows 22/27/32 — the
`Abs(Abs(oz5))` is the *renderer* printing `ProtectedSqrt (Mul (Abs a) (Exp b))`
as `sqrt(Abs(Abs(a)*exp(b)))`; the Math has one `Abs`. (`ProtectedSqrt p -> Sqrt p`
for p>=0 is UNSOUND at overflow: protected_sqrt(+inf)=0, sqrt(+inf)=+inf.)

## 1. Where the rules go

New file `src/ruleset/sign.rs`, `pub const SIGN_RULESET` declaring `(ruleset sign)`,
registered in `src/ruleset/mod.rs`, and added to the three combined-ruleset
lines: `src/extract.rs:482`, `:784` (`denoise_all guards algebra powers sign`) and
`:647` (`rational_all guards algebra powers rational sign`). Note `denoise()`
shares `denoise_all`, so the rules fire in the live operator too (bounded by
`DENOISE_ITERS = 40`).

`mod.rs` says `sympy_mined` is wired into no family to avoid the non-confluence
trap. That concern applies to its EXPANDING rules (`Abs (Mul a b) -> Mul (Abs a)
(Abs b)`, `Abs (Div ..)`, `Abs (Inv ..)`) — those stay out. The four Abs rules
lifted below are strict shrinkers with no expanding counterpart anywhere in
algebra/powers/rational (checked: rational.rs and powers.rs contain no `Neg`
and no Abs rewrite; rational only marks `Abs` as `leaf`).

```egglog
(ruleset sign)

; ===== A. ABSORBERS — each strictly removes >= 1 node =====
; A1  a - (-b) = a + b
(rewrite (Sub a (Neg b)) (Add a b) :ruleset sign)
; A2  a + (-b) = a - b ; (-a) + b = b - a
(rewrite (Add a (Neg b)) (Sub a b) :ruleset sign)
(rewrite (Add (Neg a) b) (Sub b a) :ruleset sign)
; A3  -(a - b) = b - a
(rewrite (Neg (Sub a b)) (Sub b a) :ruleset sign)
; A4  0 - x = -x
(rewrite (Sub (Num 0.0) x) (Neg x) :ruleset sign)
; A5  -(literal) folds ; -(c + x) = (-c) - x, both Add orders
(rewrite (Neg (Num a)) (Num (neg a)) :ruleset sign)
(rewrite (Neg (Add (Num c) x)) (Sub (Num (neg c)) x) :ruleset sign)
(rewrite (Neg (Add x (Num c))) (Sub (Num (neg c)) x) :ruleset sign)
; A6  sign pairs cancel through a product / quotient (incl. protected divide)
(rewrite (Mul (Neg a) (Neg b)) (Mul a b) :ruleset sign)
(rewrite (Div (Neg a) (Neg b)) (Div a b) :ruleset sign)
(rewrite (ProtectedDiv (Neg a) (Neg b)) (ProtectedDiv a b) :ruleset sign)
(rule ((= e (ProtectedDiv (Num c) (Neg b))) (< c 0.0))
      ((union e (ProtectedDiv (Num (neg c)) b))) :ruleset sign)
(rule ((= e (Div (Num c) (Neg b))) (< c 0.0))
      ((union e (Div (Num (neg c)) b))) :ruleset sign)
; A7  a negation meeting a difference inside a product flips the difference
(rewrite (Neg (Mul (Sub a b) c)) (Mul (Sub b a) c) :ruleset sign)
(rewrite (Neg (Mul c (Sub a b))) (Mul c (Sub b a)) :ruleset sign)
(rewrite (Mul (Sub a b) (Neg c)) (Mul (Sub b a) c) :ruleset sign)
(rewrite (Mul (Neg c) (Sub a b)) (Mul c (Sub b a)) :ruleset sign)
; A8  EVEN functions swallow a Neg / an Abs
(rewrite (Pow2 (Neg x)) (Pow2 x) :ruleset sign)
(rewrite (Pow2 (Abs x)) (Pow2 x) :ruleset sign)
(rewrite (Abs (Neg x)) (Abs x) :ruleset sign)
(rewrite (Cos (Neg x)) (Cos x) :ruleset sign)
(rewrite (Cos (Abs x)) (Cos x) :ruleset sign)
(rewrite (ProtectedSqrt (Neg x)) (ProtectedSqrt x) :ruleset sign)
(rewrite (ProtectedSqrt (Abs x)) (ProtectedSqrt x) :ruleset sign)
(rewrite (ProtectedLog (Neg x)) (ProtectedLog x) :ruleset sign)
(rewrite (ProtectedLog (Abs x)) (ProtectedLog x) :ruleset sign)
; A9  Abs of a function whose range is already >= 0 (or NaN)
(rewrite (Abs (Pow2 x)) (Pow2 x) :ruleset sign)
(rewrite (Abs (Sqrt x)) (Sqrt x) :ruleset sign)
(rewrite (Abs (ProtectedSqrt x)) (ProtectedSqrt x) :ruleset sign)
(rewrite (Abs (ProtectedExp x)) (ProtectedExp x) :ruleset sign)

; ===== B. FLOAT-OUTS — size-neutral; move a Neg one level ROOTWARD so an
; absorber above can delete it. Never push a Neg down. =====
; B1  (-a) - b = -(a + b)
(rewrite (Sub (Neg a) b) (Neg (Add a b)) :ruleset sign)
; B2  Neg out of a product / quotient (raw and protected divide)
(rewrite (Mul (Neg a) b) (Neg (Mul a b)) :ruleset sign)
(rewrite (Mul a (Neg b)) (Neg (Mul a b)) :ruleset sign)
(rewrite (Div (Neg a) b) (Neg (Div a b)) :ruleset sign)
(rewrite (Div a (Neg b)) (Neg (Div a b)) :ruleset sign)
(rewrite (ProtectedDiv (Neg a) b) (Neg (ProtectedDiv a b)) :ruleset sign)
(rewrite (ProtectedDiv a (Neg b)) (Neg (ProtectedDiv a b)) :ruleset sign)
; B3  ODD functions pass a Neg through (raw Inv only — NOT ProtectedInv)
(rewrite (Sin (Neg x)) (Neg (Sin x)) :ruleset sign)
(rewrite (Tan (Neg x)) (Neg (Tan x)) :ruleset sign)
(rewrite (Tanh (Neg x)) (Neg (Tanh x)) :ruleset sign)
(rewrite (Pow3 (Neg x)) (Neg (Pow3 x)) :ruleset sign)
(rewrite (Inv (Neg x)) (Neg (Inv x)) :ruleset sign)
```

`neg` on f64 and `(rule (... (< c 0.0)) ...)` are both already used in the crate
(`distribute.rs:57`, `expr.rs:102`).

## 2. Evidence per rule (Math node counts, `fuller_nodes` before -> after)

| rule | gap rows fixed (nodes) | genes.jsonl rows with the subtree |
|---|---|---|
| A1 `Sub a (Neg b)` | 23 (19->18), 24 (19->18) | 21: 106, 119, 175, 278, 343, 391, 411, 423, 560, 562, 589, 594, 612, 613, 645, 689, 720, 723, 724, 740, 800 |
| A2 `Add .. (Neg ..)` | 18 (22->21) | 10: 210, 211, 282, 292, 318, 342, 502, 647, 843, 845 |
| A3 `Neg (Sub a b)` | — | 10: 222, 223, 224, 388, 454, 492, 503, 594, 645, 740 |
| A4 `Sub 0 x` | 17 (8->7, then A7: ->6) | 4: 272, 347, 416, 772 |
| A5 `Neg (Add (Num c) x)` | 25 (24->23) | `Add` with a negative literal: 18 rows (only those under a Neg shrink) |
| A6 sign pairs | 29 (`ProtectedDiv -1 (Neg in2)`: 25->24) | 601, 769 |
| A7 Neg x difference | 5 (11->10), 26 (11->10), 17 (second step) | 564 (count 28), 693 |
| A8 `Abs (Neg x)` (+ existing `Abs (Abs x)`) | 21 (12->10) | 163, 345, 548 |
| A8 `Cos (Neg x)` | 12 (21->20) | 94, 134, 349, 615, 781 |
| A8 `Cos (Abs x)` | — | 108, 377, 427, 471, 558, 823 |
| A8 `Pow2 (Abs x)` | 16 (15->14) | 249, 618, 624, 638 |
| A8 `ProtectedSqrt (Abs x)` | 7 (19->18) | 117, 232, 595, 642, 665, 698, 774 |
| A8 `ProtectedLog (Abs x)` | 19 (24->23) | 118, 501, 504, 507, 583, 801 |
| A8 `ProtectedSqrt/Log (Neg x)` | — | 143, 454 |
| A9 `Abs (Pow2 x)` | 2 (15->14; with B1+A8: ->13) | 12: 230, 289, 377, 427, 471, 505, 522, 556, 565, 577, 697, 821 |
| A9 `Abs (Sqrt/PSqrt/Exp/PExp x)` | 1 (`Abs (ProtectedExp ..)`: 6->5) | 14: 108, 259, 273, 467, 472, 514, 546, 572, 692, 752, 755, 779, 811, 817 |
| B1 + A8 `Pow2 (Sub (Neg a) b)` -> `Pow2 (Add a b)` | 2, 4 (24->23) | 123, 504, 505, 579, 653 (`Sub (Neg a) b` anywhere: 16 rows) |
| B1 + B3 + A1 (Neg floats through `Inv`, `Tan`, absorbed by the outer `Sub`) | 3 (14->13) | odd-fn-of-Neg: 22 rows (131, 181, 218, 219, 274, 289, 350, 447, 449, 456, 561, 623, 627, 639, 660, 670, 672, 686, 721, 722, 769, 808) |
| B2 Mul / Div / ProtectedDiv | feeds A7 (rows 5, 26) | `ProtectedDiv` with a Neg arg: 11 rows (261, 286, 504, 570, 575, 601, 630, 666, 687, 741, 769); Mul: 198, 564, 693; Div: 178, 502 |

Gap rows that get a Math-node reduction: **1, 2, 3, 4, 5, 7, 12, 16, 17, 18, 19,
21, 23, 24, 25, 26, 29 = 17 of 33.** Whether the *sympy-tree* gap closes fully
needs a re-run of the ledger (not done — battery). Expected from the strings:
fully closed 2, 3, 5, 17, 21, 26 and very likely 1; partly 4, 12, 29 (the rest is
the orientation artifact of section 0); rows 7, 16, 18, 19, 23, 24, 25 shrink in
Math nodes but their sympy-tree gap has another cause (rational / log-expand /
Abs-merge families) because sympy's printer already hides e.g. `a - (-b)`.

## 3. Soundness (reals + the evaluator in `src/eval.rs`)

IEEE negation is exact and total (flips the sign bit; NaN stays NaN, inf flips),
so every identity that only moves/cancels a negation is bit-exact except
possibly the SIGN OF A ZERO result. Signed zero is unobservable in this
evaluator: `Div`/`Inv` by ±0 -> NaN, `Log(±0)` -> NaN, `Pow(±0, neg)` -> NaN,
`ProtectedInv(±0)` = 1, `ProtectedLog(±0)` = +inf, `ProtectedDiv` tests
`|b| < 1e-6`, and there is no atan2/copysign. (Same standard the crate already
applies to `Mul -1 x -> Neg x`.)

- A1–A5, B1: `a-(-b)=a+b`, `-(a-b)=b-a`, `(-a)-b=-(a+b)`, `0-x=-x`, `-(c+x)=(-c)-x`
  are exact under round-to-nearest (rounding is sign-symmetric). NaN/inf propagate
  identically.
- A6, B2 raw `Mul`/`Div`: exact (sign-symmetric rounding); `Div` by 0 is NaN on
  both sides since `-b == 0` iff `b == 0`.
- A6, B2 `ProtectedDiv`: the branch condition is `|b| < 1e-6` and `|-b| = |b|`
  exactly, so both sides take the same branch; zero branch gives 0 vs -0
  (unobservable), live branch is exact. **Sound for the protected divide — no
  guard needed.** This is a property of THIS rule, not a general licence to lift
  Div rules.
- A7: composition of B2 and A3.
- A8 `Pow2`: `(-x)*(-x)` = `x*x` bit-exact; `|x|*|x|` likewise. `Abs (Neg x)`
  exact. `ProtectedSqrt`: `sqrt(|x|) if finite else 0` — `|-x| = ||x|| = |x|` and
  finiteness is sign-blind (NaN input -> 0 on both sides). `ProtectedLog`:
  `+inf if !finite or x==0 else ln|x|` — same argument.
  `Cos (Neg x)`, `Cos (Abs x)`: cos is even mathematically; libm symmetry is
  universal in practice but not IEEE-mandated -> covered by the eval test below.
- A9: `x*x >= 0` or NaN; `Sqrt` returns >= 0 or NaN (NaN for negatives, and
  |NaN| = NaN); `ProtectedSqrt` range is [0, inf) — never NaN; `ProtectedExp`
  range is (0, +inf] — never NaN (non-finite input -> +inf, and |+inf| = +inf).
  `Abs (Exp x)` is already shed by the `is-positive` guard path; not re-added.
- B3: `Pow3` exact; `Tanh`, `Sin` odd (libm, tested); `Tan` in eval.rs is
  literally `sin/cos` with NaN at `cos == 0` — odd given sin odd/cos even, same
  NaN set. raw `Inv`: `1/(-x) = -(1/x)` exact, NaN at 0 on both sides.

### Considered and REJECTED

1. **`ProtectedInv (Neg x) -> Neg (ProtectedInv x)`** (genes rows 174, 429, 527):
   UNSOUND at x = 0. `protected_inv(-0) = 1`, `-(protected_inv(0)) = -1`.
   A regression test asserts it stays inert.
2. **`Exp (Neg x)` / `ProtectedExp (Neg x)` -> `Inv (Exp x)`**: grows, and the
   protected form differs at overflow (`protected_exp(-inf) = +inf`).
3. **`ProtectedSqrt p -> Sqrt p` when p >= 0** (would clean the rows 22/27/32
   rendering): unsound at overflow — `protected_sqrt(+inf) = 0`, `sqrt(+inf) = +inf`.
4. **`Inv (ProtectedInv x) -> x`** (row 2's outer wrapper; not my family):
   unsound at x = 0 (`Inv(1) = 1`, not 0).
5. **`Pow2 (Sub a b) <-> Pow2 (Sub b a)`** and `Sub (Num c) x` reorientation:
   sound but size-neutral and self-inverse — useless under the strict-shrink
   acceptance, see section 0.
6. **Pulling a sign out to the root** (`Mul s (Sin (Neg u))` -> `Neg (Mul ..)`
   with nothing to absorb it, sympy's row 20 form): never shrinks Math nodes.
7. **`Neg x -> Mul (Num -1.0) x`** (trig.rs:60) and **`Sub a b -> Add a (Neg b)`**
   (wide.rs:63): the opposite direction — see termination.
8. **`Abs (Mul a b) -> Mul (Abs a) (Abs b)`** and friends from sympy_mined: expanders; stay out.
9. **`Abs (Pow x (Num k))`, k even, -> `Pow x k`**: sound, zero instances in the
   ledger (even powers arrive as `Pow2`). Not proposed without evidence.

## 4. Termination / blow-up

- Group A: every rule strictly removes nodes. No bidirectional pairs.
- Group B: size-neutral, but each application moves one `Neg` strictly toward
  the root and no rule in algebra/powers/rational/sign moves a `Neg` down or
  creates one (the only Neg producers are `Mul -1 x -> Neg x`, A4 and B1, all
  shrinking or neutral). New e-nodes are bounded by (number of Neg) x (depth):
  linear, not exponential. No RHS invents a fresh operator tower or literal
  arithmetic beyond one `neg`.
- **Do NOT co-load `sign` with `trig` or `wide`.** `trig.rs:60`
  `Neg a -> Mul -1 a` + algebra `Mul -1 x -> Neg x` + B2 form a cycle, and
  `wide.rs:63-64` re-introduce `Add a (Neg b)`. They are equalities so the
  e-graph stays finite under a bounded schedule, but the node count multiplies.
  `sign` belongs only in `denoise_all` and `rational_all`.
- If the lead wants zero risk, ship group A alone first: it fixes gap rows 1, 7,
  12, 16, 17 (first step), 18, 19, 21, 23, 24, 25, 29 and part of 2. Group B is
  needed for 3, 4, 5, 26 and the second node of row 2. (A7's direct
  `Mul (Sub a b) (Neg c)` forms already cover 5/26 without B.)

## 5. Tests (for `src/ruleset/sign.rs`, identities.rs style)

```rust
#[cfg(test)]
mod tests {
    use super::SIGN_RULESET;
    use crate::eval::eval_term;
    use crate::expr::{GUARD_RELATIONS, MATH_DATATYPE};
    use crate::ruleset::identities::ALGEBRA_RULESET;
    use egglog::prelude::exprs;
    use egglog::EGraph;

    // Hard safety cap (mirrors sympy_mined.rs): a divergent rule stops here
    // instead of pegging the machine.
    const SAT_ITERS: u32 = 8;

    fn egraph() -> EGraph {
        let mut e = EGraph::default();
        for prog in [MATH_DATATYPE, GUARD_RELATIONS, ALGEBRA_RULESET, SIGN_RULESET] {
            e.parse_and_run_program(None, prog).expect("program loads");
        }
        e.parse_and_run_program(None, "(unstable-combined-ruleset sign_all guards algebra sign)")
            .expect("combined ruleset");
        e
    }

    /// Bounded run, then the lowest-cost form as a string.
    fn simplify(input: &str) -> String {
        let mut e = egraph();
        e.parse_and_run_program(
            None,
            &format!("(let __r {input})\n(run-schedule (repeat {SAT_ITERS} (run sign_all)))"),
        )
        .expect("saturate");
        let (sort, value) = e.eval_expr(&exprs::var("__r")).expect("eval");
        e.extract_value_to_string(&sort, value).expect("extract").0
    }

    #[test]
    fn absorbers_fire() {
        let cases: &[(&str, &str)] = &[
            (r#"(Sub (Var "a") (Neg (Var "b")))"#, r#"(Add (Var "a") (Var "b"))"#),
            (r#"(Add (Var "a") (Neg (Var "b")))"#, r#"(Sub (Var "a") (Var "b"))"#),
            (r#"(Add (Neg (Var "a")) (Var "b"))"#, r#"(Sub (Var "b") (Var "a"))"#),
            (r#"(Neg (Sub (Var "a") (Var "b")))"#, r#"(Sub (Var "b") (Var "a"))"#),
            (r#"(Sub (Num 0.0) (Var "x"))"#, r#"(Neg (Var "x"))"#),
            (r#"(Neg (Num 4.0))"#, "(Num -4.0)"),
            (r#"(Neg (Add (Num -1.0) (Var "x")))"#, r#"(Sub (Num 1.0) (Var "x"))"#),
            (r#"(Mul (Neg (Var "a")) (Neg (Var "b")))"#, r#"(Mul (Var "a") (Var "b"))"#),
            (r#"(ProtectedDiv (Num -1.0) (Neg (Var "x")))"#, r#"(ProtectedDiv (Num 1.0) (Var "x"))"#),
            (r#"(Pow2 (Neg (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
            (r#"(Pow2 (Abs (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
            (r#"(Cos (Neg (Var "x")))"#, r#"(Cos (Var "x"))"#),
            (r#"(Cos (Abs (Var "x")))"#, r#"(Cos (Var "x"))"#),
            (r#"(ProtectedSqrt (Abs (Var "x")))"#, r#"(ProtectedSqrt (Var "x"))"#),
            (r#"(ProtectedLog (Abs (Var "x")))"#, r#"(ProtectedLog (Var "x"))"#),
            (r#"(ProtectedLog (Neg (Var "x")))"#, r#"(ProtectedLog (Var "x"))"#),
            (r#"(Abs (Pow2 (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
            (r#"(Abs (Sqrt (Var "x")))"#, r#"(Sqrt (Var "x"))"#),
            (r#"(Abs (ProtectedSqrt (Var "x")))"#, r#"(ProtectedSqrt (Var "x"))"#),
            (r#"(Abs (ProtectedExp (Var "x")))"#, r#"(ProtectedExp (Var "x"))"#),
            // gap row 21: Abs(Neg(Abs t)) -> Abs t
            (r#"(Abs (Neg (Abs (Tanh (Var "x")))))"#, r#"(Abs (Tanh (Var "x")))"#),
        ];
        let mut failures = Vec::new();
        for (input, expected) in cases {
            let got = simplify(input);
            if got != *expected {
                failures.push(format!("{input}\n  got:      {got}\n  expected: {expected}"));
            }
        }
        assert!(failures.is_empty(), "{} failed:\n{}", failures.len(), failures.join("\n"));
    }

    /// Float-outs only pay when an absorber sits above; these are the real
    /// hall-of-fame shapes (gaps.jsonl rows 2, 3, 5, 17).
    #[test]
    fn float_outs_reach_an_absorber() {
        let cases: &[(&str, &str)] = &[
            // row 2: (-a - b)^2 = (a + b)^2
            (
                r#"(Pow2 (Sub (Neg (Var "a")) (Var "b")))"#,
                r#"(Pow2 (Add (Var "a") (Var "b")))"#,
            ),
            // row 3: m - tan(1/(-u - v)) = m + tan(1/(u + v))
            (
                r#"(Sub (Var "m") (Tan (Inv (Sub (Neg (Var "u")) (Var "v")))))"#,
                r#"(Add (Var "m") (Tan (Inv (Add (Var "u") (Var "v")))))"#,
            ),
            // rows 5/26: (1 - s) * -(pinv o) = (s - 1) * pinv o
            (
                r#"(Mul (Sub (Num 1.0) (Var "s")) (Neg (ProtectedInv (Var "o"))))"#,
                r#"(Mul (Sub (Var "s") (Num 1.0)) (ProtectedInv (Var "o")))"#,
            ),
            // row 17: 0 - a*(b - c) = a*(c - b)
            (
                r#"(Sub (Num 0.0) (Mul (Var "a") (Sub (Var "b") (Var "c"))))"#,
                r#"(Mul (Var "a") (Sub (Var "c") (Var "b")))"#,
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(simplify(input), *expected, "{input}");
        }
    }

    /// ProtectedInv is NOT odd: protected_inv(-0) = 1 but -(protected_inv 0) = -1.
    /// No sign rule may touch it.
    #[test]
    fn protected_inv_of_neg_is_inert() {
        let input = r#"(ProtectedInv (Neg (Var "x")))"#;
        assert_eq!(simplify(input), input, "ProtectedInv(Neg x) was rewritten (unsound at 0)");
    }

    /// Numeric soundness: input and simplified form agree bit-for-bit (or are
    /// both NaN) at negative, zero, sub-threshold and overflow points. Covers
    /// the libm-dependent parity rules (Cos/Sin/Tan/Tanh) and every protected
    /// branch the sign rules cross.
    #[test]
    fn sign_rules_are_sound_on_the_evaluator() {
        let inputs = [
            r#"(Cos (Neg (Var "x")))"#,
            r#"(Cos (Abs (Var "x")))"#,
            r#"(Sub (Var "y") (Sin (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Tan (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Tanh (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Pow3 (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (Inv (Neg (Var "x"))))"#,
            r#"(Sub (Var "y") (ProtectedDiv (Neg (Var "y")) (Var "x")))"#,
            r#"(Sub (Var "y") (ProtectedDiv (Var "y") (Neg (Var "x"))))"#,
            r#"(ProtectedDiv (Neg (Var "y")) (Neg (Var "x")))"#,
            r#"(ProtectedDiv (Num -1.0) (Neg (Var "x")))"#,
            r#"(ProtectedSqrt (Abs (Var "x")))"#,
            r#"(ProtectedSqrt (Neg (Exp (Var "x"))))"#,
            r#"(ProtectedLog (Abs (Var "x")))"#,
            r#"(ProtectedLog (Neg (Var "x")))"#,
            r#"(Abs (ProtectedExp (Neg (Exp (Var "x")))))"#,
            r#"(Abs (ProtectedSqrt (Var "x")))"#,
            r#"(Abs (Sqrt (Var "x")))"#,
            r#"(Pow2 (Sub (Neg (Var "x")) (Var "y")))"#,
            r#"(Mul (Sub (Num 1.0) (Var "y")) (Neg (ProtectedInv (Var "x"))))"#,
            r#"(Neg (Add (Num -1.0) (Var "x")))"#,
        ];
        // x: negative, zero, below the protected_div threshold, ordinary, and
        // large enough that Exp(x) overflows to +inf.
        let xs = [-2.3, 0.0, -1e-7, 0.7, 1000.0];
        for input in inputs {
            let simplified = simplify(input);
            for x in xs {
                let lookup = |n: &str| match n {
                    "x" => Some(x),
                    "y" => Some(-1.25),
                    _ => None,
                };
                let value = |math: &str| {
                    let mut e = EGraph::default();
                    e.parse_and_run_program(None, MATH_DATATYPE).expect("datatype");
                    e.parse_and_run_program(None, &format!("(let __v {math})")).expect("term");
                    let (s, v) = e.eval_expr(&exprs::var("__v")).expect("eval");
                    let (td, t, _) = e.extract_value(&s, v).expect("extract");
                    eval_term(&td, t, &lookup).expect("evaluates")
                };
                let (a, b) = (value(input), value(&simplified));
                assert!(
                    (a.is_nan() && b.is_nan()) || a == b,
                    "unsound at x={x}: {input} = {a} but {simplified} = {b}"
                );
            }
        }
    }
}
```

Notes for whoever lands it: `a == b` deliberately treats +0 and -0 as equal
(section 3). The expected strings in `float_outs_reach_an_absorber` assume the
extractor's tie-break lands on the shown operand order; if a same-cost sibling
is extracted instead, assert on node count (`cost_of`) rather than the string.
`eval_term`'s exact signature should be checked against `sympy_mined.rs:111-150`
(it is used there as `eval_term(&termdag, term, &lookup)`).

## 6. Already-present check

Present in the LIVE families (not re-proposed): `Neg (Neg x)`, `Abs (Abs x)`,
`Sqrt (Pow2 x) -> Abs x`, `ProtectedSqrt (Pow2 x) -> Abs x`, `Mul/Div/ProtectedDiv
by -1 -> Neg`, guarded `Abs x -> x` (covers `Abs (Exp x)`).
Present only in families `smallest_form`/`denoise` never load (so proposed here
for `sign`): `Abs (Neg x)`, `Abs (Pow2 x)`, `Abs (Sqrt x)` (sympy_mined — wired
nowhere); `Add a (Neg b) -> Sub a b`, `Neg (Sub a b) -> Sub b a` (wide — scorer
only, alongside the reverse `Sub -> Add Neg`); `Neg (Num a)` fold
(distribute, trig). Everything else in section 1 is new.
