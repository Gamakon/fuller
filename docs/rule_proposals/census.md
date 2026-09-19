# Census of the hall-of-fame genes — rewrite-rule opportunities (no sympy)

Source: `hff/notebooks/_ledgers/hof_dataset/genes.jsonl` — 847 unique genes, total `count` weight 24 344
(846 analysed; 1 errored in the builder on `(ProtectedDiv (Num 0.0) (Num 0.0))`).
`fuller_math` was produced by `smallest_form` = constant fold + `algebra+powers` pass + `algebra+powers+rational`
pass (`src/extract.rs:736`). `sympy_mined`, `wide`, `trig`, `distribute` are NOT in that path.
All weights below are `count`-weighted occurrences. "w_in" = in `math`, "w_out" = still present in `fuller_math`.
Nothing here was compiled or run through cargo; every rule is a proposal checked by hand against `src/eval.rs`.

## 1. Headline

- fuller today: 161 408 → 154 554 weighted nodes (−6 854, 4.25%). 190/847 genes reduced, 656 unchanged, 0 grew.
- The sound, unguarded, size-reducing rules in §3 tier A cover ≈ 2 540 further weighted node-savings
  (upper bound, overlaps not deduplicated) — about +37% on top of what fuller removes now.
- **Four of them are already written** in `sympy_mined` (`Abs (Neg x)`, `Abs (Pow2 x)`, `Abs (Sqrt x)`, `Abs (Exp x)`)
  or `wide` (`Neg (Sub a b)`, `Add a (Neg b)`) but never reach `smallest_form`. Recommendation: lift these few
  into `algebra`. NOTE this contradicts the stated decision in `src/ruleset/mod.rs` (sympy_mined wired into no family
  until the kingdom classifier exists). Reasoning: each of these strictly shrinks the term; the non-confluence trap
  is about the expanding rules in those sets (`Abs (Mul a b)` split, comm/assoc, distribute). Design call, not mine.
- Trig: **zero** Pythagorean / double-angle / `Sin/Cos→Tan` / `Cos*Tan` instances in 847 genes. Evolution nests trig
  (`(Tan (Tan T))` 240, `(Sin (Sin T))` 201, `(Tan (Sin T))` 210) rather than combining it. The only live trig
  win is even-function parity (`Cos (Abs x)`, `Cos (Neg x)`).
- The engine's diff_sq, seen as `(Pow2 (Sub ..))`, is the single most common depth-2 shape (≈4 300 weighted). It is
  irreducible itself but makes `Pow2` the most important even-function context for sign rules.

## 2. Tables

### 2.1 Op frequency (weighted)

| op | w | op | w | op | w |
|---|---|---|---|---|---|
| Var | 53850 | Cos | 4904 | Exp | 3520 |
| Sub | 11632 | Mul | 4834 | ProtectedInv | 3436 |
| Add | 9137 | Pow3 | 4343 | Sqrt | 2875 |
| Pow2 | 8782 | ProtectedDiv | 4287 | ProtectedLog | 2744 |
| Tanh | 5874 | Tan | 4188 | Div | 2732 |
| Num | 5845 | ProtectedExp | 4112 | Pow | 2729 |
| ProtectedSqrt | 5698 | Abs | 3561 | Inv | 2503 |
| Sin | 5457 | Neg | 3536 | Log | 973 |

No min/max ops appear. diff_sq arrives as `(Pow2 (Sub a b))`. Protected ops total 20 277 vs raw
Div/Sqrt/Log/Exp/Inv 12 603 — the ops sympy never sees natively are the majority.

### 2.2 Shapes, depth 2 (top; T = opaque subtree, v = variable, equal names = equal subtrees)

| w | shape | w | shape |
|---|---|---|---|
| 1503 | `(Pow2 (Sub T1 T2))` | 235 | `(Abs (Pow2 T1))` **reducible** |
| 1433 | `(Pow2 (Sub v1 T2))` | 212 | `(ProtectedLog (Cos T1))` |
| 1093 | `(Pow2 (Sub T1 v2))` | 211 | `(ProtectedSqrt (Pow2 T1))` reduced today |
| 390 | `(Tanh (Tanh T1))` | 210 | `(ProtectedExp (ProtectedSqrt T1))` |
| 374 | `(Pow3 (Pow2 T1))` | 208 | `(Tanh (Abs T1))` |
| 325 | `(Pow (Pow2 T1) v2)` | 180 | `(Cos (Abs T1))` **reducible** |
| 309 | `(Pow2 (Sub v1 v2))` | 180 | `(Pow3 (Pow3 T1))`, `(Pow2 (Pow2 T1))`, `(Pow2 (Exp T1))` |
| 308 | `(Sqrt (Pow2 T1))` reduced today | 150 | `(Inv (ProtectedInv v1))` guard-only |
| 301 | `(Cos (Pow2 T1))` | 150 | `(Neg (Add v1 T2))` |

Depth 1 top (153 distinct shapes): `(Pow2 T)` 7393, `(Tanh T)` 4010, `(Sub v T)` 3785, `(ProtectedSqrt T)` 3718, `(Sub T T)` 3302,
`(Tan T)` 3148, `(Sin T)` 3081, `(Pow3 T)` 2990, `(Cos T)` 2929, `(Add v T)` 2757, `(ProtectedExp T)` 2387, `(Abs T)` 2375,
`(Neg T)` 2351, `(Add T v)` 2342, `(ProtectedInv T)` 2313, `(Sin v)` 2195, `(Exp T)` 2061, `(Sub T v)` 1980, `(Sqrt T)` 1975,
`(Inv T)` 1911, `(ProtectedSqrt v)` 1889, `(Tanh v)` 1773, `(ProtectedLog T)` 1730, `(Cos v)` 1705, `(ProtectedExp v)` 1639,
`(Add T T)` 1598, `(Add v1 v2)` 1505, `(Mul v T)` 1480, `(Mul T T)` 1341, `(Pow2 v)` 1339, `(ProtectedDiv T T)` 1264,
`(ProtectedDiv v1 v2)` 569, `(Add v1 v1)` 210, `(Mul v1 v1)` 95. Depth 2 has 1541 distinct shapes, depth 3 has 2125 — the tail is long and flat
(most depth-3 shapes sit at one gene × 30).

Depth 3 top: `(Pow3 (Pow2 (Sub T1 T2)))` 212, `(Pow2 (Sub v1 (Pow2 T2)))` 150, `(Sqrt (Pow2 (Sub T1 T2)))` 120 (reduced),
`(Tanh (Tanh (Tanh T1)))` 120, `(Neg (Pow3 (Pow2 T1)))` 120, `(ProtectedSqrt (Pow2 (Sub T1 v2)))` 91 (reduced), `(Pow2 (Sub (Pow2 T1) v2))` 91, `(Cos (Pow2 (Sub v1 T2)))` 91, `(Pow2 (Sub (Inv T1) v2))` 90,
`(ProtectedInv (Pow2 (Sub v1 T2)))` 90, `(Pow2 (Sub (Abs T1) v2))` 90, `(Pow2 (Sub (Cos v1) v2))` 90, `(Sqrt (ProtectedExp (ProtectedSqrt T1)))` 90,
`(Sin (Pow2 (Sub T1 v2)))` 90, `(Sub (Cos (Abs T1)) v2)` 90 **reducible (A4)**, `(ProtectedExp (Tan (Sin T1)))` 90, `(Sub v1 (Pow2 (Sub v1 T2)))` 90,
`(Tan (ProtectedSqrt (Cos v1)))` 90, `(Abs (Pow2 (Sub T1 T2)))` 90 **reducible (A2)**, `(ProtectedInv (ProtectedInv (Mul T1 T2)))` 90 guard-only,
`(Inv (Tanh (Inv T1)))` 87, `(Pow2 (Sub v1 (Neg T2)))` 61 **reducible (A1)**, `(Add (Sqrt (Pow2 T1)) v2)` 60 (reduced), `(Inv (Add (Cos 0.0) v1))` 60 (folded).

### 2.3 nodes_in vs fuller_nodes

| nodes_in | genes | weight | w·nodes in | w·nodes fuller | reduction |
|---|---|---|---|---|---|
| 1 | 76 | 6170 | 6170 | 6170 | 0.00% |
| 2–3 | 155 | 5004 | 12141 | 12085 | 0.46% |
| 4–6 | 176 | 4552 | 22092 | 21729 | 1.64% |
| 7–10 | 146 | 3242 | 26659 | 25767 | 3.35% |
| 11–15 | 102 | 2191 | 27940 | 26864 | 3.85% |
| 16+ | 191 | 3173 | 66406 | 61939 | 6.73% |

Node-delta histogram (genes): 0→656, 1→69, 2→48, 3→22, 4→11, 5→15, 6–18→25.
(`summary.json` `weighted_tree_*` 693 887→688 274 is a different size metric; the numbers above use `nodes_in`/`fuller_nodes`.)

What fuller reduces today, by shape (w_in → w_out): constant subtrees 1569→0 (106 genes — the largest single source),
`Sqrt (Pow2 x)` 308→0, `ProtectedSqrt (Pow2 x)` 211→0, `Sub x 0` 159→0, `Add 0 x` 120→0, `Neg (Neg x)` 120→0,
`Mul x 1` 104→0, `Mul ±1 x` 92→0, `Abs (Exp x)` 62→0, `Pow 1 x` 60→0, `Pow2 (Sqrt x)` 66→6, `Pow x -1` 54→0.

sympy_capped (36 genes, w 604): 12 159 → 11 637 (4.3%), 21/36 reduced. sympy_broken (6, w 85): 1 633 → 1 129, 3/6 reduced.
The 3 unreduced broken genes all contain `(ProtectedDiv x (Num 0.0))` — see A16.

## 3. Opportunities, ranked by w_out

### Tier A — sound with no guard, size-reducing (add to `algebra`)

| # | shape | w_out | genes | capped/broken w | saves |
|---|---|---|---|---|---|
| A1 | `(Sub a (Neg b))` | 386 | 21 | 1 | 1 |
| A2 | `(Abs (Pow2 x))` | 295 | 12 | 0 | 1 |
| A3 | `(Neg (Sub a b))` | 240 | 10 | 0 | 1 |
| A4 | `(Cos (Abs x))` + `(Cos (Neg x))` | 150 + 106 | 11 | 0 | 1 |
| A5 | `(Abs (ProtectedExp x))` | 152 | 9 | 30 | 1 |
| A6 | `(ProtectedLog (Abs x))` + `(ProtectedLog (Neg x))` | 146 + 30 | 7 | 0 | 1 |
| A7 | `(ProtectedSqrt (Abs x))` + `(ProtectedSqrt (Neg x))` | 96 + 30 | 8 | 60 | 1 |
| A8 | `(Add a (Neg b))` + `(Add (Neg a) b)` | 122 + 95 | 10 | 0 | 1 |
| A9 | `(Pow2 (Sub (Neg a) b))` | 120 | 5 | 0 | 1 |
| A10 | `(Div (Num 1.0) x)` | 116 | 5 | 0 | 1 |
| A11 | `(Mul x x)` | 95 | 6 | 30 | 1 |
| A12 | `(Sub (Num 0.0) x)` | 91 | 4 | 30 | 1 |
| A13 | `(Abs (ProtectedSqrt x))` | 91 | 5 | 0 | 1 |
| A14 | `(Abs (Neg x))` | 89 | 3 | 0 | 1 |
| A15 | `(Pow2 (Abs x))` (+ `(Pow2 (Neg x))`, 0 today) | 60 | 4 | 0 | 1 |
| A16 | `(ProtectedDiv x (Num 0.0))` | 27 | 3 | 27 (all broken) | 2+size(x) |

```
; A1/A3/A8/A12 — sign plumbing. Exact in IEEE (only the sign of a zero can differ; no op in eval.rs reads it).
(rewrite (Sub a (Neg b)) (Add a b) :ruleset algebra)
(rewrite (Neg (Sub a b)) (Sub b a) :ruleset algebra)
(rewrite (Add a (Neg b)) (Sub a b) :ruleset algebra)
(rewrite (Add (Neg a) b) (Sub b a) :ruleset algebra)
(rewrite (Sub (Num 0.0) x) (Neg x) :ruleset algebra)
; A2/A5/A13/A14 — Abs of a never-negative op. NaN/±inf agree on both sides.
(rewrite (Abs (Pow2 x)) (Pow2 x) :ruleset algebra)
(rewrite (Abs (ProtectedExp x)) (ProtectedExp x) :ruleset algebra)
(rewrite (Abs (ProtectedSqrt x)) (ProtectedSqrt x) :ruleset algebra)
(rewrite (Abs (Sqrt x)) (Sqrt x) :ruleset algebra)          ; 0 today, NaN both sides when x<0
(rewrite (Abs (Neg x)) (Abs x) :ruleset algebra)
; A4/A6/A7/A15 — even functions absorb Abs/Neg.
(rewrite (Cos (Abs x)) (Cos x) :ruleset algebra)
(rewrite (Cos (Neg x)) (Cos x) :ruleset algebra)
(rewrite (Pow2 (Abs x)) (Pow2 x) :ruleset algebra)
(rewrite (Pow2 (Neg x)) (Pow2 x) :ruleset algebra)
(rewrite (ProtectedSqrt (Abs x)) (ProtectedSqrt x) :ruleset algebra)
(rewrite (ProtectedSqrt (Neg x)) (ProtectedSqrt x) :ruleset algebra)
(rewrite (ProtectedLog (Abs x)) (ProtectedLog x) :ruleset algebra)
(rewrite (ProtectedLog (Neg x)) (ProtectedLog x) :ruleset algebra)
; A9 — diff_sq of a negated left arm: (-a-b)^2 = (a+b)^2, exact.
(rewrite (Pow2 (Sub (Neg a) b)) (Pow2 (Add a b)) :ruleset algebra)
; A10/A11 — identical evaluation in eval.rs (Inv: NaN at 0, same as Div; Pow2 is a*a).
(rewrite (Div (Num 1.0) x) (Inv x) :ruleset algebra)
(rewrite (Mul x x) (Pow2 x) :ruleset algebra)
; A16 — |b| < 1e-6 -> 0.0 regardless of a (even NaN/inf a). The threshold IS the function.
; (f64 `abs` is used the same way in sympy_mined; literal written long-form, egglog's handling of `1e-6` unverified.)
(rewrite (ProtectedDiv x (Num k)) (Num 0.0) :when ((< (abs k) 0.000001)) :ruleset algebra)
```

Soundness notes:
- ProtectedSqrt/ProtectedLog test `is_finite` on the argument: `Abs`/`Neg` preserve finiteness and zero-ness, so
  both branches agree (inf→0.0 for sqrt, inf or 0→+inf for log).
- A16 is the only thing that can touch the sympy_broken genes (sympy turns `x/0` into zoo). With it
  `(Exp (ProtectedDiv wind_speed 0))` folds to `(Exp 0)` → 1 and then `Mul (Inv w) 1` → `Inv w`: the 13-node gene drops ≥4.
- A11 collapses `(Mul (Mul x x) x)` only to `(Mul (Pow2 x) x)`; add `(Mul (Pow2 x) x) → (Pow3 x)` (and the commuted form) for the follow-through — same evaluation order caveat is fp-roundoff only.

Examples (from `fuller_math`):
- A1: `(Sub oz4 (Neg oz2))` ×30; `(Sub (Sub (Inv (Tan R)) (ProtectedInv (Add S (Tan M)))) (Neg Ed))` ×24
- A2: `(Cos (Abs (Pow2 oz2)))` ×30 → with A4 becomes `(Cos (Pow2 oz2))`; `(Pow (Abs (Pow2 (Pow2 oz1))) oz1)` ×30; `(Inv (ProtectedInv (Abs (Pow2 (Sub (Neg (ProtectedLog angle)) ..)))))` ×30 (also A9)
- A3: `(Neg (Sub 1.0 oz1))` ×30; `(Neg (Sub 1.0 (Abs oz2)))` ×30
- A5: `(Pow3 (Abs (ProtectedExp (Sqrt (ProtectedExp oz5)))))` ×30; capped: `(ProtectedSqrt (Sub x1 (Abs (ProtectedExp (ProtectedExp (Tanh ..))))))` ×30
- A6: `(Tanh (ProtectedInv (ProtectedLog (Abs wind_speed))))` ×30
- A7: `(Pow (ProtectedSqrt (Abs (Pow3 oz1))) oz2)` ×30; capped `(.. (Log (ProtectedSqrt (Neg (ProtectedExp ..)))))` ×30
- A12: capped `(Add In2 (ProtectedSqrt (Sub 0.0 (ProtectedExp (Cos ..)))))` ×30 → A12 then A7 removes 2 nodes
- A16: broken `(ProtectedDiv (ProtectedDiv (Sub oz1 (Pow2 (Sub oz1 (ProtectedDiv oz3 0.0)))) oz5) oz1)` ×18

### Tier B — sound only under a guard

| shape | w_out | rule | guard |
|---|---|---|---|
| `(Inv (ProtectedInv x))` | 240 | → `x` | `is-nonzero x` (at 0: lhs = 1) |
| `(ProtectedInv (ProtectedInv x))` | 150 | → `x` | `is-nonzero x` (at 0: lhs = 1) |
| `(Inv (Inv x))` | 116 | exists in `rational`, guarded | never fires: argument is a bare Var / `Pow3 x` / `ProtectedExp x` |
| `(Div a (Inv b))` | 116 | → `(Mul a b)` | `is-nonzero b` |
| `(Log (Abs x))` | 31 | → `(ProtectedLog x)` | `is-nonzero x` (at 0: NaN vs +inf) AND finite-or-inf (x=NaN: NaN vs +inf) — really belongs with the `is-finite` gap in Tier C |
| `(ProtectedLog (Exp x))` 26, `(Log (ProtectedExp x))` 30, `(ProtectedLog (ProtectedExp x))` 7 | 63 | → `x` | same exp-overflow/underflow tail as the existing `(Log (Exp x)) → x`; no worse than current practice |

```
(rewrite (Inv (ProtectedInv x)) x :when ((is-nonzero x)) :ruleset rational)
(rewrite (ProtectedInv (ProtectedInv x)) x :when ((is-nonzero x)) :ruleset rational)
(rewrite (Div a (Inv b)) (Mul a b) :when ((is-nonzero b)) :ruleset rational)
```

These fire rarely as things stand: the 240 weight of `Inv (ProtectedInv ·)` is on bare variables
(`(Inv (ProtectedInv In3))` ×60), which are never assumed nonzero. They pay only when the caller passes
`nonzero_vars`, or when guard propagation is widened. Missing propagation seen in the data:
```
(rule ((= e (ProtectedExp x))) ((is-positive e)) :ruleset guards)     ; (Abs (Inv (Inv (ProtectedExp v))))
(rule ((is-nonzero x) (= e (Pow3 x))) ((is-nonzero e)) :ruleset guards) ; (Inv (Inv (Pow3 x2))) x25
(rule ((is-nonzero x) (= e (Neg x)))  ((is-nonzero e)) :ruleset guards)
(rule ((is-nonzero x) (= e (Inv x)))  ((is-nonzero e)) :ruleset guards)
(rule ((is-nonzero a) (is-nonzero b) (= e (Mul a b))) ((is-nonzero e)) :ruleset guards)
```
A third relation `is-nonneg` (seeded by Pow2, Abs, Sqrt, ProtectedSqrt, Exp, ProtectedExp) would replace the four
A2/A5/A13 rules by one `(Abs x) → x :when is-nonneg`, and unlock `(Pow2 (Sqrt p)) → p` on
`(Pow2 (Sqrt (ProtectedSqrt ..)))` (w 6) and `Abs (Pow2 (Pow2 ..))` chains.

### Tier C — UNSOUND under the engine's semantics: do NOT add

| shape | w_out | tempting rule | breaking input |
|---|---|---|---|
| `(Mul a (ProtectedInv b))` | 212 (+96 commuted) | → `(ProtectedDiv a b)` | b=0: `a` vs `0`; 0<\|b\|<1e-6: `a/b` vs `0` |
| `(ProtectedSqrt (ProtectedExp x))` | 217 | → `(ProtectedExp (Mul 0.5 x))` | overflow: `0.0` vs finite; also not smaller |
| `(Sqrt (Abs x))` | 59 | → `(ProtectedSqrt x)` | x=±inf: `inf` vs `0.0`; NaN: NaN vs 0.0 |
| `(Pow2 (ProtectedSqrt x))` | 66 | → `(Abs x)` | x=inf: `0` vs `inf` (finite x: fine) |
| `(ProtectedDiv (Num 1.0) x)` | 100 | → `(ProtectedInv x)` | x=0: `0` vs `1`; small \|x\|: `0` vs `1/x` |
| `(ProtectedDiv x x)` | 30 (capped) | → `1` | \|x\|<1e-6: `0` |
| `(ProtectedDiv (Num 0.0) x)` | 30 | → `0` | x=NaN → NaN; x=±inf fine. Sound only with a finiteness fact |
| `(Exp (ProtectedLog x))` 60, `(ProtectedExp (ProtectedLog x))` 30 | 90 | → `(Abs x)` | x=0: `+inf` vs `0`. OK under `is-nonzero x` + finite |
| `(ProtectedInv (Neg x))` | 90 | → `(Neg (ProtectedInv x))` | x=0: `1` vs `-1`; not smaller anyway |
| `(ProtectedDiv (ProtectedDiv a b) c)` | 82 (18 broken) | → `(ProtectedDiv a (Mul b c))` | b,c each ≥1e-6 but product <1e-6 |
| `(Pow (Num -1.0) x)` | 159 | anything | NaN for non-integer x; leave inert |

Structural gap these point at: the crate has no `is-finite` relation. `ProtectedExp` (w 4112) feeds +inf into
`ProtectedSqrt`/`ProtectedLog` chains constantly, and the Protected ops branch on finiteness, so the
`Sqrt(Abs)`/`Pow2(ProtectedSqrt)` family (≈125 w) stays blocked until finiteness can be asserted.

### Same-size shapes (no directed size win; listed so they are not re-mined)

`(Pow (Pow2 a) b)` 655, `(Tanh (Tanh x))` 480, `(Pow3 (Pow2 x))` 374 (= x^6, 3 nodes either way), `(Sub (Neg a) b)` 329,
`(Tanh (Abs x))` 239, `(Add x x)` 210, `(Pow2 (Pow2 x))` 210, `(Pow3 (Pow3 x))` 210, `(Sqrt (Exp x))` 204,
`(Abs (Pow3 x))` 211, `(Abs (ProtectedDiv a b))` 231, `(Abs (Mul a b))` 169, `(Div a (Div b c))` 149, `(Log (Pow2 x))` 150,
`(Tan/Sin/Tanh (Neg x))` 100/90/64 (odd-function pushes only help when an even op sits above; A3/A9 cover the Pow2 case).
