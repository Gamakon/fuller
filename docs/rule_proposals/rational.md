# Rule proposals — rational structure and constants

Evidence: `hff/notebooks/_ledgers/hof_dataset/{gaps,genes}.jsonl` (33 gap rows, 847 genes).
Nothing here was compiled or run (battery constraint): every rule and test below is
UNTESTED text. Gene row numbers are 0-based line indices into `genes.jsonl`; gap row
numbers are 0-based line indices into `gaps.jsonl`. Node counts are Math `node_count`
(what `smallest_form` optimises, strict-shrink), not the dataset's `gap_nodes`.

## Two findings about the dataset first

1. **`gap_nodes` is a sympy-tree size, not a Math size.** `_hof_dataset.py` computes
   `fuller_tree = tree_size(fuller.from_math(sf.expr))`. Sympify turns `Sub c x` into
   `Add(c, Mul(-1, x))` (+2) and `Neg` into `Mul(-1, .)` (+1). Gap rows 5, 7, 9, 10,
   12, 13, 14, 27, 29, 30 are `Pow2 (Sub c x)` against sympy's `(x - c)**2`: the same
   size in Math. No rewrite can make `smallest_form` accept them (it takes only a
   strictly smaller form).
2. **`verified` samples every variable on `[0.5, 4.0]`** and `smallest_form` was called
   with `positive_vars = nonzero_vars = []`. So (a) a sympy form that drops protected
   semantics at 0, or in the `|b| < 1e-6` band of `ProtectedDiv`, still "verifies";
   (b) no `is-nonzero`-guarded rule could fire in this dataset. The guarded rules
   below pay off in the live engine (where var ranges give facts), not on a re-run of
   this script as written.

Rows my family shrinks soundly with NO caller fact: **0, 6, 16, 19, 28** (−1 each), **21** (−2),
**26, 31** (−1). Guarded and reachable from a var-range fact: **4, 8, 25** (oz2 / oz1 nonzero),
**17** (col_2), **21, 26** only if P's factors oz1, oz5 are nonzero. Guarded but DEAD — the
guard target bottoms out in a `Sub` of variables nothing can prove nonzero: **1, 18**. Unsound
(sympy is wrong under protected semantics): **4, 8, 23, 24, 25, 31, and 17's PDiv step** — see the end.

Soundness vocabulary used below:
- **exact** — LHS and RHS agree at every real point including where either is NaN.
- **widening** — agree wherever LHS is defined, but RHS has a value where LHS is NaN.
  The crate guards these (`Inv (Inv x)`, `Div x x`), so I guard them too.

---

## R1. `1/x` is `Inv x` (raw only)

```lisp
; 1/x = Inv x. RAW Div only: both are NaN at x = 0, 1/x elsewhere.
; NOT ProtectedDiv: protected_div(1, x) is 0 for |x| < 1e-6 and ProtectedInv(0) is 1.
(rewrite (Div (Num 1.0) x) (Inv x) :ruleset rational)
; -1/x = -(1/x), exact in IEEE.
(rewrite (Div (Num -1.0) x) (Neg (Inv x)) :ruleset rational)
```

- Evidence: genes 416, 417, 573, 672, 756 (weighted 90). Gap row 21 (=573): 16 → 14
  (`Inv (Inv P)`); rows 26 (=672) and 31 (=756): 16 → 15; 416: 18 → 17; 417: 18 → 17. The second rule is
  equal-size (3 → 3) and only matters because it exposes `Inv (Inv P)` in row 26.
  With `is-nonzero P` the existing `Inv (Inv x) -> x` then takes 573 to 12 and 672 to 13.
- Soundness: exact. `Div` is `NaN if b == 0 else a/b`, `Inv` is `NaN if a == 0 else 1/a`.
- Termination: first rule removes one node, fires once per node. Second is directed
  (a `Num -1.0` numerator never reappears). No interaction with distribute: neither
  side is a sum or product.
- Not present: `rational.rs` has `Pow x -1 -> Inv x` but no `Div 1 x`.

## R2. `a * (1/b)` is `a / b` (raw only) — documented in `rational.rs`, never written

```lisp
; a * (1/b) = a/b. Both NaN at b = 0. RAW Inv only: a * ProtectedInv(0) is a,
; ProtectedDiv(a, 0) is 0.
(rewrite (Mul a (Inv b)) (Div a b) :ruleset rational)
(rewrite (Mul (Inv b) a) (Div a b) :ruleset rational)
```

- Evidence: genes 591, 628, 822 (weighted 34), −1 node each. The module doc of
  `rational.rs` lists "`a*(1/b) = a/b`" under point 3; the ruleset text does not contain it.
  12 more genes (weighted 308) have the `ProtectedInv` spelling — reachable only through R6.
- Soundness: exact over the reals (`a * (1/b)` vs `a/b` differ by an ulp in f64, inside
  the tests' 1e-9). One f64-only edge: `b` subnormal makes `1/b` overflow to inf.
- Termination: strictly −1 node. Overlap with the existing `(Mul (Inv a) (Inv b)) ->
  (Inv (Mul a b))`: on `Mul (Inv a) (Inv b)` both fire, giving `Div (Inv a) b` and
  `Inv (Mul a b)` in one class — finite, no new redex chain. With distribute's
  `(Mul (Inv p) (Num a)) -> (Mul (Num a) (Inv p))` it yields `Div (Num a) p`; still finite.

## R3. `0 - x` is `-x`

```lisp
(rewrite (Sub (Num 0.0) x) (Neg x) :ruleset algebra)
```

- Evidence: gap row 16 (`Sub (Log 1) (Mul ..)` folds to `Sub (Num 0.0) ..`): 8 → 7.
  Genes 271, 346, 415, 771.
- Soundness: exact (`0 - x == -x` in IEEE for every x, NaN and inf included; only the
  sign of zero differs and `0.0 == -0.0`).
- Termination: −1 node. Belongs in `algebra` beside `Sub x 0` so `denoise` gets it too.
- Not present in identities/wide/distribute (`wide` has `Sub a b <-> Add a (Neg b)`, which
  reaches it only with a zero-add fold, and `wide` is not in `smallest_form`).
- Coordination: told rules-sign I am taking this one; the follow-up
  `Neg (Mul a (Sub b c)) -> Mul a (Sub c b)` that closes the rest of row 16 is theirs.

## R4. A literal numerator absorbs the denominator's sign (raw AND protected)

```lisp
; c / (-x) = (-c) / x. Sound for the protected divide too: its guard is |b| < 1e-6
; and |-x| = |x| exactly, so both sides take the same branch.
(rewrite (Div (Num c) (Neg x)) (Div (Num (neg c)) x) :ruleset algebra)
(rewrite (ProtectedDiv (Num c) (Neg x)) (ProtectedDiv (Num (neg c)) x) :ruleset algebra)
; (-a) / (-b) = a / b, same argument.
(rewrite (Div (Neg a) (Neg b)) (Div a b) :ruleset algebra)
(rewrite (ProtectedDiv (Neg a) (Neg b)) (ProtectedDiv a b) :ruleset algebra)
```

- Evidence: gap row 28 (`ProtectedDiv (Num -1.0) (Neg in2)`): 25 → 24; genes 600, 629,
  768 (weighted 30), −1 each. `Div (Neg) (Neg)`: 0 genes — included for symmetry, drop it
  if you want evidence-only.
- Soundness: exact. `c / (-x) == (-c) / x` bit-for-bit in IEEE; zero branch gives 0 on
  both sides; raw gives NaN on both at x = 0.
- Termination: −1 node (−2 for the Neg/Neg pair). `(neg c)` is a primitive the crate already
  uses in distribute.
- WARNING for whoever generalises: `ProtectedInv (Neg x) -> Neg (ProtectedInv x)` is
  UNSOUND (`ProtectedInv(0) = 1`, the RHS gives −1).

## R5. Literal re-association folds that `fold_constant_subtrees` cannot see

`smallest_form` folds closed subtrees with the evaluator, so `Mul (Num) (Num)` never
survives (0 of 847). What survives is two literals separated by a non-constant:

```lisp
; (p * a) * b = p * (a*b) ; a + (b + q) = (a+b) + q   (a, b literals)
(rewrite (Mul (Mul p (Num a)) (Num b)) (Mul p (Num (* a b))) :ruleset rational)
(rewrite (Mul (Num a) (Mul (Num b) p)) (Mul (Num (* a b)) p) :ruleset rational)
(rewrite (Add (Num a) (Add (Num b) q)) (Add (Num (+ a b)) q) :ruleset rational)
(rewrite (Add (Add q (Num a)) (Num b)) (Add q (Num (+ a b))) :ruleset rational)
```

- Evidence — thin, stated honestly: gene 835 (count 1) `Mul (Mul (Mul g A) 0.271) 0.987`:
  10 → 8; gene 107 (count 30) `Add 1 (Add 1 X)`: 21 → 19 (sympy prints `+ 2.0`; not a
  "gap" row only because sympy's tree came out larger elsewhere).
- Soundness: exact over the reals; f64 re-association moves the last ulp, and `a*b` can
  overflow where `(p*a)*b` would not for tiny `p`. Same licence distribute already takes.
- Termination: −2 nodes each. The second rule duplicates `distribute.rs:75`; it is
  repeated here because **distribute is not in `rational_all`**, so `smallest_form`
  never sees it. In the parity Algebra family (distribute + rational co-saturated) it
  is a harmless duplicate.
- Mixed-order spellings (`Mul (Num a) (Mul p (Num b))`, 0 genes) deliberately omitted.

## R6. Protected reciprocal under a nonzero proof (GUARDED) + the guard facts it needs

```lisp
; ProtectedInv x = 1/x whenever x != 0 — that IS its definition. Same size; it
; lets every raw Inv rule (Inv(Inv x), R2, Mul x (Inv x)) reach the protected spelling.
(rewrite (ProtectedInv x) (Inv x) :when ((is-nonzero x)) :ruleset rational)
```

Guard propagation (in `GUARD_RELATIONS`, ruleset `guards`) — without these the fact
dies one level up and the rule above never chains:

```lisp
(rule ((is-nonzero a) (is-nonzero b) (= m (Mul a b))) ((is-nonzero m)) :ruleset guards)
(rule ((is-nonzero a) (is-nonzero b) (= m (Div a b))) ((is-nonzero m)) :ruleset guards)
(rule ((is-nonzero x) (= m (Inv x)))                  ((is-nonzero m)) :ruleset guards)
(rule ((is-nonzero x) (= m (Neg x)))                  ((is-nonzero m)) :ruleset guards)
(rule ((is-nonzero x) (= m (ProtectedInv x)))         ((is-nonzero m)) :ruleset guards)
(rule ((is-nonzero x) (= m (ProtectedSqrt x)))        ((is-positive m)) :ruleset guards)
(rule ((= m (ProtectedExp x)))                        ((is-positive m)) :ruleset guards)
```

- Reachability (guard target classified: a Var = needs a caller fact; Exp/ProtectedExp/literal
  = derived; anything over Sub/Add/Sin/Cos/Tanh = dead): Inv/ProtectedInv nests 8 genes fact
  (weighted 211: 40, 92, 102, 351, 418, 549, 765, 796), 10 dead (240); `Mul a (ProtectedInv b)`
  5 fact (122), 1 derived (gene 517, 30), 6 dead (156). The raw counts below overstate by about half.
- **Gap row 0 (count 30) falls to the `ProtectedExp` guard alone**: `Pow3 (Abs (ProtectedExp ..))`
  sheds the Abs through the existing `(Abs x) -> x :when is-positive`, −1, no caller fact.
  Exact: ProtectedExp is >= 0 on every branch including +inf.
- Evidence: `Inv (ProtectedInv x)` 8 genes (weighted 240: 40, 92, 299, 351, 418, 504, 549,
  765) (gap row 1, 15 → 13, is among them but DEAD); `ProtectedInv (Inv x)` 3 genes (61); `ProtectedInv
  (ProtectedInv x)` 7 genes (150); `Mul a (ProtectedInv b)` 12 genes (308). Gap row 19:
  `Mul (Sub ..) (ProtectedInv (ProtectedExp theta))` → `Div (Sub ..) (ProtectedExp theta)`,
  13 → 12, and this one needs NO caller fact (the ProtectedExp guard supplies it).
  Gap rows 4/25, 8 reduce the same way but only with `is-nonzero oz2` / `oz1` asserted.
- Soundness: for x != 0 `ProtectedInv` evaluates `1.0 / a`, identical to `Inv`. NaN in
  → NaN out on both. Unguarded it is unsound at exactly x = 0 (1 vs NaN). Each nest:
  `Inv(PInv 0) = 1`, `PInv(Inv 0) = NaN`, `PInv(PInv 0) = 1`, none equal to 0.
  The `ProtectedExp` / `ProtectedSqrt` guards hold over the reals and for finite f64;
  they share the existing `Exp` guard's blind spot (exp underflows to 0.0 below −745),
  and `ProtectedSqrt(±inf) = 0`. `ProtectedInv(x)` for nonzero x is nonzero except
  x = ±inf.
- **No analogue for `ProtectedDiv`.** `is-nonzero b` does not give `|b| >= 1e-6`
  (b = 1e-7: protected 0, raw a·1e7). The only sound lift is a literal denominator:

```lisp
(rewrite (ProtectedDiv a (Num k)) (Div a (Num k)) :when ((>= (abs k) 1e-6)) :ruleset rational)
```
  (`abs`, `>=` exist in `vendor-egglog/src/sort/f64.rs`.) 19 genes have `Div x (Num k)`
  shapes but it is equal-size — an enabler only; listed, not pushed.
- Termination: `ProtectedInv -> Inv` is one-way, equal size, once per node. Guard rules
  only add facts over existing nodes.

## R7. Reciprocal of a quotient (GUARDED)

```lisp
; 1/(a/b) = b/a for b != 0. At b = 0 the LHS is NaN and b/a is 0 — widening, so guarded.
(rewrite (Inv (Div a b)) (Div b a) :when ((is-nonzero b)) :ruleset rational)
```

- Evidence: raw/raw in 8 genes (180, 293, 458, 501, 506, 605, 628, 651) but only 2 are
  reachable (180 derived, 458 via a Var fact; weighted 60); the other 6 and gap row 18 are
  dead. Low value — keep or drop. Not for `ProtectedDiv` inside
  (`Inv(PDiv(a, tiny)) = Inv(0) = NaN` vs `tiny/a`) — and no guard we have fixes that.
- Termination: −1 node.

## R9. `(1/a)/b` is `1/(a*b)` (raw only, UNGUARDED)

```lisp
; (1/a)/b = 1/(a*b). Exact: a = 0 or b = 0 is NaN on both sides (Mul with a zero is 0,
; Inv 0 is NaN). Companion of the existing (Mul (Inv a) (Inv b)) -> (Inv (Mul a b)).
; NOT ProtectedDiv (Inv a) b: in the |b| < 1e-6 band that is 0, not 1/(ab).
(rewrite (Div (Inv a) b) (Inv (Mul a b)) :ruleset rational)
```

- Evidence: gap row 6 (count 30) `Div (Inv pc12) a65`; genes 116, 458. Equal size on its
  own (4 → 4); the shrink comes from R2 firing next: `Mul (Inv m) E -> Div E m`, row 6 −1.
- Termination: directed (Inv moves outward, never back); with R2 the class also gets
  `Div (Inv a) b` from `Mul (Inv a) (Inv b)` — this rule sends it to the same
  `Inv (Mul a b)` the existing rule produces, so the class closes.
- Partner `(Div a (Inv b)) -> (Mul a b) :when ((is-nonzero b))` is widening (b = 0: NaN vs 0);
  all 4 raw instances (416, 417, 450, 574) have dead guard targets. Not proposed.
- Test:
```rust
    #[test]
    fn div_of_inv_merges_raw_only() {
        assert!(proves_equal(
            r#"(Div (Inv (Var "a")) (Var "b"))"#,
            r#"(Inv (Mul (Var "a") (Var "b")))"#,
        ));
        assert!(!proves_equal(
            r#"(ProtectedDiv (Inv (Var "a")) (Var "b"))"#,
            r#"(Inv (Mul (Var "a") (Var "b")))"#,
        ));
        assert_sound(r#"(Div (Inv (Var "a")) (Var "b"))"#, &[("a", 1.3), ("b", -0.7)]);
    }
```

## R8. Like terms — low value, one gene

```lisp
(rewrite (Add x (Add x x)) (Mul (Num 3.0) x) :ruleset rational)
(rewrite (Add (Add x x) x) (Mul (Num 3.0) x) :ruleset rational)
```

- Evidence: gene 514 only (count 30), `Add theta (Add theta theta)`: 16 → 14. Exact in
  the reals (f64: `x+(x+x)` vs `3x` can differ by an ulp).
- I am NOT proposing `Add x x -> Mul 2 x`: all 9 occurrences have a leaf `x`
  (3 nodes → 3 nodes), so strict-shrink never takes it, and co-saturated with
  distribute's `Mul a (Add b c)` expansion it is a new redex source for no measured
  gain. `Add x (Mul (Num k) x)`: 0 genes. `Sub x x`, raw `Div x x` guarded: already in
  `identities.rs`; 0 surviving instances.

---

## Considered and rejected as unsound

| Rewrite (what sympy did) | Rows | Failing point |
|---|---|---|
| `ProtectedDiv 1 x -> ProtectedInv x` / `-> Inv x` | 31 | x = 0: 0 vs 1 / NaN; 0 < \|x\| < 1e-6: 0 vs 1/x |
| `ProtectedDiv 1 (Div 1 P) -> P` | 31 (gene 756) | \|P\| > 1e6 makes \|1/P\| < 1e-6 → LHS 0, RHS P. Wrong **even with `is-nonzero P`** |
| `ProtectedDiv x x -> 1` | genes 592, 635 | \|x\| < 1e-6: 0. `is-nonzero` does not help (x = 1e-7) |
| `Mul a (ProtectedInv b) -> ProtectedDiv a b` or `Div a b` unguarded | 4, 8, 25 | b = 0: LHS a, RHS 0 / NaN |
| `Div a (ProtectedInv b) -> Mul a b` | 23 | b = 0: LHS a, RHS 0 |
| `ProtectedDiv T (ProtectedInv S) -> Mul T S` | 24 | S = 0: LHS T, RHS 0; S > 1e6: LHS 0, RHS T·S |
| `ProtectedDiv (Pow2 (Pow2 c)) c -> Pow3 c` | 17 | 0 < \|c\| < 1e-6: LHS 0, RHS c³ (tiny but nonzero; and any factor-cancel through PDiv has this band) |
| `Inv (Inv x) -> x`, `Inv (Div a b) -> Div b a`, `Div a (Div b c) -> Div (Mul a c) b` unguarded | 17, 18, 21, 26, 28 | widening: LHS NaN at x / b / c = 0, RHS finite. Kept guarded, matching `identities.rs` |
| `ProtectedInv (Neg x) -> Neg (ProtectedInv x)` | — | x = 0: 1 vs −1 |
| `Mul (ProtectedDiv 1 b) c -> ProtectedDiv c b` | 3 | sound for finite c (both 0 in the band) but `0 * inf = NaN` vs 0 when c overflows; left to rules-sign with that caveat, since row 3 is mostly a sign flip |

Notes for other families: row 10 is not pure artifact — `Pow2 (Pow2 (Inv Bills))` is 4 nodes
against `Pow Bills (Num -4.0)` at 3, and `rational.rs:108` only rewrites the expanding
direction. Row 15 `Pow2 (Abs x) -> Pow2 x` is exact, −1, count 30, in no ruleset. Row 20's
`Abs (Neg (Abs T))` survives because `sympy_mined` is in neither `smallest_form` family.
Row 3 needs the equal-size enabler `(Mul (ProtectedDiv (Num c) x) y) -> (ProtectedDiv (Mul (Num c) y) x)`
before the sign rules can land its −1.

Enabler, listed only (agreed with rules-powers): `(Pow2 (Inv x)) -> (Inv (Pow2 x))`,
`(Pow3 (Inv x)) -> (Inv (Pow3 x))` — raw Inv, exact (NaN at 0 both sides), equal size, 5 raw
genes (164, 171, 180, 287, 341). Moves Inv outward so R2 and powers.md P1 can fire. The
row-10 reverse (`Pow2 (Pow2 (Inv x)) -> Pow x -4`) is NOT proposed: 2 genes, ulp-level only,
and it fights powers.rs:20-21 (see powers.md R9).
Cross-references: `Mul x x -> Pow2 x` is powers.md P5; `Pow2 (Abs x) -> Pow2 x` is powers.md P3.

Handed off: `Mul x x -> Pow2 x` (6 genes + rows 22/23) to rules-powers; Neg-through-Mul/Sub
flips (rows 2, 3, 4, 16 remainder) to rules-sign.

---

## Tests (for `src/ruleset/rational.rs`, same harness; R3/R4 cases go in `identities.rs` `each_rule_fires`)

```rust
    /// `proves_equal` with caller facts asserted before saturation.
    fn proves_equal_with(a: &str, b: &str, facts: &str) -> bool {
        let mut e = egraph();
        let prog = format!(
            "(let __a {a})\n(let __b {b})\n{facts}\n\
             (run-schedule (repeat {SAT_ITERS} (run guards) (run both)))\n(check (= __a __b))"
        );
        e.parse_and_run_program(None, &prog).is_ok()
    }

    #[test]
    fn one_over_x_is_inv_raw_only() {
        assert!(proves_equal(r#"(Div (Num 1.0) (Var "x"))"#, r#"(Inv (Var "x"))"#));
        assert!(proves_equal(r#"(Div (Num -1.0) (Var "x"))"#, r#"(Neg (Inv (Var "x")))"#));
        // protected_div(1, x) is 0 near zero; ProtectedInv(0) is 1. Must stay apart.
        assert!(!proves_equal(r#"(ProtectedDiv (Num 1.0) (Var "x"))"#, r#"(Inv (Var "x"))"#));
        assert!(!proves_equal(
            r#"(ProtectedDiv (Num 1.0) (Var "x"))"#,
            r#"(ProtectedInv (Var "x"))"#,
        ));
        assert_sound(r#"(Div (Num -1.0) (Var "x"))"#, &[("x", -2.3)]);
    }

    #[test]
    fn mul_by_inv_is_div_raw_only() {
        assert!(proves_equal(
            r#"(Mul (Var "a") (Inv (Var "b")))"#,
            r#"(Div (Var "a") (Var "b"))"#,
        ));
        assert!(proves_equal(
            r#"(Mul (Inv (Var "b")) (Var "a"))"#,
            r#"(Div (Var "a") (Var "b"))"#,
        ));
        // a * ProtectedInv(0) = a, ProtectedDiv(a, 0) = 0.
        assert!(!proves_equal(
            r#"(Mul (Var "a") (ProtectedInv (Var "b")))"#,
            r#"(ProtectedDiv (Var "a") (Var "b"))"#,
        ));
        assert_sound(r#"(Mul (Var "a") (Inv (Var "b")))"#, &[("a", 1.7), ("b", -0.4)]);
    }

    #[test]
    fn double_reciprocal_of_gap_row_21_needs_the_fact() {
        let (lhs, p) = (r#"(Div (Num 1.0) (Div (Num 1.0) (Var "p")))"#, r#"(Var "p")"#);
        assert!(proves_equal(lhs, r#"(Inv (Inv (Var "p")))"#));
        assert!(!proves_equal(lhs, p), "NaN at p = 0 on the left, 0 on the right");
        assert!(proves_equal_with(lhs, p, r#"(is-nonzero (Var "p"))"#));
    }

    #[test]
    fn literal_reassociation_folds() {
        assert!(proves_equal(
            r#"(Mul (Mul (Var "g") (Num 0.5)) (Num 4.0))"#,
            r#"(Mul (Var "g") (Num 2.0))"#,
        ));
        assert!(proves_equal(
            r#"(Add (Num 1.0) (Add (Num 1.0) (Var "x")))"#,
            r#"(Add (Num 2.0) (Var "x"))"#,
        ));
        assert_sound(r#"(Mul (Mul (Var "g") (Num 0.5)) (Num 4.0))"#, &[("g", 3.1)]);
    }

    #[test]
    fn protected_inv_lifts_only_under_nonzero() {
        let nest = r#"(Inv (ProtectedInv (Var "x")))"#;
        assert!(!proves_equal(nest, r#"(Var "x")"#), "Inv(ProtectedInv 0) is 1, not 0");
        assert!(proves_equal_with(nest, r#"(Var "x")"#, r#"(is-nonzero (Var "x"))"#));
        assert!(proves_equal_with(
            r#"(ProtectedInv (ProtectedInv (Var "x")))"#,
            r#"(Var "x")"#,
            r#"(is-nonzero (Var "x"))"#,
        ));
        // gap row 19: ProtectedExp is positive, so no caller fact is needed.
        assert!(proves_equal_with(
            r#"(Mul (Var "a") (ProtectedInv (ProtectedExp (Var "t"))))"#,
            r#"(Div (Var "a") (ProtectedExp (Var "t")))"#,
            "",
        ));
        // ProtectedDiv never lifts on a nonzero fact: b = 1e-7 is nonzero and gives 0.
        assert!(!proves_equal_with(
            r#"(ProtectedDiv (Var "a") (Var "b"))"#,
            r#"(Div (Var "a") (Var "b"))"#,
            r#"(is-nonzero (Var "b"))"#,
        ));
    }

    #[test]
    fn inv_of_quotient_is_guarded() {
        let (lhs, rhs) = (r#"(Inv (Div (Var "a") (Var "b")))"#, r#"(Div (Var "b") (Var "a"))"#);
        assert!(!proves_equal(lhs, rhs));
        assert!(proves_equal_with(lhs, rhs, r#"(is-nonzero (Var "b"))"#));
    }

    #[test]
    fn triple_like_term() {
        assert!(proves_equal(
            r#"(Add (Var "t") (Add (Var "t") (Var "t")))"#,
            r#"(Mul (Num 3.0) (Var "t"))"#,
        ));
    }
```

`identities.rs`, appended to the `each_rule_fires` table:

```rust
            (r#"(Sub (Num 0.0) (Var "x"))"#, r#"(Neg (Var "x"))"#),
            (r#"(ProtectedDiv (Num -1.0) (Neg (Var "x")))"#, r#"(ProtectedDiv (Num 1.0) (Var "x"))"#),
            (r#"(Div (Num 2.0) (Neg (Var "x")))"#, r#"(Div (Num -2.0) (Var "x"))"#),
            (r#"(ProtectedDiv (Neg (Var "a")) (Neg (Var "b")))"#, r#"(ProtectedDiv (Var "a") (Var "b"))"#),
```

Caveats on the tests: `proves_equal_with` runs `guards` explicitly because the
harness's `both` combined ruleset does not include it (the existing guarded rational
rules are therefore never exercised by the current tests — worth checking).
`protected_ops_are_inert` in `identities.rs` stays green: none of its four cases has a
literal numerator, a `Neg`, or a nonzero fact. The R4 extraction test assumes
`(Num 1.0)` prints as `1.0`; the first kill-guarded `cargo test` will tell.
