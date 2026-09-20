# Rule proposals — exp / log / roots / powers / Abs

Status: PROPOSAL. Nothing in `src/` was edited. **No cargo was run (battery
constraint) — every test below is unrun.** Evidence is from
`hff/notebooks/_ledgers/hof_dataset/{gaps,genes}.jsonl` (33 gap rows, 847
genes), counted on `fuller_math` with a real s-expression parser, not grep.
"w" = the `count`-weighted total (how many hall-of-fame individuals carry it).

## Read this first

1. **Only `algebra` and `powers` (+ `rational` in its own pass) are live.**
   `smallest_form` and `denoise` schedule `guards algebra powers`
   (`src/extract.rs:647,784`). `sympy_mined` is never loaded there, so its
   `Abs(Neg x)`, `Abs(Pow2 x)`, `Abs(Sqrt x)`, `Abs(Exp x)` rules are dead for
   the live path. That alone explains gap rows 1 and 20. `sympy_mined` cannot
   be scheduled whole (it holds size-INCREASING splits: `Abs(Mul a b) ->
   Mul(Abs a)(Abs b)`, `Sqrt(Mul ..)`). Every rule below names `algebra` or
   `guards` as its home.
2. **`gap_nodes` is measured on sympy's printed tree, not Math nodes.** Rows 0,
   6, 25, 32 have sympy going complex (`re/im/atan2`) because the hff renderer
   declares symbols without `real=True`; row 0's "20-node gap" is that
   artifact. Its real Math-node win is 1 (`Abs(ProtectedExp ..)`). Likewise the
   `Abs(Abs(oz5))` in rows 21/26/31 is the printer rendering `ProtectedSqrt` as
   `sqrt(Abs(..))`. Fixing the renderer (`real=True`) is an hff-side change
   that would erase most of those rows' measured gap on its own.
3. `Cbrt` and `Pow4` are not constructors in `MATH_DATATYPE`; no rules for them.
4. The textbook exp rules in the brief (`Sqrt(Exp x) = Exp(x/2)`,
   `Pow3(Exp x) = Exp(3x)`) are **size-increasing in Math nodes** (3 -> 4) and
   so can never be picked by `smallest_form`. See Rejected.

Soundness vocabulary used below: **bit-exact** = both sides return the same f64
under `src/eval.rs` for EVERY input including NaN, +-inf, +-0 (sign of zero
aside). Every proposed rule P1-P6 is bit-exact. Nothing proposed relies on the
"equal where both finite" escape hatch.

Gap-row coverage summary (Math nodes before -> after, on `fuller_math`):

| row | count | rule | nodes |
|----:|------:|------|------:|
| 0   | 30 | P1 `Abs(ProtectedExp ..)`            | 6 -> 5 |
| 1   | 30 | P1 `Abs(Pow2 ..)`                    | 15 -> 14 |
| 6   | 30 | P3 `ProtectedSqrt(Abs pc4)`          | 19 -> 18 |
| 15  | 30 | P3 `Pow2(Abs ..)`                    | 15 -> 14 |
| 18  | 30 | P3 `ProtectedLog(Abs angle)`         | 24 -> 23 |
| 20  | 29 | P2 `Abs(Neg(Abs t))` + shipped `Abs(Abs)` | 12 -> 10 |
| 21, 26, 31 | 26+3+1 | P4 `ProtectedSqrt(Mul (Abs a) b)` | 16 -> 15 |
| 22  | 15 | P5 `Mul x x` then P1 `Abs(Pow2)`     | 19 -> 17 |
| 23  | 15 | P5 `Mul x x`                         | 19 -> 18 |

11 of 33 gap rows. The other 22 are sign canonicalisation (`rules-sign`) or
sympy restructurings with no Math-node win (rows 2, 3, 17, 24), or rejected
below (rows 24, 32).

---

## P1. `is-nonneg` guard + `Abs x -> x` — the biggest lever

`is-positive (Pow2 x)` needs `is-nonzero x`, so today `Abs(Pow2 x)` never
sheds. `Abs` only needs `>= 0`.

```lisp
; ---- in GUARD_RELATIONS (src/expr.rs) ----
(relation is-nonneg (Math))
; seeds: >= 0 (or NaN) on every input
(rule ((= e (Pow2 x)))          ((is-nonneg e)) :ruleset guards)
(rule ((= e (Abs x)))           ((is-nonneg e)) :ruleset guards)
(rule ((= e (Sqrt x)))          ((is-nonneg e)) :ruleset guards)
(rule ((= e (ProtectedSqrt x))) ((is-nonneg e)) :ruleset guards)
(rule ((= e (Exp x)))           ((is-nonneg e)) :ruleset guards)
(rule ((= e (ProtectedExp x)))  ((is-nonneg e)) :ruleset guards)
(rule ((= e (Num n)) (>= n 0.0)) ((is-nonneg e)) :ruleset guards)
(rule ((is-positive m))         ((is-nonneg m)) :ruleset guards)
; propagation: sign-preserving ops over nonneg operands
(rule ((is-nonneg a) (is-nonneg b) (= m (Mul a b)))          ((is-nonneg m)) :ruleset guards)
(rule ((is-nonneg a) (is-nonneg b) (= m (Add a b)))          ((is-nonneg m)) :ruleset guards)
(rule ((is-nonneg a) (is-nonneg b) (= m (Div a b)))          ((is-nonneg m)) :ruleset guards)
(rule ((is-nonneg a) (is-nonneg b) (= m (ProtectedDiv a b))) ((is-nonneg m)) :ruleset guards)
(rule ((is-nonneg x) (= e (Pow3 x)))         ((is-nonneg e)) :ruleset guards)
(rule ((is-nonneg x) (= e (Inv x)))          ((is-nonneg e)) :ruleset guards)
(rule ((is-nonneg x) (= e (ProtectedInv x))) ((is-nonneg e)) :ruleset guards)
(rule ((is-nonneg x) (= e (Tanh x)))         ((is-nonneg e)) :ruleset guards)
(rule ((is-nonneg b) (= e (Pow b p)))        ((is-nonneg e)) :ruleset guards)

; ---- in ALGEBRA_RULESET (src/ruleset/identities.rs), next to the is-positive one ----
; |x| = x for x >= 0. Bit-exact: Abs(NaN) = NaN, Abs(+inf) = +inf.
(rewrite (Abs x) x :when ((is-nonneg x)) :ruleset algebra)
```

**Evidence.** `Abs(t)` with `t` structurally nonneg in fuller's OUTPUT: 41
genes / 42 occurrences / w700. Direct: `Abs(Pow2)` 12 genes w295 (gap row 1),
`Abs(ProtectedExp)` 9 genes w152 (gap row 0), `Abs(ProtectedSqrt)` 5 genes w91.
16 genes (w192) are INDIRECT and no finite list of `Abs(X)` rewrites covers
them — this is why it is a relation:
`Abs(ProtectedInv(ProtectedInv(Pow2 ..)))`, `Abs(Pow3(Abs ..))` (3 genes),
`Abs(Mul (ProtectedSqrt in1) (Abs (Sin in2)))`, `Abs(Tanh(Pow (ProtectedSqrt ..) ..))`
(3 genes), `Abs(Div (Pow2 ..) (Exp ..))` (4 genes), `Abs(Inv(Inv(ProtectedExp ..)))`.
Each sheds exactly 1 node. It also completes a shipped chain:
`Sqrt(Pow2(Pow2 x)) -> Abs(Pow2 x) -> Pow2 x`.

**Soundness (bit-exact).** `Abs` is the identity on `[0,+inf]` and on NaN, so
the claim needed is only "value is never in `[-inf,0)`" — NaN is never a
counterexample. Seeds: `a*a` is `>=0`, `+inf` or NaN; `Sqrt` is `>=0` or NaN;
`ProtectedSqrt` is `sqrt|a|` or `0.0`; `Exp`/`ProtectedExp` are in `[0,+inf]`
(underflow to `0.0` is still nonneg — unlike `is-positive`, this guard has NO
underflow caveat). Propagation: product/sum/quotient of values in `[0,+inf]` is
in `[0,+inf]` or NaN (`0*inf`, `inf/inf`, `Div` by 0); `ProtectedDiv` returns
`0.0` or `a/b`; `ProtectedInv` returns `1.0` or `1/a`; `a*a*a`, `tanh`,
`powf(b>=0, p)` (or the evaluator's NaN at `0^neg`) keep the sign. `Sub`, `Neg`,
`Log`, `Sin/Cos/Tan` deliberately do not propagate.

**Syntax to verify at build time:** the shipped guards use only `>` / `<` on
f64. If egglog 2.0's f64 sort lacks `>=`, write the literal seed as two rules
(`(> n 0.0)` and `(= n 0.0)`), or drop it — `is-positive => is-nonneg` already
covers positive literals.

**Termination.** Guard rules only add facts over EXISTING e-classes (no new
terms); at most one fact per class. The rewrite removes a node. No growth.

**Already present?** No `is-nonneg` anywhere in `src/`. `Abs(Pow2)`,
`Abs(Sqrt)`, `Abs(Exp)` exist only in dead-for-live `sympy_mined`; they become
redundant under P1 and can stay where they are.

**Test** (style of `identities.rs::abs_sheds_only_under_positivity`):

```rust
/// |x| sheds on a structurally non-negative argument with NO caller fact —
/// including through sign-preserving ops, which no fixed Abs(X) list reaches.
#[test]
fn abs_sheds_on_derived_nonneg() {
    let cases: &[(&str, &str)] = &[
        (r#"(Abs (Pow2 (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
        (r#"(Abs (ProtectedExp (Var "x")))"#, r#"(ProtectedExp (Var "x"))"#),
        (r#"(Abs (ProtectedSqrt (Var "x")))"#, r#"(ProtectedSqrt (Var "x"))"#),
        // gap row 0
        (
            r#"(Pow3 (Abs (ProtectedExp (Sqrt (ProtectedExp (Var "oz5"))))))"#,
            r#"(Pow3 (ProtectedExp (Sqrt (ProtectedExp (Var "oz5")))))"#,
        ),
        // indirect: nonneg flows through ProtectedInv, Pow3, Mul
        (
            r#"(Abs (ProtectedInv (ProtectedInv (Pow2 (Var "x")))))"#,
            r#"(ProtectedInv (ProtectedInv (Pow2 (Var "x"))))"#,
        ),
        (r#"(Abs (Pow3 (Abs (Var "x"))))"#, r#"(Pow3 (Abs (Var "x")))"#),
        (
            r#"(Abs (Mul (ProtectedSqrt (Var "a")) (Abs (Sin (Var "b")))))"#,
            r#"(Mul (ProtectedSqrt (Var "a")) (Abs (Sin (Var "b"))))"#,
        ),
        // Sqrt(x^4) = x^2, via the shipped Sqrt(Pow2 y) -> Abs y
        (r#"(Sqrt (Pow2 (Pow2 (Var "x"))))"#, r#"(Pow2 (Var "x"))"#),
    ];
    for (input, expected) in cases {
        assert_eq!(simplify(input).unwrap(), *expected, "{input}");
    }
    // Must NOT shed where the sign is unknown.
    for input in [
        r#"(Abs (Pow3 (Var "x")))"#,
        r#"(Abs (Sub (Pow2 (Var "x")) (Var "y")))"#,
        r#"(Abs (ProtectedLog (Var "x")))"#,
        r#"(Abs (Tanh (Var "x")))"#,
    ] {
        assert_eq!(simplify(input).unwrap(), input, "unsound shed: {input}");
    }
}
```

## P2. `Abs(Neg x) -> Abs x`

```lisp
; |-x| = |x|. Bit-exact.
(rewrite (Abs (Neg x)) (Abs x) :ruleset algebra)
```

Evidence: 3 genes / w89; gap row 20 `Abs(Neg(Abs(Tanh ..)))`: 12 -> 11, then
the shipped `Abs(Abs x)` takes it to 10 (the full measured gap). Soundness:
IEEE negation only flips the sign bit. Termination: removes one node. Present
only in `sympy_mined` (not live) — move/copy to `algebra`.

```rust
#[test]
fn abs_absorbs_neg() {
    assert_eq!(simplify(r#"(Abs (Neg (Var "x")))"#).unwrap(), r#"(Abs (Var "x"))"#);
    // gap row 20 core
    assert_eq!(
        simplify(r#"(Abs (Neg (Abs (Tanh (Var "x")))))"#).unwrap(),
        r#"(Abs (Tanh (Var "x")))"#
    );
}
```

## P3. Even heads absorb `Abs` / `Neg` of their argument

```lisp
; Pow2, ProtectedSqrt, ProtectedLog depend only on |x| (and on finiteness /
; zeroness, which Abs and Neg preserve). All bit-exact, all remove one node.
(rewrite (Pow2 (Abs x)) (Pow2 x) :ruleset algebra)
(rewrite (Pow2 (Neg x)) (Pow2 x) :ruleset algebra)
(rewrite (ProtectedSqrt (Abs x)) (ProtectedSqrt x) :ruleset algebra)
(rewrite (ProtectedSqrt (Neg x)) (ProtectedSqrt x) :ruleset algebra)
(rewrite (ProtectedLog (Abs x)) (ProtectedLog x) :ruleset algebra)
(rewrite (ProtectedLog (Neg x)) (ProtectedLog x) :ruleset algebra)
```

Evidence (genes / w): `Pow2(Abs)` 4 / 60 (gap row 15), `ProtectedSqrt(Abs)`
7 / 96 (gap row 6), `ProtectedLog(Abs)` 6 / 146 (gap row 18),
`ProtectedSqrt(Neg)` 1 / 30, `ProtectedLog(Neg)` 1 / 30. `Pow2(Neg)` has 0
surviving instances today (included for symmetry; drop if you want evidence-only).

Soundness against `eval.rs`: `Pow2` is `a*a`, and `(-a)*(-a)` / `|a|*|a|` are
the same f64 (NaN, inf included). `ProtectedSqrt x = if finite(x) sqrt|x| else
0`: `Abs`/`Neg` change neither `|x|` nor finiteness (NaN stays NaN -> 0.0 on
both sides). `ProtectedLog x = if !finite(x) || x==0 {+inf} else ln|x|`: same
argument, and `|x|==0 iff x==0`. **These are rules for the PROTECTED ops only**
— raw `Sqrt(Abs x)` / `Log(Abs x)` are different functions and get no rule
(see Rejected R4). This does not violate "protected ops are inert": the
protected op stays, only its argument shrinks; `protected_ops_are_inert`'s four
cases are untouched.

Termination: each strictly removes a node; no RHS creates a new redex.

```rust
#[test]
fn even_heads_absorb_abs_and_neg() {
    let cases: &[(&str, &str)] = &[
        (r#"(Pow2 (Abs (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
        (r#"(Pow2 (Neg (Var "x")))"#, r#"(Pow2 (Var "x"))"#),
        (r#"(ProtectedSqrt (Abs (Var "x")))"#, r#"(ProtectedSqrt (Var "x"))"#),
        (r#"(ProtectedSqrt (Neg (Var "x")))"#, r#"(ProtectedSqrt (Var "x"))"#),
        (r#"(ProtectedLog (Abs (Var "x")))"#, r#"(ProtectedLog (Var "x"))"#),
        (r#"(ProtectedLog (Neg (Var "x")))"#, r#"(ProtectedLog (Var "x"))"#),
    ];
    for (input, expected) in cases {
        assert_eq!(simplify(input).unwrap(), *expected, "{input}");
    }
    // The RAW ops are different functions (NaN vs +inf/0.0 at the edges):
    // they must not pick up the protected rule.
    for input in [r#"(Sqrt (Abs (Var "x")))"#, r#"(Log (Abs (Var "x")))"#] {
        assert_eq!(simplify(input).unwrap(), input, "raw op rewritten: {input}");
    }
}
```

Add an `eval`-level check too, because the claim is bit-exactness at the edges
(`eval.rs` test style; `Exp(1000) = +inf`, `Sqrt(-4) = NaN`):

```rust
#[test]
fn protected_even_heads_ignore_sign_even_on_non_finite() {
    let e = env(&[("x", 1000.0), ("n", -4.0), ("z", 0.0)]);
    for arg in [r#"(Var "n")"#, r#"(Var "z")"#, r#"(Exp (Var "x"))"#, r#"(Sqrt (Var "n"))"#] {
        for head in ["ProtectedSqrt", "ProtectedLog", "Pow2"] {
            let plain = eval(&format!("({head} {arg})"), &e).unwrap();
            for wrap in ["Abs", "Neg"] {
                let got = eval(&format!("({head} ({wrap} {arg}))"), &e).unwrap();
                assert!(got == plain || (got.is_nan() && plain.is_nan()), "{head}({wrap} {arg})");
            }
        }
    }
}
```

## P4. Same absorption through one product / quotient factor

Only the instantiations that occur in the data (11 genes / w240):

```lisp
; |(|a| * b)| = |a * b|: a sign flip of one factor changes only the sign of
; the product/quotient, never its magnitude, finiteness, zeroness or NaN-ness.
(rewrite (ProtectedSqrt (Mul (Abs a) b)) (ProtectedSqrt (Mul a b)) :ruleset algebra)
(rewrite (ProtectedSqrt (Mul a (Abs b))) (ProtectedSqrt (Mul a b)) :ruleset algebra)
(rewrite (ProtectedSqrt (ProtectedDiv a (Abs b))) (ProtectedSqrt (ProtectedDiv a b)) :ruleset algebra)
(rewrite (ProtectedLog (Div (Abs a) b)) (ProtectedLog (Div a b)) :ruleset algebra)
(rewrite (Pow2 (Mul (Abs a) b)) (Pow2 (Mul a b)) :ruleset algebra)
(rewrite (Pow2 (Mul a (Abs b))) (Pow2 (Mul a b)) :ruleset algebra)
(rewrite (Pow2 (ProtectedDiv a (Neg b))) (Pow2 (ProtectedDiv a b)) :ruleset algebra)
(rewrite (Abs (ProtectedDiv (Neg a) b)) (Abs (ProtectedDiv a b)) :ruleset algebra)
```

Evidence: `ProtectedSqrt(Mul (Abs oz5) (Exp oz3))` — gap rows 21, 26, 31
(26+3+1), 16 -> 15 each, the whole measured gap. `Pow2(Mul (Abs ..) oz1)` 2
genes w60; `Pow2(ProtectedDiv .. (Neg ..))` 1 / 30;
`ProtectedSqrt(ProtectedDiv .. (Abs ..))` 1 / 30; `ProtectedLog(Div (Abs ..) ..)`
2 / 30; `Abs(ProtectedDiv (Neg ..) ..)` 1 / 30.

Soundness (bit-exact): for `Mul`/`Div` IEEE sign is the XOR of operand signs and
is computed independently of magnitude, so flipping a factor's sign flips only
the result's sign; `Div`'s `b == 0.0 -> NaN` test is sign-blind. For
`ProtectedDiv` the threshold test is on `|b|`, which `Abs`/`Neg` leave alone,
so the `0.0` branch is taken on exactly the same inputs. The outer head then
discards the sign (P3's argument).

Termination: each removes one node. **If you want this general instead of 8
hand-picked shapes** (the full cross product is 4 heads x 3 ops x 2 sides x 2
wrappers = 48), use a relation — 2 seeds + 6 propagations + 4 heads:

```lisp
(relation same-mag (Math Math))   ; |lhs| = |rhs|, same finiteness/NaN
(rule ((= m (Abs x))) ((same-mag m x)) :ruleset algebra)
(rule ((= m (Neg x))) ((same-mag m x)) :ruleset algebra)
(rule ((= m (Mul a b)) (same-mag a a2)) ((same-mag m (Mul a2 b))) :ruleset algebra)
(rule ((= m (Mul a b)) (same-mag b b2)) ((same-mag m (Mul a b2))) :ruleset algebra)
; ... same pair for Div and ProtectedDiv ...
(rule ((= e (Pow2 m)) (same-mag m n)) ((union e (Pow2 n))) :ruleset algebra)
; ... same for ProtectedSqrt, ProtectedLog, Abs ...
```

This one CREATES terms (`(Mul a2 b)`), each strictly smaller than an existing
one, at most 2^k per product chain with k wrapped factors. Bounded, but it is
new e-nodes in the live denoise graph — I recommend the 8 explicit rules first
and the relation only if a re-scan shows a long tail. Flagged, not recommended.

```rust
#[test]
fn even_heads_absorb_abs_through_a_product() {
    // gap rows 21 / 26 / 31
    assert_eq!(
        simplify(r#"(ProtectedSqrt (Mul (Abs (Var "a")) (Exp (Var "b"))))"#).unwrap(),
        r#"(ProtectedSqrt (Mul (Var "a") (Exp (Var "b"))))"#
    );
    assert_eq!(
        simplify(r#"(Pow2 (Mul (Abs (Var "a")) (Var "b")))"#).unwrap(),
        r#"(Pow2 (Mul (Var "a") (Var "b")))"#
    );
    assert_eq!(
        simplify(r#"(Pow2 (ProtectedDiv (Var "a") (Neg (Var "b"))))"#).unwrap(),
        r#"(Pow2 (ProtectedDiv (Var "a") (Var "b")))"#
    );
    // NOT through a sum: |(|a| + b)| != |a + b|.
    let sum = r#"(ProtectedSqrt (Add (Abs (Var "a")) (Var "b")))"#;
    assert_eq!(simplify(sum).unwrap(), sum);
}
```

## P5. `Mul x x -> Pow2 x`

```lisp
; x * x = x^2. Bit-exact: eval's Pow2 IS `a * a`.
(rewrite (Mul x x) (Pow2 x) :ruleset algebra)
```

Evidence: 6 genes / w95, all surviving in fuller's output
(`Mul (Var "Bills") (Var "Bills")`, `In4`, `in4`, `col_2` x2). Gap rows 22 and
23 (15 each): `Abs(Mul col_2 col_2)` -> `Abs(Pow2 col_2)` -> (P1) `Pow2 col_2`,
19 -> 17; row 23 `ProtectedInv(Mul col_2 col_2)` 19 -> 18. Saves `|x|` nodes,
and it is what exposes the square to P1/P3 and to the shipped
`Sqrt(Pow2)`/`ProtectedSqrt(Pow2)` rules.

Termination: strictly size-reducing, directed. Do NOT add the reverse. Interaction
to check when it lands: `rational`'s binomial expansion emits `Mul a b` with
distinct a, b only, and `distribute` is not co-scheduled with `algebra` in
`smallest_form`/`denoise` — but the parity `Family::Algebra` schedule runs
`contract` (algebra+powers) then `expand` (distribute); run the kill-guarded
parity once after adding it. Not present (`grep "(Mul x x)"` in `src/`: nothing).

```rust
#[test]
fn self_product_is_a_square() {
    assert_eq!(simplify(r#"(Mul (Var "x") (Var "x"))"#).unwrap(), r#"(Pow2 (Var "x"))"#);
    // gap row 22 core: the square then sheds its Abs (needs P1).
    assert_eq!(
        simplify(r#"(Abs (Mul (Var "x") (Var "x")))"#).unwrap(),
        r#"(Pow2 (Var "x"))"#
    );
    // and feeds the shipped root rule
    assert_eq!(simplify(r#"(Sqrt (Mul (Var "x") (Var "x")))"#).unwrap(), r#"(Abs (Var "x"))"#);
}
```

## P6. `Exp(ProtectedLog x) -> Abs x`, guarded nonzero

```lisp
; exp(ln|x|) = |x| for x != 0. At x = 0 protected_log is +inf and the left
; side is +inf, not 0 — hence the guard. Never fires on a bare variable.
(rewrite (Exp (ProtectedLog x)) (Abs x) :when ((is-nonzero x)) :ruleset powers)
(rewrite (ProtectedExp (ProtectedLog x)) (Abs x) :when ((is-nonzero x)) :ruleset powers)
```

Evidence: `Exp(ProtectedLog ..)` 3 genes / w60, `ProtectedExp(ProtectedLog col_2)`
1 / w30. 3 -> 2 nodes each. **None fires today without a caller fact** — every
instance is over a bare variable or a `Pow3`/`Pow` of one, so the win arrives
only through `nonzero_vars`. Lowest priority of the six; included because it
is the sound form of the `Exp(Log x)` rule the brief asked about.

Soundness: for finite nonzero x, `exp(ln|x|)` equals `|x|` to within 1 ulp — NOT
bit-exact, the same rounding class as the shipped `Exp(Log x) -> x` and
`Pow2(Sqrt p) -> p`. At x = +-inf / NaN: `ProtectedLog = +inf`, `Exp(+inf) =
ProtectedExp(+inf) = +inf`; `Abs` gives +inf for +-inf (equal) but NaN for NaN
(`+inf` vs NaN — both non-finite, but distinguishable under a following
`Tanh`). If that NaN case is unacceptable, drop P6; it is the only proposal
with any edge disagreement and I would not fight for it.

```rust
#[test]
fn exp_of_protected_log_is_abs_only_when_nonzero() {
    let bare = simplify(r#"(Exp (ProtectedLog (Var "x")))"#, "");
    assert_eq!(bare, r#"(Exp (ProtectedLog (Var "x")))"#, "x = 0 gives +inf, not 0");
    let got = simplify(r#"(Exp (ProtectedLog (Var "x")))"#, r#"(is-nonzero (Var "x"))"#);
    assert_eq!(got, r#"(Abs (Var "x"))"#);
    assert_sound(r#"(Exp (ProtectedLog (Var "x")))"#, r#"(is-nonzero (Var "x"))"#, &[("x", -2.5)]);
}
```

(`powers.rs` test helpers; note its `simplify` runs `(run powers)` only — the
guard facts here are caller-asserted so that is sufficient.)

---

## Considered and REJECTED

**R1. `Sqrt(Exp x) -> Exp(x/2)`, `Pow2(Exp x) -> Exp(2x)`, `Pow3(Exp x) ->
Exp(3x)` (and the ProtectedExp spellings).** Two independent reasons.
(a) Size: `Sqrt(Exp x)` is 2+|x| nodes, `Exp(Mul (Num 0.5) x)` is 3+|x| —
every one of these GROWS the term, so `smallest_form` would never select it;
they only matter as an intermediate, and the data shows no gene where they
unlock a later collapse. (Row 0's sympy form `exp(3*exp(oz5/2))` is 8 Math
nodes against fuller's 6.) (b) Overflow: for x in (709.78, 1419] `Exp x = +inf`
so `Sqrt(Exp x) = +inf` while `Exp(x/2)` is finite (~1e154..1e308); with
`ProtectedSqrt` it is `0.0` vs finite. That is a disagreement at FINITE inputs
between a finite and a non-finite value — the engine would score one and
reject the other. For `Pow2/Pow3` the two sides overflow at the same x (bar a
denormal band), so those are merely useless, not unsound. 12 + 11 + 9 genes
carry `sqrt(exp ..)` shapes; none gets smaller.

**R2. `Log(ProtectedExp y) -> y`, `ProtectedLog(Exp y) -> y`,
`ProtectedLog(ProtectedExp y) -> y`** (2 + 2 + 1 genes; rows 20, 32). Fails at
finite y > 709.78 (`+inf` vs y) and y < -745 (underflow: NaN / `+inf` vs y),
and at non-finite y (`+inf` vs NaN/-inf). The shipped raw `Log(Exp x) -> x` has
the same overflow hole, so there IS precedent — but
`identities.rs::protected_ops_are_inert` asserts `ProtectedLog(ProtectedExp x)`
must NOT collapse, which I read as the owner's decision. Row 32's argument is
`Tanh(..)` in [-1, 1], where the identity is exact for finite input; a
bounded-argument guard (`is-unit-bounded`, seeded by Tanh/Sin/Cos) would make
it sound there except for NaN input. One gene, count 1 — not worth a relation.
Row 20's measured 1-node gap is P2, not this.

**R3. `Pow2(ProtectedSqrt x) -> Abs x`** (3 genes / w66). Wrong at x = +-inf
(`0.0` vs `+inf`) and NaN (`0.0` vs NaN). There is no `is-finite` guard and a
variable cannot be proven finite after upstream overflow. The unguarded raw
`Pow2(Sqrt x) -> x` stays rejected for the documented reason (NaN vs negative).

**R4. `Sqrt(Abs x) -> ProtectedSqrt x` (8 genes / w59, gap row 24),
`Log(Abs x) -> ProtectedLog x` (2 genes).** Tempting 1-node wins, but raw and
protected differ at the edges: `Sqrt(+inf) = +inf` vs `0.0`; `Log(0) = NaN` vs
`+inf`. Mapping raw onto protected is exactly what CLAUDE.md forbids.

**R5. `Inv(ProtectedInv y) -> y`, `ProtectedInv(ProtectedInv y) -> y`** (8 + 7
genes). At y = 0: `ProtectedInv 0 = 1`, so the left side is 1, not 0. Unsound at
a finite point. A `:when (is-nonzero y)` version is sound and belongs to the
rational family, not here.

**R6. `Exp(ProtectedLog x) -> Abs x` UNGUARDED.** x = 0 gives `+inf` vs `0`. Kept
only guarded (P6).

**R7. Equal-size canonicalisations** — `Inv(ProtectedExp x)` /
`ProtectedInv(Exp x)` -> `exp(-x)` (20 genes / w393), `ProtectedSqrt(Pow3 x)` <->
`Pow3(ProtectedSqrt x)` (24 genes), `Pow2(Pow3 x)` <-> `Pow3(Pow2 x)` (33
genes), `Pow(Exp a) b -> Exp(Mul a b)`. No node saved; several are
bidirectional by nature and would have to be run as such to be useful, which is
the blow-up pattern this crate has already been bitten by. `ProtectedInv(Exp x)`
also breaks on underflow (x < -745: `Exp x = 0.0`, `ProtectedInv 0 = 1`, while
`Exp(Neg x) = +inf`). Skip all.

**R8. `is-positive (ProtectedExp x)` guard seed.** Would unlock
`Div(ProtectedExp, ProtectedExp)` cancellation and, with a "nonneg + positive is
positive" rule, the one gene with `Exp(Log(Add (Pow2 ..) (Abs (ProtectedExp ..))))`.
It inherits the shipped `Exp`-positivity caveat (underflow to 0.0, and `+inf/+inf
= NaN` vs 1) and the data shows 0 genes with `Div(Exp, Exp)` / `Mul(Exp, Exp)`
left in fuller's output. P1's `is-nonneg` already covers every
`Abs(ProtectedExp)` instance bit-exactly, so this is not needed. Not proposed.

**R9. `Pow2(Pow2(Inv x))` / `Inv(Pow2(Pow2 x))` -> `Pow x (Num -4.0)`** (raised
by rules-rational for gap row 10; 4 nodes -> 3). `rational.rs:108` states the
equality but only ever fires from the `Pow` side, so a gene that arrives as
nested squares never gets the 3-node spelling. Whole-dataset evidence: 2 genes
(`Pow2(Pow2(Inv Bills))` w30, `Inv(Pow3(Pow3 ..))` w30) — and row 10's measured
gap is the sign flip, not this. `powf(x,-4)` vs `((1/x)^2)^2` agree at 0 (NaN
both, by eval's `0^neg` rule), at +-inf (0 both) and on overflow, but differ by
a few ulp on finite x — not bit-exact. It also trades dedicated constructors
for a general `Pow` with a literal, which the crate otherwise normalises AWAY
from (`powers.rs:20-21`). One node on two genes; not proposed. If wanted, it
belongs in `rational` as the reverse of lines 102-112, needing rational's
`Pow2 (Inv x) -> Inv (Pow2 x)` enabler first.

## Suggested landing order

P5, P2, P3 (pure one-line rewrites, bit-exact) -> P1 (guard relation; biggest
win, w700) -> P4 -> P6 (optional). After each: `RUSTFLAGS="-D warnings" cargo
test`, clippy, and a kill-guarded parity run.
