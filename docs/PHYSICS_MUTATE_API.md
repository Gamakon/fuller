# `physics_mutate` — physics-prior mutation generator

A **one-to-many generator**. Give it one gene; it returns a proliferation of
physics-shaped candidate genes — structural mutations that bias the form toward
what real physical laws look like (squared distances, axis-aligned coordinate
pairs, inverse-square, stripped wallpaper, …).

**It is NOT denoise.** Denoise tidies (behaviour-preserving identities). This
*changes* what the gene computes, on purpose, to seed the GA with forms it would
rarely reach by random mutation. It does **no evaluation, no scoring, no
fitness** — it only generates. Your GA + HFF select.

---

## 1. Install

```bash
cd /Users/andrewmorgan/Dev/kaito/fuller
maturin develop --release
python -c "from fuller import physics_mutate; print('ok')"
```

## 2. Signature

```python
from fuller import physics_mutate

candidates: list[dict] = physics_mutate(
    expr,            # str  — one gene, as a Math s-expression (see §4)
    paired_groups,   # list[list[str]] — coordinate axes (see §5)
    n=10,            # int  — max candidates to RETURN (>=1). default 10
    seed=0,          # int  — reproducible random sample
)
```

Returns a `list` of dicts, each:

```python
{
    "expr": str,          # the mutated gene, a Math s-expression
    "rule": str,          # which rule produced it: "A1" | "A2" | "E1" | "F" | ...
    "speculative": bool,  # True = structural leap (see §6)
}
```

## 3. Semantics

- **Full internal generation, then sample.** Internally it generates *all*
  distinct candidates — every rule, at every matching site, composed (outputs
  fed back through the rules) and deduplicated to a fixpoint, bounded by an
  internal safety cap (2000). It then returns up to `n` of them by **uniform
  random sample**. The cap limits what is *returned*, never what is generated.
- `n` ≥ 1. Ask for 3 → get 3 (if ≥3 generated). Ask for 50 → up to 50. Ask for
  1 → exactly 1.
- `seed` → deterministic. Same `(expr, paired_groups, n, seed)` ⇒ identical
  list, every run, every machine.
- The input gene itself is never returned, and all candidates are distinct.
- **No data crosses this boundary.** It never sees your training rows — correct,
  because selection is the engine's job, not the generator's.

## 4. The Math s-expression grammar

A gene is an s-expression over the `Math` sort. Leaves and operators:

```
leaf:   (Num <float>)              numeric literal, e.g. (Num 1.0)
        (Var "<name>")             variable, e.g. (Var "x1")

binary: (Add a b) (Sub a b) (Mul a b) (Div a b) (Pow a b)
unary:  (Neg a) (Sin a) (Cos a) (Tan a) (Tanh a) (Log a) (Exp a)
        (Sqrt a) (Abs a) (Pow2 a) (Pow3 a) (Inv a)
protected (engine pset semantics):
        (ProtectedSqrt a) (ProtectedLog a) (ProtectedExp a)
        (ProtectedInv a) (ProtectedDiv a b)
```

Example — `m1·m2 / (x2 − y1)`:

```
(Div (Mul (Var "m1") (Var "m2")) (Sub (Var "x2") (Var "y1")))
```

If you hold genes as karva chromosomes, use `denoise_karva`'s pset mapping (see
`USAGE.md`) to get to/from this form; `physics_mutate` works on the Math string.

## 5. `paired_groups` — declaring coordinate axes

The distance-family rules need to know which variables are the *same physical
quantity on different bodies* (so they can re-pair them onto an axis). Pass one
list per axis:

```python
paired_groups = [
    ["x1", "x2"],   # the x-coordinate of body 1, body 2
    ["y1", "y2"],   # y-coordinate
    ["z1", "z2"],   # z-coordinate
]
```

Index position matters: `groups[axis][i]` is body *i*'s component on that axis.
Rule A1 uses this to turn a cross-axis difference `(x2 − y1)` into same-axis
differences `(x2 − x1)` and `(y2 − y1)`. Pass `[]` to disable the
distance-family rules (other rules still fire).

## 6. The `speculative` flag — you MUST honour it

| `speculative` | meaning | how to accept it |
|---|---|---|
| `False` | a **reshape** — re-pairs/strips, closer to behaviour-preserving | HFF (TrueNorth) on the normal objective vector is fine |
| `True` | a **structural leap** — changes what the gene computes (squaring a difference, inverse-squaring a factor) | **gate on the EXTRAPOLATION objective**, never holdout alone |

Why: holdout is *gameable* — a leap that manufactures structure can fit the
training manifold yet be wrong. The extrapolation objective (train on one range,
score on an unseen range) is what separates "looks like the law" from "is the
law." The generator deliberately does not pre-judge; the extrapolation gate on
your side is the real safety mechanism.

## 7. Rules currently generated

Full catalogue with soundness dials in `physics_prior_rules.md`. Live now (18):

| Rule | Edit | `speculative` |
|---|---|---|
| **A1** | cross-coord → axis-aligned pair: `(x2 − y1)` → `(x2 − x1)`, `(y2 − y1)` | False |
| **A2** | square a difference: `(a − b)` → `(a − b)²` | True |
| **A3** | append next axis pair to a sum of squared diffs (build Euclidean r²) | True |
| **A4** | sum-of-squares divisor → its square root: `f/(Σ²)` → `f/√(Σ²)` (1/r) | True |
| **C1** | symmetrise a scalar product: `m1·x` → `m1·m2` (partner in scope) | True |
| **C2** | even-context sign kill: `(a − b)` → `\|a − b\|` | True |
| **C4** | additive-inverse fold: `a + (−b)` → `a − b` | False |
| **D1** | reduced-mass / parallel template: `a + b` → `(a·b)/(a+b)` | True |
| **E1** | inverse-square a factor: `a·b` → `a/b²`; `a/b` → `a/b²` | True |
| **E2** | exponential-decay envelope: `f` → `f·exp(−x)` | True |
| **E3** | oscillator envelope: `f` → `f·cos(x)` | True |
| **E6** | Gaussian envelope: `f` → `f·exp(−x²)` | True |
| **F**  | strip an outer wallpaper factor: `f·√g`, `f·sin g`, … → `f` | False |
| **TR11-sin** | `sin(2x)` → `2·sin x·cos x` (double-angle) | False |
| **TR11-cos** | `cos(2x)` → `cos²x − sin²x` | False |
| **TR10-sin** | `sin(a+b)` → `sin a·cos b + cos a·sin b` | False |
| **TR10-cos** | `cos(a+b)` → `cos a·cos b − sin a·sin b` | False |
| **TR5** | `sin²x` ↔ `1 − cos²x` (Pythagorean rearrange) | False |

Trig rules (TR*) are exact identities mined from SymPy `simplify/fu.py` —
reshapes, not leaps. The API does not change as more rules are added; new
`rule` tags simply start appearing in the output.

## 8. Errors

- Raises `ValueError` only if `expr` is not a parseable Math s-expression.
- Never raises on a gene that has no applicable rule — it returns `[]`.
- `n < 1` is treated as 1.

## 9. Complete example

```python
from fuller import physics_mutate

gene = '(Div (Mul (Var "m1") (Var "m2")) (Sub (Var "x2") (Var "y1")))'
axes = [["x1", "x2"], ["y1", "y2"], ["z1", "z2"]]

for c in physics_mutate(gene, axes, n=6, seed=0):
    tag = "SPECULATIVE" if c["speculative"] else "reshape"
    print(f'[{c["rule"]:>3}] {tag:11} {c["expr"]}')

# Among the candidates: (m1*m2) / (x2 - x1)^2  — the inverse-square-distance
# (Newton) shape that random GEP mutation almost never reaches.
```

## 10. Using it in the GA loop

1. Pick a gene — typically a strong HOF member that is structurally *near* the
   law (right variables, wrong shape).
2. `cands = physics_mutate(gene_expr, axes, n=…)`.
3. Inject each `cand["expr"]` into the population as a new individual.
4. Select with **HFF-TrueNorth** on your objective vector. For
   `cand["speculative"]` individuals, require the **extrapolation** objective to
   hold, not just holdout R².

The aim: flood the pool with near-physical forms (axis-aligned squared
distances, inverse-square laws) the GA rarely produces by chance, so the search
can *find* the law instead of curve-fitting near it — the last stretch of the
SRBench gap.

## 11. Rust API

```rust
fuller::physics::generate(
    gene: &str,
    paired_groups: &[Vec<String>],
    n: usize,
    seed: u64,
) -> Result<Vec<fuller::physics::Candidate>, String>
// Candidate { expr: String, rule: String, speculative: bool }
```
