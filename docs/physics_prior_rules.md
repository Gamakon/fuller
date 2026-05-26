# Physics-prior speculative rule catalogue

Non-identity, physics-shaped rewrite rules for the `physics_prior` ruleset —
**a sibling to denoise, NOT denoise.** These are *mutation proposals*, not
sound simplifications: most CHANGE what the expression computes, justified only
by the data (and extrapolation). They bias the GA's candidate pool toward forms
that real physical laws take.

## Governance contract (read before adding any rule)

- **Soundness dial** per rule:
  - `IDENTITY` — true on all inputs (rare here; could ship like denoise).
  - `DATA` — changes the function; only kept if it reproduces the parent within
    tolerance on the data. Reversible by overfit.
  - `SPECULATIVE` — a structural leap (manufactures structure the GA never
    earned). Must be **extrapolation-gated**, never holdout-only.
- **Physics bias goes in GENERATION only.** Which candidates exist is biased by
  these templates. SELECTION stays pure HFF on the real objective vector, with
  the **extrapolation objective** as the guard against physics-LOOKING overfit.
- **Screen first.** The mutator detects which rules *match* a gene before
  proposing; a rule that doesn't pattern-match produces no candidate.
- Tolerance/extrapolation thresholds are caller-supplied; a SPECULATIVE rule
  that fails extrapolation is discarded, not ranked.

Legend: **Dir** = directional rewrite shape. **Dial** = trust level.

---

## Group A — Distance / coordinate geometry (the I_9_18 family)

| # | Rule | Dir | Dial | Promotes |
|---|------|-----|------|----------|
| A1 | cross-coord → axis-aligned pair: `xᵢ op yⱼ` (i≠j) → `xᵢ op yᵢ` and `xⱼ op yⱼ` | gen variants | DATA | same-axis differences |
| A2 | raw difference → squared difference: `(a − b)` → `Pow2(Sub a b)` | → | SPECULATIVE | squared terms |
| A3 | sum of squared diffs: `{(aᵢ−bᵢ)}` over a paired-var group → `Σ Pow2(Sub aᵢ bᵢ)` | gen | SPECULATIVE | Euclidean r² |
| A4 | inverse-square law: `f / g` where g is a sum-of-squares → `f / Pow(g, 1)` kept, plus try `f / Pow(Sqrt g, 2)` | gen | SPECULATIVE | 1/r² forms |
| A5 | distance norm: `Sqrt(Σ Pow2(...))` recognised → keep as atomic `dist` motif (don't prune inside) | guard | DATA | r = ‖·‖ |
| A6 | pairing completion: if 2 of 3 axis pairs present, propose the 3rd | gen | SPECULATIVE | full 3D distance |

## Group B — Dimensional / unit consistency

| # | Rule | Dir | Dial | Promotes |
|---|------|-----|------|----------|
| B1 | drop dimensionally-inconsistent additive term (term whose unit ≠ siblings) if R² holds | → | DATA | unit homogeneity |
| B2 | factor out a constant with units (group leading numeric coefficient) `c·a + c·b` → `c·(a+b)` | → | IDENTITY | clean coefficient |
| B3 | promote a bare numeric near a known dimensionless ratio (e.g. ≈2, ≈0.5, ≈π) to the exact constant if data supports | → | DATA | clean constants |
| B4 | reciprocal-pairing: `a · Inv(b)` ↔ `Div a b` (canonical form) | ↔ | IDENTITY | division form |

## Group C — Symmetry

| # | Rule | Dir | Dial | Promotes |
|---|------|-----|------|----------|
| C1 | symmetrise product of masses/charges: `mul(m1, x)` → try `mul(m1, m2)` when m2 in scope | gen | SPECULATIVE | m₁m₂ symmetry |
| C2 | even-power sign kill: wrap a sign-bearing subterm `s` used in an even context with `Abs`/`Pow2` if R² holds | → | DATA | sign symmetry |
| C3 | exchange symmetry: if swapping two variables leaves the data fit unchanged, canonicalise their order | → | DATA | particle exchange |
| C4 | additive-inverse fold: `a + Neg b` ↔ `Sub a b` (canonical) | ↔ | IDENTITY | clean form |

## Group D — Conservation / ratio templates

| # | Rule | Dir | Dial | Promotes |
|---|------|-----|------|----------|
| D1 | product-over-sum → ratio template `(a·b)/(a+b)` recognised, kept atomic | guard | DATA | reduced mass / parallel R |
| D2 | normalise to a ratio: `a·b` → `(a·b)/c` when a per-row `c` makes it dimensionless-stable | gen | SPECULATIVE | normalised quantity |
| D3 | conserved-sum hint: if `a+b` is ~constant across rows, propose replacing with that constant | → | DATA | conservation law |

## Group E — Common functional-form templates (Feynman shapes)

| # | Rule | Dir | Dial | Promotes |
|---|------|-----|------|----------|
| E1 | inverse-square: `f` → `f / Pow2(r)` for a detected distance r | gen | SPECULATIVE | 1/r² |
| E2 | exponential decay: `f` → `f · Exp(Neg(k·x))` template when residual is monotone-decaying | gen | SPECULATIVE | e^−kx |
| E3 | oscillator: `f` → `f · Cos(w·x)` when residual is periodic | gen | SPECULATIVE | sinusoid |
| E4 | power-law: `a · b` → `a · Pow(b, p)` sweeping small integer/half p | gen | SPECULATIVE | bᵖ scaling |
| E5 | linear-in-disguise: `Log(a)·Log(b)` patterns → `Log(a·b)` (sound only if a,b>0) | → | DATA | log-linearity |
| E6 | Gaussian: `f` → `f · Exp(Neg(Pow2(x)))` when residual is bell-shaped | gen | SPECULATIVE | e^−x² |

## Group F — Wallpaper stripping (physics-template-biased)

| # | Rule | Dir | Dial | Promotes |
|---|------|-----|------|----------|
| F1 | strip outer `f(x) · Sqrt(g)` factor if R² survives (biased: try non-physics factors first) | → | DATA | (covered by denoise prune; this prioritises) |
| F2 | strip outer `f(x) · Sin(g)` / `Tan(g)` factor if R² survives | → | DATA | remove oscillatory wallpaper |
| F3 | collapse `Sin(Exp(...))`-style nested-transcendental wallpaper to a constant if ~constant on data | → | DATA | remove deep wallpaper |
| F4 | drop additive term that's small everywhere on data (sub-tree data-aware) | → | DATA | (covered by denoise prune) |

## Group G — Trigonometric-identity templates (mined from SymPy `simplify/fu.py`)

These are **exact trigonometric identities** re-expressed as one-step structural
mutations. As emitted by the generator they are reshapes (`Dial = IDENTITY`):
the implementation flags them `speculative = false`. They are NEW beyond the
A–F catalogue; their origin is the Fu et al. trig-simplification transforms
`TRn` in SymPy `sympy/simplify/fu.py` (read for the identity forms only — the
crate has no SymPy dependency).

| # | Rule | Dir | Dial | SymPy origin | Promotes |
|---|------|-----|------|--------------|----------|
| TR11-sin | `Sin(2·x)` → `2·Sin(x)·Cos(x)` | → | IDENTITY | `TR11` | double-angle expand |
| TR11-cos | `Cos(2·x)` → `Cos(x)² − Sin(x)²` | → | IDENTITY | `TR11` | double-angle expand |
| TR10-sin | `Sin(a+b)` → `Sin a·Cos b + Cos a·Sin b` | → | IDENTITY | `TR10` | angle-sum expand |
| TR10-cos | `Cos(a+b)` → `Cos a·Cos b − Sin a·Sin b` | → | IDENTITY | `TR10` | angle-sum expand |
| TR5 | `Pow2(Sin x)` → `1 − Pow2(Cos x)` and the dual `Pow2(Cos x)` → `1 − Pow2(Sin x)` | ↔ | IDENTITY | `TR5`/`TR6` | Pythagorean rearrange |

Notes on termination of the trig templates:

- **TR5** is implemented as both directions (sin² ↔ cos²). The two form a
  2-cycle; the generator's BFS `seen` set terminates it (the round-trip
  reproduces an already-seen form and is dropped). No per-rule guard is needed
  for the cycle itself.
- **TR10/TR11** consume the double-angle / angle-sum pattern and emit forms that
  do not re-match the same pattern at the same site, so they cannot re-fire on
  their own output.

## Implementation note — global composition bound (termination)

Each composing speculative rule guards against re-firing on its *own* output
(see the `parent_op` / scalar-operand / divisor-shape guards in `src/physics.rs`).
But distinct structure-builders (A2, A3, C2, D1, E1, and the Group G trig
templates) compose **multiplicatively** — one rule's output is another's input —
and can grow a tree without ever repeating an exact string, which the `seen` set
alone cannot stop. The generator therefore enforces a **global size bound**: no
generated candidate may exceed `node_count(input) + COMPOSITION_BUDGET` nodes
(`COMPOSITION_BUDGET = 12`). The set of `Math` trees under a fixed node count
over a finite constructor alphabet is finite, so generation provably terminates
regardless of rule interaction. This is the backstop that makes the whole rule
set sound w.r.t. termination; the per-rule guards keep the candidate volume
small and physically pointed.

---

## Notes on overlap with denoise

F1/F4 (and B2/B4/C4) overlap what the sound denoise ruleset + data-aware pruner
already do. Keep them here only as *prioritisation hints* for the physics
mutator (which factors to try dropping first); the actual sound drop stays in
denoise. The genuinely NEW value is Groups A, C1, D, E — the SPECULATIVE
structure-builders that denoise will never propose because they aren't
identities.

## Build priority

1. **A1** (cross-coord→axis) — DATA, safe, directly targets I_9_18. Build first.
2. **A2, A3, A6** — the squared-distance leap. SPECULATIVE; needs extrapolation
   gate. This is the rule family that could close the SRBench gap.
3. **E1, E4** — inverse-square + power-law templates. SPECULATIVE.
4. Everything else as the catalogue matures.

The mutator screens a gene against all rules, generates candidates only for
matches, evaluates on data + extrapolation, and returns ranked variants for HFF
selection.
