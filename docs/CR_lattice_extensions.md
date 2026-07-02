# CR: extend the snap lattice — missing composed forms

**To:** fuller team
**From:** HFF engine owners
**Why:** auditing our near-miss problems, the most common unsnappable shapes
all involve `1/sqrt(...)` or `1/(n·sqrt(...))` patterns. The current lattice
generates `sqrt(c)` and `1/c` and `1/(n·c)`, but **NOT** `1/sqrt(c)` or
`1/sqrt(n·c)`. These are the leading coefficients of basic physics
distributions (Gaussian PDF, oscillator amplitudes, statistical mechanics
partition normalisers), and the engine routinely produces their numeric
approximations without our snap recognising them.

## What we want added

For each base atom `c` in `master_constants()`, plus each `n ∈ INTS`, generate:

| Form | Example | Why it matters |
|---|---|---|
| `1/sqrt(c)` | `1/sqrt(2π) ≈ 0.39894228` | Gaussian PDF I_6_2, I_6_2a, I_6_2b |
| `1/sqrt(n·c)` | `1/sqrt(4π) ≈ 0.28209479` | Coulomb's law (SRBench I_27_6 territory) |
| `sqrt(c/n)` | `sqrt(g/L)`-style oscillator | Pendulum-family expressions |
| `sqrt(n/c)` | `sqrt(2/m)` | Quantum well, scattering cross-section |
| `1/(n·sqrt(c))` | `1/(2·sqrt(π))` | Statistical weight factors |
| `n/sqrt(c)` | `2/sqrt(pi)` | Diffusion coefficients |

In Math s-expression form (using `Div(1, ...)` per the earlier fix):

```
1/sqrt(c)     →  (Div (Num 1.0) (Sqrt c_form))
1/sqrt(n·c)   →  (Div (Num 1.0) (Sqrt (Mul (Num n) c_form)))
sqrt(c/n)     →  (Sqrt (Div c_form (Num n)))
sqrt(n/c)     →  (Sqrt (Div (Num n) c_form))
1/(n·sqrt(c)) →  (Div (Num 1.0) (Mul (Num n) (Sqrt c_form)))
n/sqrt(c)     →  (Div (Num n) (Sqrt c_form))
```

## Concrete misses (from our last sweep)

Verified zero matches in the current lattice (1428 entries) for these
known-physics values our near-miss chromosomes contain:

| Value | Physics meaning | Lattice match? |
|---|---|---|
| `0.39894228` | `1/sqrt(2π)` (Gaussian PDF) | ❌ no |
| `0.28209479` | `1/sqrt(4π)` (Coulomb) | ❌ no |
| `0.56418958` | `1/sqrt(π)` (error function) | ❌ no |
| `1.1283792`  | `2/sqrt(π)` | ❌ no |
| `0.79788456` | `sqrt(2/π)` | ❌ no |
| `1.5957691`  | `2·sqrt(2/π)` | ❌ no |

## Sizing impact

Current lattice: 1428 entries.

Adding the 6 sqrt-composed forms across 16 base atoms × ~20 integers (1..20)
adds **roughly 6 × 16 × 20 ≈ 1900 entries**, taking the total to ~3300.

This may double saturation cost for snap calls. If that's a concern, two
mitigations:

1. **Subset of integers.** Only powers of 2 (1, 2, 4, 8) and small primes
   (1, 2, 3, 5, 7) — the values that appear in real physics. Drops the
   multiplier to maybe 8x instead of 20x.
2. **Snap as a tiered ruleset.** Tier 1 = base lattice (1428); Tier 2 =
   sqrt-extensions. Saturation runs Tier 1 first, only adds Tier 2 rules
   if no match found.

We have no strong preference — your call which keeps perf within the 1s /
10k node budget.

## Deprecation context (why this matters now)

We are deprecating sympy.nsimplify throughout the engine and moving to
fuller snap exclusively. The two remaining nsimplify call sites
(`_clean_const` in `_extract_best` for LSM coefficients; `snap_constants`
in `hff_geppy_helpers.py` for the in-compress snap) will both be replaced
with `snap_karva`. For that to be a clean replacement, snap_karva needs
to cover the same patterns nsimplify catches with `[pi, sqrt(2*pi),
4*pi]` as basis atoms — which is exactly the missing `1/sqrt(n·c)` family.

Once these forms are in, nsimplify can be removed from the engine.

## Suggested implementation in `parity/gen_constants.py`

```python
# After the existing reciprocal block (Div(1, c)):
out.append(emit(1.0 / math.sqrt(val),
    f"(Div (Num 1.0) (Sqrt {cv}))", f"1/sqrt({name})", 3))
for n in INTS:
    ni = float(n)
    if val * ni > 0:
        out.append(emit(1.0 / math.sqrt(ni * val),
            f"(Div (Num 1.0) (Sqrt (Mul (Num {ni}) {cv})))",
            f"1/sqrt({n}*{name})", 5))
        out.append(emit(math.sqrt(val / ni),
            f"(Sqrt (Div {cv} (Num {ni})))",
            f"sqrt({name}/{n})", 4))
        out.append(emit(math.sqrt(ni / val),
            f"(Sqrt (Div (Num {ni}) {cv}))",
            f"sqrt({n}/{name})", 4))
        out.append(emit(1.0 / (ni * math.sqrt(val)),
            f"(Div (Num 1.0) (Mul (Num {ni}) (Sqrt {cv})))",
            f"1/({n}*sqrt({name}))", 5))
        out.append(emit(ni / math.sqrt(val),
            f"(Div (Num {ni}) (Sqrt {cv}))",
            f"{n}/sqrt({name})", 4))
```

(Same `Div(1, ...)` pattern as the earlier fix, so it decodes into every
real-engine pset that has `div` + `sqrt`.)

## Ask

1. Add the 6 sqrt-composed forms to the lattice generator.
2. Decide on integer multiplier scope (full INTS range vs powers-of-2 + small primes).
3. Re-emit `constants_lattice.json` and confirm `1/sqrt(2π)` now snaps.

We'll re-sweep the 10 near-misses once the lattice is updated — Gaussian-PDF
problems (I_6_2 family) are the most direct test: their truth is
`exp(-x²/2)/sqrt(2π)` and our chromosomes already produce shapes like
`0.399 · exp(...)`. Snap should now propose the coefficient as `1/sqrt(2π)`
and the candidate should cross R²≥0.999.

## What we'll do on our side after the lattice update

Replace the two remaining nsimplify call sites with snap_karva:

1. `_clean_const` in `hff_sr_engine.py` (LSM coefficient cleaner — currently
   `nsimplify(x, [pi, E], tolerance=1e-4)`). Becomes a single-atom snap_karva
   call: build `(Num x)`, snap, return the best variant's expression.

2. `snap_constants` in `hff_geppy_helpers.py` (the in-compress snap inside
   `visit_subtree`). Becomes a per-subtree snap_karva call. This may need
   an `inv` rule already in place (your earlier fix), and benefits directly
   from this CR's sqrt extensions because the in-compress sub-trees often
   contain the `1/sqrt(...)` patterns inline.

After both swaps and a verifying smoke, `import sympy` becomes the only
sympy contact in the engine, and even that's limited to expression-display
formatting.
