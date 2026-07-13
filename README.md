# fuller

**An e-graph engine for making symbolic expressions smaller — provably without changing what they compute.**

Named after Buckminster Fuller's *ephemeralization*: doing more with less. `fuller`
takes a symbolic expression (a GEP chromosome, a SymPy expression, a Brainfuck
program) into an [egglog](https://github.com/egraphs-good/egglog) e-graph,
saturates a bounded ruleset, and extracts the smallest equivalent form — with a
data-gated or exact equivalence check so a rewrite can never silently change
behaviour.

It is a Rust crate exposed to Python via [PyO3](https://github.com/PyO3/pyo3).
It was built to replace SymPy on the hot path of a symbolic-regression engine,
where SymPy's complex-domain assumptions, native signal handling, and
exponential simplification paths made expression cleanup unreliable. egglog is
the right substrate: deterministic, bounded, declarative, and fast (the same
engine family behind [Herbie](https://herbie.uwplse.org/)).

## What it does

- **Denoise** — shrink an expression to an equivalent smaller one, gated by an
  R² check against your data so the rewrite provably preserves behaviour on the
  points you care about.
- **Snap ⇄ concretize** — flip named constants (`pi`, `G`, `k_e`, …) to their
  numeric values and back. The pair works as GEP mutation operators, letting a
  population evolve the *representation* (symbolic vs numeric) under selection.
- **Prove equivalence** — `equals` / `proves_equal` decide whether two
  expressions are the same via equality saturation, not sampling.
- **SymPy bridge** — `to_math` / `from_math` convert SymPy ⇄ the internal `Math`
  form losslessly (real-domain ops), so you can drop `fuller` into a SymPy
  pipeline.
- **Brainfuck simplifier** — the same machinery applied to a second, exact
  target (`bf_simplify`), where equivalence is decidable output-by-output.

## Install

Requires a Rust toolchain (`rustup install stable`) and Python ≥ 3.9. Built with
[maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --release      # builds the Rust extension and installs `fuller`
```

## Quick start

```python
import sympy as sp
import fuller

r = sp.Symbol("r")

# SymPy <-> internal Math form
fuller.to_math(sp.pi * r**2)          # '(Mul (Var "pi") (Pow2 (Var "r")))'

# Prove two forms equal (equality saturation, not sampling)
fuller.equals(sp.pi * r**2, sp.pi * r * r)     # True

# Denoise a Math expression against data: shrink it while preserving R².
rows = [{"x": v} for v in (-2.0, 1.0, 3.0, 5.0)]
fuller.denoise('(Add (Mul (Var "x") (Num 1.0)) (Mul (Num 0.0) (Var "y")))', rows)
# -> {'expr': '(Var "x")', 'cost': 1, 'changed': True}

# Same engine, a different target: simplify a Brainfuck program.
fuller.bf_simplify("+++[-]")     # {'source': '...', 'op_count': ..., 'changed': ...}
```

For the GEP/karva-chromosome API (`denoise_karva`, `snap_karva`,
`concretize_karva`, `physics_mutate_karva`, `eclass_extract_hff_instrumented`),
see [`docs/USAGE.md`](docs/USAGE.md).

## Two proven targets: symbolic regression and Brainfuck

fuller is not tied to one language. The core is *target-agnostic*: give it a
grammar, a cost model, and a way to check equivalence, and it will minimise
programs in that grammar. Two targets ship, chosen because they check
equivalence in the two fundamentally different ways available:

**Symbolic regression (equivalence checked against data).** Evolved GEP
expressions are noisy and bloated — `x·1 + 0·y`, `√(NA/r⁴)·…` where a physical
constant is smuggled into a coefficient, `Abs(a^1.5)` where the domain is
positive. fuller folds these to the clean form (`x`, `1/r²·√NA`, `a^1.5`) and
keeps the rewrite **only if R² does not drop on your data**. This is the hot
path it was built for: replacing SymPy inside a symbolic-regression engine's
extraction step, cheaply and deterministically, so the discovered law comes out
in its simplest honest form. The `snap_karva` / `concretize_karva` pair goes
further — letting a population evolve *whether a constant is symbolic or
numeric*, so selection recovers the law's true form in a single run.

**Brainfuck (equivalence checked exactly).** A Brainfuck program's behaviour is
a decidable input→output function, so equivalence is *exact* — no data, no
tolerance. `bf_simplify` collapses run-length redundancy (`+++++-----` → nothing,
`[-]` clear-loops) while proving the simplified program produces identical output
on every input. It is the clean demonstration of the same engine where
"behaviour-preserving" is a boolean, not a statistic — the same technique
applies to any target with decidable equivalence (SQL, regex, sorting networks,
compiler IR).

## Python API

| Function | Purpose |
|---|---|
| `denoise(expr, rows, tolerance=, k_variants=, positive_vars=, nonzero_vars=)` | Shrink a `Math` string, R²-gated against `rows`. |
| `denoise_karva(head, tail, …)` | Same, as a GEP mutation operator on karva head/tail token lists. |
| `snap_karva(head, tail, …)` | Up-flip: replace fitted numbers with named lattice constants. |
| `concretize_karva(head, tail)` | Down-flip: replace named constants with their numeric values. |
| `equals(a, b)` / `proves_equal(a, b, …)` | Decide equivalence by equality saturation. |
| `to_math(expr)` / `from_math(s)` | SymPy ⇄ `Math` s-expression bridge. |
| `physics_mutate_karva(…)` | Physics-prior structural mutations (inverse-square, etc.). |
| `eclass_extract_hff_instrumented(…)` | Tournament extraction with per-variant metrics. |
| `bf_simplify(source)` / `bf_parse` / `bf_unparse` | Brainfuck program simplification. |
| `master_constants()` / `master_lattice()` / `master_pset()` | The constant lattice and primitive set. |

## Why an e-graph instead of SymPy

SymPy rewrites in place, over the complex domain, with simplification paths that
can blow up on adversarial input and native code that installs its own signal
handlers. On a hot evolutionary loop that is both slow and unreliable. An
e-graph represents *all* equivalent forms at once, applies a **bounded**
ruleset (saturation is capped, never runs away), extracts by an explicit cost
model, and is fully deterministic — the same input always yields the same
output. Equivalence is *proved* by saturation, not sampled.

## Design guarantees

- **Bounded.** Every saturation runs a capped rule schedule; no unbounded work.
- **Behaviour-preserving.** Rewrites are either exact (algebraic identities) or
  data-gated (kept only if R² does not drop) — a rewrite cannot silently change
  what an expression computes.
- **Deterministic.** No wall-clock, no randomness outside an explicit `rng_seed`.
- **Crash-safe at the boundary.** PyO3 entry points return `Err`, never abort:
  deeply nested / oversized inputs are rejected rather than overflowing the
  stack.

## Status

Core is working and used in production by a symbolic-regression engine: denoise,
the snap/concretize flip pair, the physics-prior mutations, the SymPy bridge,
and the Brainfuck simplifier all ship. The ruleset is deliberately small and
grows conservatively — a rule lands only once it is either provably sound or
data-gated. See [`docs/USAGE.md`](docs/USAGE.md) for the consumer guide and
`docs/` for per-feature notes.

## License

MIT. See [`LICENSE`](LICENSE).
