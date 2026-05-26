# `physics_mutate` — physics-prior mutation generator (API)

A **one-to-many generator**. Give it one gene; it returns a proliferation of
physics-shaped candidate genes — structural mutations that bias the form toward
what real physical laws look like (squared distances, axis-aligned coordinate
pairs, inverse-square, stripped wallpaper, …).

**It is NOT denoise.** Denoise tidies (behaviour-preserving). This *changes* what
the gene computes, on purpose, to seed the GA with forms it would rarely find
by random mutation. It does **no evaluation, no scoring, no fitness** — it only
generates. The caller (your GA + HFF) selects.

## Python

```python
from gamakAST import physics_mutate

candidates = physics_mutate(
    expr,            # a Math s-expression string (one gene)
    paired_groups,   # coordinate axes, e.g. [["x1","x2"], ["y1","y2"], ["z1","z2"]]
    n=10,            # max candidates to RETURN (>=1). default 10
    seed=0,          # makes the random sample reproducible
)
# -> list of {"expr": str, "rule": str, "speculative": bool}
```

### Semantics

- **All candidates are generated internally** (rules × every matching site ×
  composition, deduped to a fixpoint, bounded by an internal safety cap). Then,
  if more than `n` exist, a **uniform random `n`** are returned. The cap only
  limits what is *returned*, not what is generated.
- `n` from 1 upward. Ask for 3, get 3 (if ≥3 were generated); ask for 50, get up
  to 50.
- `seed` → reproducible sample. Same `(expr, groups, n, seed)` → same list.
- **No data passed in.** This function never sees your rows. Correct — selection
  is the engine's job.

### The `speculative` flag — you MUST honour it

| `speculative` | meaning | how to accept |
|---|---|---|
| `False` | a reshape (e.g. re-pairing coordinates onto the same axis) — closer to behaviour-preserving | HFF on the normal objective vector is fine |
| `True` | a **structural leap** that changes what the gene computes (e.g. squaring a difference, inverse-squaring a factor) | **gate on the EXTRAPOLATION objective**, never holdout alone — holdout is gameable, extrapolation kills physics-looking overfit |

The generator deliberately does not pre-judge speculative candidates; that's why
the extrapolation gate on your side is the real safety mechanism.

## How to use in the GA loop

1. Take a gene (e.g. a strong HOF member that's structurally *near* the law).
2. `physics_mutate(gene_expr, axes, n=…)` → candidate genes.
3. Inject them into the population as new individuals.
4. Let HFF (TrueNorth) + the extrapolation objective select — speculative
   candidates only survive if they extrapolate, not just fit holdout.

This is the mechanism aimed at closing the last SRBench gap: it floods the pool
with near-physical forms (axis-aligned squared distances, inverse-square laws)
that random GEP mutation almost never produces, so the search can *find* the law
rather than only curve-fit near it.

## Rules currently generated

See `docs/physics_prior_rules.md` for the full catalogue. Implemented so far:
- **A1** cross-coord → axis-aligned pair (`x2−y1` → `x2−x1`, `y2−y1`) — reshape.
- **A2** square a difference (`(a−b)` → `(a−b)²`) — speculative.
- **E1** inverse-square a factor (`a·b` → `a/b²`; `a/b` → `a/b²`) — speculative.
- **F** strip an outer wallpaper factor (`f·√g`, `f·sin g`, … → `f`) — reshape.

More rules (trig, symmetry, conservation, functional-form templates) are being
mined from SymPy + physics and added to the generator.

## Lower-level (Rust)

`gamakast::physics::generate(gene, paired_groups, n, seed) -> Vec<Candidate>`,
where `Candidate { expr, rule, speculative }`.
