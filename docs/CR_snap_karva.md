# CR: `snap_karva` — egglog-backed constant-snap as a karva-in/karva-out operator

**To:** fuller team
**From:** HFF engine owners
**Why:** replace our linear, single-pass post-evolution snap (`_snap_with_timeout` in `notebooks/hff_sr_engine.py`) with an egglog-backed proliferator that composes constant substitutions with algebraic rewrites in one e-graph saturation. Same architectural pattern as `denoise_karva` and `physics_mutate_karva` — karva in, karva out, pset passed alongside.

**Context:** our 49% Feynman recovery (single seed) has 10 near-misses at R²=0.99–0.999. The dominant failure mode in these is that the GA found the right shape and approximate numeric constant (e.g. `0.0796 · ...`), but the snap pass doesn't recognise it as `1/(4π)` because today's snap walks one atom at a time without composing with the surrounding algebra. Egglog's saturation naturally handles this — it can rewrite `Num(0.0796) → Inv(Mul(Num(4), Pi))` AND simultaneously apply structural simplifications, producing a Pareto frontier of variants we then score on data.

---

## What fuller already gives us

| Capability | Already on main | Used by |
|---|---|---|
| karva ↔ egglog term converter | ✅ | `denoise_karva` |
| `master_pset()` (22 ops, raw + protected) | ✅ | hff engine pset |
| Saturation with hard budgets (1s / 10k nodes) | ✅ | `denoise` |
| `extract_variants(k)` ranked by cost function | ✅ | `denoise` data-aware path |
| Algebraic identity rules (x*1, +0, etc.) | ✅ | `denoise` |
| Protected-op constructors | ✅ | both |

This CR builds on all of those — it doesn't add new substrate, it adds a new rule family and a new public entry point.

---

## Signature

```python
from fuller import snap_karva

result = snap_karva(
    head, tail,            # karva chromosome — list of (kind, value) tuples
    variables,             # list[str] — chromosome's free variables
    functions,             # {token_name: (semantic_id, arity)} — same as elsewhere
    rnc_values,            # list[float] — RNC values in the pset's Dc array
    constants_library,     # list[dict] — see §3 below
    k_variants=16,         # how many candidate variants to return
    rel_tol=1e-3,          # numeric snap match tolerance
    rng_seed=0,
)
```

**Returns** a list of dicts, each:

```python
{
    "head": list[tuple],         # karva head tokens
    "tail": list[tuple],         # karva tail tokens
    "rule_trace": list[str],     # which rules fired in order (for debugging)
    "snapped_constants": list[dict],  # which library entries matched: [{from: 0.318, to: "1/pi"}]
    "cost": int,                 # egglog's internal structural cost
    "is_original": bool,         # True if this variant matches the input
}
```

The list is sorted by ascending cost (smallest first), and includes the original input as the last entry (cost=baseline) so the caller can always compare.

---

## §3 The constants_library shape

Same idea as our current `KNOWN_CONSTANTS`, but extended so each entry carries:

- `name`: a human-readable label (`"pi"`, `"G"`, `"hbar"`)
- `value`: the numeric value (`3.14159265358979`, `6.6743e-11`)
- `sympy_form`: how it should appear in the rewritten expression (`"Pi"`, `"G"`, `"Hbar"`) — as a fuller constructor name OR a Var name we register in the egglog datatype

```python
constants_library = [
    {"name": "pi",         "value": 3.141592653589793, "sympy_form": "Pi"},
    {"name": "2pi",        "value": 6.283185307179586, "sympy_form": "Mul(Num 2.0, Pi)"},
    {"name": "1/(4pi)",    "value": 0.07957747154594766, "sympy_form": "Inv(Mul(Num 4.0, Pi))"},
    {"name": "G",          "value": 6.6743e-11,          "sympy_form": "G"},
    # ...
]
```

For HFF's purpose the caller hands you the full library each call — you don't keep state. Lets us tune the library per-problem family (Feynman vs wild) without recompiling fuller.

---

## §4 The new rule family (what we want egglog to do)

For each entry `c` in `constants_library`, two rewrites get added to the active ruleset:

**4a. Direct match.** A numeric atom in the e-graph close to `c.value`:
```
(Num x)   when   abs(x - c.value) / max(abs(c.value), 1) < rel_tol
→  c.sympy_form
```

**4b. Composed match.** Atoms scaled by another atom — `0.318 · X → 1/π · X` not just `0.318 → 1/π`:
```
(Mul (Num x) Y)   when   abs(x - c.value) / max(abs(c.value), 1) < rel_tol
→  (Mul c.sympy_form Y)
```

Both bidirectional with the existing identity rules — the e-graph finds compositions automatically (e.g. `0.0796` matches `1/(4π)` and `(1/2) · (1/(2π))` simultaneously; cost function picks the simpler).

---

## §5 Cost function

Default to the existing structural cost (tree node count). Optional second arg: **bias toward "physics-shape"** — penalise opaque numeric atoms heavier than named-constant atoms. Pseudocode:

```
cost(Num) = 5    # we'd rather see Pi than 3.14
cost(Var "Pi") = 1
cost(Var "G")  = 1
cost(Mul)     = 2
... etc
```

So `(Mul (Num 0.318) X)` costs more than `(Mul (Inv Pi) X)` even though their structures are similar. This biases extraction toward physics-shaped equivalents while still allowing the original form when no constant matches.

If exposing the cost function feels like scope creep, just use structural cost and we'll filter on our side. Either works.

---

## §6 Why this beats today's snap

Our current snap (`hff_sr_engine.py::_snap_with_timeout` → calls into `notebooks/hff_geppy_helpers.py::snap_levels`):

- Walks numeric atoms one at a time
- Single substitution per atom, no composition
- Doesn't compose with algebraic rewrites — a chromosome like `(2/π) · sin(x)` simplified would be `0.6366 · sin(x)` and snap would correctly hit `2/π`, but it MISSES `(0.6366 / 2) · 2 · sin(x)` because the `0.3183` isn't `2/π` directly
- O(n_atoms × n_library) per chromosome
- Today returns ONE snapped expression — we want a frontier

Egglog snap:
- Tries every algebraic rearrangement of the chromosome that includes a constant-rewrite as a step
- Composes with the algebra rules already on main (so `(0.6366 · x) / 2` and `(2/π · x) / 2` are in the same e-class)
- Bounded by the existing 1s / 10k node saturation budget
- Returns the K cheapest variants — caller scores them on data

---

## §7 What we do with it (HFF-side)

Replaces `_snap_with_timeout` in `hff_sr_engine.py::_extract_best`. Per pool entry:

1. Call `snap_karva` to get K variants
2. For each variant: rebuild Gene → compile_and_predict on holdout → compute HFF vec
3. Score by (R²_holdout ≥ 0.999 first, then min parsimony) — same end-phase pick logic we already have
4. Pick the winner; the rest are discarded

The 10 near-miss problems are the immediate test bed — chromosomes at R²=0.998 where a single composed constant-rewrite could cross 0.999.

---

## §8 Risk / non-goals

- **Not a search operator.** snap_karva proliferates EQUIVALENT forms only. It does not change behaviour on data within tolerance. If a snap match changes the numerical output beyond `rel_tol`, that's a bug — we'd reject downstream via the R² guard.
- **Not for non-numeric symbolic patterns.** `sin²+cos² → 1` is a `denoise` rule, not snap. Snap is specifically `numeric_atom → named_constant` (with composition through nearby algebra).
- **No new substrate.** Reuse `master_pset`, the existing egglog setup, the existing extraction harness. Just a new rule family + new entry point.

---

## §9 Cost estimate

Reading the existing `denoise_karva` Rust code:
- New entry point: 30-50 LOC
- New rule generator (takes `constants_library`, emits egglog rules at runtime): 40-60 LOC
- Two new rule patterns (direct + composed): ~10 LOC each in egglog
- Extraction wrapper to return list-of-K instead of single: trivial (uses existing `extract_variants`)
- Tests: 20-40 LOC

Total: ~120-180 LOC + tests. Smaller than physics_mutate was.

---

## §10 Ask

1. Approve the API shape (especially the `constants_library` dict format and the `rule_trace` / `snapped_constants` return fields — those are diagnostics we'd find useful but if they're awkward to compute we can drop them).
2. Confirm the cost-function biasing in §5 — happy either way.
3. Timeline estimate.
4. We'd set up a hff-side wrapper (mirror of `_denoise_op.py`) that calls `snap_karva` and integrates with the engine's end-phase. Same pattern, low risk.

---

## Appendix: where today's snap lives (for reference)

| File | Function | Purpose |
|---|---|---|
| `notebooks/hff_sr_engine.py` | `_snap_with_timeout` | The end-phase call site we'd replace |
| `notebooks/hff_geppy_helpers.py` | `snap_constants` | Per-atom numeric matcher |
| `notebooks/hff_geppy_helpers.py` | `snap_levels` | Multi-tolerance variant (deep/default/strict) |
| `notebooks/hff_geppy_helpers.py` | `score_snap_levels` | Picks best variant by holdout MSE |
| `notebooks/equation_problems.py` | `KNOWN_CONSTANTS` | The ~25-entry constants dict — this becomes `constants_library` input |

The current snap also runs **inside** `_sympy_to_karva.visit_subtree` at line ~275 (per-subtree, during chromosome decompression). That call stays for now — it's the bounded per-subtree pass and is fast. The CR is specifically about replacing the **end-phase** snap on the linker-combined sympy expression, which is where today's pass underperforms.
