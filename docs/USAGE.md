# gamakAST — usage for the SR engine team

The denoise mutation operator: takes a GEP chromosome, rewrites away algebraic
noise via egglog, returns a smaller **equivalent** chromosome. Deterministic,
real-domain, **no sympy**. This is the BRIEF.md Phase 1 skateboard, shipping.

## Install

```bash
cd /Users/andrewmorgan/Dev/kaito/gamakAST
maturin develop --release        # builds the Rust ext + installs into the active env
python -c "from gamakAST import denoise_karva; print('ok')"
```

Requires a Rust toolchain + maturin. Re-run `maturin develop --release` after
pulling new commits.

## The function you want: `denoise_karva`

Pass a chromosome directly (no string conversion):

```python
from gamakAST import denoise_karva

# Describe your pset ONCE (pure data — no geppy objects cross the boundary):
variables  = ["x", "y"]                       # variable names
functions  = {                                # token_name -> (semantic_id, arity)
    "add": ("add", 2),
    "mul": ("mul", 2),
    "sqrt": ("sqrt", 1),                      # RAW sqrt: sqrt(neg) = NaN
    "protected_sqrt": ("protected_sqrt", 1),  # geppy protected op -> the
                                              # protected_* semantic id, NEVER
                                              # the raw one (see below)
}
rnc_values = [1.0, 0.0]                        # numeric constants the tail may use

# A chromosome as head/tail token tuples: ("func"|"var"|"num", value)
head = [("func","add"), ("func","mul"), ("func","mul")]
tail = [("var","x"), ("num",1.0), ("num",0.0), ("var","y")]
#   decodes to: add(mul(x,1), mul(0,y))

rows = [{"x":1.0,"y":5.0}, {"x":2.0,"y":-3.0}, {"x":3.0,"y":0.5}]  # training data

out = denoise_karva(head, tail, variables, functions, rnc_values, rows,
                    tolerance=1e-3, k_variants=64, rng_seed=0)

# out == {"head": [("var","x")], "tail": [<2 terminal tokens>],
#         "changed": True, "expr": '(Var "x")'}
# (tail re-padded to head_len*(n_max-1)+1 = 2 terminals here — n_max is the
#  pset's max function arity, 2 — deterministic in rng_seed)
```

### Returns
`{"head": [...], "tail": [...], "changed": bool, "expr": str|None}`
- `changed=True`  → `head`/`tail` is a **smaller equivalent** chromosome; inject it.
- `changed=False` → nothing to simplify; `head`/`tail` are your originals.
- Never raises on normal input. Un-encodable chromosome → returned unchanged.
- Tail obeys the GEP rule (terminals only, `len = head_len*(n_max-1)+1` with
  `n_max` = your pset's max FUNCTION arity), re-padded deterministically from
  `rng_seed`.
- `positive_vars=[...]` / `nonzero_vars=[...]` (optional): variable names YOU
  know are `> 0` / `!= 0` (e.g. from your var_ranges). This unlocks the
  guarded rewrites — Abs-shedding (`Abs(a**1.5) -> a**1.5`), div-cancellation
  — which never fire on unproven domains. Assert only what you actually know.

### The `semantic_id` contract (important)
gamakAST rewrites on what an operator **computes**, not its geppy name. You map
your pset names → semantic ids in the `functions` dict. Valid semantic ids
(= `master_pset()`, which returns the authoritative list):
`add sub mul div neg sin cos tan log exp sqrt abs tanh pow2 pow3 pow inv
protected_sqrt protected_log protected_exp protected_inv protected_div`
(plus `diff_sq`, accepted on decode and lowered to `pow2`+`sub`).

**Protected ops are DISTINCT semantic ids — never map them to the raw op.**
`protected_sqrt(x) = sqrt(|x|)` and raw `sqrt(neg) = NaN` disagree on every
negative input; mapping `protected_sqrt -> "sqrt"` makes the rewrite engine
simplify your gene under the wrong semantics (unsound on negatives/zero).
The same holds for `protected_log/exp/inv/div`. Map `math.sqrt`-style RAW ops
to `"sqrt"`, and each `protected_*` op to its own `protected_*` id.

## When to call it in a geppy/DEAP loop

`denoise_karva` is an algebraic **tidier**, not an explorer — it only ever
returns equivalent forms, so it never improves R² on its own and never
discovers new structure. Use it to keep genetic material clean:

1. **Best — post-evolution, on the Hall of Fame.** Run it on your top-K finishers
   to strip algebraic wallpaper before reporting/reading. Zero search cost,
   pure readability/parsimony win.
2. **Periodic re-seeding (every ~10–20 generations, on the elite fraction).**
   When `changed`, inject the denoised chromosome back as a NEW individual
   (don't delete the parent). Seeds the GA with cleaner equivalents to breed.
3. **Low-probability DEAP mutation operator (~5–10%).** When it fires and
   `changed=True`, swap in the smaller chromosome.

**Do not** use it as your primary mutation, or every individual every gen
(overhead, little gain), or expect fitness jumps (it preserves behaviour).
Let the GA explore; let denoise tidy.

## Current ruleset scope (honest)

`denoise`/`denoise_karva` saturate the bounded **algebra + powers** subset:
the algebra identities (mul/add/sub identity, mul-by-zero, double-negation,
`sqrt(x^2)->|x|`, additive cancellation, guarded div-cancellation) plus the
power/log rules — and a data-guarded pruner that drops subtrees whose removal
doesn't change predictions on your rows. On top of that, an evaluator-backed
**constant-subtree fold** replaces variable-free subtrees with their literal
value (`log(|sqrt2|) -> 0.3466` — egglog has no ln/sqrt primitives, the
evaluator does), and a data-gated **additive-constant strip** drops paired
fitted offsets in one move (`pi*(r^2 + c) - pi*c -> pi*r^2`), which the
one-step pruner cannot reach. distribute / rational / trig /
trig_fu / wide rulesets exist but are used only by the parity scorer and the
e-class enumerators (`eclass_variants`, `eclass_extract_hff`, `proves_equal`
families) — they are normal-form/expansion families, deliberately kept out of
the denoise saturation (non-confluent, e-graph blow-up).

## Lower-level API: `denoise` (Math strings)

If you already have an egglog `Math` s-expression (not a chromosome), call
`denoise(expr, rows, tolerance=1e-3, k_variants=64) -> {expr, cost, changed}`.
Most consumers want `denoise_karva` instead.
