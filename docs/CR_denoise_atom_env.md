# CR: denoise / denoise_karva — bind master_constants in eval env

## Problem

`denoise_karva` returns `changed=false` even at huge tolerance (0.5) on
chromosomes whose only "noise" is data-negligible additive physics-atom
constants. Concrete I_8_14 winner:

```
gene[0] = (y1 - y2)^2 + eps0        # eps0 ≈ 8.85e-12 — invisible against y1-y2
gene[1] = hbar * pi                  # ≈ 3.31e-34 — invisible after linker
gene[2] = (G - (x1 - x2))^2 + kB     # G ≈ 6.67e-11, kB ≈ 1.38e-23 — both invisible
```

The truth shape `(y1-y2)^2 + (x1-x2)^2` is sitting inside this with an
overlay of physics atoms whose numeric contribution is below float
precision relative to the data variables. Yet `prune_on_data` can't drop
them because `eval_row` errors out:

```
EvalError::UnboundVar("eps0")
EvalError::UnboundVar("kB")
EvalError::UnboundVar("G")
EvalError::UnboundVar("hbar")
EvalError::UnboundVar("pi")
```

Every candidate that references an atom is skipped in
`extract.rs::denoise_core` (line ~129: `match preds { Err(_) => continue }`),
and `prune_on_data` likewise rejects every pruning candidate that still
contains an atom because the reference can't even be computed.

## Ask

In `denoise` and `denoise_karva`, merge `master_constants()` into the
eval environment so `eval_row` resolves atom names to their numeric
values automatically. Caller-supplied row values still take precedence
(if a row contains `pi`, the row's value wins — it's a data column, not
the constant).

Concrete patch shape in `extract.rs::denoise`:

```rust
let consts: HashMap<&str, f64> = master_constants_map();  // pi, e, G, hbar, ...

let reference: Vec<f64> = rows
    .iter()
    .map(|row| {
        // Build augmented env: row values first, fall back to constants.
        eval_row_with_consts(&termdag, ref_term, row, &consts)
    })
    .collect::<Result<_, _>>()?;
```

Same change applied to the candidate evaluation loop and to
`prune_on_data`. The constants map is the same one already used by
`snap_karva`'s `cvals` table — it's literally `master_constants()` keyed
by name.

## Functions affected

- `denoise(...)`, `denoise_karva(...)`
- internally: `denoise_core`, `prune_on_data`, `fits` — they call
  `eval_row` / `eval_term` and need the augmented env.

## Why this is sound

`master_constants()` are universal physical constants the GA gets as
SymbolTerminals via `register_atoms_in_pset`. They never vary across
data rows — they're not features, they're numeric atoms. Binding them
in the eval env matches how the GA already evaluates them at fitness
time (via `pset.globals`). The current behaviour (`UnboundVar` error)
is a bug: it silently disables prune-on-data for any chromosome the GA
naturally evolves with these atoms in it.

## Verification

After CR lands:

1. Re-run `notebooks/_prune_tune_i_8_14.py`. Expect `changed=true` at
   tolerance 1e-3 with a denoised expression that drops `eps0`, `kB`,
   `G`, `hbar*pi`, leaving `(y1-y2)^2 + (x1-x2)^2` (or close — `G`
   inside `_diff_sq(G, x1-x2)` would prune to `(x1-x2)^2`).
2. Add a unit test in `fuller/src/extract.rs` covering one Add tree
   with a tiny constant atom.

## Tolerance semantics

`tolerance` already means R² loss budget. With the env fix, the I_8_14
case should prune at tolerance ≤ 1e-10 because dropping `eps0` from
`(y1-y2)^2 + eps0` changes predictions by ~`8.85e-12`, way below any
reasonable R² threshold for floats in `[1, 5]` range.
