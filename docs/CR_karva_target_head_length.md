# CR: snap_karva / denoise_karva / physics_mutate_karva — accept `target_head_length`

> **STATUS: LANDED** (gamakAST commit `9d38596`). All three `*_karva` functions
> accept the optional `target_head_length`. Behaviour: shorter natural head →
> extended with BFS-unreachable terminal filler (semantics-preserved); longer →
> flagged `oversized` (never truncated); `None` → unchanged. `snap_karva` returns
> a per-candidate `oversized: bool`; `physics_mutate_karva` drops oversized
> candidates; `denoise_karva` returns the original unchanged with `oversized:
> true`. `maturin develop` to pick it up, then pass
> `target_head_length=gene.head_length` and drop the engine-side padding hack.
> See "Engine usage" below — the example loop is exactly the contract shipped.

## Problem

When HFF's engine wraps any of the three `*_karva` operators, the returned
`head` is sized to the rewritten term's shape — typically much shorter
than the chromosome's configured GEP head_length (e.g. 48). Plugging a
short-head gene back into a chromosome with sibling genes of head_length=48
breaks geppy's mating ops on the next generation:

```
File ".../geppy/tools/mutation.py", line 174, in is_transpose
    a, b = _choose_subsequence_indices(0, donor.head_length + donor.tail_length - 1,
                                        max_length=donee.head_length - 1)
File ".../random.py", line 319, in randrange
    raise ValueError(f"empty range in randrange({start}, {stop})")
ValueError: empty range in randrange(0, -30)
```

Root cause: GEP requires uniform `head_length` across all genes in a
chromosome. A 5-token head donor and a 48-head donee make
`is_transpose`'s slice index window empty.

We tried to fix this engine-side by padding the new head with random
terminals up to the target length. That's structurally wrong — it
pollutes the head's tail region with arbitrary tokens that mating ops
will splice into other genes, corrupting search-space connectivity.

## Ask

Add an optional parameter to each `*_karva` function:

```rust
target_head_length: Option<usize>
```

When set, gamakAST should:

1. Compute the natural head tokens from the rewritten term (current
   behaviour).
2. If the natural head is shorter than `target_head_length`, **extend
   the karva head to the target** using GEP-valid filler that BFS-decode
   would never reach. The simplest valid filler is a sequence of
   terminals appended after the k-expression's final BFS slot — every
   slot beyond the live tree is structurally a "tail" slot in BFS
   decoding, so terminals there are syntactically correct.
3. If the natural head is **longer** than `target_head_length`, refuse
   the candidate (return it with an `oversized: true` field) — the
   caller must drop it. Don't truncate, because truncation would change
   the rewritten term's semantics.
4. Update the corresponding `tail` length to satisfy the GEP rule
   `tail_length = head_length * (max_arity - 1) + 1`.

This keeps the AST-correct karva expression intact and produces a gene
of the exact head_length the chromosome expects, so geppy's mating ops
work without engine-side padding.

## Functions affected

- `snap_karva(head, tail, variables, functions, rnc_values, k_variants, rel_tol, rng_seed, *, target_head_length=None)`
- `denoise_karva(head, tail, variables, functions, rnc_values, rows, tolerance, k_variants, rng_seed, *, target_head_length=None)`
- `physics_mutate_karva(head, tail, variables, functions, rnc_values, paired_groups, n_candidates, rng_seed, *, target_head_length=None)`

Backwards-compatible: when `target_head_length=None`, behaviour is
unchanged.

## Engine usage

```python
head, tail = orig_gene.head, orig_gene.tail
cands = snap_karva(head, tail, vars, fns, rnc,
                    target_head_length=orig_gene.head_length)
for c in cands:
    if c.get("oversized"):
        continue
    new_gene = build_gene_like(orig_gene, c["head"], c["tail"], pset)
```

`build_gene_like` then drops its padding hack entirely — the new head
is already the right length.

## Notes

- The "extend with terminals" rule is the same one GEP uses to grow a
  random-init head: head slots beyond the k-expression are filler that
  the BFS decoder doesn't visit. So this is the same syntactic position
  geppy itself fills with random tokens at init time.
- We don't need symmetric truncation for shorter targets — that case
  doesn't arise in practice (the engine always passes its configured
  head_length, which is at least as large as anything snap can produce).

## Verification

After CR lands:

1. Engine smoke (`notebooks/_engine_i_6_2.py` with `physics_pset_strict=True`,
   `b_in_hff=True`) runs past gen 20 without the `randrange(0, -30)` crash.
2. Each `*_karva` call site in `notebooks/_*_op.py` passes
   `target_head_length=gene.head_length`.
3. `_gene_utils.build_gene_like` reverts its head-padding lines.
