# CR: port fuller to wgpu

Scope: fuller only, inside this crate, behind `feature = "gpu"`. nucleotable
and SQL→wgpu are out of scope. Nothing here is built yet unless marked BUILT.

## What fuller does, and where each part can run

| Part | Today | Port |
|---|---|---|
| Evaluate an expression on data rows (`eval_rows`, `eval_row`, `eval_expr_rows`) | CPU, rayon over rows | **Stage 1** — `gpu_eval` kernel (BUILT, engine-parity semantics) |
| Accept/reject a variant (`r2_loss`, `fits`, `prune_on_data`) | CPU, one variant at a time | **Stage 1** — device reduction, one number per variant |
| Snap (`snap_karva`, lattice lookup) | CPU, 0.006 ms/gene | **Stage 2** — lattice as a buffer, per-constant nearest lookup |
| Rewrite (e-matching, saturation, extraction) | egglog, CPU | **Stage 3** — bounded rewriting directly on K-expressions |

egglog is a CPU library and is not portable. Stages 1–2 move everything around
it; stage 3 replaces it for the bounded subset `denoise`/`smallest_form` use.

## Stage 1 — score every variant in one dispatch

Call sites that move (all in `src/extract.rs`): 215–216, 512–538, 571–572,
586, 857–870, 901, 922, 1198, 1253. Today each is a per-variant CPU loop.

1. Collect all variants of all genes in the batch (`denoise_candidates_assuming`
   already produces them; `denoise_karva_candidates_batch` already batches genes).
2. Encode with `gpu_eval::math_to_nodes` (BUILT) into one `GpuBatch`.
3. One dispatch: genes × variants × rows.
4. New reduction kernel: per variant, against its gene's reference column —
   Σ(ref−pred)², Σref, Σref², non-finite count → `r2_loss`. Read back 1 float
   per variant, not variants × rows.
5. Host: smallest cost within tolerance, as now.

`prune_on_data` and `additive_subset_candidates` become "generate all pruned
forms, score in the same dispatch" instead of a sequential greedy loop.

Rules: f32 on device; the *chosen* variant is re-checked in f64 on the CPU
before it is returned (same contract as the f64 re-fit). A `GpuSession` is
passed in by the caller — no device creation per call, no silent CPU path:
the GPU entry points are separate functions, the CPU ones stay as they are.

Test: parity harness — every variant's device `r2_loss` vs `eval_rows` on the
HOF dataset (847 genes). Determinism test as for `denoise`.

## Stage 2 — snap on device

Lattice (`lattice_static`) uploaded once. Per constant: nearest lattice entry
within rel-tol, parallel over genes × constants. Output feeds stage 1 as extra
variants (snapped form is just another candidate to score).

## Stage 3 — rewrite K-expressions on device, no e-graph

The e-match is a join: rule × gene × position. On a level-order array with
resolved child indices that join is data-parallel.

- Rules compiled to a table: pattern (op + child shape, depth ≤ 3), guard
  mask, replacement template. Source: the bounded algebra+powers set that
  `denoise` already uses — identities, powers, guarded rational. Not
  distribute, not trig (non-confluent; stay egglog, CPU, scorer-only).
- Kernel A, match: per (gene, position, rule) → hit bit.
- Kernel B, apply: per hit, emit one new chromosome (subtree splice + level-
  order re-layout). Size-reducing or size-neutral rules only, so output fits
  the input's slot.
- Dedup by behavioural signature (16 rows, BUILT in `chrom_score`).
- Bounded rounds (as `SMALLEST_FORM_MAX_ROUNDS = 6`) replace saturation.
- Guards: `positive_vars` / `nonzero_vars` become a per-gene bitmask over
  inputs; a guarded rule fires only when the matched subtree is a masked input
  (same soundness condition as `GUARD_RELATIONS`, narrower reach).

What is lost against egglog: sharing, and rewrites that must grow before they
shrink. Measured target before building: on the HOF dataset, what fraction of
egglog's 190 reductions does a 6-round shrink-only rewriter reproduce? That
can be measured on the CPU first with the same rule table.

Hard part: kernel B's re-layout (BFS order after a splice) — needs the arity
prefix sum from engine stage 1. Same primitive, build once.

## Order and gates

1. Stage 1 (kernel exists; reduction + plumbing). Gate: parity on 847 genes,
   13-problem set still 13/13, e-graph step time before/after.
2. CPU prototype of the stage-3 rule table → coverage number vs egglog.
3. Stage 3 kernels only if coverage justifies it. Stage 2 alongside.

Consumer: hff's join (`_nb_expand_genes`, engine `_evaluate_population`)
takes variants already scored, so e-class variants + graft can enter the
engine's join without a Python round trip.

Needs mains: every build, `cargo test --features gpu`, all gates above.


Superseded for the rewriter by `SPEC_fuller_gpu.md`. The measurements above stand.
