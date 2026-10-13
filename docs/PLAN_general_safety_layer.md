# PLAN: a general safety layer and a data-driven fitness (2026-10-13)

## Context
**Objective.** Any WGSL compute kernel — Gram, nluca's reservoir step, a future problem — goes parse → AST → Karva → evolve with no code or declaration written for that problem. The safety guarantees hold for every kernel the kingdom can express, inferred from the kernel itself.
**Why.** Today the safety layer and the fitness are specialised: the store must use the interface's declared `cell` dual (`// interface gram|cosine`), and the fitness (src/wgsl/gram.rs) hard-codes Φ, H, the 128 cases, the pairs and the timing shapes. Andrew (2026-10-13): no special components per problem; the safety layer must be general and always effective.
**Approach.** Infer roles and prove ownership per kernel; make the problem definition data.

## 1. Roles, inferred from the module
- Every `var<storage>` the entry point writes (a store, an atomic) is an output; every one it only reads is an input (whatever its declared access); every `var<uniform>` is a parameter block. No `// interface` line.
- Builtins (gid, lid, wid, lidx, nwg) and the workgroup size are known to the kingdom already.

## 2. Write ownership, proved per kernel (the one real change)
A plain store to an output buffer is legal only if its index is provably **injective across invocations** within one dispatch:
- The index, as an AST over the kingdom's duals, is reduced to a canonical affine form `Σ aₖ·tₖ + u` where tₖ are the thread's coordinates (gid.x, gid.y, gid.z, or wid·W + lid) and every aₖ, u is uniform (literals, uniform fields, products of them).
- Injective if the coordinate strides are mixed-radix: sort |aₖ| ascending, each stride ≥ (previous stride × extent of the previous coordinate's range), with the extents from the dispatch grid or a proven bound guard on the coordinate (e.g. `if (col < cfg.d)` bounds gid.x to cfg.d, so a row stride of cfg.d makes `row*cfg.d + col` injective).
- Blocked ownership (a thread owning rows r0..r0+B): the index is `f(thread) + k` with k in [0, B) a loop counter or unrolled constant, and f's stride ≥ B — the same mixed-radix test with k as an extra coordinate of extent B.
- A store whose index is not proved injective is illegal (or must be an atomic). Read-modify-write of the owned cell is allowed.
- Every proof step is a checker layer with a named failure, and is cross-checked by a dynamic test: for random shapes, enumerate the index each invocation writes on the CPU interpreter and assert no two collide (the r62-class lesson: a static proof needs an independent check).
This replaces the `cell` dual and the interface's owned-cell guard; the existing proof (cell_of over owned row/col) becomes one instance.

## 3. Unchanged general guarantees
Reads: wgpu runtime bounds checks (tests/runtime_bounds_checks.rs) plus masked workgroup and private indices. Loops: bounded, capped at 2^24. Arithmetic: no wraps, guarded integer / and % (f17ccb9). Barriers: uniform control flow. Runaways: detected and drained (ebb02db).

## 4. Fitness as data
A problem is a directory: `problem.toml` (bindings by name with sizes as functions of shape parameters; timing shapes; tolerance and its norm; steps per case for recurrent kernels) and `cases/` (inputs and expected outputs as raw little-endian bins, produced by the problem owner's reference — Gram's from gram::reference, nluca's from its host loop). One generic harness binds buffers by the inferred roles, runs N dispatches per case (feeding outputs back for recurrent kernels as `problem.toml` says), compares against the expected trajectory, and times at the shapes. The closeness columns work on any output tensor. Gram becomes the first problem directory, its saved results reproduced bit for bit.

## 5. Acceptance
- The Gram corpus, saved sets and 1,000 random genes: same legality verdicts and the same fitness values as today.
- nluca's six emitted kernels parse, are proved legal (their `out[row]` with row = gid + band offset injective), and score against their cases.
- A planted race (two threads writing one index) and a planted non-injective blocked store are rejected by the static proof and caught by the dynamic test.
