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

---

## Revision 1, after Codex's adversarial review (logs/HGP/001/codex_review/general_safety_review.md) — supersedes §2 and §4

1. **Dispatch contract.** Exact workgroup size and counts, binding sizes and element layouts, admissible uniform ranges, all part of the problem's data. Normalise gid directly; accept wid·W + lid only when W is the rendered workgroup width (safe_reader.rs:886). Prove unsigned no-wrap before any affine normalisation.
2. **Write-set disjointness, not per-store injectivity.** Each store site becomes guarded affine rectangles over the thread coordinates and loop counters (path-sensitive guards bound every coordinate and counter); prove all sites and iterations pairwise disjoint across invocations, per bound buffer. Every stride must be proved non-zero with a symbolic lower bound (a uniform stride of 0 races everything). First release is conservative: guarded affine rectangles, proven positive strides, bounded loops; non-affine indices (/, %, select, data-dependent) on a store are illegal. It must accept the 4×4 blocked Gram writes (kingdoms/wgsl/gram/blocked.wgsl:40) and nluca's out[row] with its band offset and guard as proof inputs. Exhaustive small-domain differential tests (every invocation's write index enumerated) find bugs; they are not the proof.
3. **Aliasing and ordering.** Track aliasing by actual bound buffer range, not binding name. No reads of output cells other than the invocation's own; no cross-workgroup ordering within a dispatch. Workgroup memory: read/write happens-before through barriers. Subgroup collectives in uniform control flow only. Atomics: race-free is not deterministic; allowed atomic reductions and their fitness semantics are declared.
4. **Fitness as a dispatch program.** A problem declares typed binding schemas, initialisation and clearing, the alias policy, per-step uniform updates, ping-pong or copy edges between dispatches, persistent state, dispatch geometry, output slices and comparison points. Gram's band program (clear the band accumulator, iterate item bands, copy band output into H: gram.rs:640, 657, 691) is its first instance; nluca's recurrence (state ping-pong over N steps) the second.
5. **Order.** (a) parse and infer the interface plus the dispatch and binding contract; (b) no-wrap and the guarded-rectangle write-set checker with differential tests; (c) alias, read-after-write and workgroup-memory rules; (d) atomics and subgroup policy; (e) the generic dispatch-program harness; (f) Gram migrated bit for bit, then nluca.
