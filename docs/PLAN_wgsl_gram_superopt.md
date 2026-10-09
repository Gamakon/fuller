# PLAN: the challenge kernel — the fastest WGSL for the weighted Gram H = ΦᵀΦ

## Context

**Objective.** Find, by search over sum-preserving structural variants and
measured GPU time, the fastest WGSL kernel computing the weighted Gram
`H = ΦᵀΦ` (d × d, contracted over the item axis) that reproduces the
reference within a stated numerical bound; report the speed-up against
the shipped kernel at the production size, and the compiled code that
explains it.

**The challenge.** `H_GRAM_ACCUM_WGSL` in
`~/dev/minkymorgan/qdrant/lib/blueworld/src/sphere/generator.rs:2877`
(19 lines, thread-per-output, workgroup 16 × 16, one f32 accumulator, two
strided loads of `phi` per item). The inventory
(`docs/GPU_FUNCTIONS_INVENTORY.md`) measures it at **130 s** for
d = 8,192 over 419,734 items and names "a tiled shared-memory GEMM +
SYRK upper-triangle" as the scheduled follow-up. Chosen because: the
runtime is seconds, so a 10 % gain is 13 s, two orders of magnitude
above the noise that swallowed phylu's 0.3 ms decode pass; the kernel is
arithmetic- and bandwidth-bound, not dispatch-bound; it has a CPU
reference and a Metal smoke test; and the better structure is known to
exist, so the search has something to find and a way to be judged
against hand-written tiling.

**Approach.** A harness in fuller (wgpu, Metal) that builds Φ, runs a
kernel text through the same item-band and output-band driver shape the
production code uses, checks `H` against an f64 reference under a
criterion fixed in advance, and times each build with GPU timestamp
queries, paired and randomised against the shipped kernel. A generator
that turns a genotype of structural choices into WGSL text, every choice
a transformation known to preserve the sum up to rounding order. First
the whole grid of genotypes is enumerated (the ceiling: what the best
reachable kernel is); then the genetic algorithm searches the same space
and is scored by how much of the ceiling it reaches in how many
evaluations. The winner is validated at production size inside Blue
World and its Metal compiler output diffed against the shipped kernel's.

**Unique feature.** The search space is the space of kernels that compute the same sum, not
of parameters of one kernel: register blocking, workgroup tiling,
vectorised loads, loop unrolling, symmetric half-computation and split
accumulators are composed freely by the generator, and the criterion and
the reference decide what counts, not the author's expectation of what
is fast on this GPU.

**Status.** Task 1 delivered 2026-10-09 (fuller 5a999b0, §1a). The multityped kingdom of §2 delivered 2026-10-09 (fuller f7d6d35, 304e039, §2g): the rows, the loader, the renderer, the shipped kernel as a tree (bit-identical on the device), random typed kernels valid N of N. Not started: the pair generators of §1b (twelve kinds), phylu's evolution loop over the kingdom, the grid, the search. Prompted by Andrew: "pick
a challenge, organise the whole experiment, search for the most
efficient supercompiled WGSL", after the decode pass proved too small to
measure (`PLAN_wgsl_superopt.md` §6a).

**The genotype is the chromosome, not a template** (Andrew, 2026-10-09:
"why are you not assigning random genes using symbol table types … you
need a multityped kingdom"). The first draft of this plan varied a
hand-written template over eight structural moves because the kingdom's
table held only pure expressions and kept statements as a fixed
scaffold. That is replaced by §2: statements, loops, registers,
workgroup memory and the entry point become typed rows of the kingdom's
own table, so a random typed gene is a whole kernel by construction, as
a random typed gene of the regex kingdom is a pattern. The scaffold goes.

## 1. The measurement and the criterion

- **Sizes.** Bench: d = 2,048, m = 65,536 items (Φ 512 MB f32, under the
  per-binding limit in a few bands), expected shipped time of order one
  second, so a search evaluation is seconds. Production: d = 8,192,
  m = 419,734, the inventory's 130 s, run only for the final validation
  in Blue World's own driver.
- **Φ.** Deterministic: rows drawn from the Blue World sphere
  generator's convention (unit-normalised Gaussian rows, seeded), so any
  run reproduces the same Φ; a real Φ export from Blue World for the
  final validation if one is available.
- **Reference.** `H_ref` in f64 on the host over the bench size
  (2,048² × 65,536 multiply-adds, about a minute, cached to disk by
  seed), and for production a sampled set of 4,096 (r, c) entries.
- **The criterion, fixed now.** For every entry,
  `|H_gpu[r,c] − H_ref[r,c]| ≤ τ · ‖φ_·r‖ · ‖φ_·c‖` where the norms are
  column norms of Φ (the Cauchy–Schwarz scale of the entry) and
  `τ = 4 × ε_f32 × √m` (a random-walk bound on m roundings of unit-scale
  terms, times a margin of four; at m = 65,536 that is 1.2 × 10⁻⁴). The
  shipped kernel's own error against the reference is measured first and
  must sit inside τ; a variant that reorders the sum passes under the
  same τ. Bit identity is NOT the criterion: reordering a sum changes
  bits by design, and the inventory's own banding reports parity "≈
  1.5 × 10⁻⁷", not equality. Non-finite entries fail.
- **Timing.** GPU timestamp queries around every dispatch of a build,
  summed per build (the compute only; uploads excluded and reported
  apart); each candidate against the shipped kernel in paired,
  randomised order, five rounds, median and the interval; the device's
  `WindowServer` load noted before and after each batch, as the decode
  runs taught.

## 1a. Task 1 measured (2026-10-09, `logs/WGSL_GRAM/baseline.log`)

Apple M3 Max, Metal, 128 MiB per binding (production's `effective_buffer_limit`), so the bench Φ (537 MB) runs as 4 item bands × 1 output-row band = 4 compute passes per build. Φ: unit-normalised Gaussian rows, seed 1. Five builds of the shipped kernel, back to back, 15:43 UTC, load average 4.7 before and 7.7 after.

| measurement (conditions) | expected (§5) | measured | vs expected |
|---|---|---|---|
| shipped kernel GPU time per build, d = 2,048, m = 65,536, timestamp sum of 4 passes, median of 5 | 0.8–2 s | 2.35 s (min 2.09, max 2.37) | +17 % over the upper bound |
| the same, wall clock per build, uploads excluded | — | 2.39 s | — |
| Φ upload per build, 4 bands of 134 MB | — | 86 ms | — |
| shipped kernel's max criterion ratio, error / (τ ‖φ_r‖ ‖φ_c‖), τ = 1.22 × 10⁻⁴ | 0.05–0.3 | 0.033 (at the diagonal entry 1684, |diff| 1.3 × 10⁻⁴); mean 6 × 10⁻⁵; 0 failures | below the lower bound: 30× inside τ |
| f64 reference, 1.375 × 10¹¹ multiply-adds (upper triangle), 16 threads | about a minute | 5.6 s | 10× faster than stated |
| pipeline compile of the shipped text | 80 ms | 42 ms | −48 % |

The criterion holds with a margin of 30 on the shipped kernel, so a reordered sum has room; the ratio's worst entries are the diagonal, where every term is positive and the rounding is systematic. The run-to-run spread of the GPU time is 12 % (the first build fastest), which fixes the paired protocol of §1 for every later comparison. The reference is cached at `logs/WGSL_GRAM/ref_d2048_m65536_s1.f64` (32 MB) and the harness reads it on every later run. Tests: `cargo test --features gpu,wgsl gram` (5: Φ determinism and unit rows, the reference on a hand case, the criterion's counts, and the shipped kernel on the device at d = 64, m = 1,000 banded 63 × 4 and unbanded, both inside τ).

**Is the bench the production regime?** The inventory's 130 s at d = 8,192, m = 419,734 is 2.82 × 10¹³ multiply-adds, 217 G/s; the bench runs at 117 G/s. Two more shapes of the bench's work (2.75 × 10¹¹ multiply-adds), three rounds each (`logs/WGSL_GRAM/shape_*.log`):

| shape | bands | GPU time, median of 3 | rate | max ratio |
|---|---|---|---|---|
| d = 2,048, m = 65,536 | 4 item × 1 row | 2.35 s | 117 G/s | 0.033 |
| d = 4,096, m = 16,384 | 2 item × 1 row | 2.31 s | 119 G/s | 0.007 |
| d = 8,192, m = 4,096 (production's own pass shape) | 1 item × 2 row | 2.27 s | 121 G/s | 0.002 |

The kernel's rate does not depend on d or on the banding: the bench is the production regime. At 121 G/s the production build would take 233 s, not the inventory's 130 s, so that figure was taken under other conditions (another machine or load); task 4 measures production here before and after. The criterion ratio falls with m as τ √m expects.

**The gate after the first measurement (2026-10-09, `logs/WGSL_GRAM/evolve_smoke.log`).** The shipped kernel itself fails τ on the duplicate kind (half the rows identical): max ratio 1.45 at d = 2,048, m = 8,192, and 1.40 on the validation pair of that kind. With identical terms the rounding is systematic and the random-walk model behind τ = 4 ε √m does not hold. The gate is therefore τ or twice the shipped kernel's own largest ratio on that pair, whichever is larger, with every non-finite cell still required to agree in class. ShapeDistance itself is unchanged; the HFF caps were already the shipped kernel's own errors.

## 1b. Fitness: HFF over ShapeDistance

A kernel is a map from an input dataset to an output dataset. Fitness is a distance between the output it produces and the output it should produce, over the same rows and columns, falling to zero when the kernel computes the dataset; the HFF angle over that distance on three splits, then speed. No parsimony column: validation does that work.

**ShapeDistance.** For kernel k on a split with output `H_k` and reference `H_ref` (both d × d), the per-cell distance by cell kind:

- float cell: `δ(r,c) = |H_k[r,c] − H_ref[r,c]| / (‖φ_·r‖ ‖φ_·c‖)`, the Cauchy–Schwarz-scaled error;
- non-finite cell: 0 if the same class (NaN, +∞, −∞) on both sides, else 1;
- integer or flag cell (other kernels): 0 if the bits are equal, else 1.

`ShapeDistance_mean = Σ δ / d²`, `ShapeDistance_max = max δ`. The gate of §1 is `ShapeDistance_max ≤ τ` on every split; a kernel outside it is not scored.

**Splits: input–output pairs.** Fifty different training examples, not fifty seeds of one: each pair is a distinct kind of Φ, so a kernel that is right only on well-behaved Gaussian rows fails the dataset. The kinds, each at its own shape (d ∈ {512, 1,024, 2,048, 4,096}, m from 2,048 to 65,536) and seed:

| kind of Φ | what it exercises | pairs |
|---|---|---|
| unit-normalised Gaussian rows (the sphere convention) | the common case | 6 |
| √Dₜ-weighted rows, weights log-uniform over 6 decades | production's weighting, wide dynamic range in the sum | 6 |
| low-rank rows (rank 4, 16, 64 plus noise) | near-singular H, cancellation | 4 |
| sparse rows (90 %, 99 % zeros) | zero products, skipped work | 4 |
| heavy-tailed rows (Student t, ν = 2) | a few huge terms dominating the sum | 4 |
| rows with mixed sign and exact cancellation (x and −x pairs) | sums that should be exactly zero | 4 |
| integer-valued rows (small integers, exactly representable) | bit-exact sums expected | 4 |
| tiny magnitudes (10⁻²⁰, near denormal) and huge (10¹⁵) | underflow, overflow, flush-to-zero on Metal | 4 |
| cells with NaN and ±∞ at known positions | non-finite class propagation | 4 |
| duplicate and constant rows | H with repeated rows and columns | 3 |
| m not a multiple of the item band, d not a multiple of 16 | ragged bands and partial workgroups | 4 |
| a real Φ exported from Blue World (any split, any width) | the production distribution | 3 |

Validation: 20 pairs, the same twelve kinds at other seeds and shapes, never scored on by the search. Extrapolation: 20 pairs at d = 8,192 (production's width, two output-row bands), the kinds in the same proportion. The gate (`ShapeDistance_max ≤ τ`) applies per pair; τ's norms are each pair's own column norms, so the weighted and heavy-tailed kinds are judged at their own scale.

Every pair's f64 reference is cached by `(d, m, seed)`. ShapeDistance mean and max for a split are taken over all cells of all its pairs. Cost of scoring: the 50 train pairs sum to about 10 s of GPU time per kernel, so the grid's first pass (3,840 kernels) gates and times each on a fixed sample of 5 train pairs (about 1 s of GPU time, 4,000 s for the grid), and the full three-split score with five paired rounds is paid by the top 5 % (192 kernels, about 2 min each, 6 h); the genetic algorithm scores each generation on a fresh random sample of 5 train pairs and the full set on its elite, as the SR engine samples rows.

**The columns.** Six quality, two speed:

| i | column | value v_i | cap c_i |
|---|---|---|---|
| 1 | ShapeDistance_mean, train | as above | shipped's value on the split |
| 2 | ShapeDistance_max, train | | shipped's |
| 3 | ShapeDistance_mean, validation | | shipped's |
| 4 | ShapeDistance_max, validation | | shipped's |
| 5 | ShapeDistance_mean, extrapolation | | shipped's |
| 6 | ShapeDistance_max, extrapolation | | shipped's |
| 7 | time ratio | median over 5 paired, randomised rounds of `T_k / T_shipped` (GPU timestamps, same minute) | 1 |
| 8 | time spread | `(max − min) / median` of those 5 ratios | shipped's own spread (0.12 at the baseline) |

**Normalised inputs.** Each column is scaled into [0, 1] by its caller: `x_i = log10(1 + v_i / c_i) / M_i`, with `M_i` the column's maximum over the first population scored, frozen, and the result clipped; the log because the quality columns range over decades. The quality caps are the shipped kernel's own errors, so a column under `log10(2)/M_i` means more accurate than shipped on that split.

**The angle.** phylu's HFF takes the array as it is and returns the TrueNorth angle; it does not know what the columns are. Nothing changes in fuller for this: fuller's harness produces the eight values per kernel, the evolution loop (in phylu, as every fit is) normalises them and calls HFF. Selection, the hall of fame and the report use the angle; the grid's ceiling (§3) is the smallest angle over the grid, and that kernel's time ratio is the speed-up reported.

## 2. The genotype: the WGSL kingdom made multityped over statements

The kingdom's table (`kingdoms/wgsl/gen_tables.py` → `types.tsv`,
`functions.tsv`) gains the types below. Every row has a dual, an arity
and a signature over duals, nothing else; the typed projection (Design
C) draws only well-formed genes; the renderer writes text from the typed
tree; naga validates; the harness compiles, gates and scores. A gene's
root dual is `kernel.kernel`.

### 2a. New duals

| dual | meaning |
|---|---|
| `kernel.kernel` | the entry point: a grid and a body; the root |
| `grid.grid` | workgroup size (x, y) and the block of cells each thread owns (bx, by); the driver dispatches `ceil(d / (x·bx)) × ceil(r_rows / (y·by))` |
| `block.block` | a sequence of statements |
| `stmt.stmt` | one statement |
| `u32.wgdim` | a workgroup dimension, literal from {4, 8, 16, 32, 64} |
| `u32.blk` | a block factor, literal from {1, 2, 4} |
| `u32.unroll` | an unroll factor, literal from {2, 4, 8} |
| `reg<f32>.real`, `reg<vec4<f32>>.real`, `reg<u32>.index`, `reg<bool>.flag` | a named register (a `var` declared at the top of the kernel, initialised to the form's fallback): `f0…f3`, `v0…v1`, `i0…i3`, `b0` |
| `wgmem<f32>.real` | a workgroup-memory array of 4,096 f32: `a`, `b` |

### 2b. New functions

| class.instance | in → out | renders as |
|---|---|---|
| `kernel.entry` | grid, block → kernel | the module: bindings, registers, workgroup arrays, `@workgroup_size(x, y, 1) fn main` |
| `grid.of` | wgdim, wgdim, blk, blk → grid | the attribute and the driver's dispatch |
| `literal.wgdim`, `literal.blk`, `literal.unroll` | → the code | the number |
| `block.one`, `block.seq2`, `block.seq3`, `block.seq4` | stmt… → block | the statements in order |
| `stmt.block` | block → stmt | `{ … }` |
| `stmt.if`, `stmt.if_else` | flag, block[, block] → stmt | `if (c) { … } else { … }` |
| `stmt.barrier` | → stmt | `workgroupBarrier()` |
| `loop.items` | block → stmt | `for (t = 0u; t < cfg.m; t++) { … }`, the one long loop; nested in another `loop.items` the kernel is refused (the reason counted) |
| `loop.items_unrolled` | unroll, block → stmt | the body k times with `t + j`, then the remainder loop |
| `loop.tile` | wgdim, block → stmt | `for (k = 0u; k < n; k++)`, a short loop |
| `builtin.item`, `builtin.tile` | → u32.index | `t`, `k` (0 outside their loop: both are declared registers the loop assigns) |
| `reg.get` | reg<T> → T | the name |
| `reg.set` | reg<T>, T → stmt | `name = value;` |
| `wg.get` | wgmem, u32.index → f32.real | `a[i]` |
| `wg.set` | wgmem, u32.index, f32.real → stmt | `a[i] = v;` |
| `load.phi`, `load.phi4` | u32.index → f32.real, vec4<f32>.real | `phi[i]`, `vec4(phi[i], …, phi[i+3])` |
| `load.h`, `store.h` | u32.index [, f32.real] → f32.real, stmt | `h[i]`, `h[i] = v;` |
| `load.cfg_d`, `load.cfg_m`, `load.cfg_r_rows` → u32.count; `load.cfg_r_off` → u32.index | | the uniform's fields |

The `store.h`, `load.*` rows are the per-kernel instances the kingdom
already provides for a kernel's bindings (`per_kernel_templates`), here
fixed by the Gram's three bindings. `stmt.*`, `block.*`, `loop.*`,
`reg.*`, `wg.*`, `grid.*`, `kernel.*` and the three literal forms are
general and go in the table for every kernel.

### 2c. What is expressible, and what the moves of the first draft become

Register blocking: a `grid.of` with bx, by > 1 and a body that stores
several cells from register accumulators. Workgroup tiling: `loop.items`
over a `wg.set` of a tile, `stmt.barrier`, a `loop.tile` reading
`wg.get`. Vector loads: `load.phi4`. Unrolling: `loop.items_unrolled`.
Symmetric half: `stmt.if` on `compare.ge(c, r)` with two `store.h`.
Split accumulators and pairwise sums: several registers summed at the
end. None of these is a move any more; each is a shape a random typed
gene can take and the search can find or not.

### 2d. Bounds and refusals (by construction, counted)

Every buffer index is bounds-checked by WGSL (read zero, skip write);
workgroup-array indices are masked to the array; `loop.items` is bounded
by `cfg.m` and may not nest; `loop.tile` by its literal; the workgroup
size by 1,024 threads; registers are declared, so every read is defined.
A tree the renderer refuses (nested item loops, workgroup too large) is
counted by reason, as the generator counts its refusals. A kernel that
compiles but writes the wrong cells, or none, is scored by ShapeDistance
and ranks last; it is not refused.

### 2e. The seed and the reader

The population seeds from the shipped kernel written as a typed tree
(a test proves it renders to a text that compiles and passes the
criterion on the train pairs) plus random typed genes. Reading an
arbitrary kernel's statements into the tree (the reader's statement
tree becoming `block`/`stmt`/`loop` nodes) is the next increment after
the search runs; it is not needed to seed this experiment.

### 2f. What fuller delivers and what phylu does

fuller: the rows in `gen_tables.py` and the regenerated tables, the
loader instantiating them, the renderer (`src/wgsl/render.rs`: typed
tree → text, refusals), the Gram harness scoring a rendered kernel over
the pairs of §1b, tests (the kingdom gate: the share of random typed
genes that render and compile; the shipped tree; each statement row
rendered and validated alone). phylu: the evolution loop over this
kingdom's table with `kernel.kernel` as the root dual, HFF on the eight
normalised columns, the seed population.

### 2g. Built and measured (2026-10-09, fuller f7d6d35, 304e039)

| measurement | expected | measured |
|---|---|---|
| duals, templates in the regenerated tables | 71 + 12, 197 + 42 | 83, 239 |
| the shipped kernel as a typed tree | under 128 nodes (one gene) | 82 nodes; rendered text validates; H on the device bit-identical to the shipped text at d = 64, m = 1,000, banded 63 × 4 |
| 500 random typed kernels (depth ≤ 7, mean 35 nodes), rendered and validated by naga | every rendered text valid | 440 rendered, 440 valid; 60 refused by construction, all nested long loops (m² iterations); 122 barriers masked for non-uniform control flow |
| 100 random typed kernels compiled by Metal and dispatched at d = 64, m = 100 | every rendered one compiles and runs | 94 rendered, 94 compiled, 94 ran, 93 with every cell of H finite |
| every statement row alone, every composite row alone | valid | 15 statement cases and 44 composite rows, all valid |

Decisions taken while building, each a line in `render.rs`: literals pass through `lit_f32/u32/i32` so naga's constant evaluator never sees a literal-only expression (a folded NaN or a division by a literal zero is a validation error, on the device a value); statements after a `return` in the same block are dropped (naga refuses instructions after a return); a `workgroupBarrier` inside an `if` or after a conditional return is masked and counted (WGSL's uniformity rule), the typed projection's masking at the statement level; a long loop inside a long loop is refused; the matrix product template was wrong for WGSL's `matCxR` and is corrected (27 shapes). A kernel that uses no binding still binds all three (explicit layout). The sampler never draws an `f16` dual (the prologue does not enable f16).

## 3. The search

1. **The ceiling first.** Enumerate the grid at the bench size: 3,840
   kernels, each compiled once (about 40 ms), gated and timed on the
   train split (one round), then the top 5 % scored in full (§1b: three
   splits, five paired rounds). This gives
   the best reachable kernel and the whole landscape, and it is what the
   genetic algorithm is scored against.
2. **The genetic algorithm.** Population 32 over the genotype,
   tournament selection, uniform crossover, one-move mutation, 30
   generations, fitness the HFF angle of §1b over the three splits, the
   criterion as the gate; deterministic from a seed; a ledger of every evaluation. Scored
   by the fraction of the ceiling's speed-up reached and the number of
   evaluations spent against the grid's 3,840.
3. **Three seeds** of the algorithm, reported separately.

## 4. Validation and explanation

- The winner's WGSL into Blue World's `H_GRAM_ACCUM_WGSL` slot behind a
  switch, its smoke test (`tests/gpu_inventory_smoke.rs`) run, and the
  production build timed three times against shipped three times,
  interleaved.
- naga's MSL and `xcrun -sdk macosx metal -S` AIR for shipped and winner,
  diffed and read: which of the moves the compiler had not already made.
- The report: the table of expected against measured, the ceiling, the
  algorithm's convergence, the production speed-up with its interval,
  and the criterion's measured margins.

## 5. Expectations, stated before any run

| measurement | expected |
|---|---|
| shipped kernel, bench size (d = 2,048, m = 65,536), GPU time per build | 0.8–2 s |
| shipped kernel's max criterion ratio (error / τ) | 0.05–0.3 |
| grid refusals (ill-formed combinations) | 10–25 % |
| best grid kernel against shipped | 4–12× faster (tiled GEMMs are typically 5–20× over naive) |
| the largest single move | workgroup tiling, then register blocking |
| genetic algorithm reaching 90 % of the ceiling's speed-up | within 300 evaluations (8 % of the grid) |
| production speed-up of the winner (130 s baseline) | the bench ratio within a factor of 1.5, limited by banding overhead |
| compiled-code diff | tiles and register blocks visible as workgroup memory and unrolled FMAs absent from shipped |

## 6. Order, as tasks

Every task follows `~/dev/minkymorgan/qdrant/docs/Rules.md`: no stubs,
real data, warnings fixed, every change committed with a factual message,
wall-clock and GPU time reported, timeouts from the run's own cost, one
GPU process at a time, paired randomised timing only. The wider project,
for every task: the WGSL kingdom's sum-preserving-transformation machinery
(fuller `src/wgsl/*`) applied as a superoptimiser to a production kernel
of the Blue World stack (qdrant `lib/blueworld`), the first kernel where
the gain is measurable in seconds. Tools: fuller (wgpu 0.20, Metal, the
oracle's device harness in `src/wgsl/oracle.rs`), naga, `xcrun metal`,
Blue World's generator and smoke test.

1. **Read docs/Rules.md, then: the harness and the baseline.** Purpose:
   the number everything is measured against. Architecture: fuller
   `examples/wgsl_gram.rs` under `gpu,wgsl`: seeded Φ, the banded driver
   (item bands and output-row bands from the device's binding limit, as
   `build_weighted_gram_item_banded_gpu` does), the shipped kernel text
   verbatim, GPU timestamps per dispatch, the f64 reference cached by
   seed, the criterion of §1. Done when the shipped kernel's bench time
   (five rounds) and its criterion margin are in the README with the
   expected values of §5 beside them.
2. **Read docs/Rules.md, then: the generator and the grid.** Purpose:
   the ceiling. Architecture: `src/wgsl/gram_template.rs`, a genotype
   (§2) to WGSL text, refusals stated; every generated kernel through
   naga validation, the criterion and the timer; the full grid
   enumerated at the bench size, one round each, the top 5 % five rounds
   paired. Tests: each move alone against shipped passes the criterion;
   the 1×1/none/16×16/1/1/off/1/f32 genotype reproduces the shipped
   kernel's text. Done when the landscape table (time by move) and the
   best kernel are in the README.
3. **Read docs/Rules.md, then: the genetic algorithm.** Purpose: the
   claim that search finds what enumeration finds, cheaper. Architecture:
   §3, in fuller (`examples/wgsl_gram_evolve.rs`), seeded, ledgered;
   three seeds. Done when the convergence table (evaluations against
   fraction of ceiling) is in the README.
4. **Read docs/Rules.md, then: production validation in Blue World.**
   Purpose: the speed-up that matters. Architecture: the winner's text
   behind a switch in `generator.rs`, the smoke test, three interleaved
   production builds each; a qdrant commit. Done when the production
   table with intervals is in both READMEs.
5. **Read docs/Rules.md, then: the compiled code.** Purpose: the
   explanation. Architecture: §4's diff, read line by line. Done when the
   reading is in the README.
6. **Read docs/Rules.md, then: the report**, in the table form of
   expectations against measurements, 200 words of prose at most.
