# PLAN: the challenge kernel — the fastest WGSL for the weighted Gram H = ΦᵀΦ

## Context

**Objective.** Find, by search over legal structural variants and
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

**Unique feature.** The search space is the space of legal kernels, not
of parameters of one kernel: register blocking, workgroup tiling,
vectorised loads, loop unrolling, symmetric half-computation and split
accumulators are composed freely by the generator, and the criterion and
the reference decide what counts, not the author's expectation of what
is fast on this GPU.

**Status.** Design, 2026-10-09; nothing built. Prompted by Andrew: "pick
a challenge, organise the whole experiment, search for the most
efficient supercompiled WGSL", after the decode pass proved too small to
measure (`PLAN_wgsl_superopt.md` §6a).

**Why this and not the chromosome.** The kingdom's chromosome evolves
expression trees inside a fixed scaffold; the gains here are in the
scaffold (tiles, loops, memory), which the chromosome cannot express.
So the genotype is the move vector of `PLAN_wgsl_superopt.md` §3, the
kernel is rebuilt by a template, and legality is by construction of each
move plus the reference gate. That is stated plainly, not hidden.

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

## 2. The genotype: legal moves

Each move is a transformation of the thread-per-output loop whose
result is the same sum in a different order or grouping:

| move | choices | what changes |
|---|---|---|
| register blocking | 1×1, 2×2, 4×4, 4×1 | each thread computes a block of H, loading each φ value once per block row/column |
| workgroup tile of items | none, 8, 16, 32, 64 | a tile of Φ rows staged in workgroup memory, shared by the 16 × 16 threads |
| workgroup size | 8×8, 16×16, 32×8, 8×32 | occupancy against register pressure |
| vector loads | 1, 4 (vec4 along c, contiguous in item-major Φ) | memory transactions |
| unroll of the item loop | 1, 2, 4, 8 | instruction scheduling |
| symmetric half | off, on (compute c ≥ r, mirror on write) | half the work; a write pattern the criterion still checks everywhere |
| split accumulators | 1, 2, 4 | independent partial sums, summed at the end (a reordering) |
| accumulation precision | f32, f32 with pairwise tile sums | the rounding order inside τ |

Grid size 4 × 5 × 4 × 2 × 4 × 2 × 3 × 2 = 3,840 genotypes. Not every
combination is well formed (a 4 × 4 block with an 8 × 8 workgroup and a
64-item tile may exceed workgroup memory); the generator refuses those
with a stated reason and the refusals are counted.

## 3. The search

1. **The ceiling first.** Enumerate the grid at the bench size: 3,840
   kernels, each compiled once (about 80 ms) and timed (one round, then
   the top 5 % re-timed five rounds paired against shipped). This gives
   the best reachable kernel and the whole landscape, and it is what the
   genetic algorithm is scored against.
2. **The genetic algorithm.** Population 32 over the genotype,
   tournament selection, uniform crossover, one-move mutation, 30
   generations, fitness the paired median GPU time, the criterion as the
   gate; deterministic from a seed; a ledger of every evaluation. Scored
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
for every task: the WGSL kingdom's legal-transformation machinery
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
